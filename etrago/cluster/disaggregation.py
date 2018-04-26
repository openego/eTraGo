from itertools import product
import cProfile
import time

import pandas as pd
from pyomo.environ import Constraint
from pypsa import Network


class Disaggregation:
    def __init__(self, original_network, clustered_network, clustering):
        """
        :param original_network: Initial (unclustered) network structure
        :param clustered_network: Clustered network used for the optimization
        :param clustering: The clustering object as returned by
        `pypsa.networkclustering.get_clustering_from_busmap`
        """
        self.original_network = original_network
        self.clustered_network = clustered_network
        self.clustering = clustering

        self.buses = pd.merge(original_network.buses,
                              clustering.busmap.to_frame(name='cluster'),
                              left_index=True, right_index=True)

        self.idx_prefix = '_'

    def add_constraints(self, cluster, extra_functionality=None):
        """
        Dummy function that allows the extension of `extra_functionalites` by
        custom conditions.

        :param cluster: Index of the cluster to disaggregate
        :param extra_functionality: extra_functionalities to extend
        :return: unaltered `extra_functionalites`
        """
        return extra_functionality

    def reindex_with_prefix(self, dataframe, prefix=None):
        if prefix is None:
            prefix = self.idx_prefix
            dataframe.set_index(
                dataframe.index.map(lambda x: self.idx_prefix + x),
            inplace=True)



    def construct_partial_network(self, cluster, scenario):
        """
        Compute the partial network that has been merged into a single cluster.
        The resulting network retains the external cluster buses that
        share some line with the cluster identified by `cluster`.
        These external buses will be prefixed by self.id_prefix in order to
        prevent name clashes with buses in the disaggregation

        :param cluster: Index of the cluster to disaggregate
        :return: Tuple of (partial_network, external_buses) where
        `partial_network` is the result of the partial decomposition
        and `external_buses` represent clusters adjacent to `cluster` that may
        be influenced by calculations done on the partial network.
        """

        #Create an empty network
        partial_network = scenario.build_network()

        # find all lines that have at least one bus inside the cluster
        busflags = (self.buses['cluster'] == cluster)

        def is_bus_in_cluster(conn):
            return busflags[conn]

        # Copy configurations to new network
        partial_network.snapshots = self.original_network.snapshots
        partial_network.snapshot_weightings = (self.original_network
                                                   .snapshot_weightings)
        partial_network.carriers = self.original_network.carriers

        # Collect all connectors that have some node inside the cluster

        external_buses = pd.DataFrame()

        line_types = ['lines', 'links', 'transformers']
        for line_type in line_types:
            # Copy all lines that reside entirely inside the cluster ...
            setattr(partial_network, line_type,
                    filter_internal_connector(
                        getattr(self.original_network, line_type),
                        is_bus_in_cluster))

            # ... and their time series
            # TODO: These are all time series, not just the ones from lines
            #       residing entirely in side the cluster.
            #       Is this a problem?
            setattr(partial_network, line_type + '_t',
                    getattr(self.original_network, line_type + '_t'))

            # Copy all lines whose `bus0` lies within the cluster
            left_external_connectors = filter_left_external_connector(
                getattr(self.original_network, line_type),
                is_bus_in_cluster)

            if not left_external_connectors.empty:
                f = lambda x: self.idx_prefix + self.clustering.busmap[x]
                left_external_connectors.bus0 = (left_external_connectors
                                                 .bus0
                                                 .apply(f))
                external_buses = pd.concat((external_buses,
                                            left_external_connectors.bus0))

            # Copy all lines whose `bus1` lies within the cluster
            right_external_connectors = filter_right_external_connector(
                getattr(self.original_network, line_type),
                is_bus_in_cluster)
            if not right_external_connectors.empty:
                f = lambda x: self.idx_prefix + self.clustering.busmap[x]
                right_external_connectors.bus1 = (right_external_connectors
                                                  .bus1
                                                  .apply(f))
                external_buses = pd.concat((external_buses,
                                            right_external_connectors.bus1))

        # Collect all buses that are contained in or somehow connected to the
        # cluster

        buses_in_lines = self.buses[busflags].index

        bus_types = ['loads', 'generators', 'stores', 'storage_units',
                     'shunt_impedances']

        # Copy all values that are part of the cluster
        partial_network.buses = self.original_network.buses[
            self.original_network.buses.index.isin(buses_in_lines)]

        # Collect all buses that are external, but connected to the cluster ...
        externals_to_insert = self.clustered_network.buses[
            self.clustered_network.buses.index.isin(
                map(lambda x: x[0][len(self.idx_prefix):],
                    external_buses.values))]

        # ... prefix them to avoid name clashes with buses from the original
        # network ...
        self.reindex_with_prefix(externals_to_insert)

        # .. and insert them as well as their time series
        partial_network.buses = (partial_network.buses
                                                .append(externals_to_insert))
        partial_network.buses_t = self.original_network.buses_t

        # TODO: Rename `bustype` to on_bus_type
        for bustype in bus_types:
            # Copy loads, generators, ... from original network to network copy
            setattr(partial_network, bustype,
                    filter_buses(getattr(self.original_network, bustype),
                                 buses_in_lines))

            # Collect on-bus components from external, connected clusters
            buses_to_insert = filter_buses(
                getattr(self.clustered_network, bustype),
                map(lambda x: x[0][len(self.idx_prefix):],
                    external_buses.values))

            # Prefix their external bindings
            buses_to_insert.bus = self.idx_prefix + buses_to_insert.bus

            setattr(partial_network, bustype,
                    getattr(partial_network, bustype).append(buses_to_insert))

            # Also copy their time series
            setattr(partial_network,
                    bustype + '_t',
                    getattr(self.original_network, bustype + '_t'))

        # Just a simple sanity check
        # TODO: Remove when sure that disaggregation will not go insane anymore
        for line_type in line_types:
            assert (getattr(partial_network, line_type).bus0.isin(
                partial_network.buses.index).all())
            assert (getattr(partial_network, line_type).bus1.isin(
                partial_network.buses.index).all())

        return partial_network, external_buses

    def execute(self, scenario, solver=None):
        self.solve(scenario, solver)

    def solve(self, scenario, solver):
        """
        Decompose each cluster into separate units and try to optimize them
        separately
        :param scenario:
        :param solver: Solver that may be used to optimize partial networks
        """
        clusters = set(self.clustering.busmap.values)
        n = len(clusters)
        i = 0
        profile = cProfile.Profile()
        for cluster in clusters:
            i += 1

            print('---')
            print('Decompose cluster %s (%d/%d)' % (cluster, i, n))
            profile.enable()
            t = time.time()
            partial_network, externals = self.construct_partial_network(
                    cluster,
                    scenario)
            profile.disable()
            print('Decomposed in ', (time.time() - t))
            t = time.time()
            profile.enable()
            self.solve_partial_network(cluster, partial_network, scenario,
                                       solver)
            profile.disable()
            print('Partial network solved in ', (time.time() - t))
            profile.enable()
            t = time.time()
            self.transfer_results(partial_network, externals)
            profile.disable()
            print('Results transferred in ', (time.time() - t))

        # profile.print_stats(sort='cumtime')

    def transfer_results(self, partial_network, externals,
                         bustypes=['loads', 'generators', 'stores',
                                   'storage_units', 'shunt_impedances']):
        for bustype in bustypes:
            orig_buses = getattr(self.original_network, bustype + '_t')
            part_buses = getattr(partial_network, bustype + '_t')
            for key in orig_buses.keys():
                for snap in partial_network.snapshots:
                    orig_buses[key].loc[snap].update(part_buses[key].loc[snap])


    def solve_partial_network(self, cluster, partial_network, scenario,
                              solver=None):
        extras = self.add_constraints(cluster)
        partial_network.lopf(scenario.timeindex,
                             solver_name=solver,
                             extra_functionality=extras)

class MiniSolverDisaggregation(Disaggregation):
    def add_constraints(self, cluster, extra_functionality=None):
        if extra_functionality is None:
            extra_functionality = lambda network, snapshots: None
        extra_functionality = self._validate_disaggregation_generators(
                cluster,
                extra_functionality)
        return extra_functionality

    def _validate_disaggregation_generators(self, cluster, f):
        def extra_functionality(network, snapshots):
            f(network, snapshots)
            generators = self.original_network.generators.assign(
                bus=lambda df: df.bus.map(self.clustering.busmap))
            grouper = [generators.carrier]
            def construct_constraint(model, snapshot, carrier):
                # TODO: Optimize

                generator_p = [model.generator_p[(x, snapshot)]
                               for x in generators.loc[
                                   (generators.bus == cluster) &
                                   (generators.carrier == carrier)].index]
                if not generator_p:
                    return Constraint.Feasible
                sum_generator_p = sum(generator_p)

                cluster_generators = self.clustered_network.generators[
                    (self.clustered_network.generators.bus == cluster) &
                    (self.clustered_network.generators.carrier == carrier)]
                sum_clustered_p = sum(
                    self.clustered_network.generators_t['p'].loc[snapshot, c]
                    for c in cluster_generators.index)
                return sum_generator_p == sum_clustered_p

            # TODO: Generate a better name
            network.model.validate_generators = Constraint(
                    list(snapshots),
                    set(generators.carrier),
                    rule=construct_constraint)
        return extra_functionality

    # TODO: This function is never used.
    #       Is this a problem?
    def _validate_disaggregation_buses(self, cluster, f):
        def extra_functionality(network, snapshots):
            f(network, snapshots)

            for bustype, bustype_pypsa, suffixes in [
                ('storage', 'storage_units', ['_dispatch', '_spill', '_store']),
                ('store', 'stores',[''])]:
                generators = getattr(self.original_network, bustype_pypsa).assign(
                    bus=lambda df: df.bus.map(self.clustering.busmap))
                for suffix in suffixes:
                    def construct_constraint(model, snapshot):
                        # TODO: Optimize
                        buses_p = [getattr(model,bustype + '_p' + suffix)[(x, snapshot)] for x in
                                       generators.loc[(generators.bus == cluster)].index]
                        if not buses_p:
                            return Constraint.Feasible
                        sum_bus_p = sum(buses_p)
                        cluster_buses = getattr(self.clustered_network, bustype_pypsa)[
                            (getattr(self.clustered_network, bustype_pypsa).bus == cluster)]
                        sum_clustered_p = sum(
                            getattr(self.clustered_network, bustype_pypsa + '_t')['p'].loc[snapshot,c] for
                            c in cluster_buses.index)
                        return sum_bus_p == sum_clustered_p

                    # TODO: Generate a better name
                    network.model.add_component('validate_' + bustype + suffix,Constraint(list(snapshots), rule=construct_constraint))
        return extra_functionality

class UniformDisaggregation(Disaggregation):
    def solve_partial_network(self, cluster, partial_network, scenario,
                              solver=None):
        bustypes = ('generators', 'storage_units')
        groupings = {'generators': ('carrier',),
                     'storage_units': ('carrier', 'max_hours')}
        for bustype in bustypes:
            pn_t = getattr(partial_network, bustype + '_t')
            p_max_pu_t = pn_t['p_max_pu']
            cl_t = getattr(self.clustered_network, bustype + '_t')
            pn_buses = getattr(partial_network, bustype)
            cl_buses = getattr(self.clustered_network, bustype)
            groups = product(*
                    [ [ {'key': key, 'value': value}
                        for value in set(getattr(pn_buses, key))]
                      for key in groupings[bustype]])
            for group in groups:
                clb = cl_buses[cl_buses.bus == cluster]
                query = " & ".join(["({key} == {value!r})".format(**axis)
                                    for axis in group])
                clb = clb.query(query)
                if len(clb) == 0:
                    break
                assert len(clb) == 1, (
                    "Cluster {} has {} buses for group {}.\n"
                    .format(cluster, len(clb), group) +
                    "Should be exactly one.")
                # Remove cluster buses from partial network
                cluster_bus_names = [r[0] for r in cl_buses.iterrows()]
                pnb = pn_buses.iloc[
                        [r[0] for r in enumerate(pn_buses.iterrows())
                              if r[1][0] not in cluster_bus_names]]
                pnb = pnb.query(query)
                clt = cl_t['p'].loc[:, list(clb.iterrows())[0][0]]
                timed = p_max_pu_t.columns.intersection(pnb.index)
                if not timed.empty:
                    pnmp = pnb.p_nom * p_max_pu_t.loc[:, timed]
                    index = timed
                    psum = pnmp.sum(axis=1)
                    for bus_id in index:
                        # TODO: Check whether series multiplication works as
                        #       expected.
                        pn_t['p'].loc[:, bus_id] = (
                                clt * pnmp.loc[:, bus_id] / psum)

                else:
                    pnmp = pnb.p_nom * pnb.p_max_pu
                    index = pnmp.index
                    psum = pnmp.sum()
                    for bus_id in index:
                        pn_t['p'].loc[:, bus_id] = (
                                clt * pnmp.loc[bus_id] / psum)


    def transfer_results(self, *args, **kwargs):
        kwargs['bustypes'] = ['generators', 'storage_units']
        return super().transfer_results(*args, **kwargs)


def swap_series(s):
    return pd.Series(s.index.values, index=s)


def filter_internal_connector(conn, is_bus_in_cluster):
    return conn[conn.bus0.apply(is_bus_in_cluster)
                & conn.bus1.apply(is_bus_in_cluster)]


def filter_left_external_connector(conn, is_bus_in_cluster):
    return conn[~ conn.bus0.apply(is_bus_in_cluster)
                & conn.bus1.apply(is_bus_in_cluster)]


def filter_right_external_connector(conn, is_bus_in_cluster):
    return conn[conn.bus0.apply(is_bus_in_cluster)
                & ~conn.bus1.apply(is_bus_in_cluster)]


def filter_buses(bus, buses):
    return bus[bus.bus.isin(buses)]


def filter_on_buses(connecitve, buses):
    return connecitve[connecitve.bus.isin(buses)]


def update_constraints(network, externals):
    pass
