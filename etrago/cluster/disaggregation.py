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

    def add_constraints(self, cluster, extra_functionality=None):
        """
        Dummy function that allows the extension of `extra_functionalites` by
        custom conditions.

        :param cluster: Index of the cluster to disaggregate
        :param extra_functionality: extra_functionalities to extend
        :return: unaltered `extra_functionalites`
        """
        return extra_functionality

    def construct_partial_network(self, cluster):
        """
        Compute the network partial network that has been merged into a single
        cluster

        :param cluster: Index of the cluster to disaggregate
        :return: Tuple of (partial_network, external_buses) where
        `partial_network` is the result of the partial decomposition
        and `external_buses` represent clusters adjacent to `cluster` that may
        be influenced by calculations done on the partial network.
        """
        partial_network = Network()

        # find all lines that have at least one bus inside the cluster
        busflags = (self.buses['cluster'] == cluster)

        def is_bus_in_cluster(conn):
            return busflags[conn]

        # Copy configurations to new network
        partial_network.snapshots = self.original_network.snapshots
        partial_network.snapshot_weightings = self.original_network.snapshot_weightings
        partial_network.carriers = self.original_network.carriers

        line_types = ['lines', 'links', 'transformers']

        # Collect all connectors that have some node inside the cluster

        external_buses = pd.DataFrame()

        idx_prefix = '_'

        for line_type in line_types:
            setattr(partial_network, line_type,
                    filter_internal_connector(
                        getattr(self.original_network, line_type),
                        is_bus_in_cluster))
            setattr(partial_network, line_type + '_t',
                    getattr(self.original_network, line_type + '_t'))

            left_external_connectors = filter_left_external_connector(
                getattr(self.original_network, line_type),
                is_bus_in_cluster)

            if not left_external_connectors.empty:
                left_external_connectors.bus0 = idx_prefix + left_external_connectors.bus0_s
                external_buses = pd.concat(
                    (external_buses, left_external_connectors.bus0))

            right_external_connectors = filter_right_external_connector(
                getattr(self.original_network, line_type),
                is_bus_in_cluster)
            if not right_external_connectors.empty:
                right_external_connectors.bus1 = idx_prefix + right_external_connectors.bus1_s
                external_buses = pd.concat(
                    (external_buses, right_external_connectors.bus1))

        # Collect all buses that are contained in or somehow connected to the
        # cluster

        buses_in_lines = self.buses[busflags].index

        bus_types = ['loads', 'generators', 'stores', 'storage_units',
                     'generators', 'shunt_impedances']

        partial_network.buses = self.original_network.buses[
            self.original_network.buses.index.isin(buses_in_lines)]
        partial_network.buses = partial_network.buses.append(
            self.clustered_network.buses[
                self.clustered_network.buses.index.isin(
                    map(lambda x: x[0][len(idx_prefix):],
                        external_buses.values))])
        partial_network.buses_t = self.original_network.buses_t

        for bustype in bus_types:
            # Copy loads, generators, ... from original network to network copy
            setattr(partial_network, bustype,
                    filter_buses(getattr(self.original_network, bustype),
                                 buses_in_lines))

            # Include external clusters
            buses_to_insert = filter_buses(
                getattr(self.clustered_network, bustype),
                map(lambda x: x[0][len(idx_prefix):],
                    external_buses.values))
            buses_to_insert.index = pd.Index(idx_prefix + buses_to_insert.index)
            setattr(partial_network, bustype,
                    getattr(partial_network, bustype).append(buses_to_insert))

            # Also copy t-dictionaries
            setattr(partial_network,
                    bustype + '_t',
                    getattr(self.original_network, bustype + '_t'))

        for line_type in line_types:
            assert (getattr(partial_network, line_type).bus0.isin(
                partial_network.buses.index).all())
            assert (getattr(partial_network, line_type).bus1.isin(
                partial_network.buses.index).all())

        return partial_network, external_buses

    def execute(self, scenario, solver=None):
        self.solve(scenario, solver)

    def solve(self, scenario, solver):
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
            partial_network, externals = self.construct_partial_network(cluster)
            print('Decomposed in ', (time.time() - t))
            t = time.time()
            profile.disable()
            profile.enable()
            self.solve_partial_network(cluster, partial_network, scenario, solver)
            profile.disable()

            print('Decomposition optimized in ', (time.time() - t))

        profile.print_stats(sort='cumtime')

    def solve_partial_network(self, cluster, partial_network, scenario, solver=None):
        extras = self.add_constraints(cluster)
        partial_network.lopf(scenario.timeindex,
                             solver_name=solver,
                             extra_functionality=extras)

class MiniSolverDisaggregation(Disaggregation):
    def add_constraints(self, cluster, extra_functionality=None):
        if extra_functionality is None:
            extra_functionality = lambda network, snapshots: None
        return self._validate_disaggregation_generators(cluster,
                                                        extra_functionality)

    def _validate_disaggregation_generators(self, cluster, f):
        def extra_functionality(network, snapshots):
            f(network, snapshots)
            generators = self.original_network.generators.assign(
                bus=lambda df: df.bus.map(self.clustering.busmap))
            grouper = [generators.carrier]
            i = 0
            for snapshot in snapshots:
                for carrier in generators.carrier:
                    def construct_constraint(model):
                        # TODO: Optimize

                        generator_p = [model.generator_p[(x, snapshot)] for x in
                                       generators[generators.bus == cluster][
                                           generators.carrier == carrier].index]
                        if not generator_p:
                            return Constraint.Feasible
                        sum_generator_p = sum(generator_p)

                        cluster_generators = self.clustered_network.generators[
                            self.clustered_network.generators.bus == cluster][
                            self.clustered_network.generators.carrier == carrier]
                        sum_clustered_p = sum(
                            self.clustered_network.generators_t['p'][c][
                                snapshot] for
                            c in cluster_generators.index)
                        return sum_generator_p == sum_clustered_p

                    # TODO: Generate a better name
                    network.model.add_component('validate_generators' + str(i),
                                                Constraint(
                                                    rule=construct_constraint))
                    i += 1

        return extra_functionality


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
