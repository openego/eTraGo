import networkx as nx
import pandas as pd
from pypsa import Network
import numpy as np
import time
import cProfile
from pyomo.environ import Constraint

def swap_series(s):
    return pd.Series(s.index.values, index=s)

def disaggregate(scenario, original_network, network, clustering, solver=None):
    build_cluster_maps(scenario, original_network, network, clustering, solver)

def build_cluster_maps(scenario, original_network, network, clustering, solver):
    additional_constraints = {}
    clusters = set(clustering.busmap.values)
    n = len(clusters)
    i = 0
    profile = cProfile.Profile()
    buses = pd.merge(original_network.buses, clustering.busmap.to_frame(name='cluster'), left_index=True, right_index=True)
    for cluster in clusters:
        i += 1

        print('---')
        print('Decompose cluster %s (%d/%d)'%(cluster, i, n))
        profile.enable()
        t = time.time()
        partial_network, externals, extras = construct_partial_network(original_network, network, clustering, cluster, buses)
        print('Decomposed in ', (time.time()-t))
        t = time.time()
        profile.disable()
        profile.enable()

        partial_network.lopf(scenario.timeindex, solver_name=solver,
                             extra_functionality=extras)
        profile.disable()

        update_constraints(partial_network, externals)

        print('Decomposition optimized in ', (time.time() - t))

    profile.print_stats(sort='cumtime')


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


def construct_partial_network(original_network, clustered_network, clustering, cluster, buses):
    partial_network = Network()

    # find all lines that have at least one bus inside the cluster
    busflags = (buses['cluster'] == cluster)

    def is_bus_in_cluster(conn):
        return busflags[conn]

    # Copy configurations to new network
    partial_network.snapshots = original_network.snapshots
    partial_network.snapshot_weightings = original_network.snapshot_weightings
    partial_network.carriers = original_network.carriers

    line_types = ['lines', 'links', 'transformers']

    # Collect all connectors that have some node inside the cluster

    external_buses = pd.DataFrame()

    idx_offset = int(max(original_network.buses.index)) + 1
    idx_prefix = '_'

    for line_type in line_types:
        setattr(partial_network, line_type,
                filter_internal_connector(getattr(original_network, line_type),
                                          is_bus_in_cluster))
        setattr(partial_network, line_type+'_t',
                getattr(original_network, line_type+'_t'))

        left_external_connectors = filter_left_external_connector(getattr(original_network, line_type),
                                                                  is_bus_in_cluster)

        if not left_external_connectors.empty:
            left_external_connectors.bus0 = idx_prefix + left_external_connectors.bus0_s
            external_buses = pd.concat((external_buses, left_external_connectors.bus0))

        right_external_connectors = filter_right_external_connector(getattr(original_network, line_type),
                                                                 is_bus_in_cluster)
        if not right_external_connectors.empty:
            right_external_connectors.bus1 = idx_prefix + right_external_connectors.bus1_s
            external_buses = pd.concat((external_buses, right_external_connectors.bus1))

    # Collect all buses that are contained in or somehow connected to the
    # cluster

    buses_in_lines = buses[busflags].index

    bus_types = ['loads', 'generators', 'stores', 'storage_units',
                 'generators', 'shunt_impedances']

    extra_functionality = (lambda network, externals: None)

    partial_network.buses = original_network.buses[original_network.buses.index.isin(buses_in_lines)]
    partial_network.buses = partial_network.buses.append(
        clustered_network.buses[
            clustered_network.buses.index.isin(
                map(lambda x: x[0][len(idx_prefix):], external_buses.values))])
    partial_network.buses_t = original_network.buses_t

    for bustype in bus_types:
        # Copy loads, generators, ... from original network to network copy
        setattr(partial_network, bustype, filter_buses(getattr(original_network, bustype),buses_in_lines))

        # Include external clusters
        buses_to_insert = filter_buses(getattr(clustered_network, bustype), map(lambda x: x[0][len(idx_prefix):], external_buses.values))
        buses_to_insert.index = pd.Index(idx_prefix + buses_to_insert.index)
        setattr(partial_network, bustype, getattr(partial_network, bustype).append(buses_to_insert))

        # Also copy t-dictionaries
        setattr(partial_network,
                bustype+'_t',
                getattr(original_network, bustype+'_t'))

    extra_functionality = _validate_disaggregation_generators(original_network, clustered_network, clustering, cluster,
                                                              extra_functionality)

    for line_type in line_types:
        assert (getattr(partial_network, line_type).bus0.isin(partial_network.buses.index).all())
        assert (getattr(partial_network, line_type).bus1.isin(partial_network.buses.index).all())

    return partial_network, external_buses, extra_functionality


def update_constraints(network, externals):
    pass


def _validate_disaggregation_generators(original_network, clustered_network, clustering, cluster, f):
    def extra_functionality(network, snapshots):
        f(network, snapshots)
        generators = original_network.generators.assign(
            bus=lambda df: df.bus.map(clustering.busmap))
        grouper = [generators.carrier]
        i = 0
        for snapshot in snapshots:
            for carrier in generators.carrier:
                def construct_constraint(model):
                    #TODO: Optimize

                    generator_p = [model.generator_p[(x, snapshot)] for x in generators[generators.bus == cluster][
                                generators.carrier == carrier].index]
                    if not generator_p:
                        return Constraint.Feasible
                    sum_generator_p = sum(generator_p)

                    cluster_generators = clustered_network.generators[clustered_network.generators.bus==cluster][clustered_network.generators.carrier==carrier]
                    sum_clustered_p = sum(clustered_network.generators_t['p'][c][snapshot] for c in cluster_generators.index)
                    return sum_generator_p == sum_clustered_p
                #TODO: Generate a better name
                network.model.add_component('validate_generators'+str(i), Constraint(rule=construct_constraint))
                i += 1
    return extra_functionality


def _preserve_active_power(parent, buses, c_name, f):
    def extra_functionality(network, snapshots):
        f(network, snapshots)
        model = network.model
        p_clustered = parent.p
        constraint = Constraint(rule=lambda model : sum(model.bus[buses].p) == p_clustered)
        setattr(model, c_name, constraint)
    return extra_functionality

