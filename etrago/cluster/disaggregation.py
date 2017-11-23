import networkx as nx
import pandas as pd
from pypsa import Network
import numpy as np
import time
import cProfile

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
        partial_network, externals = construct_partial_network(original_network, network, clustering, cluster, buses)
        print('Decomposed in ', (time.time()-t))
        t = time.time()
        profile.disable()
        profile.enable()

        def extras(model, snapshot):
            pass

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
    return bus[bus.index.isin(buses)]


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

    bus_types = ['buses', 'loads', 'generators', 'stores', 'storage_units',
                 'generators', 'shunt_impedances']

    for bustype in bus_types:
        setattr(partial_network, bustype, getattr(partial_network, bustype).append(filter_buses(getattr(original_network, bustype),buses_in_lines)))

        buses_to_insert = filter_buses(getattr(clustered_network, bustype), map(lambda x: x[0][len(idx_prefix):], external_buses.values))
        setattr(partial_network, bustype, getattr(partial_network, bustype).append(buses_to_insert.reindex(idx_prefix + buses_to_insert.index)))

        setattr(partial_network,
                bustype+'_t',
                getattr(original_network, bustype+'_t'))

    for line_type in line_types:
        assert (getattr(partial_network, line_type).bus0.isin(partial_network.buses.index).all())
        assert (getattr(partial_network, line_type).bus1.isin(partial_network.buses.index).all())

    return partial_network, external_buses

def update_constraints(network, externals):
    pass