import networkx as nx
import pandas as pd
from pypsa import Network
import numpy as np
import time
import cProfile

def swap_series(s):
    return pd.Series(s.index.values, index=s)

def disaggregate(scenario, original_network, network, clustering, solver=None, extras=None):
    build_cluster_maps(scenario, original_network, network, clustering, solver, extras)

def build_cluster_maps(scenario, original_network, network, clustering, solver, extras):
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
        partial_network = construct_partial_network(original_network, cluster, buses)
        print('Decomposed in ', (time.time()-t))
        t = time.time()
        profile.disable()
        profile.enable()
        partial_network.lopf(scenario.timeindex, solver_name=solver,
                             extra_functionality=extras)
        profile.disable()

        print('Decomposition optimized in ', (time.time() - t))

    profile.print_stats(sort='cumtime')

def construct_partial_network(original_network, cluster, buses):
    partial_network = Network()

    # find all lines that have at least one bus inside the cluster
    busflags = (buses['cluster'] == cluster)

    def is_bus_in_cluster(conn):
        return busflags[conn]

    def filter_connector(conn):
        return conn[conn.bus0.apply(is_bus_in_cluster) | conn.bus1.apply(is_bus_in_cluster)]

    # Copy configurations to new network
    partial_network.snapshots = original_network.snapshots
    partial_network.snapshot_weightings = original_network.snapshot_weightings
    partial_network.carriers = original_network.carriers

    # Collect all connectors that have some node inside the cluster
    partial_network.lines = filter_connector(original_network.lines)
    partial_network.lines_t = original_network.lines_t
    partial_network.links = filter_connector(original_network.links)
    partial_network.links_t = original_network.links_t
    partial_network.transformers = filter_connector(original_network.transformers)
    partial_network.transformers_t = original_network.transformers_t

    # Collect all buses that are contained in or somehow connected to the
    # cluster
    buses_in_lines = np.unique(np.concatenate((partial_network.lines.bus0.values,
                                     partial_network.lines.bus1.values,
                                     partial_network.links.bus0.values,
                                     partial_network.links.bus1.values,
                                     partial_network.transformers.bus0.values,
                                     partial_network.transformers.bus1.values,
                                     buses[busflags].index)))

    def filter_buses(bus):
        return bus[bus.index.isin(buses_in_lines)]

    def filter_on_buses(connecitve):
        return connecitve[connecitve.bus.isin(buses_in_lines)]

    partial_network.buses = filter_buses(original_network.buses)
    partial_network.buses_t = original_network.buses_t
    partial_network.loads = filter_on_buses(original_network.loads)
    partial_network.loads_t = original_network.loads_t
    partial_network.generators = filter_on_buses(original_network.generators)
    partial_network.generators_t = original_network.generators_t
    partial_network.storage_units = filter_on_buses(original_network.storage_units)
    partial_network.storage_units_t = original_network.storage_units_t
    partial_network.stores = filter_on_buses(original_network.stores)
    partial_network.stores_t = original_network.stores_t
    partial_network.generators = filter_on_buses(original_network.generators)
    partial_network.generators_t = original_network.generators_t
    partial_network.shunt_impedances = filter_on_buses(original_network.shunt_impedances)
    partial_network.shunt_impedances_t = original_network.shunt_impedances_t

    return partial_network
