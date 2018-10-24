# -*- coding: utf-8 -*-
# Copyright 2016-2018  Flensburg University of Applied Sciences,
# Europa-Universität Flensburg,
# Centre for Sustainable Energy Systems,
# DLR-Institute for Networked Energy Systems

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# File description for read-the-docs
""" Networkclustering.py defines the methods to cluster power grid networks
spatially for applications within the tool eTraGo."""

import os
if 'READTHEDOCS' not in os.environ:
    from etrago.tools.utilities import *
    from pypsa.networkclustering import (aggregatebuses, aggregateoneport,
                                         aggregategenerators,
                                         get_clustering_from_busmap,
                                         busmap_by_kmeans, busmap_by_stubs)
    from egoio.db_tables.model_draft import EgoGridPfHvBusmap

    from itertools import product
    import networkx as nx
    import multiprocessing as mp
    from math import ceil
    import pandas as pd
    from networkx import NetworkXNoPath
    from pickle import dump
    from pypsa import Network
    import pypsa.io as io
    import pypsa.components as components
    from six import iteritems
    from sqlalchemy import or_, exists

__copyright__ = ("Flensburg University of Applied Sciences, "
                 "Europa-Universität Flensburg, "
                 "Centre for Sustainable Energy Systems, "
                 "DLR-Institute for Networked Energy Systems")
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "s3pp, wolfbunke, ulfmueller, lukasol"

# TODO: Workaround because of agg


def _leading(busmap, df):
    """
    """
    def leader(x):
        ix = busmap[x.index[0]]
        return df.loc[ix, x.name]
    return leader


def cluster_on_extra_high_voltage(network, busmap, with_time=True):
    """ Main function of the EHV-Clustering approach. Creates a new clustered
    pypsa.Network given a busmap mapping all bus_ids to other bus_ids of the
    same network.

    Parameters
    ----------
    network : pypsa.Network
        Container for all network components.

    busmap : dict
        Maps old bus_ids to new bus_ids.

    with_time : bool
        If true time-varying data will also be aggregated.

    Returns
    -------
    network : pypsa.Network
        Container for all network components of the clustered network.
    """

    network_c = Network()

    buses = aggregatebuses(
        network, busmap, {
            'x': _leading(
                busmap, network.buses), 'y': _leading(
                busmap, network.buses)})

    # keep attached lines
    lines = network.lines.copy()
    mask = lines.bus0.isin(buses.index)
    lines = lines.loc[mask, :]

    # keep attached links
    links = network.links.copy()
    mask = links.bus0.isin(buses.index)
    links = links.loc[mask, :]

    # keep attached transformer
    transformers = network.transformers.copy()
    mask = transformers.bus0.isin(buses.index)
    transformers = transformers.loc[mask, :]

    io.import_components_from_dataframe(network_c, buses, "Bus")
    io.import_components_from_dataframe(network_c, lines, "Line")
    io.import_components_from_dataframe(network_c, links, "Link")
    io.import_components_from_dataframe(network_c, transformers, "Transformer")

    if with_time:
        network_c.snapshots = network.snapshots
        network_c.set_snapshots(network.snapshots)

    # dealing with generators
    network.generators.control = "PV"
    network.generators['weight'] = 1
    new_df, new_pnl = aggregategenerators(network, busmap, with_time)
    io.import_components_from_dataframe(network_c, new_df, 'Generator')
    for attr, df in iteritems(new_pnl):
        io.import_series_from_dataframe(network_c, df, 'Generator', attr)

    # dealing with all other components
    aggregate_one_ports = components.one_port_components.copy()
    aggregate_one_ports.discard('Generator')

    for one_port in aggregate_one_ports:
        new_df, new_pnl = aggregateoneport(
            network, busmap, component=one_port, with_time=with_time)
        io.import_components_from_dataframe(network_c, new_df, one_port)
        for attr, df in iteritems(new_pnl):
            io.import_series_from_dataframe(network_c, df, one_port, attr)

    network_c.determine_network_topology()

    return network_c


def graph_from_edges(edges):
    """ Constructs an undirected multigraph from a list containing data on
    weighted edges.

    Parameters
    ----------
    edges : list
        List of tuples each containing first node, second node, weight, key.

    Returns
    -------
    M : :class:`networkx.classes.multigraph.MultiGraph
    """

    M = nx.MultiGraph()

    for e in edges:

        n0, n1, weight, key = e

        M.add_edge(n0, n1, weight=weight, key=key)

    return M


def gen(nodes, n, graph):
    # TODO There could be a more convenient way of doing this. This generators
    # single purpose is to prepare data for multiprocessing's starmap function.
    """ Generator for applying multiprocessing.

    Parameters
    ----------
    nodes : list
        List of nodes in the system.

    n : int
        Number of desired multiprocessing units.

    graph : :class:`networkx.classes.multigraph.MultiGraph
        Graph representation of an electrical grid.

    Returns
    -------
    None
    """

    g = graph.copy()

    for i in range(0, len(nodes), n):
        yield (nodes[i:i + n], g)


def shortest_path(paths, graph):
    """ Finds the minimum path lengths between node pairs defined in paths.

    Parameters
    ----------
    paths : list
        List of pairs containing a source and a target node

    graph : :class:`networkx.classes.multigraph.MultiGraph
        Graph representation of an electrical grid.

    Returns
    -------
    df : pd.DataFrame
        DataFrame holding source and target node and the minimum path length.
    """

    idxnames = ['source', 'target']
    idx = pd.MultiIndex.from_tuples(paths, names=idxnames)
    df = pd.DataFrame(index=idx, columns=['path_length'])
    df.sort_index(inplace=True)

    for s, t in paths:

        try:
            df.loc[(s, t), 'path_length'] = \
                nx.dijkstra_path_length(graph, s, t)

        except NetworkXNoPath:
            continue

    return df


def busmap_by_shortest_path(network, session, scn_name, fromlvl, tolvl,
                            cpu_cores=4):
    """ Creates a busmap for the EHV-Clustering between voltage levels based
    on dijkstra shortest path. The result is automatically written to the
    `model_draft` on the <OpenEnergyPlatform>[www.openenergy-platform.org]
    database with the name `ego_grid_pf_hv_busmap` and the attributes scn_name
    (scenario name), bus0 (node before clustering), bus1 (node after
    clustering) and path_length (path length).
    An AssertionError occurs if buses with a voltage level are not covered by
    the input lists 'fromlvl' or 'tolvl'.

    Parameters
    ----------
    network : pypsa.Network object
        Container for all network components.

    session : sqlalchemy.orm.session.Session object
        Establishes interactions with the database.

    scn_name : str
        Name of the scenario.

    fromlvl : list
        List of voltage-levels to cluster.

    tolvl : list
        List of voltage-levels to remain.

    cpu_cores : int
        Number of CPU-cores.

    Returns
    -------
    None
    """

    # cpu_cores = mp.cpu_count()

    # data preperation
    s_buses = buses_grid_linked(network, fromlvl)
    lines = connected_grid_lines(network, s_buses)
    transformer = connected_transformer(network, s_buses)
    mask = transformer.bus1.isin(buses_of_vlvl(network, tolvl))

    # temporary end points, later replaced by bus1 pendant
    t_buses = transformer[mask].bus0

    # create all possible pathways
    ppaths = list(product(s_buses, t_buses))

    # graph creation
    edges = [(row.bus0, row.bus1, row.length, ix) for ix, row
             in lines.iterrows()]
    M = graph_from_edges(edges)

    # applying multiprocessing
    p = mp.Pool(cpu_cores)
    chunksize = ceil(len(ppaths) / cpu_cores)
    container = p.starmap(shortest_path, gen(ppaths, chunksize, M))
    df = pd.concat(container)
    dump(df, open('df.p', 'wb'))

    # post processing
    df.sortlevel(inplace=True)
    mask = df.groupby(level='source')['path_length'].idxmin()
    df = df.loc[mask, :]

    # rename temporary endpoints
    df.reset_index(inplace=True)
    df.target = df.target.map(dict(zip(network.transformers.bus0,
                                       network.transformers.bus1)))

    # append to busmap buses only connected to transformer
    transformer = network.transformers
    idx = list(set(buses_of_vlvl(network, fromlvl)).
               symmetric_difference(set(s_buses)))
    mask = transformer.bus0.isin(idx)

    toappend = pd.DataFrame(list(zip(transformer[mask].bus0,
                                     transformer[mask].bus1)),
                            columns=['source', 'target'])
    toappend['path_length'] = 0

    df = pd.concat([df, toappend], ignore_index=True, axis=0)

    # append all other buses
    buses = network.buses
    mask = buses.index.isin(df.source)

    assert set(buses[~mask].v_nom) == set(tolvl)

    tofill = pd.DataFrame([buses.index[~mask]] * 2).transpose()
    tofill.columns = ['source', 'target']
    tofill['path_length'] = 0

    df = pd.concat([df, tofill], ignore_index=True, axis=0)

    # prepare data for export

    print(scn_name)
    df['scn_name'] = scn_name
    print(df)

    df.rename(columns={'source': 'bus0', 'target': 'bus1'}, inplace=True)
    df.set_index(['scn_name', 'bus0', 'bus1'], inplace=True)

    for i, d in df.reset_index().iterrows():
        session.add(EgoGridPfHvBusmap(**d.to_dict()))

    session.commit()

    return


def busmap_from_psql(network, session, scn_name):
    """ Retrieves busmap from `model_draft.ego_grid_pf_hv_busmap` on the
    <OpenEnergyPlatform>[www.openenergy-platform.org] by a given scenario
    name. If this busmap does not exist, it is created with default values.

    Parameters
    ----------
    network : pypsa.Network object
        Container for all network components.

    session : sqlalchemy.orm.session.Session object
        Establishes interactions with the database.

    scn_name : str
        Name of the scenario.

    Returns
    -------
    busmap : dict
        Maps old bus_ids to new bus_ids.
    """

    def fetch():

        query = session.query(EgoGridPfHvBusmap.bus0, EgoGridPfHvBusmap.bus1).\
            filter(EgoGridPfHvBusmap.scn_name == scn_name)

        return dict(query.all())

    busmap = fetch()

    # TODO: Or better try/except/finally
    if not busmap:
        print('Busmap does not exist and will be created.\n')

        cpu_cores = input('cpu_cores (default 4): ') or '4'

        busmap_by_shortest_path(network, session, scn_name,
                                fromlvl=[110], tolvl=[220, 380],
                                cpu_cores=int(cpu_cores))
        busmap = fetch()

    return busmap


def kmean_clustering(network, n_clusters=10, load_cluster=False,
                     line_length_factor=1.25,
                     remove_stubs=False, use_reduced_coordinates=False,
                     bus_weight_tocsv=None, bus_weight_fromcsv=None,
                     n_init=10, max_iter=300, tol=1e-4,
                     n_jobs=1):
    """ Main function of the k-mean clustering approach. Maps an original
    network to a new one with adjustable number of nodes and new coordinates.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Container for all network components.

    n_clusters : int
        Desired number of clusters.

    load_cluster : boolean
        Loads cluster coordinates from a former calculation.

    line_length_factor : float
        Factor to multiply the crow-flies distance between new buses in order
        to get new line lengths.

    remove_stubs: boolean
        Removes stubs and stubby trees (i.e. sequentially reducing dead-ends).

    use_reduced_coordinates: boolean
        If True, do not average cluster coordinates, but take from busmap.

    bus_weight_tocsv : str
        Creates a bus weighting based on conventional generation and load
        and save it to a csv file.

    bus_weight_fromcsv : str
        Loads a bus weighting from a csv file to apply it to the clustering
        algorithm.

    Returns
    -------
    network : pypsa.Network object
        Container for all network components.
    """
    def weighting_for_scenario(x, save=None):
        """
        """
        b_i = x.index
        g = normed(gen.reindex(b_i, fill_value=0))
        l = normed(load.reindex(b_i, fill_value=0))

        w = g + l
        weight = ((w * (100000. / w.max())).astype(int)
                  ).reindex(network.buses.index, fill_value=1)

        if save:
            weight.to_csv(save)

        return weight

    def normed(x):
        return (x / x.sum()).fillna(0.)

    print('start k-mean clustering')
    # prepare k-mean
    # k-means clustering (first try)
    network.generators.control = "PV"
    network.storage_units.control[network.storage_units.carrier == \
                                  'extendable_storage'] = "PV"
    network.buses['v_nom'] = 380.
    # problem our lines have no v_nom. this is implicitly defined by the
    # connected buses:
    network.lines["v_nom"] = network.lines.bus0.map(network.buses.v_nom)

    # adjust the x of the lines which are not 380.
    lines_v_nom_b = network.lines.v_nom != 380
    network.lines.loc[lines_v_nom_b, 'x'] *= \
        (380. / network.lines.loc[lines_v_nom_b, 'v_nom'])**2
    network.lines.loc[lines_v_nom_b, 'v_nom'] = 380.

    trafo_index = network.transformers.index
    transformer_voltages = \
        pd.concat([network.transformers.bus0.map(network.buses.v_nom),
                   network.transformers.bus1.map(network.buses.v_nom)], axis=1)

    network.import_components_from_dataframe(
        network.transformers.loc[:, [
                'bus0', 'bus1', 'x', 's_nom', 'capital_cost']]
        .assign(x=network.transformers.x * (380. /
                transformer_voltages.max(axis=1))**2, length = 1)
        .set_index('T' + trafo_index),
        'Line')
    network.transformers.drop(trafo_index, inplace=True)

    for attr in network.transformers_t:
        network.transformers_t[attr] = network.transformers_t[attr]\
            .reindex(columns=[])

    # remove stubs
    if remove_stubs:
        network.determine_network_topology()
        busmap = busmap_by_stubs(network)
        network.generators['weight'] = network.generators['p_nom']
        aggregate_one_ports = components.one_port_components.copy()
        aggregate_one_ports.discard('Generator')
        # reset coordinates to the new reduced guys, rather than taking an
        # average (copied from pypsa.networkclustering)
        if use_reduced_coordinates:
            # TODO : FIX THIS HACK THAT HAS UNEXPECTED SIDE-EFFECTS,
            # i.e. network is changed in place!!
            network.buses.loc[busmap.index, ['x', 'y']
                              ] = network.buses.loc[busmap, ['x', 'y']].values

        clustering = get_clustering_from_busmap(
            network,
            busmap,
            aggregate_generators_weighted=True,
            aggregate_one_ports=aggregate_one_ports,
            line_length_factor=line_length_factor)
        network = clustering.network

    # define weighting based on conventional 'old' generator spatial
    # distribution
    non_conv_types = {
        'biomass',
        'wind_onshore',
        'wind_offshore',
        'solar',
        'geothermal',
        'load shedding',
        'extendable_storage'}
    # Attention: network.generators.carrier.unique()
    gen = (network.generators.loc[(network.generators.carrier
                                   .isin(non_conv_types) == False)]
           .groupby('bus').p_nom.sum()
                                .reindex(network.buses.index, fill_value=0.) +
           network.storage_units
                                .loc[(network.storage_units.carrier
                                      .isin(non_conv_types) == False)]
                  .groupby('bus').p_nom.sum()
                  .reindex(network.buses.index, fill_value=0.))

    load = network.loads_t.p_set.mean().groupby(network.loads.bus).sum()

    # k-mean clustering

    # busmap = busmap_by_kmeans(network, bus_weightings=pd.Series(np.repeat(1,
    #       len(network.buses)), index=network.buses.index) , n_clusters= 10)
    # State whether to create a bus weighting and save it, create or not save
    # it, or use a bus weighting from a csv file
    if bus_weight_tocsv is not None:
        weight = weighting_for_scenario(x=network.buses, save=bus_weight_tocsv)
    elif bus_weight_fromcsv is not None:
        weight = pd.Series.from_csv(bus_weight_fromcsv)
        weight.index = weight.index.astype(str)
    else:
        weight = weighting_for_scenario(x=network.buses, save=False)

    busmap = busmap_by_kmeans(
        network,
        bus_weightings=pd.Series(weight),
        n_clusters=n_clusters,
        load_cluster=load_cluster,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        n_jobs=n_jobs)

    # ToDo change function in order to use bus_strategies or similar
    network.generators['weight'] = network.generators['p_nom']
    aggregate_one_ports = components.one_port_components.copy()
    aggregate_one_ports.discard('Generator')
    clustering = get_clustering_from_busmap(
        network,
        busmap,
        aggregate_generators_weighted=True,
        aggregate_one_ports=aggregate_one_ports)

    return clustering
