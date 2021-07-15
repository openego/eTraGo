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
    import numpy as np
    import logging
    from sklearn.cluster import KMeans

    logger = logging.getLogger(__name__)

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
        network_c.snapshot_weightings = network.snapshot_weightings.copy()

    # dealing with generators
    network.generators.control = "PV"
    network.generators['weight'] = 1
    new_df, new_pnl = aggregategenerators(network, busmap, with_time,
                    custom_strategies={'p_nom_min':np.min,'p_nom_max': np.min,
                                       'weight': np.sum, 'p_nom': np.sum,
                                       'p_nom_opt': np.sum, 'marginal_cost':
                                           np.mean, 'capital_cost': np.mean})
    io.import_components_from_dataframe(network_c, new_df, 'Generator')
    for attr, df in iteritems(new_pnl):
        io.import_series_from_dataframe(network_c, df, 'Generator', attr)

    # dealing with all other components
    aggregate_one_ports = network.one_port_components.copy()
    aggregate_one_ports.discard('Generator')

    for one_port in aggregate_one_ports:
        one_port_strategies = {'StorageUnit': {'marginal_cost': np.mean, 'capital_cost': np.mean, 'efficiency': np.mean,
                             'efficiency_dispatch': np.mean, 'standing_loss': np.mean, 'efficiency_store': np.mean,
                             'p_min_pu': np.min}}
        new_df, new_pnl = aggregateoneport(
            network, busmap, component=one_port, with_time=with_time,
            custom_strategies=one_port_strategies.get(one_port, {}))
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

    '''
    for s, t in paths:

        try:
            df.loc[(s, t), 'path_length'] = \
                nx.dijkstra_path_length(graph, s, t)

        except NetworkXNoPath:
            continue
    '''
        
    df_isna = df.isnull()
    for s, t in paths:
        while (df_isna.loc[(s, t), 'path_length'] == True):  
            try:
                s_to_other = nx.single_source_dijkstra_path_length(graph, s) 
                for t in idx.levels[1]: 
                    if t in s_to_other:
                        df.loc[(s, t), 'path_length'] = s_to_other[t] 
                    else:
                        df.loc[(s,t),'path_length'] = np.inf
            except NetworkXNoPath:
                continue
            df_isna = df.isnull()

    return df


def busmap_by_shortest_path(etrago, scn_name, fromlvl, tolvl, cpu_cores=4):
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
    s_buses = buses_grid_linked(etrago.network, fromlvl)
    lines = connected_grid_lines(etrago.network, s_buses)
    transformer = connected_transformer(network, s_buses)
    mask = transformer.bus1.isin(buses_of_vlvl(etrago.network, tolvl))

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
    df.target = df.target.map(dict(zip(etrago.network.transformers.bus0,
                                       etrago.network.transformers.bus1)))

    # append to busmap buses only connected to transformer
    transformer = etrago.network.transformers
    idx = list(set(buses_of_vlvl(network, fromlvl)).
               symmetric_difference(set(s_buses)))
    mask = transformer.bus0.isin(idx)

    toappend = pd.DataFrame(list(zip(transformer[mask].bus0,
                                     transformer[mask].bus1)),
                            columns=['source', 'target'])
    toappend['path_length'] = 0

    df = pd.concat([df, toappend], ignore_index=True, axis=0)

    # append all other buses
    buses = etrago.network.buses
    mask = buses.index.isin(df.source)

    assert set(buses[~mask].v_nom) == set(tolvl)

    tofill = pd.DataFrame([buses.index[~mask]] * 2).transpose()
    tofill.columns = ['source', 'target']
    tofill['path_length'] = 0

    df = pd.concat([df, tofill], ignore_index=True, axis=0)

    # prepare data for export

    df['scn_name'] = scn_name
    df['version'] = etrago.args['gridversion']

    df.rename(columns={'source': 'bus0', 'target': 'bus1'}, inplace=True)
    df.set_index(['scn_name', 'bus0', 'bus1'], inplace=True)

    for i, d in df.reset_index().iterrows():
        etrago.session.add(EgoGridPfHvBusmap(**d.to_dict()))

    etrago.session.commit()

    return


def busmap_from_psql(etrago):
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
    scn_name=(etrago.args['scn_name'] if etrago.args['scn_extension']==None
                        else etrago.args['scn_name']+'_ext_'+'_'.join(
                                etrago.args['scn_extension']))
    def fetch():

        query = etrago.session.query(
            EgoGridPfHvBusmap.bus0, EgoGridPfHvBusmap.bus1).\
            filter(EgoGridPfHvBusmap.scn_name == scn_name).\
            filter(EgoGridPfHvBusmap.version == etrago.args['gridversion'])

        return dict(query.all())

    busmap = fetch()

    # TODO: Or better try/except/finally
    if not busmap:
        print('Busmap does not exist and will be created.\n')

        cpu_cores = input('cpu_cores (default 4): ') or '4'

        busmap_by_shortest_path(etrago, scn_name,
                                fromlvl=[110], tolvl=[220, 380, 400, 450],
                                cpu_cores=int(cpu_cores))
        busmap = fetch()

    return busmap

def ehv_clustering(self):


    if self.args['network_clustering_ehv']:

        logger.info('Start ehv clustering')

        self.network.generators.control = "PV"
        busmap = busmap_from_psql(self)
        self.network = cluster_on_extra_high_voltage(
            self.network, busmap, with_time=True)

        logger.info('Network clustered to EHV-grid')


def kmean_clustering(etrago):
    """ Function of the k-mean clustering approach. Maps an original
    network to a new one with adjustable number of nodes and new coordinates.
    This approach considers the geographical distribution of the network's buses
    to assign them to clusters.

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


    network = etrago.network
    kmean_settings = etrago.args['network_clustering']
    def weighting_for_scenario(x, save=None):
        """
        """
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

    # prepare k-mean
    # k-means clustering (first try)
    network.generators.control = "PV"
    network.storage_units.control[network.storage_units.carrier == \
                                  'extendable_storage'] = "PV"

    # problem our lines have no v_nom. this is implicitly defined by the
    # connected buses:
    network.lines["v_nom"] = network.lines.bus0.map(network.buses.v_nom)

    # adjust the electrical parameters of the lines which are not 380.
    lines_v_nom_b = network.lines.v_nom != 380

    voltage_factor = (network.lines.loc[lines_v_nom_b, 'v_nom'] / 380.)**2

    network.lines.loc[lines_v_nom_b, 'x'] *= 1/voltage_factor

    network.lines.loc[lines_v_nom_b, 'r'] *= 1/voltage_factor

    network.lines.loc[lines_v_nom_b, 'b'] *= voltage_factor

    network.lines.loc[lines_v_nom_b, 'g'] *= voltage_factor

    network.lines.loc[lines_v_nom_b, 'v_nom'] = 380.

    trafo_index = network.transformers.index
    transformer_voltages = \
        pd.concat([network.transformers.bus0.map(network.buses.v_nom),
                   network.transformers.bus1.map(network.buses.v_nom)], axis=1)

    network.import_components_from_dataframe(
        network.transformers.loc[:, [
                'bus0', 'bus1', 'x', 's_nom', 'capital_cost', 'sub_network', 's_max_pu']]
        .assign(x=network.transformers.x * (380. /
                transformer_voltages.max(axis=1))**2, length = 1)
        .set_index('T' + trafo_index),
        'Line')
    network.transformers.drop(trafo_index, inplace=True)

    for attr in network.transformers_t:
        network.transformers_t[attr] = network.transformers_t[attr]\
            .reindex(columns=[])

    network.buses['v_nom'] = 380.

    # State whether to create a bus weighting and save it, create or not save
    # it, or use a bus weighting from a csv file
    if kmean_settings['bus_weight_tocsv'] is not None:
        weight = weighting_for_scenario(
            x=network.buses,
            save=kmean_settings['bus_weight_tocsv'])
    elif kmean_settings['bus_weight_fromcsv'] is not None:
        weight = pd.Series.from_csv(kmean_settings['bus_weight_fromcsv'])
        weight.index = weight.index.astype(str)
    else:
        weight = weighting_for_scenario(x=network.buses, save=False)


    # remove stubs
    if kmean_settings['remove_stubs']:
        network.determine_network_topology()
        busmap = busmap_by_stubs(network)
        network.generators['weight'] = network.generators['p_nom']
        aggregate_one_ports = network.one_port_components.copy()
        aggregate_one_ports.discard('Generator')

        # reset coordinates to the new reduced guys, rather than taking an
        # average (copied from pypsa.networkclustering)
        if kmean_settings['use_reduced_coordinates']:
            # TODO : FIX THIS HACK THAT HAS UNEXPECTED SIDE-EFFECTS,
            # i.e. network is changed in place!!
            network.buses.loc[busmap.index, ['x', 'y']
                              ] = network.buses.loc[busmap, ['x', 'y']].values

        clustering = get_clustering_from_busmap(
            network,
            busmap,
            aggregate_generators_weighted=True,
            one_port_strategies={'StorageUnit': {'marginal_cost': np.mean,
                                             'capital_cost': np.mean,
                                             'efficiency': np.mean,
                                             'efficiency_dispatch': np.mean,
                                             'standing_loss': np.mean,
                                             'efficiency_store': np.mean,
                                             'p_min_pu': np.min}},
            generator_strategies={'p_nom_min':np.min,
                              'p_nom_opt': np.sum,
                              'marginal_cost': np.mean,
                              'capital_cost': np.mean},
            aggregate_one_ports=aggregate_one_ports,
            line_length_factor=kmean_settings['line_length_factor'])
        network = clustering.network

        weight = weight.groupby(busmap.values).sum()

    # k-mean clustering
    if not kmean_settings['busmap']:
        busmap = busmap_by_kmeans(
            network,
            bus_weightings=pd.Series(weight),
            n_clusters=kmean_settings['n_clusters'],
            n_init=kmean_settings['n_init'],
            max_iter=kmean_settings['max_iter'],
            tol=kmean_settings['tol'],
            n_jobs=kmean_settings['n_jobs'])
        busmap.to_csv('kmeans_busmap_' + str(kmean_settings['n_clusters']) + '_result.csv')
    else:
        df = pd.read_csv(kmean_settings['busmap'])
        df=df.astype(str)
        df = df.set_index('bus_id')
        busmap = df.squeeze('columns')

    network.generators['weight'] = network.generators['p_nom']
    aggregate_one_ports = network.one_port_components.copy()
    aggregate_one_ports.discard('Generator')
    clustering = get_clustering_from_busmap(
        network,
        busmap,
        aggregate_generators_weighted=True,
        one_port_strategies={'StorageUnit': {'marginal_cost': np.mean,
                                             'capital_cost': np.mean,
                                             'efficiency': np.mean,
                                             'efficiency_dispatch': np.mean,
                                             'standing_loss': np.mean,
                                             'efficiency_store': np.mean,
                                             'p_min_pu': np.min}},
        generator_strategies={'p_nom_min':np.min,
                              'p_nom_opt': np.sum,
                              'marginal_cost': np.mean,
                              'capital_cost': np.mean},
        aggregate_one_ports=aggregate_one_ports,
        line_length_factor=kmean_settings['line_length_factor'])

    return clustering

def dijkstras_algorithm(network, medoid_idx, busmap_kmedoid):
    """ Function for combination of k-medoids clustering and Dijkstra's algorithm.
    Creates a busmap assigning the nodes of a original network 
    to the nodes of a clustered network 
    considering the electrical distance based on Dijkstra's shortest path. 
    Parameters
    ----------
    network : pypsa.Network object
        Container for all network components.
        
    medoid_idx : pd.Series
        Indices of k-medoids centers 
    
    busmap_kmedoid: pd.Series
        Busmap based on k-medoids clustering
    Returns
    -------
    busmap (format: with labels)
    """
    
    # original data
    o_buses = network.buses.index
    # k-medoids centers
    medoid_idx=medoid_idx.astype('str')
    c_buses = medoid_idx.tolist()
    
    # k-medoids assignment 
    df_kmedoid=pd.DataFrame({'medoid_labels':busmap_kmedoid.values}, index=busmap_kmedoid.index)
    df_kmedoid['medoid_indices']=df_kmedoid['medoid_labels']
    for index, row in df_kmedoid.iterrows():
        label = int(row['medoid_labels'])
        df_kmedoid['medoid_indices'].loc[index] = c_buses[label]
             
    # list of all possible pathways
    ppathss = list(product(o_buses, c_buses))
    
    # graph creation
    lines = network.lines
    edges = [(row.bus0, row.bus1, row.length, ix) for ix, row
             in lines.iterrows()]
    M = graph_from_edges(edges)
    
    # processor count
    cpu_cores = mp.cpu_count()-1

    # calculation of shortest path between original points and k-medoids centers
    # using multiprocessing
    p = mp.Pool(cpu_cores)
    chunksize = ceil(len(ppathss) / cpu_cores)
    container = p.starmap(shortest_path, gen(ppathss, chunksize, M))
    df = pd.concat(container)
    dump(df, open('df.p', 'wb'))
     
    # assignment of data points to closest k-medoids centers
    df['path_length']=pd.to_numeric(df['path_length'])    
    mask = df.groupby(level='source')['path_length'].idxmin()
    df_dijkstra = df.loc[mask, :]
    df_dijkstra.reset_index(inplace=True)

    # delete double entries in df due to multiprocessing      
    duplicated=df_dijkstra.duplicated()
    for i in range(len(duplicated)):
        if duplicated[i]==True:
            df_dijkstra = df_dijkstra.drop([i])
    df_dijkstra.index=df_dijkstra['source']
    
    ###############################Auswertung##################################
    
    df_dijkstra['correction of assignment']=df_dijkstra['target']
    for i in range(len(df_dijkstra)):
        index=df_dijkstra['source'].iloc[i]
        if (int(df_kmedoid['medoid_indices'].loc[index]) != int(df_dijkstra['target'].iloc[i])):
            df_dijkstra['correction of assignment'].iloc[i]='True'
        else:
            df_dijkstra['correction of assignment'].iloc[i]='False'
    dfj = df_dijkstra.copy()
    dfj['kmedoid'] = df_kmedoid['medoid_indices']
    original = pd.Series(index=medoid_idx.values) # Anzahl der Originalknoten pro Cluster
    for i in medoid_idx:
        dfo = dfj[dfj['kmedoid']==i]
        original.loc[i] = len(dfo)
    dfj = dfj[dfj['correction of assignment']=='True']   
    
    print(' ')
    print("Anzahl der veränderten Zuordnungen durch Dijkstra's Algorithmus: "+str(len(dfj))) 
    pro = (len(dfj) / len(network.buses))*100
    print('Anteil der veränderten Knoten an den Originalknoten in Prozent: '+str(pro))
    print(' ')
    prozent = pd.Series(index=medoid_idx.values)
    anzahl = pd.Series(index=medoid_idx.values)
    x=0
    for i in medoid_idx:
        dfjj=dfj[dfj['kmedoid']==i]
        if len(dfjj)>0:
            x=x+1
            anzahl.loc[i] = len(dfjj)
            print('Änderung des durch k-means Clustering festgelegten Clusters (Label) '+str(medoid_idx[medoid_idx==str(i)].index[0])+': '+str(len(dfjj))+' mal')
            print('Anzahl der Originalknoten in diesem Cluster: '+str(original.loc[i]))
            prozent.loc[i] = (len(dfjj) / original.loc[i]) *100
            print('Anteil der veränderten Buses an den Originalbusses im Cluster: '+str(prozent.loc[i]))
    print(' ')
    print('Anzahl der durch Änderungen betroffene Cluster: '+str(x)+' (von) '+str(len(medoid_idx))+' Clustern insgesamt')
    print('Knotenanzahl in Durchschnittscluster: '+str(original.mean()))
    print('Mittlere Anzahl der veränderten Knoten pro Cluster (nur veränderte Cluster): '+str((anzahl.mean())))
    print('Durchschnittlicher Anteil der veränderten Knoten pro Cluster (nur veränderte Cluster) in Prozent: '+str((prozent.mean())))
    m = len(dfj) / len(c_buses)
    print("Mittlere Anzahl der veränderten Zuordnung pro Cluster (alle Cluster): "+str(m)) 
    prozent.fillna(0,inplace=True)
    print('Durchschnittlicher Anteil der veränderten Knoten pro Cluster (alle Cluster) in Prozent: '+str((prozent.mean())))

    print(' ')
    print('mittlere Pfadlänge der Originalknoten pro Cluster zu deren Medoids:')
    # dijkstra:
    df_dijkstra['path_length']=df_dijkstra['path_length'].astype(float)
    df_clusterpaths_dijkstra=df_dijkstra.groupby(df_dijkstra['target'])
    df_clusterpaths_dijkstra=df_clusterpaths_dijkstra['path_length'].aggregate(np.mean)
    #df_clusterpaths_dijkstra.to_csv('cluster_paths_dijkstra',index=True)
    print("Dijkstra's Algorithmus: "+str(df_clusterpaths_dijkstra.mean()))
    # kmean / kmedoid:
    df_kmedoid['path_length']=pd.Series(data=np.zeros)
    for i in range (0,len(df_kmedoid)):
        source=df_kmedoid.index[i]
        target=df_kmedoid['medoid_indices'].iloc[i]
        df_kmedoid['path_length'].iloc[i]=df['path_length'].loc[str(source)].loc[df.loc[str(source)].index==str(target)].values[0]
    df_kmedoid['path_length']=df_kmedoid['path_length'].astype(float)
    df_clusterpaths_kmedoid=df_kmedoid.groupby(df_kmedoid['medoid_indices'])
    df_clusterpaths_kmedoid=df_clusterpaths_kmedoid['path_length'].aggregate(np.mean)
    #df_clusterpaths_kmedoid.to_csv('cluster_paths_kmedoid',index=True)
    print('k-medoids Clustering: '+str(df_clusterpaths_kmedoid.mean()))
    verhaltnis = df_clusterpaths_kmedoid.mean() / df_clusterpaths_dijkstra.mean()
    print('Verhältnis der mittleren Pfandlängen: '+str(verhaltnis))
        
    ###############################Auswertung##################################
    
    # creation of new busmap with final assignment (format: medoids indices)
    busmap_ind=pd.Series(df_dijkstra['target'], dtype=object).rename("final_assignment", inplace=True)
    busmap_ind.index=df_dijkstra['source']
            
    # adaption of busmap to format with labels (necessary for aggregation)
    busmap=busmap_ind.copy()
    for index, item in busmap.iteritems():
        label = medoid_idx[medoid_idx==str(item)].index[0]
        busmap.loc[index] = str(label)
    busmap.index=list(busmap.index.astype(str))
                   
    return busmap
        
def kmedoids_dijkstra_clustering(etrago):
    """ Function of the k-medoids dijkstra clustering approach. Maps an original
    network to a new one with adjustable number of nodes and new coordinates.
    This approach considers the electrical distances between the network's buses
    to assign them to clusters.

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


    network = etrago.network
    settings = etrago.args['network_clustering']
    def weighting_for_scenario(x, save=None):
        """
        """
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

    # prepare k-mean
    # k-means clustering (first try)
    network.generators.control = "PV"
    network.storage_units.control[network.storage_units.carrier == \
                                  'extendable_storage'] = "PV"

    # problem our lines have no v_nom. this is implicitly defined by the
    # connected buses:
    network.lines["v_nom"] = network.lines.bus0.map(network.buses.v_nom)

    # adjust the electrical parameters of the lines which are not 380.
    lines_v_nom_b = network.lines.v_nom != 380

    voltage_factor = (network.lines.loc[lines_v_nom_b, 'v_nom'] / 380.)**2

    network.lines.loc[lines_v_nom_b, 'x'] *= 1/voltage_factor

    network.lines.loc[lines_v_nom_b, 'r'] *= 1/voltage_factor

    network.lines.loc[lines_v_nom_b, 'b'] *= voltage_factor

    network.lines.loc[lines_v_nom_b, 'g'] *= voltage_factor

    network.lines.loc[lines_v_nom_b, 'v_nom'] = 380.

    trafo_index = network.transformers.index
    transformer_voltages = \
        pd.concat([network.transformers.bus0.map(network.buses.v_nom),
                   network.transformers.bus1.map(network.buses.v_nom)], axis=1)

    network.import_components_from_dataframe(
        network.transformers.loc[:, [
                'bus0', 'bus1', 'x', 's_nom', 'capital_cost', 'sub_network', 's_max_pu']]
        .assign(x=network.transformers.x * (380. /
                transformer_voltages.max(axis=1))**2, length = 1)
        .set_index('T' + trafo_index),
        'Line')
    network.transformers.drop(trafo_index, inplace=True)

    for attr in network.transformers_t:
        network.transformers_t[attr] = network.transformers_t[attr]\
            .reindex(columns=[])

    network.buses['v_nom'] = 380.

    # State whether to create a bus weighting and save it, create or not save
    # it, or use a bus weighting from a csv file
    if settings['bus_weight_tocsv'] is not None:
        weight = weighting_for_scenario(
            x=network.buses,
            save=settings['bus_weight_tocsv'])
    elif settings['bus_weight_fromcsv'] is not None:
        weight = pd.Series.from_csv(settings['bus_weight_fromcsv'])
        weight.index = weight.index.astype(str)
    else:
        weight = weighting_for_scenario(x=network.buses, save=False)


    # remove stubs
    if settings['remove_stubs']:
        network.determine_network_topology()
        busmap = busmap_by_stubs(network)
        network.generators['weight'] = network.generators['p_nom']
        aggregate_one_ports = network.one_port_components.copy()
        aggregate_one_ports.discard('Generator')

        # reset coordinates to the new reduced guys, rather than taking an
        # average (copied from pypsa.networkclustering)
        if settings['use_reduced_coordinates']:
            # TODO : FIX THIS HACK THAT HAS UNEXPECTED SIDE-EFFECTS,
            # i.e. network is changed in place!!
            network.buses.loc[busmap.index, ['x', 'y']
                              ] = network.buses.loc[busmap, ['x', 'y']].values

        clustering = get_clustering_from_busmap(
            network,
            busmap,
            aggregate_generators_weighted=True,
            one_port_strategies={'StorageUnit': {'marginal_cost': np.mean,
                                             'capital_cost': np.mean,
                                             'efficiency': np.mean,
                                             'efficiency_dispatch': np.mean,
                                             'standing_loss': np.mean,
                                             'efficiency_store': np.mean,
                                             'p_min_pu': np.min}},
            generator_strategies={'p_nom_min':np.min,
                              'p_nom_opt': np.sum,
                              'marginal_cost': np.mean,
                              'capital_cost': np.mean},
            aggregate_one_ports=aggregate_one_ports,
            line_length_factor=settings['line_length_factor'])
        network = clustering.network

        weight = weight.groupby(busmap.values).sum()

    # k-mean clustering
    if not settings['busmap']:
        
        bus_weightings=pd.Series(weight)
        buses_i=network.buses.index
        points = (network.buses.loc[buses_i, ["x","y"]].values
                  .repeat(bus_weightings.reindex(buses_i).astype(int), axis=0))
        
        # k-means clustering
        
        kmeans = KMeans(init='k-means++', n_clusters=settings['n_clusters'], \
                    n_init=settings['n_init'], max_iter=settings['max_iter'], 
                    tol=settings['tol'], n_jobs=settings['n_jobs'])
        kmeans.fit(points)
        
        busmap = pd.Series(data=kmeans.predict(network.buses.loc[buses_i, ["x", "y"]]), 
                         index=buses_i, dtype=object)
        
        # identify medoids per cluster -> k-medoids clustering
        
        distances = pd.DataFrame(data=kmeans.transform(network.buses.loc[buses_i, ["x", "y"]].values), 
                             index=buses_i, dtype=object)
        
        medoid_idx = pd.Series(data=np.zeros(shape=settings['n_clusters'], dtype=int))
        for i in range(0, settings['n_clusters']):
            dist = pd.to_numeric(distances[i])
            index=int(dist.idxmin())
            medoid_idx[i]=index
            
        # dijkstra's algorithm
            
        busmap = dijkstras_algorithm(network, medoid_idx, busmap)
        busmap.index.name='bus_id'
        busmap.to_csv('kmedoids_dijkstra_busmap_' + str(settings['n_clusters']) + '_result.csv')
        
    else:
        df = pd.read_csv(settings['busmap'])
        df=df.astype(str)
        df = df.set_index('bus_id')
        busmap = df.squeeze('columns')
        
    network.generators['weight'] = network.generators['p_nom']
    aggregate_one_ports = network.one_port_components.copy()
    aggregate_one_ports.discard('Generator')
    clustering = get_clustering_from_busmap(
        network,
        busmap,
        aggregate_generators_weighted=True,
        one_port_strategies={'StorageUnit': {'marginal_cost': np.mean,
                                             'capital_cost': np.mean,
                                             'efficiency': np.mean,
                                             'efficiency_dispatch': np.mean,
                                             'standing_loss': np.mean,
                                             'efficiency_store': np.mean,
                                             'p_min_pu': np.min}},
        generator_strategies={'p_nom_min':np.min,
                              'p_nom_opt': np.sum,
                              'marginal_cost': np.mean,
                              'capital_cost': np.mean},
        aggregate_one_ports=aggregate_one_ports,
        line_length_factor=settings['line_length_factor'])
    
    for i in range(len(medoid_idx)):
        index=int(clustering.network.buses.index[i])
        medoid=str(medoid_idx.loc[index])
        clustering.network.buses['x'].iloc[i]=network.buses['x'].loc[medoid]     
        clustering.network.buses['y'].iloc[i]=network.buses['y'].loc[medoid]

    return clustering

def run_spatial_clustering(self):

    if self.args['network_clustering']['active']:

        self.network.generators.control = "PV"

        if self.args['network_clustering']['method'] == 'kmeans':

            logger.info('Start k-mean clustering')

            self.clustering = kmean_clustering(self)
            
        elif self.args['network_clustering']['method'] == 'kmedoids-dijkstra':
            
            logger.info('Start k-medoids dijkstra clustering')

            self.clustering = kmedoids_dijkstra_clustering(self)            

        if self.args['disaggregation'] != None:
                self.disaggregated_network = self.network.copy()

        self.network = self.clustering.network.copy()

        self.geolocation_buses()

        self.network.generators.control[self.network.generators.control == ''] = 'PV'

        logger.info("Network clustered to {} buses with k-means algorithm."
                    .format(self.args['network_clustering']['n_clusters']))