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

import pdb

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

    for s, t in paths:

        try:
            df.loc[(s, t), 'path_length'] = \
                nx.dijkstra_path_length(graph, s, t)

        except NetworkXNoPath:
            continue

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


    network = etrago.network
    kmean_settings = etrago.args['network_clustering_kmeans']
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
    if not kmean_settings['kmeans_busmap']:
        busmap = busmap_by_kmeans(
            network,
            bus_weightings=pd.Series(weight),
            n_clusters=kmean_settings['n_clusters'],
            n_init=kmean_settings['n_init'],
            max_iter=kmean_settings['max_iter'],
            tol=kmean_settings['tol'],
            n_jobs=kmean_settings['n_jobs'])
        busmap.to_csv(etrago.args['csv_export']+'/kmeans_busmap_' + str(kmean_settings['n_clusters']) + '_result.csv')
    else:
        df = pd.read_csv(kmean_settings['kmeans_busmap'])
        df=df.astype(str)
        df = df.set_index('bus_id')
        busmap = df.squeeze('columns')
        busmap.to_csv(etrago.args['csv_export']+'/kmeans_busmap_' + str(kmean_settings['n_clusters']) + '_result.csv')

    network.generators['weight'] = network.generators['p_nom']
    aggregate_one_ports = network.one_port_components.copy()
    aggregate_one_ports.discard('Generator')

    ######################################
    # add sub-sector to busmap
    sub = ' dsm_sum'    
    
    busmap_sub = busmap + sub
    busmap_sub.index = busmap_sub.index + sub
    busmap = busmap.append(busmap_sub)

    # import sub-sector network
    sub_csv_path = '/srv/ES2050/enera_region4flex/dsmlib_pypsa_export_2035/results/2011_test-2/2035/pypsa_format_sum/'
    sub_network = Network(import_name=sub_csv_path)

    #### manipulate dsm parameters
    p_max_loc = sub_network.links.index.str.contains('p_max')
    sub_network.links.marginal_cost.loc[p_max_loc] = 5.0 # marginal costs of the cheapest dsm technology
    
    # investment cost reduction from 2020 to 2035 (linear extrapolation from 2030 to 2035)
    #  FFE_Merit Order Energiespeicherung_Hauptbericht Teil 2 Technoökonomische Analyse Funktionaler Energiespeicher, p. 48 / 49 
    # linear extrapolation from 2030 to 2035: 220 + (220-310)/10 * 5 = 175
    # The imported DSM cost data refer to 2020, so they are multiplied with a correction factor
    # correction factor: 175/310 =  0.565        
    sub_network.links.capital_cost.loc[p_max_loc] = (
    sub_network.links.capital_cost.loc[p_max_loc] * 0.565 )   
    
    # cost reduction for the case that less than one year is modelled
    t_period = etrago.args['end_snapshot'] - etrago.args['start_snapshot']    
    sub_network.links.capital_cost.loc[p_max_loc] = (
    (t_period + 1) / 8760 * sub_network.links.capital_cost.loc[p_max_loc] )
    
    sub_network.links.p_nom_extendable = True    
    sub_network.links.p_nom_max = sub_network.links.p_nom
    sub_network.links.p_nom = 0
    
    sub_network.stores.e_nom_extendable = True
    sub_network.stores.e_nom_max = sub_network.stores.e_nom
    sub_network.stores.e_nom = 0
    sub_network.stores.e_initial = 0
    
    # add subsector network to main network   
    io.import_components_from_dataframe(network, sub_network.buses, "Bus")    
    io.import_components_from_dataframe(network, sub_network.links, "Link")  
    io.import_components_from_dataframe(network, sub_network.stores, "Store")
    io.import_series_from_dataframe(network, sub_network.links_t.p_max_pu,'Link','p_max_pu')
    io.import_series_from_dataframe(network, sub_network.links_t.p_min_pu,'Link','p_min_pu')
    io.import_series_from_dataframe(network, sub_network.stores_t.e_max_pu,'Store','e_max_pu')
    io.import_series_from_dataframe(network, sub_network.stores_t.e_min_pu,'Store','e_min_pu')

    ######################################
    
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
                                             'p_min_pu': np.min},
                            'Store': {'e_nom': np.sum,
                                      'e_nom_max': np.sum,
                                      'e_max_pu': np.mean,
                                      'e_min_pu': np.mean,
                                      },
                                             
                                             },
                                             
        generator_strategies={'p_nom_min':np.min,
                              'p_nom_opt': np.sum,
                              'marginal_cost': np.mean,
                              'capital_cost': np.mean},
        aggregate_one_ports=aggregate_one_ports,
        line_length_factor=kmean_settings['line_length_factor'])
    
    ############## cluster sub sector links ##############
 
    links=clustering.network.links
    
    dsm_env_list = ['p_max','p_min'] # dsm envelope

    # add main sector to linkmap
    linkmap = pd.Series()
    links_sel=links[~links.index.str.contains(sub)] # ATTENTION: adjust this when having multiple sub sectors
    linkmap_sel = pd.Series(links_sel.index.to_list(), index=links_sel.index.to_list())  
    linkmap = linkmap.append(linkmap_sel)

    for dsm_env in dsm_env_list:
        links_sel=links[links.index.str.contains(sub)&links.index.str.contains(dsm_env)]
        ind = links_sel.index
        l=links_sel.groupby(['bus0','bus1']) # dsm links all point to the dsm buses, so not necessary to take care off directionality 
    
        data=dict(
            version =l['version'].first(),  
            scn_name =l['scn_name'].first(),
            efficiency=l['efficiency'].mean(), 
            p_nom =l['p_nom'].sum(),
           p_nom_extendable =l['p_nom_extendable'].first(), 
           p_nom_min = l['p_nom_min'].sum(),
           p_nom_max = l['p_nom_max'].sum(),
           capital_cost = l['capital_cost'].mean(),
           length =l['length'].first(),
           terrain_factor = l['terrain_factor'].first(),
           geom = l['geom'].first(),
           topo = l['topo'].first(),
           type = l['type'].first(),
           p_set = l['p_set'].sum(),
           p_min_pu = l['p_min_pu'].first(),
           p_max_pu  = l['p_max_pu'].first(),
           marginal_cost  = l['marginal_cost'].mean(),
           p_nom_opt  = l['p_nom_opt'].first(),
           country = l['country'].first(),
           v_nom = l['v_nom'].first(),
        )
    
        l_cl = pd.DataFrame(data)#, index = [str(i+1) for i in range(len(ind))])
        l_cl['bus0'] = l_cl.index.get_level_values(0)  
        l_cl['bus1'] = l_cl.index.get_level_values(1) 
        l_cl['name'] = l_cl.index.get_level_values(1)+' '+dsm_env 
        
        linkmap_sel = links_sel.join(l_cl['name'], on=['bus0', 'bus1'])['name']
        linkmap = linkmap.append(linkmap_sel)
        l_cl.reset_index(drop=True, inplace=True)
        l_cl.set_index('name',inplace=True)
        
        clustering.network.links.drop(ind,inplace=True)
        clustering.network.links = clustering.network.links.append(l_cl)

    ############################

    ##### cluster links_t ############

    def _flatten_multiindex(m, join=' '):
        if m.nlevels <= 1: return m
        levels = map(m.get_level_values, range(m.nlevels))
        return reduce(lambda x, y: x+join+y, levels, next(levels))
    
    new_pnl = dict()
    grouper=linkmap
    old_pnl = clustering.network.pnl('Link') #network.pnl(component)
    for attr, df in iteritems(old_pnl):
        if not df.empty:
            pnl_df = df.groupby(grouper, axis=1).mean()
            pnl_df.columns = _flatten_multiindex(pnl_df.columns).rename("name")
            new_pnl[attr] = pnl_df
    
    for attr, df in iteritems(new_pnl):
            io.import_series_from_dataframe(clustering.network, df, 'Link', attr)
            
    ind_drop=linkmap[linkmap.index.str.contains(sub)].index
    clustering.network.links_t['p_max_pu'].drop(ind_drop,axis=1,inplace=True)
    clustering.network.links_t['p_min_pu'].drop(ind_drop,axis=1,inplace=True)

    ###############################    

    return clustering

def run_kmeans_clustering(self):

    if self.args['network_clustering_kmeans'] != False:

        self.network.generators.control = "PV"

        logger.info('Start k-mean clustering')

        self.clustering = kmean_clustering(self)

        if self.args['disaggregation'] != None:
                self.disaggregated_network = self.network.copy()

        self.network = self.clustering.network.copy()

        self.geolocation_buses()

        self.network.generators.control[self.network.generators.control == ''] = 'PV'

        logger.info("Network clustered to {} buses with k-means algorithm."
                    .format(self.args['network_clustering_kmeans']['n_clusters']))