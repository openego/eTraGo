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

def adjust_no_electric_network(network, busmap):
    
    # network2 is supposed to contain all the no electrical buses and links
    network2 = network.copy()
    network2.buses = network2.buses[(network2.buses['carrier'] == 'central_heat') |
                                  (network2.buses['carrier'] == 'rural_heat') |
                                  (network2.buses['carrier'] == 'H2') |
                                  (network2.buses['carrier'] == 'dsm-cts') |
                                  (network2.buses['carrier'] == 'dsm-ind-osm') |
                                  (network2.buses['carrier'] == 'dsm-ind-sites')]
    
    network2.links = network2.links[(network2.links['carrier'] == 'central_heat_pump') |
                                  (network2.links['carrier'] == 'individual_heat_pump') |
                                  (network2.links['carrier'] == 'power-to-H2') |
                                  (network2.links['carrier'] == 'dsm-cts') |
                                  (network2.links['carrier'] == 'dsm-ind-osm') |
                                  (network2.links['carrier'] == 'dsm-ind-sites')]
    
    #no_elec_to_eHV maps the no electrical buses to the closest eHV bus
    no_elec_to_eHV = {}
    #new_buses contains the names of all the new no electrical buses
    new_buses = {}
    #busmap2 maps all the no electrical buses to the new buses based on the
    #eHV network
    busmap2 = {}
    max_id = network.buses.index.to_series().apply(int).max()
    
    for link in network2.links.index:
        heat_bus = network2.links.loc[link, 'bus1']
        bus_hv = network2.links.loc[link, 'bus0']
        if network2.links.loc[link, 'carrier'] == 'central_heat_pump':
            carry = 'central_heat'
        elif network2.links.loc[link, 'carrier'] == 'individual_heat_pump':
            carry = 'rural_heat'
        elif network2.links.loc[link, 'carrier'] == 'power-to-H2':
            carry = 'H2'
        else:
            carry = network2.links.loc[link, 'carrier']
                
        no_elec_to_eHV[heat_bus] = busmap[str(bus_hv)]
        if busmap[str(bus_hv)] + "-" + carry not in new_buses:
            new_buses[busmap[str(bus_hv)] + "-" + carry] = str(max_id + 1)
            max_id = max_id + 1
        busmap2[heat_bus] = new_buses[busmap[str(bus_hv)] + "-" + carry]
        
    #The new buses based on the eHV network for not electrical buses are created
    for carry, df in network2.buses.groupby(by= 'carrier'):
        bus_unique = []
        for bus in df.index:
            try:
                if no_elec_to_eHV[bus] not in bus_unique:
                    bus_unique.append(no_elec_to_eHV[bus])
            except:
                print(f'Bus {bus} has no connexion to electricity network')
        
        for eHV_bus in bus_unique:    
            new_bus = pd.Series({
                'scn_name': network.buses.at[eHV_bus, 'scn_name'],
                'v_nom': np.nan,
                'carrier': carry,
                'x': network.buses.at[eHV_bus, 'x'],
                'y': network.buses.at[eHV_bus, 'y'],
                'geom': network.buses.at[eHV_bus, 'geom'],
                'type': "",
                'v_mag_pu_set': 1,
                'v_mag_pu_min': 0,
                'v_mag_pu_max': np.inf,
                'control': "PV",
                'sub_network': "",
                'country_code': network.buses.at[eHV_bus, 'country_code']},
                 name= new_buses[eHV_bus + "-" + carry],)
            network.buses = network.buses.append(new_bus)
    #busmap now includes the not electrical buses and their corresponding new
    #buses to be mapped.
    
    for CH4_bus in network.buses[network.buses['carrier'] == 'CH4'].index:
        busmap2[CH4_bus] = CH4_bus
    
    busmap = {**busmap, **busmap2}
    
    return network, busmap


def _make_consense_links(x):
    v = x.iat[0]
    assert ((x == v).all() or x.isnull().all()), (
        f"No consense in table links column {x.name}: \n {x}")
    return v


def nan_links(x):
    return np.nan

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

    network, busmap = adjust_no_electric_network(network, busmap)

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
    dc_links = links[links['carrier'] == 'AC']
    links = links[links['carrier'] != 'AC']

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
        one_port_strategies = {'StorageUnit': {'marginal_cost': np.mean, 'capital_cost': np.mean,
                             'efficiency_dispatch': np.mean, 'standing_loss': np.mean, 'efficiency_store': np.mean,
                             'p_min_pu': np.min}, 'Store': {'marginal_cost': np.mean, 'capital_cost': np.mean,
                             'standing_loss': np.mean, 'e_nom': np.sum}}
        new_df, new_pnl = aggregateoneport(
            network, busmap, component=one_port, with_time=with_time,
            custom_strategies=one_port_strategies.get(one_port, {}))
        io.import_components_from_dataframe(network_c, new_df, one_port)
        for attr, df in iteritems(new_pnl):
            io.import_series_from_dataframe(network_c, df, one_port, attr)
    
    # Dealing with links
    network2 = network.copy()
    network2.links.bus0 = network2.links.bus0.map(busmap)
    network2.links.bus1 = network2.links.bus1.map(busmap)
    network2.links.dropna(subset= ['bus0', 'bus1'], inplace= True)
    network2.links['topo'] = np.nan
    
    strategies={'scn_name': _make_consense_links,
                'bus0': _make_consense_links,
                'bus1': _make_consense_links,
                'carrier': _make_consense_links,
                'efficiency_fixed': _make_consense_links,
                'p_nom': np.sum,
                'p_nom_extendable': _make_consense_links,
                'p_nom_max': np.sum,
                'capital_cost': np.mean,
                'length': np.mean,
                'geom': nan_links,
                'topo': nan_links,
                'type': nan_links,
                'efficiency': np.mean,
                'p_nom_min': np.min,
                'p_set': np.mean,
                'p_min_pu':np.min,
                'p_max_pu': np.max,
                'marginal_cost': np.mean,
                'terrain_factor': _make_consense_links,
                'p_nom_opt': np.mean,
                'country': _make_consense_links}
    
    network_c.links = network2.links.groupby(['bus0', 'bus1', 'carrier']).agg(strategies)
    network_c.links = network_c.links.append(dc_links)
    network_c.links = network_c.links.reset_index(drop=True)
    
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
    transformer = connected_transformer(etrago.network, s_buses)
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
    df.sort_index(inplace=True)
    df = df.fillna(10000000)


    mask = df.groupby(level='source')['path_length'].idxmin()
    df = df.loc[mask, :]

    # rename temporary endpoints
    df.reset_index(inplace=True)
    df.target = df.target.map(dict(zip(etrago.network.transformers.bus0,
                                       etrago.network.transformers.bus1)))

    # append to busmap buses only connected to transformer
    transformer = etrago.network.transformers
    idx = list(set(buses_of_vlvl(etrago.network, fromlvl)).
               symmetric_difference(set(s_buses)))
    mask = transformer.bus0.isin(idx)

    toappend = pd.DataFrame(list(zip(transformer[mask].bus0,
                                     transformer[mask].bus1)),
                            columns=['source', 'target'])
    toappend['path_length'] = 0

    df = pd.concat([df, toappend], ignore_index=True, axis=0)

    # append all other buses
    buses = etrago.network.buses[etrago.network.buses.carrier=='AC']
    mask = buses.index.isin(df.source)


    assert (buses[~mask].v_nom.astype(int).isin(tolvl)).all()

    tofill = pd.DataFrame([buses.index[~mask]] * 2).transpose()
    tofill.columns = ['source', 'target']
    tofill['path_length'] = 0

    df = pd.concat([df, tofill], ignore_index=True, axis=0)

    # prepare data for export

    df['scn_name'] = scn_name
    df['version'] = etrago.args['gridversion']

    if not df.version.any():
        df.version = 'testcase'

    df.rename(columns={'source': 'bus0', 'target': 'bus1'}, inplace=True)
    df.set_index(['scn_name', 'bus0', 'bus1'], inplace=True)

    df.to_sql('egon_etrago_hv_busmap', con=etrago.engine,
                            schema='grid', if_exists='append')

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

    from saio.grid import egon_etrago_hv_busmap

    filter_version = etrago.args['gridversion']

    if not filter_version:
        filter_version = 'testcase'
    def fetch():

        query = etrago.session.query(
            egon_etrago_hv_busmap.bus0, egon_etrago_hv_busmap.bus1).\
            filter(egon_etrago_hv_busmap.scn_name == scn_name).\
            filter(egon_etrago_hv_busmap.version == filter_version)

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
        busmap.to_csv('kmeans_busmap_' + str(kmean_settings['n_clusters']) + '_result.csv')
    else:
        df = pd.read_csv(kmean_settings['kmeans_busmap'])
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

def run_kmeans_clustering(self):

    if self.args['network_clustering_kmeans']['active']:

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