# -*- coding: utf-8 -*-
""" Networkclustering.py defines the methods to cluster power grid
networks for application within the tool eTraGo. 

This program is free software; you can redistribute it and/or
modify it under the terms of the GNU Affero General Public License as
published by the Free Software Foundation; either version 3 of the
License, or (at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

__copyright__ = "Flensburg University of Applied Sciences, Europa-Universität Flensburg, Centre for Sustainable Energy Systems, DLR-Institute for Networked Energy Systems"
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "s3pp, wolfbunke, ulfmueller, lukasol"

import os
if not 'READTHEDOCS' in os.environ:
    from etrago.tools.utilities import *
    from pypsa.networkclustering import aggregatebuses, aggregateoneport, aggregategenerators, get_clustering_from_busmap, busmap_by_kmeans
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

# TODO: Workaround because of agg
def _leading(busmap, df):
    def leader(x):
        ix = busmap[x.index[0]]
        return df.loc[ix, x.name]
    return leader


def cluster_on_extra_high_voltage(network, busmap, with_time=True):
    """ Create a new clustered pypsa.Network given a busmap mapping all busids
    to other busids of the same set.

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
        Container for all network components.
    """

    network_c = Network()

    buses = aggregatebuses(network, busmap, {'x':_leading(busmap, network.buses),
                                             'y':_leading(busmap, network.buses)})

    # keep attached lines
    lines = network.lines.copy()
    mask = lines.bus0.isin(buses.index)
    lines = lines.loc[mask,:]

    # keep attached transformer
    transformers = network.transformers.copy()
    mask = transformers.bus0.isin(buses.index)
    transformers = transformers.loc[mask,:]

    io.import_components_from_dataframe(network_c, buses, "Bus")
    io.import_components_from_dataframe(network_c, lines, "Line")
    io.import_components_from_dataframe(network_c, transformers, "Transformer")

    if with_time:
        network_c.snapshots = network.snapshots
        network_c.set_snapshots(network.snapshots)

    # dealing with generators
    network.generators.control="PV"
    network.generators['weight'] = 1
    new_df, new_pnl = aggregategenerators(network, busmap, with_time)
    io.import_components_from_dataframe(network_c, new_df, 'Generator')
    for attr, df in iteritems(new_pnl):
        io.import_series_from_dataframe(network_c, df, 'Generator', attr)

    # dealing with all other components
    aggregate_one_ports = components.one_port_components.copy()
    aggregate_one_ports.discard('Generator')

    for one_port in aggregate_one_ports:
        new_df, new_pnl = aggregateoneport(network, busmap, component=one_port, with_time=with_time)
        io.import_components_from_dataframe(network_c, new_df, one_port)
        for attr, df in iteritems(new_pnl):
            io.import_series_from_dataframe(network_c, df, one_port, attr)

    network_c.determine_network_topology()

    return network_c

def graph_from_edges(edges):
    """ 
    Construct an undirected multigraph from a list containing data on
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
    """

    g = graph.copy()

    for i in range(0, len(nodes), n):
        yield (nodes[i:i + n], g)


def shortest_path(paths, graph):
    """ Finding minimum path lengths between sources and targets pairs defined
    in paths.

    Parameters
    ----------
    ways : list
        List of tuples containing a source and target node
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
    """ Create busmap between voltage levels based on dijkstra shortest path.
    The result is written to the `model_draft` on the OpenEnergy - Platform. The
    relation name is `ego_grid_pf_hv_busmap`.

    Parameters
    ----------
    network : pypsa.Network object
        Container for all network components.
    session : sqlalchemy.orm.session.Session object
        Establishes conversations with the database.
    scn_name : str
        Name of the scenario.
    fromlvl : list
        List of voltage-levels to cluster.
    tolvl : list
        List of voltage-levels to remain.
    cpu_cores : int
        Number of CPU-cores.

    Raises
    ------
    AssertionError
        If there are buses with a voltage-level not covered by fromlvl or tolvl.

    Returns
    -------
    None

    Notes
    -----

        Relation `ego_grid_pf_hv_busmap` has the following attributes:
            * scn_name
            * bus0
            * bus1,
            * path_length.

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
    chunksize = ceil(len(ppaths)/cpu_cores)
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
    df['scn_name'] = scn_name
    df.rename(columns={'source': 'bus0', 'target': 'bus1'}, inplace=True)
    df.set_index(['scn_name', 'bus0', 'bus1'], inplace=True)

    for i, d in df.reset_index().iterrows():
        session.add(EgoGridPfHvBusmap(**d.to_dict()))

    session.commit()

    return


def busmap_from_psql(network, session, scn_name):
    """ Retrieve busmap from OEP-relation `model_draft.ego_grid_pf_hv_busmap`
    by a given scenario name. If not present the busmap is created with default
    values to cluster on the EHV-level (110 --> 220, 380 kV)

    Parameters
    ----------
    network : pypsa.Network object
        Container for all network components.
    session : sqlalchemy.orm.session.Session object
        Establishes conversations with the database.
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

def kmean_clustering(network, n_clusters=10):
    """ 
    Implement k-mean clustering in existing network
   
    Parameters
    ----------
    
    network : :class:`pypsa.Network
        Overall container of PyPSA
        
    Returns
    -------
    network : pypsa.Network object
        Container for all network components.
        
    """
    def weighting_for_scenario(x):
        b_i = x.index
        g = normed(gen.reindex(b_i, fill_value=0))
        l = normed(load.reindex(b_i, fill_value=0))
      
        w= g + l
        return (w * (100000. / w.max())).astype(int)

    def normed(x):
        return (x/x.sum()).fillna(0.)
    
    print('start k-mean clustering')
    # prepare k-mean
    # k-means clustering (first try)
    network.generators.control="PV"
    network.buses['v_nom'] = 380.
    # problem our lines have no v_nom. this is implicitly defined by the connected buses:
    network.lines["v_nom"] = network.lines.bus0.map(network.buses.v_nom)

    # adjust the x of the lines which are not 380. 
    lines_v_nom_b = network.lines.v_nom != 380
    network.lines.loc[lines_v_nom_b, 'x'] *= (380./network.lines.loc[lines_v_nom_b, 'v_nom'])**2
    network.lines.loc[lines_v_nom_b, 'v_nom'] = 380.

    trafo_index = network.transformers.index
    transformer_voltages = pd.concat([network.transformers.bus0.map(network.buses.v_nom), network.transformers.bus1.map(network.buses.v_nom)], axis=1)


    network.import_components_from_dataframe(
    network.transformers.loc[:,['bus0','bus1','x','s_nom']]
    .assign(x=network.transformers.x*(380./transformer_voltages.max(axis=1))**2)
    .set_index('T' + trafo_index),
    'Line')
    network.transformers.drop(trafo_index, inplace=True)

    for attr in network.transformers_t:
      network.transformers_t[attr] = network.transformers_t[attr].reindex(columns=[])

    #define weighting based on conventional 'old' generator spatial distribution
    non_conv_types= {'biomass', 'wind', 'solar', 'geothermal', 'load shedding', 'extendable_storage'}
    # Attention: network.generators.carrier.unique() 
    gen = (network.generators.loc[(network.generators.carrier.isin(non_conv_types)==False)
        ].groupby('bus').p_nom.sum().reindex(network.buses.index, 
        fill_value=0.) + network.storage_units.loc[(network.storage_units.carrier.isin(non_conv_types)==False)
        ].groupby('bus').p_nom.sum().reindex(network.buses.index, fill_value=0.))
        
    load = network.loads_t.p_set.mean().groupby(network.loads.bus).sum()

    # k-mean clustering
    # busmap = busmap_by_kmeans(network, bus_weightings=pd.Series(np.repeat(1,
    #       len(network.buses)), index=network.buses.index) , n_clusters= 10)
    weight = weighting_for_scenario(network.buses).reindex(network.buses.index, fill_value=1)
    busmap = busmap_by_kmeans(network, bus_weightings=pd.Series(weight), n_clusters=n_clusters)


    # ToDo change function in order to use bus_strategies or similar
    network.generators['weight'] = 1
    aggregate_one_ports = components.one_port_components.copy()
    aggregate_one_ports.discard('Generator')
    clustering = get_clustering_from_busmap(network, busmap, aggregate_generators_weighted=True, aggregate_one_ports=aggregate_one_ports)
    network = clustering.network
    #network = cluster_on_extra_high_voltage(network, busmap, with_time=True)

    return network
    
