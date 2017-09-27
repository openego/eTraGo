from etrago.extras.utilities import *
from pypsa.networkclustering import aggregatebuses, aggregateoneport, aggregategenerators, get_clustering_from_busmap, busmap_by_kmeans
from egoio.db_tables.model_draft import EgoGridPfHvBusmap
from itertools import product
import networkx as nx
import multiprocessing as mp
from math import ceil
import pandas as pd
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from networkx import NetworkXNoPath
from egoio.db_tables.model_draft import EgoGridPfHvBusmap
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
        network_c.now = network.now
        network_c.set_snapshots(network.snapshots)

    # dealing with generators
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
    """ Construct an undirected multigraph from a list containing data on
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
    """
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

def kmean_clustering(network):
    """ Implement k-mean clustering in existing network
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    Returns
    -------

    """
    def weighting_for_scenario(x):
        b_i = x.index
        g = normed(gen.reindex(b_i, fill_value=0))
        l = normed(load.reindex(b_i, fill_value=0))
      
        w= g + l
        return (w * (100. / w.max())).astype(int)

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

    #ToDo: change conv to types minus wind and solar 
    conv_types = {'biomass', 'run_of_river', 'gas', 'oil','coal', 'waste','uranium'}
    # Attention: network.generators.carrier.unique() 
    # conv_types only for SH scenario defined!
    gen = (network.generators.loc[network.generators.carrier.isin(conv_types)
        ].groupby('bus').p_nom.sum().reindex(network.buses.index, 
        fill_value=0.) + network.storage_units.loc[network.storage_units.carrier.isin(conv_types)
        ].groupby('bus').p_nom.sum().reindex(network.buses.index, fill_value=0.))
        
    load = network.loads_t.p_set.mean().groupby(network.loads.bus).sum()

    # k-mean clustering
    # busmap = busmap_by_kmeans(network, bus_weightings=pd.Series(np.repeat(1,
    #       len(network.buses)), index=network.buses.index) , n_clusters= 10)
    weight = weighting_for_scenario(network.buses).reindex(network.buses.index, fill_value=1)
    busmap = busmap_by_kmeans(network, bus_weightings=pd.Series(weight), buses_i=network.buses.index , n_clusters= 10)


    # ToDo change function in order to use bus_strategies or similar
    clustering = get_clustering_from_busmap(network, busmap)
    network = clustering.network
    #network = cluster_on_extra_high_voltage(network, busmap, with_time=True)

    return network
