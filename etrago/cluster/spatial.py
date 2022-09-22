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
""" spatial.py defines the methods to run spatial clustering on networks."""

import os

if "READTHEDOCS" not in os.environ:
    import logging
    import multiprocessing as mp
    from itertools import product
    from math import ceil
    from pickle import dump

    import networkx as nx
    import numpy as np
    import pandas as pd
    import time
    from networkx import NetworkXNoPath
    from pypsa.networkclustering import (
        _flatten_multiindex,
        busmap_by_kmeans,
        busmap_by_stubs,
        busmap_by_hac,
        get_clustering_from_busmap,
    )
    from sklearn.cluster import KMeans

    from pypsa.geo import haversine

    from etrago.tools.utilities import *

    logger = logging.getLogger(__name__)

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "s3pp, wolfbunke, ulfmueller, lukasol"

# TODO: Workaround because of agg


def _make_consense_links(x):
    v = x.iat[0]
    assert (
        x == v
    ).all() or x.isnull().all(), f"No consense in table links column {x.name}: \n {x}"
    return v


def nan_links(x):
    return np.nan


def ext_storage(x):
    v = any(x[x == True])
    return v


def strategies_one_ports():
    return {
        "StorageUnit": {
            "marginal_cost": np.mean,
            "capital_cost": np.mean,
            "efficiency_dispatch": np.mean,
            "standing_loss": np.mean,
            "efficiency_store": np.mean,
            "p_min_pu": np.min,
            "p_nom_extendable": ext_storage,
        },
        "Store": {
            "marginal_cost": np.mean,
            "capital_cost": np.mean,
            "standing_loss": np.mean,
            "e_nom": np.sum,
            "e_nom_min": np.sum,
            "e_nom_max": np.sum,
            "e_initial": np.sum,
        },
    }


def agg_e_nom_max(x):
    if (x == np.inf).any():
        return np.inf
    else:
        return x.sum()


def strategies_generators():
    return {
        "p_nom_min": np.min,
        "p_nom_max": np.min,
        "weight": np.sum,
        "p_nom": np.sum,
        "p_nom_opt": np.sum,
        "marginal_cost": np.mean,
        "capital_cost": np.mean,
        "e_nom_max": agg_e_nom_max,
    }


def strategies_links():
    return {
        "scn_name": _make_consense_links,
        "bus0": _make_consense_links,
        "bus1": _make_consense_links,
        "carrier": _make_consense_links,
        "p_nom": np.sum,
        "p_nom_extendable": _make_consense_links,
        "p_nom_max": np.sum,
        "capital_cost": np.mean,
        "length": np.mean,
        "geom": nan_links,
        "topo": nan_links,
        "type": nan_links,
        "efficiency": np.mean,
        "p_nom_min": np.min,
        "p_set": np.mean,
        "p_min_pu": np.min,
        "p_max_pu": np.max,
        "marginal_cost": np.mean,
        "terrain_factor": _make_consense_links,
        "p_nom_opt": np.mean,
        "country": nan_links,
        "build_year": np.mean,
        "lifetime": np.mean,
    }


def group_links(network, with_time=True, carriers=None, cus_strateg=dict()):
    """
    Aggregate network.links and network.links_t after any kind of clustering

    Parameters
    ----------
    network : pypsa.Network object
        Container for all network components.
    with_time : bool
        says if the network object contains timedependent series.
    carriers : list of strings
        Describe which typed of carriers should be aggregated. The default is None.
    strategies : dictionary
        custom strategies to perform the aggregation

    Returns
    -------
    new_df : links aggregated based on bus0, bus1 and carrier
    new_pnl : links time series aggregated
    """
    if carriers is None:
        carriers = network.links.carrier.unique()

    links_agg_b = network.links.carrier.isin(carriers)
    links = network.links.loc[links_agg_b]
    grouper = [links.bus0, links.bus1, links.carrier]

    def normed_or_uniform(x):
        return (
            x / x.sum() if x.sum(skipna=False) > 0 else pd.Series(1.0 / len(x), x.index)
        )

    weighting = links.p_nom.groupby(grouper, axis=0).transform(normed_or_uniform)
    strategies = strategies_links()
    strategies.update(cus_strateg)
    new_df = links.groupby(grouper, axis=0).agg(strategies)
    new_df.index = _flatten_multiindex(new_df.index).rename("name")
    new_df = pd.concat([new_df, network.links.loc[~links_agg_b]], axis=0, sort=False)
    new_df["new_id"] = np.arange(len(new_df)).astype(str)
    cluster_id = new_df["new_id"].to_dict()
    new_df.set_index("new_id", inplace=True)
    new_df.index = new_df.index.rename("Link")

    new_pnl = dict()
    if with_time:
        for attr, df in network.links_t.items():
            pnl_links_agg_b = df.columns.to_series().map(links_agg_b)
            df_agg = df.loc[:, pnl_links_agg_b].astype(float)
            if not df_agg.empty:
                if attr in ["efficiency", "p_max_pu", "p_min_pu"]:
                    df_agg = df_agg.multiply(weighting.loc[df_agg.columns], axis=1)
                pnl_df = df_agg.groupby(grouper, axis=1).sum()
                pnl_df.columns = _flatten_multiindex(pnl_df.columns).rename("name")
                new_pnl[attr] = pd.concat(
                    [df.loc[:, ~pnl_links_agg_b], pnl_df], axis=1, sort=False
                )
                new_pnl[attr].columns = new_pnl[attr].columns.map(cluster_id)
            else:
                new_pnl[attr] = network.links_t[attr]

    new_pnl = pypsa.descriptors.Dict(new_pnl)

    return new_df, new_pnl


def graph_from_edges(edges):
    """Constructs an undirected multigraph from a list containing data on
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
    """Generator for applying multiprocessing.

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
        yield (nodes[i : i + n], g)


def shortest_path(paths, graph):
    """Finds the minimum path lengths between node pairs defined in paths.

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

    idxnames = ["source", "target"]
    idx = pd.MultiIndex.from_tuples(paths, names=idxnames)
    df = pd.DataFrame(index=idx, columns=["path_length"])
    df.sort_index(inplace=True)

    df_isna = df.isnull()
    for s, t in paths:
        while df_isna.loc[(s, t), "path_length"] == True:
            try:
                s_to_other = nx.single_source_dijkstra_path_length(graph, s)
                for t in idx.levels[1]:
                    if t in s_to_other:
                        df.loc[(s, t), "path_length"] = s_to_other[t]
                    else:
                        df.loc[(s, t), "path_length"] = np.inf
            except NetworkXNoPath:
                continue
            df_isna = df.isnull()

    return df


def busmap_by_shortest_path(etrago, scn_name, fromlvl, tolvl, cpu_cores=4):
    """Creates a busmap for the EHV-Clustering between voltage levels based
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
    edges = [(row.bus0, row.bus1, row.length, ix) for ix, row in lines.iterrows()]
    M = graph_from_edges(edges)

    # applying multiprocessing
    p = mp.Pool(cpu_cores)

    chunksize = ceil(len(ppaths) / cpu_cores)
    container = p.starmap(shortest_path, gen(ppaths, chunksize, M))
    df = pd.concat(container)
    dump(df, open("df.p", "wb"))

    # post processing
    df.sort_index(inplace=True)
    df = df.fillna(10000000)

    mask = df.groupby(level="source")["path_length"].idxmin()
    df = df.loc[mask, :]

    # rename temporary endpoints
    df.reset_index(inplace=True)
    df.target = df.target.map(
        dict(zip(etrago.network.transformers.bus0, etrago.network.transformers.bus1))
    )

    # append to busmap buses only connected to transformer
    transformer = etrago.network.transformers
    idx = list(
        set(buses_of_vlvl(etrago.network, fromlvl)).symmetric_difference(set(s_buses))
    )
    mask = transformer.bus0.isin(idx)

    toappend = pd.DataFrame(
        list(zip(transformer[mask].bus0, transformer[mask].bus1)),
        columns=["source", "target"],
    )
    toappend["path_length"] = 0

    df = pd.concat([df, toappend], ignore_index=True, axis=0)

    # append all other buses
    buses = etrago.network.buses[etrago.network.buses.carrier == "AC"]
    mask = buses.index.isin(df.source)

    assert (buses[~mask].v_nom.astype(int).isin(tolvl)).all()

    tofill = pd.DataFrame([buses.index[~mask]] * 2).transpose()
    tofill.columns = ["source", "target"]
    tofill["path_length"] = 0

    df = pd.concat([df, tofill], ignore_index=True, axis=0)
    df.drop_duplicates(inplace=True)

    # prepare data for export

    df["scn_name"] = scn_name
    df["version"] = etrago.args["gridversion"]

    if not df.version.any():
        df.version = "testcase"

    df.rename(columns={"source": "bus0", "target": "bus1"}, inplace=True)
    df.set_index(["scn_name", "bus0", "bus1"], inplace=True)

    df.to_sql(
        "egon_etrago_hv_busmap", con=etrago.engine, schema="grid", if_exists="append"
    )

    return


def busmap_from_psql(etrago):
    """Retrieves busmap from `model_draft.ego_grid_pf_hv_busmap` on the
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
    scn_name = (
        etrago.args["scn_name"]
        if etrago.args["scn_extension"] == None
        else etrago.args["scn_name"] + "_ext_" + "_".join(etrago.args["scn_extension"])
    )

    from saio.grid import egon_etrago_hv_busmap

    filter_version = etrago.args["gridversion"]

    if not filter_version:
        filter_version = "testcase"

    def fetch():

        query = (
            etrago.session.query(egon_etrago_hv_busmap.bus0, egon_etrago_hv_busmap.bus1)
            .filter(egon_etrago_hv_busmap.scn_name == scn_name)
            .filter(egon_etrago_hv_busmap.version == filter_version)
        )

        return dict(query.all())

    busmap = fetch()

    # TODO: Or better try/except/finally
    if not busmap:
        print("Busmap does not exist and will be created.\n")

        cpu_cores = input(f"cpu_cores (default=4, max={mp.cpu_count()}): ") or "4"
        if cpu_cores == "max":
            cpu_cores = mp.cpu_count()
        else:
            cpu_cores = int(cpu_cores)

        busmap_by_shortest_path(
            etrago,
            scn_name,
            fromlvl=[110],
            tolvl=[220, 380, 400, 450],
            cpu_cores=cpu_cores,
        )
        busmap = fetch()

    return busmap


def kmean_clustering(etrago, selected_network, weight, n_clusters):
    """Main function of the k-mean clustering approach. Maps an original
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
    kmean_settings = etrago.args["network_clustering"]
    # remove stubs
    if kmean_settings["remove_stubs"]:
        network.determine_network_topology()
        busmap = busmap_by_stubs(network)
        network.generators["weight"] = network.generators["p_nom"]
        aggregate_one_ports = network.one_port_components.copy()
        aggregate_one_ports.discard("Generator")

        # reset coordinates to the new reduced guys, rather than taking an
        # average (copied from pypsa.networkclustering)
        if kmean_settings["use_reduced_coordinates"]:
            # TODO : FIX THIS HACK THAT HAS UNEXPECTED SIDE-EFFECTS,
            # i.e. network is changed in place!!
            network.buses.loc[busmap.index, ["x", "y"]] = network.buses.loc[
                busmap, ["x", "y"]
            ].values

        clustering = get_clustering_from_busmap(
            network,
            busmap,
            aggregate_generators_weighted=True,
            one_port_strategies=strategies_one_ports(),
            generator_strategies=strategies_generators(),
            aggregate_one_ports=aggregate_one_ports,
            line_length_factor=kmean_settings["line_length_factor"],
        )
        etrago.network = clustering.network

        weight = weight.groupby(busmap.values).sum()

    # k-mean clustering
    if not kmean_settings["k_busmap"]:
        busmap = busmap_by_kmeans(
            selected_network,
            bus_weightings=pd.Series(weight),
            n_clusters=n_clusters,
            n_init=kmean_settings["n_init"],
            max_iter=kmean_settings["max_iter"],
            tol=kmean_settings["tol"],
            random_state=kmean_settings["random_state"],
        )
        busmap.to_csv(
            "kmeans_elec_busmap_" + str(kmean_settings["n_clusters_AC"]) + "_result.csv"
        )
    else:
        df = pd.read_csv(kmean_settings["k_busmap"])
        df = df.astype(str)
        df = df.set_index("Bus")
        busmap = df.squeeze("columns")

    return busmap


def dijkstras_algorithm(buses, connections, medoid_idx, busmap_kmedoid):
    """Function for combination of k-medoids Clustering and Dijkstra's algorithm.
      Creates a busmap assigning the nodes of a original network
      to the nodes of a clustered network
      considering the electrical distances based on Dijkstra's shortest path.
      Parameters
    centers
         ----------
      network : pypsa.Network object
          Container for all network components.

      medoid_idx : pd.Series
          Indices of k-medoids
      busmap_kmedoid: pd.Series
          Busmap based on k-medoids clustering
      Returns
      -------
      busmap (format: with labels)
    """

    # original data
    o_buses = buses.index
    # k-medoids centers
    medoid_idx = medoid_idx.astype("str")
    c_buses = medoid_idx.tolist()

    # list of all possible pathways
    ppathss = list(product(o_buses, c_buses))

    # graph creation
    edges = [(row.bus0, row.bus1, row.length, ix) for ix, row in connections.iterrows()]
    M = graph_from_edges(edges)

    # processor count
    cpu_cores = input(f"cpu_cores (default=4, max={mp.cpu_count()}): ") or "4"
    if cpu_cores == "max":
        cpu_cores = mp.cpu_count()
    else:
        cpu_cores = int(cpu_cores)

    # calculation of shortest path between original points and k-medoids centers
    # using multiprocessing
    p = mp.Pool(cpu_cores)
    chunksize = ceil(len(ppathss) / cpu_cores)
    container = p.starmap(shortest_path, gen(ppathss, chunksize, M))
    df = pd.concat(container)
    dump(df, open("df.p", "wb"))

    # assignment of data points to closest k-medoids centers
    df["path_length"] = pd.to_numeric(df["path_length"])
    mask = df.groupby(level="source")["path_length"].idxmin()
    df_dijkstra = df.loc[mask, :]
    df_dijkstra.reset_index(inplace=True)

    # delete double entries in df due to multiprocessing
    df_dijkstra.drop_duplicates(inplace=True)
    df_dijkstra.index = df_dijkstra["source"]

    # creation of new busmap with final assignment (format: medoids indices)
    busmap_ind = pd.Series(df_dijkstra["target"], dtype=object).rename(
        "final_assignment", inplace=True
    )
    busmap_ind.index = df_dijkstra["source"]

    # adaption of busmap to format with labels (necessary for aggregation)
    busmap = busmap_ind.copy()
    mapping = pd.Series(index=medoid_idx, data=medoid_idx.index)
    busmap = busmap_ind.map(mapping).astype(str)
    busmap.index = list(busmap.index.astype(str))

    return busmap


def kmedoids_dijkstra_clustering(etrago, buses, connections, weight, n_clusters):

    settings = etrago.args["network_clustering"]
    # remove stubs
    if settings["remove_stubs"]:

        logger.info(
            "options remove_stubs and use_reduced_coordinates not reasonable for k-medoids Dijkstra Clustering"
        )

    # k-mean clustering
    if not settings["k_busmap"]:

        bus_weightings = pd.Series(weight)
        buses_i = buses.index
        points = buses.loc[buses_i, ["x", "y"]].values.repeat(
            bus_weightings.reindex(buses_i).astype(int), axis=0
        )

        kmeans = KMeans(
            init="k-means++",
            n_clusters=n_clusters,
            n_init=settings["n_init"],
            max_iter=settings["max_iter"],
            tol=settings["tol"],
            random_state=settings["random_state"],
        )
        kmeans.fit(points)

        busmap = pd.Series(
            data=kmeans.predict(buses.loc[buses_i, ["x", "y"]]),
            index=buses_i,
            dtype=object,
        )

        # identify medoids per cluster -> k-medoids clustering

        distances = pd.DataFrame(
            data=kmeans.transform(buses.loc[buses_i, ["x", "y"]].values),
            index=buses_i,
            dtype=object,
        )
        distances = distances.apply(pd.to_numeric)

        medoid_idx = distances.idxmin()

        # dijkstra's algorithm
        busmap = dijkstras_algorithm(buses, connections, medoid_idx, busmap)
        busmap.index.name = "bus_id"

    else:
        df = pd.read_csv(settings["k_busmap"])
        df = df.astype(str)
        df = df.set_index("bus_id")
        busmap = df.squeeze("columns")
        # this method lacks the medoid_idx!

    return busmap, medoid_idx


def hac_clustering(etrago, selected_network, n_clusters):

    settings = etrago.args["network_clustering"]
    carrier = selected_network.buses.iloc[0].carrier
    branch_components = {"Line"} if carrier == "AC" else {"Link"}

    if not settings["k_busmap"]:
        a = time.time()
        D = boolDistance(etrago.network, carrier, settings)

        # make sure all lines and links a valid e. g. only connect existing buses
        bus_indeces = selected_network.buses.index
        selected_network.lines = selected_network.lines.loc[
            (selected_network.lines.bus0.isin(bus_indeces))
            & (selected_network.lines.bus1.isin(bus_indeces))
        ]
        selected_network.links = selected_network.links.loc[
            (selected_network.links.bus0.isin(bus_indeces))
            & (selected_network.links.bus1.isin(bus_indeces))
        ]

        busmap = busmap_by_hac(
            selected_network,
            n_clusters=n_clusters,
            buses_i=None,
            branch_components=branch_components,
            feature=D,
            affinity="precomputed",
            # try different linkage strategies
            linkage="complete",
        )
        busmap.to_csv(
            "kmeans_elec_busmap_" + str(settings["n_clusters_AC"]) + "_result.csv"
        )
        print(f"INFO::: Running Time HAC: {time.time()-a}")

    else:
        df = pd.read_csv(settings["k_busmap"])
        df = df.astype(str)
        df = df.set_index("Bus")
        busmap = df.squeeze("columns")

    return busmap


def get_attached_tech(network, components):
    """
    Function gathering all technologies attached to each bus (e. g. 'wind_onshore',
    'industrial_gas_CHP') and adding them as a new column in network.buses.

    Parameters
    ----------
    network : pypsa.Network object
        Container for all network components.
    components : set of strings
        Contains all network components from where attached technologies
        are gathered.

    Returns
    -------
    network : pypsa.Network object
        Object with two additional columns in network.buses containing the attached
        technologies for each bus and the respective key_indicator (e. g. p_nom or s_nom)
    """

    network.buses["tech"] = ""
    network.buses["key_indicator"] = ""
    # component-wise search for attached technologies
    for i in network.iterate_components(components):
        if i.name == "Link":
            a = i.df.set_index("bus0")
            a_ = a.groupby(a.index).carrier.apply(
                lambda x: ",".join(i.name + "_" + x)
            )
            b_ = a.groupby(a.index).p_nom.apply(lambda x: str(list(x)))

            network.buses.tech.loc[a_.index] += a_ + ","
            network.buses.key_indicator.loc[b_.index] += b_

            a = i.df.set_index("bus1")
            a_ = a.groupby(a.index).carrier.apply(
                lambda x: ",".join((i.name + "_" + x))
            )
            b_ = a.groupby(a.index).p_nom.apply(lambda x: str(list(x)))

            network.buses.tech.loc[a_.index] += a_ + ","
            network.buses.key_indicator.loc[b_.index] += b_

        elif i.name == "Line":
            a = i.df.set_index("bus0")
            a_ = a.groupby(a.index).carrier.apply(
                lambda x: ",".join((i.name + "_" + x))
            )
            b_ = a.groupby(a.index).s_nom.apply(lambda x: str(list(x)))

            network.buses.tech.loc[a_.index] += a_ + ","
            network.buses.key_indicator.loc[b_.index] += b_

            a = i.df.set_index("bus1")
            a_ = a.groupby(a.index).carrier.apply(
                lambda x: ",".join((i.name + "_" + x))
            )
            b_ = a.groupby(a.index).s_nom.apply(lambda x: str(list(x)))

            network.buses.tech.loc[a_.index] += a_ + ","
            network.buses.key_indicator.loc[b_.index] += b_

        elif i.name == "Load":
            a = i.df.set_index("bus")
            a_ = a.groupby(a.index).carrier.apply(
                lambda x: ",".join((i.name + "_" + x))
            )
            network.buses.tech.loc[a_.index] += a_ + ","

            b = i.df
            b['total_load'] = network.loads_t.p_set.transpose().sum(axis=1)
            b_ = b.groupby(b.bus).total_load.apply(lambda x: str(list(x)))
            network.buses.key_indicator.loc[b_.index] += b_
            
            # b.index
            # b.set_index('bus')
            # b_ = network.loads_t.p_set.transpose().sum(axis=1)
            # b_.index = b.index

            
            # b_ = a.groupby(a.index).p_nom.apply(lambda x: str(list(x)))

        
        else:
            a = i.df.set_index("bus")
            a_ = a.groupby(a.index).carrier.apply(
                lambda x: ",".join((i.name + "_" + x))
            )
            b_ = a.groupby(a.index).p_nom.apply(lambda x: str(list(x)))

            network.buses.tech.loc[a_.index] += a_ + ","
            network.buses.key_indicator.loc[b_.index] += b_

    # remove trailing commas and transfrom from a single string to list containg unique values
    network.buses.tech = (
        network.buses.tech.str.rstrip(",").str.split(",").apply(np.unique)
    )

    # remove all string related cluttering and cast to float values
    network.buses.key_indicator = network.buses.key_indicator.apply(lambda x: x.replace('[','').replace(']',',').replace(' ','')[:-1])
    network.buses.key_indicator = network.buses.key_indicator.str.split(",")
    network.buses.key_indicator = network.buses.key_indicator.apply(lambda x: [float(i) for i in x if (i != '')])

    return network

f = get_attached_tech(network,components)

# relevant buses as parameter?
def boolDistance(network, carrier, settings):
    """
    Function calculating a distance matrix based on the attached technologies
    (e. g. 'wind_onshore', 'industrial_gas_CHP') of each bus (one-hot encoded)
    and the haversine distance.

    Parameters
    ----------
    network : pypsa.Network object
        Container for all network components.
    carrier : string
        Contains relevant energycarrier ('AC' or 'CH4') for which to calculate
        the distance matrix.

    Returns
    -------
    D : numpy.ndarray
        Array with n_buses * n_buses entries containing the respective
        distance [0,1]
    """

    logger.info(f"Calculating distance matrix for {carrier} network")

    network = network.copy(with_time=False)

    # clean up network links and lines to prevent errors due to missing buses
    bus_indeces = network.buses.index
    network.lines = network.lines.loc[
        (network.lines.bus0.isin(bus_indeces)) & (network.lines.bus1.isin(bus_indeces))
    ]
    network.links = network.links.loc[
        (network.links.bus0.isin(bus_indeces)) & (network.links.bus1.isin(bus_indeces))
    ]

    components = {"Link", "Store", "StorageUnit", "Load", "Generator", "Line"}

    # Get all potential attached technologies
    tech = []
    for c in network.iterate_components(components):
        tech.extend(c.name + "_" + c.df.carrier.unique())

    network = get_attached_tech(network, components)

    # Convert attached technologies to bool array and add as new column to network.buses
    network.buses["tech_bool"] = network.buses.tech.apply(lambda x: np.isin(tech, x))

    # select relevant buses
    if carrier == "AC":
        if settings["cluster_foreign_AC"] == False:
            rel_buses = network.buses.loc[
                (network.buses.carrier == carrier) & (network.buses.country == "DE")
            ]
        else:
            rel_buses = network.buses.loc[network.buses.carrier == carrier]
    else:
        if settings["cluster_foreign_gas"] == False:
            rel_buses = network.buses.loc[
                (network.buses.carrier == carrier) & (network.buses.country == "DE")
            ]
        else:
            rel_buses = network.buses.loc[network.buses.carrier == carrier]

    # n_nodes * n_nodes distance matrix [sum(A&B)]/(min(sum(A), sum(B))]
    a = rel_buses.tech_bool.values
    a = np.array([i for i in a])
    D_ = (a[:, np.newaxis] & a).sum(axis=-1)
    D_ = D_ / D_.diagonal()
    D_ = np.maximum.reduce([np.tril(D_).T, np.triu(D_)])
    D_quality = D_ + D_.T - D_ * np.identity(D_.shape[0])
    D_quality = 1 - D_quality

    a = np.ones((len(rel_buses), 2))
    a[:, 0] = rel_buses.x.values
    a[:, 1] = rel_buses.y.values
    D_spatial = haversine(a, a)
    D_spatial_norm = (D_spatial + 1E-5) / D_spatial.max()
    # Combine distances based on attached technologies and spatial distance
    return  D_quality / D_spatial_norm


def capacityBasedDistance():
    """
    _______TO IMPLEMENT_______
    # calculate attached tech with p_nom and TS for Loads(max p_nom)
    # calculate D_quality
    # calculate D_spatial_norm (Can be packed in function to no double code with BoolDistance)
    """

    logger.info(f"Calculating distance matrix for {carrier} network")

    network = network.copy(with_time=True)

    # cleanup network TODO: DOUBLED CODE WITH BoolDistance -> make function
    bus_indeces = network.buses.index
    network.lines = network.lines.loc[
        (network.lines.bus0.isin(bus_indeces)) & (network.lines.bus1.isin(bus_indeces))
    ]
    network.links = network.links.loc[
        (network.links.bus0.isin(bus_indeces)) & (network.links.bus1.isin(bus_indeces))
    ]

    components = {"Link", "Store", "StorageUnit", "Load", "Generator", "Line"}

    # Get all potential attached technologies
    tech = []
    for c in network.iterate_components(components):
        tech.extend(c.name + "_" + c.df.carrier.unique())

    network = get_attached_tech(network, components)

    # Convert attached technologies to array containing all p (_nom) values and
    # add as new column to network.buses
    network.buses["tech_p"] = network.buses.tech.apply(lambda x: np.isin(tech, x))
    # lines, links, loads_t, generators, stores, storage_units,

    AT INDEX tech[] * p_nom

    # correct network.lines dataframe
    network.lines.index.name = 'Line'
    a = {"Link": network.links, "Line": network.lines, "Load": network.loads, "Generator": network.generators, "Store": etrago.network.stores, "Storage_Unit": etrago.network.storage_units}

    for i in a:
        print(i.index.name)
        if i.index.name in str(network.buses.tech)



    Link_ - etrago.network.links
    Line_ - etrago.network.lines
    Load_ - etrago.network.loads - etrago.network.loads_t
    Store_ - etrago.network.stores
    Generator_ - etrago.network.generators
    StorageUnit_ - etrago.network.storage_units

