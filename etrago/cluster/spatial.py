# -*- coding: utf-8 -*-
# Copyright 2016-2023 Flensburg University of Applied Sciences,
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
    from itertools import product
    from math import ceil
    import logging
    import multiprocessing as mp

    from networkx import NetworkXNoPath
    from pypsa.clustering.spatial import (
        busmap_by_kmeans,
        busmap_by_stubs,
        flatten_multiindex,
        get_clustering_from_busmap,
    )
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    import networkx as nx
    import numpy as np
    import pandas as pd
    import pypsa

    from etrago.tools.utilities import (
        buses_grid_linked,
        buses_of_vlvl,
        connected_grid_lines,
        connected_transformer,
    )

    logger = logging.getLogger(__name__)

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = (
    "MGlauer, MarlonSchlemminger, mariusves, BartelsJ, gnn, lukasoldi, "
    "ulfmueller, lukasol, ClaraBuettner, CarlosEpia, KathiEsterl, "
    "pieterhexen, fwitte, AmeliaNadal, cjbernal071421"
)

# TODO: Workaround because of agg


def _make_consense_links(x):
    """
    Ensure that all elements in the input Series `x` are identical, or that
    they are all NaN.

    Parameters
    ----------
    x : pandas.Series
        A Series containing the values to be checked for consensus.

    Returns
    -------
    object
        The value of the first element in the Series `x`.
    """

    v = x.iat[0]
    assert (
        x == v
    ).all() or x.isnull().all(), (
        f"No consense in table links column {x.name}: \n {x}"
    )
    return v


def nan_links(x):
    return np.nan


def ext_storage(x):
    v = any(x[x])
    return v


def sum_with_inf(x):
    if (x == np.inf).any():
        return np.inf
    else:
        return x.sum()


def strategies_buses():
    return {"geom": nan_links, "country": "first"}


def strategies_lines():
    return {
        "geom": nan_links,
    }


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
            "p_nom_max": sum_with_inf,
        },
        "Store": {
            "marginal_cost": np.mean,
            "capital_cost": np.mean,
            "standing_loss": np.mean,
            "e_nom": np.sum,
            "e_nom_min": np.sum,
            "e_nom_max": sum_with_inf,
            "e_initial": np.sum,
            "e_min_pu": np.mean,
            "e_max_pu": np.mean,
        },
    }


def strategies_generators():
    return {
        "p_nom_min": np.min,
        "p_nom_max": sum_with_inf,
        "weight": np.sum,
        "p_nom": np.sum,
        "p_nom_opt": np.sum,
        "marginal_cost": np.mean,
        "capital_cost": np.mean,
        "e_nom_max": sum_with_inf,
    }


def strategies_links():
    return {
        "scn_name": _make_consense_links,
        "bus0": _make_consense_links,
        "bus1": _make_consense_links,
        "carrier": _make_consense_links,
        "p_nom": np.sum,
        "p_nom_extendable": _make_consense_links,
        "p_nom_max": sum_with_inf,
        "capital_cost": np.mean,
        "length": np.mean,
        "geom": nan_links,
        "topo": nan_links,
        "type": nan_links,
        "efficiency": np.mean,
        "p_nom_min": np.sum,
        "p_set": np.mean,
        "p_min_pu": np.min,
        "p_max_pu": np.max,
        "marginal_cost": np.mean,
        "terrain_factor": _make_consense_links,
        "p_nom_opt": np.mean,
        "country": nan_links,
        "build_year": np.mean,
        "lifetime": np.mean,
        "min_up_time": np.mean,
        "min_down_time": np.mean,
        "up_time_before": np.mean,
        "down_time_before": np.mean,
        "committable": np.all,
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
        Describe which type of carriers should be aggregated. The default is
        None.
    strategies : dictionary
        custom strategies to perform the aggregation

    Returns
    -------
    new_df :
        links aggregated based on bus0, bus1 and carrier
    new_pnl :
        links time series aggregated
    """

    def normed_or_uniform(x):
        return (
            x / x.sum()
            if x.sum(skipna=False) > 0
            else pd.Series(1.0 / len(x), x.index)
        )

    def arrange_dc_bus0_bus1(network):
        dc_links = network.links[network.links.carrier == "DC"].copy()
        dc_links["n0"] = dc_links.apply(
            lambda x: x.bus0 if x.bus0 < x.bus1 else x.bus1, axis=1
        )
        dc_links["n1"] = dc_links.apply(
            lambda x: x.bus0 if x.bus0 > x.bus1 else x.bus1, axis=1
        )
        dc_links["bus0"] = dc_links["n0"]
        dc_links["bus1"] = dc_links["n1"]
        dc_links.drop(columns=["n0", "n1"], inplace=True)

        network.links.drop(index=dc_links.index, inplace=True)
        network.links = pd.concat([network.links, dc_links])

        return network

    network = arrange_dc_bus0_bus1(network)

    if carriers is None:
        carriers = network.links.carrier.unique()

    links_agg_b = network.links.carrier.isin(carriers)
    links = network.links.loc[links_agg_b]
    grouper = [links.bus0, links.bus1, links.carrier]

    weighting = links.p_nom.groupby(grouper, axis=0).transform(
        normed_or_uniform
    )
    strategies = strategies_links()
    strategies.update(cus_strateg)
    strategies.pop("topo")
    strategies.pop("geom")

    new_df = links.groupby(grouper, axis=0).agg(strategies)
    new_df.index = flatten_multiindex(new_df.index).rename("name")
    new_df = pd.concat(
        [new_df, network.links.loc[~links_agg_b]], axis=0, sort=False
    )
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
                    df_agg = df_agg.multiply(
                        weighting.loc[df_agg.columns], axis=1
                    )
                pnl_df = df_agg.groupby(grouper, axis=1).sum()
                pnl_df.columns = flatten_multiindex(pnl_df.columns).rename(
                    "name"
                )
                new_pnl[attr] = pd.concat(
                    [df.loc[:, ~pnl_links_agg_b], pnl_df], axis=1, sort=False
                )
                new_pnl[attr].columns = new_pnl[attr].columns.map(cluster_id)
            else:
                new_pnl[attr] = network.links_t[attr]

    new_pnl = pypsa.descriptors.Dict(new_pnl)

    return new_df, new_pnl


def graph_from_edges(edges):
    """
    Constructs an undirected multigraph from a list containing data on
    weighted edges.

    Parameters
    ----------
    edges : list
        List of tuples each containing first node, second node, weight, key.

    Returns
    -------
    M : :class:`networkx.classes.multigraph.MultiGraph`
    """

    M = nx.MultiGraph()

    for e in edges:
        n0, n1, weight, key = e

        M.add_edge(n0, n1, weight=weight, key=key)

    return M


def gen(nodes, n, graph):
    # TODO There could be a more convenient way of doing this. This generators
    # single purpose is to prepare data for multiprocessing's starmap function.
    """
    Generator for applying multiprocessing.

    Parameters
    ----------
    nodes : list
        List of nodes in the system.
    n : int
        Number of desired multiprocessing units.
    graph : :class:`networkx.classes.multigraph.MultiGraph`
        Graph representation of an electrical grid.

    Returns
    -------
    None
    """

    g = graph.copy()

    for i in range(0, len(nodes), n):
        yield (nodes[i : i + n], g)


def shortest_path(paths, graph):
    """
    Finds the minimum path lengths between node pairs defined in paths.

    Parameters
    ----------
    paths : list
        List of pairs containing a source and a target node
    graph : :class:`networkx.classes.multigraph.MultiGraph`
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
        while df_isna.loc[(s, t), "path_length"]:
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


def busmap_by_shortest_path(etrago, fromlvl, tolvl, cpu_cores=4):
    """
    Creates a busmap for the EHV-Clustering between voltage levels based
    on dijkstra shortest path. The result is automatically written to the
    `model_draft` on the <OpenEnergyPlatform>[www.openenergy-platform.org]
    database with the name `ego_grid_pf_hv_busmap` and the attributes scn_name
    (scenario name), bus0 (node before clustering), bus1 (node after
    clustering) and path_length (path length).
    An AssertionError occurs if buses with a voltage level are not covered by
    the input lists 'fromlvl' or 'tolvl'.

    Parameters
    ----------
    network : pypsa.Network
        Container for all network components.
    session : sqlalchemy.orm.session.Session object
        Establishes interactions with the database.
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

    # data preperation
    s_buses = buses_grid_linked(etrago.network, fromlvl)
    lines = connected_grid_lines(etrago.network, s_buses)
    transformer = connected_transformer(etrago.network, s_buses)
    mask = transformer.bus1.isin(buses_of_vlvl(etrago.network, tolvl))

    dc = etrago.network.links[etrago.network.links.carrier == "DC"]
    dc.index = "DC_" + dc.index
    lines_plus_dc = pd.concat([lines, dc])
    lines_plus_dc = lines_plus_dc[etrago.network.lines.columns]
    lines_plus_dc["carrier"] = "AC"

    # temporary end points, later replaced by bus1 pendant
    t_buses = transformer[mask].bus0

    # create all possible pathways
    ppaths = list(product(s_buses, t_buses))

    # graph creation
    edges = [
        (row.bus0, row.bus1, row.length, ix)
        for ix, row in lines_plus_dc.iterrows()
    ]
    M = graph_from_edges(edges)

    # applying multiprocessing
    p = mp.Pool(cpu_cores)

    chunksize = ceil(len(ppaths) / cpu_cores)
    container = p.starmap(shortest_path, gen(ppaths, chunksize, M))
    df = pd.concat(container)

    # post processing
    df.sort_index(inplace=True)
    df = df.fillna(10000000)

    mask = df.groupby(level="source")["path_length"].idxmin()
    df = df.loc[mask, :]

    # rename temporary endpoints
    df.reset_index(inplace=True)
    df.target = df.target.map(
        dict(
            zip(
                etrago.network.transformers.bus0,
                etrago.network.transformers.bus1,
            )
        )
    )

    # append to busmap buses only connected to transformer
    transformer = etrago.network.transformers
    idx = list(
        set(buses_of_vlvl(etrago.network, fromlvl)).symmetric_difference(
            set(s_buses)
        )
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

    df.rename(columns={"source": "bus0", "target": "bus1"}, inplace=True)

    busmap = pd.Series(df.bus1.values, index=df.bus0).to_dict()

    return busmap


def busmap_ehv_clustering(etrago):
    """
    Generates a busmap that can be used to cluster an electrical network to
    only extra high voltage buses. If a path to a busmap in a csv file is
    passed in the arguments, it loads the csv file and returns it.

    Parameters
    ----------
    etrago : Etrago
        An instance of the Etrago class

    Returns
    -------
    busmap : dict
        Maps old bus_ids to new bus_ids.
    """

    if etrago.args["network_clustering_ehv"]["busmap"] is False:
        cpu_cores = etrago.args["network_clustering"]["CPU_cores"]
        if cpu_cores == "max":
            cpu_cores = mp.cpu_count()
        else:
            cpu_cores = int(cpu_cores)

        busmap = busmap_by_shortest_path(
            etrago,
            fromlvl=[110],
            tolvl=[220, 380, 400, 450],
            cpu_cores=cpu_cores,
        )
        pd.DataFrame(busmap.items(), columns=["bus0", "bus1"]).to_csv(
            "ehv_elecgrid_busmap_result.csv",
            index=False,
        )
    else:
        busmap = pd.read_csv(etrago.args["network_clustering_ehv"]["busmap"])
        busmap = pd.Series(
            busmap.bus1.apply(str).values, index=busmap.bus0.apply(str)
        ).to_dict()

    return busmap


def kmean_clustering(etrago, selected_network, weight, n_clusters):
    """
    Main function of the k-mean clustering approach. Maps an original
    network to a new one with adjustable number of nodes and new coordinates.

    Parameters
    ----------
    network : pypsa.Network
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
    network : pypsa.Network
        Container for all network components.
    """
    network = etrago.network
    kmean_settings = etrago.args["network_clustering"]

    with threadpool_limits(limits=kmean_settings["CPU_cores"], user_api=None):
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
                network.buses.loc[busmap.index, ["x", "y"]] = (
                    network.buses.loc[busmap, ["x", "y"]].values
                )

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
        busmap = busmap_by_kmeans(
            selected_network,
            bus_weightings=pd.Series(weight),
            n_clusters=n_clusters,
            n_init=kmean_settings["n_init"],
            max_iter=kmean_settings["max_iter"],
            tol=kmean_settings["tol"],
            random_state=kmean_settings["random_state"],
        )

    return busmap


def dijkstras_algorithm(buses, connections, medoid_idx, cpu_cores):
    """
    Function for combination of k-medoids Clustering and Dijkstra's algorithm.
    Creates a busmap assigning the nodes of a original network to the nodes of
    a clustered network considering the electrical distances based on
    Dijkstra's shortest path.

    Parameters
    ----------
    network : pypsa.Network
        Container for all network components.
    medoid_idx : pandas.Series
        Indices of k-medoids
    busmap_kmedoid: pandas.Series
        Busmap based on k-medoids clustering
    cpu_cores: string
        numbers of cores used during multiprocessing

    Returns
    -------
    busmap : pandas.Series
        Mapping from bus ids to medoids ids
    """

    # original data
    o_buses = buses.index
    # k-medoids centers
    medoid_idx = medoid_idx.astype("str")
    c_buses = medoid_idx.tolist()

    # list of all possible pathways
    ppathss = list(product(o_buses, c_buses))

    # graph creation
    edges = [
        (row.bus0, row.bus1, row.length, ix)
        for ix, row in connections.iterrows()
    ]
    M = graph_from_edges(edges)

    # processor count
    if cpu_cores == "max":
        cpu_cores = mp.cpu_count()
    else:
        cpu_cores = int(cpu_cores)

    # calculation of shortest path between original points and k-medoids
    # centers using multiprocessing
    p = mp.Pool(cpu_cores)
    chunksize = ceil(len(ppathss) / cpu_cores)
    container = p.starmap(shortest_path, gen(ppathss, chunksize, M))
    df = pd.concat(container)

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


def kmedoids_dijkstra_clustering(
    etrago, buses, connections, weight, n_clusters
):
    """
    Applies a k-medoids clustering on the given network and calls the function
    to conduct a Dijkstra's algorithm afterwards for the consideration of the
    network's topology in the spatial clustering.

    Parameters
    ----------
    etrago : Etrago
        An instance of the Etrago class
    buses : pandas.DataFrame
        DataFrame with information about the buses of the network.
    connections : pandas.DataFrame
        DataFrame with information about the connections of the network
        (links or lines).
    weight : pandas.Series
        Series with the weight for each bus.
    n_clusters : int
        The number of clusters to create.

    Returns
    -------
    Tuple containing:
    busmap : pandas.Series
        Series containing the mapping of buses to their resp. medoids
    medoid_idx : pandas.Series
        Series containing the medoid indeces
    """

    settings = etrago.args["network_clustering"]

    # n_jobs was deprecated for the function fit(). scikit-learn recommends
    # to use threadpool_limits:
    # https://scikit-learn.org/stable/computing/parallelism.html
    with threadpool_limits(limits=settings["CPU_cores"], user_api=None):
        # remove stubs
        if settings["remove_stubs"]:
            logger.info(
                """options remove_stubs and use_reduced_coordinates not
                reasonable for k-medoids Dijkstra Clustering"""
            )

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

        if len(busmap) > n_clusters:
            # dijkstra's algorithm
            busmap = dijkstras_algorithm(
                buses,
                connections,
                medoid_idx,
                etrago.args["network_clustering"]["CPU_cores"],
            )
        elif len(busmap) < n_clusters:
            logger.warning(
                f"""
            The number supplied to the parameter n_clusters for
            {buses.carrier[0]} buses is larger than the actual number of buses
            in the network.
            """
            )

        busmap.index.name = "bus_id"

    return busmap, medoid_idx


def drop_nan_values(network):
    """
    Drops nan values after clustering an replaces output data time series with
    empty dataframes

    Parameters
    ----------
    network : pypsa.Network
        Container for all network components.

    Returns
    -------
    None.

    """

    # Drop nan values after clustering
    network.links.min_up_time.fillna(0, inplace=True)
    network.links.min_down_time.fillna(0, inplace=True)
    network.links.up_time_before.fillna(0, inplace=True)
    network.links.down_time_before.fillna(0, inplace=True)
    # Drop nan values in timeseries after clustering
    for c in network.iterate_components():
        for pnl in c.attrs[
            (c.attrs.status == "Output") & (c.attrs.varying)
        ].index:
            c.pnl[pnl] = pd.DataFrame(index=network.snapshots)
