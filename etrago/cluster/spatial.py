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
"""spatial.py defines the methods to run spatial clustering on networks."""

import json
import logging
import os

if "READTHEDOCS" not in os.environ:
    from itertools import product
    from math import ceil
    import multiprocessing as mp

    from networkx import NetworkXNoPath
    from pypsa.clustering.spatial import (
        busmap_by_kmeans,
        busmap_by_stubs,
        flatten_multiindex,
        get_clustering_from_busmap,
    )
    from shapely.geometry import Point
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    import geopandas as gpd
    import networkx as nx
    import numpy as np
    import pandas as pd
    import pypsa
    import saio

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


def ext_sto(x):
    v = any(x[x])
    return v


def sum_with_inf(x):
    if (x == np.inf).any():
        return np.inf
    else:
        return x.sum()


def strategies_buses():
    return {
        "geom": nan_links,
        "country": "first",
        "scn_name": "first",
    }


def strategies_lines():
    return {
        "geom": nan_links,
        "country": "first",
        "scn_name": "first",
        "cables": "sum",
        "topo": nan_links,
        "total_cables": "sum",
    }


def strategies_one_ports():
    return {
        "StorageUnit": {
            "scn_name": "first",
            "p_nom_extendable": ext_sto,
            "p_nom_max": sum_with_inf,
            "capital_cost": "capacity_weighted_average",
            "marginal_cost": "capacity_weighted_average",
            "p_min_pu": "capacity_weighted_average",
            "p_max_pu": "capacity_weighted_average",
            "state_of_charge_initial": "sum",
            "standing_loss": "capacity_weighted_average",
            "efficiency_dispatch": "capacity_weighted_average",
            "efficiency_store": "capacity_weighted_average",
            "max_hours": "mean",
        },
        "Store": {
            "scn_name": "first",
            "e_nom_extendable": ext_sto,
            "e_nom_max": sum_with_inf,
            "capital_cost": "capacity_weighted_average",
            "marginal_cost": "capacity_weighted_average",
            "e_initial": "sum",
            "e_initial_per_period": "sum",
            "e_min_pu": "capacity_weighted_average",
            "e_max_pu": "capacity_weighted_average",
            "standing_loss": "capacity_weighted_average",
        },
    }


def strategies_generators():
    return {
        "scn_name": "first",
        "up_time_before": "capacity_weighted_average",
        "p_nom_extendable": "any",
        "capital_cost": "capacity_weighted_average",
        "marginal_cost": "capacity_weighted_average",
        "p_min_pu": "capacity_weighted_average",
        "p_max_pu": "capacity_weighted_average",
    }


def strategies_links():
    return {
        "scn_name": "first",
        "bus0": _make_consense_links,
        "bus1": _make_consense_links,
        "length": "capacity_weighted_average",
        "carrier": _make_consense_links,
        "p_nom": "sum",
        "p_nom_extendable": "any",
        "p_nom_opt": "sum",
        "p_nom_max": sum_with_inf,
        "p_nom_min": "sum",
        "p_min_pu": "capacity_weighted_average",
        "p_max_pu": "capacity_weighted_average",
        "p_set": "sum",
        "efficiency": "capacity_weighted_average",
        "capital_cost": "capacity_weighted_average",
        "marginal_cost": "capacity_weighted_average",
        "marginal_cost_quadratic": "capacity_weighted_average",
        "min_up_time": "capacity_weighted_average",
        "min_down_time": "capacity_weighted_average",
        "up_time_before": "capacity_weighted_average",
        "down_time_before": "capacity_weighted_average",
        "committable": "all",
        "type": nan_links,
        "geom": nan_links,
        "topo": nan_links,
        "country": nan_links,
        "terrain_factor": _make_consense_links,
        "build_year": "capacity_weighted_average",
        "lifetime": "capacity_weighted_average",
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

    def align_strategies(strategies, keys):
        # Aligns the given strategies with the given keys.
        strategies |= {
            k: _make_consense_links for k in set(keys).difference(strategies)
        }
        return {k: strategies[k] for k in keys}

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

    strategies = strategies_links()
    strategies.update(cus_strateg)
    columns = links.columns
    static_strategies = align_strategies(strategies, columns)

    capacity_weights = links.p_nom.groupby(grouper, axis=0).transform(
        normed_or_uniform
    )
    for k, v in static_strategies.items():
        if v == "capacity_weighted_average":
            links[k] = links[k] * capacity_weights
            static_strategies[k] = "sum"

    new_df = links.groupby(grouper).agg(static_strategies)
    new_df.index = flatten_multiindex(new_df.index).rename("name")
    new_df = pd.concat(
        [new_df, network.links.loc[~links_agg_b]], axis=0, sort=False
    )
    new_df["new_id"] = np.arange(len(new_df)).astype(str)
    cluster_id = new_df["new_id"].to_dict()
    new_df.set_index("new_id", inplace=True)
    new_df.index = new_df.index.rename("Link")

    # timeseries
    new_pnl = dict()
    if with_time:
        dynamic_strategies = align_strategies(strategies, network.pnl("Link"))
        for attr, df in network.pnl("Link").items():
            pnl_links_agg_b = df.columns.to_series().map(links_agg_b)
            df_agg = df.loc[:, pnl_links_agg_b].astype(float)
            if not df_agg.empty:
                df_agg = network.get_switchable_as_dense("Link", attr)
                strategy = dynamic_strategies[attr]
                if strategy == "capacity_weighted_average":
                    df_agg = df_agg * capacity_weights
                    pnl_df = df_agg.T.groupby(grouper).sum().T
                else:
                    pnl_df = df_agg.T.groupby(grouper).agg(strategy).T
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
        cpu_cores = etrago.args["network_clustering_ehv"]["cpu_cores"]
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

    Returns
    -------
    network : pypsa.Network
        Container for all network components.
    """
    network = etrago.network
    kmean_settings = etrago.args["network_clustering"]

    with threadpool_limits(
        limits=kmean_settings["method"]["cpu_cores"], user_api=None
    ):
        # remove stubs
        if kmean_settings["method"]["remove_stubs"]:
            network.determine_network_topology()
            busmap = busmap_by_stubs(network)

            # reset coordinates to the new reduced guys, rather than taking an
            # average (copied from pypsa.networkclustering)
            if kmean_settings["method"]["use_reduced_coordinates"]:
                # TODO : FIX THIS HACK THAT HAS UNEXPECTED SIDE-EFFECTS,
                # i.e. network is changed in place!!
                network.buses.loc[busmap.index, ["x", "y"]] = (
                    network.buses.loc[busmap, ["x", "y"]].values
                )

            clustering = get_clustering_from_busmap(
                network,
                busmap,
                aggregate_generators_weighted=True,
                generator_strategies=strategies_generators(),
                one_port_strategies=strategies_one_ports(),
                aggregate_one_ports=network.one_port_components.copy(),
                bus_strategies=strategies_buses(),
                line_strategies=strategies_lines(),
                line_length_factor=1,
            )

            etrago.network = clustering.network

            weight = weight.groupby(busmap.values).sum()

        # k-mean clustering
        busmap = busmap_by_kmeans(
            selected_network,
            bus_weightings=pd.Series(weight),
            n_clusters=n_clusters,
            n_init=kmean_settings["method"]["n_init"],
            max_iter=kmean_settings["method"]["max_iter"],
            tol=kmean_settings["method"]["tol"],
            random_state=kmean_settings["method"]["random_state"],
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
    with threadpool_limits(
        limits=settings["method"]["cpu_cores"], user_api=None
    ):
        # remove stubs
        if settings["method"]["remove_stubs"]:
            logger.info("""options remove_stubs and use_reduced_coordinates not
                reasonable for k-medoids Dijkstra Clustering""")

        bus_weightings = pd.Series(weight)
        buses_i = buses.index
        points = buses.loc[buses_i, ["x", "y"]].values.repeat(
            bus_weightings.reindex(buses_i).astype(int), axis=0
        )

        kmeans = KMeans(
            init="k-means++",
            n_clusters=n_clusters,
            n_init=settings["method"]["n_init"],
            max_iter=settings["method"]["max_iter"],
            tol=settings["method"]["tol"],
            random_state=settings["method"]["random_state"],
        )
        kmeans.fit(points)

        busmap = pd.Series(
            data=kmeans.predict(buses.loc[buses_i, ["x", "y"]].values),
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
                etrago.args["network_clustering"]["method"]["cpu_cores"],
            )
        elif len(busmap) < n_clusters:
            logger.warning(f"""
            The number supplied to the parameter n_clusters for
            {buses.carrier[0]} buses is larger than the actual number of buses
            in the network.
            """)

        busmap.index.name = "bus_id"

    return busmap, medoid_idx


def focus_protection_requested(cluster_within):
    """Return whether buses inside the focus region must stay unclustered.

    ``False`` is the documented switch for protecting the focus region.  A
    positive numeric value is treated the same way for compatibility with
    configurations that previously used a numeric focus threshold.  ``True``
    and ``None`` retain the original weighting-only behaviour.
    """
    if cluster_within is False:
        return True

    if isinstance(cluster_within, (int, float)) and not isinstance(
        cluster_within, bool
    ):
        return cluster_within > 0

    return False


def buses_inside_focus_region(etrago, network, focus_region):
    """Return bus indices located strictly inside the configured focus."""
    if isinstance(focus_region, list):
        if "oep.iks.cs.ovgu.de" in str(etrago.engine.url):
            saio.register_schema("tables", etrago.engine)
            from saio.tables import edut_00_012 as vg250_krs
        else:
            saio.register_schema("boundaries", etrago.engine)
            from saio.boundaries import vg250_krs

        query = etrago.session.query(vg250_krs)
        focus_gdf = saio.as_pandas(query, geometry="geometry")
        missing = set(focus_region) - set(focus_gdf["gen"])
        if missing:
            raise ValueError(
                f"The following focus_region entries are not valid: {missing}"
            )
        focus_gdf = focus_gdf[focus_gdf["gen"].isin(focus_region)]
    else:
        focus_gdf = gpd.read_file(focus_region)

    buses_df = network.buses[["x", "y"]].copy()
    buses_df["geometry"] = buses_df.apply(
        lambda row: Point(row["x"], row["y"]), axis=1
    )
    buses_gdf = gpd.GeoDataFrame(
        buses_df, geometry="geometry", crs=4326
    ).to_crs(epsg=25832)

    if focus_gdf.crs is None:
        raise ValueError("CRS of your focus region must be imported.")

    focus_polygon = focus_gdf.to_crs(buses_gdf.crs).geometry.unary_union
    return buses_gdf.index[buses_gdf.geometry.within(focus_polygon)]


def _export_focus_buses(
    etrago, focus_buses, filename="buses_within_focus.csv"
):
    """Export focus buses when a result path is configured."""
    export_path = etrago.args.get("export_results_path")
    if not export_path:
        return

    os.makedirs(export_path, exist_ok=True)
    focus_buses.to_csv(os.path.join(export_path, filename))

def _allocate_component_clusters(components, weight, n_clusters):
    """Allocate an exact cluster budget across connected components."""
    n_clusters = int(n_clusters)
    n_components = len(components)
    n_buses = sum(len(component) for component in components)

    if n_clusters < n_components:
        raise ValueError(
            f"Cannot create {n_clusters} clusters for {n_components} "
            "disconnected components. Increase the regional cluster budget."
        )

    if n_clusters > n_buses:
        raise ValueError(
            f"Cannot create {n_clusters} clusters from only {n_buses} buses."
        )

    allocation = [1] * n_components
    masses = []

    for component in components:
        component_weight = pd.to_numeric(
            weight.reindex(component), errors="coerce"
        ).fillna(1)

        mass = float(component_weight.clip(lower=1).sum())

        masses.append(
            mass if np.isfinite(mass) and mass > 0 else len(component)
        )

    remaining = n_clusters - n_components

    while remaining:
        candidates = [
            index
            for index, component in enumerate(components)
            if allocation[index] < len(component)
        ]

        if not candidates:
            raise RuntimeError(
                "No component has capacity for the cluster budget."
            )

        selected = max(
            candidates,
            key=lambda index: (
                masses[index] / (allocation[index] + 1),
                -index,
            ),
        )

        allocation[selected] += 1
        remaining -= 1

    return allocation


def _cluster_partition_kmedoids(
    etrago,
    buses,
    connections,
    weight,
    n_clusters,
    label_offset,
    partition_name,
):
    """Cluster one region with an exact cluster budget."""
    if buses.empty:
        raise ValueError(
            f"The {partition_name} partition contains no buses."
        )

    # Include every bus, including buses without an internal line.
    graph = nx.Graph()
    graph.add_nodes_from(buses.index)

    # Only retain lines whose two endpoints belong to this partition.
    internal_connections = connections[
        connections.bus0.isin(buses.index)
        & connections.bus1.isin(buses.index)
    ].copy()

    graph.add_edges_from(
        internal_connections[["bus0", "bus1"]].itertuples(
            index=False,
            name=None,
        )
    )

    # Treat disconnected components separately.
    components = [
        pd.Index(component)
        for component in nx.connected_components(graph)
    ]

    # Ensure deterministic component ordering.
    components.sort(
        key=lambda component: min(map(str, component))
    )

    allocation = _allocate_component_clusters(
        components,
        weight,
        n_clusters,
    )

    busmaps = []
    medoids = []
    next_label = int(label_offset)

    for component, component_clusters in zip(
        components,
        allocation,
    ):
        component_buses = buses.loc[component].copy()

        component_connections = internal_connections[
            internal_connections.bus0.isin(component)
            & internal_connections.bus1.isin(component)
        ].copy()

        # If every bus becomes a cluster, use identity-like mappings.
        # This also handles isolated single buses.
        if component_clusters == len(component_buses):
            labels = list(
                range(
                    next_label,
                    next_label + len(component_buses),
                )
            )

            component_busmap = pd.Series(
                [str(label) for label in labels],
                index=component_buses.index.astype(str),
                name="cluster",
                dtype=str,
            )

            component_medoid = pd.Series(
                component_buses.index.astype(str),
                index=labels,
                dtype=object,
            )

        else:
            component_weight = pd.to_numeric(
                weight.reindex(component_buses.index),
                errors="coerce",
            ).fillna(1).clip(lower=1)

            (
                component_busmap,
                component_medoid,
            ) = kmedoids_dijkstra_clustering(
                etrago,
                component_buses,
                component_connections,
                component_weight,
                component_clusters,
            )

            # Replace the local labels 0...n with globally unique labels.
            local_labels = sorted(
                component_busmap.astype(int).unique().tolist()
            )

            label_map = {
                local_label: next_label + position
                for position, local_label in enumerate(local_labels)
            }

            component_busmap = (
                component_busmap.astype(int)
                .map(label_map)
                .astype(str)
            )

            component_busmap.name = "cluster"

            component_medoid.index = [
                label_map[int(label)]
                for label in component_medoid.index
            ]

        busmaps.append(component_busmap)
        medoids.append(component_medoid)

        next_label += component_clusters

    busmap = pd.concat(busmaps)
    busmap.index = busmap.index.astype(str)
    busmap.index.name = "bus_id"

    medoid_idx = pd.concat(medoids)

    # Ensure that every source bus occurs exactly once.
    if (
        busmap.index.duplicated().any()
        or len(busmap) != len(buses)
    ):
        raise RuntimeError(
            f"The {partition_name} busmap is incomplete "
            "or contains duplicates."
        )

    # Ensure that the requested number was produced.
    if busmap.nunique() != int(n_clusters):
        raise RuntimeError(
            f"The {partition_name} partition produced "
            f"{busmap.nunique()} clusters instead of "
            f"{n_clusters}."
        )

    return busmap, medoid_idx

def _first_free_numeric_cluster_label(etrago):
    """Return a label above every existing numeric bus identifier."""
    numeric_ids = pd.to_numeric(
        pd.Index(etrago.network.buses.index),
        errors="coerce",
    )

    numeric_ids = np.asarray(numeric_ids, dtype=float)
    numeric_ids = numeric_ids[np.isfinite(numeric_ids)]

    if not len(numeric_ids):
        return 0

    return max(
        0,
        int(numeric_ids.max()) + 1,
    )


def exact_focus_kmedoids_clustering(
    etrago,
    selected_network,
    weight,
    n_clusters,
    focus_region,
    n_clusters_focus,
    connection_component="lines",
    export_filename="buses_within_focus.csv",
):
    """Cluster the focus and outside-focus regions separately.

    Parameters
    ----------
    n_clusters
        German cluster budget after eTraGo has reserved the foreign
        clusters.
    n_clusters_focus
        Exact number of clusters inside the focus region.
    """
    if isinstance(n_clusters_focus, bool):
        raise ValueError(
            "n_clusters_focus must be an integer, not a boolean."
        )

    try:
        converted_focus_clusters = int(n_clusters_focus)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "n_clusters_focus must be an integer."
        ) from exc

    if converted_focus_clusters != n_clusters_focus:
        raise ValueError(
            "n_clusters_focus must be an integer."
        )

    n_clusters = int(n_clusters)
    n_clusters_focus = converted_focus_clusters
    n_clusters_outside = n_clusters - n_clusters_focus

    # Determine which German buses are inside Schleswig-Holstein.
    focus_ids = buses_inside_focus_region(
        etrago,
        selected_network,
        focus_region,
    )

    focus_ids = selected_network.buses.index[
        selected_network.buses.index.isin(focus_ids)
    ]

    outside_ids = selected_network.buses.index[
        ~selected_network.buses.index.isin(focus_ids)
    ]

    if n_clusters_focus < 1:
        raise ValueError(
            "n_clusters_focus must be at least 1."
        )

    if n_clusters_outside < 1:
        raise ValueError(
            "n_clusters_focus must be smaller than the German "
            "cluster budget remaining after foreign-country "
            "clusters are reserved."
        )

    if n_clusters_focus > len(focus_ids):
        raise ValueError(
            f"n_clusters_focus={n_clusters_focus} exceeds the "
            f"{len(focus_ids)} focus buses."
        )

    if n_clusters_outside > len(outside_ids):
        raise ValueError(
            f"The outside-focus budget {n_clusters_outside} "
            f"exceeds the {len(outside_ids)} outside-focus buses."
        )

    focus_buses = selected_network.buses.loc[focus_ids].copy()
    outside_buses = selected_network.buses.loc[outside_ids].copy()

    connections = getattr(
        selected_network,
        connection_component,
    )

    # Create labels that cannot collide with existing foreign bus IDs.
    label_offset = _first_free_numeric_cluster_label(etrago)

    # Keep the original focus-bus export.
    _export_focus_buses(
        etrago,
        focus_buses,
        export_filename,
    )

    logger.info(
        "Exact AC focus clustering active: clustering %s focus "
        "buses into %s clusters and %s outside-focus buses into "
        "%s clusters (German budget %s; first new label %s).",
        len(focus_buses),
        n_clusters_focus,
        len(outside_buses),
        n_clusters_outside,
        n_clusters,
        label_offset,
    )

    # Cluster Schleswig-Holstein.
    focus_busmap, focus_medoids = (
        _cluster_partition_kmedoids(
            etrago,
            focus_buses,
            connections,
            weight,
            n_clusters_focus,
            label_offset=label_offset,
            partition_name="focus",
        )
    )

    # Cluster the rest of Germany.
    outside_busmap, outside_medoids = (
        _cluster_partition_kmedoids(
            etrago,
            outside_buses,
            connections,
            weight,
            n_clusters_outside,
            label_offset=label_offset + n_clusters_focus,
            partition_name="outside-focus",
        )
    )

    # Export the focus-region busmap for verification.
    export_path = etrago.args.get("export_results_path")

    if export_path:
        os.makedirs(export_path, exist_ok=True)

        focus_busmap.to_csv(
            os.path.join(
                export_path,
                "busmap_within_focus.csv",
            )
        )

    busmap = pd.concat(
        [
            focus_busmap,
            outside_busmap,
        ]
    )

    medoid_idx = pd.concat(
        [
            focus_medoids,
            outside_medoids,
        ]
    )

    logger.info(
        "Exact focus clustering produced %s focus targets and "
        "%s outside-focus targets.",
        focus_busmap.nunique(),
        outside_busmap.nunique(),
    )

    return busmap, medoid_idx

def split_focus_buses_from_clustering(
    etrago,
    selected_network,
    weight,
    n_clusters,
    focus_region,
    cluster_within,
    connection_component="lines",
    export_filename="buses_within_focus.csv",
):
    """Separate protected focus buses from clustering candidates.

    Focus buses are mapped to themselves.  Outside-focus buses which become
    isolated after removing the focus-region connections are protected as
    well, because Dijkstra clustering cannot assign disconnected singletons.
    The returned network contains only buses that should actually be
    clustered.
    """
    if not focus_protection_requested(cluster_within):
        raise ValueError(
            "split_focus_buses_from_clustering requires focus protection"
        )

    focus_ids = buses_inside_focus_region(
        etrago, selected_network, focus_region
    )
    focus_ids = selected_network.buses.index[
        selected_network.buses.index.isin(focus_ids)
    ]
    focus_buses = selected_network.buses.loc[focus_ids].copy()
    _export_focus_buses(etrago, focus_buses, export_filename)

    cluster_network = selected_network.copy(with_time=False)
    outside_ids = cluster_network.buses.index[
        ~cluster_network.buses.index.isin(focus_ids)
    ]

    connections = getattr(cluster_network, connection_component).copy()
    connections = connections[
        connections.bus0.isin(outside_ids)
        & connections.bus1.isin(outside_ids)
    ].copy()

    connected_ids = pd.Index(
        pd.unique(pd.concat([connections.bus0, connections.bus1]))
    )
    isolated_ids = outside_ids[~outside_ids.isin(connected_ids)]
    isolated_buses = cluster_network.buses.loc[isolated_ids].copy()

    cluster_ids = outside_ids[~outside_ids.isin(isolated_ids)]
    cluster_network.buses = cluster_network.buses.loc[cluster_ids].copy()
    connections = connections[
        connections.bus0.isin(cluster_ids)
        & connections.bus1.isin(cluster_ids)
    ].copy()
    setattr(cluster_network, connection_component, connections)

    protected_buses = pd.concat([focus_buses, isolated_buses])

    # If there are not enough remaining buses to perform a meaningful
    # clustering, retain all of them as identity mappings as well.
    if int(n_clusters) <= 0 or len(cluster_network.buses) <= int(n_clusters):
        protected_buses = pd.concat(
            [protected_buses, cluster_network.buses]
        )
        cluster_network.buses = cluster_network.buses.iloc[0:0].copy()
        setattr(
            cluster_network,
            connection_component,
            connections.iloc[0:0].copy(),
        )
        cluster_n_clusters = 0
    else:
        cluster_n_clusters = int(n_clusters)

    protected_buses = protected_buses[
        ~protected_buses.index.duplicated(keep="first")
    ]
    protected_buses.attrs["focus_count"] = len(focus_buses)
    protected_buses.attrs["isolated_count"] = len(isolated_buses)
    protected_busmap = pd.Series(
        protected_buses.index.astype(str),
        index=protected_buses.index.astype(str),
        name="cluster",
        dtype=str,
    )
    protected_busmap.index.name = "bus_id"

    cluster_weight = weight.reindex(cluster_network.buses.index).fillna(1)

    logger.info(
        "Focus-region bus protection active: protected %s focus buses, "
        "protected %s isolated outside-focus buses, clustering %s "
        "connected outside-focus buses into %s clusters.",
        len(focus_buses),
        len(isolated_buses),
        len(cluster_network.buses),
        cluster_n_clusters,
    )

    return (
        cluster_network,
        cluster_weight,
        cluster_n_clusters,
        protected_busmap,
        protected_buses,
    )


def clean_network_after_focus_clustering(network):
    """Apply conservative consistency fixes after focus-aware clustering."""
    if network.lines.empty:
        logger.info("Focus clustering cleanup finished.")
        return

    missing_endpoint = ~(
        network.lines.bus0.isin(network.buses.index)
        & network.lines.bus1.isin(network.buses.index)
    )
    self_loops = network.lines.bus0 == network.lines.bus1
    remove_lines = network.lines.index[missing_endpoint | self_loops]
    if len(remove_lines):
        logger.warning(
            "Focus cleanup: removing %s lines with invalid endpoints or "
            "self-loops.",
            len(remove_lines),
        )
        network.mremove("Line", remove_lines)

    for column in ["x", "r"]:
        values = pd.to_numeric(network.lines[column], errors="coerce")
        valid = np.isfinite(values) & (values > 0)
        invalid = ~valid
        if invalid.any():
            if not valid.any():
                raise ValueError(
                    f"No positive finite line {column} values remain after "
                    "focus clustering."
                )
            fallback = float(values.loc[valid].median())
            logger.warning(
                "Focus cleanup: fixing %s invalid line %s values with "
                "median fallback %s.",
                int(invalid.sum()),
                column,
                fallback,
            )
            network.lines.loc[invalid, column] = fallback

    network.determine_network_topology()
    logger.info("Focus clustering cleanup finished.")


def focus_weighting(
    etrago,
    network,
    weight,
    focus_region,
    cluster_within,
    per_country,
    cpu_cores,
    func="sigmoid-50",
    save=None,
):
    """
    Apply distance-based spatial weighting relative to a defined focus region.

    This function modifies an existing bus weighting by applying a decay
    function based on the shortest-path distance (along network topology)
    between each bus and a specified focus region. Buses inside the focus
    region receive zero distance (handled as 1 m internally to avoid division
    by zero). Distances are computed using Dijkstra's algorithm and
    multiprocessing.

    The decay factor is applied multiplicatively to the input `weight`.

    Parameters
    ----------
    etrago : Etrago
        An instance of the Etrago class
    network : pypsa.Network
        Network containing buses and either lines (AC) or carrier-specific
        links (e.g. CH4, H2). Must include:
        - `network.buses` with columns ["x", "y", "carrier"]
        - `network.lines` with ["bus0", "bus1", "length"] (for AC)
        - `network.links` with ["bus0", "bus1", "length", "carrier"] (CH4/H2)
    weight : pandas.Series
        Initial bus weighting (indexed by bus names). Typically derived from
        generation capacities and/or loads. Will be modified and returned.
    focus_region : list[str] or str
        Either:
        - List of region identifiers (`gen` column in vg250_krs), or
        - Path to a geospatial file readable by GeoPandas.
        The geometry must have a valid CRS.
    cluster_within : bool
        If False, buses located inside the focus region are assigned a very
        large weight (100000) to prevent clustering within the region.
    cpu_cores : int or str
        Number of CPU cores for multiprocessing. If "max", all available
        cores are used.
    func : {"1-dist", "gauss-20", "gauss-100",
            "sigmoid-20", "sigmoid-50", "sigmoid-100"}, optional
        Distance decay function applied to shortest-path distances (in meters):

        - "1-dist"        : Inverse distance weighting (1 / d)
        - "gauss-20"      : Gaussian decay (σ = 20 km)
        - "gauss-100"     : Gaussian decay (σ = 100 km)
        - "sigmoid-20"    : Sigmoid decay (σ = 20 km)
        - "sigmoid-50"    : Sigmoid decay (σ = 50 km)
        - "sigmoid-100"   : Sigmoid decay (σ = 100 km)

        Default is "sigmoid-50".
    save : str or None, optional
        If provided, the resulting weight Series is written to this CSV path.

    Returns
    -------
    pandas.Series
        Modified bus weighting indexed by bus name. The values are rounded
        up using `np.ceil`.
    """

    # prepare focus region gdf
    if isinstance(focus_region, list):
        if "oep.iks.cs.ovgu.de" in str(etrago.engine.url):
            saio.register_schema("tables", etrago.engine)
            from saio.tables import edut_00_012 as vg250_krs
        else:
            saio.register_schema("boundaries", etrago.engine)
            from saio.boundaries import vg250_krs
        query = etrago.session.query(vg250_krs)
        krs = saio.as_pandas(query, geometry="geometry")
        missing = set(focus_region) - set(krs["gen"])
        if missing:
            raise ValueError(
                f"The following focus_region entries are not valid: {missing}"
            )
        focus_gdf = krs[krs["gen"].isin(focus_region)]
    else:
        focus_gdf = gpd.read_file(focus_region)

    # prepare buses gdf
    buses_df = network.buses[["x", "y"]].copy()
    buses_df["geometry"] = buses_df.apply(
        lambda row: Point(row["x"], row["y"]), axis=1
    )
    buses_gdf = gpd.GeoDataFrame(buses_df, geometry="geometry", crs=4326)
    buses_gdf = buses_gdf.to_crs(epsg=25832)

    # check and adapt CRS
    if focus_gdf.crs is None:
        raise ValueError("CRS of your focus region must be imported.")
    # In Bus-CRS transformieren
    focus_gdf = focus_gdf.to_crs(buses_gdf.crs)
    focus_polygon = focus_gdf.geometry.unary_union

    # identify buses within focus region
    mask = buses_gdf.geometry.within(focus_polygon)
    inside = buses_gdf[mask]
    outside = buses_gdf[~mask]

    # calc distance from buses to focus region
    # Dijkstra's algorithm will be used for path lengths of lines
    # considering buses at border of focus region to consider its size
    if "AC" in network.buses.carrier.unique():
        _export_focus_buses(etrago, inside)
        lines_cross = network.lines[
            (network.lines.bus0.isin(inside.index))
            ^ (network.lines.bus1.isin(inside.index))
        ]
        if per_country:
            lines_cross = lines_cross[lines_cross.country == "DE"]
        border_buses = buses_gdf.loc[
            list(
                set(lines_cross.bus0).union(lines_cross.bus1)
                - set(inside.index)
            )
        ]
        paths = list(product(outside.index, border_buses.index))
    elif "CH4" in network.buses.carrier.unique():
        ch4_links = network.links[network.links.carrier == "CH4"]
        links_cross = ch4_links[
            (ch4_links.bus0.isin(inside.index))
            ^ (ch4_links.bus1.isin(inside.index))
        ]
        if per_country:
            links_cross = links_cross[links_cross.country == "DE"]
        border_buses = buses_gdf.loc[
            list(
                set(links_cross.bus0).union(links_cross.bus1)
                - set(inside.index)
            )
        ]
        paths = list(product(outside.index, border_buses.index))
    elif "H2" in network.buses.carrier.unique():
        h2_links = network.links[network.links.carrier == "H2"]
        links_cross = h2_links[
            (h2_links.bus0.isin(inside.index))
            ^ (h2_links.bus1.isin(inside.index))
        ]
        if per_country:
            links_cross = links_cross[links_cross.country == "DE"]
        border_buses = buses_gdf.loc[
            list(
                set(links_cross.bus0).union(links_cross.bus1)
                - set(inside.index)
            )
        ]
        paths = list(product(outside.index, border_buses.index))

    # graph creation
    if "AC" in network.buses.carrier.unique():
        edges = [
            (row.bus0, row.bus1, row.length, ix)
            for ix, row in network.lines.iterrows()
        ]
    elif "CH4" in network.buses.carrier.unique():
        edges = [
            (row.bus0, row.bus1, row.length, ix)
            for ix, row in ch4_links.iterrows()
        ]
    elif "H2" in network.buses.carrier.unique():
        edges = [
            (row.bus0, row.bus1, row.length, ix)
            for ix, row in h2_links.iterrows()
        ]
    graph = graph_from_edges(edges)

    # processor count
    if cpu_cores == "max":
        cpu_cores = mp.cpu_count()
    else:
        cpu_cores = int(cpu_cores)

    # calculation of shortest path using multiprocessing
    p = mp.Pool(cpu_cores)
    chunksize = ceil(len(paths) / cpu_cores)
    container = p.starmap(shortest_path, gen(paths, chunksize, graph))
    dist = pd.concat(container).astype({"path_length": "float64"})

    dist = dist.loc[
        dist.groupby(level="source")["path_length"].idxmin()
    ].droplevel("target")

    # set distances to 0 for buses within focus region
    dist = pd.concat(
        [dist, pd.DataFrame({"path_length": 0}, index=inside.index)]
    )
    dist = dist["path_length"]
    dist = dist[~dist.index.duplicated(keep="first")]

    # to m
    dist = dist * 1000
    # do not allow 0
    dist[dist == 0] = 1

    # 1/dist
    if func == "1-dist":
        factor = 1 / dist

    # Gaußsche Abklingfunktion mit 20 km
    elif func == "gauss-20":
        sigma = 20000  # z. B. 20 km
        factor = np.exp(-(dist**2) / (2 * sigma**2))

    # Gaußsche Abklingfunktion mit 100 km
    elif func == "gauss-100":
        sigma = 100000  # z. B. 100 km
        factor = np.exp(-(dist**2) / (2 * sigma**2))

    # Sigmoid-Abklingfunktion mit 20 km
    elif func == "sigmoid-20":
        sigma = 20000
        factor = 1 / (1 + (dist / sigma) ** 2)

    # Sigmoid-Abklingfunktion mit 50 km
    elif func == "sigmoid-50":
        sigma = 50000
        factor = 1 / (1 + (dist / sigma) ** 2)

    # Sigmoid-Abklingfunktion mit 100 km
    elif func == "sigmoid-100":
        sigma = 100000
        factor = 1 / (1 + (dist / sigma) ** 2)

    # apply to normal weighting (based on generation capacities and loads)
    weight = weight * factor
    weight = weight.apply(np.ceil)

    if cluster_within is False:
        weight.loc[inside.index] = 100000

    if save:
        weight.to_csv(save)

    return weight


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


def export_clustering_results(etrago):

    path = etrago.args["export_results_path"]

    with open(os.path.join(path, "busmap.json"), "w") as d:
        json.dump(etrago.busmap["busmap"], d, indent=4)

    etrago.busmap["orig_network"].export_to_csv_folder(
        path + "/original_network_topology"
    )