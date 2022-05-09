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

if "READTHEDOCS" not in os.environ:
    from etrago.tools.utilities import *
    from pypsa.networkclustering import (
        aggregatebuses,
        aggregateoneport,
        aggregategenerators,
        get_clustering_from_busmap,
        busmap_by_kmeans,
        busmap_by_stubs,
        _make_consense,
        _flatten_multiindex,
    )
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
    import pdb

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


def _leading(busmap, df):
    """ """

    def leader(x):
        ix = busmap[x.index[0]]
        return df.loc[ix, x.name]

    return leader


def adjust_no_electric_network(network, busmap, cluster_met):

    # network2 is supposed to contain all the not electrical or gas buses and links
    network2 = network.copy()
    network2.buses = network2.buses[
        (network2.buses["carrier"] != "AC")
        & (network2.buses["carrier"] != "CH4")
        & (network2.buses["carrier"] != "H2_grid")
        & (network2.buses["carrier"] != "H2_ind_load")
        & (network2.buses["carrier"] != "rural_heat")
        & (network2.buses["carrier"] != "rural_heat_store")
        & (network2.buses["carrier"] != "central_heat")
        & (network2.buses["carrier"] != "central_heat_store")
    ]

    map_carrier = {
        "H2_saltcavern": "power_to_H2",
        "dsm": "dsm",
    }

    # no_elec_to_cluster maps the no electrical buses to the eHV/kmean bus
    no_elec_to_cluster = pd.DataFrame(
        columns=["cluster", "carrier", "new_bus"]
    ).set_index("new_bus")

    max_bus = max([int(item) for item in network.buses.index.to_list()])

    no_elec_conex = []
    # busmap2 maps all the no electrical buses to the new buses based on the
    # eHV network
    busmap2 = {}

    for bus_ne in network2.buses.index:
        bus_hv = -1
        carry = network2.buses.loc[bus_ne, "carrier"]

        if (
            len(
                network2.links[
                    (network2.links["bus1"] == bus_ne)
                    & (network2.links["carrier"] == map_carrier[carry])
                ]
            )
            > 0
        ):
            df = network2.links[
                (network2.links["bus1"] == bus_ne)
                & (network2.links["carrier"] == map_carrier[carry])
            ].copy()
            df["elec"] = df["bus0"].isin(busmap.keys())
            df = df[df["elec"] == True]
            if len(df) > 0:
                bus_hv = df["bus0"][0]

        if bus_hv == -1:
            busmap2[bus_ne] = str(bus_ne)
            no_elec_conex.append(bus_ne)
            continue

        if (
            (no_elec_to_cluster.cluster == busmap[bus_hv])
            & (no_elec_to_cluster.carrier == carry)
        ).any():

            bus_cluster = no_elec_to_cluster[
                (no_elec_to_cluster.cluster == busmap[bus_hv])
                & (no_elec_to_cluster.carrier == carry)
            ].index[0]
        else:
            bus_cluster = str(max_bus + 1)
            max_bus = max_bus + 1
            new = pd.Series(
                {"cluster": busmap[bus_hv], "carrier": carry}, name=bus_cluster
            )
            no_elec_to_cluster = no_elec_to_cluster.append(new)

        busmap2[bus_ne] = bus_cluster

    if no_elec_conex:
        logger.info(
            f"""There are {len(no_elec_conex)} buses that have no direct
            connection to the electric network"""
        )

    # Add the gas buses to the busmap and map them to themself
    for gas_bus in network.buses[
        (network.buses["carrier"] == "H2_grid")
        | (network.buses["carrier"] == "H2_ind_load")
        | (network.buses["carrier"] == "CH4")
        | (network.buses["carrier"] == "rural_heat")
        | (network.buses["carrier"] == "rural_heat_store")
        | (network.buses["carrier"] == "central_heat")
        | (network.buses["carrier"] == "central_heat_store")
    ].index:

        busmap2[gas_bus] = gas_bus

    busmap = {**busmap, **busmap2}

    # The new buses based on the eHV network for not electrical buses are created
    if cluster_met == "k-mean":
        for no_elec_bus in no_elec_to_cluster.index:
            cluster_bus = no_elec_to_cluster.loc[no_elec_bus, :].cluster
            carry = no_elec_to_cluster.loc[no_elec_bus, :].carrier
            new_bus = pd.Series(
                {
                    "scn_name": np.nan,
                    "v_nom": np.nan,
                    "carrier": carry,
                    "x": np.nan,
                    "y": np.nan,
                    "geom": np.nan,
                    "type": "",
                    "v_mag_pu_set": 1,
                    "v_mag_pu_min": 0,
                    "v_mag_pu_max": np.inf,
                    "control": "PV",
                    "sub_network": "",
                    "country": np.nan,
                },
                name=no_elec_bus,
            )
            network.buses = network.buses.append(new_bus)

    else:
        for no_elec_bus in no_elec_to_cluster.index:
            cluster_bus = no_elec_to_cluster.loc[no_elec_bus, :].cluster
            carry = no_elec_to_cluster.loc[no_elec_bus, :].carrier
            new_bus = pd.Series(
                {
                    "scn_name": network.buses.at[cluster_bus, "scn_name"],
                    "v_nom": np.nan,
                    "carrier": carry,
                    "x": network.buses.at[cluster_bus, "x"],
                    "y": network.buses.at[cluster_bus, "y"],
                    "geom": network.buses.at[cluster_bus, "geom"],
                    "type": "",
                    "v_mag_pu_set": 1,
                    "v_mag_pu_min": 0,
                    "v_mag_pu_max": np.inf,
                    "control": "PV",
                    "sub_network": "",
                    "country": network.buses.at[cluster_bus, "country"],
                },
                name=no_elec_bus,
            )
            network.buses = network.buses.append(new_bus)

    return network, busmap


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
        },
    }


def strategies_generators():
    return {
        "p_nom_min": np.min,
        "p_nom_max": np.min,
        "weight": np.sum,
        "p_nom": np.sum,
        "p_nom_opt": np.sum,
        "marginal_cost": np.mean,
        "capital_cost": np.mean,
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
        "country": _make_consense_links,
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
            df_agg = df.loc[:, pnl_links_agg_b]
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

    return new_df, new_pnl


def cluster_on_extra_high_voltage(network, busmap, with_time=True):
    """Main function of the EHV-Clustering approach. Creates a new clustered
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

    network, busmap = adjust_no_electric_network(network, busmap, cluster_met="ehv")
    
    pd.DataFrame(busmap.items(), columns=["bus0", "bus1"]).to_csv(
        "ehv_elecgrid_busmap_result.csv",
        index=False,
    )
    
    buses = aggregatebuses(
        network,
        busmap,
        {"x": _leading(busmap, network.buses), "y": _leading(busmap, network.buses)},
    )

    # keep attached lines
    lines = network.lines.copy()
    mask = lines.bus0.isin(buses.index)
    lines = lines.loc[mask, :]

    # keep attached transformer
    transformers = network.transformers.copy()
    mask = transformers.bus0.isin(buses.index)
    transformers = transformers.loc[mask, :]

    io.import_components_from_dataframe(network_c, buses, "Bus")
    io.import_components_from_dataframe(network_c, lines, "Line")
    io.import_components_from_dataframe(network_c, transformers, "Transformer")

    # Dealing with links
    links = network.links.copy()
    dc_links = links[links["carrier"] == "DC"]
    links = links[links["carrier"] != "DC"]

    new_links = (
        links.assign(bus0=links.bus0.map(busmap), bus1=links.bus1.map(busmap))
        .dropna(subset=["bus0", "bus1"])
        .loc[lambda df: df.bus0 != df.bus1]
    )

    new_links = new_links.append(dc_links)
    new_links["topo"] = np.nan
    io.import_components_from_dataframe(network_c, new_links, "Link")

    if with_time:
        network_c.snapshots = network.snapshots
        network_c.set_snapshots(network.snapshots)
        network_c.snapshot_weightings = network.snapshot_weightings.copy()

        for attr, df in network.lines_t.items():
            mask = df.columns[df.columns.isin(lines.index)]
            df = df.loc[:, mask]
            if not df.empty:
                io.import_series_from_dataframe(network_c, df, "Line", attr)

        for attr, df in network.links_t.items():
            mask = df.columns[df.columns.isin(links.index)]
            df = df.loc[:, mask]
            if not df.empty:
                io.import_series_from_dataframe(network_c, df, "Link", attr)

    # dealing with generators
    network.generators.control = "PV"
    network.generators["weight"] = 1

    new_df, new_pnl = aggregategenerators(
        network, busmap, with_time, custom_strategies=strategies_generators()
    )
    io.import_components_from_dataframe(network_c, new_df, "Generator")
    for attr, df in iteritems(new_pnl):
        io.import_series_from_dataframe(network_c, df, "Generator", attr)

    # dealing with all other components
    aggregate_one_ports = network.one_port_components.copy()
    aggregate_one_ports.discard("Generator")

    for one_port in aggregate_one_ports:
        one_port_strategies = strategies_one_ports()
        new_df, new_pnl = aggregateoneport(
            network,
            busmap,
            component=one_port,
            with_time=with_time,
            custom_strategies=one_port_strategies.get(one_port, {}),
        )
        io.import_components_from_dataframe(network_c, new_df, one_port)
        for attr, df in iteritems(new_pnl):
            io.import_series_from_dataframe(network_c, df, one_port, attr)

    network_c.links, network_c.links_t = group_links(network_c)

    network_c.determine_network_topology()

    return network_c.copy()


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

    for s, t in paths:

        try:
            df.loc[(s, t), "path_length"] = nx.dijkstra_path_length(graph, s, t)

        except NetworkXNoPath:
            continue

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

        cpu_cores = input("cpu_cores (default 4): ") or "4"

        busmap_by_shortest_path(
            etrago,
            scn_name,
            fromlvl=[110],
            tolvl=[220, 380, 400, 450],
            cpu_cores=int(cpu_cores),
        )
        busmap = fetch()

    return busmap


def ehv_clustering(self):

    if self.args["network_clustering_ehv"]:

        logger.info("Start ehv clustering")

        self.network.generators.control = "PV"
        busmap = busmap_from_psql(self)
        self.network = cluster_on_extra_high_voltage(
            self.network, busmap, with_time=True
        )
        logger.info("Network clustered to EHV-grid")


def kmean_clustering(etrago):
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
    kmean_settings = etrago.args["network_clustering_kmeans"]

    def weighting_for_scenario(x, save=None):
        """ """
        # define weighting based on conventional 'old' generator spatial
        # distribution

        non_conv_types = {
            "biomass",
            "central_biomass_CHP",
            "industrial_biomass_CHP",
            "wind_onshore",
            "wind_offshore",
            "solar",
            "solar_rooftop",
            "geo_thermal",
            "load shedding",
            "extendable_storage",
            "other_renewable",
            "reservoir",
            "run_of_river",
            "pumped_hydro",
        }
        # Attention: elec_network.generators.carrier.unique()
        gen = elec_network.generators.loc[
            (elec_network.generators.carrier.isin(non_conv_types) == False)
        ].groupby("bus").p_nom.sum().reindex(
            elec_network.buses.index, fill_value=0.0
        ) + elec_network.storage_units.loc[
            (elec_network.storage_units.carrier.isin(non_conv_types) == False)
        ].groupby(
            "bus"
        ).p_nom.sum().reindex(
            elec_network.buses.index, fill_value=0.0
        )

        load = elec_network.loads_t.p_set.mean().groupby(elec_network.loads.bus).sum()

        b_i = x.index
        g = normed(gen.reindex(b_i, fill_value=0))
        l = normed(load.reindex(b_i, fill_value=0))

        w = g + l
        weight = ((w * (100000.0 / w.max())).astype(int)).reindex(
            elec_network.buses.index, fill_value=1
        )

        if save:
            weight.to_csv(save)

        return weight

    def normed(x):
        return (x / x.sum()).fillna(0.0)

    # prepare k-mean
    # k-means clustering (first try)
    network.generators.control = "PV"
    network.storage_units.control[
        network.storage_units.carrier == "extendable_storage"
    ] = "PV"

    # problem our lines have no v_nom. this is implicitly defined by the
    # connected buses:
    network.lines["v_nom"] = network.lines.bus0.map(network.buses.v_nom)

    # adjust the electrical parameters of the lines which are not 380.
    lines_v_nom_b = network.lines.v_nom != 380

    voltage_factor = (network.lines.loc[lines_v_nom_b, "v_nom"] / 380.0) ** 2

    network.lines.loc[lines_v_nom_b, "x"] *= 1 / voltage_factor

    network.lines.loc[lines_v_nom_b, "r"] *= 1 / voltage_factor

    network.lines.loc[lines_v_nom_b, "b"] *= voltage_factor

    network.lines.loc[lines_v_nom_b, "g"] *= voltage_factor

    network.lines.loc[lines_v_nom_b, "v_nom"] = 380.0

    trafo_index = network.transformers.index
    transformer_voltages = pd.concat(
        [
            network.transformers.bus0.map(network.buses.v_nom),
            network.transformers.bus1.map(network.buses.v_nom),
        ],
        axis=1,
    )

    network.import_components_from_dataframe(
        network.transformers.loc[
            :, ["bus0", "bus1", "x", "s_nom", "capital_cost", "sub_network", "s_max_pu"]
        ]
        .assign(
            x=network.transformers.x * (380.0 / transformer_voltages.max(axis=1)) ** 2,
            length=1,
        )
        .set_index("T" + trafo_index),
        "Line",
    )
    network.transformers.drop(trafo_index, inplace=True)

    for attr in network.transformers_t:
        network.transformers_t[attr] = network.transformers_t[attr].reindex(columns=[])

    network.buses["v_nom"][network.buses.carrier == "AC"] = 380.0

    elec_network = select_elec_network(etrago)

    # State whether to create a bus weighting and save it, create or not save
    # it, or use a bus weighting from a csv file
    if kmean_settings["bus_weight_tocsv"] is not None:
        weight = weighting_for_scenario(
            x=elec_network.buses, save=kmean_settings["bus_weight_tocsv"]
        )
    elif kmean_settings["bus_weight_fromcsv"] is not None:
        weight = pd.Series.from_csv(kmean_settings["bus_weight_fromcsv"])
        weight.index = weight.index.astype(str)
    else:
        weight = weighting_for_scenario(x=elec_network.buses, save=False)

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
        network = clustering.network

        weight = weight.groupby(busmap.values).sum()

    # k-mean clustering
    if not kmean_settings["kmeans_busmap"]:
        busmap = busmap_by_kmeans(
            elec_network,
            bus_weightings=pd.Series(weight),
            n_clusters=kmean_settings["n_clusters"],
            n_init=kmean_settings["n_init"],
            max_iter=kmean_settings["max_iter"],
            tol=kmean_settings["tol"],
        )
        busmap.to_csv(
            "kmeans_elec_busmap_" + str(kmean_settings["n_clusters"]) + "_result.csv"
        )
    else:
        df = pd.read_csv(kmean_settings["kmeans_busmap"])
        df = df.astype(str)
        df = df.set_index("Bus")
        busmap = df.squeeze("columns")

    network, busmap = adjust_no_electric_network(network, busmap, cluster_met="k-mean")

    pd.DataFrame(busmap.items(), columns=["bus0", "bus1"]).to_csv(
        "kmeans_elecgrid_busmap_" + str(kmean_settings["n_clusters"]) + "_result.csv",
        index=False,
    )

    network.generators["weight"] = network.generators["p_nom"]
    aggregate_one_ports = network.one_port_components.copy()
    aggregate_one_ports.discard("Generator")

    clustering = get_clustering_from_busmap(
        network,
        busmap,
        aggregate_generators_weighted=True,
        one_port_strategies=strategies_one_ports(),
        generator_strategies=strategies_generators(),
        aggregate_one_ports=aggregate_one_ports,
        line_length_factor=kmean_settings["line_length_factor"],
    )

    clustering.network.links, clustering.network.links_t = group_links(
        clustering.network
    )

    return clustering


def select_elec_network(etrago):

    elec_network = etrago.network.copy()
    elec_network.buses = elec_network.buses[elec_network.buses.carrier == "AC"]
    elec_network.links = elec_network.links[
        (elec_network.links.carrier == "AC") | (elec_network.links.carrier == "DC")
    ]
    # Dealing with generators
    elec_network.generators = elec_network.generators[
        elec_network.generators.bus.isin(elec_network.buses.index)
    ]

    for attr in elec_network.generators_t:
        elec_network.generators_t[attr] = elec_network.generators_t[attr].loc[
            :,
            elec_network.generators_t[attr].columns.isin(elec_network.generators.index),
        ]

    # Dealing with loads
    elec_network.loads = elec_network.loads[
        elec_network.loads.bus.isin(elec_network.buses.index)
    ]

    for attr in elec_network.loads_t:
        elec_network.loads_t[attr] = elec_network.loads_t[attr].loc[
            :, elec_network.loads_t[attr].columns.isin(elec_network.loads.index)
        ]

    # Dealing with storage_units
    elec_network.storage_units = elec_network.storage_units[
        elec_network.storage_units.bus.isin(elec_network.buses.index)
    ]

    for attr in elec_network.storage_units_t:
        elec_network.storage_units_t[attr] = elec_network.storage_units_t[attr].loc[
            :,
            elec_network.storage_units_t[attr].columns.isin(
                elec_network.storage_units.index
            ),
        ]

    return elec_network


def run_kmeans_clustering(self):

    if self.args["network_clustering_kmeans"]["active"]:

        self.network.generators.control = "PV"

        logger.info("Start k-mean clustering")

        self.clustering = kmean_clustering(self)

        if self.args["disaggregation"] != None:
            self.disaggregated_network = self.network.copy()

        self.network = self.clustering.network.copy()

        buses_by_country(self.network)

        self.geolocation_buses()

        self.network.generators.control[self.network.generators.control == ""] = "PV"
        logger.info(
            "Network clustered to {} buses with k-means algorithm.".format(
                self.args["network_clustering_kmeans"]["n_clusters"]
            )
        )
