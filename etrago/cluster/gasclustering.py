# -*- coding: utf-8 -*-
# File description for read-the-docs
""" Gasclustering.py defines the methods to cluster gas grid networks
spatially for applications within the tool eTraGo."""

import os

if "READTHEDOCS" not in os.environ:

    from collections import Counter

    import numpy as np
    import pandas as pd
    import pypsa.io as io
    from pypsa import Network
    from pypsa.networkclustering import (
        aggregatebuses,
        aggregateoneport,
        busmap_by_kmeans,
    )
    from six import iteritems

    from etrago.cluster.networkclustering import strategies_links
    from etrago.tools.utilities import *


def create_gas_busmap(etrago):
    """
    Create a bus map from the clustering of buses in space with a
    weighting.

    Parameters
    ----------
    network : pypsa.Network
        The buses must have coordinates x,y.
    Returns
    -------
    busmap : pandas.Series
        Mapping of network.buses to k-means clusters (indexed by
        non-negative integers).
    """
    # Create network_ch4 (grid nodes in order to create the busmap basis)
    network_ch4 = Network()

    buses_ch4 = etrago.network.buses
    io.import_components_from_dataframe(network_ch4, buses_ch4, "Bus")

    num_neighboring_country = (
        (network_ch4.buses["carrier"] == "CH4") & (network_ch4.buses["country"] != "DE")
    ).sum()

    # Cluster ch4 buses
    kmean_gas_settings = etrago.args["network_clustering_kmeans"]

    if num_neighboring_country >= kmean_gas_settings["n_clusters_gas"]:
        msg = (
            "The number of clusters for the gas sector ("
            + str(kmean_gas_settings["n_clusters_gas"])
            + ") must be higher than the number of neighboring contry gas buses ("
            + str(num_neighboring_country)
            + ")."
        )
        raise ValueError(msg)

    network_ch4.buses = network_ch4.buses[
        (network_ch4.buses["carrier"] == "CH4") & (network_ch4.buses["country"] == "DE")
    ]


    # def weighting_for_scenario(x, save=None):
    #     """ """
    #     # TODO to be redefined
    #     b_i = x.index
    #     weight = pd.DataFrame([1] * len(b_i), index=b_i)

    #     if save:
    #         weight.to_csv(save)

    #     return weight


    def weighting_for_scenario(ch4_buses, save=None):
        """
        Calculate CH4-bus weightings dependant on the connected 
        CH4-loads, CH4-generators and non-transport link capacities. 
        Stores are not considered for the clustering.

        Parameters
        ----------
        ch4_buses : pandas.DataFrame
            Dataframe with CH4 etrago.network.buses to weight.
        save: path   
            Path to save weightings to as .csv
        Returns
        -------
        weightings : pandas.Series
            Integer weighting for each ch4_buses.index
        """

        to_neglect = [
            'CH4',
            'H2_to_CH4',
            'CH4_to_H2',
            'H2_feedin',
        ]

        # get all non-transport and non-H2 related links for each bus
        rel_links = {}
        for i in ch4_buses.index:
            rel_links[i] = etrago.network.links.loc[
                (etrago.network.links.bus0.isin([i])
                | etrago.network.links.bus1.isin([i]))
                & ~etrago.network.links.carrier.isin(to_neglect)].index

        # get all generators and loads related to ch4_buses
        generators_ = pd.Series(etrago.network.generators.index, index = etrago.network.generators.bus)
        buses_CH4_gen = generators_.index.intersection(rel_links.keys())
        loads_ = pd.Series(etrago.network.loads.index, index = etrago.network.loads.bus)
        buses_CH4_load = loads_.index.intersection(rel_links.keys())

        # sum up all relevant entities and cast to integer
        # Note: rel_links will hold the weightings for each bus afterwards
        for i in rel_links:
            rel_links[i] = etrago.network.links.loc[rel_links[i]].p_nom.sum()
            if i in buses_CH4_gen:
                rel_links[i] += etrago.network.generators.loc[generators_.loc[i]].p_nom.sum()
            if i in buses_CH4_load:
                rel_links[i] += etrago.network.loads_t.p_set.loc[:,loads_.loc[i]].mean().sum()
            rel_links[i] = int(rel_links[i])

        weightings = pd.DataFrame.from_dict(rel_links, orient='index')

        if save:
            weightings.to_csv(save)

        return weightings


    # State whether to create a bus weighting and save it, create or not save
    # it, or use a bus weighting from a csv file
    if kmean_gas_settings["bus_weight_tocsv"] is not None:
        weight_ch4 = weighting_for_scenario(
            network_ch4.buses,
            save="network_ch4_" + kmean_gas_settings["bus_weight_tocsv"],
        )
    elif kmean_gas_settings["bus_weight_fromcsv"] is not None:
        weight_ch4 = pd.Series.from_csv(kmean_gas_settings["bus_weight_fromcsv"])
        weight_ch4.index = weight_ch4.index.astype(str)
    else:
        weight_ch4 = weighting_for_scenario(network_ch4.buses, save=False)

    weight_ch4_s = weight_ch4.squeeze()

    # Creation of the busmap

    if not kmean_gas_settings["kmeans_gas_busmap"]:

        busmap_ch4 = busmap_by_kmeans(
            network_ch4,
            bus_weightings=weight_ch4_s,
            n_clusters=kmean_gas_settings["n_clusters_gas"]
            - num_neighboring_country,
            n_init=kmean_gas_settings["n_init"],
            max_iter=kmean_gas_settings["max_iter"],
            tol=kmean_gas_settings["tol"],
        )

        busmap_ch4.to_csv(
            "kmeans_ch4_busmap_"
            + str(kmean_gas_settings["n_clusters_gas"])
            + "_result.csv"
        )

    else:

        df = pd.read_csv(kmean_gas_settings["kmeans_gas_busmap"])
        df = df.astype(str)
        df = df.set_index("Bus")
        busmap_ch4 = df.squeeze("columns")

    # Add H2_grid buses to busmap
    df_correspondance_H2_CH4 = etrago.network.links[
        (etrago.network.links["carrier"] == "H2_feedin")
    ]
    df_correspondance_H2_CH4 = df_correspondance_H2_CH4[
        ["bus0", "bus1", "scn_name"]
    ].rename(columns={"bus0": "bus_H2", "bus1": "bus_CH4"})
    df_correspondance_H2_CH4["bus_CH4"] = df_correspondance_H2_CH4["bus_CH4"].astype(
        str
    )
    df_correspondance_H2_CH4 = df_correspondance_H2_CH4.set_index(["bus_CH4"])
    busmap_h2 = pd.concat(
        [df_correspondance_H2_CH4, busmap_ch4.rename("CH4_nodes_c")],
        axis=1,
        join="inner",
    )
    CH4_clusters = busmap_h2["CH4_nodes_c"].tolist()
    CH4_clusters_unique = list(set(CH4_clusters))
    H2_clusters = range(
        kmean_gas_settings["n_clusters_gas"],
        (kmean_gas_settings["n_clusters_gas"] + len(set(CH4_clusters))),
    )
    corr = pd.DataFrame(
        list(zip(CH4_clusters_unique, H2_clusters)),
        columns=["CH4_nodes_c", "H2_clusters"],
    )
    busmap_h2 = busmap_h2.merge(corr, on="CH4_nodes_c", how="inner")
    busmap_h2 = busmap_h2.drop(columns=["scn_name", "CH4_nodes_c"]).set_index(
        ["bus_H2"]
    )
    busmap_h2 = busmap_h2.squeeze()

    busmap = pd.concat([busmap_ch4, busmap_h2])

    # Add all other buses except H2_ind_load to busmap
    missing_idx = list(
        etrago.network.buses[(~etrago.network.buses.index.isin(busmap.index))].index
    )
    next_bus_id = highestInteger(etrago.network.buses.index) + 1
    new_gas_buses = [str(int(x) + next_bus_id) for x in busmap]

    busmap_idx = list(busmap.index) + missing_idx
    busmap_values = new_gas_buses + missing_idx
    busmap = pd.Series(busmap_values, index=busmap_idx)

    if etrago.args["sector_coupled_clustering"]["active"]:
        for name, data in etrago.args["sector_coupled_clustering"][
            "carrier_data"
        ].items():

            strategy = data["strategy"]
            if strategy == "consecutive":
                busmap_sector_coupling = consecutive_sector_coupling(
                    etrago.network,
                    busmap,
                    data["base"],
                    name,
                )
            elif strategy == "simultaneous":
                if len(data["base"]) < 2:
                    msg = (
                        "To apply simultaneous clustering for the "
                        + name
                        + " buses, at least 2 base buses must be selected."
                    )
                    raise ValueError(msg)
                busmap_sector_coupling = simultaneous_sector_coupling(
                    etrago.network,
                    busmap,
                    data["base"],
                    name,
                )
            else:
                msg = (
                    "Strategy for sector coupled clustering must be either "
                    "'consecutive' or 'coupled'."
                )
                raise ValueError(msg)

            for key, value in busmap_sector_coupling.items():
                busmap.loc[key] = value

    busmap = busmap.astype(str)
    busmap.index = busmap.index.astype(str)

    df_bm = pd.DataFrame(busmap.items(), columns=["Original bus id", "New bus id"])
    df_bm.to_csv(
        "kmeans_gasgrid_busmap_"
        + str(kmean_gas_settings["n_clusters_gas"])
        + "_result.csv",
        index=False,
    )

    return busmap


def highestInteger(potentially_numbers):
    """Fetch the highest number of a series with mixed types

    Parameters
    ----------
    potentially_numbers : pandas.core.series.Series
        Series with mixed dtypes, potentially containing numbers.

    Returns
    -------
    int
        Highest integer found in series.
    """
    highest = 0
    for number in potentially_numbers:
        try:
            num = int(number)
            if num > highest:
                highest = num
        except ValueError:
            pass

    return highest


def simultaneous_sector_coupling(network, busmap, carrier_based, carrier_to_cluster):
    """Cluster sector coupling technology based on multiple connected carriers.

    The topology of the sector coupling technology must be in a way, that the
    links connected to other sectors do only point inwards. E.g. for the heat
    sector, heat generating technologies from electricity or gas only point to
    the heat sector and not vice-versa.

    Parameters
    ----------
    network : pypsa.Network
        PyPSA network instance.
    busmap : pandas.Series
        Series with lookup table for clustered buses.
    carrier_based : list
        Carriers on which the clustering of the sector coupling is based.
    carrier_to_cluster : str
        Name of the carrier which should be clustered

    Returns
    -------
    dict
        Busmap for the sector coupling cluster.
    """
    next_bus_id = highestInteger(busmap.values) + 1
    buses_clustered = network.buses[network.buses["carrier"].isin(carrier_based)]
    buses_to_cluster = network.buses[network.buses["carrier"] == carrier_to_cluster]
    buses_to_skip = network.buses[
        network.buses["carrier"] == carrier_to_cluster + "_store"
    ]

    connected_links = network.links.loc[
        network.links["bus0"].isin(buses_clustered.index)
        & network.links["bus1"].isin(buses_to_cluster.index)
        & ~network.links["bus1"].isin(buses_to_skip.index)
        & ~network.links["bus0"].isin(buses_to_skip.index)
    ]

    busmap = busmap.to_dict()
    connected_links["bus0_clustered"] = (
        connected_links["bus0"].map(busmap).fillna(connected_links["bus0"])
    )
    connected_links["bus1_clustered"] = (
        connected_links["bus1"].map(busmap).fillna(connected_links["bus1"])
    )

    # cluster sector coupling technologies
    busmap = sc_multi_carrier_based(buses_to_cluster, connected_links)
    busmap = {bus_id: bus_num + next_bus_id for bus_id, bus_num in busmap.items()}

    # cluster appedices
    skipped_links = network.links.loc[
        (
            network.links["bus1"].isin(buses_to_skip.index)
            & network.links["bus0"].isin(buses_to_cluster.index)
        )
        | (
            network.links["bus0"].isin(buses_to_cluster.index)
            & network.links["bus1"].isin(buses_to_skip.index)
        )
    ]

    # map skipped buses after clustering
    skipped_links["bus0_clustered"] = (
        skipped_links["bus0"].map(busmap).fillna(skipped_links["bus0"])
    )
    skipped_links["bus1_clustered"] = (
        skipped_links["bus1"].map(busmap).fillna(skipped_links["bus1"])
    )

    busmap_series = pd.Series(busmap)
    next_bus_id = highestInteger(busmap_series.values) + 1

    # create clusters for skipped buses
    clusters = busmap_series.unique()
    for i in range(len(clusters)):
        buses = skipped_links.loc[
            skipped_links["bus0_clustered"] == clusters[i], "bus1_clustered"
        ]
        for bus_id in buses:
            busmap[bus_id] = next_bus_id + i
        buses = skipped_links.loc[
            skipped_links["bus1_clustered"] == clusters[i], "bus0_clustered"
        ]
        for bus_id in buses:
            busmap[bus_id] = next_bus_id + i

    return busmap


def consecutive_sector_coupling(network, busmap, carrier_based, carrier_to_cluster):
    """Cluster sector coupling technology based on single connected carriers.

    The topology of the sector coupling technology must be in a way, that the
    links connected to other sectors do only point inwards. E.g. for the heat
    sector, heat generating technologies from electricity or gas only point to
    the heat sector and not vice-versa.

    Parameters
    ----------
    network : pypsa.Network
        PyPSA network instance.
    busmap : pandas.Series
        Series with lookup table for clustered buses.
    carrier_based : list
        Carriers on which the clustering of the sector coupling is based.
    carrier_to_cluster : str
        Name of the carrier which should be clustered

    Returns
    -------
    dict
        Busmap for the sector coupling cluster.
    """
    next_bus_id = highestInteger(busmap.values) + 1
    buses_to_skip = network.buses[
        network.buses["carrier"] == carrier_to_cluster + "_store"
    ]
    buses_to_cluster = network.buses[network.buses["carrier"] == carrier_to_cluster]
    buses_clustered = network.buses[network.buses["carrier"] == carrier_based[0]]
    busmap_sc = {}

    for base in carrier_based:

        # remove already clustered buses
        buses_to_cluster = buses_to_cluster[
            ~buses_to_cluster.index.isin(busmap_sc.keys())
        ]
        buses_clustered = network.buses[network.buses["carrier"] == base]

        connected_links = network.links.loc[
            network.links["bus0"].isin(buses_clustered.index)
            & network.links["bus1"].isin(buses_to_cluster.index)
            & ~network.links["bus1"].isin(buses_to_skip.index)
            & ~network.links["bus0"].isin(buses_to_skip.index)
        ]

        connected_links["bus0_clustered"] = (
            connected_links["bus0"].map(busmap).fillna(connected_links["bus0"])
        )
        connected_links["bus1_clustered"] = (
            connected_links["bus1"].map(busmap).fillna(connected_links["bus1"])
        )

        # cluster sector coupling technologies
        busmap_by_base = sc_single_carrier_based(connected_links)
        bus_num = 0
        for bus_id, bus_num in busmap_by_base.items():

            busmap_by_base[bus_id] = bus_num + next_bus_id

        next_bus_id = bus_num + next_bus_id + 1
        busmap_sc.update(busmap_by_base)

    buses_to_cluster = buses_to_cluster[~buses_to_cluster.index.isin(busmap_sc.keys())]

    if len(buses_to_cluster) > 0:
        msg = "The following buses are not added to any cluster: " + str(
            buses_to_cluster.index
        )
        logger.warning(msg)

    # cluster appedices
    skipped_links = network.links.loc[
        (
            network.links["bus1"].isin(buses_to_skip.index)
            & network.links["bus0"].isin(busmap_sc.keys())
        )
        | (
            network.links["bus0"].isin(busmap_sc.keys())
            & network.links["bus1"].isin(buses_to_skip.index)
        )
    ]

    # map skipped buses after clustering
    skipped_links["bus0_clustered"] = (
        skipped_links["bus0"].map(busmap_sc).fillna(skipped_links["bus0"])
    )
    skipped_links["bus1_clustered"] = (
        skipped_links["bus1"].map(busmap_sc).fillna(skipped_links["bus1"])
    )

    busmap_series = pd.Series(busmap_sc)
    next_bus_id = highestInteger(busmap_series.values) + 1

    # create clusters for skipped buses
    clusters = busmap_series.unique()
    for i in range(len(clusters)):
        buses = skipped_links.loc[
            skipped_links["bus0_clustered"] == clusters[i], "bus1_clustered"
        ]
        for bus_id in buses:
            busmap_sc[bus_id] = next_bus_id + i
        buses = skipped_links.loc[
            skipped_links["bus1_clustered"] == clusters[i], "bus0_clustered"
        ]
        for bus_id in buses:
            busmap_sc[bus_id] = next_bus_id + i

    return busmap_sc


def sc_multi_carrier_based(buses_to_cluster, connected_links):
    """Create busmap for sector coupled carrier based on multiple other carriers.

    Parameters
    ----------
    buses_to_cluster : pandas.Series
        Series containing the buses of the sector coupled carrier which are
        to be clustered.
    connected_links : pandas.DataFrame
        Links that connect from the buses with other carriers to the
        buses of the sector coupled carrier.

    Returns
    -------
    dict
        Busmap for the sector cupled carrier.
    """
    clusters = pd.Series()
    for bus_id in buses_to_cluster.index:
        clusters.loc[bus_id] = tuple(
            sorted(
                connected_links.loc[
                    connected_links["bus1_clustered"] == bus_id, "bus0_clustered"
                ].tolist()
            )
        )

    duplicates = clusters.unique()

    busmap = {}
    for i in range(len(duplicates)):
        cluster = clusters[clusters == duplicates[i]].index.tolist()
        if len(cluster) > 1:
            busmap.update({bus: i for bus in cluster})

    return busmap


def sc_single_carrier_based(connected_links):
    """Create busmap for sector coupled carrier based on single other carrier.

    Parameters
    ----------
    connected_links : pandas.DataFrame
        Links that connect from the buses with other carrier to the
        buses of the sector coupled carrier.

    Returns
    -------
    dict
        Busmap for the sector cupled carrier.
    """
    busmap = {}
    clusters = connected_links["bus0_clustered"].unique()
    for i in range(len(clusters)):
        buses = connected_links.loc[
            connected_links["bus0_clustered"] == clusters[i], "bus1_clustered"
        ].unique()
        busmap.update({bus: i for bus in buses})

    return busmap


def get_clustering_from_busmap(
    network,
    busmap,
    line_length_factor=1.0,
    with_time=True,
    bus_strategies=dict(),
    one_port_strategies=dict(),
):

    network_gasgrid_c = Network()

    # Aggregate buses
    new_buses = aggregatebuses(
        network,
        busmap,
        custom_strategies=bus_strategies,
    )
    new_buses.index.name = "bus_id"

    io.import_components_from_dataframe(network_gasgrid_c, new_buses, "Bus")

    if with_time:
        network_gasgrid_c.set_snapshots(network.snapshots)
        network_gasgrid_c.snapshot_weightings = network.snapshot_weightings.copy()

    # Aggregate one port components
    one_port_components = ["Generator", "Load", "Store"]

    for one_port in one_port_components:
        one_port_components.remove(one_port)
        new_df, new_pnl = aggregateoneport(
            network,
            busmap,
            component=one_port,
            with_time=with_time,
            custom_strategies=one_port_strategies.get(one_port, {}),
        )
        io.import_components_from_dataframe(network_gasgrid_c, new_df, one_port)
        for attr, df in iteritems(new_pnl):
            io.import_series_from_dataframe(network_gasgrid_c, df, one_port, attr)

    for c in network.iterate_components(one_port_components):
        io.import_components_from_dataframe(
            network_gasgrid_c,
            c.df.assign(bus=c.df.bus.map(busmap)).dropna(subset=["bus"]),
            c.name,
        )

    if with_time:
        for c in network.iterate_components(one_port_components):
            for attr, df in iteritems(c.pnl):
                if not df.empty:
                    io.import_series_from_dataframe(network_gasgrid_c, df, c.name, attr)

    # Aggregate links
    new_links = (
        network.links.assign(
            bus0=network.links.bus0.map(busmap), bus1=network.links.bus1.map(busmap)
        )
        .dropna(subset=["bus0", "bus1"])
        .loc[lambda df: df.bus0 != df.bus1]
    )

    new_links["link_id"] = new_links.index

    strategies = strategies_links()
    strategies["link_id"] = "first"

    # aggregate CH4 pipelines
    # pipelines are treated differently compared to other links, since all of
    # them will be considered bidirectional. That means, if a pipeline exists,
    # that connects one cluster with a different one simultaneously with a
    # pipeline that connects these two clusters in reversed order (e.g. bus0=1,
    # bus1=12 and bus0=12, bus1=1) they are aggregated to a single pipeline.
    pipelines = new_links.loc[new_links["carrier"] == "CH4"]

    pipeline_combinations = pipelines.groupby(["bus0", "bus1", "carrier"]).agg(
        strategies
    )
    pipeline_combinations.reset_index(drop=True, inplace=True)
    pipeline_combinations["buscombination"] = pipeline_combinations[
        ["bus0", "bus1"]
    ].apply(lambda x: tuple(sorted([str(x.bus0), str(x.bus1)])), axis=1)
    pipeline_strategies = strategies.copy()
    pipeline_strategies.update(
        {col: "first" for col in pipeline_combinations.columns if col not in strategies}
    )
    # the order of buses for pipelines can be ignored, since the pipelines are
    # working bidirectionally
    pipeline_strategies["bus0"] = "first"
    pipeline_strategies["bus1"] = "first"
    pipelines_final = pipeline_combinations.groupby(["buscombination", "carrier"]).agg(
        pipeline_strategies
    )

    pipelines_final.set_index("link_id", inplace=True)
    pipelines_final.drop(columns="buscombination", inplace=True)
    io.import_components_from_dataframe(network_gasgrid_c, pipelines_final, "Link")

    # aggregate remaining links
    not_pipelines = new_links.loc[new_links["carrier"] != "CH4"]
    combinations = not_pipelines.groupby(["bus0", "bus1", "carrier"]).agg(strategies)
    combinations.set_index("link_id", inplace=True)

    io.import_components_from_dataframe(network_gasgrid_c, combinations, "Link")

    if with_time:
        for attr, df in iteritems(network.links_t):
            if not df.empty:
                io.import_series_from_dataframe(network_gasgrid_c, df, "Link", attr)

    return network_gasgrid_c


def kmean_clustering_gas_grid(etrago):
    """Main function of the k-mean clustering approach. Maps the original gas
    network to a new one with adjustable number of nodes and new coordinates.
    Parameters
    ----------
    network : :class:`pypsa.Network
        Container for all network components.
    n_clusters_gas : int
        Desired number of gas clusters.
    bus_weight_tocsv : str
        Creates a bus weighting based on conventional generation and load
        and save it to a csv file.
    bus_weight_fromcsv : str
        Loads a bus weighting from a csv file to apply it to the clustering
        algorithm.
    Returns
    -------
    network : pypsa.Network object
        Container for the gas network components.
    """

    gas_busmap = create_gas_busmap(etrago)

    network_gasgrid_c = get_clustering_from_busmap(
        etrago.network,
        gas_busmap,
        bus_strategies={
            "country": "first",
        },
        one_port_strategies={
            "Generator": {
                "marginal_cost": np.mean,
                "capital_cost": np.mean,
                "p_nom_max": np.sum,
                "p_nom_min": np.sum,
            },
            "Store": {
                "marginal_cost": np.mean,
                "capital_cost": np.mean,
                "e_nom": np.sum,
                "e_nom_max": np.max,
            },
            "Load": {
                "p_set": np.sum,
            },
        },
    )

    # Insert components not related to the gas clustering
    io.import_components_from_dataframe(network_gasgrid_c, etrago.network.lines, "Line")
    io.import_components_from_dataframe(
        network_gasgrid_c, etrago.network.storage_units, "StorageUnit"
    )
    io.import_components_from_dataframe(
        network_gasgrid_c, etrago.network.shunt_impedances, "ShuntImpedance"
    )
    io.import_components_from_dataframe(
        network_gasgrid_c, etrago.network.transformers, "Transformer"
    )
    io.import_components_from_dataframe(
        network_gasgrid_c, etrago.network.carriers, "Carrier"
    )

    network_gasgrid_c.determine_network_topology()

    return network_gasgrid_c


def run_kmeans_clustering_gas(self):

    if self.args["network_clustering_kmeans"]["active"]:

        self.network.generators.control = "PV"

        logger.info("Start k-mean clustering GAS")
        self.network = kmean_clustering_gas_grid(self)
        logger.info(
            "GAS Network clustered to {} buses with k-means algorithm.".format(
                self.args["network_clustering_kmeans"]["n_clusters_gas"]
            )
        )
