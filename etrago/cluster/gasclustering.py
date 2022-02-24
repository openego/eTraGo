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

    network_ch4.buses = network_ch4.buses[
        (network_ch4.buses["carrier"] == "CH4") & (network_ch4.buses["country"] == "DE")
    ]

    # Cluster ch4 buses
    kmean_gas_settings = etrago.args["network_clustering_kmeans"]

    def weighting_for_scenario(x, save=None):
        """ """
        # TODO to be redefined
        b_i = x.index
        weight = pd.DataFrame([1] * len(b_i), index=b_i)

        if save:
            weight.to_csv(save)

        return weight

    # State whether to create a bus weighting and save it, create or not save
    # it, or use a bus weighting from a csv file
    if kmean_gas_settings["bus_weight_tocsv"] is not None:
        weight_ch4 = weighting_for_scenario(
            x=network_ch4.buses,
            save="network_ch4_" + kmean_gas_settings["bus_weight_tocsv"],
        )
    elif kmean_gas_settings["bus_weight_fromcsv"] is not None:
        weight_ch4 = pd.Series.from_csv(kmean_gas_settings["bus_weight_fromcsv"])
        weight_ch4.index = weight_ch4.index.astype(str)
    else:
        weight_ch4 = weighting_for_scenario(x=network_ch4.buses, save=False)

    weight_ch4_s = weight_ch4.squeeze()

    # Creation of the busmap
    busmap_ch4 = busmap_by_kmeans(
        network_ch4,
        bus_weightings=weight_ch4_s,
        n_clusters=kmean_gas_settings["n_clusters_gas"],
        n_init=kmean_gas_settings["n_init"],
        max_iter=kmean_gas_settings["max_iter"],
        tol=kmean_gas_settings["tol"],
    )

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

    busmap = pd.concat([busmap_ch4, busmap_h2]).astype(str)

    # Add all other buses except H2_ind_load to busmap
    busmap.index = busmap.index.astype(str)
    missing_idx = list(
        etrago.network.buses[
            (~etrago.network.buses.index.isin(busmap.index))
            & (etrago.network.buses["carrier"] != "H2_ind_load")
        ].index
    )
    next_bus_id = etrago.network.buses.index.astype(int).max() + 1
    new_gas_buses = [str(int(x) + next_bus_id) for x in busmap]

    busmap_idx = list(busmap.index) + missing_idx
    busmap_values = new_gas_buses + missing_idx
    busmap = pd.Series(busmap_values, index=busmap_idx)

    busmap = busmap.astype(str)
    busmap.index = busmap.index.astype(str)

    # Add H2_ind_load buses to busmap
    H2_ind_buses = etrago.network.buses[
        (etrago.network.buses["carrier"] == "H2_ind_load")
    ]

    H2_ind_load_original_links = etrago.network.links[
        (etrago.network.links["carrier"] == "H2_ind_load")
    ]

    clustered_H2_non_ind = []
    for index, row in H2_ind_load_original_links.iterrows():
        clustered_H2_non_ind.append(busmap[row["bus0"]])

    H2_ind_load_original_links["clustered_H2_non_ind"] = clustered_H2_non_ind

    comb = []
    for index, row in H2_ind_buses.iterrows():
        df_combinaison = H2_ind_load_original_links[
            H2_ind_load_original_links["bus1"] == index
        ]
        combinaison = tuple(sorted(df_combinaison["clustered_H2_non_ind"].tolist()))
        comb.append(combinaison)

    H2_ind_buses["comb"] = comb

    counts = Counter(comb)
    non_unique_comb = [value for value, count in counts.items() if count > 1]

    new_bus_id = dict()
    for index, row in H2_ind_buses[
        ~H2_ind_buses["comb"].isin(non_unique_comb)
    ].iterrows():
        new_bus_id[index] = index

    next_bus_id = busmap.values.astype(int).max() + 1
    for i in non_unique_comb:
        c = H2_ind_buses[H2_ind_buses["comb"] == i].index.tolist()
        new_bus_id[c[0]] = next_bus_id
        new_bus_id[c[1]] = next_bus_id
        next_bus_id += 1

    busmap = {**busmap, **new_bus_id}

    df_bm = pd.DataFrame(busmap.items(), columns=["Original bus id", "New bus id"])
    df_bm.to_csv(
        "kmeans_gasgrid_busmap_"
        + str(kmean_gas_settings["n_clusters_gas"])
        + "_result.csv",
        index=False,
    )

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

    strategies = {
        "p_nom": "sum",
        "length": "mean",
    }
    strategies.update(
        {col: "first" for col in new_links.columns if col not in strategies}
    )

    gas_carriers = [  # This list should be replace by an automatic selection
        "CH4",
        "CH4_to_H2",
        "H2_feedin",
        "H2_ind_load",
        "H2_to_CH4",
        "H2_to_power",
        "power_to_H2",
        "central_gas_CHP",
        "central_gas_CHP_heat",
        "industrial_gas_CHP",
        "rural_gas_boiler",
        "central_gas_boiler",
        "OCGT",
    ]

    gas_links = new_links[new_links["carrier"].isin(gas_carriers)].copy()

    combinations = gas_links.groupby(["bus0", "bus1", "carrier"]).agg(strategies)
    combinations.set_index("link_id", inplace=True)

    io.import_components_from_dataframe(network_gasgrid_c, combinations, "Link")

    non_gas_links = (
        new_links[~new_links["carrier"].isin(gas_carriers)]
        .copy()
        .drop(columns="link_id")
    )
    io.import_components_from_dataframe(network_gasgrid_c, non_gas_links, "Link")

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
                "e_nom_max": np.sum,
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
        logger.info("Start k-mean clustering GAS")
        self.network = kmean_clustering_gas_grid(self)
        logger.info(
            "GAS Network clustered to {} buses with k-means algorithm.".format(
                self.args["network_clustering_kmeans"]["n_clusters_gas"]
            )
        )
