# -*- coding: utf-8 -*-
# File description for read-the-docs
""" Gasclustering.py defines the methods to cluster gas grid networks
spatially for applications within the tool eTraGo."""

import os

if "READTHEDOCS" not in os.environ:

    import numpy as np
    import pandas as pd
    import pypsa.io as io
    from egoio.tools import db
    from pypsa import Network
    from pypsa.networkclustering import (
        aggregatebuses,
        aggregateoneport,
        busmap_by_kmeans,
    )
    from six import iteritems

    from etrago.tools.utilities import *


def select_dataframe(sql, conn, index_col=None):
    """Select data from local database as pandas.DataFrame
    Parameters
    ----------
    sql : str
        SQL query to be executed.
    index_col : str, optional
        Column(s) to set as index(MultiIndex). The default is None.
    Returns
    -------
    df : pandas.DataFrame
        Data returned from SQL statement.
    """

    df = pd.read_sql(sql, conn, index_col=index_col)

    if df.size == 0:
        print(f"WARNING: No data returned by statement: \n {sql}")

    return df


def create_gas_busmap(etrago):
    """
    Create a bus map from the clustering of buses in space with a
    weighting.

    Parameters
    ----------
    Returns
    -------
    busmap : pandas.Series
        Mapping of network.buses to k-means clusters (indexed by
        non-negative integers).
    """

    # Download H2-CH4 correspondance table
    engine = db.connection(section=etrago.args["db"])
    sql = f"""SELECT "bus_H2", "bus_CH4", scn_name FROM grid.egon_etrago_ch4_h2;"""
    df_correspondance_H2_CH4 = select_dataframe(sql, engine)
    df_correspondance_H2_CH4["bus_CH4"] = df_correspondance_H2_CH4["bus_CH4"].astype(
        str
    )
    df_correspondance_H2_CH4 = df_correspondance_H2_CH4.set_index(["bus_CH4"])

    # Create network_ch4 (grid nodes in order to create the busmap basis)
    network_ch4 = Network()
    buses_ch4 = etrago.network.buses
    io.import_components_from_dataframe(network_ch4, buses_ch4, "Bus")

    network_ch4.buses = network_ch4.buses[
        (network_ch4.buses["carrier"] == "CH4") & (network_ch4.buses["country"] == "DE")
    ]

    # Cluster network_ch4
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
        n_jobs=kmean_gas_settings["n_jobs"],
    )

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

    ##### This part could be change to select only buses involved in links with gas buses #####
    # breakpoint()
    # print(busmap.index)
    busmap.index = busmap.index.astype(str)
    missing_idx = list(
        etrago.network.buses[~etrago.network.buses.index.isin(busmap.index)].index
    )
    # breakpoint()
    # next_bus_id = etrago.network.buses.index.astype(int).max() + 1
    next_bus_id = len(etrago.network.buses.index)
    new_gas_buses = [str(int(x) + next_bus_id) for x in busmap]

    busmap_idx = list(busmap.index) + missing_idx
    busmap_values = new_gas_buses + missing_idx
    busmap = pd.Series(busmap_values, index=busmap_idx)

    busmap = busmap.astype(str)
    busmap.index = busmap.index.astype(str)

    ### End of This part could be change to select only buses involved in links with gas buses ###

    busmap.to_csv(
        "kmeans_gasgrid_busmap_"
        + str(kmean_gas_settings["n_clusters_gas"])
        + "_result.csv"
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
    # network_gasgrid_c.buses.to_csv("network_gasgrid_c_buses.csv")

    if with_time:
        network_gasgrid_c.snapshot_weightings = network.snapshot_weightings.copy()
        network_gasgrid_c.set_snapshots(network.snapshots)

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
        "p_min_pu_fixed": "min",
        "p_max_pu_fixed": "max",
        "p_set_fixed": "mean",
        "marginal_cost_fixed": "mean",
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
    combinations.reset_index(drop=True, inplace=True)

    combinations["buscombination"] = (
        combinations[["bus0", "bus1"]].apply(sorted, axis=1).apply(lambda x: tuple(x))
    )

    strategies.update(
        {col: "first" for col in combinations.columns if col not in strategies}
    )

    combinations_final = combinations.groupby(["buscombination", "carrier"]).agg(
        strategies
    )

    combinations_final.set_index("link_id", inplace=True)
    combinations_final = combinations_final.drop(columns="buscombination")
    io.import_components_from_dataframe(network_gasgrid_c, combinations_final, "Link")

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


def kmean_clustering_gas_grid(etrago, with_time=True):
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
    for data, name in zip(
        [
            etrago.network.links,
            etrago.network.buses,
            etrago.network.generators,
            etrago.network.stores,
            etrago.network.loads,
            etrago.network.storage_units,
            etrago.network.shunt_impedances,
            etrago.network.lines,
            etrago.network.transformers,
        ],
        [
            "links",
            "buses",
            "generators",
            "stores",
            "loads",
            "storage_units",
            "shunt_impedances",
            "lines",
            "transformers",
        ],
    ):
        pd.DataFrame(data).to_csv(
            "network_not_clustered/" + name + "_not_clustered.csv"
        )

    gas_busmap = create_gas_busmap(etrago)

    network_gasgrid_c = get_clustering_from_busmap(
        etrago.network,
        gas_busmap,
        bus_strategies={
            "country": "first",
            # "scn_name": "first",
            # "v_mag_pu_set_fixed": "first",
        },
        one_port_strategies={
            "Generator": {
                # "scn_name": "first",
                "marginal_cost": np.mean,
                "capital_cost": np.mean,
                "p_nom_max": np.max,
                "p_nom_min": np.min,
            },
            "Store": {
                # "scn_name": "first",
                # "carrier": "first",
                "marginal_cost": np.mean,
                "capital_cost": np.mean,
                "e_nom": np.sum,
                "e_nom_max": np.sum,
            },
            "Load": {
                # "scn_name": "first",
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

    for data, name in zip(
        [
            network_gasgrid_c.links,
            network_gasgrid_c.buses,
            network_gasgrid_c.generators,
            network_gasgrid_c.stores,
            network_gasgrid_c.loads,
            network_gasgrid_c.storage_units,
            network_gasgrid_c.shunt_impedances,
            network_gasgrid_c.lines,
            network_gasgrid_c.transformers,
        ],
        [
            "links",
            "buses",
            "generators",
            "stores",
            "loads",
            "storage_units",
            "shunt_impedances",
            "lines",
            "transformers",
        ],
    ):
        pd.DataFrame(data).to_csv("network_clustered/" + name + "_clustered.csv")

    return network_gasgrid_c


def run_kmeans_clustering_gas(self):

    if self.args["network_clustering_kmeans"]["active"]:
        logger.info("Start k-mean clustering GAS")
        self.clustering = kmean_clustering_gas_grid(self)
        logger.info(
            "GAS Network clustered to {} buses with k-means algorithm.".format(
                self.args["network_clustering_kmeans"]["n_clusters_gas"]
            )
        )
