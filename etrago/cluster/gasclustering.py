# -*- coding: utf-8 -*-
# File description for read-the-docs
""" Gasclustering.py defines the methods to cluster gas grid networks
spatially for applications within the tool eTraGo."""

import os

if "READTHEDOCS" not in os.environ:
    import pdb

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


def get_clustering_from_busmap(
    network,
    busmap,
    with_time=True,
    bus_strategies=dict(),
    one_port_strategies=dict(),
):
    """Cluster the gas grid according to the busmap and aggregate the gas
    related technologies described as links, and the one port components
    (stores, generators and loads).
    Returns
    -------
    network_c : pypsa.Network object
        Clustered gas grid and aggregated gas related technologies.
    """

    network_c = Network()

    buses = aggregatebuses(network, busmap, bus_strategies)
    io.import_components_from_dataframe(network_c, buses, "Bus")

    if with_time:
        network_c.snapshot_weightings = network.snapshot_weightings.copy()
        network_c.set_snapshots(network.snapshots)

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
        io.import_components_from_dataframe(network_c, new_df, one_port)
        for attr, df in iteritems(new_pnl):
            io.import_series_from_dataframe(network_c, df, one_port, attr)

    for c in network.iterate_components(one_port_components):
        io.import_components_from_dataframe(
            network_c,
            c.df.assign(bus=c.df.bus.map(busmap)).dropna(subset=["bus"]),
            c.name,
        )

    if with_time:
        for c in network.iterate_components(one_port_components):
            for attr, df in iteritems(c.pnl):
                if not df.empty:
                    io.import_series_from_dataframe(network_c, df, c.name, attr)

    new_links = (
        network.links.assign(
            bus0=network.links.bus0.map(busmap), bus1=network.links.bus1.map(busmap)
        )
        .dropna(subset=["bus0", "bus1"])
        .loc[lambda df: df.bus0 != df.bus1]
    )

    new_links["link_id"] = new_links.index

    strategies = {
        "bus0": "first",
        "bus1": "first",
        "carrier": "first",
        "p_nom": "sum",
        "length": "mean",
    }
    strategies.update(
        {col: "first" for col in new_links.columns if col not in strategies}
    )

    gas_carriers = [
        "CH4",
        "H2_to_CH4",
        "CH4_to_H2",
        "H2_feedin",
        "H2_to_power",
        "power_to_H2",
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

    io.import_components_from_dataframe(network_c, combinations_final, "Link")

    non_gas_links = (
        new_links[~new_links["carrier"].isin(gas_carriers)]
        .copy()
        .drop(columns="link_id")
    )

    io.import_components_from_dataframe(network_c, non_gas_links, "Link")

    if with_time:
        for attr, df in iteritems(network.links_t):
            if not df.empty:
                io.import_series_from_dataframe(network_c, df, "Link", attr)

    io.import_components_from_dataframe(network_c, network.carriers, "Carrier")

    network_c.determine_network_topology()

    return network_c


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

    # Download H2-CH4 correspondance table
    engine = db.connection(section=etrago.args["db"])
    sql = f"""SELECT "bus_H2", "bus_CH4", scn_name FROM grid.egon_etrago_ch4_h2;"""
    df_correspondance_H2_CH4 = select_dataframe(sql, engine)
    df_correspondance_H2_CH4["bus_CH4"] = df_correspondance_H2_CH4["bus_CH4"].astype(
        str
    )
    df_correspondance_H2_CH4 = df_correspondance_H2_CH4.set_index(["bus_CH4"])

    # Create network_ch4 (grid nodes in order to create the busmap basis)
    network = etrago.network.copy()
    network_ch4 = etrago.network.copy()

    for data, name in zip([network_ch4.links, network_ch4.buses], ["links_", "buses_"]):
        pd.DataFrame(data).to_csv(name + "not_clustered.csv")

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
    busmap.index = busmap.index.astype(str)
    missing_idx = list(network.buses[~network.buses.index.isin(busmap.index)].index)
    next_bus_id = network.buses.index.astype(int).max() + 1
    new_gas_buses = [str(int(x) + next_bus_id) for x in busmap]

    busmap_idx = list(busmap.index) + missing_idx
    busmap_values = new_gas_buses + missing_idx
    busmap = pd.Series(busmap_values, index=busmap_idx)

    busmap = busmap.astype(str)
    busmap.index = busmap.index.astype(str)

    busmap.to_csv(
        "kmeans_gasgrid_busmap_"
        + str(kmean_gas_settings["n_clusters_gas"])
        + "_result.csv"
    )

    # H2 and CH4 components
    network_gas = etrago.network.copy()

    network_gasgrid_c = get_clustering_from_busmap(
        network_gas,
        busmap,
        one_port_strategies={
            "Generator": {
                "marginal_cost": np.mean,
                "capital_cost": np.mean,
                "p_nom_min": np.min,
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

    for data, name in zip(
        [network_gasgrid_c.links, network_gasgrid_c.buses], ["links_", "buses_"]
    ):
        data.to_csv(name + "clustered.csv")

    return network


def run_kmeans_clustering_gas(self):

    if self.args["network_clustering_kmeans"]["active"]:
        logger.info("Start k-mean clustering GAS")
        self.clustering = kmean_clustering_gas_grid(self)
        logger.info(
            "GAS Network clustered to {} buses with k-means algorithm.".format(
                self.args["network_clustering_kmeans"]["n_clusters_gas"]
            )
        )
