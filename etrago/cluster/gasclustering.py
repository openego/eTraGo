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
    from etrago.tools.utilities import *
    from egoio.tools import db
    from pypsa import Network
    from pypsa.networkclustering import (
        aggregatebuses,
        aggregateoneport,
        busmap_by_kmeans,
        haversine_pts,
    )
    from six import iteritems


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
    carrier,
    with_time=True,
    line_length_factor=1.0,
    bus_strategies=dict(),
    scale_link_capital_costs=True,
    aggregate_generators_weighted=False,
    # aggregate_one_ports={},
    aggregate_generators_carriers=None,
    one_port_strategies=dict(),
):
    """Adapation of the get_clustering_from_busmap from pypsa (mainly removing the clustering of the lines and adjusting the one_port_components)
    Returns
    -------
    network_c : pypsa.Network object
        Clustered CH4 grid.
    """

    network_c = Network()
    buses = aggregatebuses(network, busmap, bus_strategies)
    io.import_components_from_dataframe(network_c, buses, "Bus")

    if with_time:
        network_c.snapshot_weightings = network.snapshot_weightings.copy()
        network_c.set_snapshots(network.snapshots)

    if carrier == 'CH4':
        one_port_components = ["Generator", "Load", "Store"]
    elif carrier == 'H2':
        one_port_components = ["Load", "Store"]
    print(one_port_components)# network.one_port_components.copy()


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

    new_links["length"] = np.where(
        new_links.length.notnull() & (new_links.length > 0),
        line_length_factor
        * haversine_pts(
            buses.loc[new_links["bus0"], ["x", "y"]],
            buses.loc[new_links["bus1"], ["x", "y"]],
        ),
        0,
    )
    if scale_link_capital_costs:
        new_links["capital_cost"] *= (new_links.length / network.links.length).fillna(1)

    io.import_components_from_dataframe(network_c, new_links, "Link")

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
    H2_grid_nodes = df_correspondance_H2_CH4["bus_H2"].tolist()
    H2_grid_nodes_s = [str(x) for x in H2_grid_nodes]

    # Create network_ch4
    # network_ch4  contains all the CH4 components (buses, links, generators, stores and load)
    network = etrago.network.copy()
    network_ch4 = etrago.network.copy()

    network_ch4.buses = network_ch4.buses[(network_ch4.buses["carrier"] == "CH4")]

    network_ch4.links = network_ch4.links[
        (network_ch4.links["carrier"] == "CH4")
        | (network_ch4.links["carrier"] == "central_gas_boiler")
        | (network_ch4.links["carrier"] == "CHP")
        | (network_ch4.links["carrier"] == "OCGT")
    ]

    network_ch4.generators = network_ch4.generators[
        (network_ch4.generators["carrier"] == "CH4")
    ]
    network_ch4.stores = network_ch4.stores[(network_ch4.stores["carrier"] == "CH4")]
    network_ch4.loads = network_ch4.loads[(network_ch4.loads["carrier"] == "CH4")]

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
        bus_weightings=weight_ch4_s,  # pd.Series(weight_ch4),
        n_clusters=kmean_gas_settings["n_clusters_gas"],
        n_init=kmean_gas_settings["n_init"],
        max_iter=kmean_gas_settings["max_iter"],
        tol=kmean_gas_settings["tol"],
        n_jobs=kmean_gas_settings["n_jobs"],
    )
    busmap_ch4.to_csv(
        "kmeans_ch4_busmap_" + str(kmean_gas_settings["n_clusters_gas"]) + "_result.csv"
    )
    
    network_ch4_c = get_clustering_from_busmap(
        network_ch4,
        busmap_ch4,
        carrier='CH4',
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
        },
        aggregate_generators_weighted=False,
        line_length_factor=kmean_gas_settings["line_length_factor"],
    )

    # Create H2 busmap
    busmap_ch4 = busmap_ch4.rename("CH4_nodes_c")
    busmap_h2 = pd.concat([df_correspondance_H2_CH4, busmap_ch4], axis=1, join="inner")
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

    # Cluster H2 components
    network_h2 = etrago.network.copy()
    network_h2.buses = network_h2.buses[(network_h2.buses["carrier"] == "H2_grid")]
    network_h2.links = network_h2.links[
        (network_h2.links["carrier"] == "power_to_H2")
        | (network_h2.links["carrier"] == "H2_to_power")
        | (network_h2.links["carrier"] == "CH4_to_H2")
        | (network_h2.links["carrier"] == "H2_feedin")
        | (network_h2.links["carrier"] == "CH4_to_H2")
    ]
    network_h2.stores = network_h2.stores[
        (network_h2.stores["carrier"] == "H2_overground")
        & (
            network_h2.stores["bus"].isin(H2_grid_nodes_s)
        )  # condition could be remove when datamodel is updated
    ]
    network_h2.loads = network_h2.loads[
        (network_h2.loads["carrier"] == "H2")
        & (network_h2.loads["bus"].isin(H2_grid_nodes_s))
    ]

    network_h2_c = get_clustering_from_busmap(
        network_h2,
        busmap_h2,
        carrier='H2',
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
        },
        aggregate_generators_weighted=False,
        line_length_factor=kmean_gas_settings["line_length_factor"],
    )

    # Adjust initial network
    # Delete clustered components (old version)
    network.buses = network.buses[
        (network.buses["carrier"] != "CH4") & (network.buses["carrier"] != "H2_grid")
    ]
    network.links = network.links[
        (network.links["carrier"] != "CH4")
        & (network.links["carrier"] != "central_gas_boiler")
        & (network.links["carrier"] != "CHP")
        & (network.links["carrier"] != "OCGT")
        & (network.links["carrier"] != "power_to_H2")
        & (network.links["carrier"] != "H2_to_power")
        & (network.links["carrier"] != "CH4_to_H2")
        & (network.links["carrier"] != "H2_feedin")
        & (network.links["carrier"] != "CH4_to_H2")
    ]
    network.generators = network.generators[(network.generators["carrier"] != "CH4")]

    network.stores = network.stores[
        (
            (network.stores["carrier"] != "CH4")
            & (network.stores["carrier"] != "H2_overground")
        )
        | (
            (network.stores["carrier"] == "H2_overground")
            & ~(
                network.stores["bus"].isin(H2_grid_nodes_s)
            )  # condition could be remove when datamodel is updated
        )
    ]
    network.loads = network.loads[
        ((network.loads["carrier"] != "CH4") & (network.loads["carrier"] != "H2"))
        | (
            (network.loads["carrier"] == "H2")
            & ~(network.loads["bus"].isin(H2_grid_nodes_s))
        )
    ]

    # Add clustered components
    io.import_components_from_dataframe(network, network_ch4_c.buses, "Bus")
    io.import_components_from_dataframe(network, network_h2_c.buses, "Bus")
    io.import_components_from_dataframe(network, network_ch4_c.links, "Link")
    io.import_components_from_dataframe(network, network_h2_c.links, "Link")
    io.import_components_from_dataframe(network, network_ch4_c.stores, "Store")
    io.import_components_from_dataframe(network, network_h2_c.stores, "Store")
    io.import_components_from_dataframe(network, network_ch4_c.loads, "Load")
    io.import_components_from_dataframe(network, network_h2_c.loads, "Load")
    io.import_components_from_dataframe(network, network_ch4_c.generators, "Generator")

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
