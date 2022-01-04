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
    from pypsa.networkclustering import (  # aggregategenerators, , busmap_by_stubs,
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
    with_time=True,
    line_length_factor=1.0,
    bus_strategies=dict(),
    scale_link_capital_costs=True,
    aggregate_generators_weighted=False,
    # aggregate_one_ports={},
    aggregate_generators_carriers=None,
    one_port_strategies=dict(),
    generator_strategies=dict(),
):  # ,
    # bus_strategies=dict(),

    network_c = Network()

    buses = aggregatebuses(network, busmap, bus_strategies)
    # buses_H2 = network.buses[(network.buses['carrier'] == 'H2_grid')]

    io.import_components_from_dataframe(network_c, buses, "Bus")

    if with_time:
        network_c.snapshot_weightings = network.snapshot_weightings.copy()
        network_c.set_snapshots(network.snapshots)

    # # keep attached links
    # links = network.links.copy()
    # links = links[links['carrier'] != 'CH4']
    # io.import_components_from_dataframe(network, links, "Link")

    one_port_components = [
        "Generator",
        "Load",
        "Store",
    ]  # network.one_port_components.copy()

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

    # pdb.set_trace()
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
    H2_grid_nodes = df_correspondance_H2_CH4["bus_H2"].tolist()
    H2_grid_nodes_s = [str(x) for x in H2_grid_nodes]
    print(H2_grid_nodes_s)

    # network_ch4 is supposed to contain all the gas buses (CH4 and H2) linked to the CH4 grid (eGon2035) """and links (CH4 pipes and H2-CH4 connections)"""
    network = etrago.network.copy()
    network_ch4 = etrago.network.copy()

    network_ch4.buses = network_ch4.buses[
        (network_ch4.buses["carrier"] == "CH4")
        | (network_ch4.buses["carrier"] == "H2_grid")
    ]

    network_ch4.links = network_ch4.links[
        (network_ch4.links["carrier"] == "CH4")
        | (network_ch4.links["carrier"] == "power_to_H2")
        | (network_ch4.links["carrier"] == "H2_to_power")
        | (network_ch4.links["carrier"] == "CH4_to_H2")
        | (network_ch4.links["carrier"] == "H2_feedin")
        | (network_ch4.links["carrier"] == "CH4_to_H2")
    ]

    network_ch4.generators = network_ch4.generators[
        (network_ch4.generators["carrier"] == "CH4")
    ]
    network_ch4.stores = network_ch4.stores[
        (network_ch4.stores["carrier"] == "CH4")
        | (
            (network_ch4.stores["carrier"] == "H2_overground")
            & (network_ch4.stores["bus"].isin(H2_grid_nodes_s))
        )  # won't be necessary anymore with changes of datamodel (no overground storages for saltcavern H2 buses)
    ]
    network_ch4.loads = network_ch4.loads[
        (network_ch4.loads["carrier"] == "CH4")
        | (
            (network_ch4.loads["carrier"] == "H2")
            & (network_ch4.loads["bus"].isin(H2_grid_nodes_s))
        )
    ]

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

    # TODO: Adapt for gas if relevant
    # # remove stubs
    # if kmean_settings['remove_stubs']:
    #     network.determine_network_topology()
    #     busmap = busmap_by_stubs(network)
    #     network.generators['weight'] = network.generators['p_nom']
    #     aggregate_one_ports = network.one_port_components.copy()
    #     aggregate_one_ports.discard('Generator')

    #     # reset coordinates to the new reduced guys, rather than taking an
    #     # average (copied from pypsa.networkclustering)
    #     if kmean_settings['use_reduced_coordinates']:
    #         # TODO : FIX THIS HACK THAT HAS UNEXPECTED SIDE-EFFECTS,
    #         # i.e. network is changed in place!!
    #         network.buses.loc[busmap.index, ['x', 'y']
    #                           ] = network.buses.loc[busmap, ['x', 'y']].values

    #     clustering = get_clustering_from_busmap(
    #         network,
    #         busmap,
    #         aggregate_generators_weighted=True,
    #         one_port_strategies={'StorageUnit': {'marginal_cost': np.mean,
    #                                          'capital_cost': np.mean,
    #                                          'efficiency': np.mean,
    #                                          'efficiency_dispatch': np.mean,
    #                                          'standing_loss': np.mean,
    #                                          'efficiency_store': np.mean,
    #                                          'p_min_pu': np.min}},
    #         generator_strategies={'p_nom_min':np.min,
    #                           'p_nom_opt': np.sum,
    #                           'marginal_cost': np.mean,
    #                           'capital_cost': np.mean},
    #         aggregate_one_ports=aggregate_one_ports,
    #         line_length_factor=kmean_settings['line_length_factor'])
    #     network = clustering.network

    #     weight = weight.groupby(busmap.values).sum()

    weight_ch4_s = weight_ch4.squeeze()
    # k-mean clustering for CH4
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
        "kmeans_ch4_busmap_" + str(kmean_gas_settings["n_clusters"]) + "_result.csv"
    )

    network_c = get_clustering_from_busmap(
        network_ch4,
        busmap_ch4,
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
        generator_strategies={
            "p_nom_min": np.min,
            "p_nom_opt": np.sum,
            "marginal_cost": np.mean,
            "capital_cost": np.mean,
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
            & ~(network.stores["bus"].isin(H2_grid_nodes_s))
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
    io.import_components_from_dataframe(network, network_c.buses, "Bus")
    io.import_components_from_dataframe(network, network_c.links, "Link")
    io.import_components_from_dataframe(network, network_c.stores, "Store")
    io.import_components_from_dataframe(network, network_c.loads, "Load")
    io.import_components_from_dataframe(network, network_c.generators, "Generator")

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
