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
""" gas.py defines the methods to cluster gas grid networks
spatially for applications within the tool eTraGo."""

import os

if "READTHEDOCS" not in os.environ:
    import logging

    from pypsa import Network
    from pypsa.networkclustering import (
        aggregatebuses,
        aggregateoneport,
        busmap_by_kmeans,
    )
    from six import iteritems
    import numpy as np
    import pandas as pd
    import pypsa.io as io

    from etrago.cluster.spatial import (
        group_links,
        kmedoids_dijkstra_clustering,
        sum_with_inf,
    )
    from etrago.tools.utilities import set_control_strategies

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


def preprocessing(etrago):
    """
    Preprocesses the gas network data from the given Etrago object for the
    spatial clustering process of the CH4 grid.

    Parameters
    ----------
    etrago : Etrago
        An instance of the Etrago class

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If `settings["n_clusters_gas"]` is less than or equal to the number of
        neighboring country gas buses.
    """

    # Create network_ch4 (grid nodes in order to create the busmap basis)
    network_ch4 = Network()

    buses_ch4 = etrago.network.buses
    links_ch4 = etrago.network.links
    io.import_components_from_dataframe(network_ch4, buses_ch4, "Bus")
    io.import_components_from_dataframe(network_ch4, links_ch4, "Link")

    # Cluster ch4 buses
    settings = etrago.args["network_clustering"]

    ch4_filter = network_ch4.buses["carrier"].values == "CH4"

    num_neighboring_country = (
        ch4_filter & (network_ch4.buses["country"] != "DE")
    ).sum()

    network_ch4.links = network_ch4.links.loc[
        network_ch4.links["bus0"].isin(network_ch4.buses.loc[ch4_filter].index)
        & network_ch4.links["bus1"].isin(
            network_ch4.buses.loc[ch4_filter].index
        )
    ]

    # select buses dependent on whether they should be clustered in
    # (only DE or DE+foreign)
    if not settings["cluster_foreign_gas"]:
        network_ch4.buses = network_ch4.buses.loc[
            ch4_filter & (network_ch4.buses["country"].values == "DE")
        ]

        if settings["n_clusters_gas"] <= num_neighboring_country:
            msg = (
                "The number of clusters for the gas sector ("
                + str(settings["n_clusters_gas"])
                + ") must be higher than the number of neighboring country "
                + "gas buses ("
                + str(num_neighboring_country)
                + ")."
            )
            raise ValueError(msg)
        n_clusters = settings["n_clusters_gas"] - num_neighboring_country
    else:
        network_ch4.buses = network_ch4.buses.loc[ch4_filter]
        n_clusters = settings["n_clusters_gas"]

    def weighting_for_scenario(ch4_buses, save=None):
        """
        Calculate CH4-bus weightings dependant on the connected
        CH4-loads, CH4-generators and non-transport link capacities.
        Stores are not considered for the clustering.

        Parameters
        ----------
        ch4_buses : pandas.DataFrame
            Dataframe with CH4 etrago.network.buses to weight.
        save : str or bool
            Path to save weightings to as .csv

        Returns
        -------
        weightings : pandas.Series
            Integer weighting for each ch4_buses.index
        """

        MAX_WEIGHT = 1e5  # relevant only for foreign nodes with extra high
        # CH4 generation capacity

        to_neglect = [
            "CH4",
            "H2_to_CH4",
            "CH4_to_H2",
            "H2_feedin",
        ]

        # get all non-transport and non-H2 related links for each bus
        rel_links = {}
        for i in ch4_buses.index:
            rel_links[i] = etrago.network.links.loc[
                (
                    etrago.network.links.bus0.isin([i])
                    | etrago.network.links.bus1.isin([i])
                )
                & ~etrago.network.links.carrier.isin(to_neglect)
            ].index
        # get all generators and loads related to ch4_buses
        generators_ = pd.Series(
            etrago.network.generators[
                etrago.network.generators.carrier != "load shedding"
            ].index,
            index=etrago.network.generators[
                etrago.network.generators.carrier != "load shedding"
            ].bus,
        )
        buses_CH4_gen = generators_.index.intersection(rel_links.keys())
        loads_ = pd.Series(
            etrago.network.loads.index, index=etrago.network.loads.bus
        )
        buses_CH4_load = loads_.index.intersection(rel_links.keys())

        # sum up all relevant entities and cast to integer
        # Note: rel_links will hold the weightings for each bus afterwards
        for i in rel_links:
            rel_links[i] = etrago.network.links.loc[rel_links[i]].p_nom.sum()
            if i in buses_CH4_gen:
                rel_links[i] += etrago.network.generators.loc[
                    generators_.loc[i]
                ].p_nom.sum()
            if i in buses_CH4_load:
                rel_links[i] += (
                    etrago.network.loads_t.p_set.loc[:, loads_.loc[i]]
                    .mean()
                    .sum()
                )
            rel_links[i] = min(int(rel_links[i]), MAX_WEIGHT)
        weightings = pd.DataFrame.from_dict(rel_links, orient="index")

        if save:
            weightings.to_csv(save)
        return weightings

    # State whether to create a bus weighting and save it, create or not save
    # it, or use a bus weighting from a csv file
    if settings["gas_weight_tocsv"] is not None:
        weight_ch4 = weighting_for_scenario(
            network_ch4.buses,
            settings["gas_weight_tocsv"],
        )
    elif settings["gas_weight_fromcsv"] is not None:
        # create DataFrame with uniform weightings for all ch4_buses
        weight_ch4 = pd.DataFrame([1] * len(buses_ch4), index=buses_ch4.index)
        loaded_weights = pd.read_csv(
            settings["gas_weight_fromcsv"], index_col=0
        )
        # load weights into previously created DataFrame
        loaded_weights.index = loaded_weights.index.astype(str)
        weight_ch4.loc[loaded_weights.index] = loaded_weights
    else:
        weight_ch4 = weighting_for_scenario(network_ch4.buses, save=False)
    return network_ch4, weight_ch4.squeeze(), n_clusters


def kmean_clustering_gas(etrago, network_ch4, weight, n_clusters):
    """
    Performs K-means clustering on the gas network data in the given
    `network_ch4` pypsa.Network object.

    Parameters
    ----------
    etrago : Etrago
        An instance of the Etrago class
    network_ch4 : pypsa.Network
        A Network object containing the gas network data.
    weight : str or None
        The name of the bus weighting column to use for clustering. If None,
        unweighted clustering is performed.
    n_clusters : int
        The number of clusters to create.

    Returns
    -------
    busmap : pandas.Series
        A pandas.Series object mapping each bus in the CH4 network to its
        corresponding cluster ID
    None
        None is returned because k-means clustering makes no use of medoids
    """
    settings = etrago.args["network_clustering"]

    busmap = busmap_by_kmeans(
        network_ch4,
        bus_weightings=weight,
        n_clusters=n_clusters,
        n_init=settings["n_init"],
        max_iter=settings["max_iter"],
        tol=settings["tol"],
        random_state=settings["random_state"],
    )

    return busmap, None


def get_h2_clusters(etrago, busmap_ch4):
    """
    Maps H2 buses to CH4 cluster IDds and creates unique H2 cluster IDs.

    Parameters
    ----------
    etrago : Etrago
        An instance of the Etrago class
    busmap_ch4 : pd.Series
        A Pandas Series mapping each bus in the CH4 network to its
        corresponding cluster ID.

    Returns
    -------
    busmap : pd.Series
        A Pandas Series mapping each bus in the combined CH4 and H2 network
        to its corresponding cluster ID.
    """
    # Mapping of H2 buses to new CH4 cluster IDs
    busmap_h2 = pd.Series(
        busmap_ch4.loc[etrago.ch4_h2_mapping.index].values,
        index=etrago.ch4_h2_mapping.values,
    )

    # Create unique H2 cluster IDs
    n_gas = etrago.args["network_clustering"]["n_clusters_gas"]
    busmap_h2 = (busmap_h2.astype(int) + n_gas).astype(str)

    busmap_h2 = busmap_h2.squeeze()

    busmap = pd.concat([busmap_ch4, busmap_h2])

    return busmap


def gas_postprocessing(etrago, busmap, medoid_idx=None):
    """
    Performs the postprocessing for the gas grid clustering based on the
    provided busmap
    and returns the clustered network.

    Parameters
    ----------
    etrago : Etrago
        An instance of the Etrago class
    busmap : pd.Series
        A Pandas Series mapping each bus to its corresponding cluster ID.
    medoid_idx : pd.Series
        A pandas.Series object containing the medoid indices for the gas
        network.

    Returns
    -------
    network_gasgrid_c : pypsa.Network
        A pypsa.Network containing the clustered network.
    busmap : pd.Series
        A Pandas Series mapping each bus to its corresponding cluster ID.
    """
    settings = etrago.args["network_clustering"]

    if settings["k_gas_busmap"] is False:
        if settings["method_gas"] == "kmeans":
            busmap.index.name = "bus_id"
            busmap.name = "cluster"
            busmap.to_csv(
                "kmeans_gasgrid_busmap_"
                + str(settings["n_clusters_gas"])
                + "_result.csv"
            )

        else:
            busmap.name = "cluster"
            busmap_ind = pd.Series(
                medoid_idx[busmap.values.astype(int)].values,
                index=busmap.index,
                dtype=pd.StringDtype(),
            )
            busmap_ind.name = "medoid_idx"

            export = pd.concat([busmap, busmap_ind], axis=1)
            export.index.name = "bus_id"
            export.to_csv(
                "kmedoids-dijkstra_gasgrid_busmap_"
                + str(settings["n_clusters_gas"])
                + "_result.csv"
            )

    if "H2" in etrago.network.buses.carrier.unique():
        busmap = get_h2_clusters(etrago, busmap)

    # Add all other buses to busmap
    missing_idx = list(
        etrago.network.buses[
            (~etrago.network.buses.index.isin(busmap.index))
        ].index
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

    network_gasgrid_c = get_clustering_from_busmap(
        etrago.network,
        busmap,
        bus_strategies={
            "country": "first",
        },
        one_port_strategies={
            "Generator": {
                "marginal_cost": np.mean,
                "capital_cost": np.mean,
                "p_nom_max": np.sum,
                "p_nom_min": np.sum,
                "e_nom_max": np.sum,
            },
            "Store": {
                "marginal_cost": np.mean,
                "capital_cost": np.mean,
                "e_nom": np.sum,
                "e_nom_max": sum_with_inf,
            },
            "Load": {
                "p_set": np.sum,
            },
        },
    )

    # aggregation of the links and links time series
    network_gasgrid_c.links, network_gasgrid_c.links_t = group_links(
        network_gasgrid_c
    )

    # Overwrite p_nom of links with carrier "H2_feedin" (eGon2035 only)
    if etrago.args["scn_name"] == "eGon2035":
        H2_energy_share = 0.05053  # H2 energy share via volumetric share
        # outsourced in a mixture of H2 and CH4 with 15 %vol share
        feed_in = network_gasgrid_c.links.loc[
            network_gasgrid_c.links.carrier == "H2_feedin"
        ]
        pipeline_capacities = network_gasgrid_c.links.loc[
            network_gasgrid_c.links.carrier == "CH4"
        ]

        for bus in feed_in["bus1"].values:
            # calculate the total pipeline capacity connected to a specific bus
            nodal_capacity = pipeline_capacities.loc[
                (pipeline_capacities["bus0"] == bus)
                | (pipeline_capacities["bus1"] == bus),
                "p_nom",
            ].sum()
            # multiply total pipeline capacity with H2 energy share
            # corresponding to volumetric share
            network_gasgrid_c.links.loc[
                (network_gasgrid_c.links["bus1"].values == bus)
                & (network_gasgrid_c.links["carrier"].values == "H2_feedin"),
                "p_nom",
            ] = (
                nodal_capacity * H2_energy_share
            )
    # Insert components not related to the gas clustering
    other_components = ["Line", "StorageUnit", "ShuntImpedance", "Transformer"]

    for c in etrago.network.iterate_components(other_components):
        io.import_components_from_dataframe(
            network_gasgrid_c,
            c.df,
            c.name,
        )
        for attr, df in c.pnl.items():
            if not df.empty:
                io.import_series_from_dataframe(
                    network_gasgrid_c,
                    df,
                    c.name,
                    attr,
                )
    io.import_components_from_dataframe(
        network_gasgrid_c, etrago.network.carriers, "Carrier"
    )

    network_gasgrid_c.determine_network_topology()

    # Adjust x and y coordinates of 'CH4' and 'H2_grid' medoids
    if settings["method_gas"] == "kmedoids-dijkstra" and len(medoid_idx) > 0:
        for i in network_gasgrid_c.buses[
            network_gasgrid_c.buses.carrier == "CH4"
        ].index:
            cluster = str(i)
            if cluster in busmap[medoid_idx].values:
                medoid = busmap[medoid_idx][
                    busmap[medoid_idx] == cluster
                ].index
                h2_idx = network_gasgrid_c.buses.loc[
                    (network_gasgrid_c.buses.carrier == "H2_grid")
                    & (
                        network_gasgrid_c.buses.y
                        == network_gasgrid_c.buses.at[i, "y"]
                    )
                    & (
                        network_gasgrid_c.buses.x
                        == network_gasgrid_c.buses.at[i, "x"]
                    )
                ]
                if len(h2_idx) > 0:
                    h2_idx = h2_idx.index.tolist()[0]
                    network_gasgrid_c.buses.at[
                        h2_idx, "x"
                    ] = etrago.network.buses["x"].loc[medoid]
                    network_gasgrid_c.buses.at[
                        h2_idx, "y"
                    ] = etrago.network.buses["y"].loc[medoid]
                network_gasgrid_c.buses.at[i, "x"] = etrago.network.buses[
                    "x"
                ].loc[medoid]
                network_gasgrid_c.buses.at[i, "y"] = etrago.network.buses[
                    "y"
                ].loc[medoid]
    return (network_gasgrid_c, busmap)


def highestInteger(potentially_numbers):
    """Fetch the highest number of a series with mixed types

    Parameters
    ----------
    potentially_numbers : pandas.Series
        Series with mixed dtypes, potentially containing numbers.

    Returns
    -------
    highest : int
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


def simultaneous_sector_coupling(
    network, busmap, carrier_based, carrier_to_cluster
):
    """
    Cluster sector coupling technology based on multiple connected carriers.

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
    buses_clustered = network.buses[
        network.buses["carrier"].isin(carrier_based)
    ]
    buses_to_cluster = network.buses[
        network.buses["carrier"] == carrier_to_cluster
    ]
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
    busmap = {
        bus_id: bus_num + next_bus_id for bus_id, bus_num in busmap.items()
    }

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


def consecutive_sector_coupling(
    network, busmap, carrier_based, carrier_to_cluster
):
    """
    Cluster sector coupling technology based on single connected carriers.

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
    busmap_sc : dict
        Busmap for the sector coupled cluster.
    """
    next_bus_id = highestInteger(busmap.values) + 1
    buses_to_skip = network.buses[
        network.buses["carrier"] == carrier_to_cluster + "_store"
    ]
    buses_to_cluster = network.buses[
        network.buses["carrier"] == carrier_to_cluster
    ]
    buses_clustered = network.buses[
        network.buses["carrier"] == carrier_based[0]
    ]
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
    buses_to_cluster = buses_to_cluster[
        ~buses_to_cluster.index.isin(busmap_sc.keys())
    ]

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
    """
    Create busmap for sector coupled carrier based on multiple other carriers.

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
    busmap : dict
        Busmap for the sector coupled carrier.
    """
    clusters = pd.Series()
    for bus_id in buses_to_cluster.index:
        clusters.loc[bus_id] = tuple(
            sorted(
                connected_links.loc[
                    connected_links["bus1_clustered"] == bus_id,
                    "bus0_clustered",
                ].unique()
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
    """
    Create busmap for sector coupled carrier based on single other carrier.

    Parameters
    ----------
    connected_links : pandas.DataFrame
        Links that connect from the buses with other carrier to the
        buses of the sector coupled carrier.

    Returns
    -------
    busmap : dict
        Busmap for the sector coupled carrier.
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
    """
    Aggregates components of the given network based on a bus mapping and
    returns a clustered gas grid pypsa.Network.

    Parameters
    ----------
    network : pypsa.Network
        The input pypsa.Network object
    busmap : pandas.Sereies :
        A mapping of buses to clusters
    line_length_factor : float
        A factor used to adjust the length of new links created during
        aggregation. Default is 1.0.
    with_time : bool
        Determines whether to copy the time-dependent properties of the input
        network to the output network. Default is True.
    bus_strategies : dict
        A dictionary of custom strategies to use during the aggregation step.
        Default is an empty dictionary.
    one_port_strategies : dict
        A dictionary of custom strategies to use during the one-port component
        aggregation step. Default is an empty dictionary.

    Returns
    -------
    network_gasgrid_c : pypsa.Network
        A new gas grid pypsa.Network object with aggregated components based
        on the bus mapping.
    """
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
        network_gasgrid_c.snapshot_weightings = (
            network.snapshot_weightings.copy()
        )
    # Aggregate one port components
    one_port_components = ["Generator", "Load", "Store"]

    for one_port in one_port_components:
        new_df, new_pnl = aggregateoneport(
            network,
            busmap,
            component=one_port,
            with_time=with_time,
            custom_strategies=one_port_strategies.get(one_port, {}),
        )
        io.import_components_from_dataframe(
            network_gasgrid_c, new_df, one_port
        )
        for attr, df in iteritems(new_pnl):
            io.import_series_from_dataframe(
                network_gasgrid_c, df, one_port, attr
            )
    # Aggregate links
    new_links = (
        network.links.assign(
            bus0=network.links.bus0.map(busmap),
            bus1=network.links.bus1.map(busmap),
        )
        .dropna(subset=["bus0", "bus1"])
        .loc[lambda df: df.bus0 != df.bus1]
    )

    # preparation for CH4 pipeline aggregation:
    # pipelines are treated differently compared to other links, since all of
    # them will be considered bidirectional. That means, if a pipeline exists,
    # that connects one cluster with a different one simultaneously with a
    # pipeline that connects these two clusters in reversed order (e.g. bus0=1,
    # bus1=12 and bus0=12, bus1=1) they are aggregated to a single pipeline.
    # therefore, the order of bus0/bus1 is adjusted
    pipeline_mask = new_links["carrier"] == "CH4"
    sorted_buses = np.sort(
        new_links.loc[pipeline_mask, ["bus0", "bus1"]].values, 1
    )
    new_links.loc[pipeline_mask, ["bus0", "bus1"]] = sorted_buses

    # import the links and the respective time series with the bus0 and bus1
    # values updated from the busmap
    io.import_components_from_dataframe(network_gasgrid_c, new_links, "Link")

    if with_time:
        for attr, df in network.links_t.items():
            if not df.empty:
                io.import_series_from_dataframe(
                    network_gasgrid_c, df, "Link", attr
                )
    return network_gasgrid_c


def run_spatial_clustering_gas(self):
    """
    Performs spatial clustering on the gas network using either K-means or
    K-medoids-Dijkstra algorithm. Updates the network topology by aggregating
    buses and links, and then performs postprocessing to finalize the changes.

    Returns
    --------
    None

    Raises
    -------
    ValueError: If the selected method is not "kmeans" or "kmedoids-dijkstra".

    """
    if "CH4" in self.network.buses.carrier.values:
        settings = self.args["network_clustering"]

        if settings["active"]:
            method = settings["method_gas"]
            logger.info(f"Start {method} clustering GAS")

            gas_network, weight, n_clusters = preprocessing(self)

            if method == "kmeans":
                if settings["k_gas_busmap"]:
                    busmap = pd.read_csv(
                        settings["k_gas_busmap"],
                        index_col="bus_id",
                        dtype=pd.StringDtype(),
                    ).squeeze()
                    medoid_idx = None
                else:
                    busmap, medoid_idx = kmean_clustering_gas(
                        self, gas_network, weight, n_clusters
                    )

            elif method == "kmedoids-dijkstra":
                if settings["k_gas_busmap"]:
                    busmap = pd.read_csv(
                        settings["k_gas_busmap"],
                        index_col="bus_id",
                        dtype=pd.StringDtype(),
                    )
                    medoid_idx = pd.Series(
                        busmap["medoid_idx"].unique(),
                        index=busmap["cluster"].unique(),
                        dtype=pd.StringDtype(),
                    )
                    busmap = busmap["cluster"]

                else:
                    busmap, medoid_idx = kmedoids_dijkstra_clustering(
                        self,
                        gas_network.buses,
                        gas_network.links,
                        weight,
                        n_clusters,
                    )

            else:
                msg = (
                    'Please select "kmeans" or "kmedoids-dijkstra" as '
                    "spatial clustering method for the gas network"
                )
                raise ValueError(msg)
            self.network, busmap = gas_postprocessing(self, busmap, medoid_idx)

            self.update_busmap(busmap)

            # The control parameter is overwritten in pypsa's clustering.
            # The function network.determine_network_topology is called,
            # which sets slack bus(es).
            set_control_strategies(self.network)

            logger.info(
                """GAS Network clustered to {} DE-buses and {} foreign buses
                 with {} algorithm.""".format(
                    len(
                        self.network.buses.loc[
                            (self.network.buses.carrier == "CH4")
                            & (self.network.buses.country == "DE")
                        ]
                    ),
                    len(
                        self.network.buses.loc[
                            (self.network.buses.carrier == "CH4")
                            & (self.network.buses.country != "DE")
                        ]
                    ),
                    method,
                )
            )
