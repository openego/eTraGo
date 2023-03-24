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
""" electrical.py defines the methods to cluster power grid networks
spatially for applications within the tool eTraGo."""

import os

if "READTHEDOCS" not in os.environ:
    import logging

    from pypsa import Network
    from pypsa.networkclustering import (
        aggregatebuses,
        aggregategenerators,
        aggregateoneport,
        get_clustering_from_busmap,
    )
    from six import iteritems
    import numpy as np
    import pandas as pd
    import pypsa.io as io

    from etrago.cluster.spatial import (
        busmap_from_psql,
        group_links,
        kmean_clustering,
        kmedoids_dijkstra_clustering,
        strategies_generators,
        strategies_one_ports,
    )
    from etrago.tools.utilities import *

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


def adjust_no_electric_network(etrago, busmap, cluster_met):
    network = etrago.network
    # network2 is supposed to contain all the not electrical or gas buses and links
    network2 = network.copy(with_time=False)
    network2.buses = network2.buses[
        (network2.buses["carrier"] != "AC")
        & (network2.buses["carrier"] != "CH4")
        & (network2.buses["carrier"] != "H2_grid")
        & (network2.buses["carrier"] != "rural_heat_store")
        & (network2.buses["carrier"] != "central_heat")
        & (network2.buses["carrier"] != "central_heat_store")
    ]
    map_carrier = {
        "H2_saltcavern": "power_to_H2",
        "dsm": "dsm",
        "Li ion": "BEV charger",
        "rural_heat": "rural_heat_pump",
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

    # Map crossborder AC buses in case that they were not part of the k-mean clustering
    if (not etrago.args["network_clustering"]["cluster_foreign_AC"]) & (
        cluster_met in ["kmeans", "kmedoids-dijkstra"]
    ):
        buses_orig = network.buses.copy()
        ac_buses_out = buses_orig[
            (buses_orig["country"] != "DE") & (buses_orig["carrier"] == "AC")
        ].dropna(subset=["country", "carrier"])

        for bus_out in ac_buses_out.index:
            busmap2[bus_out] = bus_out

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
            new = pd.DataFrame(
                {"cluster": busmap[bus_hv], "carrier": carry},
                index=[bus_cluster],
            )

            no_elec_to_cluster = pd.concat([no_elec_to_cluster, new])

        busmap2[bus_ne] = bus_cluster

    if no_elec_conex:
        logger.info(
            f"""There are {len(no_elec_conex)} buses that have no direct
            connection to the electric network: {no_elec_conex}"""
        )

    # rural_heat_store buses are clustered based on the AC buses connected to
    # their corresponding rural_heat buses
    links_rural_store = etrago.network.links[
        etrago.network.links.carrier == "rural_heat_store_charger"
    ].copy()

    busmap3 = {}
    links_rural_store["to_ac"] = links_rural_store["bus0"].map(busmap2)
    for rural_heat_bus, df in links_rural_store.groupby("to_ac"):
        cluster_bus = df.bus1.iat[0]
        for rural_store_bus in df.bus1:
            busmap3[rural_store_bus] = cluster_bus

    # Add the gas buses to the busmap and map them to themself
    for gas_bus in network.buses[
        (network.buses["carrier"] == "H2_grid")
        | (network.buses["carrier"] == "CH4")
        | (network.buses["carrier"] == "central_heat")
        | (network.buses["carrier"] == "central_heat_store")
    ].index:
        busmap2[gas_bus] = gas_bus

    busmap = {**busmap, **busmap2, **busmap3}

    # The new buses based on the eHV network for not electrical buses are created
    if cluster_met in ["kmeans", "kmedoids-dijkstra"]:
        network.madd(
            "Bus",
            names=no_elec_to_cluster.index,
            carrier=no_elec_to_cluster.carrier,
        )

    else:
        network.madd(
            "Bus",
            names=no_elec_to_cluster.index,
            carrier=no_elec_to_cluster.carrier.values,
            x=network.buses.loc[no_elec_to_cluster.cluster.values, "x"].values,
            y=network.buses.loc[no_elec_to_cluster.cluster.values, "y"].values,
            country=network.buses.loc[
                no_elec_to_cluster.cluster.values, "country"
            ].values,
        )
    return network, busmap


def cluster_on_extra_high_voltage(etrago, busmap, with_time=True):
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

    busmap : dict
        Maps old bus_ids to new bus_ids including all sectors.
    """

    network_c = Network()

    network, busmap = adjust_no_electric_network(
        etrago, busmap, cluster_met="ehv"
    )

    pd.DataFrame(busmap.items(), columns=["bus0", "bus1"]).to_csv(
        "ehv_elecgrid_busmap_result.csv",
        index=False,
    )

    buses = aggregatebuses(
        network,
        busmap,
        {
            "x": _leading(busmap, network.buses),
            "y": _leading(busmap, network.buses),
        },
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

    new_links = pd.concat([new_links, dc_links])
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

    return (network_c, busmap)


def delete_ehv_buses_no_lines(network):
    """
    When there are AC buses totally isolated, this function deletes them in order
    to make possible the creation of busmaps based on electrical connections
    and other purposes. Additionally, it throws a warning to inform the user
    in case that any correction should be done.
    Parameters
    ----------
    network : pypsa.network
    Returns
    -------
    None
    """
    lines = network.lines
    buses_ac = network.buses[
        (network.buses.carrier == "AC") & (network.buses.country == "DE")
    ]
    buses_in_lines = set(list(lines.bus0) + list(lines.bus1))
    buses_ac["with_line"] = buses_ac.index.isin(buses_in_lines)
    buses_ac["with_load"] = buses_ac.index.isin(network.loads.bus)
    buses_in_links = list(network.links.bus0) + list(network.links.bus1)
    buses_ac["with_link"] = buses_ac.index.isin(buses_in_links)
    buses_ac["with_gen"] = buses_ac.index.isin(network.generators.bus)

    delete_buses = buses_ac[
        (buses_ac["with_line"] == False)
        & (buses_ac["with_load"] == False)
        & (buses_ac["with_link"] == False)
        & (buses_ac["with_gen"] == False)
    ].index

    if len(delete_buses):
        logger.info(
            f"""

                    ----------------------- WARNING ---------------------------
                    THE FOLLOWING BUSES WERE DELETED BECAUSE THEY WERE ISOLATED:
                        {delete_buses.to_list()}.
                    IT IS POTENTIALLY A SIGN OF A PROBLEM IN THE DATASET
                    ----------------------- WARNING ---------------------------

                    """
        )

    network.mremove("Bus", delete_buses)

    delete_trafo = network.transformers[
        (network.transformers.bus0.isin(delete_buses))
        | (network.transformers.bus1.isin(delete_buses))
    ].index

    network.mremove("Transformer", delete_trafo)

    delete_sto_units = network.storage_units[
        network.storage_units.bus.isin(delete_buses)
    ].index

    network.mremove("StorageUnit", delete_sto_units)

    return


def ehv_clustering(self):
    if self.args["network_clustering_ehv"]:
        logger.info("Start ehv clustering")

        self.network.generators.control = "PV"

        delete_ehv_buses_no_lines(self.network)

        busmap = busmap_from_psql(self)

        self.network, busmap = cluster_on_extra_high_voltage(
            self, busmap, with_time=True
        )

        self.update_busmap(busmap)
        self.buses_by_country()

        logger.info("Network clustered to EHV-grid")


def select_elec_network(etrago):
    elec_network = etrago.network.copy()
    settings = etrago.args["network_clustering"]
    if settings["cluster_foreign_AC"]:
        elec_network.buses = elec_network.buses[
            elec_network.buses.carrier == "AC"
        ]
        elec_network.links = elec_network.links[
            (elec_network.links.carrier == "AC")
            | (elec_network.links.carrier == "DC")
        ]
        n_clusters = settings["n_clusters_AC"]
    else:
        AC_filter = elec_network.buses.carrier.values == "AC"

        foreign_buses = elec_network.buses[
            (elec_network.buses.country != "DE")
            & (elec_network.buses.carrier == "AC")
        ]

        num_neighboring_country = len(
            foreign_buses[foreign_buses.index.isin(elec_network.loads.bus)]
        )

        elec_network.buses = elec_network.buses[
            AC_filter & (elec_network.buses.country.values == "DE")
        ]
        n_clusters = settings["n_clusters_AC"] - num_neighboring_country

    # Dealing with generators
    elec_network.generators = elec_network.generators[
        elec_network.generators.bus.isin(elec_network.buses.index)
    ]

    for attr in elec_network.generators_t:
        elec_network.generators_t[attr] = elec_network.generators_t[attr].loc[
            :,
            elec_network.generators_t[attr].columns.isin(
                elec_network.generators.index
            ),
        ]

    # Dealing with loads
    elec_network.loads = elec_network.loads[
        elec_network.loads.bus.isin(elec_network.buses.index)
    ]

    for attr in elec_network.loads_t:
        elec_network.loads_t[attr] = elec_network.loads_t[attr].loc[
            :,
            elec_network.loads_t[attr].columns.isin(elec_network.loads.index),
        ]

    # Dealing with storage_units
    elec_network.storage_units = elec_network.storage_units[
        elec_network.storage_units.bus.isin(elec_network.buses.index)
    ]

    for attr in elec_network.storage_units_t:
        elec_network.storage_units_t[attr] = elec_network.storage_units_t[
            attr
        ].loc[
            :,
            elec_network.storage_units_t[attr].columns.isin(
                elec_network.storage_units.index
            ),
        ]

    # Dealing with stores
    elec_network.stores = elec_network.stores[
        elec_network.stores.bus.isin(elec_network.buses.index)
    ]

    for attr in elec_network.stores_t:
        elec_network.stores_t[attr] = elec_network.stores_t[attr].loc[
            :,
            elec_network.stores_t[attr].columns.isin(
                elec_network.stores.index
            ),
        ]

    return elec_network, n_clusters


def preprocessing(etrago):
    def unify_foreign_buses(etrago):
        network = etrago.network.copy(with_time=False)

        foreign_buses = network.buses[
            (network.buses.country != "DE") & (network.buses.carrier == "AC")
        ]
        foreign_buses_load = foreign_buses[
            (foreign_buses.index.isin(network.loads.bus))
            & (foreign_buses.carrier == "AC")
        ]

        lines_col = network.lines.columns
        # The Dijkstra clustering works using the shortest electrical path between
        # buses. In some cases, a bus has just DC connections, which are considered
        # links. Therefore it is necessary to include temporarily the DC links
        # into the lines table.
        dc = network.links[network.links.carrier == "DC"]
        str1 = "DC_"
        dc.index = f"{str1}" + dc.index
        lines_plus_dc = lines_plus_dc = pd.concat([network.lines, dc])
        lines_plus_dc = lines_plus_dc[lines_col]
        lines_plus_dc["carrier"] = "AC"

        busmap_foreign = pd.Series(dtype=str)

        for country, df in foreign_buses.groupby(by="country"):
            weight = df.apply(
                lambda x: 1 if x.name in foreign_buses_load.index else 0,
                axis=1,
            )
            n_clusters = (foreign_buses_load.country == country).sum()
            if n_clusters < len(df):
                (
                    busmap_country,
                    medoid_idx_country,
                ) = kmedoids_dijkstra_clustering(
                    etrago, df, lines_plus_dc, weight, n_clusters
                )
                medoid_idx_country.index = medoid_idx_country.index.astype(str)
                busmap_country = busmap_country.map(medoid_idx_country)
                busmap_foreign = pd.concat([busmap_foreign, busmap_country])
            else:
                for bus in df.index:
                    busmap_foreign[bus] = bus

        busmap_foreign.name = "foreign"
        busmap_foreign.index.name = "bus"

        return busmap_foreign

    network = etrago.network
    settings = etrago.args["network_clustering"]

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
            :,
            [
                "bus0",
                "bus1",
                "x",
                "s_nom",
                "capital_cost",
                "sub_network",
                "s_max_pu",
                "lifetime",
            ],
        ]
        .assign(
            x=network.transformers.x
            * (380.0 / transformer_voltages.max(axis=1)) ** 2,
            length=1,
        )
        .set_index("T" + trafo_index),
        "Line",
    )
    network.lines.carrier = "AC"

    network.transformers.drop(trafo_index, inplace=True)

    for attr in network.transformers_t:
        network.transformers_t[attr] = network.transformers_t[attr].reindex(
            columns=[]
        )

    network.buses["v_nom"].loc[network.buses.carrier.values == "AC"] = 380.0

    if network.buses.country.isna().any():
        logger.info(
            f"""

                    ----------------------- WARNING ---------------------------
                    THE FOLLOWING BUSES HAVE NOT COUNTRY DATA:
                        {network.buses[network.buses.country.isna()].index.to_list()}.
                    THEY WILL BE ASSIGNED TO GERMANY, BUT IT IS POTENTIALLY A
                    SIGN OF A PROBLEM IN THE DATASET.
                    ----------------------- WARNING ---------------------------

                    """
        )
        network.buses.country.loc[network.buses.country.isna()] = "DE"

    if settings["k_elec_busmap"] is False:
        busmap_foreign = unify_foreign_buses(etrago)
    else:
        busmap_foreign = pd.Series(name="foreign", dtype=str)

    network_elec, n_clusters = select_elec_network(etrago)

    if settings["method"] == "kmedoids-dijkstra":
        lines_col = network_elec.lines.columns

        # The Dijkstra clustering works using the shortest electrical path between
        # buses. In some cases, a bus has just DC connections, which are considered
        # links. Therefore it is necessary to include temporarily the DC links
        # into the lines table.
        dc = network.links[network.links.carrier == "DC"]
        str1 = "DC_"
        dc.index = f"{str1}" + dc.index
        lines_plus_dc = lines_plus_dc = pd.concat([network_elec.lines, dc])
        lines_plus_dc = lines_plus_dc[lines_col]
        network_elec.lines = lines_plus_dc.copy()
        network_elec.lines["carrier"] = "AC"

    # State whether to create a bus weighting and save it, create or not save
    # it, or use a bus weighting from a csv file
    if settings["bus_weight_tocsv"] is not None:
        weight = weighting_for_scenario(
            network=network_elec, save=settings["bus_weight_tocsv"]
        )
    elif settings["bus_weight_fromcsv"] is not None:
        weight = pd.read_csv(
            settings["bus_weight_fromcsv"], index_col="Bus", squeeze=True
        )
        weight.index = weight.index.astype(str)
    else:
        weight = weighting_for_scenario(network=network_elec, save=False)

    return network_elec, weight, n_clusters, busmap_foreign


def postprocessing(etrago, busmap, busmap_foreign, medoid_idx=None):
    settings = etrago.args["network_clustering"]
    method = settings["method"]
    num_clusters = settings["n_clusters_AC"]

    if settings["k_elec_busmap"] == False:
        busmap.name = "cluster"
        busmap_elec = pd.DataFrame(busmap.copy(), dtype="string")
        busmap_elec.index.name = "bus"
        busmap_elec = busmap_elec.join(busmap_foreign, how="outer")
        busmap_elec = busmap_elec.join(
            pd.Series(
                medoid_idx.index.values.astype(str),
                medoid_idx,
                name="medoid_idx",
            )
        )

        busmap_elec.to_csv(
            f"{method}_elecgrid_busmap_{num_clusters}_result.csv"
        )

    else:
        logger.info("Import Busmap for spatial clustering")
        busmap_foreign = pd.read_csv(
            settings["k_elec_busmap"],
            dtype={"bus": str, "foreign": str},
            usecols=["bus", "foreign"],
            index_col="bus",
        ).dropna()["foreign"]
        busmap = pd.read_csv(
            settings["k_elec_busmap"],
            usecols=["bus", "cluster"],
            dtype={"bus": str, "cluster": str},
            index_col="bus",
        ).dropna()["cluster"]
        medoid_idx = pd.read_csv(
            settings["k_elec_busmap"],
            usecols=["bus", "medoid_idx"],
            index_col="bus",
        ).dropna()["medoid_idx"]

        medoid_idx = pd.Series(
            medoid_idx.index.values.astype(str), medoid_idx.values.astype(int)
        )

    network, busmap = adjust_no_electric_network(
        etrago, busmap, cluster_met=method
    )

    # merge busmap for foreign buses with the German buses
    if settings["cluster_foreign_AC"] == False:
        for bus in busmap_foreign.index:
            busmap[bus] = busmap_foreign[bus]
            if bus == busmap_foreign[bus]:
                medoid_idx[bus] = bus
            medoid_idx.index = medoid_idx.index.astype("int")

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
        line_length_factor=settings["line_length_factor"],
    )

    if method == "kmedoids-dijkstra":
        for i in clustering.network.buses[
            clustering.network.buses.carrier == "AC"
        ].index:
            cluster = int(i)
            if cluster in medoid_idx.index:
                medoid = str(medoid_idx.loc[cluster])

                clustering.network.buses.at[i, "x"] = etrago.network.buses[
                    "x"
                ].loc[medoid]
                clustering.network.buses.at[i, "y"] = etrago.network.buses[
                    "y"
                ].loc[medoid]

    clustering.network.links, clustering.network.links_t = group_links(
        clustering.network
    )

    return (clustering, busmap)


def weighting_for_scenario(network, save=None):
    """
    define bus weighting based on generation, load and storage

    Parameters
    ----------
    network : pypsa.network
        Each bus in this network will receive a weight based on the
        generator, load and storages also available in the network object.
    save : str or bool, optional
        If defined, the result of the weighting will be saved in the path
        supplied here. The default is None.

    Returns
    -------
    weight : pandas.series
        Serie with the weight assigned to each bus to perform a k-mean
        clustering.

    """

    def calc_availability_factor(gen):
        if gen["carrier"] in time_dependent:
            cf = network.generators_t["p_max_pu"].loc[:, gen.name].mean()
        else:
            try:
                cf = fixed_capacity_fac[gen["carrier"]]
            except:
                print(gen)
                cf = 1
        return cf

    time_dependent = [
        "solar_rooftop",
        "solar",
        "wind_onshore",
        "wind_offshore",
    ]
    fixed_capacity_fac = {
        # A value of 1 is given to power plants where its availability
        # does not depend on the weather
        "industrial_gas_CHP": 1,
        "industrial_biomass_CHP": 1,
        "biomass": 1,
        "central_biomass_CHP": 1,
        "central_gas_CHP": 1,
        "OCGT": 1,
        "other_non_renewable": 1,
        "run_of_river": 0.50,
        "reservoir": 1,
        "gas": 1,
        "oil": 1,
        "others": 1,
        "coal": 1,
        "lignite": 1,
        "nuclear": 1,
    }

    gen = network.generators[["bus", "carrier", "p_nom"]].copy()
    gen["cf"] = gen.apply(calc_availability_factor, axis=1)
    gen["weight"] = gen["p_nom"] * gen["cf"]

    gen = (
        gen.groupby("bus")
        .weight.sum()
        .reindex(network.buses.index, fill_value=0.0)
    )

    storage = (
        network.storage_units.groupby("bus")
        .p_nom.sum()
        .reindex(network.buses.index, fill_value=0.0)
    )

    load = (
        network.loads_t.p_set.mean()
        .groupby(network.loads.bus)
        .sum()
        .reindex(network.buses.index, fill_value=0.0)
    )

    w = gen + storage + load
    weight = ((w * (100000.0 / w.max())).astype(int)).reindex(
        network.buses.index, fill_value=1
    )

    weight[weight == 0] = 1

    if save:
        weight.to_csv(save)

    return weight


def run_spatial_clustering(self):
    if self.args["network_clustering"]["active"]:
        self.network.generators.control = "PV"

        elec_network, weight, n_clusters, busmap_foreign = preprocessing(self)

        if self.args["network_clustering"]["method"] == "kmeans":
            if self.args["network_clustering"]["k_elec_busmap"] == False:
                logger.info("Start k-means Clustering")

                busmap = kmean_clustering(
                    self, elec_network, weight, n_clusters
                )
                medoid_idx = pd.Series(dtype=str)
            else:
                busmap = pd.Series(dtype=str)
                medoid_idx = pd.Series(dtype=str)

        elif self.args["network_clustering"]["method"] == "kmedoids-dijkstra":
            if self.args["network_clustering"]["k_elec_busmap"] == False:
                logger.info("Start k-medoids Dijkstra Clustering")

                busmap, medoid_idx = kmedoids_dijkstra_clustering(
                    self,
                    elec_network.buses,
                    elec_network.lines,
                    weight,
                    n_clusters,
                )

            else:
                busmap = pd.Series(dtype=str)
                medoid_idx = pd.Series(dtype=str)

        self.clustering, busmap = postprocessing(
            self, busmap, busmap_foreign, medoid_idx
        )
        self.update_busmap(busmap)

        if self.args["disaggregation"] != None:
            self.disaggregated_network = self.network.copy()
        else:
            self.disaggregated_network = self.network.copy(with_time=False)

        self.network = self.clustering.network

        self.buses_by_country()

        self.geolocation_buses()

        self.network.generators.control[
            self.network.generators.control == ""
        ] = "PV"
        logger.info(
            "Network clustered to {} buses with ".format(
                self.args["network_clustering"]["n_clusters_AC"]
            )
            + self.args["network_clustering"]["method"]
        )
