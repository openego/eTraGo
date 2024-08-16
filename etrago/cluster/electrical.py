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
""" electrical.py defines the methods to cluster power grid networks
spatially for applications within the tool eTraGo."""

import os

if "READTHEDOCS" not in os.environ:
    import logging

    from pypsa import Network
    from pypsa.clustering.spatial import (
        aggregatebuses,
        aggregateoneport,
        get_clustering_from_busmap,
    )
    from six import iteritems
    import numpy as np
    import pandas as pd
    import pypsa.io as io

    from etrago.cluster.spatial import (
        busmap_ehv_clustering,
        drop_nan_values,
        group_links,
        kmean_clustering,
        kmedoids_dijkstra_clustering,
        strategies_buses,
        strategies_generators,
        strategies_lines,
        strategies_one_ports,
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


# TODO: Workaround because of agg


def _leading(busmap, df):
    """
    Returns a function that computes the leading bus_id for a given mapped
    list of buses.

    Parameters
    -----------
    busmap : dict
        A dictionary that maps old bus_ids to new bus_ids.
    df : pandas.DataFrame
        A DataFrame containing network.buses data. Each row corresponds
        to a unique bus

    Returns
    --------
    leader : function
        A function that returns the leading bus_id for the argument `x`.
    """

    def leader(x):
        ix = busmap[x.index[0]]
        return df.loc[ix, x.name]

    return leader


def adjust_no_electric_network(
    etrago, busmap, cluster_met, apply_on="grid_model"
):
    """
    Adjusts the non-electric network based on the electrical network
    (esp. eHV network), adds the gas buses to the busmap, and creates the
    new buses for the non-electric network.

    Parameters
    ----------
    etrago : Etrago
        An instance of the Etrago class.
    busmap : dict
        A dictionary that maps old bus_ids to new bus_ids.
    cluster_met : str
        A string indicating the clustering method to be used.

    Returns
    -------
    network : pypsa.Network
        Container for all network components of the clustered network.
    busmap : dict
        Maps old bus_ids to new bus_ids including all sectors.

    """

    def find_de_closest(network, bus_ne):
        ac_ehv = network.buses[
            (network.buses.v_nom > 110)
            & (network.buses.carrier == "AC")
            & (network.buses.country == "DE")
        ]

        bus_ne_x = network.buses.loc[bus_ne, "x"]
        bus_ne_y = network.buses.loc[bus_ne, "y"]

        ac_ehv["dist"] = ac_ehv.apply(
            lambda x: ((x.x - bus_ne_x) ** 2 + (x.y - bus_ne_y) ** 2)
            ** (1 / 2),
            axis=1,
        )

        new_ehv_bus = ac_ehv.dist.idxmin()

        return new_ehv_bus

    if apply_on == "grid_model":
        network = etrago.network.copy()
    elif apply_on == "market_model":
        network = etrago.network_tsa.copy()
    else:
        logger.warning(
            """Parameter apply_on must be either 'grid_model' or 'market_model'
            """
        )

    # network2 is supposed to contain all the not electrical or gas buses
    # and links
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
        "Li_ion": "BEV_charger",
        "rural_heat": "rural_heat_pump",
    }

    no_elec_conex = []
    # busmap2 defines how the no electrical buses directly connected to AC
    # are going to be clustered
    busmap2 = {}
    # Map crossborder AC buses in case that they were not part of the k-mean
    # clustering
    # Do not apply this part if the function is used for creating the market
    # model. It adds one bus per country, which is not useful in this case.
    if apply_on != "market_model":
        if (not etrago.args["network_clustering"]["cluster_foreign_AC"]) & (
            cluster_met in ["kmeans", "kmedoids-dijkstra"]
        ):
            buses_orig = network.buses.copy()
            ac_buses_out = buses_orig[
                (buses_orig["country"] != "DE")
                & (buses_orig["carrier"] == "AC")
            ].dropna(subset=["country", "carrier"])

            for bus_out in ac_buses_out.index:
                busmap2[bus_out] = bus_out

    foreign_hv = network.buses[
        (network.buses.country != "DE")
        & (network.buses.carrier == "AC")
        & (network.buses.v_nom > 110)
    ].index
    busmap3 = pd.DataFrame(columns=["elec_bus", "carrier", "cluster"])
    for bus_ne in network2.buses.index:
        carry = network2.buses.loc[bus_ne, "carrier"]
        busmap3.at[bus_ne, "carrier"] = carry
        try:
            df = network2.links[
                (network2.links["bus1"] == bus_ne)
                & (network2.links["carrier"] == map_carrier[carry])
            ].copy()
            df["elec"] = df["bus0"].isin(busmap.keys())
            bus_hv = df[df["elec"]]["bus0"].iloc[0]
            bus_ehv = busmap[bus_hv]
            if bus_ehv not in foreign_hv:
                busmap3.at[bus_ne, "elec_bus"] = bus_ehv
            else:
                busmap3.at[bus_ne, "elec_bus"] = find_de_closest(
                    network, bus_ne
                )
        except:
            no_elec_conex.append(bus_ne)
            busmap3.at[bus_ne, "elec_bus"] = bus_ne

    for a, df in busmap3.groupby(["elec_bus", "carrier"]):
        busmap3.loc[df.index, "cluster"] = df.index[0]

    busmap3 = busmap3["cluster"].to_dict()

    if no_elec_conex:
        logger.info(
            f"""There are {len(no_elec_conex)} buses that have no direct
            connection to the electric network: {no_elec_conex}"""
        )

    # rural_heat_store buses are clustered based on the AC buses connected to
    # their corresponding rural_heat buses. Results saved in busmap4
    links_rural_store = etrago.network.links[
        etrago.network.links.carrier == "rural_heat_store_charger"
    ].copy()

    busmap4 = {}
    links_rural_store["to_ac"] = links_rural_store["bus0"].map(busmap3)
    for rural_heat_bus, df in links_rural_store.groupby("to_ac"):
        cluster_bus = df.bus1.iat[0]
        for rural_store_bus in df.bus1:
            busmap4[rural_store_bus] = cluster_bus

    # Add the gas buses to the busmap and map them to themself
    for gas_bus in network.buses[
        (network.buses["carrier"] == "H2_grid")
        | (network.buses["carrier"] == "CH4")
        | (network.buses["carrier"] == "central_heat")
        | (network.buses["carrier"] == "central_heat_store")
    ].index:
        busmap2[gas_bus] = gas_bus

    busmap = {**busmap, **busmap2, **busmap3, **busmap4}

    return network, busmap


def cluster_on_extra_high_voltage(etrago, busmap, with_time=True):
    """
    Main function of the EHV-Clustering approach. Creates a new clustered
    pypsa.Network given a busmap mapping all bus_ids to other bus_ids of the
    same network.

    Parameters
    ----------
    etrago : Etrago
        An instance of the Etrago class
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

    buses = aggregatebuses(
        network,
        busmap,
        {
            "x": _leading(busmap, network.buses),
            "y": _leading(busmap, network.buses),
            "geom": lambda x: np.nan,
            "country": lambda x: "",
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
    # Discard links connected to buses under 220 kV
    dc_links = dc_links[dc_links.bus0.isin(buses.index)]
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
    # network.generators["weight"] = 1

    for one_port in network.one_port_components.copy():
        if one_port == "Generator":
            custom_strategies = strategies_generators()

        else:
            custom_strategies = strategies_one_ports().get(one_port, {})
        new_df, new_pnl = aggregateoneport(
            network,
            busmap,
            component=one_port,
            with_time=with_time,
            custom_strategies=custom_strategies,
        )
        io.import_components_from_dataframe(network_c, new_df, one_port)
        for attr, df in iteritems(new_pnl):
            io.import_series_from_dataframe(network_c, df, one_port, attr)

    network_c.links, network_c.links_t = group_links(network_c)
    network_c.determine_network_topology()

    return (network_c, busmap)


def delete_ehv_buses_no_lines(network):
    """
    When there are AC buses totally isolated, this function deletes them in
    order to make possible the creation of busmaps based on electrical
    connections and other purposes. Additionally, it throws a warning to
    inform the user in case that any correction should be done.

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
        (~buses_ac["with_line"])
        & (~buses_ac["with_load"])
        & (~buses_ac["with_link"])
        & (~buses_ac["with_gen"])
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
    """
    Cluster the network based on Extra High Voltage (EHV) grid.

    If 'active' in the `network_clustering_ehv` argument is True, the function
    clusters the network based on the EHV grid.

    Parameters
    ----------
    self: Etrago object pointer
        The object pointer for an Etrago object.

    Returns
    -------
    None
    """

    if self.args["network_clustering_ehv"]["active"]:
        logger.info("Start ehv clustering")

        delete_ehv_buses_no_lines(self.network)

        busmap = busmap_ehv_clustering(self)

        self.network, busmap = cluster_on_extra_high_voltage(
            self, busmap, with_time=True
        )

        self.update_busmap(busmap)
        self.buses_by_country()

        # Drop nan values in timeseries after clustering
        drop_nan_values(self.network)

        logger.info("Network clustered to EHV-grid")


def select_elec_network(etrago, apply_on="grid_model"):
    """
    Selects the electric network based on the clustering settings specified
    in the Etrago object.

    Parameters
    ----------
    etrago : Etrago
        An instance of the Etrago class
    apply_on: str
        gives information about the objective of the output network. If
        "grid_model" is provided, the value assigned in the args for
        ["network_clustering"]["cluster_foreign_AC""] will define if the
        foreign buses will be included in the network. if "market_model" is
        provided, foreign buses will be always included.

    Returns
    -------
    Tuple containing:
        elec_network : pypsa.Network
            Contains the electric network
        n_clusters : int
            number of clusters used in the clustering process.
    """
    if apply_on == "grid_model":
        elec_network = etrago.network.copy()
    elif apply_on == "market_model":
        elec_network = etrago.network_tsa.copy()
    else:
        logger.warning(
            """Parameter apply_on must be either 'grid_model' or 'market_model'
            """
        )
    settings = etrago.args["network_clustering"]

    if apply_on == "grid_model":
        include_foreign = settings["cluster_foreign_AC"]
    elif apply_on == "market_model":
        include_foreign = True
    else:
        raise ValueError(
            """Parameter apply_on must be either 'grid_model' or 'market_model'
            """
        )

    if include_foreign:
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


def unify_foreign_buses(etrago):
    """
    Unifies foreign AC buses into clusters using the k-medoids algorithm with
    Dijkstra distance as a similarity measure.

    Parameters
    ----------
    etrago : Etrago
        An instance of the Etrago class

    Returns
    -------
    busmap_foreign : pd.Series
        A pandas series that maps the foreign buses to their respective
        clusters. The series index is the bus ID and the values are the
        corresponding cluster medoid IDs.
    """
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


def preprocessing(etrago, apply_on="grid_model"):
    """
    Preprocesses an Etrago object to prepare it for network clustering.

    Parameters
    ----------
    etrago : Etrago
        An instance of the Etrago class
    apply_on : string
        provide information about the objective of the preprocessing. Which
        process is going to use the result. e.g. "grid_model", "market_model".

    Returns
    -------
    network_elec : pypsa.Network
        Container for all network components of the electrical network.
    weight : pandas.Series
        A pandas.Series with the bus weighting data.
    n_clusters : int
        The number of clusters to use for network clustering.
    busmap_foreign : pandas.Series
        The Series object with the foreign bus mapping data.
    """

    if apply_on == "grid_model":
        network = etrago.network
    elif apply_on == "market_model":
        network = etrago.network_tsa
    else:
        logger.warning(
            """Parameter apply_on must be either 'grid_model' or 'market_model'
            """
        )

    settings = etrago.args["network_clustering"]

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

    if not trafo_index.empty:
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
                v_nom=380.0,
            )
            .set_index("T" + trafo_index),
            "Line",
        )
        network.lines.carrier = "AC"

        network.transformers.drop(trafo_index, inplace=True)

        for attr in network.transformers_t:
            network.transformers_t[attr] = network.transformers_t[
                attr
            ].reindex(columns=[])
    elif trafo_index.empty:
        logging.info("Your network does not have any transformer")

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

    network_elec, n_clusters = select_elec_network(etrago, apply_on=apply_on)

    if settings["method"] == "kmedoids-dijkstra":
        lines_col = network_elec.lines.columns

        # The Dijkstra clustering works using the shortest electrical path
        # between buses. In some cases, a bus has just DC connections, which
        # are considered links. Therefore it is necessary to include
        # temporarily the DC links into the lines table.
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


def postprocessing(
    etrago,
    busmap,
    busmap_foreign,
    medoid_idx=None,
    aggregate_generators_carriers=None,
    aggregate_links=True,
    apply_on="grid_model",
):
    """
    Postprocessing function for network clustering.

    Parameters
    ----------
    etrago : Etrago
        An instance of the Etrago class
    busmap : pandas.Series
        mapping between buses and clusters
    busmap_foreign : pandas.DataFrame
        mapping between foreign buses and clusters
    medoid_idx : pandas.DataFrame
        mapping between cluster indices and medoids

    Returns
    -------
    Tuple containing:
        clustering : pypsa.network
            Network object containing the clustered network
        busmap : pandas.Series
            Updated mapping between buses and clusters
    """
    settings = etrago.args["network_clustering"]
    method = settings["method"]
    num_clusters = settings["n_clusters_AC"]

    if not settings["k_elec_busmap"]:
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
        etrago, busmap, cluster_met=method, apply_on=apply_on
    )

    # merge busmap for foreign buses with the German buses
    if not settings["cluster_foreign_AC"] and (apply_on == "grid_model"):
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
        aggregate_generators_carriers=aggregate_generators_carriers,
        one_port_strategies=strategies_one_ports(),
        generator_strategies=strategies_generators(),
        aggregate_one_ports=aggregate_one_ports,
        line_length_factor=settings["line_length_factor"],
        bus_strategies=strategies_buses(),
        line_strategies=strategies_lines(),
    )

    # Drop nan values after clustering
    drop_nan_values(clustering.network)

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

    if aggregate_links:
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
        """
        Calculate the availability factor for a given generator.

        Parameters
        -----------
        gen : pandas.DataFrame
            A `pypsa.Network.generators` DataFrame.

        Returns
        -------
        cf : float
            The availability factor of the generator.

        Notes
        -----
        Availability factor is defined as the ratio of the average power
        output of the generator over the maximum power output capacity of
        the generator. If the generator is time-dependent, its average power
        output is calculated using the `network.generators_t` DataFrame.
        Otherwise, its availability factor is obtained from the
        `fixed_capacity_fac` dictionary, which contains pre-defined factors
        for fixed capacity generators. If the generator's availability factor
        cannot be found in the dictionary, it is assumed to be 1.

        """
        if gen.name in network.generators_t.p_max_pu.columns:
            cf = network.generators_t["p_max_pu"].loc[:, gen.name].mean()
        else:
            cf = network.generators.loc[gen.name, "p_max_pu"]

        return cf

    gen = network.generators[network.generators.carrier != "load shedding"][
        ["bus", "carrier", "p_nom"]
    ].copy()
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
    """
    Main method for running spatial clustering on the electrical network.
    Allows for clustering based on k-means and k-medoids dijkstra.

    Parameters
    -----------
    self
        The object pointer for an Etrago object containing all relevant
        parameters and data

    Returns
    -------
    None
    """
    if self.args["network_clustering"]["active"]:
        if self.args["spatial_disaggregation"] is not None:
            self.disaggregated_network = self.network.copy()
        else:
            self.disaggregated_network = self.network.copy(with_time=False)

        elec_network, weight, n_clusters, busmap_foreign = preprocessing(self)

        if self.args["network_clustering"]["method"] == "kmeans":
            if not self.args["network_clustering"]["k_elec_busmap"]:
                logger.info("Start k-means Clustering")

                busmap = kmean_clustering(
                    self, elec_network, weight, n_clusters
                )
                medoid_idx = pd.Series(dtype=str)
            else:
                busmap = pd.Series(dtype=str)
                medoid_idx = pd.Series(dtype=str)

        elif self.args["network_clustering"]["method"] == "kmedoids-dijkstra":
            if not self.args["network_clustering"]["k_elec_busmap"]:
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

        clustering, busmap = postprocessing(
            self, busmap, busmap_foreign, medoid_idx
        )
        self.update_busmap(busmap)

        self.network = clustering.network

        self.buses_by_country()

        self.geolocation_buses()

        # The control parameter is overwritten in pypsa's clustering.
        # The function network.determine_network_topology is called,
        # which sets slack bus(es).
        set_control_strategies(self.network)

        logger.info(
            "Network clustered to {} buses with ".format(
                self.args["network_clustering"]["n_clusters_AC"]
            )
            + self.args["network_clustering"]["method"]
        )
