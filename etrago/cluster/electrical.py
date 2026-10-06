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
"""electrical.py defines the methods to cluster power grid networks
spatially for applications within the tool eTraGo."""

import logging
import os

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

logger = logging.getLogger(__name__)

if "READTHEDOCS" not in os.environ:

    from etrago.cluster.spatial import (
        busmap_ehv_clustering,
        drop_nan_values,
        focus_weighting,
        get_focus_protected_buses,
        group_links,
        kmean_clustering,
        kmedoids_dijkstra_clustering,
        strategies_buses,
        strategies_generators,
        strategies_lines,
        strategies_one_ports,
    )
    from etrago.tools.utilities import set_control_strategies


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
    # resp: Buses that are aggregated only based on the AC bus they are
    # connected to via a link (carriers in map_carrier)
    network2 = network.copy(with_time=False)

    if etrago.args["scn_name"] == "eGon100RE":
        map_carrier = {
            "dsm": "dsm",
            "O2": "PtH2_O2",
        }
    else:
        if etrago.args["method"]["distribution_grids"]:
            map_carrier = {
                "H2_saltcavern": "power_to_H2",
                "dsm": "dsm",
                "distribution_grid": "distribution_grid",
            }
        else:
            map_carrier = {
                "H2_saltcavern": "power_to_H2",
                "dsm": "dsm",
                "Li ion": "BEV charger",
                "Li_ion": "BEV_charger",
                "O2": "PtH2_O2",
                "rural_heat": "rural_heat_pump",
            }

    # network2 contains all busses that will be clustered only based on AC
    # connection
    network2.buses = network2.buses[
        network2.buses["carrier"].isin(map_carrier.keys())
    ]

    no_elec_conex = []
    # busmap2 defines how the no electrical buses directly connected to AC
    # are going to be clustered
    busmap2 = {}
    # Map crossborder AC buses in case that they were not part of the k-mean
    # clustering
    # Do not apply this part if the function is used for creating the market
    # model. It adds one bus per country, which is not useful in this case.
    if apply_on != "market_model":
        if (etrago.args["network_clustering"]["method"]["per_country"]) & (
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

    busmap4 = {}
    if "distribution_grid" in etrago.network.buses.carrier.unique():
        # rural_heat and BEV buses are clustered based on the AC buses
        # connected to their corresponding distribution grid buses
        for carrier in ["rural_heat_pump", "BEV_charger"]:
            links_from_dg_buses = etrago.network.links[
                etrago.network.links.carrier == carrier
            ].copy()

            links_from_dg_buses["to_ac"] = links_from_dg_buses["bus0"].map(
                busmap3
            )
            for bus, df in links_from_dg_buses.groupby("to_ac"):
                cluster_bus = df.bus1.iat[0]
                for new_bus in df.bus1:
                    busmap4[new_bus] = cluster_bus

    heat_store_links = etrago.network.links[
        etrago.network.links.carrier.isin(["rural_heat_store_charger"])
    ].copy()

    busmap5 = {}
    if "distribution_grid" in etrago.network.buses.carrier.unique():
        base_busmap = busmap4
    else:
        base_busmap = busmap3

    heat_store_links["to_ac"] = heat_store_links["bus0"].map(base_busmap)
    for bus, df in heat_store_links.groupby("to_ac"):
        cluster_bus = df.bus1.iat[0]
        for new_bus in df.bus1:
            busmap5[new_bus] = cluster_bus

    # Add the buses not related to AC to the busmap and map them to themself
    for no_ac_bus in network.buses[
        ~network.buses["carrier"].isin(
            np.append(network2.buses.carrier.unique(), "AC")
        )
    ].index:
        busmap2[no_ac_bus] = no_ac_bus
    busmap = {**busmap, **busmap2, **busmap3, **busmap4, **busmap5}

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
        logger.info(f"""

                ----------------------- WARNING ---------------------------
                THE FOLLOWING BUSES WERE DELETED BECAUSE THEY WERE ISOLATED:
                    {delete_buses.to_list()}.
                IT IS POTENTIALLY A SIGN OF A PROBLEM IN THE DATASET
                ----------------------- WARNING ---------------------------

                """)

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
        ["network_clustering"]["method"]["per_country""] will
        define if the foreign buses will be included in the network.
        If "market_model" is provided, foreign buses will be always included.

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
    settings = etrago.args["network_clustering"]["electricity_grid"]

    if apply_on == "grid_model":
        include_foreign = not etrago.args["network_clustering"]["method"][
            "per_country"
        ]
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
        n_clusters = settings["n_clusters"]
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
        n_clusters = settings["n_clusters"] - num_neighboring_country

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
                    "s_nom_extendable",
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
        logger.info(f"""

                ----------------------- WARNING ---------------------------
                THE FOLLOWING BUSES HAVE NOT COUNTRY DATA:
                {network.buses[network.buses.country.isna()].index.to_list()}.
                THEY WILL BE ASSIGNED TO GERMANY, BUT IT IS POTENTIALLY A
                SIGN OF A PROBLEM IN THE DATASET.
                ----------------------- WARNING ---------------------------

                """)
        network.buses.country.loc[network.buses.country.isna()] = "DE"

    if settings["electricity_grid"]["k_elec_busmap"] is False:
        busmap_foreign = unify_foreign_buses(etrago)
    else:
        busmap_foreign = pd.Series(name="foreign", dtype=str)

    network_elec, n_clusters = select_elec_network(etrago, apply_on=apply_on)

    if (
        settings["method"]["algorithm"] == "kmedoids-dijkstra"
        or settings["method"]["focus_region"] is not None
    ):
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

    # weight buses for clustering
    weight = weighting_for_scenario(network=network, save=False)

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
    settings = etrago.args["network_clustering"]["electricity_grid"]
    method = etrago.args["network_clustering"]["method"]["algorithm"]
    num_clusters = settings["n_clusters"]

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
    if etrago.args["network_clustering"]["method"]["per_country"] and (
        apply_on == "grid_model"
    ):
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
        line_length_factor=etrago.args["network_clustering"]["method"][
            "line_length_factor"
        ],
        bus_strategies=strategies_buses(),
        line_strategies=strategies_lines(),
    )

    # Drop nan values after clustering
    drop_nan_values(clustering.network)

    if method == "kmedoids-dijkstra":

        # ------------------------------------------------------------------
        # Restore geographical coordinates of ordinary k-medoids clusters.
        #
        # Standard k-medoids cluster IDs are numeric (e.g. "0", "1", ...).
        #
        # With explicit focus-region protection, additional cluster IDs such
        # as:
        #
        #     focus_12345
        #     boundary_10868
        #
        # are deliberately created. These are singleton clusters containing
        # original AC buses and therefore already have the correct original
        # coordinates after aggregation.
        #
        # They must NOT be interpreted as integer k-medoids cluster IDs.
        # ------------------------------------------------------------------

        if medoid_idx is None:
            medoid_idx = pd.Series(dtype=str)

        # Make lookup independent of whether medoid_idx currently uses
        # integer or string cluster labels.
        medoid_lookup = medoid_idx.copy()
        medoid_lookup.index = medoid_lookup.index.astype(str)

        ac_buses = clustering.network.buses[
            clustering.network.buses.carrier == "AC"
        ].index

        n_medoid_coordinates_restored = 0
        n_protected_focus_buses = 0
        n_protected_boundary_buses = 0
        n_other_non_medoid_buses = 0

        for i in ac_buses:

            cluster_label = str(i)

            # --------------------------------------------------------------
            # Explicitly protected study-region bus
            # --------------------------------------------------------------
            if cluster_label.startswith("focus_"):

                n_protected_focus_buses += 1

                # Singleton cluster:
                # its x/y coordinates already correspond to the original bus.
                continue

            # --------------------------------------------------------------
            # Explicitly protected first-ring boundary bus
            # --------------------------------------------------------------
            if cluster_label.startswith("boundary_"):

                n_protected_boundary_buses += 1

                # Singleton cluster:
                # its x/y coordinates already correspond to the original bus.
                continue

            # --------------------------------------------------------------
            # Ordinary k-medoids cluster
            # --------------------------------------------------------------
            if cluster_label in medoid_lookup.index:

                medoid = str(medoid_lookup.loc[cluster_label])

                if medoid not in etrago.network.buses.index.astype(str):

                    logger.warning(
                        "Medoid bus '%s' for cluster '%s' could not be found "
                        "in the original network. Cluster coordinates are "
                        "left unchanged.",
                        medoid,
                        cluster_label,
                    )

                    continue

                # Because the original network index may not itself be stored
                # as str, resolve the actual index value safely.
                original_bus_index = etrago.network.buses.index[
                    etrago.network.buses.index.astype(str) == medoid
                ][0]

                clustering.network.buses.at[
                    i,
                    "x",
                ] = etrago.network.buses.at[
                    original_bus_index,
                    "x",
                ]

                clustering.network.buses.at[
                    i,
                    "y",
                ] = etrago.network.buses.at[
                    original_bus_index,
                    "y",
                ]

                n_medoid_coordinates_restored += 1

            else:

                # This may occur for buses retained outside the normal
                # German k-medoids mapping, e.g. some foreign/special buses.
                # Their current clustered coordinates are retained.
                n_other_non_medoid_buses += 1

        logger.info(
            "\n"
            "K-MEDOIDS COORDINATE POSTPROCESSING\n"
            "---------------------------------------------\n"
            f"medoid cluster coordinates restored: "
            f"{n_medoid_coordinates_restored}\n"
            f"protected focus buses retained:       "
            f"{n_protected_focus_buses}\n"
            f"protected boundary buses retained:    "
            f"{n_protected_boundary_buses}\n"
            f"other AC buses left unchanged:        "
            f"{n_other_non_medoid_buses}\n"
        )

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

    weight = pd.Series(
        index=network.buses[network.buses.carrier == "AC"].index, data=1.0
    )

    # Add weighting of generators attached to transmission grid bus
    gen = network.generators[
        (network.generators.carrier != "load shedding")
        & (network.generators.bus.isin(weight.index))
    ][["bus", "carrier", "p_nom"]].copy()
    gen["cf"] = gen.apply(calc_availability_factor, axis=1)
    gen["weight"] = gen["p_nom"] * gen["cf"]
    weight.loc[gen.bus.unique()] += gen.groupby("bus").weight.sum()

    # Add weighting of storage units attached to transmission grid bus
    weight.loc[
        network.storage_units.bus[
            network.storage_units.bus.isin(weight.index)
        ].unique()
    ] += (
        network.storage_units[network.storage_units.bus.isin(weight.index)]
        .groupby("bus")
        .p_nom.sum()
    )

    # Add weighting of loads attached to transmission grid
    load = (
        network.loads_t.p_set.mean()
        .groupby(network.loads.bus)
        .sum()
        .reindex(network.buses.index, fill_value=0.0)
    )
    weight.loc[load.index[load.index.isin(weight.index)]] += load[weight.index]

    dg_links = network.links[
        (network.links.carrier == "distribution_grid")
        & (
            network.links.bus0.isin(
                network.buses[network.buses.carrier == "AC"].index.values
            )
        )
    ][["bus0", "bus1"]]

    if not dg_links.empty:
        # Add weighting of generators attached to distribution grid bus
        gen_dg = network.generators[
            (network.generators.carrier != "load shedding")
            & (network.generators.bus.isin(dg_links.bus1))
        ][["bus", "carrier", "p_nom"]].copy()
        gen_dg["cf"] = gen_dg.apply(calc_availability_factor, axis=1)
        gen_dg["weight"] = gen_dg["p_nom"] * gen_dg["cf"]
        gen_dg["bus_tg"] = (
            dg_links.set_index("bus1").loc[gen_dg.bus, "bus0"].values
        )
        weight.loc[gen_dg.bus_tg.unique()] += gen_dg.groupby(
            "bus_tg"
        ).weight.sum()

        # Add weighting of storage units attached to distribution grid bus
        dg_storage = network.storage_units[
            network.storage_units.bus.isin(dg_links.bus1)
        ].copy()
        dg_storage["bus_tg"] = (
            dg_links.set_index("bus1").loc[dg_storage.bus, "bus0"].values
        )
        weight.loc[dg_storage.bus_tg.unique()] += dg_storage.groupby(
            "bus_tg"
        ).p_nom.sum()

        # Add weighting of loads attached to distribution grid
        dg_load = pd.DataFrame(load.loc[dg_links.bus1])
        dg_load["bus_tg"] = (
            dg_links.set_index("bus1").loc[dg_load.index, "bus0"].values
        )
        weight.loc[dg_load["bus_tg"].unique()] += dg_load.groupby("bus_tg")[
            0
        ].sum()

    weight_normed = (
        (weight * (100000.0 / weight.max())).astype(int).clip(lower=1)
    )

    if save:
        weight_normed.to_csv(save)

    return weight_normed


def run_spatial_clustering(self):
    """
    Run spatial clustering of the electrical network.

    This implementation extends the standard eTraGo clustering workflow
    with explicit high-resolution protection of a configured focus region.

    Detailed-focus mode
    -------------------
    If a focus region is configured and
    ``cluster_within_focus == False``:

    * every AC bus inside the focus region is retained as a singleton;
    * the first AC boundary ring outside the focus region is also retained;
    * focus buses receive labels ``focus_<original_bus>``;
    * boundary buses receive labels ``boundary_<original_bus>``;
    * all remaining AC buses are clustered normally;
    * the original eTraGo hard focus weight of 100000 is deliberately
      avoided;
    * after PyPSA clustering, protected-to-protected lines are checked for
      zero/non-finite reactance;
    * damaged protected lines are restored from their actual pre-clustering
      physical branch(es), identified through mapped endpoints rather than
      line IDs.

    No artificial impedance values are introduced. If a valid source branch
    cannot be identified, clustering stops with an explicit error.

    Returns
    -------
    None
    """

    # ==================================================================
    # 0. Read clustering configuration
    # ==================================================================

    clustering_args = self.args["network_clustering"]

    electricity_args = clustering_args["electricity_grid"]

    method_args = clustering_args["method"]

    if not electricity_args.get(
        "active",
        False,
    ):
        logger.info("Electrical spatial clustering is disabled.")
        return

    focus_region = method_args.get("focus_region")

    cluster_within_focus = electricity_args.get("cluster_within_focus")

    algorithm = method_args.get("algorithm")

    k_elec_busmap = electricity_args.get("k_elec_busmap")

    per_country = method_args.get(
        "per_country",
        True,
    )

    cpu_cores = method_args.get(
        "cpu_cores",
        1,
    )

    zero_x_tolerance = 1e-12

    explicit_focus_protection = bool(
        focus_region and cluster_within_focus is False
    )

    # ==================================================================
    # 1. Preserve original network for spatial disaggregation
    # ==================================================================

    if self.args.get("spatial_disaggregation") is not None:

        self.disaggregated_network = self.network.copy()

    else:

        self.disaggregated_network = self.network.copy(with_time=False)

    # ==================================================================
    # 2. Standard eTraGo preprocessing
    # ==================================================================

    (
        elec_network,
        weight,
        n_clusters,
        busmap_foreign,
    ) = preprocessing(self)

    # IMPORTANT:
    #
    # These are the actual electrical branches entering spatial
    # clustering. They are therefore the correct source for later
    # protected-line restoration.
    #
    # Do not use line IDs from self.network after clustering to identify
    # physical branches because PyPSA may rebuild/re-index lines.
    source_lines = elec_network.lines.copy(deep=True)

    source_lines["_source_id"] = source_lines.index.astype(str)

    source_lines["bus0"] = source_lines["bus0"].astype(str)

    source_lines["bus1"] = source_lines["bus1"].astype(str)

    # ==================================================================
    # 3. Initialize focus-region containers
    # ==================================================================

    focus_buses = pd.Index(
        [],
        dtype=str,
    )

    boundary_buses = pd.Index(
        [],
        dtype=str,
    )

    protected_buses = pd.Index(
        [],
        dtype=str,
    )

    # ==================================================================
    # 4. Focus-region treatment
    # ==================================================================

    if focus_region:

        # --------------------------------------------------------------
        # 4A. Detailed focus: explicit singleton protection
        # --------------------------------------------------------------

        if explicit_focus_protection:

            logger.info(
                "Focus region enabled with explicit nodal protection. "
                "AC buses inside the focus region and the first boundary "
                "ring will remain individually represented."
            )

            if k_elec_busmap:

                raise ValueError(
                    "Explicit focus-region protection cannot be combined "
                    "with a precomputed k_elec_busmap. "
                    "Set k_elec_busmap to False."
                )

            # ----------------------------------------------------------
            # Apply the usual distance-dependent focus weighting, but
            # deliberately suppress the upstream 100000 hard focus
            # weight.
            #
            # Actual preservation is performed explicitly through the
            # busmap below.
            # ----------------------------------------------------------

            weight = focus_weighting(
                self,
                elec_network,
                weight,
                focus_region=focus_region,
                cluster_within=True,
                per_country=per_country,
                cpu_cores=cpu_cores,
            )

            (
                _protected,
                focus_buses,
                boundary_buses,
            ) = get_focus_protected_buses(
                self,
                elec_network,
                focus_region=focus_region,
                per_country=per_country,
                include_border=True,
            )

            focus_buses = pd.Index(focus_buses.astype(str))

            boundary_buses = pd.Index(boundary_buses.astype(str))

            protected_buses = focus_buses.union(boundary_buses)

            logger.info(
                "\n"
                "DETAILED FOCUS-REGION CLUSTERING\n"
                "--------------------------------------------------\n"
                f"focus regions:          {focus_region}\n"
                f"focus AC buses:         {len(focus_buses)}\n"
                f"boundary AC buses:      {len(boundary_buses)}\n"
                f"protected AC buses:     {len(protected_buses)}\n"
                f"base external clusters: {n_clusters}\n"
            )

        # --------------------------------------------------------------
        # 4B. Standard eTraGo focus clustering
        # --------------------------------------------------------------

        else:

            weight = focus_weighting(
                self,
                elec_network,
                weight,
                focus_region=focus_region,
                cluster_within=cluster_within_focus,
                per_country=per_country,
                cpu_cores=cpu_cores,
            )

    # ==================================================================
    # 5. Run configured AC clustering algorithm
    # ==================================================================

    busmap = pd.Series(dtype=str)

    medoid_idx = pd.Series(dtype=str)

    if algorithm == "kmeans":

        if not k_elec_busmap:

            logger.info("Start k-means Clustering AC")

            busmap = kmean_clustering(
                self,
                elec_network,
                weight,
                n_clusters,
            )

    elif algorithm == "kmedoids-dijkstra":

        if not k_elec_busmap:

            logger.info("Start k-medoids Dijkstra Clustering AC")

            (
                busmap,
                medoid_idx,
            ) = kmedoids_dijkstra_clustering(
                self,
                elec_network.buses,
                elec_network.lines,
                weight,
                n_clusters,
            )

    else:

        raise ValueError(
            "Unknown electrical clustering algorithm "
            f"{algorithm!r}. Expected 'kmeans' or "
            "'kmedoids-dijkstra'."
        )

    # ==================================================================
    # 6. Explicit singleton protection of focus/boundary buses
    # ==================================================================

    if explicit_focus_protection:

        if not isinstance(
            busmap,
            pd.Series,
        ):

            busmap = pd.Series(busmap)

        busmap = busmap.copy()

        busmap.index = busmap.index.astype(str)

        busmap = busmap.astype(str)

        # --------------------------------------------------------------
        # Verify all protected buses exist in the clustering busmap.
        # --------------------------------------------------------------

        missing_focus = focus_buses.difference(busmap.index)

        missing_boundary = boundary_buses.difference(busmap.index)

        if len(missing_focus):

            raise RuntimeError(
                "The following focus buses are missing from "
                "the clustering busmap:\n"
                f"{missing_focus.tolist()}"
            )

        if len(missing_boundary):

            raise RuntimeError(
                "The following boundary buses are missing from "
                "the clustering busmap:\n"
                f"{missing_boundary.tolist()}"
            )

        # --------------------------------------------------------------
        # Protect each focus bus as one singleton.
        # --------------------------------------------------------------

        for bus in focus_buses:

            busmap.loc[str(bus)] = f"focus_{bus}"

        # --------------------------------------------------------------
        # Protect each boundary bus as one singleton.
        # --------------------------------------------------------------

        for bus in boundary_buses:

            busmap.loc[str(bus)] = f"boundary_{bus}"

        # --------------------------------------------------------------
        # Validate singleton behaviour.
        # --------------------------------------------------------------

        protected_labels = busmap.reindex(protected_buses)

        if protected_labels.isna().any():

            missing = protected_labels[protected_labels.isna()].index.tolist()

            raise RuntimeError(
                "Protected buses lost their cluster mapping: " f"{missing}"
            )

        if protected_labels.nunique() != len(protected_buses):

            duplicated = protected_labels[
                protected_labels.duplicated(keep=False)
            ]

            raise RuntimeError(
                "Focus/boundary buses are not unique singleton "
                "clusters:\n"
                f"{duplicated}"
            )

        logger.info(
            "\n"
            "FOCUS BUSMAP PROTECTION APPLIED\n"
            "--------------------------------------------------\n"
            f"focus buses protected:       "
            f"{len(focus_buses)}\n"
            f"boundary buses protected:    "
            f"{len(boundary_buses)}\n"
            f"base requested clusters:     "
            f"{n_clusters}\n"
            f"final unique busmap groups:  "
            f"{busmap.nunique()}\n"
        )

    # ==================================================================
    # 7. Standard eTraGo / PyPSA postprocessing
    # ==================================================================

    (
        clustering,
        busmap,
    ) = postprocessing(
        self,
        busmap,
        busmap_foreign,
        medoid_idx,
    )

    clustered_network = clustering.network

    # ==================================================================
    # 8. Repair protected AC lines damaged during clustering
    #
    # The critical point:
    #
    # DO NOT match:
    #
    #     clustered line ID -> original line ID
    #
    # PyPSA can rebuild/re-index the line table.
    #
    # Instead identify the original physical branch through:
    #
    #     original endpoints
    #          ↓ busmap
    #     clustered endpoints
    #
    # ==================================================================

    restored_lines = []

    if explicit_focus_protection:

        # --------------------------------------------------------------
        # Normalize final busmap without modifying network indices.
        # --------------------------------------------------------------

        if isinstance(
            busmap,
            pd.Series,
        ):

            final_busmap = busmap.copy()

        else:

            final_busmap = pd.Series(busmap)

        final_busmap.index = final_busmap.index.astype(str)

        final_busmap = final_busmap.astype(str)

        # --------------------------------------------------------------
        # Determine where every source line endpoint ended up.
        # --------------------------------------------------------------

        mapped_source_lines = source_lines.copy()

        mapped_source_lines["_mapped_bus0"] = mapped_source_lines["bus0"].map(
            final_busmap
        )

        mapped_source_lines["_mapped_bus1"] = mapped_source_lines["bus1"].map(
            final_busmap
        )

        # --------------------------------------------------------------
        # Find invalid lines AFTER PyPSA clustering.
        # --------------------------------------------------------------

        clustered_x = pd.to_numeric(
            clustered_network.lines["x"],
            errors="coerce",
        )

        bad_line_mask = ~np.isfinite(clustered_x) | (
            clustered_x.abs() <= zero_x_tolerance
        )

        bad_line_indices = clustered_network.lines.index[bad_line_mask]

        protected_prefixes = (
            "focus_",
            "boundary_",
        )

        unresolved_lines = []

        # --------------------------------------------------------------
        # Helper: equivalent impedance for actual parallel source
        # branches.
        # --------------------------------------------------------------

        def _parallel_equivalent(
            series,
            tolerance=1e-12,
        ):
            """
            Equivalent value for parallel branch impedance.

            Returns NaN if one or more source values are invalid.
            """

            values = pd.to_numeric(
                series,
                errors="coerce",
            )

            if (
                values.isna().any()
                or (~np.isfinite(values)).any()
                or (values.abs() <= tolerance).any()
            ):
                return np.nan

            return 1.0 / (1.0 / values).sum()

        # --------------------------------------------------------------
        # Helper: capacity-weighted representative length
        # --------------------------------------------------------------

        def _representative_length(
            candidates,
        ):

            if "length" not in candidates.columns:
                return np.nan

            lengths = pd.to_numeric(
                candidates["length"],
                errors="coerce",
            )

            if lengths.notna().sum() == 0:
                return np.nan

            if "s_nom" in candidates.columns:

                capacities = pd.to_numeric(
                    candidates["s_nom"],
                    errors="coerce",
                ).fillna(0.0)

                total_capacity = capacities.sum()

                if total_capacity > 0:

                    return float(
                        (lengths.fillna(0.0) * capacities).sum()
                        / total_capacity
                    )

            return float(lengths.mean())

        # --------------------------------------------------------------
        # Repair each invalid clustered line
        # --------------------------------------------------------------

        for clustered_index in bad_line_indices:

            clustered_row = clustered_network.lines.loc[clustered_index]

            clustered_bus0 = str(clustered_row["bus0"])

            clustered_bus1 = str(clustered_row["bus1"])

            # ----------------------------------------------------------
            # Automatically repair ONLY branches fully contained inside
            # our explicitly protected representation.
            #
            # We do not silently change standard external clustering.
            # ----------------------------------------------------------

            protected_to_protected = clustered_bus0.startswith(
                protected_prefixes
            ) and clustered_bus1.startswith(protected_prefixes)

            if not protected_to_protected:

                logger.error(
                    "Invalid AC line %s is outside the fully protected "
                    "focus/boundary network: %s -> %s.",
                    clustered_index,
                    clustered_bus0,
                    clustered_bus1,
                )

                unresolved_lines.append(str(clustered_index))

                continue

            # ----------------------------------------------------------
            # Identify pre-clustering source branch(es) according to
            # their mapped endpoints.
            #
            # Direction can be equal or reversed.
            # ----------------------------------------------------------

            same_direction = (
                mapped_source_lines["_mapped_bus0"] == clustered_bus0
            ) & (mapped_source_lines["_mapped_bus1"] == clustered_bus1)

            reverse_direction = (
                mapped_source_lines["_mapped_bus0"] == clustered_bus1
            ) & (mapped_source_lines["_mapped_bus1"] == clustered_bus0)

            candidates = mapped_source_lines[
                same_direction | reverse_direction
            ].copy()

            # A source branch which maps both ends to the same final bus
            # cannot represent this surviving clustered line.
            candidates = candidates[
                candidates["_mapped_bus0"] != candidates["_mapped_bus1"]
            ]

            if candidates.empty:

                logger.error(
                    "No pre-clustering source line could be found for "
                    "protected clustered line %s: %s -> %s.",
                    clustered_index,
                    clustered_bus0,
                    clustered_bus1,
                )

                unresolved_lines.append(str(clustered_index))

                continue

            # ----------------------------------------------------------
            # Validate source x.
            # ----------------------------------------------------------

            source_x = pd.to_numeric(
                candidates["x"],
                errors="coerce",
            )

            invalid_source_x = (
                source_x.isna()
                | (~np.isfinite(source_x))
                | (source_x.abs() <= zero_x_tolerance)
            )

            if invalid_source_x.any():

                logger.error(
                    "\n"
                    "Cannot safely restore clustered line %s "
                    "(%s -> %s).\n"
                    "One or more matching source branches already have "
                    "invalid reactance:\n%s",
                    clustered_index,
                    clustered_bus0,
                    clustered_bus1,
                    candidates[
                        [
                            column
                            for column in [
                                "_source_id",
                                "bus0",
                                "bus1",
                                "_mapped_bus0",
                                "_mapped_bus1",
                                "x",
                                "r",
                                "s_nom",
                                "length",
                            ]
                            if column in candidates.columns
                        ]
                    ].to_string(index=False),
                )

                unresolved_lines.append(str(clustered_index))

                continue

            # ==========================================================
            # CASE A: exactly one original physical branch
            # ==========================================================

            if len(candidates) == 1:

                source_row = candidates.iloc[0]

                restored_x = float(source_row["x"])

                if "r" in source_row.index:
                    restored_r = source_row["r"]
                else:
                    restored_r = np.nan

                if "g" in source_row.index:
                    restored_g = source_row["g"]
                else:
                    restored_g = np.nan

                if "b" in source_row.index:
                    restored_b = source_row["b"]
                else:
                    restored_b = np.nan

                if "length" in source_row.index:
                    restored_length = source_row["length"]
                else:
                    restored_length = np.nan

                source_description = str(source_row["_source_id"])

            # ==========================================================
            # CASE B: several physical source branches map onto the same
            # protected endpoints.
            #
            # Because protected buses are singletons, these are parallel
            # branches, not series branches.
            # ==========================================================

            else:

                restored_x = _parallel_equivalent(candidates["x"])

                if "r" in candidates.columns:

                    restored_r = _parallel_equivalent(candidates["r"])

                else:

                    restored_r = np.nan

                # Parallel shunt admittances add.
                if "g" in candidates.columns:

                    restored_g = pd.to_numeric(
                        candidates["g"],
                        errors="coerce",
                    ).sum(min_count=1)

                else:

                    restored_g = np.nan

                if "b" in candidates.columns:

                    restored_b = pd.to_numeric(
                        candidates["b"],
                        errors="coerce",
                    ).sum(min_count=1)

                else:

                    restored_b = np.nan

                restored_length = _representative_length(candidates)

                source_description = ", ".join(
                    candidates["_source_id"].astype(str)
                )

                if (
                    not np.isfinite(restored_x)
                    or abs(restored_x) <= zero_x_tolerance
                ):

                    logger.error(
                        "Parallel source branches for clustered line %s "
                        "could not produce a valid equivalent reactance.",
                        clustered_index,
                    )

                    unresolved_lines.append(str(clustered_index))

                    continue

            # ----------------------------------------------------------
            # Save clustered values for diagnostics
            # ----------------------------------------------------------

            x_before = clustered_network.lines.at[
                clustered_index,
                "x",
            ]

            r_before = (
                clustered_network.lines.at[
                    clustered_index,
                    "r",
                ]
                if "r" in clustered_network.lines.columns
                else np.nan
            )

            length_before = (
                clustered_network.lines.at[
                    clustered_index,
                    "length",
                ]
                if "length" in clustered_network.lines.columns
                else np.nan
            )

            # ----------------------------------------------------------
            # Restore x.
            # ----------------------------------------------------------

            clustered_network.lines.at[
                clustered_index,
                "x",
            ] = restored_x

            # ----------------------------------------------------------
            # Restore r when valid.
            # ----------------------------------------------------------

            if (
                "r" in clustered_network.lines.columns
                and pd.notna(restored_r)
                and np.isfinite(restored_r)
            ):

                clustered_network.lines.at[
                    clustered_index,
                    "r",
                ] = restored_r

            # ----------------------------------------------------------
            # Restore g.
            # ----------------------------------------------------------

            if (
                "g" in clustered_network.lines.columns
                and pd.notna(restored_g)
                and np.isfinite(restored_g)
            ):

                clustered_network.lines.at[
                    clustered_index,
                    "g",
                ] = restored_g

            # ----------------------------------------------------------
            # Restore b.
            # ----------------------------------------------------------

            if (
                "b" in clustered_network.lines.columns
                and pd.notna(restored_b)
                and np.isfinite(restored_b)
            ):

                clustered_network.lines.at[
                    clustered_index,
                    "b",
                ] = restored_b

            # ----------------------------------------------------------
            # Restore representative physical length.
            # ----------------------------------------------------------

            if (
                "length" in clustered_network.lines.columns
                and pd.notna(restored_length)
                and np.isfinite(restored_length)
            ):

                clustered_network.lines.at[
                    clustered_index,
                    "length",
                ] = restored_length

            restored_lines.append(str(clustered_index))

            logger.info(
                "\n"
                "RESTORED PROTECTED AC LINE\n"
                "--------------------------------------------------\n"
                f"clustered line:       {clustered_index}\n"
                f"clustered endpoints:  "
                f"{clustered_bus0} -> {clustered_bus1}\n"
                f"source line(s):        "
                f"{source_description}\n"
                f"x before:              "
                f"{x_before}\n"
                f"x restored:            "
                f"{restored_x}\n"
                f"r before:              "
                f"{r_before}\n"
                f"r restored:            "
                f"{restored_r}\n"
                f"length before:         "
                f"{length_before}\n"
                f"length restored:       "
                f"{restored_length}\n"
            )

        # ==============================================================
        # 9. Validate post-repair line reactances
        # ==============================================================

        checked_x = pd.to_numeric(
            clustered_network.lines["x"],
            errors="coerce",
        )

        still_bad_mask = ~np.isfinite(checked_x) | (
            checked_x.abs() <= zero_x_tolerance
        )

        still_bad_lines = clustered_network.lines[still_bad_mask]

        if not still_bad_lines.empty:

            columns = [
                column
                for column in [
                    "bus0",
                    "bus1",
                    "carrier",
                    "x",
                    "r",
                    "s_nom",
                    "v_nom",
                    "length",
                ]
                if column in still_bad_lines.columns
            ]

            raise RuntimeError(
                "\n"
                "Zero or non-finite AC line reactance remains after "
                "endpoint-based protected-line restoration.\n"
                "The following branches are not safe for LOPF:\n\n"
                + still_bad_lines[columns].to_string()
            )

        logger.info(
            "\n"
            "PROTECTED LINE RESTORATION COMPLETE\n"
            "--------------------------------------------------\n"
            f"restored clustered lines: "
            f"{restored_lines}\n"
            f"unresolved lines:         "
            f"{unresolved_lines}\n"
            "PASS: all clustered AC lines have finite "
            "non-zero reactance.\n"
        )

    # ==================================================================
    # 10. Store final clustering busmap
    # ==================================================================

    self.update_busmap(busmap)

    # ==================================================================
    # 11. Replace active network with clustered network
    # ==================================================================

    self.network = clustered_network

    # ==================================================================
    # 12. Restore country/geographical information
    # ==================================================================

    self.buses_by_country()

    self.geolocation_buses()

    # ==================================================================
    # 13. Restore generator/storage control strategies
    #
    # PyPSA topology/clustering calls can overwrite control assignments.
    # ==================================================================

    set_control_strategies(self.network)

    # ==================================================================
    # 14. Final electrical integrity check
    # ==================================================================

    final_x = pd.to_numeric(
        self.network.lines["x"],
        errors="coerce",
    )

    final_bad_mask = ~np.isfinite(final_x) | (
        final_x.abs() <= zero_x_tolerance
    )

    final_bad_lines = self.network.lines[final_bad_mask]

    if not final_bad_lines.empty:

        columns = [
            column
            for column in [
                "bus0",
                "bus1",
                "carrier",
                "x",
                "r",
                "s_nom",
                "v_nom",
                "length",
            ]
            if column in final_bad_lines.columns
        ]

        raise RuntimeError(
            "\n"
            "Spatial clustering produced an electrically invalid "
            "AC network.\n"
            "Zero/non-finite reactance remains:\n\n"
            + final_bad_lines[columns].to_string()
        )

    # ==================================================================
    # 15. Final clustering diagnostics
    # ==================================================================

    final_ac_buses = self.network.buses[
        self.network.buses["carrier"].astype(str) == "AC"
    ]

    logger.info(
        "\n"
        "SPATIAL CLUSTERING FINISHED\n"
        "--------------------------------------------------\n"
        f"algorithm:                    "
        f"{algorithm}\n"
        f"requested base clusters:      "
        f"{n_clusters}\n"
        f"final AC buses:               "
        f"{len(final_ac_buses)}\n"
        f"focus region:                 "
        f"{focus_region}\n"
        f"cluster within focus:         "
        f"{cluster_within_focus}\n"
        f"protected focus buses:        "
        f"{len(focus_buses)}\n"
        f"protected boundary buses:     "
        f"{len(boundary_buses)}\n"
        f"restored protected lines:     "
        f"{len(restored_lines)}\n"
        f"invalid AC reactance lines:   "
        f"{len(final_bad_lines)}\n"
    )
