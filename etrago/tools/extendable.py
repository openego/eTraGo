# -*- coding: utf-8 -*-
# Copyright 2016-2023  Flensburg University of Applied Sciences,
# Europa-Universität Flensburg,
# Centre for Sustainable Energy Systems,
# DLR-Institute for Networked Energy Systems
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation; either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# File description
"""
Extendable.py defines function to set PyPSA components extendable.
"""
from math import sqrt
import time

import numpy as np
import pandas as pd

from etrago.cluster.snapshot import snapshot_clustering
from etrago.tools.utilities import convert_capital_costs, find_snapshots

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = """ulfmueller, s3pp, wolfbunke, mariusves, lukasol, ClaraBuettner,
 KathiEsterl, CarlosEpia"""


def extendable(
    self,
    grid_max_D=None,
    grid_max_abs_D={
        "380": {"i": 1020, "wires": 4, "circuits": 4},
        "220": {"i": 1020, "wires": 4, "circuits": 4},
        "110": {"i": 1020, "wires": 4, "circuits": 2},
        "dc": 0,
    },
    grid_max_foreign=4,
    grid_max_abs_foreign=None,
):
    """
    Function that sets selected components extendable.

    Parameters
    ----------
    grid_max_D : int, optional
        Upper bounds for electrical grid expansion relative to existing
        capacity. The default is None.
    grid_max_abs_D : dict, optional
        Absolute upper bounds for electrical grid expansion in Germany.
    grid_max_foreign : int, optional
        Upper bounds for expansion of electrical foreign lines relative to the
        existing capacity. The default is 4.
    grid_max_abs_foreign : dict, optional
        Absolute upper bounds for expansion of foreign electrical grid.
        The default is None.

    Returns
    -------
    None.

    """

    network = self.network
    extendable_settings = self.args["extendable"]

    if "as_in_db" not in extendable_settings["extendable_components"]:
        network.lines.s_nom_extendable = False
        network.transformers.s_nom_extendable = False
        network.links.p_nom_extendable = False
        network.storage_units.p_nom_extendable = False
        network.stores.e_nom_extendable = False
        network.generators.p_nom_extendable = False

    if "H2_feedin" not in extendable_settings["extendable_components"]:
        network.mremove(
            "Link", network.links[network.links.carrier == "H2_feedin"].index
        )

    if "network" in extendable_settings["extendable_components"]:
        network.lines.s_nom_extendable = True
        network.lines.s_nom_min = network.lines.s_nom

        if not network.transformers.empty:
            network.transformers.s_nom_extendable = True
            network.transformers.s_nom_min = network.transformers.s_nom
            network.transformers.s_nom_max = float("inf")

        if not network.links.empty:
            network.links.loc[
                network.links.carrier == "DC", "p_nom_extendable"
            ] = True
            network.links.loc[network.links.carrier == "DC", "p_nom_min"] = (
                network.links.p_nom
            )
            network.links.loc[network.links.carrier == "DC", "p_nom_max"] = (
                float("inf")
            )

    if "german_network" in extendable_settings["extendable_components"]:
        buses = network.buses[network.buses.country == "DE"]
        network.lines.loc[
            (network.lines.bus0.isin(buses.index))
            & (network.lines.bus1.isin(buses.index)),
            "s_nom_extendable",
        ] = True
        network.lines.loc[
            (network.lines.bus0.isin(buses.index))
            & (network.lines.bus1.isin(buses.index)),
            "s_nom_min",
        ] = network.lines.s_nom
        network.lines.loc[
            (network.lines.bus0.isin(buses.index))
            & (network.lines.bus1.isin(buses.index)),
            "s_nom_max",
        ] = float("inf")

        if not network.transformers.empty:
            network.transformers.loc[
                network.transformers.bus0.isin(buses.index), "s_nom_extendable"
            ] = True
            network.transformers.loc[
                network.transformers.bus0.isin(buses.index), "s_nom_min"
            ] = network.transformers.s_nom
            network.transformers.loc[
                network.transformers.bus0.isin(buses.index), "s_nom_max"
            ] = float("inf")

        if not network.links[network.links.carrier == "DC"].empty:
            network.links.loc[
                (network.links.bus0.isin(buses.index))
                & (network.links.bus1.isin(buses.index))
                & (network.links.carrier == "DC"),
                "p_nom_extendable",
            ] = True
            network.links.loc[
                (network.links.bus0.isin(buses.index))
                & (network.links.bus1.isin(buses.index))
                & (network.links.carrier == "DC"),
                "p_nom_min",
            ] = network.links.p_nom
            network.links.loc[
                (network.links.bus0.isin(buses.index))
                & (network.links.bus1.isin(buses.index))
                & (network.links.carrier == "DC"),
                "p_nom_max",
            ] = float("inf")

    if "foreign_network" in extendable_settings["extendable_components"]:
        buses = network.buses[network.buses.country != "DE"]
        network.lines.loc[
            network.lines.bus0.isin(buses.index)
            | network.lines.bus1.isin(buses.index),
            "s_nom_extendable",
        ] = True
        network.lines.loc[
            network.lines.bus0.isin(buses.index)
            | network.lines.bus1.isin(buses.index),
            "s_nom_min",
        ] = network.lines.s_nom
        network.lines.loc[
            network.lines.bus0.isin(buses.index)
            | network.lines.bus1.isin(buses.index),
            "s_nom_max",
        ] = float("inf")

        if not network.transformers.empty:
            network.transformers.loc[
                network.transformers.bus0.isin(buses.index)
                | network.transformers.bus1.isin(buses.index),
                "s_nom_extendable",
            ] = True
            network.transformers.loc[
                network.transformers.bus0.isin(buses.index)
                | network.transformers.bus1.isin(buses.index),
                "s_nom_min",
            ] = network.transformers.s_nom
            network.transformers.loc[
                network.transformers.bus0.isin(buses.index)
                | network.transformers.bus1.isin(buses.index),
                "s_nom_max",
            ] = float("inf")

        if not network.links[network.links.carrier == "DC"].empty:
            network.links.loc[
                (
                    (network.links.bus0.isin(buses.index))
                    | (network.links.bus1.isin(buses.index))
                )
                & (network.links.carrier == "DC"),
                "p_nom_extendable",
            ] = True
            network.links.loc[
                (
                    (network.links.bus0.isin(buses.index))
                    | (network.links.bus1.isin(buses.index))
                )
                & (network.links.carrier == "DC"),
                "p_nom_min",
            ] = network.links.p_nom
            network.links.loc[
                (
                    (network.links.bus0.isin(buses.index))
                    | (network.links.bus1.isin(buses.index))
                )
                & (network.links.carrier == "DC"),
                "p_nom_max",
            ] = float("inf")

    if "transformers" in extendable_settings["extendable_components"]:
        network.transformers.s_nom_extendable = True
        network.transformers.s_nom_min = network.transformers.s_nom
        network.transformers.s_nom_max = float("inf")

    if (
        "storages" in extendable_settings["extendable_components"]
        or "storage" in extendable_settings["extendable_components"]
        or "store" in extendable_settings["extendable_components"]
        or "stores" in extendable_settings["extendable_components"]
    ):
        network.storage_units.loc[
            network.storage_units.carrier == "battery",
            "p_nom_extendable",
        ] = True
        network.storage_units.loc[
            network.storage_units.carrier == "battery",
            "p_nom_min",
        ] = 0
        network.storage_units.loc[
            network.storage_units.carrier == "battery",
            "p_nom_max",
        ] = float("inf")
        network.storage_units.loc[
            (network.storage_units.carrier == "battery")
            & (network.storage_units.capital_cost == 0),
            "capital_cost",
        ] = network.storage_units.loc[
            network.storage_units.carrier == "battery", "capital_cost"
        ].max()

        ext_stores = [
            "H2_overground",
            "H2_underground",
            "central_heat_store",
            "rural_heat_store",
        ]
        network.stores.loc[
            network.stores.carrier.isin(ext_stores),
            "e_nom_extendable",
        ] = True
        network.stores.loc[
            network.stores.carrier.isin(ext_stores),
            "e_nom_min",
        ] = 0
        network.stores.loc[
            network.stores.carrier.isin(ext_stores),
            "e_nom_max",
        ] = float("inf")
        if (
            len(
                network.stores.loc[
                    (network.stores.carrier.isin(ext_stores))
                    & (network.stores.capital_cost == 0)
                ]
            )
            > 0
        ):
            for c in ext_stores:
                network.stores.loc[
                    (network.stores.carrier == c)
                    & (network.stores.capital_cost == 0),
                    "capital_cost",
                ] = network.stores.loc[
                    (network.stores.carrier == c), "capital_cost"
                ].max()

    if "foreign_storage" in extendable_settings["extendable_components"]:
        foreign_battery = network.storage_units[
            (
                network.storage_units.bus.isin(
                    network.buses.index[network.buses.country != "DE"]
                )
            )
            & (network.storage_units.carrier == "battery")
        ].index

        de_battery = network.storage_units[
            (
                network.storage_units.bus.isin(
                    network.buses.index[network.buses.country == "DE"]
                )
            )
            & (network.storage_units.carrier == "battery")
        ].index

        network.storage_units.loc[foreign_battery, "p_nom_extendable"] = True

        network.storage_units.loc[foreign_battery, "p_nom_max"] = (
            network.storage_units.loc[foreign_battery, "p_nom"]
        )

        network.storage_units.loc[foreign_battery, "p_nom"] = (
            network.storage_units.loc[foreign_battery, "p_nom_min"]
        )

        network.storage_units.loc[foreign_battery, "capital_cost"] = (
            network.storage_units.loc[de_battery, "capital_cost"].max()
        )

        network.storage_units.loc[foreign_battery, "marginal_cost"] = (
            network.storage_units.loc[de_battery, "marginal_cost"].max()
        )

    if (
        "foreign_storage_unlimited"
        in extendable_settings["extendable_components"]
    ):
        foreign_battery = network.storage_units[
            (
                network.storage_units.bus.isin(
                    network.buses.index[network.buses.country != "DE"]
                )
            )
            & (network.storage_units.carrier == "battery")
        ].index

        de_battery = network.storage_units[
            (
                network.storage_units.bus.isin(
                    network.buses.index[network.buses.country == "DE"]
                )
            )
            & (network.storage_units.carrier == "battery")
        ].index

        network.storage_units.loc[foreign_battery, "p_nom_extendable"] = True

        network.storage_units.loc[foreign_battery, "p_nom_max"] = np.inf

        network.storage_units.loc[foreign_battery, "p_nom"] = (
            network.storage_units.loc[foreign_battery, "p_nom_min"]
        )

        network.storage_units.loc[foreign_battery, "capital_cost"] = (
            network.storage_units.loc[de_battery, "capital_cost"].max()
        )

        network.storage_units.loc[foreign_battery, "marginal_cost"] = (
            network.storage_units.loc[de_battery, "marginal_cost"].max()
        )

    # Extension settings for extension-NEP 2035 scenarios
    if "overlay_network" in extendable_settings["extendable_components"]:
        for i in range(len(self.args["scn_extension"])):
            network.lines.loc[
                network.lines.scn_name
                == ("extension_" + self.args["scn_extension"][i]),
                "s_nom_extendable",
            ] = True

            network.lines.loc[
                network.lines.scn_name
                == ("extension_" + self.args["scn_extension"][i]),
                "s_nom_max",
            ] = network.lines.s_nom[
                network.lines.scn_name
                == ("extension_" + self.args["scn_extension"][i])
            ]

            network.links.loc[
                network.links.scn_name
                == ("extension_" + self.args["scn_extension"][i]),
                "p_nom_extendable",
            ] = True

            network.transformers.loc[
                network.transformers.scn_name
                == ("extension_" + self.args["scn_extension"][i]),
                "s_nom_extendable",
            ] = True

            network.lines.loc[
                network.lines.scn_name
                == ("extension_" + self.args["scn_extension"][i]),
                "capital_cost",
            ] = network.lines.capital_cost

    # constrain network expansion to maximum

    if grid_max_abs_D is not None:
        buses = network.buses[
            (network.buses.country == "DE") & (network.buses.carrier == "AC")
        ]

        line_max_abs(network=network, buses=buses, line_max_abs=grid_max_abs_D)

        transformer_max_abs(network=network, buses=buses)

        network.links.loc[
            (network.links.bus0.isin(buses.index))
            & (network.links.bus1.isin(buses.index))
            & (network.links.carrier == "DC"),
            "p_nom_max",
        ] = grid_max_abs_D["dc"]

    if grid_max_abs_foreign is not None:
        foreign_buses = network.buses[
            (network.buses.country != "DE") & (network.buses.carrier == "AC")
        ]

        line_max_abs(
            network=network,
            buses=foreign_buses,
            line_max_abs=grid_max_abs_foreign,
        )

        transformer_max_abs(network=network, buses=foreign_buses)

        network.links.loc[
            (
                (network.links.bus0.isin(foreign_buses.index))
                | (network.links.bus1.isin(foreign_buses.index))
            )
            & (network.links.carrier == "DC"),
            "p_nom_max",
        ] = grid_max_abs_foreign["dc"]

    if grid_max_D is not None:
        buses = network.buses[
            (network.buses.country == "DE") & (network.buses.carrier == "AC")
        ]

        network.lines.loc[
            (network.lines.bus0.isin(buses.index))
            & (network.lines.bus1.isin(buses.index)),
            "s_nom_max",
        ] = (
            grid_max_D * network.lines.s_nom
        )

        network.transformers.loc[
            network.transformers.bus0.isin(buses.index), "s_nom_max"
        ] = (grid_max_D * network.transformers.s_nom)

        network.links.loc[
            (network.links.bus0.isin(buses.index))
            & (network.links.bus1.isin(buses.index))
            & (network.links.carrier == "DC"),
            "p_nom_max",
        ] = (
            grid_max_D * network.links.p_nom
        )

    if grid_max_foreign is not None:
        foreign_buses = network.buses[
            (network.buses.country != "DE") & (network.buses.carrier == "AC")
        ]

        network.lines.loc[
            network.lines.bus0.isin(foreign_buses.index)
            | network.lines.bus1.isin(foreign_buses.index),
            "s_nom_max",
        ] = (
            grid_max_foreign * network.lines.s_nom
        )

        network.links.loc[
            (
                (network.links.bus0.isin(foreign_buses.index))
                | (network.links.bus1.isin(foreign_buses.index))
            )
            & (network.links.carrier == "DC"),
            "p_nom_max",
        ] = (
            grid_max_foreign * network.links.p_nom
        )

        network.transformers.loc[
            network.transformers.bus0.isin(foreign_buses.index)
            | network.transformers.bus1.isin(foreign_buses.index),
            "s_nom_max",
        ] = (
            grid_max_foreign * network.transformers.s_nom
        )


def snommax(i=1020, u=380, wires=4, circuits=4):
    """
    Function to calculate limitation for capacity expansion.

    Parameters
    ----------
    i : int, optional
        Current. The default is 1020.
    u : int, optional
        Voltage level. The default is 380.
    wires : int, optional
        Number of wires per line. The default is 4.
    circuits : int, optional
        Number of circuits. The default is 4.

    Returns
    -------
    s_nom_max : float
        Limitation for capacity expansion.

    """
    s_nom_max = (i * u * sqrt(3) * wires * circuits) / 1000
    return s_nom_max


def line_max_abs(
    network,
    buses,
    line_max_abs={
        "380": {"i": 1020, "wires": 4, "circuits": 4},
        "220": {"i": 1020, "wires": 4, "circuits": 4},
        "110": {"i": 1020, "wires": 4, "circuits": 2},
        "dc": 0,
    },
):
    """
    Function to calculate limitation for capacity expansion of lines in
    network.

    Parameters
    ----------
    network : pypsa.Network object
        Container for all network components.
    buses : pypsa.Network buses
        Considered buses in network.
    line_max_abs : dict, optional
        Line parameters considered to calculate maximum capacity.

    Returns
    -------
    None.

    """
    # calculate the cables of the route between two buses
    cables = network.lines.groupby(["bus0", "bus1"]).cables.sum()
    cables2 = network.lines.groupby(["bus1", "bus0"]).cables.sum()
    doubles_idx = cables.index == cables2.index
    cables3 = cables[doubles_idx] + cables2[doubles_idx]
    cables4 = cables3.swaplevel()
    cables[cables3.index] = cables3
    cables[cables4.index] = cables4
    network.lines["total_cables"] = network.lines.apply(
        lambda x: cables[(x.bus0, x.bus1)], axis=1
    )
    s_nom_max_110 = snommax(
        u=110,
        i=line_max_abs["110"]["i"],
        wires=line_max_abs["110"]["wires"],
        circuits=line_max_abs["110"]["circuits"],
    ) * (network.lines["cables"] / network.lines["total_cables"])
    s_nom_max_220 = snommax(
        u=220,
        i=line_max_abs["220"]["i"],
        wires=line_max_abs["220"]["wires"],
        circuits=line_max_abs["220"]["circuits"],
    ) * (network.lines["cables"] / network.lines["total_cables"])
    s_nom_max_380 = snommax(
        u=380,
        i=line_max_abs["380"]["i"],
        wires=line_max_abs["380"]["wires"],
        circuits=line_max_abs["380"]["circuits"],
    ) * (network.lines["cables"] / network.lines["total_cables"])
    # set the s_nom_max depending on the voltage level
    # and the share of the route
    network.lines.loc[
        (network.lines.bus0.isin(buses.index))
        & (network.lines.bus1.isin(buses.index))
        & (network.lines.v_nom == 110.0)
        & (network.lines.s_nom < s_nom_max_110),
        "s_nom_max",
    ] = s_nom_max_110

    network.lines.loc[
        (network.lines.bus0.isin(buses.index))
        & (network.lines.bus1.isin(buses.index))
        & (network.lines.v_nom == 110.0)
        & (network.lines.s_nom >= s_nom_max_110),
        "s_nom_max",
    ] = network.lines.s_nom

    network.lines.loc[
        (network.lines.bus0.isin(buses.index))
        & (network.lines.bus1.isin(buses.index))
        & (network.lines.v_nom == 220.0)
        & (network.lines.s_nom < s_nom_max_220),
        "s_nom_max",
    ] = s_nom_max_220

    network.lines.loc[
        (network.lines.bus0.isin(buses.index))
        & (network.lines.bus1.isin(buses.index))
        & (network.lines.v_nom == 220.0)
        & (network.lines.s_nom >= s_nom_max_220),
        "s_nom_max",
    ] = network.lines.s_nom

    network.lines.loc[
        (network.lines.bus0.isin(buses.index))
        & (network.lines.bus1.isin(buses.index))
        & (network.lines.v_nom == 380.0)
        & (network.lines.s_nom < s_nom_max_380),
        "s_nom_max",
    ] = s_nom_max_380

    network.lines.loc[
        (network.lines.bus0.isin(buses.index))
        & (network.lines.bus1.isin(buses.index))
        & (network.lines.v_nom == 380.0)
        & (network.lines.s_nom >= s_nom_max_380),
        "s_nom_max",
    ] = network.lines.s_nom


def transformer_max_abs(network, buses):
    """
    Function to calculate limitation for capacity expansion of transformers in
    network.

    Parameters
    ----------
    network : pypsa.Network object
        Container for all network components.
    buses : pypsa.Network buses
        Considered buses in network.

    Returns
    -------
    None.

    """

    # To determine the maximum extendable capacity of a transformer, the sum of
    # the maximum capacities of the lines connected to it is calculated for
    # each of its 2 sides. The smallest one is selected.
    smax_bus0 = network.lines.s_nom_max.groupby(network.lines.bus0).sum()
    smax_bus1 = network.lines.s_nom_max.groupby(network.lines.bus1).sum()
    smax_bus = pd.concat([smax_bus0, smax_bus1], axis=1)
    smax_bus.columns = ["s_nom_max_0", "s_nom_max_1"]
    smax_bus = smax_bus.fillna(0)
    smax_bus["s_nom_max_bus"] = smax_bus.apply(
        lambda x: x["s_nom_max_0"] + x["s_nom_max_1"], axis=1
    )

    pmax_links_bus0 = network.links.p_nom_max.groupby(network.links.bus0).sum()
    pmax_links_bus1 = network.links.p_nom_max.groupby(network.links.bus1).sum()
    pmax_links_bus = pd.concat([pmax_links_bus0, pmax_links_bus1], axis=1)
    pmax_links_bus.columns = ["p_nom_max_0", "p_nom_max_1"]
    pmax_links_bus = pmax_links_bus.fillna(0)
    pmax_links_bus["p_nom_max_bus"] = pmax_links_bus.apply(
        lambda x: x["p_nom_max_0"] + x["p_nom_max_1"], axis=1
    )

    trafo_smax_0 = network.transformers.bus0.map(smax_bus["s_nom_max_bus"])
    trafo_smax_1 = network.transformers.bus1.map(smax_bus["s_nom_max_bus"])
    trafo_pmax_0 = (
        network.transformers.bus0.map(pmax_links_bus["p_nom_max_bus"]) / 2
    )
    trafo_pmax_1 = (
        network.transformers.bus1.map(pmax_links_bus["p_nom_max_bus"]) / 2
    )
    trafo_smax = pd.concat(
        [trafo_smax_0, trafo_smax_1, trafo_pmax_0, trafo_pmax_1], axis=1
    )
    trafo_smax = trafo_smax.fillna(0)
    trafo_smax.columns = ["bus0", "bus1", "dcbus0", "dcbus1"]
    trafo_smax["s_nom_max"] = trafo_smax[trafo_smax.gt(0)].min(axis=1)
    network.transformers.loc[
        network.transformers.bus0.isin(buses.index), "s_nom_max"
    ] = trafo_smax["s_nom_max"]

    # Since the previous calculation does not depent on the min_capacity of the
    # transformer, there are few cases where the min capacity is greater than
    # the calculated maximum. For these cases, max capacity is set to be the
    # equal to the min capacity.
    network.transformers["s_nom_max"] = network.transformers.apply(
        lambda x: (
            x["s_nom_max"]
            if float(x["s_nom_max"]) > float(x["s_nom_min"])
            else x["s_nom_min"]
        ),
        axis=1,
    )


def extension_preselection(etrago, method, days=3):
    """
    Function that preselects lines which are extendend in snapshots leading to
    overloading to reduce nubmer of extension variables.

    Parameters
    ----------
    network : pypsa.Network object
        Container for all network components.
    args  : dict
        Arguments set in appl.py.
    method: str
        Choose method of selection:
        'extreme_situations' for remarkable timsteps
        (e.g. minimal resiudual load)
        'snapshot_clustering' for snapshot clustering with number of days
    days: int
        Number of clustered days, only used when method = 'snapshot_clustering'

    Returns
    -------
    network : pypsa.Network object
        Container for all network components.
    """
    network = etrago.network
    args = etrago.args
    weighting = network.snapshot_weightings

    if method == "extreme_situations":
        snapshots = find_snapshots(network, "residual load")
        snapshots = snapshots.append(find_snapshots(network, "wind_onshore"))
        snapshots = snapshots.append(find_snapshots(network, "solar"))
        snapshots = snapshots.drop_duplicates()
        snapshots = snapshots.sort_values()

    if method == "snapshot_clustering":
        network_cluster = snapshot_clustering(etrago, how="daily")
        snapshots = network_cluster.snapshots
        network.snapshot_weightings = network_cluster.snapshot_weightings

    # Set all lines and trafos extendable in network
    network.lines.loc[:, "s_nom_extendable"] = True
    network.lines.loc[:, "s_nom_min"] = network.lines.s_nom
    network.lines.loc[:, "s_nom_max"] = np.inf

    network.links.loc[:, "p_nom_extendable"] = True
    network.links.loc[:, "p_nom_min"] = network.links.p_nom
    network.links.loc[:, "p_nom_max"] = np.inf

    network.transformers.loc[:, "s_nom_extendable"] = True
    network.transformers.loc[:, "s_nom_min"] = network.transformers.s_nom
    network.transformers.loc[:, "s_nom_max"] = np.inf

    network = convert_capital_costs(network, 1, 1)
    extended_lines = network.lines.index[
        network.lines.s_nom_opt > network.lines.s_nom
    ]
    extended_links = network.links.index[
        network.links.p_nom_opt > network.links.p_nom
    ]

    x = time.time()
    for i in range(int(snapshots.value_counts().sum())):
        if i > 0:
            network.lopf(snapshots[i], solver_name=args["solver"])
            extended_lines = extended_lines.append(
                network.lines.index[
                    network.lines.s_nom_opt > network.lines.s_nom
                ]
            )
            extended_lines = extended_lines.drop_duplicates()
            extended_links = extended_links.append(
                network.links.index[
                    network.links.p_nom_opt > network.links.p_nom
                ]
            )
            extended_links = extended_links.drop_duplicates()

    print("Number of preselected lines: ", len(extended_lines))

    network.lines.loc[
        ~network.lines.index.isin(extended_lines), "s_nom_extendable"
    ] = False
    network.lines.loc[network.lines.s_nom_extendable, "s_nom_min"] = (
        network.lines.s_nom
    )
    network.lines.loc[network.lines.s_nom_extendable, "s_nom_max"] = np.inf

    network.links.loc[
        ~network.links.index.isin(extended_links), "p_nom_extendable"
    ] = False
    network.links.loc[network.links.p_nom_extendable, "p_nom_min"] = (
        network.links.p_nom
    )
    network.links.loc[network.links.p_nom_extendable, "p_nom_max"] = np.inf

    network.snapshot_weightings = weighting
    network = convert_capital_costs(
        network, args["start_snapshot"], args["end_snapshot"]
    )

    y = time.time()
    z1st = (y - x) / 60

    print("Time for first LOPF [min]:", round(z1st, 2))

    return network


def print_expansion_costs(network):
    """Function that prints network and storage investment costs.

    Parameters
    ----------
    network : pypsa.Network object
        Container for all network components.

    Returns
    -------
    None.

    """

    ext_storage = network.storage_units[network.storage_units.p_nom_extendable]
    ext_lines = network.lines[network.lines.s_nom_extendable]
    ext_links = network.links[network.links.p_nom_extendable]
    ext_trafos = network.transformers[network.transformers.s_nom_extendable]

    if not ext_storage.empty:
        storage_costs = (
            ext_storage.p_nom_opt * ext_storage.capital_cost
        ).sum()

    if not ext_lines.empty:
        network_costs = (
            (
                (ext_lines.s_nom_opt - ext_lines.s_nom)
                * ext_lines.capital_cost
            ).sum()
            + (ext_links.p_nom_opt - ext_links.p_nom) * ext_links.capital_cost
        ).sum()

    if not ext_trafos.empty:
        network_costs = (
            network_costs
            + (
                (ext_trafos.s_nom_opt - ext_trafos.s_nom)
                * ext_trafos.capital_cost
            ).sum()
        )

    if not ext_storage.empty:
        print(
            """Investment costs for all storage units in selected snapshots
            [EUR]:""",
            round(storage_costs, 2),
        )

    if not ext_lines.empty:
        print(
            """Investment costs for all lines and transformers in selected
             snapshots [EUR]:""",
            round(network_costs, 2),
        )
