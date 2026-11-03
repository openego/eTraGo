# -*- coding: utf-8 -*-
# Copyright 2016-2018  Flensburg University of Applied Sciences,
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
calc_results.py defines methods to calculate results of eTraGo
"""

import os

if "READTHEDOCS" not in os.environ:
    import logging

    from matplotlib import pyplot as plt
    import pandas as pd
    import pypsa

    from etrago.execute.grid_optimization import get_redispatch_timeseries

    logger = logging.getLogger(__name__)

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, ClaraBuettner, KathiEsterl"


def _redispatch_timeseries(self):
    """Return the time series of the redispatch components

    If the redispatch components were merged into their original components
    for the power flow, the time series stored before are used. Otherwise
    they are taken from self.network.

    Returns
    -------
    dict or None
        Time series of the redispatch components, see
        get_redispatch_timeseries. None if redispatch was not considered.

    """
    if self.redispatch_timeseries is not None:
        return self.redispatch_timeseries
    return get_redispatch_timeseries(self.network)


def calc_redispatch_cost(self):
    """Function that calculates the costs of the redispatch components

    The costs are calculated from the time series of the 'ramp_up' and
    'ramp_down' generators and links, see _redispatch_timeseries.

    Returns
    -------
    float or None
        Redispatch costs in EUR, None if redispatch was not considered

    """
    timeseries = _redispatch_timeseries(self)
    if timeseries is None:
        return None

    weightings = self.network.snapshot_weightings.objective
    costs = 0.0
    for component, p in [("Generator", "p"), ("Link", "p0")]:
        if component in timeseries:
            costs += (
                timeseries[component][p]
                .mul(weightings, axis=0)
                .mul(timeseries[component]["marginal_cost"])
                .sum()
                .sum()
            )
    return costs


def _calc_storage_expansion(self):
    """Function that calulates storage expansion in MW


    Returns
    -------
    float
        storage expansion in MW

    """
    return (
        (
            self.network.storage_units.p_nom_opt
            - self.network.storage_units.p_nom_min
        )[self.network.storage_units.p_nom_extendable]
        .groupby(self.network.storage_units.carrier)
        .sum()
    )


def _calc_store_expansion(self):
    """Function that calulates store expansion in MW

    Returns
    -------
    float
        store expansion in MW

    """
    return (self.network.stores.e_nom_opt - self.network.stores.e_nom_min)[
        self.network.stores.e_nom_extendable
    ]


def _calc_sectorcoupling_link_expansion(self):
    """Function that calulates expansion of sectorcoupling links in MW

    Returns
    -------
    float
        link expansion in MW (differentiating between technologies)

    """
    ext_links = self.network.links[self.network.links.p_nom_extendable]

    links = [0, 0, 0, 0]

    l1 = ext_links[ext_links.carrier == "H2_to_power"]
    l2 = ext_links[ext_links.carrier == "power_to_H2"]
    l3 = ext_links[ext_links.carrier == "H2_to_CH4"]
    l4 = ext_links[ext_links.carrier == "CH4_to_H2"]

    links[0] = (l1.p_nom_opt - l1.p_nom_min).sum()
    links[1] = (l2.p_nom_opt - l2.p_nom_min).sum()
    links[2] = (l3.p_nom_opt - l3.p_nom_min).sum()
    links[3] = (l4.p_nom_opt - l4.p_nom_min).sum()

    return links


def _calc_network_expansion(self):
    """Function that calulates electrical network expansion in MW

    Returns
    -------
    float
        network expansion (AC lines and DC links) in MW

    """

    network = self.network

    lines = (network.lines.s_nom_opt - network.lines.s_nom_min)[
        network.lines.s_nom_extendable
    ]

    ext_links = network.links[network.links.p_nom_extendable]
    ext_dc_lines = ext_links[ext_links.carrier == "DC"]

    dc_links = ext_dc_lines.p_nom_opt - ext_dc_lines.p_nom_min

    return lines, dc_links


def _calc_network_expansion_length(self):
    """Function that calulates electrical network expansion in MW

    Returns
    -------
    float
        network expansion (AC lines and DC links) in MW

    """

    network = self.network

    lines = (network.lines.s_nom_opt - network.lines.s_nom_min).mul(
        network.lines.length
    )[network.lines.s_nom_extendable]

    ext_links = network.links[network.links.p_nom_extendable]
    ext_dc_lines = ext_links[ext_links.carrier == "DC"]

    dc_links = (ext_dc_lines.p_nom_opt - ext_dc_lines.p_nom_min).mul(
        ext_dc_lines.length
    )

    return lines.sum(), dc_links.sum()


def annualize_capital_costs(overnight_costs, lifetime, p):
    """

    Parameters
    ----------
    overnight_costs : float
        Overnight investment costs in EUR/MW or EUR/MW/km
    lifetime : int
        Number of years in which payments will be made
    p : float
        Interest rate in p.u.

    Returns
    -------
    float
        Annualized capital costs in EUR/MW/a or EUR/MW/km/a

    """

    # Calculate present value of an annuity (PVA)
    PVA = (1 / p) - (1 / (p * (1 + p) ** lifetime))

    return overnight_costs / PVA


def calc_investment_cost(self):
    """Function that calulates overall annualized investment costs.

    Returns
    -------
    network_costs : float
        Investments in line expansion (AC+DC)
    link_costs : float
        Investments in sectorcoupling link expansion
    stor_costs : float
        Investments in storage and store expansion

    """
    network = self.network

    # electrical grid: AC lines, DC lines

    network_costs = [0, 0]

    ext_lines = network.lines[network.lines.s_nom_extendable]
    ext_trafos = network.transformers[network.transformers.s_nom_extendable]
    ext_links = network.links[network.links.p_nom_extendable]
    ext_dc_lines = ext_links[ext_links.carrier == "DC"]

    if not ext_lines.empty:
        network_costs[0] = (
            (ext_lines.s_nom_opt - ext_lines.s_nom_min)
            * ext_lines.capital_cost
        ).sum()

    if not ext_trafos.empty:
        network_costs[0] = (
            network_costs[0]
            + (
                (ext_trafos.s_nom_opt - ext_trafos.s_nom)
                * ext_trafos.capital_cost
            ).sum()
        )

    if not ext_dc_lines.empty:
        network_costs[1] = (
            (ext_dc_lines.p_nom_opt - ext_dc_lines.p_nom_min)
            * ext_dc_lines.capital_cost
        ).sum()

    # links in other sectors / coupling different sectors

    link_costs = 0

    ext_links = ext_links[ext_links.carrier != "DC"]

    if not ext_links.empty:
        link_costs = (
            (ext_links.p_nom_opt - ext_links.p_nom_min)
            * ext_links.capital_cost
        ).sum()

    # storage and store costs

    sto_costs = [0, 0]

    ext_storage = network.storage_units[network.storage_units.p_nom_extendable]
    ext_store = network.stores[network.stores.e_nom_extendable]

    if not ext_storage.empty:
        sto_costs[0] = (
            (ext_storage.p_nom_opt - ext_storage.p_nom_min)
            * ext_storage.capital_cost
        ).sum()

    if not ext_store.empty:
        sto_costs[1] = (ext_store.e_nom_opt * ext_store.capital_cost).sum()

    return network_costs, link_costs, sto_costs


def calc_marginal_cost(self):
    """
    Function that caluclates and returns marginal costs, considering
    generation and link and storage dispatch costs

    Returns
    -------
    marginal_cost : float
        Annual marginal cost in EUR

    """
    network = self.network
    gen = (
        network.generators_t.p.mul(
            network.snapshot_weightings.objective, axis=0
        )
        .mul(
            pypsa.descriptors.get_switchable_as_dense(
                network, "Generator", "marginal_cost"
            )
        )
        .sum()
        .sum()
    )
    link = (
        abs(network.links_t.p0)
        .mul(network.snapshot_weightings.objective, axis=0)
        .mul(
            pypsa.descriptors.get_switchable_as_dense(
                network, "Link", "marginal_cost"
            )
        )
        .sum()
        .sum()
    )
    stor = (
        network.storage_units_t.p.mul(
            network.snapshot_weightings.objective, axis=0
        )
        .sum(axis=0)
        .mul(network.storage_units.marginal_cost)
        .sum()
    )
    marginal_cost = gen + link + stor
    return marginal_cost


def german_network(self):
    """Cut out all network components in Germany

    Returns
    -------
    new_network : pypsa.Network
        Network with all components in Germany

    """
    keep_cntr = ["DE"]
    new_idx = self.network.buses[
        self.network.buses.country.isin(keep_cntr)
    ].index

    new_network = self.network.copy()

    # drop components of other countries
    new_network.mremove(
        "Bus", new_network.buses[~new_network.buses.index.isin(new_idx)].index
    )

    new_network.mremove(
        "Line",
        new_network.lines[
            ~new_network.lines.index.isin(
                new_network.lines[
                    (
                        new_network.lines.bus0.isin(new_idx)
                        & new_network.lines.bus1.isin(new_idx)
                    )
                ].index
            )
        ].index,
    )
    new_network.mremove(
        "Link",
        new_network.links[
            ~new_network.links.index.isin(
                new_network.links[
                    (
                        new_network.links.bus0.isin(new_idx)
                        & new_network.links.bus1.isin(new_idx)
                    )
                ].index
            )
        ].index,
    )

    new_network.mremove(
        "Transformer",
        new_network.transformers[
            ~new_network.transformers.index.isin(
                new_network.transformers[
                    (
                        new_network.transformers.bus0.isin(new_idx)
                        & new_network.transformers.bus1.isin(new_idx)
                    )
                ].index
            )
        ].index,
    )

    new_network.mremove(
        "Generator",
        new_network.generators[
            ~new_network.generators.index.isin(
                new_network.generators[
                    new_network.generators.bus.isin(new_idx)
                ].index
            )
        ].index,
    )

    new_network.mremove(
        "Load",
        new_network.loads[
            ~new_network.loads.index.isin(
                new_network.loads[new_network.loads.bus.isin(new_idx)].index
            )
        ].index,
    )

    new_network.mremove(
        "Store",
        new_network.stores[
            ~new_network.stores.index.isin(
                new_network.stores[new_network.stores.bus.isin(new_idx)].index
            )
        ].index,
    )

    new_network.mremove(
        "StorageUnit",
        new_network.storage_units[
            ~new_network.storage_units.index.isin(
                new_network.storage_units[
                    new_network.storage_units.bus.isin(new_idx)
                ].index
            )
        ].index,
    )

    return new_network


def system_costs_germany(self, electricity_only=False):
    """Calculte system costs for Germany

    Returns
    -------
    marginal_cost : float
        Marginal costs for dispatch in Germany
    invest_cost : float
        Annualized investment costs for components in Germany
    import_costs : float
        Costs for energy imported to Germany minus costs for exports

    """

    network_de = self.german_network()

    marginal_cost = 0
    invest_cost = 0

    for c in network_de.iterate_components():
        if c.name in ["Store"]:
            value = "e"
        elif c.name in ["Line", "Transformer"]:
            value = "s"
        else:
            value = "p"
        if c.name in network_de.one_port_components:
            if "marginal_cost" in c.df.columns:
                marginal_cost += (
                    c.pnl.p.mul(c.df.marginal_cost)
                    .mul(network_de.snapshot_weightings.generators, axis=0)
                    .sum()
                    .sum()
                )

        else:
            if "marginal_cost" in c.df.columns:
                marginal_cost += (
                    c.pnl.p0.mul(c.df.marginal_cost)
                    .mul(network_de.snapshot_weightings.generators, axis=0)
                    .sum()
                    .sum()
                )
        if c.name not in [
            "Bus",
            "Load",
            "LineType",
            "TransformerType",
            "Carrier",
        ]:
            invest_cost += (
                (
                    c.df[c.df[f"{value}_nom_extendable"]][f"{value}_nom_opt"]
                    - c.df[c.df[f"{value}_nom_extendable"]][f"{value}_nom_min"]
                )
                * c.df[c.df[f"{value}_nom_extendable"]]["capital_cost"]
            ).sum()

    # import and its costs
    links_export = self.network.links[
        (
            self.network.links.bus0.isin(network_de.buses.index.values)
            & ~(self.network.links.bus1.isin(network_de.buses.index.values))
        )
    ]

    export_positive = (
        self.network.links_t.p0[links_export.index]
        .clip(lower=0)
        .mul(self.network.snapshot_weightings.generators, axis=0)
        .mul(
            self.network.buses_t.marginal_price[links_export.bus1].values,
        )
        .sum()
        .sum()
    )

    export_negative = (
        self.network.links_t.p0[links_export.index]
        .clip(upper=0)
        .mul(self.network.snapshot_weightings.generators, axis=0)
        .mul(
            self.network.buses_t.marginal_price[links_export.bus1].values,
        )
        .mul(-1)
        .sum()
        .sum()
    )

    links_import = self.network.links[
        (
            self.network.links.bus1.isin(network_de.buses.index.values)
            & ~(self.network.links.bus0.isin(network_de.buses.index.values))
        )
    ]

    import_positive = (
        self.network.links_t.p0[links_import.index]
        .clip(lower=0)
        .mul(self.network.snapshot_weightings.generators, axis=0)
        .mul(
            self.network.buses_t.marginal_price[links_import.bus1].values,
        )
        .sum()
        .sum()
    )

    import_negative = (
        self.network.links_t.p0[links_import.index]
        .clip(upper=0)
        .mul(self.network.snapshot_weightings.generators, axis=0)
        .mul(
            self.network.buses_t.marginal_price[links_import.bus1].values,
        )
        .mul(-1)
        .sum()
        .sum()
    )

    import_costs = (
        export_negative + import_positive - export_positive - import_negative
    )

    return marginal_cost, invest_cost, import_costs


def electricity_system_costs_germany(self):
    """Calculte system costs for Germany

    Returns
    -------
    marginal_cost : float
        Marginal costs for dispatch in Germany
    invest_cost : float
        Annualized investment costs for components in Germany
    import_costs : float
        Costs for energy imported to Germany minus costs for exports

    """

    network_de = self.german_network()

    marginal_cost = 0
    invest_cost = 0

    for c in network_de.iterate_components():
        if c.name in ["Store"]:
            value = "e"
        elif c.name in ["Line", "Transformer"]:
            value = "s"
        else:
            value = "p"
        if c.name in network_de.one_port_components:
            if "marginal_cost" in c.df.columns:
                df = c.df[
                    c.df.bus.isin(
                        network_de.buses[
                            network_de.buses.carrier == "AC"
                        ].index
                    )
                ]
                marginal_cost += (
                    c.pnl.p[df.index]
                    .mul(df.marginal_cost)
                    .mul(network_de.snapshot_weightings.generators, axis=0)
                    .sum()
                    .sum()
                )

        else:
            if "marginal_cost" in c.df.columns:
                df = c.df[
                    c.df.bus0.isin(
                        network_de.buses[
                            network_de.buses.carrier == "AC"
                        ].index
                    )
                ]
                marginal_cost += (
                    c.pnl.p0[df.index]
                    .mul(df.marginal_cost)
                    .mul(network_de.snapshot_weightings.generators, axis=0)
                    .sum()
                    .sum()
                )
        if c.name not in [
            "Bus",
            "Load",
            "LineType",
            "TransformerType",
            "Carrier",
        ]:
            if c.name in network_de.one_port_components:
                df = c.df[
                    c.df.bus.isin(
                        network_de.buses[
                            network_de.buses.carrier == "AC"
                        ].index
                    )
                ]
            else:
                df = c.df[
                    c.df.bus0.isin(
                        network_de.buses[
                            network_de.buses.carrier == "AC"
                        ].index
                    )
                ]
            invest_cost += (
                (
                    df[df[f"{value}_nom_extendable"]][f"{value}_nom_opt"]
                    - df[df[f"{value}_nom_extendable"]][f"{value}_nom_min"]
                )
                * df[df[f"{value}_nom_extendable"]]["capital_cost"]
            ).sum()

    # import and its costs
    links_export = self.network.links[
        (
            self.network.links.bus0.isin(network_de.buses.index.values)
            & ~(self.network.links.bus1.isin(network_de.buses.index.values))
        )
    ]

    export_positive = (
        self.network.links_t.p0[links_export.index]
        .clip(lower=0)
        .mul(self.network.snapshot_weightings.generators, axis=0)
        .mul(
            self.network.buses_t.marginal_price[links_export.bus1].values,
        )
        .sum()
        .sum()
    )

    export_negative = (
        self.network.links_t.p0[links_export.index]
        .clip(upper=0)
        .mul(self.network.snapshot_weightings.generators, axis=0)
        .mul(
            self.network.buses_t.marginal_price[links_export.bus1].values,
        )
        .mul(-1)
        .sum()
        .sum()
    )

    links_import = self.network.links[
        (
            self.network.links.bus1.isin(network_de.buses.index.values)
            & ~(self.network.links.bus0.isin(network_de.buses.index.values))
        )
    ]

    import_positive = (
        self.network.links_t.p0[links_import.index]
        .clip(lower=0)
        .mul(self.network.snapshot_weightings.generators, axis=0)
        .mul(
            self.network.buses_t.marginal_price[links_import.bus1].values,
        )
        .sum()
        .sum()
    )

    import_negative = (
        self.network.links_t.p0[links_import.index]
        .clip(upper=0)
        .mul(self.network.snapshot_weightings.generators, axis=0)
        .mul(
            self.network.buses_t.marginal_price[links_import.bus1].values,
        )
        .mul(-1)
        .sum()
        .sum()
    )

    import_costs = (
        export_negative + import_positive - export_positive - import_negative
    )

    return marginal_cost, invest_cost, import_costs


def ac_export(self):
    """Calculate the balance of electricity exports and imports over AC lines

    Returns
    -------
    float
        Balance of electricity export in MWh (if negative: import from Germany)

    """
    de_buses = self.network.buses[self.network.buses.country == "DE"]
    for_buses = self.network.buses[self.network.buses.country != "DE"]
    exp = self.network.lines[
        (self.network.lines.bus0.isin(de_buses.index))
        & (self.network.lines.bus1.isin(for_buses.index))
    ]
    imp = self.network.lines[
        (self.network.lines.bus1.isin(de_buses.index))
        & (self.network.lines.bus0.isin(for_buses.index))
    ]

    return (
        self.network.lines_t.p0[exp.index]
        .sum(axis=1)
        .mul(self.network.snapshot_weightings.generators)
        .sum()
        + self.network.lines_t.p1[imp.index]
        .sum(axis=1)
        .mul(self.network.snapshot_weightings.generators)
        .sum()
    )


def ac_export_per_country(self):
    """Calculate the balance of electricity exports and imports over AC lines
    per country

    Returns
    -------
    float
        Balance of electricity exchange in TWh (if > 0: export from Germany)

    """
    de_buses = self.network.buses[self.network.buses.country == "DE"]

    for_buses = self.network.buses[self.network.buses.country != "DE"]

    result = pd.Series(index=for_buses.country.unique())

    for c in for_buses.country.unique():
        exp = self.network.lines[
            (self.network.lines.bus0.isin(de_buses.index))
            & (
                self.network.lines.bus1.isin(
                    for_buses[for_buses.country == c].index
                )
            )
        ]
        imp = self.network.lines[
            (self.network.lines.bus1.isin(de_buses.index))
            & (
                self.network.lines.bus0.isin(
                    for_buses[for_buses.country == c].index
                )
            )
        ]

        result[c] = (
            self.network.lines_t.p0[exp.index]
            .sum(axis=1)
            .mul(self.network.snapshot_weightings.generators)
            .sum()
            + self.network.lines_t.p1[imp.index]
            .sum(axis=1)
            .mul(self.network.snapshot_weightings.generators)
            .sum()
        ) * 1e-6

    return result


def dc_export(self):
    """Calculate the balance of electricity exports and imports over DC lines

    Returns
    -------
    float
        Balance of electricity exchange in MWh (if > 0: export from Germany)

    """
    de_buses = self.network.buses[self.network.buses.country == "DE"]
    for_buses = self.network.buses[self.network.buses.country != "DE"]
    exp = self.network.links[
        (self.network.links.carrier == "DC")
        & (self.network.links.bus0.isin(de_buses.index))
        & (self.network.links.bus1.isin(for_buses.index))
    ]
    imp = self.network.links[
        (self.network.links.carrier == "DC")
        & (self.network.links.bus1.isin(de_buses.index))
        & (self.network.links.bus0.isin(for_buses.index))
    ]
    return (
        self.network.links_t.p0[exp.index]
        .sum(axis=1)
        .mul(self.network.snapshot_weightings.generators)
        .sum()
        + self.network.links_t.p1[imp.index]
        .sum(axis=1)
        .mul(self.network.snapshot_weightings.generators)
        .sum()
    )


def dc_export_per_country(self):
    """Calculate the balance of electricity exports and imports over DC lines
    per country

    Returns
    -------
    float
        Balance of electricity exchange in TWh (if > 0: export from Germany)

    """
    de_buses = self.network.buses[self.network.buses.country == "DE"]

    for_buses = self.network.buses[self.network.buses.country != "DE"]

    result = pd.Series(index=for_buses.country.unique())

    for c in for_buses.country.unique():
        exp = self.network.links[
            (self.network.links.carrier == "DC")
            & (self.network.links.bus0.isin(de_buses.index))
            & (
                self.network.links.bus1.isin(
                    for_buses[for_buses.country == c].index
                )
            )
        ]
        imp = self.network.links[
            (self.network.links.carrier == "DC")
            & (self.network.links.bus1.isin(de_buses.index))
            & (
                self.network.links.bus0.isin(
                    for_buses[for_buses.country == c].index
                )
            )
        ]

        result[c] = (
            self.network.links_t.p0[exp.index]
            .sum(axis=1)
            .mul(self.network.snapshot_weightings.generators)
            .sum()
            + self.network.links_t.p1[imp.index]
            .sum(axis=1)
            .mul(self.network.snapshot_weightings.generators)
            .sum()
        ) * 1e-6

    return result


def calc_etrago_results(self):
    """Function that calculates main results of grid optimization
    and adds them to Etrago object.

    Returns
    -------
    None.

    """
    self.results = pd.DataFrame(
        columns=["unit", "value"],
        index=[
            "annual system costs",
            "annual investment costs",
            "annual marginal costs",
            "redispatch costs",
            "annual electrical grid investment costs",
            "annual ac grid investment costs",
            "annual dc grid investment costs",
            "annual links investment costs",
            "annual storage+store investment costs",
            "annual electrical storage investment costs",
            "annual store investment costs",
            "battery storage expansion",
            "store expansion",
            "H2 store expansion",
            "CH4 store expansion",
            "heat store expansion",
            "storage+store expansion",
            "fuel cell links expansion",
            "electrolyzer links expansion",
            "methanisation links expansion",
            "Steam Methane Reformation links expansion",
            "abs. electrical grid expansion",
            "abs. electrical grid expansion length",
            "abs. electrical ac grid expansion",
            "abs. electrical dc grid expansion",
            "rel. electrical ac grid expansion",
            "rel. electrical dc grid expansion",
        ],
    )

    self.results.unit[self.results.index.str.contains("cost")] = "EUR/a"
    self.results.unit[self.results.index.str.contains("expansion")] = "MW"
    self.results.unit[self.results.index.str.contains("rel.")] = "p.u."
    self.results.unit[self.results.index.str.contains("length")] = "MW*km"

    # system costs

    self.results.value["annual ac grid investment costs"] = (
        calc_investment_cost(self)[0][0]
    )
    self.results.value["annual dc grid investment costs"] = (
        calc_investment_cost(self)[0][1]
    )
    self.results.value["annual electrical grid investment costs"] = sum(
        calc_investment_cost(self)[0]
    )

    self.results.value["annual links investment costs"] = calc_investment_cost(
        self
    )[1]

    self.results.value["annual electrical storage investment costs"] = (
        calc_investment_cost(self)[2][0]
    )
    self.results.value["annual store investment costs"] = calc_investment_cost(
        self
    )[2][1]
    self.results.value["annual storage+store investment costs"] = sum(
        calc_investment_cost(self)[2]
    )

    self.results.value["annual investment costs"] = (
        sum(calc_investment_cost(self)[0])
        + calc_investment_cost(self)[1]
        + sum(calc_investment_cost(self)[2])
    )
    self.results.value["annual marginal costs"] = calc_marginal_cost(self)

    self.results.value["annual system costs"] = (
        self.results.value["annual investment costs"]
        + self.results.value["annual marginal costs"]
    )

    # storage and store expansion

    network = self.network

    if not network.storage_units[network.storage_units.p_nom_extendable].empty:
        self.results.value["battery storage expansion"] = (
            _calc_storage_expansion(self).sum()
        )

        store = _calc_store_expansion(self)
        self.results.value["store expansion"] = store.sum()
        self.results.value["H2 store expansion"] = store[
            store.index.str.contains("H2")
        ].sum()
        self.results.value["CH4 store expansion"] = store[
            store.index.str.contains("CH4")
        ].sum()
        self.results.value["heat store expansion"] = store[
            store.index.str.contains("heat")
        ].sum()

        self.results.value["storage+store expansion"] = (
            self.results.value["battery storage expansion"]
            + self.results.value["store expansion"]
        )

    # links expansion

    if not network.links[network.links.p_nom_extendable].empty:
        links = _calc_sectorcoupling_link_expansion(self)
        self.results.value["fuel cell links expansion"] = links[0]
        self.results.value["electrolyzer links expansion"] = links[1]
        self.results.value["methanisation links expansion"] = links[2]
        self.results.value["Steam Methane Reformation links expansion"] = (
            links[3]
        )

    # grid expansion

    if not network.lines[network.lines.s_nom_extendable].empty:
        self.results.value["abs. electrical ac grid expansion"] = (
            _calc_network_expansion(self)[0].sum()
        )

        self.results.value["abs. electrical grid expansion length"] = (
            _calc_network_expansion_length(self)[0]
            + _calc_network_expansion_length(self)[1]
        )

        self.results.value["abs. electrical dc grid expansion"] = (
            _calc_network_expansion(self)[1].sum()
        )
        self.results.value["abs. electrical grid expansion"] = (
            self.results.value["abs. electrical ac grid expansion"]
            + self.results.value["abs. electrical dc grid expansion"]
        )

        ext_lines = network.lines[network.lines.s_nom_extendable]
        ext_links = network.links[network.links.p_nom_extendable]
        ext_dc_lines = ext_links[ext_links.carrier == "DC"]

        self.results.value["rel. electrical ac grid expansion"] = (
            _calc_network_expansion(self)[0].sum() / ext_lines.s_nom.sum()
        )
        self.results.value["rel. electrical dc grid expansion"] = (
            _calc_network_expansion(self)[1].sum() / ext_dc_lines.p_nom.sum()
        )

    redispatch_cost = calc_redispatch_cost(self)
    if redispatch_cost is not None:
        self.results.value["redispatch costs"] = redispatch_cost
    else:
        self.results = self.results.drop(index="redispatch costs")


def total_redispatch(self, only_de=True, plot=False):
    """Calculate the redispatch volume and potential

    The time series of the redispatch components are taken from
    etrago.redispatch_timeseries after a run with power flow, otherwise from
    etrago.network, see _redispatch_timeseries.

    Parameters
    ----------
    only_de : bool, optional
        Only consider redispatch components in Germany. The default is True.
    plot : bool, optional
        Plot the redispatch and its potential over time. The default is False.

    Returns
    -------
    dict or None
        Annual volume in MWh ('total'), hourly redispatch in MW
        ('timeseries') and hourly potential in MW ('potential') for
        'ramp_up' and 'ramp_down'. None if no redispatch components are found.

    """
    network = self.network
    redispatch_timeseries = _redispatch_timeseries(self)
    if redispatch_timeseries is None:
        logger.warning("No redispatch components found.")
        return None

    de_buses = network.buses.index[network.buses.country == "DE"]

    result = {}
    for direction, limit in [
        ("ramp_up", "p_max_pu"),
        ("ramp_down", "p_min_pu"),
    ]:
        dispatch_t = pd.Series(0.0, index=network.snapshots)
        potential_t = pd.Series(0.0, index=network.snapshots)

        # Generators: feed-in p, links: feed-in on the electrical side (-p1)
        for component, p, sign, bus in [
            ("Generator", "p", 1, "bus"),
            ("Link", "p1", -1, "bus0"),
        ]:
            if component not in redispatch_timeseries:
                continue
            data = redispatch_timeseries[component]
            static = data["static"]

            idx = static.index[static.index.str.endswith(" " + direction)]
            if only_de:
                idx = idx[static.loc[idx, bus].isin(de_buses)]

            dispatch_t += data[p][idx].sum(axis=1) * sign
            potential_t += (
                data[limit][idx].mul(static.loc[idx, "p_nom"]).sum(axis=1)
            )

        if plot:
            fig, ax = plt.subplots(figsize=(15, 5))
            potential_t.plot(ax=ax, kind="area", color="lightblue")
            dispatch_t.plot(ax=ax, color="blue")

        result[direction] = {
            # Annual redispatch in MWh
            "total": dispatch_t.mul(
                network.snapshot_weightings.generators
            ).sum(),
            # Hourly redispatch in MW
            "timeseries": dispatch_t,
            # Hourly redispatch potential in MW
            "potential": potential_t,
        }

    return result
