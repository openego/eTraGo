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

    import pandas as pd

    logger = logging.getLogger(__name__)

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, s3pp, wolfbunke, mariusves, lukasol"

from etrago.tools.utilities import find_buses_area

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
        sto_costs[0] = (ext_storage.p_nom_opt * ext_storage.capital_cost).sum()

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
        .sum(axis=0)
        .mul(network.generators.marginal_cost)
        .sum()
    )
    link = (
        abs(network.links_t.p0)
        .mul(network.snapshot_weightings.objective, axis=0)
        .sum(axis=0)
        .mul(network.links.marginal_cost)
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


def system_costs_germany(self):
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


def ac_export(self):
    """Calculate electricity exports and imports over AC lines

    Returns
    -------
    float
        Electricity export (if negative: import) from Germany

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
    """Calculate electricity exports and imports over AC lines per country

    Returns
    -------
    float
        Electricity export (if negative: import) from Germany in TWh

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
    """Calculate electricity exports and imports over DC lines

    Returns
    -------
    float
        Electricity export (if negative: import) from Germany

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
    """Calculate electricity exports and imports over DC lines per country

    Returns
    -------
    float
        Electricity export (if negative: import) from Germany in TWh

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
            "abs. electrical ac grid expansion",
            "abs. electrical dc grid expansion",
            "rel. electrical ac grid expansion",
            "rel. electrical dc grid expansion",
        ],
    )

    self.results.unit[self.results.index.str.contains("cost")] = "EUR/a"
    self.results.unit[self.results.index.str.contains("expansion")] = "MW"
    self.results.unit[self.results.index.str.contains("rel.")] = "p.u."

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

    #self.results.to_csv(results_file)


#calculating results in focus regions:

def _calc_storage_expansion_areas(self):
    """Function that calculates storage expansion in MW for primary and secondary areas"""
    storage_expansion_areas = pd.DataFrame(columns=["area", "carrier", "expansion"])

    for area in ["primary", "secondary"]:
        if area == "primary":
            network_area = self.primary_network()
        else:
            network_area = self.secondary_network()

        storage_expansion = (
            (
                network_area.storage_units.p_nom_opt
                - network_area.storage_units.p_nom_min
            )[network_area.storage_units.p_nom_extendable]
            .groupby(network_area.storage_units.carrier)
            .sum()
        )

        for carrier, expansion in storage_expansion.items():
            storage_expansion_areas = storage_expansion_areas.append(
                {"area": area, "carrier": carrier, "expansion": expansion},
                ignore_index=True,
            )

    return storage_expansion_areas


def _calc_store_expansion_areas(self):
    """Function that calculates store expansion in MW for primary and secondary areas"""
    store_expansion_areas = pd.DataFrame(columns=["area", "expansion"])

    for area in ["primary", "secondary"]:
        if area == "primary":
            network_area = self.primary_network()
        else:
            network_area = self.secondary_network()

        store_expansion = (
            network_area.stores.e_nom_opt - network_area.stores.e_nom_min
        )[network_area.stores.e_nom_extendable].sum()

        store_expansion_areas = store_expansion_areas.append(
            {"area": area, "expansion": store_expansion}, ignore_index=True
        )

    return store_expansion_areas


def _calc_sectorcoupling_link_expansion_areas(self):
    """Function that calculates expansion of sectorcoupling links in MW for primary and secondary areas"""
    link_expansion_areas = pd.DataFrame(
        columns=["area", "H2_to_power", "power_to_H2", "H2_to_CH4", "CH4_to_H2"]
    )

    for area in ["primary", "secondary"]:
        if area == "primary":
            network_area = self.primary_network()
        else:
            network_area = self.secondary_network()

        ext_links = network_area.links[network_area.links.p_nom_extendable]

        links = [0, 0, 0, 0]

        l1 = ext_links[ext_links.carrier == "H2_to_power"]
        l2 = ext_links[ext_links.carrier == "power_to_H2"]
        l3 = ext_links[ext_links.carrier == "H2_to_CH4"]
        l4 = ext_links[ext_links.carrier == "CH4_to_H2"]

        links[0] = (l1.p_nom_opt - l1.p_nom_min).sum()
        links[1] = (l2.p_nom_opt - l2.p_nom_min).sum()
        links[2] = (l3.p_nom_opt - l3.p_nom_min).sum()
        links[3] = (l4.p_nom_opt - l4.p_nom_min).sum()

        link_expansion_areas = link_expansion_areas.append(
            {
                "area": area,
                "H2_to_power": links[0],
                "power_to_H2": links[1],
                "H2_to_CH4": links[2],
                "CH4_to_H2": links[3],
            },
            ignore_index=True,
        )

    return link_expansion_areas


def _calc_network_expansion_areas(self):
    """Function that calculates electrical network expansion in MW for primary and secondary areas"""
    network_expansion_areas = pd.DataFrame(columns=["area", "lines", "dc_links"])

    for area in ["primary", "secondary"]:
        if area == "primary":
            network_area = self.primary_network()
        else:
            network_area = self.secondary_network()

        lines = (network_area.lines.s_nom_opt - network_area.lines.s_nom_min)[
            network_area.lines.s_nom_extendable
        ].sum()

        ext_links = network_area.links[network_area.links.p_nom_extendable]
        ext_dc_lines = ext_links[ext_links.carrier == "DC"]

        dc_links = (ext_dc_lines.p_nom_opt - ext_dc_lines.p_nom_min).sum()

        network_expansion_areas = network_expansion_areas.append(
            {"area": area, "lines": lines, "dc_links": dc_links}, ignore_index=True
        )

    return network_expansion_areas


def calc_investment_cost_areas(self, network):
    """Function that calculates overall annualized investment costs for both primary and secondary areas.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.

    Returns
    -------
    primary_costs : tuple
        Investment costs for the primary area (network_costs, link_costs, sto_costs).
    secondary_costs : tuple
        Investment costs for the secondary area (network_costs, link_costs, sto_costs).
    """
    # Find the buses that belong to the primary and secondary areas
    primary_buses = find_buses_area(self, carrier="AC", area_type="primary")
    secondary_buses = find_buses_area(self, carrier="AC", area_type="secondary")

    # Initialize cost variables for both areas
    primary_network_costs = [0, 0]
    primary_link_costs = 0
    primary_sto_costs = [0, 0]

    secondary_network_costs = [0, 0]
    secondary_link_costs = 0
    secondary_sto_costs = [0, 0]

    # Filter the network components based on the buses in each area
    ext_lines = network.lines[network.lines.s_nom_extendable]
    ext_trafos = network.transformers[network.transformers.s_nom_extendable]
    ext_links = network.links[network.links.p_nom_extendable]
    ext_dc_lines = ext_links[ext_links.carrier == "DC"]
    ext_storage = network.storage_units[network.storage_units.p_nom_extendable]
    ext_store = network.stores[network.stores.e_nom_extendable]

    # Calculate costs for the primary area
    if not ext_lines[ext_lines.bus0.isin(primary_buses) | ext_lines.bus1.isin(primary_buses)].empty:
        primary_network_costs[0] = (
            (ext_lines[ext_lines.bus0.isin(primary_buses) | ext_lines.bus1.isin(primary_buses)].s_nom_opt -
             ext_lines[ext_lines.bus0.isin(primary_buses) | ext_lines.bus1.isin(primary_buses)].s_nom_min)
            * ext_lines[ext_lines.bus0.isin(primary_buses) | ext_lines.bus1.isin(primary_buses)].capital_cost
        ).sum()

    if not ext_trafos[ext_trafos.bus0.isin(primary_buses) | ext_trafos.bus1.isin(primary_buses)].empty:
        primary_network_costs[0] += (
            (ext_trafos[ext_trafos.bus0.isin(primary_buses) | ext_trafos.bus1.isin(primary_buses)].s_nom_opt -
             ext_trafos[ext_trafos.bus0.isin(primary_buses) | ext_trafos.bus1.isin(primary_buses)].s_nom)
            * ext_trafos[ext_trafos.bus0.isin(primary_buses) | ext_trafos.bus1.isin(primary_buses)].capital_cost
        ).sum()

    if not ext_dc_lines[ext_dc_lines.bus0.isin(primary_buses) | ext_dc_lines.bus1.isin(primary_buses)].empty:
        primary_network_costs[1] = (
            (ext_dc_lines[ext_dc_lines.bus0.isin(primary_buses) | ext_dc_lines.bus1.isin(primary_buses)].p_nom_opt -
             ext_dc_lines[ext_dc_lines.bus0.isin(primary_buses) | ext_dc_lines.bus1.isin(primary_buses)].p_nom_min)
            * ext_dc_lines[ext_dc_lines.bus0.isin(primary_buses) | ext_dc_lines.bus1.isin(primary_buses)].capital_cost
        ).sum()

    ext_links_primary = ext_links[(ext_links.bus0.isin(primary_buses) | ext_links.bus1.isin(primary_buses)) & (ext_links.carrier != "DC")]
    if not ext_links_primary.empty:
        primary_link_costs = (
            (ext_links_primary.p_nom_opt - ext_links_primary.p_nom_min)
            * ext_links_primary.capital_cost
        ).sum()

    if not ext_storage[ext_storage.bus.isin(primary_buses)].empty:
        primary_sto_costs[0] = (ext_storage[ext_storage.bus.isin(primary_buses)].p_nom_opt * ext_storage[ext_storage.bus.isin(primary_buses)].capital_cost).sum()

    if not ext_store[ext_store.bus.isin(primary_buses)].empty:
        primary_sto_costs[1] = (ext_store[ext_store.bus.isin(primary_buses)].e_nom_opt * ext_store[ext_store.bus.isin(primary_buses)].capital_cost).sum()

    # Calculate costs for the secondary area
    if not ext_lines[ext_lines.bus0.isin(secondary_buses) | ext_lines.bus1.isin(secondary_buses)].empty:
        secondary_network_costs[0] = (
            (ext_lines[ext_lines.bus0.isin(secondary_buses) | ext_lines.bus1.isin(secondary_buses)].s_nom_opt -
             ext_lines[ext_lines.bus0.isin(secondary_buses) | ext_lines.bus1.isin(secondary_buses)].s_nom_min)
            * ext_lines[ext_lines.bus0.isin(secondary_buses) | ext_lines.bus1.isin(secondary_buses)].capital_cost
        ).sum()

    if not ext_trafos[ext_trafos.bus0.isin(secondary_buses) | ext_trafos.bus1.isin(secondary_buses)].empty:
        secondary_network_costs[0] += (
            (ext_trafos[ext_trafos.bus0.isin(secondary_buses) | ext_trafos.bus1.isin(secondary_buses)].s_nom_opt -
             ext_trafos[ext_trafos.bus0.isin(secondary_buses) | ext_trafos.bus1.isin(secondary_buses)].s_nom)
            * ext_trafos[ext_trafos.bus0.isin(secondary_buses) | ext_trafos.bus1.isin(secondary_buses)].capital_cost
        ).sum()

    if not ext_dc_lines[ext_dc_lines.bus0.isin(secondary_buses) | ext_dc_lines.bus1.isin(secondary_buses)].empty:
        secondary_network_costs[1] = (
            (ext_dc_lines[ext_dc_lines.bus0.isin(secondary_buses) | ext_dc_lines.bus1.isin(secondary_buses)].p_nom_opt -
             ext_dc_lines[ext_dc_lines.bus0.isin(secondary_buses) | ext_dc_lines.bus1.isin(secondary_buses)].p_nom_min)
            * ext_dc_lines[ext_dc_lines.bus0.isin(secondary_buses) | ext_dc_lines.bus1.isin(secondary_buses)].capital_cost
        ).sum()

    ext_links_secondary = ext_links[(ext_links.bus0.isin(secondary_buses) | ext_links.bus1.isin(secondary_buses)) & (ext_links.carrier != "DC")]
    if not ext_links_secondary.empty:
        secondary_link_costs = (ext_links_secondary.p_nom_opt - ext_links_secondary.p_nom_min) * (ext_links_secondary.capital_cost).sum()

    if not ext_storage[ext_storage.bus.isin(secondary_buses)].empty:
        secondary_sto_costs[0] = (ext_storage[ext_storage.bus.isin(secondary_buses)].p_nom_opt * ext_storage[ext_storage.bus.isin(secondary_buses)].capital_cost).sum()

    if not ext_store[ext_store.bus.isin(secondary_buses)].empty:
        secondary_sto_costs[1] = (ext_store[ext_store.bus.isin(secondary_buses)].e_nom_opt * ext_store[ext_store.bus.isin(secondary_buses)].capital_cost).sum()

    primary_costs = (primary_network_costs, primary_link_costs, primary_sto_costs)
    secondary_costs = (secondary_network_costs, secondary_link_costs, secondary_sto_costs)

    return primary_costs, secondary_costs

def calc_marginal_cost_areas(self, network):
    """
    Function that calculates and returns marginal costs, considering
    generation and link and storage dispatch costs for both primary and secondary areas.

    Parameters
    ----------
    network : pypsa.Network
        The PyPSA network object.

    Returns
    -------
    primary_marginal_cost : float
        Annual marginal cost in EUR for the primary area.
    secondary_marginal_cost : float
        Annual marginal cost in EUR for the secondary area.
    """
    # Find the buses that belong to the primary and secondary areas
    primary_buses = find_buses_area(self, carrier="AC", area_type="primary")
    secondary_buses = find_buses_area(self, carrier="AC", area_type="secondary")

    # Calculate marginal costs for the primary area
    primary_gen = (
        network.generators_t.p.loc[:, network.generators.bus.isin(primary_buses)]
        .mul(network.snapshot_weightings.objective, axis=0)
        .sum(axis=0)
        .mul(network.generators[network.generators.bus.isin(primary_buses)].marginal_cost)
        .sum()
    )
    primary_link = (
        abs(network.links_t.p0.loc[:, (network.links.bus0.isin(primary_buses) | network.links.bus1.isin(primary_buses))])
        .mul(network.snapshot_weightings.objective, axis=0)
        .sum(axis=0)
        .mul(network.links[(network.links.bus0.isin(primary_buses) | network.links.bus1.isin(primary_buses))].marginal_cost)
        .sum()
    )
    primary_stor = (
        network.storage_units_t.p.loc[:, network.storage_units.bus.isin(primary_buses)]
        .mul(network.snapshot_weightings.objective, axis=0)
        .sum(axis=0)
        .mul(network.storage_units[network.storage_units.bus.isin(primary_buses)].marginal_cost)
        .sum()
    )
    primary_marginal_cost = primary_gen + primary_link + primary_stor

    # Calculate marginal costs for the secondary area
    secondary_gen = (
        network.generators_t.p.loc[:, network.generators.bus.isin(secondary_buses)]
        .mul(network.snapshot_weightings.objective, axis=0)
        .sum(axis=0)
        .mul(network.generators[network.generators.bus.isin(secondary_buses)].marginal_cost)
        .sum()
    )
    secondary_link = (
        abs(network.links_t.p0.loc[:, (network.links.bus0.isin(secondary_buses) | network.links.bus1.isin(secondary_buses))])
        .mul(network.snapshot_weightings.objective, axis=0)
        .sum(axis=0)
        .mul(network.links[(network.links.bus0.isin(secondary_buses) | network.links.bus1.isin(secondary_buses))].marginal_cost)
        .sum()
    )
    secondary_stor = (
        network.storage_units_t.p.loc[:, network.storage_units.bus.isin(secondary_buses)]
        .mul(network.snapshot_weightings.objective, axis=0)
        .sum(axis=0)
        .mul(network.storage_units[network.storage_units.bus.isin(secondary_buses)].marginal_cost)
        .sum()
    )
    secondary_marginal_cost = secondary_gen + secondary_link + secondary_stor

    return primary_marginal_cost, secondary_marginal_cost    
    
def calc_system_costs_areas(self):
    """Calculate system costs for primary and secondary areas"""
    self.system_costs_areas = pd.DataFrame(
        columns=["area", "marginal_cost", "invest_cost", "import_costs"],
        index=["primary", "secondary"],
    )

    for area in ["primary", "secondary"]:
        if area == "primary":
            network_area = self.primary_network()
        else:
            network_area = self.secondary_network()

        marginal_cost, invest_cost, import_costs = self.system_costs_area(network_area)

        self.system_costs_areas.loc[area, "area"] = area
        self.system_costs_areas.loc[area, "marginal_cost"] = marginal_cost
        self.system_costs_areas.loc[area, "invest_cost"] = invest_cost
        self.system_costs_areas.loc[area, "import_costs"] = import_costs

def system_costs_areas(self, network_area):
    """Calculate system costs for a specific area"""
    marginal_cost = 0
    invest_cost = 0

    for c in network_area.iterate_components():
        if c.name in ["Store"]:
            value = "e"
        elif c.name in ["Line", "Transformer"]:
            value = "s"
        else:
            value = "p"
        if c.name in network_area.one_port_components:
            if "marginal_cost" in c.df.columns:
                marginal_cost += (
                    c.pnl.p.mul(c.df.marginal_cost)
                    .mul(network_area.snapshot_weightings.generators, axis=0)
                    .sum()
                    .sum()
                )
        else:
            if "marginal_cost" in c.df.columns:
                marginal_cost += (
                    c.pnl.p0.mul(c.df.marginal_cost)
                    .mul(network_area.snapshot_weightings.generators, axis=0)
                    .sum()
                    .sum()
                )
        if c.name not in ["Bus", "Load", "LineType", "TransformerType", "Carrier"]:
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
            self.network.links.bus0.isin(network_area.buses.index.values)
            & ~(self.network.links.bus1.isin(network_area.buses.index.values))
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
            self.network.links.bus1.isin(network_area.buses.index.values)
            & ~(self.network.links.bus0.isin(network_area.buses.index.values))
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

    import_costs = export_negative + import_positive - export_positive - import_negative

    return marginal_cost, invest_cost, import_costs



def calc_etrago_results_areas(self):
    """
    Function that calculates and stores eTraGo results for primary and secondary areas.
    """
    network = self.network.copy()

    # Calculate investment costs for primary and secondary areas
    primary_costs, secondary_costs = calc_investment_cost_areas(self, network)

    # Calculate marginal costs for primary and secondary areas
    primary_marginal_cost, secondary_marginal_cost = calc_marginal_cost_areas(self, network)

    # Create a DataFrame to store the results
    self.results_area = pd.DataFrame(
        columns=["value", "unit"],
        index=pd.MultiIndex.from_tuples(
            [
                ("primary", "annual ac grid investment costs"),
                ("primary", "annual dc grid investment costs"),
                ("primary", "annual electrical grid investment costs"),
                ("primary", "annual generation investment costs"),
                ("primary", "annual invest costs for link electrical"),
                ("primary", "annual investment costs power to gas"),
                ("primary", "annual investment costs gas to power"),
                ("primary", "annual power to gas dispatch costs"),
                ("primary", "annual gas to power dispatch costs"),
                ("primary", "annual marginal cost"),
                ("secondary", "annual ac grid investment costs"),
                ("secondary", "annual dc grid investment costs"),
                ("secondary", "annual electrical grid investment costs"),
                ("secondary", "annual generation investment costs"),
                ("secondary", "annual invest costs for link electrical"),
                ("secondary", "annual investment costs power to gas"),
                ("secondary", "annual investment costs gas to power"),
                ("secondary", "annual power to gas dispatch costs"),
                ("secondary", "annual gas to power dispatch costs"),
                ("secondary", "annual marginal cost"),
            ],
            names=["area", "label"],
        ),
    )

    # Store investment costs for primary area
    self.results_area.loc[("primary", "annual ac grid investment costs"), "value"] = primary_costs[0][0]
    self.results_area.loc[("primary", "annual dc grid investment costs"), "value"] = primary_costs[0][1]
    self.results_area.loc[("primary", "annual electrical grid investment costs"), "value"] = sum(primary_costs[0])
    self.results_area.loc[("primary", "annual invest costs for link electrical"), "value"] = primary_costs[1]
    self.results_area.loc[("primary", "annual generation investment costs"), "value"] = primary_costs[2][0]
    self.results_area.loc[("primary", "annual investment costs power to gas"), "value"] = primary_costs[2][1]

    # Store investment costs for secondary area
    self.results_area.loc[("secondary", "annual ac grid investment costs"), "value"] = secondary_costs[0][0]
    self.results_area.loc[("secondary", "annual dc grid investment costs"), "value"] = secondary_costs[0][1]
    self.results_area.loc[("secondary", "annual electrical grid investment costs"), "value"] = sum(secondary_costs[0])
    #self.results_area.loc[("secondary", "annual invest costs for link electrical"), "value"] = secondary_costs[1]
    
    if isinstance(secondary_costs[1], pd.Series):
       self.results_area.loc[("secondary", "annual invest costs for link electrical"), "value"] = secondary_costs[1].values[0]
    else:
       self.results_area.loc[("secondary", "annual invest costs for link electrical"), "value"] = secondary_costs[1]
    
        
    self.results_area.loc[("secondary", "annual generation investment costs"), "value"] = secondary_costs[2][0]
    self.results_area.loc[("secondary", "annual investment costs power to gas"), "value"] = secondary_costs[2][1]

    # Store marginal costs for primary and secondary areas
    self.results_area.loc[("primary", "annual marginal cost"), "value"] = primary_marginal_cost
    self.results_area.loc[("secondary", "annual marginal cost"), "value"] = secondary_marginal_cost

    # Set the units for the results
    self.results_area.loc[
        [
            ("primary", "annual ac grid investment costs"),
            ("primary", "annual dc grid investment costs"),
            ("primary", "annual electrical grid investment costs"),
            ("primary", "annual generation investment costs"),
            ("primary", "annual invest costs for link electrical"),
            ("primary", "annual investment costs power to gas"),
            ("primary", "annual investment costs gas to power"),
            ("primary", "annual power to gas dispatch costs"),
            ("primary", "annual gas to power dispatch costs"),
            ("primary", "annual marginal cost"),
            ("secondary", "annual ac grid investment costs"),
            ("secondary", "annual dc grid investment costs"),
            ("secondary", "annual electrical grid investment costs"),
            ("secondary", "annual generation investment costs"),
            ("secondary", "annual invest costs for link electrical"),
            ("secondary", "annual investment costs power to gas"),
            ("secondary", "annual investment costs gas to power"),
            ("secondary", "annual power to gas dispatch costs"),
            ("secondary", "annual gas to power dispatch costs"),
            ("secondary", "annual marginal cost"),
        ],
        "unit",
    ] = "EUR"

# Save the results to a CSV file

