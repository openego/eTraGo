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
    import geopandas as gpd
    import pypsa

    logger = logging.getLogger(__name__)

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, s3pp, wolfbunke, mariusves, lukasol"


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
                df = c.df[c.df.bus.isin(network_de.buses[network_de.buses.carrier=="AC"].index)]
                marginal_cost += (
                    c.pnl.p[df.index].mul(df.marginal_cost)
                    .mul(network_de.snapshot_weightings.generators, axis=0)
                    .sum()
                    .sum()
                )

        else:
            if "marginal_cost" in c.df.columns:
                df = c.df[c.df.bus0.isin(network_de.buses[network_de.buses.carrier=="AC"].index)]
                marginal_cost += (
                    c.pnl.p0[df.index].mul(df.marginal_cost)
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
                df = c.df[c.df.bus.isin(network_de.buses[network_de.buses.carrier=="AC"].index)]
            else:
                df = c.df[c.df.bus0.isin(network_de.buses[network_de.buses.carrier=="AC"].index)]
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


def lcoe_germany(self):

    scenario = self.network.buses.scn_name.iloc[0]

    generation_capacity_costs = {
        "powerd2025": 42753843903.92459,
        "powerd2030": 43867007802.88958,
        "powerd2035": 42780310000.0,
        "eGon100RE": 39684763603.00243,
        }

    marginal_cost, invest_cost, import_costs = electricity_system_costs_germany(self)

    total_system_cost_de = marginal_cost + invest_cost #+ import_costs

    if scenario in generation_capacity_costs.keys():
        total_system_cost_de += generation_capacity_costs[scenario]

    ac_gen_de = self.network.generators_t.p[self.network.generators[
        (self.network.generators.bus.isin(
            self.network.buses[(self.network.buses.country=="DE")&(self.network.buses.carrier=="AC")].index
            ))].index].sum(axis=1).mul(self.network.snapshot_weightings["generators"]).sum()
    
    ac_link_de = self.network.links_t.p1[self.network.links[
        (self.network.links.bus1.isin(
            self.network.buses[(self.network.buses.country=="DE")&(self.network.buses.carrier=="AC")].index
            ))].index].sum(axis=1).mul(self.network.snapshot_weightings["generators"]).sum()*(-1)
    
    ac_load_de = self.network.loads_t.p_set[self.network.loads[
        (self.network.loads.carrier=="AC")
        &(self.network.loads.bus.isin(
            self.network.buses[self.network.buses.country=="DE"].index
            ))].index].sum(axis=1).mul(self.network.snapshot_weightings["generators"]).sum()

    sector_coupling_load = self.network.links_t.p0[self.network.links[
        (self.network.links.bus0.isin(
            self.network.buses[(self.network.buses.country=="DE")
                               &(self.network.buses.carrier=="AC")].index
            ))
        &(self.network.links.carrier!="DC")
        ].index].sum(axis=1).mul(self.network.snapshot_weightings["generators"]).sum()

    total_elec_demand_de = ac_load_de + sector_coupling_load

    lcoe = total_system_cost_de / (ac_gen_de+ac_link_de)

    return lcoe

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
            "redispatch cost",
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

    if not network.generators[
        network.generators.index.str.contains("ramp")
    ].empty:
        network = self.network
        gen_idx = network.generators[
            network.generators.index.str.contains("ramp")
        ].index
        gen = (
            network.generators_t.p[gen_idx]
            .mul(network.snapshot_weightings.objective, axis=0)
            .mul(network.generators_t.marginal_cost[gen_idx])
            .sum()
            .sum(axis=0)
        )

        link_idx = network.links[
            network.links.index.str.contains("ramp")
        ].index
        link = (
            network.links_t.p0[link_idx]
            .mul(network.snapshot_weightings.objective, axis=0)
            .mul(
                pypsa.descriptors.get_switchable_as_dense(
                    network, "Link", "marginal_cost"
                )[link_idx]
            )
            .sum(axis=0)
            .sum()
        )
        self.results.value["redispatch cost"] = gen + link


def total_redispatch(network, only_de=True, plot=False):

    if only_de:
        ramp_up = network.generators[
            (network.generators.index.str.contains("ramp_up"))
            & (
                network.generators.bus.isin(
                    network.buses[network.buses.country == "DE"].index.values
                )
            )
        ]

        ramp_up_links = network.links[
            (network.links.index.str.contains("ramp_up"))
            & (
                network.links.bus0.isin(
                    network.buses[network.buses.country == "DE"].index.values
                )
            )
        ]
        ramp_down = network.generators[
            (network.generators.index.str.contains("ramp_down"))
            & (
                network.generators.bus.isin(
                    network.buses[network.buses.country == "DE"].index.values
                )
            )
        ]

        ramp_down_links = network.links[
            (network.links.index.str.contains("ramp_down"))
            & (
                network.links.bus0.isin(
                    network.buses[network.buses.country == "DE"].index.values
                )
            )
        ]

    else:
        ramp_up = network.generators[
            network.generators.index.str.contains("ramp_up")
        ]

        ramp_up_links = network.links[
            network.links.index.str.contains("ramp_up")
        ]
        ramp_down = network.generators[
            network.generators.index.str.contains("ramp_down")
        ]
        ramp_down_links = network.links[
            network.links.index.str.contains("ramp_down")
        ]

    # Annual ramp up in MWh
    total_ramp_up = (
        network.links_t.p1[ramp_up_links.index]
        .sum(axis=1)
        .mul(network.snapshot_weightings.generators)
        .sum()
        * (-1)
        + network.generators_t.p[ramp_up.index]
        .sum(axis=1)
        .mul(network.snapshot_weightings.generators)
        .sum()
    )

    # Hourly ramp up during the year in MW
    total_ramp_up_t = network.links_t.p1[ramp_up_links.index].sum(axis=1) * (
        -1
    ) + network.generators_t.p[ramp_up.index].sum(axis=1)

    # Hourly ramp up potential during the year in MW
    total_ramp_up_potential = network.get_switchable_as_dense(
        "Link", "p_max_pu"
    )[ramp_up_links.index].mul(ramp_up_links.p_nom).sum(
        axis=1
    ) + network.get_switchable_as_dense(
        "Generator", "p_max_pu"
    )[
        ramp_up.index
    ].mul(
        ramp_up.p_nom
    ).sum(
        axis=1
    )

    if plot:
        # Plot potential and accutual ramp up
        fig, ax = plt.subplots(figsize=(15, 5))
        total_ramp_up_potential.plot(ax=ax, kind="area", color="lightblue")
        total_ramp_up_t.plot(ax=ax, color="blue")

    # Annual ramp down in MWh
    total_ramp_down = (
        network.links_t.p1[ramp_down_links.index]
        .sum(axis=1)
        .mul(network.snapshot_weightings.generators)
        .sum()
        * (-1)
        + network.generators_t.p[ramp_down.index]
        .sum(axis=1)
        .mul(network.snapshot_weightings.generators)
        .sum()
    )

    # Hourly ramp down during the year in MW
    total_ramp_down_t = network.links_t.p1[ramp_down_links.index].sum(
        axis=1
    ) * (-1) + network.generators_t.p[ramp_down.index].sum(axis=1)

    # Hourly ramp down potential during the year in MW
    total_ramp_down_potential = network.get_switchable_as_dense(
        "Link", "p_min_pu"
    )[ramp_down_links.index].mul(ramp_down_links.p_nom).sum(
        axis=1
    ) + network.get_switchable_as_dense(
        "Generator", "p_min_pu"
    )[
        ramp_down.index
    ].mul(
        ramp_down.p_nom
    ).sum(
        axis=1
    )

    if plot:
        fig, ax = plt.subplots(figsize=(15, 5))
        total_ramp_down_potential.plot(ax=ax, kind="area", color="lightblue")
        total_ramp_down_t.plot(ax=ax, color="blue")

    return {
        "ramp_up": {
            "total": total_ramp_up,
            "timeseries": total_ramp_up_t,
            "potential": total_ramp_up_potential,
        },
        "ramp_down": {
            "total": total_ramp_down,
            "timeseries": total_ramp_down_t,
            "potential": total_ramp_down_potential,
        },
    }

def regions_per_bus(self):
    """
    Create matching dataframe of clustered AC-buses
    and corresponding MV-grids

    Returns
    -------
    geoms : pd.DataFrame
    """
    from shapely.geometry import Point

    bus_series = pd.Series(
    index=self.network.buses[
        (self.network.buses.carrier=="AC")
        & (self.network.buses.country == "DE")
        ].index, 
    data=0.0,)

    map_buses = self.busmap["orig_network"].buses[
        [
            "carrier",
            "x",
            "y",
            "country",
        ]
    ]
    map_buses = map_buses[
        (map_buses["carrier"] == "AC") & (map_buses["country"] == "DE")
    ]
    map_buses["geom"] = map_buses.apply(
        lambda x: Point(x["x"], x["y"]), axis=1)

    map_buses["cluster"] = map_buses.index.map(self.busmap["busmap"])

    map_buses = gpd.GeoDataFrame(map_buses, geometry="geom")
    try:
        mv_grids = gpd.read_postgis(
            "SELECT bus_id, geom FROM grid.egon_mv_grid_district",
            self.engine,
        ).to_crs(4326)
        mv_grids = mv_grids.set_index("bus_id")
        mv_grids.index = mv_grids.index.astype(str)
        map_buses = map_buses[map_buses.index.isin(mv_grids.index)]
        map_buses["geom_grid"] = mv_grids.loc[map_buses.index].buffer(0.0001)
        
        geoms = gpd.GeoSeries(index=map_buses.cluster.unique())
        
        for i in map_buses.cluster.unique():
            geoms[i] = map_buses[map_buses.cluster == i].geom_grid.unary_union.simplify(0.0001)

        return geoms

    except Exception as e:
        logger.warning(
            "No egon_mv_grid_district table inside the database. To create a matching table for atlas results "
            "please add this table to your database."
        )
        logger.warning(f"Error-Message: {e}")
        
        return gpd.GeoSeries(dtype="geometry", crs=4326)


def merit_order_ely_redispatch(self):
    """
    Each hour, the electrolysers with the highest nodal prices in the grid 
    optimization are designated as dispatch until the electrolysis injection 
    from the market optimization has been met.

    Returns
    -------
    redispatch_electrolysis : pd.DataFrame
    redispatch_electrolysis_per_bus : pd.DataFrame
    df_mv_grids : gpd.GeoDataFrame
    """
    
    # Electrolysis in market optimization
    market_buses = self.market_model.buses[
        (self.market_model.buses.carrier == "AC")
    ].index
    
    ely_market = self.market_model.links[
        (self.market_model.links.carrier == "power_to_H2")
        & (self.market_model.links.bus0.isin(market_buses))
    ]
    
    # Store x, y coordinates for market electrolysis
    ely_market["x"] = self.market_model.buses.loc[ely_market.bus1.values, "x"].values
    ely_market["y"] = self.market_model.buses.loc[ely_market.bus1.values, "y"].values
    
    # Initialize market time series for each bus, grouped by country
    ely_market_t = {}
    for country in self.market_model.buses["country"].unique():
        buses_in_country = self.market_model.buses[self.market_model.buses["country"] == country].index
        ely_market_t[country] = self.market_model.links_t.p0[ely_market.index[ely_market.bus0.isin(buses_in_country)]]
    
    # Electrolysis in grid optimization
    grid_buses = self.network.buses[
        (self.network.buses.carrier == "AC")
    ].index
    
    ely_grid = self.network.links[
        (self.network.links.carrier == "power_to_H2")
        & (self.network.links.bus0.isin(grid_buses.values))
    ]
    
    # Store x, y coordinates for grid electrolysis
    ely_grid["x"] = self.network.buses.loc[ely_grid.bus0.values, "x"].values
    ely_grid["y"] = self.network.buses.loc[ely_grid.bus0.values, "y"].values
    
    # Initialize grid time series, grouped by country
    ely_grid_t = {}
    for country in self.network.buses["country"].unique():
        buses_in_country = self.network.buses[self.network.buses["country"] == country].index
        ely_grid_t[country] = self.network.links_t.p0[ely_grid.index[ely_grid.bus0.isin(buses_in_country)]]
    
    # DataFrames for dispatch and redispatch results
    highest_redispatch_price = pd.Series(index=self.network.snapshots)
    redispatch_electrolysis = pd.DataFrame(index=self.network.snapshots, 
                                           columns=ely_grid.index, 
                                           data=0.)
    dispatch_electrolysis = pd.DataFrame(index=self.network.snapshots, 
                                         columns=ely_grid.index, 
                                         data=0.)
    redispatch_electrolysis_per_bus = pd.DataFrame(
        index=self.network.snapshots,
        columns=self.network.buses[
            (self.network.buses.carrier == "AC")
        ].index, 
        data=0.0
    )
    
    # Main loop: for each snapshot
    for sn in self.network.snapshots: 
        
        for country in self.market_model.buses["country"].unique():
            market_oriented_dispatch = 0
            
            market_at_sn = ely_market_t[country].sum(axis=1)[sn]  # Get market dispatch for this country
            
            # Grid dispatch for this country
            grid_at_sn = pd.DataFrame(ely_grid_t[country].loc[sn])
            
            # Filter bus0 values for the current country
            buses_in_country = self.network.buses[self.network.buses["country"] == country].index
            relevant_buses_in_links = ely_grid.loc[ely_grid["bus0"].isin(buses_in_country)]
            
            # Now we need the marginal prices only for the buses that are in the relevant links for this country
            bus_ids_in_relevant_links = relevant_buses_in_links["bus0"].values
            
            # Extract the corresponding nodal prices
            nodal_prices = self.network.buses_t["marginal_price"].loc[sn, bus_ids_in_relevant_links].values
            
            # Assign the filtered nodal prices to the grid dispatch DataFrame
            grid_at_sn["nodal_price"] = nodal_prices
            
            # Sort grid dispatch by price
            ely_dispatch_sorted_by_price = grid_at_sn.sort_values("nodal_price", ascending=False)
            
            for ely in ely_dispatch_sorted_by_price.index:        
                if market_at_sn == 0:
                    highest_redispatch_price[sn] = ely_dispatch_sorted_by_price.iloc[0]["nodal_price"]
                    redispatch_electrolysis_per_bus.loc[
                            sn, self.network.links.loc[ely, "bus0"]
                        ] += ely_dispatch_sorted_by_price.loc[ely, sn]
                else:              
                    if market_at_sn > market_oriented_dispatch:
                        if market_at_sn >= (market_oriented_dispatch + ely_dispatch_sorted_by_price.loc[ely, sn]):
                            market_oriented_dispatch += ely_dispatch_sorted_by_price.loc[ely, sn]                    
                            dispatch_electrolysis.loc[sn, ely] = ely_dispatch_sorted_by_price.loc[ely, sn]
                        else:
                            dispatch_electrolysis.loc[sn, ely] = market_at_sn - market_oriented_dispatch
                            market_oriented_dispatch += dispatch_electrolysis.loc[sn, ely]
                            redispatch_electrolysis.loc[sn, ely] = (ely_dispatch_sorted_by_price.loc[ely, sn]
                                                                    - dispatch_electrolysis.loc[sn, ely])
                            redispatch_electrolysis_per_bus.loc[
                                sn, self.network.links.loc[ely, "bus0"]
                            ] += (ely_dispatch_sorted_by_price.loc[ely, sn]
                                                                      - dispatch_electrolysis.loc[sn, ely])
                    else:
                        redispatch_electrolysis.loc[sn, ely] = ely_dispatch_sorted_by_price.loc[ely, sn]
                        redispatch_electrolysis_per_bus.loc[
                            sn, self.network.links.loc[ely, "bus0"]
                        ] += ely_dispatch_sorted_by_price.loc[ely, sn]
                        highest_redispatch_price[sn] = ely_dispatch_sorted_by_price.loc[ely, "nodal_price"] 
    
    # matching table bus_id | corresponding mv grids
    mv_grid_geom = regions_per_bus(self)

    if not mv_grid_geom.empty:
        df_mv_grids = gpd.GeoDataFrame(
            geometry = mv_grid_geom,
            crs=4326
            )
    else: df_mv_grids = gpd.GeoDataFrame(geometry=[], crs=4326)
             
    return redispatch_electrolysis, redispatch_electrolysis_per_bus, df_mv_grids


def remaining_redispatch(self, min_flh = 3000):
    """
    Calculating the remaining redispatch per bus. Furthermore
    the method shows an electrolyzer potential based on the 
    remaining redispatch and an assumption of 3000 full-load-hours.

    Parameters
    ----------
    min_flh: int
        Assumption of minimum amount of full-load-hours

    Returns
    -------
    max_ely : pd.DataFrame
    ramp_down_per_bus : pd.DataFrame
    """
    
    ramp_down_per_bus = pd.DataFrame(
        index=self.network.snapshots,
        columns=self.network.buses[
            (self.network.buses.carrier=="AC")
            ].index, data = 0.0)
    
    for bus in ramp_down_per_bus.columns:
        
        ramp_down_per_bus[bus] += self.network.generators_t.p[self.network.generators[
            (self.network.generators.bus==bus)
            & (self.network.generators.index.str.contains("ramp_down"))
            ].index].sum(axis=1).abs()
        
        ramp_down_per_bus[bus] += self.network.links_t.p1[self.network.links[
            (self.network.links.bus1==bus)
            & (self.network.links.index.str.contains("ramp_down"))
            ].index].sum(axis=1).abs()
        
    
    
    max_ely = pd.DataFrame(
        index=self.network.buses[
            (self.network.buses.carrier=="AC")
            ].index,
        columns=["max_capacity", "x", "y"],
        data={
            "max_capacity" : 0.0,
            "x" : self.network.buses[
                (self.network.buses.carrier=="AC")
                ].x,
            "y" : self.network.buses[
                (self.network.buses.carrier=="AC")
                ].y,
            
            }
        )

    for size in range(1,200):
        for bus in max_ely.index:
            if (((ramp_down_per_bus.loc[:, bus].clip(upper=size)).sum()*(5)) >= (min_flh*size)):             
                max_ely.loc[bus, "max_capacity"] = size                              

    return max_ely, ramp_down_per_bus



def calc_atlas_results(self, filename=None):
    """
    Calculating the final results for the potential_atlas as 
    one of the main outcomes of the project. The results will
    be stored in a csv file for providing it to the project
    partners. Additonally the method creates a matching table
    for assign each clustered bus to the corresponding mv-grids.

    Parameters
    ----------
    
    Returns
    -------
    results : pd.DataFrame
    matching_mv_grids : gpd.GeoDataFrame
    """
    import numpy as np

    results = pd.DataFrame()

    heating_value_H2 = 33.33  # [kWh/kg]
    O2_calc_factor = 9.030816  # [t_O2/MWh_el] average value produced O2 per electricity, own calculation

    max_ely, ramp_down_per_bus = remaining_redispatch(self)
    redispatch_electrolysis, redispatch_electrolysis_per_bus, matching_mv_grids = merit_order_ely_redispatch(self)

    PtH2_links = self.network.links[(self.network.links.carrier == "power_to_H2") & (self.network.links.bus0.isin(
        self.network.buses[self.network.buses.country=="DE"].index
        ))]
    PtH2_links = PtH2_links[PtH2_links.p_nom_opt>10]
    AC_buses_PtH2 = self.network.buses[self.network.buses.index.isin(PtH2_links.bus0.unique())]

    # Calculate CAPEX
    p = 0.05
    scenario = self.network.buses.scn_name.iloc[0]
    lt_system = {
        "powerd2025": 20,
        "powerd2030": 25,
        "powerd2035": 25,
        "eGon100RE": 30,
        }
    lt = lt_system[scenario]
    #cost that are not included in clean CAPEX
    OPEX_STACK = 0.03 * 0.21 * 357_000 
    OPEX_SYSTEM = 0.03 * 357_000 
    OPEX_PIPES = 0.03 * 236 
    an_capex_stack = annualize_capital_costs(0.21*357_000, 20, 0.07) #interest rate for gas_sector 0.07
    
    for index, row in AC_buses_PtH2.iterrows():

        links_PtH2 = PtH2_links[PtH2_links.bus0 == index]
        
        if "H2" in self.network.buses.loc[links_PtH2.bus1, "carrier"].unique():
            at_h2_grid = False
        elif "H2_grid" in self.network.buses.loc[links_PtH2.bus1, "carrier"].unique():
            at_h2_grid = True

        #calculation for multiple_link_model
        if self.args["method"]["formulation"] == "linopy":

            # Check if elctrolyzer has coupling product usage
            links_PtH2_bus2 = links_PtH2['bus2'].replace(['', 'nan', None], np.nan).dropna()
            links_PtH2_bus3 = links_PtH2['bus3'].replace(['', 'nan', None], np.nan).dropna()
            
            buses_heat = links_PtH2_bus2.astype(float).astype(int).astype(str).tolist() 
            buses_o2 = links_PtH2_bus3.astype(float).astype(int).astype(str).tolist() 
            
            if buses_heat:
                links_waste_heat = self.network.links[self.network.links.bus0.isin(buses_heat)]
            else:
                links_waste_heat = []

            if buses_o2:
                links_o2 = self.network.links[self.network.links.bus0.isin(buses_o2)]
            else:
                links_o2 = []
                
        else: # calculation for generator model
            link_indices = links_PtH2.index.astype(str)       

            #Filter out corresponding o2 and heat generators
            gen_o2 = self.network.generators[self.network.generators.index.isin([f"{link_index}_O2" for link_index in link_indices])]
            gen_heat = self.network.generators[self.network.generators.index.isin([f"{link_index}_waste_heat" for link_index in link_indices])]

            if not gen_o2.empty:
                bus_o2 = gen_o2.bus.iloc[0]  
                links_o2 = self.network.links[self.network.links.bus0==bus_o2]
            else:
                links_o2 = []
                
            if not gen_heat.empty:
                bus_heat = gen_heat.bus.iloc[0]
                links_waste_heat = self.network.links[self.network.links.bus0==bus_heat]
            else:
                links_waste_heat = []

        # Calculate Dispatch
        AC_dispatch = self.network.links_t.p0[links_PtH2.index].mul(self.network.snapshot_weightings.objective, axis=0).sum().sum() 
        H2_dispatch = -self.network.links_t.p1[links_PtH2.index].mul(self.network.snapshot_weightings.objective, axis=0).sum().sum()      
        waste_heat_dispatch = -self.network.links_t.p1.get(links_waste_heat.index, pd.Series(0)).mul(self.network.snapshot_weightings.objective, axis=0).sum().sum()
        o2_dispatch = -self.network.links_t.p1.get(links_o2.index, pd.Series(0)).mul(self.network.snapshot_weightings.objective, axis=0).sum().sum()
        # LCOE+LCOH
        sn = self.network.snapshots[
            (self.network.links_t.p0[links_PtH2.index].sum(axis=1) > 10)]
        mean_local_electricity_cost = self.network.buses_t.marginal_price.loc[sn, row.name].mean() #[€/MWh_e]

        lcoh = (lcoe_germany(self) * (1/self.network.links.efficiency[links_PtH2.index])
                + (links_PtH2.capital_cost* links_PtH2.p_nom_opt).sum() / H2_dispatch
                ).mean()*33.33*1e-3        #[€/kg_H2]

        # H2-demand
        loads_h2 = self.network.loads[self.network.loads.carrier.str.contains('H2') 
                                 & self.network.loads.bus.isin(links_PtH2.bus1.astype(int).astype(str).tolist())]
        try: 
            H2_demand = self.network.loads_t.p_set[loads_h2.index].mul(self.network.snapshot_weightings.objective, axis=0).sum().sum()
        except: 
            H2_demand = self.network.loads.p_set[loads_h2.index].sum()

        #store_capacity
        stores_h2 = self.network.stores[self.network.stores.bus.isin(links_PtH2.bus1.astype(int).astype(str).tolist())]
        store_cap = stores_h2.e_nom_opt.sum()

        #redispatch
        ramp_down = ramp_down_per_bus.mul(self.network.snapshot_weightings.objective, axis=0).sum(axis=0)[index]
        redispatch_ely = redispatch_electrolysis_per_bus.mul(self.network.snapshot_weightings.objective, axis=0).sum(axis=0)[index]

        # specific costs for ELY
        capex_ely = (links_PtH2.capital_cost.mean()  - 
                     OPEX_PIPES - OPEX_STACK - OPEX_SYSTEM - an_capex_stack
                     ) * ((1 / p) - (1 / (p * (1 + p) ** lt)))

        #market_driven/grid_driven
        if redispatch_ely < 1e5:
            dispatch_type = 'market_driven'
        else: 
            dispatch_type = 'grid_driven'

        new_row = {
            'region': row.name,
            'Placement': "System optimization",
            'E': row.x,
            'N': row.y,
            'Type': dispatch_type,
            'Max. electrolyzer capacity [MW]': links_PtH2.p_nom_opt.sum(),
            'Max. electricity consumption [MWh]': AC_dispatch,
            'ELY investment cost [€/kW]': capex_ely/1000,
            'Max. H2-Production [ton/a]': H2_dispatch / heating_value_H2,
            'Max. heat supply [MWh/a]': waste_heat_dispatch,
            'Max. O2 supply [ton/a]': o2_dispatch * O2_calc_factor,
            'LCOH [€/kg_H2]': lcoh,
            'Mean nodal electricity cost [€/MWh_el]': mean_local_electricity_cost,
            'Max. redispatch by electrolysis [MWh/a]':redispatch_ely,
            'Remaining redispatch [MWh/a]': ramp_down,
            'Max. redispatch potential': ramp_down+redispatch_ely, 
            'H2 demand [ton/a]':  H2_demand,
            'Max. hydrogen storage capacity': store_cap,
            'At hydrogen grid': at_h2_grid,

        }
        new_row_df = pd.DataFrame([new_row])
        results = pd.concat([results, new_row_df], ignore_index=True)

    # additional ely potential calculated out of remaining redispacth
    for bus, row in max_ely[max_ely['max_capacity'] > 0].iterrows():
        new_row = {
            'region': bus,
            'Placement': "Additional redispatch potential",
            'E': row.x,
            'N': row.y,
            'Type': "grid_driven",
            'Max. electrolyzer capacity [MW]': row.max_capacity,
            'Max. electricity consumption [MWh]': None,
            'ELY investment cost [€/MW]': None,
            'Max. H2-Production [ton/a]': None,
            'Max. heat supply [MWh/a]': None,
            'Max. O2 supply [ton/a]': None,
            'LCOH [€/kg_H2]': None,
            'LCOE [€/MWh_el]': None,
            'Max. redispatch by electrolysis [MWh/a]': None,
            'Remaining redispatch [MWh/a]': None,
            'Max. redispatch potential': None, 
            'H2 demand [ton/a]': None,
            'Max. hydrogen storage capacity': None,
            'At hydrogen grid': at_h2_grid,
        }
        results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

    if filename:
        results.to_csv(filename)
        matching_mv_grids.to_file("regions.geojson")

    return results

