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
Constraints.py includes additional constraints for eTraGo-optimizations
"""
import logging
import os

from pyomo.environ import Constraint
from pypsa.descriptors import (
    expand_series,
    get_switchable_as_dense as get_as_dense,
)
from pypsa.optimization.compat import get_var, define_constraints, linexpr
import numpy as np
import pandas as pd
import pyomo.environ as po

if "READTHEDOCS" not in os.environ:
    from etrago.tools import db

logger = logging.getLogger(__name__)

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = """ulfmueller, s3pp, wolfbunke, mariusves, lukasol, AmeliaNadal,
CarlosEpia, ClaraBuettner, KathiEsterl"""


def _get_crossborder_components(network, cntr="all"):
    """
    Identifies foreign buses and crossborder ac- and dc-lines for all foreign
    countries or only one specific.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    cntr : str, optional
        Country code of the returned buses and lines. The default is 'all'.

    Returns
    -------
    buses_de : pandas.base.Index
        Index of buses located in Germany
    buses_for : pandas.base.Index
        Index of buses located in selected country
    cb0 : pandas.base.Index
        Index of ac-lines from foreign country to Germany
    cb1 : pandas.base.Index
        Index of ac-lines from Germany to foreign country
    cb0_link : pandas.base.Index
        Index of dc-lines from foreign country to Germany
    cb1_link : pandas.base.Index
        Index of dc-lines from Germany to foreign country

    """
    buses_de = network.buses.index[network.buses.country == "DE"]

    if cntr == "all":
        buses_for = network.buses.index[network.buses.country != "DE"]
    else:
        buses_for = network.buses.index[network.buses.country == cntr]

    cb0 = network.lines.index[
        (network.lines.bus0.isin(buses_for))
        & (network.lines.bus1.isin(buses_de))
    ]

    cb1 = network.lines.index[
        (network.lines.bus1.isin(buses_for))
        & (network.lines.bus0.isin(buses_de))
    ]

    cb0_link = network.links.index[
        (network.links.bus0.isin(buses_for))
        & (network.links.bus1.isin(buses_de))
        & (network.links.carrier == "DC")
    ]

    cb1_link = network.links.index[
        (network.links.bus0.isin(buses_de))
        & (network.links.bus1.isin(buses_for))
        & (network.links.carrier == "DC")
    ]

    return buses_de, buses_for, cb0, cb1, cb0_link, cb1_link


def _max_line_ext(self, network, snapshots):
    """
    Extra-functionality that limits overall line expansion to a multiple
    of existing capacity.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None

    """

    lines_snom = network.lines.s_nom_min.sum()

    links_elec = network.links[network.links.carrier == "DC"]
    links_index = links_elec.index
    links_pnom = links_elec.p_nom_min.sum()

    def _rule(m):
        lines_opt = sum(
            m.passive_branch_s_nom[index]
            for index in m.passive_branch_s_nom_index
        )

        links_opt = sum(m.link_p_nom[index] for index in links_index)

        return (lines_opt + links_opt) <= (
            lines_snom + links_pnom
        ) * self.args["extra_functionality"]["max_line_ext"]

    network.model.max_line_ext = Constraint(rule=_rule)


def _max_line_ext_nmp(self, network, snapshots):
    """
    Extra-functionality that limits overall line expansion to a multiple
    of existing capacity without pyomo.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """

    lines_snom = network.lines.s_nom.sum()

    links_elec = network.links[network.links.carrier == "DC"]
    links_index = links_elec.index
    links_pnom = links_elec.p_nom_min.sum()

    get_var(network, "Line", "s_nom")

    def _rule(m):
        lines_opt = sum(
            m.passive_branch_s_nom[index]
            for index in m.passive_branch_s_nom_index
        )

        links_opt = sum(m.link_p_nom[index] for index in links_index)

        return (lines_opt + links_opt) <= (
            lines_snom + links_pnom
        ) * self.args["extra_functionality"]["max_line_ext"]

    network.model.max_line_ext = Constraint(rule=_rule)


def _max_battery_expansion_germany(self, network, snapshots):
    """
    Set maximum expanded capacity of batteries in Germany.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """
    home_battery_capacity = network.storage_units[
        network.storage_units.carrier == "battery"
    ].p_nom_min.sum()

    batteries = network.storage_units[
        (network.storage_units.carrier == "battery")
        & (
            network.storage_units.bus.isin(
                network.buses[network.buses.country == "DE"].index
            )
        )
    ]

    def _rule_max(m):
        batteries_opt = sum(
            m.storage_p_nom[index] for index in batteries.index
        )
        return batteries_opt <= (1) * (
            self.args["extra_functionality"]["max_battery_expansion_germany"]
            + home_battery_capacity
        )

    network.model.max_battery_ext = Constraint(rule=_rule_max)


def _fixed_battery_expansion_germany(self, network, snapshots):
    """
    Define the overall expanded capacity of batteries in Germany.
    To avoid nummerical problems, a difference of 0.1% is allowed.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """
    home_battery_capacity = network.storage_units[
        network.storage_units.carrier == "battery"
    ].p_nom_min.sum()

    batteries = network.storage_units[
        (network.storage_units.carrier == "battery")
        & (
            network.storage_units.bus.isin(
                network.buses[network.buses.country == "DE"].index
            )
        )
    ]

    def _rule_min(m):
        batteries_opt = sum(
            m.storage_p_nom[index] for index in batteries.index
        )
        return (batteries_opt) >= (0.999) * (
            self.args["extra_functionality"]["fixed_battery_expansion_germany"]
            + home_battery_capacity
        )

    def _rule_max(m):
        batteries_opt = sum(
            m.storage_p_nom[index] for index in batteries.index
        )
        return batteries_opt <= (1.001) * (
            self.args["extra_functionality"]["fixed_battery_expansion_germany"]
            + home_battery_capacity
        )

    network.model.min_battery_ext = Constraint(rule=_rule_min)
    network.model.max_battery_ext = Constraint(rule=_rule_max)


def _min_renewable_share_nmp(self, network, snapshots):
    """
    Extra-functionality that limits the minimum share of renewable generation.
    Add key 'min_renewable_share' and minimal share in p.u. as float
    to args.extra_functionality.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """

    renewables = [
        "biomass",
        "central_biomass_CHP",
        "industrial_biomass_CHP",
        "solar",
        "solar_rooftop",
        "wind_offshore",
        "wind_onshore",
        "run_of_river",
        "other_renewable",
        "central_biomass_CHP_heat",
        "solar_thermal_collector",
        "geo_thermal",
    ]

    res = network.generators.index[network.generators.carrier.isin(renewables)]

    renew = (
        get_var(network, "Generator", "p")
        .loc[network.snapshots, res]
        .mul(network.snapshot_weightings.generators, axis=0)
    )
    total = get_var(network, "Generator", "p").mul(
        network.snapshot_weightings.generators, axis=0
    )

    renew_production = linexpr((1, renew)).sum().sum()
    total_production = (
        linexpr(
            (-self.args["extra_functionality"]["min_renewable_share"], total)
        )
        .sum()
        .sum()
    )

    expr = renew_production + total_production

    define_constraints(network, expr, ">=", 0, "Generator", "min_renew_share")


def _min_renewable_share(self, network, snapshots):
    """
    Extra-functionality that limits the minimum share of renewable generation.
    Add key 'min_renewable_share' and minimal share in p.u. as float
    to args.extra_functionality.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """

    renewables = [
        "biomass",
        "central_biomass_CHP",
        "industrial_biomass_CHP",
        "solar",
        "solar_rooftop",
        "wind_offshore",
        "wind_onshore",
        "run_of_river",
        "other_renewable",
        "CH4_biogas",
        "central_biomass_CHP_heat",
        "solar_thermal_collector",
        "geo_thermal",
    ]

    res = list(
        network.generators.index[network.generators.carrier.isin(renewables)]
    )

    total = list(network.generators.index)

    def _rule(m):
        renewable_production = sum(
            m.generator_p[gen, sn] * network.snapshot_weightings.generators[sn]
            for gen in res
            for sn in snapshots
        )
        total_production = sum(
            m.generator_p[gen, sn] * network.snapshot_weightings.generators[sn]
            for gen in total
            for sn in snapshots
        )

        return (
            renewable_production
            >= total_production
            * self.args["extra_functionality"]["min_renewable_share"]
        )

    network.model.min_renewable_share = Constraint(rule=_rule)


def _max_redispatch(self, network, snapshots):
    """
    Extra-functionality that limits the maximum usage of redispatch.
    Add key 'max_redispatch' and maximual amount of redispatch in MWh
    to args.extra_functionality.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """

    ramp_up = list(
        network.generators.index[
            network.generators.index.str.contains("ramp_up")
        ]
    )
    ramp_up_links = list(
        network.links.index[network.links.index.str.contains("ramp_up")]
    )

    def _rule(m):
        redispatch_gens = sum(
            m.generator_p[gen, sn] * network.snapshot_weightings.generators[sn]
            for gen in ramp_up
            for sn in snapshots
        )
        redispatch_links = sum(
            m.link_p[gen, sn]
            * network.links.loc[gen, "efficiency"]
            * network.snapshot_weightings.generators[sn]
            for gen in ramp_up_links
            for sn in snapshots
        )
        return (redispatch_gens + redispatch_links) <= self.args[
            "extra_functionality"
        ]["max_redispatch"]

    if len(ramp_up) > 0 or len(ramp_up_links) > 0:
        network.model.max_redispatch = Constraint(rule=_rule)
    else:
        print(
            """Constraint max_redispatch was not added,
              there are no redispatch generators or links."""
        )


def _max_redispatch_ramp_down(self, network, snapshots):
    """
    Extra-functionality that limits the maximum usage of redispatch.
    Add key 'max_redispatch' and maximual amount of redispatch in MWh
    to args.extra_functionality.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """

    ramp_up = list(
        network.generators.index[
            network.generators.index.str.contains("ramp_down")
        ]
    )
    ramp_up_links = list(
        network.links.index[network.links.index.str.contains("ramp_down")]
    )

    def _rule(m):
        redispatch_gens = sum(
            m.generator_p[gen, sn] * network.snapshot_weightings.generators[sn]
            for gen in ramp_up
            for sn in snapshots
        )
        redispatch_links = sum(
            m.link_p[gen, sn]
            * network.links.loc[gen, "efficiency"]
            * network.snapshot_weightings.generators[sn]
            for gen in ramp_up_links
            for sn in snapshots
        )
        return (redispatch_gens + redispatch_links) >= self.args[
            "extra_functionality"
        ]["max_redispatch_ramp_down"]

    if len(ramp_up) > 0 or len(ramp_up_links) > 0:
        network.model.max_redispatch_ramp_down = Constraint(rule=_rule)
    else:
        print(
            """Constraint max_redispatch was not added,
              there are no redispatch generators or links."""
        )


def _max_redispatch_linopy(self, network, snapshots):
    """
    Extra-functionality that limits the maximum usage of redispatch.
    Add key 'max_redispatch' and maximual amount of redispatch in MWh
    to args.extra_functionality.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """

    ramp_up = list(
        network.generators.index[
            (network.generators.index.str.contains("ramp_up"))
        ]
    )
    ramp_up_links = list(
        network.links.index[(network.links.index.str.contains("ramp_up"))]
    )

    if len(ramp_up) > 0 or len(ramp_up_links) > 0:
        define_constraints(
            network,
            get_var(network, "Generator", "p").loc[:, ramp_up].sum()
            + get_var(network, "Link", "p").loc[:, ramp_up_links].sum(),
            "<=",
            (self.args["extra_functionality"]["max_redispatch"]),
            "Global",
            "max_redispatch",
        )
    else:
        print(
            """Constraint max_redispatch_germany was not added,
              there are no redispatch generators or links."""
        )


def _max_redispatch_germany(self, network, snapshots):
    """
    Extra-functionality that limits the maximum usage of redispatch in Germany.
    Add key 'max_redispatch_germany' and maximual amount of redispatch in MWh
    in Germany to args.extra_functionality. The redispatch in other countries
    is not limited in this constraint.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """

    ramp_up = list(
        network.generators.index[
            (network.generators.index.str.contains("ramp_up"))
            & (
                network.generators.bus.isin(
                    network.buses.index[network.buses.country == "DE"]
                )
            )
        ]
    )
    ramp_up_links = list(
        network.links.index[
            (network.links.index.str.contains("ramp_up"))
            & (
                network.links.bus0.isin(
                    network.buses.index[network.buses.country == "DE"]
                )
            )
        ]
    )

    def _rule(m):
        redispatch_gens = sum(
            m.generator_p[gen, sn] * network.snapshot_weightings.generators[sn]
            for gen in ramp_up
            for sn in snapshots
        )
        redispatch_links = sum(
            m.link_p[gen, sn]
            * network.links.loc[gen, "efficiency"]
            * network.snapshot_weightings.generators[sn]
            for gen in ramp_up_links
            for sn in snapshots
        )
        return (redispatch_gens + redispatch_links) <= self.args[
            "extra_functionality"
        ]["max_redispatch_germany"]

    if len(ramp_up) > 0 or len(ramp_up_links) > 0:
        network.model.max_redispatch = Constraint(rule=_rule)
    else:
        print(
            """Constraint max_redispatch_germany was not added,
              there are no redispatch generators or links."""
        )


def _max_redispatch_germany_linopy(self, network, snapshots):
    """
    Extra-functionality that limits the maximum usage of redispatch in Germany.
    Add key 'max_redispatch_germany' and maximual amount of redispatch in MWh
    in Germany to args.extra_functionality. The redispatch in other countries
    is not limited in this constraint.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """

    ramp_up = list(
        network.generators.index[
            (network.generators.index.str.contains("ramp_up"))
            & (
                network.generators.bus.isin(
                    network.buses.index[network.buses.country == "DE"]
                )
            )
        ]
    )
    ramp_up_links = list(
        network.links.index[
            (network.links.index.str.contains("ramp_up"))
            & (
                network.links.bus0.isin(
                    network.buses.index[network.buses.country == "DE"]
                )
            )
        ]
    )

    if len(ramp_up) > 0 or len(ramp_up_links) > 0:
        define_constraints(
            network,
            get_var(network, "Generator", "p").loc[:, ramp_up].sum()
            + get_var(network, "Link", "p").loc[:, ramp_up_links].sum(),
            "<=",
            (self.args["extra_functionality"]["max_redispatch_germany"]),
            "Global",
            "max_redispatch_germany",
        )
    else:
        print(
            """Constraint max_redispatch_germany was not added,
              there are no redispatch generators or links."""
        )


def _cross_border_flow(self, network, snapshots):
    """
    Extra_functionality that limits overall AC crossborder flows from/to
    Germany. Add key 'cross_border_flow' and array with minimal and maximal
    import/export
    Example: {'cross_border_flow': [-x, y]} (with x Import, y Export)

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """

    (
        buses_de,
        buses_for,
        cb0,
        cb1,
        cb0_link,
        cb1_link,
    ) = _get_crossborder_components(network)

    export = pd.Series(
        data=self.args["extra_functionality"]["cross_border_flow"]
    )

    def _rule_min(m):
        cb_flow = (
            -sum(
                m.passive_branch_p["Line", line, sn]
                * network.snapshot_weightings.objective[sn]
                for line in cb0
                for sn in snapshots
            )
            + sum(
                m.passive_branch_p["Line", line, sn]
                * network.snapshot_weightings.objective[sn]
                for line in cb1
                for sn in snapshots
            )
            - sum(
                m.link_p[link, sn] * network.snapshot_weightings.objective[sn]
                for link in cb0_link
                for sn in snapshots
            )
            + sum(
                m.link_p[link, sn] * network.snapshot_weightings.objective[sn]
                for link in cb1_link
                for sn in snapshots
            )
        )
        return cb_flow >= export[0]

    def _rule_max(m):
        cb_flow = (
            -sum(
                m.passive_branch_p["Line", line, sn]
                * network.snapshot_weightings.objective[sn]
                for line in cb0
                for sn in snapshots
            )
            + sum(
                m.passive_branch_p["Line", line, sn]
                * network.snapshot_weightings.objective[sn]
                for line in cb1
                for sn in snapshots
            )
            - sum(
                m.link_p[link, sn] * network.snapshot_weightings.objective[sn]
                for link in cb0_link
                for sn in snapshots
            )
            + sum(
                m.link_p[link, sn] * network.snapshot_weightings.objective[sn]
                for link in cb1_link
                for sn in snapshots
            )
        )
        return cb_flow <= export[1]

    network.model.cross_border_flows_min = Constraint(rule=_rule_min)
    network.model.cross_border_flows_max = Constraint(rule=_rule_max)


def _cross_border_flow_nmp(self, network, snapshots):
    """
    Extra_functionality that limits overall crossborder flows from/to Germany.
    Add key 'cross_border_flow' and array with minimal and maximal
    import/export
    Example: {'cross_border_flow': [-x, y]} (with x Import, y Export)

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """

    (
        buses_de,
        buses_for,
        cb0,
        cb1,
        cb0_link,
        cb1_link,
    ) = _get_crossborder_components(network)

    export = pd.Series(
        data=self.args["extra_functionality"]["cross_border_flow"]
    )

    cb0_flow = (
        get_var(network, "Line", "s")
        .loc[snapshots, cb0]
        .mul(network.snapshot_weightings.objective, axis=0)
    )

    cb1_flow = (
        get_var(network, "Line", "s")
        .loc[snapshots, cb1]
        .mul(network.snapshot_weightings.objective, axis=0)
    )

    cb0_link_flow = (
        get_var(network, "Link", "p")
        .loc[snapshots, cb0_link]
        .mul(network.snapshot_weightings.objective, axis=0)
    )

    cb1_link_flow = (
        get_var(network, "Link", "p")
        .loc[snapshots, cb1_link]
        .mul(network.snapshot_weightings.objective, axis=0)
    )

    expr = (
        linexpr((-1, cb0_flow)).sum().sum()
        + linexpr((1, cb1_flow)).sum().sum()
        + linexpr((-1, cb0_link_flow)).sum().sum()
        + linexpr((1, cb1_link_flow)).sum().sum()
    )

    define_constraints(network, expr, ">=", export[0], "Line", "min_cb_flow")
    define_constraints(network, expr, "<=", export[1], "Line", "max_cb_flow")


def _cross_border_flow_per_country_nmp(self, network, snapshots):
    """
    Extra_functionality that limits AC crossborder flows for each given
    foreign country from/to Germany.
    Add key 'cross_border_flow_per_country' to args.extra_functionality and
    define dictionary of country keys and desired limitations of im/exports
    in MWh
    Example: {'cross_border_flow_per_country': {'DK':[-X, Y], 'FR':[0,0]}}

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """

    buses_de = network.buses.index[network.buses.country == "DE"]

    countries = network.buses.country.unique()

    export_per_country = pd.DataFrame(
        data=self.args["extra_functionality"]["cross_border_flow_per_country"]
    ).transpose()

    for cntr in export_per_country.index:
        if cntr in countries:
            (
                buses_de,
                buses_for,
                cb0,
                cb1,
                cb0_link,
                cb1_link,
            ) = _get_crossborder_components(network, cntr)

            cb0_flow = (
                get_var(network, "Line", "s")
                .loc[snapshots, cb0]
                .mul(network.snapshot_weightings.objective, axis=0)
            )

            cb1_flow = (
                get_var(network, "Line", "s")
                .loc[snapshots, cb1]
                .mul(network.snapshot_weightings.objective, axis=0)
            )

            cb0_link_flow = (
                get_var(network, "Link", "p")
                .loc[snapshots, cb0_link]
                .mul(network.snapshot_weightings.objective, axis=0)
            )

            cb1_link_flow = (
                get_var(network, "Link", "p")
                .loc[snapshots, cb1_link]
                .mul(network.snapshot_weightings.objective, axis=0)
            )

            expr = (
                linexpr((-1, cb0_flow)).sum().sum()
                + linexpr((1, cb1_flow)).sum().sum()
                + linexpr((-1, cb0_link_flow)).sum().sum()
                + linexpr((1, cb1_link_flow)).sum().sum()
            )

            define_constraints(
                network,
                expr,
                ">=",
                export_per_country[cntr][0],
                "Line",
                "min_cb_flow_" + cntr,
            )
            define_constraints(
                network,
                expr,
                "<=",
                export_per_country[cntr][1],
                "Line",
                "max_cb_flow_" + cntr,
            )


def _cross_border_flow_per_country(self, network, snapshots):
    """
    Extra_functionality that limits AC crossborder flows for each given
    foreign country from/to Germany.
    Add key 'cross_border_flow_per_country' to args.extra_functionality and
    define dictionary of country keys and desired limitations of im/exports
    in MWh
    Example: {'cross_border_flow_per_country': {'DK':[-X, Y], 'FR':[0,0]}}

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """

    buses_de = network.buses.index[network.buses.country == "DE"]

    countries = network.buses.country.unique()

    export_per_country = pd.DataFrame(
        data=self.args["extra_functionality"]["cross_border_flow_per_country"]
    ).transpose()

    for cntr in export_per_country.index:
        if cntr in countries:
            (
                buses_de,
                buses_for,
                cb0,
                cb1,
                cb0_link,
                cb1_link,
            ) = _get_crossborder_components(network, cntr)

            def _rule_min(m):
                cb_flow = (
                    -sum(
                        m.passive_branch_p["Line", line, sn]
                        * network.snapshot_weightings.objective[sn]
                        for line in cb0
                        for sn in snapshots
                    )
                    + sum(
                        m.passive_branch_p["Line", line, sn]
                        * network.snapshot_weightings.objective[sn]
                        for line in cb1
                        for sn in snapshots
                    )
                    - sum(
                        m.link_p[link, sn]
                        * network.snapshot_weightings.objective[sn]
                        for link in cb0_link
                        for sn in snapshots
                    )
                    + sum(
                        m.link_p[link, sn]
                        * network.snapshot_weightings.objective[sn]
                        for link in cb1_link
                        for sn in snapshots
                    )
                )
                return cb_flow >= export_per_country[0][cntr]

            setattr(
                network.model,
                "min_cross_border-" + cntr,
                Constraint(rule=_rule_min),
            )

            def _rule_max(m):
                cb_flow = (
                    -sum(
                        m.passive_branch_p["Line", line, sn]
                        * network.snapshot_weightings.objective[sn]
                        for line in cb0
                        for sn in snapshots
                    )
                    + sum(
                        m.passive_branch_p["Line", line, sn]
                        * network.snapshot_weightings.objective[sn]
                        for line in cb1
                        for sn in snapshots
                    )
                    - sum(
                        m.link_p[link, sn]
                        * network.snapshot_weightings.objective[sn]
                        for link in cb0_link
                        for sn in snapshots
                    )
                    + sum(
                        m.link_p[link, sn]
                        * network.snapshot_weightings.objective[sn]
                        for link in cb1_link
                        for sn in snapshots
                    )
                )
                return cb_flow <= export_per_country[1][cntr]

            setattr(
                network.model,
                "max_cross_border-" + cntr,
                Constraint(rule=_rule_max),
            )


def _generation_potential(network, carrier, cntr="all"):
    """
    Function that calculates the generation potential for chosen carriers and
    countries.

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    carrier : str
        Energy carrier of generator
    cntr : str, optional
        Country code or 'all'. The default is 'all'.

    Returns
    -------
    gens : pandas.base.Index
        Index of generators with given carrier in chosen country
    potential : float
        Gerneration potential in MW

    """

    if cntr == "all":
        gens = network.generators.index[network.generators.carrier == carrier]
    else:
        gens = network.generators.index[
            (network.generators.carrier == carrier)
            & (
                network.generators.bus.astype(str).isin(
                    network.buses.index[network.buses.country == cntr]
                )
            )
        ]
    if carrier in ["wind_onshore", "wind_offshore", "solar"]:
        potential = (
            (
                network.generators.p_nom[gens]
                * network.generators_t.p_max_pu[gens].mul(
                    network.snapshot_weightings.generators, axis=0
                )
            )
            .sum()
            .sum()
        )
    else:
        potential = (
            network.snapshot_weightings.generators.sum()
            * network.generators.p_nom[gens].sum()
        )
    return gens, potential


def _capacity_factor(self, network, snapshots):
    """
    Extra-functionality that limits overall dispatch of generators with chosen
    energy carrier.
    Add key 'capacity_factor' to args.extra_functionality and set limits in
    a dictonary as a fraction of generation potential.
    Example: 'capacity_factor': {'run_of_river': [0, 0.5], 'solar': [0.1, 1]}

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """
    arg = self.args["extra_functionality"]["capacity_factor"]
    carrier = arg.keys()

    for c in carrier:
        factor = arg[c]
        gens, potential = _generation_potential(network, c, cntr="all")

        def _rule_max(m):
            dispatch = sum(
                m.generator_p[gen, sn]
                * network.snapshot_weightings.generators[sn]
                for gen in gens
                for sn in snapshots
            )

            return dispatch <= factor[1] * potential

        setattr(network.model, "max_flh_" + c, Constraint(rule=_rule_max))

        def _rule_min(m):
            dispatch = sum(
                m.generator_p[gen, sn]
                * network.snapshot_weightings.generators[sn]
                for gen in gens
                for sn in snapshots
            )

            return dispatch >= factor[0] * potential

        setattr(network.model, "min_flh_" + c, Constraint(rule=_rule_min))


def _capacity_factor_nmp(self, network, snapshots):
    """
    Extra-functionality that limits overall dispatch of generators with chosen
    energy carrier.
    Add key 'capacity_factor' to args.extra_functionality and set limits in
    a dictonary as a fraction of generation potential.
    Example: 'capacity_factor': {'run_of_river': [0, 0.5], 'solar': [0.1, 1]}

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """
    arg = self.args["extra_functionality"]["capacity_factor"]
    carrier = arg.keys()

    for c in carrier:
        gens, potential = _generation_potential(network, c, cntr="all")

        generation = (
            get_var(network, "Generator", "p")
            .loc[snapshots, gens]
            .mul(network.snapshot_weightings.generators, axis=0)
        )

        define_constraints(
            network,
            linexpr((1, generation)).sum().sum(),
            ">=",
            arg[c][0] * potential,
            "Generator",
            "min_flh_" + c,
        )
        define_constraints(
            network,
            linexpr((1, generation)).sum().sum(),
            "<=",
            arg[c][1] * potential,
            "Generator",
            "max_flh_" + c,
        )


def _capacity_factor_per_cntr(self, network, snapshots):
    """
    Extra-functionality that limits dispatch of generators with chosen
    energy carrier located in the chosen country.
    Add key 'capacity_factor_per_cntr' to args.extra_functionality and set
    limits per carrier in a dictonary with country codes as keys.

    Example:
    'capacity_factor_per_cntr': {'DE':{'run_of_river': [0, 0.5],
                                       'wind_onshore': [0.1, 1]},
                                 'DK':{'wind_onshore':[0, 0.7]}}

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.
    """
    arg = self.args["extra_functionality"]["capacity_factor_per_cntr"]
    for cntr in arg.keys():
        carrier = arg[cntr].keys()
        for c in carrier:
            factor = arg[cntr][c]
            gens, potential = _generation_potential(network, c, cntr)

            if len(gens) > 0:

                def _rule_max(m):
                    dispatch = sum(
                        m.generator_p[gen, sn]
                        * network.snapshot_weightings.generators[sn]
                        for gen in gens
                        for sn in snapshots
                    )

                    return dispatch <= factor[1] * potential

                setattr(
                    network.model,
                    "max_flh_" + cntr + "_" + c,
                    Constraint(rule=_rule_max),
                )

                def _rule_min(m):
                    dispatch = sum(
                        m.generator_p[gen, sn]
                        * network.snapshot_weightings.generators[sn]
                        for gen in gens
                        for sn in snapshots
                    )

                    return dispatch >= factor[0] * potential

                setattr(
                    network.model,
                    "min_flh_" + cntr + "_" + c,
                    Constraint(rule=_rule_min),
                )

            else:
                print(
                    "Carrier "
                    + c
                    + " is not available in "
                    + cntr
                    + ". Skipping this constraint."
                )


def _capacity_factor_per_cntr_nmp(self, network, snapshots):
    """
    Extra-functionality that limits dispatch of generators with chosen
    energy carrier located in the chosen country.
    Add key 'capacity_factor_per_cntr' to args.extra_functionality and set
    limits per carrier in a dictonary with country codes as keys.

    Example:
    'capacity_factor_per_cntr': {'DE':{'run_of_river': [0, 0.5],
                                       'wind_onshore': [0.1, 1]},
                                 'DK':{'wind_onshore':[0, 0.7]}}

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.
    """
    arg = self.args["extra_functionality"]["capacity_factor_per_cntr"]
    for cntr in arg.keys():
        carrier = arg[cntr].keys()
        for c in carrier:
            gens, potential = _generation_potential(network, c, cntr)

            if len(gens) > 0:
                generation = (
                    get_var(network, "Generator", "p")
                    .loc[snapshots, gens]
                    .mul(network.snapshot_weightings.generators, axis=0)
                )

                define_constraints(
                    network,
                    linexpr((1, generation)).sum().sum(),
                    ">=",
                    arg[cntr][c][0] * potential,
                    "Generator",
                    "min_flh_" + c + "_" + cntr,
                )
                define_constraints(
                    network,
                    linexpr((1, generation)).sum().sum(),
                    "<=",
                    arg[cntr][c][1] * potential,
                    "Generator",
                    "max_flh_" + c + "_" + cntr,
                )

            else:
                print(
                    "Carrier "
                    + c
                    + " is not available in "
                    + cntr
                    + ". Skipping this constraint."
                )


def _capacity_factor_per_gen(self, network, snapshots):
    """
    Extra-functionality that limits dispatch for each generator with chosen
    energy carrier.
    Add key 'capacity_factor_per_gen' to args.extra_functionality and set
    limits in a dictonary as a fraction of generation potential.
    Example:
    'capacity_factor_per_gen': {'run_of_river': [0, 0.5], 'solar': [0.1, 1]}

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """
    arg = self.args["extra_functionality"]["capacity_factor_per_gen"]
    carrier = arg.keys()
    snapshots = network.snapshots
    for c in carrier:
        factor = arg[c]
        gens = network.generators.index[network.generators.carrier == c]
        for g in gens:
            if c in ["wind_onshore", "wind_offshore", "solar"]:
                potential = (
                    (
                        network.generators.p_nom[g]
                        * network.generators_t.p_max_pu[g].mul(
                            network.snapshot_weightings.generators, axis=0
                        )
                    )
                    .sum()
                    .sum()
                )
            else:
                potential = (
                    network.snapshot_weightings.generators.sum()
                    * network.generators.p_nom[g].sum()
                )

            def _rule_max(m):
                dispatch = sum(
                    m.generator_p[g, sn]
                    * network.snapshot_weightings.generators[sn]
                    for sn in snapshots
                )

                return dispatch <= factor[1] * potential

            setattr(network.model, "max_flh_" + g, Constraint(rule=_rule_max))

            def _rule_min(m):
                dispatch = sum(
                    m.generator_p[g, sn]
                    * network.snapshot_weightings.generators[sn]
                    for sn in snapshots
                )

                return dispatch >= factor[0] * potential

            setattr(network.model, "min_flh_" + g, Constraint(rule=_rule_min))


def _capacity_factor_per_gen_nmp(self, network, snapshots):
    """
    Extra-functionality that limits dispatch for each generator with chosen
    energy carrier.
    Add key 'capacity_factor_per_gen' to args.extra_functionality and set
    limits in a dictonary as a fraction of generation potential.
    Example:
    'capacity_factor_per_gen': {'run_of_river': [0, 0.5], 'solar': [0.1, 1]}

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.

    """
    arg = self.args["extra_functionality"]["capacity_factor_per_gen"]
    carrier = arg.keys()
    snapshots = network.snapshots
    for c in carrier:
        gens = network.generators.index[network.generators.carrier == c]
        for g in gens:
            if c in ["wind_onshore", "wind_offshore", "solar"]:
                potential = (
                    (
                        network.generators.p_nom[g]
                        * network.generators_t.p_max_pu[g].mul(
                            network.snapshot_weightings.generators, axis=0
                        )
                    )
                    .sum()
                    .sum()
                )
            else:
                potential = (
                    network.snapshot_weightings.generators.sum()
                    * network.generators.p_nom[g].sum()
                )

            generation = (
                get_var(network, "Generator", "p")
                .loc[snapshots, g]
                .mul(network.snapshot_weightings.generators, axis=0)
            )

            define_constraints(
                network,
                linexpr((1, generation)).sum(),
                ">=",
                arg[c][0] * potential,
                "Generator",
                "min_flh_" + g,
            )
            define_constraints(
                network,
                linexpr((1, generation)).sum(),
                "<=",
                arg[c][1] * potential,
                "Generator",
                "max_flh_" + g,
            )


def _capacity_factor_per_gen_cntr(self, network, snapshots):
    """
    Extra-functionality that limits dispatch of each generator with chosen
    energy carrier located in the chosen country.
    Add key 'capacity_factor_per_gen_cntr' to args.extra_functionality and set
    limits per carrier in a dictonary with country codes as keys.

    Example:
    'capacity_factor_per_gen_cntr': {'DE':{'run_of_river': [0, 0.5],
                                       'wind_onshore': [0.1, 1]},
                                 'DK':{'wind_onshore':[0, 0.7]}}

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.
    """
    arg = self.args["extra_functionality"]["capacity_factor_per_gen_cntr"]
    for cntr in arg.keys():
        carrier = arg[cntr].keys()
        snapshots = network.snapshots
        for c in carrier:
            factor = arg[cntr][c]
            gens = network.generators.index[
                (network.generators.carrier == c)
                & (
                    network.generators.bus.astype(str).isin(
                        network.buses.index[network.buses.country == cntr]
                    )
                )
            ]

            if len(gens) > 0:
                for g in gens:
                    if c in ["wind_onshore", "wind_offshore", "solar"]:
                        potential = (
                            (
                                network.generators.p_nom[g]
                                * network.generators_t.p_max_pu[g].mul(
                                    network.snapshot_weightings.generators,
                                    axis=0,
                                )
                            )
                            .sum()
                            .sum()
                        )
                    else:
                        potential = (
                            network.snapshot_weightings.generators.sum()
                            * network.generators.p_nom[g].sum()
                        )

                    def _rule_max(m):
                        dispatch = sum(
                            m.generator_p[g, sn]
                            * network.snapshot_weightings.generators[sn]
                            for sn in snapshots
                        )
                        return dispatch <= factor[1] * potential

                    setattr(
                        network.model,
                        "max_flh_" + cntr + "_" + g,
                        Constraint(rule=_rule_max),
                    )

                    def _rule_min(m):
                        dispatch = sum(
                            m.generator_p[g, sn]
                            * network.snapshot_weightings.generators[sn]
                            for sn in snapshots
                        )
                        return dispatch >= factor[0] * potential

                    setattr(
                        network.model,
                        "min_flh_" + cntr + "_" + g,
                        Constraint(rule=_rule_min),
                    )

            else:
                print(
                    "Carrier "
                    + c
                    + " is not available in "
                    + cntr
                    + ". Skipping this constraint."
                )


def _capacity_factor_per_gen_cntr_nmp(self, network, snapshots):
    """
    Extra-functionality that limits dispatch of each generator with chosen
    energy carrier located in the chosen country.
    Add key 'capacity_factor_per_gen_cntr' to args.extra_functionality and set
    limits per carrier in a dictonary with country codes as keys.

    Example:
    'capacity_factor_per_gen_cntr': {'DE':{'run_of_river': [0, 0.5],
                                       'wind_onshore': [0.1, 1]},
                                 'DK':{'wind_onshore':[0, 0.7]}}

    Parameters
    ----------
    network : :class:`pypsa.Network
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.
    """
    arg = self.args["extra_functionality"]["capacity_factor_per_gen_cntr"]
    for cntr in arg.keys():
        carrier = arg[cntr].keys()

        for c in carrier:
            gens = network.generators.index[
                (network.generators.carrier == c)
                & (
                    network.generators.bus.astype(str).isin(
                        network.buses.index[network.buses.country == cntr]
                    )
                )
            ]

            if len(gens) > 0:
                for g in gens:
                    if c in ["wind_onshore", "wind_offshore", "solar"]:
                        potential = (
                            (
                                network.generators.p_nom[g]
                                * network.generators_t.p_max_pu[g].mul(
                                    network.snapshot_weightings.generators,
                                    axis=0,
                                )
                            )
                            .sum()
                            .sum()
                        )
                    else:
                        potential = (
                            network.snapshot_weightings.generators.sum()
                            * network.generators.p_nom[g].sum()
                        )

                    generation = (
                        get_var(network, "Generator", "p")
                        .loc[snapshots, g]
                        .mul(network.snapshot_weightings.generators, axis=0)
                    )

                    define_constraints(
                        network,
                        linexpr((1, generation)).sum(),
                        ">=",
                        arg[cntr][c][0] * potential,
                        "Generator",
                        "min_flh_" + g,
                    )
                    define_constraints(
                        network,
                        linexpr((1, generation)).sum(),
                        "<=",
                        arg[cntr][c][1] * potential,
                        "Generator",
                        "max_flh_" + g,
                    )

            else:
                print(
                    "Carrier "
                    + c
                    + " is not available in "
                    + cntr
                    + ". Skipping this constraint."
                )


def read_max_gas_generation(self):
    """Return the values limiting the gas production in Germany

    Read max_gas_generation_overtheyear from
    scenario.egon_scenario_parameters if the table is available in the
    database and return the dictionnary containing the values needed
    for the constraints to limit the gas production in Germany,
    depending of the scenario.

    Returns
    -------
    arg: dict

    """
    scn_name = self.args["scn_name"]
    arg_def = {
        "eGon2035": {
            "CH4": 36000000,
            "biogas": 10000000,
        },  # [MWh] Netzentwicklungsplan Gas 2020–2030
        "eGon2035_lowflex": {
            "CH4": 36000000,
            "biogas": 10000000,
        },  # [MWh] Netzentwicklungsplan Gas 2020–2030
        "eGon100RE": {
            "biogas": 14450103
        },  # [MWh] Value from reference p-e-s run used in eGon-data
    }
    engine = db.connection(section=self.args["db"])
    try:
        sql = f"""
        SELECT gas_parameters
        FROM scenario.egon_scenario_parameters
        WHERE name = '{scn_name}';"""
        df = pd.read_sql(sql, engine)
        arg = df["max_gas_generation_overtheyear"]
    except:
        arg = arg_def[scn_name]

    return arg


def add_ch4_constraints_linopy(self, network, snapshots):
    """
    Add CH4 constraints for optimization with linopy

    Functionality that limits the dispatch of CH4 generators. In
    Germany, there is one limitation specific for biogas and one
    limitation specific for natural gas (natural gas only in eGon2035).
    Abroad, each generator has its own limitation contains in the
    column e_nom_max.

    Parameters
    ----------
    network : :class:`pypsa.Network`
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.
    """
    scn_name = self.args["scn_name"]
    n_snapshots = self.args["end_snapshot"] - self.args["start_snapshot"] + 1

    # Add constraint for Germany
    arg = read_max_gas_generation(self)
    gas_carrier = arg.keys()

    carrier_names = {
        "eGon2035": {"CH4": "CH4_NG", "biogas": "CH4_biogas"},
        "eGon2035_lowflex": {"CH4": "CH4_NG", "biogas": "CH4_biogas"},
        "eGon100RE": {"biogas": "CH4"},
    }
    for c in gas_carrier:
        gens = network.generators.index[
            (network.generators.carrier == carrier_names[scn_name][c])
            & (
                network.generators.bus.astype(str).isin(
                    network.buses.index[network.buses.country == "DE"]
                )
            )
        ]
        if not gens.empty:
            factor = arg[c]
            generation = (
                get_var(network, "Generator", "p").loc[snapshots, gens]
                * network.snapshot_weightings.generators
            )
            define_constraints(
                network,
                generation,
                "<=",
                factor * (n_snapshots / 8760),
                "Genertor",
                "max_flh_DE_" + c,
            )

    # Add contraints for neigbouring countries
    gen_abroad = network.generators[
        (network.generators.carrier == "CH4")
        & (
            network.generators.bus.astype(str).isin(
                network.buses.index[network.buses.country != "DE"]
            )
        )
        & (network.generators.e_nom_max != np.inf)
    ]
    for g in gen_abroad.index:
        factor = network.generators.e_nom_max[g]

        generation = (
            get_var(network, "Generator", "p").loc[snapshots, g]
            * network.snapshot_weightings.generators
        )
        define_constraints(
            network,
            generation,
            "<=",
            factor * (n_snapshots / 8760),
            "Genertor",
            "max_flh_abroad_" + str(g).replace(" ", "_"),
        )


def add_ch4_constraints(self, network, snapshots):
    """
    Add CH4 constraints for optimization with pyomo

    Functionality that limits the dispatch of CH4 generators. In
    Germany, there is one limitation specific for biogas and one
    limitation specific for natural gas (natural gas only in eGon2035).
    Abroad, each generator has its own limitation contains in the
    column e_nom_max.

    Parameters
    ----------
    network : :class:`pypsa.Network`
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.
    """
    scn_name = self.args["scn_name"]
    n_snapshots = self.args["end_snapshot"] - self.args["start_snapshot"] + 1

    # Add constraint for Germany
    arg = read_max_gas_generation(self)
    gas_carrier = arg.keys()

    carrier_names = {
        "eGon2035": {"CH4": "CH4_NG", "biogas": "CH4_biogas"},
        "eGon2035_lowflex": {"CH4": "CH4_NG", "biogas": "CH4_biogas"},
        "eGon100RE": {"biogas": "CH4"},
    }

    for c in gas_carrier:
        gens = network.generators.index[
            (network.generators.carrier == carrier_names[scn_name][c])
            & (
                network.generators.bus.astype(str).isin(
                    network.buses.index[network.buses.country == "DE"]
                )
            )
        ]
        if not gens.empty:
            factor = arg[c]

            def _rule_max(m):
                dispatch = sum(
                    m.generator_p[gen, sn]
                    * network.snapshot_weightings.generators[sn]
                    for gen in gens
                    for sn in snapshots
                )

                return dispatch <= factor * (n_snapshots / 8760)

            setattr(
                network.model, "max_flh_DE_" + c, Constraint(rule=_rule_max)
            )

    # Add contraints for neigbouring countries
    gen_abroad = network.generators[
        (network.generators.carrier == "CH4")
        & (
            network.generators.bus.astype(str).isin(
                network.buses.index[network.buses.country != "DE"]
            )
        )
        & (network.generators.e_nom_max != np.inf)
    ]
    for g in gen_abroad.index:
        factor = network.generators.e_nom_max[g]

        def _rule_max(m):
            dispatch = sum(
                m.generator_p[g, sn]
                * network.snapshot_weightings.generators[sn]
                for sn in snapshots
            )

            return dispatch <= factor * (n_snapshots / 8760)

        setattr(
            network.model,
            "max_flh_abroad_" + str(g).replace(" ", "_"),
            Constraint(rule=_rule_max),
        )


def add_ch4_constraints_nmp(self, network, snapshots):
    """
    Add CH4 constraints for optimization without pyomo

    Functionality that limits the dispatch of CH4 generators. In
    Germany, there is one limitation specific for biogas and one
    limitation specific for natural gas (natural gas only in eGon2035).
    Abroad, each generator has its own limitation contains in the
    column e_nom_max.

    Parameters
    ----------
    network : :class:`pypsa.Network`
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.
    """

    scn_name = self.args["scn_name"]
    n_snapshots = self.args["end_snapshot"] - self.args["start_snapshot"] + 1

    # Add constraint for Germany
    arg = read_max_gas_generation(self)
    gas_carrier = arg.keys()

    carrier_names = {
        "eGon2035": {"CH4": "CH4_NG", "biogas": "CH4_biogas"},
        "eGon100RE": {"biogas": "CH4"},
    }

    for c in gas_carrier:
        gens = network.generators.index[
            (network.generators.carrier == carrier_names[scn_name][c])
            & (
                network.generators.bus.astype(str).isin(
                    network.buses.index[network.buses.country == "DE"]
                )
            )
        ]
        if not gens.empty:
            factor = arg[c]

            generation = (
                get_var(network, "Generator", "p")
                .loc[snapshots, gens]
                .mul(network.snapshot_weightings.generators, axis=0)
            )

            define_constraints(
                network,
                linexpr((1, generation)).sum().sum(),
                "<=",
                factor * (n_snapshots / 8760),
                "Generator",
                "max_flh_DE_" + c,
            )

    # Add contraints for neigbouring countries
    gen_abroad = network.generators[
        (network.generators.carrier == "CH4")
        & (
            network.generators.bus.astype(str).isin(
                network.buses.index[network.buses.country != "DE"]
            )
        )
        & (network.generators.e_nom_max != np.inf)
    ]
    for g in gen_abroad.index:
        factor = network.generators.e_nom_max[g]

        generation = (
            get_var(network, "Generator", "p")
            .loc[snapshots, g]
            .mul(network.snapshot_weightings.generators, axis=0)
        )

        define_constraints(
            network,
            linexpr((1, generation)).sum(),
            "<=",
            factor * (n_snapshots / 8760),
            "Generator",
            "max_flh_DE_" + str(g).replace(" ", "_"),
        )


def snapshot_clustering_daily_bounds(self, network, snapshots):
    """
    Bound the storage level to 0.5 max_level every 24th hour.

    Parameters
    ----------
    network : :class:`pypsa.Network`
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps that will be constrained

    Returns
    -------
    None

    """
    sus = network.storage_units
    # take every first hour of the clustered days
    network.model.period_starts = network.snapshot_weightings.index[0::24]

    network.model.storages = sus.index

    print("Setting daily_bounds constraint")

    def day_rule(m, s, p):
        """
        Sets the soc of the every first hour to the
        soc of the last hour of the day (i.e. + 23 hours)
        """
        return (
            m.state_of_charge[s, p]
            == m.state_of_charge[s, p + pd.Timedelta(hours=23)]
        )

    network.model.period_bound = Constraint(
        network.model.storages, network.model.period_starts, rule=day_rule
    )


def snapshot_clustering_daily_bounds_nmp(self, network, snapshots):
    """
    Bound the storage level to 0.5 max_level every 24th hour.

    Parameters
    ----------
    network : :class:`pypsa.Network`
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps that will be constrained

    Returns
    -------
    None

    """

    c = "StorageUnit"

    period_starts = snapshots[0::24]
    period_ends = period_starts + pd.Timedelta(hours=23)

    eh = expand_series(
        network.snapshot_weightings.objective[period_ends],
        network.storage_units.index,
    )  # elapsed hours

    eff_stand = expand_series(1 - network.df(c).standing_loss, period_ends).T
    eff_dispatch = expand_series(
        network.df(c).efficiency_dispatch, period_ends
    ).T
    eff_store = expand_series(network.df(c).efficiency_store, period_ends).T

    soc = get_var(network, c, "state_of_charge").loc[period_ends, :]

    soc_peroid_start = get_var(network, c, "state_of_charge").loc[
        period_starts
    ]

    coeff_var = [
        (-1, soc),
        (
            -1 / eff_dispatch * eh,
            get_var(network, c, "p_dispatch").loc[period_ends, :],
        ),
        (eff_store * eh, get_var(network, c, "p_store").loc[period_ends, :]),
    ]

    lhs, *axes = linexpr(*coeff_var, return_axes=True)

    def masked_term(coeff, var, cols):
        return (
            linexpr((coeff[cols], var[cols]))
            .reindex(index=axes[0], columns=axes[1], fill_value="")
            .values
        )

    lhs += masked_term(
        eff_stand, soc_peroid_start, network.storage_units.index
    )

    rhs = -get_as_dense(network, c, "inflow", period_ends).mul(eh)

    define_constraints(network, lhs, "==", rhs, "daily_bounds")


def snapshot_clustering_seasonal_storage(
    self, network, snapshots, simplified=False
):
    """
    Depicts intertemporal dependencies of storage units and stores when using
    snapshot clustering to typical periods for temporal complexity reduction.

    According to:
        L. Kotzur et al: 'Time series aggregation for energy
        system design:
        Modeling seasonal storage', 2018

    Parameters
    ----------
    network : :class:`pypsa.Network`
        Overall container of PyPSA
    snapshots : list
        A list of datetime objects representing the timestamps of the snapshots
        to be clustered.
    simplified : bool, optional
        A flag indicating whether to use a simplified version of the model that
        does not include intra-temporal constraints and variables.

    Returns
    -------
    None
    """

    sus = network.storage_units
    sto = network.stores

    if self.args["snapshot_clustering"]["how"] == "weekly":
        network.model.period_starts = network.snapshot_weightings.index[0::168]
    elif self.args["snapshot_clustering"]["how"] == "monthly":
        network.model.period_starts = network.snapshot_weightings.index[0::720]
    else:
        network.model.period_starts = network.snapshot_weightings.index[0::24]

    network.model.storages = sus.index
    network.model.stores = sto.index

    candidates = network.cluster.index.get_level_values(0).unique()

    # create set for inter-temp constraints and variables
    network.model.candidates = po.Set(initialize=candidates, ordered=True)

    if not simplified:
        # create intra soc variable for each storage/store and each hour
        network.model.state_of_charge_intra = po.Var(
            sus.index, network.snapshots
        )
        network.model.state_of_charge_intra_store = po.Var(
            sto.index, network.snapshots
        )

    else:
        network.model.state_of_charge_intra_max = po.Var(
            sus.index, network.model.candidates
        )
        network.model.state_of_charge_intra_min = po.Var(
            sus.index, network.model.candidates
        )
        network.model.state_of_charge_intra_store_max = po.Var(
            sto.index, network.model.candidates
        )
        network.model.state_of_charge_intra_store_min = po.Var(
            sto.index, network.model.candidates
        )

        # create intra soc variable for each storage and each hour
        network.model.state_of_charge_intra = po.Var(
            sus.index, network.snapshots
        )
        network.model.state_of_charge_intra_store = po.Var(
            sto.index, network.snapshots
        )

        def intra_max(model, st, h):
            cand = network.cluster_ts["Candidate_day"][h]
            return (
                model.state_of_charge_intra_max[st, cand]
                >= model.state_of_charge_intra[st, h]
            )

        network.model.soc_intra_max = Constraint(
            network.model.storages, network.snapshots, rule=intra_max
        )

        def intra_min(model, st, h):
            cand = network.cluster_ts["Candidate_day"][h]
            return (
                model.state_of_charge_intra_min[st, cand]
                <= model.state_of_charge_intra[st, h]
            )

        network.model.soc_intra_min = Constraint(
            network.model.storages, network.snapshots, rule=intra_min
        )

        def intra_max_store(model, st, h):
            cand = network.cluster_ts["Candidate_day"][h]
            return (
                model.state_of_charge_intra_store_max[st, cand]
                >= model.state_of_charge_intra_store[st, h]
            )

        network.model.soc_intra_store_max = Constraint(
            network.model.stores, network.snapshots, rule=intra_max_store
        )

        def intra_min_store(model, st, h):
            cand = network.cluster_ts["Candidate_day"][h]
            return (
                model.state_of_charge_intra_store_min[st, cand]
                <= model.state_of_charge_intra_store[st, h]
            )

        network.model.soc_intra_store_min = Constraint(
            network.model.stores, network.snapshots, rule=intra_min_store
        )

    def intra_soc_rule(m, s, h):
        """
        Sets soc_inter of first hour of every day to 0. Other hours
        are set by technical coherences of storage units

        According to:
        L. Kotzur et al: 'Time series aggregation for energy
        system design:
        Modeling seasonal storage', 2018, equation no. 18
        """

        if (
            self.args["snapshot_clustering"]["how"] == "weekly"
            and h in network.snapshot_weightings[0::168].index
        ):
            expr = m.state_of_charge_intra[s, h] == 0
        elif (
            self.args["snapshot_clustering"]["how"] == "monthly"
            and h in network.snapshot_weightings[0::720].index
        ):
            expr = m.state_of_charge_intra[s, h] == 0
        elif (
            self.args["snapshot_clustering"]["how"] == "daily" and h.hour == 0
        ):
            expr = m.state_of_charge_intra[s, h] == 0
        else:
            expr = m.state_of_charge_intra[s, h] == m.state_of_charge_intra[
                s, h - pd.DateOffset(hours=1)
            ] * (1 - network.storage_units.at[s, "standing_loss"]) - (
                m.storage_p_dispatch[s, h - pd.DateOffset(hours=1)]
                / network.storage_units.at[s, "efficiency_dispatch"]
                - network.storage_units.at[s, "efficiency_store"]
                * m.storage_p_store[s, h - pd.DateOffset(hours=1)]
            )
        return expr

    def intra_soc_rule_store(m, s, h):
        if (
            self.args["snapshot_clustering"]["how"] == "weekly"
            and h in network.snapshot_weightings[0::168].index
        ):
            expr = m.state_of_charge_intra_store[s, h] == 0
        elif (
            self.args["snapshot_clustering"]["how"] == "monthly"
            and h in network.snapshot_weightings[0::720].index
        ):
            expr = m.state_of_charge_intra_store[s, h] == 0
        elif (
            self.args["snapshot_clustering"]["how"] == "daily" and h.hour == 0
        ):
            expr = m.state_of_charge_intra_store[s, h] == 0
        else:
            expr = (
                m.state_of_charge_intra_store[s, h]
                == m.state_of_charge_intra_store[s, h - pd.DateOffset(hours=1)]
                * (1 - network.stores.at[s, "standing_loss"])
                + m.store_p[s, h - pd.DateOffset(hours=1)]
            )
        return expr

    network.model.soc_intra = po.Constraint(
        network.model.storages, network.snapshots, rule=intra_soc_rule
    )
    network.model.soc_intra_store = po.Constraint(
        network.model.stores, network.snapshots, rule=intra_soc_rule_store
    )

    # create inter soc variable for each storage/store and each candidate
    network.model.state_of_charge_inter = po.Var(
        sus.index, network.model.candidates, within=po.NonNegativeReals
    )
    network.model.state_of_charge_inter_store = po.Var(
        sto.index, network.model.candidates, within=po.NonNegativeReals
    )

    def inter_storage_soc_rule(m, s, i):
        """
        Define the state_of_charge_inter as the state_of_charge_inter of
        the day before minus the storage losses plus the state_of_charge_intra
        of one hour after the last hour of the representative day.
        For the last reperesentive day, the soc_inter is the same as
        the first day due to cyclic soc condition

        According to:
        L. Kotzur et al: 'Time series aggregation for energy system design:
        Modeling seasonal storage', 2018, equation no. 19
        """
        if i == network.model.candidates.at(-1):
            last_hour = network.cluster["last_hour_RepresentativeDay"][i]
            expr = po.Constraint.Skip
        else:
            last_hour = network.cluster["last_hour_RepresentativeDay"][i]
            if self.args["snapshot_clustering"]["how"] == "weekly":
                hrs = 168
            elif self.args["snapshot_clustering"]["how"] == "monthly":
                hrs = 720
            else:
                hrs = 24
            expr = m.state_of_charge_inter[
                s, i + 1
            ] == m.state_of_charge_inter[s, i] * (
                1 - network.storage_units.at[s, "standing_loss"]
            ) ** hrs + m.state_of_charge_intra[
                s, last_hour
            ] * (
                1 - network.storage_units.at[s, "standing_loss"]
            ) - (
                m.storage_p_dispatch[s, last_hour]
                / network.storage_units.at[s, "efficiency_dispatch"]
                - network.storage_units.at[s, "efficiency_store"]
                * m.storage_p_store[s, last_hour]
            )
        return expr

    def inter_store_soc_rule(m, s, i):
        if i == network.model.candidates.at(-1):
            last_hour = network.cluster["last_hour_RepresentativeDay"][i]
            expr = po.Constraint.Skip
        else:
            last_hour = network.cluster["last_hour_RepresentativeDay"][i]
            if self.args["snapshot_clustering"]["how"] == "weekly":
                hrs = 168
            elif self.args["snapshot_clustering"]["how"] == "monthly":
                hrs = 720
            else:
                hrs = 24
            expr = (
                m.state_of_charge_inter_store[s, i + 1]
                == m.state_of_charge_inter_store[s, i]
                * (1 - network.stores.at[s, "standing_loss"]) ** hrs
                + m.state_of_charge_intra_store[s, last_hour]
                * (1 - network.stores.at[s, "standing_loss"])
                + m.store_p[s, last_hour]
            )
        return expr

    network.model.inter_storage_soc_constraint = po.Constraint(
        sus.index, network.model.candidates, rule=inter_storage_soc_rule
    )
    network.model.inter_store_soc_constraint = po.Constraint(
        sto.index, network.model.candidates, rule=inter_store_soc_rule
    )

    # new definition of the state_of_charge used in pypsa

    network.model.del_component("state_of_charge_constraint")
    network.model.del_component("state_of_charge_constraint_index")
    network.model.del_component("state_of_charge_constraint_index_0")
    network.model.del_component("state_of_charge_constraint_index_1")

    network.model.del_component("store_constraint")
    network.model.del_component("store_constraint_index")
    network.model.del_component("store_constraint_index_0")
    network.model.del_component("store_constraint_index_1")

    def total_state_of_charge(m, s, h):
        """
        Define the state_of_charge as the sum of state_of_charge_inter
        and state_of_charge_intra

        According to:
        L. Kotzur et al: 'Time series aggregation for energy system design:
        Modeling seasonal storage', 2018
        """

        return (
            m.state_of_charge[s, h]
            == m.state_of_charge_intra[s, h]
            + m.state_of_charge_inter[
                s, network.cluster_ts["Candidate_day"][h]
            ]
        )

    def total_state_of_charge_store(m, s, h):
        return (
            m.store_e[s, h]
            == m.state_of_charge_intra_store[s, h]
            + m.state_of_charge_inter_store[
                s, network.cluster_ts["Candidate_day"][h]
            ]
        )

    network.model.total_storage_constraint = po.Constraint(
        sus.index, network.snapshots, rule=total_state_of_charge
    )
    network.model.total_store_constraint = po.Constraint(
        sto.index, network.snapshots, rule=total_state_of_charge_store
    )

    network.model.del_component("state_of_charge_lower")
    network.model.del_component("state_of_charge_lower_index")
    network.model.del_component("state_of_charge_lower_index_0")
    network.model.del_component("state_of_charge_lower_index_1")

    network.model.del_component("store_e_lower")
    network.model.del_component("store_e_lower_index")
    network.model.del_component("store_e_lower_index_0")
    network.model.del_component("store_e_lower_index_1")

    def state_of_charge_lower(m, s, h):
        """
        Define the state_of_charge as the sum of state_of_charge_inter
        and state_of_charge_intra

        According to:
        L. Kotzur et al: 'Time series aggregation for energy system design:
        Modeling seasonal storage', 2018
        """

        # Choose datetime of representive day
        if self.args["snapshot_clustering"]["how"] == "weekly":
            hrs = 168
            candidate = network.cluster_ts["Candidate_day"][h]
            last_hour = network.cluster.loc[candidate][
                "last_hour_RepresentativeDay"
            ]
            first_hour = last_hour - pd.DateOffset(hours=167)
            period_start = network.cluster_ts.index[0::168][candidate - 1]
            delta_t = h - period_start
            intra_hour = first_hour + delta_t
        elif self.args["snapshot_clustering"]["how"] == "monthly":
            hrs = 720
            candidate = network.cluster_ts["Candidate_day"][h]
            last_hour = network.cluster.loc[candidate][
                "last_hour_RepresentativeDay"
            ]
            first_hour = last_hour - pd.DateOffset(hours=719)
            period_start = network.cluster_ts.index[0::720][candidate - 1]
            delta_t = h - period_start
            intra_hour = first_hour + delta_t
        else:
            hrs = 24
            date = str(
                network.snapshots[
                    network.snapshots.dayofyear - 1
                    == network.cluster["RepresentativeDay"][h.dayofyear]
                ][0]
            ).split(" ")[0]
            hour = str(h).split(" ")[1]
            intra_hour = pd.to_datetime(date + " " + hour)

        return (
            m.state_of_charge_intra[s, intra_hour]
            + m.state_of_charge_inter[
                s, network.cluster_ts["Candidate_day"][h]
            ]
            * (1 - network.storage_units.at[s, "standing_loss"]) ** hrs
            >= 0
        )

    def state_of_charge_lower_store(m, s, h):
        # Choose datetime of representive day
        if self.args["snapshot_clustering"]["how"] == "weekly":
            hrs = 168
            candidate = network.cluster_ts["Candidate_day"][h]
            last_hour = network.cluster.loc[candidate][
                "last_hour_RepresentativeDay"
            ]
            first_hour = last_hour - pd.DateOffset(hours=167)
            period_start = network.cluster_ts.index[0::168][candidate - 1]
            delta_t = h - period_start
            intra_hour = first_hour + delta_t
        elif self.args["snapshot_clustering"]["how"] == "monthly":
            hrs = 720
            candidate = network.cluster_ts["Candidate_day"][h]
            last_hour = network.cluster.loc[candidate][
                "last_hour_RepresentativeDay"
            ]
            first_hour = last_hour - pd.DateOffset(hours=719)
            period_start = network.cluster_ts.index[0::720][candidate - 1]
            delta_t = h - period_start
            intra_hour = first_hour + delta_t
        else:
            hrs = 24
            date = str(
                network.snapshots[
                    network.snapshots.dayofyear - 1
                    == network.cluster["RepresentativeDay"][h.dayofyear]
                ][0]
            ).split(" ")[0]
            hour = str(h).split(" ")[1]
            intra_hour = pd.to_datetime(date + " " + hour)

        if "DSM" in s:
            low = (
                network.stores.e_nom[s]
                * network.stores_t.e_min_pu.at[intra_hour, s]
            )
        else:
            low = 0

        return (
            m.state_of_charge_intra_store[s, intra_hour]
            + m.state_of_charge_inter_store[
                s, network.cluster_ts["Candidate_day"][h]
            ]
            * (1 - network.stores.at[s, "standing_loss"]) ** hrs
            >= low
        )

    def state_of_charge_lower_simplified(m, s, h):
        """
        Define the state_of_charge as the sum of state_of_charge_inter
        and state_of_charge_intra

        According to:
        L. Kotzur et al: 'Time series aggregation for energy system design:
        Modeling seasonal storage', 2018
        """
        if self.args["snapshot_clustering"]["how"] == "weekly":
            hrs = 168
        elif self.args["snapshot_clustering"]["how"] == "monthly":
            hrs = 720
        else:
            hrs = 24

        return (
            m.state_of_charge_intra_min[
                s, network.cluster_ts["Candidate_day"][h]
            ]
            + m.state_of_charge_inter[
                s, network.cluster_ts["Candidate_day"][h]
            ]
            * (1 - network.storage_units.at[s, "standing_loss"]) ** hrs
            >= 0
        )

    def state_of_charge_lower_store_simplified(m, s, h):
        if self.args["snapshot_clustering"]["how"] == "weekly":
            hrs = 168
        elif self.args["snapshot_clustering"]["how"] == "monthly":
            hrs = 720
        else:
            hrs = 24

        if "DSM" in s:
            if self.args["snapshot_clustering"]["how"] == "weekly":
                candidate = network.cluster_ts["Candidate_day"][h]
                last_hour = network.cluster.loc[candidate][
                    "last_hour_RepresentativeDay"
                ]
                first_hour = last_hour - pd.DateOffset(hours=167)
                period_start = network.cluster_ts.index[0::168][candidate - 1]
                delta_t = h - period_start
                intra_hour = first_hour + delta_t
            elif self.args["snapshot_clustering"]["how"] == "monthly":
                candidate = network.cluster_ts["Candidate_day"][h]
                last_hour = network.cluster.loc[candidate][
                    "last_hour_RepresentativeDay"
                ]
                first_hour = last_hour - pd.DateOffset(hours=719)
                period_start = network.cluster_ts.index[0::720][candidate - 1]
                delta_t = h - period_start
                intra_hour = first_hour + delta_t
            else:
                date = str(
                    network.snapshots[
                        network.snapshots.dayofyear - 1
                        == network.cluster["RepresentativeDay"][h.dayofyear]
                    ][0]
                ).split(" ")[0]
                hour = str(h).split(" ")[1]
                intra_hour = pd.to_datetime(date + " " + hour)
            low = (
                network.stores.e_nom[s]
                * network.stores_t.e_min_pu.at[intra_hour, s]
            )
        else:
            low = 0

        return (
            m.state_of_charge_intra_store_min[
                s, network.cluster_ts["Candidate_day"][h]
            ]
            + m.state_of_charge_inter_store[
                s, network.cluster_ts["Candidate_day"][h]
            ]
            * (1 - network.stores.at[s, "standing_loss"]) ** hrs
            >= low
        )

    if simplified:
        network.model.state_of_charge_lower = po.Constraint(
            sus.index,
            network.cluster_ts.index,
            rule=state_of_charge_lower_simplified,
        )
        network.model.state_of_charge_lower_store = po.Constraint(
            sto.index,
            network.cluster_ts.index,
            rule=state_of_charge_lower_store_simplified,
        )

    else:
        network.model.state_of_charge_lower = po.Constraint(
            sus.index, network.cluster_ts.index, rule=state_of_charge_lower
        )
        network.model.state_of_charge_lower_store = po.Constraint(
            sto.index,
            network.cluster_ts.index,
            rule=state_of_charge_lower_store,
        )

    network.model.del_component("state_of_charge_upper")
    network.model.del_component("state_of_charge_upper_index")
    network.model.del_component("state_of_charge_upper_index_0")
    network.model.del_component("state_of_charge_upper_index_1")

    network.model.del_component("store_e_upper")
    network.model.del_component("store_e_upper_index")
    network.model.del_component("store_e_upper_index_0")
    network.model.del_component("store_e_upper_index_1")

    def state_of_charge_upper(m, s, h):
        # Choose datetime of representive day
        if self.args["snapshot_clustering"]["how"] == "weekly":
            hrs = 168
            candidate = network.cluster_ts["Candidate_day"][h]
            last_hour = network.cluster.loc[candidate][
                "last_hour_RepresentativeDay"
            ]
            first_hour = last_hour - pd.DateOffset(hours=167)
            period_start = network.cluster_ts.index[0::168][candidate - 1]
            delta_t = h - period_start
            intra_hour = first_hour + delta_t
        elif self.args["snapshot_clustering"]["how"] == "monthly":
            hrs = 720
            candidate = network.cluster_ts["Candidate_day"][h]
            last_hour = network.cluster.loc[candidate][
                "last_hour_RepresentativeDay"
            ]
            first_hour = last_hour - pd.DateOffset(hours=719)
            period_start = network.cluster_ts.index[0::720][candidate - 1]
            delta_t = h - period_start
            intra_hour = first_hour + delta_t
        else:
            hrs = 24  # 0
            date = str(
                network.snapshots[
                    network.snapshots.dayofyear - 1
                    == network.cluster["RepresentativeDay"][h.dayofyear]
                ][0]
            ).split(" ")[0]
            hour = str(h).split(" ")[1]
            intra_hour = pd.to_datetime(date + " " + hour)

        if network.storage_units.p_nom_extendable[s]:
            p_nom = m.storage_p_nom[s]
        else:
            p_nom = network.storage_units.p_nom[s]

        return (
            m.state_of_charge_intra[s, intra_hour]
            + m.state_of_charge_inter[
                s, network.cluster_ts["Candidate_day"][h]
            ]
            * (1 - network.storage_units.at[s, "standing_loss"]) ** hrs
            <= p_nom * network.storage_units.at[s, "max_hours"]
        )

    def state_of_charge_upper_store(m, s, h):
        # Choose datetime of representive day
        if self.args["snapshot_clustering"]["how"] == "weekly":
            hrs = 168
            candidate = network.cluster_ts["Candidate_day"][h]
            last_hour = network.cluster.loc[candidate][
                "last_hour_RepresentativeDay"
            ]
            first_hour = last_hour - pd.DateOffset(hours=167)
            period_start = network.cluster_ts.index[0::168][candidate - 1]
            delta_t = h - period_start
            intra_hour = first_hour + delta_t
        elif self.args["snapshot_clustering"]["how"] == "monthly":
            hrs = 720
            candidate = network.cluster_ts["Candidate_day"][h]
            last_hour = network.cluster.loc[candidate][
                "last_hour_RepresentativeDay"
            ]
            first_hour = last_hour - pd.DateOffset(hours=719)
            period_start = network.cluster_ts.index[0::720][candidate - 1]
            delta_t = h - period_start
            intra_hour = first_hour + delta_t
        else:
            hrs = 24  # 0
            date = str(
                network.snapshots[
                    network.snapshots.dayofyear - 1
                    == network.cluster["RepresentativeDay"][h.dayofyear]
                ][0]
            ).split(" ")[0]
            hour = str(h).split(" ")[1]
            intra_hour = pd.to_datetime(date + " " + hour)

        if network.stores.e_nom_extendable[s]:
            e_nom = m.store_e_nom[s]
        else:
            if "DSM" in s:
                e_nom = (
                    network.stores.e_nom[s]
                    * network.stores_t.e_max_pu.at[intra_hour, s]
                )
            else:
                e_nom = network.stores.e_nom[s]

        return (
            m.state_of_charge_intra_store[s, intra_hour]
            + m.state_of_charge_inter_store[
                s, network.cluster_ts["Candidate_day"][h]
            ]
            * (1 - network.stores.at[s, "standing_loss"]) ** hrs
            <= e_nom
        )

    def state_of_charge_upper_simplified(m, s, h):
        if self.args["snapshot_clustering"]["how"] == "weekly":
            hrs = 168
        elif self.args["snapshot_clustering"]["how"] == "monthly":
            hrs = 720
        else:
            hrs = 24  # 0

        if network.storage_units.p_nom_extendable[s]:
            p_nom = m.storage_p_nom[s]
        else:
            p_nom = network.storage_units.p_nom[s]

        return (
            m.state_of_charge_intra_max[
                s, network.cluster_ts["Candidate_day"][h]
            ]
            + m.state_of_charge_inter[
                s, network.cluster_ts["Candidate_day"][h]
            ]
            * (1 - network.storage_units.at[s, "standing_loss"]) ** hrs
            <= p_nom * network.storage_units.at[s, "max_hours"]
        )

    def state_of_charge_upper_store_simplified(m, s, h):
        if self.args["snapshot_clustering"]["how"] == "weekly":
            hrs = 168
        elif self.args["snapshot_clustering"]["how"] == "monthly":
            hrs = 720
        else:
            hrs = 24  # 0

        if network.stores.e_nom_extendable[s]:
            e_nom = m.store_e_nom[s]
        else:
            if "DSM" in s:
                if self.args["snapshot_clustering"]["how"] == "weekly":
                    candidate = network.cluster_ts["Candidate_day"][h]
                    last_hour = network.cluster.loc[candidate][
                        "last_hour_RepresentativeDay"
                    ]
                    first_hour = last_hour - pd.DateOffset(hours=167)
                    period_start = network.cluster_ts.index[0::168][
                        candidate - 1
                    ]
                    delta_t = h - period_start
                    intra_hour = first_hour + delta_t

                elif self.args["snapshot_clustering"]["how"] == "monthly":
                    candidate = network.cluster_ts["Candidate_day"][h]
                    last_hour = network.cluster.loc[candidate][
                        "last_hour_RepresentativeDay"
                    ]
                    first_hour = last_hour - pd.DateOffset(hours=719)
                    period_start = network.cluster_ts.index[0::720][
                        candidate - 1
                    ]
                    delta_t = h - period_start
                    intra_hour = first_hour + delta_t

                else:
                    date = str(
                        network.snapshots[
                            network.snapshots.dayofyear - 1
                            == network.cluster["RepresentativeDay"][
                                h.dayofyear
                            ]
                        ][0]
                    ).split(" ")[0]
                    hour = str(h).split(" ")[1]
                    intra_hour = pd.to_datetime(date + " " + hour)
                e_nom = (
                    network.stores.e_nom[s]
                    * network.stores_t.e_max_pu.at[intra_hour, s]
                )

            else:
                e_nom = network.stores.e_nom[s]

        return (
            m.state_of_charge_intra_store_max[
                s, network.cluster_ts["Candidate_day"][h]
            ]
            + m.state_of_charge_inter_store[
                s, network.cluster_ts["Candidate_day"][h]
            ]
            * (1 - network.stores.at[s, "standing_loss"]) ** hrs
            <= e_nom
        )

    if simplified:
        network.model.state_of_charge_upper = po.Constraint(
            sus.index,
            network.cluster_ts.index,
            rule=state_of_charge_upper_simplified,
        )
        network.model.state_of_charge_upper_store = po.Constraint(
            sto.index,
            network.cluster_ts.index,
            rule=state_of_charge_upper_store_simplified,
        )

    else:
        network.model.state_of_charge_upper = po.Constraint(
            sus.index, network.cluster_ts.index, rule=state_of_charge_upper
        )
        network.model.state_of_charge_upper_store = po.Constraint(
            sto.index,
            network.cluster_ts.index,
            rule=state_of_charge_upper_store,
        )

    def cyclic_state_of_charge(m, s):
        """
        Defines cyclic condition like pypsas 'state_of_charge_contraint'.
        There are small differences to original results.
        """
        last_day = network.cluster.index[-1]
        last_calc_hour = network.cluster["last_hour_RepresentativeDay"][
            last_day
        ]
        last_inter = m.state_of_charge_inter[s, last_day]
        last_intra = m.state_of_charge_intra[s, last_calc_hour]
        first_day = network.cluster.index[0]

        if self.args["snapshot_clustering"]["how"] == "weekly":
            hrs = 167
        elif self.args["snapshot_clustering"]["how"] == "monthly":
            hrs = 719
        else:
            hrs = 23

        first_calc_hour = network.cluster["last_hour_RepresentativeDay"][
            first_day
        ] - pd.DateOffset(hours=hrs)
        first_inter = m.state_of_charge_inter[s, first_day]
        first_intra = m.state_of_charge_intra[s, first_calc_hour]

        return first_intra + first_inter == (
            (last_intra + last_inter)
            * (1 - network.storage_units.at[s, "standing_loss"])
            - (
                m.storage_p_dispatch[s, last_calc_hour]
                / network.storage_units.at[s, "efficiency_dispatch"]
                - m.storage_p_store[s, last_calc_hour]
                * network.storage_units.at[s, "efficiency_store"]
            )
        )

    def cyclic_state_of_charge_store(m, s):
        last_day = network.cluster.index[-1]
        last_calc_hour = network.cluster["last_hour_RepresentativeDay"][
            last_day
        ]
        last_inter = m.state_of_charge_inter_store[s, last_day]
        last_intra = m.state_of_charge_intra_store[s, last_calc_hour]
        first_day = network.cluster.index[0]

        if self.args["snapshot_clustering"]["how"] == "weekly":
            hrs = 167
        elif self.args["snapshot_clustering"]["how"] == "monthly":
            hrs = 719
        else:
            hrs = 23

        first_calc_hour = network.cluster["last_hour_RepresentativeDay"][
            first_day
        ] - pd.DateOffset(hours=hrs)
        first_inter = m.state_of_charge_inter_store[s, first_day]
        first_intra = m.state_of_charge_intra_store[s, first_calc_hour]

        expr = first_intra + first_inter == (
            (last_intra + last_inter)
            * (1 - network.stores.at[s, "standing_loss"])
            + m.store_p[s, last_calc_hour]
        )

        return expr

    network.model.cyclic_storage_constraint = po.Constraint(
        sus.index, rule=cyclic_state_of_charge
    )
    network.model.cyclic_store_constraint = po.Constraint(
        sto.index, rule=cyclic_state_of_charge_store
    )


def snapshot_clustering_seasonal_storage_hourly(self, network, snapshots):
    """
    Depicts intertemporal dependencies of storage units and stores when using
    snapshot clustering to typical periods for temporal complexity reduction.

    According to:
        L. Kotzur et al: 'Time series aggregation for energy
        system design:
        Modeling seasonal storage', 2018

    Parameters
    ----------
    network : :class:`pypsa.Network`
        Overall container of PyPSA
    snapshots : list
        A list of datetime objects representing the timestamps of the snapshots
        to be clustered.

    Returns
    -------
    None
    """

    # TODO: updaten mit stores (Sektorkopplung)

    network.model.del_component("state_of_charge_all")
    network.model.del_component("state_of_charge_all_index")
    network.model.del_component("state_of_charge_all_index_0")
    network.model.del_component("state_of_charge_all_index_1")
    network.model.del_component("state_of_charge_constraint")
    network.model.del_component("state_of_charge_constraint_index")
    network.model.del_component("state_of_charge_constraint_index_0")
    network.model.del_component("state_of_charge_constraint_index_1")

    candidates = network.cluster.index.get_level_values(0).unique()
    network.model.state_of_charge_all = po.Var(
        network.storage_units.index,
        candidates - 1 + self.args["start_snapshot"],
        within=po.NonNegativeReals,
    )
    network.model.storages = network.storage_units.index

    def set_soc_all(m, s, h):
        if h == self.args["start_snapshot"]:
            prev = (
                network.cluster.index.get_level_values(0)[-1]
                - 1
                + self.args["start_snapshot"]
            )

        else:
            prev = h - 1

        cluster_hour = network.cluster["Hour"][
            h + 1 - self.args["start_snapshot"]
        ]

        expr = m.state_of_charge_all[s, h] == m.state_of_charge_all[
            s, prev
        ] * (1 - network.storage_units.at[s, "standing_loss"]) - (
            m.storage_p_dispatch[s, cluster_hour]
            / network.storage_units.at[s, "efficiency_dispatch"]
            - network.storage_units.at[s, "efficiency_store"]
            * m.storage_p_store[s, cluster_hour]
        )
        return expr

    network.model.soc_all = po.Constraint(
        network.model.storages,
        candidates - 1 + self.args["start_snapshot"],
        rule=set_soc_all,
    )

    def soc_equals_soc_all(m, s, h):
        hour = (h.dayofyear - 1) * 24 + h.hour

        return m.state_of_charge_all[s, hour] == m.state_of_charge[s, h]

    network.model.soc_equals_soc_all = po.Constraint(
        network.model.storages, network.snapshots, rule=soc_equals_soc_all
    )

    network.model.del_component("state_of_charge_upper")
    network.model.del_component("state_of_charge_upper_index")
    network.model.del_component("state_of_charge_upper_index_0")
    network.model.del_component("state_of_charge_upper_index_1")

    def state_of_charge_upper(m, s, h):
        if network.storage_units.p_nom_extendable[s]:
            p_nom = m.storage_p_nom[s]
        else:
            p_nom = network.storage_units.p_nom[s]

        return (
            m.state_of_charge_all[s, h]
            <= p_nom * network.storage_units.at[s, "max_hours"]
        )

    network.model.state_of_charge_upper = po.Constraint(
        network.storage_units.index,
        candidates - 1 + self.args["start_snapshot"],
        rule=state_of_charge_upper,
    )


def snapshot_clustering_seasonal_storage_nmp(self, n, sns, simplified=False):
    """
    Depicts intertemporal dependencies of storage units and stores when using
    snapshot clustering to typical periods for temporal complexity reduction.

    According to:
        L. Kotzur et al: 'Time series aggregation for energy
        system design:
        Modeling seasonal storage', 2018

    Parameters
    ----------
    n : :class:`pypsa.Network`
        Overall container of PyPSA
    sns : list
        A list of datetime objects representing the timestamps of the snapshots
        to be clustered.
    simplified : bool, optional
        A flag indicating whether to use a simplified version of the model that
        does not include intra-temporal constraints and variables.

    Returns
    -------
    None
    """

    # TODO: so noch nicht korrekt...
    # TODO: updaten mit stores (Sektorkopplung)
    # TODO: simplified ergänzen

    sus = n.storage_units

    c = "StorageUnit"

    period_starts = sns[0::24]

    candidates = n.cluster.index.get_level_values(0).unique()

    soc_total = get_var(n, c, "state_of_charge")

    # inter_soc
    # Set lower and upper bound for soc_inter
    lb = pd.DataFrame(index=candidates, columns=sus.index, data=0)
    ub = pd.DataFrame(index=candidates, columns=sus.index, data=np.inf)

    # Create soc_inter variable for each storage and each day
    define_variables(n, lb, ub, "StorageUnit", "soc_inter")

    # Define soc_intra
    # Set lower and upper bound for soc_intra
    lb = pd.DataFrame(index=sns, columns=sus.index, data=-np.inf)
    ub = pd.DataFrame(index=sns, columns=sus.index, data=np.inf)

    # Set soc_intra to 0 at first hour of every day
    lb.loc[period_starts, :] = 0
    ub.loc[period_starts, :] = 0

    # Create intra soc variable for each storage and each hour
    define_variables(n, lb, ub, "StorageUnit", "soc_intra")
    soc_intra = get_var(n, c, "soc_intra")

    last_hour = n.cluster["last_hour_RepresentativeDay"].values

    soc_inter = get_var(n, c, "soc_inter")
    next_soc_inter = soc_inter.shift(-1).fillna(soc_inter.loc[candidates[0]])

    last_soc_intra = soc_intra.loc[last_hour].set_index(candidates)

    eff_stand = expand_series(1 - n.df(c).standing_loss, candidates).T
    eff_dispatch = expand_series(n.df(c).efficiency_dispatch, candidates).T
    eff_store = expand_series(n.df(c).efficiency_store, candidates).T

    dispatch = get_var(n, c, "p_dispatch").loc[last_hour].set_index(candidates)
    store = get_var(n, c, "p_store").loc[last_hour].set_index(candidates)

    coeff_var = [
        (-1, next_soc_inter),
        (eff_stand.pow(24), soc_inter),
        (eff_stand, last_soc_intra),
        (-1 / eff_dispatch, dispatch),
        (eff_store, store),
    ]

    lhs, *axes = linexpr(*coeff_var, return_axes=True)

    define_constraints(n, lhs, "==", 0, c, "soc_inter_constraints")

    coeff_var = [
        (-1, soc_total),
        (1, soc_intra),
        (
            1,
            soc_inter.loc[n.cluster_ts.loc[sns, "Candidate_day"]].set_index(
                sns
            ),
        ),
    ]
    lhs, *axes = linexpr(*coeff_var, return_axes=True)

    define_constraints(n, lhs, "==", 0, c, "soc_intra_constraints")


def snapshot_clustering_seasonal_storage_hourly_nmp(self, n, sns):
    """
    Depicts intertemporal dependencies of storage units and stores when using
    snapshot clustering to typical periods for temporal complexity reduction.

    According to:
        L. Kotzur et al: 'Time series aggregation for energy
        system design:
        Modeling seasonal storage', 2018

    Parameters
    ----------
    n : :class:`pypsa.Network`
        Overall container of PyPSA
    sns : list
        A list of datetime objects representing the timestamps of the snapshots
        to be clustered.

    Returns
    -------
    None
    """

    print("TODO")

    # TODO: implementieren


def split_dispatch_disaggregation_constraints(self, n, sns):
    """
    Add constraints for state of charge of storage units and stores
    when separating the optimization into smaller subproblems
    while conducting thedispatch_disaggregation in temporally fully resolved
    network

    The state of charge at the end of each slice is set to the value
    calculated in the optimization with the temporally reduced network
    to account to ensure compatibility and to reproduce saisonality

    Parameters
    ----------
    network : :class:`pypsa.Network`
        Overall container of PyPSA
    snapshots : pandas.DatetimeIndex
        List of timesteps considered in the optimization

    Returns
    -------
    None.
    """
    tsa_hour = sns[sns.isin(self.conduct_dispatch_disaggregation.index)]
    if len(tsa_hour) > 1:
        tsa_hour = tsa_hour[-1]
    else:
        tsa_hour = tsa_hour[0]
    n.model.soc_values = self.conduct_dispatch_disaggregation.loc[tsa_hour]

    sus = n.storage_units.index
    # for stores, exclude emob and dsm because of their special constraints
    sto = n.stores[
        ~n.stores.carrier.isin(["battery storage", "battery_storage", "dsm"])
    ].index

    def disaggregation_sus_soc(m, s, h):
        """
        Sets soc at the end of the time slice in disptach_disaggregation
        to value calculated in temporally reduced lopf without slices.
        """
        return m.state_of_charge[s, h] == m.soc_values[s]

    n.model.split_dispatch_sus_soc = po.Constraint(
        sus, sns[-1:], rule=disaggregation_sus_soc
    )

    def disaggregation_sto_soc(m, s, h):
        """
        Sets soc at the end of the time slice in disptach_disaggregation
        to value calculated in temporally reduced lopf without slices.
        """
        return m.store_e[s, h] == m.soc_values[s]

    n.model.split_dispatch_sto_soc = po.Constraint(
        sto, sns[-1:], rule=disaggregation_sto_soc
    )


def split_dispatch_disaggregation_constraints_nmp(self, n, sns):
    print("TODO")

    # TODO: implementieren


def fixed_storage_unit_soc_at_the_end(n, sns):
    """
    Defines energy balance constraints for storage units. In principal the
    constraints states:

    previous_soc + p_store - p_dispatch + inflow - spill == soc
    """
    from xarray import DataArray

    sns = n.snapshots[-1]
    m = n.model
    c = "StorageUnit"
    assets = n.df(c)
    if assets.empty:
        return

    # elapsed hours
    eh = n.snapshot_weightings.stores[sns]
    # efficiencies
    eff_stand = (1 - n.storage_units.standing_loss).pow(eh)
    eff_dispatch = n.storage_units.efficiency_dispatch
    eff_store = n.storage_units.efficiency_store

    # SOC first hour of the year
    post_soc = n.storage_units_t.state_of_charge.loc[n.snapshots[0]]

    # SOC last hour of the year
    soc = m[f"{c}-state_of_charge"].loc[sns]

    lhs = [
        (1, soc),
        (-1 / eff_dispatch * eh, m[f"{c}-p_dispatch"].loc[sns]),
        (eff_store * eh, m[f"{c}-p_store"].loc[sns]),
    ]

    if f"{c}-spill" in m.variables:
        lhs += [(-eh, m[f"{c}-spill"])]

    # We add inflow and initial soc for noncyclic assets to rhs
    rhs = DataArray((-n.storage_units.inflow).mul(eh) + (eff_stand * post_soc))

    m.add_constraints(lhs, "=", rhs, name=f"{c}-energy_balance_end")


class Constraints:
    def __init__(
        self, args, conduct_dispatch_disaggregation, apply_on="grid_model"
    ):
        self.args = args
        self.conduct_dispatch_disaggregation = conduct_dispatch_disaggregation
        self.apply_on = apply_on

    def functionality(self, network, snapshots):
        """Add constraints to pypsa-model using extra-functionality.
        Serveral constraints can be choosen at once. Possible constraints are
        set and described in the above functions.

        Parameters
        ----------
        network : :class:`pypsa.Network`
            Overall container of PyPSA
        snapshots : pandas.DatetimeIndex
            List of timesteps considered in the optimization

        """
        if "CH4" in network.buses.carrier.values:
            if self.args["method"]["formulation"] == "pyomo":
                add_chp_constraints(network, snapshots)
                if (self.args["scn_name"] != "status2019") & (
                    len(snapshots) > 1500
                ):
                    add_ch4_constraints(self, network, snapshots)
            elif self.args["method"]["formulation"] == "linopy":
                if (self.args["scn_name"] != "status2019") & (
                    len(snapshots) > 1500
                ):
                    add_ch4_constraints_linopy(self, network, snapshots)

                if self.apply_on == "last_market_model":
                    fixed_storage_unit_soc_at_the_end(network, snapshots)
                add_chp_constraints_linopy(network, snapshots)
            else:
                add_chp_constraints_nmp(network)
                if self.args["scn_name"] != "status2019":
                    add_ch4_constraints_nmp(self, network, snapshots)

        for constraint in self.args["extra_functionality"].keys():
            if self.args["method"]["formulation"] == "pyomo":
                try:
                    eval("_" + constraint + "(self, network, snapshots)")
                    logger.info(
                        "Added extra_functionality {}".format(constraint)
                    )
                except:
                    logger.warning(
                        "Constraint {} not defined".format(constraint)
                        + ". New constraints can be defined in"
                        + " etrago/tools/constraint.py."
                    )
            elif self.args["method"]["formulation"] == "linopy":
                try:
                    eval(
                        "_" + constraint + "_linopy(self, network, snapshots)"
                    )
                    logger.info(
                        "Added extra_functionality {}".format(constraint)
                    )
                except:
                    logger.warning(
                        "Constraint {} not defined for linopy formulation".format(
                            constraint
                        )
                        + ". New constraints can be defined in"
                        + " etrago/tools/constraint.py."
                    )
            else:
                try:
                    eval("_" + constraint + "_nmp(self, network, snapshots)")
                    logger.info(
                        "Added extra_functionality {} without pyomo".format(
                            constraint
                        )
                    )
                except:
                    logger.warning(
                        "Constraint {} not defined".format(constraint)
                    )

        if (
            self.args["snapshot_clustering"]["active"]
            and self.args["snapshot_clustering"]["method"] == "typical_periods"
        ):
            if (
                self.args["snapshot_clustering"]["storage_constraints"]
                == "daily_bounds"
            ):
                if self.args["method"]["pyomo"]:
                    snapshot_clustering_daily_bounds(self, network, snapshots)
                else:
                    snapshot_clustering_daily_bounds_nmp(
                        self, network, snapshots
                    )

            elif (
                self.args["snapshot_clustering"]["storage_constraints"]
                == "soc_constraints"
            ):
                if self.args["snapshot_clustering"]["how"] == "hourly":
                    if self.args["method"]["pyomo"]:
                        snapshot_clustering_seasonal_storage_hourly(
                            self, network, snapshots
                        )
                    else:
                        snapshot_clustering_seasonal_storage_hourly_nmp(
                            self, network, snapshots
                        )
                else:
                    if self.args["method"]["pyomo"]:
                        snapshot_clustering_seasonal_storage(
                            self, network, snapshots
                        )
                    else:
                        snapshot_clustering_seasonal_storage_nmp(
                            self, network, snapshots
                        )

            elif (
                self.args["snapshot_clustering"]["storage_constraints"]
                == "soc_constraints_simplified"
            ):
                if self.args["snapshot_clustering"]["how"] == "hourly":
                    logger.info(
                        """soc_constraints_simplified not possible while hourly
                        clustering -> changed to soc_constraints"""
                    )

                    if self.args["method"]["pyomo"]:
                        snapshot_clustering_seasonal_storage_hourly(
                            self, network, snapshots
                        )
                    else:
                        snapshot_clustering_seasonal_storage_hourly_nmp(
                            self, network, snapshots
                        )

                if self.args["method"]["pyomo"]:
                    snapshot_clustering_seasonal_storage(
                        self, network, snapshots, simplified=True
                    )
                else:
                    snapshot_clustering_seasonal_storage_nmp(
                        self, network, snapshots, simplified=True
                    )

            else:
                logger.error(
                    """If you want to use constraints considering the storage
                    behaviour, snapshot clustering constraints must be in
                    [daily_bounds, soc_constraints,
                     soc_constraints_simplified]"""
                )

        if self.conduct_dispatch_disaggregation is not False:
            if self.args["method"]["pyomo"]:
                split_dispatch_disaggregation_constraints(
                    self, network, snapshots
                )
            else:
                split_dispatch_disaggregation_constraints_nmp(
                    self, network, snapshots
                )


def add_chp_constraints_nmp(n):
    """
    Limits the dispatch of combined heat and power links based on
    T.Brown et. al : Synergies of sector coupling and transmission
    reinforcement in a cost-optimised, highly renewable European energy system,
    2018

    Parameters
    ----------
    n : pypsa.Network
        Network container

    Returns
    -------
    None.

    """
    # backpressure limit
    c_m = 0.75

    # marginal loss for each additional generation of heat
    c_v = 0.15
    electric_bool = n.links.carrier == "central_gas_CHP"
    heat_bool = n.links.carrier == "central_gas_CHP_heat"

    electric = n.links.index[electric_bool]
    heat = n.links.index[heat_bool]

    n.links.loc[heat, "efficiency"] = (
        n.links.loc[electric, "efficiency"] / c_v
    ).values.mean()

    ch4_nodes_with_chp = n.buses.loc[
        n.links.loc[electric, "bus0"].values
    ].index.unique()

    for i in ch4_nodes_with_chp:
        elec_chp = n.links[
            (n.links.carrier == "central_gas_CHP") & (n.links.bus0 == i)
        ].index

        heat_chp = n.links[
            (n.links.carrier == "central_gas_CHP_heat") & (n.links.bus0 == i)
        ].index

        link_p = get_var(n, "Link", "p")
        # backpressure

        lhs_1 = sum(
            c_m * n.links.at[h_chp, "efficiency"] * link_p[h_chp]
            for h_chp in heat_chp
        )

        lhs_2 = sum(
            n.links.at[e_chp, "efficiency"] * link_p[e_chp]
            for e_chp in elec_chp
        )

        lhs = linexpr((1, lhs_1), (-1, lhs_2))

        define_constraints(
            n, lhs, "<=", 0, "chplink_" + str(i), "backpressure"
        )

        # top_iso_fuel_line
        lhs, *ax = linexpr(
            (1, sum(link_p[h_chp] for h_chp in heat_chp)),
            (1, sum(link_p[h_e] for h_e in elec_chp)),
            return_axes=True,
        )

        define_constraints(
            n,
            lhs,
            "<=",
            n.links.loc[elec_chp].p_nom.sum(),
            "chplink_" + str(i),
            "top_iso_fuel_line_fix",
            axes=ax,
        )


def add_chp_constraints(network, snapshots):
    """
    Limits the dispatch of combined heat and power links based on
    T.Brown et. al : Synergies of sector coupling and transmission
    reinforcement in a cost-optimised, highly renewable European energy system,
    2018

    Parameters
    ----------
    network : pypsa.Network
        Network container
    snapshots : pandas.DataFrame
        Timesteps to optimize

    Returns
    -------
    None.

    """

    # backpressure limit
    c_m = 0.75

    # marginal loss for each additional generation of heat
    c_v = 0.15
    electric_bool = network.links.carrier == "central_gas_CHP"
    heat_bool = network.links.carrier == "central_gas_CHP_heat"

    electric = network.links.index[electric_bool]
    heat = network.links.index[heat_bool]

    network.links.loc[heat, "efficiency"] = (
        network.links.loc[electric, "efficiency"] / c_v
    ).values.mean()

    ch4_nodes_with_chp = network.buses.loc[
        network.links.loc[electric, "bus0"].values
    ].index.unique()

    for i in ch4_nodes_with_chp:
        elec_chp = network.links[
            (network.links.carrier == "central_gas_CHP")
            & (network.links.bus0 == i)
        ].index

        heat_chp = network.links[
            (network.links.carrier == "central_gas_CHP_heat")
            & (network.links.bus0 == i)
        ].index

        # Guarantees c_m p_b1  \leq p_g1
        def backpressure(model, snapshot):
            lhs = sum(
                c_m
                * network.links.at[h_chp, "efficiency"]
                * model.link_p[h_chp, snapshot]
                for h_chp in heat_chp
            )

            rhs = sum(
                network.links.at[e_chp, "efficiency"]
                * model.link_p[e_chp, snapshot]
                for e_chp in elec_chp
            )

            return lhs <= rhs

        setattr(
            network.model,
            "backpressure_" + str(i),
            Constraint(list(snapshots), rule=backpressure),
        )

        # Guarantees p_g1 +c_v p_b1 \leq p_g1_nom
        def top_iso_fuel_line(model, snapshot):
            lhs = sum(
                model.link_p[h_chp, snapshot] for h_chp in heat_chp
            ) + sum(model.link_p[e_chp, snapshot] for e_chp in elec_chp)

            rhs = network.links[
                (network.links.carrier == "central_gas_CHP")
                & (network.links.bus0 == i)
            ].p_nom.sum()

            return lhs <= rhs

        setattr(
            network.model,
            "top_iso_fuel_line_" + str(i),
            Constraint(list(snapshots), rule=top_iso_fuel_line),
        )


def add_chp_constraints_linopy(network, snapshots):
    """
    Limits the dispatch of combined heat and power links based on
    T.Brown et. al : Synergies of sector coupling and transmission
    reinforcement in a cost-optimised, highly renewable European energy system,
    2018

    Parameters
    ----------
    network : pypsa.Network
        Network container
    snapshots : pandas.DataFrame
        Timesteps to optimize

    Returns
    -------
    None.

    """

    # backpressure limit
    c_m = 0.75

    # marginal loss for each additional generation of heat
    c_v = 0.15
    electric_bool = network.links.carrier == "central_gas_CHP"
    heat_bool = network.links.carrier == "central_gas_CHP_heat"

    electric = network.links.index[electric_bool]
    heat = network.links.index[heat_bool]

    network.links.loc[heat, "efficiency"] = (
        network.links.loc[electric, "efficiency"] / c_v
    ).values.mean()

    ch4_nodes_with_chp = network.buses.loc[
        network.links.loc[electric, "bus0"].values
    ].index.unique()

    for i in ch4_nodes_with_chp:
        elec_chp = network.links[
            (network.links.carrier == "central_gas_CHP")
            & (network.links.bus0 == i)
        ].index

        heat_chp = network.links[
            (network.links.carrier == "central_gas_CHP_heat")
            & (network.links.bus0 == i)
        ].index

        for snapshot in snapshots:
            dispatch_heat = (
                c_m
                * get_var(network, "Link", "p").loc[snapshot, heat_chp]
                * network.links.loc[heat_chp, "efficiency"]
            ).sum()
            dispatch_elec = (
                get_var(network, "Link", "p").loc[snapshot, elec_chp]
                * network.links.loc[elec_chp, "efficiency"]
            ).sum()

            define_constraints(
                network,
                (dispatch_heat - dispatch_elec),
                "<=",
                0,
                "Link",
                "backpressure_" + i + "_" + str(snapshot),
            )

            define_constraints(
                network,
                get_var(network, "Link", "p").loc[snapshot, heat_chp].sum()
                + get_var(network, "Link", "p").loc[snapshot, elec_chp].sum(),
                "<=",
                network.links[
                    (network.links.carrier == "central_gas_CHP")
                    & (network.links.bus0 == i)
                ].p_nom.sum(),
                "Link",
                "top_iso_fuel_line_" + i + "_" + str(snapshot),
            )
