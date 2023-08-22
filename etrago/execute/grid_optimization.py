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
Defines the market optimization within eTraGo
"""
import os

if "READTHEDOCS" not in os.environ:
    import logging
    import time

    import numpy as np
    import pandas as pd

    logger = logging.getLogger(__name__)

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ulfmueller, ClaraBuettner, CarlosEpia"


def grid_optimization(self):
    add_redispatch_generators(self)

    self.network.lopf(
        solver_name=self.args["solver"],
        solver_options=self.args["solver_options"],
        pyomo=True,
        extra_functionality=extra_functionality(),
        formulation=self.args["model_formulation"],
    )


def add_redispatch_generators(self):
    """Add components and parameters to model redispatch with costs

    Returns
    -------
    None.

    """

    # Select generator and link components that are considered in redispatch
    # all others can be redispatched without any extra costs
    gens_redispatch = self.network.generators[
        self.network.generators.carrier.isin(
            [
                "coal",
                "lignite",
                "nuclear",
                "oil",
                "others",
                "reservoir",
                "run_of_river",
                "solar",
                "wind_offshore",
                "wind_onshore",
                "solar_rooftop",
                "central_biomass_CHP",
                "industrial_biomass_CHP",
                "biomass",
            ]
        )
    ].index

    links_redispatch = self.network.links[
        self.network.links.carrier.isin(["OCGT"])
    ].index

    # Fix generator dispatch from market simulation
    self.network.generators_t.p_max_pu.loc[
        :, gens_redispatch
    ] = self.network.generators_t.p[gens_redispatch].mul(
        1 / self.network.generators.p_nom[gens_redispatch]
    )

    self.network.generators_t.p_min_pu.loc[
        :, gens_redispatch
    ] = self.network.generators_t.p[gens_redispatch].mul(
        1 / self.network.generators.p_nom[gens_redispatch]
    )

    # Fix link dispatch from market simulation
    self.network.links_t.p_max_pu.loc[
        :, links_redispatch
    ] = self.network.links_t.p0[links_redispatch].mul(
        1 / self.network.links.p_nom[links_redispatch]
    )

    self.network.links_t.p_min_pu.loc[
        :, gens_redispatch
    ] = self.network.links_t.p0[links_redispatch].mul(
        1 / self.network.links.p_nom[links_redispatch]
    )

    # Calculate costs for redispatch
    market_price_per_bus = self.network.buses_t.marginal_price

    self.network.generators.loc[gens_redispatch, "marginal_cost"]

    market_price_per_generator = market_price_per_bus.loc[
        :, self.network.generators.loc[gens_redispatch, "bus"]
    ].median()
    market_price_per_generator.index = gens_redispatch

    ramp_up_costs = self.network.generators.loc[
        gens_redispatch, "marginal_cost"
    ]

    ramp_up_costs[
        market_price_per_generator
        > self.network.generators.loc[gens_redispatch, "marginal_cost"]
    ] = market_price_per_generator

    ramp_down_costs = (
        market_price_per_generator
        - self.network.generators.loc[gens_redispatch, "marginal_cost"]
    )

    # Add ramp up generators
    self.network.madd(
        "Generator",
        gens_redispatch + " ramp_up",
        bus=self.network.generators.loc[gens_redispatch, "bus"].values,
        p_nom=self.network.generators.loc[gens_redispatch, "p_nom"].values,
        marginal_cost=ramp_up_costs.values + 4,
        carrier=self.network.generators.loc[gens_redispatch, "carrier"].values,
    )

    self.network.generators_t.p_max_pu.loc[:, gens_redispatch + " ramp_up"] = (
        (
            self.network.generators_t.p_max_pu.loc[:, gens_redispatch].mul(
                self.network.generators.loc[gens_redispatch, "p_nom"]
            )
            - (self.network.generators_t.p.loc[:, gens_redispatch])
        )
        .clip(lower=0.0)
        .values
    )

    # Add ramp up links
    self.network.madd(
        "Link",
        links_redispatch + " ramp_up",
        bus0=self.network.links.loc[links_redispatch, "bus0"].values,
        bus1=self.network.links.loc[links_redispatch, "bus1"].values,
        p_nom=self.network.links.loc[links_redispatch, "p_nom"].values,
        marginal_cost=self.network.links.loc[
            links_redispatch, "marginal_cost"
        ].values
        + 4,
        carrier=self.network.links.loc[links_redispatch, "carrier"].values,
    )

    self.network.links_t.p_max_pu.loc[:, links_redispatch + " ramp_up"] = (
        (
            self.network.links.loc[links_redispatch, "p_nom"]
            - (self.network.links_t.p0.loc[:, links_redispatch])
        )
        .clip(lower=0.0)
        .values
    )

    # Add ramp down generators
    self.network.madd(
        "Generator",
        gens_redispatch + " ramp_down",
        bus=self.network.generators.loc[gens_redispatch, "bus"].values,
        p_nom=self.network.generators.loc[gens_redispatch, "p_nom"].values,
        marginal_cost=-(ramp_down_costs.values + 4),
        carrier=self.network.generators.loc[gens_redispatch, "carrier"].values,
    )

    self.network.generators_t.p_max_pu.loc[
        :, gens_redispatch + " ramp_down"
    ] = 0

    self.network.generators_t.p_min_pu.loc[
        :, gens_redispatch + " ramp_down"
    ] = (-(self.network.generators_t.p.loc[:, gens_redispatch])).values

    # Add ramp down links
    self.network.madd(
        "Link",
        links_redispatch + " ramp_down",
        bus0=self.network.links.loc[links_redispatch, "bus0"].values,
        bus1=self.network.links.loc[links_redispatch, "bus1"].values,
        p_nom=self.network.links.loc[links_redispatch, "p_nom"].values,
        marginal_cost=-(4),
        carrier=self.network.links.loc[links_redispatch, "carrier"].values,
    )

    self.network.links_t.p_max_pu.loc[:, links_redispatch + " ramp_down"] = 0

    self.network.links_t.p_min_pu.loc[:, links_redispatch + " ramp_down"] = (
        -(self.network.links_t.p0.loc[:, links_redispatch])
    ).values


def extra_functionality():
    return None
