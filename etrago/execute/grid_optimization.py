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


def grid_optimization(
    self,
    factor_redispatch_cost=1,
    management_cost=0,
    time_depended_cost=True,
    fre_mangement_fee=0,
):
    logger.info("Start building grid optimization model")

    # Drop existing ramping generators
    self.network.mremove(
        "Generator",
        self.network.generators[
            self.network.generators.index.str.contains("ramp")
        ].index,
    )
    self.network.mremove(
        "Link",
        self.network.links[
            self.network.links.index.str.contains("ramp")
        ].index,
    )

    fix_chp_generation(self)

    add_redispatch_generators(
        self,
        factor_redispatch_cost,
        management_cost,
        time_depended_cost,
        fre_mangement_fee,
    )

    if not self.args["method"]["market_optimization"]["redispatch"]:
        self.network.mremove(
            "Generator",
            self.network.generators[
                self.network.generators.index.str.contains("ramp")
            ].index,
        )
        self.network.mremove(
            "Link",
            self.network.links[
                self.network.links.index.str.contains("ramp")
            ].index,
        )
    logger.info("Start solving grid optimization model")

    # Replace NaN values in quadratic costs to keep problem linear
    self.network.generators.marginal_cost_quadratic.fillna(0.0, inplace=True)
    self.network.links.marginal_cost_quadratic.fillna(0.0, inplace=True)

    # Replacevery small values with zero to avoid numerical problems
    self.network.generators_t.p_max_pu.where(
        self.network.generators_t.p_max_pu.abs() > 1e-7,
        other=0.0,
        inplace=True,
    )
    self.network.generators_t.p_min_pu.where(
        self.network.generators_t.p_min_pu.abs() > 1e-7,
        other=0.0,
        inplace=True,
    )
    self.network.links_t.p_max_pu.where(
        self.network.links_t.p_max_pu.abs() > 1e-7, other=0.0, inplace=True
    )
    self.network.links_t.p_min_pu.where(
        self.network.links_t.p_min_pu > 1e-7, other=0.0, inplace=True
    )

    self.network.links.loc[
        (
            self.network.links.bus0.isin(
                self.network.buses[self.network.buses.country == "GB"].index
            )
        )
        & (
            self.network.links.bus1.isin(
                self.network.buses[self.network.buses.country == "GB"].index
            )
        )
        & (self.network.links.carrier == "DC"),
        "p_nom_max",
    ] = np.inf

    self.network.storage_units.loc[
        (
            self.network.storage_units.bus.isin(
                self.network.buses[self.network.buses.country != "DE"].index
            )
        )
        & (self.network.storage_units.carrier == "battery"),
        "p_nom_max",
    ] = np.inf

    self.args["method"]["formulation"] = "pyomo"
    if self.args["method"]["type"] == "lopf":
        self.lopf()
    else:
        self.sclopf(
            post_lopf=False,
            n_process=4,
            delta=0.01,
            n_overload=0,
            div_ext_lines=False,
        )


def fix_chp_generation(self):
    # Select generator and link components that are fixed after
    # the market optimization.
    gens_fixed = self.network.generators[
        self.network.generators.carrier.str.endswith("_CHP")
    ].index

    links_fixed = self.network.links[
        self.network.links.carrier.str.endswith("_CHP")
    ].index

    # Fix generator dispatch from market simulation:
    # Set p_max_pu of generators using results from (disaggregated) market
    # model
    self.network.generators_t.p_max_pu.loc[:, gens_fixed] = (
        self.market_model.generators_t.p[gens_fixed].mul(
            1 / self.market_model.generators.p_nom[gens_fixed]
        )
    )

    # Set p_min_pu of generators using results from (disaggregated) market
    # model
    self.network.generators_t.p_min_pu.loc[:, gens_fixed] = (
        self.market_model.generators_t.p[gens_fixed].mul(
            1 / self.market_model.generators.p_nom[gens_fixed]
        )
    )

    # Fix link dispatch (gas turbines) from market simulation
    # Set p_max_pu of links using results from (disaggregated) market model
    self.network.links_t.p_max_pu.loc[:, links_fixed] = (
        self.market_model.links_t.p0[links_fixed].mul(
            1 / self.market_model.links.p_nom[links_fixed]
        )
    )

    # Set p_min_pu of links using results from (disaggregated) market model
    self.network.links_t.p_min_pu.loc[:, links_fixed] = (
        self.market_model.links_t.p0[links_fixed].mul(
            1 / self.market_model.links.p_nom[links_fixed]
        )
    )


def add_redispatch_generators(
    self,
    factor_redispatch_cost,
    management_cost,
    time_depended_cost,
    fre_mangement_fee,
):
    """Add components and parameters to model redispatch with costs

    This function currently assumes that the market_model includes all
    generators and links for the spatial resolution of the grid optimization

    Returns
    -------
    None.

    """

    # Select generator and link components that are considered in redispatch
    # all others can be redispatched without any extra costs
    gens_redispatch = self.network.generators[
        (
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
                    "biomass",
                    "OCGT",
                ]
            )
            & (~self.network.generators.index.str.contains("ramp"))
        )
    ].index

    links_redispatch = self.network.links[
        (
            self.network.links.carrier.isin(["OCGT"])
            & (~self.network.links.index.str.contains("ramp"))
        )
    ].index

    management_cost_carrier = pd.Series(
        index=self.network.generators.loc[gens_redispatch].carrier.unique(),
        data=management_cost,
    )
    management_cost_carrier["OCGT"] = management_cost
    if fre_mangement_fee:
        management_cost_carrier[
            ["wind_onshore", "wind_offshore", "solar", "solar_rooftop"]
        ] = fre_mangement_fee

    management_cost_per_generator = management_cost_carrier.loc[
        self.network.generators.loc[gens_redispatch, "carrier"].values
    ]
    management_cost_per_generator.index = gens_redispatch

    management_cost_per_link = management_cost_carrier.loc[
        self.network.links.loc[links_redispatch, "carrier"].values
    ]
    management_cost_per_link.index = links_redispatch

    if time_depended_cost:
        management_cost_per_generator = pd.DataFrame(
            index=self.network.snapshots,
            columns=management_cost_per_generator.index,
        )
        management_cost_per_link = pd.DataFrame(
            index=self.network.snapshots,
            columns=management_cost_per_link.index,
        )
        for i in self.network.snapshots:
            management_cost_per_generator.loc[i, :] = (
                management_cost_carrier.loc[
                    self.network.generators.loc[
                        gens_redispatch, "carrier"
                    ].values
                ].values
            )

            management_cost_per_link.loc[i, :] = management_cost_carrier.loc[
                self.network.links.loc[links_redispatch, "carrier"].values
            ].values

    # Fix generator dispatch from market simulation:
    # Set p_max_pu of generators using results from (disaggregated) market
    # model
    self.network.generators_t.p_max_pu.loc[:, gens_redispatch] = (
        self.market_model.generators_t.p[gens_redispatch].mul(
            1 / self.market_model.generators.p_nom[gens_redispatch]
        )
    )

    # Set p_min_pu of generators using results from (disaggregated) market
    # model
    self.network.generators_t.p_min_pu.loc[:, gens_redispatch] = (
        self.market_model.generators_t.p[gens_redispatch].mul(
            1 / self.market_model.generators.p_nom[gens_redispatch]
        )
    )

    # Fix link dispatch (gas turbines) from market simulation
    # Set p_max_pu of links using results from (disaggregated) market model
    self.network.links_t.p_max_pu.loc[:, links_redispatch] = (
        self.market_model.links_t.p0[links_redispatch].mul(
            1 / self.market_model.links.p_nom[links_redispatch]
        )
    )

    # Set p_min_pu of links using results from (disaggregated) market model
    self.network.links_t.p_min_pu.loc[:, links_redispatch] = (
        self.market_model.links_t.p0[links_redispatch].mul(
            1 / self.market_model.links.p_nom[links_redispatch]
        )
    )

    # Calculate costs for redispatch
    # Extract prices per market zone from market model results
    market_price_per_bus = self.market_model.buses_t.marginal_price.copy()

    # Set market price for each disaggregated generator according to the bus
    # can be reduced liner by setting a factor_redispatch_cost
    market_price_per_generator = (
        market_price_per_bus.loc[
            :, self.market_model.generators.loc[gens_redispatch, "bus"]
        ]
        * factor_redispatch_cost
    )

    market_price_per_link = (
        market_price_per_bus.loc[
            :, self.market_model.links.loc[links_redispatch, "bus1"]
        ]
        * factor_redispatch_cost
    )

    if not time_depended_cost:
        market_price_per_generator = market_price_per_generator.median()
        market_price_per_generator.index = gens_redispatch
        market_price_per_link = market_price_per_link.median()
        market_price_per_link.index = links_redispatch
    else:
        market_price_per_generator.columns = gens_redispatch
        market_price_per_link.columns = links_redispatch
        market_price_per_generator = market_price_per_generator.loc[
            self.network.snapshots
        ]

    # Costs for ramp_up generators are first set the marginal_cost for each
    # generator
    if time_depended_cost:
        ramp_up_costs = pd.DataFrame(
            index=self.network.snapshots,
            columns=gens_redispatch,
        )
        for i in ramp_up_costs.index:
            ramp_up_costs.loc[i, gens_redispatch] = (
                self.network.generators.loc[
                    gens_redispatch, "marginal_cost"
                ].values
            )

    else:
        ramp_up_costs = self.network.generators.loc[
            gens_redispatch, "marginal_cost"
        ]

    # In case the market price is higher than the marginal_cost (e.g. for
    # renewables) ramp up costs are set to the market price. This way,
    # every generator gets at least the costs at the market.
    # In case the marginal cost are higher, e.g. because of fuel costs,
    # the real marginal price is payed for redispatch

    if time_depended_cost:
        ramp_up_costs[market_price_per_generator > ramp_up_costs] = (
            market_price_per_generator
        )

    else:
        ramp_up_costs[
            market_price_per_generator
            > self.network.generators.loc[gens_redispatch, "marginal_cost"]
        ] = market_price_per_generator

    ramp_up_costs = ramp_up_costs + management_cost_per_generator.values

    # Costs for ramp down generators consist of the market price
    # which is still payed for the generation. Fuel costs can be saved,
    # therefore the ramp down costs are reduced by the marginal costs
    if time_depended_cost:
        ramp_down_costs = (
            market_price_per_generator
            - self.network.generators.loc[
                gens_redispatch, "marginal_cost"
            ].values
        )
        ramp_down_costs.columns = gens_redispatch + " ramp_down"
    else:
        ramp_down_costs = (
            market_price_per_generator
            - self.network.generators.loc[
                gens_redispatch, "marginal_cost"
            ].values
        )
    ramp_down_costs = ramp_down_costs + management_cost_per_generator.values
    # Add ramp up generators to the network for the grid optimization
    # Marginal cost are incread by a management fee of 4 EUR/MWh
    self.network.madd(
        "Generator",
        gens_redispatch + " ramp_up",
        bus=self.network.generators.loc[gens_redispatch, "bus"].values,
        p_nom=self.network.generators.loc[gens_redispatch, "p_nom"].values,
        carrier=self.network.generators.loc[gens_redispatch, "carrier"].values,
    )

    if time_depended_cost:
        ramp_up_costs.columns += " ramp_up"
        self.network.generators_t.marginal_cost = pd.concat(
            [self.network.generators_t.marginal_cost, ramp_up_costs], axis=1
        )
    else:
        self.network.generators.loc[
            gens_redispatch + " ramp_up", "marginal_cost"
        ] = ramp_up_costs

    # Set maximum feed-in limit for ramp up generators based on feed-in of
    # (disaggregated) generators from the market optimization and potential
    # feedin time series
    p_max_pu_all = self.network.get_switchable_as_dense(
        "Generator", "p_max_pu"
    )

    self.network.generators_t.p_max_pu.loc[:, gens_redispatch + " ramp_up"] = (
        (
            p_max_pu_all.loc[:, gens_redispatch].mul(
                self.network.generators.loc[gens_redispatch, "p_nom"]
            )
            - (
                self.market_model.generators_t.p.loc[
                    self.network.snapshots, gens_redispatch
                ]
            )
        )
        .clip(lower=0.0)
        .mul(1 / self.network.generators.loc[gens_redispatch, "p_nom"])
        .values
    )

    # Add ramp up links to the network for the grid optimization
    # Marginal cost are incread by a management fee of 4 EUR/MWh
    if time_depended_cost:
        ramp_up_costs_links = pd.DataFrame(
            index=self.network.snapshots,
            columns=links_redispatch,
        )
        for i in ramp_up_costs.index:
            ramp_up_costs_links.loc[i, links_redispatch] = (
                self.network.links.loc[
                    links_redispatch, "marginal_cost"
                ].values
            )

        ramp_up_costs_links[
            market_price_per_link.loc[self.network.snapshots]
            > ramp_up_costs_links
        ] = market_price_per_link

    else:
        ramp_up_costs_links = self.network.links.loc[
            links_redispatch + " ramp_up", "marginal_cost"
        ]

        ramp_up_costs_links[
            market_price_per_link
            > self.network.links.loc[links_redispatch, "marginal_cost"]
        ] = market_price_per_link

    ramp_up_costs_links = ramp_up_costs_links + management_cost_per_link.values

    self.network.madd(
        "Link",
        links_redispatch + " ramp_up",
        bus0=self.network.links.loc[links_redispatch, "bus0"].values,
        bus1=self.network.links.loc[links_redispatch, "bus1"].values,
        p_nom=self.network.links.loc[links_redispatch, "p_nom"].values,
        carrier=self.network.links.loc[links_redispatch, "carrier"].values,
        efficiency=self.network.links.loc[
            links_redispatch, "efficiency"
        ].values,
    )

    if time_depended_cost:
        ramp_up_costs_links.columns += " ramp_up"
        self.network.links_t.marginal_cost = pd.concat(
            [self.network.links_t.marginal_cost, ramp_up_costs_links], axis=1
        )
    else:
        self.network.links.loc[
            links_redispatch + " ramp_up", "marginal_cost"
        ] = ramp_up_costs_links

    # Set maximum feed-in limit for ramp up links based on feed-in of
    # (disaggregated) links from the market optimization
    self.network.links_t.p_max_pu.loc[:, links_redispatch + " ramp_up"] = (
        (
            self.network.links.loc[links_redispatch, "p_nom"]
            - (
                self.market_model.links_t.p0.loc[
                    self.network.snapshots, links_redispatch
                ]
            )
        )
        .clip(lower=0.0)
        .mul(1 / self.network.links.loc[links_redispatch, "p_nom"])
        .values
    )

    # Add ramp down generators to the network for the grid optimization
    # Marginal cost are incread by a management fee of 4 EUR/MWh, since the
    # feedin is negative, the costs are multiplyed by (-1)
    self.network.madd(
        "Generator",
        gens_redispatch + " ramp_down",
        bus=self.network.generators.loc[gens_redispatch, "bus"].values,
        p_nom=self.network.generators.loc[gens_redispatch, "p_nom"].values,
        carrier=self.network.generators.loc[gens_redispatch, "carrier"].values,
    )

    if time_depended_cost:
        self.network.generators_t.marginal_cost = pd.concat(
            [self.network.generators_t.marginal_cost, -ramp_down_costs], axis=1
        )
    else:
        self.network.generators.loc[
            gens_redispatch + " ramp_down", "marginal_cost"
        ] = -(ramp_down_costs.values)

    # Ramp down generators can not feed-in addtional energy
    self.network.generators_t.p_max_pu.loc[
        :, gens_redispatch + " ramp_down"
    ] = 0.0
    # Ramp down can be at maximum as high as the feed-in of the
    # (disaggregated) generators in the market model
    self.network.generators_t.p_min_pu.loc[
        :, gens_redispatch + " ramp_down"
    ] = (
        -(
            self.market_model.generators_t.p.loc[
                self.network.snapshots, gens_redispatch
            ]
            .clip(lower=0.0)
            .mul(1 / self.network.generators.loc[gens_redispatch, "p_nom"])
        )
    ).values

    # Add ramp down links to the network for the grid optimization
    # Marginal cost are currently only the management fee of 4 EUR/MWh,
    # other costs are somehow complicated due to the gas node and fuel costs
    # this is still an open ToDO.
    self.network.madd(
        "Link",
        links_redispatch + " ramp_down",
        bus0=self.network.links.loc[links_redispatch, "bus0"].values,
        bus1=self.network.links.loc[links_redispatch, "bus1"].values,
        p_nom=self.network.links.loc[links_redispatch, "p_nom"].values,
        marginal_cost=-(management_cost),
        carrier=self.network.links.loc[links_redispatch, "carrier"].values,
        efficiency=self.network.links.loc[
            links_redispatch, "efficiency"
        ].values,
    )

    # Ramp down links can not feed-in addtional energy
    self.network.links_t.p_max_pu.loc[:, links_redispatch + " ramp_down"] = 0.0

    # Ramp down can be at maximum as high as the feed-in of the
    # (disaggregated) links in the market model
    self.network.links_t.p_min_pu.loc[:, links_redispatch + " ramp_down"] = (
        -(
            self.market_model.links_t.p0.loc[
                self.network.snapshots, links_redispatch
            ]
            .clip(lower=0.0)
            .mul(1 / self.network.links.loc[links_redispatch, "p_nom"])
        )
    ).values

    # Check if the network contains any problems
    self.network.consistency_check()

    # just for the current status2019 scenario a quick fix for buses which
    # do not have a connection
    # self.network.buses.drop(
    #     self.network.buses[
    #         self.network.buses.index.isin(['47085', '47086', '37865', '37870'
    #                                        ])].index, inplace=True)


def extra_functionality():
    return None
