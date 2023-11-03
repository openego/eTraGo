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
    
    logger.info("Start building grid optimization model")
    fix_chp_generation(self)
    add_redispatch_generators(self)
    #self.network.generators.drop(self.network.generators[self.network.generators.index.str.contains('ramp')].index, inplace=True)
    #self.network.links.drop(self.network.links[self.network.links.index.str.contains('ramp')].index, inplace=True)
    logger.info("Start solving grid optimization model")
    self.lopf()

def fix_chp_generation(self):

    # Select generator and link components that are fixed after
    # the market optimization.
    gens_fixed = self.network.generators[
        self.network.generators.carrier.str.endswith("_CHP")].index

    links_fixed = self.network.links[
        self.network.links.carrier.str.endswith("_CHP")].index

    # Fix generator dispatch from market simulation:
    ## Set p_max_pu of generators using results from (disaggregated) market model
    self.network.generators_t.p_max_pu.loc[
        :, gens_fixed
    ] = self.market_model.generators_t.p[gens_fixed].mul(
        1 / self.market_model.generators.p_nom[gens_fixed]
    )

    ## Set p_min_pu of generators using results from (disaggregated) market model
    self.network.generators_t.p_min_pu.loc[
        :, gens_fixed
    ] = self.market_model.generators_t.p[gens_fixed].mul(
        1 / self.market_model.generators.p_nom[gens_fixed]
    )

    # Fix link dispatch (gas turbines) from market simulation
    ## Set p_max_pu of links using results from (disaggregated) market model
    self.network.links_t.p_max_pu.loc[
        :, links_fixed
    ] = self.market_model.links_t.p0[links_fixed].mul(
        1 / self.market_model.links.p_nom[links_fixed]
    )

    ## Set p_min_pu of links using results from (disaggregated) market model
    self.network.links_t.p_min_pu.loc[
        :, links_fixed
    ] = self.market_model.links_t.p0[links_fixed].mul(
        1 / self.market_model.links.p_nom[links_fixed]
    )

def add_redispatch_generators(self):
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
            ]
        )
    ].index

    links_redispatch = self.network.links[
        self.network.links.carrier.isin(["OCGT"])
    ].index

    # Fix generator dispatch from market simulation:
    ## Set p_max_pu of generators using results from (disaggregated) market model
    self.network.generators_t.p_max_pu.loc[
        :, gens_redispatch
    ] = self.market_model.generators_t.p[gens_redispatch].mul(
        1 / self.market_model.generators.p_nom[gens_redispatch]
    )

    ## Set p_min_pu of generators using results from (disaggregated) market model
    self.network.generators_t.p_min_pu.loc[
        :, gens_redispatch
    ] = self.market_model.generators_t.p[gens_redispatch].mul(
        1 / self.market_model.generators.p_nom[gens_redispatch]
    )

    # Fix link dispatch (gas turbines) from market simulation
    ## Set p_max_pu of links using results from (disaggregated) market model
    self.network.links_t.p_max_pu.loc[
        :, links_redispatch
    ] = self.market_model.links_t.p0[links_redispatch].mul(
        1 / self.market_model.links.p_nom[links_redispatch]
    )

    ## Set p_min_pu of links using results from (disaggregated) market model
    self.network.links_t.p_min_pu.loc[
        :, links_redispatch
    ] = self.market_model.links_t.p0[links_redispatch].mul(
        1 / self.market_model.links.p_nom[links_redispatch]
    )

    # Calculate costs for redispatch
    ## Extract prices per market zone from market model results
    market_price_per_bus = self.market_model.buses_t.marginal_price

    # Set market price for each disaggregated generator according to the bus
    market_price_per_generator = market_price_per_bus.loc[
        :, self.market_model.generators.loc[gens_redispatch, "bus"]
    ].median()
    market_price_per_generator.index = gens_redispatch

    # Costs for ramp_up generators are first set the marginal_cost for each
    # generator
    ramp_up_costs = self.network.generators.loc[
        gens_redispatch, "marginal_cost"
    ]

    # In case the market price is higher than the marginal_cost (e.g. for renewables)
    # ram up costs are set to the market price. This way, every generator gets at
    # least the costs at the market. In case the marginal cost are higher, e.g.
    # Because of fuel costs, the real marginal price is payed for redispatch
    ramp_up_costs[
        market_price_per_generator
        > self.network.generators.loc[gens_redispatch, "marginal_cost"]
    ] = market_price_per_generator

    # Costs for ramp down generators consist of the market price
    # which is still payed for the generation. Fuel costs can be saved,
    # therefore the ramp down costs are reduced by the marginal costs
    ramp_down_costs = (
        market_price_per_generator
        - self.network.generators.loc[gens_redispatch, "marginal_cost"]
    )

    # Add ramp up generators to the network for the grid optimization
    # Marginal cost are incread by a management fee of 4 EUR/MWh
    self.network.madd(
        "Generator",
        gens_redispatch + " ramp_up",
        bus=self.network.generators.loc[gens_redispatch, "bus"].values,
        p_nom=self.network.generators.loc[gens_redispatch, "p_nom"].values,
        marginal_cost=ramp_up_costs.values + 4,
        carrier=self.network.generators.loc[gens_redispatch, "carrier"].values,
    )

    # Set maximum feed-in limit for ramp up generators based on feed-in of
    # (disaggregated) generators from the market optimization and potential
    # feedin time series
    self.network.generators_t.p_max_pu.loc[:, gens_redispatch + " ramp_up"] = (
        (
            self.network.generators_t.p_max_pu.loc[:, gens_redispatch].mul(
                self.network.generators.loc[gens_redispatch, "p_nom"]
            )
            - (self.market_model.generators_t.p.loc[:, gens_redispatch])
        )
        .clip(lower=0.0)
        .values
    )

    # Add ramp up links to the network for the grid optimization
    # Marginal cost are incread by a management fee of 4 EUR/MWh
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

    # Set maximum feed-in limit for ramp up links based on feed-in of
    # (disaggregated) links from the market optimization
    self.network.links_t.p_max_pu.loc[:, links_redispatch + " ramp_up"] = (
        (
            self.network.links.loc[links_redispatch, "p_nom"]
            - (self.market_model.links_t.p0.loc[:, links_redispatch])
        )
        .clip(lower=0.0)
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
        marginal_cost=-(ramp_down_costs.values + 4),
        carrier=self.network.generators.loc[gens_redispatch, "carrier"].values,
    )

    # Ramp down generators can not feed-in addtional energy
    self.network.generators_t.p_max_pu.loc[
        :, gens_redispatch + " ramp_down"
    ] = 0.0
    # Ramp down can be at maximum as high as the feed-in of the
    # (disaggregated) generators in the market model
    self.network.generators_t.p_min_pu.loc[
        :, gens_redispatch + " ramp_down"
    ] = (-(self.market_model.generators_t.p.loc[:, gens_redispatch])).values

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
        marginal_cost=-(4),
        carrier=self.network.links.loc[links_redispatch, "carrier"].values,
    )

    # Ramp down links can not feed-in addtional energy
    self.network.links_t.p_max_pu.loc[:, links_redispatch + " ramp_down"] = 0.0

    # Ramp down can be at maximum as high as the feed-in of the
    # (disaggregated) links in the market model
    self.network.links_t.p_min_pu.loc[:, links_redispatch + " ramp_down"] = (
        -(self.market_model.links_t.p0.loc[:, links_redispatch])
    ).values

    # Check if the network contains any problems
    self.network.consistency_check()

    # just for the current status2019 scenario a quick fix for buses which do not have a connection
    #self.network.buses.drop(self.network.buses[self.network.buses.index.isin(['47085', '47086', '37865', '37870'])].index, inplace=True)
    
    # TEMPORAL
    self.network.generators.loc[self.network.generators.index.str.contains('run_of_river'), 'p_max_pu'] = 0.65
    self.network.generators.loc[self.network.generators.index.str.contains('reservoir'), 'p_max_pu'] = 0.65
    

def extra_functionality():
    return None
