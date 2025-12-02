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
electrolysis_results.py defines methods used to calculate results for the
potential atlas for electrolysis grid connection points, which was one of the
main outcomes of the PoWerD-project
"""
import os
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point

if "READTHEDOCS" not in os.environ:    
    from etrago.analyze.calc_results import (
        annualize_capital_costs, 
        electricity_system_costs_germany
        )

    logger = logging.getLogger(__name__)

__copyright__ = (
    "Flensburg University of Applied Sciences, "
    "Europa-Universität Flensburg, "
    "Centre for Sustainable Energy Systems, "
    "DLR-Institute for Networked Energy Systems"
)
__license__ = "GNU Affero General Public License Version 3 (AGPL-3.0)"
__author__ = "ClaraBuettner, lenzim97"


def lcoe_germany(etrago):

    scenario = etrago.network.buses.scn_name.iloc[0]

    generation_capacity_costs = {
        "powerd2025": 42753843903.92459,
        "powerd2030": 43867007802.88958,
        "powerd2035": 42780310000.0,
        "eGon100RE": 39684763603.00243,
    }

    marginal_cost, invest_cost, import_costs = (
        electricity_system_costs_germany(etrago)
    )

    total_system_cost_de = marginal_cost + invest_cost  # + import_costs

    if scenario in generation_capacity_costs.keys():
        total_system_cost_de += generation_capacity_costs[scenario]

    ac_gen_de = (
        etrago.network.generators_t.p[
            etrago.network.generators[
                (
                    etrago.network.generators.bus.isin(
                        etrago.network.buses[
                            (etrago.network.buses.country == "DE")
                            & (etrago.network.buses.carrier == "AC")
                        ].index
                    )
                )
            ].index
        ]
        .sum(axis=1)
        .mul(etrago.network.snapshot_weightings["generators"])
        .sum()
    )

    ac_link_de = etrago.network.links_t.p1[
        etrago.network.links[
            (
                etrago.network.links.bus1.isin(
                    etrago.network.buses[
                        (etrago.network.buses.country == "DE")
                        & (etrago.network.buses.carrier == "AC")
                    ].index
                )
            )
        ].index
    ].sum(axis=1).mul(etrago.network.snapshot_weightings["generators"]).sum() * (
        -1
    )

    lcoe = total_system_cost_de / (ac_gen_de + ac_link_de)

    return lcoe

def regions_per_bus(etrago):
    """
    Create matching dataframe of clustered AC-buses
    and corresponding MV-grids

    Returns
    -------
    geoms : pd.DataFrame
    """

    map_buses = etrago.busmap["orig_network"].buses[
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
        lambda x: Point(x["x"], x["y"]), axis=1
    )

    map_buses["cluster"] = map_buses.index.map(etrago.busmap["busmap"])

    map_buses = gpd.GeoDataFrame(map_buses, geometry="geom")
    try:
        mv_grids = gpd.read_postgis(
            "SELECT bus_id, geom FROM grid.egon_mv_grid_district",
            etrago.engine,
        ).to_crs(4326)
        mv_grids = mv_grids.set_index("bus_id")
        mv_grids.index = mv_grids.index.astype(str)
        map_buses = map_buses[map_buses.index.isin(mv_grids.index)]
        map_buses["geom_grid"] = mv_grids.loc[map_buses.index].buffer(0.0001)

        geoms = gpd.GeoSeries(index=map_buses.cluster.unique())

        for i in map_buses.cluster.unique():
            geoms[i] = map_buses[
                map_buses.cluster == i
            ].geom_grid.unary_union.simplify(0.0001)

        return geoms

    except Exception as e:
        logger.warning(
            "No egon_mv_grid_district table inside the database. "
            "To create a matching table for atlas results "
            "please add this table to your database."
        )
        logger.warning(f"Error-Message: {e}")

        return gpd.GeoSeries(dtype="geometry", crs=4326)


def merit_order_ely_redispatch(etrago):
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
    market_buses = etrago.market_model.buses[
        (etrago.market_model.buses.carrier == "AC")
    ].index

    ely_market = etrago.market_model.links[
        (etrago.market_model.links.carrier == "power_to_H2")
        & (etrago.market_model.links.bus0.isin(market_buses))
    ]

    # Store x, y coordinates for market electrolysis
    ely_market["x"] = etrago.market_model.buses.loc[
        ely_market.bus1.values, "x"
    ].values
    ely_market["y"] = etrago.market_model.buses.loc[
        ely_market.bus1.values, "y"
    ].values

    # Initialize market time series for each bus, grouped by country
    ely_market_t = {}
    for country in etrago.market_model.buses["country"].unique():
        buses_in_country = etrago.market_model.buses[
            etrago.market_model.buses["country"] == country
        ].index
        ely_market_t[country] = etrago.market_model.links_t.p0[
            ely_market.index[ely_market.bus0.isin(buses_in_country)]
        ]

    # Electrolysis in grid optimization
    grid_buses = etrago.network.buses[(etrago.network.buses.carrier == "AC")].index

    ely_grid = etrago.network.links[
        (etrago.network.links.carrier == "power_to_H2")
        & (etrago.network.links.bus0.isin(grid_buses.values))
    ]

    # Store x, y coordinates for grid electrolysis
    ely_grid["x"] = etrago.network.buses.loc[ely_grid.bus0.values, "x"].values
    ely_grid["y"] = etrago.network.buses.loc[ely_grid.bus0.values, "y"].values

    # Initialize grid time series, grouped by country
    ely_grid_t = {}
    for country in etrago.network.buses["country"].unique():
        buses_in_country = etrago.network.buses[
            etrago.network.buses["country"] == country
        ].index
        ely_grid_t[country] = etrago.network.links_t.p0[
            ely_grid.index[ely_grid.bus0.isin(buses_in_country)]
        ]

    # DataFrames for dispatch and redispatch results
    highest_redispatch_price = pd.Series(index=etrago.network.snapshots)
    redispatch_electrolysis = pd.DataFrame(
        index=etrago.network.snapshots, columns=ely_grid.index, data=0.0
    )
    dispatch_electrolysis = pd.DataFrame(
        index=etrago.network.snapshots, columns=ely_grid.index, data=0.0
    )
    redispatch_electrolysis_per_bus = pd.DataFrame(
        index=etrago.network.snapshots,
        columns=etrago.network.buses[(etrago.network.buses.carrier == "AC")].index,
        data=0.0,
    )

    # Main loop: for each snapshot
    for sn in etrago.network.snapshots:

        for country in etrago.market_model.buses["country"].unique():
            market_oriented_dispatch = 0

            market_at_sn = ely_market_t[country].sum(axis=1)[
                sn
            ]  # Get market dispatch for this country

            # Grid dispatch for this country
            grid_at_sn = pd.DataFrame(ely_grid_t[country].loc[sn])

            # Filter bus0 values for the current country
            buses_in_country = etrago.network.buses[
                etrago.network.buses["country"] == country
            ].index
            relevant_buses_in_links = ely_grid.loc[
                ely_grid["bus0"].isin(buses_in_country)
            ]

            # Now we need the marginal prices only for the buses that are in
            # the relevant links for this country
            bus_ids_in_relevant_links = relevant_buses_in_links["bus0"].values

            # Extract the corresponding nodal prices
            nodal_prices = (
                etrago.network.buses_t["marginal_price"]
                .loc[sn, bus_ids_in_relevant_links]
                .values
            )

            # Assign the filtered nodal prices to the grid dispatch DataFrame
            grid_at_sn["nodal_price"] = nodal_prices

            # Sort grid dispatch by price
            ely_dispatch_sorted_by_price = grid_at_sn.sort_values(
                "nodal_price", ascending=False
            )

            for ely in ely_dispatch_sorted_by_price.index:
                if market_at_sn == 0:
                    highest_redispatch_price[sn] = (
                        ely_dispatch_sorted_by_price.iloc[0]["nodal_price"]
                    )
                    redispatch_electrolysis_per_bus.loc[
                        sn, etrago.network.links.loc[ely, "bus0"]
                    ] += ely_dispatch_sorted_by_price.loc[ely, sn]
                else:
                    if market_at_sn > market_oriented_dispatch:
                        if market_at_sn >= (
                            market_oriented_dispatch
                            + ely_dispatch_sorted_by_price.loc[ely, sn]
                        ):
                            market_oriented_dispatch += (
                                ely_dispatch_sorted_by_price.loc[ely, sn]
                            )
                            dispatch_electrolysis.loc[sn, ely] = (
                                ely_dispatch_sorted_by_price.loc[ely, sn]
                            )
                        else:
                            dispatch_electrolysis.loc[sn, ely] = (
                                market_at_sn - market_oriented_dispatch
                            )
                            market_oriented_dispatch += (
                                dispatch_electrolysis.loc[sn, ely]
                            )
                            redispatch_electrolysis.loc[sn, ely] = (
                                ely_dispatch_sorted_by_price.loc[ely, sn]
                                - dispatch_electrolysis.loc[sn, ely]
                            )
                            redispatch_electrolysis_per_bus.loc[
                                sn, etrago.network.links.loc[ely, "bus0"]
                            ] += (
                                ely_dispatch_sorted_by_price.loc[ely, sn]
                                - dispatch_electrolysis.loc[sn, ely]
                            )
                    else:
                        redispatch_electrolysis.loc[sn, ely] = (
                            ely_dispatch_sorted_by_price.loc[ely, sn]
                        )
                        redispatch_electrolysis_per_bus.loc[
                            sn, etrago.network.links.loc[ely, "bus0"]
                        ] += ely_dispatch_sorted_by_price.loc[ely, sn]
                        highest_redispatch_price[sn] = (
                            ely_dispatch_sorted_by_price.loc[
                                ely, "nodal_price"
                            ]
                        )

    # matching table bus_id | corresponding mv grids
    mv_grid_geom = regions_per_bus(etrago)

    if not mv_grid_geom.empty:
        df_mv_grids = gpd.GeoDataFrame(geometry=mv_grid_geom, crs=4326)
    else:
        df_mv_grids = gpd.GeoDataFrame(geometry=[], crs=4326)

    return (
        redispatch_electrolysis,
        redispatch_electrolysis_per_bus,
        df_mv_grids,
    )


def remaining_redispatch(etrago, min_flh=3000):
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
        index=etrago.network.snapshots,
        columns=etrago.network.buses[(etrago.network.buses.carrier == "AC")].index,
        data=0.0,
    )

    for bus in ramp_down_per_bus.columns:

        ramp_down_per_bus[bus] += (
            etrago.network.generators_t.p[
                etrago.network.generators[
                    (etrago.network.generators.bus == bus)
                    & (etrago.network.generators.index.str.contains("ramp_down"))
                ].index
            ]
            .sum(axis=1)
            .abs()
        )

        ramp_down_per_bus[bus] += (
            etrago.network.links_t.p1[
                etrago.network.links[
                    (etrago.network.links.bus1 == bus)
                    & (etrago.network.links.index.str.contains("ramp_down"))
                ].index
            ]
            .sum(axis=1)
            .abs()
        )

    max_ely = pd.DataFrame(
        index=etrago.network.buses[(etrago.network.buses.carrier == "AC")].index,
        columns=["max_capacity", "x", "y"],
        data={
            "max_capacity": 0.0,
            "x": etrago.network.buses[(etrago.network.buses.carrier == "AC")].x,
            "y": etrago.network.buses[(etrago.network.buses.carrier == "AC")].y,
        },
    )

    for size in range(1, 200):
        for bus in max_ely.index:
            if (
                (ramp_down_per_bus.loc[:, bus].clip(upper=size)).sum() * (5)
            ) >= (min_flh * size):
                max_ely.loc[bus, "max_capacity"] = size

    return max_ely, ramp_down_per_bus


def calc_atlas_results(etrago, filename=None):
    """
    Calculating the final results for the potential_atlas as
    one of the main outcomes of the PoWerD-project. The results will
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
    

    results = pd.DataFrame()

    heating_value_H2 = 33.33  # [kWh/kg]
    # average value produced O2 per electricity, own calculation
    O2_calc_factor = 9.030816  # [t_O2/MWh_el]

    max_ely, ramp_down_per_bus = remaining_redispatch(etrago)
    (
        redispatch_electrolysis,
        redispatch_electrolysis_per_bus,
        matching_mv_grids,
    ) = merit_order_ely_redispatch(etrago)

    PtH2_links = etrago.network.links[
        (etrago.network.links.carrier == "power_to_H2")
        & (
            etrago.network.links.bus0.isin(
                etrago.network.buses[etrago.network.buses.country == "DE"].index
            )
        )
    ]
    PtH2_links = PtH2_links[PtH2_links.p_nom_opt > 10]
    AC_buses_PtH2 = etrago.network.buses[
        etrago.network.buses.index.isin(PtH2_links.bus0.unique())
    ]

    # Calculate CAPEX
    p = 0.05
    scenario = etrago.network.buses.scn_name.iloc[0]
    lt_system = {
        "powerd2025": 20,
        "powerd2030": 25,
        "powerd2035": 25,
        "eGon100RE": 30,
    }
    lt = lt_system[scenario]
    # cost that are not included in clean CAPEX
    OPEX_STACK = 0.03 * 0.21 * 357_000
    OPEX_SYSTEM = 0.03 * 357_000
    OPEX_PIPES = 0.03 * 236
    an_capex_stack = annualize_capital_costs(
        0.21 * 357_000, 20, 0.07
    )  # interest rate for gas_sector 0.07

    for index, row in AC_buses_PtH2.iterrows():

        links_PtH2 = PtH2_links[PtH2_links.bus0 == index]

        if "H2" in etrago.network.buses.loc[links_PtH2.bus1, "carrier"].unique():
            at_h2_grid = False
        if (
            "H2_grid"
            in etrago.network.buses.loc[links_PtH2.bus1, "carrier"].unique()
        ):
            at_h2_grid = True

        # calculation for multiple_link_model
        if etrago.args["method"]["formulation"] == "linopy":

            # Check if elctrolyzer has coupling product usage
            links_PtH2_bus2 = (
                links_PtH2["bus2"].replace(["", "nan", None], np.nan).dropna()
            )
            links_PtH2_bus3 = (
                links_PtH2["bus3"].replace(["", "nan", None], np.nan).dropna()
            )

            buses_heat = (
                links_PtH2_bus2.astype(float).astype(int).astype(str).tolist()
            )
            buses_o2 = (
                links_PtH2_bus3.astype(float).astype(int).astype(str).tolist()
            )

            if buses_heat:
                links_waste_heat = etrago.network.links[
                    etrago.network.links.bus0.isin(buses_heat)
                ]
            else:
                links_waste_heat = []

            if buses_o2:
                links_o2 = etrago.network.links[
                    etrago.network.links.bus0.isin(buses_o2)
                ]
            else:
                links_o2 = []

        else:  # calculation for generator model
            link_indices = links_PtH2.index.astype(str)

            # Filter out corresponding o2 and heat generators
            gen_o2 = etrago.network.generators[
                etrago.network.generators.index.isin(
                    [f"{link_index}_O2" for link_index in link_indices]
                )
            ]
            gen_heat = etrago.network.generators[
                etrago.network.generators.index.isin(
                    [f"{link_index}_waste_heat" for link_index in link_indices]
                )
            ]

            if not gen_o2.empty:
                bus_o2 = gen_o2.bus.iloc[0]
                links_o2 = etrago.network.links[
                    etrago.network.links.bus0 == bus_o2
                ]
            else:
                links_o2 = []

            if not gen_heat.empty:
                bus_heat = gen_heat.bus.iloc[0]
                links_waste_heat = etrago.network.links[
                    etrago.network.links.bus0 == bus_heat
                ]
            else:
                links_waste_heat = []

        # Calculate Dispatch
        AC_dispatch = (
            etrago.network.links_t.p0[links_PtH2.index]
            .mul(etrago.network.snapshot_weightings.objective, axis=0)
            .sum()
            .sum()
        )
        H2_dispatch = (
            -etrago.network.links_t.p1[links_PtH2.index]
            .mul(etrago.network.snapshot_weightings.objective, axis=0)
            .sum()
            .sum()
        )
        waste_heat_dispatch = (
            -etrago.network.links_t.p1.get(links_waste_heat.index, pd.Series(0))
            .mul(etrago.network.snapshot_weightings.objective, axis=0)
            .sum()
            .sum()
        )
        o2_dispatch = (
            -etrago.network.links_t.p1.get(links_o2.index, pd.Series(0))
            .mul(etrago.network.snapshot_weightings.objective, axis=0)
            .sum()
            .sum()
        )
        # LCOE+LCOH
        sn = etrago.network.snapshots[
            (etrago.network.links_t.p0[links_PtH2.index].sum(axis=1) > 10)
        ]
        mean_local_cost = etrago.network.buses_t.marginal_price.loc[
            sn, row.name
        ].mean()  # [€/MWh_e]

        lcoh = (
            (
                lcoe_germany(etrago)
                * (1 / etrago.network.links.efficiency[links_PtH2.index])
                + (links_PtH2.capital_cost * links_PtH2.p_nom_opt).sum()
                / H2_dispatch
            ).mean()
            * 33.33
            * 1e-3
        )  # [€/kg_H2]

        # H2-demand
        loads_h2 = etrago.network.loads[
            etrago.network.loads.carrier.str.contains("H2")
            & etrago.network.loads.bus.isin(
                links_PtH2.bus1.astype(int).astype(str).tolist()
            )
        ]
        try:
            H2_demand = (
                etrago.network.loads_t.p_set[loads_h2.index]
                .mul(etrago.network.snapshot_weightings.objective, axis=0)
                .sum()
                .sum()
            )
        except:
            H2_demand = etrago.network.loads.p_set[loads_h2.index].sum()

        # store_capacity
        stores_h2 = etrago.network.stores[
            etrago.network.stores.bus.isin(
                links_PtH2.bus1.astype(int).astype(str).tolist()
            )
        ]
        store_cap = stores_h2.e_nom_opt.sum()

        # redispatch
        ramp_down = ramp_down_per_bus.mul(
            etrago.network.snapshot_weightings.objective, axis=0
        ).sum(axis=0)[index]
        redispatch_ely = redispatch_electrolysis_per_bus.mul(
            etrago.network.snapshot_weightings.objective, axis=0
        ).sum(axis=0)[index]

        # specific costs for ELY
        capex_ely = (
            links_PtH2.capital_cost.mean()
            - OPEX_PIPES
            - OPEX_STACK
            - OPEX_SYSTEM
            - an_capex_stack
        ) * ((1 / p) - (1 / (p * (1 + p) ** lt)))

        # market_driven/grid_driven
        if redispatch_ely < 1e5:
            dispatch_type = "market_driven"
        else:
            dispatch_type = "grid_driven"

        new_row = {
            "region": row.name,
            "Placement": "System optimization",
            "E": row.x,
            "N": row.y,
            "Type": dispatch_type,
            "Max. electrolyzer capacity [MW]": links_PtH2.p_nom_opt.sum(),
            "Max. electricity consumption [MWh]": AC_dispatch,
            "ELY investment cost [€/kW]": capex_ely / 1000,
            "Max. H2-Production [ton/a]": H2_dispatch / heating_value_H2,
            "Max. heat supply [MWh/a]": waste_heat_dispatch,
            "Max. O2 supply [ton/a]": o2_dispatch * O2_calc_factor,
            "LCOH [€/kg_H2]": lcoh,
            "Mean nodal electricity cost [€/MWh_el]": mean_local_cost,
            "Max. redispatch by electrolysis [MWh/a]": redispatch_ely,
            "Remaining redispatch [MWh/a]": ramp_down,
            "Max. redispatch potential": ramp_down + redispatch_ely,
            "H2 demand [ton/a]": H2_demand,
            "Max. hydrogen storage capacity": store_cap,
            "At hydrogen grid": at_h2_grid,
        }
        new_row_df = pd.DataFrame([new_row])
        results = pd.concat([results, new_row_df], ignore_index=True)

    # additional ely potential calculated out of remaining redispacth
    for bus, row in max_ely[max_ely["max_capacity"] > 0].iterrows():
        new_row = {
            "region": bus,
            "Placement": "Additional redispatch potential",
            "E": row.x,
            "N": row.y,
            "Type": "grid_driven",
            "Max. electrolyzer capacity [MW]": row.max_capacity,
            "Max. electricity consumption [MWh]": None,
            "ELY investment cost [€/MW]": None,
            "Max. H2-Production [ton/a]": None,
            "Max. heat supply [MWh/a]": None,
            "Max. O2 supply [ton/a]": None,
            "LCOH [€/kg_H2]": None,
            "LCOE [€/MWh_el]": None,
            "Max. redispatch by electrolysis [MWh/a]": None,
            "Remaining redispatch [MWh/a]": None,
            "Max. redispatch potential": None,
            "H2 demand [ton/a]": None,
            "Max. hydrogen storage capacity": None,
            "At hydrogen grid": at_h2_grid,
        }
        results = pd.concat(
            [results, pd.DataFrame([new_row])], ignore_index=True
        )

    if filename:
        results.to_csv(filename)
        matching_mv_grids.to_file(f"regions_{scenario}.geojson")

    return results