# -*- coding: utf-8 -*-
# Copyright 2016-2026  Flensburg University of Applied Sciences,
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
Defines function to seperate transmission grid from distribution grid in eTraGo
"""

import saio
import pandas as pd
from sqlalchemy import func
from shapely.geometry import Point
import geopandas as gpd


def get_ac_nodes_germany(self):
    """
    Filters AC buses within Germany

    Returns
    -------
    pd.Series
        Indices of AC buses within Germany.

    """
    return self.network.buses[
        (self.network.buses.carrier == "AC")
        & (self.network.buses.country == "DE")
    ].index


def distribution_grid_buses_and_links(self, mv_grids, seperate_dg_link=True):
    """Adds buses and links modeling simplified distribution grids.
    The parametrization is derived from bottom-up simulations of distribution
    grids with eDisGo/eGo.

    Parameters
    ----------
    mv_grids : pd.DataFrame
        Medium voltage grid districts.
    seperate_dg_link : boolean, optional
        Select if distribution grids are modeled as one bi-directional link or
        as two coupled uni-directional link. The default is True.

    Returns
    -------
    None.

    """
    # Create distribution grid (DG) nodes
    self.network.madd(
        "Bus",
        names=mv_grids.bus_id.astype(str) + "_distribution_grid",
        carrier="distribution_grid",
        country="DE",
        x=self.network.buses.loc[mv_grids.bus_id.astype(str), "x"].values,
        y=self.network.buses.loc[mv_grids.bus_id.astype(str), "y"].values,
    )

    edisgo_results = pd.read_csv(
        self.args["method"]["distribution_grids"]
    ).set_index("bus_id")

    # Create link between transmission an distribution grid
    # Either one bidirectional link is created or two unidirectional.
    # Two links allow setting different starting capacities for load and
    # feedin case in the distribution grid. The expansion is coupled end
    # always extends the capacity in both directions.
    if seperate_dg_link:
        self.network.madd(
            "Link",
            names=mv_grids.bus_id.astype(str) + "_to_distribution_grid",
            carrier="distribution_grid",
            bus0=mv_grids.bus_id.astype(str).values,
            bus1=(mv_grids.bus_id.astype(str) + "_distribution_grid").values,
            p_nom_min=edisgo_results.loc[mv_grids.bus_id, "p_nom_load"].values,
            p_nom_max=(
                edisgo_results.loc[mv_grids.bus_id, "p_nom_load"]
                + (edisgo_results.loc[mv_grids.bus_id, "p_nom_load"] * 4).clip(
                    lower=200.0
                )
            ).values,
            p_nom_extendable=True,
            capital_cost=edisgo_results.loc[
                mv_grids.bus_id, "capital_cost_worst_case"
            ].values,
            marginal_cost=0.1,
            p_min_pu=0,
        )

        self.network.madd(
            "Link",
            names=mv_grids.bus_id.astype(str) + "_from_distribution_grid",
            carrier="distribution_grid",
            bus0=(mv_grids.bus_id.astype(str) + "_distribution_grid").values,
            bus1=mv_grids.bus_id.astype(str).values,
            p_nom_min=edisgo_results.loc[mv_grids.bus_id, "p_nom_feedin"]
            .abs()
            .values,
            p_nom_max=(
                edisgo_results.loc[mv_grids.bus_id, "p_nom_feedin"].abs()
                + (edisgo_results.loc[mv_grids.bus_id, "p_nom_load"] * 4).clip(
                    lower=200.0
                )
            ).values,
            p_nom_extendable=True,
            capital_cost=0,
            marginal_cost=0.1,
            p_min_pu=0,
        )
    else:
        self.network.madd(
            "Link",
            names=mv_grids.bus_id.astype(str) + "_distribution_grid",
            carrier="distribution_grid",
            bus0=mv_grids.bus_id.astype(str).values,
            bus1=(mv_grids.bus_id.astype(str) + "_distribution_grid").values,
            p_nom_min=edisgo_results.loc[mv_grids.bus_id, "p_nom_load"].values,
            p_nom_max=(edisgo_results.loc[mv_grids.bus_id, "p_nom_load"] * 4)
            .clip(lower=200.0)
            .values,
            p_nom_extendable=True,
            capital_cost=edisgo_results.loc[
                mv_grids.bus_id, "capital_cost_worst_case"
            ].values,
            p_min_pu=-1,
        )


def seperate_power_plants(self, egon_power_plants, old_network):
    """
    Divides power plants by grid level (transmission or distribution grid) and
    connects them to the corresponding bus.

    Parameters
    ----------
    egon_power_plants : sqlalchemy.ext.declarative.api.DeclarativeMeta
        Pointer to table containing non-aggregated power plants.
    old_network : pypsa.Network
        Previous network without adjustments.

    Returns
    -------
    None.

    """
    ac_nodes_germany = get_ac_nodes_germany(self)
    # Import power plants table
    power_plants = saio.as_pandas(
        query=(
            self.session.query(egon_power_plants)
            .filter(egon_power_plants.scenario == "eGon2035")
            .filter(egon_power_plants.carrier != "gas")
        )
    )
    # Drop power plants in Germany
    self.network.mremove(
        "Generator",
        self.network.generators[
            (self.network.generators.bus.isin(ac_nodes_germany))
            & (
                self.network.generators.carrier.isin(
                    power_plants.carrier.unique()
                )
            )
        ].index,
    )

    # Import generators in transmission grid
    tg_generators = (
        power_plants[power_plants.voltage_level < 4]
        .groupby(["bus_id", "carrier"])
        .el_capacity.sum()
        .reset_index()
    )

    self.network.madd(
        "Generator",
        names=(
            tg_generators.bus_id.astype(str) + "_" + tg_generators.carrier
        ).values,
        bus=tg_generators.bus_id.astype(str).values,
        carrier=tg_generators.carrier.values,
        p_nom=tg_generators.el_capacity.values,
        marginal_cost=old_network.generators.groupby("carrier")
        .marginal_cost.mean()
        .loc[tg_generators.carrier]
        .values,
    )

    # Import generators in distribution grid
    dg_generators = (
        power_plants[power_plants.voltage_level >= 4]
        .groupby(["bus_id", "carrier"])
        .el_capacity.sum()
        .reset_index()
    )
    self.network.madd(
        "Generator",
        names=(
            "distribution_grid_"
            + dg_generators.bus_id.astype(str)
            + "_"
            + dg_generators.carrier
        ).values,
        bus=(dg_generators.bus_id.astype(str) + "_distribution_grid").values,
        carrier=dg_generators.carrier.values,
        p_nom=dg_generators.el_capacity.values,
        marginal_cost=old_network.generators.groupby("carrier")
        .marginal_cost.mean()
        .loc[dg_generators.carrier]
        .values,
    )
    # Add solar rooftop, that is not in the power plants tables
    self.network.mremove(
        "Generator",
        self.network.generators[
            (self.network.generators.bus.isin(ac_nodes_germany))
            & (self.network.generators.carrier == "solar_rooftop")
        ].index,
    )

    solar_rooftop = old_network.generators[
        (old_network.generators.carrier == "solar_rooftop")
        & (old_network.generators.bus.isin(ac_nodes_germany))
    ]

    self.network.madd(
        "Generator",
        names=(
            "distribution_grid_" + solar_rooftop.bus + "_solar_rooftop"
        ).values,
        bus=(solar_rooftop.bus.astype(str) + "_distribution_grid").values,
        carrier=solar_rooftop.carrier.values,
        p_nom=solar_rooftop.p_nom.values,
        marginal_cost=old_network.generators.groupby("carrier")
        .marginal_cost.mean()
        .loc["solar_rooftop"],
    )

    ## Generation timeseries
    # Store timeseries per carrier and bus
    p_max_pu_ts = {}
    for c in ["solar", "wind_onshore", "wind_offshore", "solar_rooftop"]:
        p_max_pu_ts[c] = old_network.generators_t.p_max_pu[
            old_network.generators[old_network.generators.carrier == c].index
        ]
        p_max_pu_ts[c].rename(
            old_network.generators.bus, axis="columns", inplace=True
        )

        dg_generators_carrier = dg_generators[dg_generators.carrier == c]

        if c == "solar_rooftop":
            dg_generators_carrier = (
                solar_rooftop.groupby(["bus", "carrier"])
                .p_nom.sum()
                .reset_index()
            )
            dg_generators_carrier.loc[:, "bus_id"] = dg_generators_carrier.loc[
                :, "bus"
            ]

        self.network.generators_t.p_max_pu.loc[
            :,
            "distribution_grid_"
            + dg_generators_carrier.bus_id.astype(str)
            + "_"
            + dg_generators_carrier.carrier,
        ] = p_max_pu_ts[c][
            dg_generators_carrier.bus_id.astype(str).values
        ].values

        tg_generators_carrier = tg_generators[tg_generators.carrier == c]
        self.network.generators_t.p_max_pu.loc[
            :,
            tg_generators_carrier.bus_id.astype(str)
            + "_"
            + tg_generators_carrier.carrier,
        ] = p_max_pu_ts[c][
            tg_generators_carrier.bus_id.astype(str).values
        ].values


def seperate_chp(self, egon_chp_plants, egon_district_heating_areas):
    """
    Divides CHP plants by grid level (transmission or distribution grid) and
    connects them to the corresponding bus.

    Parameters
    ----------
    egon_chp_plants : sqlalchemy.ext.declarative.api.DeclarativeMeta
        Pointer to table containing non-aggregated CHP plants.
    egon_district_heating_areas : sqlalchemy.ext.declarative.api.DeclarativeMeta
        Pointer to table containing district heating areas.

    Returns
    -------
    None

    """
    ac_nodes_germany = get_ac_nodes_germany(self)

    def drop_chp(network, ac_nodes):
        """
        Drop current CHP plants in Germany

        Parameters
        ----------
        network : pypsa.Network
            Network container
        ac_nodes : pd.Series
            List of AC nodes within Germany.

        Returns
        -------
        None.

        """
        gen = network.generators
        link = network.links

        network.mremove(
            "Generator",
            gen[
                (gen.bus.isin(ac_nodes) & gen.carrier.str.endswith("CHP"))
                | gen.carrier.str.endswith("CHP_heat")
            ].index,
        )

        network.mremove(
            "Link",
            link[
                (link.bus1.isin(ac_nodes) & link.carrier.str.endswith("CHP"))
                | link.carrier.str.endswith("CHP_heat")
            ].index,
        )

    drop_chp(self.network, ac_nodes_germany)

    def dg_bus(bus_id):
        """
        Select corresponting distribution grid node

        Parameters
        ----------
        bus_id : Index
            Transmission grid bus.

        Returns
        -------
        str
            Distribution grid bus.

        """
        return bus_id.astype(str) + "_distribution_grid"

    def add_generators(df, carrier, bus, p_nom, mc, suffix):
        """
        Add CHP as generator component

        Parameters
        ----------
        df : pd.DataFrame
            CHP plants that should be modeled as generator.
        carrier : str
            Energy carrier.
        bus : str
            Bus the CHP is connected to.
        p_nom : float
            Nominal power.
        mc : float
            Marginal cost.
        suffix : str
            Addition to index.

        Returns
        -------
        None.

        """
        if df.empty:
            return
        self.network.madd(
            "Generator",
            names=(df.index.astype(str) + suffix).values,
            bus=bus,
            carrier=carrier,
            p_nom=p_nom,
            marginal_cost=mc,
        )

    def add_links(df, carrier, bus0, bus1, p_nom, mc, suffix, efficiency=1.0):
        """
        Add CHP as link component

        Parameters
        ----------
        df : pd.DataFrame
            CHP plants that should be modeled as link.
        carrier : str
            Energy carrier.
        bus0 : str
            CH4-bus the CHP is connected to.
        bus1 : str
            Electrical or heat bus the CHP is connected to.
        p_nom : float
            Nominal power.
        mc : float
            Marginal cost.
        suffix : str
            Addition to index.
        efficiency : float, optional
            Efficiency of bus01 to bus1. The default is 1.0.

        Returns
        -------
        None.

        """
        if df.empty:
            return
        self.network.madd(
            "Link",
            names=(df.index.astype(str) + suffix).values,
            bus0=bus0,
            bus1=bus1,
            carrier=carrier,
            p_nom=p_nom,
            efficiency=efficiency,
            marginal_cost=mc,
        )

    def select_chp(scenario):
        """
        Select and prepare CHP plants from the database.

        Parameters
        ----------
        scenario : str
            Scenario name.

        Returns
        -------
        chp_plants : pd.DataFrame
            CHP plants in the specific scenario.

        """
        chp_plants = saio.as_pandas(
            query=self.session.query(egon_chp_plants).filter(
                egon_chp_plants.scenario == scenario
            )
        )

        areas = saio.as_pandas(
            self.session.query(
                egon_district_heating_areas.area_id,
                func.ST_X(
                    func.ST_Transform(
                        func.ST_Centroid(
                            egon_district_heating_areas.geom_polygon
                        ),
                        4326,
                    )
                ).label("centroid_x"),
                func.ST_Y(
                    func.ST_Transform(
                        func.ST_Centroid(
                            egon_district_heating_areas.geom_polygon
                        ),
                        4326,
                    )
                ).label("centroid_y"),
            ).filter(egon_district_heating_areas.scenario == "eGon2035")
        )
        areas["geometry"] = areas.apply(
            lambda row: Point(row.centroid_x, row.centroid_y), axis=1
        )

        areas_gdf = gpd.GeoDataFrame(
            areas,
            geometry="geometry",
            crs="EPSG:4326",  # match your bus coordinates
        )
        # Only central_heat buses
        buses = self.network.buses[
            self.network.buses.carrier == "central_heat"
        ].copy()
        buses["bus_id"] = buses.index

        # Create Shapely geometry
        buses["geom"] = buses.apply(lambda row: Point(row.x, row.y), axis=1)

        buses_gdf = gpd.GeoDataFrame(
            buses,
            geometry="geom",
            crs="EPSG:4326",  # match your bus coordinates
        )

        areas_gdf["geometry"] = areas_gdf["geometry"].buffer(0.0001)

        district_heating_mapping = areas_gdf.sjoin(buses_gdf)[
            ["area_id", "bus_id"]
        ].set_index("area_id")

        chp_plants["heating_bus"] = 0
        chp_plants.loc[
            chp_plants.district_heating_area_id.notnull(), "heating_bus"
        ] = (
            district_heating_mapping.loc[
                chp_plants[chp_plants.district_heating_area_id.notnull()]
                .district_heating_area_id.astype(int)
                .values
            ]
            .bus_id.astype(int)
            .values
        )
        return chp_plants

    # Select CHP plants from database and add heating bus
    chp_plants = select_chp("eGon2035")

    # Group CHPs into all different categories
    is_tg = chp_plants.voltage_level < 4
    is_dg = ~is_tg
    is_gas = chp_plants.ch4_bus_id.notnull()
    has_heat = chp_plants.heating_bus != 0

    groups = {
        "tg_gen_heat": chp_plants[is_tg & ~is_gas & has_heat],
        "tg_gen_wo": chp_plants[is_tg & ~is_gas & ~has_heat],
        "dg_gen_heat": chp_plants[is_dg & ~is_gas & has_heat],
        "dg_gen_wo": chp_plants[is_dg & ~is_gas & ~has_heat],
        "tg_link_heat": chp_plants[is_tg & is_gas & has_heat],
        "tg_link_wo": chp_plants[is_tg & is_gas & ~has_heat],
        "dg_link_heat": chp_plants[is_dg & is_gas & has_heat],
        "dg_link_wo": chp_plants[is_dg & is_gas & ~has_heat],
    }

    # Add all kinds of CHP plants
    add_generators(
        groups["tg_gen_wo"],
        "industrial_biomass_CHP",
        groups["tg_gen_wo"].electrical_bus_id.astype(str).values,
        groups["tg_gen_wo"].el_capacity.values,
        42.1,
        "_chp",
    )

    add_generators(
        groups["tg_gen_heat"],
        "central_biomass_CHP",
        groups["tg_gen_heat"].electrical_bus_id.astype(str).values,
        groups["tg_gen_heat"].el_capacity.values,
        42.1,
        "_chp",
    )

    add_generators(
        groups["tg_gen_heat"],
        "central_biomass_CHP_heat",
        groups["tg_gen_heat"].heating_bus.astype(str).values,
        groups["tg_gen_heat"].th_capacity.values,
        0.0,
        "_chp_heat",
    )

    add_generators(
        groups["dg_gen_wo"],
        "industrial_biomass_CHP",
        dg_bus(groups["dg_gen_wo"].electrical_bus_id).values,
        groups["dg_gen_wo"].el_capacity.values,
        42.1,
        "_chp",
    )

    add_generators(
        groups["dg_gen_heat"],
        "central_biomass_CHP",
        dg_bus(groups["dg_gen_heat"].electrical_bus_id).values,
        groups["dg_gen_heat"].el_capacity.values,
        42.1,
        "_chp",
    )

    add_generators(
        groups["dg_gen_heat"],
        "central_biomass_CHP_heat",
        groups["dg_gen_heat"].heating_bus.astype(str).values,
        groups["dg_gen_heat"].th_capacity.values,
        0.0,
        "_chp_heat",
    )

    add_links(
        groups["tg_link_wo"],
        "industrial_gas_CHP",
        groups["tg_link_wo"].ch4_bus_id.astype(int).astype(str).values,
        groups["tg_link_wo"].electrical_bus_id.astype(str).values,
        groups["tg_link_wo"].el_capacity.values,
        4.15,
        "_chp",
    )

    add_links(
        groups["tg_link_heat"],
        "central_gas_CHP",
        groups["tg_link_heat"].ch4_bus_id.astype(int).astype(str).values,
        groups["tg_link_heat"].electrical_bus_id.astype(str).values,
        groups["tg_link_heat"].el_capacity.mul(1 / 0.2794561726625968).values,
        4.15,
        "_chp",
        efficiency=0.2794561726625968,
    )

    add_links(
        groups["tg_link_heat"],
        "central_gas_CHP_heat",
        groups["tg_link_heat"].ch4_bus_id.astype(int).astype(str).values,
        groups["tg_link_heat"].heating_bus.astype(str).values,
        groups["tg_link_heat"].th_capacity.values,
        0.0,
        "_chp_heat",
    )

    add_links(
        groups["dg_link_wo"],
        "industrial_gas_CHP",
        groups["dg_link_wo"].ch4_bus_id.astype(int).astype(str).values,
        dg_bus(groups["dg_link_wo"].electrical_bus_id).values,
        groups["dg_link_wo"].el_capacity.values,
        4.15,
        "_chp",
    )

    add_links(
        groups["dg_link_heat"],
        "central_gas_CHP",
        groups["dg_link_heat"].ch4_bus_id.astype(int).astype(str).values,
        dg_bus(groups["dg_link_heat"].electrical_bus_id).values,
        groups["dg_link_heat"].el_capacity.mul(1 / 0.2794561726625968).values,
        4.15,
        "_chp",
        efficiency=0.2794561726625968,
    )

    add_links(
        groups["dg_link_heat"],
        "central_gas_CHP_heat",
        groups["dg_link_heat"].ch4_bus_id.astype(int).astype(str).values,
        groups["dg_link_heat"].heating_bus.astype(str).values,
        groups["dg_link_heat"].th_capacity.values,
        0.0,
        "_chp_heat",
    )


def seperate_demands(
    self,
    mv_grids,
    egon_osm_ind_load_curves_individual,
    egon_sites_ind_load_curves_individual,
):
    """
    Divides electrical loads by grid level (transmission or distribution grid)
    and connects them to the corresponding bus.

    Parameters
    ----------
    mv_grids : pd.DataFrame
        Medium voltage grid districts.
    egon_osm_ind_load_curves_individual :
        sqlalchemy.ext.declarative.api.DeclarativeMeta
        Pointer to table containing industrial load curves at OSM areas.
    egon_sites_ind_load_curves_individual :
        sqlalchemy.ext.declarative.api.DeclarativeMeta
        Pointer to table containing industrial load curves at points.

    Returns
    -------
    None.

    """
    # Import industrial demands attached to transmission grid
    hv_ind_loads = pd.concat(
        [
            saio.as_pandas(
                query=self.session.query(
                    egon_sites_ind_load_curves_individual.bus_id,
                    egon_sites_ind_load_curves_individual.p_set,
                ).filter(
                    egon_sites_ind_load_curves_individual.scn_name
                    == "eGon2035",
                    egon_sites_ind_load_curves_individual.voltage_level < 4,
                )
            ),
            saio.as_pandas(
                query=self.session.query(
                    egon_osm_ind_load_curves_individual.bus_id,
                    egon_osm_ind_load_curves_individual.p_set,
                ).filter(
                    egon_osm_ind_load_curves_individual.scn_name == "eGon2035",
                    egon_osm_ind_load_curves_individual.voltage_level < 4,
                )
            ),
        ]
    )

    # Slice industrail loads to selected snapshots and group by bus
    hv_ind_loads_p = (
        hv_ind_loads.set_index("bus_id")["p_set"]
        .apply(pd.Series)
        .transpose()[
            self.args["start_snapshot"] - 1 : self.args["end_snapshot"]
        ]
    )

    hv_ind_loads_p = (
        hv_ind_loads_p.transpose()
        .reset_index()
        .groupby("bus_id")
        .sum()
        .transpose()
    )

    hv_ind_loads_p.index = self.network.loads_t.p_set.index

    # Reduce current load
    for c in hv_ind_loads_p.columns:
        dg_load = self.network.loads[
            (self.network.loads.bus == str(c))
            & (self.network.loads.carrier == "AC")
        ].index[0]
        self.network.loads_t.p_set.loc[:, dg_load] -= hv_ind_loads_p.loc[:, c]

    # Connect existing load to distribution grid
    self.network.loads.loc[
        (self.network.loads.carrier == "AC")
        & self.network.loads.bus.isin(mv_grids.bus_id.astype(str)),
        "bus",
    ] += "_distribution_grid"

    # Add loads at transmission grid
    self.network.madd(
        "Load",
        names=hv_ind_loads_p.columns.astype(str) + "_transmission_grid",
        bus=hv_ind_loads_p.columns.values,
        carrier="AC",
        scn_name="eGon2035",
    )

    self.network.loads_t.p_set.loc[
        :, hv_ind_loads_p.columns.astype(str) + "_transmission_grid"
    ] = (hv_ind_loads_p).values

    ## Connect rural heat and BEV charger to distribution grids
    self.network.links.loc[
        self.network.links.carrier == "rural_heat_pump", "bus0"
    ] = (
        self.network.links.loc[
            self.network.links.carrier == "rural_heat_pump", "bus0"
        ]
        + "_distribution_grid"
    )

    self.network.links.loc[
        self.network.links.carrier == "BEV_charger", "bus0"
    ] = (
        self.network.links.loc[
            self.network.links.carrier == "BEV_charger", "bus0"
        ]
        + "_distribution_grid"
    )

    if "lowflex" in self.args["scn_name"]:
        self.network.loads.loc[
            self.network.loads.carrier == "land_transport_EV", "bus"
        ] = (
            self.network.loads.loc[
                self.network.loads.carrier == "land_transport_EV", "bus"
            ]
            + "_distribution_grid"
        )


def seperate_storage_units(self, mv_grids):
    """
    Divides storage units by grid level (transmission or distribution grid)
    and connects them to the corresponding bus.

    Parameters
    ----------
    mv_grids : pd.DataFrame
        Medium voltage grid districts.

    Returns
    -------
    None.

    """
    ac_nodes_germany = get_ac_nodes_germany(self)

    # Add PV home storage units to distribution grid
    battery_storages = self.network.storage_units[
        (self.network.storage_units.carrier == "battery")
        & (self.network.storage_units.bus.isin(ac_nodes_germany))
    ]
    self.network.madd(
        "StorageUnit",
        names=(
            battery_storages[
                battery_storages.bus.isin(mv_grids.bus_id.astype(str))
            ].bus
            + "_home_storage"
        ).values,
        bus=(
            battery_storages.set_index("bus")
            .loc[mv_grids.bus_id.astype(str)]
            .index
            + "_distribution_grid"
        ).values,
        p_nom=battery_storages.set_index("bus")
        .loc[mv_grids.bus_id.astype(str)]
        .p_nom_min.values,
        p_nom_extendable=False,
        max_hours=2,
        carrier="home_battery",
    )
    self.network.storage_units.loc[battery_storages.index, "p_nom_min"] = 0


def add_simplified_distribution_grids(self):
    """
    Adds simplified distrubution grids to each HV/MV substation in the
    transmission grid model. Load, generation and storage units are connected
    to the corresponding grid level based on not-aggregated egon-data results.

    Returns
    -------
    None.

    """

    # Define and import tables in high spatial resolution
    if self.args["method"]["distribution_grids"]:
        if "oep.iks.cs.ovgu.de" in str(self.engine.url):
            from saio.tables import (
                edut_00_080 as egon_mv_grid_district,
                edut_00_153 as egon_power_plants,
                edut_00_146 as egon_chp_plants,
                edut_00_168 as egon_district_heating_areas,
                edut_00_042 as egon_osm_ind_load_curves_individual,
                edut_00_047 as egon_sites_ind_load_curves_individual,
            )
        else:
            saio.register_schema("supply", self.engine)
            saio.register_schema("demand", self.engine)
            from saio.grid import egon_mv_grid_district
            from saio.supply import egon_power_plants, egon_chp_plants
            from saio.demand import (
                egon_district_heating_areas,
                egon_osm_ind_load_curves_individual,
                egon_sites_ind_load_curves_individual,
            )
    # Copy previous network to derive data from it later on
    old_network = self.network.copy()

    # import mv grid districts
    mv_grids = saio.as_pandas(
        query=self.session.query(egon_mv_grid_district.bus_id),
    )

    # Add distribution grid buses and links
    distribution_grid_buses_and_links(self, mv_grids)

    # Seperate power plants conected to transmission and distribution grids
    seperate_power_plants(self, egon_power_plants, old_network)
    assert (
        abs(
            self.network.generators.p_nom.sum()
            - old_network.generators.p_nom.sum()
        )
        < 1e-6
    ), "Installed capacity of power plants differs from original network."

    # Seperate chp plants conected to transmission and distribution grids
    seperate_chp(self, egon_chp_plants, egon_district_heating_areas)
    assert (
        abs(
            self.network.generators[
                self.network.generators.carrier.str.contains("CHP")
            ].p_nom.sum()
            - old_network.generators[
                old_network.generators.carrier.str.contains("CHP")
            ].p_nom.sum()
        )
        < 1e-6
    ), "Installed capacity of generator CHP differs from original network."
    assert (
        abs(
            self.network.links[
                self.network.links.carrier.str.contains("CHP")
            ].p_nom.sum()
            - old_network.links[
                old_network.links.carrier.str.contains("CHP")
            ].p_nom.sum()
        )
        < 1e-6
    ), "Installed capacity of link CHP differs from original network."

    # Seperate demands conected to transmission and distribution grids
    seperate_demands(
        self,
        mv_grids,
        egon_osm_ind_load_curves_individual,
        egon_sites_ind_load_curves_individual,
    )
    assert (
        abs(
            self.network.loads_t.p_set.sum().sum()
            - old_network.loads_t.p_set.sum().sum()
        )
        < 1e-6
    ), "Loads differ from original network."

    # Add pv home storage units to distribution grid node
    seperate_storage_units(self, mv_grids)
