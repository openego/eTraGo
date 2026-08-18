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

    relax_battery_stores_for_grid_optimization(
        self,
        initial_soc=0.5,
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
        self.network.links_t.p_min_pu.abs() > 1e-7,
        other=0.0,
        inplace=True,
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


def _market_prices_for_components(
    market_price_per_bus,
    component_bus,
    component_index,
    snapshots,
    factor_redispatch_cost=1.0,
    label="component",
):
    """
    Build a time-dependent market-price dataframe for components.

    Parameters
    ----------
    market_price_per_bus : pd.DataFrame
        Market prices with snapshots as index and market buses as columns.
    component_bus : pd.Series
        Maps each component to the bus whose market price should be used.
        Index must be component names.
    component_index : pd.Index
        Components for which prices are requested.
    snapshots : pd.Index
        Target snapshots.
    factor_redispatch_cost : float
        Scaling factor for redispatch cost.
    label : str
        Label used in warnings.

    Returns
    -------
    pd.DataFrame
        index   = snapshots
        columns = component_index
    """
    if market_price_per_bus is None or market_price_per_bus.empty:
        logger.warning(
            "market_price_per_bus is empty. Using zero prices for %s.",
            label,
        )
        return pd.DataFrame(
            0.0,
            index=snapshots,
            columns=component_index,
        )

    prices = market_price_per_bus.copy()

    # Important for 300-cluster cases:
    # normalize all bus labels to strings on both sides.
    prices.columns = prices.columns.astype(str)
    prices.index = pd.Index(prices.index)

    component_index = pd.Index(component_index)
    component_bus = component_bus.reindex(component_index).astype(str)

    result = pd.DataFrame(
        index=prices.index,
        columns=component_index,
        dtype=float,
    )

    available_bus_mask = component_bus.isin(prices.columns)

    available_components = component_bus.index[available_bus_mask]
    missing_components = component_bus.index[~available_bus_mask]

    # Direct assignment for components whose bus exists in market prices.
    for component in available_components:
        bus = component_bus.loc[component]
        result.loc[:, component] = prices.loc[:, bus].astype(float).values

    # Fallback for missing buses:
    # use the time-dependent median market price across available market buses.
    # This avoids crashing and makes the fallback visible via warning.
    if len(missing_components) > 0:
        missing_buses = component_bus.loc[missing_components].dropna().unique()

        logger.warning(
            "%s: %s of %s components refer to buses that are missing from "
            "market_price_per_bus.columns. Missing bus examples: %s. "
            "Using time-dependent median market price as fallback.",
            label,
            len(missing_components),
            len(component_index),
            list(missing_buses[:10]),
        )

        fallback_price = prices.median(axis=1).astype(float)

        for component in missing_components:
            result.loc[:, component] = fallback_price.values

    result = result * factor_redispatch_cost

    # Align to target network snapshots.
    result = result.reindex(snapshots)

    if result.isna().any().any():
        logger.warning(
            "%s: market-price dataframe still contains NaNs after reindexing. "
            "Filling remaining NaNs with 0.0.",
            label,
        )
        result = result.fillna(0.0)

    return result


def relax_battery_stores_for_grid_optimization(
    self,
    initial_soc=0.5,
):
    """Allow battery_storage Stores to redispatch in grid optimisation.

    This avoids infeasibilities caused by forcing the market battery energy
    trajectory into a more detailed physical grid model.

    Parameters
    ----------
    self : Etrago
        eTraGo object.
    initial_soc : float
        Initial state of charge as share of grid e_nom.
        Use 0.5 for a neutral initial state.
    """

    if self.network.stores.empty:
        logger.info("Battery relaxation skipped: grid network has no stores.")
        return

    battery_stores = self.network.stores.index[
        self.network.stores.carrier.astype(str).eq("battery_storage")
    ]

    if len(battery_stores) == 0:
        logger.info("Battery relaxation skipped: no battery_storage stores.")
        return

    snapshots = self.network.snapshots

    e_nom = pd.to_numeric(
        self.network.stores.loc[battery_stores, "e_nom"],
        errors="coerce",
    )

    if "e_nom_opt" in self.network.stores.columns:
        e_nom_opt = pd.to_numeric(
            self.network.stores.loc[battery_stores, "e_nom_opt"],
            errors="coerce",
        )

        e_nom = e_nom.where(
            e_nom.notna() & (e_nom > 0),
            e_nom_opt,
        )

    e_nom = e_nom.where(e_nom.notna() & (e_nom > 0), 1.0)

    # Remove fixed market energy trajectory.
    for store in battery_stores:
        if store not in self.network.stores_t.e_min_pu.columns:
            self.network.stores_t.e_min_pu[store] = 0.0

        if store not in self.network.stores_t.e_max_pu.columns:
            self.network.stores_t.e_max_pu[store] = 1.0

    self.network.stores_t.e_min_pu.loc[
        snapshots,
        battery_stores,
    ] = 0.0

    self.network.stores_t.e_max_pu.loc[
        snapshots,
        battery_stores,
    ] = 1.0

    # Neutral feasible initial energy state.
    initial_soc = max(0.0, min(1.0, initial_soc))

    self.network.stores.loc[
        battery_stores,
        "e_initial",
    ] = (initial_soc * e_nom).values

    # Avoid an additional cyclic condition conflicting with
    # the chosen initial state.
    self.network.stores.loc[
        battery_stores,
        "e_cyclic",
    ] = False

    # If p_set exists and was filled before, clear it.
    if hasattr(self.network.stores_t, "p_set"):
        for store in battery_stores:
            if store in self.network.stores_t.p_set.columns:
                self.network.stores_t.p_set.loc[
                    snapshots,
                    store,
                ] = pd.NA

    logger.warning(
        "Relaxed %s battery_storage Stores for grid optimisation. "
        "Battery energy bounds set to [0, e_nom], e_initial set to %.2f SOC. "
        "This means batteries are allowed to redispatch in the grid model.",
        len(battery_stores),
        initial_soc,
    )


def fix_battery_stores_from_market_model(
    self,
    fix_power=False,
    tolerance=1e-6,
    initial_mode="previous_market_soc",
):
    """Fix battery Store states from the market model using SOC ratios.

    This function is for batteries represented as PyPSA Stores with carrier
    ``battery_storage``.

    Important:
    ---------
    Do not copy absolute market energy values directly into the grid model.
    In high-cluster cases, the market and grid store capacities can differ.
    Therefore this function transfers the market state-of-charge ratio:

        market_soc = market_e / market_e_nom
        grid_e     = market_soc * grid_e_nom

    Parameters
    ----------
    self : Etrago
        eTraGo object.
    fix_power : bool, default False
        If True, also fixes Store-p via stores_t.p_set using scaled market
        dispatch. Start with False unless you explicitly want to force
        battery dispatch.
    tolerance : float, default 1e-6
        Small relaxation around fixed energy trajectory in e_min_pu/e_max_pu.
    initial_mode : {"previous_market_soc", "first_grid_state"}
        - "previous_market_soc": e_initial follows the previous market SOC,
          scaled to grid e_nom. This makes first-step battery dispatch follow
          the market trajectory.
        - "first_grid_state": e_initial equals the first fixed grid energy
          state. This avoids a first-step battery jump and is more relaxed.
    """

    if self.network.stores.empty:
        logger.info("Battery SOC fix skipped: grid network has no stores.")
        return

    if self.market_model.stores.empty:
        logger.info("Battery SOC fix skipped: market model has no stores.")
        return

    def _effective_e_nom(stores, idx, label):
        idx = pd.Index(idx)

        if len(idx) == 0:
            return pd.Series(dtype=float)

        e_nom = pd.to_numeric(
            (
                stores.loc[idx, "e_nom"]
                if "e_nom" in stores.columns
                else pd.Series(index=idx)
            ),
            errors="coerce",
        )
        e_nom.index = idx

        if "e_nom_opt" in stores.columns:
            e_nom_opt = pd.to_numeric(
                stores.loc[idx, "e_nom_opt"],
                errors="coerce",
            )
            e_nom = e_nom.where(e_nom.notna() & (e_nom > 0), e_nom_opt)

        bad = e_nom.isna() | (e_nom <= 0)
        if bad.any():
            logger.warning(
                "%s: %s stores have missing/non-positive e_nom. "
                "Using 1.0 MWh fallback for examples: %s",
                label,
                int(bad.sum()),
                list(e_nom.index[bad][:10]),
            )
            e_nom.loc[bad] = 1.0

        return e_nom.astype(float)

    snapshots = pd.Index(self.network.snapshots)

    grid_batteries = self.network.stores.index[
        self.network.stores.carrier.astype(str).eq("battery_storage")
    ].astype(str)

    market_batteries = self.market_model.stores.index[
        self.market_model.stores.carrier.astype(str).eq("battery_storage")
    ].astype(str)

    batteries = pd.Index(grid_batteries).intersection(
        pd.Index(market_batteries)
    )

    if len(batteries) == 0:
        logger.info(
            "Battery SOC fix skipped: no matching battery_storage Stores "
            "between grid network and market model."
        )
        return

    market_e_all = self.market_model.stores_t.e.copy()
    market_e_all.columns = market_e_all.columns.astype(str)

    batteries = batteries.intersection(market_e_all.columns)

    if len(batteries) == 0:
        logger.warning(
            "Battery SOC fix skipped: matching battery stores are missing "
            "from market_model.stores_t.e."
        )
        return

    grid_e_nom = _effective_e_nom(
        self.network.stores,
        batteries,
        "grid battery stores",
    )

    market_e_nom = _effective_e_nom(
        self.market_model.stores,
        batteries,
        "market battery stores",
    )

    market_e = (
        market_e_all.reindex(
            index=snapshots,
            columns=batteries,
        )
        .ffill()
        .bfill()
    )

    if market_e.isna().any().any():
        logger.warning(
            "Battery SOC fix: market_e contains NaNs after ffill/bfill; "
            "filling remaining NaNs with 0.0."
        )
        market_e = market_e.fillna(0.0)

    # Transfer SOC ratio, not absolute MWh.
    market_soc = market_e.div(market_e_nom, axis=1).clip(
        lower=0.0,
        upper=1.0,
    )

    grid_e = market_soc.mul(grid_e_nom, axis=1)

    e_ratio = grid_e.div(grid_e_nom, axis=1).clip(lower=0.0, upper=1.0)
    e_min = (e_ratio - tolerance).clip(lower=0.0)
    e_max = (e_ratio + tolerance).clip(upper=1.0)

    for store in batteries:
        if store not in self.network.stores_t.e_min_pu.columns:
            self.network.stores_t.e_min_pu[store] = 0.0

        if store not in self.network.stores_t.e_max_pu.columns:
            self.network.stores_t.e_max_pu[store] = 1.0

    self.network.stores_t.e_min_pu.loc[
        snapshots,
        batteries,
    ] = e_min.values

    self.network.stores_t.e_max_pu.loc[
        snapshots,
        batteries,
    ] = e_max.values

    if initial_mode == "first_grid_state":
        e_initial = grid_e.loc[snapshots[0], batteries].astype(float)
        initial_source = "first_grid_state"

    elif initial_mode == "previous_market_soc":
        market_index = pd.Index(market_e_all.index)
        first_snapshot = snapshots[0]

        if first_snapshot in market_index:
            first_pos = market_index.get_loc(first_snapshot)

            if isinstance(first_pos, slice):
                first_pos = first_pos.start

            previous_pos = (
                first_pos - 1 if first_pos > 0 else len(market_index) - 1
            )
        else:
            previous_pos = len(market_index) - 1

        previous_snapshot = market_index[previous_pos]

        previous_market_e = market_e_all.loc[
            previous_snapshot,
            batteries,
        ].astype(float)

        previous_market_soc = previous_market_e.div(market_e_nom).clip(
            lower=0.0, upper=1.0
        )

        e_initial = previous_market_soc.mul(grid_e_nom).astype(float)
        initial_source = f"previous_market_soc at {previous_snapshot}"

    else:
        raise ValueError(
            "initial_mode must be either 'previous_market_soc' "
            "or 'first_grid_state'."
        )

    # Numerical safety: never let e_initial exceed grid capacity.
    e_initial = e_initial.clip(lower=0.0, upper=grid_e_nom)

    self.network.stores.loc[batteries, "e_initial"] = e_initial.values
    self.network.stores.loc[batteries, "e_cyclic"] = False

    ratio_initial = e_initial.div(grid_e_nom).replace(
        [np.inf, -np.inf],
        np.nan,
    )

    logger.info(
        "Battery SOC fix: fixed %s battery_storage stores from market SOC. "
        "Initial source=%s. Max initial SOC ratio=%.4f, "
        "max grid e_nom=%.3f MWh.",
        len(batteries),
        initial_source,
        float(ratio_initial.max()) if len(ratio_initial) else float("nan"),
        float(grid_e_nom.max()) if len(grid_e_nom) else float("nan"),
    )

    if fix_power:
        if not hasattr(self.network.stores_t, "p_set"):
            logger.warning(
                "fix_power=True requested, but self.network.stores_t has no "
                "p_set. Only battery energy trajectory was fixed."
            )
            return

        if not hasattr(self.market_model.stores_t, "p"):
            logger.warning(
                "fix_power=True requested, but self.market_model.stores_t "
                "has no p. Only battery energy trajectory was fixed."
            )
            return

        market_p = (
            self.market_model.stores_t.p.reindex(
                index=snapshots,
                columns=batteries,
            )
            .ffill()
            .bfill()
            .fillna(0.0)
        )

        # Scale Store-p consistently with energy-capacity ratio.
        scale = (
            grid_e_nom.div(market_e_nom)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )

        grid_p = market_p.mul(scale, axis=1)

        for store in batteries:
            if store not in self.network.stores_t.p_set.columns:
                self.network.stores_t.p_set[store] = pd.NA

        self.network.stores_t.p_set.loc[
            snapshots,
            batteries,
        ] = grid_p.values

        logger.info(
            "Battery SOC fix: also fixed Store-p dispatch for %s "
            "battery_storage stores.",
            len(batteries),
        )


def add_redispatch_generators(
    self,
    factor_redispatch_cost,
    management_cost,
    time_depended_cost,
    fre_mangement_fee,
):
    """Add components and parameters to model redispatch with costs.

    Robust version for high AC-cluster cases.

    The original function assumes that the market_model includes all
    generators and links for the spatial resolution of the grid optimization.
    This version keeps the same modelling logic, but aligns components and
    market-price columns safely before indexing.

    Returns
    -------
    None.
    """

    snapshots = self.network.snapshots

    def _as_str_index(index):
        return pd.Index(index.astype(str), name=index.name)

    def _safe_nominal(series, index):
        out = pd.to_numeric(series.reindex(index), errors="coerce")
        out = out.replace(0.0, float("nan"))
        return out

    def _ensure_ts_columns(ts_df, columns, fill_value=0.0):
        """Ensure all requested columns exist in a time-series dataframe."""
        if ts_df is None:
            return ts_df

        columns = pd.Index(columns)

        for col in columns.difference(ts_df.columns):
            ts_df[col] = fill_value

        return ts_df

    def _component_market_prices(
        market_price_per_bus,
        component_bus,
        component_index,
        label,
    ):
        """Return market prices per component.

        Output shape:
            index   = self.network.snapshots
            columns = component_index

        This fixes the crash where bus labels like '11', '12', '13', '14'
        were requested but not found in market_price_per_bus.columns.
        """
        component_index = pd.Index(component_index)

        if len(component_index) == 0:
            return pd.DataFrame(index=snapshots, columns=component_index)

        if market_price_per_bus is None or market_price_per_bus.empty:
            logger.warning(
                "%s: market_price_per_bus is empty. Using zero fallback.",
                label,
            )
            return pd.DataFrame(
                0.0,
                index=snapshots,
                columns=component_index,
            )

        prices = market_price_per_bus.copy()

        # Important: normalize market bus labels to strings.
        prices.columns = prices.columns.astype(str)

        # Align snapshots. Remaining gaps are filled by median fallback later.
        prices = prices.reindex(snapshots)

        # If some snapshot rows are missing completely, use neighbouring values
        # where possible.
        prices = prices.ffill().bfill()

        component_bus = component_bus.reindex(component_index).astype(str)

        out = pd.DataFrame(
            index=snapshots,
            columns=component_index,
            dtype=float,
        )

        fallback_price = prices.median(axis=1).fillna(0.0)

        missing_buses = []

        for bus in component_bus.dropna().unique():
            comps = component_bus.index[component_bus == bus]

            if bus in prices.columns:
                out.loc[:, comps] = (
                    prices.loc[:, bus].astype(float).values[:, None]
                )
            else:
                missing_buses.append(bus)
                out.loc[:, comps] = fallback_price.values[:, None]

        if missing_buses:
            logger.warning(
                "%s: %s market-price buses missing from "
                "market_price_per_bus.columns. Examples: %s. "
                "Using time-dependent median market price as fallback.",
                label,
                len(missing_buses),
                missing_buses[:10],
            )

        out = out.fillna(0.0) * factor_redispatch_cost
        return out

    def _static_or_time_dependent_management_cost(
        carriers,
        component_index,
        management_cost_carrier,
    ):
        component_index = pd.Index(component_index)

        if len(component_index) == 0:
            if time_depended_cost:
                return pd.DataFrame(index=snapshots, columns=component_index)
            return pd.Series(index=component_index, dtype=float)

        static_cost = management_cost_carrier.loc[
            carriers.reindex(component_index).values
        ]
        static_cost.index = component_index

        if not time_depended_cost:
            return static_cost

        return pd.DataFrame(
            data=[static_cost.values] * len(snapshots),
            index=snapshots,
            columns=component_index,
        )

    def _time_dependent_marginal_cost(static_df, component_index):
        component_index = pd.Index(component_index)

        if len(component_index) == 0:
            return pd.DataFrame(index=snapshots, columns=component_index)

        mc = pd.to_numeric(
            static_df.loc[component_index, "marginal_cost"],
            errors="coerce",
        ).fillna(0.0)

        return pd.DataFrame(
            data=[mc.values] * len(snapshots),
            index=snapshots,
            columns=component_index,
        )

    # ------------------------------------------------------------------
    # Select generator and link components considered for redispatch
    # ------------------------------------------------------------------

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

    # Keep only components also available in the market model.
    missing_gens_market = gens_redispatch.difference(
        self.market_model.generators.index
    )
    if len(missing_gens_market) > 0:
        logger.warning(
            "Skipping %s redispatch generators missing from market_model. "
            "Examples: %s",
            len(missing_gens_market),
            list(missing_gens_market[:10]),
        )

    missing_links_market = links_redispatch.difference(
        self.market_model.links.index
    )
    if len(missing_links_market) > 0:
        logger.warning(
            "Skipping %s redispatch links missing from market_model. "
            "Examples: %s",
            len(missing_links_market),
            list(missing_links_market[:10]),
        )

    gens_redispatch = gens_redispatch.intersection(
        self.market_model.generators.index
    )
    links_redispatch = links_redispatch.intersection(
        self.market_model.links.index
    )

    gens_ramp_up = gens_redispatch.astype(str) + " ramp_up"
    gens_ramp_down = gens_redispatch.astype(str) + " ramp_down"
    links_ramp_up = links_redispatch.astype(str) + " ramp_up"
    links_ramp_down = links_redispatch.astype(str) + " ramp_down"

    # This function is called before p_max_pu is modified to fix dispatch
    # values from the market optimization.
    p_max_pu_all = self.network.get_switchable_as_dense(
        "Generator",
        "p_max_pu",
    ).copy()

    p_max_pu_all = p_max_pu_all.reindex(
        index=snapshots,
        columns=gens_redispatch,
    ).fillna(0.0)

    # ------------------------------------------------------------------
    # Management costs
    # ------------------------------------------------------------------

    generator_carriers = (
        self.network.generators.loc[gens_redispatch, "carrier"]
        if len(gens_redispatch) > 0
        else pd.Series(dtype=object)
    )

    link_carriers = (
        self.network.links.loc[links_redispatch, "carrier"]
        if len(links_redispatch) > 0
        else pd.Series(dtype=object)
    )

    all_carriers = pd.Index(
        list(generator_carriers.unique()) + list(link_carriers.unique())
    ).drop_duplicates()

    management_cost_carrier = pd.Series(
        index=all_carriers,
        data=management_cost,
        dtype=float,
    )

    if "OCGT" not in management_cost_carrier.index:
        management_cost_carrier.loc["OCGT"] = management_cost

    if fre_mangement_fee:
        for carrier in [
            "wind_onshore",
            "wind_offshore",
            "solar",
            "solar_rooftop",
        ]:
            if carrier in management_cost_carrier.index:
                management_cost_carrier.loc[carrier] = fre_mangement_fee

    management_cost_per_generator = (
        _static_or_time_dependent_management_cost(
            self.network.generators.loc[:, "carrier"],
            gens_redispatch,
            management_cost_carrier,
        )
        if len(gens_redispatch) > 0
        else (
            pd.DataFrame(index=snapshots, columns=gens_redispatch)
            if time_depended_cost
            else pd.Series(index=gens_redispatch, dtype=float)
        )
    )

    management_cost_per_link = (
        _static_or_time_dependent_management_cost(
            self.network.links.loc[:, "carrier"],
            links_redispatch,
            management_cost_carrier,
        )
        if len(links_redispatch) > 0
        else (
            pd.DataFrame(index=snapshots, columns=links_redispatch)
            if time_depended_cost
            else pd.Series(index=links_redispatch, dtype=float)
        )
    )

    # ------------------------------------------------------------------
    # Market dispatch used to fix original generators/links
    # ------------------------------------------------------------------

    market_gen_dispatch = self.market_model.generators_t.p.reindex(
        index=snapshots,
        columns=gens_redispatch,
    ).fillna(0.0)

    market_link_dispatch = self.market_model.links_t.p0.reindex(
        index=snapshots,
        columns=links_redispatch,
    ).fillna(0.0)

    # ------------------------------------------------------------------
    # Fix generator dispatch from market simulation
    # ------------------------------------------------------------------

    if len(gens_redispatch) > 0:
        market_gen_p_nom = _safe_nominal(
            self.market_model.generators["p_nom"],
            gens_redispatch,
        )

        fixed_gen_pu = market_gen_dispatch.div(
            market_gen_p_nom, axis=1
        ).fillna(0.0)

        self.network.generators_t.p_max_pu = _ensure_ts_columns(
            self.network.generators_t.p_max_pu,
            gens_redispatch,
            fill_value=0.0,
        )
        self.network.generators_t.p_min_pu = _ensure_ts_columns(
            self.network.generators_t.p_min_pu,
            gens_redispatch,
            fill_value=0.0,
        )

        self.network.generators_t.p_max_pu.loc[
            snapshots,
            gens_redispatch,
        ] = fixed_gen_pu.values

        self.network.generators_t.p_min_pu.loc[
            snapshots,
            gens_redispatch,
        ] = fixed_gen_pu.values

    # ------------------------------------------------------------------
    # Fix link dispatch from market simulation
    # ------------------------------------------------------------------

    if len(links_redispatch) > 0:
        market_link_p_nom = _safe_nominal(
            self.market_model.links["p_nom"],
            links_redispatch,
        )

        fixed_link_pu = market_link_dispatch.div(
            market_link_p_nom, axis=1
        ).fillna(0.0)

        self.network.links_t.p_max_pu = _ensure_ts_columns(
            self.network.links_t.p_max_pu,
            links_redispatch,
            fill_value=0.0,
        )
        self.network.links_t.p_min_pu = _ensure_ts_columns(
            self.network.links_t.p_min_pu,
            links_redispatch,
            fill_value=0.0,
        )

        self.network.links_t.p_max_pu.loc[
            snapshots,
            links_redispatch,
        ] = fixed_link_pu.values

        self.network.links_t.p_min_pu.loc[
            snapshots,
            links_redispatch,
        ] = fixed_link_pu.values

    # ------------------------------------------------------------------
    # Calculate market-price reference for redispatch costs
    # ------------------------------------------------------------------

    market_price_per_bus = self.market_model.buses_t.marginal_price.copy()
    market_price_per_bus.columns = market_price_per_bus.columns.astype(str)

    if len(gens_redispatch) > 0:
        market_generator_buses = self.market_model.generators.loc[
            gens_redispatch, "bus"
        ].astype(str)

        market_price_per_generator_ts = _component_market_prices(
            market_price_per_bus,
            market_generator_buses,
            gens_redispatch,
            "redispatch generators",
        )
    else:
        market_price_per_generator_ts = pd.DataFrame(
            index=snapshots,
            columns=gens_redispatch,
        )

    if len(links_redispatch) > 0:
        market_link_buses = self.market_model.links.loc[
            links_redispatch, "bus1"
        ].astype(str)

        market_price_per_link_ts = _component_market_prices(
            market_price_per_bus,
            market_link_buses,
            links_redispatch,
            "redispatch links",
        )
    else:
        market_price_per_link_ts = pd.DataFrame(
            index=snapshots,
            columns=links_redispatch,
        )

    if not time_depended_cost:
        market_price_per_generator = market_price_per_generator_ts.median()
        market_price_per_generator.index = gens_redispatch

        market_price_per_link = market_price_per_link_ts.median()
        market_price_per_link.index = links_redispatch

    else:
        market_price_per_generator = market_price_per_generator_ts
        market_price_per_link = market_price_per_link_ts

    # ------------------------------------------------------------------
    # Ramp-up and ramp-down costs for generators
    # ------------------------------------------------------------------

    if len(gens_redispatch) > 0:
        generator_marginal_cost = pd.to_numeric(
            self.network.generators.loc[gens_redispatch, "marginal_cost"],
            errors="coerce",
        ).fillna(0.0)

        if time_depended_cost:
            ramp_up_costs = _time_dependent_marginal_cost(
                self.network.generators,
                gens_redispatch,
            )

            ramp_up_costs = ramp_up_costs.mask(
                market_price_per_generator > ramp_up_costs,
                market_price_per_generator,
            )

            ramp_up_costs = ramp_up_costs.add(
                management_cost_per_generator,
                fill_value=0.0,
            )

            ramp_down_costs = market_price_per_generator.sub(
                generator_marginal_cost, axis=1
            ).add(management_cost_per_generator, fill_value=0.0)

        else:
            ramp_up_costs = pd.concat(
                [
                    generator_marginal_cost,
                    market_price_per_generator,
                ],
                axis=1,
            ).max(axis=1)

            ramp_up_costs = (
                ramp_up_costs
                + management_cost_per_generator.reindex(gens_redispatch)
            )

            ramp_down_costs = (
                market_price_per_generator
                - generator_marginal_cost
                + management_cost_per_generator.reindex(gens_redispatch)
            )

        # --------------------------------------------------------------
        # Add ramp-up generators
        # --------------------------------------------------------------

        self.network.madd(
            "Generator",
            gens_ramp_up,
            bus=self.network.generators.loc[gens_redispatch, "bus"].values,
            p_nom=self.network.generators.loc[
                gens_redispatch,
                "p_nom",
            ].values,
            carrier=self.network.generators.loc[
                gens_redispatch,
                "carrier",
            ].values,
        )

        if time_depended_cost:
            ramp_up_costs = ramp_up_costs.copy()
            ramp_up_costs.columns = gens_ramp_up

            self.network.generators_t.marginal_cost = pd.concat(
                [
                    self.network.generators_t.marginal_cost,
                    ramp_up_costs,
                ],
                axis=1,
            )
        else:
            self.network.generators.loc[
                gens_ramp_up,
                "marginal_cost",
            ] = ramp_up_costs.values

        # Maximum ramp-up feed-in:
        # available feed-in minus market dispatch.
        gen_p_nom_grid = _safe_nominal(
            self.network.generators["p_nom"],
            gens_redispatch,
        )

        available_gen_feed_in = p_max_pu_all.mul(
            gen_p_nom_grid,
            axis=1,
        )

        ramp_up_gen_pu = (
            (available_gen_feed_in - market_gen_dispatch)
            .clip(lower=0.0)
            .div(gen_p_nom_grid, axis=1)
            .fillna(0.0)
        )

        self.network.generators_t.p_max_pu = _ensure_ts_columns(
            self.network.generators_t.p_max_pu,
            gens_ramp_up,
            fill_value=0.0,
        )

        self.network.generators_t.p_max_pu.loc[
            snapshots,
            gens_ramp_up,
        ] = ramp_up_gen_pu.values

    # ------------------------------------------------------------------
    # Ramp-up costs and components for links
    # ------------------------------------------------------------------

    if len(links_redispatch) > 0:
        link_marginal_cost = pd.to_numeric(
            self.network.links.loc[links_redispatch, "marginal_cost"],
            errors="coerce",
        ).fillna(0.0)

        if time_depended_cost:
            ramp_up_costs_links = _time_dependent_marginal_cost(
                self.network.links,
                links_redispatch,
            )

            ramp_up_costs_links = ramp_up_costs_links.mask(
                market_price_per_link > ramp_up_costs_links,
                market_price_per_link,
            )

            ramp_up_costs_links = ramp_up_costs_links.add(
                management_cost_per_link,
                fill_value=0.0,
            )

        else:
            ramp_up_costs_links = pd.concat(
                [
                    link_marginal_cost,
                    market_price_per_link,
                ],
                axis=1,
            ).max(axis=1)

            ramp_up_costs_links = (
                ramp_up_costs_links
                + management_cost_per_link.reindex(links_redispatch)
            )

        self.network.madd(
            "Link",
            links_ramp_up,
            bus0=self.network.links.loc[links_redispatch, "bus0"].values,
            bus1=self.network.links.loc[links_redispatch, "bus1"].values,
            p_nom=self.network.links.loc[links_redispatch, "p_nom"].values,
            carrier=self.network.links.loc[links_redispatch, "carrier"].values,
            efficiency=self.network.links.loc[
                links_redispatch,
                "efficiency",
            ].values,
        )

        if time_depended_cost:
            ramp_up_costs_links = ramp_up_costs_links.copy()
            ramp_up_costs_links.columns = links_ramp_up

            self.network.links_t.marginal_cost = pd.concat(
                [
                    self.network.links_t.marginal_cost,
                    ramp_up_costs_links,
                ],
                axis=1,
            )
        else:
            self.network.links.loc[
                links_ramp_up,
                "marginal_cost",
            ] = ramp_up_costs_links.values

        link_p_nom_grid = _safe_nominal(
            self.network.links["p_nom"],
            links_redispatch,
        )

        ramp_up_link_pu = (
            link_p_nom_grid.reindex(links_redispatch)
            .to_frame()
            .T.reindex(index=snapshots)
        )

        for link in links_redispatch:
            ramp_up_link_pu.loc[:, link] = link_p_nom_grid.loc[link]

        ramp_up_link_pu = (
            (ramp_up_link_pu - market_link_dispatch)
            .clip(lower=0.0)
            .div(link_p_nom_grid, axis=1)
            .fillna(0.0)
        )

        self.network.links_t.p_max_pu = _ensure_ts_columns(
            self.network.links_t.p_max_pu,
            links_ramp_up,
            fill_value=0.0,
        )

        self.network.links_t.p_max_pu.loc[
            snapshots,
            links_ramp_up,
        ] = ramp_up_link_pu.values

    # ------------------------------------------------------------------
    # Add ramp-down generators
    # ------------------------------------------------------------------

    if len(gens_redispatch) > 0:
        self.network.madd(
            "Generator",
            gens_ramp_down,
            bus=self.network.generators.loc[gens_redispatch, "bus"].values,
            p_nom=self.network.generators.loc[
                gens_redispatch,
                "p_nom",
            ].values,
            carrier=self.network.generators.loc[
                gens_redispatch,
                "carrier",
            ].values,
        )

        if time_depended_cost:
            ramp_down_costs = ramp_down_costs.copy()
            ramp_down_costs.columns = gens_ramp_down

            self.network.generators_t.marginal_cost = pd.concat(
                [
                    self.network.generators_t.marginal_cost,
                    -ramp_down_costs,
                ],
                axis=1,
            )
        else:
            self.network.generators.loc[
                gens_ramp_down,
                "marginal_cost",
            ] = -ramp_down_costs.values

        self.network.generators_t.p_max_pu = _ensure_ts_columns(
            self.network.generators_t.p_max_pu,
            gens_ramp_down,
            fill_value=0.0,
        )
        self.network.generators_t.p_min_pu = _ensure_ts_columns(
            self.network.generators_t.p_min_pu,
            gens_ramp_down,
            fill_value=0.0,
        )

        self.network.generators_t.p_max_pu.loc[
            snapshots,
            gens_ramp_down,
        ] = 0.0

        gen_p_nom_grid = _safe_nominal(
            self.network.generators["p_nom"],
            gens_redispatch,
        )

        ramp_down_gen_pu = (
            -market_gen_dispatch.clip(lower=0.0)
            .div(gen_p_nom_grid, axis=1)
            .fillna(0.0)
        )

        self.network.generators_t.p_min_pu.loc[
            snapshots,
            gens_ramp_down,
        ] = ramp_down_gen_pu.values

    # ------------------------------------------------------------------
    # Add ramp-down links
    # ------------------------------------------------------------------

    if len(links_redispatch) > 0:
        self.network.madd(
            "Link",
            links_ramp_down,
            bus0=self.network.links.loc[links_redispatch, "bus0"].values,
            bus1=self.network.links.loc[links_redispatch, "bus1"].values,
            p_nom=self.network.links.loc[links_redispatch, "p_nom"].values,
            marginal_cost=-(management_cost),
            carrier=self.network.links.loc[links_redispatch, "carrier"].values,
            efficiency=self.network.links.loc[
                links_redispatch,
                "efficiency",
            ].values,
        )

        self.network.links_t.p_max_pu = _ensure_ts_columns(
            self.network.links_t.p_max_pu,
            links_ramp_down,
            fill_value=0.0,
        )
        self.network.links_t.p_min_pu = _ensure_ts_columns(
            self.network.links_t.p_min_pu,
            links_ramp_down,
            fill_value=0.0,
        )

        self.network.links_t.p_max_pu.loc[
            snapshots,
            links_ramp_down,
        ] = 0.0

        link_p_nom_grid = _safe_nominal(
            self.network.links["p_nom"],
            links_redispatch,
        )

        ramp_down_link_pu = (
            -market_link_dispatch.clip(lower=0.0)
            .div(link_p_nom_grid, axis=1)
            .fillna(0.0)
        )

        self.network.links_t.p_min_pu.loc[
            snapshots,
            links_ramp_down,
        ] = ramp_down_link_pu.values

    # ------------------------------------------------------------------
    # Final consistency check
    # ------------------------------------------------------------------

    self.network.consistency_check()


def extra_functionality():
    return None
