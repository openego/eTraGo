"""Scenario configuration helpers for the Biogas.SH / SWFL eTraGo model."""

from __future__ import annotations

import argparse
import copy
import csv
import itertools
import os
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import yaml


CONFIG_VERSION = 3


class ScenarioConfigError(ValueError):
    """Raised when the scenario configuration is incomplete or inconsistent."""


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ScenarioConfigError(f"Missing '{key}' in {context}.")
    return mapping[key]


def _require_mapping(
    mapping: Mapping[str, Any],
    key: str,
    context: str,
) -> Mapping[str, Any]:
    value = _require(mapping, key, context)
    if not isinstance(value, Mapping):
        raise ScenarioConfigError(f"'{key}' in {context} must be a mapping.")
    return value


def _named_case(
    parent: Mapping[str, Any],
    section: str,
    name: str,
) -> Mapping[str, Any]:
    cases = _require_mapping(parent, section, "configuration")
    if name not in cases:
        options = ", ".join(sorted(map(str, cases)))
        raise ScenarioConfigError(
            f"Unknown case '{name}' in '{section}'. Available: {options}"
        )
    value = cases[name]
    if not isinstance(value, Mapping):
        raise ScenarioConfigError(
            f"Case '{section}.{name}' must be a mapping."
        )
    return value


def _selection_with_environment_overrides(
    config: Mapping[str, Any],
) -> Dict[str, str]:
    selection = {
        str(key): str(value)
        for key, value in _require_mapping(
            config,
            "selection",
            "configuration",
        ).items()
    }

    environment_variables = {
        "fossil_gas_price_case": (
            "ETRAGO_FOSSIL_GAS_PRICE_CASE"
        ),
        "support_case": (
            "ETRAGO_SUPPORT_CASE"
        ),
        "biomethane_price_case": (
            "ETRAGO_BIOMETHANE_PRICE_CASE"
        ),
        "co2_sale_case": (
            "ETRAGO_CO2_SALE_CASE"
        ),
        "heat_pump_case": "ETRAGO_HEAT_PUMP_CASE",
        "swfl_unit_case": "ETRAGO_SWFL_UNIT_CASE",
        "biomethane_use_case": (
            "ETRAGO_BIOMETHANE_USE_CASE"
        ),
        "biogas_route_case": "ETRAGO_BIOGAS_ROUTE_CASE",
    }

    for key, variable in environment_variables.items():
        override = os.getenv(variable)
        if override:
            selection[key] = override

    return selection


def _validate_selection(
    config: Mapping[str, Any],
    selection: Mapping[str, str],
) -> None:
    required_dimensions = (
        "fossil_gas_price_case",
        "support_case",
        "biomethane_price_case",
        "heat_pump_case",
        "swfl_unit_case",
        "biomethane_use_case",
        "biogas_route_case",
        "co2_sale_case",
    )
    missing = [key for key in required_dimensions if key not in selection]
    if missing:
        raise ScenarioConfigError(
            f"Missing selection dimensions: {', '.join(missing)}"
        )

    price_cases = _require_mapping(config, "price_cases", "configuration")
    _named_case(
        price_cases,
        "fossil_gas",
        selection["fossil_gas_price_case"],
    )
    _named_case(
        price_cases,
        "biomethane",
        selection["biomethane_price_case"],
    )

    heat_pump_case = _named_case(
        config,
        "heat_pump_cases",
        selection["heat_pump_case"],
    )
    unit_case = _named_case(
        config,
        "swfl_unit_cases",
        selection["swfl_unit_case"],
    )
    biomethane_case = _named_case(
        config,
        "biomethane_use_cases",
        selection["biomethane_use_case"],
    )
    support_case = _named_case(
        config,
        "support_cases",
        selection["support_case"],
    )

    # Validate support-policy parameters.
    eeg_active = bool(
        support_case.get(
            "eeg_active",
            False,
        )
    )
    flexibility_active = bool(
        support_case.get(
            "flexibility_active",
            False,
        )
    )
    supported_hours = float(
        support_case.get(
            "supported_hours_per_year",
            0.0,
        )
    )
    capacity_multiplier = float(
        support_case.get(
            "chp_capacity_multiplier",
            1.0,
        )
    )
    flexibility_payment = float(
        support_case.get(
            "flexibility_payment_eur_per_kw_year",
            0.0,
        )
    )

    if capacity_multiplier <= 0:
        raise ScenarioConfigError(
            "support_case.chp_capacity_multiplier "
            "must be greater than zero."
        )

    if supported_hours < 0:
        raise ScenarioConfigError(
            "support_case.supported_hours_per_year "
            "must be non-negative."
        )

    if eeg_active and supported_hours <= 0:
        raise ScenarioConfigError(
            "An active EEG support case must define "
            "supported_hours_per_year > 0."
        )

    if not eeg_active and supported_hours != 0:
        raise ScenarioConfigError(
            "A support case with eeg_active=false must set "
            "supported_hours_per_year to 0."
        )

    if flexibility_payment < 0:
        raise ScenarioConfigError(
            "support_case.flexibility_payment_eur_per_kw_year "
            "must be non-negative."
        )

    if flexibility_active:
        if float(support_case.get("flex_capex_eur_per_kw", 0.0)) <= 0:
            raise ScenarioConfigError(
                "A flexibility case must define "
                "flex_capex_eur_per_kw > 0."
            )
        if float(support_case.get("flex_lifetime_years", 0.0)) <= 0:
            raise ScenarioConfigError(
                "A flexibility case must define "
                "flex_lifetime_years > 0."
            )
        if float(support_case.get("flex_discount_rate", 0.0)) < 0:
            raise ScenarioConfigError(
                "flex_discount_rate must be non-negative."
            )
        if float(support_case.get("flex_fixed_om_fraction", 0.0)) < 0:
            raise ScenarioConfigError(
                "flex_fixed_om_fraction must be non-negative."
            )

    route_case = _named_case(
        config,
        "biogas_route_cases",
        selection["biogas_route_case"],
    )

    technical = _require_mapping(config, "technical", "configuration")
    swfl = _require_mapping(technical, "swfl", "technical")
    biogas = _require_mapping(technical, "biogas_sh", "technical")

    available_heat_pumps = set(
        _require_mapping(swfl, "heat_pumps", "technical.swfl")
    )
    available_boilers = set(
        _require_mapping(swfl, "boilers", "technical.swfl")
    )
    available_resistive = set(
        _require_mapping(swfl, "resistive_heaters", "technical.swfl")
    )

    selected_heat_pumps = set(map(str, heat_pump_case.get("active_units", [])))
    selected_boilers = set(map(str, unit_case.get("boilers", [])))
    selected_resistive = set(map(str, unit_case.get("resistive_heaters", [])))
    eligible_biomethane = set(
        map(str, biomethane_case.get("eligible_units", []))
    )

    unknown_heat_pumps = selected_heat_pumps - available_heat_pumps
    unknown_boilers = selected_boilers - available_boilers
    unknown_resistive = selected_resistive - available_resistive

    if unknown_heat_pumps:
        raise ScenarioConfigError(
            "Unknown heat pumps in selected case: "
            f"{sorted(unknown_heat_pumps)}"
        )
    if unknown_boilers:
        raise ScenarioConfigError(
            f"Unknown boilers in selected case: {sorted(unknown_boilers)}"
        )
    if unknown_resistive:
        raise ScenarioConfigError(
            "Unknown resistive heaters in selected case: "
            f"{sorted(unknown_resistive)}"
        )

    inactive_eligible = eligible_biomethane - selected_boilers
    if inactive_eligible:
        raise ScenarioConfigError(
            "Biomethane-eligible boilers are inactive in the selected "
            f"SWFL unit case: {sorted(inactive_eligible)}"
        )

    storage = _require_mapping(biogas, "storage", "technical.biogas_sh")
    storage_required = bool(
        route_case.get("add_gas_grid_generation", False)
        or route_case.get("add_swfl_direct_supply", False)
    )
    if storage_required and not bool(storage.get("active", False)):
        raise ScenarioConfigError(
            "The selected Biogas.SH route requires central storage, "
            "but technical.biogas_sh.storage.active is false."
        )


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the selected scenario and the configured matrix dimensions."""
    _validate_selection(config, _selection_with_environment_overrides(config))

    matrix = config.get("scenario_matrix", {})
    if not isinstance(matrix, Mapping) or not matrix.get("enabled", False):
        return

    dimensions = matrix.get("dimensions", {})
    if not isinstance(dimensions, Mapping) or not dimensions:
        raise ScenarioConfigError(
            "scenario_matrix.enabled is true, but no dimensions are defined."
        )

    base_selection = _selection_with_environment_overrides(config)
    for key, values in dimensions.items():
        if key not in base_selection:
            raise ScenarioConfigError(
                f"Unknown scenario-matrix dimension: {key}"
            )
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ScenarioConfigError(
                f"scenario_matrix.dimensions.{key} must be a list."
            )
        for value in values:
            candidate = dict(base_selection)
            candidate[str(key)] = str(value)
            _validate_selection(config, candidate)


def load_config(path: str | Path) -> Dict[str, Any]:
    """Load and validate a version-3 YAML scenario configuration."""
    config_path = Path(path)
    if not config_path.exists():
        raise ScenarioConfigError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    if not isinstance(config, dict):
        raise ScenarioConfigError("The YAML root must be a mapping.")

    version = int(config.get("version", 0))
    if version != CONFIG_VERSION:
        raise ScenarioConfigError(
            f"Unsupported config version {version}; expected {CONFIG_VERSION}."
        )

    validate_config(config)
    return config


def resolve_config(
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    """Resolve one selected scenario into a reproducibility record."""

    selection = _selection_with_environment_overrides(
        config
    )

    _validate_selection(
        config,
        selection,
    )

    # ============================================================
    # PRICE CASES
    # ============================================================

    price_cases = _require_mapping(
        config,
        "price_cases",
        "configuration",
    )

    fossil_gas = _named_case(
        price_cases,
        "fossil_gas",
        selection["fossil_gas_price_case"],
    )

    biomethane = _named_case(
        price_cases,
        "biomethane",
        selection["biomethane_price_case"],
    )

    onsite = _require_mapping(
        price_cases,
        "onsite",
        "price_cases",
    )

    # ============================================================
    # CO2 SALE CASE
    # ============================================================

    co2_sale = _named_case(
        config,
        "co2_sale_cases",
        selection["co2_sale_case"],
    )

    # ============================================================
    # SUPPORT CASE
    # ============================================================

    support = _named_case(
        config,
        "support_cases",
        selection["support_case"],
    )

    # ============================================================
    # ONSITE BIOGAS COSTS
    # ============================================================

    raw_biogas_cost = float(
        onsite[
            "raw_biogas_cost_eur_per_mwh_hs"
        ]
    )

    onsite_el_efficiency = float(
        onsite[
            "electricity_efficiency"
        ]
    )

    onsite_heat_efficiency = float(
        onsite[
            "heat_efficiency"
        ]
    )

    merchant_el_cost = float(
        onsite[
            "merchant_electricity_marginal_cost_eur_per_mwh"
        ]
    )

    onsite_heat_cost = float(
        onsite[
            "heat_marginal_cost_eur_per_mwh"
        ]
    )

    eeg_premium = float(
        onsite[
            "eeg_premium_eur_per_mwh"
        ]
    )

    supported_el_cost = (
        merchant_el_cost
        - eeg_premium
    )

    if supported_el_cost < 0:
        raise ScenarioConfigError(
            "The EEG premium produces a negative supported "
            "electricity marginal cost. "
            "Check price_cases.onsite."
        )

    # ============================================================
    # FOSSIL NATURAL-GAS COST
    # ============================================================

    gas_commodity_price = float(
        fossil_gas[
            "gas_commodity_price_eur_per_mwh_fuel"
        ]
    )

    fossil_co2_price = float(
        fossil_gas[
            "co2_price_eur_per_tco2"
        ]
    )

    emission_factor = float(
        fossil_gas[
            "emission_factor_tco2_per_mwh_fuel"
        ]
    )

    co2_cost_on_gas = (
        fossil_co2_price
        * emission_factor
    )

    final_ch4_ng_cost = (
        gas_commodity_price
        + co2_cost_on_gas
    )

    swfl_import_adder = float(
        fossil_gas.get(
            "swfl_import_adder_eur_per_mwh_fuel",
            0.0,
        )
    )

    # ============================================================
    # BIOMETHANE BASE COST
    # ============================================================

    biomethane_base_cost = float(
        biomethane[
            "marginal_cost_eur_per_mwh_hs"
        ]
    )

    # ============================================================
    # BIOGENIC CO2 SALE
    # ============================================================
    #
    # CO2 sale is represented as a revenue credit on biomethane
    # production.
    #
    # Example:
    #
    #   CO2 yield = 0.12365 tCO2 / MWh_Hs biomethane
    #   CO2 price = 60 EUR/tCO2
    #
    #   revenue credit =
    #       0.12365 * 60
    #       = 7.419 EUR/MWh_Hs biomethane
    #
    #   effective biomethane cost =
    #       92.9 - 7.419
    #       = 85.481 EUR/MWh_Hs
    #
    # ============================================================

    co2_sale_active = bool(
        co2_sale.get(
            "active",
            False,
        )
    )

    co2_sale_price = float(
        co2_sale.get(
            "sale_price_eur_per_t",
            0.0,
        )
    )

    co2_additional_cost = float(
        co2_sale.get(
            "additional_cost_eur_per_t",
            0.0,
        )
    )

    co2_marketable_fraction = float(
        co2_sale.get(
            "marketable_fraction",
            1.0,
        )
    )

    co2_yield = float(
        co2_sale.get(
            "co2_yield_t_per_mwh_biomethane",
            0.0,
        )
    )

    # ------------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------------

    if not 0.0 <= co2_marketable_fraction <= 1.0:
        raise ScenarioConfigError(
            "co2_sale_cases."
            f"{selection['co2_sale_case']}."
            "marketable_fraction must be between 0 and 1."
        )

    if co2_sale_price < 0:
        raise ScenarioConfigError(
            "CO2 sale price cannot be negative."
        )

    if co2_additional_cost < 0:
        raise ScenarioConfigError(
            "CO2 additional cost cannot be negative."
        )

    if co2_yield < 0:
        raise ScenarioConfigError(
            "CO2 yield cannot be negative."
        )

    # ------------------------------------------------------------
    # CO2 revenue calculation
    # ------------------------------------------------------------

    if co2_sale_active:

        co2_net_sale_price = (
            co2_sale_price
            - co2_additional_cost
        )

        co2_revenue_credit = (
            co2_yield
            * co2_marketable_fraction
            * co2_net_sale_price
        )

    else:

        co2_net_sale_price = 0.0
        co2_revenue_credit = 0.0

    # ------------------------------------------------------------
    # Effective biomethane marginal cost
    # ------------------------------------------------------------

    biomethane_effective_cost = (
        biomethane_base_cost
        - co2_revenue_credit
    )

    if biomethane_effective_cost < 0:
        raise ScenarioConfigError(
            "CO2 sales revenue produces a negative effective "
            "biomethane marginal cost. Check the CO2 assumptions."
        )

    # ============================================================
    # SCENARIO NAME
    # ============================================================

    ordered_dimensions = (
        "support_case",
        "fossil_gas_price_case",
        "biomethane_price_case",
        "co2_sale_case",
        "heat_pump_case",
        "swfl_unit_case",
        "biomethane_use_case",
        "biogas_route_case",
    )

    scenario_name = "__".join(
        selection[key]
        for key in ordered_dimensions
    )

    # ============================================================
    # RESOLVED CONFIGURATION
    # ============================================================

    return {
        "config_version": CONFIG_VERSION,

        "scenario_name": scenario_name,

        "selection": selection,

        "prices": {

            # ----------------------------------------------------
            # Fossil natural gas
            # ----------------------------------------------------

            "gas_commodity_price_eur_per_mwh_fuel": (
                gas_commodity_price
            ),

            "co2_price_eur_per_tco2": (
                fossil_co2_price
            ),

            "emission_factor_tco2_per_mwh_fuel": (
                emission_factor
            ),

            "co2_cost_on_gas_eur_per_mwh_fuel": (
                co2_cost_on_gas
            ),

            "final_ch4_ng_marginal_cost_eur_per_mwh_fuel": (
                final_ch4_ng_cost
            ),

            "swfl_import_adder_eur_per_mwh_fuel": (
                swfl_import_adder
            ),

            "gas_source": str(
                fossil_gas.get(
                    "gas_source",
                    "",
                )
            ),

            "co2_source": str(
                fossil_gas.get(
                    "co2_source",
                    "",
                )
            ),

            "gas_value_status": str(
                fossil_gas.get(
                    "gas_value_status",
                    "",
                )
            ),

            # ----------------------------------------------------
            # Biomethane
            # ----------------------------------------------------

            "biomethane_base_cost_eur_per_mwh_hs": (
                biomethane_base_cost
            ),

            "biomethane_marginal_cost_eur_per_mwh_hs": (
                biomethane_effective_cost
            ),

            # ----------------------------------------------------
            # Biogenic CO2 sale
            # ----------------------------------------------------

            "co2_sale_active": (
                co2_sale_active
            ),

            "co2_sale_price_eur_per_t": (
                co2_sale_price
            ),

            "co2_additional_cost_eur_per_t": (
                co2_additional_cost
            ),

            "co2_net_sale_price_eur_per_t": (
                co2_net_sale_price
            ),

            "co2_yield_t_per_mwh_biomethane": (
                co2_yield
            ),

            "co2_marketable_fraction": (
                co2_marketable_fraction
            ),

            "co2_revenue_credit_eur_per_mwh_biomethane": (
                co2_revenue_credit
            ),

            # ----------------------------------------------------
            # Raw biogas / onsite CHP
            # ----------------------------------------------------

            "raw_biogas_cost_eur_per_mwh_hs": (
                raw_biogas_cost
            ),

            "onsite_electricity_marginal_cost_eur_per_mwh": (
                merchant_el_cost
            ),

            "onsite_supported_electricity_marginal_cost_eur_per_mwh": (
                supported_el_cost
            ),

            "onsite_heat_marginal_cost_eur_per_mwh": (
                onsite_heat_cost
            ),

            "eeg_premium_eur_per_mwh": (
                eeg_premium
            ),
        },

        # ========================================================
        # SUPPORT
        # ========================================================

        "support": {
            "case": selection["support_case"],
            **copy.deepcopy(support),
        },

        # ========================================================
        # CO2 SALE CASE
        # ========================================================

        "co2_sale": {
            "case": selection["co2_sale_case"],
            **copy.deepcopy(co2_sale),
        },

        # ========================================================
        # HEAT PUMPS
        # ========================================================

        "heat_pumps": copy.deepcopy(
            _named_case(
                config,
                "heat_pump_cases",
                selection["heat_pump_case"],
            )
        ),

        # ========================================================
        # SWFL UNITS
        # ========================================================

        "swfl_units": copy.deepcopy(
            _named_case(
                config,
                "swfl_unit_cases",
                selection["swfl_unit_case"],
            )
        ),

        # ========================================================
        # BIOMETHANE USE
        # ========================================================

        "biomethane_use": copy.deepcopy(
            _named_case(
                config,
                "biomethane_use_cases",
                selection["biomethane_use_case"],
            )
        ),

        # ========================================================
        # BIOGAS ROUTES
        # ========================================================

        "biogas_routes": copy.deepcopy(
            _named_case(
                config,
                "biogas_route_cases",
                selection["biogas_route_case"],
            )
        ),

        # ========================================================
        # TECHNICAL PARAMETERS
        # ========================================================

        "technical": copy.deepcopy(
            _require_mapping(
                config,
                "technical",
                "configuration",
            )
        ),

        # ========================================================
        # RUN SETTINGS
        # ========================================================

        "run": copy.deepcopy(
            config.get(
                "run",
                {},
            )
        ),
    }

def _mutable_mapping(
    mapping: MutableMapping[str, Any],
    key: str,
    context: str,
) -> MutableMapping[str, Any]:
    value = _require(mapping, key, context)
    if not isinstance(value, MutableMapping):
        raise ScenarioConfigError(f"'{key}' in {context} must be a mapping.")
    return value


def _update_named_assets(
    assets: Iterable[MutableMapping[str, Any]],
    selected_names: set[str],
    technical_data: Mapping[str, Mapping[str, Any]],
) -> None:
    by_name = {
        str(asset.get("name")): asset
        for asset in assets
        if isinstance(asset, MutableMapping) and asset.get("name") is not None
    }

    missing = selected_names - set(by_name)
    if missing:
        raise ScenarioConfigError(
            "Selected units are absent from the eTraGo args: "
            f"{sorted(missing)}"
        )

    for name, asset in by_name.items():
        asset["active"] = name in selected_names
        settings = technical_data.get(name, {})

        if "heat_capacity_mw" in settings:
            asset["heat_capacity_mw"] = float(settings["heat_capacity_mw"])
        if "efficiency" in settings:
            asset["efficiency"] = float(settings["efficiency"])
        if "cop" in settings:
            asset["cop"] = float(settings["cop"])
        if "planned_year" in settings:
            asset["planned_year"] = settings["planned_year"]


def _effective_ac_clusters(
    args: Mapping[str, Any],
) -> int | None:
    clustering = args.get(
        "network_clustering"
    )

    if not isinstance(
        clustering,
        Mapping,
    ):
        return None

    electricity_grid = clustering.get(
        "electricity_grid"
    )

    if not isinstance(
        electricity_grid,
        Mapping,
    ):
        return None

    value = electricity_grid.get(
        "n_clusters"
    )

    return (
        int(value)
        if value is not None
        else None
    )


def _apply_run_settings(
    args: MutableMapping[str, Any],
    resolved: MutableMapping[str, Any],
) -> None:
    """Apply snapshot, AC-clustering and result-directory settings."""
    run = resolved.get("run", {})

    if not isinstance(run, Mapping):
        raise ScenarioConfigError(
            "'run' must be a mapping."
        )

    start_override = run.get("start_snapshot")
    end_override = run.get("end_snapshot")
    ac_clusters_override = run.get("ac_clusters")

    if start_override is not None:
        args["start_snapshot"] = int(
            start_override
        )

    if end_override is not None:
        args["end_snapshot"] = int(
            end_override
        )

    if ac_clusters_override is not None:
        clustering = _mutable_mapping(
            args,
            "network_clustering",
            "eTraGo args",
        )

        electricity_grid = _mutable_mapping(
            clustering,
            "electricity_grid",
            "args.network_clustering",
        )

        electricity_grid["n_clusters"] = int(
            ac_clusters_override
        )

    start = args.get("start_snapshot")
    end = args.get("end_snapshot")

    represented_hours = None

    if start is not None and end is not None:
        start = int(start)
        end = int(end)

        represented_hours = end - start + 1

        if represented_hours <= 0:
            raise ScenarioConfigError(
                "end_snapshot must be greater than or "
                "equal to start_snapshot."
            )

    ac_clusters = _effective_ac_clusters(
        args
    )

    result_name_template = run.get(
        "result_name_template"
    )

    if result_name_template:
        if represented_hours is None:
            raise ScenarioConfigError(
                "result_name_template requires "
                "start_snapshot and end_snapshot."
            )

        if ac_clusters is None:
            raise ScenarioConfigError(
                "result_name_template requires a "
                "configured AC cluster count at "
                "args['network_clustering']"
                "['electricity_grid']['n_clusters']."
            )

        args["csv_export"] = str(
            result_name_template
        ).format(
            scenario=resolved["scenario_name"],
            hours=represented_hours,
            ac_clusters=ac_clusters,
        )

    resolved["effective_run"] = {
        "start_snapshot": start,
        "end_snapshot": end,
        "represented_hours": represented_hours,
        "ac_clusters": ac_clusters,
        "csv_export": args.get("csv_export"),
    }


def apply_network_price_scenario(
    network,
    resolved: Mapping[str, Any],
    biogas_sh_active: bool = True,
) -> None:
    """
    Apply scenario-dependent fuel and biomethane marginal costs.

    The function performs two separate price assignments:

    1. Fossil natural gas
       ------------------
       The selected CH4_NG marginal cost is assigned to all
       generators with carrier ``CH4_NG``.

       The final fossil-gas cost already includes:

           gas commodity price
           + CO2 certificate price * emission factor

    2. Biogas.SH biomethane
       ---------------------
       If Biogas.SH is active, the effective biomethane
       marginal cost is assigned to the 21 custom
       ``CH4_biogas`` generators.

       The effective biomethane cost may include a revenue
       credit from selling separated biogenic CO2:

           effective biomethane cost
           = base biomethane cost
           - biogenic CO2 revenue credit

    This function should run:
        - after adjust_CH4_gen_carriers()
        - after Biogas.SH assets have been added
        - before spatial clustering
    """

    # ============================================================
    # RESOLVED SCENARIO DATA
    # ============================================================

    prices = resolved["prices"]
    selection = resolved["selection"]

    # ============================================================
    # FOSSIL NATURAL-GAS PRICE
    # ============================================================

    gas_commodity_price = float(
        prices[
            "gas_commodity_price_eur_per_mwh_fuel"
        ]
    )

    fossil_co2_price = float(
        prices[
            "co2_price_eur_per_tco2"
        ]
    )

    fossil_emission_factor = float(
        prices[
            "emission_factor_tco2_per_mwh_fuel"
        ]
    )

    fossil_co2_cost = float(
        prices[
            "co2_cost_on_gas_eur_per_mwh_fuel"
        ]
    )

    fossil_gas_cost = float(
        prices[
            "final_ch4_ng_marginal_cost_eur_per_mwh_fuel"
        ]
    )

    # ------------------------------------------------------------
    # Find all fossil CH4_NG generators
    # ------------------------------------------------------------

    ch4_ng_ids = network.generators.index[
        network.generators["carrier"]
        .astype(str)
        .eq("CH4_NG")
    ]

    if len(ch4_ng_ids) == 0:
        raise ScenarioConfigError(
            "No CH4_NG generators were found. "
            "adjust_CH4_gen_carriers() must run before "
            "apply_network_price_scenario()."
        )

    # ------------------------------------------------------------
    # Apply fossil-gas marginal cost
    # ------------------------------------------------------------

    network.generators.loc[
        ch4_ng_ids,
        "marginal_cost",
    ] = fossil_gas_cost

    # ============================================================
    # BASELINE RUN WITHOUT BIOGAS.SH
    # ============================================================

    if not biogas_sh_active:

        print(
            "\n"
            "============================================================"
        )
        print(
            "APPLIED NETWORK PRICE SCENARIO"
        )
        print(
            "============================================================"
        )

        print(
            "\nFOSSIL NATURAL GAS"
        )
        print(
            "------------------------------------------------------------"
        )

        print(
            "Fossil-gas case:",
            selection[
                "fossil_gas_price_case"
            ],
        )

        print(
            "Gas commodity price:",
            f"{gas_commodity_price:.4f}",
            "EUR/MWh_fuel",
        )

        print(
            "Fossil CO2 price:",
            f"{fossil_co2_price:.4f}",
            "EUR/tCO2",
        )

        print(
            "Natural-gas emission factor:",
            f"{fossil_emission_factor:.4f}",
            "tCO2/MWh_fuel",
        )

        print(
            "CO2 cost on natural gas:",
            f"{fossil_co2_cost:.4f}",
            "EUR/MWh_fuel",
        )

        print(
            "Final CH4_NG marginal cost:",
            f"{fossil_gas_cost:.4f}",
            "EUR/MWh_fuel",
        )

        print(
            "CH4_NG generators updated:",
            len(ch4_ng_ids),
        )

        print(
            "\nBIOGAS.SH"
        )
        print(
            "------------------------------------------------------------"
        )

        print(
            "Biogas.SH inactive: "
            "biomethane price assignment skipped."
        )

        print(
            "============================================================\n"
        )

        return

    # ============================================================
    # BIOMETHANE PRICE
    # ============================================================

    biomethane_base_cost = float(
        prices[
            "biomethane_base_cost_eur_per_mwh_hs"
        ]
    )

    biomethane_effective_cost = float(
        prices[
            "biomethane_marginal_cost_eur_per_mwh_hs"
        ]
    )

    # ============================================================
    # BIOGENIC CO2 SALE ASSUMPTIONS
    # ============================================================

    co2_sale_active = bool(
        prices.get(
            "co2_sale_active",
            False,
        )
    )

    biogenic_co2_sale_price = float(
        prices.get(
            "co2_sale_price_eur_per_t",
            0.0,
        )
    )

    biogenic_co2_additional_cost = float(
        prices.get(
            "co2_additional_cost_eur_per_t",
            0.0,
        )
    )

    biogenic_co2_net_price = float(
        prices.get(
            "co2_net_sale_price_eur_per_t",
            0.0,
        )
    )

    biogenic_co2_yield = float(
        prices.get(
            "co2_yield_t_per_mwh_biomethane",
            0.0,
        )
    )

    co2_marketable_fraction = float(
        prices.get(
            "co2_marketable_fraction",
            1.0,
        )
    )

    co2_revenue_credit = float(
        prices.get(
            "co2_revenue_credit_eur_per_mwh_biomethane",
            0.0,
        )
    )

    # ============================================================
    # FIND CUSTOM BIOGAS.SH BIOMETHANE GENERATORS
    # ============================================================

    generator_names = (
        network.generators.index
        .to_series()
        .astype(str)
    )

    generator_buses = (
        network.generators["bus"]
        .astype(str)
    )

    generator_carriers = (
        network.generators["carrier"]
        .astype(str)
    )

    custom_biomethane_mask = (
        generator_carriers.eq(
            "CH4_biogas"
        )
    )

    custom_biomethane_mask &= (
        generator_names.str.startswith(
            "biogas_sh_ch4_bus_"
        )
        |
        generator_buses.str.startswith(
            "biogas_sh_ch4_bus_"
        )
    )

    custom_biomethane_ids = (
        network.generators.index[
            custom_biomethane_mask
        ]
    )

    # ------------------------------------------------------------
    # Validate expected Biogas.SH plant fleet
    # ------------------------------------------------------------

    expected_biomethane_generators = 21

    if (
        len(custom_biomethane_ids)
        != expected_biomethane_generators
    ):
        raise ScenarioConfigError(
            "Expected "
            f"{expected_biomethane_generators} "
            "custom Biogas.SH biomethane generators, "
            f"but found {len(custom_biomethane_ids)}."
        )

    # ============================================================
    # APPLY EFFECTIVE BIOMETHANE PRICE
    # ============================================================

    network.generators.loc[
        custom_biomethane_ids,
        "marginal_cost",
    ] = biomethane_effective_cost

    # ============================================================
    # FINAL VERIFICATION
    # ============================================================

    assigned_biomethane_costs = (
        network.generators.loc[
            custom_biomethane_ids,
            "marginal_cost",
        ]
        .astype(float)
    )

    if not (
        assigned_biomethane_costs
        .sub(
            biomethane_effective_cost
        )
        .abs()
        .le(1e-9)
        .all()
    ):
        raise ScenarioConfigError(
            "Biomethane marginal-cost assignment failed. "
            "Not all custom CH4_biogas generators received "
            "the expected effective biomethane cost."
        )

    # ============================================================
    # REPORT APPLIED PRICES
    # ============================================================

    print(
        "\n"
        "============================================================"
    )
    print(
        "APPLIED NETWORK PRICE SCENARIO"
    )
    print(
        "============================================================"
    )

    # ------------------------------------------------------------
    # Fossil natural gas
    # ------------------------------------------------------------

    print(
        "\nFOSSIL NATURAL GAS"
    )
    print(
        "------------------------------------------------------------"
    )

    print(
        "Fossil-gas case:",
        selection[
            "fossil_gas_price_case"
        ],
    )

    print(
        "Gas commodity price:",
        f"{gas_commodity_price:.4f}",
        "EUR/MWh_fuel",
    )

    print(
        "Fossil CO2 price:",
        f"{fossil_co2_price:.4f}",
        "EUR/tCO2",
    )

    print(
        "Natural-gas emission factor:",
        f"{fossil_emission_factor:.4f}",
        "tCO2/MWh_fuel",
    )

    print(
        "CO2 cost on natural gas:",
        f"{fossil_co2_cost:.4f}",
        "EUR/MWh_fuel",
    )

    print(
        "Final CH4_NG marginal cost:",
        f"{fossil_gas_cost:.4f}",
        "EUR/MWh_fuel",
    )

    print(
        "CH4_NG generators updated:",
        len(ch4_ng_ids),
    )

    # ------------------------------------------------------------
    # Biomethane
    # ------------------------------------------------------------

    print(
        "\nBIOGAS.SH BIOMETHANE"
    )
    print(
        "------------------------------------------------------------"
    )

    print(
        "Biomethane price case:",
        selection[
            "biomethane_price_case"
        ],
    )

    print(
        "Base biomethane cost:",
        f"{biomethane_base_cost:.4f}",
        "EUR/MWh_Hs",
    )

    # ------------------------------------------------------------
    # Biogenic CO2 sale
    # ------------------------------------------------------------

    print(
        "\nBIOGENIC CO2 SALE"
    )
    print(
        "------------------------------------------------------------"
    )

    print(
        "CO2 sale case:",
        selection[
            "co2_sale_case"
        ],
    )

    print(
        "CO2 sale active:",
        "yes"
        if co2_sale_active
        else "no",
    )

    print(
        "Biogenic CO2 sale price:",
        f"{biogenic_co2_sale_price:.4f}",
        "EUR/tCO2",
    )

    print(
        "Additional CO2 handling cost:",
        f"{biogenic_co2_additional_cost:.4f}",
        "EUR/tCO2",
    )

    print(
        "Net biogenic CO2 sale price:",
        f"{biogenic_co2_net_price:.4f}",
        "EUR/tCO2",
    )

    print(
        "Biogenic CO2 yield:",
        f"{biogenic_co2_yield:.5f}",
        "tCO2/MWh_Hs biomethane",
    )

    print(
        "Marketable CO2 fraction:",
        f"{100.0 * co2_marketable_fraction:.1f}",
        "%",
    )

    print(
        "CO2 revenue credit:",
        f"{co2_revenue_credit:.4f}",
        "EUR/MWh_Hs biomethane",
    )

    # ------------------------------------------------------------
    # Final biomethane price
    # ------------------------------------------------------------

    print(
        "\nFINAL BIOMETHANE COST"
    )
    print(
        "------------------------------------------------------------"
    )

    print(
        "Base biomethane cost:",
        f"{biomethane_base_cost:.4f}",
        "EUR/MWh_Hs",
    )

    print(
        "Minus CO2 revenue credit:",
        f"{co2_revenue_credit:.4f}",
        "EUR/MWh_Hs",
    )

    print(
        "Effective biomethane marginal cost:",
        f"{biomethane_effective_cost:.4f}",
        "EUR/MWh_Hs",
    )

    print(
        "Custom biomethane generators updated:",
        len(custom_biomethane_ids),
    )

    print(
        "============================================================\n"
    )


def apply_config_to_args(
    args: MutableMapping[str, Any],
    resolved: MutableMapping[str, Any],
) -> MutableMapping[str, Any]:
    """
    Apply the resolved Biogas.SH / SWFL scenario configuration to eTraGo args.

    The resolved YAML controls:

        - fossil natural-gas prices
        - biomethane prices
        - biogenic CO2-sale assumptions
        - EEG / flexibility support
        - Biogas.SH route activation
        - direct raw-biogas supply to SWFL
        - SWFL unit availability
        - biomethane eligibility
        - heat-pump configuration
        - regional raw-biogas resource constraint
        - central biomethane storage
        - run settings

    Biogas.SH fuel pathways
    -----------------------

    1. Onsite CHP electricity

        raw biogas
            -> onsite CHP
            -> electricity


    2. Onsite heat

        raw biogas
            -> onsite CHP / heat use
            -> heat


    3. Upgraded biomethane -> public gas grid

        raw biogas
            -> collection
            -> upgrading
            -> biomethane
            -> central storage
            -> public CH4 grid

       Biogenic CO2 is separated only on this upgrading pathway.


    4. Direct raw biogas -> SWFL

        raw biogas
            -> collection / transport
            -> SWFL raw-biogas bus
            -> eligible SWFL boilers
            -> heat

       This pathway:

            - does NOT use biomethane upgrading
            - does NOT apply eta_upgrade
            - does NOT receive a CO2-sale credit
            - uses raw-biogas cost + direct delivery cost


    Important
    ---------
    The direct raw-biogas route and all other Biogas.SH uses must share
    the same regional raw-biogas resource constraint. Therefore the
    resource constraint is required whenever direct raw-biogas supply
    to SWFL is active.
    """

    # =====================================================================
    # 1. TOP-LEVEL ARGUMENT SECTIONS
    # =====================================================================

    swfl = _mutable_mapping(
        args,
        "swfl_real_system",
        "eTraGo args",
    )

    biogas = _mutable_mapping(
        args,
        "biogas_sh",
        "eTraGo args",
    )

    extra_functionality = args.get(
        "extra_functionality"
    )

    if not isinstance(
        extra_functionality,
        MutableMapping,
    ):
        extra_functionality = {}
        args["extra_functionality"] = (
            extra_functionality
        )


    # =====================================================================
    # 2. RESOLVED SCENARIO SECTIONS
    # =====================================================================

    selection = resolved["selection"]

    prices = resolved["prices"]

    support = resolved["support"]

    units = resolved["swfl_units"]

    heat_pump_case = resolved[
        "heat_pumps"
    ]

    biomethane_use = resolved[
        "biomethane_use"
    ]

    routes = resolved[
        "biogas_routes"
    ]

    technical = resolved[
        "technical"
    ]

    technical_swfl = technical[
        "swfl"
    ]

    technical_biogas = technical[
        "biogas_sh"
    ]


    # =====================================================================
    # 3. RESOLVE BIOGAS.SH ROUTE SWITCHES
    # =====================================================================

    route_name = str(
        selection[
            "biogas_route_case"
        ]
    )

    add_local_generation = bool(
        routes.get(
            "add_local_generation",
            False,
        )
    )

    add_gas_grid_generation = bool(
        routes.get(
            "add_gas_grid_generation",
            False,
        )
    )

    add_swfl_biomethane_supply = bool(
        routes.get(
            "add_swfl_direct_supply",
            False,
        )
    )

    add_swfl_raw_biogas_supply = bool(
        routes.get(
            "add_swfl_raw_biogas_supply",
            False,
        )
    )


    # ---------------------------------------------------------------------
    # Existing biogas_sh.py understands "hybrid" as a special mode.
    #
    # New route combinations use "custom" and the explicit booleans below.
    # ---------------------------------------------------------------------

    if route_name == "hybrid":

        biogas[
            "scenario_mode"
        ] = "hybrid"

    else:

        biogas[
            "scenario_mode"
        ] = "custom"


    biogas[
        "add_local_generation"
    ] = add_local_generation

    biogas[
        "add_gas_grid_generation"
    ] = add_gas_grid_generation

    biogas[
        "add_swfl_direct_supply"
    ] = add_swfl_biomethane_supply

    biogas[
        "add_swfl_raw_biogas_supply"
    ] = add_swfl_raw_biogas_supply


    # ---------------------------------------------------------------------
    # SWFL still requires access to the public CH4 grid even when the
    # upgraded-biomethane-to-SWFL route is disabled.
    #
    # biogas_sh.py should later use this flag when deciding whether the
    # SWFL natural-gas bus / public-grid supply link must exist.
    # ---------------------------------------------------------------------

    biogas[
        "ensure_swfl_gas_access"
    ] = bool(
        add_swfl_biomethane_supply
        or add_swfl_raw_biogas_supply
    )


    # =====================================================================
    # 4. GENERAL BIOGAS.SH PRICES
    # =====================================================================

    swfl_direct = _mutable_mapping(
        biogas,
        "swfl_direct",
        "args.biogas_sh",
    )


    # ---------------------------------------------------------------------
    # Public-grid -> SWFL natural-gas link.
    #
    # Fossil-gas commodity + CO2 costs are already assigned upstream to
    # CH4_NG generators. Only the configured import adder belongs here.
    # ---------------------------------------------------------------------

    swfl_direct[
        "grid_supply_marginal_cost"
    ] = float(
        prices[
            "swfl_import_adder_eur_per_mwh_fuel"
        ]
    )


    # ---------------------------------------------------------------------
    # Raw-biogas fuel price
    # ---------------------------------------------------------------------

    raw_biogas_cost = float(
        prices[
            "raw_biogas_cost_eur_per_mwh_hs"
        ]
    )

    biogas[
        "raw_biogas_cost_eur_per_mwh_hs"
    ] = raw_biogas_cost


    # ---------------------------------------------------------------------
    # Effective biomethane cost.
    #
    # This may already contain the CO2 revenue credit calculated in
    # resolve_config().
    # ---------------------------------------------------------------------

    effective_biomethane_cost = float(
        prices[
            "biomethane_marginal_cost_eur_per_mwh_hs"
        ]
    )

    biogas[
        "biomethane_price_override_eur_per_mwh"
    ] = effective_biomethane_cost

    biogas[
        "default_biomethane_cost"
    ] = effective_biomethane_cost


    # ---------------------------------------------------------------------
    # Onsite Biogas.SH costs
    # ---------------------------------------------------------------------

    biogas[
        "electricity_marginal_cost"
    ] = float(
        prices[
            "onsite_electricity_marginal_cost_eur_per_mwh"
        ]
    )

    biogas[
        "heat_marginal_cost"
    ] = float(
        prices[
            "onsite_heat_marginal_cost_eur_per_mwh"
        ]
    )


    # =====================================================================
    # 5. BIOGAS.SH EEG / FLEXIBILITY SUPPORT
    # =====================================================================

    support_args = biogas.get(
        "support"
    )

    if not isinstance(
        support_args,
        MutableMapping,
    ):
        support_args = {}
        biogas[
            "support"
        ] = support_args


    support_args[
        "case"
    ] = str(
        support[
            "case"
        ]
    )

    support_args[
        "eeg_active"
    ] = bool(
        support.get(
            "eeg_active",
            False,
        )
    )

    support_args[
        "market_electricity_marginal_cost"
    ] = float(
        prices[
            "onsite_electricity_marginal_cost_eur_per_mwh"
        ]
    )

    support_args[
        "supported_electricity_marginal_cost"
    ] = float(
        prices[
            "onsite_supported_electricity_marginal_cost_eur_per_mwh"
        ]
    )

    support_args[
        "supported_hours_per_year"
    ] = float(
        support.get(
            "supported_hours_per_year",
            0.0,
        )
    )

    support_args[
        "flexibility_active"
    ] = bool(
        support.get(
            "flexibility_active",
            False,
        )
    )

    support_args[
        "chp_capacity_multiplier"
    ] = float(
        support.get(
            "chp_capacity_multiplier",
            1.0,
        )
    )

    support_args[
        "flex_capex_eur_per_kw"
    ] = float(
        support.get(
            "flex_capex_eur_per_kw",
            800.0,
        )
    )

    support_args[
        "flex_lifetime_years"
    ] = float(
        support.get(
            "flex_lifetime_years",
            15.0,
        )
    )

    support_args[
        "flex_discount_rate"
    ] = float(
        support.get(
            "flex_discount_rate",
            0.05,
        )
    )

    support_args[
        "flex_fixed_om_fraction"
    ] = float(
        support.get(
            "flex_fixed_om_fraction",
            0.02,
        )
    )

    support_args[
        "flexibility_payment_eur_per_kw_year"
    ] = float(
        support.get(
            "flexibility_payment_eur_per_kw_year",
            0.0,
        )
    )


    # ---------------------------------------------------------------------
    # EEG support constraint
    # ---------------------------------------------------------------------

    if support_args[
        "eeg_active"
    ]:

        extra_functionality[
            "biogas_sh_support"
        ] = {
            "active": True,
            "supported_hours_per_year": (
                support_args[
                    "supported_hours_per_year"
                ]
            ),
            "ignore_missing_components": False,
        }

    else:

        extra_functionality.pop(
            "biogas_sh_support",
            None,
        )


    # =====================================================================
    # 6. SWFL GAS-TO-POWER
    # =====================================================================

    central_gas_to_power = _mutable_mapping(
        swfl,
        "central_gas_chp",
        "args.swfl_real_system",
    )

    gas_to_power_active = bool(
        units[
            "central_gas_to_power"
        ]
    )

    central_gas_to_power[
        "active"
    ] = gas_to_power_active

    central_gas_to_power[
        "add_electric_link"
    ] = gas_to_power_active


    # =====================================================================
    # 7. SWFL CENTRAL HEAT SYSTEM
    # =====================================================================

    central_heat = _mutable_mapping(
        swfl,
        "central_heat_units",
        "args.swfl_real_system",
    )


    selected_boilers = set(
        map(
            str,
            units.get(
                "boilers",
                [],
            ),
        )
    )

    selected_resistive = set(
        map(
            str,
            units.get(
                "resistive_heaters",
                [],
            ),
        )
    )


    boiler_data = technical_swfl[
        "boilers"
    ]

    resistive_data = technical_swfl[
        "resistive_heaters"
    ]


    _update_named_assets(
        central_heat[
            "boilers"
        ],
        selected_boilers,
        boiler_data,
    )

    _update_named_assets(
        central_heat[
            "resistive_heaters"
        ],
        selected_resistive,
        resistive_data,
    )


    selected_heat_capacity = sum(
        float(
            boiler_data[
                name
            ][
                "heat_capacity_mw"
            ]
        )
        for name in selected_boilers
    )


    selected_heat_capacity += sum(
        float(
            resistive_data[
                name
            ][
                "heat_capacity_mw"
            ]
        )
        for name in selected_resistive
    )


    central_heat[
        "active"
    ] = bool(
        selected_boilers
        or selected_resistive
    )

    central_heat[
        "expected_total_heat_capacity_mw"
    ] = selected_heat_capacity


    # =====================================================================
    # 8. RESERVE GAS BOILER
    # =====================================================================

    reserve = _mutable_mapping(
        swfl,
        "reserve_gas_boiler",
        "args.swfl_real_system",
    )

    reserve[
        "active"
    ] = bool(
        units.get(
            "reserve_gas_boiler",
            False,
        )
    )


    # =====================================================================
    # 9. BIOMETHANE-ELIGIBLE SWFL BOILERS
    # =====================================================================
    #
    # If the upgraded biomethane -> SWFL route is disabled, do not create
    # unnecessary biomethane supply links at K12/K13.
    # =====================================================================

    configured_biomethane_units = list(
        map(
            str,
            biomethane_use.get(
                "eligible_units",
                [],
            ),
        )
    )


    if add_swfl_biomethane_supply:

        biomethane_units = (
            configured_biomethane_units
        )

        central_heat[
            "biomethane_mode"
        ] = str(
            biomethane_use[
                "mode"
            ]
        )

    else:

        biomethane_units = []

        central_heat[
            "biomethane_mode"
        ] = "off"


    central_heat[
        "planned_biomethane_units"
    ] = list(
        biomethane_units
    )

    central_heat[
        "custom_biomethane_units"
    ] = list(
        biomethane_units
    )


    # =====================================================================
    # 10. SWFL HEAT-PUMP CONFIGURATION
    # =====================================================================

    future_heat_pumps = _mutable_mapping(
        swfl,
        "future_heat_pumps",
        "args.swfl_real_system",
    )


    selected_heat_pumps = list(
        map(
            str,
            heat_pump_case.get(
                "active_units",
                [],
            ),
        )
    )


    future_heat_pumps[
        "active"
    ] = bool(
        selected_heat_pumps
    )

    future_heat_pumps[
        "active_units"
    ] = selected_heat_pumps


    _update_named_assets(
        future_heat_pumps[
            "units"
        ],
        set(
            selected_heat_pumps
        ),
        technical_swfl[
            "heat_pumps"
        ],
    )


    # =====================================================================
    # 11. SWFL BUS CONFIGURATION
    # =====================================================================

    buses = technical_swfl[
        "buses"
    ]


    required_buses = {
        "ac",
        "heat",
        "natural_gas",
        "biomethane",
        "public_ch4",
    }


    if add_swfl_raw_biogas_supply:

        required_buses.add(
            "raw_biogas"
        )


    missing_bus_keys = (
        required_buses
        - set(
            buses.keys()
        )
    )


    if missing_bus_keys:

        raise ScenarioConfigError(
            "Missing SWFL bus definitions in "
            "technical.swfl.buses: "
            f"{sorted(missing_bus_keys)}"
        )


    swfl[
        "swfl_ac_bus"
    ] = str(
        buses[
            "ac"
        ]
    )

    swfl[
        "swfl_heat_bus"
    ] = str(
        buses[
            "heat"
        ]
    )

    swfl[
        "swfl_ch4_bus"
    ] = str(
        buses[
            "natural_gas"
        ]
    )

    swfl[
        "selected_mv_grid_district_ids"
    ] = list(
        map(
            str,
            technical_swfl[
                "selected_mv_grid_district_ids"
            ],
        )
    )


    # ---------------------------------------------------------------------
    # Central heat-system buses
    # ---------------------------------------------------------------------

    central_heat[
        "natural_gas_bus"
    ] = str(
        buses[
            "natural_gas"
        ]
    )

    central_heat[
        "biomethane_bus"
    ] = str(
        buses[
            "biomethane"
        ]
    )


    if (
        "raw_biogas"
        in buses
    ):

        central_heat[
            "raw_biogas_bus"
        ] = str(
            buses[
                "raw_biogas"
            ]
        )


    central_heat[
        "ac_bus"
    ] = str(
        buses[
            "ac"
        ]
    )

    central_heat[
        "heat_bus"
    ] = str(
        buses[
            "heat"
        ]
    )


    # ---------------------------------------------------------------------
    # Gas-to-power buses
    # ---------------------------------------------------------------------

    central_gas_to_power[
        "gas_bus"
    ] = str(
        buses[
            "natural_gas"
        ]
    )

    central_gas_to_power[
        "ac_bus"
    ] = str(
        buses[
            "ac"
        ]
    )

    central_gas_to_power[
        "heat_bus"
    ] = str(
        buses[
            "heat"
        ]
    )


    # =====================================================================
    # 12. DIRECT RAW-BIOGAS SUPPLY TO SWFL
    # =====================================================================
    #
    # Economic structure:
    #
    #     delivered raw-biogas cost
    #
    #       = raw-biogas fuel cost
    #       + collection / transport cost
    #
    # No biomethane upgrading cost and no CO2-sale credit are included.
    # =====================================================================

    direct_raw_swfl = (
        technical_biogas.get(
            "direct_raw_biogas_to_swfl",
            {},
        )
        or {}
    )


    if (
        add_swfl_raw_biogas_supply
        and not isinstance(
            direct_raw_swfl,
            Mapping,
        )
    ):

        raise ScenarioConfigError(
            "technical.biogas_sh."
            "direct_raw_biogas_to_swfl "
            "must be a mapping."
        )


    if (
        add_swfl_raw_biogas_supply
        and not direct_raw_swfl
    ):

        raise ScenarioConfigError(
            "The selected route activates direct raw-biogas "
            "supply to SWFL, but "
            "technical.biogas_sh.direct_raw_biogas_to_swfl "
            "is missing."
        )


    raw_swfl_args = biogas.get(
        "raw_biogas_to_swfl"
    )

    if raw_swfl_args is None:
        raw_swfl_args = {}
        biogas[
            "raw_biogas_to_swfl"
        ] = raw_swfl_args

    elif not isinstance(
            raw_swfl_args,
            MutableMapping,
    ):
        raise ScenarioConfigError(
            "'raw_biogas_to_swfl' in args.biogas_sh "
            "must be a mapping."
        )


    raw_swfl_args[
        "active"
    ] = add_swfl_raw_biogas_supply


    # ---------------------------------------------------------------------
    # Dedicated fuel bus
    # ---------------------------------------------------------------------

    if add_swfl_raw_biogas_supply:

        raw_swfl_bus = str(
            direct_raw_swfl.get(
                "target_bus",
                buses[
                    "raw_biogas"
                ],
            )
        )

    else:

        raw_swfl_bus = str(
            buses.get(
                "raw_biogas",
                "swfl_real_raw_biogas_bus",
            )
        )


    raw_swfl_args[
        "target_bus"
    ] = raw_swfl_bus

    central_heat[
        "raw_biogas_bus"
    ] = raw_swfl_bus


    # ---------------------------------------------------------------------
    # Raw-biogas fuel cost
    #
    # Prefer the common project raw-biogas price from price_cases.onsite.
    # The technical block should contain only route-specific additions.
    # ---------------------------------------------------------------------

    raw_swfl_args[
        "raw_biogas_cost_eur_per_mwh_hs"
    ] = raw_biogas_cost


    # ---------------------------------------------------------------------
    # Collection / direct-delivery cost
    #
    # Current source-derived main assumption:
    #
    #       8.78 EUR/MWh_Hs
    #
    # derived separately from Treurat's investment and P&L values.
    # ---------------------------------------------------------------------

    raw_swfl_transport_cost = float(
        direct_raw_swfl.get(
            "transport_cost_eur_per_mwh_hs",
            0.0,
        )
    )


    if raw_swfl_transport_cost < 0:

        raise ScenarioConfigError(
            "direct_raw_biogas_to_swfl."
            "transport_cost_eur_per_mwh_hs "
            "must be non-negative."
        )


    raw_swfl_args[
        "transport_cost_eur_per_mwh_hs"
    ] = raw_swfl_transport_cost


    # ---------------------------------------------------------------------
    # Final direct-delivery marginal cost.
    #
    # Example:
    #
    #     75.00 + 8.78
    #     = 83.78 EUR/MWh_Hs
    # ---------------------------------------------------------------------

    raw_swfl_args[
        "marginal_cost_eur_per_mwh_hs"
    ] = (
        raw_biogas_cost
        + raw_swfl_transport_cost
    )


    # ---------------------------------------------------------------------
    # Optional aggregate delivery capacity
    #
    # None means biogas_sh.py may derive the capacity from the regional
    # annual raw-biogas potential.
    # ---------------------------------------------------------------------

    configured_raw_swfl_capacity = (
        direct_raw_swfl.get(
            "power_capacity_mw",
            None,
        )
    )


    if (
        configured_raw_swfl_capacity
        is None
    ):

        raw_swfl_args[
            "power_capacity_mw"
        ] = None

    else:

        configured_raw_swfl_capacity = float(
            configured_raw_swfl_capacity
        )

        if configured_raw_swfl_capacity < 0:

            raise ScenarioConfigError(
                "direct_raw_biogas_to_swfl."
                "power_capacity_mw "
                "must be non-negative."
            )

        raw_swfl_args[
            "power_capacity_mw"
        ] = configured_raw_swfl_capacity


    # ---------------------------------------------------------------------
    # Stable component names / carriers
    # ---------------------------------------------------------------------

    raw_swfl_args[
        "generator_name"
    ] = str(
        direct_raw_swfl.get(
            "generator_name",
            "biogas_sh_raw_biogas_swfl_supply",
        )
    )

    raw_swfl_args[
        "generator_carrier"
    ] = str(
        direct_raw_swfl.get(
            "generator_carrier",
            "biogas_sh_raw_biogas_swfl",
        )
    )

    raw_swfl_args[
        "bus_carrier"
    ] = str(
        direct_raw_swfl.get(
            "bus_carrier",
            "raw_biogas",
        )
    )


    # =====================================================================
    # 13. RAW-BIOGAS-ELIGIBLE SWFL BOILERS
    # =====================================================================

    configured_raw_biogas_units = list(
        map(
            str,
            direct_raw_swfl.get(
                "eligible_units",
                [],
            ),
        )
    )


    if add_swfl_raw_biogas_supply:

        raw_biogas_units = (
            configured_raw_biogas_units
        )

    else:

        raw_biogas_units = []


    # ---------------------------------------------------------------------
    # Check that raw-biogas units actually exist.
    # ---------------------------------------------------------------------

    unknown_raw_units = (
        set(
            raw_biogas_units
        )
        - set(
            boiler_data.keys()
        )
    )


    if unknown_raw_units:

        raise ScenarioConfigError(
            "Unknown raw-biogas SWFL boiler units: "
            f"{sorted(unknown_raw_units)}"
        )


    # ---------------------------------------------------------------------
    # Check that all raw-biogas-eligible boilers are active in this run.
    # ---------------------------------------------------------------------

    inactive_raw_units = (
        set(
            raw_biogas_units
        )
        - selected_boilers
    )


    if inactive_raw_units:

        raise ScenarioConfigError(
            "Raw-biogas-eligible SWFL boilers are inactive "
            "in the selected swfl_unit_case: "
            f"{sorted(inactive_raw_units)}"
        )


    raw_swfl_args[
        "eligible_units"
    ] = list(
        raw_biogas_units
    )

    central_heat[
        "raw_biogas_units"
    ] = list(
        raw_biogas_units
    )


    # =====================================================================
    # 14. SWFL LOAD ASSUMPTIONS
    # =====================================================================

    loads = technical_swfl[
        "loads"
    ]


    heat_load = _mutable_mapping(
        swfl,
        "heat_load",
        "args.swfl_real_system",
    )

    ac_load = _mutable_mapping(
        swfl,
        "ac_load",
        "args.swfl_real_system",
    )


    if loads.get(
        "heat_csv_path"
    ):

        heat_load[
            "csv_path"
        ] = str(
            loads[
                "heat_csv_path"
            ]
        )


    heat_load[
        "year"
    ] = int(
        loads[
            "heat_year"
        ]
    )

    heat_load[
        "datetime_column"
    ] = str(
        loads[
            "heat_datetime_column"
        ]
    )

    heat_load[
        "column"
    ] = str(
        loads[
            "heat_column"
        ]
    )


    ac_load[
        "target_annual_demand_mwh"
    ] = float(
        loads[
            "ac_target_annual_demand_mwh"
        ]
    )


    # =====================================================================
    # 15. SWFL TECHNOLOGY ASSUMPTIONS
    # =====================================================================

    gas_to_power_data = technical_swfl[
        "central_gas_to_power"
    ]


    central_gas_to_power[
        "electric_capacity_mw"
    ] = float(
        gas_to_power_data[
            "electric_capacity_mw"
        ]
    )

    central_gas_to_power[
        "electric_efficiency"
    ] = float(
        gas_to_power_data[
            "electric_efficiency"
        ]
    )

    central_gas_to_power[
        "marginal_cost"
    ] = float(
        gas_to_power_data[
            "marginal_cost_eur_per_mwh"
        ]
    )


    reserve_data = technical_swfl[
        "reserve_gas_boiler"
    ]


    reserve[
        "heat_capacity_mw"
    ] = float(
        reserve_data[
            "heat_capacity_mw"
        ]
    )

    reserve[
        "efficiency"
    ] = float(
        reserve_data[
            "efficiency"
        ]
    )


    # =====================================================================
    # 16. PUBLIC-GRID NATURAL-GAS SUPPLY TO SWFL
    # =====================================================================

    public_supply = technical_biogas[
        "public_grid_to_swfl"
    ]


    swfl_direct[
        "grid_supply_p_nom"
    ] = float(
        public_supply[
            "power_capacity_mw"
        ]
    )

    swfl_direct[
        "grid_supply_efficiency"
    ] = float(
        public_supply[
            "efficiency"
        ]
    )

    swfl_direct[
        "grid_supply_capital_cost"
    ] = float(
        public_supply[
            "capital_cost_eur_per_mw"
        ]
    )


    # =====================================================================
    # 17. SHARED REGIONAL RAW-BIOGAS RESOURCE CONSTRAINT
    # =====================================================================

    resource_settings = technical_biogas[
        "resource_constraint"
    ]


    resource_active = bool(
        resource_settings.get(
            "active",
            True,
        )
    )


    # ---------------------------------------------------------------------
    # The new raw-biogas -> SWFL route MUST be constrained by the same
    # regional resource as onsite generation and biomethane upgrading.
    # ---------------------------------------------------------------------

    if (
        add_swfl_raw_biogas_supply
        and not resource_active
    ):

        raise ScenarioConfigError(
            "Direct raw-biogas supply to SWFL is active, but "
            "technical.biogas_sh.resource_constraint.active "
            "is false. Enable the shared Biogas.SH resource "
            "constraint to prevent double counting of raw biogas."
        )


    if resource_active:

        resource = extra_functionality.setdefault(
            "biogas_sh_resource",
            {},
        )


        if not isinstance(
            resource,
            MutableMapping,
        ):

            raise ScenarioConfigError(
                "args.extra_functionality."
                "biogas_sh_resource "
                "must be a mapping."
            )


        csv_path = (
            resource.get(
                "csv_path"
            )
            or biogas.get(
                "csv_path"
            )
        )


        if not csv_path:

            raise ScenarioConfigError(
                "The active Biogas.SH resource constraint "
                "requires csv_path in "
                "args.extra_functionality.biogas_sh_resource "
                "or args.biogas_sh.csv_path."
            )


        resource[
            "csv_path"
        ] = csv_path


        efficiencies = technical_biogas[
            "efficiencies"
        ]


        resource[
            "eta_el"
        ] = float(
            efficiencies[
                "onsite_electricity"
            ]
        )

        resource[
            "eta_heat"
        ] = float(
            efficiencies[
                "onsite_heat"
            ]
        )

        resource[
            "eta_upgrade"
        ] = float(
            efficiencies[
                "upgrading"
            ]
        )


        # -------------------------------------------------------------
        # Direct raw-biogas delivery uses no upgrading.
        #
        # The resource equation therefore contains:
        #
        #     E_raw_to_SWFL / 1.0
        #
        # rather than:
        #
        #     E_raw_to_SWFL / eta_upgrade
        # -------------------------------------------------------------

        resource[
            "eta_raw_swfl"
        ] = 1.0

        resource[
            "raw_swfl_generator_carrier"
        ] = str(
            raw_swfl_args[
                "generator_carrier"
            ]
        )


        resource[
            "ignore_missing_components"
        ] = bool(
            resource_settings[
                "ignore_missing_components"
            ]
        )

    else:

        extra_functionality.pop(
            "biogas_sh_resource",
            None,
        )


    # =====================================================================
    # 18. CENTRAL BIOGAS.SH BIOMETHANE STORAGE
    # =====================================================================
    #
    # The central storage remains a BIOMETHANE storage.
    #
    # Raw biogas sent directly to SWFL does NOT pass through this storage.
    # =====================================================================

    storage = technical_biogas[
        "storage"
    ]


    storage_args = _mutable_mapping(
        biogas,
        "gas_storage",
        "args.biogas_sh",
    )


    storage_args[
        "active"
    ] = bool(
        storage[
            "active"
        ]
    )

    storage_args[
        "e_nom_mwh"
    ] = float(
        storage[
            "energy_capacity_mwh"
        ]
    )

    storage_args[
        "e_initial"
    ] = float(
        storage[
            "initial_energy_mwh"
        ]
    )

    storage_args[
        "e_cyclic"
    ] = bool(
        storage[
            "cyclic"
        ]
    )

    storage_args[
        "standing_loss"
    ] = float(
        storage[
            "standing_loss"
        ]
    )


    # ---------------------------------------------------------------------
    # Plant biomethane -> central storage
    # ---------------------------------------------------------------------

    storage_args[
        "input_link_efficiency"
    ] = float(
        storage[
            "plant_to_storage"
        ][
            "efficiency"
        ]
    )

    storage_args[
        "input_link_marginal_cost"
    ] = float(
        storage[
            "plant_to_storage"
        ][
            "marginal_cost_eur_per_mwh"
        ]
    )


    # ---------------------------------------------------------------------
    # Central biomethane storage -> public gas grid
    # ---------------------------------------------------------------------

    storage_args[
        "grid_link_p_nom_mw"
    ] = float(
        storage[
            "storage_to_public_grid"
        ][
            "power_capacity_mw"
        ]
    )

    storage_args[
        "grid_link_efficiency"
    ] = float(
        storage[
            "storage_to_public_grid"
        ][
            "efficiency"
        ]
    )

    storage_args[
        "grid_link_marginal_cost"
    ] = float(
        storage[
            "storage_to_public_grid"
        ][
            "marginal_cost_eur_per_mwh"
        ]
    )


    # ---------------------------------------------------------------------
    # Existing biomethane storage -> SWFL route.
    #
    # The parameters remain available for old scenarios. Whether this link
    # is actually created is controlled by add_swfl_direct_supply.
    # ---------------------------------------------------------------------

    storage_to_swfl = storage.get(
        "storage_to_swfl",
        {},
    ) or {}


    if storage_to_swfl:

        storage_args[
            "swfl_link_p_nom_mw"
        ] = float(
            storage_to_swfl[
                "power_capacity_mw"
            ]
        )

        storage_args[
            "swfl_link_efficiency"
        ] = float(
            storage_to_swfl[
                "efficiency"
            ]
        )

        storage_args[
            "swfl_link_marginal_cost"
        ] = float(
            storage_to_swfl[
                "marginal_cost_eur_per_mwh"
            ]
        )


    storage_args[
        "swfl_target_bus"
    ] = str(
        buses[
            "biomethane"
        ]
    )


    # =====================================================================
    # 19. RUN SETTINGS AND SCENARIO IDENTIFICATION
    # =====================================================================

    _apply_run_settings(
        args,
        resolved,
    )


    args[
        "biogas_sh_scenario_name"
    ] = str(
        resolved[
            "scenario_name"
        ]
    )


    # =====================================================================
    # 20. DIAGNOSTIC SUMMARY
    # =====================================================================

    print(
        "\n"
        "============================================================"
    )

    print(
        "APPLIED BIOGAS.SH ROUTE CONFIGURATION"
    )

    print(
        "============================================================"
    )

    print(
        "Route case:",
        route_name,
    )

    print(
        "Onsite generation:",
        add_local_generation,
    )

    print(
        "Biomethane -> public grid:",
        add_gas_grid_generation,
    )

    print(
        "Biomethane -> SWFL:",
        add_swfl_biomethane_supply,
    )

    print(
        "Raw biogas -> SWFL:",
        add_swfl_raw_biogas_supply,
    )


    if add_swfl_raw_biogas_supply:

        print(
            "\nDIRECT RAW-BIOGAS -> SWFL"
        )

        print(
            "------------------------------------------------------------"
        )

        print(
            "SWFL raw-biogas bus:",
            raw_swfl_bus,
        )

        print(
            "Eligible SWFL units:",
            ", ".join(
                raw_biogas_units
            )
            if raw_biogas_units
            else "none",
        )

        print(
            "Raw-biogas cost:",
            f"{raw_biogas_cost:.2f}",
            "EUR/MWh_Hs",
        )

        print(
            "Collection / transport cost:",
            f"{raw_swfl_transport_cost:.2f}",
            "EUR/MWh_Hs",
        )

        print(
            "Delivered raw-biogas cost:",
            (
                f"{raw_swfl_args['marginal_cost_eur_per_mwh_hs']:.2f}"
            ),
            "EUR/MWh_Hs",
        )

        print(
            "Upgrading efficiency applied:",
            "no",
        )

        print(
            "CO2-sale credit applied:",
            "no",
        )

        print(
            "Shared raw-biogas constraint:",
            "active",
        )


    print(
        "============================================================\n"
    )


    return args


def load_and_apply_config(
    args: MutableMapping[str, Any],
    path: str | Path = "config.yaml",
) -> Tuple[MutableMapping[str, Any], Dict[str, Any]]:
    """Load, resolve and apply one YAML scenario."""
    config = load_config(path)
    resolved = resolve_config(config)
    apply_config_to_args(args, resolved)
    return args, resolved


def scenario_summary(
    resolved: Mapping[str, Any],
) -> str:
    """Return a readable summary of the resolved scenario and run settings."""

    selection = resolved["selection"]
    prices = resolved["prices"]
    support = resolved["support"]
    heat_pumps = resolved["heat_pumps"]
    units = resolved["swfl_units"]
    biomethane_use = resolved["biomethane_use"]
    biogas_routes = resolved["biogas_routes"]
    technical_biogas = resolved["technical"]["biogas_sh"]
    effective_run = resolved.get("effective_run", {})

    active_heat_pumps = list(
        map(
            str,
            heat_pumps.get("active_units", []),
        )
    )

    active_boilers = list(
        map(
            str,
            units.get("boilers", []),
        )
    )

    active_resistive_heaters = list(
        map(
            str,
            units.get("resistive_heaters", []),
        )
    )

    biomethane_eligible_units = list(
        map(
            str,
            biomethane_use.get("eligible_units", []),
        )
    )

    gas_source = (
        str(prices.get("gas_source", "")).strip()
        or "not specified"
    )

    co2_source = (
        str(prices.get("co2_source", "")).strip()
        or "not specified"
    )

    gas_value_status = (
        str(prices.get("gas_value_status", "")).strip()
        or "not specified"
    )

    def value_or_none(values: list[str]) -> str:
        return ", ".join(values) if values else "none"

    def yes_no(value: Any) -> str:
        return "yes" if bool(value) else "no"

    lines = [
        "",
        "============================================================",
        "BIOGAS.SH / SWFL SCENARIO",
        "============================================================",
        "",
        "SCENARIO IDENTIFICATION",
        "------------------------------------------------------------",
        f"Scenario name:                  {resolved['scenario_name']}",
        f"Configuration version:          {resolved['config_version']}",
        "",
        "BIOGAS.SH SUPPORT REGIME",
        "------------------------------------------------------------",
        (
            "Support case:                   "
            f"{selection['support_case']}"
        ),
        (
            "EEG support active:             "
            f"{yes_no(support.get('eeg_active', False))}"
        ),
        (
            "Supported hours per year:       "
            f"{float(support.get('supported_hours_per_year', 0.0)):.1f}"
        ),
        (
            "CHP capacity multiplier:        "
            f"{float(support.get('chp_capacity_multiplier', 1.0)):.2f}"
        ),
        (
            "Flexibilisation active:         "
            f"{yes_no(support.get('flexibility_active', False))}"
        ),
        (
            "Flexibility payment:            "
            f"{float(support.get('flexibility_payment_eur_per_kw_year', 0.0)):.2f} "
            "EUR/kW_el/a"
        ),
        "",
        "FOSSIL NATURAL-GAS PRICE",
        "------------------------------------------------------------",
        (
            "Price scenario:                 "
            f"{selection['fossil_gas_price_case']}"
        ),
        (
            "Gas commodity price:            "
            f"{prices['gas_commodity_price_eur_per_mwh_fuel']:.4f} "
            "EUR/MWh_fuel"
        ),
        (
            "CO2 certificate price:          "
            f"{prices['co2_price_eur_per_tco2']:.4f} "
            "EUR/tCO2"
        ),
        (
            "Natural-gas emission factor:    "
            f"{prices['emission_factor_tco2_per_mwh_fuel']:.4f} "
            "tCO2/MWh_fuel"
        ),
        (
            "CO2 cost on natural gas:        "
            f"{prices['co2_cost_on_gas_eur_per_mwh_fuel']:.4f} "
            "EUR/MWh_fuel"
        ),
        (
            "Final CH4_NG marginal cost:     "
            f"{prices['final_ch4_ng_marginal_cost_eur_per_mwh_fuel']:.4f} "
            "EUR/MWh_fuel"
        ),
        (
            "SWFL gas-import adder:          "
            f"{prices['swfl_import_adder_eur_per_mwh_fuel']:.4f} "
            "EUR/MWh_fuel"
        ),
        f"Gas-price source:              {gas_source}",
        f"CO2-price source:              {co2_source}",
        f"Gas-value status:              {gas_value_status}",
        "",
        "BIOGAS.SH COST ASSUMPTIONS",
        "------------------------------------------------------------",
        (
            "Biomethane price case:          "
            f"{selection['biomethane_price_case']}"
        ),
        (
            "Raw biogas cost:                "
            f"{prices['raw_biogas_cost_eur_per_mwh_hs']:.4f} "
            "EUR/MWh_Hs"
        ),
        (
            "Biomethane marginal cost:       "
            f"{prices['biomethane_marginal_cost_eur_per_mwh_hs']:.4f} "
            "EUR/MWh_Hs"
        ),
        (
            "Merchant onsite electricity:    "
            f"{prices['onsite_electricity_marginal_cost_eur_per_mwh']:.4f} "
            "EUR/MWh_el"
        ),
        (
            "Supported onsite electricity:   "
            f"{prices['onsite_supported_electricity_marginal_cost_eur_per_mwh']:.4f} "
            "EUR/MWh_el"
        ),
        (
            "EEG premium assumption:         "
            f"{prices['eeg_premium_eur_per_mwh']:.4f} "
            "EUR/MWh_el"
        ),
        (
            "Onsite heat cost:               "
            f"{prices['onsite_heat_marginal_cost_eur_per_mwh']:.4f} "
            "EUR/MWh_th"
        ),
        "",
        "SWFL TECHNOLOGY CONFIGURATION",
        "------------------------------------------------------------",
        (
            "Heat-pump case:                 "
            f"{selection['heat_pump_case']}"
        ),
        (
            "Active heat pumps:              "
            f"{value_or_none(active_heat_pumps)}"
        ),
        (
            "SWFL unit case:                 "
            f"{selection['swfl_unit_case']}"
        ),
        (
            "Gas-to-power active:            "
            f"{yes_no(units.get('central_gas_to_power', False))}"
        ),
        (
            "Active gas boilers:             "
            f"{value_or_none(active_boilers)}"
        ),
        (
            "Active resistive heaters:       "
            f"{value_or_none(active_resistive_heaters)}"
        ),
        (
            "Reserve gas boiler active:      "
            f"{yes_no(units.get('reserve_gas_boiler', False))}"
        ),
        "",
        "BIOMETHANE USE AT SWFL",
        "------------------------------------------------------------",
        (
            "Biomethane-use case:            "
            f"{selection['biomethane_use_case']}"
        ),
        (
            "Biomethane-use mode:            "
            f"{biomethane_use.get('mode', 'not specified')}"
        ),
        (
            "Biomethane-eligible units:      "
            f"{value_or_none(biomethane_eligible_units)}"
        ),
        "",
        "BIOGAS.SH ROUTES",
        "------------------------------------------------------------",
        (
            "Route case:                     "
            f"{selection['biogas_route_case']}"
        ),
        (
            "Onsite electricity/heat:        "
            f"{yes_no(biogas_routes.get('add_local_generation', False))}"
        ),
        (
            "Storage to public gas grid:     "
            f"{yes_no(biogas_routes.get('add_gas_grid_generation', False))}"
        ),
        (
            "Storage to SWFL:                "
            f"{yes_no(biogas_routes.get('add_swfl_direct_supply', False))}"
        ),
        (
            "Regional resource constraint:   "
            f"{yes_no(technical_biogas['resource_constraint']['active'])}"
        ),
        (
            "Central biomethane storage:     "
            f"{yes_no(technical_biogas['storage']['active'])}"
        ),
    ]

    if effective_run:
        lines.extend(
            [
                "",
                "RUN SETTINGS",
                "------------------------------------------------------------",
                (
                    "Start snapshot:                 "
                    f"{effective_run.get('start_snapshot')}"
                ),
                (
                    "End snapshot:                   "
                    f"{effective_run.get('end_snapshot')}"
                ),
                (
                    "Represented hours:              "
                    f"{effective_run.get('represented_hours')}"
                ),
                (
                    "AC clusters:                    "
                    f"{effective_run.get('ac_clusters')}"
                ),
                (
                    "Result directory:               "
                    f"{effective_run.get('csv_export')}"
                ),
            ]
        )

    lines.extend(
        [
            "",
            "PRICE ACCOUNTING",
            "------------------------------------------------------------",
            (
                "CH4_NG cost = gas commodity price "
                "+ CO2 price × emission factor"
            ),
            (
                "Fuel and CO2 costs are assigned upstream to "
                "CH4_NG generators."
            ),
            (
                "The SWFL boiler and gas-to-power Links therefore "
                "do not repeat these costs."
            ),
            (
                "Biomethane full cost is assigned upstream to the custom "
                "CH4_biogas generators; storage and route links do not "
                "repeat this infrastructure cost."
            ),
            "============================================================",
        ]
    )

    return "\n".join(lines)


def write_resolved_config(
    resolved: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Write the effective scenario beside the results."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(
            dict(resolved),
            handle,
            sort_keys=False,
            allow_unicode=True,
        )
    return output_path


def expand_scenario_matrix(
    config: Mapping[str, Any],
) -> list[Dict[str, str]]:
    """Expand and validate the configured factorial scenario matrix."""
    matrix = config.get("scenario_matrix", {})
    if not isinstance(matrix, Mapping) or not matrix.get("enabled", False):
        return []

    dimensions = matrix.get("dimensions", {})
    if not isinstance(dimensions, Mapping) or not dimensions:
        return []

    keys = list(dimensions)
    values = [list(dimensions[key]) for key in keys]
    base_selection = _selection_with_environment_overrides(config)
    records: list[Dict[str, str]] = []

    for combination in itertools.product(*values):
        selection = dict(base_selection)
        record = dict(zip(keys, map(str, combination)))
        selection.update(record)
        _validate_selection(config, selection)

        record["scenario_name"] = "__".join(
            selection[key]
            for key in (
                    "support_case",
                    "fossil_gas_price_case",
                    "biomethane_price_case",
                    "heat_pump_case",
                    "swfl_unit_case",
                    "biomethane_use_case",
                    "biogas_route_case",
            )
        )
        records.append(record)

    return records


def write_scenario_matrix(
    config: Mapping[str, Any],
    path: str | Path,
) -> Path:
    """Write the expanded factorial matrix as CSV."""
    records = expand_scenario_matrix(config)
    if not records:
        raise ScenarioConfigError(
            "scenario_matrix is disabled or has no dimensions."
        )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["scenario_name"] + [
        key for key in records[0] if key != "scenario_name"
    ]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    return output_path


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect or expand the Biogas.SH / SWFL YAML configuration."
    )
    parser.add_argument(
        "config",
        nargs="?",
        default="config.yaml",
        help="Path to config.yaml",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate", help="Validate the YAML configuration.")
    subparsers.add_parser("show", help="Print the resolved selected scenario.")

    matrix_parser = subparsers.add_parser(
        "matrix",
        help="Expand scenario_matrix into a CSV file.",
    )
    matrix_parser.add_argument(
        "--output",
        default="scenario_matrix.csv",
        help="Output CSV path.",
    )

    cli = parser.parse_args()
    config = load_config(cli.config)

    if cli.command == "validate":
        print(f"PASS: {cli.config} is valid.")
        return

    if cli.command == "show":
        resolved = resolve_config(config)
        print(scenario_summary(resolved))
        return

    if cli.command == "matrix":
        output = write_scenario_matrix(config, cli.output)
        print(
            f"Wrote {len(expand_scenario_matrix(config))} scenarios to {output}"
        )
        return


if __name__ == "__main__":
    _main()