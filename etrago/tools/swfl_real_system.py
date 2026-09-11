"""
Stadtwerke Flensburg real-system replacement module for eTraGo/PyPSA.

What it does
------------
1. Removes generic eGon AC/heat loads and generators/links in the Flensburg/SWFL area.
2. Adds one real SWFL heat load from an hourly Stadtwerke heat time series.
3. Adds one scaled SWFL AC load using the existing eGon Flensburg AC profile shape.
4. Adds one central gas CHP electricity link and one central gas CHP heat link using SWFL capacities.
5. Removes old eGon central heat pumps and optionally adds future SWFL heat pumps.

Recommended call in appl.py
---------------------------
    from etrago.tools.swfl_real_system import apply_swfl_real_system

    etrago.adjust_network()
    apply_swfl_real_system(etrago.network, args.get("swfl_real_system", {}))
    etrago.ehv_clustering()
    etrago.spatial_clustering()
    etrago.spatial_clustering_gas()

Heat-pump flexibility
---------------------
You can add none, one, or both planned heat pumps:

    # none
    "future_heat_pumps": {"active": False, ...}

    # both
    "future_heat_pumps": {"active": True, "units": [{...}, {...}]}

    # only GWP 1, method A
    "future_heat_pumps": {
        "active": True,
        "units": [
            {"name": "swfl_gwp_1", "active": True, ...},
            {"name": "swfl_gwp_2", "active": False, ...},
        ],
    }

    # only GWP 1, method B
    "future_heat_pumps": {
        "active": True,
        "active_units": ["swfl_gwp_1"],
        "units": [{"name": "swfl_gwp_1", ...}, {"name": "swfl_gwp_2", ...}],
    }
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Main public function
# =============================================================================

def resolve_biomethane_units(
    cfg: Dict[str, Any],
    available_boiler_names: Sequence[str],
) -> Set[str]:
    """
    Return the boiler units allowed to use biomethane.

    Modes:
        off
        k12_k13_only
        all_gas_units
        custom
    """

    mode = str(
        cfg.get(
            "biomethane_mode",
            "k12_k13_only",
        )
    ).strip().lower()

    available = {
        str(name)
        for name in available_boiler_names
    }

    if mode == "off":
        selected = set()

    elif mode == "k12_k13_only":
        selected = {
            str(name)
            for name in cfg.get(
                "planned_biomethane_units",
                [
                    "swfl_real_k12",
                    "swfl_real_k13",
                ],
            )
        }

    elif mode == "all_gas_units":
        selected = set(available)

    elif mode == "custom":
        selected = {
            str(name)
            for name in cfg.get(
                "custom_biomethane_units",
                [],
            )
        }

    else:
        raise ValueError(
            "Unsupported central_heat_units.biomethane_mode. "
            "Use 'off', 'k12_k13_only', "
            "'all_gas_units', or 'custom'."
        )

    unknown = selected - available

    if unknown:
        raise ValueError(
            "Unknown biomethane boiler units: "
            + ", ".join(sorted(unknown))
        )

    return selected

def apply_swfl_real_system(
    network,
    settings: Optional[Dict[str, Any]] = None,
):
    """
    Replace the generic eGon representation of the Flensburg/SWFL system
    with project-specific loads and generation technologies.

    The function performs the following steps:

    1. Identify the Flensburg/SWFL replacement area.
    2. Preserve the existing eGon AC-load profile shape.
    3. Remove generic eGon loads, generators, and conversion technologies.
    4. Create the project-specific SWFL AC, heat, and natural-gas buses.
    5. Add the real SWFL heat-load profile.
    6. Add the scaled SWFL AC-load profile.
    7. Optionally add the aggregate SWFL gas-to-power link.
    8. Add detailed SWFL heat-production units:
       K5, K11, K12, K13, EHK1, and EHK2.
    9. Allow the detailed fuel-fired units to receive separately:
       - fossil natural gas,
       - upgraded biomethane,
       - direct raw biogas,
       - optional HEL.
    10. Optionally add the reserve heating plant.
    11. Optionally add future SWFL heat pumps.

    Important
    ---------
    Natural gas, upgraded biomethane, and raw biogas are represented on
    separate buses so their physical and economic flows remain distinguishable.

    When ``central_heat_units.active=True``, the old aggregate SWFL heat
    link in ``central_gas_chp`` must be disabled:

        central_gas_chp.add_heat_link = False

    Otherwise SWFL heat-production capacity would be represented twice.
    """

    settings = settings or {}

    # ==================================================================
    # 0. ACTIVATION
    # ==================================================================

    if not _as_bool(
        settings.get(
            "active",
            False,
        ),
        False,
    ):
        logger.info(
            "SWFL real system inactive; network remains unchanged."
        )
        return network

    _ensure_timeseries_tables(
        network
    )

    snapshots = pd.Index(
        network.snapshots
    )


    # ==================================================================
    # 1. IDENTIFY SWFL AREA
    # ==================================================================

    area_buses = get_swfl_area_buses(
        network=network,
        settings=settings,
    )


    # ==================================================================
    # 2. PRESERVE EXISTING AC LOAD PROFILE SHAPE
    # ==================================================================

    ac_cfg = (
        settings.get(
            "ac_load",
            {},
        )
        or {}
    )

    ac_active = _as_bool(
        ac_cfg.get(
            "active",
            True,
        ),
        True,
    )

    ac_shape = None

    if ac_active:

        ac_shape = build_existing_ac_profile_shape(
            network=network,
            area_buses=area_buses,
            cfg=ac_cfg,
        )


    # ==================================================================
    # 3. REMOVE OLD FLENSBURG / SWFL ASSETS
    # ==================================================================

    if _as_bool(
        settings.get(
            "remove_existing_flensburg_assets",
            True,
        ),
        True,
    ):

        remove_existing_flensburg_assets(
            network=network,
            area_buses=area_buses,
            settings=settings,
        )


    # ==================================================================
    # 4. REMOVE LEGACY HEAT PUMPS
    # ==================================================================

    hp_cfg = (
        settings.get(
            "future_heat_pumps",
            {},
        )
        or {}
    )

    if _as_bool(
        hp_cfg.get(
            "remove_existing_central_heat_pumps",
            True,
        ),
        True,
    ):

        remove_existing_heat_pumps(
            network=network,
            area_buses=area_buses,
            settings=settings,
        )


    # ==================================================================
    # 5. MAIN SWFL BUSES
    # ==================================================================

    swfl_ac_bus = str(
        settings.get(
            "swfl_ac_bus",
            "swfl_ac_bus",
        )
    )

    swfl_heat_bus = str(
        settings.get(
            "swfl_heat_bus",
            "swfl_central_heat_bus",
        )
    )

    swfl_ch4_bus = str(
        settings.get(
            "swfl_ch4_bus",
            "biogas_sh_swfl_ch4_bus",
        )
    )


    x, y = get_swfl_coordinates(
        settings
    )


    ensure_bus(
        network=network,
        name=swfl_ac_bus,
        carrier="AC",
        x=x,
        y=y,
    )

    ensure_bus(
        network=network,
        name=swfl_heat_bus,
        carrier=str(
            settings.get(
                "heat_carrier",
                "central_heat",
            )
        ),
        x=x,
        y=y,
    )

    ensure_bus(
        network=network,
        name=swfl_ch4_bus,
        carrier="CH4",
        x=x,
        y=y,
    )


    # ==================================================================
    # 6. REAL SWFL HEAT LOAD
    # ==================================================================

    heat_cfg = (
        settings.get(
            "heat_load",
            {},
        )
        or {}
    )

    heat_active = _as_bool(
        heat_cfg.get(
            "active",
            True,
        ),
        True,
    )

    heat_profile = None

    if heat_active:

        heat_profile = read_heat_profile_for_snapshots(
            snapshots=snapshots,
            cfg=heat_cfg,
        )

        add_or_replace_load(
            network=network,
            name=str(
                heat_cfg.get(
                    "name",
                    "swfl_real_heat_load",
                )
            ),
            bus=swfl_heat_bus,
            carrier=str(
                heat_cfg.get(
                    "carrier",
                    settings.get(
                        "heat_carrier",
                        "central_heat",
                    ),
                )
            ),
            p_set=heat_profile,
        )


    # ==================================================================
    # 7. REAL / SCALED SWFL ELECTRICITY LOAD
    # ==================================================================

    if ac_active:

        if ac_shape is None:
            raise ValueError(
                "SWFL AC-load configuration is active, "
                "but no existing eGon AC profile shape was created."
            )

        ac_profile = scale_ac_profile_to_target(
            profile=ac_shape,
            network=network,
            cfg=ac_cfg,
            snapshots=snapshots,
        )

        add_or_replace_load(
            network=network,
            name=str(
                ac_cfg.get(
                    "name",
                    "swfl_real_ac_load",
                )
            ),
            bus=swfl_ac_bus,
            carrier=str(
                ac_cfg.get(
                    "carrier",
                    "AC",
                )
            ),
            p_set=ac_profile,
        )


    # ==================================================================
    # 8. CENTRAL GENERATION CONFIGURATION
    # ==================================================================

    chp_cfg = (
        settings.get(
            "central_gas_chp",
            {},
        )
        or {}
    )

    heat_units_cfg = (
        settings.get(
            "central_heat_units",
            {},
        )
        or {}
    )


    chp_active = _as_bool(
        chp_cfg.get(
            "active",
            True,
        ),
        True,
    )

    detailed_heat_active = _as_bool(
        heat_units_cfg.get(
            "active",
            False,
        ),
        False,
    )

    aggregate_heat_active = (
        chp_active
        and _as_bool(
            chp_cfg.get(
                "add_heat_link",
                True,
            ),
            True,
        )
    )


    if (
        detailed_heat_active
        and aggregate_heat_active
    ):

        raise ValueError(
            "Both the aggregate SWFL heat link and the detailed "
            "central heat units are active. This would double-count "
            "SWFL heat capacity. Set "
            "args['swfl_real_system']['central_gas_chp']"
            "['add_heat_link'] = False."
        )


    # ==================================================================
    # 9. SWFL GAS-TO-POWER
    # ==================================================================

    if chp_active:

        add_central_gas_chp_links(
            network=network,
            cfg=chp_cfg,
            gas_bus=str(
                chp_cfg.get(
                    "gas_bus",
                    swfl_ch4_bus,
                )
            ),
            ac_bus=str(
                chp_cfg.get(
                    "ac_bus",
                    swfl_ac_bus,
                )
            ),
            heat_bus=str(
                chp_cfg.get(
                    "heat_bus",
                    swfl_heat_bus,
                )
            ),
        )


    # ==================================================================
    # 10. DETAILED SWFL CENTRAL HEAT SYSTEM
    # ==================================================================

    if detailed_heat_active:

        natural_gas_bus = str(
            heat_units_cfg.get(
                "natural_gas_bus",
                swfl_ch4_bus,
            )
        )

        biomethane_bus = str(
            heat_units_cfg.get(
                "biomethane_bus",
                "swfl_real_biomethane_ch4_bus",
            )
        )

        raw_biogas_bus = str(
            heat_units_cfg.get(
                "raw_biogas_bus",
                "swfl_real_raw_biogas_bus",
            )
        )


        # --------------------------------------------------------------
        # Natural gas is the standard fuel and is always required by
        # the detailed SWFL gas boilers.
        # --------------------------------------------------------------

        ensure_bus(
            network=network,
            name=natural_gas_bus,
            carrier="CH4",
            x=x,
            y=y,
        )


        # --------------------------------------------------------------
        # IMPORTANT:
            #
            # Do NOT create the biomethane or raw-biogas buses here
            # unconditionally.
            #
            # add_central_heat_units() knows which fuels are actually
            # enabled and creates only the buses that are needed.
            #
            # This prevents isolated CH4 buses from entering gas clustering.
        # --------------------------------------------------------------

        add_central_heat_units(
            network=network,
            cfg=heat_units_cfg,
            natural_gas_bus=natural_gas_bus,
            biomethane_bus=biomethane_bus,
            raw_biogas_bus=raw_biogas_bus,
            ac_bus=str(
                heat_units_cfg.get(
                    "ac_bus",
                    swfl_ac_bus,
                )
            ),
            heat_bus=str(
                heat_units_cfg.get(
                    "heat_bus",
                    swfl_heat_bus,
                )
            ),
            x=x,
            y=y,
        )


    # ==================================================================
    # 11. OPTIONAL RESERVE GAS BOILER
    # ==================================================================

    reserve_cfg = (
        settings.get(
            "reserve_gas_boiler",
            {},
        )
        or {}
    )

    if _as_bool(
        reserve_cfg.get(
            "active",
            False,
        ),
        False,
    ):

        reserve_gas_bus = str(
            reserve_cfg.get(
                "gas_bus",
                swfl_ch4_bus,
            )
        )

        ensure_bus(
            network=network,
            name=reserve_gas_bus,
            carrier="CH4",
            x=x,
            y=y,
        )

        add_reserve_gas_boiler(
            network=network,
            cfg=reserve_cfg,
            gas_bus=reserve_gas_bus,
            heat_bus=str(
                reserve_cfg.get(
                    "heat_bus",
                    swfl_heat_bus,
                )
            ),
        )


    # ==================================================================
    # 12. FUTURE LARGE SWFL HEAT PUMPS
    # ==================================================================

    if _as_bool(
        hp_cfg.get(
            "active",
            False,
        ),
        False,
    ):

        add_future_heat_pumps(
            network=network,
            cfg=hp_cfg,
            ac_bus=swfl_ac_bus,
            heat_bus=swfl_heat_bus,
        )


    # ==================================================================
    # 13. SUMMARY
    # ==================================================================

    print_swfl_real_system_summary(
        network=network,
        settings=settings,
        area_buses=area_buses,
        heat_profile=heat_profile,
        ac_shape=ac_shape,
    )

    return network


# =============================================================================
# Area selection
# =============================================================================


def get_swfl_coordinates(settings: Dict[str, Any]) -> Tuple[float, float]:
    """Default coordinates close to Stadtwerke Flensburg / Flensburg."""
    return (
        float(settings.get("swfl_ch4_bus_x", 9.436502119171873)),
        float(settings.get("swfl_ch4_bus_y", 54.79233181101448)),
    )


def get_swfl_area_buses(network, settings: Dict[str, Any]) -> Set[str]:
    """
    Identify buses in the Flensburg/SWFL replacement area.

    Preferred mode for this project:
        area_mode = "ding0_mv_grid_districts"

    This selects eTraGo buses located inside selected DING0 MV grid districts.
    For the current SWFL case:

        selected_mv_grid_district_ids = ["33935", "33543", "35906"]

    No radius-based selection is used.
    """
    area_mode = str(settings.get("area_mode", "ding0_mv_grid_districts"))

    if area_mode == "explicit_buses":
        explicit = settings.get("area_buses", [])
        existing = set(network.buses.index.astype(str))
        return {str(b) for b in explicit if str(b) in existing}

    if area_mode == "ding0_mv_grid_districts":
        return get_swfl_area_buses_from_ding0_mv_grid_districts(network, settings)

    raise ValueError(
        f"Unsupported swfl_real_system area_mode={area_mode!r}. "
        "Use 'ding0_mv_grid_districts' or 'explicit_buses'."
    )


def get_swfl_area_buses_from_ding0_mv_grid_districts(
    network,
    settings: Dict[str, Any],
) -> Set[str]:
    """
    Select network buses located inside selected DING0 MV grid districts.

    Required settings:
        mv_grid_districts_gpkg
        selected_mv_grid_district_ids

    Optional settings:
        mv_grid_layer
        mv_grid_id_column
        bus_crs
        district_crs
        expand_area_through_local_links
    """
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError(
            "geopandas is required for "
            "swfl_real_system.area_mode='ding0_mv_grid_districts'."
        ) from exc

    gpkg = settings.get("mv_grid_districts_gpkg")
    if not gpkg:
        raise ValueError(
            "swfl_real_system.area_mode='ding0_mv_grid_districts' requires "
            "'mv_grid_districts_gpkg'."
        )

    selected_ids = [
        str(x) for x in settings.get("selected_mv_grid_district_ids", [])
    ]
    if not selected_ids:
        raise ValueError(
            "swfl_real_system.area_mode='ding0_mv_grid_districts' requires "
            "'selected_mv_grid_district_ids'."
        )

    layer = settings.get("mv_grid_layer", None)
    id_col = str(settings.get("mv_grid_id_column", "name"))

    if layer is None:
        districts = gpd.read_file(gpkg)
    else:
        districts = gpd.read_file(gpkg, layer=layer)

    if id_col not in districts.columns:
        raise KeyError(
            f"Column {id_col!r} not found in DING0 MV grid district file. "
            f"Available columns: {list(districts.columns)}"
        )

    districts = districts.copy()
    districts[id_col] = districts[id_col].astype(str)

    selected_districts = districts[districts[id_col].isin(selected_ids)].copy()

    if selected_districts.empty:
        raise ValueError(
            f"No DING0 MV grid districts found for selected IDs {selected_ids} "
            f"using column {id_col!r}."
        )

    buses = network.buses.copy()

    if "x" not in buses.columns or "y" not in buses.columns:
        raise ValueError(
            "network.buses must contain x/y coordinates for DING0 spatial selection."
        )

    buses["x"] = pd.to_numeric(buses["x"], errors="coerce")
    buses["y"] = pd.to_numeric(buses["y"], errors="coerce")
    buses = buses.dropna(subset=["x", "y"])

    bus_gdf = gpd.GeoDataFrame(
        buses,
        geometry=gpd.points_from_xy(buses["x"], buses["y"]),
        crs=settings.get("bus_crs", "EPSG:4326"),
    )

    if selected_districts.crs is None:
        selected_districts = selected_districts.set_crs(
            settings.get("district_crs", "EPSG:4326")
        )

    if bus_gdf.crs != selected_districts.crs:
        bus_gdf = bus_gdf.to_crs(selected_districts.crs)

    joined = gpd.sjoin(
        bus_gdf,
        selected_districts[[id_col, "geometry"]],
        how="inner",
        predicate="within",
    )

    area_buses = set(joined.index.astype(str))

    # Also include the district IDs themselves if they are actual eTraGo buses.
    # In this case, 33935, 33543, 35906 are relevant AC buses.
    network_bus_ids = set(network.buses.index.astype(str))
    area_buses |= {x for x in selected_ids if x in network_bus_ids}

    # Expand from selected AC buses to directly connected local heat/CHP/boiler buses.
    # This catches the central/rural heat buses attached to these districts.
    if _as_bool(settings.get("expand_area_through_local_links", True), True):
        area_buses = expand_area_buses_through_local_links(
            network=network,
            area_buses=area_buses,
            settings=settings,
        )

    print("\nSWFL area selection using DING0 MV grid districts")
    print(f"  selected MV districts: {selected_ids}")
    print(f"  selected area buses:   {len(area_buses)}")

    return area_buses


def expand_area_buses_through_local_links(
    network,
    area_buses: Set[str],
    settings: Dict[str, Any],
) -> Set[str]:
    """
    Expand selected area buses through local heat/CHP/boiler links.

    This is needed because the selected DING0 MV district IDs are usually AC
    buses, while relevant heat buses may be connected through local conversion
    links such as central_gas_CHP_heat, central_gas_boiler, heat_pump, etc.
    """
    patterns = settings.get(
        "area_expansion_link_carrier_patterns",
        [
            "central_gas",
            "central_heat",
            "rural_heat",
            "heat_pump",
            "CHP",
            "boiler",
        ],
    )

    links = network.links.copy()
    if links.empty:
        return area_buses

    bus_cols = link_bus_columns(links)

    carrier_mask = pd.Series(False, index=links.index)
    if "carrier" in links.columns:
        carrier = links["carrier"].astype(str)
        for pat in patterns:
            carrier_mask |= carrier.str.contains(str(pat), case=False, na=False)
    else:
        carrier_mask[:] = True

    connected_mask = pd.Series(False, index=links.index)
    for col in bus_cols:
        if col in links.columns:
            connected_mask |= links[col].astype(str).isin(area_buses)

    local_links = links[carrier_mask & connected_mask]

    expanded = set(area_buses)
    for col in bus_cols:
        if col in local_links.columns:
            expanded |= set(local_links[col].astype(str))

    return expanded


# =============================================================================
# Removing old eGon assets
# =============================================================================


def remove_existing_flensburg_assets(network, area_buses: Set[str], settings: Dict[str, Any]) -> None:
    """Remove existing eGon loads, generators, and selected links in SWFL area."""
    protected_prefixes = tuple(settings.get("protected_prefixes", ["biogas_sh_", "swfl_real_", "swfl_gwp_"]))
    keep_components = {str(x) for x in settings.get("keep_components", [])}

    # Loads connected to area buses.
    load_ids = component_indices_connected_to_buses(
        network.loads,
        bus_columns=["bus"],
        area_buses=area_buses,
        protected_prefixes=protected_prefixes,
        keep_components=keep_components,
    )
    remove_components(network, "Load", load_ids)

    # Generators connected to area buses.
    gen_ids = component_indices_connected_to_buses(
        network.generators,
        bus_columns=["bus"],
        area_buses=area_buses,
        protected_prefixes=protected_prefixes,
        keep_components=keep_components,
    )
    remove_components(network, "Generator", gen_ids)

    # Conversion links connected to area buses. Carrier filter prevents deleting
    # unrelated transmission links just because one endpoint is in Flensburg.
    patterns = settings.get(
        "remove_link_carrier_patterns",
        ["central_gas", "central_heat", "rural_heat", "heat_pump", "CHP", "boiler", "central_resistive_heater"],
    )
    link_ids = component_indices_connected_to_buses(
        network.links,
        bus_columns=link_bus_columns(network.links),
        area_buses=area_buses,
        protected_prefixes=protected_prefixes,
        keep_components=keep_components,
        carrier_patterns=patterns,
    )
    remove_components(network, "Link", link_ids)


def remove_existing_heat_pumps(
    network,
    area_buses: Set[str],
    settings: Dict[str, Any],
    remove_at_planned_connections: bool = False,
) -> None:
    """
    Remove legacy eGon heat-pump Links relevant to the SWFL system.

    Two modes are supported:

    1. Before clustering:
       Remove generic heat pumps connected to the selected SWFL area buses.

    2. After spatial clustering:
       Remove generic heat pumps that share the exact bus0/bus1 connection
       of an active planned SWFL heat pump.

    The second mode avoids removing heat pumps elsewhere in Germany.
    """

    protected_prefixes = tuple(
        settings.get(
            "protected_prefixes",
            [
                "biogas_sh_",
                "swfl_real_",
                "swfl_gwp_",
            ],
        )
    )

    keep_components = {
        str(value)
        for value in settings.get(
            "keep_components",
            [],
        )
    }

    patterns = settings.get(
        "remove_heat_pump_carrier_patterns",
        [
            "central_heat_pump",
            "rural_heat_pump",
            "heat_pump",
        ],
    )

    links = network.links

    ids_to_remove: Set[str] = set()

    # --------------------------------------------------------------
    # A. Existing behaviour: remove heat pumps connected to SWFL area
    # --------------------------------------------------------------
    if area_buses:
        area_ids = component_indices_connected_to_buses(
            links,
            bus_columns=link_bus_columns(links),
            area_buses=area_buses,
            protected_prefixes=protected_prefixes,
            keep_components=keep_components,
            carrier_patterns=patterns,
        )

        ids_to_remove.update(area_ids)

    # --------------------------------------------------------------
    # B. Post-clustering cleanup
    # --------------------------------------------------------------
    if remove_at_planned_connections:
        hp_cfg = settings.get(
            "future_heat_pumps",
            {},
        ) or {}

        if not _as_bool(
            hp_cfg.get("active", False),
            False,
        ):
            print(
                "\nLegacy heat-pump cleanup skipped: "
                "planned SWFL heat pumps are inactive."
            )

        else:
            units = hp_cfg.get(
                "units",
                [],
            ) or []

            configured_active_units = hp_cfg.get(
                "active_units",
                None,
            )

            if configured_active_units is not None:
                active_names = {
                    str(name)
                    for name in configured_active_units
                }
            else:
                active_names = {
                    str(
                        unit.get(
                            "name",
                            "",
                        )
                    ).strip()
                    for unit in units
                    if _as_bool(
                        unit.get(
                            "active",
                            True,
                        ),
                        True,
                    )
                }

            active_names.discard("")

            planned_names = set()
            planned_carriers = set()

            for unit in units:
                name = str(
                    unit.get(
                        "name",
                        "",
                    )
                ).strip()

                if not name or name not in active_names:
                    continue

                carrier = str(
                    unit.get("carrier")
                    or f"{name}_heat_pump"
                )

                planned_names.add(name)
                planned_carriers.add(carrier)

            link_names = links.index.astype(str)
            link_carriers = links[
                "carrier"
            ].astype(str)

            # Link names usually exist before clustering.
            # Unique carriers remain usable after numeric renaming.
            planned_mask = (
                link_names.isin(planned_names)
                | link_carriers.isin(
                    planned_carriers
                )
            )

            planned_links = links[
                planned_mask
            ]

            if planned_links.empty:
                raise ValueError(
                    "No planned SWFL heat-pump Links were found "
                    "during post-clustering legacy cleanup. "
                    f"Expected carriers: "
                    f"{sorted(planned_carriers)}"
                )

            if (
                "bus0" not in planned_links.columns
                or "bus1" not in planned_links.columns
            ):
                raise ValueError(
                    "network.links must contain bus0 and bus1 "
                    "for heat-pump cleanup."
                )

            planned_connections = {
                (
                    str(row["bus0"]),
                    str(row["bus1"]),
                )
                for _, row in planned_links.iterrows()
            }

            generic_carrier_mask = pd.Series(
                False,
                index=links.index,
            )

            for pattern in patterns:
                generic_carrier_mask |= (
                    link_carriers.str.contains(
                        str(pattern),
                        case=False,
                        regex=False,
                        na=False,
                    )
                )

            same_connection_mask = pd.Series(
                [
                    (
                        str(row["bus0"]),
                        str(row["bus1"]),
                    )
                    in planned_connections
                    for _, row in links.iterrows()
                ],
                index=links.index,
            )

            protected_mask = pd.Series(
                False,
                index=links.index,
            )

            for prefix in protected_prefixes:
                protected_mask |= (
                    link_names.str.startswith(
                        prefix
                    )
                )

            if keep_components:
                protected_mask |= (
                    link_names.isin(
                        keep_components
                    )
                )

            # Planned Links may have numeric IDs after clustering,
            # so protect them explicitly through their carriers.
            protected_mask |= planned_mask

            legacy_mask = (
                generic_carrier_mask
                & same_connection_mask
                & ~protected_mask
            )

            legacy_ids = links.index[
                legacy_mask
            ].tolist()

            if legacy_ids:
                diagnostic_columns = [
                    column
                    for column in [
                        "bus0",
                        "bus1",
                        "carrier",
                        "p_nom",
                        "p_nom_opt",
                        "efficiency",
                    ]
                    if column in links.columns
                ]

                print(
                    "\nRemoving legacy heat pumps at planned "
                    "SWFL heat-pump connections:"
                )

                print(
                    links.loc[
                        legacy_ids,
                        diagnostic_columns,
                    ].to_string()
                )

                ids_to_remove.update(
                    legacy_ids
                )

            else:
                print(
                    "\nNo legacy heat pump found at planned "
                    "SWFL heat-pump connections."
                )

    remove_components(
        network,
        "Link",
        list(ids_to_remove),
    )


def component_indices_connected_to_buses(
    df: pd.DataFrame,
    bus_columns: Sequence[str],
    area_buses: Set[str],
    protected_prefixes: Tuple[str, ...],
    keep_components: Set[str],
    carrier_patterns: Optional[Sequence[str]] = None,
) -> List[str]:
    if df is None or len(df) == 0:
        return []

    idx = df.index.astype(str)
    protected = pd.Series(False, index=df.index)
    for prefix in protected_prefixes:
        protected |= idx.str.startswith(prefix)
    if keep_components:
        protected |= idx.isin(keep_components)

    connected = pd.Series(False, index=df.index)
    for col in bus_columns:
        if col in df.columns:
            connected |= df[col].astype(str).isin(area_buses)

    if carrier_patterns is not None and "carrier" in df.columns:
        carrier = df["carrier"].astype(str)
        carrier_mask = pd.Series(False, index=df.index)
        for pat in carrier_patterns:
            carrier_mask |= carrier.str.contains(str(pat), case=False, na=False)
        connected &= carrier_mask

    return df.index[connected & ~protected].astype(str).tolist()


def link_bus_columns(links: pd.DataFrame) -> List[str]:
    return [c for c in links.columns if c.startswith("bus")]


def remove_components(
    network,
    component: str,
    names: Sequence[Any],
) -> None:
    """
    Remove PyPSA components safely.

    Requested names are matched through their string representation, but the
    original index values are passed to PyPSA. This supports both string IDs
    and numeric IDs created by clustering.
    """
    requested_names = [
        name
        for name in names
        if str(name)
    ]

    if not requested_names:
        return

    component_table = {
        "Bus": "buses",
        "Load": "loads",
        "Generator": "generators",
        "Link": "links",
        "Store": "stores",
        "StorageUnit": "storage_units",
        "Line": "lines",
        "Transformer": "transformers",
    }

    table_name = component_table.get(component)

    if table_name is None or not hasattr(
        network,
        table_name,
    ):
        logger.warning(
            "Unknown PyPSA component type: %s",
            component,
        )
        return

    table = getattr(
        network,
        table_name,
    )

    # Map string representations back to the actual index values.
    index_lookup = {
        str(index_value): index_value
        for index_value in table.index
    }

    existing_names = []
    missing_names = []

    for requested_name in requested_names:
        key = str(requested_name)

        if key in index_lookup:
            existing_names.append(
                index_lookup[key]
            )
        else:
            missing_names.append(key)

    # Remove duplicates while preserving original index types.
    existing_names = list(
        dict.fromkeys(existing_names)
    )

    if missing_names:
        logger.debug(
            "Skipping %d missing %s components: %s",
            len(missing_names),
            component,
            missing_names[:10],
        )

    if not existing_names:
        return

    logger.info(
        "Removing %d %s components: %s",
        len(existing_names),
        component,
        [
            str(name)
            for name in existing_names[:10]
        ],
    )

    if hasattr(network, "mremove"):
        network.mremove(
            component,
            existing_names,
        )
    else:
        for name in existing_names:
            try:
                network.remove(
                    component,
                    name,
                )
            except Exception as exc:
                logger.warning(
                    "Could not remove %s %s: %s",
                    component,
                    name,
                    exc,
                )
# =============================================================================
# Load profiles
# =============================================================================


def build_existing_ac_profile_shape(network, area_buses: Set[str], cfg: Dict[str, Any]) -> pd.Series:
    """Use current eGon Flensburg AC loads as temporal profile shape."""
    source_ids = cfg.get("source_load_ids")
    if source_ids:
        load_ids = [str(i) for i in source_ids if str(i) in network.loads.index.astype(str)]
    else:
        carrier = str(cfg.get("source_carrier", cfg.get("carrier", "AC")))
        mask = network.loads["bus"].astype(str).isin(area_buses)
        if "carrier" in network.loads.columns:
            mask &= network.loads["carrier"].astype(str).str.contains(carrier, case=False, na=False)
        load_ids = network.loads.index[mask].astype(str).tolist()

    if not load_ids:
        raise ValueError(
            "No existing Flensburg AC loads found. Adjust area selection or set "
            "swfl_real_system.ac_load.source_load_ids."
        )

    p_set = get_load_p_set(network, load_ids)
    profile = p_set.sum(axis=1).astype(float).reindex(network.snapshots).fillna(0.0)
    if profile.sum() <= 0:
        raise ValueError("Selected eGon AC profile has zero sum.")
    return profile


def get_load_p_set(network, load_ids: Sequence[str]) -> pd.DataFrame:
    snapshots = pd.Index(network.snapshots)
    out = pd.DataFrame(index=snapshots)

    # Time-varying load profiles.
    if hasattr(network, "loads_t") and hasattr(network.loads_t, "p_set"):
        pset = network.loads_t.p_set
        if isinstance(pset, pd.DataFrame):
            for lid in load_ids:
                if lid in pset.columns:
                    out[lid] = pd.to_numeric(pset[lid], errors="coerce").reindex(snapshots)

    # Static p_set fallback.
    for lid in load_ids:
        if lid not in out.columns and lid in network.loads.index:
            val = network.loads.at[lid, "p_set"] if "p_set" in network.loads.columns else 0.0
            val = pd.to_numeric(pd.Series([val]), errors="coerce").fillna(0.0).iloc[0]
            out[lid] = float(val)

    return out.fillna(0.0)


def scale_ac_profile_to_target(profile, network, cfg, snapshots=None):
    """
    Scale an AC load profile to the target electricity demand.

    The function is called as:

        scale_ac_profile_to_target(ac_shape, network, ac_cfg, snapshots)

    For full-year runs:
        target_annual_demand_mwh is used directly.

    For short test runs:
        scale_annual_target_to_snapshot_hours=True prevents the full annual
        demand from being compressed into only a few snapshots.
    """
    profile = profile.copy()

    if profile.empty:
        raise ValueError("Cannot scale empty SWFL AC profile.")

    profile = profile.astype(float)

    if _as_bool(cfg.get("clip_negative", True), True):
        profile = profile.clip(lower=0.0)

    target = cfg.get("target_annual_demand_mwh", None)

    if target is None:
        raise ValueError(
            "ac_load.target_annual_demand_mwh is required for SWFL AC scaling."
        )

    target = float(target)

    if snapshots is None:
        snapshots = profile.index

    # Align profile to the selected snapshots
    profile = profile.reindex(snapshots)

    if profile.isna().any():
        profile = profile.interpolate(method="time").ffill().bfill()

    weights = snapshot_weights(network, profile.index)
    weights = weights.reindex(profile.index).fillna(1.0)

    represented_hours = float(weights.sum())

    if _as_bool(cfg.get("target_is_annual", True), True):
        if _as_bool(cfg.get("scale_annual_target_to_snapshot_hours", True), True):
            if represented_hours > 0 and represented_hours < 8760.0:
                original_target = target
                target = target * represented_hours / 8760.0

                print(
                    "\nSWFL AC annual-demand scaling for short run"
                    f"\n  original annual target: {original_target:.3f} MWh/a"
                    f"\n  represented hours:      {represented_hours:.3f} h"
                    f"\n  scaled target:          {target:.3f} MWh"
                )

    current = float((profile * weights).sum())

    if current <= 0:
        raise ValueError(
            "Existing eGon AC profile shape has zero weighted energy. "
            "Cannot scale SWFL AC load."
        )

    factor = target / current
    scaled = profile * factor

    print(
        "\nSWFL AC load scaling"
        f"\n  current weighted energy: {current:.3f} MWh"
        f"\n  target weighted energy:  {target:.3f} MWh"
        f"\n  scaling factor:          {factor:.6f}"
        f"\n  mean load:               {scaled.mean():.3f} MW"
        f"\n  peak load:               {scaled.max():.3f} MW"
    )

    return scaled


def read_heat_profile_for_snapshots(snapshots: pd.Index, cfg: Dict[str, Any]) -> pd.Series:
    """Read Stadtwerke hourly heat profile for one year and map to network snapshots."""
    year = int(cfg.get("year"))
    col_name = str(cfg.get("column", "HKW Wärmeleistung Gesamt"))
    unit = str(cfg.get("unit", "MW")).lower()

    if cfg.get("csv_path"):
        df = read_table_flexible(Path(cfg["csv_path"]))
    elif cfg.get("xlsx_path"):
        df = pd.read_excel(cfg["xlsx_path"], sheet_name=cfg.get("sheet_name", 0))
    else:
        raise ValueError("heat_load requires csv_path or xlsx_path.")

    df = normalise_columns(df)
    heat_col = find_column(df, col_name)
    dt = extract_datetime_index(df, cfg)

    values = pd.to_numeric(df[heat_col].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    s = pd.Series(values.values, index=dt, name="swfl_heat_mw").sort_index()
    s = s[~s.index.duplicated(keep="first")]
    s = s[s.index.year == year]

    if s.empty:
        raise ValueError(f"No heat data found for year {year}.")

    if unit in {"kw", "kwh/h"}:
        s = s / 1000.0
    elif unit in {"mw", "mwh/h"}:
        pass
    else:
        raise ValueError(f"Unsupported heat unit {unit!r}; use MW or kW.")

    s8760 = make_8760_hourly_year(s, year, fill_method=cfg.get("fill_method", "time_interpolate"))
    mapped = map_year_profile_to_network_snapshots(s8760, snapshots)
    if _as_bool(cfg.get("clip_negative", True), True):
        mapped = mapped.clip(lower=0.0)
    return mapped.astype(float)


# =============================================================================
# Adding components
# =============================================================================


def ensure_bus(network, name: str, carrier: str, x: float, y: float) -> None:
    name = str(name)
    if name in network.buses.index.astype(str):
        if "carrier" in network.buses.columns:
            network.buses.loc[name, "carrier"] = carrier
        if "x" in network.buses.columns:
            network.buses.loc[name, "x"] = x
        if "y" in network.buses.columns:
            network.buses.loc[name, "y"] = y
        return
    network.add("Bus", name, carrier=carrier, x=x, y=y)


def ensure_carrier(network, name: str) -> None:
    """Add a PyPSA carrier if it does not already exist."""
    name = str(name)

    existing = set(network.carriers.index.astype(str))

    if name not in existing:
        network.add("Carrier", name)


def add_or_replace_load(network, name: str, bus: str, carrier: str, p_set: pd.Series) -> None:
    remove_components(network, "Load", [name])
    network.add("Load", name, bus=bus, carrier=carrier)
    _ensure_timeseries_tables(network)
    network.loads_t.p_set[name] = p_set.reindex(network.snapshots).astype(float).fillna(0.0)


def add_central_gas_chp_links(
    network,
    cfg: Dict[str, Any],
    gas_bus: str,
    ac_bus: str,
    heat_bus: str,
) -> None:
    """
    Add configurable aggregate SWFL gas-to-power and gas-to-heat links.

    The electrical link can be retained while the aggregate heat link is
    disabled and replaced by the detailed boiler/EHK representation.

    This is still not a physically coupled CHP representation.
    """

    add_electric_link = _as_bool(
        cfg.get("add_electric_link", True),
        True,
    )

    add_heat_link = _as_bool(
        cfg.get("add_heat_link", True),
        True,
    )

    p_nom_is_output = _as_bool(
        cfg.get("p_nom_is_output_capacity", True),
        True,
    )

    el_name = str(
        cfg.get(
            "electric_link_name",
            "swfl_real_central_gas_CHP",
        )
    )

    heat_name = str(
        cfg.get(
            "heat_link_name",
            "swfl_real_central_gas_CHP_heat",
        )
    )

    # Always remove previous versions before rebuilding.
    remove_components(
        network,
        "Link",
        [el_name, heat_name],
    )

    # ------------------------------------------------------------------
    # Aggregated electricity-generation link
    # ------------------------------------------------------------------
    if add_electric_link:
        el_cap = float(
            cfg.get(
                "electric_capacity_mw",
                241.0,
            )
        )

        el_eff = float(
            cfg.get(
                "electric_efficiency",
                0.40,
            )
        )

        if el_eff <= 0:
            raise ValueError(
                "central_gas_chp.electric_efficiency must be greater than zero."
            )

        # PyPSA Link p_nom is on the bus0/input side.
        el_p_nom = (
            el_cap / el_eff
            if p_nom_is_output
            else el_cap
        )

        el_carrier = str(
            cfg.get(
                "carrier_el",
                "swfl_real_gas_to_power",
            )
        )

        ensure_carrier(network, el_carrier)

        network.add(
            "Link",
            el_name,
            bus0=gas_bus,
            bus1=ac_bus,
            carrier=el_carrier,
            p_nom=el_p_nom,
            p_nom_extendable=_as_bool(
                cfg.get("extendable", False),
                False,
            ),
            p_min_pu=float(
                cfg.get("p_min_pu", 0.0)
            ),
            p_max_pu=float(
                cfg.get("p_max_pu", 1.0)
            ),
            efficiency=el_eff,
            marginal_cost=float(
                cfg.get(
                    "electric_marginal_cost",
                    cfg.get("marginal_cost", 0.0),
                )
            ),
            capital_cost=float(
                cfg.get(
                    "electric_capital_cost",
                    cfg.get("capital_cost", 0.0),
                )
            ),
        )

    # ------------------------------------------------------------------
    # Optional aggregate heat link
    # ------------------------------------------------------------------
    # For the new detailed SWFL unit representation this must be False.
    if add_heat_link:
        heat_cap = float(
            cfg.get(
                "heat_capacity_mw",
                370.0,
            )
        )

        heat_eff = float(
            cfg.get(
                "heat_efficiency",
                0.90,
            )
        )

        if heat_eff <= 0:
            raise ValueError(
                "central_gas_chp.heat_efficiency must be greater than zero."
            )

        heat_p_nom = (
            heat_cap / heat_eff
            if p_nom_is_output
            else heat_cap
        )

        heat_carrier = str(
            cfg.get(
                "carrier_heat",
                "swfl_real_gas_to_heat",
            )
        )

        ensure_carrier(network, heat_carrier)

        network.add(
            "Link",
            heat_name,
            bus0=gas_bus,
            bus1=heat_bus,
            carrier=heat_carrier,
            p_nom=heat_p_nom,
            p_nom_extendable=_as_bool(
                cfg.get("extendable", False),
                False,
            ),
            p_min_pu=float(
                cfg.get("p_min_pu", 0.0)
            ),
            p_max_pu=float(
                cfg.get("p_max_pu", 1.0)
            ),
            efficiency=heat_eff,
            marginal_cost=float(
                cfg.get(
                    "heat_marginal_cost",
                    cfg.get("marginal_cost", 0.0),
                )
            ),
            capital_cost=float(
                cfg.get(
                    "heat_capital_cost",
                    cfg.get("capital_cost", 0.0),
                )
            ),
        )

def add_central_heat_units(
    network,
    cfg: Dict[str, Any],
    natural_gas_bus: str,
    biomethane_bus: str,
    raw_biogas_bus: str,
    ac_bus: str,
    heat_bus: str,
    x: float,
    y: float,
) -> None:
    """
    Add detailed SWFL central heat-generation units.

    Fuel-fired units
    ----------------
        K5   60 MWth
        K11  70 MWth
        K12  80 MWth
        K13  90 MWth

    Electric units
    --------------
        EHK1 30 MWth
        EHK2 40 MWth

    Fuel topology
    -------------
    Each fuel-fired unit receives one internal fuel bus and one heat
    conversion Link.

    Separate supply Links may feed that common fuel bus from:

        natural gas
        upgraded biomethane
        raw biogas
        optional HEL

    Example for K12:

        natural gas --------\
                             \
        biomethane ----------- > K12 fuel bus -> K12 boiler -> heat
                             /
        raw biogas ----------/

    This architecture guarantees that all fuels compete for the SAME
    physical boiler capacity.

    Fuel prices are represented upstream at the corresponding source.
    The supply links in this function therefore contain only optional
    route-specific transport costs.
    """

    # ==================================================================
    # 1. REQUIRED SYSTEM BUSES
    # ==================================================================

    existing_buses = set(
        network.buses.index.astype(str)
    )


    if natural_gas_bus not in existing_buses:

        raise ValueError(
            f"Natural-gas bus {natural_gas_bus!r} "
            "does not exist in network.buses."
        )


    if heat_bus not in existing_buses:

        raise ValueError(
            f"SWFL heat bus {heat_bus!r} "
            "does not exist in network.buses."
        )


    if ac_bus not in existing_buses:

        raise ValueError(
            f"SWFL AC bus {ac_bus!r} "
            "does not exist in network.buses."
        )


    # ==================================================================
    # 2. UNIT CONFIGURATION
    # ==================================================================

    boilers = (
        cfg.get(
            "boilers",
            [],
        )
        or []
    )

    resistive_heaters = (
        cfg.get(
            "resistive_heaters",
            [],
        )
        or []
    )


    boiler_names = [
        str(
            unit.get(
                "name",
                "",
            )
        ).strip()
        for unit in boilers
        if str(
            unit.get(
                "name",
                "",
            )
        ).strip()
    ]


    available_boilers = set(
        boiler_names
    )


    # ==================================================================
    # 3. BIOMETHANE ELIGIBILITY
    # ==================================================================

    biomethane_units = resolve_biomethane_units(
        cfg,
        boiler_names,
    )


    # ==================================================================
    # 4. RAW-BIOGAS ELIGIBILITY
    # ==================================================================

    raw_biogas_units = {
        str(name).strip()
        for name in cfg.get(
            "raw_biogas_units",
            [],
        )
        if str(name).strip()
    }


    unknown_raw_biogas_units = (
        raw_biogas_units
        - available_boilers
    )


    if unknown_raw_biogas_units:

        raise ValueError(
            "Unknown raw-biogas boiler units: "
            + ", ".join(
                sorted(
                    unknown_raw_biogas_units
                )
            )
        )


    # ==================================================================
    # 6. ENSURE DEDICATED FUEL BUSES
    # ==================================================================

    ensure_carrier(
        network,
        "CH4",
    )


    if biomethane_units:

        if biomethane_bus == natural_gas_bus:

            raise ValueError(
                "Natural-gas and biomethane buses must be different."
            )

        ensure_bus(
            network=network,
            name=biomethane_bus,
            carrier="CH4",
            x=x,
            y=y,
        )


    # ------------------------------------------------------------------
    # Raw-biogas bus
    #
    # Raw biogas uses its own carrier and therefore stays outside the
    # normal CH4 gas-clustering topology.
    # ------------------------------------------------------------------

    if raw_biogas_units:

        if raw_biogas_bus == natural_gas_bus:

            raise ValueError(
                "Raw-biogas and natural-gas buses must be different."
            )

        if (
                biomethane_units
                and raw_biogas_bus == biomethane_bus
        ):

                raise ValueError(
                    "Raw-biogas and biomethane buses must be different."
                )

        ensure_carrier(
            network,
            "raw_biogas",
        )

        ensure_bus(
            network=network,
            name=raw_biogas_bus,
            carrier="raw_biogas",
            x=x,
            y=y,
        )


    ensure_carrier(
        network,
        "swfl_real_boiler_fuel",
    )


    # ==================================================================
    # 7. OPTIONAL HEL SUPPLY
    # ==================================================================

    allow_hel_backup = _as_bool(
        cfg.get(
            "allow_hel_backup",
            False,
        ),
        False,
    )


    hel_bus = str(
        cfg.get(
            "hel_bus",
            "swfl_real_hel_bus",
        )
    )

    hel_generator = str(
        cfg.get(
            "hel_supply_generator",
            "swfl_real_hel_supply",
        )
    )


    remove_components(
        network,
        "Generator",
        [
            hel_generator,
        ],
    )


    if allow_hel_backup:

        ensure_carrier(
            network,
            "heating_oil",
        )

        ensure_bus(
            network=network,
            name=hel_bus,
            carrier="heating_oil",
            x=x,
            y=y,
        )

        network.add(
            "Generator",
            hel_generator,
            bus=hel_bus,
            carrier="heating_oil",
            p_nom=float(
                cfg.get(
                    "hel_supply_p_nom_mw",
                    1.0e6,
                )
            ),
            p_nom_extendable=False,
            p_min_pu=0.0,
            p_max_pu=1.0,
            marginal_cost=float(
                cfg.get(
                    "hel_marginal_cost",
                    0.0,
                )
            ),
            capital_cost=0.0,
        )


    # ==================================================================
    # 8. HELPER: FUEL-SUPPLY LINK
    # ==================================================================

    def add_fuel_supply_link(
        name: str,
        source_bus: str,
        target_bus: str,
        p_nom: float,
        carrier: str,
        marginal_cost: float = 0.0,
    ) -> None:
        """
        Connect one external fuel bus to one boiler's internal fuel bus.

        p_nom is on the fuel-input side.
        """

        if source_bus not in network.buses.index.astype(str):

            raise ValueError(
                f"Fuel source bus {source_bus!r} "
                f"for Link {name!r} does not exist."
            )


        if target_bus not in network.buses.index.astype(str):

            raise ValueError(
                f"Fuel target bus {target_bus!r} "
                f"for Link {name!r} does not exist."
            )


        if p_nom < 0:

            raise ValueError(
                f"Fuel Link {name!r} has negative "
                f"capacity {p_nom} MW."
            )


        ensure_carrier(
            network,
            carrier,
        )


        remove_components(
            network,
            "Link",
            [
                name,
            ],
        )


        network.add(
            "Link",
            name,
            bus0=source_bus,
            bus1=target_bus,
            carrier=carrier,
            p_nom=float(
                p_nom
            ),
            p_nom_extendable=False,
            p_min_pu=0.0,
            p_max_pu=1.0,
            efficiency=1.0,
            marginal_cost=float(
                marginal_cost
            ),
            capital_cost=0.0,
        )


    # ==================================================================
    # 9. FUEL-FIRED BOILERS
    # ==================================================================

    total_boiler_heat_capacity = 0.0

    biomethane_heat_capacity = 0.0

    raw_biogas_heat_capacity = 0.0


    for unit in boilers:

        name = str(
            unit.get(
                "name",
                "",
            )
        ).strip()


        if not name:

            raise ValueError(
                "Each central heat boiler needs a name."
            )


        unit_active = _as_bool(
            unit.get(
                "active",
                True,
            ),
            True,
        )


        fuel_bus = str(
            unit.get(
                "fuel_bus",
                f"{name}_fuel_bus",
            )
        )


        boiler_link = str(
            unit.get(
                "heat_link_name",
                f"{name}_to_heat",
            )
        )


        natural_gas_link = (
            f"{name}_natural_gas_supply"
        )

        biomethane_link = (
            f"{name}_biomethane_supply"
        )

        raw_biogas_link = (
            f"{name}_raw_biogas_supply"
        )

        hel_link = (
            f"{name}_hel_supply"
        )


        # --------------------------------------------------------------
        # Remove stale versions before rebuilding this unit.
        # --------------------------------------------------------------

        remove_components(
            network,
            "Link",
            [
                boiler_link,
                natural_gas_link,
                biomethane_link,
                raw_biogas_link,
                hel_link,
            ],
        )


        if not unit_active:
            continue


        # --------------------------------------------------------------
        # Physical boiler parameters
        # --------------------------------------------------------------

        heat_capacity = float(
            unit.get(
                "heat_capacity_mw",
                0.0,
            )
        )


        efficiency = float(
            unit.get(
                "efficiency",
                cfg.get(
                    "default_boiler_efficiency",
                    0.90,
                ),
            )
        )


        if heat_capacity <= 0:

            raise ValueError(
                f"Boiler {name!r} has invalid heat capacity "
                f"{heat_capacity} MW."
            )


        if efficiency <= 0:

            raise ValueError(
                f"Boiler {name!r} has invalid efficiency "
                f"{efficiency}."
            )


        fuel_input_capacity = (
            heat_capacity
            / efficiency
        )


        # --------------------------------------------------------------
        # Internal common fuel bus.
        #
        # The boiler Link downstream of this bus enforces the shared
        # physical capacity for all fuels.
        # --------------------------------------------------------------

        ensure_bus(
            network=network,
            name=fuel_bus,
            carrier="swfl_real_boiler_fuel",
            x=x,
            y=y,
        )


        # --------------------------------------------------------------
        # Base fuels
        # --------------------------------------------------------------

        base_fuels = {
            str(fuel)
            .strip()
            .lower()
            for fuel in unit.get(
                "base_fuels",
                [
                    "natural_gas",
                ],
            )
        }


        # --------------------------------------------------------------
        # Natural gas -> internal boiler fuel bus
        # --------------------------------------------------------------

        if "natural_gas" in base_fuels:

            add_fuel_supply_link(
                name=natural_gas_link,
                source_bus=natural_gas_bus,
                target_bus=fuel_bus,
                p_nom=fuel_input_capacity,
                carrier=(
                    "swfl_real_natural_gas_to_boiler"
                ),
                marginal_cost=float(
                    unit.get(
                        "natural_gas_transport_cost",
                        0.0,
                    )
                ),
            )


        # --------------------------------------------------------------
        # Upgraded biomethane -> internal boiler fuel bus
        # --------------------------------------------------------------

        if name in biomethane_units:

            add_fuel_supply_link(
                name=biomethane_link,
                source_bus=biomethane_bus,
                target_bus=fuel_bus,
                p_nom=fuel_input_capacity,
                carrier=(
                    "swfl_real_biomethane_to_boiler"
                ),
                marginal_cost=float(
                    unit.get(
                        "biomethane_transport_cost",
                        0.0,
                    )
                ),
            )

            biomethane_heat_capacity += (
                heat_capacity
            )


        # --------------------------------------------------------------
        # Direct raw biogas -> internal boiler fuel bus
        #
        # IMPORTANT:
        # Fuel price + collection cost are represented upstream at the
        # dedicated raw-biogas source Generator.
        #
        # Therefore this Link must not add the 83.78 EUR/MWh_Hs again.
        # --------------------------------------------------------------

        if name in raw_biogas_units:

            add_fuel_supply_link(
                name=raw_biogas_link,
                source_bus=raw_biogas_bus,
                target_bus=fuel_bus,
                p_nom=fuel_input_capacity,
                carrier=(
                    "swfl_real_raw_biogas_to_boiler"
                ),
                marginal_cost=float(
                    unit.get(
                        "raw_biogas_transport_cost",
                        0.0,
                    )
                ),
            )

            raw_biogas_heat_capacity += (
                heat_capacity
            )


        # --------------------------------------------------------------
        # Optional HEL
        # --------------------------------------------------------------

        optional_fuels = {
            str(fuel)
            .strip()
            .lower()
            for fuel in unit.get(
                "optional_fuels",
                [],
            )
        }


        if (
            allow_hel_backup
            and "hel" in optional_fuels
        ):

            add_fuel_supply_link(
                name=hel_link,
                source_bus=hel_bus,
                target_bus=fuel_bus,
                p_nom=fuel_input_capacity,
                carrier=(
                    "swfl_real_hel_to_boiler"
                ),
                marginal_cost=0.0,
            )


        # --------------------------------------------------------------
        # Boiler conversion Link:
        #
        # common fuel bus -> SWFL heat bus
        # --------------------------------------------------------------

        boiler_carrier = str(
            unit.get(
                "carrier",
                "central_gas_boiler",
            )
        )


        ensure_carrier(
            network,
            boiler_carrier,
        )


        network.add(
            "Link",
            boiler_link,
            bus0=fuel_bus,
            bus1=heat_bus,
            carrier=boiler_carrier,

            # PyPSA Link p_nom is fuel-input capacity.
            p_nom=fuel_input_capacity,

            p_nom_extendable=_as_bool(
                unit.get(
                    "extendable",
                    False,
                ),
                False,
            ),

            p_min_pu=float(
                unit.get(
                    "p_min_pu",
                    0.0,
                )
            ),

            p_max_pu=float(
                unit.get(
                    "p_max_pu",
                    1.0,
                )
            ),

            efficiency=efficiency,

            # Fuel cost is represented upstream.
            marginal_cost=float(
                unit.get(
                    "marginal_cost",
                    0.0,
                )
            ),

            capital_cost=float(
                unit.get(
                    "capital_cost",
                    0.0,
                )
            ),
        )


        # --------------------------------------------------------------
        # Reporting metadata
        # --------------------------------------------------------------

        try:

            network.links.loc[
                boiler_link,
                "heat_capacity_mw",
            ] = heat_capacity

            network.links.loc[
                boiler_link,
                "boiler_efficiency",
            ] = efficiency

        except Exception:
            pass


        total_boiler_heat_capacity += (
            heat_capacity
        )


    # ==================================================================
    # 10. ELECTRODE / RESISTIVE BOILERS
    # ==================================================================

    total_resistive_heat_capacity = 0.0


    for unit in resistive_heaters:

        name = str(
            unit.get(
                "name",
                "",
            )
        ).strip()


        if not name:

            raise ValueError(
                "Each resistive heater needs a name."
            )


        remove_components(
            network,
            "Link",
            [
                name,
            ],
        )


        if not _as_bool(
            unit.get(
                "active",
                True,
            ),
            True,
        ):
            continue


        heat_capacity = float(
            unit.get(
                "heat_capacity_mw",
                0.0,
            )
        )


        efficiency = float(
            unit.get(
                "efficiency",
                cfg.get(
                    "default_resistive_efficiency",
                    0.99,
                ),
            )
        )


        if heat_capacity <= 0:

            raise ValueError(
                f"Resistive heater {name!r} has invalid "
                f"heat capacity {heat_capacity} MW."
            )


        if efficiency <= 0:

            raise ValueError(
                f"Resistive heater {name!r} has invalid "
                f"efficiency {efficiency}."
            )


        electric_input_capacity = (
            heat_capacity
            / efficiency
        )


        carrier = str(
            unit.get(
                "carrier",
                "central_resistive_heater",
            )
        )


        ensure_carrier(
            network,
            carrier,
        )


        network.add(
            "Link",
            name,
            bus0=str(
                unit.get(
                    "ac_bus",
                    ac_bus,
                )
            ),
            bus1=str(
                unit.get(
                    "heat_bus",
                    heat_bus,
                )
            ),
            carrier=carrier,
            p_nom=electric_input_capacity,
            p_nom_extendable=_as_bool(
                unit.get(
                    "extendable",
                    False,
                ),
                False,
            ),
            p_min_pu=float(
                unit.get(
                    "p_min_pu",
                    0.0,
                )
            ),
            p_max_pu=float(
                unit.get(
                    "p_max_pu",
                    1.0,
                )
            ),
            efficiency=efficiency,
            marginal_cost=float(
                unit.get(
                    "marginal_cost",
                    0.0,
                )
            ),
            capital_cost=float(
                unit.get(
                    "capital_cost",
                    0.0,
                )
            ),
        )


        try:

            network.links.loc[
                name,
                "heat_capacity_mw",
            ] = heat_capacity

        except Exception:
            pass


        total_resistive_heat_capacity += (
            heat_capacity
        )


    # ==================================================================
    # 11. CAPACITY VALIDATION
    # ==================================================================

    total_heat_capacity = (
        total_boiler_heat_capacity
        + total_resistive_heat_capacity
    )


    expected_capacity = float(
        cfg.get(
            "expected_total_heat_capacity_mw",
            370.0,
        )
    )


    tolerance = float(
        cfg.get(
            "capacity_validation_tolerance_mw",
            1.0e-6,
        )
    )


    if (
        abs(
            total_heat_capacity
            - expected_capacity
        )
        > tolerance
    ):

        logger.warning(
            "SWFL central heat capacity is %.3f MW, "
            "but expected %.3f MW.",
            total_heat_capacity,
            expected_capacity,
        )


    # ==================================================================
    # 12. SUMMARY
    # ==================================================================

    print(
        "\nSWFL detailed central heat units added"
    )

    print(
        f"  boiler heat capacity MW:       "
        f"{total_boiler_heat_capacity:.3f}"
    )

    print(
        f"  resistive heat capacity MW:    "
        f"{total_resistive_heat_capacity:.3f}"
    )

    print(
        f"  total heat capacity MW:        "
        f"{total_heat_capacity:.3f}"
    )

    print(
        f"  biomethane-enabled units:      "
        f"{sorted(biomethane_units) if biomethane_units else 'none'}"
    )

    print(
        f"  biomethane heat capacity MW:   "
        f"{biomethane_heat_capacity:.3f}"
    )

    print(
        f"  raw-biogas-enabled units:      "
        f"{sorted(raw_biogas_units) if raw_biogas_units else 'none'}"
    )

    print(
        f"  raw-biogas heat capacity MW:   "
        f"{raw_biogas_heat_capacity:.3f}"
    )

    print(
        f"  HEL backup active:             "
        f"{allow_hel_backup}"
    )

def add_reserve_gas_boiler(network, cfg: Dict[str, Any], gas_bus: str, heat_bus: str) -> None:
    name = str(cfg.get("name", "swfl_real_reserve_gas_boiler"))
    heat_cap = float(cfg.get("heat_capacity_mw", 203.0))
    eff = float(cfg.get("efficiency", 1.0))
    p_nom_is_output = _as_bool(cfg.get("p_nom_is_output_capacity", True), True)
    p_nom = heat_cap / eff if p_nom_is_output and eff else heat_cap

    remove_components(network, "Link", [name])
    network.add(
        "Link",
        name,
        bus0=gas_bus,
        bus1=heat_bus,
        carrier=str(cfg.get("carrier", "central_gas_boiler")),
        p_nom=p_nom,
        p_nom_extendable=_as_bool(cfg.get("extendable", False), False),
        p_min_pu=float(cfg.get("p_min_pu", 0.0)),
        p_max_pu=float(cfg.get("p_max_pu", 1.0)),
        efficiency=eff,
        marginal_cost=float(cfg.get("marginal_cost", 0.0)),
        capital_cost=float(cfg.get("capital_cost", 0.0)),
    )


def add_future_heat_pumps(
    network,
    cfg: Dict[str, Any],
    ac_bus: str,
    heat_bus: str,
) -> None:
    """
    Add future SWFL heat pumps; each unit can be activated individually.

    Each planned unit receives its own carrier by default. This is important
    because eTraGo may aggregate Links that have the same carrier and clustered
    endpoint buses. Separate carriers preserve GWP 1 and GWP 2 as distinct
    Links and prevent them from being merged with generic eGon heat pumps.
    """
    units = cfg.get("units", []) or []
    active_units = cfg.get("active_units", None)

    active_units_set = (
        {str(value) for value in active_units}
        if active_units is not None
        else None
    )

    for unit in units:
        name = str(unit.get("name", "")).strip()

        if not name:
            raise ValueError(
                "Each future heat-pump unit needs a name."
            )

        if active_units_set is not None:
            unit_active = name in active_units_set
        else:
            unit_active = _as_bool(
                unit.get("active", True),
                True,
            )

        if not unit_active:
            remove_components(
                network,
                "Link",
                [name],
            )
            continue

        heat_capacity = float(
            unit.get(
                "heat_capacity_mw",
                60.0,
            )
        )

        cop = float(
            unit.get(
                "cop",
                cfg.get("default_cop", 3.0),
            )
        )

        if heat_capacity <= 0:
            raise ValueError(
                f"Heat pump {name} has invalid heat capacity "
                f"{heat_capacity} MWth."
            )

        if cop <= 0:
            raise ValueError(
                f"Heat pump {name} has invalid COP {cop}."
            )

        # PyPSA Link p_nom is electricity-input capacity.
        # Useful heat output is p0 * COP.
        electric_input_capacity = heat_capacity / cop

        # Do not fall back to the shared generic central_heat_pump carrier.
        # A unique carrier prevents aggregation with other heat pumps.
        unit_carrier = str(
            unit.get("carrier")
            or f"{name}_heat_pump"
        )

        ensure_carrier(
            network,
            unit_carrier,
        )

        remove_components(
            network,
            "Link",
            [name],
        )

        network.add(
            "Link",
            name,
            bus0=str(
                unit.get(
                    "ac_bus",
                    ac_bus,
                )
            ),
            bus1=str(
                unit.get(
                    "heat_bus",
                    heat_bus,
                )
            ),
            carrier=unit_carrier,
            p_nom=electric_input_capacity,
            p_nom_extendable=_as_bool(
                unit.get(
                    "extendable",
                    cfg.get("extendable", False),
                ),
                False,
            ),
            p_min_pu=float(
                unit.get(
                    "p_min_pu",
                    cfg.get("p_min_pu", 0.0),
                )
            ),
            p_max_pu=float(
                unit.get(
                    "p_max_pu",
                    cfg.get("p_max_pu", 1.0),
                )
            ),
            efficiency=cop,
            marginal_cost=float(
                unit.get(
                    "marginal_cost",
                    cfg.get("marginal_cost", 0.0),
                )
            ),
            capital_cost=float(
                unit.get(
                    "capital_cost",
                    cfg.get("capital_cost", 0.0),
                )
            ),
        )

        # Optional metadata for reporting.
        metadata = {
            "heat_capacity_mw": heat_capacity,
            "cop": cop,
            "planned_year": unit.get("planned_year", np.nan),
            "swfl_heat_pump_unit": name,
        }

        for column, value in metadata.items():
            try:
                network.links.loc[name, column] = value
            except Exception:
                pass


def validate_swfl_heat_pumps(
    network,
    hp_cfg: Dict[str, Any],
    stage: str,
) -> pd.DataFrame:
    """
    Validate active planned SWFL heat pumps before or after clustering.

    Matching is carrier-based, so validation still works when clustering
    replaces the original Link names with numeric IDs.
    """
    if not _as_bool(
        hp_cfg.get("active", False),
        False,
    ):
        print(
            f"\nSWFL heat-pump validation [{stage}]: inactive"
        )
        return pd.DataFrame()

    units = hp_cfg.get("units", []) or []
    configured_active_units = hp_cfg.get(
        "active_units",
        None,
    )

    if configured_active_units is not None:
        active_names = {
            str(name)
            for name in configured_active_units
        }
    else:
        active_names = {
            str(unit.get("name", "")).strip()
            for unit in units
            if _as_bool(
                unit.get("active", True),
                True,
            )
        }

    active_names.discard("")

    rows = []
    links = network.links

    for unit in units:
        name = str(
            unit.get("name", "")
        ).strip()

        if not name or name not in active_names:
            continue

        heat_capacity = float(
            unit.get(
                "heat_capacity_mw",
                60.0,
            )
        )

        cop = float(
            unit.get(
                "cop",
                hp_cfg.get("default_cop", 3.0),
            )
        )

        expected_p_nom = heat_capacity / cop

        expected_carrier = str(
            unit.get("carrier")
            or f"{name}_heat_pump"
        )

        matches = links[
            links["carrier"]
            .astype(str)
            .eq(expected_carrier)
        ]

        # Before clustering, also accept the original component name.
        if matches.empty and name in links.index.astype(str):
            matches = links.loc[[name]]

        if matches.empty:
            raise ValueError(
                f"[{stage}] Missing planned SWFL heat pump "
                f"{name!r} with carrier {expected_carrier!r}."
            )

        if len(matches) != 1:
            available_columns = [
                column
                for column in [
                    "bus0",
                    "bus1",
                    "carrier",
                    "p_nom",
                    "p_nom_opt",
                    "efficiency",
                ]
                if column in matches.columns
            ]

            raise ValueError(
                f"[{stage}] Expected exactly one Link for {name!r}, "
                f"but found {len(matches)}:\n"
                + matches[available_columns].to_string()
            )

        link_id = matches.index[0]
        row = matches.iloc[0]

        # Validate configured capacity before optimisation.
        actual_p_nom = float(row["p_nom"])
        actual_cop = float(row["efficiency"])

        rows.append(
            {
                "unit": name,
                "link_id": str(link_id),
                "carrier": str(row["carrier"]),
                "bus0": str(row["bus0"]),
                "bus1": str(row["bus1"]),
                "expected_p_nom_mwel": expected_p_nom,
                "actual_p_nom_mwel": actual_p_nom,
                "expected_cop": cop,
                "actual_cop": actual_cop,
            }
        )

        if abs(actual_p_nom - expected_p_nom) > 1.0e-3:
            raise ValueError(
                f"[{stage}] Invalid electricity-input capacity for "
                f"{name}: expected {expected_p_nom:.6f} MW, "
                f"found {actual_p_nom:.6f} MW."
            )

        if abs(actual_cop - cop) > 1.0e-6:
            raise ValueError(
                f"[{stage}] Invalid COP for {name}: "
                f"expected {cop:.6f}, found {actual_cop:.6f}."
            )

    if len(rows) != len(active_names):
        found_names = {
            row["unit"]
            for row in rows
        }

        raise ValueError(
            f"[{stage}] Active heat-pump configuration and validated "
            f"units differ. Active={sorted(active_names)}, "
            f"validated={sorted(found_names)}."
        )

    result = pd.DataFrame(rows)

    print(
        f"\nSWFL heat-pump validation [{stage}]"
    )
    print(
        result.to_string(index=False)
    )

    # Diagnostic only. Generic heat pumps elsewhere in the network may remain,
    # but they must no longer be merged with the planned SWFL units.
    generic = links[
        links["carrier"]
        .astype(str)
        .eq("central_heat_pump")
    ]

    if not generic.empty:
        available_columns = [
            column
            for column in [
                "bus0",
                "bus1",
                "carrier",
                "p_nom",
                "p_nom_opt",
                "efficiency",
            ]
            if column in generic.columns
        ]

        print(
            f"\nRemaining generic central heat pumps [{stage}]"
        )
        print(
            generic[available_columns].to_string()
        )

    return result


# =============================================================================
# Time-series IO helpers
# =============================================================================


def read_table_flexible(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    attempts = [
        {"sep": None, "engine": "python"},
        {"sep": ";", "decimal": ","},
        {"sep": ";", "decimal": "."},
        {"sep": ",", "decimal": "."},
        {"sep": ",", "decimal": ","},
        {"sep": "\t", "decimal": ","},
        {"sep": "\t", "decimal": "."},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(path, **kwargs)
            if df.shape[1] >= 2:
                return df
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not read {path}: {last_error}")


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().replace("\ufeff", "") for c in out.columns]
    return out


def normalise_col_name(s: str) -> str:
    return "".join(str(s).strip().lower().replace("_", " ").split())


def find_column(df: pd.DataFrame, requested: str) -> str:
    req = normalise_col_name(requested)
    exact = [c for c in df.columns if normalise_col_name(c) == req]
    if exact:
        return exact[0]
    fuzzy = [c for c in df.columns if req in normalise_col_name(c) or normalise_col_name(c) in req]
    if fuzzy:
        return fuzzy[0]
    raise KeyError(f"Could not find column {requested!r}. Available: {list(df.columns)}")


def extract_datetime_index(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DatetimeIndex:
    if cfg.get("datetime_column"):
        candidates = [cfg["datetime_column"]]
    else:
        candidates = ["timestamp", "datetime", "time", "date", "datum", "zeit", "DateTime", "Datum", "Zeit"]

    for c in candidates:
        if c in df.columns:
            dt = pd.to_datetime(df[c], errors="coerce", dayfirst=True)
            if dt.notna().sum() > 0:
                return pd.DatetimeIndex(dt)

    date_cols = [c for c in df.columns if normalise_col_name(c) in {"date", "datum"}]
    time_cols = [c for c in df.columns if normalise_col_name(c) in {"time", "zeit", "hour", "stunde"}]
    if date_cols and time_cols:
        dt = pd.to_datetime(df[date_cols[0]].astype(str) + " " + df[time_cols[0]].astype(str), errors="coerce", dayfirst=True)
        if dt.notna().sum() > 0:
            return pd.DatetimeIndex(dt)

    first = df.columns[0]
    dt = pd.to_datetime(df[first], errors="coerce", dayfirst=True)
    if dt.notna().sum() > 0:
        return pd.DatetimeIndex(dt)

    raise ValueError("Could not detect datetime column. Set datetime_column in heat_load config.")


def make_8760_hourly_year(s: pd.Series, year: int, fill_method: str = "time_interpolate") -> pd.Series:
    s = s.copy().sort_index()
    s.index = pd.DatetimeIndex(s.index)
    if s.index.tz is not None:
        s.index = s.index.tz_localize(None)

    # Remove leap day if present.
    s = s[~((s.index.month == 2) & (s.index.day == 29))]

    idx = pd.date_range(f"{year}-01-01 00:00:00", f"{year}-12-31 23:00:00", freq="h")
    idx = idx[~((idx.month == 2) & (idx.day == 29))]
    s = s.reindex(idx)

    if fill_method == "zero":
        s = s.fillna(0.0)
    elif fill_method == "ffill":
        s = s.ffill().bfill()
    elif fill_method == "time_interpolate":
        s = s.interpolate(method="time").ffill().bfill()
    else:
        raise ValueError(f"Unsupported fill_method {fill_method!r}")

    if len(s) != 8760:
        raise ValueError(f"Expected 8760 values, got {len(s)}")
    return s.astype(float)


def map_year_profile_to_network_snapshots(profile_8760: pd.Series, snapshots: pd.Index) -> pd.Series:
    profile_8760 = profile_8760.copy()
    profile_8760.index = pd.DatetimeIndex(profile_8760.index)

    lookup = {
        (ts.month, ts.day, ts.hour): float(val)
        for ts, val in profile_8760.items()
        if not (ts.month == 2 and ts.day == 29)
    }

    values = []
    for sn in pd.DatetimeIndex(snapshots):
        key = (sn.month, sn.day, sn.hour)
        values.append(lookup.get(key, np.nan))

    return pd.Series(values, index=snapshots, dtype=float).interpolate().ffill().bfill()


def snapshot_weights(network, snapshots: pd.Index) -> pd.Series:
    try:
        sw = network.snapshot_weightings
        if isinstance(sw, pd.DataFrame):
            if "generators" in sw.columns:
                w = sw["generators"].reindex(snapshots)
            elif "objective" in sw.columns:
                w = sw["objective"].reindex(snapshots)
            else:
                w = sw.iloc[:, 0].reindex(snapshots)
        else:
            w = pd.Series(sw, index=snapshots)
        return pd.to_numeric(w, errors="coerce").fillna(1.0).astype(float)
    except Exception:
        return pd.Series(1.0, index=snapshots, dtype=float)


def _ensure_timeseries_tables(network) -> None:
    if not hasattr(network, "loads_t"):
        return
    if not hasattr(network.loads_t, "p_set") or network.loads_t.p_set is None:
        network.loads_t.p_set = pd.DataFrame(index=network.snapshots)
    if not isinstance(network.loads_t.p_set, pd.DataFrame):
        network.loads_t.p_set = pd.DataFrame(network.loads_t.p_set, index=network.snapshots)


# =============================================================================
# Summary and helpers
# =============================================================================


def print_swfl_real_system_summary(
    network,
    settings: Dict[str, Any],
    area_buses: Set[str],
    heat_profile: Optional[pd.Series],
    ac_shape: Optional[pd.Series],
) -> None:
    hp_cfg = settings.get("future_heat_pumps", {}) or {}
    active_hps = []
    if _as_bool(hp_cfg.get("active", False), False):
        active_units = hp_cfg.get("active_units")
        active_set = {str(x) for x in active_units} if active_units is not None else None
        for u in hp_cfg.get("units", []) or []:
            name = str(u.get("name"))
            if active_set is not None:
                if name in active_set:
                    active_hps.append(name)
            elif _as_bool(u.get("active", True), True):
                active_hps.append(name)

    print("\nSWFL real system added")
    print(f"  area buses detected:       {len(area_buses)}")
    print(f"  swfl_ac_bus:              {settings.get('swfl_ac_bus', 'swfl_ac_bus')}")
    print(f"  swfl_heat_bus:            {settings.get('swfl_heat_bus', 'swfl_central_heat_bus')}")
    print(f"  swfl_ch4_bus:             {settings.get('swfl_ch4_bus', 'biogas_sh_swfl_ch4_bus')}")
    if heat_profile is not None:
        print(f"  heat load max MW:         {float(heat_profile.max()):.3f}")
        print(f"  heat load mean MW:        {float(heat_profile.mean()):.3f}")
    if ac_shape is not None:
        print(f"  AC source profile points: {len(ac_shape)}")
    print(f"  future heat pumps active: {active_hps if active_hps else 'none'}")


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"true", "1", "yes", "y", "on"}:
        return True
    if s in {"false", "0", "no", "n", "off"}:
        return False
    return default


# =============================================================================
# Example config block for appl.py
# =============================================================================


SWFL_REAL_SYSTEM_EXAMPLE = {
    "active": True,

    # ------------------------------------------------------------------
    # Area definition: DING0 MV grid districts, not radius
    # ------------------------------------------------------------------
    # Flensburg/SWFL assets are selected from these DING0 MV grid districts:
    #   33935, 33543, 35906
    #
    # No radius-based area selection is used.
    "area_mode": "ding0_mv_grid_districts",

    # In appl.py, use:
    # "mv_grid_districts_gpkg": str(DING0_MV_GPKG),
    #
    # If this example stays inside swfl_real_system.py, keep it as a string path.
    "mv_grid_districts_gpkg": "/path/to/ding0_mv_grid_districts.gpkg",
    "mv_grid_layer": None,
    "mv_grid_id_column": "name",

    "selected_mv_grid_district_ids": [
        "33935",
        "33543",
        "35906",
    ],

    # Also include directly connected local heat/CHP/boiler buses.
    # This is important because the MV district IDs are mainly AC buses,
    # while the heat buses are often connected via conversion links.
    "expand_area_through_local_links": True,

    # Remove old generic eGon assets inside the selected MV districts.
    "remove_existing_flensburg_assets": True,

    # Do not delete Biogas-SH or newly created SWFL components.
    "protected_prefixes": [
        "biogas_sh_",
        "swfl_real_",
        "swfl_gwp_",
    ],

    # Old eGon conversion links to remove in the selected MV districts.
    "remove_link_carrier_patterns": [
        "central_gas",
        "central_heat",
        "rural_heat",
        "heat_pump",
        "CHP",
        "boiler",
    ],

    # Old eGon heat pumps to remove in the selected MV districts.
    "remove_heat_pump_carrier_patterns": [
        "central_heat_pump",
        "rural_heat_pump",
        "heat_pump",
    ],

    # ------------------------------------------------------------------
    # Main replacement buses
    # ------------------------------------------------------------------
    "swfl_ac_bus": "33935",
    "swfl_heat_bus": "swfl_real_central_heat_bus",
    "swfl_ch4_bus": "biogas_sh_swfl_ch4_bus",

    "swfl_ch4_bus_x": 9.436502119171873,
    "swfl_ch4_bus_y": 54.79233181101448,

    # ------------------------------------------------------------------
    # Real SWFL heat load
    # ------------------------------------------------------------------
    "heat_load": {
        "active": True,

        # In appl.py, use:
        # "csv_path": str(SWFL_HEAT_CSV),
        #
        # If this example stays inside swfl_real_system.py, keep it as a string path.
        "csv_path": "/path/to/stadtwerke_hourly_heat.csv",

        "year": 2023,
        "column": "HKW Wärmeleistung Gesamt",
        "unit": "MW",
        "carrier": "central_heat",
        "name": "swfl_real_heat_load",

        # Optional if automatic detection fails:
        # "datetime_column": "Zeitstempel",

        "fill_method": "time_interpolate",
        "clip_negative": True,
    },

    # ------------------------------------------------------------------
    # Real-scaled Flensburg/SWFL AC load
    # ------------------------------------------------------------------
    # The temporal profile shape comes from existing eGon AC loads in the
    # selected MV districts. The annual sum is scaled to real electricity demand.
    "ac_load": {
        "active": True,
        "use_existing_egon_profile_shape": True,

        # Real annual electricity demand:
        # 381.516 GWh/a = 381,516 MWh/a
        "target_annual_demand_mwh": 381516.0,

        "carrier": "AC",
        "source_carrier": "AC",
        "name": "swfl_real_ac_load",
        "clip_negative": True,

        # Optional if automatic MV-district selection misses the correct loads:
        # "source_load_ids": ["load_id_1", "load_id_2"],
    },

    # ------------------------------------------------------------------
    # Real SWFL central gas CHP
    # ------------------------------------------------------------------
    # Represented as two simple links:
    #   CH4 bus -> AC bus
    #   CH4 bus -> central heat bus
    #
    # This version does not yet force a fixed heat-to-power coupling.
    "central_gas_chp": {
        "active": True,

        "electric_link_name": "swfl_real_central_gas_CHP",
        "heat_link_name": "swfl_real_central_gas_CHP_heat",

        "electric_capacity_mw": 241.0,
        "heat_capacity_mw": 370.0,

        "gas_bus": "biogas_sh_swfl_ch4_bus",
        "ac_bus": "33935",
        "heat_bus": "swfl_real_central_heat_bus",

        "carrier_el": "central_gas_CHP",
        "carrier_heat": "central_gas_CHP_heat",

        "p_nom_is_output_capacity": True,
        "electric_efficiency": 1.0,
        "heat_efficiency": 1.0,

        "extendable": False,
        "p_min_pu": 0.0,
        "p_max_pu": 1.0,

        "marginal_cost": 0.0,
        "capital_cost": 0.0,
    },

    # ------------------------------------------------------------------
    # Optional reserve gas boiler
    # ------------------------------------------------------------------
    "reserve_gas_boiler": {
        "active": False,

        "name": "swfl_real_reserve_gas_boiler",
        "heat_capacity_mw": 203.0,

        "gas_bus": "biogas_sh_swfl_ch4_bus",
        "heat_bus": "swfl_real_central_heat_bus",

        "carrier": "central_gas_boiler",

        "p_nom_is_output_capacity": True,
        "efficiency": 1.0,

        "extendable": False,
        "p_min_pu": 0.0,
        "p_max_pu": 1.0,

        "marginal_cost": 0.0,
        "capital_cost": 0.0,
    },

    # ------------------------------------------------------------------
    # Future SWFL large heat pumps
    # ------------------------------------------------------------------
    "future_heat_pumps": {
        # First remove existing eGon heat pumps inside the selected MV districts.
        "remove_existing_central_heat_pumps": True,

        # False = remove old eGon heat pumps but add no new SWFL heat pumps.
        # True  = add selected units below.
        "active": False,

        # Flexible options:
        #
        # none:
        # "active": False
        #
        # only GWP 1:
        # "active": True,
        # "active_units": ["swfl_gwp_1"]
        #
        # only GWP 2:
        # "active": True,
        # "active_units": ["swfl_gwp_2"]
        #
        # both:
        # "active": True,
        # "active_units": ["swfl_gwp_1", "swfl_gwp_2"]

        # No shared carrier: each planned unit has its own carrier.
        "default_cop": 3.0,

        "extendable": False,
        "p_min_pu": 0.0,
        "p_max_pu": 1.0,

        "marginal_cost": 0.0,
        "capital_cost": 0.0,

        "units": [
            {
                "name": "swfl_gwp_1",
                "carrier": "swfl_gwp_1_heat_pump",
                "active": True,
                "heat_capacity_mw": 60.0,
                "cop": 3.0,
                "planned_year": 2028,
                "ac_bus": "33935",
                "heat_bus": "swfl_real_central_heat_bus",
            },
            {
                "name": "swfl_gwp_2",
                "carrier": "swfl_gwp_2_heat_pump",
                "active": True,
                "heat_capacity_mw": 60.0,
                "cop": 3.0,
                "planned_year": None,
                "ac_bus": "33935",
                "heat_bus": "swfl_real_central_heat_bus",
            },
        ],
    },
}


def remove_known_legacy_swfl_heat_pump_before_clustering(
    network,
) -> None:
    """
    Remove the known small legacy eGon heat pump before spatial
    clustering.

    Similar-capacity central heat pumps elsewhere in the network
    are preserved.
    """
    import numpy as np

    target_p_nom = 0.283661
    links = network.links

    candidates = links.loc[
        links["carrier"].astype(str).eq(
            "central_heat_pump"
        )
    ].copy()

    candidates["_p_nom_numeric"] = pd.to_numeric(
        candidates["p_nom"],
        errors="coerce",
    )

    candidates["_capacity_difference"] = (
        candidates["_p_nom_numeric"]
        - target_p_nom
    ).abs()

    # Find plausible components near the expected capacity.
    candidates = candidates.loc[
        candidates["_capacity_difference"]
        <= 1.0e-4
    ].copy()

    print(
        "\nLegacy SWFL heat-pump search "
        "[before spatial clustering]"
    )

    display_columns = [
        column
        for column in [
            "bus0",
            "bus1",
            "carrier",
            "p_nom",
            "p_nom_opt",
            "efficiency",
            "_capacity_difference",
        ]
        if column in candidates.columns
    ]

    if candidates.empty:
        raise RuntimeError(
            "No central heat pump close to "
            f"{target_p_nom:.6f} MW was found before "
            "spatial clustering."
        )

    candidates = candidates.sort_values(
        "_capacity_difference"
    )

    print("\nSimilar-capacity candidates:")
    print(
        candidates[display_columns].to_string()
    )

    minimum_difference = candidates[
        "_capacity_difference"
    ].min()

    legacy = candidates.loc[
        np.isclose(
            candidates["_capacity_difference"],
            minimum_difference,
            atol=1.0e-12,
            rtol=0.0,
        )
    ]

    if len(legacy) != 1:
        raise RuntimeError(
            "Could not identify one unique closest legacy "
            "SWFL heat pump:\n"
            + legacy[display_columns].to_string()
        )

    legacy_id = legacy.index[0]

    print("\nSelected legacy SWFL heat pump:")
    print(
        legacy[display_columns].to_string()
    )

    # Preserve the actual PyPSA index type.
    network.remove(
        "Link",
        legacy_id,
    )

    # Verify only the selected Link itself was removed.
    if legacy_id in network.links.index:
        raise RuntimeError(
            f"Legacy heat-pump Link {legacy_id!r} "
            "still exists after removal."
        )

    print(
        f"Removed legacy heat-pump Link {legacy_id!r} "
        "before spatial clustering: PASS"
    )

def purge_legacy_swfl_heat_pumps(
    network,
    stage: str,
) -> None:
    """
    Remove generic eGon heat pumps that share the clustered connection
    of the planned SWFL GWP Links.

    Uses actual index values and verifies removal immediately.
    """
    links = network.links

    planned_carriers = {
        "swfl_gwp_1_heat_pump",
        "swfl_gwp_2_heat_pump",
    }

    carrier = links["carrier"].astype(str)

    planned = links[
        carrier.isin(planned_carriers)
    ]

    if planned.empty:
        raise RuntimeError(
            f"[{stage}] No planned SWFL heat pumps found."
        )

    planned_connections = {
        (
            str(row["bus0"]),
            str(row["bus1"]),
        )
        for _, row in planned.iterrows()
    }

    generic_mask = carrier.isin(
        {
            "central_heat_pump",
            "rural_heat_pump",
        }
    )

    same_connection_mask = pd.Series(
        [
            (
                str(row["bus0"]),
                str(row["bus1"]),
            )
            in planned_connections
            for _, row in links.iterrows()
        ],
        index=links.index,
    )

    legacy_ids = links.index[
        generic_mask & same_connection_mask
    ].tolist()

    print(
        f"\nLegacy SWFL heat-pump cleanup [{stage}]"
    )

    print(
        "  planned connections:",
        sorted(planned_connections),
    )

    if legacy_ids:
        columns = [
            column
            for column in [
                "bus0",
                "bus1",
                "carrier",
                "p_nom",
                "p_nom_opt",
                "efficiency",
            ]
            if column in links.columns
        ]

        print(
            "\n  Removing these legacy Links:"
        )
        print(
            links.loc[
                legacy_ids,
                columns,
            ].to_string()
        )

        # Pass actual index values directly to PyPSA.
        for link_id in legacy_ids:
            network.remove(
                "Link",
                link_id,
            )
    else:
        print(
            "  No legacy Link found."
        )

    # Hard verification after removal.
    links_after = network.links
    carrier_after = (
        links_after["carrier"].astype(str)
    )

    generic_after = carrier_after.isin(
        {
            "central_heat_pump",
            "rural_heat_pump",
        }
    )

    same_connection_after = pd.Series(
        [
            (
                str(row["bus0"]),
                str(row["bus1"]),
            )
            in planned_connections
            for _, row in links_after.iterrows()
        ],
        index=links_after.index,
    )

    remaining = links_after[
        generic_after
        & same_connection_after
    ]

    if not remaining.empty:
        raise RuntimeError(
            f"[{stage}] Legacy SWFL heat pump remains "
            "after direct removal:\n"
            + remaining[
                [
                    "bus0",
                    "bus1",
                    "carrier",
                    "p_nom",
                    "efficiency",
                ]
            ].to_string()
        )

    print(
        f"  PASS: no legacy heat pump remains [{stage}]"
    )
