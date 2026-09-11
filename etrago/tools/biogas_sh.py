# -*- coding: utf-8 -*-
"""
Biogas-SH integration for eTraGo.

This module adds project-specific Biogas-SH plants to an eTraGo/PyPSA network.
It follows the modelling decisions agreed for the Biogas-SH project:

Scenario dimensions
-------------------
1. Gas connection target:
   - "swfl"
   - "gas_grid"
   In the current project setup both can connect to target CH4 bus 47538.
   The distinction is kept as scenario metadata and for future cost/demand
   extensions.

2. Local generation switch:
   - add_local_generation = True/False

3. Gas generation switch:
   - add_gas_generation = True/False

Plant treatment
---------------
- AC-only plants are represented as industrial_biomass_CHP.
- AC + heat plants are represented as two separate fixed-capacity generators:
    central_biomass_CHP       -> electricity
    central_biomass_CHP_heat  -> heat
- Gas/SWFL option is represented as CH4_biogas production.

Demand-side connection
----------------------
- AC generators are connected to the AC load bus in the same ding0 MV grid
  district. If no AC load bus exists there, the nearest/neighbouring district
  with AC load is used.
- Heat generators are connected to the rural_heat bus/load in the same ding0
  MV grid district. In eTraGo this is usually the bus1 side of a
  rural_heat_pump link. If no rural_heat bus exists there, the nearest/
  neighbouring district with rural_heat is used.
- Multiple CHP_heat generators in the same MV district are connected to the
  same rural_heat bus. No additional heat link is required; PyPSA nodal
  balance lets generators on the same bus supply the load.

Gas topology
------------
Two gas topologies are supported:

1. direct_at_target
   Add CH4_biogas generator directly at target_ch4_bus.

2. producer_bus_link  [recommended]
   Add a plant-specific CH4 bus at the plant location, add the CH4_biogas
   generator there, and connect it to target_ch4_bus by a CH4 link. This is
   closer to the existing eGon CH4_biogas producer-bus topology.

Resource constraint
-------------------
This module only adds assets. The shared annual raw-biogas resource constraint
must be added in etrago/tools/constraints.py and activated through
args["extra_functionality"]["biogas_sh_resource"].

Expected minimum CSV columns
----------------------------
plant_id
plant_name
raw_biogas_mwh_hs_a
biomethane_mwh_hs_a_at_96pct
biomethane_price_eur_per_mwh_hs_at_96pct
lat
lon

Recommended CSV columns
-----------------------
hbl_95_mwel
installed_electric_capacity_mwel
useful_heat_mwh_a
plant_category            # optional: ac_only or ac_heat
ac_bus_for_model          # optional, can be auto-mapped
heat_bus_for_model        # optional, can be auto-mapped

Author: Biogas-SH project-specific eTraGo extension
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# Generic helpers
# =============================================================================


def _is_missing(value) -> bool:
    """Return True for NaN/None/empty-string values."""
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return str(value).strip() == ""



def _clean_id(value) -> Optional[str]:
    """Convert numeric ids such as 30872.0 to '30872'."""
    if _is_missing(value):
        return None
    try:
        f = float(value)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(value).strip()



def _safe_float(row, column: str, default=0.0) -> float:
    """Read a float from a dataframe row; return default if missing."""
    if column not in row.index or _is_missing(row[column]):
        return float(default)
    try:
        return float(row[column])
    except (TypeError, ValueError):
        return float(default)



def _safe_str(row, column: str, default=None) -> Optional[str]:
    """Read a string from a dataframe row; return default if missing."""
    if column not in row.index or _is_missing(row[column]):
        return default
    return str(row[column]).strip()



def _as_bool(value, default=False) -> bool:
    """Robust bool parser for args values."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)



def _ensure_carrier(network, carrier: str) -> None:
    """Add carrier if it does not yet exist."""
    if carrier not in network.carriers.index:
        network.add("Carrier", carrier)



def _remove_component_if_exists(network, component: str, name: str) -> None:
    """Remove a PyPSA component if it already exists."""
    table = getattr(network, component.lower() + "s")
    if name in table.index:
        network.mremove(component, [name])



def _read_biogas_sh_csv(csv_path: str | Path) -> pd.DataFrame:
    """Read and validate the Biogas-SH plant CSV."""
    csv_path = Path(csv_path).expanduser()
    if not csv_path.exists():
        raise FileNotFoundError(f"Biogas-SH CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    required = [
        "plant_id",
        "plant_name",
        "raw_biogas_mwh_hs_a",
        "biomethane_mwh_hs_a_at_96pct",
        "biomethane_price_eur_per_mwh_hs_at_96pct",
        "lat",
        "lon",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            "Biogas-SH CSV is missing required columns: " + ", ".join(missing)
        )

    df = df[df["raw_biogas_mwh_hs_a"].fillna(0) > 0].copy()
    df["plant_id"] = df["plant_id"].astype(int)
    return df



def _distance_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Approximate haversine distance in km."""
    r = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return float(r * c)


# =============================================================================
# MV grid district and demand-bus mapping
# =============================================================================


def _read_mv_grid_districts(settings):
    """Read ding0 MV grid district polygons from GeoPackage."""
    try:
        import geopandas as gpd  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "geopandas is required for Biogas-SH auto_map_demand_buses=True. "
            "Install geopandas or provide ac_bus_for_model and heat_bus_for_model "
            "directly in the CSV."
        ) from exc

    gpkg = settings.get("mv_grid_districts_gpkg")
    if gpkg is None:
        raise ValueError(
            "args['biogas_sh']['mv_grid_districts_gpkg'] is required when "
            "auto_map_demand_buses=True."
        )

    gpkg = Path(gpkg).expanduser()
    if not gpkg.exists():
        raise FileNotFoundError(f"MV grid districts GeoPackage not found: {gpkg}")


    layer = settings.get("mv_grid_layer")
    districts = gpd.read_file(gpkg) if not layer else gpd.read_file(gpkg, layer=layer)

    if districts.empty:
        raise ValueError(f"MV grid districts are empty: {gpkg}")

    districts = districts.copy()
    districts["geometry"] = districts.geometry.buffer(0)
    return districts



def _detect_column(df: pd.DataFrame, preferred: Optional[str], candidates: List[str]) -> str:
    """Return preferred column if present, otherwise first candidate present."""
    if preferred and preferred in df.columns:
        return preferred
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        "Could not detect required column. Tried: " + ", ".join([str(preferred)] + candidates)
    )



def _district_id_column(districts: pd.DataFrame, settings) -> tuple[pd.DataFrame, str]:
    """Return districts with a usable district id column."""
    district_id_col = settings.get("mv_grid_id_column")
    if district_id_col and district_id_col in districts.columns:
        return districts, district_id_col

    districts = districts.reset_index().rename(columns={"index": "_mv_grid_id"})
    return districts, "_mv_grid_id"



def _plant_points_gdf(df: pd.DataFrame, target_crs):
    """Return plant points as GeoDataFrame in target CRS."""
    import geopandas as gpd

    plants = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326",
    )
    if target_crs is not None:
        plants = plants.to_crs(target_crs)
    return plants



def _map_plants_to_districts(df, districts, settings):
    """
    Map Biogas-SH plant points to ding0 MV grid district polygons.

    The GeoPackage may have only one identifier column, e.g. 'name'.
    This function avoids duplicate column labels by renaming district
    columns before the spatial join.
    """

    import geopandas as gpd

    def _find_col(dataframe, candidates, label):
        """Return first matching column name from candidates."""
        for col in candidates:
            if col and col in dataframe.columns:
                return col
        raise ValueError(
            f"Could not detect {label}. Tried: {candidates}. "
            f"Available columns are: {list(dataframe.columns)}"
        )

    df = df.copy()
    districts = districts.copy()

    # Detect plant coordinate columns
    lon_col = _find_col(
        df,
        ["lon", "longitude", "x", "plant_lon", "bus_x"],
        "plant longitude column",
    )

    lat_col = _find_col(
        df,
        ["lat", "latitude", "y", "plant_lat", "bus_y"],
        "plant latitude column",
    )

    # Use configured MV grid ID column, normally 'name'
    id_col = settings.get("mv_grid_id_column") or "name"

    if id_col not in districts.columns:
        raise ValueError(
            f"Configured mv_grid_id_column='{id_col}' not found in "
            f"MV grid districts. Available columns are: "
            f"{list(districts.columns)}"
        )

    # Bus column may be the same as ID column, e.g. both are 'name'
    bus_col = settings.get("mv_grid_bus_column")

    if bus_col is not None and bus_col not in districts.columns:
        raise ValueError(
            f"Configured mv_grid_bus_column='{bus_col}' not found in "
            f"MV grid districts. Available columns are: "
            f"{list(districts.columns)}"
        )

    # Build clean district table with unique internal names
    districts_join = districts[[id_col, "geometry"]].copy()
    districts_join = districts_join.rename(
        columns={id_col: "__mv_grid_id__"}
    )

    if bus_col is not None:
        districts_join["__mv_grid_bus__"] = districts[bus_col].astype(str)
    else:
        districts_join["__mv_grid_bus__"] = districts_join[
            "__mv_grid_id__"
        ].astype(str)

    districts_join["__mv_grid_id__"] = districts_join[
        "__mv_grid_id__"
    ].astype(str)

    # Build plant GeoDataFrame
    plants_gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )

    # Make CRS consistent
    if districts_join.crs is None:
        districts_join = districts_join.set_crs("EPSG:4326")
    else:
        districts_join = districts_join.to_crs("EPSG:4326")

    # Spatial join: plant inside MV district
    joined = gpd.sjoin(
        plants_gdf,
        districts_join,
        how="left",
        predicate="within",
    )

    # If a plant lies exactly on a border or outside due to geometry precision,
    # use nearest MV district as fallback.
    missing = joined["__mv_grid_id__"].isna()

    if missing.any():
        nearest = gpd.sjoin_nearest(
            plants_gdf.loc[missing],
            districts_join,
            how="left",
            distance_col="__mv_grid_distance__",
        )

        joined.loc[missing, "__mv_grid_id__"] = nearest[
            "__mv_grid_id__"
        ].values
        joined.loc[missing, "__mv_grid_bus__"] = nearest[
            "__mv_grid_bus__"
        ].values

    # Store results back into the normal DataFrame
    df["ding0_mv_grid_id"] = joined["__mv_grid_id__"].astype(str).values
    df["ding0_mv_grid_bus_id"] = joined["__mv_grid_bus__"].astype(str).values

    return df



def _network_bus_points(network, bus_ids: Iterable[str], target_crs):
    """Return selected network buses as GeoDataFrame."""
    import geopandas as gpd

    ids = [str(b) for b in bus_ids if str(b) in network.buses.index]
    if not ids:
        return gpd.GeoDataFrame(columns=["bus", "geometry"], geometry="geometry", crs=target_crs)

    buses = network.buses.loc[ids].copy()
    buses["bus"] = buses.index.astype(str)
    gdf = gpd.GeoDataFrame(
        buses[["bus", "x", "y"]],
        geometry=gpd.points_from_xy(buses["x"], buses["y"]),
        crs="EPSG:4326",
    )
    if target_crs is not None:
        gdf = gdf.to_crs(target_crs)
    return gdf



def _candidate_ac_load_buses(network, settings) -> List[str]:
    """Return buses with AC load."""
    ac_carrier = settings.get("ac_load_carrier", "AC")
    candidates = set()

    if not network.loads.empty:
        loads = network.loads.copy()
        if "carrier" in loads.columns:
            candidates.update(loads.loc[loads.carrier == ac_carrier, "bus"].astype(str))

        ac_buses = network.buses.index[network.buses.carrier == "AC"].astype(str)
        candidates.update(loads.loc[loads.bus.astype(str).isin(ac_buses), "bus"].astype(str))

    return sorted([b for b in candidates if b in network.buses.index])



def _candidate_rural_heat_buses(network, settings) -> List[str]:
    """
    Return buses associated with rural heat demand.

    Primary source: bus1 of rural_heat_pump links.
    Fallbacks: loads/carrier or bus carrier containing rural_heat.
    """
    heat_carrier = settings.get("heat_demand_carrier", "rural_heat")
    heat_pump_carrier = settings.get("rural_heat_pump_carrier", "rural_heat_pump")

    candidates = set()

    if hasattr(network, "links") and not network.links.empty:
        hp = network.links[network.links.carrier.astype(str) == heat_pump_carrier]
        if not hp.empty and "bus1" in hp.columns:
            candidates.update(hp.bus1.astype(str))

    if not network.loads.empty:
        loads = network.loads.copy()
        if "carrier" in loads.columns:
            candidates.update(
                loads.loc[
                    loads.carrier.astype(str).str.contains(heat_carrier, na=False),
                    "bus",
                ].astype(str)
            )

        rural_heat_buses = network.buses.index[
            network.buses.carrier.astype(str).str.contains(heat_carrier, na=False)
        ].astype(str)
        candidates.update(loads.loc[loads.bus.astype(str).isin(rural_heat_buses), "bus"].astype(str))

    return sorted([b for b in candidates if b in network.buses.index])



def _load_energy_by_bus(network, buses: Iterable[str], carrier_contains: Optional[str] = None) -> Dict[str, float]:
    """Approximate load energy by bus; used to choose among multiple demand buses."""
    buses = [str(b) for b in buses]
    energy = {b: 0.0 for b in buses}

    if network.loads.empty:
        return energy

    loads = network.loads.copy()
    if carrier_contains and "carrier" in loads.columns:
        loads = loads[loads.carrier.astype(str).str.contains(carrier_contains, na=False)]

    for bus in buses:
        load_ids = loads.index[loads.bus.astype(str) == bus]
        if len(load_ids) == 0:
            continue

        if hasattr(network, "loads_t") and hasattr(network.loads_t, "p_set") and not network.loads_t.p_set.empty:
            existing = [l for l in load_ids if l in network.loads_t.p_set.columns]
            if existing:
                try:
                    weights = network.snapshot_weightings.generators.loc[network.snapshots]
                    energy[bus] = float(network.loads_t.p_set[existing].sum(axis=1).mul(weights, axis=0).sum())
                except Exception:
                    energy[bus] = float(network.loads_t.p_set[existing].sum().sum())
            else:
                energy[bus] = float(len(load_ids))
        else:
            energy[bus] = float(len(load_ids))

    return energy



def _map_candidate_buses_to_districts(network, districts, candidate_buses: List[str], settings) -> pd.DataFrame:
    """
    Assign candidate demand buses to MV districts.

    This version avoids duplicate column names like 'name' / 'name_right'
    by renaming the MV district id column before spatial joins.
    """
    import geopandas as gpd
    import pandas as pd

    if not candidate_buses:
        return pd.DataFrame(columns=["bus", "ding0_mv_grid_id"])

    districts, district_id_col = _district_id_column(districts, settings)

    if district_id_col not in districts.columns:
        raise ValueError(
            f"District id column '{district_id_col}' not found. "
            f"Available columns: {list(districts.columns)}"
        )

    # Rename district id to an internal unique name before joining.
    districts_join = districts[[district_id_col, "geometry"]].copy()
    districts_join = districts_join.rename(
        columns={district_id_col: "__mv_grid_id__"}
    )
    districts_join["__mv_grid_id__"] = districts_join[
        "__mv_grid_id__"
    ].astype(str)

    bus_points = _network_bus_points(network, candidate_buses, districts_join.crs)

    if bus_points.empty:
        return pd.DataFrame(columns=["bus", "ding0_mv_grid_id"])

    # First try: bus point inside MV district polygon.
    joined = gpd.sjoin(
        bus_points,
        districts_join[["__mv_grid_id__", "geometry"]],
        how="left",
        predicate="within",
    )

    # Fallback for buses on boundaries or just outside polygons.
    missing_mask = joined["__mv_grid_id__"].isna()

    if missing_mask.any():
        missing_index = joined.index[missing_mask]

        # Important: use the original bus_points, not the already joined frame.
        # Otherwise columns like 'name' get duplicated/renamed by GeoPandas.
        nearest = gpd.sjoin_nearest(
            bus_points.loc[missing_index],
            districts_join[["__mv_grid_id__", "geometry"]],
            how="left",
            distance_col="_bus_join_distance",
        )

        joined.loc[missing_index, "__mv_grid_id__"] = nearest[
            "__mv_grid_id__"
        ].values

    out = pd.DataFrame(
        {
            "bus": joined["bus"].astype(str).values,
            "ding0_mv_grid_id": joined["__mv_grid_id__"].astype(str).values,
        }
    )

    # Avoid duplicate bus rows if a point hits multiple polygons.
    out = out.dropna(subset=["ding0_mv_grid_id"])
    out = out.drop_duplicates(subset=["bus"])

    return out



def _choose_bus_for_district(
    plant_district_id,
    candidate_map: pd.DataFrame,
    districts,
    demand_energy: Dict[str, float],
    settings,
) -> Optional[str]:
    """Choose same-district demand bus; fallback to nearest candidate district."""
    if candidate_map.empty:
        return None

    same = candidate_map[candidate_map["ding0_mv_grid_id"] == plant_district_id]
    if not same.empty:
        return max(same.bus.astype(str), key=lambda b: demand_energy.get(str(b), 0.0))

    if not _as_bool(settings.get("fallback_to_neighbor_area", True), True):
        return None

    # Fallback: choose largest-load candidate bus if geometry lookup fails.
    districts, district_id_col = _district_id_column(districts, settings)
    plant_poly = districts.loc[districts[district_id_col] == plant_district_id]
    if plant_poly.empty:
        return max(candidate_map.bus.astype(str), key=lambda b: demand_energy.get(str(b), 0.0))

    plant_centroid = plant_poly.geometry.iloc[0].centroid
    candidate_districts = candidate_map.drop_duplicates("ding0_mv_grid_id").merge(
        districts[[district_id_col, "geometry"]],
        left_on="ding0_mv_grid_id",
        right_on=district_id_col,
        how="left",
    )
    candidate_districts = candidate_districts.dropna(subset=["geometry"])
    if candidate_districts.empty:
        return max(candidate_map.bus.astype(str), key=lambda b: demand_energy.get(str(b), 0.0))

    candidate_districts["_distance"] = candidate_districts.geometry.centroid.distance(plant_centroid)
    nearest_district = candidate_districts.sort_values("_distance").iloc[0]["ding0_mv_grid_id"]
    nearest_buses = candidate_map[candidate_map["ding0_mv_grid_id"] == nearest_district].bus.astype(str)
    return max(nearest_buses, key=lambda b: demand_energy.get(str(b), 0.0))



def _auto_map_demand_buses(self, df: pd.DataFrame, settings) -> pd.DataFrame:
    """Add ac_bus_for_model and heat_bus_for_model from MV districts and demand buses."""
    districts = _read_mv_grid_districts(settings)
    df = _map_plants_to_districts(df, districts, settings)

    ac_candidates = _candidate_ac_load_buses(self.network, settings)
    heat_candidates = _candidate_rural_heat_buses(self.network, settings)

    ac_map = _map_candidate_buses_to_districts(self.network, districts, ac_candidates, settings)
    heat_map = _map_candidate_buses_to_districts(self.network, districts, heat_candidates, settings)

    ac_energy = _load_energy_by_bus(self.network, ac_candidates, carrier_contains=settings.get("ac_load_carrier", "AC"))
    heat_energy = _load_energy_by_bus(self.network, heat_candidates, carrier_contains=settings.get("heat_demand_carrier", "rural_heat"))

    ac_buses = []
    heat_buses = []
    for _, row in df.iterrows():
        district_id = row.get("ding0_mv_grid_id")
        ac_buses.append(_choose_bus_for_district(district_id, ac_map, districts, ac_energy, settings))
        heat_buses.append(_choose_bus_for_district(district_id, heat_map, districts, heat_energy, settings))

    if "ac_bus_for_model" not in df.columns:
        df["ac_bus_for_model"] = ac_buses
    else:
        df["ac_bus_for_model"] = [
            _clean_id(existing) if not _is_missing(existing) else auto
            for existing, auto in zip(df["ac_bus_for_model"], ac_buses)
        ]

    if "heat_bus_for_model" not in df.columns:
        df["heat_bus_for_model"] = heat_buses
    else:
        df["heat_bus_for_model"] = [
            _clean_id(existing) if not _is_missing(existing) else auto
            for existing, auto in zip(df["heat_bus_for_model"], heat_buses)
        ]

    return df


# =============================================================================
# Capacity and plant category helpers
# =============================================================================


def _plant_has_heat(row, settings) -> bool:
    """Return True for AC+heat plants."""
    if "plant_category" in row.index and not _is_missing(row["plant_category"]):
        return str(row["plant_category"]).strip().lower() in {
            "ac_heat",
            "heat",
            "chp",
            "central_biomass_chp",
        }
    return _safe_float(row, "useful_heat_mwh_a", 0.0) > float(settings.get("heat_threshold_mwh_a", 0.0))



def _get_electric_p_nom(row, settings) -> float:
    """Return electricity p_nom in MW_el."""
    p_nom_column = settings.get("electric_capacity_column", "hbl_95_mwel")
    p_nom = _safe_float(row, p_nom_column, default=np.nan)
    if not np.isfinite(p_nom) or p_nom <= 0:
        p_nom = _safe_float(row, "installed_electric_capacity_mwel", default=0.0)
    return max(float(p_nom), 0.0)



def _get_heat_p_nom(row, settings) -> float:
    """Return heat p_nom in MW_th."""
    if "p_nom_heat_mw" in row.index and not _is_missing(row["p_nom_heat_mw"]):
        return max(_safe_float(row, "p_nom_heat_mw"), 0.0)

    method = settings.get("heat_capacity_method", "annual_heat_div_8760")
    if method != "annual_heat_div_8760":
        raise ValueError(f"Unsupported heat_capacity_method: {method}")

    heat_mwh_a = _safe_float(row, "useful_heat_mwh_a", default=0.0)
    hours = float(settings.get("hours_for_heat_capacity", 8760.0))
    return max(heat_mwh_a / hours, 0.0) if hours > 0 else 0.0



def _get_ch4_p_nom(row, settings) -> float:
    """Return CH4_biogas p_nom in MW_CH4."""
    if "p_nom_ch4_mw" in row.index and not _is_missing(row["p_nom_ch4_mw"]):
        return max(_safe_float(row, "p_nom_ch4_mw"), 0.0)

    biomethane_mwh_a = _safe_float(row, "biomethane_mwh_hs_a_at_96pct", default=0.0)
    hours = float(settings.get("hours_for_gas_capacity", 8760.0))
    return max(biomethane_mwh_a / hours, 0.0) if hours > 0 else 0.0


# =============================================================================
# Asset creation
# =============================================================================


def _add_electricity_generator(
    network,
    plant_id,
    ac_bus,
    carrier,
    p_nom,
    mc,
    tranche="market",
) -> str:
    """Add plant electricity generator."""

    if tranche == "market":
        # Preserve old name for backwards compatibility.
        name = f"biogas_sh_el_{plant_id}"
    elif tranche == "supported":
        name = f"biogas_sh_el_supported_{plant_id}"
    else:
        raise ValueError(
            f"Unsupported Biogas.SH electricity tranche: {tranche}"
        )

    _remove_component_if_exists(
        network,
        "Generator",
        name,
    )

    network.add(
        "Generator",
        name,
        bus=ac_bus,
        carrier=carrier,
        p_nom=p_nom,
        p_nom_extendable=False,
        p_nom_min=0.0,
        p_min_pu=0.0,
        p_max_pu=1.0,
        marginal_cost=mc,
        capital_cost=0.0,
        efficiency=1.0,
        build_year=0,
        lifetime=np.inf,
        committable=False,
    )

    return name



def _add_heat_generator(network, plant_id: int, heat_bus: str, carrier: str, p_nom: float, mc: float) -> None:
    """Add plant heat generator."""
    name = f"biogas_sh_heat_{plant_id}"
    _remove_component_if_exists(network, "Generator", name)
    network.add(
        "Generator",
        name,
        bus=heat_bus,
        carrier=carrier,
        p_nom=p_nom,
        p_nom_extendable=False,
        p_nom_min=0.0,
        p_min_pu=0.0,
        p_max_pu=1.0,
        marginal_cost=mc,
        capital_cost=0.0,
        efficiency=1.0,
        build_year=0,
        lifetime=np.inf,
        committable=False,
    )



def _add_ch4_generator_direct(network, plant_id: int, ch4_bus: str, p_nom: float, mc: float) -> None:
    """Add CH4_biogas generator directly at target CH4 bus."""
    name = f"biogas_sh_ch4_{plant_id}"
    _remove_component_if_exists(network, "Generator", name)
    network.add(
        "Generator",
        name,
        bus=ch4_bus,
        carrier="CH4_biogas",
        p_nom=p_nom,
        p_nom_extendable=False,
        p_nom_min=0.0,
        p_min_pu=0.0,
        p_max_pu=1.0,
        marginal_cost=mc,
        capital_cost=0.0,
        efficiency=1.0,
        build_year=0,
        lifetime=np.inf,
        committable=False,
    )



def _add_ch4_generator_with_link(network, row, plant_id: int, target_ch4_bus: str, p_nom: float, mc: float, settings) -> None:
    """Add plant CH4 bus, CH4_biogas generator, and CH4 link to target bus."""
    plant_bus = f"biogas_sh_ch4_bus_{plant_id}"
    gen_name = f"biogas_sh_ch4_{plant_id}"
    link_name = f"biogas_sh_ch4_link_{plant_id}_to_{target_ch4_bus}"

    lon = _safe_float(row, "lon")
    lat = _safe_float(row, "lat")

    _remove_component_if_exists(network, "Generator", gen_name)
    _remove_component_if_exists(network, "Link", link_name)

    if plant_bus in network.buses.index:
        # Keep bus but update coordinates/carrier if needed.
        network.buses.loc[plant_bus, "carrier"] = "CH4"
        network.buses.loc[plant_bus, "country"] = "DE"
        network.buses.loc[plant_bus, "x"] = lon
        network.buses.loc[plant_bus, "y"] = lat
    else:
        network.add(
            "Bus",
            plant_bus,
            carrier="CH4",
            country="DE",
            x=lon,
            y=lat,
        )

    network.add(
        "Generator",
        gen_name,
        bus=plant_bus,
        carrier="CH4_biogas",
        p_nom=p_nom,
        p_nom_extendable=False,
        p_nom_min=0.0,
        p_min_pu=0.0,
        p_max_pu=1.0,
        marginal_cost=mc,
        capital_cost=0.0,
        efficiency=1.0,
        build_year=0,
        lifetime=np.inf,
        committable=False,
    )

    link_factor = float(settings.get("ch4_link_p_nom_factor", 1.0))
    link_p_nom = max(p_nom * link_factor, 0.0)

    p_min_pu = float(settings.get("ch4_link_p_min_pu", 0.0))  # one-way injection by default
    efficiency = float(settings.get("ch4_link_efficiency", 1.0))
    link_mc = float(settings.get("ch4_link_marginal_cost", 0.0))
    link_capital_cost = float(settings.get("ch4_link_capital_cost", 0.0))

    network.add(
        "Link",
        link_name,
        bus0=plant_bus,
        bus1=target_ch4_bus,
        carrier=settings.get("ch4_link_carrier", "biogas_sh_gas_grid_injection"),
        p_nom=link_p_nom,
        p_nom_extendable=_as_bool(settings.get("ch4_link_extendable", False), False),
        p_nom_min=0.0,
        p_min_pu=p_min_pu,
        p_max_pu=1.0,
        efficiency=efficiency,
        marginal_cost=link_mc,
        capital_cost=link_capital_cost,
    )

    # Optional extra metadata if the target bus has coordinates.
    try:
        tx = float(network.buses.loc[target_ch4_bus, "x"])
        ty = float(network.buses.loc[target_ch4_bus, "y"])
        network.links.loc[link_name, "length"] = _distance_km(lon, lat, tx, ty)
    except Exception:
        pass



# =============================================================================
# Flexible gas route and SWFL-direct helpers
# =============================================================================


def _resolve_biogas_sh_route_switches(
    settings,
):
    """
    Resolve the active Biogas.SH physical routes.

    Returns
    -------
    tuple
        (
            mode,
            add_local_generation,
            add_gas_grid_generation,
            add_swfl_biomethane_supply,
            add_swfl_raw_biogas_supply,
        )

    Route meanings
    --------------
    onsite
        Raw biogas -> onsite electricity / heat only.

    gas_grid
        Raw biogas -> upgrading -> biomethane -> public gas grid.

    swfl
        Legacy route:
        raw biogas -> upgrading -> biomethane -> SWFL.

    raw_swfl
        Raw biogas -> direct delivery -> SWFL without upgrading.

    hybrid
        Legacy production topology:
        onsite + public-grid biomethane + SWFL biomethane.

    hybrid_raw_swfl
        New production topology:
        onsite + public-grid biomethane + direct raw biogas -> SWFL.

    custom
        Read all route switches explicitly from settings.
    """

    mode = str(
        settings.get(
            "scenario_mode",
            "custom",
        )
    ).strip().lower()


    # ------------------------------------------------------------------
    # Legacy predefined modes
    # ------------------------------------------------------------------

    if mode == "onsite":

        return (
            mode,
            True,   # local
            False,  # public grid
            False,  # biomethane -> SWFL
            False,  # raw biogas -> SWFL
        )


    if mode in {
        "gas_grid",
        "grid",
    }:

        return (
            mode,
            False,
            True,
            False,
            False,
        )


    if mode in {
        "swfl",
        "swfl_direct",
    }:

        # Preserve historical meaning:
        # upgraded biomethane -> SWFL.
        return (
            mode,
            False,
            False,
            True,
            False,
        )


    if mode in {
        "hybrid",
        "all",
        "competition",
    }:

        # Preserve historical hybrid definition.
        return (
            mode,
            True,
            True,
            True,
            False,
        )


    # ------------------------------------------------------------------
    # New raw-biogas modes
    # ------------------------------------------------------------------

    if mode in {
        "raw_swfl",
        "swfl_raw",
    }:

        return (
            mode,
            False,
            False,
            False,
            True,
        )


    if mode == "hybrid_raw_swfl":

        return (
            mode,
            True,
            True,
            False,
            True,
        )


    # ------------------------------------------------------------------
    # Explicit custom configuration
    # ------------------------------------------------------------------

    if mode != "custom":

        raise ValueError(
            "Unsupported biogas_sh.scenario_mode. "
            "Use 'custom', 'onsite', 'gas_grid', 'swfl', "
            "'raw_swfl', 'hybrid', or 'hybrid_raw_swfl'."
        )


    add_local = _as_bool(
        settings.get(
            "add_local_generation",
            True,
        ),
        True,
    )


    add_grid = _as_bool(
        settings.get(
            "add_gas_grid_generation",
            settings.get(
                "add_gas_generation",
                True,
            ),
        ),
        True,
    )


    swfl_cfg = (
        settings.get(
            "swfl_direct",
            {},
        )
        or {}
    )


    add_swfl_biomethane = _as_bool(
        settings.get(
            "add_swfl_direct_supply",
            swfl_cfg.get(
                "active",
                False,
            ),
        ),
        False,
    )


    add_swfl_raw_biogas = _as_bool(
        settings.get(
            "add_swfl_raw_biogas_supply",
            False,
        ),
        False,
    )


    return (
        mode,
        add_local,
        add_grid,
        add_swfl_biomethane,
        add_swfl_raw_biogas,
    )

def _swfl_direct_settings(settings) -> dict:
    cfg = settings.get("swfl_direct", {})
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise TypeError("biogas_sh.swfl_direct must be a dictionary if provided.")
    return cfg


def _consumer_link_ids_from_settings(swfl_cfg: dict) -> List[str]:
    ids = swfl_cfg.get("consumer_link_ids", []) or []
    return [str(_clean_id(i)) for i in ids if _clean_id(i) is not None]


def _infer_swfl_bus_coordinates(network, swfl_cfg: dict, public_ch4_bus: str, consumer_link_ids: List[str]):
    if "swfl_ch4_bus_x" in swfl_cfg and "swfl_ch4_bus_y" in swfl_cfg:
        return float(swfl_cfg["swfl_ch4_bus_x"]), float(swfl_cfg["swfl_ch4_bus_y"])
    xs, ys = [], []
    for link_id in consumer_link_ids:
        if link_id not in network.links.index:
            continue
        bus1 = str(network.links.loc[link_id, "bus1"])
        if bus1 in network.buses.index:
            xs.append(float(network.buses.loc[bus1, "x"]))
            ys.append(float(network.buses.loc[bus1, "y"]))
    if xs and ys:
        return float(np.mean(xs)), float(np.mean(ys))
    if public_ch4_bus in network.buses.index:
        return float(network.buses.loc[public_ch4_bus, "x"]), float(network.buses.loc[public_ch4_bus, "y"])
    return 9.436502119171873, 54.79233181101448


def _ensure_swfl_ch4_bus_and_access(network, settings):
    """Create artificial SWFL CH4 bus and optional public-grid supply link."""
    swfl_cfg = _swfl_direct_settings(settings)
    public_ch4_bus = _clean_id(swfl_cfg.get("public_ch4_bus", settings.get("target_ch4_bus", "47538")))
    swfl_bus = _clean_id(swfl_cfg.get("swfl_ch4_bus", "biogas_sh_swfl_ch4_bus")) or "biogas_sh_swfl_ch4_bus"
    if public_ch4_bus not in network.buses.index:
        raise ValueError(f"SWFL public_ch4_bus {public_ch4_bus} not found in network.buses.")
    consumer_link_ids = _consumer_link_ids_from_settings(swfl_cfg)
    x, y = _infer_swfl_bus_coordinates(network, swfl_cfg, public_ch4_bus, consumer_link_ids)
    _ensure_carrier(network, "CH4")
    _ensure_carrier(network, "biogas_sh_swfl_direct")
    _ensure_carrier(network, "biogas_sh_swfl_grid_supply")
    if swfl_bus in network.buses.index:
        network.buses.loc[swfl_bus, "carrier"] = "CH4"
        network.buses.loc[swfl_bus, "country"] = "DE"
        network.buses.loc[swfl_bus, "x"] = x
        network.buses.loc[swfl_bus, "y"] = y
    else:
        network.add("Bus", swfl_bus, carrier="CH4", country="DE", x=x, y=y)
    redirected = []
    if _as_bool(swfl_cfg.get("redirect_consumer_links", True), True):
        for link_id in consumer_link_ids:
            if link_id not in network.links.index:
                logger.warning("SWFL consumer link %s not found; cannot redirect to %s.", link_id, swfl_bus)
                continue
            old_bus0 = str(network.links.loc[link_id, "bus0"])
            if old_bus0 == public_ch4_bus or _as_bool(swfl_cfg.get("redirect_even_if_bus0_differs", False), False):
                network.links.loc[link_id, "bus0"] = swfl_bus
                redirected.append(link_id)
    grid_supply_added = False
    if _as_bool(swfl_cfg.get("grid_supply_active", True), True):
        grid_supply_name = _clean_id(swfl_cfg.get("grid_supply_link", f"biogas_sh_swfl_grid_supply_{public_ch4_bus}_to_{swfl_bus}"))
        _remove_component_if_exists(network, "Link", grid_supply_name)
        if "grid_supply_p_nom" in swfl_cfg and not _is_missing(swfl_cfg["grid_supply_p_nom"]):
            grid_p_nom = float(swfl_cfg["grid_supply_p_nom"])
        else:
            p_nom_sum = 0.0
            for link_id in consumer_link_ids:
                if link_id in network.links.index and "p_nom" in network.links.columns:
                    try:
                        p_nom_sum += float(network.links.loc[link_id, "p_nom"])
                    except Exception:
                        pass
            grid_p_nom = p_nom_sum if p_nom_sum > 0 else float(swfl_cfg.get("grid_supply_p_nom_default", 1e6))
            grid_p_nom *= float(swfl_cfg.get("grid_supply_p_nom_factor", 1.0))
        network.add(
            "Link", grid_supply_name, bus0=public_ch4_bus, bus1=swfl_bus,
            carrier="biogas_sh_swfl_grid_supply", p_nom=grid_p_nom,
            p_nom_extendable=_as_bool(swfl_cfg.get("grid_supply_extendable", False), False),
            p_nom_min=0.0, p_min_pu=0.0, p_max_pu=1.0,
            efficiency=float(swfl_cfg.get("grid_supply_efficiency", 1.0)),
            marginal_cost=float(swfl_cfg.get("grid_supply_marginal_cost", 0.0)),
            capital_cost=float(swfl_cfg.get("grid_supply_capital_cost", 0.0)),
        )
        grid_supply_added = True
    if _as_bool(swfl_cfg.get("add_swfl_gas_load", False), False):
        load_name = _clean_id(swfl_cfg.get("swfl_gas_load", "biogas_sh_swfl_ch4_load"))
        _remove_component_if_exists(network, "Load", load_name)
        mwh_a = float(swfl_cfg.get("swfl_demand_mwh_a", 0.0))
        flat_mw = mwh_a / 8760.0 if mwh_a > 0 else 0.0
        network.add("Load", load_name, bus=swfl_bus, carrier="CH4", p_set=flat_mw)
    return swfl_bus, public_ch4_bus, redirected, grid_supply_added


def _ensure_plant_ch4_bus_and_generator(network, row, plant_id: int, p_nom: float, mc: float):
    plant_bus = f"biogas_sh_ch4_bus_{plant_id}"
    gen_name = f"biogas_sh_ch4_{plant_id}"
    lon = _safe_float(row, "lon")
    lat = _safe_float(row, "lat")
    _remove_component_if_exists(network, "Generator", gen_name)
    if plant_bus in network.buses.index:
        network.buses.loc[plant_bus, "carrier"] = "CH4"
        network.buses.loc[plant_bus, "country"] = "DE"
        network.buses.loc[plant_bus, "x"] = lon
        network.buses.loc[plant_bus, "y"] = lat
    else:
        network.add("Bus", plant_bus, carrier="CH4", country="DE", x=lon, y=lat)
    network.add(
        "Generator", gen_name, bus=plant_bus, carrier="CH4_biogas", p_nom=p_nom,
        p_nom_extendable=False, p_nom_min=0.0, p_min_pu=0.0, p_max_pu=1.0,
        marginal_cost=mc, capital_cost=0.0, efficiency=1.0, build_year=0,
        lifetime=np.inf, committable=False,
    )
    return plant_bus, gen_name


def _add_ch4_grid_link_from_plant(
    network,
    plant_id: int,
    plant_bus: str,
    target_ch4_bus: str,
    p_nom: float,
    settings,
) -> str:
    """
    Add a one-way link from a Biogas.SH plant CH4 bus to the public CH4 grid.

    This function is only intended for direct public-grid injection:

        plant CH4 bus -> public CH4 grid
    """
    plant_bus = str(plant_bus)
    target_ch4_bus = str(target_ch4_bus)

    if plant_bus not in network.buses.index:
        raise ValueError(
            f"Biogas.SH plant CH4 bus {plant_bus!r} does not exist."
        )

    if target_ch4_bus not in network.buses.index:
        raise ValueError(
            f"Public CH4 target bus {target_ch4_bus!r} does not exist."
        )

    link_name = (
        f"biogas_sh_ch4_grid_link_"
        f"{plant_id}_to_{target_ch4_bus}"
    )

    link_carrier = str(
        settings.get(
            "ch4_link_carrier",
            "biogas_sh_gas_grid_injection",
        )
    )

    _ensure_carrier(
        network,
        link_carrier,
    )

    _remove_component_if_exists(
        network,
        "Link",
        link_name,
    )

    link_factor = float(
        settings.get(
            "ch4_link_p_nom_factor",
            1.0,
        )
    )

    link_p_nom = max(
        float(p_nom) * link_factor,
        0.0,
    )

    network.add(
        "Link",
        link_name,
        bus0=plant_bus,
        bus1=target_ch4_bus,
        carrier=link_carrier,
        p_nom=link_p_nom,
        p_nom_extendable=_as_bool(
            settings.get(
                "ch4_link_extendable",
                False,
            ),
            False,
        ),
        p_nom_min=float(
            settings.get(
                "ch4_link_p_nom_min",
                0.0,
            )
        ),
        p_min_pu=float(
            settings.get(
                "ch4_link_p_min_pu",
                0.0,
            )
        ),
        p_max_pu=float(
            settings.get(
                "ch4_link_p_max_pu",
                1.0,
            )
        ),
        efficiency=float(
            settings.get(
                "ch4_link_efficiency",
                1.0,
            )
        ),
        marginal_cost=float(
            settings.get(
                "ch4_link_marginal_cost",
                0.0,
            )
        ),
        capital_cost=float(
            settings.get(
                "ch4_link_capital_cost",
                0.0,
            )
        ),
    )

    return link_name

def _add_ch4_storage_input_link_from_plant(
    network,
    plant_id: int,
    plant_bus: str,
    storage_bus: str,
    p_nom: float,
    settings,
) -> str:
    """
    Add a one-way collection link from a Biogas.SH plant CH4 bus
    to the central Biogas.SH storage bus.

    Topology:

        plant-specific CH4 bus
                  |
                  v
        central Biogas.SH storage bus

    This link is not a public gas-grid injection link. It represents
    collection and transport of biomethane from a plant to the common
    Biogas.SH storage.
    """
    cfg = _biogas_sh_storage_settings(settings)

    plant_bus = str(plant_bus)
    storage_bus = str(storage_bus)

    if plant_bus not in network.buses.index:
        raise ValueError(
            f"Biogas.SH plant CH4 bus {plant_bus!r} does not exist."
        )

    if storage_bus not in network.buses.index:
        raise ValueError(
            f"Biogas.SH storage bus {storage_bus!r} does not exist."
        )

    name_prefix = str(
        cfg.get(
            "input_link_name_prefix",
            "biogas_sh_storage_input",
        )
    ).strip()

    if not name_prefix:
        raise ValueError(
            "biogas_sh.gas_storage.input_link_name_prefix "
            "must not be empty."
        )

    link_name = (
        f"{name_prefix}_{plant_id}_to_{storage_bus}"
    )

    link_carrier = str(
        cfg.get(
            "input_link_carrier",
            "biogas_sh_collection_to_storage",
        )
    ).strip()

    if not link_carrier:
        raise ValueError(
            "biogas_sh.gas_storage.input_link_carrier "
            "must not be empty."
        )

    _ensure_carrier(
        network,
        link_carrier,
    )

    _remove_component_if_exists(
        network,
        "Link",
        link_name,
    )

    # Normally one MW of plant biomethane-production capacity receives
    # one MW of transport capacity to the storage.
    link_factor = float(
        cfg.get(
            "input_link_p_nom_factor",
            1.0,
        )
    )

    if link_factor < 0:
        raise ValueError(
            "biogas_sh.gas_storage.input_link_p_nom_factor "
            "must be non-negative."
        )

    link_p_nom = max(
        float(p_nom) * link_factor,
        0.0,
    )

    efficiency = float(
        cfg.get(
            "input_link_efficiency",
            1.0,
        )
    )

    if efficiency <= 0:
        raise ValueError(
            "biogas_sh.gas_storage.input_link_efficiency "
            "must be greater than zero."
        )

    p_min_pu = float(
        cfg.get(
            "input_link_p_min_pu",
            0.0,
        )
    )

    p_max_pu = float(
        cfg.get(
            "input_link_p_max_pu",
            1.0,
        )
    )

    if p_min_pu > p_max_pu:
        raise ValueError(
            "biogas_sh.gas_storage.input_link_p_min_pu "
            "must not exceed input_link_p_max_pu."
        )

    network.add(
        "Link",
        link_name,
        bus0=plant_bus,
        bus1=storage_bus,
        carrier=link_carrier,

        # PyPSA Link p_nom is the input-side gas-flow capacity.
        p_nom=link_p_nom,

        p_nom_extendable=_as_bool(
            cfg.get(
                "input_link_extendable",
                False,
            ),
            False,
        ),
        p_nom_min=float(
            cfg.get(
                "input_link_p_nom_min",
                0.0,
            )
        ),

        # One-way plant-to-storage flow by default.
        p_min_pu=p_min_pu,
        p_max_pu=p_max_pu,

        efficiency=efficiency,

        marginal_cost=float(
            cfg.get(
                "input_link_marginal_cost",
                0.0,
            )
        ),
        capital_cost=float(
            cfg.get(
                "input_link_capital_cost",
                0.0,
            )
        ),
    )

    # Optional metadata for easier reporting.
    try:
        network.links.loc[
            link_name,
            "biogas_sh_route",
        ] = "plant_to_storage"

    except Exception:
        pass

    return link_name


def _add_swfl_direct_link_from_plant(network, plant_id: int, plant_bus: str, swfl_bus: str, p_nom: float, settings) -> None:
    swfl_cfg = _swfl_direct_settings(settings)
    link_name = f"biogas_sh_swfl_direct_link_{plant_id}_to_{swfl_bus}"
    _remove_component_if_exists(network, "Link", link_name)
    link_p_nom = max(p_nom * float(swfl_cfg.get("direct_link_p_nom_factor", 1.0)), 0.0)
    network.add(
        "Link", link_name, bus0=plant_bus, bus1=swfl_bus,
        carrier="biogas_sh_swfl_direct", p_nom=link_p_nom,
        p_nom_extendable=_as_bool(swfl_cfg.get("direct_link_extendable", False), False),
        p_nom_min=0.0, p_min_pu=0.0, p_max_pu=1.0,
        efficiency=float(swfl_cfg.get("direct_link_efficiency", 1.0)),
        marginal_cost=float(swfl_cfg.get("direct_link_marginal_cost", swfl_cfg.get("swfl_direct_transport_cost", 0.0))),
        capital_cost=float(swfl_cfg.get("direct_link_capital_cost", 0.0)),
    )


def _biogas_sh_storage_settings(settings) -> dict:
    """Read optional Biogas.SH gas-storage settings."""
    cfg = settings.get("gas_storage", {})
    if cfg is None:
        return {}
    if not isinstance(cfg, dict):
        raise TypeError("biogas_sh.gas_storage must be a dictionary.")
    return cfg


def _ensure_biogas_sh_single_storage(
    network,
    settings,
) -> tuple[Optional[str], bool]:
    """
    Add or update one central Biogas.SH CH4 storage.

    Topology
    --------
        Biogas.SH plant CH4 buses
                    |
                    v
        biogas_sh_storage_ch4_bus
                    |
                    <-> biogas_sh_ch4_store
                    |
                    +----> public CH4 grid
                    |
                    +----> dedicated SWFL biomethane bus

    Returns
    -------
    tuple[Optional[str], bool]
        storage_bus:
            Name of the storage bus when storage is active, otherwise None.

        storage_active:
            True when the storage was created or updated, otherwise False.
    """
    cfg = _biogas_sh_storage_settings(settings)

    # ------------------------------------------------------------------
    # 1. Activation check
    # ------------------------------------------------------------------
    if not _as_bool(
        cfg.get("active", False),
        False,
    ):
        return None, False

    # ------------------------------------------------------------------
    # 2. Component names and location
    # ------------------------------------------------------------------
    storage_bus = str(
        cfg.get(
            "bus",
            "biogas_sh_storage_ch4_bus",
        )
    ).strip()

    storage_name = str(
        cfg.get(
            "store",
            "biogas_sh_ch4_store",
        )
    ).strip()

    if not storage_bus:
        raise ValueError(
            "biogas_sh.gas_storage.bus must not be empty."
        )

    if not storage_name:
        raise ValueError(
            "biogas_sh.gas_storage.store must not be empty."
        )

    # Approximate location near Flensburg/SWFL unless configured otherwise.
    x = float(
        cfg.get(
            "x",
            9.436502119171873,
        )
    )

    y = float(
        cfg.get(
            "y",
            54.79233181101448,
        )
    )

    country = str(
        cfg.get(
            "country",
            "DE",
        )
    )

    # ------------------------------------------------------------------
    # 3. Storage parameters
    # ------------------------------------------------------------------
    e_nom = float(
        cfg.get(
            "e_nom_mwh",
            500.0,
        )
    )

    e_nom_extendable = _as_bool(
        cfg.get(
            "e_nom_extendable",
            False,
        ),
        False,
    )

    e_nom_min = float(
        cfg.get(
            "e_nom_min",
            0.0,
        )
    )

    e_initial = float(
        cfg.get(
            "e_initial",
            0.0,
        )
    )

    e_cyclic = _as_bool(
        cfg.get(
            "e_cyclic",
            True,
        ),
        True,
    )

    standing_loss = float(
        cfg.get(
            "standing_loss",
            0.0,
        )
    )

    marginal_cost = float(
        cfg.get(
            "marginal_cost",
            0.0,
        )
    )

    capital_cost = float(
        cfg.get(
            "capital_cost",
            0.0,
        )
    )

    # ------------------------------------------------------------------
    # 4. Parameter validation
    # ------------------------------------------------------------------
    if e_nom < 0:
        raise ValueError(
            "biogas_sh.gas_storage.e_nom_mwh must be non-negative."
        )

    if e_nom_min < 0:
        raise ValueError(
            "biogas_sh.gas_storage.e_nom_min must be non-negative."
        )

    if not e_nom_extendable and e_nom_min > e_nom:
        raise ValueError(
            "For fixed storage capacity, "
            "biogas_sh.gas_storage.e_nom_min must not exceed e_nom_mwh."
        )

    if e_initial < 0:
        raise ValueError(
            "biogas_sh.gas_storage.e_initial must be non-negative."
        )

    if not e_nom_extendable and e_initial > e_nom:
        raise ValueError(
            "For fixed storage capacity, "
            "biogas_sh.gas_storage.e_initial must not exceed e_nom_mwh."
        )

    if not 0.0 <= standing_loss < 1.0:
        raise ValueError(
            "biogas_sh.gas_storage.standing_loss must be "
            "greater than or equal to 0 and smaller than 1."
        )

    # ------------------------------------------------------------------
    # 5. Ensure CH4 carrier
    # ------------------------------------------------------------------
    _ensure_carrier(
        network,
        "CH4",
    )

    # ------------------------------------------------------------------
    # 6. Add or update the storage bus
    # ------------------------------------------------------------------
    if storage_bus in network.buses.index:
        network.buses.loc[
            storage_bus,
            "carrier",
        ] = "CH4"

        network.buses.loc[
            storage_bus,
            "country",
        ] = country

        network.buses.loc[
            storage_bus,
            "x",
        ] = x

        network.buses.loc[
            storage_bus,
            "y",
        ] = y

    else:
        network.add(
            "Bus",
            storage_bus,
            carrier="CH4",
            country=country,
            x=x,
            y=y,
        )

    # ------------------------------------------------------------------
    # 7. Recreate the Store component
    # ------------------------------------------------------------------
    _remove_component_if_exists(
        network,
        "Store",
        storage_name,
    )

    network.add(
        "Store",
        storage_name,
        bus=storage_bus,
        carrier="CH4",

        # Energy capacity in MWh_CH4.
        e_nom=e_nom,
        e_nom_extendable=e_nom_extendable,
        e_nom_min=e_nom_min,

        # Initial and terminal storage conditions.
        e_initial=e_initial,
        e_cyclic=e_cyclic,

        # Storage losses and costs.
        standing_loss=standing_loss,
        marginal_cost=marginal_cost,
        capital_cost=capital_cost,
    )

    # ------------------------------------------------------------------
    # 8. Optional metadata for reporting
    # ------------------------------------------------------------------
    try:
        network.stores.loc[
            storage_name,
            "storage_type",
        ] = "biogas_sh_central_ch4_storage"

        network.stores.loc[
            storage_name,
            "configured_e_nom_mwh",
        ] = e_nom
    except Exception:
        # Metadata are optional and must not prevent model construction.
        pass

    logger.info(
        "Biogas.SH CH4 storage created or updated: "
        "store=%s, bus=%s, e_nom=%.3f MWh, extendable=%s",
        storage_name,
        storage_bus,
        e_nom,
        e_nom_extendable,
    )

    return storage_bus, True


def _add_biogas_sh_storage_output_links(
    network,
    storage_bus: str,
    settings,
    add_gas_grid_generation: bool,
    add_swfl_direct_supply: bool,
    target_ch4_bus: str = None,
    swfl_bus: str = None,
) -> None:
    """
    Add one-way output links from the central Biogas.SH storage bus.

    Routes
    ------
    1. Storage -> public CH4 grid
    2. Storage -> dedicated SWFL biomethane bus

    Natural gas and biomethane must remain on separate SWFL buses.
    """
    cfg = _biogas_sh_storage_settings(settings)

    storage_bus = str(storage_bus).strip()

    if storage_bus not in network.buses.index:
        raise ValueError(
            f"Biogas.SH storage bus {storage_bus!r} does not exist."
        )

    # =========================================================================
    # 1. Storage -> public CH4 grid
    # =========================================================================
    if add_gas_grid_generation:
        target_ch4_bus = _clean_id(target_ch4_bus)

        if target_ch4_bus is None:
            raise ValueError(
                "target_ch4_bus is required for the "
                "storage-to-public-grid route."
            )

        if target_ch4_bus not in network.buses.index:
            raise ValueError(
                f"Public CH4 target bus {target_ch4_bus!r} "
                "does not exist."
            )

        grid_link = str(
            cfg.get(
                "grid_link",
                f"biogas_sh_storage_to_grid_{target_ch4_bus}",
            )
        ).strip()

        grid_carrier = str(
            cfg.get(
                "grid_link_carrier",
                "biogas_sh_storage_to_grid",
            )
        ).strip()

        if not grid_link:
            raise ValueError(
                "biogas_sh.gas_storage.grid_link must not be empty."
            )

        if not grid_carrier:
            raise ValueError(
                "biogas_sh.gas_storage.grid_link_carrier "
                "must not be empty."
            )

        _ensure_carrier(
            network,
            grid_carrier,
        )

        _remove_component_if_exists(
            network,
            "Link",
            grid_link,
        )

        network.add(
            "Link",
            grid_link,
            bus0=storage_bus,
            bus1=target_ch4_bus,
            carrier=grid_carrier,
            p_nom=float(
                cfg.get(
                    "grid_link_p_nom_mw",
                    50.0,
                )
            ),
            p_nom_extendable=_as_bool(
                cfg.get(
                    "grid_link_extendable",
                    False,
                ),
                False,
            ),
            p_nom_min=float(
                cfg.get(
                    "grid_link_p_nom_min",
                    0.0,
                )
            ),
            p_min_pu=float(
                cfg.get(
                    "grid_link_p_min_pu",
                    0.0,
                )
            ),
            p_max_pu=float(
                cfg.get(
                    "grid_link_p_max_pu",
                    1.0,
                )
            ),
            efficiency=float(
                cfg.get(
                    "grid_link_efficiency",
                    1.0,
                )
            ),
            marginal_cost=float(
                cfg.get(
                    "grid_link_marginal_cost",
                    0.0,
                )
            ),
            capital_cost=float(
                cfg.get(
                    "grid_link_capital_cost",
                    0.0,
                )
            ),
        )

    # =========================================================================
    # 2. Storage -> dedicated SWFL biomethane bus
    # =========================================================================
    if add_swfl_direct_supply:
        configured_target = cfg.get(
            "swfl_target_bus",
            None,
        )

        if configured_target is None or _is_missing(configured_target):
            raise ValueError(
                "Biogas.SH storage-to-SWFL supply is active, but "
                "biogas_sh.gas_storage.swfl_target_bus is not set. "
                "Set it to 'swfl_real_biomethane_ch4_bus'."
            )

        swfl_target_bus = str(
            configured_target
        ).strip()

        if not swfl_target_bus:
            raise ValueError(
                "biogas_sh.gas_storage.swfl_target_bus "
                "must not be empty."
            )

        # Prevent accidental mixing of biomethane and public natural gas.
        if (
            swfl_bus is not None
            and swfl_target_bus == str(swfl_bus)
        ):
            raise ValueError(
                "The storage SWFL target bus is identical to the "
                "SWFL natural-gas bus. Set "
                "gas_storage.swfl_target_bus to "
                "'swfl_real_biomethane_ch4_bus'."
            )

        target_x = float(
            cfg.get(
                "swfl_target_bus_x",
                9.436502119171873,
            )
        )

        target_y = float(
            cfg.get(
                "swfl_target_bus_y",
                54.79233181101448,
            )
        )

        if swfl_target_bus in network.buses.index:
            network.buses.loc[
                swfl_target_bus,
                "carrier",
            ] = "CH4"

            network.buses.loc[
                swfl_target_bus,
                "country",
            ] = "DE"

            network.buses.loc[
                swfl_target_bus,
                "x",
            ] = target_x

            network.buses.loc[
                swfl_target_bus,
                "y",
            ] = target_y

        else:
            network.add(
                "Bus",
                swfl_target_bus,
                carrier="CH4",
                country="DE",
                x=target_x,
                y=target_y,
            )

        swfl_link = str(
            cfg.get(
                "swfl_link",
                (
                    "biogas_sh_storage_to_"
                    f"{swfl_target_bus}"
                ),
            )
        ).strip()

        swfl_carrier = str(
            cfg.get(
                "swfl_link_carrier",
                "biogas_sh_storage_to_swfl",
            )
        ).strip()

        if not swfl_link:
            raise ValueError(
                "biogas_sh.gas_storage.swfl_link must not be empty."
            )

        if not swfl_carrier:
            raise ValueError(
                "biogas_sh.gas_storage.swfl_link_carrier "
                "must not be empty."
            )

        _ensure_carrier(
            network,
            swfl_carrier,
        )

        _remove_component_if_exists(
            network,
            "Link",
            swfl_link,
        )

        network.add(
            "Link",
            swfl_link,
            bus0=storage_bus,
            bus1=swfl_target_bus,
            carrier=swfl_carrier,
            p_nom=float(
                cfg.get(
                    "swfl_link_p_nom_mw",
                    50.0,
                )
            ),
            p_nom_extendable=_as_bool(
                cfg.get(
                    "swfl_link_extendable",
                    False,
                ),
                False,
            ),
            p_nom_min=float(
                cfg.get(
                    "swfl_link_p_nom_min",
                    0.0,
                )
            ),
            p_min_pu=float(
                cfg.get(
                    "swfl_link_p_min_pu",
                    0.0,
                )
            ),
            p_max_pu=float(
                cfg.get(
                    "swfl_link_p_max_pu",
                    1.0,
                )
            ),
            efficiency=float(
                cfg.get(
                    "swfl_link_efficiency",
                    1.0,
                )
            ),
            marginal_cost=float(
                cfg.get(
                    "swfl_link_marginal_cost",
                    0.0,
                )
            ),
            capital_cost=float(
                cfg.get(
                    "swfl_link_capital_cost",
                    0.0,
                )
            ),
        )

        # Final topology validation.
        actual_target = str(
            network.links.at[
                swfl_link,
                "bus1",
            ]
        )

        if actual_target != swfl_target_bus:
            raise RuntimeError(
                f"Storage-to-SWFL link {swfl_link!r} has bus1="
                f"{actual_target!r}, expected {swfl_target_bus!r}."
            )

        logger.info(
            "Biogas.SH storage-to-SWFL link created: "
            "%s -> %s, carrier=%s, p_nom=%.3f MW",
            storage_bus,
            swfl_target_bus,
            swfl_carrier,
            float(network.links.at[swfl_link, "p_nom"]),
        )

# =============================================================================
# Public function attached to Etrago in network.py
# =============================================================================

def _add_raw_biogas_swfl_supply(
    network,
    df,
    settings,
) -> str:
    """
    Add one aggregate regional raw-biogas supply Generator for SWFL.

    Topology
    --------
        regional raw-biogas resource
                    |
                    v
        swfl_real_raw_biogas_bus
                    |
                    +--> K12 fuel bus
                    |
                    +--> K13 fuel bus

    Economics
    ---------
    The Generator represents the delivered raw-biogas cost:

        raw-biogas fuel cost
        + collection / transport cost

    For the current project assumptions:

        75.00 + 8.78
        = 83.78 EUR/MWh_Hs

    Important
    ---------
    - No upgrading efficiency is applied here.
    - No biomethane cost is applied here.
    - No biogenic CO2-sale credit is applied here.
    - Dispatch must additionally be counted in the shared regional
      raw-biogas resource constraint.
    """

    cfg = (
        settings.get(
            "raw_biogas_to_swfl",
            {},
        )
        or {}
    )


    if not isinstance(
        cfg,
        dict,
    ):

        raise TypeError(
            "biogas_sh.raw_biogas_to_swfl "
            "must be a dictionary."
        )


    generator_name = str(
        cfg.get(
            "generator_name",
            "biogas_sh_raw_biogas_swfl_supply",
        )
    ).strip()


    generator_carrier = str(
        cfg.get(
            "generator_carrier",
            "biogas_sh_raw_biogas_swfl",
        )
    ).strip()


    bus_carrier = str(
        cfg.get(
            "bus_carrier",
            "raw_biogas",
        )
    ).strip()


    # ------------------------------------------------------------------
    # Inactive route: remove stale component if present.
    # ------------------------------------------------------------------

    if not _as_bool(
        cfg.get(
            "active",
            False,
        ),
        False,
    ):

        _remove_component_if_exists(
            network,
            "Generator",
            generator_name,
        )

        return ""


    if not generator_name:
        raise ValueError(
            "raw_biogas_to_swfl.generator_name "
            "must not be empty."
        )


    if not generator_carrier:
        raise ValueError(
            "raw_biogas_to_swfl.generator_carrier "
            "must not be empty."
        )


    if not bus_carrier:
        raise ValueError(
            "raw_biogas_to_swfl.bus_carrier "
            "must not be empty."
        )


    # ------------------------------------------------------------------
    # Dedicated SWFL raw-biogas bus
    # ------------------------------------------------------------------

    target_bus = str(
        cfg.get(
            "target_bus",
            "swfl_real_raw_biogas_bus",
        )
    ).strip()


    if not target_bus:

        raise ValueError(
            "raw_biogas_to_swfl.target_bus "
            "must not be empty."
        )


    x = float(
        cfg.get(
            "x",
            9.436502119171873,
        )
    )

    y = float(
        cfg.get(
            "y",
            54.79233181101448,
        )
    )


    _ensure_carrier(
        network,
        bus_carrier,
    )

    _ensure_carrier(
        network,
        generator_carrier,
    )


    if target_bus in network.buses.index:

        current_carrier = str(
            network.buses.at[
                target_bus,
                "carrier",
            ]
        ).strip()


        if (
            current_carrier
            and current_carrier != bus_carrier
        ):

            raise ValueError(
                f"Raw-biogas target bus {target_bus!r} already "
                f"exists with carrier {current_carrier!r}; "
                f"expected {bus_carrier!r}."
            )


        network.buses.loc[
            target_bus,
            "carrier",
        ] = bus_carrier

        network.buses.loc[
            target_bus,
            "country",
        ] = "DE"

        network.buses.loc[
            target_bus,
            "x",
        ] = x

        network.buses.loc[
            target_bus,
            "y",
        ] = y

    else:

        network.add(
            "Bus",
            target_bus,
            carrier=bus_carrier,
            country="DE",
            x=x,
            y=y,
        )


    # ------------------------------------------------------------------
    # Total regional raw-biogas potential
    # ------------------------------------------------------------------

    if "raw_biogas_mwh_hs_a" not in df.columns:

        raise ValueError(
            "Biogas.SH plant data do not contain "
            "'raw_biogas_mwh_hs_a'."
        )


    annual_raw_biogas = float(
        pd.to_numeric(
            df[
                "raw_biogas_mwh_hs_a"
            ],
            errors="coerce",
        )
        .fillna(
            0.0
        )
        .clip(
            lower=0.0
        )
        .sum()
    )


    if annual_raw_biogas <= 0:

        raise ValueError(
            "Regional raw-biogas potential is zero or negative; "
            "cannot create direct SWFL supply."
        )


    # ------------------------------------------------------------------
    # Delivery power capacity
    #
    # Default:
    #
    #     annual raw-biogas potential / 8760
    #
    # This represents continuous average raw-biogas availability.
    # ------------------------------------------------------------------

    configured_p_nom = cfg.get(
        "power_capacity_mw",
        None,
    )


    if (
        configured_p_nom is None
        or _is_missing(
            configured_p_nom
        )
    ):

        p_nom = (
            annual_raw_biogas
            / 8760.0
        )

    else:

        p_nom = float(
            configured_p_nom
        )


    if p_nom <= 0:

        raise ValueError(
            "Direct raw-biogas SWFL supply capacity "
            f"must be positive; got {p_nom} MW."
        )


    # ------------------------------------------------------------------
    # Economics
    # ------------------------------------------------------------------

    raw_cost = float(
        cfg.get(
            "raw_biogas_cost_eur_per_mwh_hs",
            settings.get(
                "raw_biogas_cost_eur_per_mwh_hs",
                75.0,
            ),
        )
    )


    transport_cost = float(
        cfg.get(
            "transport_cost_eur_per_mwh_hs",
            0.0,
        )
    )


    if raw_cost < 0:

        raise ValueError(
            "raw_biogas_to_swfl."
            "raw_biogas_cost_eur_per_mwh_hs "
            "must be non-negative."
        )


    if transport_cost < 0:

        raise ValueError(
            "raw_biogas_to_swfl."
            "transport_cost_eur_per_mwh_hs "
            "must be non-negative."
        )


    calculated_marginal_cost = (
        raw_cost
        + transport_cost
    )


    configured_marginal_cost = cfg.get(
        "marginal_cost_eur_per_mwh_hs",
        None,
    )


    if (
        configured_marginal_cost is not None
        and not _is_missing(
            configured_marginal_cost
        )
    ):

        configured_marginal_cost = float(
            configured_marginal_cost
        )


        if not np.isclose(
            configured_marginal_cost,
            calculated_marginal_cost,
            atol=1.0e-6,
            rtol=0.0,
        ):

            raise ValueError(
                "Inconsistent raw-biogas SWFL cost: "
                f"configured final marginal cost is "
                f"{configured_marginal_cost:.6f} EUR/MWh_Hs, "
                "but raw-biogas cost + transport cost gives "
                f"{calculated_marginal_cost:.6f} EUR/MWh_Hs."
            )


    marginal_cost = (
        calculated_marginal_cost
    )


    # ------------------------------------------------------------------
    # Recreate supply Generator
    # ------------------------------------------------------------------

    _remove_component_if_exists(
        network,
        "Generator",
        generator_name,
    )


    network.add(
        "Generator",
        generator_name,
        bus=target_bus,
        carrier=generator_carrier,
        p_nom=p_nom,
        p_nom_extendable=False,
        p_nom_min=0.0,
        p_min_pu=0.0,
        p_max_pu=1.0,
        marginal_cost=marginal_cost,
        capital_cost=0.0,
        efficiency=1.0,
        build_year=0,
        lifetime=np.inf,
        committable=False,
    )


    # ------------------------------------------------------------------
    # Metadata for reporting / debugging
    # ------------------------------------------------------------------

    try:

        network.generators.loc[
            generator_name,
            "annual_raw_biogas_potential_mwh_hs"
        ] = annual_raw_biogas

        network.generators.loc[
            generator_name,
            "raw_biogas_fuel_cost_eur_per_mwh_hs"
        ] = raw_cost

        network.generators.loc[
            generator_name,
            "raw_biogas_transport_cost_eur_per_mwh_hs"
        ] = transport_cost

        network.generators.loc[
            generator_name,
            "biogas_sh_route"
        ] = "raw_biogas_to_swfl"

    except Exception:
        pass


    print(
        "\nDirect raw-biogas supply to SWFL"
    )

    print(
        f"  Generator:              "
        f"{generator_name}"
    )

    print(
        f"  carrier:                "
        f"{generator_carrier}"
    )

    print(
        f"  target bus:             "
        f"{target_bus}"
    )

    print(
        f"  annual raw potential:   "
        f"{annual_raw_biogas:.2f} MWh_Hs/a"
    )

    print(
        f"  supply capacity:        "
        f"{p_nom:.6f} MW_Hs"
    )

    print(
        f"  raw-biogas cost:        "
        f"{raw_cost:.2f} EUR/MWh_Hs"
    )

    print(
        f"  collection/transport:   "
        f"{transport_cost:.2f} EUR/MWh_Hs"
    )

    print(
        f"  delivered cost:         "
        f"{marginal_cost:.2f} EUR/MWh_Hs"
    )

    print(
        "  upgrading applied:      no"
    )

    print(
        "  CO2-sale credit:        no"
    )


    return generator_name


def apply_biogas_sh_assets(
    self,
) -> None:
    """
    Add project-specific Biogas.SH assets to the PyPSA network.

    Supported physical routes
    -------------------------

    1. Onsite generation

        raw biogas
            -> onsite electricity
            -> onsite heat


    2. Upgraded biomethane -> public CH4 grid

        raw biogas
            -> upgrading
            -> plant CH4 generator
            -> central biomethane storage
            -> public CH4 grid


    3. Legacy upgraded biomethane -> SWFL

        raw biogas
            -> upgrading
            -> central biomethane storage
            -> dedicated SWFL biomethane bus


    4. Direct raw biogas -> SWFL

        regional raw biogas
            -> collection / transport
            -> dedicated SWFL raw-biogas bus
            -> eligible SWFL boilers

        No upgrading efficiency and no CO2-sale credit are applied
        to this direct raw-biogas route.


    Resource constraint
    -------------------
    All Biogas.SH routes must consume the same regional raw-biogas
    potential through the shared constraint in constraints.py.
    """

    # ==================================================================
    # 1. SETTINGS
    # ==================================================================

    settings = (
        self.args.get(
            "biogas_sh",
            {},
        )
        or {}
    )


    if not _as_bool(
        settings.get(
            "active",
            False,
        ),
        False,
    ):

        logger.info(
            "Biogas.SH assets inactive; "
            "network remains unchanged."
        )

        return


    csv_path = settings.get(
        "csv_path"
    )


    if (
        csv_path is None
        or not str(
            csv_path
        ).strip()
    ):

        raise ValueError(
            "args['biogas_sh']['csv_path'] "
            "must be set."
        )


    (
        route_mode,
        add_local_generation,
        add_gas_grid_generation,
        add_swfl_biomethane_supply,
        add_swfl_raw_biogas_supply,
    ) = _resolve_biogas_sh_route_switches(
        settings
    )


    network = self.network


    # ==================================================================
    # 2. PLANT DATA
    # ==================================================================

    df = _read_biogas_sh_csv(
        csv_path
    )


    if _as_bool(
        settings.get(
            "auto_map_demand_buses",
            True,
        ),
        True,
    ):

        df = _auto_map_demand_buses(
            self,
            df,
            settings,
        )


    mapped_csv = settings.get(
        "write_mapped_csv"
    )


    if mapped_csv:

        mapped_csv = Path(
            mapped_csv
        ).expanduser()

        mapped_csv.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_csv(
            mapped_csv,
            index=False,
        )


    # ==================================================================
    # 3. CARRIERS
    # ==================================================================

    ac_only_carrier = str(
        settings.get(
            "ac_only_carrier",
            "biogas_sh_onsite_el",
        )
    ).strip()


    chp_el_carrier = str(
        settings.get(
            "chp_el_carrier",
            "biogas_sh_onsite_chp_el",
        )
    ).strip()


    supported_ac_only_carrier = str(
        settings.get(
            "supported_ac_only_carrier",
            "biogas_sh_onsite_el_supported",
        )
    ).strip()


    supported_chp_el_carrier = str(
        settings.get(
            "supported_chp_el_carrier",
            "biogas_sh_onsite_chp_el_supported",
        )
    ).strip()


    chp_heat_carrier = str(
        settings.get(
            "chp_heat_carrier",
            "biogas_sh_onsite_chp_heat",
        )
    ).strip()


    ch4_link_carrier = str(
        settings.get(
            "ch4_link_carrier",
            "biogas_sh_gas_grid_injection",
        )
    ).strip()


    storage_cfg = _biogas_sh_storage_settings(
        settings
    )


    storage_input_carrier = str(
        storage_cfg.get(
            "input_link_carrier",
            "biogas_sh_collection_to_storage",
        )
    ).strip()


    storage_grid_carrier = str(
        storage_cfg.get(
            "grid_link_carrier",
            "biogas_sh_storage_to_grid",
        )
    ).strip()


    storage_swfl_carrier = str(
        storage_cfg.get(
            "swfl_link_carrier",
            "biogas_sh_storage_to_swfl",
        )
    ).strip()


    required_carriers = [
        ac_only_carrier,
        chp_el_carrier,
        supported_ac_only_carrier,
        supported_chp_el_carrier,
        chp_heat_carrier,
        "CH4",
        "CH4_biogas",
        ch4_link_carrier,
        "biogas_sh_swfl_direct",
        "biogas_sh_swfl_grid_supply",
        storage_input_carrier,
        storage_grid_carrier,
        storage_swfl_carrier,
    ]


    for carrier in required_carriers:

        carrier = str(
            carrier
        ).strip()

        if not carrier:

            raise ValueError(
                "Biogas.SH carrier names "
                "must not be empty."
            )

        _ensure_carrier(
            network,
            carrier,
        )


    # ==================================================================
    # 4. SUPPORT / COST SETTINGS
    # ==================================================================

    support_cfg = (
        settings.get(
            "support",
            {},
        )
        or {}
    )


    support_case = str(
        support_cfg.get(
            "case",
            "post_eeg",
        )
    ).strip()


    eeg_active = _as_bool(
        support_cfg.get(
            "eeg_active",
            False,
        ),
        False,
    )


    chp_capacity_multiplier = float(
        support_cfg.get(
            "chp_capacity_multiplier",
            1.0,
        )
    )


    if chp_capacity_multiplier <= 0:

        raise ValueError(
            "biogas_sh.support.chp_capacity_multiplier "
            "must be greater than zero."
        )


    market_el_mc = float(
        support_cfg.get(
            "market_electricity_marginal_cost",
            settings.get(
                "electricity_marginal_cost",
                197.37,
            ),
        )
    )


    supported_el_mc = float(
        support_cfg.get(
            "supported_electricity_marginal_cost",
            110.31,
        )
    )


    supported_hours_per_year = float(
        support_cfg.get(
            "supported_hours_per_year",
            0.0,
        )
    )


    heat_mc = float(
        settings.get(
            "heat_marginal_cost",
            166.67,
        )
    )


    default_biomethane_cost = float(
        settings.get(
            "default_biomethane_cost",
            92.9,
        )
    )


    biomethane_price_override = settings.get(
        "biomethane_price_override_eur_per_mwh",
        None,
    )


    if (
        biomethane_price_override is not None
        and not _is_missing(
            biomethane_price_override
        )
    ):

        biomethane_price_override = float(
            biomethane_price_override
        )

    else:

        biomethane_price_override = None


    # ==================================================================
    # 5. PUBLIC CH4 GRID
    # ==================================================================

    target_ch4_bus = _clean_id(
        settings.get(
            "target_ch4_bus",
            "47538",
        )
    )


    gas_topology = str(
        settings.get(
            "gas_topology",
            "producer_bus_link",
        )
    ).strip().lower()


    valid_gas_topologies = {
        "direct_at_target",
        "producer_bus_link",
    }


    if gas_topology not in valid_gas_topologies:

        raise ValueError(
            "biogas_sh.gas_topology must be "
            "'direct_at_target' or "
            "'producer_bus_link'."
        )


    if add_gas_grid_generation:

        if target_ch4_bus is None:

            raise ValueError(
                "biogas_sh.target_ch4_bus "
                "is missing."
            )


        if (
            target_ch4_bus
            not in network.buses.index
        ):

            raise ValueError(
                "Biogas.SH target_ch4_bus "
                f"{target_ch4_bus!r} was not "
                "found in network.buses."
            )


    # ==================================================================
    # 6. SWFL NATURAL-GAS ACCESS
    # ==================================================================
    #
    # This is deliberately independent of whether upgraded biomethane
    # is delivered to SWFL.
    #
    # In the new raw-biogas scenario SWFL still requires fossil natural
    # gas as an alternative fuel.
    # ==================================================================

    swfl_bus = None
    swfl_public_ch4_bus = None
    swfl_redirected_links = []
    swfl_grid_supply_added = False


    ensure_swfl_gas_access = _as_bool(
        settings.get(
            "ensure_swfl_gas_access",
            (
                add_swfl_biomethane_supply
                or add_swfl_raw_biogas_supply
            ),
        ),
        (
            add_swfl_biomethane_supply
            or add_swfl_raw_biogas_supply
        ),
    )


    if ensure_swfl_gas_access:

        (
            swfl_bus,
            swfl_public_ch4_bus,
            swfl_redirected_links,
            swfl_grid_supply_added,
        ) = _ensure_swfl_ch4_bus_and_access(
            network,
            settings,
        )


    # ==================================================================
    # 7. DIRECT RAW-BIOGAS SUPPLY TO SWFL
    # ==================================================================

    raw_swfl_generator = ""


    if add_swfl_raw_biogas_supply:

        raw_swfl_generator = (
            _add_raw_biogas_swfl_supply(
                network=network,
                df=df,
                settings=settings,
            )
        )


        if not raw_swfl_generator:

            raise RuntimeError(
                "Direct raw-biogas supply to SWFL "
                "was requested, but no supply "
                "Generator was created."
            )


    # ==================================================================
    # 8. OPTIONAL CENTRAL BIOMETHANE STORAGE
    # ==================================================================

    use_biogas_sh_storage = _as_bool(
        storage_cfg.get(
            "active",
            False,
        ),
        False,
    )


    biomethane_output_active = (
        add_gas_grid_generation
        or add_swfl_biomethane_supply
    )


    if (
        use_biogas_sh_storage
        and not biomethane_output_active
    ):

        raise ValueError(
            "Biogas.SH biomethane storage is active, "
            "but neither public-grid biomethane injection "
            "nor biomethane-to-SWFL is active."
        )


    storage_bus = None
    storage_added = False


    if use_biogas_sh_storage:

        (
            storage_bus,
            storage_added,
        ) = _ensure_biogas_sh_single_storage(
            network=network,
            settings=settings,
        )


        if storage_bus is None:

            raise RuntimeError(
                "Biogas.SH storage is active, "
                "but no storage bus was returned."
            )


        _add_biogas_sh_storage_output_links(
            network=network,
            storage_bus=storage_bus,
            settings=settings,
            add_gas_grid_generation=(
                add_gas_grid_generation
            ),
            add_swfl_direct_supply=(
                add_swfl_biomethane_supply
            ),
            target_ch4_bus=target_ch4_bus,
            swfl_bus=swfl_bus,
        )


    # ==================================================================
    # 9. COUNTERS
    # ==================================================================

    added_el = 0
    added_heat = 0
    added_ch4 = 0

    added_storage_input_links = 0
    added_grid_links = 0
    added_swfl_links = 0

    added_el_capacity = 0.0
    added_heat_capacity = 0.0
    added_ch4_capacity = 0.0

    added_storage_input_capacity = 0.0
    added_grid_link_capacity = 0.0
    added_swfl_link_capacity = 0.0

    potential_el_capacity = 0.0
    potential_heat_capacity = 0.0
    potential_ch4_capacity = 0.0


    debug_local = {
        "rows_total": 0,
        "local_block_entered": 0,
        "has_heat_true": 0,
        "ac_bus_missing": 0,
        "ac_bus_not_in_network": 0,
        "p_nom_el_zero": 0,
        "heat_bus_missing": 0,
        "heat_bus_not_in_network": 0,
        "p_nom_heat_zero": 0,
        "ch4_p_nom_zero": 0,
    }


    skipped = []


    storage_input_factor = float(
        storage_cfg.get(
            "input_link_p_nom_factor",
            1.0,
        )
    )


    grid_link_factor = float(
        settings.get(
            "ch4_link_p_nom_factor",
            1.0,
        )
    )


    swfl_cfg = _swfl_direct_settings(
        settings
    )


    direct_swfl_factor = float(
        swfl_cfg.get(
            "direct_link_p_nom_factor",
            1.0,
        )
    )


    # ==================================================================
    # 10. PLANT-SPECIFIC ASSETS
    # ==================================================================

    for _, row in df.iterrows():

        debug_local[
            "rows_total"
        ] += 1


        plant_id = int(
            row[
                "plant_id"
            ]
        )

        plant_name = str(
            row[
                "plant_name"
            ]
        )


        has_heat = _plant_has_heat(
            row,
            settings,
        )


        # ==============================================================
        # 10A. ONSITE ELECTRICITY / HEAT
        # ==============================================================

        if add_local_generation:

            debug_local[
                "local_block_entered"
            ] += 1


            ac_bus = _clean_id(
                _safe_str(
                    row,
                    "ac_bus_for_model",
                )
            )


            base_p_nom_el = (
                _get_electric_p_nom(
                    row,
                    settings,
                )
            )


            p_nom_el = (
                base_p_nom_el
                * chp_capacity_multiplier
            )


            potential_el_capacity += max(
                float(
                    p_nom_el
                ),
                0.0,
            )


            if ac_bus is None:

                debug_local[
                    "ac_bus_missing"
                ] += 1

                skipped.append(
                    (
                        plant_id,
                        plant_name,
                        "electricity_bus_missing",
                        ac_bus,
                    )
                )


            elif (
                ac_bus
                not in network.buses.index
            ):

                debug_local[
                    "ac_bus_not_in_network"
                ] += 1

                skipped.append(
                    (
                        plant_id,
                        plant_name,
                        "electricity_bus_not_in_network",
                        ac_bus,
                    )
                )


            elif p_nom_el <= 0:

                debug_local[
                    "p_nom_el_zero"
                ] += 1

                skipped.append(
                    (
                        plant_id,
                        plant_name,
                        "electricity_p_nom_zero",
                        p_nom_el,
                    )
                )


            else:

                market_carrier = (
                    chp_el_carrier
                    if has_heat
                    else ac_only_carrier
                )


                _add_electricity_generator(
                    network=network,
                    plant_id=plant_id,
                    ac_bus=ac_bus,
                    carrier=market_carrier,
                    p_nom=p_nom_el,
                    mc=market_el_mc,
                    tranche="market",
                )


                if eeg_active:

                    supported_carrier = (
                        supported_chp_el_carrier
                        if has_heat
                        else supported_ac_only_carrier
                    )


                    _add_electricity_generator(
                        network=network,
                        plant_id=plant_id,
                        ac_bus=ac_bus,
                        carrier=supported_carrier,
                        p_nom=p_nom_el,
                        mc=supported_el_mc,
                        tranche="supported",
                    )


                added_el += 1

                added_el_capacity += float(
                    p_nom_el
                )


            # ----------------------------------------------------------
            # Onsite heat
            # ----------------------------------------------------------

            if has_heat:

                debug_local[
                    "has_heat_true"
                ] += 1


                heat_bus = _clean_id(
                    _safe_str(
                        row,
                        "heat_bus_for_model",
                    )
                )


                p_nom_heat = _get_heat_p_nom(
                    row,
                    settings,
                )


                potential_heat_capacity += max(
                    float(
                        p_nom_heat
                    ),
                    0.0,
                )


                if heat_bus is None:

                    debug_local[
                        "heat_bus_missing"
                    ] += 1

                    skipped.append(
                        (
                            plant_id,
                            plant_name,
                            "heat_bus_missing",
                            heat_bus,
                        )
                    )


                elif (
                    heat_bus
                    not in network.buses.index
                ):

                    debug_local[
                        "heat_bus_not_in_network"
                    ] += 1

                    skipped.append(
                        (
                            plant_id,
                            plant_name,
                            "heat_bus_not_in_network",
                            heat_bus,
                        )
                    )


                elif p_nom_heat <= 0:

                    debug_local[
                        "p_nom_heat_zero"
                    ] += 1

                    skipped.append(
                        (
                            plant_id,
                            plant_name,
                            "heat_p_nom_zero",
                            p_nom_heat,
                        )
                    )


                else:

                    _add_heat_generator(
                        network=network,
                        plant_id=plant_id,
                        heat_bus=heat_bus,
                        carrier=chp_heat_carrier,
                        p_nom=p_nom_heat,
                        mc=heat_mc,
                    )


                    added_heat += 1

                    added_heat_capacity += float(
                        p_nom_heat
                    )


        # ==============================================================
        # 10B. UPGRADED BIOMETHANE
        # ==============================================================
        #
        # Direct raw-biogas-to-SWFL does NOT enter this block.
        # ==============================================================

        if not biomethane_output_active:

            continue


        p_nom_ch4 = _get_ch4_p_nom(
            row,
            settings,
        )


        potential_ch4_capacity += max(
            float(
                p_nom_ch4
            ),
            0.0,
        )


        if (
            biomethane_price_override
            is not None
        ):

            biomethane_cost = (
                biomethane_price_override
            )

        else:

            biomethane_cost = _safe_float(
                row,
                "biomethane_price_eur_per_mwh_hs_at_96pct",
                default=default_biomethane_cost,
            )


        if p_nom_ch4 <= 0:

            debug_local[
                "ch4_p_nom_zero"
            ] += 1

            skipped.append(
                (
                    plant_id,
                    plant_name,
                    "ch4_p_nom_zero",
                    p_nom_ch4,
                )
            )

            continue


        # --------------------------------------------------------------
        # Grid-only direct-at-target topology
        # --------------------------------------------------------------

        if (
            add_gas_grid_generation
            and not add_swfl_biomethane_supply
            and gas_topology
            == "direct_at_target"
            and not use_biogas_sh_storage
        ):

            _add_ch4_generator_direct(
                network=network,
                plant_id=plant_id,
                ch4_bus=target_ch4_bus,
                p_nom=p_nom_ch4,
                mc=biomethane_cost,
            )


            added_ch4 += 1

            added_ch4_capacity += float(
                p_nom_ch4
            )

            continue


        # --------------------------------------------------------------
        # Plant biomethane Generator
        # --------------------------------------------------------------

        (
            plant_bus,
            _,
        ) = _ensure_plant_ch4_bus_and_generator(
            network=network,
            row=row,
            plant_id=plant_id,
            p_nom=p_nom_ch4,
            mc=biomethane_cost,
        )


        added_ch4 += 1

        added_ch4_capacity += float(
            p_nom_ch4
        )


        # --------------------------------------------------------------
        # Biomethane -> central storage
        # --------------------------------------------------------------

        if use_biogas_sh_storage:

            _add_ch4_storage_input_link_from_plant(
                network=network,
                plant_id=plant_id,
                plant_bus=plant_bus,
                storage_bus=storage_bus,
                p_nom=p_nom_ch4,
                settings=settings,
            )


            added_storage_input_links += 1

            added_storage_input_capacity += (
                float(
                    p_nom_ch4
                )
                * storage_input_factor
            )


            continue


        # --------------------------------------------------------------
        # Biomethane -> public grid
        # --------------------------------------------------------------

        if add_gas_grid_generation:

            _add_ch4_grid_link_from_plant(
                network=network,
                plant_id=plant_id,
                plant_bus=plant_bus,
                target_ch4_bus=target_ch4_bus,
                p_nom=p_nom_ch4,
                settings=settings,
            )


            added_grid_links += 1

            added_grid_link_capacity += (
                float(
                    p_nom_ch4
                )
                * grid_link_factor
            )


        # --------------------------------------------------------------
        # Legacy biomethane -> SWFL
        # --------------------------------------------------------------

        if add_swfl_biomethane_supply:

            if swfl_bus is None:

                raise RuntimeError(
                    "Biomethane-to-SWFL route is active "
                    "but the SWFL CH4 bus was not created."
                )


            _add_swfl_direct_link_from_plant(
                network=network,
                plant_id=plant_id,
                plant_bus=plant_bus,
                swfl_bus=swfl_bus,
                p_nom=p_nom_ch4,
                settings=settings,
            )


            added_swfl_links += 1

            added_swfl_link_capacity += (
                float(
                    p_nom_ch4
                )
                * direct_swfl_factor
            )


    # ==================================================================
    # 11. SUMMARY
    # ==================================================================

    print(
        "\nBiogas.SH assets added"
    )


    print(
        f"  scenario mode:                    "
        f"{route_mode}"
    )

    print(
        f"  local generation active:          "
        f"{add_local_generation}"
    )

    print(
        f"  public biomethane grid route:     "
        f"{add_gas_grid_generation}"
    )

    print(
        f"  upgraded biomethane -> SWFL:      "
        f"{add_swfl_biomethane_supply}"
    )

    print(
        f"  direct raw biogas -> SWFL:        "
        f"{add_swfl_raw_biogas_supply}"
    )

    print(
        f"  gas topology:                     "
        f"{gas_topology}"
    )

    print(
        f"  public target CH4 bus:            "
        f"{target_ch4_bus}"
    )


    if biomethane_price_override is not None:

        print(
            f"  biomethane price override:        "
            f"{biomethane_price_override:.2f} "
            "EUR/MWh_Hs"
        )


    if add_swfl_raw_biogas_supply:

        raw_cfg = (
            settings.get(
                "raw_biogas_to_swfl",
                {},
            )
            or {}
        )

        print(
            f"  raw-biogas SWFL generator:        "
            f"{raw_swfl_generator}"
        )

        print(
            f"  raw-biogas delivered cost:        "
            f"{float(raw_cfg.get('marginal_cost_eur_per_mwh_hs', 0.0)):.2f} "
            "EUR/MWh_Hs"
        )


    print(
        f"  support case:                     "
        f"{support_case}"
    )

    print(
        f"  EEG support active:               "
        f"{eeg_active}"
    )

    print(
        f"  CHP capacity multiplier:          "
        f"{chp_capacity_multiplier:.2f}"
    )

    print(
        f"  onsite electricity generators:    "
        f"{added_el}"
    )

    print(
        f"  onsite heat generators:           "
        f"{added_heat}"
    )

    print(
        f"  biomethane generators:            "
        f"{added_ch4}"
    )

    print(
        f"  plant-to-storage links:           "
        f"{added_storage_input_links}"
    )

    print(
        f"  direct plant-to-grid links:       "
        f"{added_grid_links}"
    )

    print(
        f"  legacy biomethane->SWFL links:    "
        f"{added_swfl_links}"
    )

    print(
        f"  onsite electricity capacity MW:   "
        f"{added_el_capacity:.6f}"
    )

    print(
        f"  onsite heat capacity MW:          "
        f"{added_heat_capacity:.6f}"
    )

    print(
        f"  biomethane capacity MW:           "
        f"{added_ch4_capacity:.6f}"
    )

    print(
        f"  central biomethane storage:       "
        f"{use_biogas_sh_storage}"
    )


    if ensure_swfl_gas_access:

        print(
            f"  SWFL natural-gas bus:             "
            f"{swfl_bus}"
        )

        print(
            f"  SWFL public CH4 bus:              "
            f"{swfl_public_ch4_bus}"
        )

        print(
            f"  public-grid-to-SWFL link added:   "
            f"{swfl_grid_supply_added}"
        )


    print(
        f"  skipped components:               "
        f"{len(skipped)}"
    )


    if (
        skipped
        and _as_bool(
            settings.get(
                "print_skipped_components",
                True,
            ),
            True,
        )
    ):

        print(
            "  skipped details:"
        )


        for (
            plant_id,
            plant_name,
            kind,
            value,
        ) in skipped[:30]:

            print(
                f"    plant {plant_id} "
                f"({plant_name}), "
                f"{kind}, value={value}"
            )


        if len(
            skipped
        ) > 30:

            print(
                f"    ... {len(skipped) - 30} "
                "more skipped components"
            )


def validate_biogas_sh_storage_topology(
    network,
    args,
) -> None:
    """
    Validate active Biogas.SH biomethane-storage routes before clustering.

    The validator checks only routes that are actually active.

    Possible storage outputs
    ------------------------
    1. central biomethane storage -> public CH4 grid
    2. central biomethane storage -> dedicated SWFL biomethane bus

    The direct raw-biogas -> SWFL route does not use this storage and
    must therefore not be validated here.
    """

    biogas_cfg = (
        args.get(
            "biogas_sh",
            {},
        )
        or {}
    )


    storage_cfg = (
        biogas_cfg.get(
            "gas_storage",
            {},
        )
        or {}
    )


    storage_active = _as_bool(
        storage_cfg.get(
            "active",
            False,
        ),
        False,
    )


    if not storage_active:

        print(
            "\nBiogas.SH biomethane storage inactive; "
            "storage topology validation skipped."
        )

        return


    add_grid = _as_bool(
        biogas_cfg.get(
            "add_gas_grid_generation",
            False,
        ),
        False,
    )


    add_swfl_biomethane = _as_bool(
        biogas_cfg.get(
            "add_swfl_direct_supply",
            False,
        ),
        False,
    )


    add_swfl_raw = _as_bool(
        biogas_cfg.get(
            "add_swfl_raw_biogas_supply",
            False,
        ),
        False,
    )


    expected_storage_bus = str(
        storage_cfg.get(
            "bus",
            "biogas_sh_storage_ch4_bus",
        )
    ).strip()


    if (
        expected_storage_bus
        not in network.buses.index
    ):

        raise RuntimeError(
            f"Biogas.SH storage bus "
            f"{expected_storage_bus!r} does not "
            "exist before clustering."
        )


    print(
        "\nBiogas.SH storage topology validation"
    )

    print(
        f"  storage bus:                 "
        f"{expected_storage_bus}"
    )

    print(
        f"  public-grid route active:    "
        f"{add_grid}"
    )

    print(
        f"  biomethane -> SWFL active:   "
        f"{add_swfl_biomethane}"
    )

    print(
        f"  raw biogas -> SWFL active:   "
        f"{add_swfl_raw}"
    )


    if not (
        add_grid
        or add_swfl_biomethane
    ):

        raise RuntimeError(
            "Biogas.SH biomethane storage is active but "
            "has no active biomethane output route."
        )


    # ==================================================================
    # Helper
    # ==================================================================

    def validate_link(
        expected_name,
        expected_bus0,
        expected_bus1,
        expected_carrier,
        label,
    ):

        expected_name = str(
            expected_name
        ).strip()

        expected_bus0 = str(
            expected_bus0
        ).strip()

        expected_bus1 = str(
            expected_bus1
        ).strip()

        expected_carrier = str(
            expected_carrier
        ).strip()


        if (
            expected_bus1
            not in network.buses.index
        ):

            raise RuntimeError(
                f"{label} target bus "
                f"{expected_bus1!r} does not exist."
            )


        actual_link = None


        if expected_name in network.links.index:

            actual_link = (
                expected_name
            )

        else:

            mask = (
                network.links[
                    "bus0"
                ]
                .astype(str)
                .eq(
                    expected_bus0
                )
                &
                network.links[
                    "bus1"
                ]
                .astype(str)
                .eq(
                    expected_bus1
                )
                &
                network.links[
                    "carrier"
                ]
                .astype(str)
                .eq(
                    expected_carrier
                )
            )


            matches = (
                network.links.index[
                    mask
                ]
            )


            if len(matches) == 0:

                relevant = network.links.loc[
                    network.links[
                        "carrier"
                    ]
                    .astype(str)
                    .str.contains(
                        "biogas_sh_storage",
                        case=False,
                        na=False,
                    ),
                    [
                        column
                        for column in [
                            "bus0",
                            "bus1",
                            "carrier",
                            "p_nom",
                        ]
                        if column
                        in network.links.columns
                    ],
                ]


                raise RuntimeError(
                    f"Missing {label}.\n"
                    f"Expected name: {expected_name!r}\n"
                    f"Expected topology: "
                    f"{expected_bus0} -> {expected_bus1}\n"
                    f"Expected carrier: "
                    f"{expected_carrier!r}\n"
                    f"Storage-related links found:\n"
                    f"{relevant}"
                )


            if len(matches) > 1:

                raise RuntimeError(
                    f"Multiple links match {label}: "
                    f"{matches.astype(str).tolist()}"
                )


            actual_link = (
                matches[0]
            )


            logger.warning(
                "Expected %s link name %s was not found, "
                "but equivalent topology exists as %s.",
                label,
                expected_name,
                actual_link,
            )


        actual_bus0 = str(
            network.links.at[
                actual_link,
                "bus0",
            ]
        )

        actual_bus1 = str(
            network.links.at[
                actual_link,
                "bus1",
            ]
        )

        actual_carrier = str(
            network.links.at[
                actual_link,
                "carrier",
            ]
        )


        if actual_bus0 != expected_bus0:

            raise RuntimeError(
                f"{label}: wrong source bus "
                f"{actual_bus0!r}; expected "
                f"{expected_bus0!r}."
            )


        if actual_bus1 != expected_bus1:

            raise RuntimeError(
                f"{label}: wrong target bus "
                f"{actual_bus1!r}; expected "
                f"{expected_bus1!r}."
            )


        if actual_carrier != expected_carrier:

            raise RuntimeError(
                f"{label}: wrong carrier "
                f"{actual_carrier!r}; expected "
                f"{expected_carrier!r}."
            )


        print(
            f"  PASS: {label}"
        )

        print(
            f"        {actual_bus0} "
            f"-> {actual_bus1}"
        )

        print(
            f"        carrier="
            f"{actual_carrier}"
        )


    # ==================================================================
    # 1. STORAGE -> PUBLIC GAS GRID
    # ==================================================================

    if add_grid:

        public_bus = _clean_id(
            biogas_cfg.get(
                "target_ch4_bus",
                "47538",
            )
        )


        if public_bus is None:

            raise RuntimeError(
                "Public-grid biomethane route is active "
                "but target_ch4_bus is missing."
            )


        grid_link = str(
            storage_cfg.get(
                "grid_link",
                f"biogas_sh_storage_to_grid_{public_bus}",
            )
        )


        grid_carrier = str(
            storage_cfg.get(
                "grid_link_carrier",
                "biogas_sh_storage_to_grid",
            )
        )


        validate_link(
            expected_name=grid_link,
            expected_bus0=expected_storage_bus,
            expected_bus1=public_bus,
            expected_carrier=grid_carrier,
            label=(
                "storage -> public CH4 grid"
            ),
        )


    # ==================================================================
    # 2. STORAGE -> SWFL BIOMETHANE
    # ==================================================================

    if add_swfl_biomethane:

        swfl_target_bus = str(
            storage_cfg.get(
                "swfl_target_bus",
                "swfl_real_biomethane_ch4_bus",
            )
        ).strip()


        swfl_link = str(
            storage_cfg.get(
                "swfl_link",
                "biogas_sh_storage_to_swfl_biomethane",
            )
        )


        swfl_carrier = str(
            storage_cfg.get(
                "swfl_link_carrier",
                "biogas_sh_storage_to_swfl",
            )
        )


        validate_link(
            expected_name=swfl_link,
            expected_bus0=expected_storage_bus,
            expected_bus1=swfl_target_bus,
            expected_carrier=swfl_carrier,
            label=(
                "storage -> SWFL biomethane"
            ),
        )


    # ==================================================================
    # 3. RAW-BIOGAS ROUTE NOTICE
    # ==================================================================

    if (
        add_swfl_raw
        and not add_swfl_biomethane
    ):

        print(
            "  INFO: direct raw-biogas -> SWFL "
            "bypasses biomethane storage."
        )


    print(
        "\nBiogas.SH storage topology "
        "validation successful."
    )