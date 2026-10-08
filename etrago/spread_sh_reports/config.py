"""Static configuration: scenario discovery, carrier groups and palettes."""

from dataclasses import dataclass
from pathlib import Path
import re

PACKAGE_DIR = Path(__file__).resolve().parent
ETRAGO_DIR = PACKAGE_DIR.parent
DEFAULT_RESULTS_DIR = ETRAGO_DIR / "results_071026"
DEFAULT_DATA_DIR = ETRAGO_DIR / "data"

STAGES = (
    "pre_market_optimization",
    "market_optimization",
    "grid_optimization",
)

# Fixed display order. Colour follows the configuration, never its rank.
SCENARIO_ORDER = ["status_quo", "DE2", "DE3", "DE4", "DE5"]
SCENARIO_LABELS = {
    "status_quo": "Status quo (DE/LU)",
    "DE2": "DE2 - two zones",
    "DE3": "DE3 - three zones",
    "DE4": "DE4 - four zones",
    "DE5": "DE5 - five zones",
}

# Shapefiles that were used by etrago.tools.market_zones (same MD5 set).
ZONE_SHAPEFILES = {
    "DE2": "shapes_biddingzones/BZR_config_2_DE2.shp",
    "DE3": "shapes_biddingzones/BZR_config_12_DE3.shp",
    "DE4": "shapes_biddingzones/BZR_config_13_DE4.shp",
    "DE5": "shapes_biddingzones/BZR_config_14_DE5.shp",
}
FOCUS_SHAPE = "shapes_focus_region/vg250_focus_region.geojson"
COUNTRY_SHAPE = "shapes_europe/ne_110m_admin_0_countries.shp"

# Countries in the eGon2035 electrical model.
MODEL_COUNTRIES = [
    "DE", "AT", "BE", "CH", "CZ", "DK", "FR", "GB", "LU", "NL", "NO", "PL",
    "SE",
]

# --------------------------------------------------------------------------
# Carrier groups (max. eight per chart - see dataviz rules)
# --------------------------------------------------------------------------
GEN_GROUPS = {
    "wind_onshore": "Wind onshore",
    "wind_offshore": "Wind offshore",
    "solar": "Solar",
    "solar_rooftop": "Solar",
    "run_of_river": "Hydro",
    "reservoir": "Hydro",
    "pumped_hydro": "Hydro",
    "biomass": "Biomass",
    "central_biomass_CHP": "Biomass",
    "industrial_biomass_CHP": "Biomass",
    "OCGT": "Gas & H2",
    "CCGT": "Gas & H2",
    "central_gas_CHP": "Gas & H2",
    "industrial_gas_CHP": "Gas & H2",
    "nuclear": "Nuclear",
    "coal": "Coal, oil & other",
    "lignite": "Coal, oil & other",
    "oil": "Coal, oil & other",
    "others": "Coal, oil & other",
    "H2_to_power": "Gas & H2",
}
GEN_GROUP_ORDER = [
    "Wind onshore", "Wind offshore", "Solar", "Hydro", "Biomass", "Gas & H2",
    "Nuclear", "Coal, oil & other",
]
VRES_CARRIERS = ["wind_onshore", "wind_offshore", "solar", "solar_rooftop"]
CURTAIL_GROUPS = {
    "wind_onshore": "Wind onshore",
    "wind_offshore": "Wind offshore",
    "solar": "Solar",
    "solar_rooftop": "Solar",
}
CURTAIL_GROUP_ORDER = ["Wind onshore", "Wind offshore", "Solar"]

# Electrical generator links feeding an AC bus (electric output = -p1).
POWER_LINKS = [
    "OCGT", "CCGT", "central_gas_CHP", "industrial_gas_CHP", "H2_to_power",
]

# Flexibility options: (label, component, carriers, measure)
#   measure "consumption"  -> sum of p0 > 0 (electricity drawn)
#   measure "discharge"    -> storage-unit dispatch
#   measure "shift"        -> |p0| / 2 (energy moved in time)
FLEX_OPTIONS = [
    ("Batteries", "storage_units", ["battery"], "discharge"),
    ("Pumped hydro", "storage_units", ["pumped_hydro"], "discharge"),
    ("Electrolysis", "links", ["power_to_H2"], "consumption"),
    ("Heat pumps & e-boilers", "links",
     ["central_heat_pump", "rural_heat_pump", "central_resistive_heater"],
     "consumption"),
    ("E-mobility charging", "links", ["BEV_charger"], "consumption"),
    ("Demand-side management", "links", ["dsm"], "shift"),
    ("H2-to-power", "links", ["H2_to_power"], "output"),
]
FLEX_ORDER = [f[0] for f in FLEX_OPTIONS]

# --------------------------------------------------------------------------
# Palette roles. Hex values are resolved in the browser so that light and
# dark mode each use their own validated steps (dataviz reference palette).
# --------------------------------------------------------------------------
SCENARIO_COLOR = {
    "status_quo": "@s1", "DE2": "@s2", "DE3": "@s3", "DE4": "@s4",
    "DE5": "@s5",
}
GEN_GROUP_COLOR = dict(zip(GEN_GROUP_ORDER, [f"@s{i}" for i in range(1, 9)]))
CURTAIL_COLOR = {
    "Wind onshore": "@s1", "Wind offshore": "@s2", "Solar": "@s4",
}
FLEX_COLOR = dict(zip(FLEX_ORDER, [f"@s{i}" for i in range(1, 9)]))

# Thresholds
CONGESTION_SHARE = 0.98  # |flow| >= 98 % of the limit counts as congested
HIGH_LOADING = 0.70


@dataclass
class Scenario:
    key: str           # status_quo | DE2 | ...
    path: Path
    run_name: str

    @property
    def label(self):
        return SCENARIO_LABELS.get(self.key, self.key)

    @property
    def is_split(self):
        return self.key in ZONE_SHAPEFILES

    def stage(self, name):
        return self.path / name


def discover_scenarios(results_dir):
    """Find one result directory per market-zone configuration."""
    results_dir = Path(results_dir)
    found = {}
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir() or not (child / "grid_optimization").exists():
            continue
        match = re.match(r"results_(status_quo|DE\d)_", child.name)
        if not match:
            continue
        key = match.group(1)
        if key in found:
            # Keep the most recent run of a configuration.
            if child.name < found[key].path.name:
                continue
        found[key] = Scenario(key, child, child.name.replace("results_", ""))
    ordered = [found[k] for k in SCENARIO_ORDER if k in found]
    ordered += [s for k, s in found.items() if k not in SCENARIO_ORDER]
    return ordered
