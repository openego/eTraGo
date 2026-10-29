#!/bin/bash
set -o pipefail

# --------------------------------------------------------------
# Make sure Python imports THIS eTraGo checkout
# --------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$SCRIPT_DIR" || exit 1

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "Python executable:"
which python

echo
echo "Checking imported eTraGo files..."

python - <<'PY'
import etrago
import etrago.network
import etrago.execute.market_optimization as mo

print("etrago:")
print(etrago.__file__)

print("network:")
print(etrago.network.__file__)

print("market_optimization:")
print(mo.__file__)
PY

if [ $? -ne 0 ]; then
    echo "Failed to import eTraGo."
    exit 1
fi

CONFIG="$SCRIPT_DIR/args.json"

ZONES=(
     #"status_quo"
     #"DE2"
     "DE3"
     #"DE4"
     "DE5"
)

AC_CLUSTERS="100"
FOCUS_CLUSTERS="50"
SNAPSHOTS="168"

# Optional overrides from the command line, e.g.
#   RUN_ZONES="DE3" RUN_SNAPSHOTS=48 ./run_all_market_zones.sh
#   RUN_START=8017 RUN_SNAPSHOTS=8760 (= December; RUN_SNAPSHOTS is the last hour)
if [ -n "${RUN_ZONES:-}" ]; then
    read -ra ZONES <<< "$RUN_ZONES"
fi
SNAPSHOTS="${RUN_SNAPSHOTS:-$SNAPSHOTS}"
START_SNAPSHOT="${RUN_START:-1}"

# Transfer capacity between bidding zones in the market model
# LINE_LIMITS: "true"  = hourly limits of the underlying lines (branch
#                        capacity factor + dynamic line rating)
#              "false" = old eTraGo behaviour (sum of line ratings)
# CAPACITY_FACTOR: extra factor on links between two German zones
#              (1.0 = off, e.g. 0.7 = 70 % of the line limit tradable)
# LINE_LIMITS_SCOPE: links that get the line limits: "all", "german"
#              (at least one German zone) or "internal" (two German zones)
LINE_LIMITS="true"
LINE_LIMITS_SCOPE="${LINE_LIMITS_SCOPE:-german}"
CAPACITY_FACTOR="0.7"

# H2_to_power, H2_to_CH4 and CH4_to_H2 keep the market investment in the
# grid optimisation ("false" = the grid stage decides them again)
FIX_H2_CONVERSION="${FIX_H2_CONVERSION:-true}"

# Electrolysers keep the market total per zone and H2 bus, the grid
# optimisation chooses the nodes ("false" = market locations are kept);
# H2-to-power and H2 stores move with them
# ELY_LOCATION_GROUP: "zone" or "zone_h2" (zone and H2 bus)
# ELY_LOCATION_MAX_SHARE: max. share of a group total per node (1.0 = off)
OPT_ELY_LOCATION="${OPT_ELY_LOCATION:-true}"
ELY_LOCATION_GROUP="${ELY_LOCATION_GROUP:-zone_h2}"
ELY_LOCATION_MAX_SHARE="${ELY_LOCATION_MAX_SHARE:-0.25}"

# Preserve the original configuration
CONFIG_BACKUP="$(mktemp)"
cp "$CONFIG" "$CONFIG_BACKUP"

restore_config() {
    cp "$CONFIG_BACKUP" "$CONFIG"
    rm -f "$CONFIG_BACKUP"
    echo "Original args.json restored."
}

trap restore_config EXIT


for ZONE in "${ZONES[@]}"; do

    echo "=========================================="
    echo "Starting run for ${ZONE}"
    echo "=========================================="

    RUN_NAME="${ZONE}_ac${AC_CLUSTERS}_focus${FOCUS_CLUSTERS}_s${SNAPSHOTS}_$(date +%Y%m%d_%H%M%S)"
    RUN_DIR="results_${RUN_NAME}"

    if [ -e "$RUN_DIR" ]; then
        echo "Result directory already exists: $RUN_DIR"
        exit 1
    fi

    mkdir -p "$RUN_DIR"

    # Update the correct fields in args.json
    ZONE="$ZONE" \
    RUN_DIR="$RUN_DIR" \
    RUN_NAME="$RUN_NAME" \
    AC_CLUSTERS="$AC_CLUSTERS" \
    FOCUS_CLUSTERS="$FOCUS_CLUSTERS" \
    SNAPSHOTS="$SNAPSHOTS" \
    START_SNAPSHOT="$START_SNAPSHOT" \
    LINE_LIMITS="$LINE_LIMITS" \
    LINE_LIMITS_SCOPE="$LINE_LIMITS_SCOPE" \
    CAPACITY_FACTOR="$CAPACITY_FACTOR" \
    FIX_H2_CONVERSION="$FIX_H2_CONVERSION" \
    OPT_ELY_LOCATION="$OPT_ELY_LOCATION" \
    ELY_LOCATION_GROUP="$ELY_LOCATION_GROUP" \
    ELY_LOCATION_MAX_SHARE="$ELY_LOCATION_MAX_SHARE" \
    CONFIG="$CONFIG" \
    python - <<'PY'
import json
import os
from pathlib import Path

config_path = Path(os.environ["CONFIG"])

with config_path.open(encoding="utf-8") as handle:
    args = json.load(handle)

zone = os.environ["ZONE"]
run_dir = str(Path(os.environ["RUN_DIR"]).resolve())
run_name = os.environ["RUN_NAME"]

FOCUS_REGION = [
    "Flensburg",
    "Kiel",
    "Lübeck",
    "Neumünster",
    "Dithmarschen",
    "Herzogtum Lauenburg",
    "Nordfriesland",
    "Ostholstein",
    "Pinneberg",
    "Plön",
    "Rendsburg-Eckernförde",
    "Schleswig-Flensburg",
    "Segeberg",
    "Steinburg",
    "Stormarn",
]

market = args["method"]["market_optimization"]
clustering = args["network_clustering"]
cluster_method = clustering["method"]
electricity = clustering["electricity_grid"]

# Bidding-zone and temporal configuration
market["market_zones"] = zone
market["snapshot_step"] = 5

# Transfer capacity between bidding zones
line_limits = os.environ["LINE_LIMITS"].strip().lower()
if line_limits not in ("true", "false"):
    raise SystemExit(f"ERROR: LINE_LIMITS must be true or false, found {line_limits!r}.")
market["interzonal_line_limits"] = line_limits == "true"
line_limits_scope = os.environ["LINE_LIMITS_SCOPE"].strip().lower()
if line_limits_scope not in ("all", "german", "internal"):
    raise SystemExit(f"ERROR: LINE_LIMITS_SCOPE must be all, german or internal, found {line_limits_scope!r}.")
market["interzonal_line_limits_scope"] = line_limits_scope
try:
    capacity_factor = float(os.environ["CAPACITY_FACTOR"])
except ValueError:
    raise SystemExit("ERROR: CAPACITY_FACTOR must be a number.")
if not 0.0 < capacity_factor <= 1.0:
    raise SystemExit(f"ERROR: CAPACITY_FACTOR must be in (0, 1], found {capacity_factor}.")
market["interzonal_capacity_factor"] = capacity_factor

fix_conversion = os.environ["FIX_H2_CONVERSION"].strip().lower()
if fix_conversion not in ("true", "false"):
    raise SystemExit(f"ERROR: FIX_H2_CONVERSION must be true or false, found {fix_conversion!r}.")
market["fix_h2_conversion_investments_in_grid"] = fix_conversion == "true"

opt_ely_location = os.environ["OPT_ELY_LOCATION"].strip().lower()
if opt_ely_location not in ("true", "false"):
    raise SystemExit(f"ERROR: OPT_ELY_LOCATION must be true or false, found {opt_ely_location!r}.")
market["optimize_electrolyzer_location_in_grid"] = opt_ely_location == "true"
ely_group = os.environ["ELY_LOCATION_GROUP"].strip().lower()
if ely_group not in ("zone", "zone_h2"):
    raise SystemExit(f"ERROR: ELY_LOCATION_GROUP must be zone or zone_h2, found {ely_group!r}.")
market["electrolyzer_location_group"] = ely_group
try:
    max_share = float(os.environ["ELY_LOCATION_MAX_SHARE"])
except ValueError:
    raise SystemExit("ERROR: ELY_LOCATION_MAX_SHARE must be a number.")
if not 0.0 < max_share <= 1.0:
    raise SystemExit(f"ERROR: ELY_LOCATION_MAX_SHARE must be in (0, 1], found {max_share}.")
market["electrolyzer_location_max_share"] = max_share

args["start_snapshot"] = int(os.environ["START_SNAPSHOT"])
args["end_snapshot"] = int(os.environ["SNAPSHOTS"])
args["skip_snapshots"] = 5

# Focus-region configuration
cluster_method["focus_region"] = FOCUS_REGION
focus_clusters = int(os.environ["FOCUS_CLUSTERS"])

electricity["cluster_within_focus"] = True
electricity["n_clusters_focus"] = focus_clusters
electricity["n_clusters"] = int(os.environ["AC_CLUSTERS"])
if "gas_grids" in clustering:
    clustering["gas_grids"]["cluster_within_focus"] = False

# Required for ordinary and negative load shedding
args["load_shedding"] = True

# Separate output directory
args["export_results_path"] = run_dir

# Validate production solver configuration
solver_options = args.setdefault("solver_options", {})

required_solver_options = {
    "method": 2,
    "threads": 4,
    "NumericFocus": 3,
    "ScaleFlag": 2,
    "BarHomogeneous": 1,
}

for key, expected in required_solver_options.items():
    actual = solver_options.get(key)

    if actual != expected:
        raise SystemExit(
            f"ERROR: solver option {key!r} must be {expected!r}, "
            f"found {actual!r}."
        )

# Separate solver log
solver_log = str(Path(run_dir) / f"solver_{run_name}.log")

if "logFile" in solver_options:
    solver_options["logFile"] = solver_log
else:
    solver_options["LogFile"] = solver_log

# Validate the effective configuration
focus_region = cluster_method.get("focus_region")
cluster_within_focus = electricity.get("cluster_within_focus")
n_clusters_focus = electricity.get("n_clusters_focus")

if len(focus_region or []) != 15:
    raise SystemExit(
        f"ERROR: expected 15 focus districts, found {len(focus_region or [])}."
    )

if cluster_within_focus is not True:
    raise SystemExit(
        "ERROR: cluster_within_focus must be True."
    )

if n_clusters_focus != focus_clusters:
    raise SystemExit(
        "ERROR: n_clusters_focus does not match "
        f"FOCUS_CLUSTERS={focus_clusters}."
    )

if not 1 <= focus_clusters < electricity["n_clusters"]:
    raise SystemExit(
        "ERROR: FOCUS_CLUSTERS must be between 1 and "
        "AC_CLUSTERS - 1."
    )

if args.get("load_shedding") is not True:
    raise SystemExit(
        "ERROR: load_shedding must be True."
    )

if Path(args["export_results_path"]).resolve() != Path(run_dir).resolve():
    raise SystemExit(
        "ERROR: export_results_path does not match RUN_DIR."
    )
# Write the effective configuration
with config_path.open("w", encoding="utf-8") as handle:
    json.dump(args, handle, indent=4, ensure_ascii=False)
    handle.write("\n")

print("Effective settings:")
print("  market_zones:", market["market_zones"])
print("  total AC clusters:", electricity["n_clusters"])
print("  focus AC clusters:", n_clusters_focus)
print("  snapshots:", args["start_snapshot"], "-", args["end_snapshot"])
print("  snapshot_step:", market["snapshot_step"])
print("  interzonal_line_limits:", market["interzonal_line_limits"])
print("  interzonal_line_limits_scope:", market["interzonal_line_limits_scope"])
print("  interzonal_capacity_factor:", market["interzonal_capacity_factor"])
print("  fix_h2_conversion_investments_in_grid:", market["fix_h2_conversion_investments_in_grid"])
print("  optimize_electrolyzer_location_in_grid:", market["optimize_electrolyzer_location_in_grid"])
print("  electrolyzer_location_group:", market["electrolyzer_location_group"])
print("  electrolyzer_location_max_share:", market["electrolyzer_location_max_share"])
print("  skip_snapshots:", args["skip_snapshots"])
print("  focus districts:", len(focus_region))
print("  cluster_within_focus:", cluster_within_focus)
print("  load_shedding:", args["load_shedding"])
print("  solver:", args["solver"])
print("  solver options:")

for key, value in args["solver_options"].items():
    print(f"    {key}: {value}")

print("  results:", args["export_results_path"])
PY


    CONFIG_EDIT_EXIT_CODE=$?

    if [ "$CONFIG_EDIT_EXIT_CODE" -ne 0 ]; then
        echo "ERROR: failed to prepare the effective args.json."
        echo "The eTraGo run will not be started."
        exit "$CONFIG_EDIT_EXIT_CODE"
    fi

    cp "$CONFIG" "$RUN_DIR/args_input.json"

    echo "Running ${ZONE}..."

    ETRAGO_CONFIG="$CONFIG" \
    python -u appl.py 2>&1 | tee "${RUN_DIR}/${RUN_NAME}.log"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "Run ${ZONE} failed with exit code ${EXIT_CODE}."
        echo "Stopping loop."
        exit "$EXIT_CODE"
    fi

    REQUIRED_STAGES=(
        "original_network_topology"
        "pre_market_optimization"
        "market_optimization"
        "grid_optimization"
    )

    for STAGE in "${REQUIRED_STAGES[@]}"; do
        REQUIRED_FILE="${RUN_DIR}/${STAGE}/buses.csv"

        if [ ! -f "$REQUIRED_FILE" ]; then
            echo "ERROR: expected export is missing:"
            echo "  $REQUIRED_FILE"
            echo "Stopping batch."
            exit 1
        fi
    done

    echo "All required network exports verified."


    echo "Finished ${ZONE}"
    echo "Results: ${RUN_DIR}"
    echo "Log: ${RUN_DIR}/${RUN_NAME}.log"

done

echo "All five runs finished."