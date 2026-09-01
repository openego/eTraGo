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
    "status_quo"
    "DE2"
    "DE3"
    "DE4"
    "DE5"
)

AC_CLUSTERS="150"
FOCUS_CLUSTERS="100"
SNAPSHOTS="10"

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

args["start_snapshot"] = 1
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

# Separate solver log
solver_options = args.setdefault("solver_options", {})
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
print("  skip_snapshots:", args["skip_snapshots"])
print("  focus districts:", len(focus_region))
print("  cluster_within_focus:", cluster_within_focus)
print("  load_shedding:", args["load_shedding"])
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