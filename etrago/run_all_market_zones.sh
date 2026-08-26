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


CONFIG="args.json"

ZONES=(
    "status_quo"
    "DE2"
    "DE3"
    "DE4"
    "DE5"
)

AC_CLUSTERS="100"
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

    RUN_NAME="${ZONE}_ac${AC_CLUSTERS}_s${SNAPSHOTS}_$(date +%Y%m%d_%H%M%S)"
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

# Bidding-zone configuration
args["method"]["market_optimization"]["market_zones"] = zone

# AC clustering
args["network_clustering"]["electricity_grid"]["n_clusters"] = int(
    os.environ["AC_CLUSTERS"]
)

# Snapshot range
args["start_snapshot"] = 1
args["end_snapshot"] = int(os.environ["SNAPSHOTS"])

# Separate result folder
args["export_results_path"] = run_dir

# Separate solver log
solver_options = args.setdefault("solver_options", {})
solver_log = str(Path(run_dir) / f"solver_{run_name}.log")

if "logFile" in solver_options:
    solver_options["logFile"] = solver_log
else:
    solver_options["LogFile"] = solver_log

with config_path.open("w", encoding="utf-8") as handle:
    json.dump(args, handle, indent=4, ensure_ascii=False)
    handle.write("\n")

print("Effective settings:")
print("  market_zones:", zone)
print("  AC clusters:", args["network_clustering"]["electricity_grid"]["n_clusters"])
print("  snapshots:", args["start_snapshot"], "-", args["end_snapshot"])
print("  snapshot_step:",
      args["method"]["market_optimization"]["snapshot_step"])
print("  results:", run_dir)
PY

    cp "$CONFIG" "$RUN_DIR/args_input.json"

    echo "Running ${ZONE}..."

    python -u appl.py 2>&1 | tee "${RUN_DIR}/${RUN_NAME}.log"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "Run ${ZONE} failed with exit code ${EXIT_CODE}."
        echo "Stopping loop."
        exit "$EXIT_CODE"
    fi

    echo "Finished ${ZONE}"
    echo "Results: ${RUN_DIR}"
    echo "Log: ${RUN_DIR}/${RUN_NAME}.log"

done

echo "All five runs finished."