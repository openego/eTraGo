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

AC_CLUSTERS="50"
SNAPSHOTS="8760"


for ZONE in "${ZONES[@]}"; do

    echo "=========================================="
    echo "Starting run for ${ZONE}"
    echo "=========================================="

    RUN_NAME="${ZONE}_ac${AC_CLUSTERS}_s${SNAPSHOTS}_$(date +%Y%m%d_%H%M%S)"

    # Change market zone in config
    sed -i \
        "s/\"market_zones\": *\"[^\"]*\"/\"market_zones\": \"${ZONE}\"/" \
        "$CONFIG"

    # Change solver logfile name
    sed -i \
        "s/\"LogFile\": *\"[^\"]*\"/\"LogFile\": \"solver_${RUN_NAME}.log\"/" \
        "$CONFIG"

    sed -i \
        "s/\"logFile\": *\"[^\"]*\"/\"logFile\": \"solver_${RUN_NAME}.log\"/" \
        "$CONFIG"

    echo "Running ${ZONE}..."

    python -u appl.py 2>&1 | tee "${RUN_NAME}.log"

    EXIT_CODE=${PIPESTATUS[0]}

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "Run ${ZONE} failed with exit code ${EXIT_CODE}."
        echo "Stopping loop."
        exit "$EXIT_CODE"
    fi

    echo "Finished ${ZONE}"
    echo "Log saved to ${RUN_NAME}.log"

done

echo "All runs finished."