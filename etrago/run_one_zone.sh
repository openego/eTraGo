#!/bin/bash
set -o pipefail

# --------------------------------------------------------------
# Paths
# --------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$SCRIPT_DIR" || exit 1

export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# --------------------------------------------------------------
# Run information
# --------------------------------------------------------------

ZONE="DE5"
AC_CLUSTERS="50"
SNAPSHOTS="168"

RUN_NAME="${ZONE}_ac${AC_CLUSTERS}_s${SNAPSHOTS}_$(date +%Y%m%d_%H%M%S)"

echo "=========================================="
echo "Starting run: ${RUN_NAME}"
echo "=========================================="

echo
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

print("\nnetwork:")
print(etrago.network.__file__)

print("\nmarket_optimization:")
print(mo.__file__)
PY

EXIT_CODE=$?

if [ "$EXIT_CODE" -ne 0 ]; then
    echo "Failed to import eTraGo."
    exit "$EXIT_CODE"
fi

echo
echo "=========================================="
echo "Running ${ZONE}"
echo "=========================================="

python -u appl.py 2>&1 | tee "${RUN_NAME}.log"

EXIT_CODE=${PIPESTATUS[0]}

if [ "$EXIT_CODE" -ne 0 ]; then
    echo
    echo "Run ${ZONE} failed with exit code ${EXIT_CODE}."
    exit "$EXIT_CODE"
fi

echo
echo "=========================================="
echo "Finished ${ZONE}"
echo "Log saved to ${RUN_NAME}.log"
echo "=========================================="