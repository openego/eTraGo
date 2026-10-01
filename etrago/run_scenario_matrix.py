#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 29 16:30:25 2026

@author: mohsenmansouri
"""

#!/usr/bin/env python3

import os
import subprocess
import sys
from pathlib import Path

from scenario_config import (
    expand_scenario_matrix,
    load_config,
)


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.yaml"
APPL_PATH = ROOT / "appl.py"
LOG_DIR = ROOT / "batch_logs"


ENVIRONMENT_VARIABLES = {
    "fossil_gas_price_case": "ETRAGO_FOSSIL_GAS_PRICE_CASE",
    "support_case": "ETRAGO_SUPPORT_CASE",
    "biomethane_price_case": "ETRAGO_BIOMETHANE_PRICE_CASE",
    "co2_sale_case": "ETRAGO_CO2_SALE_CASE",
    "heat_pump_case": "ETRAGO_HEAT_PUMP_CASE",
    "swfl_unit_case": "ETRAGO_SWFL_UNIT_CASE",
    "biomethane_use_case": "ETRAGO_BIOMETHANE_USE_CASE",
    "biogas_route_case": "ETRAGO_BIOGAS_ROUTE_CASE",
}


def run_scenario(record, index, total):
    scenario_name = record["scenario_name"]

    environment = os.environ.copy()

    # Remove any old overrides inherited from the shell.
    for variable in ENVIRONMENT_VARIABLES.values():
        environment.pop(variable, None)

    # Apply this scenario through the environment-override interface
    # already implemented in scenario_config.py.
    for dimension, variable in ENVIRONMENT_VARIABLES.items():
        if dimension in record:
            environment[variable] = str(record[dimension])

    log_path = LOG_DIR / f"{index:02d}_{scenario_name}.log"

    print()
    print("=" * 100)
    print(f"RUN {index}/{total}")
    print(f"Scenario: {scenario_name}")
    print(f"Log:      {log_path}")
    print("=" * 100)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [
                sys.executable,
                "-u",
                str(APPL_PATH),
            ],
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        if process.stdout is None:
            raise RuntimeError("Could not capture appl.py output.")

        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)

        return process.wait()


def main():
    config = load_config(CONFIG_PATH)

    batch_config = config.get("batch", {})

    if not batch_config.get("enabled", False):
        raise RuntimeError(
            "Batch execution is disabled. Set batch.enabled: true "
            "in config.yaml."
        )

    scenarios = expand_scenario_matrix(config)

    if not scenarios:
        raise RuntimeError(
            "No scenarios were generated. Check "
            "scenario_matrix.enabled and scenario_matrix.dimensions."
        )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    stop_on_error = bool(
        batch_config.get(
            "stop_on_error",
            True,
        )
    )

    completed = []
    failed = []

    print(f"Sequential scenario batch contains {len(scenarios)} runs.")

    for index, scenario in enumerate(
        scenarios,
        start=1,
    ):
        return_code = run_scenario(
            scenario,
            index,
            len(scenarios),
        )

        scenario_name = scenario["scenario_name"]

        if return_code == 0:
            completed.append(scenario_name)
            print(f"COMPLETED: {scenario_name}")
        else:
            failed.append(
                (
                    scenario_name,
                    return_code,
                )
            )

            print(
                f"FAILED: {scenario_name} "
                f"(exit code {return_code})"
            )

            if stop_on_error:
                break

    print()
    print("=" * 100)
    print("BATCH SUMMARY")
    print("=" * 100)
    print(f"Completed: {len(completed)}")
    print(f"Failed:    {len(failed)}")

    for scenario_name, return_code in failed:
        print(
            f"  - {scenario_name}: "
            f"exit code {return_code}"
        )

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())