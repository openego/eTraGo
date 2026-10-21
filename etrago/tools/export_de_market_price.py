"""Export the German-zone market price of a finished run for the EEG premium.

The sliding EEG market premium of the Biogas.SH follow-on support needs the
annual market value JW (mean German day-ahead price) and the hours with a
price at or below 20 EUR/MWh. Both come from the hourly price of the German
bidding zone in the market stage of a reference run. The price depends on the
fossil-gas price case and hardly on the Biogas.SH/SWFL settings, so one
reference run per gas case is enough.

Usage:
    python -m etrago.tools.export_de_market_price RUN_FOLDER [RUN_FOLDER ...]
        [--out-dir DIR] [--min-price 20]

Writes de_price_<gas_case>.csv (columns snapshot, price_eur_per_mwh) to the
Biogas.SH data folder, the name config.yaml expects in
price_cases.fossil_gas.<case>.de_price_series.
"""

import argparse
from pathlib import Path

import pandas as pd
import yaml

from etrago.tools.import_data import get_data_paths


def german_market_price(run: Path) -> pd.Series:
    """Hourly price of the German AC bus in the market stage of a run."""

    market = run / "market"
    buses = pd.read_csv(market / "buses.csv", index_col=0, low_memory=False)
    de = buses.index[(buses.carrier == "AC") & (buses.country == "DE")]
    if len(de) != 1:
        raise ValueError(
            f"{market}: expected one German AC bus in the market model, "
            f"found {len(de)}."
        )

    prices = pd.read_csv(market / "buses-marginal_price.csv", index_col=0)
    snapshots = pd.read_csv(market / "snapshots.csv")["snapshot"]
    if len(snapshots) != 8760 or len(prices) != 8760:
        raise ValueError(
            f"{market}: the reference run must cover 8760 hourly snapshots."
        )

    series = prices[str(de[0])].astype(float)
    series.index = pd.to_datetime(snapshots.to_numpy())
    series.index.name = "snapshot"
    return series.rename("price_eur_per_mwh")


def gas_case(run: Path) -> str:
    resolved = yaml.safe_load((run / "resolved_config.yaml").read_text())
    return resolved["selection"]["fossil_gas_price_case"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("runs", nargs="+", type=Path, help="finished run folders")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--min-price", type=float, default=20.0)
    a = ap.parse_args()

    out_dir = a.out_dir or get_data_paths(validate=False).BIOGAS_SH_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    for run in a.runs:
        case = gas_case(run)
        series = german_market_price(run)
        path = out_dir / f"de_price_{case}.csv"
        series.to_csv(path, float_format="%.4f")
        print(
            f"{case}: JW {series.mean():.2f} EUR/MWh, "
            f"{int((series > a.min_price).sum())} h above "
            f"{a.min_price:g} EUR/MWh -> {path}"
        )


if __name__ == "__main__":
    main()
