"""Populate and validate the RFNBO CSV inputs used by eTraGo/Pyomo.

The CSV templates are deliberately conservative: missing evidence is treated as
non-compliant. This helper derives the bus-zone mapping from a PyPSA network and
builds the hourly eligibility mask from externally supplied evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pypsa


REQUIRED_EVIDENCE_COLUMNS = {
    "snapshot_position",
    "bidding_zone",
    "day_ahead_price_eur_per_mwh",
    "ets_price_eur_per_tco2",
    "ppa_additionality_ok",
    "geographical_correlation_ok",
    "temporal_correlation_ok",
}


def export_bus_zones(network, output_path: Path, zone_column: str) -> pd.DataFrame:
    """Export one bidding-zone label for every electricity-input bus."""
    if zone_column not in network.buses.columns:
        raise KeyError(
            f"network.buses has no {zone_column!r} column. Available columns: "
            f"{sorted(network.buses.columns)}"
        )

    mapping = (
        network.buses[[zone_column]]
        .rename(columns={zone_column: "bidding_zone"})
        .rename_axis("bus")
        .reset_index()
    )
    if mapping["bidding_zone"].isna().any():
        missing = mapping.loc[mapping["bidding_zone"].isna(), "bus"].tolist()
        raise ValueError(f"Bidding-zone assignment is missing for buses: {missing[:20]}")

    mapping.to_csv(output_path, index=False)
    return mapping


def validate_electrolyser_zones(network, mapping: pd.DataFrame) -> None:
    """Confirm each electrolyser inherits the zone of its electricity bus0."""
    zone_by_bus = mapping.set_index("bus")["bidding_zone"]
    carriers = network.links.carrier.astype(str).str.lower()
    electrolysers = network.links[
        carriers.str.contains("electroly", regex=False)
        | carriers.str.contains("power-to-h2", regex=False)
    ]
    missing = electrolysers.loc[~electrolysers.bus0.isin(zone_by_bus.index)]
    if not missing.empty:
        raise ValueError(
            "No bidding-zone assignment exists for these electrolyser bus0 values: "
            f"{missing.bus0.unique().tolist()}"
        )


def build_hourly_mask(evidence_path: Path, output_path: Path) -> pd.DataFrame:
    """Compute the exogenous RFNBO eligibility mask.

    Article 6's low-price condition makes temporal correlation satisfied when
    the day-ahead price is <= EUR 20/MWh OR < 0.36 times the ETS allowance
    price. It does not itself waive additionality/PPA or geographical evidence.
    """
    evidence = pd.read_csv(evidence_path)
    missing = REQUIRED_EVIDENCE_COLUMNS.difference(evidence.columns)
    if missing:
        raise ValueError(f"Hourly evidence is missing columns: {sorted(missing)}")

    duplicate = evidence.duplicated(["snapshot_position", "bidding_zone"])
    if duplicate.any():
        raise ValueError(
            "Hourly evidence contains duplicate (snapshot_position, bidding_zone) rows."
        )

    price = pd.to_numeric(
        evidence["day_ahead_price_eur_per_mwh"], errors="coerce"
    )
    ets = pd.to_numeric(evidence["ets_price_eur_per_tco2"], errors="coerce")
    evidence["low_price_threshold_eur_per_mwh"] = 0.36 * ets
    evidence["low_price_exception"] = (
        price.le(20.0) | price.lt(evidence["low_price_threshold_eur_per_mwh"])
    ).fillna(False).astype(int)

    for column in [
        "ppa_additionality_ok",
        "geographical_correlation_ok",
        "temporal_correlation_ok",
    ]:
        evidence[column] = (
            pd.to_numeric(evidence[column], errors="coerce")
            .fillna(0)
            .clip(0, 1)
            .astype(int)
        )

    effective_temporal = evidence[
        ["temporal_correlation_ok", "low_price_exception"]
    ].max(axis=1)
    evidence["rfnbo_eligible"] = (
        evidence["ppa_additionality_ok"]
        * evidence["geographical_correlation_ok"]
        * effective_temporal
    ).astype(int)
    evidence.to_csv(output_path, index=False)
    return evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--hourly-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--zone-column", default="market_zone")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    network = pypsa.Network(args.network)
    mapping = export_bus_zones(
        network,
        args.output_dir / "bus_bidding_zones.csv",
        args.zone_column,
    )
    validate_electrolyser_zones(network, mapping)
    build_hourly_mask(
        args.hourly_evidence,
        args.output_dir / "rfnbo_hourly_eligibility.csv",
    )


if __name__ == "__main__":
    main()
