#!/usr/bin/env python3
"""Quick check of the Biogas.SH flows in one exported stage of a run.

    python check_biogas_flows.py <run_folder>/market

Prints energy (GWh), mean and peak MW and capacity per Biogas.SH carrier,
including the biomethane offtake links (biogas_sh_offtake_*). Links are
matched by carrier, because clustering renames them to numbers.
"""

import sys
from pathlib import Path

import pandas as pd


def main(stage: Path) -> None:
    links = pd.read_csv(stage / "links.csv", index_col=0)
    links.index = links.index.astype(str)
    p0 = pd.read_csv(stage / "links-p0.csv", index_col=0)
    p0.columns = p0.columns.astype(str)
    hours = len(p0)

    sel = links[
        links.carrier.astype(str).str.contains(
            "biogas_sh|swfl_real_raw|biomethane"
        )
    ]
    flows = p0.reindex(columns=sel.index).fillna(0.0)
    table = pd.DataFrame(
        {
            "GWh": flows.sum().groupby(sel.carrier).sum() / 1e3,
            "mean_MW": flows.mean().groupby(sel.carrier).sum(),
            "p_nom_MW": sel.groupby("carrier").p_nom.sum(),
        }
    ).round(3)
    print(f"{stage}  ({hours} snapshots)\n")
    print(table.to_string())

    gens = pd.read_csv(stage / "generators.csv", index_col=0)
    gens.index = gens.index.astype(str)
    gen_p = pd.read_csv(stage / "generators-p.csv", index_col=0)
    gen_p.columns = gen_p.columns.astype(str)
    own = gens[
        gens.carrier.astype(str).str.startswith("biogas_sh")
        | (
            gens.index.str.startswith("biogas_sh")
            & (gens.carrier == "CH4_biogas")
        )
    ]
    energy = gen_p.reindex(columns=own.index).fillna(0.0).sum() / 1e3
    print("\nBiogas.SH generators (GWh)")
    print(energy.groupby(own.carrier).sum().round(3).to_string())

    offtake = table[table.index.str.startswith("biogas_sh_offtake")]
    if not offtake.empty:
        share = offtake.mean_MW.sum() / offtake.p_nom_MW.sum()
        print(f"\nOfftake: {offtake.GWh.sum():.1f} GWh, "
              f"{share:.0%} of the caps on average")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
