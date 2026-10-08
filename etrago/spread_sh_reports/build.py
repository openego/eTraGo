"""Build the SPREAD.SH bidding-zone HTML reports.

Usage (from the ``etrago`` directory)::

    python -m spread_sh_reports                       # results_071026 -> results_071026/html_reports
    python -m spread_sh_reports --results results_071026 --out reports_071026
    python -m spread_sh_reports --plotly-js plotly.min.js   # fully offline pages
"""

import argparse
import logging
import time
import warnings
from pathlib import Path

from .config import DEFAULT_DATA_DIR, DEFAULT_RESULTS_DIR, SCENARIO_LABELS, discover_scenarios
from .metrics import ScenarioMetrics
from .report import ComparisonReport, ScenarioReport
from .side_by_side import SideBySideReport

logger = logging.getLogger("spread_sh_reports")


def build(results_dir=DEFAULT_RESULTS_DIR, out_dir=None, data_dir=DEFAULT_DATA_DIR,
          plotly_js=None, only=None):
    results_dir = Path(results_dir)
    out_dir = Path(out_dir) if out_dir else results_dir / "html_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    js = Path(plotly_js).read_text(encoding="utf-8") if plotly_js else None

    scenarios = discover_scenarios(results_dir)
    if only:
        scenarios = [s for s in scenarios if s.key in only]
    if not scenarios:
        raise SystemExit(f"No result directories with grid_optimization found in {results_dir}")

    metrics = {}
    for sc in scenarios:
        t = time.time()
        logger.info("Computing metrics for %s (%s)", sc.key, sc.path.name)
        metrics[sc.key] = ScenarioMetrics(sc, data_dir).compute_all()
        logger.info("  done in %.1f s", time.time() - t)

    files = {k: f"report_{k}.html" for k in metrics}
    multi = len(metrics) > 1

    def nav(current):
        items = ([("Comparison", "index.html", current == "index"),
                  ("Side by side", "side_by_side.html", current == "sbs")] if multi else [])
        items += [(k if k != "status_quo" else "Status quo", files[k], current == k) for k in metrics]
        return items

    written = []
    for k, m in metrics.items():
        page = ScenarioReport(m, metrics, nav(k), js).build()
        path = out_dir / files[k]
        path.write_text(page, encoding="utf-8")
        written.append(path)
        logger.info("Wrote %s (%.1f MB)", path, path.stat().st_size / 1e6)
    if multi:
        page = ComparisonReport(metrics, nav("index"), files, js).build()
        path = out_dir / "index.html"
        path.write_text(page, encoding="utf-8")
        written.append(path)
        logger.info("Wrote %s (%.1f MB)", path, path.stat().st_size / 1e6)
        page = SideBySideReport(metrics, nav("sbs"), files, data_dir, js).build()
        path = out_dir / "side_by_side.html"
        path.write_text(page, encoding="utf-8")
        written.append(path)
        logger.info("Wrote %s (%.1f MB)", path, path.stat().st_size / 1e6)
    return written


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default=str(DEFAULT_RESULTS_DIR),
                   help="directory containing one results_<zone>_... folder per configuration")
    p.add_argument("--out", default=None, help="output directory (default: <results>/html_reports)")
    p.add_argument("--data", default=str(DEFAULT_DATA_DIR), help="eTraGo data directory with the shapefiles")
    p.add_argument("--plotly-js", default=None,
                   help="path to a local plotly.min.js to inline (offline use); default loads it from the CDN")
    p.add_argument("--only", nargs="*", choices=list(SCENARIO_LABELS), help="restrict to some configurations")
    p.add_argument("-q", "--quiet", action="store_true")
    a = p.parse_args(argv)
    logging.basicConfig(level=logging.WARNING if a.quiet else logging.INFO,
                        format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    for path in build(a.results, a.out, a.data, a.plotly_js, a.only):
        print(path)


if __name__ == "__main__":
    main()
