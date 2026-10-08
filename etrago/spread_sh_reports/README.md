# spread_sh_reports

Interactive HTML reports for the SPREAD.SH bidding-zone runs of eTraGo
(`status_quo`, `DE2` … `DE5`). The tool reads the CSV exports in each
`results_<zone>_…` folder. It writes one report per configuration plus a
comparison page.

## Usage

Run from the `etrago/` directory, with the eTraGo conda environment active
(it needs only pandas, numpy, geopandas and shapely):

```bash
python -m spread_sh_reports                      # results_071026 -> results_071026/html_reports
python -m spread_sh_reports --results results_071026 --out results_071026/html_reports
python -m spread_sh_reports --only DE3 DE5       # subset
python -m spread_sh_reports --plotly-js plotly.min.js   # inline plotly.js for offline use
```

Open `html_reports/index.html` (comparison) or `html_reports/side_by_side.html`
(every configuration next to each other). The pages load plotly.js 2.35.2 from
`cdn.plot.ly`. Pass `--plotly-js` with a local copy for machines without
internet access. A full build of the five runs takes about 20 s.

URL options: `?all` renders every chart at once (for printing and
screenshots). On the side-by-side page, `?extent=sh` opens all German maps
zoomed to Schleswig-Holstein. `?theme=dark` or `?theme=light` forces a theme. Otherwise the
page follows the system setting, and the header button cycles
system → light → dark.

## Input layout

```
results_071026/
  results_status_quo_ac100_focus50_s8760_…/
    args.json  busmap_within_focus.csv
    pre_market_optimization/  market_optimization/  grid_optimization/
  results_DE2_…/ …
```

Bidding-zone shapes come from `data/shapes_biddingzones` (the same files used
by `etrago.tools.market_zones`). The detailed outline of Germany and
Schleswig-Holstein comes from `data/shapes_focus_region/vg250_focus_region.geojson`,
and neighbouring countries from `data/shapes_europe`.

## Report contents

| Section | Main content |
|---|---|
| Overview | KPI tiles with change vs. status quo, auto-generated key observations, data notes |
| Bidding zones | zone map with clustered AC nodes, zonal generation mix vs. demand, zone table |
| Electricity prices | zonal price map, price-duration curves, monthly heatmap, DE price spread, daily profile, nodal shadow prices |
| Cross-zonal exchange | corridor map (capacity, congestion share), net positions, flow-duration curves, congestion rent |
| Redispatch | node map, by technology / zone, weekly series, top nodes, cost |
| Curtailment | node map, market vs. after-redispatch by technology / zone / month, duration curve |
| Flexibility | use of batteries, pumped hydro, electrolysis, heat pumps, BEV, DSM, H2-to-power; prices captured; electrolyser map |
| Batteries | node map, scenario minimum vs. added capacity, month × hour dispatch heatmap |
| Grid | expansion / mean-loading / hours-at-limit maps, expansion by line type and zone, loading distribution and duration, most congested lines |
| Costs | breakdown, change vs. status quo, wholesale cost per zone |
| Schleswig-Holstein | regional KPIs, zoomed map (loading, curtailment, electrolysers), table of the 50 SH nodes |

The comparison page (`index.html`) puts the KPIs, zone layouts, prices,
redispatch, curtailment, grid expansion (including line-by-line change maps
vs. status quo), flexibility, costs and SH indicators side by side.

### Side-by-side page (`side_by_side.html`)

* **KPI matrix:** 20 indicators × 5 configurations. Cells are coloured by the
  change against the status quo (blue = better, red = worse, saturating at
  ±15 %), and the values carry their difference to the status quo.
* **KPI charts:** one small dot chart per indicator, with each configuration
  in the same colour on every chart.
* **Map matrix:** one row per topic, one column per configuration. Rows:
  zone layout, zonal price, nodal shadow price, redispatch, curtailment,
  electrolysers, batteries, line loading, hours at limit, grid expansion,
  change in expansion vs. status quo, and cross-zonal corridors. Every map in
  a row shares the same colour and size scale. Panning or zooming one map
  moves its whole row, and the header switch zooms all German maps to
  Schleswig-Holstein. Each map has the row's KPI and its change vs. status
  quo underneath.

## Method notes

* Snapshots are 5-hour steps with weight 5, so annual sums cover 8760 h.
* Grid nodes are assigned to zones by point-in-polygon, with the nearest zone
  for German nodes outside every polygon (same rule as eTraGo). Market buses
  are mapped to zones through the generators they share with the grid model.
  This matters because the exported market bus labelled `LU` actually carries
  the German zone that contains Luxembourg.
* Redispatch and redispatch cost follow `calc_etrago_results`
  (`ramp_up` / `ramp_down` generators and gas-turbine links, p × marginal_cost × weight).
* Curtailment = p_nom × p_max_pu (market model) − infeed. The grid-stage infeed
  is the market dispatch plus ramp_up and ramp_down of the same unit.
* Line loading = |p0| / (s_nom_opt × s_max_pu(t)). A line counts as congested
  at ≥ 98 % loading.
* Investments are annualised capital costs:
  (opt − min) × capital_cost for extendable components; electrolysers use
  p_nom × capital_cost, since they are sized in the pre-market model and fixed in the grid model.
* Natural-gas imports (`CH4_NG`) are reported separately from the other
  dispatch cost. In the market export, CH4 store dispatch is not consistent
  with the stored energy over the year (several hundred TWh of net discharge
  with an unchanged state of charge). Gas imports absorb the difference and
  vary between runs for reasons unrelated to the zones.

## Code layout

| File | Role |
|---|---|
| `config.py` | scenario discovery, carrier groups, colour roles |
| `loader.py` | lazy CSV access per optimisation stage |
| `geo.py` | zone polygons and labels, bus→zone assignment, map outlines |
| `metrics.py` | all calculations (`ScenarioMetrics`) |
| `charts.py` | Plotly figure specs as plain dicts, with theme colour tokens |
| `html.py` | page template, CSS, chart runtime (lazy rendering, light/dark, data tables) |
| `report.py` | scenario report and comparison page |
| `side_by_side.py` | KPI matrix and map matrix page |
| `build.py` | CLI |
