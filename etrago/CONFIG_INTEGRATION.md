# Biogas.SH scenario configuration v3

Frequently changed scenario assumptions live in `config.yaml` instead of the
large `args` dictionary in `appl.py`. Static topology, component names,
carriers, and file paths stay in `appl.py` and in the `technical` section of
the YAML file.

The configuration covers fossil-gas prices, the Biogas.SH support regime
(post-EEG, follow-on EEG, flexibilisation), the biomethane cost, biogenic CO2
sales, planned SWFL heat pumps, SWFL unit availability, biomethane eligibility
at SWFL, and the Biogas.SH utilisation routes (onsite CHP, biomethane to the
public CH4 grid, direct raw biogas to SWFL, and biomethane for HGV transport).

## Files

| File | Purpose |
| --- | --- |
| `config.yaml` | Selection, batch list, run settings, case definitions, technical assumptions, scenario matrix. |
| `scenario_config.py` | Validation, environment overrides, scenario resolution, updates to `args`, network-price application, summaries, matrix expansion, CLI. |
| `appl.py` | eTraGo entry point; loads `config.yaml` from its own directory. |
| `tools/swfl_real_system.py` | Detailed Stadtwerke Flensburg (SWFL) plant, loads, and heat-pump handling. |
| `tools/biogas_sh.py` | Biogas.SH plants, routes, central storage, raw-biogas and transport routes. |
| `tools/constraints.py` | `biogas_sh_resource` and `biogas_sh_support` extra-functionality constraints. |
| `cluster/gas.py` | Keeps custom Biogas.SH / SWFL CH4 buses out of gas clustering. |
| `run_scenario_matrix.py` | Runs every `scenario_matrix` combination sequentially. |
| `scenario_matrix.csv` | Generated factorial table (written by the `matrix` command). |
| `resolved_config.yaml` | Effective configuration written beside each result export. |

## Pipeline in `appl.py`

`run_etrago()` performs these steps in order. The order matters: prices must
be applied after all custom generators exist, and the transport route must be
added after gas clustering.

```text
 1. load_and_apply_config(args, config.yaml)   -> args, resolved_scenario
    print(scenario_summary(resolved_scenario))
 2. Etrago(args)
    build_network_from_db()
    adjust_network()
 3. apply_swfl_real_system(network, args["swfl_real_system"])
 4. if Biogas.SH inactive: disable biogas_sh_support / biogas_sh_resource
 5. apply_biogas_sh_assets(etrago)
 6. if Biogas.SH inactive and SWFL active:
      keep public-grid CH4 supply 47538 -> biogas_sh_swfl_ch4_bus
      remove swfl_real_biomethane_ch4_bus and its links
 7. fix_custom_component_scn_names(network)
 8. apply_network_price_scenario(network, resolved_scenario, biogas_sh_active)
 9. validate_biogas_sh_storage_topology(...)        (Biogas.SH active only)
10. remove_known_legacy_swfl_heat_pump_before_clustering(...)  (SWFL active)
11. spatial_clustering()
12. purge_legacy_swfl_heat_pumps(stage="after spatial clustering")
      (SWFL active and future_heat_pumps.active)
13. spatial_clustering_gas()
14. apply_biogas_sh_transport_route(etrago)
15. snapshot_clustering()
    skip_snapshots()
16. consistency_check()
    restore_load_shedding_after_clustering(etrago, negative_load_shedding=("Li_ion",))
    consistency_check()
17. optimize()
    temporal_disaggregation()
    pf_post_lopf()
    spatial_disaggregation()
    calc_results()
18. write <csv_export>/resolved_config.yaml
    export <csv_export>/network.nc
```

EHV clustering (`network_clustering_ehv`) is disabled and is not called in
this pipeline.

## Biogas.SH routes

All active routes compete for one regional raw-biogas resource; the optimiser
allocates it.

```text
raw biogas (Biogas.SH plants)
 ├── onsite CHP electricity                 (eta_el      = 0.38)
 ├── onsite heat                            (eta_heat    = 0.45)
 ├── upgrading -> plant CH4 bus             (eta_upgrade = 0.96)
 │       └── biogas_sh_storage_ch4_bus <-> biogas_sh_ch4_store
 │               ├── -> public CH4 grid (bus 47538)
 │               ├── -> swfl_real_biomethane_ch4_bus   (legacy route)
 │               └── -> HGV transport-energy bus       (transport route)
 └── direct raw biogas -> swfl_real_raw_biogas_bus -> K12 / K13
                                            (eta_raw_swfl = 1.00)
```

### Transport route

`apply_biogas_sh_transport_route()` runs after gas clustering. In the
configured MV grid districts (`transport_biomethane.eligible_mv_grid_ids`) it
moves only `H2_hgv_load` loads from their H2 bus to a dedicated
`biogas_sh_hgv_transport_energy_*` bus, which can be supplied by an H2 link
and by a biomethane link from `biogas_sh_storage_ch4_bus`. Other H2
components (industry demand, storage, electrolysis, pipelines) are untouched.

### Optimisation constraints

Both are registered in `args["extra_functionality"]` by
`apply_config_to_args()`, and the run uses the linopy formulation
(`_biogas_sh_*_linopy`). Unlike standard eTraGo constraints, a failing
`biogas_sh_*` constraint stops the run instead of logging a warning.

- `biogas_sh_resource` limits the shared raw-biogas use:

  ```text
  onsite electricity / 0.38
  + onsite heat / 0.45
  + upgraded biomethane / 0.96
  + direct raw biogas to SWFL / 1.00
  <= available regional raw biogas
  ```

- `biogas_sh_support` (only when the support case has `eeg_active: true`)
  splits CHP output into a merchant and a supported tranche that share the
  same physical capacity, and caps supported output at
  `supported_hours_per_year` full-load hours.

## Normal use

Normally edit only `selection`, `batch`, and, when needed, `run`:

```yaml
selection:
  fossil_gas_price_case: "crisis_2035"
  biomethane_price_case: "project_92_9"
  co2_sale_case: "medium_60"
  support_case: "followon_eeg_flex"
  heat_pump_case: "two"
  swfl_unit_case: "all_operational"
  biomethane_use_case: "off"
  biogas_route_case: "hybrid_raw_swfl_transport"

run:
  start_snapshot: 1
  end_snapshot: 8760
  ac_clusters: 50
  result_name_template: "{scenario}_{hours}h_{ac_clusters}ac"
```

Use `null` for a `run` value to keep the value already defined in `appl.py`.

In the core analysis only `fossil_gas_price_case` and `support_case` vary.
The route is always `hybrid_raw_swfl_transport`, so zero flow on a route is an
optimisation result, not a missing connection.

## Available cases

### `fossil_gas_price_case`

`CH4_NG` marginal cost = gas commodity price + CO2 price × 0.201 tCO2/MWh_fuel.

| Case | Gas [EUR/MWh] | CO2 [EUR/t] | Final `CH4_NG` [EUR/MWh_fuel] |
| --- | ---: | ---: | ---: |
| `low_2035` | 13.0 | 50.0 | 23.0500 |
| `legacy_egon` | 25.6 | 76.5 | 40.9765 |
| `high_2035` | 34.125 | 144.0 | 63.0690 |
| `crisis_2035` | 60.0 | 144.0 | 88.9440 |

### `support_case`

| Case | EEG | Supported h/a | Flexibilisation | CHP capacity multiplier |
| --- | --- | ---: | --- | ---: |
| `post_eeg` | no | 0 | no | 1.0 |
| `followon_eeg_no_flex` | yes | 2920 | no | 1.0 |
| `followon_eeg_flex` | yes | 2920 | yes (100 EUR/kW/a payment) | 3.0 |

With flexibilisation, electrical CHP capacity is tripled while the annual
raw-biogas availability stays unchanged.

### `biomethane_price_case`

| Case | Base biomethane cost [EUR/MWh_Hs] |
| --- | ---: |
| `project_92_9` | 92.9 (all-in: raw biogas, collection, upgrading, storage) |
| `debug_10` | 10.0 (debugging only) |

### `co2_sale_case`

Revenue from separated biogenic CO2 is credited against the biomethane cost:

```text
effective biomethane cost
= base biomethane cost
- (sale price - additional cost) × marketable fraction
  × 0.14981 tCO2/MWh_biomethane
```

The credit is applied only when the case has `active: true`.

| Case | Sale price [EUR/t] | Effective cost with `project_92_9` [EUR/MWh_Hs] |
| --- | ---: | ---: |
| `none` | 0 | 92.9000 |
| `low_50` | 50 | 85.4095 |
| `medium_60` | 60 | 83.9114 |
| `high_144` | 144 | 71.3274 |

The credit does not apply to the direct raw-biogas route. A negative effective
cost is rejected.

### Onsite prices (`price_cases.onsite`, not a selection)

| Parameter | Value |
| --- | ---: |
| Raw biogas cost | 75.00 EUR/MWh_Hs |
| Merchant onsite electricity | 197.37 EUR/MWh_el |
| Supported onsite electricity | 110.31 EUR/MWh_el |
| EEG premium | 87.06 EUR/MWh_el |
| Onsite heat | 166.67 EUR/MWh_th |

### `heat_pump_case`

| Case | Active planned heat pumps |
| --- | --- |
| `none` | none (legacy eGon heat pump is still removed) |
| `one_gwp1` | `swfl_gwp_1` |
| `one_gwp2` | `swfl_gwp_2` |
| `two` | `swfl_gwp_1`, `swfl_gwp_2` |

### `swfl_unit_case`

| Case | Gas-to-power | Boilers | Resistive heaters |
| --- | --- | --- | --- |
| `all_operational` | yes | K5, K11, K12, K13 | EHK1, EHK2 |
| `gas_units_only` | yes | K5, K11, K12, K13 | — |
| `k12_k13_plus_gas_to_power` | yes | K12, K13 | — |
| `k12_k13_only` | no | K12, K13 | — |
| `electric_heat_only` | no | — | EHK1, EHK2 |

### `biomethane_use_case`

Which SWFL units may burn upgraded biomethane from the SWFL biomethane bus.

| Case | Eligible units |
| --- | --- |
| `off` | none |
| `k12_k13_only` | K12, K13 |
| `all_gas_units` | K5, K11, K12, K13 |

### `biogas_route_case`

| Case | Onsite | Storage → grid | Storage → SWFL biomethane | Raw biogas → SWFL | HGV transport |
| --- | :-: | :-: | :-: | :-: | :-: |
| `onsite` | ✓ | | | | |
| `storage_to_grid` | | ✓ | | | |
| `storage_to_swfl` (legacy) | | | ✓ | | |
| `hybrid` (legacy) | ✓ | ✓ | ✓ | | |
| `hybrid_raw_swfl_transport` (main) | ✓ | ✓ | | ✓ | ✓ |

## Validate and inspect

Use a conda environment that has `pyyaml` (for example `etrago-biogas`).
Run these checks before starting eTraGo:

```bash
python scenario_config.py config.yaml validate
python scenario_config.py config.yaml show
```

For the selection above, `show` reports among others:

```text
Scenario name:                  followon_eeg_flex__crisis_2035__project_92_9__medium_60__two__all_operational__off__hybrid_raw_swfl_transport
Final CH4_NG marginal cost:     88.9440 EUR/MWh_fuel
Biomethane marginal cost:       83.9114 EUR/MWh_Hs
Route case:                     hybrid_raw_swfl_transport
Storage to public gas grid:     yes
Storage to SWFL:                no
Regional resource constraint:   yes
```

## Scenario names and result directories

The scenario name joins the eight selected cases in this order:

```text
support_case__fossil_gas_price_case__biomethane_price_case__co2_sale_case
__heat_pump_case__swfl_unit_case__biomethane_use_case__biogas_route_case
```

`run.result_name_template` sets `args["csv_export"]` and overrides the value
in `appl.py`. With the default template, the result directory (relative to
`etrago/`) is, for example:

```text
followon_eeg_flex__crisis_2035__project_92_9__medium_60__two__all_operational__off__hybrid_raw_swfl_transport_8760h_50ac
```

Hours are `end_snapshot - start_snapshot + 1`. Each directory contains the
eTraGo CSV export, `resolved_config.yaml`, and `network.nc`.

`run.ac_clusters` only updates
`args["network_clustering"]["electricity_grid"]["n_clusters"]`; the rest of the
clustering dictionary is kept.

For a genuine consecutive short test, keep snapshot clustering disabled and
set `"skip_snapshots": False` in `appl.py`.

## Single run

```bash
python -u appl.py 2>&1 | tee "results_biogas_SH_$(date +%Y%m%d_%H%M%S).log"
```

## Environment-variable overrides

Any selection value can be overridden for one run:

```bash
ETRAGO_SUPPORT_CASE=post_eeg \
ETRAGO_FOSSIL_GAS_PRICE_CASE=high_2035 \
python -u appl.py
```

| Variable | Selection key |
| --- | --- |
| `ETRAGO_FOSSIL_GAS_PRICE_CASE` | `fossil_gas_price_case` |
| `ETRAGO_SUPPORT_CASE` | `support_case` |
| `ETRAGO_BIOMETHANE_PRICE_CASE` | `biomethane_price_case` |
| `ETRAGO_CO2_SALE_CASE` | `co2_sale_case` |
| `ETRAGO_HEAT_PUMP_CASE` | `heat_pump_case` |
| `ETRAGO_SWFL_UNIT_CASE` | `swfl_unit_case` |
| `ETRAGO_BIOMETHANE_USE_CASE` | `biomethane_use_case` |
| `ETRAGO_BIOGAS_ROUTE_CASE` | `biogas_route_case` |

Unset these in your shell before normal runs; leftover values silently change
the selected scenario.

## Multi-scenario runs

`run_scenario_matrix.py` runs every combination in `scenario_matrix` and
starts one `appl.py` process per scenario, so runs do not share state. Each
scenario is passed to its run through the eight `ETRAGO_*` variables; any
values already set in the shell are cleared first.

The runner requires `batch.enabled: true` and honours `batch.stop_on_error`.
The `batch.scenarios` list in `config.yaml` is not used.

`scenario_matrix.dimensions` defines the factorial design. The current matrix
contains 3 support cases × 3 fossil-gas cases (`legacy_egon`, `high_2035`,
`crisis_2035`) with all other dimensions fixed, giving 9 scenarios.
Dimensions not listed in the matrix take their value from `selection`.

```bash
# write the combinations to CSV only (does not run eTraGo)
python scenario_config.py config.yaml matrix --output scenario_matrix.csv

# run all combinations; per-run logs go to batch_logs/NN_<scenario>.log
python -u run_scenario_matrix.py
```

## Important `args` mappings

`apply_config_to_args()` updates:

```python
args["swfl_real_system"]                       # units, heat pumps, loads, buses
args["biogas_sh"]                              # routes, prices, support,
                                               # swfl_direct, gas_storage,
                                               # transport_biomethane, raw-biogas route
args["extra_functionality"]["biogas_sh_resource"]
args["extra_functionality"]["biogas_sh_support"]   # removed when EEG is inactive
args["biogas_sh_scenario_name"]
args["start_snapshot"], args["end_snapshot"], args["csv_export"]
args["network_clustering"]["electricity_grid"]["n_clusters"]
```

## Technical assumptions (`technical`)

Rarely changed; edit only for deliberate model changes.

- `technical.swfl.buses`: AC `33935`, heat `swfl_real_central_heat_bus`,
  natural gas `biogas_sh_swfl_ch4_bus`, biomethane
  `swfl_real_biomethane_ch4_bus`, raw biogas `swfl_real_raw_biogas_bus`,
  public CH4 `47538`.
- `technical.biogas_sh.efficiencies`: onsite electricity 0.38, onsite heat
  0.45, upgrading 0.96.
- `technical.biogas_sh.direct_raw_biogas_to_swfl`: delivered cost =
  raw biogas 75.00 + transport 8.78 = 83.78 EUR/MWh_Hs; eligible units K12,
  K13. No upgrading efficiency and no CO2 credit are applied.
- `technical.biogas_sh.transport_biomethane`: HGV route settings; the THG
  quota credit is controlled by `thg_quota_active` (currently `false`).
- `technical.biogas_sh.storage`: 500 MWh central biomethane store (cyclic),
  50 MW output links to the public grid and to SWFL.
- `technical.biogas_sh.public_grid_to_swfl`: 1500 MW natural-gas link
  `47538 -> biogas_sh_swfl_ch4_bus`.
- `technical.biogas_sh.resource_constraint.active`: must stay `true` whenever
  more than one route uses raw biogas, otherwise raw biogas is double counted.

## Price accounting

Fuel and CO2 costs are assigned upstream:

- `apply_network_price_scenario()` sets the selected cost on all `CH4_NG`
  generators and the effective biomethane cost on the custom `CH4_biogas`
  generators.
- SWFL gas-to-power, boiler, reserve-boiler, storage, and routing links must
  not repeat the gas commodity, CO2, or biomethane infrastructure cost.
- Keep `swfl_import_adder_eur_per_mwh_fuel: 0.0` unless a separate SWFL
  transport or network charge is intentionally modelled.

## Running without Biogas.SH

When `args["biogas_sh"]["active"]` is false, `appl.py` disables the
`biogas_sh_support` and `biogas_sh_resource` constraints, skips the storage
validation, keeps the public-grid CH4 supply to SWFL, and removes the unused
SWFL biomethane bus and its links. SWFL itself is still modelled when
`args["swfl_real_system"]["active"]` is true.
