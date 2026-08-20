# AUDIT REPORT

## 1) Packaging metadata
- Repository commit hash: `0ebb6f5551b81297c5443c1546ffc15a8d847de5`
- Packaging timestamp (UTC): `2026-08-20 17:57:57 UTC`
- No experiments were rerun during this packaging process.

## 2) Run-classification table
| Category | Folder/file | Classification | Evidence note |
|---|---|---|---|
| Principal seven-contract (m=12) | `Not found in repository` | missing | No stage8 principal m=12 seven-contract output folder or required principal CSV set found. |
| Principal seven-contract (m=252) | `Not found in repository` | missing | No stage8 principal m=252 seven-contract output folder or required principal CSV set found. |
| 2x2 sensitivity experiment | `experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z` | included as primary | Matches dissertation profile: 30 reps, train=5000, pilot=1000, pricing=50000, m∈{12,252}, epochs∈{25,1000}, validation passed. |
| (m=12) training-curve experiment | `Not found in repository` | missing | No training-curve output folder or training-curve validation report found. |
| (m=252) training-curve experiment | `Not found in repository` | missing | No training-curve output folder or training-curve validation report found. |
| High-precision reference-price calculations | `Not found in repository` | missing | No reference_precision_diagnostics.csv or standalone high-precision reference-price outputs found. |
| Older/smoke/superseded/unresolved | `experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T095404Z; experiment_runs/stage8_sensitivity_2x2_smoke_20260818T095313Z; experiment_runs/stage8_sensitivity_2x2_smoke_20260818T101239Z; asian_options_*_comparison.csv` | included in unresolved_candidates | Older duplicate dissertation 2x2 run and smoke/single-contract outputs retained for auditability. |

## 3) Principal-run configuration status
- Principal seven-contract m=12 run: **not found**.
- Principal seven-contract m=252 run: **not found**.
- Therefore the expected seven-contract method grid (MC, AV, GCV, contract-specific NCV, frozen β=1 NCV, frozen estimated-β NCV) cannot be fully audited from repository-resident outputs.

## 4) Included primary 2×2 sensitivity run details
- Folder: `experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z`
- Created-at (config): `20260818T101252Z`
- Profile: `dissertation`; replications: `30`; train/validation/pilot/pricing paths: `5000` / `10000` / `1000` / `50000`
- Hidden width: `32`; checkpoint grid: `[0, 10, 25, 50, 100, 200, 500, 1000]`
- Formal design cells: `[{'monitoring_dates': 12, 'ncv_epoch': 25}, {'monitoring_dates': 12, 'ncv_epoch': 1000}, {'monitoring_dates': 252, 'ncv_epoch': 25}, {'monitoring_dates': 252, 'ncv_epoch': 1000}]`
- Validation passed: `True`; errors: `[]`; warnings: `[]`
- Contracts present: reference contract only (single contract config)
- Methods present in outputs: MC, GCV, NCV (implied via ncv_* columns)

### Row counts (primary 2×2 run major CSVs)
- `replication_level_results.csv`: 120 rows
- `cell_summary.csv`: 4 rows
- `paired_contrasts.csv`: 3 rows
- `runtime_results.csv`: 120 rows
- `checkpoint_curve_results.csv`: 480 rows
- `seed_manifest.csv`: 300 rows

### NCV numerical-completeness check (primary 2×2 run)
- Numeric NCV rows in `replication_level_results.csv`: `120/120`
- No `torch_not_available` errors in this run output.

## 5) Dissertation cross-check reconciliation (using existing files only)
### 2×2 expected NCV/GCV advantages
| Cell (m, epoch) | Observed advantage_geometric_mean | Expected approx | Difference |
|---|---:|---:|---:|
| (12, 25) | 0.500871 | 0.501 | -0.000129 |
| (12, 1000) | 17.744767 | 17.745 | -0.000233 |
| (252, 25) | 0.179272 | 0.179 | +0.000272 |
| (252, 1000) | 0.169731 | 0.170 | -0.000269 |

### Principal 12-date and 252-date seven-contract tables
- Not reproducible from repository-resident outputs because the corresponding principal seven-contract run folders/files are missing.

### AV checks for main contract tables
- Contract-level AV values for the seven-contract principal tables are **not available** in repository-resident outputs.
- Only older single-contract comparison files exist, and these are included under `unresolved_candidates/single_contract_candidates/`.

### Frozen estimated-β and β=1 checks
- Replication-level frozen estimated-β target evidence for the reported values is **not found** in repository-resident outputs.
- Frozen β=1 target-wise NCV/GCV evidence for the claim “below 0.016 for every target contract” is **not found** in repository-resident outputs.

### Break-even checks
- The specific seven-contract m=12 matched-accuracy break-even values (609–703 by contract) are **not traceable** to repository-resident files.
- In the available 2×2 reference-contract run, break-even is finite only for (m=12, epoch=1000) and mostly non-finite otherwise; this is a different experiment and was not substituted for the missing seven-contract principal break-even evidence.
- For m=252 in the available 2×2 run, break-even is consistently non-finite with `proposed_marginal_not_below_baseline_no_finite_break_even`, supporting the qualitative non-finite-break-even conclusion for that run design.

## 6) Missing required evidence files (not found anywhere in repository)
- `per_replication_results.csv`
- `aggregate_statistical_results.csv`
- `aggregate_results.csv`
- `per_replication_variance_ratios.csv`
- `variance_ratio_summary.csv`
- `transfer_diagnostics.csv`
- `transfer_diagnostics_summary.csv`
- `shared_reference_training.csv`
- `runtime_raw.csv`
- `runtime_summary.csv`
- `equal_pricing_observations_summary.csv`
- `equal_budget_projected_results.csv`
- `equal_budget_empirical_results.csv`
- `matched_accuracy_results.csv`
- `break_even_equal_observations.csv`
- `break_even_matched_accuracy.csv`
- `break_even_fixed_accuracy.csv`
- `portfolio_break_even.csv`
- `reference_precision_diagnostics.csv`

## 7) High-precision reference prices and uncertainty
- No standalone high-precision reference-price output files were found in this repository snapshot (e.g., no `reference_precision_diagnostics.csv` or equivalent run output).
- Relevant code paths are included in `code_snapshot/asian_options/frozen_transfer.py` and `code_snapshot/scripts/run_stage8.py`, but output evidence files are absent.

## 8) Included unresolved candidates explanation
- `stage8_sensitivity_2x2_dissertation_20260818T095404Z`: earlier dissertation-profile 2×2 run; included due ambiguity over which duplicate run was final.
- `stage8_sensitivity_2x2_smoke_*`: smoke-test runs; included as superseded, non-principal evidence.
- `asian_options_*_comparison.csv`: older single-contract summaries with NCV error rows; included for transparency only.

## 9) Integrity checks during packaging
- Source-vs-copied hash mismatches: `0`
- Copy operation preserved original files; no source file content was edited, regenerated, rounded, or overwritten.
- Bundle excludes virtual environments, caches, credentials, and dependency folders.
