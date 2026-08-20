# AUDIT_REPORT

## 1) Repository commit hash
- `0ebb6f5551b81297c5443c1546ffc15a8d847de5`

## 2) Packaging timestamp (UTC)
- `2026-08-20T17:56:56Z`

## 3) Run-classification table (all plausible discovered run folders)

| Category | Classification target | Folder(s) found | Status |
|---|---|---|---|
| 1 | Principal seven-contract (m=12) | None found | **Missing in repository snapshot** |
| 2 | Principal seven-contract (m=252) | None found | **Missing in repository snapshot** |
| 3 | 2 × 2 sensitivity experiment | `experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z` (selected), `experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T095404Z` (plausible alternate) | **Present; final-candidate ambiguity retained** |
| 4 | Training-curve experiment (m=12) | None found (`ncv_training_curve_*` not present) | **Missing** |
| 5 | Training-curve experiment (m=252) | None found (`ncv_training_curve_*` not present) | **Missing** |
| 6 | High-precision reference-price calculations | No output folder/file found (e.g., `high_precision_references.csv`) | **Missing** |
| 7 | Older/smoke/superseded/unresolved | `stage8_sensitivity_2x2_smoke_20260818T095313Z`, `stage8_sensitivity_2x2_smoke_20260818T101239Z`, plus `asian_options_*_comparison.csv` legacy files | **Present (not final dissertation principal evidence)** |

### Included run placement in this bundle
- `sensitivity_2x2/`: `stage8_sensitivity_2x2_dissertation_20260818T101252Z`
- `unresolved_candidates/`: `stage8_sensitivity_2x2_dissertation_20260818T095404Z`, both smoke runs, and legacy Stage-6 comparison CSVs.

## 4) Configuration of each principal run
No principal seven-contract Stage-8 run directory was found, so principal m=12/m=252 configuration snapshots are unavailable.

## 5) Methods and contracts present in each included run

### `stage8_sensitivity_2x2_dissertation_20260818T101252Z`
- Profile: `dissertation`
- Monitoring profiles present: `12`, `252`
- Epoch cells present: `25`, `1000` (full 2×2)
- Replications: `30`
- Sample sizes: train `5000`, validation `10000`, pilot `1000`, pricing `50000`
- Hidden width: `32`
- Contract scope: single reference contract only (`S0=100`, `K=100`, `r=0.05`, `sigma=0.2`, `T=1.0`) from `config.json`
- Methods represented in replication-level output columns: `MC`, `GCV`, `NCV`
- Not present in this run: AV rows, seven-contract target grid, frozen beta-one, frozen estimated-beta transfer rows.

### `stage8_sensitivity_2x2_dissertation_20260818T095404Z` (unresolved alternate)
- Same formal design and same non-runtime numerical metrics as the selected run.
- Runtime/projection columns differ (timing noise / rerun timing variation).

### Smoke runs
- Same 2×2 structure but smoke profile (`replications=2`, tiny sample sizes), not dissertation-scale.

## 6) Row counts for major CSVs

### Selected sensitivity run (`...101252Z`)
- `replication_level_results.csv`: 120
- `cell_summary.csv`: 4
- `paired_contrasts.csv`: 3
- `runtime_results.csv`: 120
- `checkpoint_curve_results.csv`: 480
- `seed_manifest.csv`: 300

### Required principal-output filenames (requested) availability
The following required files were **not found anywhere** in this repository snapshot:
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

## 7) NCV numerical completeness check
- Selected and alternate dissertation 2×2 runs: NCV fields are numerical in all replication rows (no `torch_not_available` rows).
- Legacy Stage-6 comparison CSVs in `unresolved_candidates/legacy_stage6_comparisons/` contain `NCV=ERROR` with `torch_not_available` messaging.

## 8) Reconciliation against dissertation values (cross-check only)

### Principal 12-date and 252-date seven-contract tables
- Cannot be reproduced from this repository snapshot because principal seven-contract run outputs are missing.

### 2×2 sensitivity expected NCV/GCV advantage cross-check
From selected run `cell_summary.csv`:
- (m=12, 25): `0.5008707163` (expected ≈ `0.501`)
- (m=12, 1000): `17.7447673892` (expected ≈ `17.745`)
- (m=252, 25): `0.1792723981` (expected ≈ `0.179`)
- (m=252, 1000): `0.1697306537` (expected ≈ `0.170`)

These match the expected approximate sensitivity values.

## 9) Mismatches and gaps found
- No principal seven-contract m=12/m=252 Stage-8 run directories found.
- No training-curve output folders found.
- No high-precision reference-price output files found.
- No principal break-even output files with the reported 609–703 values were found.
- No frozen beta-one or frozen estimated-beta replication-level output files were found.
- Available dissertation-profile run evidence is sensitivity 2×2 for the reference contract only.

## 10) Required files/results still missing
- All required principal filenames listed in section 6 are missing.
- Missing output categories: principal m=12, principal m=252, training-curve m=12, training-curve m=252, reference-price outputs.

## 11) AV contract-level data availability
- Not available for seven-contract principal tables in this snapshot.
- Only legacy single-contract Stage-6 AV entries are available (`~4.2230`, `~4.2460`), insufficient for the dissertation’s stated AV ranges across seven contracts/profiles.

## 12) Frozen beta-one and estimated-beta availability
- No replication-level frozen-reuse output files found in this repository snapshot.
- Therefore, the reported frozen estimated-beta values and “beta-one NCV/GCV < 0.016 for every target contract” claim cannot be verified from available output files.

## 13) Break-even 609–703 traceability check
- No file containing the principal seven-contract m=12 matched-accuracy break-even values (609–703) was found.
- Available 2×2 sensitivity run contains reference-contract break-even outputs only and is not a substitute for principal seven-contract evidence.

## 14) High-precision reference price + uncertainty evidence
- No high-precision reference-price output files were found in this repository snapshot.
- No file containing uncertainty diagnostics for high-precision reference prices was found.

## 15) Explicit non-rerun statement
- No experiments were rerun for this packaging task.
- Existing files were copied as-is; no result files were altered, regenerated, rounded, overwritten, moved, or deleted.

## Additional notes on ambiguity
- Two dissertation-profile 2×2 runs exist (`095404Z` and `101252Z`) with identical core statistical outputs and differing runtime/projection columns.
- Because both are plausible final sensitivity artifacts, the later run was selected under `sensitivity_2x2/` and the earlier run was preserved under `unresolved_candidates/`.
