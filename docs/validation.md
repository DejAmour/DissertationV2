# Stage 7 Validation Workflow

Run the validation profile:

```bash
python scripts/run_experiments.py \
  --output-dir /tmp/stage7_runs \
  --profile validation_minimal
```

## What Stage 7 adds

- Multi-config validation matrix (`validation_minimal`) across:
  - option difficulty (low/high volatility + short/long maturity),
  - budget scale (small/large),
  - deterministic seeds/replications.
- Validation checks with actionable failures:
  - finite/non-negative checks,
  - budget consistency (MC/AV/CV/NCV),
  - mode/schema NA rules,
  - MC reference consistency with SE-aware tolerance,
  - VRR/efficiency formula recomputation checks.
- Aggregation outputs:
  - `validation_aggregate.csv`
  - `validation_aggregate.md`

## CI and formula definitions

- `variance_reduction_ratio = MC_observation_variance / method_observation_variance`
- `efficiency_gain_vs_mc = (MC_estimator_variance * MC_runtime_seconds) / (method_estimator_variance * method_runtime_seconds)`
- 95% CI for aggregate means:
  - `mean ± 1.96 * sample_std / sqrt(n)`
  - if `n < 2`, CI is `NA` with explanation.

## Mode interpretation guidance

- **Mode A** (`A_equal_obs`): compare estimators under equal pricing observations.
- **Mode B** (`B_equal_budget`): compare under equal total simulated-path budget.
- **Mode C** (`C_runtime`): compare runtime/efficiency metrics.
- NCV timing caveat: runtime interpretation depends on `--timing-scope-policy`.

## Troubleshooting

- Missing torch: NCV execution may fail; verify `torch` installation in `asian_options/requirements.txt`.
- Invalid budgets: ensure Mode B constraints (`total_path_budget - pilot_paths > 0`, `total_path_budget - training_paths > 0`).
- Nondeterminism: keep fixed seeds and avoid changing Python/PyTorch/hardware stack between runs.

## Safe claims

- Supported:
  - relative variance reduction (`variance_reduction_ratio`),
  - combined runtime+precision efficiency (`efficiency_gain_vs_mc`) under explicit timing scope,
  - reproducibility of bundle artifacts under fixed config/seeds/environment.
- Not supported:
  - treating variance reduction as runtime speedup,
  - extrapolating results beyond tested config matrix without new runs.
