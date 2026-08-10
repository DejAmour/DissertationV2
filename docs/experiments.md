# Stage 6 Experiments: Reporting, Semantics, and Reproducibility

## Canonical runner

Use `scripts/run_experiments.py` for all Stage 6 experiment packaging.

```bash
python scripts/run_experiments.py \
  --output-dir experiment_runs \
  --modes A B C \
  --seeds 42 \
  --replications 1 \
  --n-paths 50000 \
  --total-path-budget 50000 \
  --pilot-paths 1000 \
  --training-paths 5000 \
  --timing-scope-policy exclude_ncv_training
```

Minimal deterministic smoke run:

```bash
python scripts/run_experiments.py \
  --output-dir experiment_runs \
  --modes A B C \
  --seeds 7 \
  --replications 1 \
  --n-paths 50 \
  --total-path-budget 80 \
  --pilot-paths 10 \
  --training-paths 20 \
  --timing-scope-policy exclude_ncv_training
```

## Comparison modes (do not mix semantics)

- **Mode A (`A_equal_obs`)**: equal pricing observations for all methods.
- **Mode B (`B_equal_budget`)**: equal total simulated-path budget; pilot/training paths consume budget.
- **Mode C (`C_runtime`)**: runtime/efficiency metrics with explicit timing-scope policy.

## Metric definitions (copyable)

- `observation_variance = Var_ddof1(corrected_observations)`
- `estimator_variance = observation_variance / pricing_observations`
- `standard_error = sqrt(estimator_variance)`
- `variance_reduction_ratio = MC_observation_variance / method_observation_variance`
- `time_per_observation = runtime_seconds / pricing_observations`
- `time_per_simulated_path = runtime_seconds / pricing_simulated_paths`
- `efficiency_gain_vs_mc = (MC_estimator_variance * MC_runtime_seconds) / (method_estimator_variance * method_runtime_seconds)`

## Common pitfalls

- AV uses **2 simulated paths per pair observation**.
- CV pilot paths and NCV training paths are part of **total path budget** in Mode B.
- `variance_reduction_ratio` is a variance metric, **not** a speed metric.
- `efficiency_gain_vs_mc` is a runtime+precision metric, **not** a pure variance metric.

## Output contract

Each run writes a timestamped folder `run_YYYYMMDDTHHMMSSffffffZ` under `--output-dir` with:

- `mode_ab_statistical_raw.csv`
- `mode_c_runtime_raw.csv`
- `merged_summary.csv`
- `summary_stable.csv`
- `paper_table.csv`
- `paper_table.md`
- `paper_table_notes.txt`
- `metadata.json`
- `manifest.json`
- `README.txt`

`manifest.json` includes schema version (`RESULT_SCHEMA_VERSION = "1.0"`), file purposes, canonical columns, metric definitions, and units.

## Reproducibility guardrails

- Same seed + same config should produce identical `summary_stable.csv` bytes.
- Different seed should change at least one stochastic metric row.
- `metadata.json` captures `seeds` and `config_hash`.
