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

Stage 7 validation profile:

```bash
python scripts/run_experiments.py \
  --output-dir /tmp/stage7_runs \
  --profile validation_minimal
```

This profile runs a compact deterministic config matrix and emits validation
checks, aggregate confidence intervals, and a camera-ready bundle.

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

## Stage 8 / NCV training-curve cost scope note

- Use end-to-end pricing cost for matched-accuracy projections:
  RNG + GBM path generation + arithmetic payoff + control evaluation +
  estimator averaging.
- Record bounded empirical timing rows in
  `training_curve_runtime_benchmarks.csv` (replication × method × timing size ×
  repeat) and use those exact rows as the sole input for timing-repeat
  aggregation, linearity diagnostics, projection-method choice, and runtime
  cost projection.
- Keep component timings as diagnostics (`path_and_payoff_runtime_s`,
  `control_evaluation_runtime_s`, `estimator_reduction_runtime_s`) but do not
  replace end-to-end timing with a sum of duplicated components. If components
  are not separately measured, record them as `NA` with
  `component_runtime_measurement_status=not_separately_measured`.
- One-time costs are amortised once (`setup_cost_s`), pricing is multiplied by
  `Q` (`projected_total_cost_s = setup_cost_s + Q * marginal_pricing_cost_s`).
- For NCV training-curve checkpoints, operational setup is
  `ncv_setup_cost_s = training_data_generation_runtime_s + optimizer_cumulative_training_runtime_s`
  for checkpoint > 0 and `0` for checkpoint = 0.
- Validation generation/evaluation runtime is tracked separately as
  research/tuning overhead and excluded from operational setup cost.
- Stage 8 fixed checkpoint policy is 25 epochs with
  `ncv_epoch_source=training_curve_validation_tuning`; this is separate from
  the generic `neural_cv.train_network` default (200 epochs), and the generic
  training function does not perform online early stopping.
- For training-curve summaries, confidence intervals are two-sided Student-t
  intervals (`n>=2`); `n=1` rows are marked CI undefined.
- Legacy `gcv_pricing_runtime_s` and `gcv_per_observation_runtime_s` remain
  diagnostic control-only fields; primary costing uses explicit end-to-end
  runtime fields.

## Dissertation environment reproducibility reminder

Run dissertation-profile experiments from the project virtual environment and
record the interpreter path:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
```
