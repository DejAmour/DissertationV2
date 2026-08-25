# Brent GBM Analysis — Stage 1

This repository contains Stage 1 of an MSc dissertation workflow evaluating where geometric Brownian motion (GBM) is and is not a reasonable model for Brent crude oil spot prices.

## Project structure

```text
brent_gbm_analysis/
    README.md
    requirements.txt
    data/
        raw/
        processed/
    notebooks/
        brent_gbm_analysis.ipynb
    src/
        __init__.py
        download_data.py
        prepare_data.py
        descriptive_statistics.py
        distribution_analysis.py
        volatility_analysis.py
        mean_reversion.py
        sample_comparison.py
    tests/
        test_data_quality.py
    outputs/
        figures/
        tables/
        reports/
```

## Installation (Python 3.11+)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r brent_gbm_analysis/requirements.txt
```

## Run Stage 1

```bash
cd brent_gbm_analysis
python src/download_data.py
python src/prepare_data.py
pytest tests/test_data_quality.py -q
```

## Asian options — Stage 1 reproducibility

This repository also contains a separate Asian-options project under
`asian_options/`. Its dependency capture is intentionally scoped to the
Asian-options code only and does not change the Brent-analysis environment.

### Validated dependency baseline

- Python 3.13.1
- numpy 2.5.2
- scipy 1.18.0
- torch 2.13.0+cpu
- pytest 9.1.1
- matplotlib 3.10.7

Install into a clean virtual environment from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r asian_options/requirements.txt
```

Windows PowerShell activation:

```powershell
.\.venv\Scripts\Activate.ps1
```

`asian_options/requirements.txt` is the authoritative direct dependency spec.
`asian_options/requirements-lock.txt` captures the full transitive dependency
graph from the sandbox validation environment. Reproducibility metadata,
including platform, Python, package versions, and deterministic-seeding
settings, is stored in `asian_options/environment_metadata.md`.

### Stage 1 validation commands

```bash
pytest -q
python run_method_comparison.py
```

### Stage 6 canonical experiment packaging

For publication-ready Stage 6 outputs (raw mode CSVs, merged summary, stable
summary for reproducibility, metadata, manifest, and run README), use:

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

See `docs/experiments.md` for exact metric definitions, mode semantics, and
reproducibility guidance.

### Stage 7 validation + packaging profile

Run the Stage 7 validation milestone bundle:

```bash
python scripts/run_experiments.py \
  --output-dir /tmp/stage7_runs \
  --profile validation_minimal
```

This writes a camera-ready bundle including raw mode CSVs, merged/stable
summaries, publication tables, validation aggregates with 95% CIs,
manifest/metadata, and a validation pass/fail report.
See `docs/validation.md` for formulas, interpretation, troubleshooting, and
safe-claims boundaries.

`run_method_comparison.py` writes `asian_options_method_comparison.csv` in the
repository root after running MC, AV, CV, and NCV.

### Deterministic seeding

`asian_options.seed_everything()` seeds Python `random`, NumPy, and PyTorch,
seeds CUDA when available, and enables deterministic PyTorch algorithms where
practical in warn-only mode. Determinism can still vary across hardware,
drivers, and PyTorch builds, especially on CUDA-enabled systems.

### Stage 8 runner commands

```bash
python scripts/run_stage8.py --profile smoke --output-dir experiment_runs
python scripts/run_stage8.py --profile dissertation --output-dir experiment_runs
python scripts/run_stage8.py --profile dissertation --output-dir experiment_runs --empirical-equal-budget
python scripts/run_stage8.py --profile m252_w16_n20000_r10_final --output-dir experiment_runs
python scripts/run_stage8.py --profile m252_w16_n20000_r10_frozen_reuse --output-dir experiment_runs
python scripts/run_stage8.py --profile pcncv_smoke --output-dir experiment_runs
python scripts/run_stage8.py --profile pcncv_dissertation --output-dir experiment_runs
```

Supported Stage 8 overrides (defaults preserved): `--monitoring-dates`,
`--ncv-epoch`, `--ncv-epoch-source`, `--experiment-role`, `--output-dir`.

Stage 8 fixes now use a fixed NCV checkpoint of **25 epochs** sourced from
the auxiliary training-curve validation tuning study
(`ncv_epoch_source=training_curve_validation_tuning`). Training still uses
MSE/Adam with the same architecture and option specification; only the fixed
checkpoint policy changed. Final evaluation seeds are deterministically
separated from training-curve train/validation/test streams.
The generic `neural_cv.train_network` default remains 200 epochs; Stage 8
explicitly overrides that generic default with the fixed 25-epoch policy.
`neural_cv.train_network` does not perform online validation early stopping.
The 25-epoch choice comes from the separate validation-based training-curve
study, and final Stage 8 test/pricing runs do not select epoch online.
The tuning study documented that 25 epochs improved held-out NCV residual
variance by roughly 26% versus 100 epochs (about 27.6% with validation-selected
stopping), while GCV remained substantially stronger (about 5.8–5.9× lower
residual variance than the best NCV checkpoint).

### NCV training-curve experiment (reference contract)

```bash
python scripts/run_ncv_training_curve.py --profile smoke --output-dir experiment_runs
python scripts/run_ncv_training_curve.py --profile dissertation --output-dir experiment_runs
```

Training-curve reporting distinguishes:
- optimizer objective: training MSE;
- checkpoint criterion: validation residual variance
  `Var(f(Z)-H_theta(Z))`;
- held-out test set: final evaluation only.

Projected matched-accuracy costs now use **end-to-end pricing runtime** (fresh
RNG + path simulation + payoff + control evaluation + estimator averaging),
with one-time setup costs (NCV training, GCV pilot where reusable) counted once
and marginal pricing costs multiplied by `Q`. Runtime projection metadata
records basis sizes and whether values are empirical or projected.
Raw bounded timing observations are exported to
`training_curve_runtime_benchmarks.csv` (one row per replication/method/timing
size/repeat) and are the direct source for repeat aggregation, linearity ratios,
projection-method selection, and marginal-cost projection inputs.
Diagnostic component runtime fields are only populated when separately measured;
otherwise they are recorded as `NA` with
`component_runtime_measurement_status=not_separately_measured`.
Legacy `gcv_pricing_runtime_s` / `gcv_per_observation_runtime_s` fields remain
diagnostic control-only labels with
`gcv_pricing_runtime_scope=legacy_control_only_diagnostic_not_used_for_costing`.
Stage 8 equal-observation break-even costing uses mutually exclusive setup and
pricing-only marginal runtime components (no setup/marginal double counting).
Stage 8 equal-budget projections use amortised-`Q` setup reuse: reusable setup
paths are counted once and pricing paths scale with `Q`.
Validation emits non-fatal runtime-regime warnings when robust early/late
replication medians differ materially.
Stage 8 reproducibility output reports both canonical stable-summary hash and
raw written CSV-byte hash.
For NCV training-curve outputs, `ncv_setup_cost_s` is defined as
`training_data_generation_runtime_s + optimizer_cumulative_training_runtime_s`
for checkpoints above zero, and `0` at checkpoint zero; validation
generation/evaluation runtime is explicitly labeled as research/tuning overhead
excluded from operational setup cost.
Summary confidence intervals now use two-sided Student-t intervals for finite
replication counts (`n>=2`), with explicit CI metadata and `n=1` marked
undefined (no synthetic interval).
The training-curve config field `default_epochs` is explicitly scoped to the
training-curve default checkpoint horizon (`default_epochs_scope`) and is not
the Stage 8 fixed checkpoint policy.

Before final dissertation runs, activate the environment and record the Python
interpreter:

```powershell
.\.venv\Scripts\Activate.ps1
python -c "import sys; print(sys.executable)"
```

Full Stage 8 (m=12, epoch=1000) joint monitoring-frequency/input-dimensionality
sensitivity run:

```powershell
python .\scripts\run_stage8.py --profile dissertation --output-dir .\experiment_runs_m12 --monitoring-dates 12 --ncv-epoch 1000 --ncv-epoch-source training_curve_validation_tuning_m12 --experiment-role monitoring_frequency_and_input_dimension_sensitivity
```

### NCV output-rescaling note

The existing transferred control coefficient `beta` already performs scalar
output rescaling. If `H̃(Z)=aH(Z)`, then
`H̃(Z)-E[H̃(Z)] = a(H(Z)-E[H(Z)])`, so multiplying final-layer output weights
by `a` is equivalent to setting `beta=a`. A separate physical output-weight
rescaling estimator is therefore intentionally not added.

## Data source attribution

- Series: `DCOILBRENTEU` (Daily Europe Brent Spot Price FOB, USD/barrel)
- Original publisher: U.S. Energy Information Administration (EIA)
- Accessed via FRED direct CSV:
  `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU&cosd=2000-01-01&coed=2025-12-31`
