# Methodology Audit

This audit maps each computation to source/function/input/formula/package/options/output.

## Computation mapping
- Stage 1 retrieval: `src/download_data.py::download_brent_data` -> raw CSV + provenance JSON.
- Stage 1 preparation: `src/prepare_data.py::prepare_data` -> cleaned processed CSV and validation summary.
- Stage 2 estimation: `src/estimate_gbm.py::{compute_log_returns, estimate_gbm_parameters}` -> `outputs/tables/gbm_parameters.csv`.
- Stage 3 simulation: `src/simulate_gbm.py::{simulate_gbm_paths, compute_simulation_quantiles}` -> simulation outputs.
- Stage 4 backtest: `src/evaluate_gbm.py::{compute_forecast_quantiles, compute_backtest_metrics}` -> backtest metrics/comparison.
- Handover diagnostics: `src/historical_diagnostics.py` -> handoff tables, figures, report, audit, logs.

## Formula/package/options/output
- Returns: `r_t = ln(S_t/S_{t-1})` using pandas/numpy.
- Annualized volatility: `std_daily * sqrt(252)`.
- Jarque-Bera/tails: scipy.stats.
- Rolling vol: pandas rolling std with windows 30/90 and annualization factor 252.
- ACF: statsmodels `acf` and `plot_acf`.
- Ljung-Box: statsmodels `acorr_ljungbox` at lags [5,10,20].
- AR(1): statsmodels OLS on `log_price_t ~ 1 + log_price_{t-1}`.
- ADF: statsmodels `adfuller(..., regression in {'c','ct'}, autolag='AIC')`.

## Audit flags
- Silent NA drops: **Flagged and explicit** (`dropna()` used intentionally for returns/tests).
- Interpolation: **Not used**.
- Look-ahead bias: **Flagged caveat** for Stage 4 because mu/sigma come from Stage 2 full sample.
- Date inconsistency: **Checked** via validation + period slicing checks.
- Hard-coded results: **Not used for computed diagnostics**; values are read from generated files for report tables.
- Annualization correctness: **Checked** against `sqrt(252)`.
- Arithmetic-vs-log returns: **Log returns only** in diagnostics.
- Assumption caveats: **Explicitly included** (normality, constant volatility, continuity limits, physical vs risk-neutral).
- Figure-table sample mismatch: **Checked in tests where feasible**.
- Notebook-script discrepancy: **Not relied upon**; workflow is script-based.
- Reproducibility gaps: **Logged** in `logs/analysis_run_log.txt` and `logs/environment.txt`.
