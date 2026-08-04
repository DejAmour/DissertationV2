# Methodology Audit V3

This audit maps the corrected v3 computations to the code that produces them and records the principal methodological controls.

## Computation mapping
- Stage 1 preparation: `src/prepare_data.py::prepare_data` reads the inherited raw Brent CSV, applies the requested 2000-01-01 to 2025-12-31 window, removes invalid prices, and writes the cleaned processed CSV.
- Full-sample parameter estimation: `src/estimate_gbm.py::estimate_gbm` computes log returns on the full cleaned sample and writes `outputs/tables/gbm_parameters.csv`.
- Fixed-origin held-out path forecast: `src/evaluate_gbm.py::evaluate_gbm` performs the split **before** estimation, computes training returns from training prices only, estimates mu and sigma on the training sample only, sets `S0` to the final training price, and excludes all test observations from parameter estimation.
- Rolling-origin evaluation: `src/rolling_origin_backtest.py::run_rolling_origin_backtest` re-estimates mu and sigma on the expanding sample available at each origin and generates 1-, 5-, and 20-day endpoint forecasts for targets in the final 252 valid prices only.
- Handover packaging: `src/historical_diagnostics.py::build_handoff_package` regenerates the v3 tables, figures, report, methodology audit, corrections log, copied data/code, and archive.

## Parameter distinctions
1. **Full-sample parameters for separate forward simulation** are written to `tables/gbm_simulation_parameters.csv`.
2. **Training-only fixed-origin parameters** are written to `tables/fixed_origin_training_parameters.csv`.
3. **Rolling re-estimated parameters at each origin** are stored forecast-by-forecast in `tables/rolling_origin_forecasts.csv` and summarised in `tables/rolling_origin_metrics_by_horizon.csv`.

## Formula/package/options/output
- Returns: `r_t = ln(S_t / S_{t-1})` using pandas/NumPy.
- Annualised volatility: `std_daily * sqrt(252)`.
- GBM annualised drift: `mu = mean_daily * 252 + 0.5 * sigma^2`.
- Jarque-Bera, skewness, kurtosis, tail frequencies: SciPy.
- Autocorrelation and Ljung-Box: statsmodels.
- AR(1) and ADF diagnostics: statsmodels OLS and `adfuller(..., regression in {'c','ct'}, autolag='AIC')`.
- Fixed-origin and rolling-origin predictive quantiles: exact analytical lognormal GBM quantiles via `src/evaluate_gbm.py::compute_forecast_quantiles`.

## Key audit outcomes
- Split-before-estimation control for the fixed-origin backtest: **implemented**.
- No test observations in fixed-origin parameter estimation: **implemented**.
- S0 for the fixed-origin backtest equals the final training price: **implemented**.
- Rolling-origin forecasts use only information available on or before each origin: **implemented**.
- The old stale statement claiming a fixed-origin look-ahead caveat from full-sample Stage 2 parameters has been removed.
- Historical diagnostics remain code-generated from the processed dataset; reported values are read from generated outputs rather than manually hard-coded into the report.
- Very small p-values are formatted in prose as inequalities rather than literal zero p-values.
- The Q-Q figure explicitly labels fit-based standardisation as `fit=True`.
