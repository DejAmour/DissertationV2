# Brent GBM Handover Report

## 1) Objective and assumptions tested
This handover adds a historical empirical diagnostics component alongside existing GBM Stage 3/4 simulation and backtesting outputs. Diagnostics evaluate the assumptions of normal returns, constant volatility, independent returns, continuous price changes, absence of mean reversion, and strictly positive prices.

## 2) Data provenance and validation
- Source URL: https://raw.githubusercontent.com/datasets/oil-prices/main/data/brent-daily.csv
- Retrieval date: 2026-07-27
- Raw filename: DCOILBRENTEU_fred_3.csv
- Units: USD/barrel
- Frequency: Daily
- First/last observation in raw: 1987-05-20 to 2026-07-20
- Missing: 0
- Non-numeric: 0
- Duplicates: 0
- Non-positive: 0
- Chronological order in raw: True

No interpolation was applied. Missing values are not filled.

## 3) Return construction
Daily log returns are constructed exactly as:
\[r_t = \ln(S_t / S_{t-1})\]
with annualisation factor \(\sqrt{252}\) for volatility. No trimming or winsorisation was applied.

## 4) Descriptive statistics by required period
- **full_sample** (2000-01-04 to 2025-12-31): n_prices=6599, n_returns=6598, mean=0.000143, std=0.026026, annualized vol=0.4132, skew=-1.9363, excess kurtosis=76.3718.
- **pre-pandemic comparison** (2010-01-04 to 2019-12-31): n_prices=2530, n_returns=2529, mean=-0.000061, std=0.019100, annualized vol=0.3032, skew=0.1941, excess kurtosis=2.7699.
- **COVID-19 stress period** (2020-02-03 to 2020-06-30): n_prices=103, n_returns=102, mean=-0.002548, std=0.110876, annualized vol=1.7601, skew=-1.4892, excess kurtosis=12.6058.

## 5) Normality and tails
- Jarque-Bera and fitted normal parameters are reported in `tables/normality_tests.csv`.
- Tail exceedance frequencies outside 2σ/3σ/4σ/5σ, theoretical normal probabilities, and empirical/theoretical ratios are in `tables/empirical_tail_frequencies.csv`.
- Top-10 absolute daily log returns by period are in `tables/largest_absolute_returns.csv`.
- Caveat: daily data cannot prove path discontinuity.

## 6) Volatility and dependence
- Rolling annualized volatility summaries for 30-day and 90-day windows are in `tables/rolling_volatility_summary.csv`.
- ACF tables are in `tables/autocorrelations.csv` for returns, squared returns, and absolute returns.
- Ljung-Box results at lags 5/10/20 are in `tables/ljung_box_results.csv`.

## 7) Mean reversion and unit root diagnostics
- AR(1) on log prices is reported in `tables/ar1_results.csv`.
- ADF tests with `regression='c'` and `regression='ct'` and `autolag='AIC'` are in `tables/adf_results.csv`.
- Half-life is only reported when statistically defensible.

## 8) Sample-period comparison
`tables/sample_period_comparison.csv` and `figures/sample_period_comparison.png` compare the required periods.
The COVID-19 stress period (2020-02-01 to 2020-06-30) is selected ex ante based on external event timing.

## 9) GBM assumption assessment matrix
See `tables/gbm_assumption_assessment.csv`.

## 10) Stage 3/4 simulation and backtest assessment (separate component)
- Simulation parameters extracted from existing generated outputs are in `tables/gbm_simulation_parameters.csv`.
- Stage 4 metrics extracted from existing output are in `tables/gbm_backtest_metrics.csv`.
- Interpretation table with caveats is in `tables/gbm_backtest_interpretation.csv`.

Preservation check:
All checked Stage 3/4 reference values matched expected constants.

## 11) Limitations
- Spot vs futures basis differences are not modeled.
- Structural breaks and regime changes can violate constant-parameter assumptions.
- Statistical test outcomes depend on sample window and test power.
- Daily observations cannot conclusively identify jump discontinuities.
- Backtest is under physical-measure historical dynamics, distinct from risk-neutral pricing contexts.
- Required full-sample coverage 2000-01-01 to 2025-12-31 not satisfied by processed data.

## 12) Reproducibility details
- Analysis script: `brent_gbm_analysis/src/historical_diagnostics.py`
- Generated tables: `handoff_brent_analysis/tables/`
- Generated figures: `handoff_brent_analysis/figures/`
- Run log: `handoff_brent_analysis/logs/analysis_run_log.txt`
- Environment log: `handoff_brent_analysis/logs/environment.txt`
- Test log: `handoff_brent_analysis/logs/test_results.txt`

## 13) Concise factual summary
- Full-sample processed coverage: 2000-01-04 to 2025-12-31 with 6599 prices.
- Full-sample return mean=0.000143, std=0.026026, annualized vol=0.4132.
- Full-sample return range: min=-0.643699, max=0.412023.
- Jarque-Bera statistic=1605146.1747, p-value=0.
- Empirical outside-3σ frequency=0.009700, theoretical=0.002700.
- Ljung-Box returns lag10 p-value=4.492e-06; squared returns lag10 p-value=1.162e-296.
- ADF(full sample, regression='c'): stat=-2.6062, p=0.09172, usedlag=33.
- Stage 3 inputs extracted: S0=61.35, mu=0.121273, sigma=0.413151.
- Backtest MAE=9.023263, RMSE=10.319288, MAPE=13.644789%.
- Backtest coverage_p05_p95=1.000000, avg_interval_width=74.044446 USD.
- Directional accuracy=0.477912 (<0.50 indicates weak short-horizon direction capture).
- COVID-19 stress period selected ex ante based on external event timing (2020-02-01 to 2020-06-30).
