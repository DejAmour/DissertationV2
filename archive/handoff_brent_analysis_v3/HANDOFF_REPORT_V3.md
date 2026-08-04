# Brent GBM Handover Report V3

## 1) Historical GBM assumption diagnostics
- Requested sample window: 2000-01-01 to 2025-12-31.
- The requested sample window begins on 1 January 2000. The first available reported price observation within that window occurs on 2000-01-04.
- Processed effective sample: 2000-01-04 to 2025-12-31 with 6599 prices and 6598 daily log returns.
- Full-sample mean daily log return = 0.000142562420021; daily standard deviation = 0.02602606310655; annualised volatility = 0.4131509435.
- Full-sample return range: min = -0.6436989051459642; max = 0.4120225086543235.
- Jarque-Bera statistic = 1605146.17, p < 0.001.
- Empirical 3σ tail frequency = 0.009700 versus Gaussian benchmark 0.002700.
- Ljung-Box on returns at lag 10: p < 0.001; on squared returns at lag 10: p < 0.001.
- ADF with intercept: statistic = -2.6062, p = 0.0917; ADF with intercept and trend: statistic = -2.6659, p = 0.2505.
- AR(1) half-life is not interpreted; `half_life_days` is left missing because the ADF evidence is not robust at the 5% level. Failure to reject a unit root is not proof that a unit root exists, and structural breaks can weaken conventional ADF power.

Required-period descriptive summaries:
- **full_sample** (2000-01-04 to 2025-12-31): 6599 prices, 6598 returns, mean daily log return 0.000142562420021, daily std 0.02602606310655, annualised volatility 0.4131509435, skewness -1.9363, excess kurtosis 76.3718.
- **pre-pandemic comparison** (2010-01-04 to 2019-12-31): 2530 prices, 2529 returns, mean daily log return -0.000060878190790, daily std 0.01909957811531, annualised volatility 0.3031964030, skewness 0.1941, excess kurtosis 2.7699.
- **COVID-19 stress period** (2020-02-03 to 2020-06-30): 103 prices, 102 returns, mean daily log return -0.002548262772721, daily std 0.11087621503717, annualised volatility 1.7601053478, skewness -1.4892, excess kurtosis 12.6058.

The corrected GBM assumption matrix is provided in `tables/gbm_assumption_assessment.csv`.

## 2) Fixed-origin held-out path forecast
This section reports a **held-out fixed-origin path forecast**, not an independent repeated-forecast calibration exercise.

- **Split before estimation**: training sample 2000-01-04 to 2025-01-02; held-out test sample 2025-01-03 to 2025-12-31.
- **Counts**: 6347 training prices, 6346 training returns, 252 held-out test prices.
- **Training-only estimation**: mean daily log return = 0.000182257395; daily std = 0.026258553169; annualised mu = 0.1328073269; annualised sigma = 0.4168416088.
- **Forecast origin**: S0 = 76.14, equal to the final training price.
- No test observations enter fixed-origin parameter estimation.
- Fixed-origin median-path MAE = 9.39218049; RMSE = 10.74862092; MAPE = 14.20497660%.
- Fixed-origin 90% interval coverage = 1.000000; average interval width = 75.22448154; lower-tail violation frequency = 0.000000; upper-tail violation frequency = 0.000000.
- Directional accuracy is omitted from the dissertation-facing principal result table because the GBM median path is near-constant in sign and does not evidence market-timing skill.
- Figures: `figures/fixed_origin_observed_vs_median.png` and `figures/fixed_origin_prediction_interval.png`.

## 3) Rolling-origin forecast evaluation
Rolling-origin evaluation re-estimates GBM parameters on an expanding sample at each forecast origin and keeps all targets inside the final 252 valid prices.

Parameter regimes are explicitly separated:
1. `tables/gbm_simulation_parameters.csv` records **full-sample parameters** for the separate forward-simulation context.
2. `tables/fixed_origin_training_parameters.csv` records **training-only parameters** for the held-out fixed-origin path forecast.
3. `tables/rolling_origin_metrics_by_horizon.csv` summarises **rolling re-estimated parameters** across origins for horizons 1, 5, and 20 trading days.

Rolling-origin results:
- **1-day horizon**: n=252, MAE=1.022271, RMSE=1.346480, MAPE=1.479346%, coverage_90=0.964286, avg width=5.953421, median width=5.851603, lower-tail freq=0.019841, upper-tail freq=0.015873.
- **5-day horizon**: n=252, MAE=2.426677, RMSE=3.307608, MAPE=3.505216%, coverage_90=0.952381, avg width=13.372478, median width=13.134083, lower-tail freq=0.035714, upper-tail freq=0.011905.
- **20-day horizon**: n=252, MAE=3.768844, RMSE=4.806135, MAPE=5.411830%, coverage_90=0.984127, avg width=27.217783, median width=26.840032, lower-tail freq=0.011905, upper-tail freq=0.003968.

Overlap caveat: the 5-day and 20-day forecast errors overlap when every valid origin is used, so the 1-day-ahead results are the cleanest repeated-origin calibration diagnostic.

Supporting outputs:
- `tables/rolling_origin_forecasts.csv`
- `tables/rolling_origin_metrics_by_horizon.csv`
- `figures/rolling_origin_one_day_forecasts.png`
- `figures/rolling_origin_coverage_by_horizon.png`
- `figures/rolling_origin_interval_width_by_horizon.png`
- `figures/rolling_origin_errors_by_horizon.png`

## 4) Physical vs risk-neutral interpretation
- The historical diagnostics and backtests assess **physical-measure predictive behaviour** under historically estimated drift and volatility.
- The full-sample forward simulation uses mu = 0.1212725809 and sigma = 0.4131509435 estimated from the complete sample, which is distinct from the fixed-origin and rolling-origin backtests.
- Historical drift is not automatically the risk-neutral drift used for option valuation.
- Commodity risk-neutral pricing can require interest rates, storage costs, convenience yield, and market-price-of-risk adjustments.
- Good historical forecasting would not by itself validate a risk-neutral derivative-pricing model, and poor historical point forecasting does not by itself invalidate GBM for derivative-pricing approximations.
- MAE, RMSE, and MAPE evaluate median forecasts only; coverage and tail-frequency diagnostics are reported separately for interval behaviour.

## 5) Limitations
- Normality and constant-volatility assumptions are empirically poor approximations for Brent daily returns.
- Daily observations cannot distinguish jumps from extreme continuous shocks.
- ADF-based mean-reversion conclusions remain sensitive to structural breaks and low power near a unit root.
- Fixed-origin 100% coverage arises from one expanding path interval and should not be interpreted as calibrated independent forecast evidence.
- Rolling 5-day and 20-day results use overlapping targets and therefore produce dependent forecast errors.
- No additional unresolved pipeline errors were encountered.

## 6) Reproducibility
- Operating system and package versions are logged in `logs/environment.txt`.
- Exact environment versions are listed in `requirements-lock.txt`.
- Key regeneration commands:
  - Cleaned data: `python src/prepare_data.py`
  - Historical/full-sample parameter estimation: `python src/estimate_gbm.py`
  - Fixed-origin held-out backtest: `python src/evaluate_gbm.py`
  - Rolling-origin evaluation: `python src/rolling_origin_backtest.py`
  - Full v3 handover package: `python src/historical_diagnostics.py`
  - Full test run: `python -m pytest -q`

## 7) Test evidence
- Full pytest output is archived in `logs/test_results_all.txt`.
- Component logs are archived in `logs/test_results_data.txt`, `logs/test_results_historical.txt`, `logs/test_results_fixed_origin.txt`, `logs/test_results_rolling_origin.txt`, and `logs/test_results_handover.txt`.
- These logs are generated from actual pytest execution and should be reviewed directly for pass/fail/skip/warning details.
