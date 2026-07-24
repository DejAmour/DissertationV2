# Brent GBM Analysis

This subproject implements a two-stage Geometric Brownian Motion (GBM) analysis pipeline
for Europe Brent crude oil spot prices (2000–2025).

Use the repository root README for full setup instructions.

---

## Stage 1 — Data retrieval and validation

Downloads and cleans daily Brent spot prices from FRED.

```bash
cd brent_gbm_analysis
python src/download_data.py    # fetch raw CSV from FRED
python src/prepare_data.py     # clean, validate, produce Stage 1 outputs
pytest tests/test_data_quality.py -q
```

**Stage 1 outputs**

| File | Description |
|------|-------------|
| `data/processed/brent_prices_2000_2025_clean.csv` | Cleaned daily prices (columns: `Date`, `Price_USD_per_barrel`) |
| `outputs/tables/data_validation_summary.csv` | Data-quality metrics |
| `outputs/figures/brent_price_series.png` | Price time-series chart |

---

## Stage 2 — GBM parameter estimation

Estimates GBM drift (`mu`) and volatility (`sigma`) from the Stage 1 cleaned data.

```bash
cd brent_gbm_analysis
python src/estimate_gbm.py
pytest tests/test_gbm_estimation.py -q
```

> **Prerequisites**: Stage 1 must be run first so that
> `data/processed/brent_prices_2000_2025_clean.csv` exists.

### Estimation formulas

Daily log returns are computed as:

```
r_t = ln(P_t / P_{t-1})
```

Annualised parameters (T = 252 trading days per year):

```
sigma_annual = std(r_t) × sqrt(T)

mu_annual    = mean(r_t) × T + 0.5 × sigma_annual²
```

The Ito correction (`+ 0.5 × sigma²`) converts the mean log-return
(which estimates `mu − 0.5σ²` under GBM) back to the true GBM drift `mu`.

### Stage 2 outputs

| File | Description |
|------|-------------|
| `outputs/tables/gbm_parameters.csv` | Estimated `sigma_annual`, `mu_annual`, `n_returns` |
| `outputs/tables/return_diagnostics.csv` | Mean, std, min, max (and optionally skew/kurtosis) of daily log returns |
| `outputs/figures/log_returns_histogram.png` | Histogram of daily log returns |

---

## Stage 3 — GBM simulation

Uses the Stage 2 parameters to simulate GBM price paths and produces
dissertation-ready fan charts and terminal distribution summaries.

```bash
cd brent_gbm_analysis
python src/simulate_gbm.py
pytest tests/test_gbm_simulation.py -q
```

> **Prerequisites**: Stages 1 and 2 must be run first so that
> `data/processed/brent_prices_2000_2025_clean.csv` and
> `outputs/tables/gbm_parameters.csv` both exist.

### Simulation formula

Exact GBM discretization:

```
S_{t+1} = S_t × exp((mu − 0.5×sigma²)×dt + sigma×sqrt(dt)×Z_t)
```

where `Z_t ~ N(0, 1)` i.i.d. and `dt = 1/252` (one trading day).

### Parameterization defaults

| Parameter | Default | Description |
|-----------|---------|-------------|
| `S0` | last observed close | Initial price loaded from Stage 1 cleaned CSV |
| `horizon_days` | 252 | Number of trading days to simulate |
| `n_paths` | 5000 | Number of independent Monte Carlo paths |
| `dt` | 1/252 | Length of each time step (in years) |
| `random_seed` | 42 | NumPy seed for reproducibility |

### Stage 3 outputs

| File | Description |
|------|-------------|
| `outputs/tables/simulation_quantiles.csv` | Day-by-day price quantiles (q05, q25, q50, q75, q95) across all paths |
| `outputs/tables/terminal_distribution_summary.csv` | Mean, std, min, max and percentiles of the terminal (day-252) price distribution |
| `outputs/figures/gbm_fan_chart.png` | Median + percentile bands fan chart over the simulation horizon |
| `outputs/figures/terminal_price_histogram.png` | Histogram of simulated terminal prices |
