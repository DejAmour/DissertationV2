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
pip install -r /home/runner/work/DissertationV2/DissertationV2/brent_gbm_analysis/requirements.txt
```

## Run Stage 1

```bash
cd /home/runner/work/DissertationV2/DissertationV2/brent_gbm_analysis
python src/download_data.py
python src/prepare_data.py
pytest tests/test_data_quality.py -q
```

## Data source attribution

- Series: `DCOILBRENTEU` (Daily Europe Brent Spot Price FOB, USD/barrel)
- Original publisher: U.S. Energy Information Administration (EIA)
- Accessed via FRED direct CSV:
  `https://fred.stlouisfed.org/graph/fredgraph.csv?id=DCOILBRENTEU&cosd=2000-01-01&coed=2025-12-31`
