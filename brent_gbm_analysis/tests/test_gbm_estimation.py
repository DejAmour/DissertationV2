"""Stage 2 tests: GBM parameter estimation from Brent spot price data."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "brent_prices_2000_2025_clean.csv"
PARAMS_FILE = PROJECT_ROOT / "outputs" / "tables" / "gbm_parameters.csv"
DIAG_FILE = PROJECT_ROOT / "outputs" / "tables" / "return_diagnostics.csv"
HISTOGRAM_FILE = PROJECT_ROOT / "outputs" / "figures" / "log_returns_histogram.png"

_PREREQ_MISSING = not PROCESSED_FILE.exists()
_PREREQ_SKIP_REASON = (
    "Run `python src/download_data.py` and `python src/prepare_data.py` "
    "before Stage 2 integration tests."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_estimation() -> None:
    """Run Stage 2 estimation if output files are missing."""
    if not PARAMS_FILE.exists() or not DIAG_FILE.exists() or not HISTOGRAM_FILE.exists():
        from src.estimate_gbm import estimate_gbm  # noqa: PLC0415
        estimate_gbm()


def _load_params() -> pd.DataFrame:
    _run_estimation()
    return pd.read_csv(PARAMS_FILE)


def _load_diagnostics() -> pd.DataFrame:
    _run_estimation()
    return pd.read_csv(DIAG_FILE)


# ---------------------------------------------------------------------------
# Integration tests (require Stage 1 processed data)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_output_files_exist() -> None:
    """All three Stage 2 output files should be created after running estimate_gbm."""
    _run_estimation()
    assert PARAMS_FILE.exists(), f"Missing: {PARAMS_FILE}"
    assert DIAG_FILE.exists(), f"Missing: {DIAG_FILE}"
    assert HISTOGRAM_FILE.exists(), f"Missing: {HISTOGRAM_FILE}"


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_sigma_is_positive_and_finite() -> None:
    """Estimated annualised volatility must be strictly positive and finite."""
    params = _load_params()
    sigma_row = params.loc[params["Parameter"] == "sigma_annual", "Value"]
    assert len(sigma_row) == 1, "sigma_annual row not found in gbm_parameters.csv"
    sigma = float(sigma_row.iloc[0])
    assert math.isfinite(sigma), f"sigma_annual is not finite: {sigma}"
    assert sigma > 0, f"sigma_annual is not positive: {sigma}"


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_mu_is_finite() -> None:
    """Estimated annualised drift must be finite."""
    params = _load_params()
    mu_row = params.loc[params["Parameter"] == "mu_annual", "Value"]
    assert len(mu_row) == 1, "mu_annual row not found in gbm_parameters.csv"
    mu = float(mu_row.iloc[0])
    assert math.isfinite(mu), f"mu_annual is not finite: {mu}"


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_return_sample_count_nonzero() -> None:
    """Number of log-return observations must be greater than zero."""
    params = _load_params()
    n_row = params.loc[params["Parameter"] == "n_returns", "Value"]
    assert len(n_row) == 1, "n_returns row not found in gbm_parameters.csv"
    n = int(float(n_row.iloc[0]))
    assert n > 0, f"n_returns is not positive: {n}"


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_gbm_parameters_schema() -> None:
    """gbm_parameters.csv must contain exactly the expected Parameter names."""
    params = _load_params()
    assert set(params.columns) == {"Parameter", "Value"}, (
        f"Unexpected columns in gbm_parameters.csv: {list(params.columns)}"
    )
    expected_params = {"sigma_annual", "mu_annual", "n_returns"}
    found_params = set(params["Parameter"].tolist())
    assert expected_params.issubset(found_params), (
        f"Missing parameters in gbm_parameters.csv: {expected_params - found_params}"
    )


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_return_diagnostics_schema() -> None:
    """return_diagnostics.csv must contain exactly the expected Metric names."""
    diag = _load_diagnostics()
    assert set(diag.columns) == {"Metric", "Value"}, (
        f"Unexpected columns in return_diagnostics.csv: {list(diag.columns)}"
    )
    required_metrics = {"n_returns", "mean_daily", "std_daily", "min_daily", "max_daily"}
    found_metrics = set(diag["Metric"].tolist())
    assert required_metrics.issubset(found_metrics), (
        f"Missing metrics in return_diagnostics.csv: {required_metrics - found_metrics}"
    )


# ---------------------------------------------------------------------------
# Unit tests: estimation functions (no file I/O, always run)
# ---------------------------------------------------------------------------

def test_compute_log_returns_length() -> None:
    """Log returns should have one fewer element than the price series."""
    from src.estimate_gbm import compute_log_returns  # noqa: PLC0415

    prices = pd.Series([100.0, 102.0, 101.0, 105.0])
    returns = compute_log_returns(prices)
    assert len(returns) == len(prices) - 1


def test_compute_log_returns_values() -> None:
    """Log returns should equal ln(P_t / P_{t-1})."""
    from src.estimate_gbm import compute_log_returns  # noqa: PLC0415

    prices = pd.Series([100.0, 110.0])
    returns = compute_log_returns(prices)
    expected = math.log(110.0 / 100.0)
    assert abs(float(returns.iloc[0]) - expected) < 1e-10


def test_estimate_gbm_parameters_known_input() -> None:
    """sigma_annual should equal std(r)*sqrt(252); mu should use Ito correction."""
    from src.estimate_gbm import estimate_gbm_parameters, TRADING_DAYS_PER_YEAR  # noqa: PLC0415

    rng = np.random.default_rng(42)
    r = pd.Series(rng.normal(0.0002, 0.01, 1000))
    result = estimate_gbm_parameters(r)

    expected_sigma = float(r.std(ddof=1)) * math.sqrt(TRADING_DAYS_PER_YEAR)
    expected_mu = float(r.mean()) * TRADING_DAYS_PER_YEAR + 0.5 * expected_sigma ** 2

    assert abs(result["sigma_annual"] - expected_sigma) < 1e-10
    assert abs(result["mu_annual"] - expected_mu) < 1e-10
    assert result["n_returns"] == 1000
    assert math.isfinite(result["sigma_annual"])
    assert math.isfinite(result["mu_annual"])


def test_estimate_gbm_parameters_raises_on_empty() -> None:
    """estimate_gbm_parameters should raise ValueError for an empty series."""
    from src.estimate_gbm import estimate_gbm_parameters  # noqa: PLC0415

    with pytest.raises(ValueError, match="No valid log returns"):
        estimate_gbm_parameters(pd.Series([], dtype=float))
