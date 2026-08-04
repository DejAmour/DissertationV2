"""Stage 4 tests: GBM backtest / out-of-sample evaluation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "brent_prices_2000_2025_clean.csv"
PARAMS_FILE = PROJECT_ROOT / "outputs" / "tables" / "gbm_parameters.csv"

COMPARISON_FILE = PROJECT_ROOT / "outputs" / "tables" / "backtest_path_comparison.csv"
METRICS_FILE = PROJECT_ROOT / "outputs" / "tables" / "backtest_metrics.csv"
FORECAST_VS_ACTUAL_FILE = PROJECT_ROOT / "outputs" / "figures" / "backtest_forecast_vs_actual.png"
ERROR_DIST_FILE = PROJECT_ROOT / "outputs" / "figures" / "backtest_error_distribution.png"

_PREREQ_MISSING = not PROCESSED_FILE.exists() or not PARAMS_FILE.exists()
_PREREQ_SKIP_REASON = (
    "Run `python src/download_data.py`, `python src/prepare_data.py`, and "
    "`python src/estimate_gbm.py` before Stage 4 integration tests."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_evaluation() -> None:
    """Run Stage 4 evaluation if output files are missing."""
    if (
        not COMPARISON_FILE.exists()
        or not METRICS_FILE.exists()
        or not FORECAST_VS_ACTUAL_FILE.exists()
        or not ERROR_DIST_FILE.exists()
    ):
        from src.evaluate_gbm import evaluate_gbm  # noqa: PLC0415
        evaluate_gbm()


def _load_comparison() -> pd.DataFrame:
    _run_evaluation()
    return pd.read_csv(COMPARISON_FILE)


def _load_metrics() -> pd.DataFrame:
    _run_evaluation()
    return pd.read_csv(METRICS_FILE)


# ---------------------------------------------------------------------------
# Integration tests (require Stage 1 + Stage 2 prerequisite files)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_output_files_exist() -> None:
    """All four Stage 4 output files should be created after running evaluate_gbm."""
    _run_evaluation()
    assert COMPARISON_FILE.exists(), f"Missing: {COMPARISON_FILE}"
    assert METRICS_FILE.exists(), f"Missing: {METRICS_FILE}"
    assert FORECAST_VS_ACTUAL_FILE.exists(), f"Missing: {FORECAST_VS_ACTUAL_FILE}"
    assert ERROR_DIST_FILE.exists(), f"Missing: {ERROR_DIST_FILE}"


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_comparison_schema() -> None:
    """backtest_path_comparison.csv must contain the expected columns."""
    df = _load_comparison()
    expected = {"date", "day_index", "actual_price", "forecast_p05", "forecast_p50", "forecast_p95"}
    assert expected.issubset(set(df.columns)), (
        f"Missing columns in backtest_path_comparison.csv: {expected - set(df.columns)}"
    )


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_comparison_non_empty() -> None:
    """backtest_path_comparison.csv must have at least one row."""
    df = _load_comparison()
    assert len(df) > 0, "backtest_path_comparison.csv is empty."


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_comparison_prices_positive() -> None:
    """All price columns must be strictly positive."""
    df = _load_comparison()
    for col in ["actual_price", "forecast_p05", "forecast_p50", "forecast_p95"]:
        assert (df[col] > 0).all(), f"Non-positive value found in column '{col}'."


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_comparison_forecast_quantile_ordering() -> None:
    """Forecast quantiles must satisfy p05 <= p50 <= p95 at every row."""
    df = _load_comparison()
    assert (df["forecast_p05"] <= df["forecast_p50"]).all(), (
        "forecast_p05 exceeds forecast_p50 in some rows."
    )
    assert (df["forecast_p50"] <= df["forecast_p95"]).all(), (
        "forecast_p50 exceeds forecast_p95 in some rows."
    )


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_metrics_schema() -> None:
    """backtest_metrics.csv must contain expected metric rows."""
    df = _load_metrics()
    assert set(df.columns) == {"Metric", "Value"}, (
        f"Unexpected columns in backtest_metrics.csv: {list(df.columns)}"
    )
    required = {"MAE", "RMSE", "MAPE_pct", "coverage_p05_p95", "avg_interval_width",
                "directional_accuracy"}
    found = set(df["Metric"].tolist())
    assert required.issubset(found), (
        f"Missing metrics in backtest_metrics.csv: {required - found}"
    )


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_metrics_finite() -> None:
    """All metric values must be finite numbers."""
    df = _load_metrics()
    for _, row in df.iterrows():
        assert np.isfinite(row["Value"]), (
            f"Metric '{row['Metric']}' has non-finite value: {row['Value']}"
        )


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_coverage_in_unit_interval() -> None:
    """Interval coverage must be in [0, 1]."""
    df = _load_metrics()
    coverage = float(df.loc[df["Metric"] == "coverage_p05_p95", "Value"].iloc[0])
    assert 0.0 <= coverage <= 1.0, f"Coverage out of [0, 1]: {coverage}"


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_directional_accuracy_in_unit_interval() -> None:
    """Directional accuracy must be in [0, 1]."""
    df = _load_metrics()
    da = float(df.loc[df["Metric"] == "directional_accuracy", "Value"].iloc[0])
    assert 0.0 <= da <= 1.0, f"Directional accuracy out of [0, 1]: {da}"


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_mae_rmse_positive() -> None:
    """MAE and RMSE must be strictly positive (non-trivial forecast errors expected)."""
    df = _load_metrics()
    mae = float(df.loc[df["Metric"] == "MAE", "Value"].iloc[0])
    rmse = float(df.loc[df["Metric"] == "RMSE", "Value"].iloc[0])
    assert mae >= 0, f"MAE should be non-negative, got {mae}"
    assert rmse >= 0, f"RMSE should be non-negative, got {rmse}"


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_rmse_ge_mae() -> None:
    """RMSE must be >= MAE (standard property of L2 vs L1 error)."""
    df = _load_metrics()
    mae = float(df.loc[df["Metric"] == "MAE", "Value"].iloc[0])
    rmse = float(df.loc[df["Metric"] == "RMSE", "Value"].iloc[0])
    assert rmse >= mae - 1e-9, f"RMSE ({rmse:.4f}) should be >= MAE ({mae:.4f})."


# ---------------------------------------------------------------------------
# Unit tests — no file I/O, always run
# ---------------------------------------------------------------------------

def test_split_train_test_sizes() -> None:
    """split_train_test should produce correct train/test sizes."""
    from src.evaluate_gbm import split_train_test  # noqa: PLC0415

    df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=300), "Price_USD_per_barrel": np.ones(300)})
    train, test = split_train_test(df, test_window=50)
    assert len(train) == 250
    assert len(test) == 50


def test_split_train_test_no_overlap() -> None:
    """Train and test sets must be temporally separated (no date overlap)."""
    from src.evaluate_gbm import split_train_test  # noqa: PLC0415

    n = 400
    df = pd.DataFrame({
        "Date": pd.date_range("2020-01-01", periods=n),
        "Price_USD_per_barrel": np.arange(n, dtype=float),
    })
    train, test = split_train_test(df, test_window=100)
    last_train_date = pd.to_datetime(train["Date"].iloc[-1])
    first_test_date = pd.to_datetime(test["Date"].iloc[0])
    assert last_train_date < first_test_date, (
        f"Train/test overlap: last train date {last_train_date} >= "
        f"first test date {first_test_date}."
    )


def test_split_train_test_raises_on_small_dataset() -> None:
    """split_train_test should raise ValueError when test_window >= total rows."""
    from src.evaluate_gbm import split_train_test  # noqa: PLC0415

    df = pd.DataFrame({"Date": pd.date_range("2020-01-01", periods=10), "Price_USD_per_barrel": np.ones(10)})
    with pytest.raises(ValueError, match="test_window"):
        split_train_test(df, test_window=10)


def test_compute_forecast_quantiles_shape() -> None:
    """compute_forecast_quantiles should return a DataFrame with horizon_days rows."""
    from src.evaluate_gbm import compute_forecast_quantiles  # noqa: PLC0415

    df = compute_forecast_quantiles(S0=80.0, mu=0.12, sigma=0.40, horizon_days=20)
    assert len(df) == 20, f"Expected 20 rows, got {len(df)}"


def test_compute_forecast_quantiles_schema() -> None:
    """compute_forecast_quantiles should include day_index and quantile columns."""
    from src.evaluate_gbm import compute_forecast_quantiles  # noqa: PLC0415

    df = compute_forecast_quantiles(S0=80.0, mu=0.12, sigma=0.40, horizon_days=10,
                                    quantiles=[0.05, 0.50, 0.95])
    expected = {"day_index", "forecast_p05", "forecast_p50", "forecast_p95"}
    assert expected.issubset(set(df.columns)), (
        f"Missing columns: {expected - set(df.columns)}"
    )


def test_compute_forecast_quantiles_ordering() -> None:
    """Forecast quantiles must satisfy p05 <= p50 <= p95 at every row."""
    from src.evaluate_gbm import compute_forecast_quantiles  # noqa: PLC0415

    df = compute_forecast_quantiles(S0=80.0, mu=0.12, sigma=0.40, horizon_days=252,
                                    quantiles=[0.05, 0.50, 0.95])
    assert (df["forecast_p05"] <= df["forecast_p50"]).all()
    assert (df["forecast_p50"] <= df["forecast_p95"]).all()


def test_compute_forecast_quantiles_positive() -> None:
    """All forecast quantile values must be strictly positive."""
    from src.evaluate_gbm import compute_forecast_quantiles  # noqa: PLC0415

    df = compute_forecast_quantiles(S0=80.0, mu=0.12, sigma=0.40, horizon_days=252,
                                    quantiles=[0.05, 0.50, 0.95])
    for col in ["forecast_p05", "forecast_p50", "forecast_p95"]:
        assert (df[col] > 0).all(), f"Non-positive forecast value in '{col}'."


def test_compute_forecast_quantiles_day1_near_s0() -> None:
    """At day 1 with zero mu, the p50 equals S0 * exp(-0.5*sigma^2*dt), close to S0."""
    from src.evaluate_gbm import compute_forecast_quantiles  # noqa: PLC0415

    S0 = 75.0
    sigma = 0.40
    dt = 1.0 / 252
    # With mu=0: p50(day 1) = S0 * exp(-0.5*sigma^2*dt) ≈ S0 * exp(-0.000317) ≈ 0.99968*S0
    expected_p50 = S0 * np.exp(-0.5 * sigma ** 2 * dt)
    df = compute_forecast_quantiles(S0=S0, mu=0.0, sigma=sigma, horizon_days=5,
                                    quantiles=[0.50])
    p50_day1 = float(df.loc[df["day_index"] == 1, "forecast_p50"].iloc[0])
    assert abs(p50_day1 - expected_p50) < 1e-10, (
        f"p50 at day 1 ({p50_day1:.8f}) does not match expected ({expected_p50:.8f})"
    )


def test_compute_forecast_quantiles_raises_on_invalid_s0() -> None:
    """compute_forecast_quantiles should raise ValueError for non-positive S0."""
    from src.evaluate_gbm import compute_forecast_quantiles  # noqa: PLC0415

    with pytest.raises(ValueError, match="S0 must be positive"):
        compute_forecast_quantiles(S0=0.0, mu=0.12, sigma=0.40, horizon_days=10)


def test_compute_forecast_quantiles_raises_on_invalid_sigma() -> None:
    """compute_forecast_quantiles should raise ValueError for non-positive sigma."""
    from src.evaluate_gbm import compute_forecast_quantiles  # noqa: PLC0415

    with pytest.raises(ValueError, match="sigma must be positive"):
        compute_forecast_quantiles(S0=80.0, mu=0.12, sigma=0.0, horizon_days=10)


def test_compute_backtest_metrics_schema() -> None:
    """compute_backtest_metrics should return DataFrame with Metric/Value columns."""
    from src.evaluate_gbm import compute_backtest_metrics  # noqa: PLC0415

    df = pd.DataFrame({
        "actual_price":  np.array([80.0, 82.0, 78.0, 85.0, 81.0]),
        "forecast_p05":  np.array([70.0, 71.0, 69.0, 72.0, 70.0]),
        "forecast_p50":  np.array([80.5, 81.5, 80.0, 82.0, 81.0]),
        "forecast_p95":  np.array([90.0, 91.0, 89.0, 93.0, 91.0]),
    })
    metrics = compute_backtest_metrics(df)
    assert set(metrics.columns) == {"Metric", "Value"}
    required = {"MAE", "RMSE", "MAPE_pct", "coverage_p05_p95", "avg_interval_width",
                "directional_accuracy"}
    found = set(metrics["Metric"].tolist())
    assert required.issubset(found), f"Missing metrics: {required - found}"


def test_compute_backtest_metrics_values() -> None:
    """Spot-check metric values with a simple example."""
    from src.evaluate_gbm import compute_backtest_metrics  # noqa: PLC0415

    # Perfect forecast: p50 == actual
    actual = np.array([80.0, 82.0, 78.0])
    df = pd.DataFrame({
        "actual_price": actual,
        "forecast_p05": actual - 10.0,
        "forecast_p50": actual,
        "forecast_p95": actual + 10.0,
    })
    metrics = compute_backtest_metrics(df)
    metrics_map = dict(zip(metrics["Metric"], metrics["Value"]))
    assert abs(metrics_map["MAE"]) < 1e-10, "MAE should be 0 for perfect forecast."
    assert abs(metrics_map["RMSE"]) < 1e-10, "RMSE should be 0 for perfect forecast."
    assert abs(metrics_map["MAPE_pct"]) < 1e-10, "MAPE should be 0 for perfect forecast."
    assert metrics_map["coverage_p05_p95"] == pytest.approx(1.0), (
        "Coverage should be 1.0 when all actuals inside interval."
    )


def test_compute_backtest_metrics_coverage_zero() -> None:
    """Coverage should be 0.0 when all actuals are outside [p05, p95]."""
    from src.evaluate_gbm import compute_backtest_metrics  # noqa: PLC0415

    df = pd.DataFrame({
        "actual_price": np.array([200.0, 210.0, 220.0]),
        "forecast_p05": np.array([70.0, 71.0, 72.0]),
        "forecast_p50": np.array([80.0, 81.0, 82.0]),
        "forecast_p95": np.array([90.0, 91.0, 92.0]),
    })
    metrics = compute_backtest_metrics(df)
    metrics_map = dict(zip(metrics["Metric"], metrics["Value"]))
    assert metrics_map["coverage_p05_p95"] == pytest.approx(0.0)


def test_compute_backtest_metrics_mape_robust_zero_actual() -> None:
    """MAPE computation should not crash when actual prices are zero."""
    from src.evaluate_gbm import compute_backtest_metrics  # noqa: PLC0415

    # All actuals are zero — MAPE should be NaN (no valid entries)
    df = pd.DataFrame({
        "actual_price": np.array([0.0, 0.0, 0.0]),
        "forecast_p05": np.array([70.0, 71.0, 72.0]),
        "forecast_p50": np.array([80.0, 81.0, 82.0]),
        "forecast_p95": np.array([90.0, 91.0, 92.0]),
    })
    metrics = compute_backtest_metrics(df)
    metrics_map = dict(zip(metrics["Metric"], metrics["Value"]))
    assert np.isnan(metrics_map["MAPE_pct"]), (
        "MAPE should be NaN when all actual prices are zero."
    )


def test_load_price_history_raises_on_missing_file() -> None:
    """load_price_history should raise FileNotFoundError for a non-existent path."""
    from src.evaluate_gbm import load_price_history  # noqa: PLC0415

    with pytest.raises(FileNotFoundError):
        load_price_history(Path("/nonexistent/path/brent_prices.csv"))


def test_load_gbm_parameters_raises_on_missing_file() -> None:
    """load_gbm_parameters should raise FileNotFoundError for a non-existent path."""
    from src.evaluate_gbm import load_gbm_parameters  # noqa: PLC0415

    with pytest.raises(FileNotFoundError):
        load_gbm_parameters(Path("/nonexistent/path/gbm_parameters.csv"))


def test_reproducibility_analytical_quantiles() -> None:
    """Analytical quantiles are deterministic — two calls with same args must match."""
    from src.evaluate_gbm import compute_forecast_quantiles  # noqa: PLC0415

    df1 = compute_forecast_quantiles(S0=75.0, mu=0.12, sigma=0.40, horizon_days=50,
                                     quantiles=[0.05, 0.50, 0.95])
    df2 = compute_forecast_quantiles(S0=75.0, mu=0.12, sigma=0.40, horizon_days=50,
                                     quantiles=[0.05, 0.50, 0.95])
    pd.testing.assert_frame_equal(df1, df2)
