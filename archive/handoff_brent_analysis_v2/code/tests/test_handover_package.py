"""Handover package acceptance tests."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HANDOFF_ROOT = REPO_ROOT / "handoff_brent_analysis"
TABLES_DIR = HANDOFF_ROOT / "tables"
FIGURES_DIR = HANDOFF_ROOT / "figures"
RAW_DIR = HANDOFF_ROOT / "data" / "raw"
PROCESSED_DIR = HANDOFF_ROOT / "data" / "processed"
REPORT_PATH = HANDOFF_ROOT / "HANDOFF_REPORT.md"
AUDIT_PATH = HANDOFF_ROOT / "METHODOLOGY_AUDIT.md"

if not HANDOFF_ROOT.exists():
    pytestmark = pytest.mark.skip(
        reason="Run `python src/historical_diagnostics.py` before handover package tests."
    )


REQUIRED_CSV_COLUMNS = {
    "data_validation.csv": {
        "source_url", "retrieval_date", "raw_filename", "units", "frequency",
        "first_observation", "last_observation", "missing_values", "non_numeric_values",
        "duplicate_dates", "non_positive_values", "chronological_order",
    },
    "descriptive_statistics.csv": {
        "period", "start_date", "end_date", "observations_prices", "observations_returns",
        "mean", "median", "std", "annualized_vol", "min", "max", "skewness",
        "pearson_kurtosis", "excess_kurtosis", "p01", "p05", "p25", "p50", "p75", "p95", "p99",
    },
    "normality_tests.csv": {
        "period", "jarque_bera_statistic", "jarque_bera_pvalue", "fitted_normal_mean", "fitted_normal_std",
    },
    "empirical_tail_frequencies.csv": {
        "period", "sigma_threshold", "empirical_frequency", "theoretical_normal_probability", "empirical_to_theoretical_ratio",
    },
    "largest_absolute_returns.csv": {
        "period", "date", "price", "previous_price", "log_return", "abs_log_return",
    },
    "rolling_volatility_summary.csv": {
        "window_days", "min", "median", "mean", "max", "date_of_max",
    },
    "autocorrelations.csv": {"series", "lag", "acf"},
    "ljung_box_results.csv": {"series", "lag", "ljung_box_stat", "ljung_box_pvalue"},
    "ar1_results.csv": {
        "series", "intercept", "phi", "intercept_se", "phi_se", "intercept_t", "phi_t", "intercept_p", "phi_p",
        "nobs", "r_squared", "half_life_days", "half_life_note",
    },
    "adf_results.csv": {
        "period", "series", "regression", "autolag", "adf_statistic", "pvalue", "usedlag", "nobs",
        "critical_value_1pct", "critical_value_5pct", "critical_value_10pct", "reject_1pct", "reject_5pct", "reject_10pct", "note",
    },
    "sample_period_comparison.csv": {
        "period", "observations", "mean_return", "annualized_vol", "skewness", "excess_kurtosis",
        "jarque_bera_stat", "jarque_bera_pvalue", "freq_outside_3sigma", "freq_outside_5sigma",
        "acf1_returns", "acf1_squared_returns", "adf_stat_c", "adf_pvalue_c",
    },
    "gbm_assumption_assessment.csv": {
        "assumption", "diagnostic_used", "numerical_result", "interpretation", "assessment", "qualification",
    },
    "gbm_simulation_parameters.csv": {"S0", "mu_annual", "sigma_annual", "horizon_days", "n_paths", "seed"},
    "gbm_backtest_metrics.csv": {"metric", "value"},
    "gbm_backtest_interpretation.csv": {
        "metric", "definition", "preferred_direction", "observed_value", "interpretation", "limitation",
    },
}

REQUIRED_FIGURES = {
    "brent_price_history.png",
    "brent_log_returns.png",
    "return_histogram_normal_overlay.png",
    "normal_qq_plot.png",
    "rolling_volatility_30_90_day.png",
    "acf_returns.png",
    "acf_squared_returns.png",
    "acf_absolute_returns.png",
    "log_price_history.png",
    "sample_period_comparison.png",
    "gbm_backtest_observed_vs_simulated.png",
    "gbm_prediction_intervals.png",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_required_csv_files_exist_and_nonempty() -> None:
    for filename, required_cols in REQUIRED_CSV_COLUMNS.items():
        path = TABLES_DIR / filename
        assert path.exists(), f"Missing required table: {path}"
        df = pd.read_csv(path)
        assert len(df) > 0, f"Table is empty: {filename}"
        assert required_cols.issubset(df.columns), (
            f"Missing columns in {filename}: {required_cols - set(df.columns)}"
        )


def test_required_figures_exist_and_nonzero() -> None:
    for fig in REQUIRED_FIGURES:
        path = FIGURES_DIR / fig
        assert path.exists(), f"Missing required figure: {path}"
        assert path.stat().st_size > 0, f"Figure is zero-size: {path}"


def test_full_sample_dates_consistent_across_tables() -> None:
    desc = pd.read_csv(TABLES_DIR / "descriptive_statistics.csv")
    full = desc.loc[desc["period"] == "full_sample"].iloc[0]
    sample = pd.read_csv(TABLES_DIR / "sample_period_comparison.csv")
    full_cmp = sample.loc[sample["period"] == "full_sample"].iloc[0]
    validation = pd.read_csv(TABLES_DIR / "data_validation.csv").iloc[0]
    assert int(full["observations_prices"]) == int(full_cmp["observations"])
    assert full["start_date"] >= "2000-01-01"
    assert full["end_date"] <= "2025-12-31"
    assert full["start_date"] >= validation["first_observation"]
    assert full["end_date"] <= validation["last_observation"]


def test_annualized_volatility_uses_sqrt_252() -> None:
    proc = pd.read_csv(PROCESSED_DIR / "brent_prices_2000_2025_with_log_features.csv", parse_dates=["Date"])
    desc = pd.read_csv(TABLES_DIR / "descriptive_statistics.csv")
    full = desc.loc[desc["period"] == "full_sample"].iloc[0]
    returns = proc.loc[
        (proc["Date"] >= "2000-01-01") & (proc["Date"] <= "2025-12-31"), "Log_Return"
    ].dropna()
    expected = float(returns.std(ddof=1) * math.sqrt(252))
    assert np.isclose(float(full["annualized_vol"]), expected, atol=1e-12, rtol=0)


def test_returns_are_log_difference_based() -> None:
    proc = pd.read_csv(PROCESSED_DIR / "brent_prices_2000_2025_with_log_features.csv")
    prices = proc["Price_USD_per_barrel"].astype(float)
    expected = np.log(prices / prices.shift(1))
    assert np.allclose(proc["Log_Return"].iloc[1:].to_numpy(), expected.iloc[1:].to_numpy(), equal_nan=True)


def test_raw_data_not_overwritten_in_handoff_copy() -> None:
    validation = pd.read_csv(TABLES_DIR / "data_validation.csv").iloc[0]
    raw_name = validation["raw_filename"]
    source_file = PROJECT_ROOT / "data" / "raw" / raw_name
    copied_file = RAW_DIR / raw_name
    assert source_file.exists() and copied_file.exists()
    assert _sha(source_file) == _sha(copied_file)


def test_processed_dates_sorted_and_unique() -> None:
    proc = pd.read_csv(PROCESSED_DIR / "brent_prices_2000_2025_with_log_features.csv", parse_dates=["Date"])
    assert proc["Date"].is_monotonic_increasing
    assert not proc["Date"].duplicated().any()


def test_report_references_generated_values() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")
    metrics = pd.read_csv(TABLES_DIR / "gbm_backtest_metrics.csv")
    mae = float(metrics.loc[metrics["metric"] == "MAE", "value"].iloc[0])
    rmse = float(metrics.loc[metrics["metric"] == "RMSE", "value"].iloc[0])
    assert f"{mae:.6f}" in report
    assert f"{rmse:.6f}" in report


def test_package_excludes_forbidden_dirs_and_patterns() -> None:
    forbidden = {".venv", "__pycache__", ".ipynb_checkpoints"}
    secret_markers = ["secret", "token", "credential", "password"]
    for path in HANDOFF_ROOT.rglob("*"):
        parts = set(path.parts)
        assert forbidden.isdisjoint(parts), f"Forbidden directory found in package: {path}"
        name_lower = path.name.lower()
        assert not any(marker in name_lower for marker in secret_markers), (
            f"Potential secret-pattern filename found: {path}"
        )


def test_report_and_audit_exist() -> None:
    assert REPORT_PATH.exists() and REPORT_PATH.stat().st_size > 0
    assert AUDIT_PATH.exists() and AUDIT_PATH.stat().st_size > 0
