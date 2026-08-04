"""V3 handover package acceptance tests."""

from __future__ import annotations

import math
from pathlib import Path
import zipfile

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HANDOFF_ROOT = REPO_ROOT / "handoff_brent_analysis_v3"
TABLES_DIR = HANDOFF_ROOT / "tables"
FIGURES_DIR = HANDOFF_ROOT / "figures"
LOGS_DIR = HANDOFF_ROOT / "logs"
REPORT_PATH = HANDOFF_ROOT / "HANDOFF_REPORT_V3.md"
AUDIT_PATH = HANDOFF_ROOT / "METHODOLOGY_AUDIT_V3.md"
CORRECTIONS_PATH = HANDOFF_ROOT / "CORRECTIONS_LOG.md"
ARCHIVE_PATH = REPO_ROOT / "brent_gbm_handover_v3.zip"

REQUIRED_TOP_LEVEL = {
    REPORT_PATH.name,
    AUDIT_PATH.name,
    CORRECTIONS_PATH.name,
    "requirements.txt",
    "requirements-lock.txt",
}

REQUIRED_TABLE_COLUMNS = {
    "data_validation.csv": {
        "source_url", "source_description", "retrieval_date", "raw_filename", "units", "frequency",
        "requested_start", "requested_end", "first_observation", "last_observation",
        "first_valid_observation_in_window", "first_processed_observation", "last_processed_observation",
        "missing_values", "non_numeric_values", "duplicate_dates", "non_positive_values",
        "chronological_order", "window_applied_correctly", "end_date_present",
    },
    "fixed_origin_training_parameters.csv": {
        "training_start", "training_end", "test_start", "test_end", "n_training_prices",
        "n_training_returns", "n_test_prices", "S0", "mean_daily_log_return_training",
        "std_daily_log_return_training", "mu_annual_training", "sigma_annual_training",
        "annualisation_factor",
    },
    "fixed_origin_backtest_metrics.csv": {"metric_name", "value"},
    "gbm_backtest_interpretation.csv": {
        "metric_name", "definition", "observed_value", "preferred_direction", "interpretation", "limitation",
    },
    "rolling_origin_forecasts.csv": {
        "horizon", "origin_date", "target_date", "S0", "n_training_prices", "n_training_returns",
        "estimated_mu", "estimated_sigma", "observed_target_price", "forecast_p05", "forecast_p50",
        "forecast_p95", "interval_width", "below_p05", "inside_interval", "above_p95",
        "absolute_error", "squared_error", "absolute_percentage_error",
    },
    "rolling_origin_metrics_by_horizon.csv": {
        "horizon", "n_forecasts", "MAE", "RMSE", "MAPE", "coverage_90", "avg_interval_width",
        "median_interval_width", "lower_tail_violation_freq", "upper_tail_violation_freq",
        "mean_estimated_mu", "median_estimated_mu", "mean_estimated_sigma", "median_estimated_sigma",
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
    "fixed_origin_observed_vs_median.png",
    "fixed_origin_prediction_interval.png",
    "rolling_origin_one_day_forecasts.png",
    "rolling_origin_coverage_by_horizon.png",
    "rolling_origin_interval_width_by_horizon.png",
    "rolling_origin_errors_by_horizon.png",
}

REQUIRED_LOGS = {
    "test_results_all.txt",
    "test_results_data.txt",
    "test_results_historical.txt",
    "test_results_fixed_origin.txt",
    "test_results_rolling_origin.txt",
    "test_results_handover.txt",
}


def test_required_top_level_artifacts_exist() -> None:
    assert HANDOFF_ROOT.exists()
    assert REQUIRED_TOP_LEVEL.issubset({path.name for path in HANDOFF_ROOT.iterdir()})
    assert ARCHIVE_PATH.exists() and ARCHIVE_PATH.stat().st_size > 0


def test_required_csv_files_exist_nonempty_and_match_schema() -> None:
    for filename, required_columns in REQUIRED_TABLE_COLUMNS.items():
        path = TABLES_DIR / filename
        assert path.exists(), f"Missing required table: {path}"
        df = pd.read_csv(path)
        assert not df.empty, f"Table is empty: {filename}"
        assert required_columns.issubset(df.columns), (
            f"Missing columns in {filename}: {required_columns - set(df.columns)}"
        )


def test_required_figure_files_exist_and_are_nonzero() -> None:
    for filename in REQUIRED_FIGURES:
        path = FIGURES_DIR / filename
        assert path.exists(), f"Missing required figure: {path}"
        assert path.stat().st_size > 0, f"Figure is empty: {path}"


def test_required_log_files_contain_actual_pytest_output() -> None:
    for filename in REQUIRED_LOGS:
        path = LOGS_DIR / filename
        assert path.exists(), f"Missing required log: {path}"
        content = path.read_text(encoding="utf-8")
        assert "Test results will be populated" not in content
        assert len(content.strip()) > 0
        if filename not in {"test_results_handover.txt", "test_results_all.txt"}:
            assert "passed" in content or "failed" in content or "skipped" in content


def test_report_values_match_csv_values() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")
    fixed_params = pd.read_csv(TABLES_DIR / "fixed_origin_training_parameters.csv").iloc[0]
    fixed_metrics = pd.read_csv(TABLES_DIR / "fixed_origin_backtest_metrics.csv")
    rolling_metrics = pd.read_csv(TABLES_DIR / "rolling_origin_metrics_by_horizon.csv")

    mae = float(fixed_metrics.loc[fixed_metrics["metric_name"] == "MAE", "value"].iloc[0])
    rmse = float(fixed_metrics.loc[fixed_metrics["metric_name"] == "RMSE", "value"].iloc[0])
    assert fixed_params["training_start"] in report
    assert fixed_params["test_end"] in report
    assert f"{mae:.8f}" in report
    assert f"{rmse:.8f}" in report
    for horizon in (1, 5, 20):
        row = rolling_metrics.loc[rolling_metrics["horizon"] == horizon].iloc[0]
        assert f"**{horizon}-day horizon**" in report
        assert f"{row['MAE']:.6f}" in report


def test_report_audit_and_corrections_are_nonempty() -> None:
    for path in [REPORT_PATH, AUDIT_PATH, CORRECTIONS_PATH]:
        assert path.exists()
        assert path.stat().st_size > 0


def test_archive_contains_v3_handoff_files() -> None:
    with zipfile.ZipFile(ARCHIVE_PATH) as zf:
        names = set(zf.namelist())
    for required in [
        f"handoff_brent_analysis_v3/{REPORT_PATH.name}",
        f"handoff_brent_analysis_v3/{AUDIT_PATH.name}",
        "handoff_brent_analysis_v3/CORRECTIONS_LOG.md",
        "handoff_brent_analysis_v3/tables/rolling_origin_metrics_by_horizon.csv",
        "handoff_brent_analysis_v3/figures/rolling_origin_errors_by_horizon.png",
    ]:
        assert required in names


def test_handover_package_excludes_forbidden_directories_and_secret_patterns() -> None:
    forbidden = {".venv", "__pycache__", ".ipynb_checkpoints", ".pytest_cache"}
    secret_markers = ("secret", "token", "credential", "password")
    for path in HANDOFF_ROOT.rglob("*"):
        assert forbidden.isdisjoint(path.parts), f"Forbidden directory found in package: {path}"
        lower_name = path.name.lower()
        assert not any(marker in lower_name for marker in secret_markers), (
            f"Potential secret-pattern file found in package: {path}"
        )


def test_fixed_origin_metrics_principal_table_omits_directional_accuracy() -> None:
    metrics = pd.read_csv(TABLES_DIR / "fixed_origin_backtest_metrics.csv")
    assert "directional_accuracy" not in set(metrics["metric_name"])


def test_rolling_metrics_cover_required_horizons_only() -> None:
    metrics = pd.read_csv(TABLES_DIR / "rolling_origin_metrics_by_horizon.csv")
    assert set(metrics["horizon"]) == {1, 5, 20}


def test_methodology_audit_has_corrected_fixed_origin_language() -> None:
    audit = AUDIT_PATH.read_text(encoding="utf-8").lower()
    assert "split **before** estimation" in audit
    assert "training-only" in audit or "training only" in audit
    assert "full-sample stage 2 parameters" in audit
    assert "removed" in audit


def test_full_sample_dates_preserved_in_descriptive_statistics() -> None:
    desc = pd.read_csv(TABLES_DIR / "descriptive_statistics.csv")
    full = desc.loc[desc["period"] == "full_sample"].iloc[0]
    assert full["start_date"] == "2000-01-04"
    assert full["end_date"] == "2025-12-31"
    assert math.isclose(float(full["observations_prices"]), 6599)
