"""Fixed-origin held-out backtest tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_gbm import (
    estimate_params_from_training,
    evaluate_gbm,
    load_price_history,
    split_train_test,
)
from src.prepare_data import prepare_data
from src.estimate_gbm import estimate_gbm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "brent_prices_2000_2025_clean.csv"


def _ensure_fixed_origin_result() -> dict[str, object]:
    if not PROCESSED_FILE.exists():
        prepare_data()
    estimate_gbm()
    return evaluate_gbm()


def test_training_end_precedes_test_start() -> None:
    result = _ensure_fixed_origin_result()
    assert pd.Timestamp(result["training_end"]) < pd.Timestamp(result["test_start"])


def test_s0_equals_final_training_price() -> None:
    result = _ensure_fixed_origin_result()
    prices = load_price_history()
    train_df, _ = split_train_test(prices, test_window=252)
    assert float(result["S0"]) == float(train_df["Price_USD_per_barrel"].iloc[-1])


def test_training_params_unchanged_when_test_prices_altered() -> None:
    prices = load_price_history() if PROCESSED_FILE.exists() else load_price_history(prepare_data()[0])
    train_df, test_df = split_train_test(prices, test_window=252)
    original = estimate_params_from_training(train_df)

    altered = prices.copy()
    altered.loc[altered.index[-len(test_df):], "Price_USD_per_barrel"] *= 10
    altered_train_df, _ = split_train_test(altered, test_window=252)
    altered_params = estimate_params_from_training(altered_train_df)

    assert original["n_returns"] == altered_params["n_returns"]
    assert original["mean_daily"] == altered_params["mean_daily"]
    assert original["std_daily"] == altered_params["std_daily"]
    assert original["mu_annual"] == altered_params["mu_annual"]
    assert original["sigma_annual"] == altered_params["sigma_annual"]


def test_test_prices_are_excluded_from_estimation_returns() -> None:
    prices = load_price_history() if PROCESSED_FILE.exists() else load_price_history(prepare_data()[0])
    train_df, test_df = split_train_test(prices, test_window=252)
    params = estimate_params_from_training(train_df)

    train_returns = np.log(train_df["Price_USD_per_barrel"] / train_df["Price_USD_per_barrel"].shift(1)).dropna()
    cross_boundary_return = np.log(
        float(test_df["Price_USD_per_barrel"].iloc[0]) / float(train_df["Price_USD_per_barrel"].iloc[-1])
    )

    assert params["n_returns"] == len(train_returns)
    assert np.isclose(params["mean_daily"], float(train_returns.mean()))
    assert not np.isclose(params["mean_daily"], float((train_returns.sum() + cross_boundary_return) / (len(train_returns) + 1)))


def test_corrected_metrics_match_independent_recalculation() -> None:
    result = _ensure_fixed_origin_result()
    comparison = result["comparison_df"]
    actual = comparison["actual_price"].to_numpy(dtype=float)
    p05 = comparison["forecast_p05"].to_numpy(dtype=float)
    p50 = comparison["forecast_p50"].to_numpy(dtype=float)
    p95 = comparison["forecast_p95"].to_numpy(dtype=float)
    metrics = dict(zip(result["metrics_df"]["Metric"], result["metrics_df"]["Value"]))

    errors = actual - p50
    assert np.isclose(metrics["MAE"], np.mean(np.abs(errors)))
    assert np.isclose(metrics["RMSE"], np.sqrt(np.mean(errors ** 2)))
    assert np.isclose(metrics["MAPE_pct"], np.mean(np.abs(errors / actual)) * 100)
    assert np.isclose(metrics["coverage_p05_p95"], np.mean((actual >= p05) & (actual <= p95)))
    assert np.isclose(metrics["avg_interval_width"], np.mean(p95 - p05))
