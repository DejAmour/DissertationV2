"""Rolling-origin backtest tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.evaluate_gbm import load_price_history
from src.prepare_data import prepare_data
from src.rolling_origin_backtest import (
    compute_rolling_origin_forecasts,
    compute_rolling_origin_metrics,
    run_rolling_origin_backtest,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "brent_prices_2000_2025_clean.csv"


def _ensure_prices() -> pd.DataFrame:
    if not PROCESSED_FILE.exists():
        prepare_data()
    return load_price_history()


def _ensure_rolling_outputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    result = run_rolling_origin_backtest()
    return result["forecasts_df"], result["metrics_df"]


def test_origin_date_precedes_target_date() -> None:
    forecasts, _ = _ensure_rolling_outputs()
    assert (pd.to_datetime(forecasts["origin_date"]) < pd.to_datetime(forecasts["target_date"])).all()


def test_valid_observation_distance_equals_horizon() -> None:
    prices = _ensure_prices().reset_index(drop=True)
    date_to_index = {row.Date.date().isoformat(): idx for idx, row in prices.iterrows()}
    forecasts, _ = _ensure_rolling_outputs()
    distances = forecasts.apply(
        lambda row: date_to_index[row["target_date"]] - date_to_index[row["origin_date"]],
        axis=1,
    )
    assert (distances == forecasts["horizon"]).all()


def test_no_target_or_later_observation_enters_estimation() -> None:
    prices = _ensure_prices().reset_index(drop=True)
    date_to_index = {row.Date.date().isoformat(): idx for idx, row in prices.iterrows()}
    forecasts, _ = _ensure_rolling_outputs()
    expected_training_prices = forecasts["origin_date"].map(lambda x: date_to_index[x] + 1)
    assert (forecasts["n_training_prices"] == expected_training_prices).all()
    assert (forecasts["n_training_returns"] == expected_training_prices - 1).all()


def test_altering_future_observations_does_not_change_earlier_forecast() -> None:
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(
        {
            "Date": pd.date_range("2020-01-01", periods=400, freq="B"),
            "Price_USD_per_barrel": 75 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 400))),
        }
    )
    forecasts = compute_rolling_origin_forecasts(prices, horizons=(1, 5, 20), heldout_window=50)
    row = forecasts.loc[(forecasts["horizon"] == 5)].iloc[0]

    altered = prices.copy()
    changed_dates = altered["Date"] > pd.Timestamp(row["target_date"])
    altered.loc[changed_dates, "Price_USD_per_barrel"] *= 4
    altered_forecasts = compute_rolling_origin_forecasts(altered, horizons=(1, 5, 20), heldout_window=50)
    altered_row = altered_forecasts.loc[
        (altered_forecasts["horizon"] == row["horizon"])
        & (altered_forecasts["origin_date"] == row["origin_date"])
        & (altered_forecasts["target_date"] == row["target_date"])
    ].iloc[0]

    for column in ["estimated_mu", "estimated_sigma", "forecast_p05", "forecast_p50", "forecast_p95"]:
        assert np.isclose(row[column], altered_row[column])


def test_s0_equals_origin_price() -> None:
    prices = _ensure_prices().reset_index(drop=True)
    date_to_price = {row.Date.date().isoformat(): float(row.Price_USD_per_barrel) for _, row in prices.iterrows()}
    forecasts, _ = _ensure_rolling_outputs()
    expected = forecasts["origin_date"].map(date_to_price)
    assert np.allclose(forecasts["S0"], expected)


def test_quantiles_are_ordered() -> None:
    forecasts, _ = _ensure_rolling_outputs()
    assert (forecasts["forecast_p05"] <= forecasts["forecast_p50"]).all()
    assert (forecasts["forecast_p50"] <= forecasts["forecast_p95"]).all()


def test_interval_width_is_positive() -> None:
    forecasts, _ = _ensure_rolling_outputs()
    assert (forecasts["interval_width"] > 0).all()


def test_coverage_indicators_are_mutually_exclusive() -> None:
    forecasts, _ = _ensure_rolling_outputs()
    indicator_sum = forecasts[["below_p05", "inside_interval", "above_p95"]].sum(axis=1)
    assert (indicator_sum == 1).all()


def test_tail_probabilities_sum_to_one() -> None:
    forecasts, _ = _ensure_rolling_outputs()
    grouped = forecasts.groupby("horizon")[["below_p05", "inside_interval", "above_p95"]].mean()
    sums = grouped.sum(axis=1)
    assert np.allclose(sums.to_numpy(), np.ones(len(sums)))


def test_targets_all_lie_inside_heldout_period() -> None:
    prices = _ensure_prices()
    heldout_dates = set(prices.iloc[-252:]["Date"].dt.date.astype(str))
    forecasts, _ = _ensure_rolling_outputs()
    assert set(forecasts["target_date"]).issubset(heldout_dates)


def test_horizons_are_exactly_1_5_20() -> None:
    forecasts, metrics = _ensure_rolling_outputs()
    assert set(forecasts["horizon"]) == {1, 5, 20}
    assert set(metrics["horizon"]) == {1, 5, 20}


def test_metrics_table_matches_forecast_level_aggregation() -> None:
    forecasts, metrics = _ensure_rolling_outputs()
    expected = compute_rolling_origin_metrics(forecasts).sort_values("horizon").reset_index(drop=True)
    observed = metrics.sort_values("horizon").reset_index(drop=True)
    pd.testing.assert_frame_equal(observed, expected)
