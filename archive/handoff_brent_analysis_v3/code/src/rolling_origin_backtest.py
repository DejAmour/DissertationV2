"""Rolling-origin GBM forecast evaluation on a held-out Brent target period."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from evaluate_gbm import (
        TRADING_DAYS_PER_YEAR,
        compute_forecast_quantiles,
        estimate_params_from_training,
        load_price_history,
    )
except ModuleNotFoundError:  # pragma: no cover - import path varies between script and package usage
    from src.evaluate_gbm import (
        TRADING_DAYS_PER_YEAR,
        compute_forecast_quantiles,
        estimate_params_from_training,
        load_price_history,
    )

DEFAULT_HORIZONS = (1, 5, 20)
DEFAULT_HELDOUT_WINDOW = 252


def project_root() -> Path:
    """Return the project root directory (brent_gbm_analysis/)."""
    return Path(__file__).resolve().parents[1]


def _source_note() -> str:
    return "Source: Europe Brent Spot Price FOB (USD/barrel); underlying EIA series (DCOILBRENTEU)."


def compute_rolling_origin_forecasts(
    prices_df: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    heldout_window: int = DEFAULT_HELDOUT_WINDOW,
) -> pd.DataFrame:
    """Compute rolling-origin GBM forecasts for each held-out target date."""
    if heldout_window <= 0:
        raise ValueError("heldout_window must be positive.")
    if len(prices_df) <= heldout_window:
        raise ValueError("heldout_window must be smaller than the number of prices.")
    if set(horizons) != {1, 5, 20}:
        raise ValueError(f"horizons must equal {{1, 5, 20}}, got {set(horizons)}")

    prices_df = prices_df.sort_values("Date").reset_index(drop=True)
    heldout_start_idx = len(prices_df) - heldout_window

    records: list[dict[str, object]] = []
    for horizon in sorted(horizons):
        for target_idx in range(heldout_start_idx, len(prices_df)):
            origin_idx = target_idx - horizon
            if origin_idx < 0:
                raise ValueError("Insufficient history for requested horizon.")

            training_df = prices_df.iloc[: origin_idx + 1].reset_index(drop=True)
            target_row = prices_df.iloc[target_idx]
            origin_row = prices_df.iloc[origin_idx]

            params = estimate_params_from_training(training_df)
            quantiles = compute_forecast_quantiles(
                S0=float(origin_row["Price_USD_per_barrel"]),
                mu=float(params["mu_annual"]),
                sigma=float(params["sigma_annual"]),
                horizon_days=horizon,
                quantiles=[0.05, 0.50, 0.95],
            ).iloc[-1]

            observed_target_price = float(target_row["Price_USD_per_barrel"])
            forecast_p05 = float(quantiles["forecast_p05"])
            forecast_p50 = float(quantiles["forecast_p50"])
            forecast_p95 = float(quantiles["forecast_p95"])
            absolute_error = abs(observed_target_price - forecast_p50)
            squared_error = (observed_target_price - forecast_p50) ** 2
            absolute_percentage_error = absolute_error / observed_target_price * 100

            below_p05 = int(observed_target_price < forecast_p05)
            above_p95 = int(observed_target_price > forecast_p95)
            inside_interval = int((below_p05 == 0) and (above_p95 == 0))

            records.append(
                {
                    "horizon": int(horizon),
                    "origin_date": pd.Timestamp(origin_row["Date"]).date().isoformat(),
                    "target_date": pd.Timestamp(target_row["Date"]).date().isoformat(),
                    "S0": float(origin_row["Price_USD_per_barrel"]),
                    "n_training_prices": int(len(training_df)),
                    "n_training_returns": int(params["n_returns"]),
                    "estimated_mu": float(params["mu_annual"]),
                    "estimated_sigma": float(params["sigma_annual"]),
                    "observed_target_price": observed_target_price,
                    "forecast_p05": forecast_p05,
                    "forecast_p50": forecast_p50,
                    "forecast_p95": forecast_p95,
                    "interval_width": float(forecast_p95 - forecast_p05),
                    "below_p05": below_p05,
                    "inside_interval": inside_interval,
                    "above_p95": above_p95,
                    "absolute_error": float(absolute_error),
                    "squared_error": float(squared_error),
                    "absolute_percentage_error": float(absolute_percentage_error),
                }
            )

    forecasts = pd.DataFrame.from_records(records)
    if forecasts.empty:
        raise ValueError("No rolling-origin forecasts were generated.")
    return forecasts


def compute_rolling_origin_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate forecast-level rolling-origin results by horizon."""
    rows: list[dict[str, float | int]] = []
    for horizon, group in forecasts.groupby("horizon", sort=True):
        lower = float(group["below_p05"].mean())
        coverage = float(group["inside_interval"].mean())
        upper = float(group["above_p95"].mean())
        if not np.isclose(lower + coverage + upper, 1.0, atol=1e-12, rtol=0):
            raise ValueError(
                f"Rolling-origin tail frequencies do not sum to 1 for horizon {horizon}: "
                f"{lower + coverage + upper}"
            )

        rows.append(
            {
                "horizon": int(horizon),
                "n_forecasts": int(len(group)),
                "MAE": float(group["absolute_error"].mean()),
                "RMSE": float(np.sqrt(group["squared_error"].mean())),
                "MAPE": float(group["absolute_percentage_error"].mean()),
                "coverage_90": coverage,
                "avg_interval_width": float(group["interval_width"].mean()),
                "median_interval_width": float(group["interval_width"].median()),
                "lower_tail_violation_freq": lower,
                "upper_tail_violation_freq": upper,
                "mean_estimated_mu": float(group["estimated_mu"].mean()),
                "median_estimated_mu": float(group["estimated_mu"].median()),
                "mean_estimated_sigma": float(group["estimated_sigma"].mean()),
                "median_estimated_sigma": float(group["estimated_sigma"].median()),
            }
        )
    return pd.DataFrame(rows)


def _plot_one_day_forecasts(forecasts: pd.DataFrame, figures_dir: Path) -> Path:
    one_day = forecasts.loc[forecasts["horizon"] == 1].copy()
    one_day["target_date"] = pd.to_datetime(one_day["target_date"])

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.fill_between(
        one_day["target_date"],
        one_day["forecast_p05"],
        one_day["forecast_p95"],
        alpha=0.22,
        color="steelblue",
        label="1-day 90% interval (p05-p95)",
    )
    ax.plot(one_day["target_date"], one_day["forecast_p50"], linestyle="--", color="navy", linewidth=1.3, label="1-day median forecast")
    ax.plot(one_day["target_date"], one_day["observed_target_price"], color="firebrick", linewidth=1.3, label="Observed price")
    ax.set_title(
        "Rolling-Origin 1-Day GBM Forecasts\n"
        f"Forecast period: {one_day['target_date'].min().date()} to {one_day['target_date'].max().date()}"
    )
    ax.set_xlabel("Target date")
    ax.set_ylabel("Price (USD/barrel)")
    ax.legend(loc="upper left")
    fig.text(0.01, 0.01, _source_note(), fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = figures_dir / "rolling_origin_one_day_forecasts.png"
    fig.savefig(path, dpi=320)
    plt.close(fig)
    return path


def _plot_coverage(metrics: pd.DataFrame, figures_dir: Path, heldout_dates: pd.Series) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(metrics["horizon"].astype(str), metrics["coverage_90"], color="steelblue", label="Observed coverage")
    ax.axhline(0.90, color="firebrick", linestyle="--", linewidth=1.3, label="90% reference")
    ax.set_title(
        "Rolling-Origin 90% Interval Coverage by Horizon\n"
        f"Forecast period: {heldout_dates.min().date()} to {heldout_dates.max().date()}"
    )
    ax.set_xlabel("Forecast horizon (trading days)")
    ax.set_ylabel("Coverage frequency")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    fig.text(0.01, 0.01, _source_note(), fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = figures_dir / "rolling_origin_coverage_by_horizon.png"
    fig.savefig(path, dpi=320)
    plt.close(fig)
    return path


def _plot_interval_width(metrics: pd.DataFrame, figures_dir: Path, heldout_dates: pd.Series) -> Path:
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, metrics["avg_interval_width"], width=width, color="steelblue", label="Average width")
    ax.bar(x + width / 2, metrics["median_interval_width"], width=width, color="darkorange", label="Median width")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["horizon"].astype(str))
    ax.set_title(
        "Rolling-Origin Interval Width by Horizon\n"
        f"Forecast period: {heldout_dates.min().date()} to {heldout_dates.max().date()}"
    )
    ax.set_xlabel("Forecast horizon (trading days)")
    ax.set_ylabel("Interval width (USD/barrel)")
    ax.legend(loc="upper left")
    fig.text(0.01, 0.01, _source_note(), fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = figures_dir / "rolling_origin_interval_width_by_horizon.png"
    fig.savefig(path, dpi=320)
    plt.close(fig)
    return path


def _plot_errors(metrics: pd.DataFrame, figures_dir: Path, heldout_dates: pd.Series) -> Path:
    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width / 2, metrics["MAE"], width=width, color="seagreen", label="MAE")
    ax.bar(x + width / 2, metrics["RMSE"], width=width, color="mediumpurple", label="RMSE")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics["horizon"].astype(str))
    ax.set_title(
        "Rolling-Origin GBM Forecast Errors by Horizon\n"
        f"Forecast period: {heldout_dates.min().date()} to {heldout_dates.max().date()}"
    )
    ax.set_xlabel("Forecast horizon (trading days)")
    ax.set_ylabel("Forecast error (USD/barrel)")
    ax.legend(loc="upper left")
    fig.text(0.01, 0.01, _source_note(), fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    path = figures_dir / "rolling_origin_errors_by_horizon.png"
    fig.savefig(path, dpi=320)
    plt.close(fig)
    return path


def run_rolling_origin_backtest(
    cleaned_csv: Path | None = None,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    heldout_window: int = DEFAULT_HELDOUT_WINDOW,
) -> dict[str, object]:
    """Run the rolling-origin evaluation and persist outputs under outputs/."""
    root = project_root()
    tables_dir = root / "outputs" / "tables"
    figures_dir = root / "outputs" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    prices_df = load_price_history(cleaned_csv)
    forecasts = compute_rolling_origin_forecasts(prices_df, horizons=horizons, heldout_window=heldout_window)
    metrics = compute_rolling_origin_metrics(forecasts)

    forecasts_path = tables_dir / "rolling_origin_forecasts.csv"
    metrics_path = tables_dir / "rolling_origin_metrics_by_horizon.csv"
    forecasts.to_csv(forecasts_path, index=False)
    metrics.to_csv(metrics_path, index=False)

    heldout_dates = prices_df.iloc[-heldout_window:]["Date"]
    figure_paths = {
        "one_day": _plot_one_day_forecasts(forecasts, figures_dir),
        "coverage": _plot_coverage(metrics, figures_dir, heldout_dates),
        "interval_width": _plot_interval_width(metrics, figures_dir, heldout_dates),
        "errors": _plot_errors(metrics, figures_dir, heldout_dates),
    }

    print("=== Rolling-Origin GBM Backtest ===")
    print(f"  Held-out target period : {heldout_dates.min().date()} to {heldout_dates.max().date()}")
    print(f"  Trading-day horizons   : {', '.join(str(h) for h in sorted(horizons))}")
    print(f"  Annualisation factor   : {TRADING_DAYS_PER_YEAR}")
    print(f"  Forecast table         : {forecasts_path}")
    print(f"  Metrics table          : {metrics_path}")
    for name, path in figure_paths.items():
        print(f"  Figure ({name})         : {path}")

    return {
        "forecasts_df": forecasts,
        "metrics_df": metrics,
        "forecasts_path": forecasts_path,
        "metrics_path": metrics_path,
        "figure_paths": figure_paths,
    }


if __name__ == "__main__":
    run_rolling_origin_backtest()
