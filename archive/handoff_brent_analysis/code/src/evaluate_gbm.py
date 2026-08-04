"""Stage 4: Out-of-sample backtesting and evaluation of GBM forecasts.

Evaluates GBM forecast performance against realised Brent crude prices using
an analytical lognormal quantile approach.  Under exact GBM discretisation the
*t*-day-ahead price distribution is:

    S_t | S_0 ~ LogNormal(
        log(S_0) + (mu - 0.5*sigma^2) * t_years,
        sigma^2 * t_years
    )

where t_years = t / 252.  Quantile forecasts are therefore exact and fully
reproducible without any Monte Carlo seed.

Train / test split
------------------
The last ``test_window`` trading days of the cleaned price history form the
hold-out test set.  ``S_0`` is set to the final training price.  ``mu`` and
``sigma`` are taken from the Stage 2 estimates (estimated on the full sample);
this is documented as an assumption and is noted in the printed summary.

Outputs
-------
tables/backtest_path_comparison.csv   – date, actual_price, forecast_p05/50/95
tables/backtest_metrics.csv           – MAE, RMSE, MAPE, coverage, interval_width,
                                        directional_accuracy
figures/backtest_forecast_vs_actual.png  – actual vs p50 + p05-p95 band
figures/backtest_error_distribution.png – histogram of (actual - p50) errors
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252
DEFAULT_TEST_WINDOW = 252  # last N trading days used as hold-out test set

PROCESSED_FILE = (
    Path(__file__).resolve().parents[1]
    / "data" / "processed" / "brent_prices_2000_2025_clean.csv"
)
PARAMS_FILE = (
    Path(__file__).resolve().parents[1]
    / "outputs" / "tables" / "gbm_parameters.csv"
)


def project_root() -> Path:
    """Return the project root directory (brent_gbm_analysis/)."""
    return Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_price_history(path: Path | None = None) -> pd.DataFrame:
    """Load cleaned price history from Stage 1.

    Parameters
    ----------
    path:
        Override path to the cleaned price CSV.  Defaults to the canonical
        Stage 1 output location.

    Returns
    -------
    DataFrame with columns ``Date`` (datetime) and
    ``Price_USD_per_barrel`` (float), sorted chronologically.
    """
    csv_path = path if path is not None else PROCESSED_FILE
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cleaned price file not found: {csv_path}\n"
            "Run `python src/download_data.py` then `python src/prepare_data.py` "
            "(Stage 1) first."
        )
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_gbm_parameters(path: Path | None = None) -> dict[str, float]:
    """Load Stage 2 estimated GBM parameters from CSV.

    Parameters
    ----------
    path:
        Override path to ``gbm_parameters.csv``.  Defaults to the canonical
        Stage 2 output location.

    Returns
    -------
    Dictionary with keys ``mu_annual`` and ``sigma_annual``.
    """
    csv_path = path if path is not None else PARAMS_FILE
    if not csv_path.exists():
        raise FileNotFoundError(
            f"GBM parameter file not found: {csv_path}\n"
            "Run `python src/estimate_gbm.py` (Stage 2) first."
        )
    df = pd.read_csv(csv_path)
    param_map = dict(zip(df["Parameter"], df["Value"]))
    missing = {"mu_annual", "sigma_annual"} - set(param_map.keys())
    if missing:
        raise ValueError(
            f"Required parameters missing from {csv_path}: {missing}"
        )
    return {k: float(param_map[k]) for k in ("mu_annual", "sigma_annual")}


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def split_train_test(
    df: pd.DataFrame,
    test_window: int = DEFAULT_TEST_WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split price history into training and test sets.

    Parameters
    ----------
    df:
        Full price history DataFrame (sorted chronologically).
    test_window:
        Number of trading days to hold out as the test set.

    Returns
    -------
    Tuple ``(train_df, test_df)`` where ``test_df`` contains the last
    ``test_window`` rows of ``df``.
    """
    if test_window <= 0:
        raise ValueError(f"test_window must be positive, got {test_window}.")
    if len(df) <= test_window:
        raise ValueError(
            f"test_window ({test_window}) must be smaller than total rows "
            f"({len(df)})."
        )
    train_df = df.iloc[:-test_window].reset_index(drop=True)
    test_df = df.iloc[-test_window:].reset_index(drop=True)
    return train_df, test_df


# ---------------------------------------------------------------------------
# Analytical GBM forecast quantiles
# ---------------------------------------------------------------------------

def compute_forecast_quantiles(
    S0: float,
    mu: float,
    sigma: float,
    horizon_days: int,
    quantiles: list[float] | None = None,
    dt: float = 1.0 / TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """Compute analytical GBM lognormal quantile forecasts.

    Under GBM, the *t*-step price distribution is exactly lognormal:

        log(S_t / S_0) ~ N((mu - 0.5*sigma^2)*t_yrs, sigma^2 * t_yrs)

    where t_yrs = t * dt.

    Parameters
    ----------
    S0:
        Initial asset price (last training price).
    mu:
        Annualised GBM drift (from Stage 2).
    sigma:
        Annualised GBM volatility (from Stage 2).
    horizon_days:
        Number of trading days to forecast.
    quantiles:
        List of quantile levels to compute.  Defaults to [0.05, 0.50, 0.95].
    dt:
        Time step in years (default: 1/252).

    Returns
    -------
    DataFrame with columns ``day_index``, and one column per quantile
    (e.g. ``forecast_p05``, ``forecast_p50``, ``forecast_p95``).
    Day index runs from 1 to ``horizon_days`` (forecast origin = day 0 not
    included).
    """
    if S0 <= 0:
        raise ValueError(f"S0 must be positive, got {S0}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}.")
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}.")

    if quantiles is None:
        quantiles = [0.05, 0.50, 0.95]

    days = np.arange(1, horizon_days + 1)
    t_years = days * dt  # shape: (horizon_days,)

    drift = (mu - 0.5 * sigma ** 2) * t_years   # shape: (horizon_days,)
    vol = sigma * np.sqrt(t_years)               # shape: (horizon_days,)

    records: list[dict] = []
    for i, day in enumerate(days):
        row: dict[str, float | int] = {"day_index": int(day)}
        for q in quantiles:
            z = float(norm.ppf(q))
            col = f"forecast_p{int(round(q * 100)):02d}"
            row[col] = float(S0 * np.exp(drift[i] + vol[i] * z))
        records.append(row)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def compute_backtest_metrics(
    comparison_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute evaluation metrics from aligned actual vs forecast DataFrame.

    Parameters
    ----------
    comparison_df:
        DataFrame with columns ``actual_price``, ``forecast_p05``,
        ``forecast_p50``, ``forecast_p95``.

    Returns
    -------
    DataFrame with columns ``Metric`` and ``Value``.
    """
    actual = comparison_df["actual_price"].to_numpy(dtype=float)
    p50 = comparison_df["forecast_p50"].to_numpy(dtype=float)
    p05 = comparison_df["forecast_p05"].to_numpy(dtype=float)
    p95 = comparison_df["forecast_p95"].to_numpy(dtype=float)

    errors = actual - p50

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    # MAPE: robust against zero actual prices
    nonzero_mask = actual != 0
    if nonzero_mask.any():
        mape = float(np.mean(np.abs(errors[nonzero_mask] / actual[nonzero_mask])) * 100)
    else:
        mape = float("nan")

    # Interval coverage: fraction of actuals within [p05, p95]
    coverage = float(np.mean((actual >= p05) & (actual <= p95)))

    # Average interval width
    avg_interval_width = float(np.mean(p95 - p05))

    # Directional accuracy: sign of daily actual change vs sign of daily p50 change.
    # Days where either the actual or forecast change is exactly zero are excluded to
    # avoid artefacts from np.sign(0) == 0 never matching a non-zero direction.
    actual_changes = np.diff(actual)
    p50_changes = np.diff(p50)
    if len(actual_changes) > 0 and len(p50_changes) > 0:
        nonzero_mask = (actual_changes != 0) & (p50_changes != 0)
        if nonzero_mask.any():
            dir_acc = float(
                np.mean(
                    np.sign(actual_changes[nonzero_mask])
                    == np.sign(p50_changes[nonzero_mask])
                )
            )
        else:
            dir_acc = float("nan")
    else:
        dir_acc = float("nan")

    rows = [
        ("MAE", mae),
        ("RMSE", rmse),
        ("MAPE_pct", mape),
        ("coverage_p05_p95", coverage),
        ("avg_interval_width", avg_interval_width),
        ("directional_accuracy", dir_acc),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Value"])


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def plot_forecast_vs_actual(
    comparison_df: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    """Save a backtest forecast vs actual chart.

    Parameters
    ----------
    comparison_df:
        DataFrame with columns ``date`` (or ``day_index``), ``actual_price``,
        ``forecast_p05``, ``forecast_p50``, ``forecast_p95``.
    figures_dir:
        Directory where the PNG is written.

    Returns
    -------
    Path to the saved figure.
    """
    fig_path = figures_dir / "backtest_forecast_vs_actual.png"

    x_col = "date" if "date" in comparison_df.columns else "day_index"
    x = comparison_df[x_col]

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.fill_between(
        x,
        comparison_df["forecast_p05"],
        comparison_df["forecast_p95"],
        alpha=0.20,
        color="steelblue",
        label="GBM 5th–95th percentile",
    )
    ax.plot(
        x,
        comparison_df["forecast_p50"],
        color="steelblue",
        linewidth=1.5,
        linestyle="--",
        label="GBM median (p50) forecast",
    )
    ax.plot(
        x,
        comparison_df["actual_price"],
        color="firebrick",
        linewidth=1.5,
        label="Actual price",
    )

    ax.set_xlabel("Date" if x_col == "date" else "Trading day (test horizon)")
    ax.set_ylabel("Brent crude price (USD/barrel)")
    ax.set_title(
        "GBM Backtest: Forecast vs Actual — Europe Brent Spot Price"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    return fig_path


def plot_error_distribution(
    comparison_df: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    """Save a histogram of forecast errors (actual − p50).

    Parameters
    ----------
    comparison_df:
        DataFrame with columns ``actual_price`` and ``forecast_p50``.
    figures_dir:
        Directory where the PNG is written.

    Returns
    -------
    Path to the saved figure.
    """
    fig_path = figures_dir / "backtest_error_distribution.png"

    errors = comparison_df["actual_price"] - comparison_df["forecast_p50"]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(errors, bins=30, edgecolor="none", color="steelblue", alpha=0.8)
    ax.axvline(
        float(errors.mean()),
        color="firebrick",
        linewidth=1.5,
        linestyle="--",
        label=f"Mean error = {errors.mean():.2f}",
    )
    ax.axvline(0, color="black", linewidth=1.0, linestyle=":", label="Zero error")

    ax.set_xlabel("Forecast error: actual − p50 (USD/barrel)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of GBM Forecast Errors — Backtest")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    return fig_path


# ---------------------------------------------------------------------------
# Top-level pipeline function
# ---------------------------------------------------------------------------

def evaluate_gbm(
    cleaned_csv: Path | None = None,
    params_csv: Path | None = None,
    test_window: int = DEFAULT_TEST_WINDOW,
    dt: float = 1.0 / TRADING_DAYS_PER_YEAR,
) -> dict:
    """Run the full Stage 4 GBM backtesting pipeline.

    Parameters
    ----------
    cleaned_csv:
        Optional override path to the Stage 1 cleaned price CSV.
    params_csv:
        Optional override path to the Stage 2 GBM parameters CSV.
    test_window:
        Number of trading days to hold out as the out-of-sample test set
        (default: 252).
    dt:
        Length of each time step in years (default: 1/252).

    Notes
    -----
    ``mu`` and ``sigma`` are the Stage 2 estimates computed from the full
    price history.  Using full-sample parameters is a common simplifying
    assumption for backtesting GBM; it slightly overfits the drift and
    volatility but is standard practice in the dissertation context.

    Returns
    -------
    Dictionary with keys: ``S0``, ``mu``, ``sigma``, ``train_size``,
    ``test_size``, ``comparison_df``, ``metrics_df``,
    ``comparison_csv_path``, ``metrics_csv_path``,
    ``forecast_vs_actual_path``, ``error_distribution_path``.
    """
    root = project_root()
    figures_dir = root / "outputs" / "figures"
    tables_dir = root / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # -- Load inputs ----------------------------------------------------------
    prices_df = load_price_history(cleaned_csv)
    gbm_params = load_gbm_parameters(params_csv)
    mu = gbm_params["mu_annual"]
    sigma = gbm_params["sigma_annual"]

    # -- Train / test split ---------------------------------------------------
    train_df, test_df = split_train_test(prices_df, test_window=test_window)

    S0 = float(train_df["Price_USD_per_barrel"].iloc[-1])
    horizon_days = len(test_df)

    # -- Analytical GBM forecast quantiles ------------------------------------
    forecast_df = compute_forecast_quantiles(
        S0=S0,
        mu=mu,
        sigma=sigma,
        horizon_days=horizon_days,
        quantiles=[0.05, 0.50, 0.95],
        dt=dt,
    )

    # -- Align with actual prices ---------------------------------------------
    assert len(test_df) == len(forecast_df), (
        f"test_df length ({len(test_df)}) does not match forecast_df length "
        f"({len(forecast_df)}); cannot align actual and forecast series."
    )
    comparison_df = pd.DataFrame(
        {
            "date": test_df["Date"].values,
            "day_index": forecast_df["day_index"].values,
            "actual_price": test_df["Price_USD_per_barrel"].values,
            "forecast_p05": forecast_df["forecast_p05"].values,
            "forecast_p50": forecast_df["forecast_p50"].values,
            "forecast_p95": forecast_df["forecast_p95"].values,
        }
    )

    # -- Compute evaluation metrics -------------------------------------------
    metrics_df = compute_backtest_metrics(comparison_df)

    # -- Save tables ----------------------------------------------------------
    comparison_csv_path = tables_dir / "backtest_path_comparison.csv"
    metrics_csv_path = tables_dir / "backtest_metrics.csv"
    comparison_df.to_csv(comparison_csv_path, index=False)
    metrics_df.to_csv(metrics_csv_path, index=False)

    # -- Save figures ---------------------------------------------------------
    forecast_vs_actual_path = plot_forecast_vs_actual(comparison_df, figures_dir)
    error_distribution_path = plot_error_distribution(comparison_df, figures_dir)

    # -- Console summary ------------------------------------------------------
    metrics_map = dict(zip(metrics_df["Metric"], metrics_df["Value"]))
    print("=== Stage 4: GBM Backtest Evaluation ===")
    print(f"  Total observations   : {len(prices_df)}")
    print(f"  Training observations: {len(train_df)}")
    print(f"  Test window          : {horizon_days} trading days")
    print(f"  Train period         : {train_df['Date'].iloc[0].date()} "
          f"to {train_df['Date'].iloc[-1].date()}")
    print(f"  Test period          : {test_df['Date'].iloc[0].date()} "
          f"to {test_df['Date'].iloc[-1].date()}")
    print(f"  S0 (forecast origin) : {S0:.4f} USD/barrel")
    print(f"  mu_annual            : {mu:.6f}  ({mu * 100:.2f} %)")
    print(f"  sigma_annual         : {sigma:.6f}  ({sigma * 100:.2f} %)")
    print(f"  NOTE: mu/sigma estimated from full sample (Stage 2 assumption).")
    print()
    print("--- Key evaluation metrics ---")
    print(f"  MAE                  : {metrics_map.get('MAE', float('nan')):.4f}")
    print(f"  RMSE                 : {metrics_map.get('RMSE', float('nan')):.4f}")
    print(f"  MAPE                 : {metrics_map.get('MAPE_pct', float('nan')):.2f} %")
    print(f"  Coverage [p05,p95]   : {metrics_map.get('coverage_p05_p95', float('nan')):.2%}")
    print(f"  Avg interval width   : {metrics_map.get('avg_interval_width', float('nan')):.4f}")
    print(f"  Directional accuracy : {metrics_map.get('directional_accuracy', float('nan')):.2%}")
    print()
    print(f"Backtest comparison   : {comparison_csv_path}")
    print(f"Backtest metrics      : {metrics_csv_path}")
    print(f"Forecast vs actual    : {forecast_vs_actual_path}")
    print(f"Error distribution    : {error_distribution_path}")

    return {
        "S0": S0,
        "mu": mu,
        "sigma": sigma,
        "train_size": len(train_df),
        "test_size": horizon_days,
        "comparison_df": comparison_df,
        "metrics_df": metrics_df,
        "comparison_csv_path": comparison_csv_path,
        "metrics_csv_path": metrics_csv_path,
        "forecast_vs_actual_path": forecast_vs_actual_path,
        "error_distribution_path": error_distribution_path,
    }


if __name__ == "__main__":
    evaluate_gbm()
