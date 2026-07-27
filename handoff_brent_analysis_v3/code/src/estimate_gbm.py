"""Stage 2: Estimate GBM parameters from cleaned Brent crude oil price data.

Geometric Brownian Motion (GBM) model:
    dS = mu * S * dt + sigma * S * dW

where:
    S     = asset price
    mu    = drift (annualized, continuously compounded)
    sigma = volatility (annualized)
    dW    = Wiener process increment

Estimation approach (using daily log returns r_t = ln(S_t / S_{t-1})):

    sigma_annual = std(r_t) * sqrt(T)
        where T = 252 (trading days per year)
        This is the annualized historical volatility.

    mu_annual = mean(r_t) * T + 0.5 * sigma_annual^2
        The second term (Ito correction) converts the mean of log returns
        (which estimates mu - 0.5*sigma^2 under GBM) back to the true
        drift mu.  This matches the GBM parameter directly.

        Equivalently: mu_annual = mean(r_t) * T + 0.5 * var(r_t) * T
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for reproducibility
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

PROCESSED_FILE = (
    Path(__file__).resolve().parents[1]
    / "data" / "processed" / "brent_prices_2000_2025_clean.csv"
)


def project_root() -> Path:
    """Return the project root directory (brent_gbm_analysis/)."""
    return Path(__file__).resolve().parents[1]


def load_cleaned_prices(path: Path | None = None) -> pd.DataFrame:
    """Load Stage 1 cleaned prices.

    Parameters
    ----------
    path:
        Override path to the cleaned CSV.  Defaults to the canonical Stage 1
        output location.

    Returns
    -------
    pd.DataFrame with columns ``Date`` (datetime) and
    ``Price_USD_per_barrel`` (float).
    """
    csv_path = path if path is not None else PROCESSED_FILE
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cleaned price file not found: {csv_path}\n"
            "Run `python src/download_data.py` then `python src/prepare_data.py` first."
        )
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute daily log returns from a price series.

    log_return_t = ln(P_t / P_{t-1})
    """
    return np.log(prices / prices.shift(1)).dropna()


def estimate_gbm_parameters(log_returns: pd.Series) -> dict[str, float]:
    """Estimate annualized GBM drift and volatility from daily log returns.

    Parameters
    ----------
    log_returns:
        Series of daily log returns (NaN values are excluded automatically).

    Returns
    -------
    Dictionary with keys:
        ``sigma_annual`` – annualized volatility
        ``mu_annual``    – annualized GBM drift (Ito-corrected)
        ``n_returns``    – number of return observations used
        ``mean_daily``   – mean daily log return
        ``std_daily``    – std dev of daily log returns
        ``min_daily``    – minimum daily log return
        ``max_daily``    – maximum daily log return

    """
    r = log_returns.dropna()
    n = len(r)
    if n == 0:
        raise ValueError("No valid log returns to estimate GBM parameters.")

    mean_r = float(r.mean())
    std_r = float(r.std(ddof=1))
    min_r = float(r.min())
    max_r = float(r.max())

    T = TRADING_DAYS_PER_YEAR

    # Annualized volatility: sigma = std(r) * sqrt(252)
    sigma_annual = std_r * math.sqrt(T)

    # Annualized drift: mu = mean(r)*T + 0.5*sigma^2
    # Under GBM: E[r_t] = (mu - 0.5*sigma^2) / T  =>  mu = mean(r)*T + 0.5*sigma^2
    mu_annual = mean_r * T + 0.5 * sigma_annual ** 2

    return {
        "sigma_annual": sigma_annual,
        "mu_annual": mu_annual,
        "n_returns": n,
        "mean_daily": mean_r,
        "std_daily": std_r,
        "min_daily": min_r,
        "max_daily": max_r,
    }


def _try_skew_kurtosis(log_returns: pd.Series) -> dict[str, float]:
    """Compute skew and excess kurtosis if scipy is available."""
    try:
        from scipy import stats as sp_stats  # noqa: PLC0415

        skew = float(sp_stats.skew(log_returns.dropna()))
        excess_kurt = float(sp_stats.kurtosis(log_returns.dropna(), fisher=True))
        return {"skew": skew, "excess_kurtosis": excess_kurt}
    except ImportError:
        return {}


def save_outputs(
    params: dict[str, float],
    log_returns: pd.Series,
    figures_dir: Path,
    tables_dir: Path,
) -> tuple[Path, Path, Path]:
    """Persist GBM parameter table, return diagnostics, and returns histogram.

    Parameters
    ----------
    params:
        Dictionary returned by :func:`estimate_gbm_parameters`.
    log_returns:
        Series of daily log returns.
    figures_dir:
        Directory where the histogram PNG is saved.
    tables_dir:
        Directory where CSV tables are saved.

    Returns
    -------
    Tuple of (gbm_parameters_path, return_diagnostics_path, figure_path).
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # -- GBM parameters table -----------------------------------------------
    param_rows = [
        ("sigma_annual", params["sigma_annual"]),
        ("mu_annual", params["mu_annual"]),
        ("n_returns", params["n_returns"]),
    ]
    param_df = pd.DataFrame(param_rows, columns=["Parameter", "Value"])
    params_path = tables_dir / "gbm_parameters.csv"
    param_df.to_csv(params_path, index=False)

    # -- Return diagnostics table -------------------------------------------
    extra = _try_skew_kurtosis(log_returns)
    diag_rows: list[tuple[str, float]] = [
        ("n_returns", params["n_returns"]),
        ("mean_daily", params["mean_daily"]),
        ("std_daily", params["std_daily"]),
        ("min_daily", params["min_daily"]),
        ("max_daily", params["max_daily"]),
    ]
    if "skew" in extra:
        diag_rows.append(("skew", extra["skew"]))
    if "excess_kurtosis" in extra:
        diag_rows.append(("excess_kurtosis", extra["excess_kurtosis"]))

    diag_df = pd.DataFrame(diag_rows, columns=["Metric", "Value"])
    diag_path = tables_dir / "return_diagnostics.csv"
    diag_df.to_csv(diag_path, index=False)

    # -- Histogram of daily log returns -------------------------------------
    fig_path = figures_dir / "log_returns_histogram.png"
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(log_returns.dropna(), bins=80, edgecolor="none", color="steelblue", alpha=0.8)
    ax.axvline(params["mean_daily"], color="firebrick", linewidth=1.5, linestyle="--",
               label=f"Mean = {params['mean_daily']:.4f}")
    ax.set_xlabel("Daily log return")
    ax.set_ylabel("Frequency")
    ax.set_title(
        "Distribution of daily log returns — Europe Brent Spot Price (2000–2025)"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    return params_path, diag_path, fig_path


def estimate_gbm(cleaned_csv: Path | None = None) -> dict[str, float]:
    """Run the full Stage 2 GBM estimation pipeline.

    Parameters
    ----------
    cleaned_csv:
        Optional override path to the cleaned Stage 1 CSV.

    Returns
    -------
    Dictionary of estimated GBM parameters and diagnostics.
    """
    root = project_root()
    figures_dir = root / "outputs" / "figures"
    tables_dir = root / "outputs" / "tables"

    df = load_cleaned_prices(cleaned_csv)
    log_returns = compute_log_returns(df["Price_USD_per_barrel"])
    params = estimate_gbm_parameters(log_returns)

    params_path, diag_path, fig_path = save_outputs(
        params, log_returns, figures_dir, tables_dir
    )

    # -- Console summary ----------------------------------------------------
    print("=== Stage 2: GBM Parameter Estimation ===")
    print(f"  Return observations : {params['n_returns']}")
    print(f"  Mean daily log-ret  : {params['mean_daily']:.6f}")
    print(f"  Std  daily log-ret  : {params['std_daily']:.6f}")
    print(f"  Min  daily log-ret  : {params['min_daily']:.6f}")
    print(f"  Max  daily log-ret  : {params['max_daily']:.6f}")
    print(f"  sigma_annual        : {params['sigma_annual']:.6f}  ({params['sigma_annual']*100:.2f} %)")
    print(f"  mu_annual           : {params['mu_annual']:.6f}  ({params['mu_annual']*100:.2f} %)")
    print(f"GBM parameters saved  : {params_path}")
    print(f"Return diagnostics    : {diag_path}")
    print(f"Histogram figure      : {fig_path}")

    return params


if __name__ == "__main__":
    estimate_gbm()
