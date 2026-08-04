"""Stage 3: Simulate GBM price paths for Brent crude oil.

Uses parameters estimated in Stage 2 to simulate Geometric Brownian Motion
price paths using exact discretization:

    S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_t)

where Z_t ~ N(0, 1) i.i.d.

Outputs
-------
tables/simulation_quantiles.csv        – day, q05, q25, q50, q75, q95
tables/terminal_distribution_summary.csv – mean/std/min/max + quantiles of
                                           terminal prices
figures/gbm_fan_chart.png              – median + percentile bands over horizon
figures/terminal_price_histogram.png   – histogram of terminal simulated prices
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for reproducibility
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

TRADING_DAYS_PER_YEAR = 252

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


def load_initial_price(path: Path | None = None) -> float:
    """Load the latest observed close price (S0) from the Stage 1 cleaned CSV.

    Parameters
    ----------
    path:
        Override path to the cleaned price CSV.  Defaults to the canonical
        Stage 1 output location.

    Returns
    -------
    Latest ``Price_USD_per_barrel`` value as a float.
    """
    csv_path = path if path is not None else PROCESSED_FILE
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cleaned price file not found: {csv_path}\n"
            "Run `python src/download_data.py` then `python src/prepare_data.py` first."
        )
    df = pd.read_csv(csv_path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    return float(df["Price_USD_per_barrel"].iloc[-1])


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------

def simulate_gbm_paths(
    S0: float,
    mu: float,
    sigma: float,
    horizon_days: int = TRADING_DAYS_PER_YEAR,
    n_paths: int = 5000,
    dt: float = 1.0 / TRADING_DAYS_PER_YEAR,
    random_seed: int = 42,
) -> np.ndarray:
    """Simulate GBM price paths using exact discretization.

    Parameters
    ----------
    S0:
        Initial asset price.
    mu:
        Annualized GBM drift.
    sigma:
        Annualized GBM volatility.
    horizon_days:
        Number of time steps to simulate (default: 252 trading days).
    n_paths:
        Number of independent simulation paths (default: 5000).
    dt:
        Length of each time step in years (default: 1/252).
    random_seed:
        NumPy random seed for reproducibility (default: 42).

    Returns
    -------
    np.ndarray of shape ``(horizon_days + 1, n_paths)`` where row 0 is ``S0``
    for all paths and each subsequent row is one time step forward.
    """
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}.")
    if horizon_days <= 0:
        raise ValueError(f"horizon_days must be positive, got {horizon_days}.")
    if S0 <= 0:
        raise ValueError(f"Initial price S0 must be positive, got {S0}.")

    rng = np.random.default_rng(random_seed)

    # Shape: (horizon_days, n_paths)
    Z = rng.standard_normal((horizon_days, n_paths))

    # Exact discretization increments
    drift_term = (mu - 0.5 * sigma ** 2) * dt
    diffusion_term = sigma * np.sqrt(dt)

    log_increments = drift_term + diffusion_term * Z   # (horizon_days, n_paths)
    log_paths = np.cumsum(log_increments, axis=0)       # cumulative log returns

    # Prepend zeros for t=0 (S0 for all paths)
    log_paths = np.vstack([np.zeros((1, n_paths)), log_paths])  # (horizon_days+1, n_paths)
    paths = S0 * np.exp(log_paths)

    return paths


# ---------------------------------------------------------------------------
# Summary / output helpers
# ---------------------------------------------------------------------------

def compute_simulation_quantiles(paths: np.ndarray) -> pd.DataFrame:
    """Compute price quantiles across paths at each time step.

    Parameters
    ----------
    paths:
        Array of shape ``(T+1, n_paths)`` returned by
        :func:`simulate_gbm_paths`.

    Returns
    -------
    DataFrame with columns: ``day``, ``q05``, ``q25``, ``q50``, ``q75``, ``q95``.
    """
    quantile_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
    quantile_matrix = np.quantile(paths, quantile_levels, axis=1).T  # (T+1, 5)
    df = pd.DataFrame(
        quantile_matrix,
        columns=["q05", "q25", "q50", "q75", "q95"],
    )
    df.insert(0, "day", np.arange(len(paths)))
    return df


def compute_terminal_summary(paths: np.ndarray) -> pd.DataFrame:
    """Compute summary statistics of the terminal (final-day) price distribution.

    Parameters
    ----------
    paths:
        Array of shape ``(T+1, n_paths)`` returned by
        :func:`simulate_gbm_paths`.

    Returns
    -------
    DataFrame with columns ``Metric`` and ``Value``.
    """
    terminal = paths[-1, :]
    quantile_levels = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    rows: list[tuple[str, float]] = [
        ("mean", float(np.mean(terminal))),
        ("std", float(np.std(terminal, ddof=1))),
        ("min", float(np.min(terminal))),
        ("max", float(np.max(terminal))),
    ]
    for q in quantile_levels:
        rows.append((f"q{int(q * 100):02d}", float(np.quantile(terminal, q))))
    return pd.DataFrame(rows, columns=["Metric", "Value"])


# ---------------------------------------------------------------------------
# Figure helpers
# ---------------------------------------------------------------------------

def plot_fan_chart(
    quantiles_df: pd.DataFrame,
    S0: float,
    figures_dir: Path,
) -> Path:
    """Save a GBM fan chart (median + percentile bands) to disk.

    Parameters
    ----------
    quantiles_df:
        DataFrame returned by :func:`compute_simulation_quantiles`.
    S0:
        Initial price (used in the chart title).
    figures_dir:
        Directory where the PNG is written.

    Returns
    -------
    Path to the saved figure.
    """
    fig_path = figures_dir / "gbm_fan_chart.png"
    days = quantiles_df["day"].values

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(
        days,
        quantiles_df["q05"],
        quantiles_df["q95"],
        alpha=0.20,
        color="steelblue",
        label="5th–95th percentile",
    )
    ax.fill_between(
        days,
        quantiles_df["q25"],
        quantiles_df["q75"],
        alpha=0.35,
        color="steelblue",
        label="25th–75th percentile",
    )
    ax.plot(
        days,
        quantiles_df["q50"],
        color="steelblue",
        linewidth=2,
        label="Median (50th percentile)",
    )
    ax.axhline(S0, color="black", linewidth=1, linestyle="--", label=f"S0 = {S0:.2f}")

    ax.set_xlabel("Trading days ahead")
    ax.set_ylabel("Brent crude price (USD/barrel)")
    ax.set_title(
        "GBM Simulated Price Fan Chart — Europe Brent Spot Price"
    )
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    return fig_path


def plot_terminal_histogram(
    paths: np.ndarray,
    S0: float,
    figures_dir: Path,
) -> Path:
    """Save a histogram of the terminal (final-day) simulated price distribution.

    Parameters
    ----------
    paths:
        Array of shape ``(T+1, n_paths)``.
    S0:
        Initial price (shown as a vertical reference line).
    figures_dir:
        Directory where the PNG is written.

    Returns
    -------
    Path to the saved figure.
    """
    fig_path = figures_dir / "terminal_price_histogram.png"
    terminal = paths[-1, :]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(terminal, bins=80, edgecolor="none", color="steelblue", alpha=0.8)
    ax.axvline(
        float(np.median(terminal)),
        color="firebrick",
        linewidth=1.5,
        linestyle="--",
        label=f"Median = {np.median(terminal):.2f}",
    )
    ax.axvline(
        S0,
        color="black",
        linewidth=1.5,
        linestyle=":",
        label=f"S0 = {S0:.2f}",
    )
    ax.set_xlabel("Terminal price (USD/barrel)")
    ax.set_ylabel("Frequency")
    ax.set_title(
        "Distribution of Simulated Terminal Prices — Europe Brent Spot Price"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)

    return fig_path


# ---------------------------------------------------------------------------
# Top-level pipeline function
# ---------------------------------------------------------------------------

def simulate_gbm(
    cleaned_csv: Path | None = None,
    params_csv: Path | None = None,
    horizon_days: int = TRADING_DAYS_PER_YEAR,
    n_paths: int = 5000,
    dt: float = 1.0 / TRADING_DAYS_PER_YEAR,
    random_seed: int = 42,
) -> dict:
    """Run the full Stage 3 GBM simulation pipeline.

    Parameters
    ----------
    cleaned_csv:
        Optional override path to the Stage 1 cleaned price CSV.
    params_csv:
        Optional override path to the Stage 2 GBM parameters CSV.
    horizon_days:
        Number of trading days to simulate (default: 252).
    n_paths:
        Number of independent simulation paths (default: 5000).
    dt:
        Length of each time step in years (default: 1/252).
    random_seed:
        NumPy random seed for reproducibility (default: 42).

    Returns
    -------
    Dictionary with keys: ``S0``, ``mu``, ``sigma``, ``paths``,
    ``quantiles_df``, ``terminal_summary_df``,
    ``fan_chart_path``, ``histogram_path``,
    ``quantiles_csv_path``, ``terminal_csv_path``.
    """
    root = project_root()
    figures_dir = root / "outputs" / "figures"
    tables_dir = root / "outputs" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # -- Load inputs ----------------------------------------------------------
    gbm_params = load_gbm_parameters(params_csv)
    mu = gbm_params["mu_annual"]
    sigma = gbm_params["sigma_annual"]
    S0 = load_initial_price(cleaned_csv)

    # -- Simulate -------------------------------------------------------------
    paths = simulate_gbm_paths(
        S0=S0,
        mu=mu,
        sigma=sigma,
        horizon_days=horizon_days,
        n_paths=n_paths,
        dt=dt,
        random_seed=random_seed,
    )

    # -- Compute summaries ----------------------------------------------------
    quantiles_df = compute_simulation_quantiles(paths)
    terminal_df = compute_terminal_summary(paths)

    # -- Save tables ----------------------------------------------------------
    quantiles_csv_path = tables_dir / "simulation_quantiles.csv"
    terminal_csv_path = tables_dir / "terminal_distribution_summary.csv"
    quantiles_df.to_csv(quantiles_csv_path, index=False)
    terminal_df.to_csv(terminal_csv_path, index=False)

    # -- Save figures ---------------------------------------------------------
    fan_chart_path = plot_fan_chart(quantiles_df, S0, figures_dir)
    histogram_path = plot_terminal_histogram(paths, S0, figures_dir)

    # -- Console summary ------------------------------------------------------
    print("=== Stage 3: GBM Simulation ===")
    print(f"  S0 (initial price)  : {S0:.4f} USD/barrel")
    print(f"  mu_annual           : {mu:.6f}  ({mu * 100:.2f} %)")
    print(f"  sigma_annual        : {sigma:.6f}  ({sigma * 100:.2f} %)")
    print(f"  Horizon             : {horizon_days} trading days")
    print(f"  Paths               : {n_paths}")
    print(f"  Random seed         : {random_seed}")
    print(f"Simulation quantiles  : {quantiles_csv_path}")
    print(f"Terminal distribution : {terminal_csv_path}")
    print(f"Fan chart figure      : {fan_chart_path}")
    print(f"Terminal histogram    : {histogram_path}")

    return {
        "S0": S0,
        "mu": mu,
        "sigma": sigma,
        "paths": paths,
        "quantiles_df": quantiles_df,
        "terminal_summary_df": terminal_df,
        "fan_chart_path": fan_chart_path,
        "histogram_path": histogram_path,
        "quantiles_csv_path": quantiles_csv_path,
        "terminal_csv_path": terminal_csv_path,
    }


if __name__ == "__main__":
    simulate_gbm()
