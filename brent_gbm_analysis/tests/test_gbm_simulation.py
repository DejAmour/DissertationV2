"""Stage 3 tests: GBM simulation of Brent crude oil price paths."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "brent_prices_2000_2025_clean.csv"
PARAMS_FILE = PROJECT_ROOT / "outputs" / "tables" / "gbm_parameters.csv"
QUANTILES_FILE = PROJECT_ROOT / "outputs" / "tables" / "simulation_quantiles.csv"
TERMINAL_FILE = PROJECT_ROOT / "outputs" / "tables" / "terminal_distribution_summary.csv"
FAN_CHART_FILE = PROJECT_ROOT / "outputs" / "figures" / "gbm_fan_chart.png"
TERMINAL_HIST_FILE = PROJECT_ROOT / "outputs" / "figures" / "terminal_price_histogram.png"

_PREREQ_MISSING = not PROCESSED_FILE.exists() or not PARAMS_FILE.exists()
_PREREQ_SKIP_REASON = (
    "Run `python src/download_data.py`, `python src/prepare_data.py`, and "
    "`python src/estimate_gbm.py` before Stage 3 integration tests."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_simulation() -> None:
    """Run Stage 3 simulation if output files are missing."""
    if (
        not QUANTILES_FILE.exists()
        or not TERMINAL_FILE.exists()
        or not FAN_CHART_FILE.exists()
        or not TERMINAL_HIST_FILE.exists()
    ):
        from src.simulate_gbm import simulate_gbm  # noqa: PLC0415
        simulate_gbm()


def _load_quantiles() -> pd.DataFrame:
    _run_simulation()
    return pd.read_csv(QUANTILES_FILE)


def _load_terminal_summary() -> pd.DataFrame:
    _run_simulation()
    return pd.read_csv(TERMINAL_FILE)


# ---------------------------------------------------------------------------
# Integration tests (require Stage 1 processed data + Stage 2 parameters)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_output_files_exist() -> None:
    """All four Stage 3 output files should be created after running simulate_gbm."""
    _run_simulation()
    assert QUANTILES_FILE.exists(), f"Missing: {QUANTILES_FILE}"
    assert TERMINAL_FILE.exists(), f"Missing: {TERMINAL_FILE}"
    assert FAN_CHART_FILE.exists(), f"Missing: {FAN_CHART_FILE}"
    assert TERMINAL_HIST_FILE.exists(), f"Missing: {TERMINAL_HIST_FILE}"


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_quantiles_schema() -> None:
    """simulation_quantiles.csv must contain the expected columns."""
    df = _load_quantiles()
    expected_columns = {"day", "q05", "q25", "q50", "q75", "q95"}
    assert expected_columns.issubset(set(df.columns)), (
        f"Missing columns in simulation_quantiles.csv: "
        f"{expected_columns - set(df.columns)}"
    )


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_quantiles_row_count() -> None:
    """simulation_quantiles.csv should have horizon_days + 1 rows (t=0 to t=T)."""
    df = _load_quantiles()
    # Default horizon is 252 trading days -> 253 rows (0..252 inclusive)
    assert len(df) == 253, (
        f"Expected 253 rows in simulation_quantiles.csv, got {len(df)}"
    )


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_quantiles_day_zero_is_s0() -> None:
    """At day 0 all quantiles should equal S0 (initial price)."""
    from src.simulate_gbm import load_initial_price  # noqa: PLC0415

    df = _load_quantiles()
    row0 = df[df["day"] == 0].iloc[0]
    S0 = load_initial_price()
    # All quantile columns should equal S0 (within floating-point tolerance)
    for col in ["q05", "q25", "q50", "q75", "q95"]:
        assert abs(row0[col] - S0) < 1e-6, (
            f"{col} at day 0 ({row0[col]:.6f}) differs from S0 ({S0:.6f})"
        )


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_quantiles_ordering() -> None:
    """Quantile columns must be monotonically non-decreasing at every row."""
    df = _load_quantiles()
    for _, row in df.iterrows():
        assert row["q05"] <= row["q25"] <= row["q50"] <= row["q75"] <= row["q95"], (
            f"Quantile ordering violated at day {row['day']}: {row.tolist()}"
        )


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_quantiles_prices_positive() -> None:
    """All quantile values must be strictly positive (GBM never crosses zero)."""
    df = _load_quantiles()
    for col in ["q05", "q25", "q50", "q75", "q95"]:
        assert (df[col] > 0).all(), (
            f"Non-positive value found in {col} column of simulation_quantiles.csv"
        )


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_terminal_summary_schema() -> None:
    """terminal_distribution_summary.csv must contain required metric rows."""
    df = _load_terminal_summary()
    assert set(df.columns) == {"Metric", "Value"}, (
        f"Unexpected columns in terminal_distribution_summary.csv: {list(df.columns)}"
    )
    required_metrics = {"mean", "std", "min", "max", "q50"}
    found_metrics = set(df["Metric"].tolist())
    assert required_metrics.issubset(found_metrics), (
        f"Missing metrics in terminal_distribution_summary.csv: "
        f"{required_metrics - found_metrics}"
    )


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_terminal_mean_positive() -> None:
    """Terminal mean price must be strictly positive."""
    df = _load_terminal_summary()
    mean_val = float(df.loc[df["Metric"] == "mean", "Value"].iloc[0])
    assert mean_val > 0, f"Terminal mean is not positive: {mean_val}"


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_reproducibility_same_seed() -> None:
    """Two runs with identical parameters and seed must produce identical quantile CSVs."""
    from src.simulate_gbm import simulate_gbm_paths, compute_simulation_quantiles  # noqa: PLC0415

    params = pd.read_csv(PARAMS_FILE)
    param_map = dict(zip(params["Parameter"], params["Value"]))
    mu = float(param_map["mu_annual"])
    sigma = float(param_map["sigma_annual"])

    paths_a = simulate_gbm_paths(S0=75.0, mu=mu, sigma=sigma, random_seed=42)
    paths_b = simulate_gbm_paths(S0=75.0, mu=mu, sigma=sigma, random_seed=42)
    np.testing.assert_array_equal(paths_a, paths_b)

    q_a = compute_simulation_quantiles(paths_a)
    q_b = compute_simulation_quantiles(paths_b)
    pd.testing.assert_frame_equal(q_a, q_b)


@pytest.mark.skipif(_PREREQ_MISSING, reason=_PREREQ_SKIP_REASON)
def test_different_seeds_differ() -> None:
    """Different seeds must produce different simulation results."""
    from src.simulate_gbm import simulate_gbm_paths  # noqa: PLC0415

    params = pd.read_csv(PARAMS_FILE)
    param_map = dict(zip(params["Parameter"], params["Value"]))
    mu = float(param_map["mu_annual"])
    sigma = float(param_map["sigma_annual"])

    paths_a = simulate_gbm_paths(S0=75.0, mu=mu, sigma=sigma, random_seed=1)
    paths_b = simulate_gbm_paths(S0=75.0, mu=mu, sigma=sigma, random_seed=2)
    assert not np.array_equal(paths_a, paths_b), (
        "Paths with different seeds should not be identical."
    )


# ---------------------------------------------------------------------------
# Unit tests (no file I/O, always run)
# ---------------------------------------------------------------------------

def test_simulate_gbm_paths_shape() -> None:
    """simulate_gbm_paths should return array of shape (horizon+1, n_paths)."""
    from src.simulate_gbm import simulate_gbm_paths  # noqa: PLC0415

    paths = simulate_gbm_paths(S0=80.0, mu=0.12, sigma=0.40, horizon_days=10, n_paths=50)
    assert paths.shape == (11, 50), f"Unexpected shape: {paths.shape}"


def test_simulate_gbm_paths_initial_price() -> None:
    """Row 0 of the paths array must equal S0 for all paths."""
    from src.simulate_gbm import simulate_gbm_paths  # noqa: PLC0415

    S0 = 75.0
    paths = simulate_gbm_paths(S0=S0, mu=0.12, sigma=0.40, horizon_days=20, n_paths=100)
    np.testing.assert_allclose(paths[0, :], S0)


def test_simulate_gbm_paths_finite_positive() -> None:
    """All simulated prices must be finite and strictly positive."""
    from src.simulate_gbm import simulate_gbm_paths  # noqa: PLC0415

    paths = simulate_gbm_paths(S0=80.0, mu=0.12, sigma=0.40, horizon_days=252, n_paths=200)
    assert np.all(np.isfinite(paths)), "Non-finite values found in simulated paths."
    assert np.all(paths > 0), "Non-positive prices found in simulated paths."


def test_simulate_gbm_paths_reproducibility() -> None:
    """Same seed must reproduce identical paths."""
    from src.simulate_gbm import simulate_gbm_paths  # noqa: PLC0415

    paths_a = simulate_gbm_paths(S0=80.0, mu=0.12, sigma=0.40, random_seed=99)
    paths_b = simulate_gbm_paths(S0=80.0, mu=0.12, sigma=0.40, random_seed=99)
    np.testing.assert_array_equal(paths_a, paths_b)


def test_simulate_gbm_paths_raises_on_invalid_n_paths() -> None:
    """simulate_gbm_paths should raise ValueError for n_paths <= 0."""
    from src.simulate_gbm import simulate_gbm_paths  # noqa: PLC0415

    with pytest.raises(ValueError, match="n_paths must be positive"):
        simulate_gbm_paths(S0=80.0, mu=0.12, sigma=0.40, n_paths=0)


def test_simulate_gbm_paths_raises_on_invalid_horizon() -> None:
    """simulate_gbm_paths should raise ValueError for horizon_days <= 0."""
    from src.simulate_gbm import simulate_gbm_paths  # noqa: PLC0415

    with pytest.raises(ValueError, match="horizon_days must be positive"):
        simulate_gbm_paths(S0=80.0, mu=0.12, sigma=0.40, horizon_days=0)


def test_simulate_gbm_paths_raises_on_invalid_s0() -> None:
    """simulate_gbm_paths should raise ValueError for non-positive S0."""
    from src.simulate_gbm import simulate_gbm_paths  # noqa: PLC0415

    with pytest.raises(ValueError, match="Initial price S0 must be positive"):
        simulate_gbm_paths(S0=0.0, mu=0.12, sigma=0.40)


def test_compute_simulation_quantiles_schema() -> None:
    """compute_simulation_quantiles should return DataFrame with expected columns."""
    from src.simulate_gbm import simulate_gbm_paths, compute_simulation_quantiles  # noqa: PLC0415

    paths = simulate_gbm_paths(S0=80.0, mu=0.12, sigma=0.40, horizon_days=10, n_paths=50)
    df = compute_simulation_quantiles(paths)
    assert set(df.columns) == {"day", "q05", "q25", "q50", "q75", "q95"}, (
        f"Unexpected columns: {list(df.columns)}"
    )
    assert len(df) == 11  # horizon_days + 1


def test_compute_simulation_quantiles_ordering() -> None:
    """Quantile columns should be monotonically non-decreasing within each row."""
    from src.simulate_gbm import simulate_gbm_paths, compute_simulation_quantiles  # noqa: PLC0415

    paths = simulate_gbm_paths(S0=80.0, mu=0.12, sigma=0.40, horizon_days=50, n_paths=500)
    df = compute_simulation_quantiles(paths)
    for _, row in df.iterrows():
        assert row["q05"] <= row["q25"] <= row["q50"] <= row["q75"] <= row["q95"]


def test_compute_terminal_summary_schema() -> None:
    """compute_terminal_summary should return DataFrame with Metric/Value columns."""
    from src.simulate_gbm import simulate_gbm_paths, compute_terminal_summary  # noqa: PLC0415

    paths = simulate_gbm_paths(S0=80.0, mu=0.12, sigma=0.40, horizon_days=10, n_paths=50)
    df = compute_terminal_summary(paths)
    assert set(df.columns) == {"Metric", "Value"}, (
        f"Unexpected columns: {list(df.columns)}"
    )
    required = {"mean", "std", "min", "max", "q50"}
    found = set(df["Metric"].tolist())
    assert required.issubset(found), f"Missing metrics: {required - found}"


def test_load_gbm_parameters_raises_on_missing_file() -> None:
    """load_gbm_parameters should raise FileNotFoundError for a non-existent path."""
    from src.simulate_gbm import load_gbm_parameters  # noqa: PLC0415

    with pytest.raises(FileNotFoundError):
        load_gbm_parameters(Path("/nonexistent/path/gbm_parameters.csv"))


def test_load_initial_price_raises_on_missing_file() -> None:
    """load_initial_price should raise FileNotFoundError for a non-existent path."""
    from src.simulate_gbm import load_initial_price  # noqa: PLC0415

    with pytest.raises(FileNotFoundError):
        load_initial_price(Path("/nonexistent/path/brent_prices.csv"))
