"""Stage 1 data quality tests for Brent spot price data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_FILE = PROJECT_ROOT / "data" / "processed" / "brent_prices_2000_2025_clean.csv"
SUMMARY_FILE = PROJECT_ROOT / "outputs" / "tables" / "data_validation_summary.csv"
RAW_DIR = PROJECT_ROOT / "data" / "raw"

if not PROCESSED_FILE.exists() or not SUMMARY_FILE.exists():
    pytestmark = pytest.mark.skip(
        reason="Run `python src/download_data.py` and `python src/prepare_data.py` before tests."
    )


def _load_processed() -> pd.DataFrame:
    return pd.read_csv(PROCESSED_FILE, parse_dates=["Date"])


def _load_summary() -> pd.DataFrame:
    return pd.read_csv(SUMMARY_FILE)


def _load_latest_raw() -> pd.DataFrame:
    csv_files = sorted(RAW_DIR.glob("DCOILBRENTEU*.csv"), key=lambda p: p.stat().st_mtime)
    if not csv_files:
        raise FileNotFoundError("No raw CSV found. Run `python src/download_data.py` first.")
    return pd.read_csv(csv_files[-1])


def _expected_missing_count_from_raw() -> int:
    raw = _load_latest_raw().copy()
    raw["DATE"] = pd.to_datetime(raw["DATE"], errors="coerce")
    raw = raw[(raw["DATE"] >= "2000-01-01") & (raw["DATE"] <= "2025-12-31")].copy()
    raw_values = raw["DCOILBRENTEU"].astype("string").str.strip()
    missing_mask = raw["DCOILBRENTEU"].isna() | raw_values.eq(".") | raw_values.eq("")
    return int(missing_mask.sum())


def test_dates_strictly_ascending() -> None:
    """Dates should be strictly increasing in the cleaned dataset."""
    df = _load_processed()
    assert df["Date"].is_monotonic_increasing
    assert (df["Date"].diff().dropna().dt.days > 0).all()


def test_no_duplicated_dates() -> None:
    """Cleaned dataset should not contain duplicated dates."""
    df = _load_processed()
    assert not df["Date"].duplicated().any()


def test_missing_price_count_matches_expected() -> None:
    """Missing-price count in summary should match expected count from raw data."""
    summary = _load_summary()
    missing_count = int(summary.loc[summary["Metric"] == "Missing prices", "Value"].iloc[0])
    expected_missing_count = _expected_missing_count_from_raw()
    assert missing_count == expected_missing_count


def test_all_valid_prices_strictly_positive() -> None:
    """All retained prices should be strictly positive."""
    df = _load_processed()
    assert (df["Price_USD_per_barrel"] > 0).all()
