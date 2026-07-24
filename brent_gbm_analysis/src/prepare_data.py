"""Prepare and validate Brent crude oil spot-price data for Stage 1."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

START_DATE = "2000-01-01"
END_DATE = "2025-12-31"


@dataclass
class ValidationSummary:
    """Container for Stage 1 validation outputs."""

    source_url: str
    retrieval_date: str
    earliest_observation: str
    latest_observation: str
    total_rows_in_window: int
    rows_with_valid_prices: int
    missing_prices: int
    non_numeric_prices: int
    duplicated_dates: int
    non_positive_prices: int
    min_price: float
    max_price: float
    gaps_over_4_days: str


def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def _find_latest_raw_csv(raw_dir: Path) -> Path:
    """Find the most recently modified raw CSV file."""
    csv_files = sorted(raw_dir.glob("DCOILBRENTEU*.csv"), key=lambda p: p.stat().st_mtime)
    if not csv_files:
        raise FileNotFoundError("No raw CSV found. Run src/download_data.py first.")
    return csv_files[-1]


def _load_provenance(raw_dir: Path) -> dict[str, str]:
    """Load provenance metadata if present."""
    provenance_path = raw_dir / "data_provenance.json"
    if not provenance_path.exists():
        return {"download_url": "", "retrieval_date": ""}
    return json.loads(provenance_path.read_text(encoding="utf-8"))


def _calculate_gap_report(cleaned: pd.DataFrame) -> str:
    """Report gaps greater than 4 calendar days between valid consecutive observations."""
    date_diff_days = cleaned["Date"].diff().dt.days
    gap_indices = date_diff_days[date_diff_days > 4].index.tolist()
    if not gap_indices:
        return "None"

    gaps: list[str] = []
    for idx in gap_indices:
        prev_date = cleaned.loc[idx - 1, "Date"].date().isoformat()
        curr_date = cleaned.loc[idx, "Date"].date().isoformat()
        gap_days = int(date_diff_days.loc[idx])
        gaps.append(f"{prev_date} to {curr_date} ({gap_days} days)")
    return " | ".join(gaps)


def prepare_data() -> tuple[Path, Path, Path]:
    """Clean the Brent series and generate Stage 1 validation outputs."""
    root = project_root()
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    figures_dir = root / "outputs" / "figures"
    tables_dir = root / "outputs" / "tables"
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    raw_csv_path = _find_latest_raw_csv(raw_dir)
    provenance = _load_provenance(raw_dir)
    source_url = provenance.get("download_url", "")
    retrieval_date = provenance.get("retrieval_date", "")

    raw_df = pd.read_csv(raw_csv_path)
    raw_df.columns = raw_df.columns.str.strip().str.upper()
    if "OBSERVATION_DATE" in raw_df.columns and "DATE" not in raw_df.columns:
        raw_df = raw_df.rename(columns={"OBSERVATION_DATE": "DATE"})
    required = {"DATE", "DCOILBRENTEU"}
    if not required.issubset(raw_df.columns):
        missing = required - set(raw_df.columns)
        raise ValueError(
            f"Raw CSV missing required columns: {', '.join(sorted(missing))}. "
            f"Discovered columns: {list(raw_df.columns)}"
        )

    df = raw_df.copy()
    df["Date"] = pd.to_datetime(df["DATE"], errors="coerce")
    df = df[df["Date"].notna()].copy()
    df = df[(df["Date"] >= START_DATE) & (df["Date"] <= END_DATE)].copy()
    df = df.sort_values("Date").reset_index(drop=True)

    duplicated_dates = int(df.duplicated(subset=["Date"]).sum())
    df = df.drop_duplicates(subset=["Date"], keep="first").copy()

    raw_price = df["DCOILBRENTEU"]
    raw_price_str = raw_price.astype("string").str.strip()
    numeric_price = pd.to_numeric(raw_price, errors="coerce")
    missing_mask = raw_price.isna() | raw_price_str.eq(".") | raw_price_str.eq("")
    non_numeric_mask = numeric_price.isna() & ~missing_mask
    valid_price_mask = numeric_price.notna()

    cleaned = pd.DataFrame(
        {
            "Date": df["Date"],
            "Price_USD_per_barrel": numeric_price,
        }
    )
    total_rows = int(len(cleaned))
    cleaned = cleaned[valid_price_mask].copy()
    cleaned = cleaned.sort_values("Date").reset_index(drop=True)

    non_positive_prices = int((cleaned["Price_USD_per_barrel"] <= 0).sum())
    min_price = float(cleaned["Price_USD_per_barrel"].min())
    max_price = float(cleaned["Price_USD_per_barrel"].max())
    earliest = cleaned["Date"].min().date().isoformat()
    latest = cleaned["Date"].max().date().isoformat()
    gap_report = _calculate_gap_report(cleaned)

    processed_csv_path = processed_dir / "brent_prices_2000_2025_clean.csv"
    cleaned.to_csv(processed_csv_path, index=False)

    figure_path = figures_dir / "brent_price_series.png"
    plt.figure(figsize=(12, 6))
    plt.plot(cleaned["Date"], cleaned["Price_USD_per_barrel"], linewidth=1.2)
    plt.xlabel("Date")
    plt.ylabel("Price (USD per barrel)")
    plt.title("Europe Brent Spot Price FOB (USD/barrel), 2000-01-01 to 2025-12-31")
    plt.tight_layout()
    plt.savefig(figure_path, dpi=300)
    plt.close()

    summary = ValidationSummary(
        source_url=source_url,
        retrieval_date=retrieval_date,
        earliest_observation=earliest,
        latest_observation=latest,
        total_rows_in_window=total_rows,
        rows_with_valid_prices=int(len(cleaned)),
        missing_prices=int(missing_mask.sum()),
        non_numeric_prices=int(non_numeric_mask.sum()),
        duplicated_dates=duplicated_dates,
        non_positive_prices=non_positive_prices,
        min_price=min_price,
        max_price=max_price,
        gaps_over_4_days=gap_report,
    )

    summary_rows = [
        ("Source URL", summary.source_url),
        ("Retrieval date", summary.retrieval_date),
        ("Earliest observation", summary.earliest_observation),
        ("Latest observation", summary.latest_observation),
        ("Total rows in sample window", summary.total_rows_in_window),
        ("Rows with valid prices", summary.rows_with_valid_prices),
        ("Missing prices", summary.missing_prices),
        ("Non-numeric prices", summary.non_numeric_prices),
        ("Duplicated dates", summary.duplicated_dates),
        ("Non-positive prices", summary.non_positive_prices),
        ("Minimum price", summary.min_price),
        ("Maximum price", summary.max_price),
        ("Gaps over 4 days", summary.gaps_over_4_days),
    ]
    summary_df = pd.DataFrame(summary_rows, columns=["Metric", "Value"])
    summary_csv_path = tables_dir / "data_validation_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)

    for metric, value in summary_rows:
        print(f"{metric}: {value}")

    print(f"Cleaned data saved to: {processed_csv_path}")
    print(f"Validation summary saved to: {summary_csv_path}")
    print(f"Figure saved to: {figure_path}")

    return processed_csv_path, summary_csv_path, figure_path


if __name__ == "__main__":
    prepare_data()
