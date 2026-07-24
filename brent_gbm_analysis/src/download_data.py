"""Download Brent crude oil spot-price data from FRED (EIA-republished series)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import requests

SERIES_NAME = "DCOILBRENTEU"
DOWNLOAD_URL = (
    "https://fred.stlouisfed.org/graph/fredgraph.csv"
    "?id=DCOILBRENTEU&cosd=2000-01-01&coed=2025-12-31"
)
SOURCE_DESCRIPTION = "US EIA Europe Brent Spot Price FOB (republished via FRED)"
DEFAULT_FILENAME = "DCOILBRENTEU_fred.csv"


def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


def _unique_raw_path(raw_dir: Path, base_filename: str) -> Path:
    """Return a non-overwriting path inside the raw directory."""
    candidate = raw_dir / base_filename
    if not candidate.exists():
        return candidate

    stem = Path(base_filename).stem
    suffix = Path(base_filename).suffix
    index = 1
    while True:
        candidate = raw_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def download_brent_data() -> tuple[Path, Path]:
    """Download the CSV and write provenance metadata."""
    root = project_root()
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    response = requests.get(DOWNLOAD_URL, timeout=60)
    response.raise_for_status()

    csv_path = _unique_raw_path(raw_dir, DEFAULT_FILENAME)
    csv_path.write_bytes(response.content)

    retrieval_date = datetime.now(timezone.utc).date().isoformat()
    provenance = {
        "original_source": SOURCE_DESCRIPTION,
        "download_url": DOWNLOAD_URL,
        "series_name": SERIES_NAME,
        "retrieval_date": retrieval_date,
        "raw_csv_filename": csv_path.name,
    }
    provenance_path = raw_dir / "data_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    print(f"Downloaded raw data to: {csv_path}")
    print(f"Saved provenance to: {provenance_path}")
    return csv_path, provenance_path


if __name__ == "__main__":
    download_brent_data()
