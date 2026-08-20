#!/usr/bin/env python3
"""Build dissertation_handover.zip with required results, configs, and scripts.

Notes:
- `replication_level_results.csv` is intentionally included in both
  `principal_results` and `frozen_reuse` categories because it contains both
  principal and frozen-transfer outputs for the requested handover.
- Relative source paths are preserved under each category to keep provenance
  visible in the archive layout.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

REPO_ROOT = Path(__file__).resolve().parents[1]
ZIP_NAME = "dissertation_handover.zip"

CATEGORIES: dict[str, list[str]] = {
    "results/principal_results": [
        "experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z/replication_level_results.csv",
        "experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z/cell_summary.csv",
        "experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z/paired_contrasts.csv",
    ],
    "results/equal_observation_and_budget": [
        "asian_options_equal_obs_comparison.csv",
        "asian_options_equal_budget_comparison.csv",
    ],
    "results/frozen_reuse": [
        "experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z/replication_level_results.csv",
        "asian_options/frozen_transfer.py",
    ],
    "results/runtime_and_break_even": [
        "asian_options_runtime_comparison.csv",
        "experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z/runtime_results.csv",
        "experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z/checkpoint_curve_results.csv",
    ],
    "results/reference_price_calculations": [
        "experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z/dissertation_summary.md",
    ],
    "config_and_scripts/config_files": [
        "experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z/config.json",
        "experiment_runs/stage8_sensitivity_2x2_dissertation_20260818T101252Z/environment.json",
        "asian_options/config.py",
        "asian_options/results.py",
    ],
    "config_and_scripts/execution_scripts": [
        "scripts/run_stage8.py",
        "scripts/run_experiments.py",
        "scripts/run_stage8_sensitivity_2x2.py",
    ],
}


def _validate_files(repo_root: Path) -> list[tuple[str, Path]]:
    """Return (archive_path, source_path) pairs after verifying required files."""
    mapped_files: list[tuple[str, Path]] = []
    missing: list[Path] = []

    for category, rel_paths in CATEGORIES.items():
        for rel_path in rel_paths:
            source_path = repo_root / rel_path
            if not source_path.exists():
                missing.append(source_path)
                continue
            archive_path = Path(category) / rel_path
            mapped_files.append((str(archive_path), source_path))

    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required files:\n{missing_text}")

    return mapped_files


def build_handover_zip(repo_root: Path, zip_path: Path) -> None:
    files_to_add = _validate_files(repo_root)

    if zip_path.exists():
        zip_path.unlink()

    with ZipFile(zip_path, mode="w", compression=ZIP_DEFLATED) as zf:
        for archive_path, source_path in files_to_add:
            zf.write(source_path, archive_path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create dissertation handover zip with principal results and scripts."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help=f"Repository root (default: {REPO_ROOT})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / ZIP_NAME,
        help=f"Output zip path (default: {REPO_ROOT / ZIP_NAME})",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output = args.output.resolve()
    build_handover_zip(repo_root, output)
    print(f"Created handover zip: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
