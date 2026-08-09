"""
results.py
==========
Reproducible result output: CSV serialisation and console summaries.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def save_results_csv(
    results: Iterable[dict],
    path: Path,
    fieldnames: list[str] | None = None,
) -> None:
    """
    Write a sequence of result dictionaries to a CSV file.

    If the file does not exist it is created with a header row.  If it exists
    it is overwritten.  Pass ``fieldnames`` explicitly when the result dicts
    may have inconsistent key ordering across Python versions.

    Parameters
    ----------
    results : Iterable[dict]
        Result records to write.
    path : Path
        Destination CSV file.  Parent directories must exist.
    fieldnames : list[str] or None
        Column order.  If None, the keys of the first record are used.
    """
    rows = list(results)
    if not rows:
        return

    path = Path(path)
    if fieldnames is None:
        fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_comparison_table(results: Iterable[dict]) -> None:
    """
    Print a formatted comparison table to stdout.

    Displays budget-accounting fields and variance semantics introduced in
    Stage 4.  The ``speed_ratio_vs_mc`` column is no longer shown; the
    variance-reduction ratio (``variance_reduction_ratio``) is displayed
    instead.

    Parameters
    ----------
    results : Iterable[dict]
        Result records with at least the keys: ``method``, ``price``,
        ``observation_variance``, ``variance_reduction_ratio``,
        ``pricing_observations``, ``total_simulated_paths``.
        Falls back gracefully when optional keys are absent.
    """
    rows = list(results)
    if not rows:
        print("(no results to display)")
        return

    header = "{:<6} {:>10} {:>16} {:>10} {:>14} {:>20}".format(
        "Method", "Price", "ObsVariance", "PricingObs", "TotalPaths", "VarReductionRatio"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        print("{:<6} {:>10} {:>16} {:>10} {:>14} {:>20}".format(
            row.get("method", ""),
            row.get("price", ""),
            row.get("observation_variance", row.get("variance", "")),
            row.get("pricing_observations", ""),
            row.get("total_simulated_paths", ""),
            row.get("variance_reduction_ratio", ""),
        ))
