"""
results.py
==========
Reproducible result output: CSV serialisation and console summaries.

Stage 1 placeholder: interfaces defined; implementations deferred to Stage 10.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from asian_options.estimators import EstimateResult


def save_results_csv(
    results: Iterable[dict],
    path: Path,
) -> None:
    """
    Append a sequence of result dictionaries to a CSV file.

    Each dictionary must contain at least the fields defined in
    ``EstimateResult`` plus an ``estimator`` label and the experiment
    configuration fields.

    Parameters
    ----------
    results : Iterable[dict]
        Result records to write.
    path : Path
        Destination CSV file.  Parent directories must exist.

    Raises
    ------
    NotImplementedError
        Stage 10 will implement this function.
    """
    raise NotImplementedError("CSV output will be implemented in Stage 10.")


def print_comparison_table(results: Iterable[dict]) -> None:
    """
    Print a formatted comparison table to stdout.

    Parameters
    ----------
    results : Iterable[dict]
        Result records (same format as ``save_results_csv``).

    Raises
    ------
    NotImplementedError
        Stage 10 will implement this function.
    """
    raise NotImplementedError(
        "Comparison table output will be implemented in Stage 10."
    )
