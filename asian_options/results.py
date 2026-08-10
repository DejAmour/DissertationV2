"""
results.py
==========
Reproducible result output: CSV serialisation and console summaries.

Stage 5 additions
-----------------
- ``efficiency_gain_vs_mc``: combines estimator variance and runtime into a
  single efficiency metric.  Formula::

      efficiency_gain = (MC_estimator_variance * MC_runtime_seconds)
                      / (method_estimator_variance * method_runtime_seconds)

  A value > 1 means the method is more efficient than plain MC per unit of
  compute time.  This is distinct from the *variance-reduction ratio* which
  is a ratio of observation variances only and ignores runtime.

- ``make_runtime_row``: build a runtime/efficiency result dict suitable for
  CSV serialisation.  Timing scope: wall-clock time from the start of path
  simulation to the end of the pricing step (training excluded for NCV
  unless ``runtime_seconds`` is provided from a scope that includes training).
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path
from typing import Iterable

RESULT_SCHEMA_VERSION = "1.0"

STATISTICAL_OUTPUT_COLUMNS = [
    "comparison_mode",
    "method",
    "price",
    "std_error",
    "observation_variance",
    "estimator_variance",
    "variance_reduction_ratio",
    "pricing_observations",
    "pricing_simulated_paths",
    "pilot_paths",
    "training_paths",
    "total_simulated_paths",
    "runtime_s",
    "notes",
    "seed",
    "replication",
]

RUNTIME_OUTPUT_COLUMNS = [
    "comparison_mode",
    "method",
    "runtime_seconds",
    "time_per_observation",
    "time_per_simulated_path",
    "efficiency_gain_vs_mc",
    "timing_scope",
    "pricing_observations",
    "pricing_simulated_paths",
    "pilot_paths",
    "training_paths",
    "total_simulated_paths",
    "price",
    "std_error",
    "observation_variance",
    "estimator_variance",
    "notes",
    "seed",
    "replication",
]

PUBLICATION_TABLE_COLUMNS = [
    "method",
    "mode",
    "price_estimate",
    "standard_error",
    "observation_variance",
    "estimator_variance",
    "variance_reduction_ratio",
    "runtime_seconds",
    "time_per_observation",
    "time_per_simulated_path",
    "efficiency_gain_vs_mc",
    "pricing_observations",
    "pricing_simulated_paths",
    "pilot_paths",
    "training_paths",
    "total_simulated_paths",
    "timing_scope_note",
]

METRIC_DEFINITIONS = {
    "observation_variance": "sample variance (ddof=1) of per-observation corrected payoffs",
    "estimator_variance": "observation_variance / pricing_observations",
    "variance_reduction_ratio": "MC_observation_variance / method_observation_variance",
    "standard_error": "sqrt(estimator_variance)",
    "time_per_observation": "runtime_seconds / pricing_observations",
    "time_per_simulated_path": "runtime_seconds / pricing_simulated_paths",
    "efficiency_gain_vs_mc": (
        "(MC_estimator_variance * MC_runtime_seconds) / "
        "(method_estimator_variance * method_runtime_seconds)"
    ),
}

METRIC_UNITS = {
    "runtime_seconds": "seconds",
    "time_per_observation": "seconds/observation",
    "time_per_simulated_path": "seconds/path",
    "pricing_observations": "observations",
    "pricing_simulated_paths": "paths",
    "pilot_paths": "paths",
    "training_paths": "paths",
    "total_simulated_paths": "paths",
    "observation_variance": "price^2",
    "estimator_variance": "price^2",
    "variance_reduction_ratio": "dimensionless",
    "efficiency_gain_vs_mc": "dimensionless",
}

# Columns written to summary_stable.csv.  Runtime-sensitive fields
# (runtime_seconds, time_per_observation, time_per_simulated_path,
# efficiency_gain_vs_mc) are intentionally excluded so that
# summary_stable.csv is bit-for-bit identical across runs with the same
# seed and profile.  Runtime metrics are preserved in mode_c_runtime_raw.csv,
# merged_summary.csv, and paper_table.csv for reporting purposes.
STABLE_PUBLICATION_COLUMNS = [
    "method",
    "mode",
    "price_estimate",
    "standard_error",
    "observation_variance",
    "estimator_variance",
    "variance_reduction_ratio",
    "pricing_observations",
    "pricing_simulated_paths",
    "pilot_paths",
    "training_paths",
    "total_simulated_paths",
]

PUBLICATION_TABLE_NOTES = """Metric notes:
- variance_reduction_ratio = MC_observation_variance / method_observation_variance (variance metric only).
- efficiency_gain_vs_mc = (MC_estimator_variance * MC_runtime_seconds) / (method_estimator_variance * method_runtime_seconds) (speed+precision metric).
- AV uses 2 simulated paths per pair observation.
- CV pilot paths and NCV training paths consume total path budget in equal-budget mode.
- Variance reduction is not speedup; do not compare variance_reduction_ratio and efficiency_gain_vs_mc as interchangeable metrics.
- summary_stable.csv excludes runtime fields (runtime_seconds, time_per_observation, time_per_simulated_path, efficiency_gain_vs_mc) for deterministic reproducibility; runtime metrics are reported in mode_c_runtime_raw.csv and merged_summary.csv.
"""


def _normalize_rows(rows: Iterable[dict], columns: Sequence[str]) -> list[dict]:
    normalized = []
    for row in rows:
        normalized.append({col: row.get(col, "NA") for col in columns})
    return normalized


def normalize_statistical_rows(rows: Iterable[dict]) -> list[dict]:
    return _normalize_rows(rows, STATISTICAL_OUTPUT_COLUMNS)


def normalize_runtime_rows(rows: Iterable[dict]) -> list[dict]:
    return _normalize_rows(rows, RUNTIME_OUTPUT_COLUMNS)


def write_stable_csv(rows: Iterable[dict], path: Path, fieldnames: Sequence[str], sort_keys: Sequence[str]) -> None:
    normalized = _normalize_rows(rows, fieldnames)
    normalized.sort(key=lambda row: tuple(str(row.get(k, "")) for k in sort_keys))
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(normalized)


def build_publication_summary_rows(stat_rows: Iterable[dict], runtime_rows: Iterable[dict]) -> list[dict]:
    rows = []
    for row in stat_rows:
        rows.append({
            "method": row.get("method", "NA"),
            "mode": row.get("comparison_mode", "NA"),
            "price_estimate": row.get("price", "NA"),
            "standard_error": row.get("std_error", "NA"),
            "observation_variance": row.get("observation_variance", "NA"),
            "estimator_variance": row.get("estimator_variance", "NA"),
            "variance_reduction_ratio": row.get("variance_reduction_ratio", "NA"),
            "runtime_seconds": "NA",
            "time_per_observation": "NA",
            "time_per_simulated_path": "NA",
            "efficiency_gain_vs_mc": "NA",
            "pricing_observations": row.get("pricing_observations", "NA"),
            "pricing_simulated_paths": row.get("pricing_simulated_paths", "NA"),
            "pilot_paths": row.get("pilot_paths", "NA"),
            "training_paths": row.get("training_paths", "NA"),
            "total_simulated_paths": row.get("total_simulated_paths", "NA"),
            "timing_scope_note": "NA",
        })
    for row in runtime_rows:
        rows.append({
            "method": row.get("method", "NA"),
            "mode": row.get("comparison_mode", "NA"),
            "price_estimate": row.get("price", "NA"),
            "standard_error": row.get("std_error", "NA"),
            "observation_variance": row.get("observation_variance", "NA"),
            "estimator_variance": row.get("estimator_variance", "NA"),
            "variance_reduction_ratio": "NA",
            "runtime_seconds": row.get("runtime_seconds", "NA"),
            "time_per_observation": row.get("time_per_observation", "NA"),
            "time_per_simulated_path": row.get("time_per_simulated_path", "NA"),
            "efficiency_gain_vs_mc": row.get("efficiency_gain_vs_mc", "NA"),
            "pricing_observations": row.get("pricing_observations", "NA"),
            "pricing_simulated_paths": row.get("pricing_simulated_paths", "NA"),
            "pilot_paths": row.get("pilot_paths", "NA"),
            "training_paths": row.get("training_paths", "NA"),
            "total_simulated_paths": row.get("total_simulated_paths", "NA"),
            "timing_scope_note": row.get("timing_scope", "NA"),
        })
    return _normalize_rows(rows, PUBLICATION_TABLE_COLUMNS)


def write_publication_markdown(rows: Iterable[dict], path: Path) -> None:
    rows = _normalize_rows(rows, PUBLICATION_TABLE_COLUMNS)
    header = "| " + " | ".join(PUBLICATION_TABLE_COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] * len(PUBLICATION_TABLE_COLUMNS)) + " |"
    lines = [header, separator]
    for row in rows:
        lines.append("| " + " | ".join(str(row[col]) for col in PUBLICATION_TABLE_COLUMNS) + " |")
    lines.append("")
    lines.append(PUBLICATION_TABLE_NOTES.strip())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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


# ---------------------------------------------------------------------------
# Stage 5: Runtime-efficiency helpers
# ---------------------------------------------------------------------------

def efficiency_gain_vs_mc(
    mc_estimator_variance: float,
    mc_runtime_s: float,
    method_estimator_variance: float,
    method_runtime_s: float,
) -> float:
    """
    Compute the efficiency gain of a variance-reduction method vs plain MC.

    Combines estimator variance *and* wall-clock runtime into a single metric::

        efficiency_gain = (MC_estimator_variance * MC_runtime_seconds)
                        / (method_estimator_variance * method_runtime_seconds)

    A value > 1 means the method achieves the same precision as MC in less
    compute time (or achieves better precision in the same time).

    This is **distinct** from the variance-reduction ratio, which only
    compares observation variances.

    Parameters
    ----------
    mc_estimator_variance : float
        Estimator variance (``observation_variance / pricing_observations``) of
        the plain Monte Carlo baseline.
    mc_runtime_s : float
        Wall-clock runtime of the MC baseline in seconds.
    method_estimator_variance : float
        Estimator variance of the method being compared.
    method_runtime_s : float
        Wall-clock runtime of the method in seconds.

    Returns
    -------
    float
        Efficiency gain (dimensionless, > 1 is better than MC).

    Raises
    ------
    ValueError
        If ``method_estimator_variance`` or ``method_runtime_s`` is zero.
    """
    if method_estimator_variance == 0.0:
        raise ValueError("method_estimator_variance is zero; efficiency gain is undefined.")
    if method_runtime_s == 0.0:
        raise ValueError("method_runtime_s is zero; efficiency gain is undefined.")
    return (mc_estimator_variance * mc_runtime_s) / (method_estimator_variance * method_runtime_s)


def make_runtime_row(
    comparison_mode: str,
    method: str,
    result,
    runtime_seconds: float,
    mc_estimator_variance: float,
    mc_runtime_s: float | None = None,
    timing_scope: str = "pricing only (excludes training/pilot)",
) -> dict:
    """
    Build a runtime/efficiency result dict for a single method.

    Parameters
    ----------
    comparison_mode : str
        Identifier for the comparison mode (e.g. ``"A_equal_obs"``).
    method : str
        Method name (e.g. ``"MC"``, ``"AV"``, ``"CV"``, ``"NCV"``).
    result : EstimateResult or CVEstimateResult
        Pricing result from the estimator.
    runtime_seconds : float
        Measured wall-clock runtime for this method run.
    mc_estimator_variance : float
        Estimator variance of the MC baseline (for efficiency computation).
    mc_runtime_s : float or None
        Runtime of the MC baseline.  If None, ``runtime_seconds`` is used
        (so efficiency_gain_vs_mc = 1.0 for MC itself).
    timing_scope : str
        Human-readable description of what is included in ``runtime_seconds``.

    Returns
    -------
    dict
        Row dict with runtime and efficiency fields.
    """
    if mc_runtime_s is None:
        mc_runtime_s = runtime_seconds

    n_obs = getattr(result, "pricing_observations", getattr(result, "n_paths", 1))
    n_paths = getattr(result, "pricing_simulated_paths", n_obs)
    pilot_paths = getattr(result, "pilot_paths", 0)
    training_paths = getattr(result, "training_paths", 0)
    total_paths = getattr(result, "total_simulated_paths", n_paths + pilot_paths + training_paths)
    obs_var = getattr(result, "observation_variance", getattr(result, "variance", float("nan")))
    est_var = getattr(result, "estimator_variance", result.variance / max(n_obs, 1))

    tpo = runtime_seconds / max(n_obs, 1)
    tpsp = runtime_seconds / max(n_paths, 1)

    try:
        eff = efficiency_gain_vs_mc(mc_estimator_variance, mc_runtime_s, est_var, runtime_seconds)
        eff_str = f"{eff:.6f}"
    except (ValueError, ZeroDivisionError):
        eff_str = "nan"
        eff = float("nan")

    return {
        "comparison_mode": comparison_mode,
        "method": method,
        "runtime_seconds": runtime_seconds,
        "pricing_observations": n_obs,
        "pricing_simulated_paths": n_paths,
        "pilot_paths": pilot_paths,
        "training_paths": training_paths,
        "total_simulated_paths": total_paths,
        "price": getattr(result, "price", float("nan")),
        "std_error": getattr(result, "std_error", float("nan")),
        "observation_variance": obs_var,
        "time_per_observation": tpo,
        "time_per_simulated_path": tpsp,
        "estimator_variance": est_var,
        "efficiency_gain_vs_mc": eff,
        "notes": "",
        "timing_scope": timing_scope,
    }


def print_runtime_table(results: Iterable[dict]) -> None:
    """
    Print a formatted runtime/efficiency comparison table to stdout.

    Parameters
    ----------
    results : Iterable[dict]
        Runtime result records from ``make_runtime_row``.
    """
    rows = list(results)
    if not rows:
        print("(no runtime results to display)")
        return

    header = "{:<6} {:>14} {:>14} {:>14} {:>20}".format(
        "Method", "Runtime(s)", "TimePerObs", "TimePerPath", "EfficiencyGainVsMC"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in rows:
        def _fmt(v, spec):
            try:
                return format(float(v), spec)
            except (TypeError, ValueError):
                return str(v)
        print("{:<6} {:>14} {:>14} {:>14} {:>20}".format(
            row.get("method", ""),
            _fmt(row.get("runtime_seconds", ""), ".6f"),
            _fmt(row.get("time_per_observation", ""), ".8e"),
            _fmt(row.get("time_per_simulated_path", ""), ".8e"),
            _fmt(row.get("efficiency_gain_vs_mc", ""), ".4f"),
        ))
