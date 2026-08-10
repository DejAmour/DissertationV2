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
        "runtime_seconds": f"{runtime_seconds:.6f}",
        "pricing_observations": n_obs,
        "pricing_simulated_paths": n_paths,
        "time_per_observation": f"{tpo:.8e}",
        "time_per_simulated_path": f"{tpsp:.8e}",
        "estimator_variance": f"{est_var:.8e}",
        "efficiency_gain_vs_mc": eff_str,
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
        print("{:<6} {:>14} {:>14} {:>14} {:>20}".format(
            row.get("method", ""),
            row.get("runtime_seconds", ""),
            row.get("time_per_observation", ""),
            row.get("time_per_simulated_path", ""),
            row.get("efficiency_gain_vs_mc", ""),
        ))
