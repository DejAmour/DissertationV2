"""
metrics.py
==========
Summary statistics and variance-reduction metrics.

Stage 1 placeholder: interfaces defined; implementations deferred to Stage 3+.
"""

from __future__ import annotations

import numpy as np


def summarise_estimates(
    observations: np.ndarray,
    discount_factor: float,
    runtime_s: float,
) -> dict:
    """
    Compute standard Monte Carlo summary statistics from per-path observations.

    Parameters
    ----------
    observations : np.ndarray
        Per-path (already discounted) payoff observations, shape ``(n,)``.
    discount_factor : float
        Retained for documentation; observations are assumed pre-discounted.
    runtime_s : float
        Wall-clock runtime for the pricing step (seconds).

    Returns
    -------
    dict
        Keys: price, variance, std_dev, std_error, ci_lower, ci_upper,
        n_paths, runtime_s.

    Raises
    ------
    NotImplementedError
        Stage 3 will implement this function.
    """
    raise NotImplementedError("Summary statistics will be implemented in Stage 3.")


def variance_reduction_ratio(
    variance_baseline: float,
    variance_reduced: float,
) -> float:
    """
    Variance-reduction ratio VRR = Var(X) / Var(Y).

    A value greater than 1 indicates variance reduction.

    Parameters
    ----------
    variance_baseline : float
        Observation variance of the baseline estimator (standard MC).
    variance_reduced : float
        Observation variance of the reduced estimator.

    Returns
    -------
    float
        VRR.

    Raises
    ------
    NotImplementedError
        Stage 4 will implement this function.
    """
    raise NotImplementedError(
        "Variance-reduction ratio will be implemented in Stage 4."
    )
