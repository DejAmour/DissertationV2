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

    The 95 % confidence interval uses the normal approximation with
    ``z = 1.96`` (i.e. ``CI = mean ± 1.96 * SE``).

    Parameters
    ----------
    observations : np.ndarray
        Per-path (already discounted) payoff observations, shape ``(n,)``.
        Must contain at least two elements to compute sample variance.
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
    ValueError
        If ``observations`` has fewer than two elements or is not 1-D.
    """
    obs = np.asarray(observations, dtype=np.float64)
    if obs.ndim != 1:
        raise ValueError(
            f"observations must be 1-D, got ndim={obs.ndim}"
        )
    n = len(obs)
    if n < 2:
        raise ValueError(
            f"observations must have at least 2 elements to compute variance, got {n}"
        )

    price = float(obs.mean())
    variance = float(obs.var(ddof=1))
    std_dev = float(np.sqrt(variance))
    std_error = std_dev / np.sqrt(n)
    z = 1.96
    ci_lower = price - z * std_error
    ci_upper = price + z * std_error

    return {
        "price": price,
        "variance": variance,
        "std_dev": std_dev,
        "std_error": std_error,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_paths": n,
        "runtime_s": runtime_s,
    }


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
