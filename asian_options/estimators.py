"""
estimators.py
=============
Monte Carlo estimator entry points.

Stage 1 placeholder: function signatures defined; implementations deferred.

Each estimator returns an ``EstimateResult`` named-tuple that carries all
summary statistics needed for the final comparison tables.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from asian_options.config import ModelConfig


class EstimateResult(NamedTuple):
    """
    Summary statistics produced by every Monte Carlo estimator.

    Attributes
    ----------
    price : float
        Estimated option price.
    variance : float
        Sample variance of the per-path observations.
    std_dev : float
        Sample standard deviation.
    std_error : float
        Standard error of the mean estimate (std_dev / sqrt(n_paths)).
    ci_lower : float
        Lower bound of the 95 % confidence interval.
    ci_upper : float
        Upper bound of the 95 % confidence interval.
    n_paths : int
        Number of paths used in the pricing sample.
    runtime_s : float
        Wall-clock runtime in seconds.
    """

    price: float
    variance: float
    std_dev: float
    std_error: float
    ci_lower: float
    ci_upper: float
    n_paths: int
    runtime_s: float


def standard_monte_carlo(cfg: ModelConfig) -> EstimateResult:
    """
    Plain arithmetic Asian call price via Monte Carlo.

    Parameters
    ----------
    cfg : ModelConfig
        Validated experiment configuration.

    Returns
    -------
    EstimateResult
        Pricing summary statistics.

    Raises
    ------
    NotImplementedError
        Stage 3 will implement this estimator.
    """
    raise NotImplementedError("Standard MC will be implemented in Stage 3.")


def antithetic_variates(cfg: ModelConfig) -> EstimateResult:
    """
    Antithetic-variate estimator for the arithmetic Asian call.

    For each shock vector Z, the paired observation is formed from -Z.
    ``cfg.n_paths`` refers to the number of *antithetic pairs*; the total
    number of payoff evaluations is therefore 2 * cfg.n_paths.

    Parameters
    ----------
    cfg : ModelConfig
        Validated experiment configuration.

    Returns
    -------
    EstimateResult
        Pricing summary statistics (variance of pair-average observations).

    Raises
    ------
    NotImplementedError
        Stage 4 will implement this estimator.
    """
    raise NotImplementedError("Antithetic variates will be implemented in Stage 4.")


def geometric_control_variate(
    cfg: ModelConfig,
    n_pilot: int = 1000,
) -> EstimateResult:
    """
    Geometric Asian control-variate estimator.

    Uses the discounted geometric Asian payoff G as a control with its
    analytical expectation E[G].  The coefficient beta is estimated on
    a separate pilot sample of size ``n_pilot`` and frozen before the
    independent pricing sample is evaluated.

    Parameters
    ----------
    cfg : ModelConfig
        Validated experiment configuration.
    n_pilot : int
        Number of paths for the pilot (beta estimation) phase.

    Returns
    -------
    EstimateResult
        Pricing summary statistics.

    Raises
    ------
    NotImplementedError
        Stage 4 will implement this estimator.
    """
    raise NotImplementedError(
        "Geometric control variate will be implemented in Stage 4."
    )
