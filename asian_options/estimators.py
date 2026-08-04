"""
estimators.py
=============
Monte Carlo estimator entry points.

Stage 1 placeholder: function signatures defined; implementations deferred.

Each estimator returns an ``EstimateResult`` named-tuple that carries all
summary statistics needed for the final comparison tables.

Stage 4 adds:
- ``antithetic_variates``: antithetic-variate estimator for arithmetic Asian call.
- ``geometric_control_variate``: control-variate estimator using geometric Asian
  payoff as control with closed-form expectation from Stage 3.
- ``CVEstimateResult``: extended result type carrying CV metadata (beta_hat,
  corr_estimate).
"""

from __future__ import annotations

from typing import NamedTuple, Optional

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


class CVEstimateResult(NamedTuple):
    """
    Extended summary statistics for the control-variate estimator.

    Carries all fields of ``EstimateResult`` plus CV-specific metadata.

    Attributes
    ----------
    price : float
        CV-corrected estimated option price.
    variance : float
        Sample variance of the CV-corrected per-path observations.
    std_dev : float
        Sample standard deviation of corrected observations.
    std_error : float
        Standard error of the CV mean estimate (std_dev / sqrt(n_paths)).
    ci_lower : float
        Lower bound of the 95 % confidence interval.
    ci_upper : float
        Upper bound of the 95 % confidence interval.
    n_paths : int
        Number of paths used in the pricing (main) sample.
    runtime_s : float
        Wall-clock runtime in seconds.
    beta_hat : float
        Estimated (frozen) control-variate coefficient from the pilot sample.
    corr_estimate : float
        Sample correlation between arithmetic and geometric payoffs on the
        pilot sample.  Close to 1 indicates strong variance reduction potential.
    """

    price: float
    variance: float
    std_dev: float
    std_error: float
    ci_lower: float
    ci_upper: float
    n_paths: int
    runtime_s: float
    beta_hat: float
    corr_estimate: float


def standard_monte_carlo(cfg: ModelConfig) -> EstimateResult:
    """
    Plain arithmetic Asian call price via Monte Carlo.

    Simulates ``cfg.n_paths`` GBM paths using ``cfg.seed`` for reproducibility,
    computes the discounted arithmetic Asian call payoff for each path, and
    returns the standard Monte Carlo estimator summary.

    Parameters
    ----------
    cfg : ModelConfig
        Validated experiment configuration.

    Returns
    -------
    EstimateResult
        Pricing summary statistics.
    """
    import time
    from asian_options.simulate_gbm import simulate_paths
    from asian_options.payoffs import arithmetic_asian_call_payoff
    from asian_options.metrics import summarise_estimates

    t0 = time.perf_counter()
    paths = simulate_paths(cfg)
    payoffs = arithmetic_asian_call_payoff(paths, cfg)
    runtime_s = time.perf_counter() - t0

    stats = summarise_estimates(payoffs, cfg.discount_factor, runtime_s)
    return EstimateResult(
        price=stats["price"],
        variance=stats["variance"],
        std_dev=stats["std_dev"],
        std_error=stats["std_error"],
        ci_lower=stats["ci_lower"],
        ci_upper=stats["ci_upper"],
        n_paths=stats["n_paths"],
        runtime_s=stats["runtime_s"],
    )


def antithetic_variates(cfg: ModelConfig) -> EstimateResult:
    """
    Antithetic-variate estimator for the arithmetic Asian call.

    For each shock vector Z, the paired observation is formed from -Z.
    ``cfg.n_paths`` refers to the number of *antithetic pairs*; the total
    number of payoff evaluations is therefore ``2 * cfg.n_paths``.

    The estimator averages each pair:

    .. math::

        Y_i = \\frac{f(Z_i) + f(-Z_i)}{2}

    and returns summary statistics computed from the ``n_paths`` pair-averages.

    Parameters
    ----------
    cfg : ModelConfig
        Validated experiment configuration.

    Returns
    -------
    EstimateResult
        Pricing summary statistics (variance of pair-average observations).
    """
    import time
    from asian_options.simulate_gbm import simulate_paths
    from asian_options.payoffs import arithmetic_asian_call_payoff
    from asian_options.metrics import summarise_estimates

    t0 = time.perf_counter()
    rng = np.random.default_rng(cfg.seed)
    n, m = cfg.n_paths, cfg.m
    Z = rng.standard_normal((n, m))

    # Build positive and negative shock paths from cfg
    from dataclasses import replace
    import dataclasses

    # Simulate positive paths
    paths_pos = simulate_paths(cfg, shocks=Z)
    # Simulate negative (antithetic) paths
    paths_neg = simulate_paths(cfg, shocks=-Z)

    payoffs_pos = arithmetic_asian_call_payoff(paths_pos, cfg)
    payoffs_neg = arithmetic_asian_call_payoff(paths_neg, cfg)
    pair_averages = 0.5 * (payoffs_pos + payoffs_neg)

    runtime_s = time.perf_counter() - t0
    stats = summarise_estimates(pair_averages, cfg.discount_factor, runtime_s)
    return EstimateResult(
        price=stats["price"],
        variance=stats["variance"],
        std_dev=stats["std_dev"],
        std_error=stats["std_error"],
        ci_lower=stats["ci_lower"],
        ci_upper=stats["ci_upper"],
        n_paths=stats["n_paths"],
        runtime_s=stats["runtime_s"],
    )


def geometric_control_variate(
    cfg: ModelConfig,
    n_pilot: int = 1000,
) -> CVEstimateResult:
    """
    Geometric Asian control-variate estimator for the arithmetic Asian call.

    Uses the discounted geometric Asian payoff G as a control with its
    analytical expectation E[G] (from ``geometric_asian_call_price`` in
    Stage 3).  The optimal coefficient beta is estimated on a separate pilot
    sample of ``n_pilot`` paths and frozen before the independent pricing
    sample is evaluated.

    Algorithm
    ---------
    1. Draw a pilot sample of ``n_pilot`` paths (seed ``cfg.seed``).
    2. Compute arithmetic payoffs X_pilot and geometric payoffs G_pilot.
    3. Estimate beta = Cov(X_pilot, G_pilot) / Var(G_pilot).
       If Var(G_pilot) == 0 (degenerate), beta = 0 (fallback to plain MC).
    4. Draw an independent main sample of ``cfg.n_paths`` paths
       (seed ``cfg.seed + 1`` to ensure independence).
    5. Compute corrected observations Y = X - beta*(G - E[G]).
    6. Return summary statistics of Y plus CV metadata.

    Notes
    -----
    * E[G] is the *discounted* analytical price from ``geometric_asian_call_price``,
      consistent with the discounted payoff convention used throughout.
    * corr_estimate is the sample Pearson correlation between X_pilot and
      G_pilot; values close to 1 indicate strong variance reduction potential.

    Parameters
    ----------
    cfg : ModelConfig
        Validated experiment configuration.
    n_pilot : int
        Number of paths for the pilot (beta estimation) phase.
        Must be >= 2.  Defaults to 1000.

    Returns
    -------
    CVEstimateResult
        Pricing summary statistics plus beta_hat and corr_estimate.

    Raises
    ------
    ValueError
        If ``n_pilot`` < 2.
    """
    import time
    from asian_options.simulate_gbm import simulate_paths
    from asian_options.payoffs import (
        arithmetic_asian_call_payoff,
        geometric_asian_call_payoff,
    )
    from asian_options.analytical import geometric_asian_call_price
    from asian_options.variance_reduction import estimate_beta, apply_control_variate
    from asian_options.metrics import summarise_estimates
    import dataclasses

    if n_pilot < 2:
        raise ValueError(f"n_pilot must be >= 2, got {n_pilot}")

    t0 = time.perf_counter()

    # --- Analytical expectation of the geometric Asian payoff (discounted) ---
    eg = geometric_asian_call_price(cfg)

    # --- Pilot sample: estimate beta ---
    pilot_cfg = dataclasses.replace(cfg, n_paths=n_pilot, seed=cfg.seed)
    pilot_paths = simulate_paths(pilot_cfg)
    x_pilot = arithmetic_asian_call_payoff(pilot_paths, pilot_cfg)
    g_pilot = geometric_asian_call_payoff(pilot_paths, pilot_cfg)

    beta_hat = estimate_beta(x_pilot, g_pilot)

    # Compute pilot correlation for metadata
    if np.std(g_pilot, ddof=1) == 0.0 or np.std(x_pilot, ddof=1) == 0.0:
        corr_estimate = 0.0
    else:
        corr_estimate = float(np.corrcoef(x_pilot, g_pilot)[0, 1])

    # --- Main sample: independent seed to avoid pilot correlation bias ---
    main_cfg = dataclasses.replace(cfg, seed=cfg.seed + 1)
    main_paths = simulate_paths(main_cfg)
    x_main = arithmetic_asian_call_payoff(main_paths, main_cfg)
    g_main = geometric_asian_call_payoff(main_paths, main_cfg)

    corrected = apply_control_variate(x_main, g_main, beta_hat, eg)

    runtime_s = time.perf_counter() - t0
    stats = summarise_estimates(corrected, cfg.discount_factor, runtime_s)

    return CVEstimateResult(
        price=stats["price"],
        variance=stats["variance"],
        std_dev=stats["std_dev"],
        std_error=stats["std_error"],
        ci_lower=stats["ci_lower"],
        ci_upper=stats["ci_upper"],
        n_paths=stats["n_paths"],
        runtime_s=stats["runtime_s"],
        beta_hat=beta_hat,
        corr_estimate=corr_estimate,
    )
