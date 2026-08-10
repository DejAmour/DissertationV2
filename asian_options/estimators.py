"""
estimators.py
=============
Monte Carlo estimator entry points.

Each estimator returns an ``EstimateResult`` (or ``CVEstimateResult``)
named-tuple that carries all summary statistics needed for comparison tables,
including Stage 4 budget-accounting fields.

Budget conventions
------------------
- MC with N observations uses N paths.
- AV with N pair-observations uses 2N simulated paths.
- CV uses ``pilot_paths`` plus ``pricing_simulated_paths`` main paths.
- NCV uses ``training_paths`` plus ``pricing_simulated_paths`` main paths.

Variance conventions
--------------------
- ``observation_variance``: sample variance (ddof=1) of the corrected
  per-observation values (equals the ``variance`` field).
- ``estimator_variance``: ``observation_variance / pricing_observations``
  — the variance of the mean price estimator.
"""

from __future__ import annotations

from typing import NamedTuple, Optional

import numpy as np

from asian_options.config import ModelConfig


class EstimateResult(NamedTuple):
    """
    Summary statistics produced by every Monte Carlo estimator.

    Core fields
    -----------
    price : float
        Estimated option price (mean of corrected observations).
    variance : float
        Sample variance (ddof=1) of the per-observation values.
        Alias: ``observation_variance``.
    std_dev : float
        Sample standard deviation of the observations.
    std_error : float
        Standard error of the mean price estimate
        (``sqrt(observation_variance / pricing_observations)``).
    ci_lower, ci_upper : float
        Bounds of the 95 % confidence interval (``price ± 1.96 * std_error``).
    n_paths : int
        Kept for backward compatibility; equals ``pricing_observations``
        for MC and NCV, and for AV equals the number of antithetic *pairs*.
    runtime_s : float
        Wall-clock runtime in seconds.

    Budget fields (Stage 4)
    -----------------------
    pricing_observations : int
        Number of independent estimator observations contributing to ``price``.
        For MC/NCV this equals the number of pricing paths simulated.
        For AV this is the number of antithetic pairs (each pair = 2 paths).
    pricing_simulated_paths : int
        Paths simulated in the pricing (non-pilot/non-training) step.
        For AV this is ``2 * pricing_observations``.
    pilot_paths : int
        Paths used in a pilot phase (0 for MC, AV, NCV).
    training_paths : int
        Paths used in a training phase (0 for MC, AV, CV).
    total_simulated_paths : int
        ``pilot_paths + training_paths + pricing_simulated_paths``.

    Variance fields (Stage 4)
    -------------------------
    observation_variance : float
        Sample variance (ddof=1) of the corrected observations; equals
        ``variance``.  Named explicitly to avoid ambiguity with estimator
        variance.
    estimator_variance : float
        ``observation_variance / pricing_observations`` — variance of the mean
        price estimator.
    """

    price: float
    variance: float
    std_dev: float
    std_error: float
    ci_lower: float
    ci_upper: float
    n_paths: int
    runtime_s: float
    # Stage 4 budget fields (default 0 for backward compatibility)
    pricing_observations: int = 0
    pricing_simulated_paths: int = 0
    pilot_paths: int = 0
    training_paths: int = 0
    total_simulated_paths: int = 0
    # Stage 4 variance fields
    observation_variance: float = 0.0
    estimator_variance: float = 0.0
    # Stage 7 runtime-scope fields
    # pricing_runtime_seconds: simulation + payoff + correction only (no training/pilot)
    # training_runtime_seconds: pilot (CV) or network training (NCV); 0.0 for MC/AV
    # end_to_end_runtime_seconds: training + pricing + all prep (>= pricing + training)
    pricing_runtime_seconds: float = 0.0
    training_runtime_seconds: float = 0.0
    end_to_end_runtime_seconds: float = 0.0


class CVEstimateResult(NamedTuple):
    """
    Extended summary statistics for the control-variate estimator.

    Carries all fields of ``EstimateResult`` plus CV-specific metadata.

    Core fields
    -----------
    price : float
        CV-corrected estimated option price.
    variance : float
        Sample variance of the CV-corrected per-path observations.
        Alias: ``observation_variance``.
    std_dev, std_error, ci_lower, ci_upper, n_paths, runtime_s
        As in ``EstimateResult``.
    beta_hat : float
        Estimated control-variate coefficient from the pilot sample.
    corr_estimate : float
        Sample correlation between arithmetic and geometric payoffs on the
        pilot sample.

    Budget / variance fields (Stage 4)
    ------------------------------------
    pricing_observations, pricing_simulated_paths, pilot_paths,
    training_paths, total_simulated_paths, observation_variance,
    estimator_variance : same semantics as in ``EstimateResult``.
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
    # Stage 4 budget fields
    pricing_observations: int = 0
    pricing_simulated_paths: int = 0
    pilot_paths: int = 0
    training_paths: int = 0
    total_simulated_paths: int = 0
    # Stage 4 variance fields
    observation_variance: float = 0.0
    estimator_variance: float = 0.0
    # Stage 7 runtime-scope fields
    pricing_runtime_seconds: float = 0.0
    training_runtime_seconds: float = 0.0
    end_to_end_runtime_seconds: float = 0.0


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
    n_obs = stats["n_paths"]
    obs_var = stats["variance"]
    return EstimateResult(
        price=stats["price"],
        variance=obs_var,
        std_dev=stats["std_dev"],
        std_error=stats["std_error"],
        ci_lower=stats["ci_lower"],
        ci_upper=stats["ci_upper"],
        n_paths=n_obs,
        runtime_s=stats["runtime_s"],
        # Budget fields: MC — observations == simulated paths
        pricing_observations=n_obs,
        pricing_simulated_paths=n_obs,
        pilot_paths=0,
        training_paths=0,
        total_simulated_paths=n_obs,
        # Variance fields
        observation_variance=obs_var,
        estimator_variance=obs_var / n_obs,
        # Stage 7 runtime-scope fields: MC has no training/pilot phase
        pricing_runtime_seconds=runtime_s,
        training_runtime_seconds=0.0,
        end_to_end_runtime_seconds=runtime_s,
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
    n_pairs = stats["n_paths"]          # number of antithetic pairs = pricing_observations
    obs_var = stats["variance"]          # variance of pair-average observations
    return EstimateResult(
        price=stats["price"],
        variance=obs_var,
        std_dev=stats["std_dev"],
        std_error=stats["std_error"],
        ci_lower=stats["ci_lower"],
        ci_upper=stats["ci_upper"],
        n_paths=n_pairs,
        runtime_s=stats["runtime_s"],
        # Budget fields: AV — each pair observation requires 2 simulated paths
        pricing_observations=n_pairs,
        pricing_simulated_paths=2 * n_pairs,
        pilot_paths=0,
        training_paths=0,
        total_simulated_paths=2 * n_pairs,
        # Variance fields
        observation_variance=obs_var,
        estimator_variance=obs_var / n_pairs,
        # Stage 7 runtime-scope fields: AV has no training/pilot phase
        pricing_runtime_seconds=runtime_s,
        training_runtime_seconds=0.0,
        end_to_end_runtime_seconds=runtime_s,
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

    t_after_pilot = time.perf_counter()
    training_runtime_s = t_after_pilot - t0  # pilot simulation + beta estimation

    # --- Main sample: independent seed to avoid pilot correlation bias ---
    main_cfg = dataclasses.replace(cfg, seed=cfg.seed + 1)
    main_paths = simulate_paths(main_cfg)
    x_main = arithmetic_asian_call_payoff(main_paths, main_cfg)
    g_main = geometric_asian_call_payoff(main_paths, main_cfg)

    corrected = apply_control_variate(x_main, g_main, beta_hat, eg)

    t_end = time.perf_counter()
    pricing_runtime_s = t_end - t_after_pilot  # main sample only
    runtime_s = t_end - t0  # end-to-end: pilot + pricing
    stats = summarise_estimates(corrected, cfg.discount_factor, runtime_s)
    n_pricing = stats["n_paths"]
    obs_var = stats["variance"]

    return CVEstimateResult(
        price=stats["price"],
        variance=obs_var,
        std_dev=stats["std_dev"],
        std_error=stats["std_error"],
        ci_lower=stats["ci_lower"],
        ci_upper=stats["ci_upper"],
        n_paths=n_pricing,
        runtime_s=runtime_s,
        beta_hat=beta_hat,
        corr_estimate=corr_estimate,
        # Budget fields: CV uses pilot paths then independent pricing paths
        pricing_observations=n_pricing,
        pricing_simulated_paths=n_pricing,
        pilot_paths=n_pilot,
        training_paths=0,
        total_simulated_paths=n_pilot + n_pricing,
        # Variance fields
        observation_variance=obs_var,
        estimator_variance=obs_var / n_pricing,
        # Stage 7 runtime-scope fields: pilot is training for CV
        pricing_runtime_seconds=pricing_runtime_s,
        training_runtime_seconds=training_runtime_s,
        end_to_end_runtime_seconds=runtime_s,
    )
