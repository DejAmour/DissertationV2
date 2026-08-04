"""
analytical.py
=============
Analytical pricing utilities for Asian options.

Stage 1 placeholder: the closed-form geometric Asian option price and the
analytical neural-network expectation will be implemented in Stage 3 / Stage 7.

Stage 2 adds: ``black_scholes_call`` — European call benchmark for simulator
validation.
"""

from __future__ import annotations

from asian_options.config import ModelConfig


def black_scholes_call(cfg: ModelConfig) -> float:
    """
    Black-Scholes price for a European call option.

    Uses the standard formula::

        d1 = (log(S0/K) + (r - q + 0.5*sigma^2)*T) / (sigma*sqrt(T))
        d2 = d1 - sigma*sqrt(T)
        C  = S0*exp(-q*T)*N(d1) - K*exp(-r*T)*N(d2)

    Parameters
    ----------
    cfg : ModelConfig
        Validated model and simulation configuration.

    Returns
    -------
    float
        European call price.
    """
    from math import log, exp, sqrt
    from scipy.stats import norm  # type: ignore[import]

    sqrtT = sqrt(cfg.T)
    d1 = (
        log(cfg.S0 / cfg.K) + (cfg.r - cfg.q + 0.5 * cfg.sigma ** 2) * cfg.T
    ) / (cfg.sigma * sqrtT)
    d2 = d1 - cfg.sigma * sqrtT
    return (
        cfg.S0 * exp(-cfg.q * cfg.T) * norm.cdf(d1)
        - cfg.K * exp(-cfg.r * cfg.T) * norm.cdf(d2)
    )


def geometric_asian_call_price(cfg: ModelConfig) -> float:
    """
    Closed-form price for a discrete geometric Asian call option.

    Under risk-neutral GBM the geometric average is log-normally distributed,
    which admits an exact Black-Scholes-type formula.  The derivation adjusts
    the drift and volatility to account for the averaging.

    Parameters
    ----------
    cfg : ModelConfig
        Validated model and simulation configuration.

    Returns
    -------
    float
        Analytical price of the geometric Asian call.

    Raises
    ------
    NotImplementedError
        Stage 3 will implement the closed-form formula.
    """
    raise NotImplementedError(
        "Analytical geometric Asian price will be implemented in Stage 3."
    )


def relu_expected_value(mu: float, sigma: float) -> float:
    """
    Analytical expectation of max(a, 0) where a ~ N(mu, sigma^2).

    E[a^+] = sigma * phi(mu/sigma) + mu * Phi(mu/sigma)

    When sigma == 0, E[a^+] = max(mu, 0).

    Parameters
    ----------
    mu : float
        Mean of the pre-activation.
    sigma : float
        Standard deviation of the pre-activation (>= 0).

    Returns
    -------
    float
        E[max(a, 0)].

    Raises
    ------
    NotImplementedError
        Stage 7 will implement the full analytical network expectation.
    """
    raise NotImplementedError(
        "Analytical ReLU expectation will be implemented in Stage 7."
    )
