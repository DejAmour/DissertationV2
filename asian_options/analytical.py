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
    Closed-form price for a discretely monitored geometric Asian call option.

    Under risk-neutral GBM with continuous dividend yield ``q``, the geometric
    average of m equally-spaced monitoring dates t_j = j*dt (j=1..m) is
    log-normally distributed.  The closed-form price is a Black-Scholes-type
    formula on the adjusted parameters.

    Derivation
    ----------
    Let G = (S_{t_1} * ... * S_{t_m})^{1/m}.  Under Q::

        log G = (1/m) * sum_{j=1}^{m} log S_{t_j}
              = log(S0) + (1/m) * sum_{j=1}^{m} [mu_adj * j*dt + sigma * B_{j*dt}]

    where ``mu_adj = r - q - 0.5*sigma^2``.

    The sum of j*dt for j=1..m is dt * m*(m+1)/2, so:

        E[log G] = log(S0) + mu_adj * dt * (m+1)/2
        Var[log G] = sigma^2 * dt * (m+1)*(2m+1) / (6m)

    Define::

        mu_G  = log(S0) + (r - q - 0.5*sigma^2) * T * (m+1)/(2m)
        sigma_G = sigma * sqrt(T * (m+1)*(2m+1) / (6*m^2))

    The price is then (Kemna & Vorst, 1990 style)::

        d1 = (mu_G - log(K) + sigma_G^2) / sigma_G
        d2 = d1 - sigma_G
        C  = exp(-r*T) * (exp(mu_G + 0.5*sigma_G^2) * N(d1) - K * N(d2))

    References
    ----------
    Kemna, A. G. Z. & Vorst, A. C. F. (1990).  "A Pricing Method for Options
    Based on Average Asset Values."  Journal of Banking & Finance, 14(1), 113–129.

    Parameters
    ----------
    cfg : ModelConfig
        Validated model and simulation configuration.

    Returns
    -------
    float
        Analytical price of the discretely monitored geometric Asian call.
    """
    from math import log, exp, sqrt
    from scipy.stats import norm  # type: ignore[import]

    m = cfg.m
    dt = cfg.dt
    sigma = cfg.sigma
    r = cfg.r
    q = cfg.q
    S0 = cfg.S0
    K = cfg.K
    T = cfg.T

    # Adjusted drift for log(G)
    mu_G = log(S0) + (r - q - 0.5 * sigma ** 2) * T * (m + 1) / (2 * m)

    # Variance of log(G)
    var_G = sigma ** 2 * T * (m + 1) * (2 * m + 1) / (6 * m ** 2)
    sigma_G = sqrt(var_G)

    if sigma_G == 0.0:
        # Degenerate: G is deterministic
        G_val = exp(mu_G)
        return exp(-r * T) * max(G_val - K, 0.0)

    d1 = (mu_G - log(K) + sigma_G ** 2) / sigma_G
    d2 = d1 - sigma_G

    price = exp(-r * T) * (
        exp(mu_G + 0.5 * sigma_G ** 2) * norm.cdf(d1)
        - K * norm.cdf(d2)
    )
    return float(price)


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
