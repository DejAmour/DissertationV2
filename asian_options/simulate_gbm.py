"""
simulate_gbm.py
===============
Risk-neutral GBM path simulation for Asian option pricing.

Stage 1 placeholder: function signatures and docstrings are defined;
the full numerical implementation is deferred to Stage 2.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from asian_options.config import ModelConfig


def simulate_paths(
    cfg: ModelConfig,
    shocks: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate risk-neutral GBM paths at the monitoring dates.

    Under risk-neutral measure each step follows::

        S_{t_j} = S_{t_{j-1}} * exp((r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_j)

    where Z_j ~ N(0,1) i.i.d.

    Parameters
    ----------
    cfg : ModelConfig
        Validated model and simulation configuration.
    shocks : np.ndarray, optional
        Pre-generated standard-normal shock matrix of shape
        ``(cfg.n_paths, cfg.m)``.  If supplied, ``rng`` is ignored.
    rng : np.random.Generator, optional
        Explicit NumPy random-number generator.  If neither ``shocks`` nor
        ``rng`` is provided, one is created from ``cfg.seed``.

    Returns
    -------
    np.ndarray
        Simulated spot prices at monitoring dates,
        shape ``(cfg.n_paths, cfg.m)``.
        The initial spot ``S0`` is **not** included in the output array.

    Raises
    ------
    ValueError
        If ``shocks`` is supplied with the wrong shape.

    Notes
    -----
    Convention: the returned array contains prices at monitoring dates
    t_1, t_2, ..., t_m only.  The initial spot S0 is **not** included, so
    the output shape is always ``(n_paths, m)``.
    """
    n, m = cfg.n_paths, cfg.m

    if shocks is not None:
        Z = np.asarray(shocks, dtype=np.float64)
        if Z.shape != (n, m):
            raise ValueError(
                f"shocks must have shape ({n}, {m}), got {Z.shape}"
            )
    else:
        if rng is None:
            rng = np.random.default_rng(cfg.seed)
        Z = rng.standard_normal((n, m))

    dt = cfg.dt
    drift = (cfg.r - cfg.q - 0.5 * cfg.sigma ** 2) * dt
    diffusion = cfg.sigma * np.sqrt(dt)

    # Compute log-increments for all steps, then cumsum for log-prices
    log_increments = drift + diffusion * Z          # shape (n_paths, m)
    log_paths = np.cumsum(log_increments, axis=1)   # shape (n_paths, m)

    paths: np.ndarray = cfg.S0 * np.exp(log_paths)
    return paths.astype(np.float64)
