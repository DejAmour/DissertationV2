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
    NotImplementedError
        Stage 2 will replace this placeholder with the full implementation.
    """
    raise NotImplementedError("GBM simulation will be implemented in Stage 2.")
