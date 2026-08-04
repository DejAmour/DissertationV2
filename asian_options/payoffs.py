"""
payoffs.py
==========
Asian option payoff calculations.

Stage 1 placeholder: interfaces defined; implementations deferred to Stage 3.
"""

from __future__ import annotations

import numpy as np

from asian_options.config import ModelConfig


def arithmetic_average(paths: np.ndarray) -> np.ndarray:
    """
    Compute the arithmetic average of spot prices across monitoring dates.

    Parameters
    ----------
    paths : np.ndarray
        Simulated spot prices, shape ``(n_paths, m)``.

    Returns
    -------
    np.ndarray
        Arithmetic average per path, shape ``(n_paths,)``.
    """
    raise NotImplementedError("Payoff functions will be implemented in Stage 3.")


def geometric_average(paths: np.ndarray) -> np.ndarray:
    """
    Compute the geometric average of spot prices across monitoring dates.

    Parameters
    ----------
    paths : np.ndarray
        Simulated spot prices, shape ``(n_paths, m)``.

    Returns
    -------
    np.ndarray
        Geometric average per path, shape ``(n_paths,)``.
    """
    raise NotImplementedError("Payoff functions will be implemented in Stage 3.")


def arithmetic_asian_call_payoff(
    paths: np.ndarray, cfg: ModelConfig
) -> np.ndarray:
    """
    Discounted arithmetic Asian call payoff for each path.

    f(Z) = e^{-rT} * max(A - K, 0)

    where A is the arithmetic average of monitored spot prices.

    Parameters
    ----------
    paths : np.ndarray
        Simulated spot prices, shape ``(n_paths, m)``.
    cfg : ModelConfig
        Configuration providing strike K, rate r, and maturity T.

    Returns
    -------
    np.ndarray
        Discounted payoff per path, shape ``(n_paths,)``.
    """
    raise NotImplementedError("Payoff functions will be implemented in Stage 3.")


def geometric_asian_call_payoff(
    paths: np.ndarray, cfg: ModelConfig
) -> np.ndarray:
    """
    Discounted geometric Asian call payoff for each path.

    Used as the Monte Carlo control variate target.

    Parameters
    ----------
    paths : np.ndarray
        Simulated spot prices, shape ``(n_paths, m)``.
    cfg : ModelConfig
        Configuration providing strike K, rate r, and maturity T.

    Returns
    -------
    np.ndarray
        Discounted geometric payoff per path, shape ``(n_paths,)``.
    """
    raise NotImplementedError("Payoff functions will be implemented in Stage 3.")
