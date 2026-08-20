"""
payoffs.py
==========
Asian option payoff calculations.

Convention
----------
* ``paths`` is always shape ``(n_paths, m)`` where column j corresponds to
  monitoring date t_{j+1} = (j+1)*dt.  The initial spot S0 is **not** stored
  in this array (consistent with ``simulate_paths`` in simulate_gbm.py).
* Averaging is over all m monitoring dates t_1, ..., t_m (columns 0..m-1).
* Discounting uses the risk-neutral discount factor e^{-rT} from ``cfg``.
"""

from __future__ import annotations

import numpy as np

from asian_options.config import ModelConfig


def _validate_paths(paths: np.ndarray) -> None:
    """Raise ValueError if paths has wrong ndim, invalid shape, or non-finite values."""
    if paths.ndim != 2:
        raise ValueError(
            f"paths must be 2-D with shape (n_paths, m), got ndim={paths.ndim}"
        )
    if paths.shape[0] < 1 or paths.shape[1] < 1:
        raise ValueError(
            f"paths must have at least one row and one column, got shape {paths.shape}"
        )
    if not np.all(np.isfinite(paths)):
        raise ValueError("paths contains non-finite values (NaN or Inf)")


def arithmetic_average(paths: np.ndarray) -> np.ndarray:
    """
    Compute the arithmetic average of spot prices across monitoring dates.

    Parameters
    ----------
    paths : np.ndarray
        Simulated spot prices, shape ``(n_paths, m)``.  S0 is **not** included;
        column j is the price at monitoring date t_{j+1}.

    Returns
    -------
    np.ndarray
        Arithmetic average per path, shape ``(n_paths,)``.

    Raises
    ------
    ValueError
        If ``paths`` is not 2-D, has an empty dimension, or contains non-finite
        values.
    """
    _validate_paths(paths)
    return np.mean(paths, axis=1)


def geometric_average(paths: np.ndarray) -> np.ndarray:
    """
    Compute the geometric average of spot prices across monitoring dates.

    Uses the log-space identity:  G = exp(mean(log(S_{t_j}))).

    Parameters
    ----------
    paths : np.ndarray
        Simulated spot prices, shape ``(n_paths, m)``.  S0 is **not** included;
        column j is the price at monitoring date t_{j+1}.

    Returns
    -------
    np.ndarray
        Geometric average per path, shape ``(n_paths,)``.

    Raises
    ------
    ValueError
        If ``paths`` is not 2-D, has an empty dimension, or contains non-finite
        values.
    """
    _validate_paths(paths)
    return np.exp(np.mean(np.log(paths), axis=1))


def arithmetic_asian_call_payoff(
    paths: np.ndarray, cfg: ModelConfig
) -> np.ndarray:
    """
    Discounted arithmetic Asian call payoff for each path.

    .. math::

        f(Z) = e^{-rT} \\cdot \\max\\!\\left(\\frac{1}{m}\\sum_{j=1}^{m} S_{t_j} - K,\\; 0\\right)

    Parameters
    ----------
    paths : np.ndarray
        Simulated spot prices, shape ``(n_paths, m)``.  S0 is **not** included.
        Averaging is over all m columns (monitoring dates t_1, ..., t_m).
    cfg : ModelConfig
        Configuration providing strike K, risk-free rate r, and maturity T.

    Returns
    -------
    np.ndarray
        Discounted payoff per path, shape ``(n_paths,)``.

    Raises
    ------
    ValueError
        If ``paths`` has the wrong shape or contains non-finite values.
    """
    _validate_paths(paths)
    A = arithmetic_average(paths)
    return cfg.discount_factor * np.maximum(A - cfg.K, 0.0)


def geometric_asian_call_payoff(
    paths: np.ndarray, cfg: ModelConfig
) -> np.ndarray:
    """
    Discounted geometric Asian call payoff for each path.

    .. math::

        f(Z) = e^{-rT} \\cdot \\max\\!\\left(\\left(\\prod_{j=1}^{m} S_{t_j}\\right)^{1/m} - K,\\; 0\\right)

    Used as the control variate in Stage 4.

    Parameters
    ----------
    paths : np.ndarray
        Simulated spot prices, shape ``(n_paths, m)``.  S0 is **not** included.
        Averaging is over all m columns (monitoring dates t_1, ..., t_m).
    cfg : ModelConfig
        Configuration providing strike K, risk-free rate r, and maturity T.

    Returns
    -------
    np.ndarray
        Discounted geometric payoff per path, shape ``(n_paths,)``.

    Raises
    ------
    ValueError
        If ``paths`` has the wrong shape or contains non-finite values.
    """
    _validate_paths(paths)
    G = geometric_average(paths)
    return cfg.discount_factor * np.maximum(G - cfg.K, 0.0)
