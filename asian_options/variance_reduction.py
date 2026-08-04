"""
variance_reduction.py
=====================
Classical variance-reduction utilities (antithetic variates, control variates).

Stage 4 implements:
- ``estimate_beta``: OLS control-variate coefficient from a pilot sample.
- ``apply_control_variate``: corrected observation Y = X - beta*(G - E[G]).

Control-variate formula
-----------------------
Given arithmetic Asian payoffs X and geometric Asian payoffs G on the same
paths, with known analytical expectation E[G], the CV-corrected observation is:

    Y = X - beta * (G - E[G])

The optimal beta minimises Var(Y):

    beta* = Cov(X, G) / Var(G)

When Var(G) == 0 (degenerate control, e.g. zero-vol or deep ITM/OTM corner),
``estimate_beta`` returns 0.0 so that Y = X (fallback to plain MC).
"""

from __future__ import annotations

import numpy as np


def estimate_beta(
    payoffs_x: np.ndarray,
    payoffs_g: np.ndarray,
) -> float:
    """
    Estimate the control-variate coefficient beta via OLS on a pilot sample.

    .. math::

        \\hat{\\beta} = \\frac{\\mathrm{Cov}(X, G)}{\\mathrm{Var}(G)}

    When ``Var(G) == 0`` (degenerate control), returns ``0.0`` so that the
    corrected observations fall back to the plain arithmetic payoffs.

    Parameters
    ----------
    payoffs_x : np.ndarray
        Arithmetic Asian payoffs on the pilot sample, shape ``(n_pilot,)``.
    payoffs_g : np.ndarray
        Geometric Asian payoffs on the pilot sample, shape ``(n_pilot,)``.

    Returns
    -------
    float
        Estimated coefficient beta (0.0 when Var(G) is zero).

    Raises
    ------
    ValueError
        If the arrays have different lengths or fewer than 2 elements.
    """
    x = np.asarray(payoffs_x, dtype=np.float64).ravel()
    g = np.asarray(payoffs_g, dtype=np.float64).ravel()
    if x.shape != g.shape:
        raise ValueError(
            f"payoffs_x and payoffs_g must have the same shape, "
            f"got {x.shape} and {g.shape}"
        )
    if len(x) < 2:
        raise ValueError(
            f"Need at least 2 observations to estimate beta, got {len(x)}"
        )
    var_g = float(np.var(g, ddof=1))
    if var_g == 0.0:
        # Degenerate control: G is constant, no variance to exploit.
        return 0.0
    cov_xg = float(np.cov(x, g, ddof=1)[0, 1])
    return cov_xg / var_g


def apply_control_variate(
    payoffs_x: np.ndarray,
    payoffs_g: np.ndarray,
    beta: float,
    eg: float,
) -> np.ndarray:
    """
    Apply the control-variate correction.

    .. math::

        Y_i = X_i - \\beta \\cdot (G_i - \\mathbb{E}[G])

    Parameters
    ----------
    payoffs_x : np.ndarray
        Arithmetic Asian payoffs, shape ``(n_paths,)``.
    payoffs_g : np.ndarray
        Geometric Asian payoffs, shape ``(n_paths,)``.
    beta : float
        Frozen control-variate coefficient (estimated on a separate pilot).
    eg : float
        Analytical expectation of the geometric Asian payoff (E[G]).

    Returns
    -------
    np.ndarray
        Corrected observations Y, shape ``(n_paths,)``.

    Raises
    ------
    ValueError
        If the arrays have different lengths.
    """
    x = np.asarray(payoffs_x, dtype=np.float64).ravel()
    g = np.asarray(payoffs_g, dtype=np.float64).ravel()
    if x.shape != g.shape:
        raise ValueError(
            f"payoffs_x and payoffs_g must have the same shape, "
            f"got {x.shape} and {g.shape}"
        )
    return x - beta * (g - eg)
