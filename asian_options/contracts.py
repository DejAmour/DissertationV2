"""
contracts.py
============
Seven-contract grid for Stage 8 frozen-NCV transfer experiments.

Contract set
------------
S0=100, r=0.05, m=12 fixed across all contracts; one parameter varies
at a time relative to the reference contract.

Contract labels and parameters::

    reference:       K=100, sigma=0.20, T=1.0
    strike_low:      K=80,  sigma=0.20, T=1.0
    strike_high:     K=120, sigma=0.20, T=1.0
    volatility_low:  K=100, sigma=0.10, T=1.0
    volatility_high: K=100, sigma=0.50, T=1.0
    maturity_short:  K=100, sigma=0.20, T=0.5
    maturity_long:   K=100, sigma=0.20, T=2.0

For every T, monitoring dates are t_j = j*T/m, j=1..m (encoded in ModelConfig
as dt=T/m with m equally-spaced steps).

Usage
-----
>>> from asian_options.contracts import CONTRACT_GRID, REFERENCE_ID
>>> ref_params = CONTRACT_GRID[REFERENCE_ID]
"""
from __future__ import annotations

import dataclasses
from typing import Dict, List, Tuple

from asian_options.config import ModelConfig

# ---------------------------------------------------------------------------
# Fixed parameters shared across all contracts
# ---------------------------------------------------------------------------
_S0 = 100.0
_r = 0.05
_m = 12

# Reference contract id
REFERENCE_ID: str = "reference"

# ---------------------------------------------------------------------------
# Contract parameter table
# ---------------------------------------------------------------------------
# Each entry: (K, sigma, T)
_CONTRACT_PARAMS: Dict[str, Tuple[float, float, float]] = {
    "reference":       (100.0, 0.20, 1.0),
    "strike_low":      ( 80.0, 0.20, 1.0),
    "strike_high":     (120.0, 0.20, 1.0),
    "volatility_low":  (100.0, 0.10, 1.0),
    "volatility_high": (100.0, 0.50, 1.0),
    "maturity_short":  (100.0, 0.20, 0.5),
    "maturity_long":   (100.0, 0.20, 2.0),
}

# Ordered list of contract ids: reference first, then the six targets.
CONTRACT_IDS: List[str] = [
    "reference",
    "strike_low",
    "strike_high",
    "volatility_low",
    "volatility_high",
    "maturity_short",
    "maturity_long",
]

TARGET_IDS: List[str] = [cid for cid in CONTRACT_IDS if cid != REFERENCE_ID]


def make_contract_cfg(
    contract_id: str,
    n_paths: int = 50_000,
    seed: int = 42,
) -> ModelConfig:
    """
    Build a ModelConfig for a named contract in the Stage 8 grid.

    Parameters
    ----------
    contract_id : str
        One of the keys in CONTRACT_IDS.
    n_paths : int
        Number of pricing paths.
    seed : int
        Base seed (callers typically derive this from a replication-level seed).

    Returns
    -------
    ModelConfig
        Validated configuration for the contract.

    Raises
    ------
    KeyError
        If contract_id is not in the grid.
    """
    if contract_id not in _CONTRACT_PARAMS:
        raise KeyError(
            f"Unknown contract_id '{contract_id}'. "
            f"Valid options: {list(_CONTRACT_PARAMS)}"
        )
    K, sigma, T = _CONTRACT_PARAMS[contract_id]
    return ModelConfig(
        S0=_S0,
        K=K,
        r=_r,
        sigma=sigma,
        T=T,
        m=_m,
        n_paths=n_paths,
        seed=seed,
    )


# ---------------------------------------------------------------------------
# Grid dict: contract_id -> (K, sigma, T) for lightweight introspection
# ---------------------------------------------------------------------------
CONTRACT_GRID: Dict[str, Tuple[float, float, float]] = dict(_CONTRACT_PARAMS)


def validate_contract_grid() -> None:
    """
    Validate that the contract grid satisfies the one-parameter-change rule
    and that all monitoring-date parameters are correct.

    Raises
    ------
    AssertionError
        If any contract violates the grid specification.
    """
    ref_K, ref_sigma, ref_T = _CONTRACT_PARAMS[REFERENCE_ID]
    assert ref_K == 100.0, "Reference K must be 100"
    assert ref_sigma == 0.20, "Reference sigma must be 0.20"
    assert ref_T == 1.0, "Reference T must be 1.0"

    for cid, (K, sigma, T) in _CONTRACT_PARAMS.items():
        if cid == REFERENCE_ID:
            continue
        # Exactly one parameter differs from reference
        diffs = sum([K != ref_K, sigma != ref_sigma, T != ref_T])
        assert diffs == 1, (
            f"Contract '{cid}' must differ from reference in exactly one "
            f"parameter; found diffs={diffs} (K={K}, sigma={sigma}, T={T})"
        )
        # All parameters must be valid ModelConfig values
        cfg = make_contract_cfg(cid, n_paths=10, seed=0)
        assert cfg.m == _m, f"Contract '{cid}': m must be {_m}, got {cfg.m}"
        assert cfg.dt == T / _m, (
            f"Contract '{cid}': dt={cfg.dt} must equal T/m={T/_m}"
        )

    assert set(CONTRACT_IDS) == set(_CONTRACT_PARAMS.keys()), (
        "CONTRACT_IDS does not match _CONTRACT_PARAMS keys"
    )
