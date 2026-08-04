"""
config.py
=========
Validated configuration dataclass for Asian option pricing experiments.

All financial and simulation parameters live here so that every module
receives an identical, already-validated snapshot of the experiment setup.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """
    Parameters for arithmetic Asian option pricing under risk-neutral GBM.

    Attributes
    ----------
    S0 : float
        Initial spot price (must be strictly positive).
    K : float
        Strike price (must be strictly positive).
    r : float
        Continuously-compounded risk-free rate.  May be zero or negative.
    q : float
        Continuous dividend / convenience-yield rate.  Defaults to zero.
    sigma : float
        Annualised volatility (must be strictly positive).
    T : float
        Option maturity in years (must be strictly positive).
    m : int
        Number of equally-spaced monitoring dates (must be >= 1).
        The time step is dt = T / m.
    n_paths : int
        Number of Monte Carlo sample paths (must be >= 1).
    seed : int
        Non-negative integer seed for reproducible random-number generation.
    """

    S0: float
    K: float
    r: float
    sigma: float
    T: float
    m: int
    n_paths: int
    q: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate all parameters immediately after construction."""
        if self.S0 <= 0:
            raise ValueError(f"S0 must be strictly positive, got {self.S0}")
        if self.K <= 0:
            raise ValueError(f"K must be strictly positive, got {self.K}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {self.sigma}")
        if self.T <= 0:
            raise ValueError(f"T must be strictly positive, got {self.T}")
        if self.m < 1:
            raise ValueError(f"m must be >= 1, got {self.m}")
        if self.n_paths < 1:
            raise ValueError(f"n_paths must be >= 1, got {self.n_paths}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

    @property
    def dt(self) -> float:
        """Uniform time step between consecutive monitoring dates (years)."""
        return self.T / self.m

    @property
    def discount_factor(self) -> float:
        """Risk-neutral discount factor e^{-rT}."""
        import math
        return math.exp(-self.r * self.T)
