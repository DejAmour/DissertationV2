"""
test_analytical.py
==================
Stage 3 tests for the geometric Asian analytical pricing formula.

Tests cover:
C. Geometric analytical consistency — closed-form vs large Monte Carlo.
   Tolerance is justified from the MC standard error (5 × SE).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from asian_options.config import ModelConfig
from asian_options.analytical import geometric_asian_call_price
from asian_options.simulate_gbm import simulate_paths
from asian_options.payoffs import geometric_asian_call_payoff

BASE_CFG = dict(S0=100.0, K=100.0, r=0.05, q=0.02, sigma=0.2, T=1.0, m=12,
                n_paths=10_000, seed=42)


def _cfg(**kw) -> ModelConfig:
    return ModelConfig(**{**BASE_CFG, **kw})


def _mc_geo_price(cfg: ModelConfig) -> tuple[float, float]:
    """Return (MC geometric Asian price, MC SE) using simulate_paths."""
    paths = simulate_paths(cfg)
    payoffs = geometric_asian_call_payoff(paths, cfg)
    price = float(payoffs.mean())
    se = float(payoffs.std(ddof=1) / math.sqrt(cfg.n_paths))
    return price, se


# ---------------------------------------------------------------------------
# C. Geometric analytical consistency
# ---------------------------------------------------------------------------

class TestGeometricAnalyticalVsMC:
    """
    Compares geometric_asian_call_price to large-sample Monte Carlo.
    Tolerance = 5 × MC SE (roughly ±5σ, p < 3e-7 for a false alarm).
    """

    N = 300_000

    def _check(self, cfg_kw: dict) -> None:
        cfg = _cfg(n_paths=self.N, **cfg_kw)
        analytical = geometric_asian_call_price(cfg)
        mc_price, se = _mc_geo_price(cfg)
        tol = 5 * se
        assert abs(analytical - mc_price) < tol, (
            f"Analytical {analytical:.5f} vs MC {mc_price:.5f}; "
            f"diff={abs(analytical-mc_price):.5f}, tol (5×SE)={tol:.5f}"
        )

    def test_atm(self):
        self._check({})

    def test_itm(self):
        self._check({"K": 90.0})

    def test_otm(self):
        self._check({"K": 110.0})

    def test_nonzero_dividend(self):
        self._check({"q": 0.03})

    def test_high_vol(self):
        self._check({"sigma": 0.4})

    def test_short_maturity(self):
        self._check({"T": 0.25, "m": 3})

    def test_many_monitoring_dates(self):
        self._check({"m": 252})


# ---------------------------------------------------------------------------
# Additional analytical property tests
# ---------------------------------------------------------------------------

class TestGeometricAnalyticalProperties:
    def test_returns_float(self):
        cfg = _cfg()
        assert isinstance(geometric_asian_call_price(cfg), float)

    def test_non_negative(self):
        cfg = _cfg()
        assert geometric_asian_call_price(cfg) >= 0.0

    def test_deep_otm_near_zero(self):
        cfg = _cfg(K=300.0)  # S0=100 far OTM
        assert geometric_asian_call_price(cfg) < 1e-4

    def test_deep_itm_positive(self):
        cfg = _cfg(K=10.0)  # deep ITM
        assert geometric_asian_call_price(cfg) > 0.0

    def test_geom_leq_arith_call(self):
        """Geometric Asian call <= arithmetic Asian call (AM-GM inequality)."""
        from asian_options.estimators import standard_monte_carlo

        cfg = _cfg(n_paths=50_000, seed=0)
        geo = geometric_asian_call_price(cfg)
        arith_result = standard_monte_carlo(cfg)
        # Geometric Asian <= Arithmetic Asian (with large MC tolerance)
        se = arith_result.std_error
        assert geo <= arith_result.price + 5 * se

    def test_monotone_in_strike(self):
        """Price decreases as K increases (standard call monotonicity)."""
        prices = [
            geometric_asian_call_price(_cfg(K=k))
            for k in [80.0, 100.0, 120.0]
        ]
        assert prices[0] > prices[1] > prices[2]

    def test_m1_single_date_matches_bs_style(self):
        """With m=1, monitoring = maturity, geometric Asian ≈ European call structure."""
        cfg = _cfg(m=1, n_paths=200_000)
        # Just verify it returns a reasonable positive value
        price = geometric_asian_call_price(cfg)
        assert price > 0.0
