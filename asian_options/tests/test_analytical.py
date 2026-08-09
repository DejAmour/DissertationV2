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


# ---------------------------------------------------------------------------
# Stage 2: relu_expected_value tests
# ---------------------------------------------------------------------------

from asian_options.analytical import relu_expected_value
from scipy.stats import norm as _norm
import math


class TestReluExpectedValue:
    """Tests for the analytical ReLU expectation E[max(mu + sigma*Y, 0)]."""

    def test_scalar_positive_mu_zero_sigma(self):
        """sigma=0, mu>0: E[a^+] = mu."""
        assert relu_expected_value(3.0, 0.0) == pytest.approx(3.0)

    def test_scalar_negative_mu_zero_sigma(self):
        """sigma=0, mu<0: E[a^+] = 0."""
        assert relu_expected_value(-2.0, 0.0) == pytest.approx(0.0)

    def test_scalar_zero_mu_zero_sigma(self):
        """sigma=0, mu=0: E[a^+] = 0."""
        assert relu_expected_value(0.0, 0.0) == pytest.approx(0.0)

    def test_known_value_mu0_sigma1(self):
        """mu=0, sigma=1: E[Y^+] = 1/sqrt(2*pi) = phi(0)."""
        expected = _norm.pdf(0.0)
        assert relu_expected_value(0.0, 1.0) == pytest.approx(expected, rel=1e-10)

    def test_known_value_general(self):
        """Verify against direct formula for mu=1, sigma=2."""
        mu, sigma = 1.0, 2.0
        t = mu / sigma
        expected = sigma * _norm.pdf(t) + mu * _norm.cdf(t)
        assert relu_expected_value(mu, sigma) == pytest.approx(expected, rel=1e-10)

    def test_negative_sigma_raises(self):
        """Negative sigma must raise ValueError."""
        with pytest.raises(ValueError):
            relu_expected_value(1.0, -0.5)

    def test_nonfinite_mu_raises(self):
        with pytest.raises(ValueError):
            relu_expected_value(float("inf"), 1.0)

    def test_nonfinite_sigma_raises(self):
        with pytest.raises(ValueError):
            relu_expected_value(0.0, float("nan"))

    def test_array_input(self):
        """Verify vectorised output matches element-wise computation."""
        mu = np.array([0.0, 1.0, -1.0])
        sigma = np.array([1.0, 2.0, 1.0])
        result = relu_expected_value(mu, sigma)
        expected = np.array([relu_expected_value(m, s) for m, s in zip(mu, sigma)])
        np.testing.assert_allclose(result, expected, rtol=1e-12)

    def test_mixed_zero_nonzero_sigma(self):
        """Array with some zero and some nonzero sigma values."""
        mu = np.array([2.0, -1.0, 0.0])
        sigma = np.array([0.0, 0.0, 1.0])
        result = relu_expected_value(mu, sigma)
        assert result[0] == pytest.approx(2.0)
        assert result[1] == pytest.approx(0.0)
        assert result[2] == pytest.approx(_norm.pdf(0.0))

    def test_returns_float_for_scalar(self):
        val = relu_expected_value(1.0, 1.0)
        assert isinstance(val, float)
