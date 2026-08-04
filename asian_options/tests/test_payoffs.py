"""
test_payoffs.py
===============
Stage 3 tests for Asian option payoff functions.

Convention confirmed here
-------------------------
* ``paths`` shape is ``(n_paths, m)`` with S0 **excluded**.
* Averaging is over all m columns (monitoring dates t_1, ..., t_m).
* Discounting uses ``cfg.discount_factor = exp(-r*T)``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from asian_options.config import ModelConfig
from asian_options.payoffs import (
    arithmetic_average,
    geometric_average,
    arithmetic_asian_call_payoff,
    geometric_asian_call_payoff,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

BASE_CFG = dict(S0=100.0, K=100.0, r=0.05, q=0.02, sigma=0.2, T=1.0, m=4,
                n_paths=3, seed=0)


def _cfg(**kw) -> ModelConfig:
    return ModelConfig(**{**BASE_CFG, **kw})


# ---------------------------------------------------------------------------
# A. Arithmetic / geometric average — hand-crafted exact cases
# ---------------------------------------------------------------------------

class TestArithmeticAverage:
    def test_constant_paths(self):
        paths = np.full((3, 4), 100.0)
        result = arithmetic_average(paths)
        np.testing.assert_allclose(result, [100.0, 100.0, 100.0])

    def test_known_values(self):
        paths = np.array([[1.0, 2.0, 3.0, 4.0],
                          [2.0, 4.0, 6.0, 8.0]])
        result = arithmetic_average(paths)
        np.testing.assert_allclose(result, [2.5, 5.0])

    def test_single_path_single_date(self):
        paths = np.array([[7.5]])
        result = arithmetic_average(paths)
        np.testing.assert_allclose(result, [7.5])

    def test_output_shape(self):
        paths = np.ones((5, 10))
        assert arithmetic_average(paths).shape == (5,)


class TestGeometricAverage:
    def test_constant_paths(self):
        paths = np.full((3, 4), 100.0)
        result = geometric_average(paths)
        np.testing.assert_allclose(result, [100.0, 100.0, 100.0])

    def test_known_values(self):
        # G = (1*2*4*8)^(1/4) = 64^(1/4) = (2^6)^(1/4) = 2^(6/4) = 2^1.5
        paths = np.array([[1.0, 2.0, 4.0, 8.0]])
        expected = 2 ** 1.5
        result = geometric_average(paths)
        np.testing.assert_allclose(result, [expected], rtol=1e-12)

    def test_arith_geq_geom(self):
        """AM-GM inequality: arithmetic average >= geometric average."""
        rng = np.random.default_rng(42)
        paths = rng.uniform(50.0, 150.0, size=(100, 12))
        assert np.all(arithmetic_average(paths) >= geometric_average(paths) - 1e-10)

    def test_output_shape(self):
        paths = np.ones((5, 10))
        assert geometric_average(paths).shape == (5,)


# ---------------------------------------------------------------------------
# B. Payoff functions — ITM / OTM / ATM edge cases
# ---------------------------------------------------------------------------

class TestArithmeticCallPayoff:
    def test_itm(self):
        """Average > K: payoff > 0."""
        cfg = _cfg(K=90.0)
        paths = np.full((1, 4), 100.0)  # avg = 100 > 90
        p = arithmetic_asian_call_payoff(paths, cfg)
        expected = cfg.discount_factor * (100.0 - 90.0)
        np.testing.assert_allclose(p, [expected])

    def test_otm(self):
        """Average < K: payoff = 0."""
        cfg = _cfg(K=110.0)
        paths = np.full((1, 4), 100.0)  # avg = 100 < 110
        p = arithmetic_asian_call_payoff(paths, cfg)
        np.testing.assert_allclose(p, [0.0])

    def test_atm(self):
        """Average == K: payoff = 0."""
        cfg = _cfg(K=100.0)
        paths = np.full((1, 4), 100.0)
        p = arithmetic_asian_call_payoff(paths, cfg)
        np.testing.assert_allclose(p, [0.0])

    def test_discounting(self):
        """Discount factor applied correctly."""
        cfg = _cfg(K=90.0, r=0.10, T=2.0, m=4)
        paths = np.full((1, 4), 100.0)  # avg = 100
        p = arithmetic_asian_call_payoff(paths, cfg)
        expected = math.exp(-0.10 * 2.0) * (100.0 - 90.0)
        np.testing.assert_allclose(p, [expected], rtol=1e-12)

    def test_output_shape(self):
        cfg = _cfg()
        paths = np.ones((7, 4)) * 105.0
        assert arithmetic_asian_call_payoff(paths, cfg).shape == (7,)


class TestGeometricCallPayoff:
    def test_itm(self):
        cfg = _cfg(K=90.0)
        # geometric avg of constant 100 = 100 > 90
        paths = np.full((1, 4), 100.0)
        p = geometric_asian_call_payoff(paths, cfg)
        expected = cfg.discount_factor * (100.0 - 90.0)
        np.testing.assert_allclose(p, [expected])

    def test_otm(self):
        cfg = _cfg(K=110.0)
        paths = np.full((1, 4), 100.0)
        p = geometric_asian_call_payoff(paths, cfg)
        np.testing.assert_allclose(p, [0.0])

    def test_known_geom_avg(self):
        # paths = [1, 2, 4, 8] → G = 2^1.5 ≈ 2.828
        cfg = _cfg(K=2.0, S0=1.0, n_paths=1, m=4)
        paths = np.array([[1.0, 2.0, 4.0, 8.0]])
        G = 2 ** 1.5
        expected = cfg.discount_factor * max(G - 2.0, 0.0)
        p = geometric_asian_call_payoff(paths, cfg)
        np.testing.assert_allclose(p, [expected], rtol=1e-12)

    def test_output_shape(self):
        cfg = _cfg()
        paths = np.ones((7, 4)) * 105.0
        assert geometric_asian_call_payoff(paths, cfg).shape == (7,)


# ---------------------------------------------------------------------------
# C. Shape / validation guards
# ---------------------------------------------------------------------------

class TestValidation:
    def test_1d_raises(self):
        paths_1d = np.ones(10)
        with pytest.raises(ValueError, match="2-D"):
            arithmetic_average(paths_1d)

    def test_3d_raises(self):
        paths_3d = np.ones((2, 3, 4))
        with pytest.raises(ValueError, match="2-D"):
            geometric_average(paths_3d)

    def test_empty_paths_rows_raises(self):
        paths = np.ones((0, 4))
        with pytest.raises(ValueError):
            arithmetic_average(paths)

    def test_empty_paths_cols_raises(self):
        paths = np.ones((4, 0))
        with pytest.raises(ValueError):
            arithmetic_average(paths)

    def test_nan_raises(self):
        paths = np.array([[1.0, np.nan, 3.0, 4.0]])
        with pytest.raises(ValueError, match="non-finite"):
            arithmetic_average(paths)

    def test_inf_raises(self):
        paths = np.array([[1.0, np.inf, 3.0, 4.0]])
        with pytest.raises(ValueError, match="non-finite"):
            geometric_average(paths)

    def test_payoff_invalid_shape_raises(self):
        cfg = _cfg()
        paths_bad = np.ones(10)
        with pytest.raises(ValueError, match="2-D"):
            arithmetic_asian_call_payoff(paths_bad, cfg)
