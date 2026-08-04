"""
test_estimators.py
==================
Stage 3 tests for the baseline Monte Carlo estimator and metrics.

Tests cover:
B. Estimator statistics sanity — fixed sample vectors, exact computations.
C. MC estimator end-to-end using standard_monte_carlo().
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from asian_options.config import ModelConfig
from asian_options.metrics import summarise_estimates
from asian_options.estimators import standard_monte_carlo, EstimateResult

BASE_CFG = dict(S0=100.0, K=100.0, r=0.05, q=0.02, sigma=0.2, T=1.0, m=12,
                n_paths=10_000, seed=42)


def _cfg(**kw) -> ModelConfig:
    return ModelConfig(**{**BASE_CFG, **kw})


# ---------------------------------------------------------------------------
# B. summarise_estimates — exact hand-crafted cases
# ---------------------------------------------------------------------------

class TestSummariseEstimates:
    def _fixed(self):
        """Small fixed sample: [1, 2, 3, 4, 5]."""
        return np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    def test_price_is_mean(self):
        obs = self._fixed()
        stats = summarise_estimates(obs, 1.0, 0.0)
        assert math.isclose(stats["price"], 3.0)

    def test_variance_ddof1(self):
        obs = self._fixed()
        stats = summarise_estimates(obs, 1.0, 0.0)
        expected_var = np.var(obs, ddof=1)  # 2.5
        assert math.isclose(stats["variance"], expected_var)

    def test_std_dev(self):
        obs = self._fixed()
        stats = summarise_estimates(obs, 1.0, 0.0)
        assert math.isclose(stats["std_dev"], math.sqrt(2.5))

    def test_std_error(self):
        obs = self._fixed()
        stats = summarise_estimates(obs, 1.0, 0.0)
        expected_se = math.sqrt(2.5) / math.sqrt(5)
        assert math.isclose(stats["std_error"], expected_se)

    def test_ci_uses_196(self):
        """CI must be mean ± 1.96 * SE."""
        obs = self._fixed()
        stats = summarise_estimates(obs, 1.0, 0.0)
        se = stats["std_error"]
        assert math.isclose(stats["ci_lower"], 3.0 - 1.96 * se)
        assert math.isclose(stats["ci_upper"], 3.0 + 1.96 * se)

    def test_n_paths(self):
        obs = self._fixed()
        stats = summarise_estimates(obs, 1.0, 0.0)
        assert stats["n_paths"] == 5

    def test_runtime_passed_through(self):
        obs = self._fixed()
        stats = summarise_estimates(obs, 1.0, 3.14)
        assert math.isclose(stats["runtime_s"], 3.14)

    def test_constant_array_zero_variance(self):
        obs = np.array([5.0, 5.0, 5.0, 5.0])
        stats = summarise_estimates(obs, 1.0, 0.0)
        assert stats["variance"] == 0.0
        assert stats["std_error"] == 0.0
        assert math.isclose(stats["ci_lower"], 5.0)
        assert math.isclose(stats["ci_upper"], 5.0)

    def test_single_element_raises(self):
        with pytest.raises(ValueError):
            summarise_estimates(np.array([1.0]), 1.0, 0.0)

    def test_2d_raises(self):
        with pytest.raises(ValueError, match="1-D"):
            summarise_estimates(np.ones((3, 3)), 1.0, 0.0)

    def test_ci_lower_lt_upper(self):
        obs = np.random.default_rng(0).uniform(0, 10, 100)
        stats = summarise_estimates(obs, 1.0, 0.0)
        assert stats["ci_lower"] < stats["ci_upper"]


# ---------------------------------------------------------------------------
# C. standard_monte_carlo end-to-end
# ---------------------------------------------------------------------------

class TestStandardMonteCarlo:
    def test_returns_estimate_result(self):
        cfg = _cfg()
        result = standard_monte_carlo(cfg)
        assert isinstance(result, EstimateResult)

    def test_price_positive(self):
        cfg = _cfg()
        result = standard_monte_carlo(cfg)
        assert result.price >= 0.0

    def test_variance_positive(self):
        cfg = _cfg()
        result = standard_monte_carlo(cfg)
        assert result.variance >= 0.0

    def test_n_paths_matches_config(self):
        cfg = _cfg(n_paths=500)
        result = standard_monte_carlo(cfg)
        assert result.n_paths == 500

    def test_ci_contains_price(self):
        cfg = _cfg()
        result = standard_monte_carlo(cfg)
        assert result.ci_lower <= result.price <= result.ci_upper

    def test_se_equals_std_over_sqrt_n(self):
        cfg = _cfg()
        result = standard_monte_carlo(cfg)
        expected_se = result.std_dev / math.sqrt(result.n_paths)
        assert math.isclose(result.std_error, expected_se, rel_tol=1e-9)

    def test_reproducible(self):
        cfg = _cfg(seed=7)
        r1 = standard_monte_carlo(cfg)
        r2 = standard_monte_carlo(cfg)
        assert r1.price == r2.price
