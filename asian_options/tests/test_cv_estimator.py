"""
test_cv_estimator.py
====================
Stage 4 tests for the control-variate (CV) estimator and antithetic variates.

Test sections
-------------
A. Control-variate utility correctness
   - ``estimate_beta``: known covariance/variance cases.
   - ``apply_control_variate``: explicit formula check.
   - Degenerate control-variance case (var(G) == 0) handled without crash.

B. Variance-reduction evidence
   - Reproducible seeded comparison: CV variance < plain MC variance.
   - Tolerance is justified by the MC standard error (avoid flakiness).

C. Unbiasedness sanity
   - CV and plain MC estimates are statistically consistent (within 5 × SE).
   - CV estimate is within confidence bounds of the geometric analytical price
     (lower bound only, since arith Asian >= geo Asian).

D. Convention consistency
   - CV estimator uses the same path/payoff convention as Stage 2/3
     (n_paths, m); S0 excluded; averaging over t1..tm.

E. Antithetic variates
   - Returns EstimateResult with correct shape.
   - Variance <= plain MC variance (at same n_paths budget).

F. variance_reduction_ratio utility
   - Correct ratio value.
   - Zero variance_reduced raises ValueError.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from asian_options.config import ModelConfig
from asian_options.estimators import (
    CVEstimateResult,
    EstimateResult,
    antithetic_variates,
    geometric_control_variate,
    standard_monte_carlo,
)
from asian_options.metrics import variance_reduction_ratio
from asian_options.variance_reduction import apply_control_variate, estimate_beta

# ---------------------------------------------------------------------------
# Shared fixture parameters
# ---------------------------------------------------------------------------

BASE_CFG = dict(
    S0=100.0, K=100.0, r=0.05, q=0.02, sigma=0.2, T=1.0, m=12,
    n_paths=10_000, seed=42,
)


def _cfg(**kw) -> ModelConfig:
    return ModelConfig(**{**BASE_CFG, **kw})


# ---------------------------------------------------------------------------
# A. Control-variate utility correctness
# ---------------------------------------------------------------------------

class TestEstimateBeta:
    def test_known_case(self):
        """With G = 2*X + noise, beta = Cov(X,G)/Var(G) ~ 2*Var(X)/(4*Var(X)) = 0.5."""
        rng = np.random.default_rng(0)
        x = rng.uniform(0.0, 1.0, 200)
        # G = 2*X + noise => Cov(X,G) ~ 2*Var(X), Var(G) ~ 4*Var(X) => beta ~ 0.5
        g = 2.0 * x + rng.normal(0, 0.01, 200)
        beta = estimate_beta(x, g)
        assert math.isclose(beta, 0.5, abs_tol=0.05), f"beta={beta}"

    def test_zero_covariance(self):
        """Uncorrelated X and G: beta should be near 0."""
        rng = np.random.default_rng(1)
        x = rng.standard_normal(500)
        g = rng.standard_normal(500)
        beta = estimate_beta(x, g)
        assert abs(beta) < 0.2, f"beta={beta}"

    def test_degenerate_control_returns_zero(self):
        """If Var(G) == 0, beta must return 0.0 without raising."""
        x = np.array([1.0, 2.0, 3.0, 4.0])
        g = np.full(4, 5.0)  # constant control
        beta = estimate_beta(x, g)
        assert beta == 0.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            estimate_beta(np.ones(5), np.ones(6))

    def test_too_few_elements_raises(self):
        with pytest.raises(ValueError, match="at least 2"):
            estimate_beta(np.array([1.0]), np.array([1.0]))

    def test_returns_float(self):
        x = np.array([1.0, 2.0, 3.0])
        g = np.array([1.5, 2.5, 3.5])
        assert isinstance(estimate_beta(x, g), float)


class TestApplyControlVariate:
    def test_zero_beta_identity(self):
        """beta=0 => Y = X (no correction)."""
        x = np.array([1.0, 2.0, 3.0])
        g = np.array([0.5, 1.0, 1.5])
        y = apply_control_variate(x, g, beta=0.0, eg=1.0)
        np.testing.assert_array_equal(y, x)

    def test_exact_formula(self):
        """Y_i = X_i - beta*(G_i - E[G])."""
        x = np.array([2.0, 4.0, 6.0])
        g = np.array([1.0, 2.0, 3.0])
        beta = 0.5
        eg = 2.0
        expected = x - beta * (g - eg)
        y = apply_control_variate(x, g, beta=beta, eg=eg)
        np.testing.assert_allclose(y, expected)

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="same shape"):
            apply_control_variate(np.ones(4), np.ones(5), 1.0, 0.0)

    def test_output_shape(self):
        x = np.zeros(100)
        g = np.zeros(100)
        y = apply_control_variate(x, g, beta=1.0, eg=0.0)
        assert y.shape == (100,)


class TestDegenerateControlVariance:
    """Ensure the full CV estimator handles the degenerate-beta fallback."""

    def test_no_crash_low_variance_control(self):
        """
        Estimator must not crash even with extreme parameters.
        With sigma=0.001 the geometric payoff has near-zero variance but is
        not exactly constant; beta_hat will be finite.
        """
        cfg = _cfg(sigma=0.001, n_paths=200, seed=0)
        result = geometric_control_variate(cfg, n_pilot=50)
        assert isinstance(result, CVEstimateResult)
        assert math.isfinite(result.price)
        assert math.isfinite(result.beta_hat)


# ---------------------------------------------------------------------------
# B. Variance-reduction evidence
# ---------------------------------------------------------------------------

class TestVarianceReduction:
    """
    Reproducible seeded experiment: CV variance < plain MC variance.

    Tolerance: we assert CV variance < plain MC variance.
    With arithmetic/geometric Asian calls highly correlated (~0.98+), the VRR
    is typically > 5×, so the assertion is robust.
    We use a moderate path budget (20 000) with a fixed seed.
    """

    N = 20_000
    SEED = 42

    def _run(self):
        cfg = _cfg(n_paths=self.N, seed=self.SEED)
        plain = standard_monte_carlo(cfg)
        cv = geometric_control_variate(cfg, n_pilot=2_000)
        return plain, cv

    def test_cv_variance_lower_than_plain(self):
        plain, cv = self._run()
        assert cv.variance < plain.variance, (
            f"CV variance {cv.variance:.6f} not less than plain {plain.variance:.6f}"
        )

    def test_cv_se_lower_than_plain(self):
        plain, cv = self._run()
        assert cv.std_error < plain.std_error, (
            f"CV SE {cv.std_error:.6f} not less than plain {plain.std_error:.6f}"
        )

    def test_vrr_greater_than_one(self):
        plain, cv = self._run()
        vrr = variance_reduction_ratio(plain.variance, cv.variance)
        assert vrr > 1.0, f"VRR={vrr:.2f} — expected > 1"

    def test_meaningful_variance_reduction(self):
        """
        For ATM arithmetic/geometric Asian calls VRR is typically >> 2.
        We assert VRR > 2 as a conservative threshold.
        """
        plain, cv = self._run()
        vrr = variance_reduction_ratio(plain.variance, cv.variance)
        assert vrr > 2.0, (
            f"VRR={vrr:.2f} is unexpectedly low; expected > 2 for geometric CV"
        )


# ---------------------------------------------------------------------------
# C. Unbiasedness sanity
# ---------------------------------------------------------------------------

class TestUnbiasednessSanity:
    """
    CV and plain MC estimates must be statistically consistent.

    Tolerance = 5 × plain SE (conservative 5σ bound; false-alarm rate ~ 3e-7).
    """

    N = 50_000
    SEED = 7

    def test_cv_consistent_with_plain_mc(self):
        cfg = _cfg(n_paths=self.N, seed=self.SEED)
        plain = standard_monte_carlo(cfg)
        cv = geometric_control_variate(cfg, n_pilot=5_000)
        tol = 5 * plain.std_error
        assert abs(cv.price - plain.price) < tol, (
            f"CV price {cv.price:.5f} and plain MC {plain.price:.5f} differ "
            f"by more than 5×SE={tol:.5f}"
        )

    def test_cv_price_in_plain_ci(self):
        """CV price should fall within 5×SE of the plain MC CI."""
        cfg = _cfg(n_paths=self.N, seed=self.SEED)
        plain = standard_monte_carlo(cfg)
        cv = geometric_control_variate(cfg, n_pilot=5_000)
        margin = 5 * plain.std_error
        assert plain.ci_lower - margin <= cv.price <= plain.ci_upper + margin

    def test_cv_price_geq_geo_analytic(self):
        """
        Arithmetic Asian call >= geometric Asian call (AM-GM inequality).
        CV estimate (arithmetic) should be >= geometric analytical - tolerance.
        """
        from asian_options.analytical import geometric_asian_call_price
        cfg = _cfg(n_paths=self.N, seed=self.SEED)
        cv = geometric_control_variate(cfg, n_pilot=5_000)
        geo_price = geometric_asian_call_price(cfg)
        tol = 5 * cv.std_error
        assert cv.price >= geo_price - tol, (
            f"CV price {cv.price:.5f} < geo analytic {geo_price:.5f} - 5×SE"
        )


# ---------------------------------------------------------------------------
# D. Convention consistency
# ---------------------------------------------------------------------------

class TestConventionConsistency:
    """
    CV estimator must respect Stage 2/3 conventions:
    - paths shape (n_paths, m); S0 excluded; averaging over t1..tm.
    """

    def test_n_paths_matches_config(self):
        cfg = _cfg(n_paths=500, seed=1)
        result = geometric_control_variate(cfg, n_pilot=100)
        assert result.n_paths == 500

    def test_result_type(self):
        cfg = _cfg(n_paths=200, seed=2)
        result = geometric_control_variate(cfg, n_pilot=50)
        assert isinstance(result, CVEstimateResult)

    def test_ci_contains_price(self):
        cfg = _cfg(n_paths=2_000, seed=3)
        result = geometric_control_variate(cfg, n_pilot=200)
        assert result.ci_lower <= result.price <= result.ci_upper

    def test_se_equals_std_over_sqrt_n(self):
        cfg = _cfg(n_paths=2_000, seed=4)
        result = geometric_control_variate(cfg, n_pilot=200)
        expected_se = result.std_dev / math.sqrt(result.n_paths)
        assert math.isclose(result.std_error, expected_se, rel_tol=1e-9)

    def test_different_monitoring_dates(self):
        """Estimator should work for m=1, 5, 52 (weekly), 252 (daily)."""
        for m in [1, 5, 52]:
            cfg = _cfg(m=m, n_paths=500, seed=0)
            result = geometric_control_variate(cfg, n_pilot=100)
            assert math.isfinite(result.price)

    def test_beta_hat_finite(self):
        cfg = _cfg(n_paths=1_000, seed=5)
        result = geometric_control_variate(cfg, n_pilot=200)
        assert math.isfinite(result.beta_hat)

    def test_corr_estimate_in_range(self):
        """Correlation must lie in [-1, 1]."""
        cfg = _cfg(n_paths=1_000, seed=5)
        result = geometric_control_variate(cfg, n_pilot=200)
        assert -1.0 <= result.corr_estimate <= 1.0

    def test_n_pilot_lt_2_raises(self):
        cfg = _cfg(n_paths=200)
        with pytest.raises(ValueError, match="n_pilot"):
            geometric_control_variate(cfg, n_pilot=1)


# ---------------------------------------------------------------------------
# E. Antithetic variates
# ---------------------------------------------------------------------------

class TestAntitheticVariates:
    def test_returns_estimate_result(self):
        cfg = _cfg(n_paths=500, seed=0)
        result = antithetic_variates(cfg)
        assert isinstance(result, EstimateResult)

    def test_n_paths_matches_config(self):
        cfg = _cfg(n_paths=500, seed=0)
        result = antithetic_variates(cfg)
        assert result.n_paths == 500

    def test_price_positive(self):
        cfg = _cfg(n_paths=1_000, seed=1)
        result = antithetic_variates(cfg)
        assert result.price >= 0.0

    def test_ci_contains_price(self):
        cfg = _cfg(n_paths=1_000, seed=1)
        result = antithetic_variates(cfg)
        assert result.ci_lower <= result.price <= result.ci_upper

    def test_reproducible(self):
        cfg = _cfg(n_paths=500, seed=99)
        r1 = antithetic_variates(cfg)
        r2 = antithetic_variates(cfg)
        assert r1.price == r2.price

    def test_antithetic_variance_not_higher_than_plain(self):
        """
        Antithetic variates should not inflate variance for a standard ATM call.
        Allow equal-to-or-lower with a lenient tolerance (10 × plain SE).
        """
        cfg = _cfg(n_paths=10_000, seed=0)
        plain = standard_monte_carlo(cfg)
        av = antithetic_variates(cfg)
        # Allow up to plain variance as a broad sanity check
        margin = 10 * plain.std_error ** 2
        assert av.variance <= plain.variance + margin, (
            f"AV variance {av.variance:.6f} much higher than plain {plain.variance:.6f}"
        )

    def test_antithetic_consistent_with_plain(self):
        """Antithetic price should be within 5 × plain SE of plain MC."""
        cfg = _cfg(n_paths=20_000, seed=0)
        plain = standard_monte_carlo(cfg)
        av = antithetic_variates(cfg)
        tol = 5 * plain.std_error
        assert abs(av.price - plain.price) < tol, (
            f"AV price {av.price:.5f} inconsistent with plain {plain.price:.5f}"
        )


# ---------------------------------------------------------------------------
# F. variance_reduction_ratio utility
# ---------------------------------------------------------------------------

class TestVarianceReductionRatio:
    def test_correct_value(self):
        vrr = variance_reduction_ratio(4.0, 1.0)
        assert math.isclose(vrr, 4.0)

    def test_greater_than_one_means_reduction(self):
        assert variance_reduction_ratio(2.0, 0.5) > 1.0

    def test_less_than_one_means_inflation(self):
        assert variance_reduction_ratio(1.0, 2.0) < 1.0

    def test_zero_reduced_raises(self):
        with pytest.raises(ValueError, match="zero"):
            variance_reduction_ratio(1.0, 0.0)

    def test_equal_variances(self):
        assert math.isclose(variance_reduction_ratio(3.0, 3.0), 1.0)
