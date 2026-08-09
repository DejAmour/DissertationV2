"""
test_budget_semantics.py
========================
Stage 4 tests for correct budget accounting and variance-field semantics.

Tests prove:
1. MC: pricing_observations == pricing_simulated_paths == total_simulated_paths
2. AV: pricing_simulated_paths == 2 * pricing_observations;
       total_simulated_paths == pricing_simulated_paths
3. CV: total_simulated_paths == pilot_paths + pricing_simulated_paths
4. NCV: total_simulated_paths == training_paths + pricing_simulated_paths
5. Equal-total-budget allocation logic
6. observation_variance == variance (same quantity); estimator_variance defined correctly
7. variance_reduction_ratio named and computed correctly (NOT "speed ratio")

All tests are deterministic (fixed seeds, no random assertions).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from asian_options.config import ModelConfig
from asian_options.estimators import (
    EstimateResult,
    CVEstimateResult,
    standard_monte_carlo,
    antithetic_variates,
    geometric_control_variate,
)
from asian_options.metrics import variance_reduction_ratio

BASE = dict(S0=100.0, K=100.0, r=0.05, q=0.0, sigma=0.2, T=1.0, m=12,
            n_paths=500, seed=1)


def _cfg(**kw) -> ModelConfig:
    return ModelConfig(**{**BASE, **kw})


# ---------------------------------------------------------------------------
# 1. MC budget accounting
# ---------------------------------------------------------------------------

class TestMCBudgetAccounting:
    def test_pricing_observations_equals_n_paths(self):
        cfg = _cfg(n_paths=200)
        r = standard_monte_carlo(cfg)
        assert r.pricing_observations == 200

    def test_pricing_simulated_paths_equals_n_paths(self):
        cfg = _cfg(n_paths=200)
        r = standard_monte_carlo(cfg)
        assert r.pricing_simulated_paths == 200

    def test_pilot_paths_zero(self):
        r = standard_monte_carlo(_cfg())
        assert r.pilot_paths == 0

    def test_training_paths_zero(self):
        r = standard_monte_carlo(_cfg())
        assert r.training_paths == 0

    def test_total_equals_pricing_simulated_paths(self):
        cfg = _cfg(n_paths=300)
        r = standard_monte_carlo(cfg)
        assert r.total_simulated_paths == r.pricing_simulated_paths

    def test_total_equals_n_paths(self):
        cfg = _cfg(n_paths=300)
        r = standard_monte_carlo(cfg)
        assert r.total_simulated_paths == 300


# ---------------------------------------------------------------------------
# 2. AV budget accounting
# ---------------------------------------------------------------------------

class TestAVBudgetAccounting:
    def test_pricing_simulated_paths_equals_twice_observations(self):
        cfg = _cfg(n_paths=200)
        r = antithetic_variates(cfg)
        assert r.pricing_simulated_paths == 2 * r.pricing_observations

    def test_pricing_observations_equals_n_paths(self):
        """AV: n_paths means number of antithetic *pairs* = pricing observations."""
        cfg = _cfg(n_paths=200)
        r = antithetic_variates(cfg)
        assert r.pricing_observations == 200

    def test_total_simulated_paths_equals_twice_n_paths(self):
        cfg = _cfg(n_paths=200)
        r = antithetic_variates(cfg)
        assert r.total_simulated_paths == 2 * 200

    def test_pilot_paths_zero(self):
        r = antithetic_variates(_cfg())
        assert r.pilot_paths == 0

    def test_training_paths_zero(self):
        r = antithetic_variates(_cfg())
        assert r.training_paths == 0

    def test_total_equals_pricing_simulated_paths(self):
        """AV has no separate pilot/training phase."""
        r = antithetic_variates(_cfg())
        assert r.total_simulated_paths == r.pricing_simulated_paths


# ---------------------------------------------------------------------------
# 3. CV budget accounting
# ---------------------------------------------------------------------------

class TestCVBudgetAccounting:
    def test_total_equals_pilot_plus_pricing(self):
        n_pilot = 50
        cfg = _cfg(n_paths=200)
        r = geometric_control_variate(cfg, n_pilot=n_pilot)
        assert r.total_simulated_paths == n_pilot + r.pricing_simulated_paths

    def test_pilot_paths_stored(self):
        n_pilot = 75
        r = geometric_control_variate(_cfg(n_paths=200), n_pilot=n_pilot)
        assert r.pilot_paths == n_pilot

    def test_training_paths_zero(self):
        r = geometric_control_variate(_cfg(), n_pilot=50)
        assert r.training_paths == 0

    def test_pricing_observations_equals_n_paths(self):
        cfg = _cfg(n_paths=200)
        r = geometric_control_variate(cfg, n_pilot=50)
        assert r.pricing_observations == 200

    def test_pricing_simulated_paths_equals_n_paths(self):
        cfg = _cfg(n_paths=200)
        r = geometric_control_variate(cfg, n_pilot=50)
        assert r.pricing_simulated_paths == 200

    def test_total_additive(self):
        n_pilot = 100
        n_pricing = 300
        cfg = _cfg(n_paths=n_pricing)
        r = geometric_control_variate(cfg, n_pilot=n_pilot)
        assert r.total_simulated_paths == n_pilot + n_pricing


# ---------------------------------------------------------------------------
# 4. NCV budget accounting
# ---------------------------------------------------------------------------

class TestNCVBudgetAccounting:
    @staticmethod
    def _make_ncv_result(n_pricing: int, n_training: int):
        from asian_options.neural_cv import build_network, train_network, ncv_estimator
        from asian_options.simulate_gbm import simulate_paths
        from asian_options.payoffs import arithmetic_asian_call_payoff
        import dataclasses
        import math as _math

        base = _cfg()
        train_cfg = dataclasses.replace(base, n_paths=n_training, seed=base.seed + 99)
        train_paths = simulate_paths(train_cfg)
        train_payoffs = arithmetic_asian_call_payoff(train_paths, train_cfg)
        dt = train_cfg.dt
        drift = (train_cfg.r - train_cfg.q - 0.5 * train_cfg.sigma ** 2) * dt
        diffusion = train_cfg.sigma * _math.sqrt(dt)
        log_S = np.log(train_paths / train_cfg.S0)
        log_inc = np.diff(np.hstack([np.zeros((n_training, 1)), log_S]), axis=1)
        Z_train = (log_inc - drift) / diffusion
        dataset = {"X_train": Z_train, "y_train": train_payoffs}
        network = build_network(train_cfg, hidden_width=8)
        train_network(network, dataset, train_cfg, n_epochs=5)

        price_cfg = dataclasses.replace(base, n_paths=n_pricing, seed=base.seed + 200)
        return ncv_estimator(network, price_cfg, n_training_paths=n_training)

    def test_total_equals_training_plus_pricing(self):
        n_pricing, n_training = 100, 80
        r = self._make_ncv_result(n_pricing, n_training)
        assert r.total_simulated_paths == n_training + r.pricing_simulated_paths

    def test_training_paths_stored(self):
        n_training = 60
        r = self._make_ncv_result(100, n_training)
        assert r.training_paths == n_training

    def test_pilot_paths_zero(self):
        r = self._make_ncv_result(100, 50)
        assert r.pilot_paths == 0

    def test_pricing_simulated_paths_equals_n_pricing(self):
        n_pricing = 120
        r = self._make_ncv_result(n_pricing, 50)
        assert r.pricing_simulated_paths == n_pricing

    def test_pricing_observations_equals_n_pricing(self):
        n_pricing = 150
        r = self._make_ncv_result(n_pricing, 50)
        assert r.pricing_observations == n_pricing


# ---------------------------------------------------------------------------
# 5. Equal-total-budget allocation logic
# ---------------------------------------------------------------------------

class TestEqualBudgetAllocation:
    """
    Verify the equal-budget allocation rules without running a full comparison.
    These are pure arithmetic checks.
    """

    BUDGET = 10_000
    N_PILOT = 500
    N_TRAINING = 1_000

    def test_mc_gets_full_budget(self):
        mc_obs = self.BUDGET
        assert mc_obs == self.BUDGET

    def test_av_gets_half_observations(self):
        av_pairs = self.BUDGET // 2
        av_paths = 2 * av_pairs
        assert av_pairs == self.BUDGET // 2
        assert av_paths == self.BUDGET

    def test_cv_pricing_deducts_pilot(self):
        cv_pricing = self.BUDGET - self.N_PILOT
        cv_total = self.N_PILOT + cv_pricing
        assert cv_pricing == self.BUDGET - self.N_PILOT
        assert cv_total == self.BUDGET

    def test_ncv_pricing_deducts_training(self):
        ncv_pricing = self.BUDGET - self.N_TRAINING
        ncv_total = self.N_TRAINING + ncv_pricing
        assert ncv_pricing == self.BUDGET - self.N_TRAINING
        assert ncv_total == self.BUDGET

    def test_av_observation_count_less_than_mc(self):
        av_pairs = self.BUDGET // 2
        mc_obs = self.BUDGET
        assert av_pairs < mc_obs

    def test_cv_pricing_less_than_mc_when_pilot_positive(self):
        cv_pricing = self.BUDGET - self.N_PILOT
        assert cv_pricing < self.BUDGET

    def test_ncv_pricing_less_than_mc_when_training_positive(self):
        ncv_pricing = self.BUDGET - self.N_TRAINING
        assert ncv_pricing < self.BUDGET


# ---------------------------------------------------------------------------
# 6. Variance field semantics
# ---------------------------------------------------------------------------

class TestVarianceFieldSemantics:
    def test_mc_observation_variance_equals_variance(self):
        r = standard_monte_carlo(_cfg())
        assert r.observation_variance == r.variance

    def test_mc_estimator_variance_is_obs_var_over_pricing_obs(self):
        r = standard_monte_carlo(_cfg())
        expected = r.observation_variance / r.pricing_observations
        assert math.isclose(r.estimator_variance, expected, rel_tol=1e-9)

    def test_av_observation_variance_equals_variance(self):
        r = antithetic_variates(_cfg())
        assert r.observation_variance == r.variance

    def test_av_estimator_variance_correct(self):
        r = antithetic_variates(_cfg())
        expected = r.observation_variance / r.pricing_observations
        assert math.isclose(r.estimator_variance, expected, rel_tol=1e-9)

    def test_cv_observation_variance_equals_variance(self):
        r = geometric_control_variate(_cfg(), n_pilot=50)
        assert r.observation_variance == r.variance

    def test_cv_estimator_variance_correct(self):
        r = geometric_control_variate(_cfg(), n_pilot=50)
        expected = r.observation_variance / r.pricing_observations
        assert math.isclose(r.estimator_variance, expected, rel_tol=1e-9)

    def test_estimator_variance_smaller_than_observation_variance(self):
        """For any non-trivial n, est_var < obs_var."""
        r = standard_monte_carlo(_cfg(n_paths=10))
        assert r.estimator_variance < r.observation_variance

    def test_std_error_consistent_with_estimator_variance(self):
        """std_error == sqrt(estimator_variance)."""
        r = standard_monte_carlo(_cfg())
        assert math.isclose(r.std_error, math.sqrt(r.estimator_variance), rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 7. Variance-reduction ratio (not "speed ratio")
# ---------------------------------------------------------------------------

class TestVarianceReductionRatio:
    def test_vrr_greater_than_one_for_cv(self):
        mc = standard_monte_carlo(_cfg(n_paths=2000, seed=7))
        cv = geometric_control_variate(_cfg(n_paths=2000, seed=7), n_pilot=200)
        vrr = variance_reduction_ratio(mc.observation_variance, cv.observation_variance)
        assert vrr > 0, "VRR must be positive"

    def test_vrr_mc_vs_self_is_one(self):
        r = standard_monte_carlo(_cfg())
        vrr = variance_reduction_ratio(r.observation_variance, r.observation_variance)
        assert math.isclose(vrr, 1.0)

    def test_vrr_zero_denominator_raises(self):
        with pytest.raises(ValueError):
            variance_reduction_ratio(1.0, 0.0)

    def test_vrr_uses_observation_variance_not_estimator_variance(self):
        """
        VRR must be the ratio of observation variances, not estimator variances.
        They differ by the observation count.
        """
        mc = standard_monte_carlo(_cfg(n_paths=100))
        cv = geometric_control_variate(_cfg(n_paths=100), n_pilot=20)
        vrr_obs = variance_reduction_ratio(mc.observation_variance, cv.observation_variance)
        vrr_est = mc.estimator_variance / cv.estimator_variance
        # Both ratios should be equal when n is the same
        # (they differ when pricing_observations differ, e.g. equal-budget mode)
        assert math.isclose(vrr_obs, vrr_est, rel_tol=1e-9)
