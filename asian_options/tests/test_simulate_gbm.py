"""
test_simulate_gbm.py
====================
Stage 2 tests for the risk-neutral GBM path simulator.

Conventions validated here
--------------------------
* ``simulate_paths`` returns an array of shape ``(n_paths, m)``.
* The initial spot S0 is **not** included in the output.
* Prices are float64.
* Each column j (0-indexed) corresponds to monitoring date t_{j+1} = (j+1)*dt.

Random seed
-----------
All stochastic tests use ``seed=42`` via ``ModelConfig`` or via explicit
``np.random.default_rng(42)`` calls so results are fully reproducible.

Tolerance justification
-----------------------
For a quantity X with MC standard error SE, we use a tolerance of
``5 * SE`` (roughly ±5σ, one-sided p < 3e-7 for a false alarm), which keeps
the test statistically meaningful while staying computationally cheap.
``n_paths=200_000`` is used for moment / pricing tests to keep SE small
enough that a 1 % relative error is detectable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.stats import norm  # type: ignore[import]

from asian_options.config import ModelConfig
from asian_options.simulate_gbm import simulate_paths
from asian_options.analytical import black_scholes_call

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

BASE_CFG = dict(S0=100.0, K=100.0, r=0.05, q=0.02, sigma=0.2, T=1.0, m=12,
                n_paths=10_000, seed=42)


def _cfg(**overrides) -> ModelConfig:
    return ModelConfig(**{**BASE_CFG, **overrides})


# ---------------------------------------------------------------------------
# 1. Output shape
# ---------------------------------------------------------------------------

class TestOutputShape:
    def test_shape_default(self):
        cfg = _cfg()
        paths = simulate_paths(cfg)
        assert paths.shape == (cfg.n_paths, cfg.m)

    def test_shape_various_m(self):
        for m in [1, 5, 52, 250]:
            cfg = _cfg(m=m, n_paths=100)
            assert simulate_paths(cfg).shape == (100, m)

    def test_shape_various_n_paths(self):
        for n in [1, 10, 1000]:
            cfg = _cfg(n_paths=n)
            assert simulate_paths(cfg).shape == (n, cfg.m)

    def test_dtype_float64(self):
        paths = simulate_paths(_cfg())
        assert paths.dtype == np.float64

    def test_s0_not_in_output(self):
        """Column 0 should be prices at t_1 = dt, not S0."""
        cfg = _cfg(sigma=0.0001, n_paths=1)  # near-zero vol → prices ≈ S0*exp(drift*dt)
        paths = simulate_paths(cfg)
        # With tiny vol, first column is approximately S0*exp((r-q)*dt), not S0 exactly
        expected_t1 = cfg.S0 * math.exp((cfg.r - cfg.q) * cfg.dt)
        assert paths.shape[1] == cfg.m
        assert abs(paths[0, 0] - expected_t1) < 0.05  # very small vol → close


# ---------------------------------------------------------------------------
# 2. Reproducibility and explicit shocks
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_output(self):
        cfg = _cfg(seed=7)
        p1 = simulate_paths(cfg)
        p2 = simulate_paths(cfg)
        np.testing.assert_array_equal(p1, p2)

    def test_different_seeds_different_output(self):
        p1 = simulate_paths(_cfg(seed=1))
        p2 = simulate_paths(_cfg(seed=2))
        assert not np.array_equal(p1, p2)

    def test_explicit_rng_reproducible(self):
        cfg = _cfg()
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        p1 = simulate_paths(cfg, rng=rng1)
        p2 = simulate_paths(cfg, rng=rng2)
        np.testing.assert_array_equal(p1, p2)

    def test_explicit_shocks_identical_output(self):
        cfg = _cfg()
        Z = np.random.default_rng(0).standard_normal((cfg.n_paths, cfg.m))
        p1 = simulate_paths(cfg, shocks=Z)
        p2 = simulate_paths(cfg, shocks=Z)
        np.testing.assert_array_equal(p1, p2)

    def test_shocks_override_rng(self):
        """Supplying shocks must ignore any rng argument."""
        cfg = _cfg()
        Z = np.random.default_rng(5).standard_normal((cfg.n_paths, cfg.m))
        rng_ignored = np.random.default_rng(999)
        p_shocks = simulate_paths(cfg, shocks=Z)
        p_rng_ignored = simulate_paths(cfg, shocks=Z, rng=rng_ignored)
        np.testing.assert_array_equal(p_shocks, p_rng_ignored)

    def test_wrong_shock_shape_raises(self):
        cfg = _cfg()
        bad_Z = np.zeros((cfg.n_paths + 1, cfg.m))
        with pytest.raises(ValueError, match="shape"):
            simulate_paths(cfg, shocks=bad_Z)

    def test_no_accidental_shock_reuse(self):
        """Each path must use its own distinct shocks (no broadcasting error)."""
        cfg = _cfg(n_paths=500, m=5, seed=0)
        paths = simulate_paths(cfg)
        # If shocks were shared across paths, paths would be identical
        assert not np.all(paths == paths[0])


# ---------------------------------------------------------------------------
# 3. Zero-volatility limiting behaviour
# ---------------------------------------------------------------------------

class TestZeroVolLimit:
    def test_deterministic_paths_at_zero_vol(self):
        """As sigma → 0, all paths collapse to the deterministic forward curve."""
        cfg = _cfg(sigma=1e-8, n_paths=100, seed=0)
        paths = simulate_paths(cfg)
        # Expected price at step j+1: S0 * exp((r - q) * (j+1) * dt)
        ts = np.arange(1, cfg.m + 1) * cfg.dt
        expected = cfg.S0 * np.exp((cfg.r - cfg.q) * ts)  # shape (m,)
        # Broadcast expected to (n_paths, m) for comparison
        expected_2d = np.broadcast_to(expected, paths.shape)
        np.testing.assert_allclose(paths, expected_2d, rtol=1e-4, atol=1e-4)

    def test_all_paths_equal_at_zero_vol(self):
        cfg = _cfg(sigma=1e-8, n_paths=50, seed=0)
        paths = simulate_paths(cfg)
        # All paths should match the first path within floating-point noise
        first = np.broadcast_to(paths[0], paths.shape)
        np.testing.assert_allclose(paths, first, rtol=1e-5, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Moment tests  (empirical vs analytical)
# ---------------------------------------------------------------------------

class TestMoments:
    """
    Under risk-neutral GBM: E[S_T] = S0 * exp((r - q) * T).
    Var[S_T] = S0^2 * exp(2*(r-q)*T) * (exp(sigma^2*T) - 1).

    With n_paths = 200_000 the MC SE of the mean is ≈ std(S_T) / sqrt(N),
    and we use a 5σ tolerance band.
    """

    N = 200_000

    def test_terminal_mean(self):
        cfg = _cfg(n_paths=self.N, seed=42)
        paths = simulate_paths(cfg)
        terminal = paths[:, -1]

        analytical_mean = cfg.S0 * math.exp((cfg.r - cfg.q) * cfg.T)
        # Variance of S_T
        var_ST = (cfg.S0 ** 2
                  * math.exp(2 * (cfg.r - cfg.q) * cfg.T)
                  * (math.exp(cfg.sigma ** 2 * cfg.T) - 1))
        se = math.sqrt(var_ST / self.N)
        tol = 5 * se

        assert abs(terminal.mean() - analytical_mean) < tol, (
            f"Terminal mean {terminal.mean():.4f} not within 5 SE of "
            f"analytical {analytical_mean:.4f} (tol={tol:.4f})"
        )

    def test_terminal_variance(self):
        cfg = _cfg(n_paths=self.N, seed=42)
        paths = simulate_paths(cfg)
        terminal = paths[:, -1]

        analytical_var = (cfg.S0 ** 2
                          * math.exp(2 * (cfg.r - cfg.q) * cfg.T)
                          * (math.exp(cfg.sigma ** 2 * cfg.T) - 1))
        # SE of sample variance ≈ Var * sqrt(2/(N-1)) for log-normal is approximate;
        # we use a generous 5 % relative tolerance which is statistically safe at N=200k.
        rtol = 0.05
        assert abs(terminal.var() - analytical_var) / analytical_var < rtol, (
            f"Terminal variance {terminal.var():.4f} not within 5% of "
            f"analytical {analytical_var:.4f}"
        )


# ---------------------------------------------------------------------------
# 5. European call price vs Black–Scholes
# ---------------------------------------------------------------------------

class TestEuropeanCallVsBS:
    """
    Price a European call using simulated terminal prices and compare to the
    Black-Scholes formula.

    MC estimator: C_MC = discount * mean(max(S_T - K, 0))

    SE of the MC estimator ≈ discount * std(max(S_T - K, 0)) / sqrt(N).
    We verify |C_MC - C_BS| < 5 * SE.
    """

    N = 200_000

    def test_atm_call(self):
        cfg = _cfg(n_paths=self.N, seed=42)
        self._check(cfg)

    def test_itm_call(self):
        cfg = _cfg(K=90.0, n_paths=self.N, seed=42)
        self._check(cfg)

    def test_otm_call(self):
        cfg = _cfg(K=110.0, n_paths=self.N, seed=42)
        self._check(cfg)

    def test_nonzero_dividend(self):
        cfg = _cfg(q=0.03, n_paths=self.N, seed=42)
        self._check(cfg)

    def _check(self, cfg: ModelConfig) -> None:
        paths = simulate_paths(cfg)
        terminal = paths[:, -1]
        payoffs = np.maximum(terminal - cfg.K, 0.0)
        mc_price = cfg.discount_factor * payoffs.mean()

        bs_price = black_scholes_call(cfg)
        se = cfg.discount_factor * payoffs.std() / math.sqrt(cfg.n_paths)
        tol = 5 * se

        assert abs(mc_price - bs_price) < tol, (
            f"MC call {mc_price:.5f} not within 5 SE ({tol:.5f}) of "
            f"Black-Scholes {bs_price:.5f}"
        )
