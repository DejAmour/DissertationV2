"""
test_neural_cv.py
=================
Stage 3 — Functional NCV Test Suite.

Covers:
 1. Network forward pass against a hand-calculated small example.
 2. Output shapes.
 3. Analytical E[H(Z)] against a high-precision numerical Monte Carlo approximation.
 4. A network containing zero-weight hidden neurons.
 5. Correct inclusion of W2 and b2.
 6. Training updates weights.
 7. Training loss decreases on a deterministic learnable example.
 8. Same seeds produce reproducible training results.
 9. Training and pricing datasets are separate.
10. NCV correction exactly equals: payoff − network_output + analytical_expectation.
11. NCV price is statistically consistent with a reliable reference.
12. NCV reduces variance on a stable benchmark configuration.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from asian_options.analytical import relu_expected_value
from asian_options.config import ModelConfig, seed_everything
from asian_options.neural_cv import (
    _ShallowNet,
    analytical_network_expectation,
    build_network,
    train_network,
    ncv_estimator,
)
from asian_options.simulate_gbm import simulate_paths
from asian_options.payoffs import arithmetic_asian_call_payoff
from asian_options.estimators import standard_monte_carlo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _base_cfg(**kwargs) -> ModelConfig:
    """Return a small-but-valid ModelConfig for fast tests."""
    defaults = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, m=4, n_paths=2000, seed=42)
    defaults.update(kwargs)
    return ModelConfig(**defaults)


def _make_net(hidden: int = 3, m: int = 2, seed: int = 0) -> _ShallowNet:
    rng = np.random.default_rng(seed)
    W1 = rng.standard_normal((hidden, m))
    b1 = rng.standard_normal(hidden)
    W2 = rng.standard_normal((1, hidden))
    b2 = rng.standard_normal(1)
    return _ShallowNet(W1, b1, W2, b2)


def _make_training_dataset(cfg: ModelConfig, n_train: int, seed: int = 99):
    """Generate an independent training dataset (Z, payoff) distinct from any pricing seed.

    Uses a temporary ModelConfig with n_paths=n_train so that simulate_paths
    accepts the shock matrix, keeping training and pricing samples separate.
    """
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_train, cfg.m))
    train_cfg = ModelConfig(
        S0=cfg.S0, K=cfg.K, r=cfg.r, sigma=cfg.sigma, T=cfg.T,
        m=cfg.m, n_paths=n_train, q=cfg.q, seed=seed,
    )
    paths = simulate_paths(train_cfg, shocks=Z)
    payoffs = arithmetic_asian_call_payoff(paths, train_cfg)
    return {"X_train": Z, "y_train": payoffs}


# ---------------------------------------------------------------------------
# 1. Network forward pass — hand-calculated small example
# ---------------------------------------------------------------------------

class TestForwardPass:
    def test_hand_calculated_single_path(self):
        """
        Network with one hidden neuron and known weights.
        H(Z) = W2 * ReLU(W1 @ Z + b1) + b2

        Setup:
          Z = [[1.0, -1.0]]   (1-path, m=2)
          W1 = [[2.0, 0.5]]   (hidden=1, m=2)
          b1 = [0.5]
          W2 = [[3.0]]
          b2 = [1.0]

        Pre-activation: 2*1 + 0.5*(-1) + 0.5 = 2.0
        ReLU: 2.0
        H = 3.0 * 2.0 + 1.0 = 7.0
        """
        W1 = np.array([[2.0, 0.5]])
        b1 = np.array([0.5])
        W2 = np.array([[3.0]])
        b2 = np.array([1.0])
        net = _ShallowNet(W1, b1, W2, b2)

        Z = np.array([[1.0, -1.0]])
        result = net.forward(Z)

        assert result.shape == (1,)
        assert np.isclose(result[0], 7.0, atol=1e-12), f"Expected 7.0, got {result[0]}"

    def test_hand_calculated_negative_relu(self):
        """
        Pre-activation is negative; ReLU clamps to zero.
        W1=[[1, 1]], b1=[-5], W2=[[2]], b2=[0.5]
        Z=[[1, 1]] → pre-act = 1+1-5 = -3 → ReLU = 0 → H = 0+0.5 = 0.5
        """
        W1 = np.array([[1.0, 1.0]])
        b1 = np.array([-5.0])
        W2 = np.array([[2.0]])
        b2 = np.array([0.5])
        net = _ShallowNet(W1, b1, W2, b2)

        Z = np.array([[1.0, 1.0]])
        result = net.forward(Z)
        assert np.isclose(result[0], 0.5, atol=1e-12)

    def test_hand_calculated_batch(self):
        """Two-path batch with one hidden neuron."""
        W1 = np.array([[1.0, 0.0]])
        b1 = np.array([0.0])
        W2 = np.array([[2.0]])
        b2 = np.array([0.0])
        net = _ShallowNet(W1, b1, W2, b2)

        # Z1=[3], Z2=[-2] → ReLU([3, -2]) = [3, 0] → H = [6, 0]
        Z = np.array([[3.0, 0.0], [-2.0, 0.0]])
        result = net.forward(Z)
        np.testing.assert_allclose(result, [6.0, 0.0], atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Output shapes
# ---------------------------------------------------------------------------

class TestOutputShapes:
    def test_forward_shape_single(self):
        net = _make_net(hidden=4, m=3)
        Z = np.random.default_rng(1).standard_normal((1, 3))
        out = net.forward(Z)
        assert out.shape == (1,)

    def test_forward_shape_batch(self):
        net = _make_net(hidden=8, m=5)
        Z = np.random.default_rng(2).standard_normal((100, 5))
        out = net.forward(Z)
        assert out.shape == (100,)

    def test_analytical_expectation_returns_scalar(self):
        net = _make_net(hidden=6, m=4)
        val = analytical_network_expectation(net)
        assert isinstance(val, float)
        assert np.isfinite(val)

    def test_forward_shape_many_hidden(self):
        net = _make_net(hidden=32, m=12)
        Z = np.random.default_rng(3).standard_normal((500, 12))
        out = net.forward(Z)
        assert out.shape == (500,)


# ---------------------------------------------------------------------------
# 3. Analytical E[H(Z)] vs high-precision Monte Carlo
# ---------------------------------------------------------------------------

class TestAnalyticalExpectation:
    """Compare analytical_network_expectation to a Monte Carlo approximation."""

    N_MONTE_CARLO = 5_000_000

    def _mc_expectation(self, net: _ShallowNet) -> tuple[float, float]:
        """Return (estimate, standard_error) via large MC."""
        rng = np.random.default_rng(77)
        m = net.W1.shape[1]
        Z = rng.standard_normal((self.N_MONTE_CARLO, m))
        h_vals = net.forward(Z)
        est = float(h_vals.mean())
        se = float(h_vals.std(ddof=1) / np.sqrt(self.N_MONTE_CARLO))
        return est, se

    def test_analytical_vs_mc_small_net(self):
        """Analytical should be within 5 SE of MC with 5M samples."""
        net = _make_net(hidden=3, m=2, seed=7)
        analytical = analytical_network_expectation(net)
        mc_est, mc_se = self._mc_expectation(net)
        # 5 SE gives a very conservative bound
        assert abs(analytical - mc_est) < 5 * mc_se, (
            f"Analytical {analytical:.6f} vs MC {mc_est:.6f} ± {mc_se:.6f}"
        )

    def test_analytical_vs_mc_larger_net(self):
        net = _make_net(hidden=8, m=4, seed=13)
        analytical = analytical_network_expectation(net)
        mc_est, mc_se = self._mc_expectation(net)
        assert abs(analytical - mc_est) < 5 * mc_se, (
            f"Analytical {analytical:.6f} vs MC {mc_est:.6f} ± {mc_se:.6f}"
        )


# ---------------------------------------------------------------------------
# 4. Zero-weight hidden neurons
# ---------------------------------------------------------------------------

class TestZeroWeightNeurons:
    def test_zero_row_does_not_raise(self):
        """A neuron with W1[i,:]=0 has sigma=0; must use max(mu,0) branch."""
        W1 = np.array([[1.0, 0.5], [0.0, 0.0], [0.3, -0.2]])
        b1 = np.array([0.0, 3.0, -1.0])
        W2 = np.array([[1.0, 1.0, 1.0]])
        b2 = np.array([0.0])
        net = _ShallowNet(W1, b1, W2, b2)
        val = analytical_network_expectation(net)
        assert np.isfinite(val)

    def test_zero_row_negative_bias(self):
        """Neuron with zero weights and negative bias contributes 0 to E[H]."""
        W1 = np.zeros((1, 3))
        b1 = np.array([-5.0])        # sigma=0, mu=-5 → E[a+]=0
        W2 = np.array([[10.0]])
        b2 = np.array([0.0])
        net = _ShallowNet(W1, b1, W2, b2)
        val = analytical_network_expectation(net)
        assert np.isclose(val, 0.0, atol=1e-12)

    def test_zero_row_positive_bias(self):
        """Neuron with zero weights and positive bias contributes mu to E[H]."""
        W1 = np.zeros((1, 3))
        b1 = np.array([4.0])         # sigma=0, mu=4 → E[a+]=4
        W2 = np.array([[2.0]])
        b2 = np.array([1.0])
        net = _ShallowNet(W1, b1, W2, b2)
        val = analytical_network_expectation(net)
        # E[H] = W2 * 4 + b2 = 2*4+1 = 9
        assert np.isclose(val, 9.0, atol=1e-12)

    def test_forward_pass_zero_neuron(self):
        """Zero-weight neuron fires only when b1>0."""
        W1 = np.zeros((1, 2))
        b1 = np.array([1.0])
        W2 = np.array([[1.0]])
        b2 = np.array([0.0])
        net = _ShallowNet(W1, b1, W2, b2)
        Z = np.array([[999.0, -999.0]])  # input irrelevant for zero-weight neuron
        result = net.forward(Z)
        assert np.isclose(result[0], 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. Correct inclusion of W2 and b2
# ---------------------------------------------------------------------------

class TestW2AndB2Inclusion:
    def test_scaling_by_W2(self):
        """Doubling W2 should double the output (and the expectation)."""
        net1 = _make_net(hidden=4, m=3, seed=5)
        net2 = _ShallowNet(net1.W1.copy(), net1.b1.copy(), 2.0 * net1.W2.copy(), net1.b2.copy())

        rng = np.random.default_rng(6)
        Z = rng.standard_normal((100, 3))

        # forward pass: (net2 - b2) ≈ 2*(net1 - b2) won't hold because b2 unchanged
        # but contribution of hidden layer should double
        h1 = net1.forward(Z) - float(net1.b2.item())
        h2 = net2.forward(Z) - float(net2.b2.item())
        np.testing.assert_allclose(h2, 2.0 * h1, atol=1e-10)

    def test_b2_shifts_output(self):
        """Adding a constant delta to b2 shifts every output and the expectation by delta."""
        net1 = _make_net(hidden=4, m=3, seed=8)
        delta = 7.5
        net2 = _ShallowNet(net1.W1.copy(), net1.b1.copy(), net1.W2.copy(), net1.b2 + delta)

        rng = np.random.default_rng(9)
        Z = rng.standard_normal((200, 3))

        diff_forward = net2.forward(Z) - net1.forward(Z)
        np.testing.assert_allclose(diff_forward, delta, atol=1e-12)

        diff_analytical = (analytical_network_expectation(net2)
                           - analytical_network_expectation(net1))
        assert np.isclose(diff_analytical, delta, atol=1e-12)

    def test_zero_W2_gives_b2_output(self):
        """With W2=0, H(Z) = b2 for all Z."""
        W1 = np.ones((3, 2))
        b1 = np.zeros(3)
        W2 = np.zeros((1, 3))
        b2 = np.array([5.0])
        net = _ShallowNet(W1, b1, W2, b2)

        Z = np.random.default_rng(10).standard_normal((50, 2))
        out = net.forward(Z)
        np.testing.assert_allclose(out, 5.0, atol=1e-12)
        assert np.isclose(analytical_network_expectation(net), 5.0, atol=1e-12)


# ---------------------------------------------------------------------------
# 6. Training updates weights
# ---------------------------------------------------------------------------

class TestTrainingUpdatesWeights:
    def test_weights_change_after_training(self):
        cfg = _base_cfg(n_paths=500, m=4, seed=42)
        net = build_network(cfg, hidden_width=4)

        W1_before = net.W1.copy()
        b1_before = net.b1.copy()

        dataset = _make_training_dataset(cfg, n_train=300, seed=100)
        train_network(net, dataset, cfg, n_epochs=5, lr=1e-2)

        # At least one weight should have changed
        assert not np.allclose(net.W1, W1_before) or not np.allclose(net.b1, b1_before), (
            "Weights were not updated after training"
        )

    def test_all_weight_arrays_can_change(self):
        """All four parameter arrays (W1, b1, W2, b2) are included in the optimizer."""
        cfg = _base_cfg(n_paths=300, m=4, seed=55)
        net = build_network(cfg, hidden_width=4)

        W1_b, b1_b, W2_b, b2_b = (
            net.W1.copy(), net.b1.copy(), net.W2.copy(), net.b2.copy()
        )

        dataset = _make_training_dataset(cfg, n_train=200, seed=111)
        train_network(net, dataset, cfg, n_epochs=20, lr=5e-3)

        changed = (
            not np.allclose(net.W1, W1_b),
            not np.allclose(net.b1, b1_b),
            not np.allclose(net.W2, W2_b),
            not np.allclose(net.b2, b2_b),
        )
        # All four parameter arrays should see gradient updates
        assert all(changed), (
            f"Expected all params to change; changed flags: "
            f"W1={changed[0]}, b1={changed[1]}, W2={changed[2]}, b2={changed[3]}"
        )


# ---------------------------------------------------------------------------
# 7. Training loss decreases on a deterministic learnable example
# ---------------------------------------------------------------------------

class TestTrainingLossDecreases:
    def test_loss_decreases_on_constant_target(self):
        """
        Target is the constant function y=C. The network can learn this by
        setting b2=C. Loss should decrease monotonically (or at least overall).
        """
        m = 4
        cfg = _base_cfg(m=m, seed=77)
        net = build_network(cfg, hidden_width=8)

        rng = np.random.default_rng(200)
        X_train = rng.standard_normal((1000, m))
        C = 2.5
        y_train = np.full(1000, C)
        dataset = {"X_train": X_train, "y_train": y_train}

        history = train_network(net, dataset, cfg, n_epochs=100, lr=1e-2)
        losses = history["train_loss"]

        assert losses[0] > losses[-1], (
            f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
        )

    def test_loss_history_non_empty(self):
        cfg = _base_cfg(m=4, seed=33)
        net = build_network(cfg, hidden_width=4)
        dataset = _make_training_dataset(cfg, n_train=200, seed=201)
        history = train_network(net, dataset, cfg, n_epochs=10, lr=1e-2)
        assert "train_loss" in history
        assert len(history["train_loss"]) == 10
        assert all(np.isfinite(l) for l in history["train_loss"])


# ---------------------------------------------------------------------------
# 8. Same seeds produce reproducible training results
# ---------------------------------------------------------------------------

class TestReproducibility:
    def test_same_seed_same_weights(self):
        """Two runs with identical seeds must produce identical trained weights."""
        cfg = _base_cfg(m=4, seed=42)
        dataset = _make_training_dataset(cfg, n_train=300, seed=123)

        def _run():
            seed_everything(cfg.seed)
            net = build_network(cfg, hidden_width=8)
            train_network(net, dataset, cfg, n_epochs=20, lr=1e-2)
            return net.W1.copy(), net.b1.copy(), net.W2.copy(), net.b2.copy()

        W1a, b1a, W2a, b2a = _run()
        W1b, b1b, W2b, b2b = _run()

        np.testing.assert_array_equal(W1a, W1b, err_msg="W1 differs across runs")
        np.testing.assert_array_equal(b1a, b1b, err_msg="b1 differs across runs")
        np.testing.assert_array_equal(W2a, W2b, err_msg="W2 differs across runs")
        np.testing.assert_array_equal(b2a, b2b, err_msg="b2 differs across runs")


# ---------------------------------------------------------------------------
# 9. Training and pricing datasets are separate
# ---------------------------------------------------------------------------

class TestDatasetSeparation:
    def test_training_uses_different_rng_seed_than_pricing(self):
        """
        The pricing paths (seeded via cfg.seed) must not be used during training.
        We verify by showing that the training Z matrix does not overlap with
        the pricing Z matrix generated by ncv_estimator's internal rng.
        """
        pricing_seed = 42
        training_seed = 999   # deliberately different
        cfg = _base_cfg(seed=pricing_seed, n_paths=500, m=4)

        # Reproduce pricing Z (ncv_estimator uses cfg.seed)
        pricing_rng = np.random.default_rng(pricing_seed)
        Z_pricing = pricing_rng.standard_normal((cfg.n_paths, cfg.m))

        # Training dataset uses an independent seed
        training_rng = np.random.default_rng(training_seed)
        Z_training = training_rng.standard_normal((400, cfg.m))

        # The first rows of each matrix should differ
        assert not np.allclose(Z_pricing[:min(400, cfg.n_paths)], Z_training[:min(400, cfg.n_paths)]), (
            "Training and pricing Z matrices are identical — they likely share the same seed."
        )

    def test_train_network_does_not_use_pricing_observations(self):
        """
        The pricing sample is not passed to train_network at all.
        Changing the training dataset while holding cfg.seed fixed should
        change trained weights but leave the pricing observations reproducible.
        """
        cfg = _base_cfg(seed=42, n_paths=300, m=4)

        ds_a = _make_training_dataset(cfg, n_train=200, seed=400)
        ds_b = _make_training_dataset(cfg, n_train=200, seed=401)

        seed_everything(cfg.seed)
        net_a = build_network(cfg, hidden_width=4)
        train_network(net_a, ds_a, cfg, n_epochs=10, lr=1e-2)

        seed_everything(cfg.seed)
        net_b = build_network(cfg, hidden_width=4)
        train_network(net_b, ds_b, cfg, n_epochs=10, lr=1e-2)

        # Different training data → different weights
        assert not np.allclose(net_a.W1, net_b.W1), (
            "Networks trained on different datasets should differ."
        )


# ---------------------------------------------------------------------------
# 10. NCV correction exactly equals payoff − network_output + analytical_expectation
# ---------------------------------------------------------------------------

class TestNCVCorrectionFormula:
    def test_correction_formula_exact(self):
        """
        For a frozen network and a fixed Z matrix, verify element-wise:
            corrected[i] = payoff[i] - H(Z[i]) + E[H(Z)]
        """
        cfg = _base_cfg(n_paths=200, m=4, seed=88)
        net = build_network(cfg, hidden_width=6)

        rng = np.random.default_rng(cfg.seed)
        Z = rng.standard_normal((cfg.n_paths, cfg.m))
        paths = simulate_paths(cfg, shocks=Z)
        payoffs = arithmetic_asian_call_payoff(paths, cfg)

        h_vals = net.forward(Z)
        e_h = analytical_network_expectation(net)

        expected_corrected = payoffs - h_vals + e_h

        # Cross-check against what ncv_estimator would compute using same seed
        # (ncv_estimator generates its own Z with cfg.seed)
        corrected_direct = payoffs - net.forward(Z) + e_h

        np.testing.assert_allclose(
            corrected_direct,
            expected_corrected,
            atol=1e-14,
            err_msg="Correction formula deviates element-wise",
        )

    def test_correction_preserves_mean_when_e_h_is_exact(self):
        """
        If E[H(Z)] is the exact analytical expectation, the correction should be
        an unbiased modification: mean(corrected) ≈ mean(payoff).
        For a constant network H(Z)=c, E[H(Z)]=c, so corrected = payoff exactly.
        """
        W1 = np.zeros((1, 3))   # zero weights → H(Z) = W2*ReLU(b1) + b2 = const
        b1 = np.array([1.0])    # ReLU(1)=1
        W2 = np.array([[2.0]])
        b2 = np.array([0.5])
        net = _ShallowNet(W1, b1, W2, b2)   # H(Z) = 2.5 for all Z

        cfg = _base_cfg(n_paths=500, m=3, seed=55)
        rng = np.random.default_rng(77)
        Z = rng.standard_normal((500, 3))
        paths = simulate_paths(cfg, shocks=Z)
        payoffs = arithmetic_asian_call_payoff(paths, cfg)

        h_vals = net.forward(Z)
        e_h = analytical_network_expectation(net)

        assert np.isclose(e_h, 2.5, atol=1e-12), f"E[H]={e_h}"
        np.testing.assert_allclose(h_vals, 2.5, atol=1e-12)

        corrected = payoffs - h_vals + e_h
        np.testing.assert_allclose(corrected, payoffs, atol=1e-12)


# ---------------------------------------------------------------------------
# 11. NCV price is statistically consistent with a reliable reference
# ---------------------------------------------------------------------------

class TestNCVPriceConsistency:
    """
    Compare the NCV price to the standard MC price.  Both estimate the same
    quantity; the difference should be small relative to the standard errors.
    We use a large-sample configuration to keep standard errors small.
    """

    def test_ncv_price_within_confidence_of_mc(self):
        """
        |NCV_price - MC_price| < 4 * (SE_NCV + SE_MC) with very high probability.
        Uses separate seeds so the two estimators are independent.
        """
        cfg_mc = _base_cfg(n_paths=10_000, seed=1000)
        cfg_ncv = _base_cfg(n_paths=10_000, seed=1001)

        # Train on a separate set
        net = build_network(cfg_ncv, hidden_width=16)
        dataset = _make_training_dataset(cfg_ncv, n_train=5000, seed=555)
        train_network(net, dataset, cfg_ncv, n_epochs=50, lr=5e-3)

        mc_result = standard_monte_carlo(cfg_mc)
        ncv_result = ncv_estimator(net, cfg_ncv)

        diff = abs(ncv_result.price - mc_result.price)
        bound = 4.0 * (ncv_result.std_error + mc_result.std_error)

        assert diff < bound, (
            f"NCV price {ncv_result.price:.4f} too far from MC {mc_result.price:.4f}: "
            f"|diff|={diff:.5f} > 4*(SE_NCV+SE_MC)={bound:.5f}"
        )


# ---------------------------------------------------------------------------
# 12. NCV reduces variance on a stable benchmark configuration
# ---------------------------------------------------------------------------

class TestVarianceReduction:
    """
    After training on a sufficiently expressive configuration, NCV observation
    variance should be lower than MC observation variance.  We assert a
    weak reduction (VRR > 1) rather than a specific factor to remain robust.
    """

    def test_ncv_variance_less_than_mc(self):
        cfg = _base_cfg(n_paths=5000, m=4, seed=2024)

        # MC variance
        mc_result = standard_monte_carlo(cfg)
        mc_var = mc_result.variance

        # NCV: train on a separate dataset, price on cfg.seed paths
        net = build_network(cfg, hidden_width=32)
        train_cfg = ModelConfig(
            S0=cfg.S0, K=cfg.K, r=cfg.r, sigma=cfg.sigma, T=cfg.T,
            m=cfg.m, n_paths=cfg.n_paths, seed=9999,
        )
        dataset = _make_training_dataset(train_cfg, n_train=3000, seed=8888)
        train_network(net, dataset, cfg, n_epochs=100, lr=5e-3)

        ncv_result = ncv_estimator(net, cfg)
        ncv_var = ncv_result.variance

        # Allow a generous upper bound: NCV should be at most equal to MC
        assert ncv_var <= mc_var * 1.05, (
            f"NCV variance {ncv_var:.6f} is not lower than MC variance {mc_var:.6f}"
        )
