"""
test_neural_cv_shapes.py
========================
Stage 2 tests for network shape validation in neural_cv.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from asian_options.neural_cv import _ShallowNet, analytical_network_expectation, build_network
from asian_options.config import ModelConfig


def _make_net(hidden=4, m=3):
    rng = np.random.default_rng(0)
    W1 = rng.standard_normal((hidden, m))
    b1 = rng.standard_normal(hidden)
    W2 = rng.standard_normal((1, hidden))
    b2 = rng.standard_normal(1)
    return _ShallowNet(W1, b1, W2, b2)


class TestNetworkShapeValidation:
    def test_bad_b1_shape_raises(self):
        net = _make_net()
        net.b1 = np.zeros(net.W1.shape[0] + 1)  # wrong size
        with pytest.raises(ValueError, match="b1 shape"):
            analytical_network_expectation(net)

    def test_bad_W2_shape_raises(self):
        net = _make_net()
        net.W2 = np.ones((2, net.W1.shape[0]))  # wrong rows
        with pytest.raises(ValueError, match="W2 shape"):
            analytical_network_expectation(net)

    def test_bad_b2_shape_raises(self):
        net = _make_net()
        net.b2 = np.zeros(2)  # should be (1,)
        with pytest.raises(ValueError, match="b2 shape"):
            analytical_network_expectation(net)

    def test_nonfinite_W1_raises(self):
        net = _make_net()
        net.W1[0, 0] = float("inf")
        with pytest.raises(ValueError, match="W1"):
            analytical_network_expectation(net)

    def test_nonfinite_b1_raises(self):
        net = _make_net()
        net.b1[0] = float("nan")
        with pytest.raises(ValueError, match="b1"):
            analytical_network_expectation(net)

    def test_valid_network_returns_float(self):
        net = _make_net()
        result = analytical_network_expectation(net)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_zero_W1_row_handled(self):
        """A hidden neuron with zero weights has sigma=0; must not raise."""
        net = _make_net(hidden=3, m=4)
        net.W1[1, :] = 0.0  # zero-weight neuron
        net.b1[1] = 2.0     # mu=2, should contribute max(2,0)*W2[0,1]
        result = analytical_network_expectation(net)
        assert np.isfinite(result)

    def test_build_network_correct_shapes(self):
        cfg = ModelConfig(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, m=12, n_paths=1000, seed=0)
        net = build_network(cfg, hidden_width=16)
        assert net.W1.shape == (16, 12)
        assert net.b1.shape == (16,)
        assert net.W2.shape == (1, 16)
        assert net.b2.shape == (1,)
