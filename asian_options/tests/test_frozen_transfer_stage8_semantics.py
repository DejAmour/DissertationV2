from __future__ import annotations

import math

import numpy as np

from asian_options.contracts import make_contract_cfg
from asian_options.frozen_transfer import compute_network_hash, ncv_transfer_beta, ncv_transfer_beta1


class _DummyFrozenNetwork:
    def __init__(self) -> None:
        self.W1 = np.array([[1.0]], dtype=np.float64)
        self.b1 = np.array([0.0], dtype=np.float64)
        self.W2 = np.array([[1.0]], dtype=np.float64)
        self.b2 = np.array([0.0], dtype=np.float64)

    def forward(self, Z: np.ndarray) -> np.ndarray:
        return Z[:, 0]


def _patch_simulation_and_payoff(monkeypatch):
    from asian_options import payoffs, simulate_gbm

    monkeypatch.setattr(simulate_gbm, "simulate_paths", lambda cfg, shocks=None: shocks)
    monkeypatch.setattr(payoffs, "arithmetic_asian_call_payoff", lambda paths, cfg: 2.0 * paths[:, 0] + paths[:, 1])


def test_transfer_beta_one_reports_oracle_optimal_not_observed(monkeypatch):
    _patch_simulation_and_payoff(monkeypatch)
    net = _DummyFrozenNetwork()
    frozen_hash = compute_network_hash(net)
    cfg = make_contract_cfg("strike_low", n_paths=5000, seed=1)
    out = ncv_transfer_beta1(
        frozen_network=net,
        e_h0=0.0,
        frozen_hash=frozen_hash,
        target_cfg=cfg,
        pricing_seed=123,
        n_pricing=5000,
        training_runtime_s=0.0,
    )
    expected_opt = out["payoff_variance"] - (out["payoff_control_covariance"] ** 2) / out["control_variance"]
    assert math.isclose(out["optimal_residual_variance"], expected_opt, rel_tol=1e-12, abs_tol=1e-12)
    assert out["observed_residual_variance"] == out["observation_variance"]
    assert out["residual_variance_beta_one"] == out["observation_variance"]
    assert out["optimal_residual_variance"] <= out["observed_residual_variance"]


def test_transfer_beta_reports_pricing_sample_oracle_formula(monkeypatch):
    _patch_simulation_and_payoff(monkeypatch)
    net = _DummyFrozenNetwork()
    frozen_hash = compute_network_hash(net)
    cfg = make_contract_cfg("strike_high", n_paths=4000, seed=2)
    out = ncv_transfer_beta(
        frozen_network=net,
        e_h0=0.0,
        frozen_hash=frozen_hash,
        target_cfg=cfg,
        pilot_seed=456,
        pricing_seed=789,
        n_pilot=2000,
        n_pricing=4000,
        training_runtime_s=0.0,
    )
    expected_opt = out["payoff_variance"] - (out["payoff_control_covariance"] ** 2) / out["control_variance"]
    assert math.isclose(out["optimal_residual_variance"], expected_opt, rel_tol=1e-12, abs_tol=1e-12)
    assert out["observed_residual_variance"] == out["observation_variance"]
    assert out["residual_variance_beta_one"] > out["observed_residual_variance"]
