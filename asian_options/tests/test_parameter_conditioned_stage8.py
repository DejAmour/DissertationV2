from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from asian_options.config import ModelConfig
from asian_options.neural_cv import _ShallowNet, analytical_network_expectation_conditional, build_network
from asian_options.parameter_conditioned_stage8 import (
    INPUT_DIM,
    PARAM_BOUNDS,
    PCNCV_PARAM_COUNT,
    SHOCK_DIM,
    build_parameter_inputs,
    build_seed_manifest,
    generate_parameter_conditioned_training_data,
    profile_config,
    run_parameter_conditioned_stage8,
    trainable_parameter_count,
    transform_contract_parameters,
    _build_variance_ratio_rows,
)

try:
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


def test_parameter_transform_bounds_and_center():
    u_low = transform_contract_parameters(75.0, 0.08, 0.4)
    u_high = transform_contract_parameters(125.0, 0.55, 2.2)
    assert np.allclose(u_low, np.array([-1.0, -1.0, -1.0]), atol=1e-12)
    assert np.allclose(u_high, np.array([1.0, 1.0, 1.0]), atol=1e-12)


def test_build_parameter_inputs_and_dimension():
    z = np.zeros((4, SHOCK_DIM))
    u = np.zeros((4, 3))
    x = build_parameter_inputs(z, u)
    assert x.shape == (4, INPUT_DIM)


def test_trainable_parameter_count_is_545():
    assert trainable_parameter_count(INPUT_DIM, 32) == PCNCV_PARAM_COUNT == 545


def test_conditional_expectation_tau_zero_case():
    W1 = np.zeros((1, INPUT_DIM), dtype=np.float64)
    W1[0, SHOCK_DIM] = 2.0
    b1 = np.array([-1.0], dtype=np.float64)
    W2 = np.array([[3.0]], dtype=np.float64)
    b2 = np.array([0.5], dtype=np.float64)
    net = _ShallowNet(W1=W1, b1=b1, W2=W2, b2=b2)

    p = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    # mu = 2*1 - 1 = 1, tau=0 => E[ReLU]=1
    expected = 3.0 * 1.0 + 0.5
    got = analytical_network_expectation_conditional(net, p, shock_dim=SHOCK_DIM)
    assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-12)


def test_conditional_analytic_expectation_matches_monte_carlo():
    cfg = ModelConfig(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, m=SHOCK_DIM, n_paths=2, seed=7)
    net = build_network(cfg, hidden_width=8, input_dim=INPUT_DIM)
    p = np.array([0.2, -0.1, 0.7], dtype=np.float64)

    n = 20000
    z = np.random.default_rng(123).standard_normal((n, SHOCK_DIM))
    x = build_parameter_inputs(z, np.repeat(p.reshape(1, -1), n, axis=0))
    mc = float(np.mean(net.forward(x)))
    an = analytical_network_expectation_conditional(net, p, shock_dim=SHOCK_DIM)
    assert abs(mc - an) < 0.05


def test_training_data_ranges_and_reproducibility():
    d1 = generate_parameter_conditioned_training_data(
        n_training=50,
        training_parameter_seed=11,
        training_shock_seed=12,
    )
    d2 = generate_parameter_conditioned_training_data(
        n_training=50,
        training_parameter_seed=11,
        training_shock_seed=12,
    )

    assert np.allclose(d1["K"], d2["K"])
    assert np.allclose(d1["sigma"], d2["sigma"])
    assert np.allclose(d1["T"], d2["T"])
    assert np.allclose(d1["X_train"], d2["X_train"])
    assert np.allclose(d1["y_train"], d2["y_train"])

    assert np.min(d1["K"]) >= 75.0
    assert np.max(d1["K"]) <= 125.0
    assert np.min(d1["sigma"]) >= PARAM_BOUNDS["sigma_min"]
    assert np.max(d1["sigma"]) <= PARAM_BOUNDS["sigma_max"]
    assert np.min(d1["T"]) >= PARAM_BOUNDS["t_min"]
    assert np.max(d1["T"]) <= PARAM_BOUNDS["t_max"]
    assert np.all(d1["u_params"] <= 1.0000001)
    assert np.all(d1["u_params"] >= -1.0000001)


def test_seed_manifest_contains_required_streams_and_contract_streams():
    rows = build_seed_manifest(base_seed=42, replications=1)
    streams = {r["stream"] for r in rows if "contract_id" not in r}
    assert {
        "training_parameters",
        "training_shocks",
        "validation_contracts",
        "validation_shocks",
        "gcv_pilot",
        "conditional_ncv_pilot",
        "frozen_ncv_pilot",
        "final_pricing_shocks",
        "reference_training",
    }.issubset(streams)


def test_variance_ratio_summary_log_ci_rows():
    rows = []
    for rep in (0, 1):
        rows.extend(
            [
                {"replication": rep, "contract_id": "reference", "method": "GCV", "observation_variance": 2.0},
                {"replication": rep, "contract_id": "reference", "method": "NCV_TRANSFER_BETA1", "observation_variance": 3.0},
                {"replication": rep, "contract_id": "reference", "method": "NCV_TRANSFER_BETA", "observation_variance": 2.5},
                {"replication": rep, "contract_id": "reference", "method": "PCNCV_BETA1", "observation_variance": 1.0},
                {"replication": rep, "contract_id": "reference", "method": "PCNCV_BETA", "observation_variance": 0.8},
            ]
        )
    per_rep, summary = _build_variance_ratio_rows(rows)
    assert per_rep
    target = [r for r in summary if r["method"] == "PCNCV_BETA" and r["comparator"] == "GCV"]
    assert target
    assert float(target[0]["geometric_mean"]) > 1.0


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch required")
def test_smoke_runner_outputs_required_files(tmp_path):
    cfg = profile_config("smoke", output_dir=str(tmp_path), base_seed=9)
    cfg = dataclasses.replace(cfg, replications=1, n_training=20, n_validation=20, n_pilot=10, n_pricing=20, checkpoints=(0, 2), max_epochs=2, frozen_reference_epochs=2)
    out_dir = run_parameter_conditioned_stage8(cfg)

    required = {
        "config_snapshot.json",
        "environment.json",
        "seed_manifest.csv",
        "training_checkpoint_results.csv",
        "per_replication_results.csv",
        "per_replication_variance_ratios.csv",
        "variance_ratio_summary.csv",
        "runtime_raw.csv",
        "runtime_summary.csv",
        "portfolio_break_even.csv",
        "validation_report.json",
        "handover.md",
    }
    written = {p.name for p in out_dir.iterdir() if p.is_file()}
    assert required.issubset(written)
