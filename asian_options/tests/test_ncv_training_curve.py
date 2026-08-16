from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from asian_options.ncv_training_curve import (
    TrainingCurveConfig,
    build_seed_manifest,
    compute_ncv_split_diagnostics,
    compute_required_paths,
    compute_total_cost,
    compute_gcv_benchmark,
    ncv_training_facts,
    profile_config,
    replication_seeds,
    run_training_curve_experiment,
    simulate_split_dataset,
    validate_checkpoints,
)

try:
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


def test_smoke_configuration_is_valid():
    cfg = profile_config("smoke", output_dir="/tmp/out", base_seed=1)
    assert cfg.replications == 2
    assert cfg.train_paths == 100
    assert cfg.validation_paths == 200
    assert cfg.test_paths == 500
    assert list(cfg.checkpoints) == [0, 2, 5, 10]


def test_checkpoint_grid_strict_and_zero_start():
    validate_checkpoints((0, 1, 2))
    with pytest.raises(ValueError):
        validate_checkpoints((1, 2, 3))
    with pytest.raises(ValueError):
        validate_checkpoints((0, 2, 2))


def test_dissertation_configuration_uses_252_monitoring_dates():
    cfg = profile_config("dissertation", output_dir="/tmp/out", base_seed=1)
    assert cfg.monitoring_dates == 252


def test_training_validation_test_seeds_are_distinct():
    seeds = replication_seeds(base_seed=42, replication=0)
    assert len({seeds["train"], seeds["validation"], seeds["test"], seeds["gcv_pilot"]}) == 4


def test_seed_manifest_has_all_replication_phase_rows():
    rows = build_seed_manifest(base_seed=123, replications=3)
    assert len(rows) == 12
    phases = {r["phase"] for r in rows}
    assert phases == {"train", "validation", "test", "gcv_pilot"}


def test_residual_variance_invariant_to_adding_analytical_expectation():
    payoff = np.array([2.0, 4.0, 5.0, 9.0])
    h = np.array([1.0, 2.0, 1.0, 3.0])
    out = compute_ncv_split_diagnostics(payoff, h, e_h=10.0)
    assert math.isfinite(out["analytical_eh"])
    assert abs(out["residual_variance"] - out["residual_variance_shift_check"]) < 1e-12


def test_vrr_formula_matches_definition():
    payoff = np.array([1.0, 3.0, 2.0, 8.0])
    h = np.array([0.5, 2.5, 1.5, 5.5])
    out = compute_ncv_split_diagnostics(payoff, h, e_h=0.1)
    expected = np.var(payoff, ddof=1) / np.var(payoff - h, ddof=1)
    assert abs(out["vrr_ncv_vs_mc"] - expected) < 1e-12


def test_required_paths_ceiling_formula_and_floor_of_two():
    n = compute_required_paths(0.25, 0.1)
    assert n == math.ceil(0.25 / (0.1**2))
    assert compute_required_paths(float("nan"), 0.1) == 2
    assert compute_required_paths(0.0, 0.1) == 2


def test_cost_formula_training_once_pricing_q_times():
    c = compute_total_cost(training_runtime=5.0, required_paths=10, per_obs_runtime=0.2, reuse_q=3)
    assert abs(c - (5.0 + 3 * (10 * 0.2))) < 1e-12


def test_gcv_benchmark_uses_same_contract_and_monitoring_schedule():
    val = simulate_split_dataset(monitoring_dates=252, n_paths=20, seed=1)
    test = simulate_split_dataset(monitoring_dates=252, n_paths=20, seed=2)
    pilot = simulate_split_dataset(monitoring_dates=252, n_paths=20, seed=3)
    rows = compute_gcv_benchmark(val, test, pilot, n_reporting=50_000)
    assert {r["split"] for r in rows} == {"validation", "test"}
    assert val["cfg"].m == 252 and test["cfg"].m == 252


def test_ncv_training_facts_reflect_current_objective_and_architecture():
    facts = ncv_training_facts()
    assert "ReLU" in facts["activation"]
    assert "MSELoss" in facts["loss_function"]
    assert "Adam" in facts["optimizer"]
    assert facts["hidden_layer_width_default"] == 32


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch required")
def test_tiny_run_outputs_schema_and_checkpoint_properties(tmp_path, monkeypatch):
    from asian_options import ncv_training_curve as tc

    monkeypatch.setattr(
        tc,
        "_plot_summary_figure",
        lambda output_dir, *args, **kwargs: (output_dir / "ncv_training_curve_summary.png").write_text("x", encoding="utf-8"),
    )

    cfg = TrainingCurveConfig(
        profile="smoke",
        base_seed=7,
        replications=1,
        train_paths=16,
        validation_paths=20,
        test_paths=24,
        monitoring_dates=12,
        checkpoints=(0, 1, 2),
        hidden_width=8,
        learning_rate=1e-2,
        default_epochs=2,
        train_batch_size=8,
        runtime_repeats=2,
        pricing_observations_for_reporting=50_000,
        q_values=(1, 10),
        se_targets=(0.001,),
        output_dir=str(tmp_path),
    )

    out_dir = run_training_curve_experiment(cfg)

    required = [
        "training_curve_config.json",
        "training_curve_environment.json",
        "training_curve_seed_manifest.csv",
        "training_curve_per_replication.csv",
        "training_curve_summary.csv",
        "training_curve_diminishing_returns.csv",
        "training_curve_optimal_checkpoints.csv",
        "training_curve_gcv_benchmark.csv",
        "training_curve_validation_report.json",
        "TRAINING_CURVE_HANDOVER.md",
    ]
    for name in required:
        assert (out_dir / name).exists()

    with (out_dir / "training_curve_per_replication.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert rows

    checkpoints = sorted({int(r["checkpoint"]) for r in rows if r["split"] == "validation"})
    assert checkpoints == [0, 1, 2]

    epoch0_rows = [r for r in rows if int(r["checkpoint"]) == 0]
    assert all(float(r["cumulative_training_runtime_s"]) == 0.0 for r in epoch0_rows)

    for split in ("validation", "test"):
        split_rows = sorted(
            [r for r in rows if r["split"] == split],
            key=lambda r: int(r["checkpoint"]),
        )
        times = [float(r["cumulative_training_runtime_s"]) for r in split_rows]
        assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))
        assert all(r["evaluation_no_grad"] == "True" for r in split_rows)
        assert all(math.isfinite(float(r["analytical_eh"])) for r in split_rows)

    with (out_dir / "training_curve_optimal_checkpoints.csv").open() as fh:
        opts = list(csv.DictReader(fh))
    assert opts
    assert all(r["selection_split"] == "validation" for r in opts)
    grid = {0, 1, 2}
    assert all(int(r["checkpoint"]) in grid for r in opts)

    with (out_dir / "training_curve_validation_report.json").open() as fh:
        report = json.load(fh)
    assert report["checkpoint_grid_starts_at_zero"] is True
    assert report["checkpoint_grid_strictly_increasing"] is True
    assert report["all_present"] is True


def test_invalid_or_zero_residual_variance_handled_explicitly():
    payoff = np.array([1.0, 1.0, 1.0, 1.0])
    h = np.array([1.0, 1.0, 1.0, 1.0])
    out = compute_ncv_split_diagnostics(payoff, h, e_h=0.0)
    assert not math.isfinite(out["vrr_ncv_vs_mc"]) or out["vrr_ncv_vs_mc"] > 0
    assert compute_required_paths(out["residual_variance"], 0.001) == 2
