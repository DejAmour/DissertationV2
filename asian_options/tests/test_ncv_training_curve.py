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
    required_output_files,
    replication_seeds,
    run_training_curve_experiment,
    simulate_split_dataset,
    validate_numeric_content,
    validate_output_schema,
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
    assert cfg.pilot_paths == 50
    assert list(cfg.timing_path_counts) == [100, 500]
    assert cfg.timing_repeats == 1
    assert 50_000 not in cfg.timing_path_counts


def test_checkpoint_grid_strict_and_zero_start():
    validate_checkpoints((0, 1, 2))
    with pytest.raises(ValueError):
        validate_checkpoints((1, 2, 3))
    with pytest.raises(ValueError):
        validate_checkpoints((0, 2, 2))


def test_dissertation_configuration_uses_252_monitoring_dates():
    cfg = profile_config("dissertation", output_dir="/tmp/out", base_seed=1)
    assert cfg.monitoring_dates == 252
    assert cfg.train_paths == 5_000
    assert cfg.validation_paths == 10_000
    assert cfg.test_paths == 50_000
    assert cfg.replications == 10
    assert list(cfg.checkpoints) == [0, 10, 25, 50, 100, 200, 500, 1000]
    assert cfg.pilot_paths == 1_000
    assert list(cfg.timing_path_counts) == [1_000, 5_000, 10_000]
    assert cfg.timing_repeats == 3


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
    assert compute_required_paths(0.011, 0.1) == 2


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
        pilot_paths=10,
        timing_path_counts=(8, 16),
        timing_repeats=1,
        pricing_observations_for_reporting=200,
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
    assert report["passed"] is True
    assert report["report_present_post_write"] is True
    assert report["exists"]["training_curve_validation_report.json"] is True


def test_validation_report_self_check_pre_and_post_write(tmp_path):
    for name in required_output_files():
        if name != "training_curve_validation_report.json":
            (tmp_path / name).write_text("x", encoding="utf-8")
    pre = validate_output_schema(tmp_path, include_report_in_presence_check=False)
    assert pre["all_present"] is False
    assert pre["exists"]["training_curve_validation_report.json"] is False
    assert pre["passed"] is True
    (tmp_path / "training_curve_validation_report.json").write_text("{}", encoding="utf-8")
    post = validate_output_schema(tmp_path, include_report_in_presence_check=True)
    assert post["all_present"] is True
    assert post["passed"] is True


def test_validation_report_missing_required_file_fails(tmp_path):
    for name in required_output_files():
        if name != "training_curve_summary.csv":
            (tmp_path / name).write_text("x", encoding="utf-8")
    report = validate_output_schema(tmp_path, include_report_in_presence_check=True)
    assert report["all_present"] is False
    assert report["passed"] is False
    assert any("training_curve_summary.csv" in err for err in report["errors"])


def test_numeric_validation_fails_for_non_finite_timing_or_variance():
    rows = [{
        "replication": 0,
        "checkpoint": 25,
        "split": "validation",
        "residual_variance": float("nan"),
        "path_and_payoff_runtime_s": 1.0,
        "control_evaluation_runtime_s": 1.0,
        "estimator_reduction_runtime_s": 0.0,
        "end_to_end_pricing_runtime_s": 2.0,
        "ncv_end_to_end_runtime_per_observation_s": 0.01,
        "cumulative_training_runtime_s": 1.0,
        "training_data_generation_runtime_s": 1.0,
        "optimizer_training_runtime_s": 1.0,
        "validation_generation_and_evaluation_runtime_s": 1.0,
        "runtime_projection_is_sufficiently_linear": True,
    }]
    gcv_rows = [{"replication": 0, "split": "validation", "gcv_residual_variance": 1.0, "end_to_end_pricing_runtime_s": 1.0, "end_to_end_runtime_per_observation_s": float("nan")}]
    optimal_rows = [{"replication": 0, "checkpoint": 25, "Q": 1, "required_pricing_observations": 10, "setup_cost_s": 1.0, "marginal_pricing_cost_s": 1.0, "projected_total_cost_s": 2.0}]
    errors, warnings = validate_numeric_content(rows, gcv_rows, optimal_rows)
    assert warnings == []
    assert any("non-finite residual_variance" in e for e in errors)
    assert any("non-finite end_to_end_runtime_per_observation_s" in e for e in errors)


def test_invalid_or_zero_residual_variance_handled_explicitly():
    payoff = np.array([1.0, 1.0, 1.0, 1.0])
    h = np.array([1.0, 1.0, 1.0, 1.0])
    out = compute_ncv_split_diagnostics(payoff, h, e_h=0.0)
    assert not math.isfinite(out["vrr_ncv_vs_mc"]) or out["vrr_ncv_vs_mc"] > 0
    assert compute_required_paths(out["residual_variance"], 0.001) == 2


def test_gcv_pilot_not_repeated_inside_marginal_timing(monkeypatch):
    from asian_options import ncv_training_curve as tc

    cfg = tc._make_reference_cfg(monitoring_dates=4, n_paths=5, seed=11)
    calls = {"pilot": 0}

    def fake_fit(_cfg, n_pilot):
        calls["pilot"] += 1
        assert n_pilot == 7
        return {"beta": 0.5, "eg": 1.0, "pilot_runtime_s": 0.25}

    monkeypatch.setattr(tc, "_fit_gcv_pilot_once", fake_fit)
    out = tc._timed_mc_av_gcv("GCV", cfg, n_pilot=7, repeats=3)
    assert calls["pilot"] == 1
    assert out["setup_runtime_s"] == 0.25

    calls2 = {"pilot": 0}

    def fail_fit(_cfg, _n_pilot):
        calls2["pilot"] += 1
        raise AssertionError("pilot should not be refit when fitted beta is supplied")

    monkeypatch.setattr(tc, "_fit_gcv_pilot_once", fail_fit)
    out2 = tc._timed_mc_av_gcv(
        "GCV",
        cfg,
        n_pilot=7,
        repeats=4,
        gcv_pilot_fit={"beta": 0.5, "eg": 1.0, "pilot_runtime_s": 0.125},
    )
    assert calls2["pilot"] == 0
    assert out2["setup_runtime_s"] == 0.125


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch required")
def test_runtime_timing_loops_are_bounded_and_not_checkpoint_or_q_driven(tmp_path, monkeypatch):
    from asian_options import ncv_training_curve as tc

    monkeypatch.setattr(
        tc,
        "_plot_summary_figure",
        lambda output_dir, *args, **kwargs: (output_dir / "ncv_training_curve_summary.png").write_text("x", encoding="utf-8"),
    )

    timing_calls = {"baseline": 0, "ncv": 0}
    build_network_calls = {"count": 0}
    timed_path_counts = []
    projected_ns = []

    original_build_network = tc.build_network

    def tracked_build_network(*args, **kwargs):
        build_network_calls["count"] += 1
        return original_build_network(*args, **kwargs)

    def fake_build_runtime_profiles(**kwargs):
        timing_calls["baseline"] += 1
        counts = tuple(int(x) for x in kwargs["timing_path_counts"])
        timed_path_counts.extend(counts)
        rows = []
        for method in kwargs["methods"]:
            for n in counts:
                rows.append(
                    {
                        "method": method,
                        "n_paths": n,
                        "end_to_end_pricing_runtime_s": n / 1000.0,
                        "end_to_end_runtime_per_observation_s": 1 / 1000.0,
                    }
                )
        return rows

    def fake_measure_end_to_end_pricing_runtime_profile(**kwargs):
        timing_calls["ncv"] += 1
        n = int(kwargs["n_paths"])
        timed_path_counts.append(n)
        return {
            "method": "NCV",
            "n_paths": n,
            "timing_repeats": 1,
            "timing_path_counts": str(n),
            "path_and_payoff_runtime_s": n / 1000.0,
            "control_evaluation_runtime_s": 0.0,
            "estimator_reduction_runtime_s": 0.0,
            "end_to_end_pricing_runtime_s": n / 1000.0,
            "torch_tensor_conversion_inside_pricing_timing": True,
            "end_to_end_runtime_per_observation_s": 1 / 1000.0,
        }

    def fake_project_runtime_at_n(_profiles, _method, n_required):
        projected_ns.append(int(n_required))
        return {
            "runtime_projection_basis_n": [4, 8],
            "runtime_projection_method": "linear_per_observation_rate",
            "runtime_projection_is_empirical_or_projected": "projected_from_empirical_multi_n",
            "runtime_projection_linearity_ratio": 1.0,
            "runtime_projection_is_sufficiently_linear": True,
            "projected_runtime_s": int(n_required) / 1000.0,
            "projected_per_observation_s": 1 / 1000.0,
            "runtime_at_required_n_is_empirical_or_projected": "projected",
        }

    def fake_compute_ncv_split_diagnostics(*args, **kwargs):
        return {
            "arithmetic_payoff_mean": 1.0,
            "network_output_mean": 1.0,
            "analytical_eh": 1.0,
            "payoff_variance": 1.0,
            "network_output_variance": 1.0,
            "payoff_network_covariance": 1.0,
            "payoff_network_correlation": 0.5,
            "residual_mean": 0.0,
            "residual_variance": 1.0,
            "ncv_price_estimate": 1.0,
            "estimator_variance_at_reporting_n": 1.0,
            "standard_error_at_reporting_n": 1.0,
            "vrr_ncv_vs_mc": 2.0,
            "residual_variance_shift_check": 1.0,
            "residual_shift_delta": 0.0,
        }

    monkeypatch.setattr(tc, "build_runtime_profiles", fake_build_runtime_profiles)
    monkeypatch.setattr(tc, "measure_end_to_end_pricing_runtime_profile", fake_measure_end_to_end_pricing_runtime_profile)
    monkeypatch.setattr(tc, "project_runtime_at_n", fake_project_runtime_at_n)
    monkeypatch.setattr(tc, "compute_ncv_split_diagnostics", fake_compute_ncv_split_diagnostics)
    monkeypatch.setattr(tc, "build_network", tracked_build_network)

    cfg = TrainingCurveConfig(
        profile="smoke",
        base_seed=9,
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
        runtime_repeats=1,
        pilot_paths=10,
        timing_path_counts=(4, 8),
        timing_repeats=1,
        pricing_observations_for_reporting=200,
        q_values=(1, 10, 1000),
        se_targets=(0.001,),
        output_dir=str(tmp_path),
    )

    out_dir = run_training_curve_experiment(cfg)
    assert timing_calls["baseline"] == 1
    assert timing_calls["ncv"] == len(cfg.timing_path_counts)
    assert build_network_calls["count"] == 1
    assert all(n in set(cfg.timing_path_counts) for n in timed_path_counts)
    assert 50_000 not in timed_path_counts

    with (out_dir / "training_curve_optimal_checkpoints.csv").open() as fh:
        optimal_rows = list(csv.DictReader(fh))
    assert optimal_rows
    required_ns = [int(r["required_pricing_observations"]) for r in optimal_rows]
    assert max(required_ns) > max(cfg.timing_path_counts)
    assert any(n > max(cfg.timing_path_counts) for n in projected_ns)
    assert all(int(r["Q"]) in (1, 10, 1000) for r in optimal_rows)
    assert all(float(r["projected_total_cost_s"]) == float(r["setup_cost_s"]) + int(r["Q"]) * float(r["marginal_pricing_cost_s"]) for r in optimal_rows)
