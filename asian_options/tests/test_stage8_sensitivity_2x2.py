from __future__ import annotations

import csv
import math

import pytest

from asian_options.stage8_sensitivity_2x2 import (
    FORMAL_EPOCHS,
    MONITORING_DATES_GRID,
    SensitivityConfig,
    _build_seed_manifest,
    _validate_seed_independence,
    compute_paired_contrasts,
    formal_design_cells,
    log_ratio_summary,
    profile_config,
    required_ncv_observations_to_match_gcv,
    run_sensitivity_study,
    solve_break_even_q,
)

try:
    import torch  # noqa: F401

    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


def test_formal_design_is_exactly_four_cells_and_fixed_epochs():
    cells = formal_design_cells()
    assert len(cells) == 4
    assert {tuple((c["monitoring_dates"], c["ncv_epoch"])) for c in cells} == {
        (12, 25),
        (12, 1000),
        (252, 25),
        (252, 1000),
    }
    assert FORMAL_EPOCHS == (25, 1000)
    assert MONITORING_DATES_GRID == (12, 252)


def test_required_observations_and_ratio_direction():
    n, reason = required_ncv_observations_to_match_gcv(obs_var_ncv=4.0, obs_var_gcv=2.0, gcv_observations=50_000)
    assert reason == ""
    assert n == math.ceil(4.0 / (2.0 / 50_000.0))


def test_log_ratio_summary_uses_geometric_ci_on_log_scale():
    out = log_ratio_summary([2.0, 8.0])
    assert out["count"] == 2
    assert math.isclose(out["geometric_mean"], 4.0, rel_tol=1e-12)
    assert out["geometric_ci95_lower"] > 0.0
    assert out["geometric_ci95_upper"] >= out["geometric_ci95_lower"]


def test_paired_log_contrasts_and_interaction():
    rows = [
        {"replication": 0, "monitoring_dates": 12, "ncv_epoch": 25, "ncv_to_gcv_advantage": 1.0},
        {"replication": 0, "monitoring_dates": 12, "ncv_epoch": 1000, "ncv_to_gcv_advantage": 2.0},
        {"replication": 0, "monitoring_dates": 252, "ncv_epoch": 25, "ncv_to_gcv_advantage": 4.0},
        {"replication": 0, "monitoring_dates": 252, "ncv_epoch": 1000, "ncv_to_gcv_advantage": 8.0},
    ]
    contrasts = {r["contrast"]: r for r in compute_paired_contrasts(rows)}
    expected = math.log(2.0) - math.log(1.0)
    assert math.isclose(float(contrasts["delta_12"]["estimate_log_scale"]), expected, rel_tol=1e-12)
    expected_252 = math.log(8.0) - math.log(4.0)
    assert math.isclose(float(contrasts["delta_252"]["estimate_log_scale"]), expected_252, rel_tol=1e-12)
    assert math.isclose(float(contrasts["delta_interaction"]["estimate_log_scale"]), expected - expected_252, rel_tol=1e-12)


def test_break_even_q1_boundary_and_no_finite_case():
    q1 = solve_break_even_q(
        baseline_setup_cost=0.0,
        baseline_marginal_cost=10.0,
        proposed_setup_cost=2.0,
        proposed_marginal_cost=5.0,
    )
    assert q1["break_even_q"] == 1
    assert q1["verified_q"] is True

    no_finite = solve_break_even_q(
        baseline_setup_cost=0.0,
        baseline_marginal_cost=5.0,
        proposed_setup_cost=1.0,
        proposed_marginal_cost=6.0,
    )
    assert no_finite["break_even_q"] == "NA"
    assert no_finite["failure_reason"] == "proposed_marginal_not_below_baseline_no_finite_break_even"


def test_seed_manifest_streams_are_independent_within_replication_monitoring():
    cfg = SensitivityConfig(
        profile="unit",
        base_seed=42,
        replications=2,
        train_paths=10,
        validation_paths=10,
        pilot_paths=10,
        pricing_paths=10,
        hidden_width=32,
        learning_rate=1e-2,
        batch_size=8,
        checkpoints=(0, 25, 1000),
        output_dir="/tmp/out",
        direct_timing_max_paths=100,
        direct_timing_repeats=1,
    )
    manifest = _build_seed_manifest(cfg)
    ok, failures = _validate_seed_independence(manifest)
    assert ok is True
    assert failures == []


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch required")
def test_smoke_run_outputs_required_files_and_pairing(tmp_path):
    cfg = profile_config("smoke", output_dir=str(tmp_path), base_seed=7)
    cfg = SensitivityConfig(**{**cfg.__dict__, "replications": 1})
    out_dir = run_sensitivity_study(cfg)

    required = {
        "config.json",
        "replication_level_results.csv",
        "cell_summary.csv",
        "paired_contrasts.csv",
        "checkpoint_curve_results.csv",
        "runtime_results.csv",
        "validation_report.json",
        "figure_checkpoint_curves.png",
        "figure_checkpoint_curves.pdf",
        "figure_2x2_interaction.png",
        "figure_2x2_interaction.pdf",
        "dissertation_summary.md",
    }
    written = {p.name for p in out_dir.iterdir() if p.is_file()}
    assert required.issubset(written)

    with (out_dir / "replication_level_results.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 4
    pairs = {(int(r["monitoring_dates"]), int(r["ncv_epoch"])) for r in rows}
    assert pairs == {(12, 25), (12, 1000), (252, 25), (252, 1000)}

    by_rep_m = {}
    for r in rows:
        key = (int(r["replication"]), int(r["monitoring_dates"]))
        by_rep_m.setdefault(key, set()).add(r["trajectory_id"])
    assert all(len(v) == 1 for v in by_rep_m.values())
