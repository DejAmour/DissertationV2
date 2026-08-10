from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

import scripts.run_experiments as run_experiments
from asian_options.validation import validate_profile_outputs


def _stat_rows_fixture() -> list[dict]:
    rows = []
    mc_obs = 0.8
    for mode in ("A_equal_obs", "B_equal_budget"):
        for method, obs_var, obs, sim, pilot, train, total in (
            ("MC", mc_obs, 100, 100, 0, 0, 100),
            ("AV", 0.5, 100, 200, 0, 0, 200),
            ("CV", 0.4, 90 if mode == "B_equal_budget" else 100, 90 if mode == "B_equal_budget" else 100, 10 if mode == "B_equal_budget" else 10, 0, 100 if mode == "B_equal_budget" else 110),
            ("NCV", 0.3, 80 if mode == "B_equal_budget" else 100, 80 if mode == "B_equal_budget" else 100, 0, 20 if mode == "B_equal_budget" else 20, 100 if mode == "B_equal_budget" else 120),
        ):
            rows.append(
                {
                    "comparison_mode": mode,
                    "method": method,
                    "price": "10.0",
                    "std_error": "0.1",
                    "observation_variance": f"{obs_var}",
                    "estimator_variance": f"{obs_var / obs}",
                    "variance_reduction_ratio": f"{(mc_obs / obs_var):.4f}",
                    "pricing_observations": obs,
                    "pricing_simulated_paths": sim,
                    "pilot_paths": pilot,
                    "training_paths": train,
                    "total_simulated_paths": total,
                    "runtime_s": "0.01",
                    "notes": "",
                    "seed": 11,
                    "replication": 0,
                    "profile_config_id": "cfg1",
                }
            )
    return rows


def _runtime_rows_fixture() -> list[dict]:
    mc_est_var = 0.8 / 100
    mc_runtime = 2.0
    rows = []
    for method, est_var, runtime, obs, sim in (
        ("MC", mc_est_var, mc_runtime, 100, 100),
        ("AV", 0.005, 1.8, 100, 200),
        ("CV", 0.004, 1.6, 100, 100),
        ("NCV", 0.003, 1.4, 100, 100),
    ):
        rows.append(
            {
                "comparison_mode": "C_runtime",
                "method": method,
                "runtime_seconds": f"{runtime}",
                "time_per_observation": f"{runtime / obs}",
                "time_per_simulated_path": f"{runtime / sim}",
                "efficiency_gain_vs_mc": f"{(mc_est_var * mc_runtime) / (est_var * runtime):.6f}",
                "timing_scope": "pricing only",
                "pricing_observations": obs,
                "pricing_simulated_paths": sim,
                "pilot_paths": 0,
                "training_paths": 0 if method != "NCV" else 20,
                "total_simulated_paths": sim if method != "NCV" else sim + 20,
                "price": "10.0",
                "std_error": "0.1",
                "observation_variance": "0.8",
                "estimator_variance": f"{est_var}",
                "notes": "",
                "seed": 11,
                "replication": 0,
                "profile_config_id": "cfg1",
            }
        )
    return rows


def _publication_rows_fixture() -> list[dict]:
    rows = []
    for method in ("MC", "AV", "CV", "NCV"):
        rows.append(
            {
                "mode": "A_equal_obs",
                "method": method,
                "runtime_seconds": "NA",
                "time_per_observation": "NA",
                "time_per_simulated_path": "NA",
                "efficiency_gain_vs_mc": "NA",
                "timing_scope_note": "NA",
                "variance_reduction_ratio": "1.0000",
                "profile_config_id": "cfg1",
                "seed": 11,
                "replication": 0,
            }
        )
    for method in ("MC", "AV", "CV", "NCV"):
        rows.append(
            {
                "mode": "C_runtime",
                "method": method,
                "runtime_seconds": "1.0",
                "time_per_observation": "0.01",
                "time_per_simulated_path": "0.01",
                "efficiency_gain_vs_mc": "1.0",
                "timing_scope_note": "pricing only",
                "variance_reduction_ratio": "NA",
                "profile_config_id": "cfg1",
                "seed": 11,
                "replication": 0,
            }
        )
    return rows


def test_stage7_validation_checks_pass_on_good_fixture():
    report = validate_profile_outputs(_stat_rows_fixture(), _runtime_rows_fixture(), _publication_rows_fixture())
    assert report.passed
    assert not report.failures


def test_stage7_validation_checks_fail_with_actionable_message():
    stat_rows = _stat_rows_fixture()
    stat_rows[1]["pricing_simulated_paths"] = 199
    report = validate_profile_outputs(stat_rows, _runtime_rows_fixture(), _publication_rows_fixture())
    assert not report.passed
    assert any("AV must satisfy pricing_simulated_paths == 2 * pricing_observations" in failure for failure in report.failures)


def _fake_stat_row(mode: str, method: str, seed: int, n_paths: int, pilot: int = 0, training: int = 0, mc_obs: float = 1.0) -> dict:
    obs_var = {"MC": mc_obs, "AV": 0.6, "CV": 0.5, "NCV": 0.4}[method]
    return {
        "comparison_mode": mode,
        "method": method,
        "pricing_observations": n_paths,
        "pricing_simulated_paths": n_paths if method != "AV" else 2 * n_paths,
        "pilot_paths": pilot if method == "CV" else 0,
        "training_paths": training if method == "NCV" else 0,
        "total_simulated_paths": (n_paths if method != "AV" else 2 * n_paths) + (pilot if method == "CV" else 0) + (training if method == "NCV" else 0),
        "price": "10.000000",
        "observation_variance": f"{obs_var:.8e}",
        "estimator_variance": f"{(obs_var / n_paths):.8e}",
        "variance_reduction_ratio": f"{(mc_obs / obs_var):.4f}",
        "std_error": "0.100000",
        "ci_lower": "9.000000",
        "ci_upper": "11.000000",
        "runtime_s": "0.0100",
        "notes": "",
    }


def _fake_runtime_rows(seed: int, n_paths: int, policy: str) -> list[dict]:
    mc_est_var = 1.0 / n_paths
    rows = []
    for method, est_var, runtime, sim, training in (
        ("MC", mc_est_var, 2.0, n_paths, 0),
        ("AV", 0.006, 1.8, 2 * n_paths, 0),
        ("CV", 0.005, 1.6, n_paths, 0),
        ("NCV", 0.004, 1.4, n_paths, 5),
    ):
        rows.append(
            {
                "comparison_mode": "C_runtime",
                "method": method,
                "runtime_seconds": f"{runtime:.6f}",
                "pricing_observations": n_paths,
                "pricing_simulated_paths": sim,
                "pilot_paths": 0,
                "training_paths": training,
                "total_simulated_paths": sim + training,
                "time_per_observation": f"{runtime / n_paths:.8e}",
                "time_per_simulated_path": f"{runtime / sim:.8e}",
                "price": "10.000000",
                "std_error": "0.100000",
                "observation_variance": "1.00000000e+00",
                "estimator_variance": f"{est_var:.8e}",
                "efficiency_gain_vs_mc": f"{((mc_est_var * 2.0) / (est_var * runtime)):.6f}",
                "timing_scope": policy,
                "notes": "",
            }
        )
    return rows


def test_validation_profile_generation_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    def fake_equal_obs(cfg=None, n_pilot=None, n_training=None):
        return [
            _fake_stat_row("A_equal_obs", "MC", cfg.seed, cfg.n_paths),
            _fake_stat_row("A_equal_obs", "AV", cfg.seed, cfg.n_paths),
            _fake_stat_row("A_equal_obs", "CV", cfg.seed, cfg.n_paths, pilot=n_pilot),
            _fake_stat_row("A_equal_obs", "NCV", cfg.seed, cfg.n_paths, training=n_training),
        ]

    def fake_equal_budget(cfg=None, n_pilot=None, n_training=None, total_path_budget=None):
        cv_n = total_path_budget - n_pilot
        ncv_n = total_path_budget - n_training
        return [
            _fake_stat_row("B_equal_budget", "MC", cfg.seed, total_path_budget),
            _fake_stat_row("B_equal_budget", "AV", cfg.seed, total_path_budget // 2),
            _fake_stat_row("B_equal_budget", "CV", cfg.seed, cv_n, pilot=n_pilot),
            _fake_stat_row("B_equal_budget", "NCV", cfg.seed, ncv_n, training=n_training),
        ]

    def fake_runtime(cfg=None, n_pilot=None, n_training=None, timing_scope_policy="exclude_ncv_training"):
        return _fake_runtime_rows(cfg.seed, cfg.n_paths, timing_scope_policy)

    monkeypatch.setattr(run_experiments, "run_equal_obs_comparison", fake_equal_obs)
    monkeypatch.setattr(run_experiments, "run_equal_budget_comparison", fake_equal_budget)
    monkeypatch.setattr(run_experiments, "run_runtime_comparison", fake_runtime)
    monkeypatch.setattr(run_experiments, "_commit_hash", lambda: "deadbeef")
    monkeypatch.setattr(run_experiments, "collect_environment_metadata", lambda: {"python_version": "test"})

    out_dir = tmp_path / "runs"
    argv = [
        "run_experiments.py",
        "--output-dir",
        str(out_dir),
        "--profile",
        "validation_minimal",
        "--profile-seeds",
        "1,2",
        "--profile-replications",
        "1",
        "--pilot-paths",
        "2",
        "--training-paths",
        "5",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert run_experiments.main() == 0

    run_dirs = sorted(out_dir.glob("run_*"))
    assert run_dirs
    run_dir = run_dirs[-1]

    expected_files = {
        "mode_ab_statistical_raw.csv",
        "mode_c_runtime_raw.csv",
        "merged_summary.csv",
        "summary_stable.csv",
        "paper_table.csv",
        "paper_table.md",
        "paper_table_notes.txt",
        "validation_aggregate.csv",
        "validation_aggregate.md",
        "validation_report.md",
        "metadata.json",
        "manifest.json",
        "README.txt",
    }
    assert expected_files.issubset({p.name for p in run_dir.iterdir() if p.is_file()})

    metadata = json.loads((run_dir / "metadata.json").read_text())
    assert metadata["profile_name"] == "validation_minimal"
    assert metadata["profile_version"] == run_experiments.VALIDATION_PROFILE_VERSION
    assert metadata["seeds"] == [1, 2]
    assert len(metadata["profile_config_matrix"]) >= 4

    with open(run_dir / "validation_aggregate.csv", newline="") as fh:
        agg_rows = list(csv.DictReader(fh))
    assert agg_rows
