from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

import scripts.run_experiments as run_experiments
from asian_options.results import (
    RESULT_SCHEMA_VERSION,
    STATISTICAL_OUTPUT_COLUMNS,
    RUNTIME_OUTPUT_COLUMNS,
    PUBLICATION_TABLE_COLUMNS,
)


def _fake_stat_row(mode: str, method: str, seed: int, n_paths: int, pilot: int = 0, training: int = 0) -> dict:
    base = 10.0 + seed * 0.01
    return {
        "comparison_mode": mode,
        "method": method,
        "pricing_observations": n_paths,
        "pricing_simulated_paths": n_paths if method != "AV" else 2 * n_paths,
        "pilot_paths": pilot,
        "training_paths": training,
        "total_simulated_paths": (n_paths if method != "AV" else 2 * n_paths) + pilot + training,
        "price": f"{base:.6f}",
        "observation_variance": f"{(0.50 + seed * 0.001):.8e}",
        "estimator_variance": f"{(0.05 + seed * 0.001):.8e}",
        "variance_reduction_ratio": "1.0000",
        "std_error": f"{(0.20 + seed * 0.001):.6f}",
        "ci_lower": "9.000000",
        "ci_upper": "11.000000",
        "runtime_s": "0.0100",
        "notes": "",
    }


def _fake_runtime_row(seed: int, n_paths: int, policy: str) -> dict:
    return {
        "comparison_mode": "C_runtime",
        "method": "MC",
        "runtime_seconds": f"{(0.5 + seed * 0.001):.6f}",
        "pricing_observations": n_paths,
        "pricing_simulated_paths": n_paths,
        "pilot_paths": 0,
        "training_paths": 0,
        "total_simulated_paths": n_paths,
        "time_per_observation": "1.00000000e-03",
        "time_per_simulated_path": "1.00000000e-03",
        "price": "10.000000",
        "std_error": "0.100000",
        "observation_variance": "5.00000000e-01",
        "estimator_variance": "5.00000000e-02",
        "efficiency_gain_vs_mc": "1.000000",
        "timing_scope": policy,
        "notes": "",
    }


def _run_with_fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, seeds: str) -> Path:
    def fake_equal_obs(cfg=None, n_pilot=None, n_training=None):
        return [_fake_stat_row("A_equal_obs", "MC", cfg.seed, cfg.n_paths, pilot=0, training=0)]

    def fake_equal_budget(cfg=None, n_pilot=None, n_training=None, total_path_budget=None):
        return [_fake_stat_row("B_equal_budget", "MC", cfg.seed, total_path_budget, pilot=n_pilot, training=0)]

    def fake_runtime(cfg=None, n_pilot=None, n_training=None, timing_scope_policy="exclude_ncv_training"):
        return [_fake_runtime_row(cfg.seed, cfg.n_paths, timing_scope_policy)]

    monkeypatch.setattr(run_experiments, "run_equal_obs_comparison", fake_equal_obs)
    monkeypatch.setattr(run_experiments, "run_equal_budget_comparison", fake_equal_budget)
    monkeypatch.setattr(run_experiments, "run_runtime_comparison", fake_runtime)
    monkeypatch.setattr(run_experiments, "_commit_hash", lambda: "deadbeef")
    monkeypatch.setattr(run_experiments, "collect_environment_metadata", lambda: {"python_version": "test"})

    out_dir = tmp_path / "runs"
    argv = [
        "run_experiments.py",
        "--output-dir", str(out_dir),
        "--modes", "A", "B", "C",
        "--seeds", seeds,
        "--replications", "1",
        "--n-paths", "10",
        "--total-path-budget", "12",
        "--pilot-paths", "2",
        "--training-paths", "3",
        "--timing-scope-policy", "exclude_ncv_training",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    assert run_experiments.main() == 0

    run_dirs = sorted(out_dir.glob("run_*"))
    assert run_dirs
    return run_dirs[-1]


@pytest.mark.parametrize(
    "argv, expected_msg",
    [
        (
            ["run_experiments.py", "--modes", "B", "--total-path-budget", "1"],
            "--total-path-budget must be >= 2 for AV in Mode B",
        ),
        (
            ["run_experiments.py", "--n-paths", "0"],
            "--n-paths must be > 0",
        ),
        (
            ["run_experiments.py", "--pilot-paths", "-1"],
            "--pilot-paths must be >= 0",
        ),
    ],
)
def test_cli_validation_errors(monkeypatch, capsys, argv, expected_msg):
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        run_experiments.main()
    assert exc.value.code == 2
    assert expected_msg in capsys.readouterr().err


def test_output_contract_and_manifest_schema(tmp_path, monkeypatch):
    run_dir = _run_with_fakes(tmp_path, monkeypatch, seeds="7")

    expected = {
        "mode_ab_statistical_raw.csv",
        "mode_c_runtime_raw.csv",
        "merged_summary.csv",
        "summary_stable.csv",
        "paper_table.csv",
        "paper_table.md",
        "paper_table_notes.txt",
        "metadata.json",
        "manifest.json",
        "README.txt",
    }
    actual = {p.name for p in run_dir.iterdir() if p.is_file()}
    assert expected.issubset(actual)

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["schema_version"] == RESULT_SCHEMA_VERSION
    assert "statistical outputs" in manifest["metric_definitions"]["observation_variance"] or "variance" in manifest["metric_definitions"]["observation_variance"]


def test_canonical_columns_and_publication_na_rules(tmp_path, monkeypatch):
    run_dir = _run_with_fakes(tmp_path, monkeypatch, seeds="8")

    with open(run_dir / "mode_ab_statistical_raw.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == STATISTICAL_OUTPUT_COLUMNS

    with open(run_dir / "mode_c_runtime_raw.csv", newline="") as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames == RUNTIME_OUTPUT_COLUMNS

    with open(run_dir / "paper_table.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    assert list(rows[0].keys()) == PUBLICATION_TABLE_COLUMNS

    stat_row = next(r for r in rows if r["mode"] == "A_equal_obs")
    runtime_row = next(r for r in rows if r["mode"] == "C_runtime")
    assert stat_row["runtime_seconds"] == "NA"
    assert runtime_row["variance_reduction_ratio"] == "NA"


def test_reproducibility_same_seed_stable_and_different_seed_changes(tmp_path, monkeypatch):
    run_dir_1 = _run_with_fakes(tmp_path, monkeypatch, seeds="10")
    stable_1 = (run_dir_1 / "summary_stable.csv").read_bytes()

    run_dir_2 = _run_with_fakes(tmp_path, monkeypatch, seeds="10")
    stable_2 = (run_dir_2 / "summary_stable.csv").read_bytes()

    run_dir_3 = _run_with_fakes(tmp_path, monkeypatch, seeds="11")
    stable_3 = (run_dir_3 / "summary_stable.csv").read_bytes()

    assert stable_1 == stable_2
    assert stable_1 != stable_3

    metadata = json.loads((run_dir_1 / "metadata.json").read_text())
    assert metadata["seeds"] == [10]
    assert metadata["config_hash"]
