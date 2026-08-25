from __future__ import annotations

import csv
import json
import math

import numpy as np
import pytest

from asian_options.ncv_capacity_data_sensitivity import (
    CapacityCell,
    CapacityDataConfig,
    _build_paired_contrasts,
    _default_cells,
    _default_paired_comparisons,
    _build_seed_manifest,
    _expected_parameter_count,
    _generate_nested_training_splits,
    _make_reference_cfg,
    _network_seed,
    _selected_checkpoints,
    _split_seeds,
    _student_t_ci,
    _torch_model_from_initial_network,
    _train_continuous_snapshots,
    profile_config,
    run_capacity_data_sensitivity,
)
from asian_options.neural_cv import build_network

try:
    import torch

    _TORCH_AVAILABLE = True
except Exception:
    _TORCH_AVAILABLE = False


def test_profiles_have_exactly_five_configurations():
    smoke = profile_config("smoke", output_dir="/tmp/out", base_seed=42)
    dissertation = profile_config("dissertation", output_dir="/tmp/out", base_seed=42)
    assert len(smoke.cells) == 5
    assert len(dissertation.cells) == 5
    assert [c.config_id for c in dissertation.cells] == [
        "w32_n5000",
        "w16_n5000",
        "w8_n5000",
        "w32_n10000",
        "w32_n20000",
    ]


def test_dissertation_missing_cells_configurations_are_present_and_unique():
    cfg = profile_config("dissertation", output_dir="/tmp/out", base_seed=42, dissertation_cell_set="missing_cells")
    assert [c.config_id for c in cfg.cells] == ["w8_n10000", "w8_n20000", "w16_n10000", "w16_n20000"]
    assert len({c.config_id for c in cfg.cells}) == 4
    assert {c.train_paths for c in cfg.cells} == {10_000, 20_000}
    assert {c.hidden_width for c in cfg.cells} == {8, 16}


def test_full_grid_cell_set_has_exactly_nine_unique_cells():
    cells = _default_cells("dissertation", dissertation_cell_set="full_grid")
    assert len(cells) == 9
    assert len({c.config_id for c in cells}) == 9
    assert {(c.hidden_width, c.train_paths) for c in cells} == {
        (8, 5_000), (8, 10_000), (8, 20_000),
        (16, 5_000), (16, 10_000), (16, 20_000),
        (32, 5_000), (32, 10_000), (32, 20_000),
    }


def test_expected_parameter_counts_and_ratios_for_m252():
    m = 252
    cfg = _make_reference_cfg(monitoring_dates=m, n_paths=10, seed=1)
    for h, expected in ((8, 2033), (16, 4065), (32, 8129)):
        net = build_network(cfg, hidden_width=h)
        model = _torch_model_from_initial_network(torch if _TORCH_AVAILABLE else __import__("torch"), net)
        count = sum(int(p.numel()) for p in model.parameters())
        assert count == expected
        assert _expected_parameter_count(m, h) == expected
    assert math.isclose(4065 / 5000, 8129 / 10000, rel_tol=2e-3)
    assert math.isclose(2033 / 5000, 8129 / 20000, rel_tol=2e-3)
    assert math.isclose(10_000 / 2033, 4.91884, rel_tol=1e-4)
    assert math.isclose(20_000 / 2033, 9.83768, rel_tol=1e-4)
    assert math.isclose(10_000 / 4065, 2.46002, rel_tol=1e-4)
    assert math.isclose(20_000 / 4065, 4.92005, rel_tol=1e-4)


def test_seed_generation_is_deterministic_and_no_hash_needed():
    s0 = _split_seeds(42, 0)
    s1 = _split_seeds(42, 1)
    assert s0 != s1
    assert s0["train"] == 1042
    manifest = _build_seed_manifest(profile_config("smoke", output_dir="/tmp/out", base_seed=42))
    assert len(manifest) == 2 * (5 + 5)


def test_network_seed_rule_for_width32_shared_across_data_sizes():
    seeds = _split_seeds(42, 0)
    w32_a = _network_seed(seeds, CapacityCell("a", 32, 100))
    w32_b = _network_seed(seeds, CapacityCell("b", 32, 400))
    w16 = _network_seed(seeds, CapacityCell("c", 16, 100))
    assert w32_a == w32_b
    assert w16 != w32_a


def test_nested_training_samples_prefix_and_independence():
    splits, _rows, diag = _generate_nested_training_splits(
        monitoring_dates=252,
        training_seed=123,
        training_sizes=[100, 200, 400],
    )
    assert diag["training_nested_prefix_ok"] is True
    assert np.array_equal(splits[100]["Z"], splits[400]["Z"][:100])
    assert np.array_equal(splits[200]["Z"], splits[400]["Z"][:200])


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch required")
def test_continuous_training_snapshots_and_epoch_zero_runtime_semantics():
    splits, _, _ = _generate_nested_training_splits(
        monitoring_dates=12,
        training_seed=321,
        training_sizes=[20],
    )
    train = splits[20]
    vcfg = _make_reference_cfg(12, 25, 222)
    trng = np.random.default_rng(222)
    val = {
        "cfg": vcfg,
        "Z": trng.standard_normal((25, 12)),
    }
    from asian_options.payoffs import arithmetic_asian_call_payoff, geometric_asian_call_payoff
    from asian_options.simulate_gbm import simulate_paths

    vpaths = simulate_paths(vcfg, shocks=val["Z"])
    val["payoff_arithmetic"] = arithmetic_asian_call_payoff(vpaths, vcfg)
    val["payoff_geometric"] = geometric_asian_call_payoff(vpaths, vcfg)

    tcfg = _make_reference_cfg(12, 30, 333)
    trng2 = np.random.default_rng(333)
    test = {
        "cfg": tcfg,
        "Z": trng2.standard_normal((30, 12)),
    }
    tpaths = simulate_paths(tcfg, shocks=test["Z"])
    test["payoff_arithmetic"] = arithmetic_asian_call_payoff(tpaths, tcfg)
    test["payoff_geometric"] = geometric_asian_call_payoff(tpaths, tcfg)

    snaps = _train_continuous_snapshots(
        torch_mod=torch,
        train_split=train,
        validation_split=val,
        test_split=test,
        hidden_width=8,
        learning_rate=1e-2,
        batch_size=16,
        checkpoints=(0, 1, 2),
        n_reporting=30,
        init_seed=999,
    )
    assert snaps[0].optimizer_cumulative_runtime_s == 0.0
    assert snaps[2].optimizer_cumulative_runtime_s >= snaps[1].optimizer_cumulative_runtime_s
    assert math.isfinite(snaps[1].e_h)
    assert snaps[1].test_diag["residual_variance"] > 0.0


def test_validation_only_checkpoint_selection_and_tie_break_earlier():
    cfg = profile_config("smoke", output_dir="/tmp/out", base_seed=42)
    rows = []
    for rep in range(2):
        for cp, val_var in ((0, 5.0), (1, 2.0), (2, 2.0)):
            rows.append(
                {
                    "config_id": "w32_n100",
                    "replication": rep,
                    "checkpoint": cp,
                    "split": "validation",
                    "centered_residual_variance": val_var,
                }
            )
    selected_rows, selected_map = _selected_checkpoints(rows, cfg)
    assert selected_map["w32_n100"] == 1
    assert selected_rows[0]["selected_checkpoint"] == 1


def test_changing_test_results_does_not_change_selected_checkpoint():
    cfg = profile_config("smoke", output_dir="/tmp/out", base_seed=42)
    base = []
    for rep in range(2):
        for cp, val_var in ((0, 4.0), (1, 2.0), (2, 3.0)):
            base.append(
                {
                    "config_id": "w32_n100",
                    "replication": rep,
                    "checkpoint": cp,
                    "split": "validation",
                    "centered_residual_variance": val_var,
                }
            )
            base.append(
                {
                    "config_id": "w32_n100",
                    "replication": rep,
                    "checkpoint": cp,
                    "split": "test",
                    "centered_residual_variance": 999.0 - 100.0 * cp,
                }
            )
    _, sel1 = _selected_checkpoints(base, cfg)
    for row in base:
        if row["split"] == "test":
            row["centered_residual_variance"] *= 1000.0
    _, sel2 = _selected_checkpoints(base, cfg)
    assert sel1 == sel2


def test_paired_contrast_direction_is_unambiguous():
    rows = []
    for rep in range(2):
        rows.extend(
            [
                {
                    "replication": rep,
                    "config_id": "a",
                    "checkpoint": 1,
                    "split": "test",
                    "mse": 1.0,
                    "centered_residual_variance": 1.0,
                    "ncv_vrr_vs_mc": 2.0,
                    "generalization_gap_log": 0.1,
                    "ncv_setup_cost_s": 10.0,
                },
                {
                    "replication": rep,
                    "config_id": "b",
                    "checkpoint": 2,
                    "split": "test",
                    "mse": 2.0,
                    "centered_residual_variance": 4.0,
                    "ncv_vrr_vs_mc": 1.0,
                    "generalization_gap_log": 0.3,
                    "ncv_setup_cost_s": 13.0,
                },
            ]
        )
    out = _build_paired_contrasts(rows, {"a": 1, "b": 2}, [("a", "b")])
    lookup = {r["metric"]: r for r in out}
    assert lookup["paired_difference_test_mse"]["mean"] == -1.0
    assert lookup["paired_log_ratio_test_residual_variance"]["mean"] < 0.0
    assert lookup["paired_ratio_test_vrr"]["mean"] > 1.0


def test_default_paired_comparisons_cover_full_grid():
    cells = _default_cells("dissertation", dissertation_cell_set="full_grid")
    pairs = _default_paired_comparisons(cells)
    assert len(pairs) == 18
    required = {
        ("w8_n5000", "w16_n5000"),
        ("w8_n5000", "w32_n5000"),
        ("w16_n5000", "w32_n5000"),
        ("w8_n10000", "w16_n10000"),
        ("w8_n10000", "w32_n10000"),
        ("w16_n10000", "w32_n10000"),
        ("w8_n20000", "w16_n20000"),
        ("w8_n20000", "w32_n20000"),
        ("w16_n20000", "w32_n20000"),
        ("w8_n10000", "w8_n5000"),
        ("w8_n20000", "w8_n5000"),
        ("w8_n20000", "w8_n10000"),
        ("w16_n10000", "w16_n5000"),
        ("w16_n20000", "w16_n5000"),
        ("w16_n20000", "w16_n10000"),
        ("w32_n10000", "w32_n5000"),
        ("w32_n20000", "w32_n5000"),
        ("w32_n20000", "w32_n10000"),
    }
    assert required.issubset(set(pairs))


def test_student_t_ci_safe_for_n_less_than_two():
    none = _student_t_ci([])
    one = _student_t_ci([3.0])
    assert none["ci_status"] == "no_observations"
    assert one["ci_status"] == "undefined_n_equals_1"


@pytest.mark.skipif(not _TORCH_AVAILABLE, reason="torch required")
def test_smoke_run_output_schema_row_counts_and_no_overwrite(tmp_path):
    cfg = profile_config("smoke", output_dir=str(tmp_path), base_seed=7)
    out1 = run_capacity_data_sensitivity(cfg)
    out2 = run_capacity_data_sensitivity(cfg)
    assert out1 != out2

    required = {
        "capacity_data_config.json",
        "capacity_data_environment.json",
        "capacity_data_seed_manifest.csv",
        "capacity_data_per_replication.csv",
        "capacity_data_checkpoint_summary.csv",
        "capacity_data_selected_checkpoints.csv",
        "capacity_data_paired_contrasts.csv",
        "capacity_data_runtime_summary.csv",
        "capacity_data_gcv_benchmark.csv",
        "capacity_data_validation_report.json",
        "CAPACITY_DATA_HANDOVER.md",
        "ncv_capacity_data_summary.png",
    }
    produced = {p.name for p in out1.iterdir() if p.is_file()}
    assert required.issubset(produced)

    with (out1 / "capacity_data_per_replication.csv").open() as fh:
        rows = list(csv.DictReader(fh))
    expected = cfg.replications * len(cfg.cells) * len(cfg.checkpoints) * 3
    assert len(rows) == expected

    with (out1 / "capacity_data_validation_report.json").open() as fh:
        report = json.load(fh)
    assert report["passed"] is True
    assert report["row_count_matches_expectation"] is True
