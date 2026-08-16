from __future__ import annotations

import json
import math

import pytest

from asian_options.contracts import CONTRACT_IDS, REFERENCE_ID, TARGET_IDS, make_contract_cfg
from scripts import run_stage8 as s8


def _base_success_row(rep: int, cid: str, method: str) -> dict:
    return {
        "base_seed": 42,
        "replication": rep,
        "contract_id": cid,
        "method": method,
        "price": 1.0,
        "observation_variance": 4.0,
        "estimator_variance": 0.04,
        "std_error": 0.2,
        "ci_lower": 0.5,
        "ci_upper": 1.5,
        "pricing_observations": 100,
        "pricing_simulated_paths": 100,
        "pilot_paths": 0,
        "pilot_runtime_s": 0.0,
        "target_training_paths": 0,
        "target_training_runtime_s": 0.0,
        "shared_reference_training_paths": 0,
        "shared_reference_training_runtime_s": 0.0,
        "training_paths": 0,
        "total_simulated_paths": 100,
        "pricing_runtime_s": 1.0,
        "marginal_runtime_s": 1.0,
        "standalone_runtime_s": 1.0,
        "beta": float("nan"),
        "corr_f_c0": float("nan"),
        "hash_verified": "",
        "error": "",
    }


def _fixture_rows(n_replications: int = 1) -> list[dict]:
    rows = []
    for rep in range(n_replications):
        for cid in CONTRACT_IDS:
            for method in s8.BASE_METHODS:
                row = _base_success_row(rep, cid, method)
                if method == "AV":
                    row["pricing_simulated_paths"] = 2 * row["pricing_observations"]
                if method == "GCV":
                    row["pilot_paths"] = 10
                    row["pilot_runtime_s"] = 0.2
                    row["marginal_runtime_s"] = 1.2
                    row["standalone_runtime_s"] = 1.2
                if method == "NCV_SCRATCH":
                    row["target_training_paths"] = 50
                    row["target_training_runtime_s"] = 2.0
                    row["training_paths"] = 50
                    row["standalone_runtime_s"] = 3.0
                rows.append(row)
        for cid in TARGET_IDS:
            tb1 = _base_success_row(rep, cid, "NCV_TRANSFER_BETA1")
            tb1["shared_reference_training_paths"] = 50
            tb1["shared_reference_training_runtime_s"] = 2.5
            tb1["target_training_paths"] = 0
            tb1["target_training_runtime_s"] = 0.0
            tb1["pilot_paths"] = 0
            tb1["pilot_runtime_s"] = 0.0
            tb1["marginal_runtime_s"] = tb1["pricing_runtime_s"]
            tb1["standalone_runtime_s"] = 3.5
            tb1["hash_verified"] = True
            rows.append(tb1)

            tb = _base_success_row(rep, cid, "NCV_TRANSFER_BETA")
            tb["shared_reference_training_paths"] = 50
            tb["shared_reference_training_runtime_s"] = 2.5
            tb["target_training_paths"] = 0
            tb["target_training_runtime_s"] = 0.0
            tb["pilot_paths"] = 10
            tb["pilot_runtime_s"] = 0.2
            tb["marginal_runtime_s"] = 1.2
            tb["standalone_runtime_s"] = 3.7
            tb["hash_verified"] = True
            rows.append(tb)
    return rows


def test_monitoring_dates_value_and_snapshot_field(tmp_path):
    cfg = make_contract_cfg(REFERENCE_ID, n_paths=10, seed=1)
    assert cfg.m == 252

    run_dir = s8.run_stage8(profile="smoke", base_seed=1, output_dir=str(tmp_path), n_replications_override=1)
    snap = json.loads((run_dir / "config_snapshot.json").read_text())
    assert snap["monitoring_dates"] == 252


def test_dissertation_profile_requires_torch(monkeypatch):
    monkeypatch.setattr(s8, "_try_import_torch", lambda: False)
    with pytest.raises(RuntimeError, match="requires PyTorch"):
        s8.run_stage8(profile="dissertation", base_seed=1, output_dir="/tmp/stage8_test_dissertation")


def test_all_expected_contract_method_replication_rows_required():
    rows = _fixture_rows(1)
    ok, failures = s8._all_required_rows_present(rows, n_replications=1)
    assert ok is True
    assert failures == []

    rows = rows[:-1]
    ok2, failures2 = s8._all_required_rows_present(rows, n_replications=1)
    assert ok2 is False
    assert failures2


def test_strict_success_check_rejects_failed_rows():
    rows = _fixture_rows(1)
    ok, failures = s8._check_required_success(rows, n_replications=1)
    assert ok is True
    assert failures == []

    rows[0]["error"] = "boom"
    ok2, failures2 = s8._check_required_success(rows, n_replications=1)
    assert ok2 is False
    assert failures2


def test_reference_training_row_per_replication():
    shared = []
    for rep in range(2):
        _, _, _, sr = s8.run_replication(
            base_seed=1,
            replication=rep,
            n_training=10,
            n_pilot=2,
            n_pricing=5,
            torch_available=False,
        )
        shared.append(sr)
    assert len(shared) == 2
    assert {r["replication"] for r in shared} == {0, 1}


def test_transfer_target_training_zero_and_costs():
    rows = _fixture_rows(1)
    for row in rows:
        if row["method"] == "NCV_TRANSFER_BETA1":
            assert row["target_training_paths"] == 0
            assert row["target_training_runtime_s"] == 0.0
            assert row["marginal_runtime_s"] == row["pricing_runtime_s"]
            assert row["standalone_runtime_s"] == row["shared_reference_training_runtime_s"] + row["marginal_runtime_s"]


def test_shared_reference_training_counted_once_in_equal_budget():
    agg = [{
        "contract_id": "strike_low", "method": "NCV_TRANSFER_BETA1", "mean_observation_variance": 4.0
    }]
    rows = s8.compute_equal_budget_projected_results(
        aggregate_rows=agg,
        n_training=5000,
        n_pilot=1000,
        budget=50000,
        q_values=[10],
    )
    row = next(r for r in rows if r["contract_id"] == "strike_low" and r["method"] == "NCV_TRANSFER_BETA1")
    assert row["total_paths_used"] == 5000 + 10 * row["pricing_simulated_paths"]
    assert row["total_paths_used"] <= 10 * 50000


def test_scratch_training_one_time_reuse_formula():
    alloc = s8._equal_budget_allocation("NCV_SCRATCH", B=50000, Q=10, n_training=5000, n_pilot=1000)
    expected = math.floor((10 * 50000 - 5000) / 10)
    assert alloc["pricing_observations"] == expected
    assert alloc["total_paths_used"] == 5000 + 10 * expected


def test_gcv_pilot_inside_budget_and_av_pair_paths():
    gcv = s8._equal_budget_allocation("GCV", B=50000, Q=1, n_training=5000, n_pilot=1000)
    assert gcv["pricing_observations"] == 49000
    assert gcv["total_paths_used"] == 50000

    av = s8._equal_budget_allocation("AV", B=50000, Q=1, n_training=5000, n_pilot=1000)
    assert av["pricing_simulated_paths"] == 2 * av["pricing_observations"]


def test_every_equal_budget_row_respects_total_budget():
    rows = s8.compute_equal_budget_projected_results(
        aggregate_rows=[],
        n_training=5000,
        n_pilot=1000,
        budget=50000,
        q_values=[1, 10, 100],
    )
    for r in rows:
        if isinstance(r.get("total_paths_used"), (int, float)):
            assert r["total_paths_used"] <= r["total_budget_paths"]


def test_per_replication_vrr_before_aggregation_and_log_ci_finite():
    rows = _fixture_rows(3)
    per_rep, summary = s8.compute_variance_ratio_summary(rows)
    assert per_rep

    valid = [r for r in summary if r["n_valid_replications"] > 0 and r["method"] == "AV" and r["comparator"] == "MC"]
    assert valid
    for r in valid:
        lo = r["log_vrr_ci95_lower"]
        hi = r["log_vrr_ci95_upper"]
        assert lo != "NA"
        assert hi != "NA"
        assert math.isfinite(float(lo))
        assert math.isfinite(float(hi))


def test_matched_accuracy_ceiling_formula():
    n, reason = s8._required_observations(obs_variance=10.0, target_se=0.1)
    assert reason == ""
    assert n == math.ceil(10.0 / (0.1 ** 2))


def test_combined_reference_uncertainty_calculation():
    agg = [{
        "contract_id": "reference",
        "method": "MC",
        "successful_replications": 4,
        "mean_price": 10.0,
        "empirical_std_across_replications": 2.0,
        "mean_reported_estimator_standard_error": 0.5,
    }]
    refs = [{"contract_id": "reference", "price": 9.5, "std_error": 0.3}]
    rows = s8.compute_reference_precision_diagnostics(agg, refs)
    row = rows[0]
    expected = math.sqrt((2.0 ** 2) / 4 + 0.3 ** 2)
    assert abs(row["se_combined"] - expected) < 1e-12


def test_break_even_verified_at_q_minus_1_and_q():
    out = s8._solve_break_even(initial_cost=100.0, baseline_marginal=10.0, proposed_marginal=5.0)
    assert out["break_even_q"] == 20
    assert out["verified_q_minus_1"] is True
    assert out["verified_q"] is True


def test_no_finite_break_even_has_reason():
    out = s8._solve_break_even(initial_cost=100.0, baseline_marginal=5.0, proposed_marginal=5.0)
    assert out["break_even_q"] == "NA"
    assert out["failure_reason"]


def test_runtime_summary_contains_required_statistics():
    rows = _fixture_rows(1)
    shared = [{"replication": 0, "training_runtime_s": 2.0, "error": ""}]
    summary, _ = s8._runtime_summary([], shared, rows)
    assert summary
    required = {"count", "mean", "std_dev", "median", "minimum", "maximum"}
    for r in summary:
        assert required.issubset(r.keys())


def test_portfolio_break_even_counts_shared_training_once():
    aggregate = []
    for cid in TARGET_IDS:
        aggregate.extend(
            [
                {"contract_id": cid, "method": "GCV", "marginal_runtime_s_median": 10.0},
                {"contract_id": cid, "method": "NCV_TRANSFER_BETA1", "marginal_runtime_s_median": 7.0},
                {"contract_id": cid, "method": "NCV_TRANSFER_BETA", "marginal_runtime_s_median": 8.0},
                {"contract_id": cid, "method": "NCV_SCRATCH", "marginal_runtime_s_median": 9.0, "target_training_runtime_s_median": 30.0},
            ]
        )
    matched = []
    shared = [{"training_runtime_s": 20.0, "error": ""}]
    _, _, _, portfolio = s8.compute_break_even_tables(aggregate, matched, shared)
    row = portfolio[0]
    assert row["portfolio"] == "six_target_contract_cycle"
    assert row["shared_reference_training_counted_once"] is True


def test_handover_torch_matches_environment_snapshot(tmp_path):
    run_dir = s8.run_stage8(profile="smoke", base_seed=11, output_dir=str(tmp_path), n_replications_override=1)
    env = json.loads((run_dir / "environment.json").read_text())
    handover = (run_dir / "handover.md").read_text()
    if env.get("torch_available"):
        assert "Torch: available" in handover
    else:
        assert "Torch: unavailable" in handover


def test_empirical_equal_budget_flag_optional_and_off_by_default():
    parser = s8._build_parser()
    args = parser.parse_args([])
    assert args.empirical_equal_budget is False
