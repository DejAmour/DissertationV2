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
        "ncv_epoch": 25 if method.startswith("NCV") else "NA",
        "ncv_epoch_source": "training_curve_validation_tuning" if method.startswith("NCV") else "NA",
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


def test_stage8_fixed_epoch_and_source_constants():
    assert s8.STAGE8_FIXED_NCV_EPOCH == 25
    assert s8.STAGE8_NCV_EPOCH_SOURCE == "training_curve_validation_tuning"
    row = s8._replication_base_row(42, 0, "reference", "NCV_SCRATCH")
    assert row["ncv_epoch"] == 25
    assert row["ncv_epoch_source"] == "training_curve_validation_tuning"


def test_training_curve_and_final_seed_namespaces_are_disjoint():
    assert s8.seed_namespaces_are_disjoint(base_seed=42, replication=0) is True
    assert s8.seed_namespaces_are_disjoint(base_seed=42, replication=3) is True


def test_matched_accuracy_cost_fields_use_end_to_end_pricing_and_setup_once():
    aggregate = [
        {"contract_id": "reference", "method": "MC", "mean_observation_variance": 4.0, "pricing_observations_mean": 100, "pricing_runtime_s_median": 10.0},
        {"contract_id": "reference", "method": "AV", "mean_observation_variance": 2.0, "pricing_observations_mean": 100, "pricing_runtime_s_median": 12.0},
        {"contract_id": "reference", "method": "GCV", "mean_observation_variance": 1.0, "mean_reported_estimator_standard_error": 0.1, "pricing_observations_mean": 100, "pilot_runtime_s_median": 5.0, "pricing_runtime_s_median": 4.0},
        {"contract_id": "reference", "method": "NCV_SCRATCH", "mean_observation_variance": 1.5, "mean_reported_estimator_standard_error": 0.12, "pricing_observations_mean": 100, "pricing_runtime_s_median": 3.0, "target_training_runtime_s_median": 7.0, "ncv_setup_cost_s_median": 7.0},
    ]
    per_rep = [
        {"contract_id": "reference", "method": "GCV", "pricing_runtime_s": 4.0, "pricing_observations": 100, "error": ""},
        {"contract_id": "reference", "method": "NCV_SCRATCH", "pricing_runtime_s": 3.0, "pricing_observations": 100, "error": ""},
        {"contract_id": "reference", "method": "MC", "pricing_runtime_s": 10.0, "pricing_observations": 100, "error": ""},
        {"contract_id": "reference", "method": "AV", "pricing_runtime_s": 12.0, "pricing_observations": 100, "error": ""},
    ]
    rows = s8.compute_matched_accuracy_results(aggregate, per_rep, n_training=5000, n_pilot=1000, shared_train_runtime_median=2.0)
    gcv_row = next(r for r in rows if r["contract_id"] == "reference" and r["method"] == "GCV" and r["target_definition"] == "fixed_se_0.001")
    assert gcv_row["cost_scope"] == "end_to_end"
    assert gcv_row["setup_cost_s"] == 5.0
    assert gcv_row["marginal_pricing_cost_s"] == gcv_row["projected_pricing_runtime_s_median"]
    assert gcv_row["projected_total_cost_s"] == gcv_row["setup_cost_s"] + gcv_row["Q"] * gcv_row["marginal_pricing_cost_s"]
    ncv_row = next(r for r in rows if r["contract_id"] == "reference" and r["method"] == "NCV_SCRATCH" and r["target_definition"] == "fixed_se_0.001")
    assert ncv_row["setup_cost_s"] == 7.0
    assert ncv_row["ncv_setup_cost_s"] == 7.0
    assert "training-data generation + optimizer runtime" in ncv_row["setup_reuse_assumption"]
    assert ncv_row["runtime_projection_is_empirical_or_projected"] == "projected_from_empirical_single_n"


def test_matched_accuracy_prefers_explicit_ncv_setup_cost_over_legacy_training_field():
    aggregate = [
        {"contract_id": "reference", "method": "NCV_SCRATCH", "mean_observation_variance": 1.0, "pricing_observations_mean": 100, "pricing_runtime_s_median": 2.0, "target_training_runtime_s_median": 3.0, "ncv_setup_cost_s_median": 9.0},
        {"contract_id": "reference", "method": "GCV", "mean_observation_variance": 1.0, "mean_reported_estimator_standard_error": 0.1, "pricing_observations_mean": 100, "pilot_runtime_s_median": 1.0, "pricing_runtime_s_median": 2.0},
    ]
    per_rep = [
        {"contract_id": "reference", "method": "NCV_SCRATCH", "pricing_runtime_s": 2.0, "pricing_observations": 100, "error": ""},
        {"contract_id": "reference", "method": "GCV", "pricing_runtime_s": 2.0, "pricing_observations": 100, "error": ""},
    ]
    rows = s8.compute_matched_accuracy_results(aggregate, per_rep, n_training=10, n_pilot=5, shared_train_runtime_median=0.0)
    ncv_row = next(r for r in rows if r["contract_id"] == "reference" and r["method"] == "NCV_SCRATCH" and r["target_definition"] == "fixed_se_0.001")
    assert ncv_row["setup_cost_s"] == 9.0
    assert ncv_row["ncv_setup_cost_s"] == 9.0


def test_matched_accuracy_target_definitions_include_fixed_and_gcv_matched():
    aggregate = [{"contract_id": "reference", "method": "GCV", "mean_observation_variance": 1.0, "mean_reported_estimator_standard_error": 0.1, "pricing_observations_mean": 100}]
    per_rep = [{"contract_id": "reference", "method": "GCV", "pricing_runtime_s": 1.0, "pricing_observations": 100, "error": ""}]
    rows = s8.compute_matched_accuracy_results(aggregate, per_rep, n_training=10, n_pilot=5, shared_train_runtime_median=0.0)
    targets = {r["target_definition"] for r in rows}
    assert "fixed_se_0.001" in targets
    assert "match_gcv_50000_se" in targets


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
    assert out["q_minus_1_verification_status"] == "verified_against_q_minus_1"


def test_break_even_q_equals_one_uses_minimum_boundary_verification():
    out = s8._solve_break_even(initial_cost=2.0, baseline_marginal=10.0, proposed_marginal=5.0)
    assert out["break_even_q"] == 1
    assert out["verified_q"] is True
    assert out["verified_q_minus_1"] is True
    assert out["failure_reason"] == ""
    assert out["baseline_setup_cost_s"] == 0.0
    assert out["proposed_setup_cost_s"] == 2.0
    assert out["q_minus_1_verification_status"] == "not_applicable_minimum_q_boundary"
    assert out["cost_baseline_q_minus_1"] == "NA"
    assert out["cost_proposed_q_minus_1"] == "NA"
    assert out["cost_proposed_q"] == 7.0
    assert out["cost_baseline_q"] == 10.0


def test_no_finite_break_even_has_reason():
    out = s8._solve_break_even(initial_cost=100.0, baseline_marginal=5.0, proposed_marginal=5.0)
    assert out["break_even_q"] == "NA"
    assert out["failure_reason"]


def test_break_even_handles_float_ratio_just_above_integer():
    out = s8._solve_break_even(
        initial_cost=40.00000000000004,
        baseline_marginal=10.0,
        proposed_marginal=5.0,
    )
    assert out["break_even_q"] == 8
    assert out["verified_q"] is True
    assert out["verified_q_minus_1"] is True


def test_break_even_equal_marginal_cases():
    out_ok = s8._solve_break_even(
        initial_cost=5.0,
        baseline_setup_cost=6.0,
        proposed_setup_cost=5.0,
        baseline_marginal=10.0,
        proposed_marginal=10.0,
    )
    assert out_ok["break_even_q"] == 1
    assert out_ok["verified_q"] is True
    assert out_ok["verified_q_minus_1"] is True
    assert out_ok["q_minus_1_verification_status"] == "not_applicable_minimum_q_boundary"

    out_na = s8._solve_break_even(
        initial_cost=7.0,
        baseline_setup_cost=6.0,
        proposed_setup_cost=7.0,
        baseline_marginal=10.0,
        proposed_marginal=10.0,
    )
    assert out_na["break_even_q"] == "NA"
    assert out_na["failure_reason"] == "equal_marginal_proposed_setup_above_baseline"


def test_break_even_proposed_marginal_above_baseline_is_no_finite_break_even():
    out = s8._solve_break_even(
        initial_cost=1.0,
        baseline_setup_cost=0.0,
        proposed_setup_cost=1.0,
        baseline_marginal=5.0,
        proposed_marginal=6.0,
    )
    assert out["break_even_q"] == "NA"
    assert out["failure_reason"] == "proposed_marginal_above_baseline_no_long_run_break_even"


def test_break_even_proposed_setup_below_baseline_returns_q_one():
    out = s8._solve_break_even(
        initial_cost=1.0,
        baseline_setup_cost=4.0,
        proposed_setup_cost=1.0,
        baseline_marginal=10.0,
        proposed_marginal=9.0,
    )
    assert out["break_even_q"] == 1
    assert out["verified_q"] is True
    assert out["verified_q_minus_1"] is True


def test_break_even_missing_or_non_finite_inputs_are_rejected():
    out_missing = s8._solve_break_even(
        initial_cost=None,
        baseline_marginal=10.0,
        proposed_marginal=5.0,
    )
    assert out_missing["break_even_q"] == "NA"
    assert out_missing["failure_reason"] == "missing_or_non_finite_runtime_input"

    out_non_finite = s8._solve_break_even(
        initial_cost=2.0,
        baseline_marginal=float("nan"),
        proposed_marginal=5.0,
    )
    assert out_non_finite["break_even_q"] == "NA"
    assert out_non_finite["failure_reason"] == "missing_or_non_finite_runtime_input"


def test_runtime_summary_contains_required_statistics():
    rows = _fixture_rows(1)
    shared = [{"replication": 0, "training_runtime_s": 2.0, "error": ""}]
    summary, _ = s8._runtime_summary([], shared, rows)
    assert summary
    required = {"count", "mean", "std_dev", "median", "minimum", "maximum"}
    for r in summary:
        assert required.issubset(r.keys())


@pytest.mark.parametrize("field", ["pricing_runtime_s", "observation_variance", "estimator_variance"])
def test_validation_report_fails_on_non_finite_runtime_or_variance(field):
    rows = _fixture_rows(1)
    rows[0][field] = float("nan")
    seeds = s8._build_seed_manifest(base_seed=42, n_replications=1)
    report = s8._build_validation_report(rows, seeds, n_replications=1, base_seed=42)
    assert report["passed"] is False
    assert any(f"non-finite {field}" in x for x in report["failures"])


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


def test_break_even_changes_when_ncv_setup_cost_changes():
    aggregate_low = []
    aggregate_high = []
    for cid in TARGET_IDS:
        aggregate_low.extend(
            [
                {
                    "contract_id": cid,
                    "method": "GCV",
                    "marginal_runtime_s_median": 10.0,
                    "pilot_runtime_s_median": 0.0,
                    "pricing_observations_mean": 100,
                    "pricing_runtime_s_median": 10.0,
                },
                {
                    "contract_id": cid,
                    "method": "NCV_SCRATCH",
                    "marginal_runtime_s_median": 5.0,
                    "target_training_runtime_s_median": 2.0,
                    "ncv_setup_cost_s_median": 2.0,
                    "pricing_observations_mean": 100,
                    "pricing_runtime_s_median": 5.0,
                },
            ]
        )
        aggregate_high.extend(
            [
                {
                    "contract_id": cid,
                    "method": "GCV",
                    "marginal_runtime_s_median": 10.0,
                    "pilot_runtime_s_median": 0.0,
                    "pricing_observations_mean": 100,
                    "pricing_runtime_s_median": 10.0,
                },
                {
                    "contract_id": cid,
                    "method": "NCV_SCRATCH",
                    "marginal_runtime_s_median": 5.0,
                    "target_training_runtime_s_median": 2.0,
                    "ncv_setup_cost_s_median": 20.0,
                    "pricing_observations_mean": 100,
                    "pricing_runtime_s_median": 5.0,
                },
            ]
        )
    be_low, _, _, _ = s8.compute_break_even_tables(aggregate_low, [], [])
    be_high, _, _, _ = s8.compute_break_even_tables(aggregate_high, [], [])
    row_low = next(r for r in be_low if r["contract_id"] == TARGET_IDS[0] and r["method"] == "NCV_SCRATCH")
    row_high = next(r for r in be_high if r["contract_id"] == TARGET_IDS[0] and r["method"] == "NCV_SCRATCH")
    assert row_low["break_even_q"] == 1
    assert row_low["verified_q"] is True
    assert row_low["verified_q_minus_1"] is True
    assert row_low["failure_reason"] == ""
    assert row_low["q_minus_1_verification_status"] == "not_applicable_minimum_q_boundary"
    assert row_low["baseline_setup_cost_s"] == 0.0
    assert row_low["proposed_setup_cost_s"] == 2.0
    assert row_low["cost_proposed_q"] == 7.0
    assert row_low["cost_baseline_q"] == 10.0
    assert row_high["break_even_q"] == 4
    assert row_high["verified_q"] is True
    assert row_high["verified_q_minus_1"] is True
    assert row_high["baseline_setup_cost_s"] == 0.0
    assert row_high["proposed_setup_cost_s"] == 20.0
    assert row_high["cost_proposed_q_minus_1"] == 35.0
    assert row_high["cost_baseline_q_minus_1"] == 30.0
    assert row_high["cost_proposed_q"] == 40.0
    assert row_high["cost_baseline_q"] == 40.0
    assert row_high["failure_reason"] == ""


def test_break_even_uses_non_zero_gcv_setup_cost():
    aggregate = [
        {
            "contract_id": TARGET_IDS[0],
            "method": "GCV",
            "marginal_runtime_s_median": 10.0,
            "pilot_runtime_s_median": 3.0,
        },
        {
            "contract_id": TARGET_IDS[0],
            "method": "NCV_SCRATCH",
            "marginal_runtime_s_median": 5.0,
            "ncv_setup_cost_s_median": 20.0,
        },
    ]
    for cid in TARGET_IDS[1:]:
        aggregate.extend(
            [
                {"contract_id": cid, "method": "GCV", "marginal_runtime_s_median": 10.0, "pilot_runtime_s_median": 3.0},
                {"contract_id": cid, "method": "NCV_SCRATCH", "marginal_runtime_s_median": 5.0, "ncv_setup_cost_s_median": 20.0},
            ]
        )
    rows, _, _, _ = s8.compute_break_even_tables(aggregate, [], [])
    row = next(r for r in rows if r["contract_id"] == TARGET_IDS[0] and r["method"] == "NCV_SCRATCH")
    assert row["baseline_setup_cost_s"] == 3.0
    assert row["break_even_q"] == 4


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
