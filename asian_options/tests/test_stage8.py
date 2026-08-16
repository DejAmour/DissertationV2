from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from scripts.run_stage8 import (
    CONTRACT_IDS,
    PROFILES,
    TARGET_IDS,
    _replication_seeds,
    compute_amortised_costs,
    compute_break_even,
    run_stage8,
)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists() or path.read_text().strip() == "":
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except Exception:
        return None


def test_profiles_valid():
    smoke = PROFILES["smoke"]
    diss = PROFILES["dissertation"]

    assert smoke["n_replications"] == 2
    assert diss["n_replications"] >= 30
    assert diss["n_training"] >= 10_000
    assert diss["n_pilot"] >= 2_000
    assert diss["n_pricing"] >= 10_000
    assert diss["n_high_precision"] >= 1_000_000
    assert diss["m_monitoring"] == 252
    assert diss["amortised_q_values"] == [1, 5, 10, 25, 50, 100, 250, 500, 1000]


def test_seed_independence_and_common_design():
    seeds = _replication_seeds(42, 0)
    all_vals = list(seeds.values())
    assert len(all_vals) == len(set(all_vals))

    pricing = [seeds[f"pricing_{cid}"] for cid in CONTRACT_IDS]
    pilot = [seeds[f"pilot_{cid}"] for cid in CONTRACT_IDS]
    train = [seeds[f"target_training_{cid}"] for cid in CONTRACT_IDS]

    assert len(pricing) == len(set(pricing))
    assert len(pilot) == len(set(pilot))
    assert len(train) == len(set(train))
    assert set(pricing).isdisjoint(set(pilot))
    assert set(pricing).isdisjoint(set(train))


def test_amortised_costs_shape():
    rows = compute_amortised_costs(10_000, 2_000, 10_000, [1, 5, 10])
    assert [int(r["Q"]) for r in rows] == [1, 5, 10]
    for r in rows:
        assert int(r["tb1_total_paths_for_Q"]) >= 0
        assert int(r["tb_total_paths_for_Q"]) >= 0


def test_break_even_formula_smoke():
    assert compute_break_even(100.0, 10.0, 5.0) == "20"
    assert compute_break_even(100.0, 5.0, 5.0) == "No finite break-even under the measured configuration."


def test_stage8_smoke_outputs_and_accounting(tmp_path):
    run_dir = run_stage8(profile="smoke", base_seed=7, output_dir=str(tmp_path), n_replications_override=1)

    expected_files = [
        "per_replication_results.csv",
        "aggregate_statistical_results.csv",
        "equal_observation_results.csv",
        "equal_budget_empirical_results.csv",
        "matched_accuracy_results.csv",
        "runtime_raw_results.csv",
        "runtime_summary.csv",
        "break_even_by_contract.csv",
        "portfolio_break_even.csv",
        "transfer_diagnostics.csv",
        "high_precision_references.csv",
        "seed_manifest.csv",
        "configuration_snapshot.json",
        "environment_snapshot.json",
        "validation_report.json",
        "reproducibility_report.json",
        "handover.md",
    ]
    for name in expected_files:
        assert (run_dir / name).exists(), f"missing {name}"

    per_rep = _read_csv(run_dir / "per_replication_results.csv")
    shared = _read_csv(run_dir / "shared_reference_training.csv")
    equal_budget = _read_csv(run_dir / "equal_budget_empirical_results.csv")
    matched = _read_csv(run_dir / "matched_accuracy_results.csv")
    break_even = _read_csv(run_dir / "break_even_by_contract.csv")
    agg = _read_csv(run_dir / "aggregate_statistical_results.csv")

    # Shared training row exactly once per replication
    assert len(shared) == 1

    # Transfer accounting checks on successful rows
    for r in per_rep:
        if r.get("method") not in ("NCV_TRANSFER_BETA1", "NCV_TRANSFER_BETA"):
            continue
        if r.get("error"):
            continue
        assert int(float(r["target_training_paths"])) == 0
        assert float(r["target_training_runtime_s"]) == 0.0
        e2e = float(r["end_to_end_runtime_s"])
        pricing = float(r["pricing_runtime_s"])
        pilot = float(r.get("pilot_runtime_s") or 0.0)
        assert abs(e2e - (pricing + pilot)) < 1e-8
        assert str(r.get("hash_verified", "")).lower() in ("true", "1")

    # Equal-budget accounting exact
    for r in equal_budget:
        if str(r.get("feasible", "")).lower() not in ("true", "1"):
            continue
        used = float(r["total_used_paths"])
        budget = float(r["declared_budget_paths"])
        assert used <= budget + 1e-9

    # Matched-accuracy sample size formula
    for r in matched:
        t = _to_float(r.get("target_standard_error"))
        v = _to_float(r.get("observation_variance_estimate"))
        n = r.get("required_pricing_observations")
        if t is None or v is None or t <= 0 or v <= 0 or n == "NA":
            continue
        expected = math.ceil(v / (t * t))
        assert int(n) == expected

    # Break-even verification contains q-1/q checks where finite
    for r in break_even:
        q = r.get("q_star", "")
        if q.isdigit():
            assert str(r.get("verified", "")).lower() in ("true", "1")

    # Aggregate contains required metrics
    required_cols = {
        "mean_estimated_price",
        "empirical_std_estimates",
        "mean_model_reported_standard_error",
        "observation_variance",
        "estimator_variance",
        "bias_vs_reference",
        "absolute_bias",
        "rmse",
        "mae",
        "ci95_coverage",
        "mean_ci_width",
        "variance_reduction_ratio_vs_mc_mean",
        "variance_reduction_ratio_vs_gcv_mean",
        "computational_efficiency_runtime",
        "computational_efficiency_paths",
        "n_successful_replications",
        "n_failed_replications",
    }
    assert agg, "aggregate file is empty"
    assert required_cols.issubset(set(agg[0].keys()))

    # failed rows must keep explicit error text
    failed = [r for r in per_rep if r.get("error")]
    for r in failed:
        assert str(r["error"]).strip() != ""

    # successful rows must be finite on core outputs
    for r in per_rep:
        if r.get("error"):
            continue
        for f in ("price", "observation_variance", "estimator_variance", "std_error", "pricing_runtime_s"):
            assert _to_float(r[f]) is not None


def test_smoke_reproducibility_hash(tmp_path):
    run1 = run_stage8(profile="smoke", base_seed=11, output_dir=str(tmp_path), n_replications_override=1)
    run2 = run_stage8(profile="smoke", base_seed=11, output_dir=str(tmp_path), n_replications_override=1)

    h1 = json.loads((run1 / "reproducibility_report.json").read_text())["stable_summary_sha256"]
    h2 = json.loads((run2 / "reproducibility_report.json").read_text())["stable_summary_sha256"]
    assert h1 == h2


def test_handover_torch_statement_consistent(tmp_path):
    run_dir = run_stage8(profile="smoke", base_seed=17, output_dir=str(tmp_path), n_replications_override=1)
    env = json.loads((run_dir / "environment_snapshot.json").read_text())
    handover = (run_dir / "handover.md").read_text()

    expected_phrase = f"Torch available: {env.get('torch_available')}"
    assert expected_phrase in handover


def test_transfer_outputs_for_all_targets(tmp_path):
    run_dir = run_stage8(profile="smoke", base_seed=23, output_dir=str(tmp_path), n_replications_override=1)
    diag = _read_csv(run_dir / "transfer_diagnostics.csv")
    present = {r["contract_id"] for r in diag if r.get("contract_id") in TARGET_IDS}
    assert present == set(TARGET_IDS)


def test_validation_report_present_and_structured(tmp_path):
    run_dir = run_stage8(profile="smoke", base_seed=29, output_dir=str(tmp_path), n_replications_override=1)
    report = json.loads((run_dir / "validation_report.json").read_text())
    assert set(report.keys()) >= {"passed", "n_failures", "n_warnings", "failures", "warnings"}
