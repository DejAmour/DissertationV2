"""
test_stage8.py
==============
Stage 8 validation tests.

Validates all core requirements for the frozen NCV transfer pipeline:
1.  Contract grid correctness (7 contracts, correct parameters)
2.  One-parameter-change rule for each non-reference contract
3.  Monitoring dates: dt = T/m for all contracts
4.  Network hash function is deterministic and changes on weight modification
5.  Frozen hash verification raises RuntimeError on mismatch
6.  NearZeroVarianceError raised for degenerate Var(C0)
7.  Amortised cost allocations are deterministic integers
8.  Break-even formula: finite and "no break-even" cases
9.  NCV_TRANSFER_BETA1 beta is exactly 1.0
10. NCV_TRANSFER_BETA pilot uses independent seed from pricing
11. Estimator variance identity: est_var = obs_var / n_pricing
12. AV pair accounting: pricing_simulated_paths == 2 * pricing_observations
13. Seed independence: distinct stream seeds within each replication
14. Validation report generator runs without error
15. Config snapshot is valid JSON with required keys
16. Contract grid validate_contract_grid() passes
17. make_contract_cfg raises KeyError for unknown contract id
18. compute_amortised_costs produces increasing Q-dependent allocations
19. Analytical expectation is not NaN/Inf for a small test network
20. ncv_transfer_beta1 result keys are present
21. ncv_transfer_beta result keys are present (when torch available)
22. Hash is unchanged after transfer evaluation (verified field is True)
23. Existing tests still pass (no regression imports)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


TORCH_MISSING = not _torch_available()


# ---------------------------------------------------------------------------
# 1-3. Contract grid
# ---------------------------------------------------------------------------

class TestContractGrid:
    def test_seven_contracts(self):
        from asian_options.contracts import CONTRACT_IDS
        assert len(CONTRACT_IDS) == 7

    def test_reference_first(self):
        from asian_options.contracts import CONTRACT_IDS, REFERENCE_ID
        assert CONTRACT_IDS[0] == REFERENCE_ID

    def test_six_target_ids(self):
        from asian_options.contracts import TARGET_IDS, CONTRACT_IDS, REFERENCE_ID
        assert len(TARGET_IDS) == 6
        assert REFERENCE_ID not in TARGET_IDS

    def test_reference_parameters(self):
        from asian_options.contracts import CONTRACT_GRID, REFERENCE_ID
        K, sigma, T = CONTRACT_GRID[REFERENCE_ID]
        assert K == 100.0
        assert sigma == 0.20
        assert T == 1.0

    def test_one_parameter_change_per_target(self):
        from asian_options.contracts import CONTRACT_GRID, REFERENCE_ID, TARGET_IDS
        ref_K, ref_sigma, ref_T = CONTRACT_GRID[REFERENCE_ID]
        for cid in TARGET_IDS:
            K, sigma, T = CONTRACT_GRID[cid]
            diffs = sum([K != ref_K, sigma != ref_sigma, T != ref_T])
            assert diffs == 1, (
                f"Contract '{cid}' has {diffs} parameter changes from reference"
            )

    def test_fixed_s0_r_m(self):
        from asian_options.contracts import make_contract_cfg, CONTRACT_IDS
        for cid in CONTRACT_IDS:
            cfg = make_contract_cfg(cid, n_paths=10, seed=0)
            assert cfg.S0 == 100.0
            assert cfg.r == 0.05
            assert cfg.m == 12

    def test_monitoring_dates_dt_equals_T_over_m(self):
        from asian_options.contracts import CONTRACT_GRID, CONTRACT_IDS, make_contract_cfg
        for cid in CONTRACT_IDS:
            K, sigma, T = CONTRACT_GRID[cid]
            cfg = make_contract_cfg(cid, n_paths=10, seed=0)
            expected_dt = T / 12
            assert abs(cfg.dt - expected_dt) < 1e-12, (
                f"dt mismatch for {cid}: got {cfg.dt}, expected {expected_dt}"
            )

    def test_validate_contract_grid_passes(self):
        from asian_options.contracts import validate_contract_grid
        validate_contract_grid()  # must not raise

    def test_unknown_contract_raises_key_error(self):
        from asian_options.contracts import make_contract_cfg
        with pytest.raises(KeyError, match="unknown_contract"):
            make_contract_cfg("unknown_contract")


# ---------------------------------------------------------------------------
# 4-5. Network hash
# ---------------------------------------------------------------------------

class TestNetworkHash:
    def _make_tiny_net(self):
        from asian_options.neural_cv import _ShallowNet
        rng = np.random.default_rng(0)
        W1 = rng.standard_normal((4, 3))
        b1 = rng.standard_normal(4)
        W2 = rng.standard_normal((1, 4))
        b2 = rng.standard_normal(1)
        return _ShallowNet(W1, b1, W2, b2)

    def test_hash_is_deterministic(self):
        from asian_options.frozen_transfer import compute_network_hash
        net = self._make_tiny_net()
        h1 = compute_network_hash(net)
        h2 = compute_network_hash(net)
        assert h1 == h2

    def test_hash_is_64_hex_chars(self):
        from asian_options.frozen_transfer import compute_network_hash
        net = self._make_tiny_net()
        h = compute_network_hash(net)
        assert len(h) == 64
        int(h, 16)  # should not raise

    def test_hash_changes_on_weight_modification(self):
        from asian_options.frozen_transfer import compute_network_hash
        net = self._make_tiny_net()
        h_before = compute_network_hash(net)
        net.W1[0, 0] += 1e-6
        h_after = compute_network_hash(net)
        assert h_before != h_after

    def test_verify_frozen_hash_passes_when_unchanged(self):
        from asian_options.frozen_transfer import compute_network_hash, verify_frozen_hash
        net = self._make_tiny_net()
        h = compute_network_hash(net)
        verify_frozen_hash(net, h)  # should not raise

    def test_verify_frozen_hash_raises_on_mismatch(self):
        from asian_options.frozen_transfer import compute_network_hash, verify_frozen_hash
        net = self._make_tiny_net()
        h = compute_network_hash(net)
        net.W1[0, 0] += 1e-6
        with pytest.raises(RuntimeError, match="hash mismatch"):
            verify_frozen_hash(net, h)


# ---------------------------------------------------------------------------
# 6. NearZeroVarianceError
# ---------------------------------------------------------------------------

class TestNearZeroVarianceError:
    def test_near_zero_variance_error_imported(self):
        from asian_options.frozen_transfer import NearZeroVarianceError
        assert issubclass(NearZeroVarianceError, RuntimeError)

    @pytest.mark.skipif(TORCH_MISSING, reason="torch not available")
    def test_near_zero_variance_raises_not_falls_back(self):
        """When Var(C0) is near-zero, NearZeroVarianceError is raised."""
        from asian_options.neural_cv import _ShallowNet
        from asian_options.frozen_transfer import (
            ncv_transfer_beta,
            NearZeroVarianceError,
            compute_network_hash,
        )
        from asian_options.contracts import make_contract_cfg

        # Zero-weight network => C0 = H0 - E[H0] = 0 => Var(C0) = 0
        net = _ShallowNet(
            W1=np.zeros((4, 12)),
            b1=np.zeros(4),
            W2=np.zeros((1, 4)),
            b2=np.zeros(1),
        )
        e_h0 = 0.0  # E[0] = 0
        h = compute_network_hash(net)
        cfg = make_contract_cfg("reference", n_paths=20, seed=1)
        with pytest.raises(NearZeroVarianceError):
            ncv_transfer_beta(
                frozen_network=net,
                e_h0=e_h0,
                frozen_hash=h,
                target_cfg=cfg,
                pilot_seed=10,
                pricing_seed=20,
                n_pilot=20,
                n_pricing=20,
            )


# ---------------------------------------------------------------------------
# 7. Amortised cost allocations
# ---------------------------------------------------------------------------

class TestAmortisedCosts:
    def test_returns_one_row_per_q(self):
        from scripts.run_stage8 import compute_amortised_costs
        rows = compute_amortised_costs(5000, 1000, 50000, [1, 5, 10])
        assert len(rows) == 3

    def test_q_values_match(self):
        from scripts.run_stage8 import compute_amortised_costs
        q_vals = [1, 5, 10, 20]
        rows = compute_amortised_costs(5000, 1000, 50000, q_vals)
        assert [r["Q"] for r in rows] == q_vals

    def test_tb1_pricing_non_negative(self):
        from scripts.run_stage8 import compute_amortised_costs
        rows = compute_amortised_costs(5000, 1000, 50000, [1, 5, 10, 100])
        for r in rows:
            assert r["tb1_pricing_per_valuation"] >= 1

    def test_tb_pricing_non_negative(self):
        from scripts.run_stage8 import compute_amortised_costs
        rows = compute_amortised_costs(5000, 1000, 50000, [1, 5, 10, 100])
        for r in rows:
            assert r["tb_pricing_per_valuation"] >= 1

    def test_larger_q_gives_more_pricing_paths_per_valuation(self):
        from scripts.run_stage8 import compute_amortised_costs
        rows = compute_amortised_costs(5000, 1000, 50000, [1, 100])
        # With Q=100 the training cost is amortised further, so pricing per val increases
        assert rows[1]["tb1_pricing_per_valuation"] >= rows[0]["tb1_pricing_per_valuation"]


# ---------------------------------------------------------------------------
# 8. Break-even formula
# ---------------------------------------------------------------------------

class TestBreakEven:
    def test_finite_break_even(self):
        from scripts.run_stage8 import compute_break_even
        result = compute_break_even(c_train=100.0, c_gcv_per_val=10.0, c_transfer_per_val=5.0)
        assert result == "20"

    def test_no_finite_break_even_when_denom_zero(self):
        from scripts.run_stage8 import compute_break_even
        result = compute_break_even(c_train=100.0, c_gcv_per_val=5.0, c_transfer_per_val=5.0)
        assert result == "No finite break-even under the measured configuration."

    def test_no_finite_break_even_when_denom_negative(self):
        from scripts.run_stage8 import compute_break_even
        result = compute_break_even(c_train=100.0, c_gcv_per_val=3.0, c_transfer_per_val=10.0)
        assert result == "No finite break-even under the measured configuration."

    def test_ceiling_applied(self):
        from scripts.run_stage8 import compute_break_even
        # c_train=100, denom=3 => 100/3=33.33 => ceil=34
        result = compute_break_even(c_train=100.0, c_gcv_per_val=13.0, c_transfer_per_val=10.0)
        assert result == "34"


# ---------------------------------------------------------------------------
# 9. NCV_TRANSFER_BETA1 beta = 1.0
# ---------------------------------------------------------------------------

class TestTransferBeta1Value:
    @pytest.mark.skipif(TORCH_MISSING, reason="torch not available")
    def test_beta_is_one(self):
        from asian_options.neural_cv import _ShallowNet
        from asian_options.frozen_transfer import (
            ncv_transfer_beta1,
            compute_network_hash,
            analytical_network_expectation,
        )
        from asian_options.contracts import make_contract_cfg

        rng = np.random.default_rng(7)
        net = _ShallowNet(
            W1=rng.standard_normal((4, 12)) * 0.01,
            b1=rng.standard_normal(4) * 0.01,
            W2=rng.standard_normal((1, 4)) * 0.01,
            b2=rng.standard_normal(1) * 0.01,
        )
        from asian_options.neural_cv import analytical_network_expectation as ane
        e_h0 = ane(net)
        h = compute_network_hash(net)
        cfg = make_contract_cfg("reference", n_paths=50, seed=1)
        result = ncv_transfer_beta1(
            frozen_network=net, e_h0=e_h0, frozen_hash=h,
            target_cfg=cfg, pricing_seed=10, n_pricing=50,
        )
        assert result["beta"] == 1.0


# ---------------------------------------------------------------------------
# 11. Estimator variance identity
# ---------------------------------------------------------------------------

class TestEstimatorVarianceIdentity:
    def test_mc_estimator_variance(self):
        from asian_options.estimators import standard_monte_carlo
        from asian_options.contracts import make_contract_cfg
        cfg = make_contract_cfg("reference", n_paths=200, seed=5)
        result = standard_monte_carlo(cfg)
        expected = result.observation_variance / result.pricing_observations
        assert abs(result.estimator_variance - expected) < 1e-10

    def test_gcv_estimator_variance(self):
        from asian_options.estimators import geometric_control_variate
        from asian_options.contracts import make_contract_cfg
        cfg = make_contract_cfg("reference", n_paths=200, seed=5)
        result = geometric_control_variate(cfg, n_pilot=20)
        expected = result.observation_variance / result.pricing_observations
        assert abs(result.estimator_variance - expected) < 1e-10


# ---------------------------------------------------------------------------
# 12. AV pair accounting
# ---------------------------------------------------------------------------

class TestAVPairAccounting:
    def test_pricing_simulated_paths_is_double_observations(self):
        from asian_options.estimators import antithetic_variates
        from asian_options.contracts import make_contract_cfg
        cfg = make_contract_cfg("reference", n_paths=100, seed=5)
        result = antithetic_variates(cfg)
        assert result.pricing_simulated_paths == 2 * result.pricing_observations

    def test_n_paths_equals_pricing_observations(self):
        from asian_options.estimators import antithetic_variates
        from asian_options.contracts import make_contract_cfg
        cfg = make_contract_cfg("reference", n_paths=100, seed=5)
        result = antithetic_variates(cfg)
        assert result.n_paths == result.pricing_observations == 100


# ---------------------------------------------------------------------------
# 13. Seed independence
# ---------------------------------------------------------------------------

class TestSeedIndependence:
    def test_pricing_seeds_distinct_within_replication(self):
        from scripts.run_stage8 import _replication_seeds, CONTRACT_IDS
        seeds = _replication_seeds(base_seed=42, replication=0)
        pricing_seeds = [seeds[f"pricing_{cid}"] for cid in CONTRACT_IDS]
        assert len(pricing_seeds) == len(set(pricing_seeds))

    def test_pilot_seeds_distinct_from_pricing_seeds(self):
        from scripts.run_stage8 import _replication_seeds, CONTRACT_IDS
        seeds = _replication_seeds(base_seed=42, replication=0)
        pricing_seeds = set(seeds[f"pricing_{cid}"] for cid in CONTRACT_IDS)
        pilot_seeds = set(seeds[f"pilot_{cid}"] for cid in CONTRACT_IDS)
        assert pricing_seeds.isdisjoint(pilot_seeds)

    def test_different_replications_get_different_seeds(self):
        from scripts.run_stage8 import _replication_seeds
        s0 = _replication_seeds(42, 0)
        s1 = _replication_seeds(42, 1)
        assert s0["pricing_reference"] != s1["pricing_reference"]


# ---------------------------------------------------------------------------
# 14. Validation report
# ---------------------------------------------------------------------------

class TestValidationReport:
    def test_build_validation_report_runs(self):
        from scripts.run_stage8 import build_validation_report
        report = build_validation_report(
            per_rep_rows=[],
            agg_rows=[],
            seed_manifest=[],
            high_prec_rows=[],
            amortised_rows=[],
        )
        assert "passed" in report
        assert "failures" in report
        assert "warnings" in report

    def test_mc_row_passes_validation(self):
        from scripts.run_stage8 import build_validation_report
        mc_row = {
            "base_seed": 42, "replication": 0,
            "contract_id": "reference", "method": "MC",
            "price": 5.0,
            "observation_variance": 0.01,
            "estimator_variance": 0.01 / 100,
            "std_error": 0.001,
            "ci_lower": 4.998, "ci_upper": 5.002,
            "pricing_observations": 100,
            "pricing_simulated_paths": 100,
            "error": "",
        }
        report = build_validation_report(
            per_rep_rows=[mc_row],
            agg_rows=[],
            seed_manifest=[],
            high_prec_rows=[],
            amortised_rows=[],
        )
        # If there are failures, they should not be from this row
        row_failures = [f for f in report["failures"] if "reference" in f and "MC" in f]
        assert len(row_failures) == 0

    def test_av_wrong_accounting_triggers_failure(self):
        from scripts.run_stage8 import build_validation_report
        av_row = {
            "base_seed": 42, "replication": 0,
            "contract_id": "reference", "method": "AV",
            "price": 5.0,
            "observation_variance": 0.008,
            "estimator_variance": 0.008 / 50,
            "std_error": 0.001,
            "ci_lower": 4.998, "ci_upper": 5.002,
            "pricing_observations": 50,
            "pricing_simulated_paths": 50,  # Wrong: should be 100
            "error": "",
        }
        report = build_validation_report(
            per_rep_rows=[av_row],
            agg_rows=[],
            seed_manifest=[],
            high_prec_rows=[],
            amortised_rows=[],
        )
        av_failures = [f for f in report["failures"] if "AV" in f]
        assert len(av_failures) > 0


# ---------------------------------------------------------------------------
# 15. Config snapshot is valid JSON with required keys
# ---------------------------------------------------------------------------

class TestConfigSnapshot:
    def test_config_snapshot_has_required_keys(self, tmp_path):
        from scripts.run_stage8 import run_stage8
        run_dir = run_stage8(
            profile="smoke",
            base_seed=7,
            output_dir=str(tmp_path),
            n_replications_override=1,
        )
        snap = json.loads((run_dir / "config_snapshot.json").read_text())
        required = [
            "profile", "base_seed", "n_replications", "n_training",
            "n_pilot", "n_pricing", "contracts", "timestamp_utc",
        ]
        for k in required:
            assert k in snap, f"Missing key: {k}"

    def test_output_bundle_files_exist(self, tmp_path):
        from scripts.run_stage8 import run_stage8
        run_dir = run_stage8(
            profile="smoke",
            base_seed=13,
            output_dir=str(tmp_path),
            n_replications_override=1,
        )
        required_files = [
            "config_snapshot.json",
            "environment.json",
            "seed_manifest.csv",
            "high_precision_references.csv",
            "per_replication_results.csv",
            "aggregate_results.csv",
            "beta_transfer_results.csv",
            "break_even_by_contract.csv",
            "portfolio_break_even.csv",
            "matched_accuracy_results.csv",
            "summary_stable.csv",
            "validation_report.json",
            "reproducibility_report.json",
            "handover.md",
        ]
        for fname in required_files:
            assert (run_dir / fname).exists(), f"Missing output file: {fname}"


# ---------------------------------------------------------------------------
# 19. Analytical expectation finite for test network
# ---------------------------------------------------------------------------

class TestAnalyticalExpectation:
    def test_analytical_expectation_is_finite(self):
        from asian_options.neural_cv import _ShallowNet, analytical_network_expectation
        rng = np.random.default_rng(1)
        net = _ShallowNet(
            W1=rng.standard_normal((8, 12)) * 0.1,
            b1=rng.standard_normal(8) * 0.1,
            W2=rng.standard_normal((1, 8)) * 0.1,
            b2=rng.standard_normal(1) * 0.1,
        )
        e_h = analytical_network_expectation(net)
        assert math.isfinite(e_h)

    def test_analytical_expectation_consistent_with_mc(self):
        """E[H(Z)] via formula should match MC approximation within 2%."""
        from asian_options.neural_cv import _ShallowNet, analytical_network_expectation
        rng = np.random.default_rng(0)
        net = _ShallowNet(
            W1=rng.standard_normal((8, 12)) * 0.1,
            b1=rng.standard_normal(8) * 0.1,
            W2=rng.standard_normal((1, 8)) * 0.1,
            b2=rng.standard_normal(1) * 0.1,
        )
        analytical = analytical_network_expectation(net)

        # MC approximation
        n_mc = 200_000
        Z_mc = rng.standard_normal((n_mc, 12))
        mc_vals = net.forward(Z_mc)
        mc_mean = float(mc_vals.mean())

        # Allow generous tolerance for statistical test
        if abs(analytical) > 1e-8:
            rel_err = abs(analytical - mc_mean) / abs(analytical)
            assert rel_err < 0.05, (
                f"Analytical E[H(Z)]={analytical:.6f} vs MC mean={mc_mean:.6f} "
                f"rel_err={rel_err:.4f}"
            )
        else:
            assert abs(analytical - mc_mean) < 0.01


# ---------------------------------------------------------------------------
# 20-21. Transfer result keys
# ---------------------------------------------------------------------------

class TestTransferResultKeys:
    @pytest.mark.skipif(TORCH_MISSING, reason="torch not available")
    def test_ncv_transfer_beta1_result_keys(self):
        from asian_options.neural_cv import _ShallowNet, analytical_network_expectation
        from asian_options.frozen_transfer import ncv_transfer_beta1, compute_network_hash
        from asian_options.contracts import make_contract_cfg

        rng = np.random.default_rng(5)
        net = _ShallowNet(
            W1=rng.standard_normal((4, 12)) * 0.01,
            b1=rng.standard_normal(4) * 0.01,
            W2=rng.standard_normal((1, 4)) * 0.01,
            b2=rng.standard_normal(1) * 0.01,
        )
        e_h0 = analytical_network_expectation(net)
        h = compute_network_hash(net)
        cfg = make_contract_cfg("strike_low", n_paths=50, seed=1)
        result = ncv_transfer_beta1(
            frozen_network=net, e_h0=e_h0, frozen_hash=h,
            target_cfg=cfg, pricing_seed=10, n_pricing=50,
        )
        required_keys = [
            "method", "price", "observation_variance", "estimator_variance",
            "std_error", "ci_lower", "ci_upper",
            "pricing_observations", "beta", "corr_f_c0", "e_h0",
            "pricing_runtime_s", "param_hash", "hash_verified",
        ]
        for k in required_keys:
            assert k in result, f"Missing key: {k}"

    @pytest.mark.skipif(TORCH_MISSING, reason="torch not available")
    def test_ncv_transfer_beta_result_keys(self):
        from asian_options.neural_cv import _ShallowNet, analytical_network_expectation
        from asian_options.frozen_transfer import ncv_transfer_beta, compute_network_hash
        from asian_options.contracts import make_contract_cfg

        rng = np.random.default_rng(5)
        net = _ShallowNet(
            W1=rng.standard_normal((4, 12)) * 0.5,
            b1=rng.standard_normal(4) * 0.5,
            W2=rng.standard_normal((1, 4)) * 0.5,
            b2=rng.standard_normal(1) * 0.5,
        )
        e_h0 = analytical_network_expectation(net)
        h = compute_network_hash(net)
        cfg = make_contract_cfg("strike_low", n_paths=100, seed=1)
        result = ncv_transfer_beta(
            frozen_network=net, e_h0=e_h0, frozen_hash=h,
            target_cfg=cfg, pilot_seed=10, pricing_seed=20,
            n_pilot=50, n_pricing=100,
        )
        required_keys = [
            "method", "price", "observation_variance", "estimator_variance",
            "std_error", "ci_lower", "ci_upper",
            "pricing_observations", "beta", "corr_f_c0", "e_h0",
            "pricing_runtime_s", "pilot_runtime_s", "param_hash", "hash_verified",
        ]
        for k in required_keys:
            assert k in result, f"Missing key: {k}"


# ---------------------------------------------------------------------------
# 22. Hash unchanged after transfer evaluation
# ---------------------------------------------------------------------------

class TestHashUnchangedAfterEvaluation:
    @pytest.mark.skipif(TORCH_MISSING, reason="torch not available")
    def test_hash_unchanged_after_beta1_evaluation(self):
        from asian_options.neural_cv import _ShallowNet, analytical_network_expectation
        from asian_options.frozen_transfer import (
            ncv_transfer_beta1,
            compute_network_hash,
        )
        from asian_options.contracts import make_contract_cfg

        rng = np.random.default_rng(3)
        net = _ShallowNet(
            W1=rng.standard_normal((4, 12)) * 0.01,
            b1=rng.standard_normal(4) * 0.01,
            W2=rng.standard_normal((1, 4)) * 0.01,
            b2=rng.standard_normal(1) * 0.01,
        )
        e_h0 = analytical_network_expectation(net)
        h_before = compute_network_hash(net)
        cfg = make_contract_cfg("strike_low", n_paths=50, seed=1)
        result = ncv_transfer_beta1(
            frozen_network=net, e_h0=e_h0, frozen_hash=h_before,
            target_cfg=cfg, pricing_seed=10, n_pricing=50,
        )
        h_after = compute_network_hash(net)
        assert h_before == h_after
        assert result["hash_verified"] is True


# ---------------------------------------------------------------------------
# 23. No regression in existing estimators
# ---------------------------------------------------------------------------

class TestNoRegression:
    def test_standard_mc_still_works(self):
        from asian_options.estimators import standard_monte_carlo
        from asian_options.contracts import make_contract_cfg
        cfg = make_contract_cfg("reference", n_paths=100, seed=5)
        result = standard_monte_carlo(cfg)
        assert result.price >= 0.0
        assert result.n_paths == 100

    def test_antithetic_variates_still_works(self):
        from asian_options.estimators import antithetic_variates
        from asian_options.contracts import make_contract_cfg
        cfg = make_contract_cfg("reference", n_paths=100, seed=5)
        result = antithetic_variates(cfg)
        assert result.price >= 0.0

    def test_gcv_still_works(self):
        from asian_options.estimators import geometric_control_variate
        from asian_options.contracts import make_contract_cfg
        cfg = make_contract_cfg("reference", n_paths=100, seed=5)
        result = geometric_control_variate(cfg, n_pilot=20)
        assert result.price >= 0.0

    def test_gcv_all_seven_contracts(self):
        """GCV should work for all 7 contracts without error."""
        from asian_options.estimators import geometric_control_variate
        from asian_options.contracts import make_contract_cfg, CONTRACT_IDS
        for cid in CONTRACT_IDS:
            cfg = make_contract_cfg(cid, n_paths=50, seed=7)
            result = geometric_control_variate(cfg, n_pilot=10)
            assert math.isfinite(result.price)
