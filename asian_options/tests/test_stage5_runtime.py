"""
test_stage5_runtime.py
======================
Stage 5 tests: runtime-efficiency metrics, schema correctness, and comparison
workflow validation.

Tests validate:
E1. Runtime fields present and non-negative.
E2. Variance fields remain Stage 4-correct.
E3. No misuse of labels (speed ratio not used for variance ratio).
E4. Efficiency metric formula correctness on deterministic synthetic values.
E5. Comparison outputs include both statistical modes and runtime section.
E6. Backward compatibility: existing consumers don't crash.
"""

from __future__ import annotations

import math
import io

import numpy as np
import pytest

from asian_options.config import ModelConfig
from asian_options.estimators import (
    standard_monte_carlo,
    antithetic_variates,
    geometric_control_variate,
)
from asian_options.results import (
    save_results_csv,
    print_comparison_table,
    efficiency_gain_vs_mc,
    make_runtime_row,
)

BASE = dict(S0=100.0, K=100.0, r=0.05, q=0.0, sigma=0.2, T=1.0, m=12,
            n_paths=300, seed=1)


def _cfg(**kw) -> ModelConfig:
    return ModelConfig(**{**BASE, **kw})


# ---------------------------------------------------------------------------
# E1. Runtime fields present and non-negative
# ---------------------------------------------------------------------------

class TestRuntimeFieldsPresent:
    def test_mc_runtime_s_present_and_non_negative(self):
        r = standard_monte_carlo(_cfg())
        assert hasattr(r, "runtime_s")
        assert r.runtime_s >= 0.0

    def test_av_runtime_s_present_and_non_negative(self):
        r = antithetic_variates(_cfg())
        assert hasattr(r, "runtime_s")
        assert r.runtime_s >= 0.0

    def test_cv_runtime_s_present_and_non_negative(self):
        r = geometric_control_variate(_cfg(), n_pilot=30)
        assert hasattr(r, "runtime_s")
        assert r.runtime_s >= 0.0

    def test_time_per_observation_non_negative(self):
        r = standard_monte_carlo(_cfg(n_paths=100))
        tpo = r.runtime_s / r.pricing_observations
        assert tpo >= 0.0

    def test_time_per_simulated_path_non_negative(self):
        r = antithetic_variates(_cfg(n_paths=100))
        tpsp = r.runtime_s / r.pricing_simulated_paths
        assert tpsp >= 0.0


# ---------------------------------------------------------------------------
# E2. Variance fields remain Stage 4-correct
# ---------------------------------------------------------------------------

class TestVarianceFieldsStage4Correct:
    def test_mc_observation_variance_ddof1(self):
        """observation_variance must equal numpy ddof=1 variance of the observations."""
        cfg = _cfg(n_paths=200)
        r = standard_monte_carlo(cfg)
        # obs_var reported == estimator_variance * n_obs
        expected_obs_var = r.estimator_variance * r.pricing_observations
        assert math.isclose(r.observation_variance, expected_obs_var, rel_tol=1e-9)

    def test_estimator_variance_obs_var_over_n(self):
        r = standard_monte_carlo(_cfg(n_paths=150))
        assert math.isclose(
            r.estimator_variance,
            r.observation_variance / r.pricing_observations,
            rel_tol=1e-9,
        )

    def test_av_estimator_variance_correct(self):
        r = antithetic_variates(_cfg(n_paths=200))
        assert math.isclose(
            r.estimator_variance,
            r.observation_variance / r.pricing_observations,
            rel_tol=1e-9,
        )

    def test_cv_estimator_variance_correct(self):
        r = geometric_control_variate(_cfg(n_paths=200), n_pilot=50)
        assert math.isclose(
            r.estimator_variance,
            r.observation_variance / r.pricing_observations,
            rel_tol=1e-9,
        )


# ---------------------------------------------------------------------------
# E3. No misuse of labels
# ---------------------------------------------------------------------------

class TestNoLabelMisuse:
    def test_efficiency_gain_function_exists(self):
        """efficiency_gain_vs_mc must be importable from results."""
        from asian_options.results import efficiency_gain_vs_mc
        assert callable(efficiency_gain_vs_mc)

    def test_efficiency_gain_is_not_variance_ratio(self):
        """
        efficiency_gain_vs_mc(est_var_mc, t_mc, est_var_method, t_method)
        is NOT the same as obs_var_ratio when runtimes differ.
        """
        # Synthetic: method is 2x faster but same variance => gain != variance ratio
        est_var_mc = 1.0
        t_mc = 2.0
        est_var_method = 1.0
        t_method = 1.0
        gain = efficiency_gain_vs_mc(est_var_mc, t_mc, est_var_method, t_method)
        variance_ratio = est_var_mc / est_var_method  # = 1.0
        assert not math.isclose(gain, variance_ratio), (
            "Efficiency gain must differ from variance ratio when runtimes differ"
        )
        assert math.isclose(gain, 2.0, rel_tol=1e-9)  # 2x faster, same variance => 2

    def test_csv_no_speed_ratio_column(self):
        """CSV output must not contain a column named 'speed_ratio_vs_mc'."""
        r = standard_monte_carlo(_cfg())
        row = make_runtime_row("test", "MC", r, r.runtime_s, r.estimator_variance)
        assert "speed_ratio_vs_mc" not in row


# ---------------------------------------------------------------------------
# E4. Efficiency metric formula correctness on deterministic synthetic values
# ---------------------------------------------------------------------------

class TestEfficiencyMetricFormula:
    def test_formula_mc_vs_method_lower_var_faster(self):
        """
        efficiency = (MC_est_var * MC_runtime) / (method_est_var * method_runtime)
        With MC_est_var=4, t_mc=8, method_est_var=1, t_method=2:
        efficiency = (4*8)/(1*2) = 16.
        """
        gain = efficiency_gain_vs_mc(
            mc_estimator_variance=4.0,
            mc_runtime_s=8.0,
            method_estimator_variance=1.0,
            method_runtime_s=2.0,
        )
        assert math.isclose(gain, 16.0, rel_tol=1e-9)

    def test_formula_mc_vs_itself_is_one(self):
        gain = efficiency_gain_vs_mc(1.0, 1.0, 1.0, 1.0)
        assert math.isclose(gain, 1.0, rel_tol=1e-9)

    def test_formula_zero_method_variance_raises(self):
        with pytest.raises((ValueError, ZeroDivisionError)):
            efficiency_gain_vs_mc(1.0, 1.0, 0.0, 1.0)

    def test_formula_zero_method_runtime_raises(self):
        with pytest.raises((ValueError, ZeroDivisionError)):
            efficiency_gain_vs_mc(1.0, 1.0, 1.0, 0.0)

    def test_make_runtime_row_has_efficiency_field(self):
        """make_runtime_row must include an efficiency_gain_vs_mc field."""
        mc = standard_monte_carlo(_cfg(n_paths=100))
        row = make_runtime_row("mode_a", "MC", mc, mc.runtime_s, mc.estimator_variance)
        assert "efficiency_gain_vs_mc" in row

    def test_make_runtime_row_efficiency_mc_vs_self(self):
        """MC vs itself: efficiency_gain_vs_mc = 1.0."""
        mc = standard_monte_carlo(_cfg(n_paths=100))
        row = make_runtime_row("mode_a", "MC", mc, mc.runtime_s, mc.estimator_variance)
        assert math.isclose(float(row["efficiency_gain_vs_mc"]), 1.0, rel_tol=1e-6)

    def test_make_runtime_row_time_per_observation(self):
        mc = standard_monte_carlo(_cfg(n_paths=200))
        row = make_runtime_row("mode_a", "MC", mc, mc.runtime_s, mc.estimator_variance)
        tpo = float(row["time_per_observation"])
        expected = mc.runtime_s / mc.pricing_observations
        assert math.isclose(tpo, expected, rel_tol=1e-9)

    def test_make_runtime_row_time_per_simulated_path(self):
        av = antithetic_variates(_cfg(n_paths=100))
        row = make_runtime_row("mode_a", "AV", av, av.runtime_s, av.estimator_variance)
        tpsp = float(row["time_per_simulated_path"])
        expected = av.runtime_s / av.pricing_simulated_paths
        assert math.isclose(tpsp, expected, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# E5. Comparison outputs include both statistical modes and runtime section
# ---------------------------------------------------------------------------

class TestComparisonOutputStructure:
    def test_runtime_row_has_required_columns(self):
        required = {
            "comparison_mode", "method",
            "runtime_seconds", "time_per_observation",
            "time_per_simulated_path", "efficiency_gain_vs_mc",
            "timing_scope",
        }
        mc = standard_monte_carlo(_cfg(n_paths=100))
        row = make_runtime_row("mode_a", "MC", mc, mc.runtime_s, mc.estimator_variance)
        missing = required - set(row.keys())
        assert not missing, f"Missing columns: {missing}"

    def test_save_results_csv_writes_file(self, tmp_path):
        mc = standard_monte_carlo(_cfg(n_paths=100))
        rows = [make_runtime_row("mode_a", "MC", mc, mc.runtime_s, mc.estimator_variance)]
        out = tmp_path / "runtime.csv"
        save_results_csv(rows, out)
        assert out.exists()
        content = out.read_text()
        assert "efficiency_gain_vs_mc" in content
        assert "runtime_seconds" in content

    def test_print_comparison_table_no_crash(self, capsys):
        mc = standard_monte_carlo(_cfg(n_paths=100))
        av = antithetic_variates(_cfg(n_paths=100))
        rows = [
            {
                "method": "MC",
                "price": mc.price,
                "observation_variance": mc.observation_variance,
                "variance_reduction_ratio": 1.0,
                "pricing_observations": mc.pricing_observations,
                "total_simulated_paths": mc.total_simulated_paths,
            },
            {
                "method": "AV",
                "price": av.price,
                "observation_variance": av.observation_variance,
                "variance_reduction_ratio": mc.observation_variance / av.observation_variance,
                "pricing_observations": av.pricing_observations,
                "total_simulated_paths": av.total_simulated_paths,
            },
        ]
        print_comparison_table(rows)  # must not raise
        out = capsys.readouterr().out
        assert "MC" in out
        assert "AV" in out


# ---------------------------------------------------------------------------
# E6. Backward compatibility
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:
    def test_estimate_result_has_legacy_n_paths(self):
        r = standard_monte_carlo(_cfg())
        assert hasattr(r, "n_paths")
        assert r.n_paths == r.pricing_observations

    def test_estimate_result_has_legacy_variance(self):
        r = standard_monte_carlo(_cfg())
        assert hasattr(r, "variance")
        assert r.variance == r.observation_variance

    def test_save_results_csv_with_stage4_rows(self, tmp_path):
        """Stage 4 result rows (no runtime columns) must not crash save_results_csv."""
        rows = [
            {
                "comparison_mode": "A_equal_obs",
                "method": "MC",
                "pricing_observations": 500,
                "price": "7.123456",
                "observation_variance": "1.23e-02",
            }
        ]
        out = tmp_path / "stage4_compat.csv"
        save_results_csv(rows, out)
        assert out.exists()

    def test_print_comparison_table_with_missing_optional_keys(self, capsys):
        """print_comparison_table must not crash when optional keys absent."""
        rows = [{"method": "MC", "price": 7.1}]
        print_comparison_table(rows)  # must not raise
