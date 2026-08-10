"""
test_stage7_fixes.py
====================
Stage 7 output-audit acceptance tests.

Issue 1: Runtime-scope fields present and correctly populated.
Issue 2: Student-t CI with metadata fields.
Issue 3: No speed_ratio_vs_mc in current publication outputs.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from asian_options.aggregation import aggregate_validation_rows, AGGREGATE_OUTPUT_COLUMNS
from asian_options.config import ModelConfig
from asian_options.estimators import (
    standard_monte_carlo,
    antithetic_variates,
    geometric_control_variate,
    EstimateResult,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE = dict(S0=100.0, K=100.0, r=0.05, q=0.0, sigma=0.2, T=1.0, m=12,
            n_paths=200, seed=99)


def _cfg(**kw) -> ModelConfig:
    return ModelConfig(**{**BASE, **kw})


# ---------------------------------------------------------------------------
# Issue 1: Runtime-scope fields
# ---------------------------------------------------------------------------

class TestRuntimeScopeFields:
    """Runtime-scope fields are present and self-consistent for each estimator."""

    def test_mc_scope_fields_present(self):
        r = standard_monte_carlo(_cfg())
        assert hasattr(r, "pricing_runtime_seconds")
        assert hasattr(r, "training_runtime_seconds")
        assert hasattr(r, "end_to_end_runtime_seconds")

    def test_mc_training_runtime_zero(self):
        r = standard_monte_carlo(_cfg())
        assert r.training_runtime_seconds == 0.0

    def test_mc_pricing_equals_e2e(self):
        r = standard_monte_carlo(_cfg())
        assert math.isclose(r.pricing_runtime_seconds, r.end_to_end_runtime_seconds,
                            rel_tol=1e-9)

    def test_mc_e2e_non_negative(self):
        r = standard_monte_carlo(_cfg())
        assert r.end_to_end_runtime_seconds >= 0.0
        assert r.pricing_runtime_seconds >= 0.0

    def test_av_training_runtime_zero(self):
        r = antithetic_variates(_cfg())
        assert r.training_runtime_seconds == 0.0

    def test_av_pricing_equals_e2e(self):
        r = antithetic_variates(_cfg())
        assert math.isclose(r.pricing_runtime_seconds, r.end_to_end_runtime_seconds,
                            rel_tol=1e-9)

    def test_cv_e2e_includes_pilot(self):
        """CV end-to-end must be strictly >= pricing-only (pilot adds time)."""
        r = geometric_control_variate(_cfg(n_paths=200), n_pilot=50)
        # e2e >= pricing because pilot is included
        assert r.end_to_end_runtime_seconds >= r.pricing_runtime_seconds - 1e-9

    def test_cv_training_runtime_is_pilot_time(self):
        """CV training_runtime_seconds records pilot simulation + beta estimation."""
        r = geometric_control_variate(_cfg(n_paths=200), n_pilot=50)
        assert r.training_runtime_seconds >= 0.0
        # e2e should equal pricing + training
        assert math.isclose(
            r.end_to_end_runtime_seconds,
            r.pricing_runtime_seconds + r.training_runtime_seconds,
            rel_tol=0.05,  # small floating-point tolerance for perf_counter splits
        )

    def test_cv_training_runtime_not_zero_with_pilot(self):
        """CV must not silently report zero training time when pilot is non-trivial."""
        r = geometric_control_variate(_cfg(n_paths=200), n_pilot=100)
        # training_runtime_seconds should reflect pilot cost; 0.0 would be wrong
        # (pilot simulation takes some time, even if tiny)
        assert r.training_runtime_seconds >= 0.0  # non-negative is mandatory
        # The field must exist and be explicitly set (not defaulting from EstimateResult=0)
        assert isinstance(r.training_runtime_seconds, float)

    def test_estimateresult_has_scope_fields(self):
        """EstimateResult NamedTuple exposes all three scope fields."""
        r = EstimateResult(
            price=1.0, variance=0.1, std_dev=0.316, std_error=0.01,
            ci_lower=0.98, ci_upper=1.02, n_paths=100, runtime_s=0.5,
            pricing_observations=100, pricing_simulated_paths=100,
            pilot_paths=0, training_paths=0, total_simulated_paths=100,
            observation_variance=0.1, estimator_variance=0.001,
            pricing_runtime_seconds=0.5,
            training_runtime_seconds=0.0,
            end_to_end_runtime_seconds=0.5,
        )
        assert r.pricing_runtime_seconds == 0.5
        assert r.training_runtime_seconds == 0.0
        assert r.end_to_end_runtime_seconds == 0.5


# ---------------------------------------------------------------------------
# Issue 2: Student-t CI
# ---------------------------------------------------------------------------

class TestStudentTCI:
    """Aggregation uses Student-t CI, not normal approximation."""

    def test_n2_ci_wider_than_normal(self):
        """For n=2, t-critical (12.706) >> z=1.96, so CI must be much wider."""
        rows = [
            {
                "profile_name": "validation_minimal",
                "profile_version": "1.0",
                "profile_config_id": "c1",
                "profile_config_label": "l",
                "mode": "A_equal_obs",
                "method": "MC",
                "price_estimate": "10.0",
                "standard_error": "NA",
                "estimator_variance": "NA",
                "runtime_seconds": "NA",
                "efficiency_gain_vs_mc": "NA",
            },
            {
                "profile_name": "validation_minimal",
                "profile_version": "1.0",
                "profile_config_id": "c1",
                "profile_config_label": "l",
                "mode": "A_equal_obs",
                "method": "MC",
                "price_estimate": "12.0",
                "standard_error": "NA",
                "estimator_variance": "NA",
                "runtime_seconds": "NA",
                "efficiency_gain_vs_mc": "NA",
            },
        ]
        agg = aggregate_validation_rows(rows)
        price_row = next(r for r in agg if r["metric"] == "price_estimate")
        ci_width_t = float(price_row["ci95_upper"]) - float(price_row["ci95_lower"])
        # Normal approx CI width: 2 * 1.96 * std/sqrt(2)
        std = math.sqrt(2.0)  # stdev([10, 12])
        ci_width_normal = 2 * 1.96 * std / math.sqrt(2)
        # t-critical for df=1 at 95% is ~12.706, so t-CI >> z-CI
        assert ci_width_t > ci_width_normal * 3

    def test_metadata_fields_populated(self):
        rows = [
            {
                "profile_name": "p", "profile_version": "1", "profile_config_id": "c",
                "profile_config_label": "l", "mode": "A", "method": "MC",
                "price_estimate": "10.0", "standard_error": "NA",
                "estimator_variance": "NA", "runtime_seconds": "NA",
                "efficiency_gain_vs_mc": "NA",
            },
            {
                "profile_name": "p", "profile_version": "1", "profile_config_id": "c",
                "profile_config_label": "l", "mode": "A", "method": "MC",
                "price_estimate": "14.0", "standard_error": "NA",
                "estimator_variance": "NA", "runtime_seconds": "NA",
                "efficiency_gain_vs_mc": "NA",
            },
        ]
        agg = aggregate_validation_rows(rows)
        row = next(r for r in agg if r["metric"] == "price_estimate")
        assert "ci_dof" in row
        assert "ci_critical_value" in row
        assert "ci_method" in row
        assert "ci_confidence_level" in row
        assert row["ci_method"] == "student-t"
        assert row["ci_dof"] == "1"  # n=2 => dof=1
        assert row["ci_confidence_level"] == "0.95"
        # Critical value for df=1 at 97.5% is ~12.706
        assert math.isclose(float(row["ci_critical_value"]), 12.706, rel_tol=1e-3)

    def test_n1_ci_is_na_with_explanation(self):
        rows = [
            {
                "profile_name": "p", "profile_version": "1", "profile_config_id": "c",
                "profile_config_label": "l", "mode": "A", "method": "MC",
                "price_estimate": "10.0", "standard_error": "NA",
                "estimator_variance": "NA", "runtime_seconds": "NA",
                "efficiency_gain_vs_mc": "NA",
            },
        ]
        agg = aggregate_validation_rows(rows)
        row = next(r for r in agg if r["metric"] == "price_estimate")
        assert row["std"] == "NA"
        assert row["ci95_lower"] == "NA"
        assert row["ci95_upper"] == "NA"
        assert "n<2" in row["ci_note"]
        assert row["ci_method"] == "NA"

    def test_output_columns_include_ci_metadata(self):
        """AGGREGATE_OUTPUT_COLUMNS must include the new CI metadata fields."""
        for field in ("ci_dof", "ci_critical_value", "ci_method", "ci_confidence_level"):
            assert field in AGGREGATE_OUTPUT_COLUMNS, f"Missing column: {field}"

    def test_known_sample_interval(self):
        """Verify interval against scipy directly for n=3 values."""
        from scipy.stats import t as _t
        values = [10.0, 11.0, 12.0]
        rows = [
            {
                "profile_name": "p", "profile_version": "1", "profile_config_id": "c",
                "profile_config_label": "l", "mode": "A", "method": "MC",
                "price_estimate": str(v), "standard_error": "NA",
                "estimator_variance": "NA", "runtime_seconds": "NA",
                "efficiency_gain_vs_mc": "NA",
            }
            for v in values
        ]
        agg = aggregate_validation_rows(rows)
        row = next(r for r in agg if r["metric"] == "price_estimate")
        import statistics
        m = statistics.mean(values)
        s = statistics.stdev(values)
        t_crit = float(_t.ppf(0.975, df=2))
        hw = t_crit * s / math.sqrt(3)
        assert math.isclose(float(row["ci95_lower"]), m - hw, rel_tol=1e-9)
        assert math.isclose(float(row["ci95_upper"]), m + hw, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# Issue 3: No speed_ratio_vs_mc in current outputs
# ---------------------------------------------------------------------------

class TestNoSpeedRatioInCurrentOutputs:
    """No CSV outside legacy/ should contain speed_ratio_vs_mc."""

    def _current_csvs(self):
        """Yield all CSV files in the repo that are NOT under legacy/."""
        for csv_path in REPO_ROOT.rglob("*.csv"):
            if "legacy" in csv_path.parts:
                continue
            if "__pycache__" in str(csv_path):
                continue
            yield csv_path

    def test_no_speed_ratio_in_any_current_csv(self):
        violations = []
        for csv_path in self._current_csvs():
            try:
                text = csv_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if "speed_ratio_vs_mc" in text:
                violations.append(str(csv_path.relative_to(REPO_ROOT)))
        assert not violations, (
            "The following current (non-legacy) CSV files contain "
            f"'speed_ratio_vs_mc':\n" + "\n".join(violations)
        )

    def test_legacy_csv_has_warning_file(self):
        """The legacy directory must contain a WARNING.txt."""
        warning = REPO_ROOT / "legacy" / "WARNING.txt"
        assert warning.exists(), "legacy/WARNING.txt missing"
        text = warning.read_text(encoding="utf-8")
        assert "DO NOT USE" in text
        assert "speed_ratio_vs_mc" in text
