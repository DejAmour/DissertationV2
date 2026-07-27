"""Historical diagnostics preservation and reporting tests."""

from __future__ import annotations

from pathlib import Path

import math
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
HANDOFF_ROOT = REPO_ROOT / "handoff_brent_analysis_v3"
TABLES_DIR = HANDOFF_ROOT / "tables"
REPORT_PATH = HANDOFF_ROOT / "HANDOFF_REPORT_V3.md"
AUDIT_PATH = HANDOFF_ROOT / "METHODOLOGY_AUDIT_V3.md"


def test_verified_full_sample_descriptive_results_preserved() -> None:
    desc = pd.read_csv(TABLES_DIR / "descriptive_statistics.csv")
    full = desc.loc[desc["period"] == "full_sample"].iloc[0]
    assert int(full["observations_prices"]) == 6599
    assert int(full["observations_returns"]) == 6598
    assert full["start_date"] == "2000-01-04"
    assert full["end_date"] == "2025-12-31"
    assert math.isclose(float(full["mean"]), 0.000142562420021218, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(float(full["std"]), 0.02602606310655274, rel_tol=0, abs_tol=1e-14)
    assert math.isclose(float(full["annualized_vol"]), 0.4131509435160701, rel_tol=0, abs_tol=1e-13)


def test_half_life_is_missing_and_mean_reversion_is_inconclusive() -> None:
    ar1 = pd.read_csv(TABLES_DIR / "ar1_results.csv").iloc[0]
    assumptions = pd.read_csv(TABLES_DIR / "gbm_assumption_assessment.csv")
    mean_reversion = assumptions.loc[assumptions["assumption"] == "Absence of mean reversion"].iloc[0]
    assert pd.isna(ar1["half_life_days"])
    assert "Half-life not interpreted" in ar1["half_life_note"]
    assert mean_reversion["assessment"] == "inconclusive/not contradicted at 5%"


def test_report_and_audit_use_nonliteral_small_pvalue_language() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")
    audit = AUDIT_PATH.read_text(encoding="utf-8")
    assert "p=0" not in report
    assert "p=0" not in audit
    assert "p < 0.001" in report


def test_report_documents_requested_sample_window_note() -> None:
    report = REPORT_PATH.read_text(encoding="utf-8")
    assert "The requested sample window begins on 1 January 2000." in report
    assert "first available reported price observation within that window occurs on 2000-01-04" in report
