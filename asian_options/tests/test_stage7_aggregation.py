from __future__ import annotations

import math

from asian_options.aggregation import aggregate_validation_rows


def test_aggregation_mean_std_ci_on_deterministic_values():
    rows = [
        {
            "profile_name": "validation_minimal",
            "profile_version": "1.0",
            "profile_config_id": "cfg1",
            "profile_config_label": "label",
            "mode": "A_equal_obs",
            "method": "MC",
            "price_estimate": "10.0",
            "standard_error": "0.1",
            "estimator_variance": "0.01",
            "runtime_seconds": "NA",
            "efficiency_gain_vs_mc": "NA",
        },
        {
            "profile_name": "validation_minimal",
            "profile_version": "1.0",
            "profile_config_id": "cfg1",
            "profile_config_label": "label",
            "mode": "A_equal_obs",
            "method": "MC",
            "price_estimate": "14.0",
            "standard_error": "0.2",
            "estimator_variance": "0.04",
            "runtime_seconds": "NA",
            "efficiency_gain_vs_mc": "NA",
        },
    ]

    aggregated = aggregate_validation_rows(rows)
    price_row = next(r for r in aggregated if r["metric"] == "price_estimate")
    assert int(price_row["sample_count"]) == 2
    assert math.isclose(float(price_row["mean"]), 12.0, rel_tol=1e-12)
    assert math.isclose(float(price_row["std"]), math.sqrt(8.0), rel_tol=1e-8)
    assert float(price_row["ci95_lower"]) < 12.0
    assert float(price_row["ci95_upper"]) > 12.0


def test_aggregation_n_lt_2_marks_ci_as_na():
    rows = [
        {
            "profile_name": "validation_minimal",
            "profile_version": "1.0",
            "profile_config_id": "cfg1",
            "profile_config_label": "label",
            "mode": "C_runtime",
            "method": "MC",
            "price_estimate": "NA",
            "standard_error": "NA",
            "estimator_variance": "0.01",
            "runtime_seconds": "1.0",
            "efficiency_gain_vs_mc": "1.0",
        }
    ]
    aggregated = aggregate_validation_rows(rows)
    runtime_row = next(r for r in aggregated if r["metric"] == "runtime_seconds")
    assert runtime_row["std"] == "NA"
    assert runtime_row["ci95_lower"] == "NA"
    assert runtime_row["ci95_upper"] == "NA"
    assert "n<2" in runtime_row["ci_note"]
