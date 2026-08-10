from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ValidationReport:
    passed: bool
    failures: list[str]


def _parse_float(value) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text in {"", "NA", "nan", "NaN", "ERROR"}:
        return None
    try:
        parsed = float(text)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _parse_int(value) -> int | None:
    parsed = _parse_float(value)
    if parsed is None:
        return None
    if not float(parsed).is_integer():
        return None
    return int(parsed)


def _row_id(row: dict) -> str:
    return (
        f"config={row.get('profile_config_id', 'NA')}, mode={row.get('comparison_mode', row.get('mode', 'NA'))}, "
        f"method={row.get('method', 'NA')}, seed={row.get('seed', 'NA')}, rep={row.get('replication', 'NA')}"
    )


def _group_key(row: dict, mode_key: str = "comparison_mode") -> tuple[str, str, str, str]:
    return (
        str(row.get("profile_config_id", "NA")),
        str(row.get(mode_key, "NA")),
        str(row.get("seed", "NA")),
        str(row.get("replication", "NA")),
    )


def validate_profile_outputs(
    stat_rows: Iterable[dict],
    runtime_rows: Iterable[dict],
    publication_rows: Iterable[dict],
    reference_zscore: float = 6.0,
) -> ValidationReport:
    failures: list[str] = []
    stat_rows = list(stat_rows)
    runtime_rows = list(runtime_rows)
    publication_rows = list(publication_rows)

    for row in stat_rows:
        for field in (
            "price",
            "std_error",
            "observation_variance",
            "estimator_variance",
            "variance_reduction_ratio",
            "runtime_s",
        ):
            value = _parse_float(row.get(field))
            if value is None:
                failures.append(f"{_row_id(row)}: field '{field}' must be finite and defined.")
                continue
            if field != "price" and value < 0:
                failures.append(f"{_row_id(row)}: field '{field}' must be non-negative.")

        for field in ("pricing_observations", "pricing_simulated_paths", "pilot_paths", "training_paths", "total_simulated_paths"):
            value = _parse_int(row.get(field))
            if value is None:
                failures.append(f"{_row_id(row)}: integer field '{field}' is invalid.")
                continue
            if value < 0:
                failures.append(f"{_row_id(row)}: integer field '{field}' must be non-negative.")

    for row in runtime_rows:
        for field in (
            "runtime_seconds",
            "time_per_observation",
            "time_per_simulated_path",
            "efficiency_gain_vs_mc",
            "std_error",
            "observation_variance",
            "estimator_variance",
        ):
            value = _parse_float(row.get(field))
            if value is None:
                failures.append(f"{_row_id(row)}: field '{field}' must be finite and defined.")
                continue
            if value < 0:
                failures.append(f"{_row_id(row)}: field '{field}' must be non-negative.")

        for field in ("pricing_observations", "pricing_simulated_paths", "pilot_paths", "training_paths", "total_simulated_paths"):
            value = _parse_int(row.get(field))
            if value is None:
                failures.append(f"{_row_id(row)}: integer field '{field}' is invalid.")
                continue
            if value < 0:
                failures.append(f"{_row_id(row)}: integer field '{field}' must be non-negative.")

    expected_methods = {"MC", "AV", "CV", "NCV"}
    mode_method_map: dict[tuple[str, str, str, str], set[str]] = {}
    for row in stat_rows:
        mode = row.get("comparison_mode")
        if mode not in {"A_equal_obs", "B_equal_budget"}:
            failures.append(f"{_row_id(row)}: unexpected statistical comparison_mode '{mode}'.")
        mode_method_map.setdefault(_group_key(row), set()).add(str(row.get("method", "NA")))
    for row in runtime_rows:
        mode = row.get("comparison_mode")
        if mode != "C_runtime":
            failures.append(f"{_row_id(row)}: unexpected runtime comparison_mode '{mode}'.")
        mode_method_map.setdefault(_group_key(row), set()).add(str(row.get("method", "NA")))
    for key, methods in mode_method_map.items():
        missing = expected_methods - methods
        if missing:
            failures.append(
                f"config={key[0]}, mode={key[1]}, seed={key[2]}, rep={key[3]}: missing methods {sorted(missing)}."
            )

    for row in [*stat_rows, *runtime_rows]:
        method = str(row.get("method", "NA"))
        obs = _parse_int(row.get("pricing_observations"))
        sim = _parse_int(row.get("pricing_simulated_paths"))
        pilot = _parse_int(row.get("pilot_paths"))
        training = _parse_int(row.get("training_paths"))
        total = _parse_int(row.get("total_simulated_paths"))
        if None in {obs, sim, pilot, training, total}:
            continue
        if total != sim + pilot + training:
            failures.append(f"{_row_id(row)}: total_simulated_paths must equal pricing_simulated_paths + pilot_paths + training_paths.")
        if method == "MC" and not (obs == sim == total):
            failures.append(f"{_row_id(row)}: MC must satisfy pricing_observations == pricing_simulated_paths == total_simulated_paths.")
        if method == "AV" and sim != 2 * obs:
            failures.append(f"{_row_id(row)}: AV must satisfy pricing_simulated_paths == 2 * pricing_observations.")
        if method == "CV" and total != pilot + sim:
            failures.append(f"{_row_id(row)}: CV must satisfy total_simulated_paths == pilot_paths + pricing_simulated_paths.")
        if method == "NCV" and total != training + sim:
            failures.append(f"{_row_id(row)}: NCV must satisfy total_simulated_paths == training_paths + pricing_simulated_paths.")

    for row in publication_rows:
        mode = row.get("mode")
        if mode in {"A_equal_obs", "B_equal_budget"}:
            for field in ("runtime_seconds", "time_per_observation", "time_per_simulated_path", "efficiency_gain_vs_mc", "timing_scope_note"):
                if str(row.get(field, "NA")) != "NA":
                    failures.append(f"{_row_id(row)}: field '{field}' must be NA for mode {mode}.")
        if mode == "C_runtime" and str(row.get("variance_reduction_ratio", "NA")) != "NA":
            failures.append(f"{_row_id(row)}: variance_reduction_ratio must be NA for mode C_runtime.")

    stat_mc_lookup: dict[tuple[str, str, str, str], tuple[float, float]] = {}
    for row in stat_rows:
        if row.get("method") != "MC":
            continue
        obs_var = _parse_float(row.get("observation_variance"))
        se = _parse_float(row.get("std_error"))
        if obs_var is not None and se is not None:
            stat_mc_lookup[_group_key(row)] = (obs_var, se)
    for row in stat_rows:
        key = _group_key(row)
        mc_values = stat_mc_lookup.get(key)
        if mc_values is None:
            failures.append(f"{_row_id(row)}: missing MC baseline row for variance/reference checks.")
            continue
        method_obs_var = _parse_float(row.get("observation_variance"))
        vrr = _parse_float(row.get("variance_reduction_ratio"))
        if method_obs_var is not None and vrr is not None and method_obs_var > 0:
            expected_vrr = mc_values[0] / method_obs_var
            if abs(vrr - expected_vrr) > 5e-4:
                failures.append(f"{_row_id(row)}: variance_reduction_ratio mismatch; expected {expected_vrr:.6f}, got {vrr:.6f}.")
        method_price = _parse_float(row.get("price"))
        method_se = _parse_float(row.get("std_error"))
        mc_row = next((r for r in stat_rows if _group_key(r) == key and r.get("method") == "MC"), None)
        if mc_row is not None:
            mc_price = _parse_float(mc_row.get("price"))
            mc_se = _parse_float(mc_row.get("std_error"))
            if None not in {method_price, method_se, mc_price, mc_se}:
                tol = reference_zscore * (method_se + mc_se)
                if abs(method_price - mc_price) > tol:
                    failures.append(
                        f"{_row_id(row)}: price deviates from MC reference beyond SE-aware tolerance "
                        f"(abs_diff={abs(method_price - mc_price):.6f}, tolerance={tol:.6f})."
                    )

    runtime_mc_lookup: dict[tuple[str, str, str, str], tuple[float, float]] = {}
    for row in runtime_rows:
        if row.get("method") != "MC":
            continue
        mc_est_var = _parse_float(row.get("estimator_variance"))
        mc_runtime = _parse_float(row.get("runtime_seconds"))
        if mc_est_var is not None and mc_runtime is not None:
            runtime_mc_lookup[_group_key(row)] = (mc_est_var, mc_runtime)
    for row in runtime_rows:
        key = _group_key(row)
        baseline = runtime_mc_lookup.get(key)
        if baseline is None:
            failures.append(f"{_row_id(row)}: missing MC baseline row for efficiency checks.")
            continue
        est_var = _parse_float(row.get("estimator_variance"))
        runtime_s = _parse_float(row.get("runtime_seconds"))
        stored_eff = _parse_float(row.get("efficiency_gain_vs_mc"))
        if None in {est_var, runtime_s, stored_eff}:
            continue
        if est_var == 0 or runtime_s == 0:
            continue
        expected_eff = (baseline[0] * baseline[1]) / (est_var * runtime_s)
        if abs(stored_eff - expected_eff) > 1e-5:
            failures.append(f"{_row_id(row)}: efficiency_gain_vs_mc mismatch; expected {expected_eff:.6f}, got {stored_eff:.6f}.")

    return ValidationReport(passed=not failures, failures=failures)


def write_validation_report(path, report: ValidationReport) -> None:
    lines = [
        "# Stage 7 Validation Report",
        "",
        f"- Overall: {'PASS' if report.passed else 'FAIL'}",
        f"- Total failed checks: {len(report.failures)}",
        "",
        "## Checklist",
        f"- [{'x' if report.passed else ' '}] Non-negativity/finite checks",
        f"- [{'x' if report.passed else ' '}] Budget consistency checks",
        f"- [{'x' if report.passed else ' '}] Mode-consistency checks",
        f"- [{'x' if report.passed else ' '}] Reference-consistency checks",
        f"- [{'x' if report.passed else ' '}] VRR/efficiency formula checks",
    ]
    if report.failures:
        lines.extend(["", "## Failures"])
        for failure in report.failures:
            lines.append(f"- {failure}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
