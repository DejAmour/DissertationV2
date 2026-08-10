from __future__ import annotations

import math
from statistics import mean, stdev
from typing import Iterable

from asian_options.results import save_results_csv

AGGREGATE_OUTPUT_COLUMNS = [
    "profile_name",
    "profile_version",
    "profile_config_id",
    "profile_config_label",
    "mode",
    "method",
    "metric",
    "sample_count",
    "mean",
    "std",
    "ci95_lower",
    "ci95_upper",
    "ci_note",
]


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


def aggregate_validation_rows(summary_rows: Iterable[dict]) -> list[dict]:
    metric_map = {
        "price_estimate": "price_estimate",
        "standard_error": "standard_error",
        "estimator_variance": "estimator_variance",
        "runtime_seconds": "runtime_seconds",
        "efficiency_gain_vs_mc": "efficiency_gain_vs_mc",
    }
    grouped: dict[tuple[str, str, str, str, str], dict[str, object]] = {}

    for row in summary_rows:
        for metric_name, column_name in metric_map.items():
            value = _parse_float(row.get(column_name))
            if value is None:
                continue
            key = (
                str(row.get("profile_config_id", "NA")),
                str(row.get("profile_config_label", "NA")),
                str(row.get("mode", "NA")),
                str(row.get("method", "NA")),
                metric_name,
            )
            bucket = grouped.setdefault(
                key,
                {
                    "profile_name": row.get("profile_name", "NA"),
                    "profile_version": row.get("profile_version", "NA"),
                    "profile_config_id": key[0],
                    "profile_config_label": key[1],
                    "mode": key[2],
                    "method": key[3],
                    "metric": key[4],
                    "values": [],
                },
            )
            bucket["values"].append(value)

    aggregates: list[dict] = []
    for _, bucket in sorted(grouped.items(), key=lambda item: item[0]):
        values = list(bucket["values"])
        n = len(values)
        avg = mean(values)
        if n < 2:
            std_val = "NA"
            ci_low = "NA"
            ci_high = "NA"
            ci_note = "n<2; CI undefined"
        else:
            std_float = stdev(values)
            half_width = 1.96 * std_float / math.sqrt(n)
            std_val = f"{std_float:.10g}"
            ci_low = f"{(avg - half_width):.10g}"
            ci_high = f"{(avg + half_width):.10g}"
            ci_note = ""
        aggregates.append(
            {
                "profile_name": bucket["profile_name"],
                "profile_version": bucket["profile_version"],
                "profile_config_id": bucket["profile_config_id"],
                "profile_config_label": bucket["profile_config_label"],
                "mode": bucket["mode"],
                "method": bucket["method"],
                "metric": bucket["metric"],
                "sample_count": n,
                "mean": f"{avg:.10g}",
                "std": std_val,
                "ci95_lower": ci_low,
                "ci95_upper": ci_high,
                "ci_note": ci_note,
            }
        )
    return aggregates


def write_validation_aggregate_markdown(rows: Iterable[dict], path) -> None:
    rows = list(rows)
    header = "| " + " | ".join(AGGREGATE_OUTPUT_COLUMNS) + " |"
    separator = "| " + " | ".join(["---"] * len(AGGREGATE_OUTPUT_COLUMNS)) + " |"
    lines = [header, separator]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "NA")) for col in AGGREGATE_OUTPUT_COLUMNS) + " |")
    lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_validation_aggregate_csv(rows: Iterable[dict], path) -> None:
    save_results_csv(rows, path, fieldnames=AGGREGATE_OUTPUT_COLUMNS)
