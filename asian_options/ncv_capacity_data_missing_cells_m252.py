from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from asian_options.ncv_capacity_data_sensitivity import (
    CapacityCell,
    _build_checkpoint_summary,
    _build_paired_contrasts,
    _build_runtime_summary,
    _default_cells,
    _default_paired_comparisons,
    _student_t_ci,
    profile_config,
    run_capacity_data_sensitivity,
)


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _discover_existing_dissertation_run(repo_root: Path) -> Path:
    expected = set(c.config_id for c in _default_cells("dissertation", dissertation_cell_set="baseline"))
    candidates = sorted(repo_root.glob("experiment_runs/**/ncv_capacity_data_m252/capacity_data_selected_checkpoints.csv"))
    valid: list[Path] = []
    for csv_path in candidates:
        rows = _read_csv(csv_path)
        ids = {row["config_id"] for row in rows}
        if ids == expected:
            valid.append(csv_path.parent)
    if not valid:
        raise RuntimeError("unable to find existing five-cell dissertation capacity run")
    valid.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return valid[0]


def _as_float(v: Any) -> float:
    try:
        return float(v)
    except Exception:
        return float("nan")


def _selected_checkpoint_map(rows: list[dict[str, Any]]) -> dict[str, int]:
    return {row["config_id"]: int(row["selected_checkpoint"]) for row in rows}


def _combine_rows(existing_rows: list[dict[str, Any]], new_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return list(existing_rows) + list(new_rows)


def _full_grid_cells() -> tuple[CapacityCell, ...]:
    return _default_cells("dissertation", dissertation_cell_set="full_grid")


def _summary_table(
    per_replication_rows: list[dict[str, Any]],
    selected: dict[str, int],
    cells: tuple[CapacityCell, ...],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cell in cells:
        cp = int(selected[cell.config_id])
        test_rows = [
            r
            for r in per_replication_rows
            if r["config_id"] == cell.config_id and r["split"] == "test" and int(r["checkpoint"]) == cp
        ]
        resid = [_as_float(r["centered_residual_variance"]) for r in test_rows]
        vrr = [_as_float(r["ncv_vrr_vs_mc"]) for r in test_rows]
        corr = [_as_float(r["payoff_network_correlation"]) for r in test_rows]
        gen_gap = [_as_float(r["generalization_gap_log"]) for r in test_rows]
        runtime = [_as_float(r["ncv_setup_cost_s"]) for r in test_rows]
        resid_ci = _student_t_ci([x for x in resid if math.isfinite(x)])
        out.append(
            {
                "config_id": cell.config_id,
                "hidden_width": int(cell.hidden_width),
                "parameter_count": int(cell.hidden_width * (252 + 2) + 1),
                "train_paths": int(cell.train_paths),
                "paths_per_parameter": float(cell.train_paths) / float(cell.hidden_width * (252 + 2) + 1),
                "selected_checkpoint": cp,
                "mean_test_residual_variance": statistics.mean([x for x in resid if math.isfinite(x)]),
                "median_test_residual_variance": statistics.median([x for x in resid if math.isfinite(x)]),
                "mean_test_vrr": statistics.mean([x for x in vrr if math.isfinite(x)]),
                "median_test_vrr": statistics.median([x for x in vrr if math.isfinite(x)]),
                "ci95_test_residual_variance_lower": resid_ci["ci95_lower"],
                "ci95_test_residual_variance_upper": resid_ci["ci95_upper"],
                "mean_test_correlation": statistics.mean([x for x in corr if math.isfinite(x)]),
                "mean_log_generalization_gap": statistics.mean([x for x in gen_gap if math.isfinite(x)]),
                "mean_setup_runtime_s": statistics.mean([x for x in runtime if math.isfinite(x)]),
            }
        )
    return out


def _make_full_grid_plot(
    out_path: Path,
    per_replication_rows: list[dict[str, Any]],
    selected: dict[str, int],
    cells: tuple[CapacityCell, ...],
    existing_ids: set[str],
    gcv_rows: list[dict[str, Any]],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [
        ("ncv_vrr_vs_mc", "Test VRR vs MC", False),
        ("centered_residual_variance", "Test residual variance", True),
        ("generalization_gap_log", "Log generalization gap", False),
        ("ncv_setup_cost_s", "Setup runtime (s)", True),
    ]
    widths = sorted(set(int(c.hidden_width) for c in cells))
    color_by_width = {8: "#1f77b4", 16: "#ff7f0e", 32: "#2ca02c"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    gcv_test = [r for r in gcv_rows if str(r["split"]) == "test"]
    gcv_vrr_mean = statistics.mean([_as_float(r["gcv_vrr_vs_mc"]) for r in gcv_test])
    gcv_resid_mean = statistics.mean([_as_float(r["gcv_residual_variance"]) for r in gcv_test])

    for ax, (metric, title, logy) in zip(axes.flatten(), metrics):
        for w in widths:
            xs: list[int] = []
            ys: list[float] = []
            for n in sorted({int(c.train_paths) for c in cells if int(c.hidden_width) == w}):
                cfg_id = next(c.config_id for c in cells if int(c.hidden_width) == w and int(c.train_paths) == n)
                cp = int(selected[cfg_id])
                vals = [
                    _as_float(r[metric])
                    for r in per_replication_rows
                    if r["config_id"] == cfg_id and r["split"] == "test" and int(r["checkpoint"]) == cp
                ]
                vals = [v for v in vals if math.isfinite(v)]
                xs.append(n)
                ys.append(float(statistics.mean(vals)))
                marker = "o" if cfg_id in existing_ids else "s"
                ax.scatter([n], [ys[-1]], color=color_by_width.get(w, "#333333"), marker=marker, s=35, zorder=3)
            ax.plot(xs, ys, color=color_by_width.get(w, "#333333"), linewidth=1.2, label=f"width={w}")
        if metric == "ncv_vrr_vs_mc":
            ax.axhline(gcv_vrr_mean, color="#d62728", linestyle="--", linewidth=1.2, label="GCV benchmark")
        if metric == "centered_residual_variance":
            ax.axhline(gcv_resid_mean, color="#d62728", linestyle="--", linewidth=1.2, label="GCV benchmark")
        ax.set_title(title)
        ax.set_xlabel("Training paths")
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    marker_existing = plt.Line2D([0], [0], marker="o", color="black", linestyle="None", label="Existing cells")
    marker_new = plt.Line2D([0], [0], marker="s", color="black", linestyle="None", label="New cells")
    fig.legend(handles + [marker_existing, marker_new], labels + ["Existing cells", "New cells"], loc="upper center", ncol=4, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def _question_answers(
    summary_rows: list[dict[str, Any]],
    selected_new_rows: list[dict[str, Any]],
    gcv_rows: list[dict[str, Any]],
) -> list[str]:
    by_cfg = {r["config_id"]: r for r in summary_rows}

    def _best_at(n: int) -> dict[str, Any]:
        rows = [r for r in summary_rows if int(r["train_paths"]) == n]
        return min(rows, key=lambda r: float(r["mean_test_residual_variance"]))

    best_10 = _best_at(10_000)
    best_20 = _best_at(20_000)
    best_all = min(summary_rows, key=lambda r: float(r["mean_test_residual_variance"]))
    gcv_test = [r for r in gcv_rows if str(r["split"]) == "test"]
    gcv_resid = statistics.mean([_as_float(r["gcv_residual_variance"]) for r in gcv_test])

    q: list[str] = []
    q.append(
        "1) Increasing data for widths 8 and 16: assess from within-width residual-variance declines across 5k→10k→20k in combined table."
    )
    q.append(f"2) At 10,000 paths, best-performing configuration among the nine width–training-size combinations examined: {best_10['config_id']}.")
    q.append(f"3) At 20,000 paths, best-performing configuration among the nine width–training-size combinations examined: {best_20['config_id']}.")
    q.append(f"4) Width-32 at 20,000 remains best at 20,000 paths: {'yes' if best_20['config_id']=='w32_n20000' else 'no'}.")
    q.append("5) Association pattern: compare train-size trends within width and cross-width comparisons at fixed data using paired_contrasts.")
    q.append(f"6) Any new config outperforming existing w32_n20000: {'yes' if best_all['config_id'] != 'w32_n20000' else 'no'}.")
    q.append(
        "7) Any new config outperforming GCV benchmark (test residual variance): "
        + ("yes" if any(float(r["mean_test_residual_variance"]) < gcv_resid for r in summary_rows if r["config_id"] in {"w8_n10000","w8_n20000","w16_n10000","w16_n20000"}) else "no")
        + "."
    )
    q.append("8) Consistency across 10 replications: use paired-contrast CI signs and medians in combined paired-contrast output.")
    edge_cells = [r["config_id"] for r in selected_new_rows if int(r["selected_checkpoint"]) == 200]
    q.append(f"9) Selected checkpoint at edge (200) for new cells: {', '.join(edge_cells) if edge_cells else 'none'}.")
    return q


def run_missing_cells_and_combine(
    *,
    repo_root: Path,
    output_dir: Path,
    base_seed: int,
    existing_run_dir: Path | None,
    new_run_dir: Path | None,
) -> dict[str, str]:
    existing_dir = existing_run_dir or _discover_existing_dissertation_run(repo_root)
    gcv_rows = _read_csv(existing_dir / "capacity_data_gcv_benchmark.csv")
    computed_new_run_dir = new_run_dir
    if computed_new_run_dir is None:
        missing_cfg = profile_config(
            "dissertation",
            output_dir=str(output_dir),
            base_seed=base_seed,
            dissertation_cell_set="missing_cells",
        )
        computed_new_run_dir = run_capacity_data_sensitivity(missing_cfg, gcv_benchmark_rows=gcv_rows)
    assert computed_new_run_dir is not None

    combined_root = computed_new_run_dir.parent / "combined_full_grid_m252"
    combined_root.mkdir(parents=True, exist_ok=False)

    existing_per_rep = _read_csv(existing_dir / "capacity_data_per_replication.csv")
    existing_sel = _read_csv(existing_dir / "capacity_data_selected_checkpoints.csv")
    new_per_rep = _read_csv(computed_new_run_dir / "capacity_data_per_replication.csv")
    new_sel = _read_csv(computed_new_run_dir / "capacity_data_selected_checkpoints.csv")

    combined_per_rep = _combine_rows(existing_per_rep, new_per_rep)
    combined_sel = _combine_rows(existing_sel, new_sel)
    selected = _selected_checkpoint_map(combined_sel)
    cells = _full_grid_cells()
    if len(selected) != 9:
        raise RuntimeError("combined selected checkpoint map must contain exactly nine cells")

    comparisons = _default_paired_comparisons(cells)
    combined_pairs = _build_paired_contrasts(combined_per_rep, selected, comparisons)

    combined_cfg = profile_config(
        "dissertation",
        output_dir=str(combined_root),
        base_seed=base_seed,
        dissertation_cell_set="full_grid",
    )
    combined_checkpoint_summary = _build_checkpoint_summary(combined_per_rep, combined_cfg)
    combined_runtime_summary = _build_runtime_summary(combined_per_rep, selected, combined_cfg)
    combined_table = _summary_table(combined_per_rep, selected, cells)

    _write_csv(combined_root / "capacity_data_per_replication_combined.csv", combined_per_rep)
    _write_csv(combined_root / "capacity_data_selected_checkpoints_combined.csv", combined_sel)
    _write_csv(combined_root / "capacity_data_paired_contrasts_combined.csv", combined_pairs)
    _write_csv(combined_root / "capacity_data_checkpoint_summary_combined.csv", combined_checkpoint_summary)
    _write_csv(combined_root / "capacity_data_runtime_summary_combined.csv", combined_runtime_summary)
    _write_csv(combined_root / "capacity_data_full_grid_summary.csv", combined_table)
    _write_csv(combined_root / "capacity_data_gcv_benchmark.csv", gcv_rows)

    existing_ids = set(row["config_id"] for row in existing_sel)
    _make_full_grid_plot(
        combined_root / "ncv_capacity_data_full_grid_summary.png",
        combined_per_rep,
        selected,
        cells,
        existing_ids,
        gcv_rows,
    )

    questions = _question_answers(combined_table, new_sel, gcv_rows)
    handover_lines = ["# NCV Capacity Data Missing Cells (m=252)", "", "## Final questions", *[f"- {line}" for line in questions]]
    (combined_root / "CAPACITY_DATA_MISSING_CELLS_HANDOVER.md").write_text("\n".join(handover_lines), encoding="utf-8")

    _write_json(
        combined_root / "capacity_data_combined_validation_report.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "existing_run_dir": str(existing_dir),
            "new_run_dir": str(computed_new_run_dir),
            "combined_run_dir": str(combined_root),
            "combined_unique_cells": sorted(selected.keys()),
            "combined_unique_cell_count": len(selected),
        },
    )
    return {
        "existing_run_dir": str(existing_dir),
        "new_run_dir": str(computed_new_run_dir),
        "combined_run_dir": str(combined_root),
    }


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run the four missing m=252 capacity-data cells and combine to nine-cell grid.")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--output-dir", default="experiment_runs/capacity_data_missing_cells_m252")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--existing-run-dir", default=None)
    p.add_argument("--new-run-dir", default=None)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    result = run_missing_cells_and_combine(
        repo_root=Path(args.repo_root).resolve(),
        output_dir=Path(args.output_dir),
        base_seed=args.base_seed,
        existing_run_dir=Path(args.existing_run_dir).resolve() if args.existing_run_dir else None,
        new_run_dir=Path(args.new_run_dir).resolve() if args.new_run_dir else None,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
