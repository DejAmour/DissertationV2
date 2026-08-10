from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asian_options.config import collect_environment_metadata
from asian_options.results import (
    RESULT_SCHEMA_VERSION,
    STATISTICAL_OUTPUT_COLUMNS,
    RUNTIME_OUTPUT_COLUMNS,
    PUBLICATION_TABLE_COLUMNS,
    METRIC_DEFINITIONS,
    METRIC_UNITS,
    PUBLICATION_TABLE_NOTES,
    normalize_statistical_rows,
    normalize_runtime_rows,
    build_publication_summary_rows,
    save_results_csv,
    write_stable_csv,
    write_publication_markdown,
)
from run_method_comparison import (
    CFG,
    run_equal_obs_comparison,
    run_equal_budget_comparison,
    run_runtime_comparison,
)


MODE_MAP = {
    "A": "A_equal_obs",
    "B": "B_equal_budget",
    "C": "C_runtime",
}


def _parse_seeds(value: str) -> list[int]:
    seeds: list[int] = []
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            seed = int(chunk)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid seed '{chunk}'") from exc
        seeds.append(seed)
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def _commit_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return "unavailable"
    return out.strip()


def _hash_config(config: dict) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.n_paths <= 0:
        parser.error("--n-paths must be > 0")
    if args.total_path_budget <= 0:
        parser.error("--total-path-budget must be > 0")
    if args.total_path_budget < 2 and "B" in args.modes:
        parser.error("--total-path-budget must be >= 2 for AV in Mode B")
    if args.pilot_paths < 0:
        parser.error("--pilot-paths must be >= 0")
    if args.training_paths < 0:
        parser.error("--training-paths must be >= 0")
    if args.replications <= 0:
        parser.error("--replications must be > 0")
    if any(seed < 0 for seed in args.seeds):
        parser.error("all seeds must be non-negative")

    if "B" in args.modes:
        if args.total_path_budget - args.pilot_paths <= 0:
            parser.error("Mode B invalid: total-path-budget - pilot-paths must be > 0")
        if args.total_path_budget - args.training_paths <= 0:
            parser.error("Mode B invalid: total-path-budget - training-paths must be > 0")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Canonical Stage 6 experiment runner for Asian-options MC/AV/CV/NCV comparisons "
            "across Mode A (equal observations), Mode B (equal total path budget), and Mode C (runtime/efficiency)."
        )
    )
    parser.add_argument("--output-dir", default="experiment_runs", help="Base output directory for timestamped run folders")
    parser.add_argument("--modes", nargs="+", choices=["A", "B", "C"], default=["A", "B", "C"], help="Modes to run")
    parser.add_argument("--seeds", type=_parse_seeds, default=[CFG.seed], help="Comma-separated base seed list")
    parser.add_argument("--replications", type=int, default=1, help="Replications per base seed")
    parser.add_argument("--n-paths", type=int, default=CFG.n_paths, help="Pricing observations for Modes A/C")
    parser.add_argument("--total-path-budget", type=int, default=50_000, help="Total simulated path budget for Mode B")
    parser.add_argument("--pilot-paths", type=int, default=1_000, help="Pilot paths for CV")
    parser.add_argument("--training-paths", type=int, default=5_000, help="Training paths for NCV")
    parser.add_argument(
        "--timing-scope-policy",
        choices=["exclude_ncv_training", "include_ncv_training"],
        default="exclude_ncv_training",
        help="Whether NCV training time is excluded or included in Mode C runtime_seconds",
    )
    return parser


def _with_run_tags(rows: Iterable[dict], seed: int, replication: int) -> list[dict]:
    tagged = []
    for row in rows:
        row_copy = dict(row)
        row_copy["seed"] = seed
        row_copy["replication"] = replication
        tagged.append(row_copy)
    return tagged


def _write_run_readme(path: Path, metadata: dict, manifest: dict) -> None:
    lines = [
        "Stage 6 experiment run",
        "======================",
        "",
        f"schema_version: {metadata['schema_version']}",
        f"created_at_utc: {metadata['created_at_utc']}",
        f"commit_hash: {metadata['commit_hash']}",
        f"timing_scope_policy: {metadata['timing_scope_policy']}",
        f"seeds: {metadata['seeds']}",
        f"replications: {metadata['replications']}",
        f"config_hash: {metadata['config_hash']}",
        "",
        "Files:",
    ]
    for f in manifest["files"]:
        lines.append(f"- {f['path']}: {f['purpose']}")
    lines.append("")
    lines.append("Metric notes:")
    lines.append(PUBLICATION_TABLE_NOTES.strip())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(parser, args)

    now = datetime.now(timezone.utc)
    run_stamp = now.strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = Path(args.output_dir).expanduser().resolve() / f"run_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)

    statistical_rows: list[dict] = []
    runtime_rows: list[dict] = []

    run_config = {
        "modes": args.modes,
        "seeds": args.seeds,
        "replications": args.replications,
        "n_paths": args.n_paths,
        "total_path_budget": args.total_path_budget,
        "pilot_paths": args.pilot_paths,
        "training_paths": args.training_paths,
        "timing_scope_policy": args.timing_scope_policy,
        "model_config_base": dataclasses.asdict(CFG),
    }

    for base_seed in args.seeds:
        for replication in range(args.replications):
            run_seed = base_seed + replication
            cfg = dataclasses.replace(CFG, n_paths=args.n_paths, seed=run_seed)

            if "A" in args.modes:
                rows = run_equal_obs_comparison(cfg=cfg, n_pilot=args.pilot_paths, n_training=args.training_paths)
                statistical_rows.extend(_with_run_tags(rows, run_seed, replication))

            if "B" in args.modes:
                rows = run_equal_budget_comparison(
                    cfg=cfg,
                    n_pilot=args.pilot_paths,
                    n_training=args.training_paths,
                    total_path_budget=args.total_path_budget,
                )
                statistical_rows.extend(_with_run_tags(rows, run_seed, replication))

            if "C" in args.modes:
                rows = run_runtime_comparison(
                    cfg=cfg,
                    n_pilot=args.pilot_paths,
                    n_training=args.training_paths,
                    timing_scope_policy=args.timing_scope_policy,
                )
                runtime_rows.extend(_with_run_tags(rows, run_seed, replication))

    stat_norm = normalize_statistical_rows(statistical_rows)
    runtime_norm = normalize_runtime_rows(runtime_rows)

    files: list[dict] = []
    if stat_norm:
        stats_path = run_dir / "mode_ab_statistical_raw.csv"
        save_results_csv(stat_norm, stats_path, fieldnames=STATISTICAL_OUTPUT_COLUMNS)
        files.append({"path": stats_path.name, "purpose": "Raw statistical outputs for Modes A/B", "columns": STATISTICAL_OUTPUT_COLUMNS})

    if runtime_norm:
        runtime_path = run_dir / "mode_c_runtime_raw.csv"
        save_results_csv(runtime_norm, runtime_path, fieldnames=RUNTIME_OUTPUT_COLUMNS)
        files.append({"path": runtime_path.name, "purpose": "Raw runtime outputs for Mode C", "columns": RUNTIME_OUTPUT_COLUMNS})

    publication_rows = build_publication_summary_rows(stat_norm, runtime_norm)

    merged_summary_path = run_dir / "merged_summary.csv"
    save_results_csv(publication_rows, merged_summary_path, fieldnames=PUBLICATION_TABLE_COLUMNS)
    files.append({"path": merged_summary_path.name, "purpose": "Merged publication-ready summary across selected modes", "columns": PUBLICATION_TABLE_COLUMNS})

    summary_stable_path = run_dir / "summary_stable.csv"
    write_stable_csv(
        publication_rows,
        summary_stable_path,
        fieldnames=PUBLICATION_TABLE_COLUMNS,
        sort_keys=("mode", "method", "pricing_observations", "seed", "replication"),
    )
    files.append({"path": summary_stable_path.name, "purpose": "Stable deterministic summary for reproducibility checks", "columns": PUBLICATION_TABLE_COLUMNS})

    paper_csv = run_dir / "paper_table.csv"
    save_results_csv(publication_rows, paper_csv, fieldnames=PUBLICATION_TABLE_COLUMNS)
    files.append({"path": paper_csv.name, "purpose": "Consolidated paper table (CSV)", "columns": PUBLICATION_TABLE_COLUMNS})

    paper_md = run_dir / "paper_table.md"
    write_publication_markdown(publication_rows, paper_md)
    files.append({"path": paper_md.name, "purpose": "Consolidated paper table (Markdown)", "columns": PUBLICATION_TABLE_COLUMNS})

    notes_path = run_dir / "paper_table_notes.txt"
    notes_path.write_text(PUBLICATION_TABLE_NOTES, encoding="utf-8")
    files.append({"path": notes_path.name, "purpose": "Metric footnotes and interpretation notes"})

    metadata = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "created_at_utc": now.isoformat(),
        "commit_hash": _commit_hash(),
        "timing_scope_policy": args.timing_scope_policy,
        "seeds": args.seeds,
        "replications": args.replications,
        "config_hash": _hash_config(run_config),
        "config": run_config,
        "environment": {
            **collect_environment_metadata(),
            "platform_summary": platform.platform(),
            "python": sys.version,
        },
    }

    manifest = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "mode_identifiers": MODE_MAP,
        "timing_scope_policy": args.timing_scope_policy,
        "files": files,
        "metric_definitions": METRIC_DEFINITIONS,
        "metric_units": METRIC_UNITS,
    }

    metadata_path = run_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    files.append({"path": metadata_path.name, "purpose": "Machine-readable run metadata (config, seed, environment, config hash)"})

    manifest_path = run_dir / "manifest.json"
    files.append({"path": manifest_path.name, "purpose": "Run manifest (schema version, files, metric definitions, units)"})
    files.append({"path": "README.txt", "purpose": "Human-readable run summary and usage notes"})
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _write_run_readme(run_dir / "README.txt", metadata=metadata, manifest=manifest)

    print(f"Created run artifacts in: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
