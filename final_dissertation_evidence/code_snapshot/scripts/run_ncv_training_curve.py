from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from asian_options.ncv_training_curve import (
    TrainingCurveConfig,
    profile_config,
    run_training_curve_experiment,
    validate_checkpoints,
)


def _parse_checkpoints(raw: str | None) -> tuple[int, ...] | None:
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    values = tuple(int(p) for p in parts)
    validate_checkpoints(values)
    return values


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run NCV training-curve experiment for the reference arithmetic Asian option")
    p.add_argument("--profile", choices=["smoke", "dissertation"], default="smoke")
    p.add_argument("--output-dir", default="experiment_runs")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--replications", type=int)
    p.add_argument("--train-paths", type=int)
    p.add_argument("--validation-paths", type=int)
    p.add_argument("--test-paths", type=int)
    p.add_argument("--monitoring-dates", type=int)
    p.add_argument("--checkpoints", type=str, help="Comma-separated checkpoints, e.g. 0,10,25,50")
    p.add_argument("--hidden-width", type=int)
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--default-epochs", type=int)
    p.add_argument("--train-batch-size", type=int)
    p.add_argument("--runtime-repeats", type=int)
    p.add_argument("--pilot-paths", type=int)
    p.add_argument("--timing-path-counts", type=str, help="Comma-separated bounded timing path counts")
    p.add_argument("--timing-repeats", type=int)
    p.add_argument("--pricing-observations-for-reporting", type=int)
    p.add_argument("--q-values", type=str, help="Comma-separated Q values")
    return p


def _apply_overrides(base: TrainingCurveConfig, args: argparse.Namespace) -> TrainingCurveConfig:
    updates = dataclasses.asdict(base)
    if args.replications is not None:
        updates["replications"] = args.replications
    if args.train_paths is not None:
        updates["train_paths"] = args.train_paths
    if args.validation_paths is not None:
        updates["validation_paths"] = args.validation_paths
    if args.test_paths is not None:
        updates["test_paths"] = args.test_paths
    if args.monitoring_dates is not None:
        updates["monitoring_dates"] = args.monitoring_dates
    cps = _parse_checkpoints(args.checkpoints)
    if cps is not None:
        updates["checkpoints"] = cps
    if args.hidden_width is not None:
        updates["hidden_width"] = args.hidden_width
    if args.learning_rate is not None:
        updates["learning_rate"] = args.learning_rate
    if args.default_epochs is not None:
        updates["default_epochs"] = args.default_epochs
    if args.train_batch_size is not None:
        updates["train_batch_size"] = args.train_batch_size
    if args.runtime_repeats is not None:
        updates["runtime_repeats"] = args.runtime_repeats
    if args.pilot_paths is not None:
        updates["pilot_paths"] = args.pilot_paths
    if args.timing_path_counts:
        updates["timing_path_counts"] = tuple(int(x.strip()) for x in args.timing_path_counts.split(",") if x.strip())
    if args.timing_repeats is not None:
        updates["timing_repeats"] = args.timing_repeats
    if args.pricing_observations_for_reporting is not None:
        updates["pricing_observations_for_reporting"] = args.pricing_observations_for_reporting
    if args.q_values:
        updates["q_values"] = tuple(int(x.strip()) for x in args.q_values.split(",") if x.strip())
    cfg = TrainingCurveConfig(**updates)
    validate_checkpoints(cfg.checkpoints)
    return cfg


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    base = profile_config(args.profile, output_dir=args.output_dir, base_seed=args.base_seed)
    cfg = _apply_overrides(base, args)
    out_dir = run_training_curve_experiment(cfg)
    print(f"NCV training-curve output: {Path(out_dir).resolve()}")


if __name__ == "__main__":
    main()
