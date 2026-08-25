from __future__ import annotations

import argparse

from asian_options.parameter_conditioned_stage8 import profile_config, run_parameter_conditioned_stage8


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 8 parameter-conditioned NCV experiment (m=12)")
    p.add_argument("--profile", choices=["smoke", "dissertation"], default="smoke")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--output-dir", default="experiment_runs")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    cfg = profile_config(args.profile, output_dir=args.output_dir, base_seed=args.base_seed)
    out = run_parameter_conditioned_stage8(cfg)
    print(out)
