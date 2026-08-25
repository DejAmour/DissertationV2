from __future__ import annotations

import argparse
import csv
import dataclasses
import itertools
import json
import math
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as _t_dist

from asian_options.config import ModelConfig, collect_environment_metadata, seed_everything
from asian_options.contracts import REFERENCE_ID, make_contract_cfg
from asian_options.ncv_training_curve import (
    _network_from_torch_model,
    _torch_model_from_initial_network,
    _write_csv,
    _write_json,
    compute_gcv_benchmark,
    compute_ncv_setup_cost,
    compute_ncv_split_diagnostics,
    measure_inference_runtime,
)
from asian_options.neural_cv import analytical_network_expectation, build_network
from asian_options.payoffs import arithmetic_asian_call_payoff, geometric_asian_call_payoff
from asian_options.simulate_gbm import simulate_paths


SEED_OFFSETS = {
    "train": 1_000,
    "validation": 2_000,
    "test": 3_000,
    "pilot": 4_000,
    "init": 5_000,
}


@dataclass(frozen=True)
class CapacityCell:
    config_id: str
    hidden_width: int
    train_paths: int


@dataclass(frozen=True)
class CapacityDataConfig:
    profile: str
    base_seed: int
    replications: int
    monitoring_dates: int
    validation_paths: int
    test_paths: int
    pilot_paths: int
    checkpoints: tuple[int, ...]
    learning_rate: float
    train_batch_size: int
    pricing_observations_for_reporting: int
    cells: tuple[CapacityCell, ...]
    output_dir: str


@dataclass(frozen=True)
class _CheckpointSnapshot:
    epoch: int
    network: Any
    train_mse: float
    validation_mse: float
    test_mse: float
    optimizer_cumulative_runtime_s: float
    e_h: float
    training_diag: dict[str, float]
    validation_diag: dict[str, float]
    test_diag: dict[str, float]
    inference: dict[str, float]


def _default_cells(profile: str, *, dissertation_cell_set: str = "baseline") -> tuple[CapacityCell, ...]:
    p = profile.lower()
    if p == "smoke":
        return (
            CapacityCell("w32_n100", 32, 100),
            CapacityCell("w16_n100", 16, 100),
            CapacityCell("w8_n100", 8, 100),
            CapacityCell("w32_n200", 32, 200),
            CapacityCell("w32_n400", 32, 400),
        )
    if p == "dissertation":
        cell_set = dissertation_cell_set.lower()
        if cell_set == "missing_cells":
            return (
                CapacityCell("w8_n10000", 8, 10_000),
                CapacityCell("w8_n20000", 8, 20_000),
                CapacityCell("w16_n10000", 16, 10_000),
                CapacityCell("w16_n20000", 16, 20_000),
            )
        if cell_set == "full_grid":
            return (
                CapacityCell("w8_n5000", 8, 5_000),
                CapacityCell("w8_n10000", 8, 10_000),
                CapacityCell("w8_n20000", 8, 20_000),
                CapacityCell("w16_n5000", 16, 5_000),
                CapacityCell("w16_n10000", 16, 10_000),
                CapacityCell("w16_n20000", 16, 20_000),
                CapacityCell("w32_n5000", 32, 5_000),
                CapacityCell("w32_n10000", 32, 10_000),
                CapacityCell("w32_n20000", 32, 20_000),
            )
        return (
            CapacityCell("w32_n5000", 32, 5_000),
            CapacityCell("w16_n5000", 16, 5_000),
            CapacityCell("w8_n5000", 8, 5_000),
            CapacityCell("w32_n10000", 32, 10_000),
            CapacityCell("w32_n20000", 32, 20_000),
        )
    raise ValueError(f"unknown profile: {profile}")


def profile_config(
    profile: str,
    output_dir: str,
    base_seed: int = 42,
    dissertation_cell_set: str = "baseline",
) -> CapacityDataConfig:
    p = profile.lower()
    if p == "smoke":
        return CapacityDataConfig(
            profile="smoke",
            base_seed=base_seed,
            replications=2,
            monitoring_dates=252,
            validation_paths=200,
            test_paths=500,
            pilot_paths=100,
            checkpoints=(0, 1, 2),
            learning_rate=1e-2,
            train_batch_size=256,
            pricing_observations_for_reporting=500,
            cells=_default_cells("smoke", dissertation_cell_set="baseline"),
            output_dir=output_dir,
        )
    if p == "dissertation":
        return CapacityDataConfig(
            profile="dissertation",
            base_seed=base_seed,
            replications=10,
            monitoring_dates=252,
            validation_paths=10_000,
            test_paths=50_000,
            pilot_paths=1_000,
            checkpoints=(0, 10, 25, 50, 100, 200),
            learning_rate=1e-2,
            train_batch_size=256,
            pricing_observations_for_reporting=50_000,
            cells=_default_cells("dissertation", dissertation_cell_set=dissertation_cell_set),
            output_dir=output_dir,
        )
    raise ValueError(f"unknown profile: {profile}")


def _ensure_unique_run_dir(base_dir: Path, stem: str) -> Path:
    candidate = base_dir / stem
    if not candidate.exists():
        candidate.mkdir(parents=True, exist_ok=False)
        return candidate
    for idx in range(1, 10_000):
        alt = base_dir / f"{stem}_{idx}"
        if not alt.exists():
            alt.mkdir(parents=True, exist_ok=False)
            return alt
    raise RuntimeError(f"unable to create unique run dir under {base_dir}")


def _validate_checkpoints(checkpoints: tuple[int, ...]) -> None:
    if not checkpoints:
        raise ValueError("checkpoints cannot be empty")
    if checkpoints[0] != 0:
        raise ValueError("checkpoints must start at 0")
    for i in range(1, len(checkpoints)):
        if checkpoints[i] <= checkpoints[i - 1]:
            raise ValueError("checkpoints must be strictly increasing")


def _split_seeds(base_seed: int, replication: int) -> dict[str, int]:
    rep_offset = int(replication) * 100_000
    s = int(base_seed) + rep_offset
    return {k: s + v for k, v in SEED_OFFSETS.items()}


def _network_seed(replication_seeds: dict[str, int], cell: CapacityCell) -> int:
    # Width-32 data-size cells must share the same init seed.
    if cell.hidden_width == 32:
        return int(replication_seeds["init"])
    return int(replication_seeds["init"] + cell.hidden_width * 10_000 + cell.train_paths)


def _build_seed_manifest(config: CapacityDataConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rep in range(config.replications):
        seeds = _split_seeds(config.base_seed, rep)
        for stream, value in seeds.items():
            rows.append({"replication": rep, "seed_stream": stream, "seed": int(value)})
        for cell in config.cells:
            rows.append(
                {
                    "replication": rep,
                    "seed_stream": f"network_init::{cell.config_id}",
                    "seed": _network_seed(seeds, cell),
                }
            )
    return rows


def _sample_var(x: np.ndarray) -> float:
    if x.size <= 1:
        return float("nan")
    return float(np.var(x, ddof=1))


def _student_t_ci(values: list[float]) -> dict[str, Any]:
    if len(values) == 0:
        return {
            "n": 0,
            "mean": "NA",
            "std_dev": "NA",
            "median": "NA",
            "ci95_lower": "NA",
            "ci95_upper": "NA",
            "ci_method": "NA",
            "ci_status": "no_observations",
        }
    mean = float(statistics.mean(values))
    std_dev = float(statistics.stdev(values)) if len(values) >= 2 else 0.0
    median = float(statistics.median(values))
    if len(values) == 1:
        return {
            "n": 1,
            "mean": mean,
            "std_dev": std_dev,
            "median": median,
            "ci95_lower": "NA",
            "ci95_upper": "NA",
            "ci_method": "NA",
            "ci_status": "undefined_n_equals_1",
        }
    dof = len(values) - 1
    t_crit = float(_t_dist.ppf(0.975, df=dof))
    half = t_crit * std_dev / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "std_dev": std_dev,
        "median": median,
        "ci95_lower": mean - half,
        "ci95_upper": mean + half,
        "ci_method": "student-t",
        "ci_status": "ok",
    }


def _make_reference_cfg(monitoring_dates: int, n_paths: int, seed: int) -> ModelConfig:
    base_cfg = make_contract_cfg(REFERENCE_ID, n_paths=n_paths, seed=seed)
    return dataclasses.replace(base_cfg, m=monitoring_dates, n_paths=n_paths, seed=seed)


def _simulate_split_dataset(cfg: ModelConfig, z: np.ndarray) -> dict[str, Any]:
    paths = simulate_paths(cfg, shocks=z)
    payoff_arith = arithmetic_asian_call_payoff(paths, cfg)
    payoff_geom = geometric_asian_call_payoff(paths, cfg)
    return {
        "cfg": cfg,
        "Z": z,
        "payoff_arithmetic": payoff_arith,
        "payoff_geometric": payoff_geom,
    }


def _generate_nested_training_splits(
    *,
    monitoring_dates: int,
    training_seed: int,
    training_sizes: list[int],
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    sizes = sorted(set(int(x) for x in training_sizes))
    max_n = sizes[-1]
    rows: list[dict[str, Any]] = []

    generated: dict[int, np.ndarray] = {}
    runtimes: dict[int, float] = {}
    for n in sizes:
        cfg = _make_reference_cfg(monitoring_dates, n, training_seed)
        t0 = time.perf_counter()
        rng = np.random.default_rng(training_seed)
        z = rng.standard_normal((n, monitoring_dates))
        runtimes[n] = time.perf_counter() - t0
        generated[n] = z
        rows.append(
            {
                "check": "training_nested_generation",
                "train_seed": int(training_seed),
                "n_paths": int(n),
                "generation_runtime_s": float(runtimes[n]),
            }
        )

    nested_equal = True
    nested_failures: list[str] = []
    z_max = generated[max_n]
    for n in sizes[:-1]:
        ok = np.array_equal(generated[n], z_max[:n])
        nested_equal = nested_equal and bool(ok)
        if not ok:
            nested_failures.append(f"size {n} is not a prefix of size {max_n}")
        rows.append(
            {
                "check": "training_nested_prefix",
                "small_n": int(n),
                "large_n": int(max_n),
                "is_equal_prefix": bool(ok),
            }
        )

    splits: dict[int, dict[str, Any]] = {}
    for n in sizes:
        cfg = _make_reference_cfg(monitoring_dates, n, training_seed)
        z = z_max[:n].copy()
        splits[n] = _simulate_split_dataset(cfg, z)

    diagnostics = {
        "training_nested_prefix_ok": nested_equal,
        "training_nested_prefix_failures": nested_failures,
        "training_sizes": sizes,
        "max_training_paths": int(max_n),
        "generation_runtime_by_size_s": {str(k): float(v) for k, v in runtimes.items()},
    }
    return splits, rows, diagnostics


def _train_continuous_snapshots(
    *,
    torch_mod,
    train_split: dict[str, Any],
    validation_split: dict[str, Any],
    test_split: dict[str, Any],
    hidden_width: int,
    learning_rate: float,
    batch_size: int,
    checkpoints: tuple[int, ...],
    n_reporting: int,
    init_seed: int,
) -> dict[int, _CheckpointSnapshot]:
    cfg_train = dataclasses.replace(train_split["cfg"], seed=int(init_seed))
    network = build_network(cfg_train, hidden_width=hidden_width)
    model = _torch_model_from_initial_network(torch_mod, network)
    optimizer = torch_mod.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch_mod.nn.MSELoss()

    x_train = torch_mod.tensor(train_split["Z"], dtype=torch_mod.float64)
    y_train = torch_mod.tensor(train_split["payoff_arithmetic"], dtype=torch_mod.float64)
    x_val = torch_mod.tensor(validation_split["Z"], dtype=torch_mod.float64)
    y_val = torch_mod.tensor(validation_split["payoff_arithmetic"], dtype=torch_mod.float64)
    x_test = torch_mod.tensor(test_split["Z"], dtype=torch_mod.float64)
    y_test = torch_mod.tensor(test_split["payoff_arithmetic"], dtype=torch_mod.float64)

    snapshots: dict[int, _CheckpointSnapshot] = {}
    batch = min(int(batch_size), len(x_train))

    def capture(epoch: int, cumulative_runtime_s: float) -> None:
        model.eval()
        with torch_mod.no_grad():
            train_mse = float(loss_fn(model(x_train), y_train).item())
            val_mse = float(loss_fn(model(x_val), y_val).item())
            test_mse = float(loss_fn(model(x_test), y_test).item())

        net = _network_from_torch_model(model)
        e_h = float(analytical_network_expectation(net))

        train_h = net.forward(train_split["Z"])
        val_h = net.forward(validation_split["Z"])
        test_h = net.forward(test_split["Z"])

        train_diag = compute_ncv_split_diagnostics(
            payoff=train_split["payoff_arithmetic"],
            h_vals=train_h,
            e_h=e_h,
            n_reporting=n_reporting,
        )
        val_diag = compute_ncv_split_diagnostics(
            payoff=validation_split["payoff_arithmetic"],
            h_vals=val_h,
            e_h=e_h,
            n_reporting=n_reporting,
        )
        test_diag = compute_ncv_split_diagnostics(
            payoff=test_split["payoff_arithmetic"],
            h_vals=test_h,
            e_h=e_h,
            n_reporting=n_reporting,
        )
        inference = measure_inference_runtime(net, test_split["Z"], repeats=3)
        snapshots[int(epoch)] = _CheckpointSnapshot(
            epoch=int(epoch),
            network=net,
            train_mse=train_mse,
            validation_mse=val_mse,
            test_mse=test_mse,
            optimizer_cumulative_runtime_s=float(cumulative_runtime_s),
            e_h=e_h,
            training_diag=train_diag,
            validation_diag=val_diag,
            test_diag=test_diag,
            inference=inference,
        )

    capture(0, 0.0)

    cp_set = set(int(cp) for cp in checkpoints)
    t_train = time.perf_counter()
    for epoch in range(1, int(max(checkpoints)) + 1):
        model.train()
        perm = torch_mod.randperm(len(x_train))
        for start in range(0, len(x_train), batch):
            idx = perm[start : start + batch]
            xb = x_train[idx]
            yb = y_train[idx]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        elapsed = time.perf_counter() - t_train
        if epoch in cp_set:
            capture(epoch, elapsed)

    missing = [cp for cp in checkpoints if int(cp) not in snapshots]
    if missing:
        raise RuntimeError(f"missing checkpoints: {missing}")
    return snapshots


def _parameter_count(model) -> int:
    return int(sum(int(p.numel()) for p in model.parameters()))


def _expected_parameter_count(m: int, h: int) -> int:
    return int(h * (m + 2) + 1)


def _safe_log_ratio(num: float, den: float) -> float:
    if not (math.isfinite(num) and math.isfinite(den)):
        return float("nan")
    if num <= 0.0 or den <= 0.0:
        return float("nan")
    return float(math.log(num / den))


def _safe_ratio(num: float, den: float) -> float:
    if not (math.isfinite(num) and math.isfinite(den)):
        return float("nan")
    if den == 0.0:
        return float("nan")
    return float(num / den)


def _safe_generalization_gap(validation_mse: float, training_mse: float) -> tuple[float, float]:
    ratio = _safe_ratio(validation_mse, training_mse)
    log_ratio = float("nan")
    if math.isfinite(ratio) and ratio > 0.0:
        log_ratio = float(math.log(ratio))
    return ratio, log_ratio


def _pick_checkpoint_from_rows(rows: list[dict[str, Any]], checkpoints: tuple[int, ...]) -> int:
    finite = [r for r in rows if math.isfinite(float(r["centered_residual_variance"]))]
    if not finite:
        return int(checkpoints[0])
    # min() over checkpoint key ensures earliest tie-break.
    return int(min(finite, key=lambda r: (float(r["centered_residual_variance"]), int(r["checkpoint"])))["checkpoint"])


def _selected_checkpoints(per_replication_rows: list[dict[str, Any]], config: CapacityDataConfig) -> tuple[list[dict[str, Any]], dict[str, int]]:
    selected_rows: list[dict[str, Any]] = []
    selected_by_config: dict[str, int] = {}

    for cell in config.cells:
        rows = [
            r
            for r in per_replication_rows
            if r["config_id"] == cell.config_id and r["split"] == "validation"
        ]
        by_cp: dict[int, list[float]] = {}
        for row in rows:
            cp = int(row["checkpoint"])
            by_cp.setdefault(cp, []).append(float(row["centered_residual_variance"]))

        cp_summary: list[tuple[int, float]] = []
        checkpoints_sorted = sorted(by_cp.keys())
        for cp in checkpoints_sorted:
            vals = [x for x in by_cp[cp] if math.isfinite(x)]
            if vals:
                cp_summary.append((cp, float(statistics.mean(vals))))
        if cp_summary:
            selected_cp = min(cp_summary, key=lambda x: (x[1], x[0]))[0]
        else:
            selected_cp = int(config.checkpoints[0])
        selected_by_config[cell.config_id] = int(selected_cp)

        rep_optimal = {}
        for rep in range(config.replications):
            rep_rows = [r for r in rows if int(r["replication"]) == rep]
            rep_optimal[rep] = _pick_checkpoint_from_rows(rep_rows, config.checkpoints)

        selected_rows.append(
            {
                "config_id": cell.config_id,
                "hidden_width": int(cell.hidden_width),
                "train_paths": int(cell.train_paths),
                "selected_checkpoint": int(selected_cp),
                "selection_metric": "mean_validation_centered_residual_variance",
                "tie_break_rule": "earlier_checkpoint",
                "per_replication_validation_optimal_checkpoints": "|".join(
                    f"rep{rep}:{rep_optimal[rep]}" for rep in range(config.replications)
                ),
                "checkpoint_grid": "|".join(str(int(cp)) for cp in config.checkpoints),
            }
        )
    return selected_rows, selected_by_config


def _paired_metric_summary(values: list[float]) -> dict[str, Any]:
    finite = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v))]
    ci = _student_t_ci(finite)
    return {
        "n_pairs": ci["n"],
        "mean": ci["mean"],
        "std_dev": ci["std_dev"],
        "median": ci["median"],
        "ci95_lower": ci["ci95_lower"],
        "ci95_upper": ci["ci95_upper"],
        "ci_method": ci["ci_method"],
        "ci_status": ci["ci_status"],
    }


def _build_paired_contrasts(
    per_replication_rows: list[dict[str, Any]],
    selected_by_config: dict[str, int],
    comparisons: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    lookup = {}
    for row in per_replication_rows:
        if row["split"] != "test":
            continue
        lookup[(int(row["replication"]), row["config_id"], int(row["checkpoint"]))] = row

    out: list[dict[str, Any]] = []
    for left, right in comparisons:
        left_cp = int(selected_by_config[left])
        right_cp = int(selected_by_config[right])

        d_test_mse: list[float] = []
        d_log_resid_var_ratio: list[float] = []
        d_vrr_ratio: list[float] = []
        d_log_gen_gap: list[float] = []
        d_setup_runtime: list[float] = []

        for rep in sorted(set(int(r["replication"]) for r in per_replication_rows)):
            row_l = lookup.get((rep, left, left_cp))
            row_r = lookup.get((rep, right, right_cp))
            if row_l is None or row_r is None:
                continue
            d_test_mse.append(float(row_l["mse"]) - float(row_r["mse"]))
            d_log_resid_var_ratio.append(
                _safe_log_ratio(float(row_l["centered_residual_variance"]), float(row_r["centered_residual_variance"]))
            )
            d_vrr_ratio.append(_safe_ratio(float(row_l["ncv_vrr_vs_mc"]), float(row_r["ncv_vrr_vs_mc"])))
            d_log_gen_gap.append(float(row_l["generalization_gap_log"]) - float(row_r["generalization_gap_log"]))
            d_setup_runtime.append(float(row_l["ncv_setup_cost_s"]) - float(row_r["ncv_setup_cost_s"]))

        for metric_name, values, direction in (
            ("paired_difference_test_mse", d_test_mse, "left_minus_right; lower_is_better"),
            (
                "paired_log_ratio_test_residual_variance",
                d_log_resid_var_ratio,
                "log(left/right); negative_favors_left",
            ),
            ("paired_ratio_test_vrr", d_vrr_ratio, "left_div_right; above_1_favors_left"),
            (
                "paired_difference_log_generalization_gap",
                d_log_gen_gap,
                "left_minus_right; lower_is_better",
            ),
            ("paired_difference_setup_runtime_s", d_setup_runtime, "left_minus_right; lower_is_better"),
        ):
            summary = _paired_metric_summary(values)
            out.append(
                {
                    "comparison": f"{left}_vs_{right}",
                    "left_config_id": left,
                    "right_config_id": right,
                    "left_selected_checkpoint": left_cp,
                    "right_selected_checkpoint": right_cp,
                    "metric": metric_name,
                    "direction": direction,
                    **summary,
                }
            )
    return out


def _default_paired_comparisons(cells: tuple[CapacityCell, ...]) -> list[tuple[str, str]]:
    by_key = {(int(c.hidden_width), int(c.train_paths)): c.config_id for c in cells}
    by_train: dict[int, list[int]] = {}
    by_width: dict[int, list[int]] = {}
    for c in cells:
        by_train.setdefault(int(c.train_paths), []).append(int(c.hidden_width))
        by_width.setdefault(int(c.hidden_width), []).append(int(c.train_paths))

    out: list[tuple[str, str]] = []
    for n in sorted(by_train):
        widths = sorted(set(by_train[n]))
        for wl, wr in itertools.combinations(widths, 2):
            out.append((by_key[(wl, n)], by_key[(wr, n)]))
    for w in sorted(by_width):
        sizes = sorted(set(by_width[w]))
        for nl, nr in itertools.combinations(sizes, 2):
            out.append((by_key[(w, nr)], by_key[(w, nl)]))
    return out


def _build_checkpoint_summary(rows: list[dict[str, Any]], config: CapacityDataConfig) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cell in config.cells:
        for cp in config.checkpoints:
            for split in ("training", "validation", "test"):
                subset = [
                    r
                    for r in rows
                    if r["config_id"] == cell.config_id and int(r["checkpoint"]) == int(cp) and r["split"] == split
                ]
                if not subset:
                    continue
                for metric in (
                    "mse",
                    "centered_residual_variance",
                    "ncv_vrr_vs_mc",
                    "generalization_gap_ratio",
                    "generalization_gap_log",
                    "ncv_setup_cost_s",
                ):
                    vals = [float(r[metric]) for r in subset if math.isfinite(float(r[metric]))]
                    if not vals:
                        continue
                    ci = _student_t_ci(vals)
                    out.append(
                        {
                            "config_id": cell.config_id,
                            "hidden_width": cell.hidden_width,
                            "train_paths": cell.train_paths,
                            "checkpoint": int(cp),
                            "split": split,
                            "metric": metric,
                            "count": ci["n"],
                            "mean": ci["mean"],
                            "std_dev": ci["std_dev"],
                            "median": ci["median"],
                            "ci95_lower": ci["ci95_lower"],
                            "ci95_upper": ci["ci95_upper"],
                            "ci_method": ci["ci_method"],
                            "ci_status": ci["ci_status"],
                        }
                    )
    return out


def _build_runtime_summary(rows: list[dict[str, Any]], selected: dict[str, int], config: CapacityDataConfig) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for cell in config.cells:
        cp = int(selected[cell.config_id])
        subset = [
            r
            for r in rows
            if r["config_id"] == cell.config_id and int(r["checkpoint"]) == cp and r["split"] == "test"
        ]
        if not subset:
            continue
        for metric in (
            "training_data_generation_runtime_s",
            "optimizer_cumulative_runtime_s",
            "validation_tuning_overhead_runtime_s",
            "ncv_setup_cost_s",
            "inference_runtime_per_observation_s",
        ):
            vals = [float(r[metric]) for r in subset if math.isfinite(float(r[metric]))]
            if not vals:
                continue
            ci = _student_t_ci(vals)
            out.append(
                {
                    "config_id": cell.config_id,
                    "hidden_width": cell.hidden_width,
                    "train_paths": cell.train_paths,
                    "selected_checkpoint": cp,
                    "metric": metric,
                    "count": ci["n"],
                    "mean": ci["mean"],
                    "std_dev": ci["std_dev"],
                    "median": ci["median"],
                    "ci95_lower": ci["ci95_lower"],
                    "ci95_upper": ci["ci95_upper"],
                }
            )
    return out


def _plot_summary(run_dir: Path, rows: list[dict[str, Any]], selected: dict[str, int], config: CapacityDataConfig) -> None:
    import matplotlib.pyplot as plt

    def series(config_id: str, split: str, metric: str) -> tuple[list[int], list[float], list[float], list[float]]:
        cps = list(config.checkpoints)
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for cp in cps:
            vals = [
                float(r[metric])
                for r in rows
                if r["config_id"] == config_id and r["split"] == split and int(r["checkpoint"]) == int(cp)
                and math.isfinite(float(r[metric]))
            ]
            ci = _student_t_ci(vals)
            means.append(float(ci["mean"]) if isinstance(ci["mean"], (int, float)) else float("nan"))
            lows.append(float(ci["ci95_lower"]) if isinstance(ci["ci95_lower"], (int, float)) else float("nan"))
            highs.append(float(ci["ci95_upper"]) if isinstance(ci["ci95_upper"], (int, float)) else float("nan"))
        return cps, means, lows, highs

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    cells = [c.config_id for c in config.cells]
    palette = list(plt.cm.tab10.colors) + list(plt.cm.Set3.colors) + list(plt.cm.Dark2.colors)
    colors = {cid: palette[i % len(palette)] for i, cid in enumerate(cells)}

    # Panel 1: train/val/test MSE curves by configuration
    ax = axes[0, 0]
    for cell in config.cells:
        for split, style in (("training", "-"), ("validation", "--"), ("test", ":")):
            cps, means, lows, highs = series(cell.config_id, split, "mse")
            label = f"{cell.config_id} {split}"
            ax.plot(cps, means, linestyle=style, color=colors[cell.config_id], linewidth=1.2, label=label)
            ax.fill_between(cps, lows, highs, color=colors[cell.config_id], alpha=0.08)
    ax.set_title("Loss by checkpoint")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("MSE")

    # Panel 2: selected checkpoint test VRR
    ax = axes[0, 1]
    x = np.arange(len(config.cells))
    means = []
    lows = []
    highs = []
    labels = []
    for cell in config.cells:
        cp = int(selected[cell.config_id])
        vals = [
            float(r["ncv_vrr_vs_mc"])
            for r in rows
            if r["config_id"] == cell.config_id and r["split"] == "test" and int(r["checkpoint"]) == cp
            and math.isfinite(float(r["ncv_vrr_vs_mc"]))
        ]
        ci = _student_t_ci(vals)
        means.append(float(ci["mean"]) if isinstance(ci["mean"], (int, float)) else float("nan"))
        lows.append(float(ci["ci95_lower"]) if isinstance(ci["ci95_lower"], (int, float)) else float("nan"))
        highs.append(float(ci["ci95_upper"]) if isinstance(ci["ci95_upper"], (int, float)) else float("nan"))
        labels.append(f"{cell.config_id}\ncp={cp}")
    ax.bar(x, means, color=[colors[c.config_id] for c in config.cells], alpha=0.8)
    yerr = np.array([np.array(means) - np.array(lows), np.array(highs) - np.array(means)])
    ax.errorbar(x, means, yerr=yerr, fmt="none", color="black", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("Selected-checkpoint test VRR")
    ax.set_ylabel("VRR vs MC")

    # Panel 3: generalization gap log at selected checkpoint
    ax = axes[1, 0]
    g_means = []
    g_lows = []
    g_highs = []
    for cell in config.cells:
        cp = int(selected[cell.config_id])
        vals = [
            float(r["generalization_gap_log"])
            for r in rows
            if r["config_id"] == cell.config_id and r["split"] == "test" and int(r["checkpoint"]) == cp
            and math.isfinite(float(r["generalization_gap_log"]))
        ]
        ci = _student_t_ci(vals)
        g_means.append(float(ci["mean"]) if isinstance(ci["mean"], (int, float)) else float("nan"))
        g_lows.append(float(ci["ci95_lower"]) if isinstance(ci["ci95_lower"], (int, float)) else float("nan"))
        g_highs.append(float(ci["ci95_upper"]) if isinstance(ci["ci95_upper"], (int, float)) else float("nan"))
    ax.bar(x, g_means, color=[colors[c.config_id] for c in config.cells], alpha=0.8)
    yerr = np.array([np.array(g_means) - np.array(g_lows), np.array(g_highs) - np.array(g_means)])
    ax.errorbar(x, g_means, yerr=yerr, fmt="none", color="black", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("Generalization gap log(val/train)")
    ax.set_ylabel("log gap")

    # Panel 4: setup runtime at selected checkpoint
    ax = axes[1, 1]
    r_means = []
    r_lows = []
    r_highs = []
    for cell in config.cells:
        cp = int(selected[cell.config_id])
        vals = [
            float(r["ncv_setup_cost_s"])
            for r in rows
            if r["config_id"] == cell.config_id and r["split"] == "test" and int(r["checkpoint"]) == cp
            and math.isfinite(float(r["ncv_setup_cost_s"]))
        ]
        ci = _student_t_ci(vals)
        r_means.append(float(ci["mean"]) if isinstance(ci["mean"], (int, float)) else float("nan"))
        r_lows.append(float(ci["ci95_lower"]) if isinstance(ci["ci95_lower"], (int, float)) else float("nan"))
        r_highs.append(float(ci["ci95_upper"]) if isinstance(ci["ci95_upper"], (int, float)) else float("nan"))
    ax.bar(x, r_means, color=[colors[c.config_id] for c in config.cells], alpha=0.8)
    yerr = np.array([np.array(r_means) - np.array(r_lows), np.array(r_highs) - np.array(r_means)])
    ax.errorbar(x, r_means, yerr=yerr, fmt="none", color="black", capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_title("Setup runtime")
    ax.set_ylabel("seconds")

    handles, labels_ = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=3, fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(run_dir / "ncv_capacity_data_summary.png", dpi=250)
    plt.close(fig)


def _required_output_files() -> list[str]:
    return [
        "capacity_data_config.json",
        "capacity_data_environment.json",
        "capacity_data_seed_manifest.csv",
        "capacity_data_per_replication.csv",
        "capacity_data_checkpoint_summary.csv",
        "capacity_data_selected_checkpoints.csv",
        "capacity_data_paired_contrasts.csv",
        "capacity_data_runtime_summary.csv",
        "capacity_data_gcv_benchmark.csv",
        "capacity_data_validation_report.json",
        "CAPACITY_DATA_HANDOVER.md",
        "ncv_capacity_data_summary.png",
    ]


def _validate_outputs(run_dir: Path, diagnostics: dict[str, Any]) -> dict[str, Any]:
    required = _required_output_files()
    exists = {name: (run_dir / name).exists() for name in required}
    errors: list[str] = []
    warnings: list[str] = []
    if not all(exists.values()):
        for name, ok in exists.items():
            if not ok:
                errors.append(f"missing required output file: {name}")

    if not diagnostics.get("training_nested_prefix_all_ok", False):
        errors.append("nested training sample prefix checks failed")
    if not diagnostics.get("seed_streams_unique_within_replication", False):
        errors.append("seed streams are not unique within replication")
    if diagnostics.get("extension_recommended"):
        warnings.append("extension_recommended_true")

    report = {
        "required_files": required,
        "exists": exists,
        "errors": errors,
        "warnings": warnings,
        "passed": len(errors) == 0,
        **diagnostics,
    }
    return report


def _build_handover(
    *,
    config: CapacityDataConfig,
    selected_rows: list[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    extension_recommended: bool,
    validation_report: dict[str, Any],
) -> str:
    lines: list[str] = []
    lines.append("# NCV Capacity-vs-Data Sensitivity Handover")
    lines.append("")
    lines.append("## Experiment")
    lines.append(f"- profile={config.profile}")
    lines.append(f"- replications={config.replications}")
    lines.append(f"- monitoring_dates={config.monitoring_dates}")
    lines.append(f"- checkpoints={list(config.checkpoints)}")
    lines.append(f"- configurations={len(config.cells)}")
    lines.append("")
    lines.append("## Selected checkpoints by configuration")
    for row in selected_rows:
        lines.append(f"- {row['config_id']}: checkpoint {row['selected_checkpoint']}")
    lines.append("")
    lines.append("## Paired-contrast outputs")
    lines.append(f"- paired contrast rows: {len(paired_rows)}")
    lines.append("- See capacity_data_paired_contrasts.csv for means, medians, and 95% confidence intervals.")
    lines.append("")
    lines.append(f"## Epoch-200 extension flag\n- extension_recommended={str(bool(extension_recommended)).lower()}")
    lines.append("")
    lines.append("## Warnings and limitations")
    if validation_report.get("warnings"):
        for w in validation_report["warnings"]:
            lines.append(f"- {w}")
    else:
        lines.append("- None from automated validation.")
    lines.append("- Training/validation/test/pilot are independent by construction and seeds are fixed deterministically.")
    lines.append("- This is a sensitivity analysis rather than proof that parameter count alone caused the m=252 behaviour.")
    lines.append("")
    return "\n".join(lines)


def run_capacity_data_sensitivity(
    config: CapacityDataConfig,
    *,
    gcv_benchmark_rows: list[dict[str, Any]] | None = None,
) -> Path:
    _validate_checkpoints(config.checkpoints)
    if len(config.cells) < 1:
        raise ValueError("experiment must run at least one configuration")

    seed_everything(config.base_seed)
    torch = __import__("torch")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = _ensure_unique_run_dir(Path(config.output_dir), f"ncv_capacity_data_{config.profile}_{ts}")
    run_dir = run_root / "ncv_capacity_data_m252"
    run_dir.mkdir(parents=True, exist_ok=False)

    config_payload = dataclasses.asdict(config)
    config_payload.update(
        {
            "contract": {"S0": 100.0, "K": 100.0, "r": 0.05, "sigma": 0.20, "T": 1.0},
            "risk_neutral_simulation": "existing_gbm_simulate_paths",
            "architecture": "one_hidden_layer_relu_linear_output",
            "optimizer": "adam",
            "objective": "mse",
            "seed_offsets": SEED_OFFSETS,
            "created_at_utc": ts,
        }
    )
    _write_json(run_dir / "capacity_data_config.json", config_payload)

    env = collect_environment_metadata()
    _write_json(run_dir / "capacity_data_environment.json", env)

    seed_manifest = _build_seed_manifest(config)
    _write_csv(run_dir / "capacity_data_seed_manifest.csv", seed_manifest)

    per_replication_rows: list[dict[str, Any]] = []
    gcv_rows: list[dict[str, Any]] = []
    gcv_rows_by_rep_split: dict[tuple[int, str], dict[str, Any]] = {}
    if gcv_benchmark_rows is not None:
        for row in gcv_benchmark_rows:
            rep = int(row["replication"])
            split = str(row["split"])
            gcv_rows_by_rep_split[(rep, split)] = dict(row)

    training_sizes = sorted(set(int(cell.train_paths) for cell in config.cells))
    all_nested_rows: list[dict[str, Any]] = []
    nested_ok_per_rep: list[bool] = []

    for rep in range(config.replications):
        rep_seeds = _split_seeds(config.base_seed, rep)

        t_data_validation = time.perf_counter()
        val_cfg = _make_reference_cfg(config.monitoring_dates, config.validation_paths, rep_seeds["validation"])
        val_rng = np.random.default_rng(rep_seeds["validation"])
        val_z = val_rng.standard_normal((config.validation_paths, config.monitoring_dates))
        val_split = _simulate_split_dataset(val_cfg, val_z)

        test_cfg = _make_reference_cfg(config.monitoring_dates, config.test_paths, rep_seeds["test"])
        test_rng = np.random.default_rng(rep_seeds["test"])
        test_z = test_rng.standard_normal((config.test_paths, config.monitoring_dates))
        test_split = _simulate_split_dataset(test_cfg, test_z)

        pilot_cfg = _make_reference_cfg(config.monitoring_dates, config.pilot_paths, rep_seeds["pilot"])
        pilot_rng = np.random.default_rng(rep_seeds["pilot"])
        pilot_z = pilot_rng.standard_normal((config.pilot_paths, config.monitoring_dates))
        pilot_split = _simulate_split_dataset(pilot_cfg, pilot_z)
        validation_generation_overhead_s = time.perf_counter() - t_data_validation

        train_splits, nested_rows, nested_diag = _generate_nested_training_splits(
            monitoring_dates=config.monitoring_dates,
            training_seed=rep_seeds["train"],
            training_sizes=training_sizes,
        )
        for row in nested_rows:
            row["replication"] = rep
        all_nested_rows.extend(nested_rows)
        nested_ok_per_rep.append(bool(nested_diag["training_nested_prefix_ok"]))

        if gcv_rows_by_rep_split:
            for split in ("validation", "test"):
                key = (rep, split)
                if key not in gcv_rows_by_rep_split:
                    raise RuntimeError(f"missing reused GCV row for replication={rep} split={split}")
                row = dict(gcv_rows_by_rep_split[key])
                row.update(
                    {
                        "replication": rep,
                        "split": split,
                        "train_seed": rep_seeds["train"],
                        "validation_seed": rep_seeds["validation"],
                        "test_seed": rep_seeds["test"],
                        "pilot_seed": rep_seeds["pilot"],
                        "validation_paths": config.validation_paths,
                        "test_paths": config.test_paths,
                        "pilot_paths": config.pilot_paths,
                    }
                )
                gcv_rows.append(row)
        else:
            gcv_for_rep = compute_gcv_benchmark(
                validation_split=val_split,
                test_split=test_split,
                pilot_split=pilot_split,
                n_reporting=config.pricing_observations_for_reporting,
            )
            for row in gcv_for_rep:
                gcv_rows.append(
                    {
                        "replication": rep,
                        "split": row["split"],
                        "train_seed": rep_seeds["train"],
                        "validation_seed": rep_seeds["validation"],
                        "test_seed": rep_seeds["test"],
                        "pilot_seed": rep_seeds["pilot"],
                        "validation_paths": config.validation_paths,
                        "test_paths": config.test_paths,
                        "pilot_paths": config.pilot_paths,
                        **row,
                    }
                )

        for cell in config.cells:
            print(f"[capacity-data] rep={rep+1}/{config.replications} config={cell.config_id}", flush=True)
            train_split = train_splits[int(cell.train_paths)]
            init_seed = _network_seed(rep_seeds, cell)

            snapshots = _train_continuous_snapshots(
                torch_mod=torch,
                train_split=train_split,
                validation_split=val_split,
                test_split=test_split,
                hidden_width=cell.hidden_width,
                learning_rate=config.learning_rate,
                batch_size=config.train_batch_size,
                checkpoints=config.checkpoints,
                n_reporting=config.pricing_observations_for_reporting,
                init_seed=init_seed,
            )

            count_model = _torch_model_from_initial_network(torch, snapshots[0].network)
            p_count = _parameter_count(count_model)
            p_formula = _expected_parameter_count(config.monitoring_dates, cell.hidden_width)
            if p_count != p_formula:
                raise RuntimeError(
                    f"parameter count mismatch for {cell.config_id}: model={p_count} formula={p_formula}"
                )
            parameters_per_training_path = p_count / float(cell.train_paths)
            paths_per_parameter = float(cell.train_paths) / p_count

            train_gen_runtime = float(nested_diag["generation_runtime_by_size_s"][str(cell.train_paths)])
            val_tuning_overhead = float(validation_generation_overhead_s)

            for cp in config.checkpoints:
                snap = snapshots[int(cp)]
                if int(cp) == 0 and abs(float(snap.optimizer_cumulative_runtime_s)) > 1e-12:
                    raise RuntimeError("checkpoint 0 optimizer runtime must be zero")
                setup_runtime = compute_ncv_setup_cost(
                    training_data_generation_runtime_s=train_gen_runtime,
                    optimizer_cumulative_training_runtime_s=snap.optimizer_cumulative_runtime_s,
                    checkpoint=int(cp),
                )
                for split_name, mse, diag in (
                    ("training", snap.train_mse, snap.training_diag),
                    ("validation", snap.validation_mse, snap.validation_diag),
                    ("test", snap.test_mse, snap.test_diag),
                ):
                    gap_ratio, gap_log = _safe_generalization_gap(snap.validation_mse, snap.train_mse)
                    per_replication_rows.append(
                        {
                            "config_id": cell.config_id,
                            "replication": rep,
                            "checkpoint": int(cp),
                            "split": split_name,
                            "train_seed": rep_seeds["train"],
                            "validation_seed": rep_seeds["validation"],
                            "test_seed": rep_seeds["test"],
                            "pilot_seed": rep_seeds["pilot"],
                            "network_init_seed": init_seed,
                            "train_paths": int(cell.train_paths),
                            "validation_paths": int(config.validation_paths),
                            "test_paths": int(config.test_paths),
                            "hidden_width": int(cell.hidden_width),
                            "parameter_count": int(p_count),
                            "parameter_count_formula": int(p_formula),
                            "parameter_to_training_path_ratio": float(parameters_per_training_path),
                            "parameters_per_training_path": float(parameters_per_training_path),
                            "paths_per_parameter": float(paths_per_parameter),
                            "mse": float(mse),
                            "payoff_mean": float(diag["arithmetic_payoff_mean"]),
                            "network_output_mean": float(diag["network_output_mean"]),
                            "analytic_eh": float(diag["analytical_eh"]),
                            "payoff_variance": float(diag["payoff_variance"]),
                            "network_output_variance": float(diag["network_output_variance"]),
                            "payoff_network_covariance": float(diag["payoff_network_covariance"]),
                            "payoff_network_correlation": float(diag["payoff_network_correlation"]),
                            "centered_residual_mean": float(diag["residual_mean"]),
                            "centered_residual_variance": float(diag["residual_variance"]),
                            "ncv_price_estimate": float(diag["ncv_price_estimate"]),
                            "estimator_variance_at_reporting_n": float(diag["estimator_variance_at_reporting_n"]),
                            "standard_error": float(diag["standard_error_at_reporting_n"]),
                            "ncv_vrr_vs_mc": float(diag["vrr_ncv_vs_mc"]),
                            "generalization_gap_ratio": float(gap_ratio),
                            "generalization_gap_log": float(gap_log),
                            "optimizer_cumulative_runtime_s": float(snap.optimizer_cumulative_runtime_s),
                            "training_data_generation_runtime_s": float(train_gen_runtime),
                            "validation_tuning_overhead_runtime_s": float(val_tuning_overhead),
                            "total_ncv_setup_runtime_s": float(train_gen_runtime + float(snap.optimizer_cumulative_runtime_s)),
                            "ncv_setup_cost_s": float(setup_runtime),
                            "setup_cost_excludes_validation_generation_and_evaluation_runtime": True,
                            "inference_runtime_per_observation_s": float(snap.inference["inference_runtime_per_observation_median_s"]),
                            "training_continuous_no_reinit_between_checkpoints": True,
                        }
                    )

    selected_rows, selected_by_config = _selected_checkpoints(per_replication_rows, config)

    comparisons = _default_paired_comparisons(config.cells)

    paired_rows = _build_paired_contrasts(per_replication_rows, selected_by_config, comparisons)
    checkpoint_summary = _build_checkpoint_summary(per_replication_rows, config)
    runtime_summary = _build_runtime_summary(per_replication_rows, selected_by_config, config)

    # Extension recommendation rule from validation only.
    extension_recommended = False
    for cell in config.cells:
        val_cp100 = [
            float(r["centered_residual_variance"])
            for r in per_replication_rows
            if r["config_id"] == cell.config_id and r["split"] == "validation" and int(r["checkpoint"]) == 100
            and math.isfinite(float(r["centered_residual_variance"]))
        ]
        val_cp200 = [
            float(r["centered_residual_variance"])
            for r in per_replication_rows
            if r["config_id"] == cell.config_id and r["split"] == "validation" and int(r["checkpoint"]) == 200
            and math.isfinite(float(r["centered_residual_variance"]))
        ]
        if not val_cp100 or not val_cp200:
            continue
        mean100 = float(statistics.mean(val_cp100))
        mean200 = float(statistics.mean(val_cp200))
        if selected_by_config[cell.config_id] == 200 and mean100 > 0.0 and (mean100 - mean200) / mean100 >= 0.01:
            extension_recommended = True

    _write_csv(run_dir / "capacity_data_per_replication.csv", per_replication_rows)
    _write_csv(run_dir / "capacity_data_checkpoint_summary.csv", checkpoint_summary)
    _write_csv(run_dir / "capacity_data_selected_checkpoints.csv", selected_rows)
    _write_csv(run_dir / "capacity_data_paired_contrasts.csv", paired_rows)
    _write_csv(run_dir / "capacity_data_runtime_summary.csv", runtime_summary)
    _write_csv(run_dir / "capacity_data_gcv_benchmark.csv", gcv_rows)

    _plot_summary(run_dir, per_replication_rows, selected_by_config, config)

    seed_unique = True
    for rep in range(config.replications):
        rep_rows = [r for r in seed_manifest if int(r["replication"]) == rep and "network_init::" not in str(r["seed_stream"])]
        values = [int(r["seed"]) for r in rep_rows]
        seed_unique = seed_unique and (len(values) == len(set(values)))

    diagnostics = {
        "profile": config.profile,
        "replications": config.replications,
        "n_configurations": len(config.cells),
        "checkpoint_grid": list(config.checkpoints),
        "checkpoint_grid_starts_at_zero": config.checkpoints[0] == 0,
        "checkpoint_grid_strictly_increasing": all(
            config.checkpoints[i] < config.checkpoints[i + 1] for i in range(len(config.checkpoints) - 1)
        ),
        "training_nested_prefix_all_ok": all(nested_ok_per_rep),
        "seed_streams_unique_within_replication": seed_unique,
        "seed_generation_uses_builtin_hash": False,
        "extension_recommended": bool(extension_recommended),
        "row_count_per_replication": len(per_replication_rows) // max(1, config.replications),
        "expected_per_replication_rows": len(config.cells) * len(config.checkpoints) * 3,
        "row_count_matches_expectation": len(per_replication_rows)
        == config.replications * len(config.cells) * len(config.checkpoints) * 3,
    }

    preliminary = _validate_outputs(run_dir, diagnostics)
    _write_json(run_dir / "capacity_data_validation_report.json", preliminary)

    handover = _build_handover(
        config=config,
        selected_rows=selected_rows,
        paired_rows=paired_rows,
        extension_recommended=extension_recommended,
        validation_report=preliminary,
    )
    (run_dir / "CAPACITY_DATA_HANDOVER.md").write_text(handover, encoding="utf-8")

    final_report = _validate_outputs(run_dir, diagnostics)
    final_report["warnings"] = sorted(set(final_report["warnings"]).union(preliminary.get("warnings", [])))
    _write_json(run_dir / "capacity_data_validation_report.json", final_report)

    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run NCV capacity-vs-data m=252 sensitivity experiment"
    )
    p.add_argument("--profile", choices=["smoke", "dissertation"], default="smoke")
    p.add_argument(
        "--dissertation-cell-set",
        choices=["baseline", "missing_cells", "full_grid"],
        default="baseline",
    )
    p.add_argument("--output-dir", default="experiment_runs")
    p.add_argument("--base-seed", type=int, default=42)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    cfg = profile_config(
        args.profile,
        output_dir=args.output_dir,
        base_seed=args.base_seed,
        dissertation_cell_set=args.dissertation_cell_set,
    )
    out = run_capacity_data_sensitivity(cfg)
    print(f"NCV capacity-data sensitivity output: {out.resolve()}")


if __name__ == "__main__":
    main()
