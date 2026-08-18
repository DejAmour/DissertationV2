from __future__ import annotations

import argparse
import csv
import dataclasses
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
    _timed_mc_av_gcv,
    _timed_ncv_end_to_end,
    _torch_model_from_initial_network,
    compute_ncv_setup_cost,
    compute_ncv_split_diagnostics,
)
from asian_options.neural_cv import analytical_network_expectation, build_network
from asian_options.payoffs import arithmetic_asian_call_payoff, geometric_asian_call_payoff
from asian_options.simulate_gbm import simulate_paths


CHECKPOINT_GRID = (0, 10, 25, 50, 100, 200, 500, 1000)
FORMAL_EPOCHS = (25, 1000)
MONITORING_DATES_GRID = (12, 252)
SEED_OFFSETS = {
    "train": 1_000,
    "validation": 2_000,
    "pilot": 3_000,
    "pricing": 4_000,
    "runtime": 5_000,
}


@dataclass(frozen=True)
class SensitivityConfig:
    profile: str
    base_seed: int
    replications: int
    train_paths: int
    validation_paths: int
    pilot_paths: int
    pricing_paths: int
    hidden_width: int
    learning_rate: float
    batch_size: int
    checkpoints: tuple[int, ...]
    output_dir: str
    direct_timing_max_paths: int
    direct_timing_repeats: int


@dataclass(frozen=True)
class _CheckpointSnapshot:
    epoch: int
    network: Any
    e_h: float
    train_mse: float
    validation_mse: float
    optimizer_cumulative_runtime_s: float


@dataclass(frozen=True)
class _CellRuntime:
    mc_pricing_runtime_s: float
    gcv_setup_runtime_s: float
    gcv_pricing_runtime_s: float
    ncv_setup_runtime_s: float
    ncv_marginal_runtime_50000_s: float
    ncv_marginal_runtime_matched_measured_s: float | None
    ncv_marginal_runtime_matched_projected_s: float | None


def profile_config(profile: str, output_dir: str, base_seed: int = 42) -> SensitivityConfig:
    p = profile.lower()
    if p == "smoke":
        return SensitivityConfig(
            profile="smoke",
            base_seed=base_seed,
            replications=2,
            train_paths=64,
            validation_paths=128,
            pilot_paths=64,
            pricing_paths=512,
            hidden_width=32,
            learning_rate=1e-2,
            batch_size=64,
            checkpoints=(0, 10, 25, 1000),
            output_dir=output_dir,
            direct_timing_max_paths=2_000,
            direct_timing_repeats=1,
        )
    if p == "dissertation":
        return SensitivityConfig(
            profile="dissertation",
            base_seed=base_seed,
            replications=30,
            train_paths=5_000,
            validation_paths=10_000,
            pilot_paths=1_000,
            pricing_paths=50_000,
            hidden_width=32,
            learning_rate=1e-2,
            batch_size=256,
            checkpoints=CHECKPOINT_GRID,
            output_dir=output_dir,
            direct_timing_max_paths=200_000,
            direct_timing_repeats=1,
        )
    raise ValueError(f"unknown profile: {profile}")


def formal_design_cells() -> list[dict[str, int]]:
    return [
        {"monitoring_dates": m, "ncv_epoch": epoch}
        for m in MONITORING_DATES_GRID
        for epoch in FORMAL_EPOCHS
    ]


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
    raise RuntimeError(f"unable to create unique output directory under {base_dir}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    cols: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                cols.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _sample_var(x: np.ndarray) -> float:
    if x.size <= 1:
        return float("nan")
    return float(np.var(x, ddof=1))


def _student_t_ci(values: list[float], alpha: float = 0.05) -> dict[str, Any]:
    if len(values) == 0:
        return {
            "n": 0,
            "mean": "NA",
            "ci95_lower": "NA",
            "ci95_upper": "NA",
            "status": "no_observations",
        }
    m = float(statistics.mean(values))
    if len(values) == 1:
        return {
            "n": 1,
            "mean": m,
            "ci95_lower": m,
            "ci95_upper": m,
            "status": "single_observation",
        }
    s = float(statistics.stdev(values))
    dof = len(values) - 1
    t_crit = float(_t_dist.ppf(1.0 - alpha / 2.0, dof))
    half = t_crit * s / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": m,
        "ci95_lower": m - half,
        "ci95_upper": m + half,
        "status": "ok",
        "dof": dof,
        "t_critical": t_crit,
    }


def log_ratio_summary(values: list[float]) -> dict[str, Any]:
    positive = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0.0]
    if not positive:
        return {
            "count": 0,
            "geometric_mean": "NA",
            "geometric_ci95_lower": "NA",
            "geometric_ci95_upper": "NA",
            "median": "NA",
            "minimum": "NA",
            "maximum": "NA",
        }
    logs = [math.log(v) for v in positive]
    ci = _student_t_ci(logs)
    lo = ci["ci95_lower"]
    hi = ci["ci95_upper"]
    return {
        "count": len(positive),
        "geometric_mean": math.exp(ci["mean"]),
        "geometric_ci95_lower": math.exp(lo) if isinstance(lo, (int, float)) else "NA",
        "geometric_ci95_upper": math.exp(hi) if isinstance(hi, (int, float)) else "NA",
        "median": float(statistics.median(positive)),
        "minimum": float(min(positive)),
        "maximum": float(max(positive)),
    }


def _paired_log_contrast(rows: list[dict[str, Any]], monitoring_dates: int) -> list[float]:
    by_rep: dict[int, dict[int, float]] = {}
    for row in rows:
        if int(row["monitoring_dates"]) != int(monitoring_dates):
            continue
        epoch = int(row["ncv_epoch"])
        advantage = row.get("ncv_to_gcv_advantage")
        if not isinstance(advantage, (int, float)) or not math.isfinite(float(advantage)) or float(advantage) <= 0.0:
            continue
        rep = int(row["replication"])
        by_rep.setdefault(rep, {})[epoch] = float(advantage)
    deltas: list[float] = []
    for rep, mapping in by_rep.items():
        if 25 in mapping and 1000 in mapping:
            deltas.append(math.log(mapping[1000]) - math.log(mapping[25]))
    return deltas


def compute_paired_contrasts(replication_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    d12 = _paired_log_contrast(replication_rows, 12)
    d252 = _paired_log_contrast(replication_rows, 252)
    n = min(len(d12), len(d252))
    interaction_vals = [d12[i] - d252[i] for i in range(n)]

    rows: list[dict[str, Any]] = []
    for label, values in (
        ("delta_12", d12),
        ("delta_252", d252),
        ("delta_interaction", interaction_vals),
    ):
        ci = _student_t_ci(values)
        mean = ci["mean"] if isinstance(ci["mean"], (int, float)) else float("nan")
        if len(values) >= 2 and isinstance(mean, (int, float)) and math.isfinite(mean):
            sd = statistics.stdev(values)
            se = sd / math.sqrt(len(values))
            t_stat = mean / se if se > 0 else float("nan")
            p_value = 2.0 * (1.0 - float(_t_dist.cdf(abs(t_stat), df=len(values) - 1))) if math.isfinite(t_stat) else float("nan")
        else:
            t_stat = float("nan")
            p_value = float("nan")
        rows.append(
            {
                "contrast": label,
                "n_pairs": len(values),
                "estimate_log_scale": ci["mean"],
                "ci95_lower_log_scale": ci["ci95_lower"],
                "ci95_upper_log_scale": ci["ci95_upper"],
                "estimate_ratio_scale": math.exp(ci["mean"]) if isinstance(ci["mean"], (int, float)) else "NA",
                "ci95_lower_ratio_scale": math.exp(ci["ci95_lower"]) if isinstance(ci["ci95_lower"], (int, float)) else "NA",
                "ci95_upper_ratio_scale": math.exp(ci["ci95_upper"]) if isinstance(ci["ci95_upper"], (int, float)) else "NA",
                "paired_t_statistic": t_stat if math.isfinite(t_stat) else "NA",
                "paired_t_p_value": p_value if math.isfinite(p_value) else "NA",
            }
        )
    return rows


def required_ncv_observations_to_match_gcv(obs_var_ncv: float, obs_var_gcv: float, gcv_observations: int = 50_000) -> tuple[int | None, str]:
    if not math.isfinite(obs_var_ncv) or not math.isfinite(obs_var_gcv):
        return None, "non_finite_variance"
    if obs_var_ncv <= 0.0 or obs_var_gcv <= 0.0:
        return None, "non_positive_variance"
    target_estimator_var = obs_var_gcv / float(gcv_observations)
    if target_estimator_var <= 0.0:
        return None, "invalid_target_estimator_variance"
    n_required = max(2, int(math.ceil(obs_var_ncv / target_estimator_var)))
    return n_required, ""


def solve_break_even_q(
    *,
    baseline_setup_cost: float,
    baseline_marginal_cost: float,
    proposed_setup_cost: float,
    proposed_marginal_cost: float,
) -> dict[str, Any]:
    tol = 1e-9
    vals = [baseline_setup_cost, baseline_marginal_cost, proposed_setup_cost, proposed_marginal_cost]
    if any((not isinstance(v, (int, float))) or (not math.isfinite(float(v))) for v in vals):
        return {"break_even_q": "NA", "failure_reason": "missing_or_non_finite_runtime_input", "verified_q": False, "verified_q_minus_1": False}
    b0 = float(baseline_setup_cost)
    bm = float(baseline_marginal_cost)
    p0 = float(proposed_setup_cost)
    pm = float(proposed_marginal_cost)
    if pm >= bm - tol:
        return {
            "break_even_q": "NA",
            "failure_reason": "proposed_marginal_not_below_baseline_no_finite_break_even",
            "verified_q": False,
            "verified_q_minus_1": False,
        }
    raw_q = (p0 - b0) / (bm - pm)
    q = max(1, int(math.ceil(raw_q - 1e-12 * max(1.0, abs(raw_q)))))
    def c_b(k: int) -> float:
        return b0 + k * bm
    def c_p(k: int) -> float:
        return p0 + k * pm
    verified_q = c_p(q) <= c_b(q) + tol
    if q == 1:
        verified_q_minus_1 = True
        status = "not_applicable_minimum_q_boundary"
    else:
        verified_q_minus_1 = c_p(q - 1) > c_b(q - 1) + tol
        status = "verified_against_q_minus_1" if verified_q_minus_1 else "failed_at_q_minus_1"
    if not (verified_q and verified_q_minus_1):
        return {
            "break_even_q": "NA",
            "failure_reason": "verification_failed",
            "verified_q": verified_q,
            "verified_q_minus_1": verified_q_minus_1,
            "q_minus_1_verification_status": status,
        }
    return {
        "break_even_q": q,
        "failure_reason": "",
        "verified_q": True,
        "verified_q_minus_1": True,
        "q_minus_1_verification_status": status,
        "cost_baseline_q": c_b(q),
        "cost_proposed_q": c_p(q),
        "cost_baseline_q_minus_1": "NA" if q == 1 else c_b(q - 1),
        "cost_proposed_q_minus_1": "NA" if q == 1 else c_p(q - 1),
    }


def _split_seeds(base_seed: int, replication: int) -> dict[str, int]:
    rep_offset = int(replication) * 100_000
    s = int(base_seed) + rep_offset
    return {k: s + v for k, v in SEED_OFFSETS.items()}


def _build_seed_manifest(config: SensitivityConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rep in range(config.replications):
        seeds = _split_seeds(config.base_seed, rep)
        for m in MONITORING_DATES_GRID:
            for stream, value in seeds.items():
                rows.append(
                    {
                        "replication": rep,
                        "monitoring_dates": m,
                        "stream": stream,
                        "seed": int(value),
                        "shared_across_monitoring_profiles": True,
                    }
                )
    return rows


def _validate_seed_independence(seed_manifest: list[dict[str, Any]]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    by_key: dict[tuple[int, int], list[int]] = {}
    for row in seed_manifest:
        key = (int(row["replication"]), int(row["monitoring_dates"]))
        by_key.setdefault(key, []).append(int(row["seed"]))
    for key, values in by_key.items():
        if len(values) != len(set(values)):
            failures.append(f"duplicate stream seeds within replication/monitoring pair: {key}")
    return len(failures) == 0, failures


def _make_reference_cfg(monitoring_dates: int, n_paths: int, seed: int) -> ModelConfig:
    base_cfg = make_contract_cfg(REFERENCE_ID, n_paths=n_paths, seed=seed)
    return dataclasses.replace(base_cfg, m=monitoring_dates, n_paths=n_paths, seed=seed)


def _simulate_split_dataset_shared(monitoring_dates: int, n_paths: int, seed: int) -> dict[str, Any]:
    max_m = max(MONITORING_DATES_GRID)
    cfg = _make_reference_cfg(monitoring_dates=monitoring_dates, n_paths=n_paths, seed=seed)
    rng = np.random.default_rng(seed)
    z_full = rng.standard_normal((n_paths, max_m))
    z = z_full[:, :monitoring_dates].copy()
    paths = simulate_paths(cfg, shocks=z)
    payoff_arith = arithmetic_asian_call_payoff(paths, cfg)
    payoff_geom = geometric_asian_call_payoff(paths, cfg)
    return {
        "cfg": cfg,
        "Z": z,
        "payoff_arithmetic": payoff_arith,
        "payoff_geometric": payoff_geom,
    }


def _fit_gcv_pilot_from_split(pilot_split: dict[str, Any]) -> dict[str, float]:
    t0 = time.perf_counter()
    x = pilot_split["payoff_arithmetic"]
    g = pilot_split["payoff_geometric"]
    var_g = _sample_var(g)
    cov_xg = float(np.cov(x, g, ddof=1)[0, 1]) if x.size > 1 else float("nan")
    beta = cov_xg / var_g if math.isfinite(var_g) and var_g > 0 else float("nan")
    pilot_runtime = time.perf_counter() - t0
    from asian_options.analytical import geometric_asian_call_price

    return {
        "beta": beta,
        "eg": float(geometric_asian_call_price(pilot_split["cfg"])),
        "pilot_runtime_s": pilot_runtime,
    }


def _evaluate_gcv_on_split(split: dict[str, Any], gcv_fit: dict[str, float], n_reporting: int) -> dict[str, float]:
    t0 = time.perf_counter()
    x = split["payoff_arithmetic"]
    g = split["payoff_geometric"]
    beta = float(gcv_fit["beta"])
    eg = float(gcv_fit["eg"])
    corrected = x - beta * (g - eg) if math.isfinite(beta) else x
    pricing_runtime = time.perf_counter() - t0
    obs_var = _sample_var(corrected)
    estimator_var = obs_var / n_reporting if math.isfinite(obs_var) else float("nan")
    se = math.sqrt(estimator_var) if math.isfinite(estimator_var) and estimator_var >= 0.0 else float("nan")
    return {
        "price": float(np.mean(corrected)),
        "observation_variance": obs_var,
        "estimator_variance": estimator_var,
        "standard_error": se,
        "pricing_runtime_control_only_s": pricing_runtime,
    }


def _train_snapshots(
    *,
    torch_mod,
    train_split: dict[str, Any],
    validation_split: dict[str, Any],
    checkpoints: tuple[int, ...],
    hidden_width: int,
    learning_rate: float,
    batch_size: int,
) -> tuple[dict[int, _CheckpointSnapshot], float]:
    cfg = train_split["cfg"]
    t_data = time.perf_counter()
    x_train_np = train_split["Z"]
    y_train_np = train_split["payoff_arithmetic"]
    x_val_np = validation_split["Z"]
    y_val_np = validation_split["payoff_arithmetic"]
    data_generation_runtime_s = time.perf_counter() - t_data

    network = build_network(cfg, hidden_width=hidden_width)
    model = _torch_model_from_initial_network(torch_mod, network)
    optimizer = torch_mod.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch_mod.nn.MSELoss()

    x_train = torch_mod.tensor(x_train_np, dtype=torch_mod.float64)
    y_train = torch_mod.tensor(y_train_np, dtype=torch_mod.float64)
    x_val = torch_mod.tensor(x_val_np, dtype=torch_mod.float64)
    y_val = torch_mod.tensor(y_val_np, dtype=torch_mod.float64)

    batch = min(batch_size, len(x_train))
    snapshots: dict[int, _CheckpointSnapshot] = {}

    def capture(epoch: int, opt_runtime: float) -> None:
        model.eval()
        with torch_mod.no_grad():
            train_mse = float(loss_fn(model(x_train), y_train).item())
            validation_mse = float(loss_fn(model(x_val), y_val).item())
        net = _network_from_torch_model(model)
        snapshots[int(epoch)] = _CheckpointSnapshot(
            epoch=int(epoch),
            network=net,
            e_h=float(analytical_network_expectation(net)),
            train_mse=train_mse,
            validation_mse=validation_mse,
            optimizer_cumulative_runtime_s=float(opt_runtime),
        )

    capture(0, 0.0)
    max_cp = int(max(checkpoints))
    cp_set = set(int(x) for x in checkpoints)
    t_train = time.perf_counter()
    for epoch in range(1, max_cp + 1):
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
    return snapshots, float(data_generation_runtime_s)


def _direct_ncv_runtime_at_n(
    *,
    torch_mod,
    snapshot: _CheckpointSnapshot,
    monitoring_dates: int,
    pricing_seed: int,
    n_paths: int,
    repeats: int,
) -> float:
    cfg = _make_reference_cfg(monitoring_dates=monitoring_dates, n_paths=n_paths, seed=pricing_seed)
    timed = _timed_ncv_end_to_end(snapshot.network, cfg, torch_mod, repeats=max(1, repeats))
    return float(timed["end_to_end_pricing_runtime_s"])


def _build_cell_summary(replication_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in replication_rows:
        key = (int(row["monitoring_dates"]), int(row["ncv_epoch"]))
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for (m, epoch), rows in sorted(grouped.items()):
        advantages = [float(r["ncv_to_gcv_advantage"]) for r in rows if isinstance(r.get("ncv_to_gcv_advantage"), (int, float))]
        ncv_vrr = [float(r["ncv_vrr_vs_mc"]) for r in rows if isinstance(r.get("ncv_vrr_vs_mc"), (int, float))]
        gcv_vrr = [float(r["gcv_vrr_vs_mc"]) for r in rows if isinstance(r.get("gcv_vrr_vs_mc"), (int, float))]

        summary_adv = log_ratio_summary(advantages)
        summary_ncv_vrr = log_ratio_summary(ncv_vrr)
        summary_gcv_vrr = log_ratio_summary(gcv_vrr)

        out.append(
            {
                "monitoring_dates": m,
                "ncv_epoch": epoch,
                "replications": len(rows),
                "ncv_beats_gcv_count": sum(1 for r in rows if bool(r.get("ncv_beats_gcv"))),
                "advantage_geometric_mean": summary_adv["geometric_mean"],
                "advantage_ci95_lower": summary_adv["geometric_ci95_lower"],
                "advantage_ci95_upper": summary_adv["geometric_ci95_upper"],
                "advantage_median": summary_adv["median"],
                "advantage_min": summary_adv["minimum"],
                "advantage_max": summary_adv["maximum"],
                "ncv_vrr_geometric_mean": summary_ncv_vrr["geometric_mean"],
                "ncv_vrr_ci95_lower": summary_ncv_vrr["geometric_ci95_lower"],
                "ncv_vrr_ci95_upper": summary_ncv_vrr["geometric_ci95_upper"],
                "gcv_vrr_geometric_mean": summary_gcv_vrr["geometric_mean"],
                "gcv_vrr_ci95_lower": summary_gcv_vrr["geometric_ci95_lower"],
                "gcv_vrr_ci95_upper": summary_gcv_vrr["geometric_ci95_upper"],
            }
        )
    return out


def _plot_checkpoint_curves(run_dir: Path, checkpoint_curve_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    def _curve_for_m(m: int) -> tuple[list[int], list[float], list[float], list[float]]:
        by_cp: dict[int, list[float]] = {}
        for row in checkpoint_curve_rows:
            if int(row["monitoring_dates"]) != int(m):
                continue
            cp = int(row["checkpoint"])
            v = row.get("advantage")
            if isinstance(v, (int, float)) and math.isfinite(float(v)) and float(v) > 0.0:
                by_cp.setdefault(cp, []).append(float(v))
        cps = sorted(by_cp.keys())
        means: list[float] = []
        lows: list[float] = []
        highs: list[float] = []
        for cp in cps:
            logs = [math.log(v) for v in by_cp[cp]]
            ci = _student_t_ci(logs)
            means.append(math.exp(ci["mean"]))
            lows.append(math.exp(ci["ci95_lower"]))
            highs.append(math.exp(ci["ci95_upper"]))
        return cps, means, lows, highs

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, m in zip(axes, (252, 12), strict=True):
        cps, means, lows, highs = _curve_for_m(m)
        ax.plot(cps, means, color="#1f4e79", linewidth=1.8)
        ax.fill_between(cps, lows, highs, color="#1f4e79", alpha=0.2)
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
        ax.axvline(25, color="#7f7f7f", linestyle=":", linewidth=1.0)
        ax.axvline(1000, color="#7f7f7f", linestyle=":", linewidth=1.0)
        ax.set_title(f"m={m}")
        ax.set_xlabel("Epoch")
        ax.set_yscale("log")
    axes[0].set_ylabel("Held-out NCV/GCV variance advantage")
    fig.tight_layout()
    fig.savefig(run_dir / "figure_checkpoint_curves.png", dpi=300)
    fig.savefig(run_dir / "figure_checkpoint_curves.pdf")
    plt.close(fig)


def _plot_interaction(run_dir: Path, replication_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for m, color in ((12, "#1f4e79"), (252, "#8b1a1a")):
        xs: list[int] = []
        ys: list[float] = []
        lo: list[float] = []
        hi: list[float] = []
        for epoch in FORMAL_EPOCHS:
            vals = [
                float(r["ncv_to_gcv_advantage"])
                for r in replication_rows
                if int(r["monitoring_dates"]) == m
                and int(r["ncv_epoch"]) == epoch
                and isinstance(r.get("ncv_to_gcv_advantage"), (int, float))
                and float(r["ncv_to_gcv_advantage"]) > 0.0
            ]
            logs = [math.log(v) for v in vals]
            ci = _student_t_ci(logs)
            xs.append(epoch)
            ys.append(math.exp(ci["mean"]))
            lo.append(math.exp(ci["ci95_lower"]))
            hi.append(math.exp(ci["ci95_upper"]))
        ax.plot(xs, ys, marker="o", color=color, label=f"m={m}")
        ax.fill_between(xs, lo, hi, color=color, alpha=0.2)

    ax.axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Held-out NCV/GCV variance advantage")
    ax.set_xticks([25, 1000])
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(run_dir / "figure_2x2_interaction.png", dpi=300)
    fig.savefig(run_dir / "figure_2x2_interaction.pdf")
    plt.close(fig)


def _build_dissertation_summary(
    *,
    config: SensitivityConfig,
    cell_summary: list[dict[str, Any]],
    contrasts: list[dict[str, Any]],
    validation_report: dict[str, Any],
) -> str:
    cell_map = {(int(r["monitoring_dates"]), int(r["ncv_epoch"])): r for r in cell_summary}

    def _fmt(v: Any) -> str:
        if isinstance(v, float):
            if math.isfinite(v):
                return f"{v:.6g}"
            return "NA"
        return str(v)

    delta_lookup = {r["contrast"]: r for r in contrasts}
    d12 = delta_lookup.get("delta_12", {})
    d252 = delta_lookup.get("delta_252", {})
    dint = delta_lookup.get("delta_interaction", {})

    def _answer_rescue() -> str:
        key_252_25 = cell_map.get((252, 25), {})
        key_252_1000 = cell_map.get((252, 1000), {})
        gm25 = key_252_25.get("advantage_geometric_mean")
        gm1000 = key_252_1000.get("advantage_geometric_mean")
        if isinstance(gm25, (int, float)) and isinstance(gm1000, (int, float)):
            if gm1000 > 1.0:
                return "At m=252, the 1,000-epoch checkpoint reaches A>1 on average, so longer training rescues NCV versus GCV in this sensitivity design."
            return "At m=252, the 1,000-epoch checkpoint remains at or below parity (A≤1 on average), so longer training does not rescue NCV versus GCV in this sensitivity design."
        return "The m=252 rescue conclusion is indeterminate from this run because geometric-mean advantage values were not finite."

    def _answer_reversal() -> str:
        key12_25 = cell_map.get((12, 25), {})
        key12_1000 = cell_map.get((12, 1000), {})
        key252_25 = cell_map.get((252, 25), {})
        key252_1000 = cell_map.get((252, 1000), {})
        vals = [key12_25.get("advantage_geometric_mean"), key12_1000.get("advantage_geometric_mean"), key252_25.get("advantage_geometric_mean"), key252_1000.get("advantage_geometric_mean")]
        if not all(isinstance(v, (int, float)) for v in vals):
            return "The cross-profile reversal assessment is indeterminate because one or more cell geometric means are not finite."
        better_12 = max(float(key12_25["advantage_geometric_mean"]), float(key12_1000["advantage_geometric_mean"]))
        better_252 = max(float(key252_25["advantage_geometric_mean"]), float(key252_1000["advantage_geometric_mean"]))
        if better_12 > better_252:
            return "After crossing both profiles with both checkpoints, the 12-date profile still shows the stronger NCV/GCV advantage."
        if better_252 > better_12:
            return "After crossing both profiles with both checkpoints, the 252-date profile shows the stronger NCV/GCV advantage."
        return "After crossing both profiles with both checkpoints, the two monitoring profiles are effectively tied on the NCV/GCV advantage summary used here."

    lines: list[str] = []
    lines.append("# 2×2 Monitoring-Profile × Training-Horizon Sensitivity Study")
    lines.append("")
    lines.append("## 1) Design")
    lines.append(
        "This run fixes a 2×2 design across monitoring profiles (m=12, m=252) and NCV checkpoints (25, 1,000), "
        "with all other model, optimiser, seed, and sampling choices held constant. Checkpoints were fixed ex ante and not selected using final pricing data."
    )
    lines.append("")
    lines.append("## 2) Four formal cells")
    lines.append("")
    lines.append("| m | checkpoint | geometric mean A=Var(GCV)/Var(NCV) | 95% CI | NCV beats GCV count |")
    lines.append("|---:|---:|---:|---:|---:|")
    for m, epoch in ((12, 25), (12, 1000), (252, 25), (252, 1000)):
        row = cell_map.get((m, epoch), {})
        lines.append(
            f"| {m} | {epoch} | {_fmt(row.get('advantage_geometric_mean'))} | "
            f"[{_fmt(row.get('advantage_ci95_lower'))}, {_fmt(row.get('advantage_ci95_upper'))}] | "
            f"{_fmt(row.get('ncv_beats_gcv_count'))}/{config.replications} |"
        )
    lines.append("")
    lines.append("## 3) Paired contrasts (log-scale, then exponentiated)")
    for label, row in (("Δ12", d12), ("Δ252", d252), ("Δinteraction", dint)):
        lines.append(
            f"- {label}: estimate={_fmt(row.get('estimate_log_scale'))}, "
            f"95% CI=[{_fmt(row.get('ci95_lower_log_scale'))}, {_fmt(row.get('ci95_upper_log_scale'))}], "
            f"ratio-scale={_fmt(row.get('estimate_ratio_scale'))}"
        )
    lines.append("")
    lines.append("## 4) Does 1,000 epochs rescue m=252 NCV?")
    lines.append(_answer_rescue())
    lines.append("")
    lines.append("## 5) Does the monitoring-profile reversal remain under the full 2×2 cross?")
    lines.append(_answer_reversal())
    lines.append("")
    lines.append("## 6) Limitation")
    lines.append(
        "The monitoring profile m jointly changes payoff discretisation, network input dimension, and parameter count; "
        "this sensitivity analysis quantifies pattern robustness but is not a causal identification of any single mechanism."
    )
    lines.append("")
    lines.append("## 7) Dissertation-ready epoch-selection replacement text")
    lines.append(
        "Epoch choice should be treated as a fixed design input selected off-line from held-out training/validation evidence. "
        "In the crossed 2×2 sensitivity analysis, longer NCV training can materially change relative NCV/GCV variance performance, "
        "so conclusions should be reported conditionally on both monitoring profile and checkpoint horizon rather than extrapolated from a single profile-horizon pair."
    )
    lines.append("")
    lines.append("## 8) Suggested figure captions")
    lines.append("- Figure 1 (checkpoint curves): Held-out NCV/GCV variance-advantage trajectories across epochs for m=252 and m=12, with geometric means and 95% log-scale CIs.")
    lines.append("- Figure 2 (2×2 interaction): Crossed monitoring-profile × checkpoint comparison of held-out NCV/GCV variance advantage with 95% CIs on a log scale.")
    lines.append("")
    lines.append("## 9) Integrity note")
    lines.append("All numeric claims in this summary are sourced from this run’s generated CSV outputs; no unsupported values were introduced.")
    if validation_report.get("warnings"):
        lines.append("")
        lines.append("## Validation warnings")
        for warning in validation_report["warnings"]:
            lines.append(f"- {warning}")
    return "\n".join(lines) + "\n"


def _cell_runtime(
    *,
    torch_mod,
    monitoring_dates: int,
    pricing_seed: int,
    pricing_paths: int,
    ncv_snapshot: _CheckpointSnapshot,
    gcv_fit: dict[str, float],
    n_required_ncv: int | None,
    direct_timing_max_paths: int,
    direct_timing_repeats: int,
    data_generation_runtime_s: float,
) -> _CellRuntime:
    cfg_pricing = _make_reference_cfg(monitoring_dates, pricing_paths, pricing_seed)

    mc_timed = _timed_mc_av_gcv("MC", cfg_pricing, repeats=1)
    gcv_timed = _timed_mc_av_gcv("GCV", cfg_pricing, n_pilot=0, repeats=1, gcv_pilot_fit=gcv_fit)
    ncv_timed_50000 = _timed_ncv_end_to_end(ncv_snapshot.network, cfg_pricing, torch_mod, repeats=1)

    projected = None
    measured = None
    if n_required_ncv is not None and n_required_ncv >= 2:
        projected = float(ncv_timed_50000["end_to_end_pricing_runtime_s"]) * (float(n_required_ncv) / float(pricing_paths))
        if n_required_ncv <= int(direct_timing_max_paths):
            measured = _direct_ncv_runtime_at_n(
                torch_mod=torch_mod,
                snapshot=ncv_snapshot,
                monitoring_dates=monitoring_dates,
                pricing_seed=pricing_seed + 777,
                n_paths=int(n_required_ncv),
                repeats=direct_timing_repeats,
            )

    ncv_setup = compute_ncv_setup_cost(
        training_data_generation_runtime_s=float(data_generation_runtime_s),
        optimizer_cumulative_training_runtime_s=float(ncv_snapshot.optimizer_cumulative_runtime_s),
        checkpoint=int(ncv_snapshot.epoch),
    )

    return _CellRuntime(
        mc_pricing_runtime_s=float(mc_timed["end_to_end_pricing_runtime_s"]),
        gcv_setup_runtime_s=float(gcv_fit["pilot_runtime_s"]),
        gcv_pricing_runtime_s=float(gcv_timed["end_to_end_pricing_runtime_s"]),
        ncv_setup_runtime_s=float(ncv_setup),
        ncv_marginal_runtime_50000_s=float(ncv_timed_50000["end_to_end_pricing_runtime_s"]),
        ncv_marginal_runtime_matched_measured_s=measured,
        ncv_marginal_runtime_matched_projected_s=projected,
    )


def run_sensitivity_study(config: SensitivityConfig) -> Path:
    if set(FORMAL_EPOCHS).difference(config.checkpoints):
        raise ValueError("checkpoints must include 25 and 1000")
    seed_everything(config.base_seed)
    torch = __import__("torch")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = _ensure_unique_run_dir(Path(config.output_dir), f"stage8_sensitivity_2x2_{config.profile}_{ts}")

    config_payload = dataclasses.asdict(config)
    config_payload.update(
        {
            "contract": {"S0": 100.0, "K": 100.0, "r": 0.05, "sigma": 0.20, "T": 1.0},
            "formal_design_cells": formal_design_cells(),
            "fixed_architecture": "one_hidden_layer_relu_linear_output",
            "initialisation": "xavier_uniform_zero_biases",
            "optimizer": "adam",
            "seed_offsets": SEED_OFFSETS,
            "created_at_utc": ts,
        }
    )
    _write_json(run_dir / "config.json", config_payload)

    env = collect_environment_metadata()
    _write_json(run_dir / "environment.json", env)

    seed_manifest = _build_seed_manifest(config)
    _write_csv(run_dir / "seed_manifest.csv", seed_manifest)

    checkpoint_curve_rows: list[dict[str, Any]] = []
    replication_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for rep in range(config.replications):
        rep_seeds = _split_seeds(config.base_seed, rep)
        for m in MONITORING_DATES_GRID:
            print(f"[study] rep={rep+1}/{config.replications} m={m}", flush=True)
            train_split = _simulate_split_dataset_shared(m, config.train_paths, rep_seeds["train"])
            val_split = _simulate_split_dataset_shared(m, config.validation_paths, rep_seeds["validation"])
            pilot_split = _simulate_split_dataset_shared(m, config.pilot_paths, rep_seeds["pilot"])
            pricing_split = _simulate_split_dataset_shared(m, config.pricing_paths, rep_seeds["pricing"])

            snapshots, data_generation_runtime_s = _train_snapshots(
                torch_mod=torch,
                train_split=train_split,
                validation_split=val_split,
                checkpoints=config.checkpoints,
                hidden_width=config.hidden_width,
                learning_rate=config.learning_rate,
                batch_size=config.batch_size,
            )

            gcv_fit = _fit_gcv_pilot_from_split(pilot_split)
            gcv_val = _evaluate_gcv_on_split(val_split, gcv_fit, config.pricing_paths)
            gcv_price = _evaluate_gcv_on_split(pricing_split, gcv_fit, config.pricing_paths)

            trajectory_id = f"rep{rep}_m{m}"
            for cp in config.checkpoints:
                snap = snapshots[int(cp)]
                val_diag = compute_ncv_split_diagnostics(
                    payoff=val_split["payoff_arithmetic"],
                    h_vals=snap.network.forward(val_split["Z"]),
                    e_h=snap.e_h,
                    n_reporting=config.pricing_paths,
                )
                price_diag = compute_ncv_split_diagnostics(
                    payoff=pricing_split["payoff_arithmetic"],
                    h_vals=snap.network.forward(pricing_split["Z"]),
                    e_h=snap.e_h,
                    n_reporting=config.pricing_paths,
                )
                advantage_val = (
                    float(gcv_val["observation_variance"]) / float(val_diag["residual_variance"])
                    if math.isfinite(float(gcv_val["observation_variance"]))
                    and math.isfinite(float(val_diag["residual_variance"]))
                    and float(val_diag["residual_variance"]) > 0.0
                    else float("nan")
                )
                checkpoint_curve_rows.append(
                    {
                        "replication": rep,
                        "monitoring_dates": m,
                        "checkpoint": int(cp),
                        "trajectory_id": trajectory_id,
                        "training_seed": rep_seeds["train"],
                        "validation_seed": rep_seeds["validation"],
                        "pilot_seed": rep_seeds["pilot"],
                        "pricing_seed": rep_seeds["pricing"],
                        "train_mse": snap.train_mse,
                        "validation_mse": snap.validation_mse,
                        "validation_ncv_observation_variance": val_diag["residual_variance"],
                        "validation_gcv_observation_variance": gcv_val["observation_variance"],
                        "advantage": advantage_val,
                        "pricing_ncv_observation_variance": price_diag["residual_variance"],
                        "pricing_gcv_observation_variance": gcv_price["observation_variance"],
                        "optimizer_cumulative_training_runtime_s": snap.optimizer_cumulative_runtime_s,
                        "training_data_generation_runtime_s": data_generation_runtime_s,
                        "ncv_setup_cost_s": compute_ncv_setup_cost(
                            data_generation_runtime_s,
                            snap.optimizer_cumulative_runtime_s,
                            cp,
                        ),
                    }
                )

            mc_price = float(np.mean(pricing_split["payoff_arithmetic"]))
            mc_obs_var = _sample_var(pricing_split["payoff_arithmetic"])
            mc_est_var = mc_obs_var / config.pricing_paths if math.isfinite(mc_obs_var) else float("nan")
            mc_se = math.sqrt(mc_est_var) if math.isfinite(mc_est_var) and mc_est_var >= 0.0 else float("nan")

            for epoch in FORMAL_EPOCHS:
                snap = snapshots[int(epoch)]
                price_diag = compute_ncv_split_diagnostics(
                    payoff=pricing_split["payoff_arithmetic"],
                    h_vals=snap.network.forward(pricing_split["Z"]),
                    e_h=snap.e_h,
                    n_reporting=config.pricing_paths,
                )

                obs_var_ncv = float(price_diag["residual_variance"])
                obs_var_gcv = float(gcv_price["observation_variance"])
                obs_var_mc = float(mc_obs_var)

                gcv_vrr = obs_var_mc / obs_var_gcv if obs_var_gcv > 0.0 else float("nan")
                ncv_vrr = obs_var_mc / obs_var_ncv if obs_var_ncv > 0.0 else float("nan")
                advantage = obs_var_gcv / obs_var_ncv if obs_var_ncv > 0.0 else float("nan")
                n_required, n_reason = required_ncv_observations_to_match_gcv(obs_var_ncv, obs_var_gcv, config.pricing_paths)

                rt = _cell_runtime(
                    torch_mod=torch,
                    monitoring_dates=m,
                    pricing_seed=rep_seeds["runtime"],
                    pricing_paths=config.pricing_paths,
                    ncv_snapshot=snap,
                    gcv_fit=gcv_fit,
                    n_required_ncv=n_required,
                    direct_timing_max_paths=config.direct_timing_max_paths,
                    direct_timing_repeats=config.direct_timing_repeats,
                    data_generation_runtime_s=data_generation_runtime_s,
                )

                marginal_ncv_for_break_even = (
                    rt.ncv_marginal_runtime_matched_measured_s
                    if isinstance(rt.ncv_marginal_runtime_matched_measured_s, (int, float))
                    else rt.ncv_marginal_runtime_matched_projected_s
                )
                be = solve_break_even_q(
                    baseline_setup_cost=float(rt.gcv_setup_runtime_s),
                    baseline_marginal_cost=float(rt.gcv_pricing_runtime_s),
                    proposed_setup_cost=float(rt.ncv_setup_runtime_s),
                    proposed_marginal_cost=float(marginal_ncv_for_break_even)
                    if isinstance(marginal_ncv_for_break_even, (int, float))
                    else float("nan"),
                )

                row = {
                    "replication": rep,
                    "monitoring_dates": m,
                    "ncv_epoch": int(epoch),
                    "trajectory_id": trajectory_id,
                    "training_seed": rep_seeds["train"],
                    "validation_seed": rep_seeds["validation"],
                    "pilot_seed": rep_seeds["pilot"],
                    "pricing_seed": rep_seeds["pricing"],
                    "mc_price": mc_price,
                    "mc_observation_variance": obs_var_mc,
                    "mc_standard_error": mc_se,
                    "gcv_price": gcv_price["price"],
                    "gcv_observation_variance": obs_var_gcv,
                    "gcv_estimator_variance": gcv_price["estimator_variance"],
                    "gcv_standard_error": gcv_price["standard_error"],
                    "ncv_price": price_diag["ncv_price_estimate"],
                    "ncv_observation_variance": obs_var_ncv,
                    "ncv_estimator_variance": price_diag["estimator_variance_at_reporting_n"],
                    "ncv_standard_error": price_diag["standard_error_at_reporting_n"],
                    "gcv_vrr_vs_mc": gcv_vrr,
                    "ncv_vrr_vs_mc": ncv_vrr,
                    "ncv_to_gcv_advantage": advantage,
                    "ncv_beats_gcv": bool(isinstance(advantage, (int, float)) and math.isfinite(float(advantage)) and float(advantage) > 1.0),
                    "required_ncv_observations_to_match_gcv_50000": n_required if n_required is not None else "NA",
                    "required_ncv_observations_reason": n_reason,
                    "optimizer_cumulative_training_runtime_s": snap.optimizer_cumulative_runtime_s,
                    "training_data_generation_runtime_s": data_generation_runtime_s,
                    "ncv_setup_cost_s": rt.ncv_setup_runtime_s,
                    "gcv_setup_cost_s": rt.gcv_setup_runtime_s,
                    "mc_marginal_pricing_runtime_s": rt.mc_pricing_runtime_s,
                    "gcv_marginal_pricing_runtime_s": rt.gcv_pricing_runtime_s,
                    "ncv_marginal_pricing_runtime_50000_s": rt.ncv_marginal_runtime_50000_s,
                    "ncv_marginal_pricing_runtime_matched_measured_s": rt.ncv_marginal_runtime_matched_measured_s if rt.ncv_marginal_runtime_matched_measured_s is not None else "NA",
                    "ncv_marginal_pricing_runtime_matched_projected_s": rt.ncv_marginal_runtime_matched_projected_s if rt.ncv_marginal_runtime_matched_projected_s is not None else "NA",
                    "break_even_q": be.get("break_even_q", "NA"),
                    "break_even_failure_reason": be.get("failure_reason", ""),
                    "break_even_verified_q": be.get("verified_q", False),
                    "break_even_verified_q_minus_1": be.get("verified_q_minus_1", False),
                    "break_even_q_minus_1_verification_status": be.get("q_minus_1_verification_status", ""),
                }
                replication_rows.append(row)
                runtime_rows.append(
                    {
                        "replication": rep,
                        "monitoring_dates": m,
                        "ncv_epoch": int(epoch),
                        "trajectory_id": trajectory_id,
                        "training_data_generation_runtime_s": data_generation_runtime_s,
                        "optimizer_cumulative_training_runtime_s": snap.optimizer_cumulative_runtime_s,
                        "ncv_setup_cost_s": rt.ncv_setup_runtime_s,
                        "gcv_setup_cost_s": rt.gcv_setup_runtime_s,
                        "gcv_marginal_pricing_runtime_s": rt.gcv_pricing_runtime_s,
                        "ncv_marginal_pricing_runtime_50000_s": rt.ncv_marginal_runtime_50000_s,
                        "ncv_marginal_pricing_runtime_matched_measured_s": rt.ncv_marginal_runtime_matched_measured_s if rt.ncv_marginal_runtime_matched_measured_s is not None else "NA",
                        "ncv_marginal_pricing_runtime_matched_projected_s": rt.ncv_marginal_runtime_matched_projected_s if rt.ncv_marginal_runtime_matched_projected_s is not None else "NA",
                        "break_even_q": be.get("break_even_q", "NA"),
                        "break_even_failure_reason": be.get("failure_reason", ""),
                    }
                )

    cell_summary = _build_cell_summary(replication_rows)
    paired_contrasts = compute_paired_contrasts(replication_rows)

    _write_csv(run_dir / "replication_level_results.csv", replication_rows)
    _write_csv(run_dir / "cell_summary.csv", cell_summary)
    _write_csv(run_dir / "paired_contrasts.csv", paired_contrasts)
    _write_csv(run_dir / "checkpoint_curve_results.csv", checkpoint_curve_rows)
    _write_csv(run_dir / "runtime_results.csv", runtime_rows)

    _plot_checkpoint_curves(run_dir, checkpoint_curve_rows)
    _plot_interaction(run_dir, replication_rows)

    seed_ok, seed_failures = _validate_seed_independence(seed_manifest)
    design_cells = {(int(r["monitoring_dates"]), int(r["ncv_epoch"])) for r in replication_rows}
    expected_cells = {(12, 25), (12, 1000), (252, 25), (252, 1000)}
    checkpoints_present = {int(r["checkpoint"]) for r in checkpoint_curve_rows}

    validation_errors: list[str] = []
    if design_cells != expected_cells:
        validation_errors.append(f"formal design cells mismatch: expected={sorted(expected_cells)} got={sorted(design_cells)}")
    if not seed_ok:
        validation_errors.extend(seed_failures)
    if not set(FORMAL_EPOCHS).issubset(checkpoints_present):
        validation_errors.append("formal checkpoints 25/1000 missing from checkpoint results")

    validation_report = {
        "passed": len(validation_errors) == 0,
        "errors": validation_errors,
        "warnings": [],
        "formal_design_cells": sorted([list(x) for x in expected_cells]),
        "observed_design_cells": sorted([list(x) for x in design_cells]),
        "checkpoint_grid": list(config.checkpoints),
        "formal_epochs": list(FORMAL_EPOCHS),
        "replications": config.replications,
        "seed_streams_independent_within_replication": seed_ok,
        "pricing_sample_size": config.pricing_paths,
    }
    _write_json(run_dir / "validation_report.json", validation_report)

    summary_text = _build_dissertation_summary(
        config=config,
        cell_summary=cell_summary,
        contrasts=paired_contrasts,
        validation_report=validation_report,
    )
    (run_dir / "dissertation_summary.md").write_text(summary_text, encoding="utf-8")

    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run focused 2x2 monitoring-profile x training-horizon sensitivity study")
    p.add_argument("--profile", choices=["smoke", "dissertation"], default="smoke")
    p.add_argument("--output-dir", default="experiment_runs")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--replications", type=int, default=None)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    cfg = profile_config(profile=args.profile, output_dir=args.output_dir, base_seed=args.base_seed)
    if args.replications is not None:
        cfg = dataclasses.replace(cfg, replications=int(args.replications))
    out = run_sensitivity_study(cfg)
    print(f"Stage 8 2x2 sensitivity output: {out.resolve()}")


if __name__ == "__main__":
    main()
