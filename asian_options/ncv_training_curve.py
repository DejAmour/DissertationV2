from __future__ import annotations

import copy
import dataclasses
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from asian_options.analytical import geometric_asian_call_price
from asian_options.config import ModelConfig, collect_environment_metadata, seed_everything
from asian_options.contracts import make_contract_cfg
from asian_options.neural_cv import _ShallowNet, analytical_network_expectation, build_network
from asian_options.payoffs import arithmetic_asian_call_payoff, geometric_asian_call_payoff
from asian_options.simulate_gbm import simulate_paths


SEED_OFFSETS = {
    "train": 1_000,
    "validation": 2_000,
    "test": 3_000,
    "gcv_pilot": 4_000,
}


@dataclass(frozen=True)
class TrainingCurveConfig:
    profile: str
    base_seed: int
    replications: int
    train_paths: int
    validation_paths: int
    test_paths: int
    monitoring_dates: int
    checkpoints: tuple[int, ...]
    hidden_width: int
    learning_rate: float
    default_epochs: int
    train_batch_size: int
    runtime_repeats: int
    pricing_observations_for_reporting: int
    q_values: tuple[int, ...]
    se_targets: tuple[float, ...]
    output_dir: str


def ncv_training_facts() -> dict[str, Any]:
    return {
        "architecture": "One-hidden-layer feed-forward net: H_theta(Z)=W2*ReLU(W1@Z+b1)+b2",
        "hidden_layer_width_default": 32,
        "activation": "ReLU",
        "loss_function": "torch.nn.MSELoss (mean squared error)",
        "optimizer": "torch.optim.Adam",
        "learning_rate_default": 1e-2,
        "epoch_count_defaults": {
            "neural_cv.train_network": 200,
            "stage8_scratch_and_reference": 100,
        },
        "weight_initialization": "Xavier-uniform for W1/W2, zero biases",
        "input_dimension": "m monitoring shocks (cfg.m; Stage 8 reference uses m=252)",
        "training_targets": "Discounted arithmetic Asian call payoff",
        "validation_or_early_stopping": "No validation split and no early stopping in current training path",
        "objective_preserved_for_training_curve": "MSE objective preserved",
    }


def profile_config(profile: str, output_dir: str, base_seed: int = 42) -> TrainingCurveConfig:
    p = profile.lower()
    if p == "smoke":
        return TrainingCurveConfig(
            profile="smoke",
            base_seed=base_seed,
            replications=2,
            train_paths=100,
            validation_paths=200,
            test_paths=500,
            monitoring_dates=252,
            checkpoints=(0, 2, 5, 10),
            hidden_width=32,
            learning_rate=1e-2,
            default_epochs=100,
            train_batch_size=256,
            runtime_repeats=3,
            pricing_observations_for_reporting=50_000,
            q_values=(1, 10, 100, 1000),
            se_targets=(0.001,),
            output_dir=output_dir,
        )
    if p == "dissertation":
        return TrainingCurveConfig(
            profile="dissertation",
            base_seed=base_seed,
            replications=10,
            train_paths=5_000,
            validation_paths=10_000,
            test_paths=50_000,
            monitoring_dates=252,
            checkpoints=(0, 10, 25, 50, 100, 200, 500, 1000),
            hidden_width=32,
            learning_rate=1e-2,
            default_epochs=100,
            train_batch_size=256,
            runtime_repeats=5,
            pricing_observations_for_reporting=50_000,
            q_values=(1, 10, 100, 1000),
            se_targets=(0.001,),
            output_dir=output_dir,
        )
    raise ValueError(f"Unknown profile: {profile}")


def validate_checkpoints(checkpoints: list[int] | tuple[int, ...]) -> None:
    if not checkpoints:
        raise ValueError("checkpoints cannot be empty")
    if checkpoints[0] != 0:
        raise ValueError("checkpoints must start at 0")
    for i in range(1, len(checkpoints)):
        if checkpoints[i] <= checkpoints[i - 1]:
            raise ValueError("checkpoints must be strictly increasing")


def replication_seeds(base_seed: int, replication: int) -> dict[str, int]:
    rep_offset = replication * 10_000
    out: dict[str, int] = {}
    for phase, offset in SEED_OFFSETS.items():
        out[phase] = base_seed + rep_offset + offset
    return out


def build_seed_manifest(base_seed: int, replications: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rep in range(replications):
        seeds = replication_seeds(base_seed, rep)
        for phase in ("train", "validation", "test", "gcv_pilot"):
            rows.append({"replication": rep, "phase": phase, "seed": seeds[phase]})
    return rows


def _make_reference_cfg(monitoring_dates: int, n_paths: int, seed: int) -> ModelConfig:
    base_cfg = make_contract_cfg("reference", n_paths=n_paths, seed=seed)
    return dataclasses.replace(base_cfg, m=monitoring_dates, n_paths=n_paths, seed=seed)


def simulate_split_dataset(monitoring_dates: int, n_paths: int, seed: int) -> dict[str, np.ndarray]:
    cfg = _make_reference_cfg(monitoring_dates=monitoring_dates, n_paths=n_paths, seed=seed)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_paths, cfg.m))
    paths = simulate_paths(cfg, shocks=z)
    arithmetic = arithmetic_asian_call_payoff(paths, cfg)
    geometric = geometric_asian_call_payoff(paths, cfg)
    return {
        "Z": z,
        "paths": paths,
        "payoff_arithmetic": arithmetic,
        "payoff_geometric": geometric,
        "cfg": cfg,
    }


def _network_from_torch_model(model) -> _ShallowNet:
    w1 = model.linear1.weight.detach().cpu().numpy().copy()
    b1 = model.linear1.bias.detach().cpu().numpy().copy()
    w2 = model.linear2.weight.detach().cpu().numpy().copy()
    b2 = model.linear2.bias.detach().cpu().numpy().copy()
    return _ShallowNet(w1, b1, w2, b2)


def _torch_model_from_initial_network(torch, network: _ShallowNet):
    class TorchShallowNet(torch.nn.Module):
        def __init__(self, m: int, hidden_width: int):
            super().__init__()
            self.linear1 = torch.nn.Linear(m, hidden_width)
            self.linear2 = torch.nn.Linear(hidden_width, 1)

        def forward(self, x):
            return self.linear2(torch.relu(self.linear1(x))).squeeze(1)

    model = TorchShallowNet(network.W1.shape[1], network.W1.shape[0]).double()
    with torch.no_grad():
        model.linear1.weight.copy_(torch.tensor(network.W1, dtype=torch.float64))
        model.linear1.bias.copy_(torch.tensor(network.b1, dtype=torch.float64))
        model.linear2.weight.copy_(torch.tensor(network.W2, dtype=torch.float64))
        model.linear2.bias.copy_(torch.tensor(network.b2, dtype=torch.float64))
    return model


def _sample_var(x: np.ndarray) -> float:
    if x.size <= 1:
        return float("nan")
    return float(np.var(x, ddof=1))


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    vx = _sample_var(x)
    vy = _sample_var(y)
    if (not math.isfinite(vx)) or (not math.isfinite(vy)) or vx <= 0.0 or vy <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_ncv_split_diagnostics(payoff: np.ndarray, h_vals: np.ndarray, e_h: float, n_reporting: int = 50_000) -> dict[str, float]:
    residual = payoff - h_vals
    corrected = residual + e_h
    payoff_var = _sample_var(payoff)
    h_var = _sample_var(h_vals)
    cov = float(np.cov(payoff, h_vals, ddof=1)[0, 1]) if payoff.size > 1 else float("nan")
    residual_mean = float(np.mean(residual))
    residual_var = _sample_var(residual)
    vrr = float("nan")
    if math.isfinite(payoff_var) and math.isfinite(residual_var) and residual_var > 0.0:
        vrr = payoff_var / residual_var
    estimator_var = float("nan")
    std_error = float("nan")
    if math.isfinite(residual_var) and residual_var >= 0.0 and n_reporting > 0:
        estimator_var = residual_var / n_reporting
        std_error = math.sqrt(estimator_var)

    return {
        "arithmetic_payoff_mean": float(np.mean(payoff)),
        "network_output_mean": float(np.mean(h_vals)),
        "analytical_eh": float(e_h),
        "payoff_variance": payoff_var,
        "network_output_variance": h_var,
        "payoff_network_covariance": cov,
        "payoff_network_correlation": _safe_corr(payoff, h_vals),
        "residual_mean": residual_mean,
        "residual_variance": residual_var,
        "ncv_price_estimate": float(np.mean(corrected)),
        "estimator_variance_at_reporting_n": estimator_var,
        "standard_error_at_reporting_n": std_error,
        "vrr_ncv_vs_mc": vrr,
        "residual_variance_shift_check": _sample_var(corrected),
        "residual_shift_delta": _sample_var(corrected) - residual_var if math.isfinite(residual_var) else float("nan"),
    }


def compute_required_paths(residual_variance: float, target_se: float) -> int:
    if not math.isfinite(residual_variance) or residual_variance <= 0.0:
        return 2
    return max(2, int(math.ceil(residual_variance / (target_se ** 2))))


def compute_total_cost(training_runtime: float, required_paths: int, per_obs_runtime: float, reuse_q: int) -> float:
    return training_runtime + reuse_q * (required_paths * per_obs_runtime)


def compute_gcv_benchmark(validation_split: dict[str, np.ndarray], test_split: dict[str, np.ndarray], pilot_split: dict[str, np.ndarray], n_reporting: int = 50_000) -> list[dict[str, Any]]:
    t0 = time.perf_counter()
    pilot_x = pilot_split["payoff_arithmetic"]
    pilot_g = pilot_split["payoff_geometric"]
    var_g = _sample_var(pilot_g)
    cov_xg = float(np.cov(pilot_x, pilot_g, ddof=1)[0, 1]) if pilot_x.size > 1 else float("nan")
    beta = cov_xg / var_g if math.isfinite(var_g) and var_g > 0 else float("nan")
    pilot_runtime = time.perf_counter() - t0

    out: list[dict[str, Any]] = []
    eg = geometric_asian_call_price(validation_split["cfg"])
    for split_name, split in (("validation", validation_split), ("test", test_split)):
        t1 = time.perf_counter()
        x = split["payoff_arithmetic"]
        g = split["payoff_geometric"]
        corrected = x - beta * (g - eg) if math.isfinite(beta) else x
        pricing_runtime = time.perf_counter() - t1
        payoff_var = _sample_var(x)
        residual_var = _sample_var(corrected)
        vrr = payoff_var / residual_var if math.isfinite(residual_var) and residual_var > 0 else float("nan")
        se = math.sqrt(residual_var / n_reporting) if math.isfinite(residual_var) and residual_var >= 0 else float("nan")
        out.append(
            {
                "split": split_name,
                "gcv_beta": beta,
                "arith_geom_correlation": _safe_corr(x, g),
                "gcv_residual_variance": residual_var,
                "gcv_vrr_vs_mc": vrr,
                "gcv_standard_error_at_reporting_n": se,
                "gcv_pilot_runtime_s": pilot_runtime,
                "gcv_pricing_runtime_s": pricing_runtime,
                "gcv_per_observation_runtime_s": pricing_runtime / max(1, x.size),
                "gcv_price_estimate": float(np.mean(corrected)),
                "gcv_analytical_eg": float(eg),
            }
        )
    return out


def measure_inference_runtime(network: _ShallowNet, z_inputs: np.ndarray, repeats: int) -> dict[str, float]:
    times: list[float] = []
    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        _ = network.forward(z_inputs)
        times.append(time.perf_counter() - t0)
    return {
        "timed_repeats": int(max(1, repeats)),
        "inference_runtime_median_s": float(statistics.median(times)),
        "inference_runtime_mean_s": float(statistics.mean(times)),
        "inference_runtime_std_s": float(statistics.stdev(times)) if len(times) > 1 else 0.0,
        "inference_runtime_per_observation_median_s": float(statistics.median(times) / max(1, z_inputs.shape[0])),
    }


def _confidence95(values: list[float]) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    if len(values) == 1:
        return values[0], values[0]
    m = statistics.mean(values)
    s = statistics.stdev(values)
    half = 1.96 * s / math.sqrt(len(values))
    return m - half, m + half


def summarize_rows(rows: list[dict[str, Any]], group_keys: list[str], metrics: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        grouped.setdefault(key, []).append(row)

    out: list[dict[str, Any]] = []
    for key, items in grouped.items():
        base = {k: v for k, v in zip(group_keys, key)}
        for metric in metrics:
            vals = [float(r[metric]) for r in items if isinstance(r.get(metric), (int, float)) and math.isfinite(float(r[metric]))]
            if not vals:
                continue
            lo, hi = _confidence95(vals)
            out.append(
                {
                    **base,
                    "metric": metric,
                    "count": len(vals),
                    "mean": float(statistics.mean(vals)),
                    "std_dev": float(statistics.stdev(vals)) if len(vals) > 1 else 0.0,
                    "median": float(statistics.median(vals)),
                    "minimum": float(min(vals)),
                    "maximum": float(max(vals)),
                    "ci95_lower": float(lo),
                    "ci95_upper": float(hi),
                }
            )
    out.sort(key=lambda r: tuple(r[k] for k in group_keys) + (r["metric"],))
    return out


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

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


def validate_output_schema(output_dir: Path) -> dict[str, Any]:
    required = [
        "training_curve_config.json",
        "training_curve_environment.json",
        "training_curve_seed_manifest.csv",
        "training_curve_per_replication.csv",
        "training_curve_summary.csv",
        "training_curve_diminishing_returns.csv",
        "training_curve_optimal_checkpoints.csv",
        "training_curve_gcv_benchmark.csv",
        "training_curve_validation_report.json",
        "TRAINING_CURVE_HANDOVER.md",
        "ncv_training_curve_summary.png",
    ]
    exists = {name: (output_dir / name).exists() for name in required}
    return {
        "required_files": required,
        "exists": exists,
        "all_present": all(exists.values()),
    }


def build_handover_text(config: TrainingCurveConfig, facts: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# TRAINING_CURVE_HANDOVER",
            "",
            "## Current NCV training facts (verified from code)",
            f"- Architecture: {facts['architecture']}",
            f"- Hidden width: {facts['hidden_layer_width_default']}",
            f"- Activation: {facts['activation']}",
            f"- Loss: {facts['loss_function']}",
            f"- Optimizer: {facts['optimizer']}",
            f"- Learning rate: {facts['learning_rate_default']}",
            f"- Epoch defaults: {facts['epoch_count_defaults']}",
            f"- Initialization: {facts['weight_initialization']}",
            f"- Input dimension: {facts['input_dimension']}",
            f"- Targets: {facts['training_targets']}",
            f"- Validation/early stopping in current path: {facts['validation_or_early_stopping']}",
            "",
            "## Training-curve experiment scope",
            f"- Profile: {config.profile}",
            f"- Replications: {config.replications}",
            f"- Checkpoints: {list(config.checkpoints)}",
            "- Uses independent training, validation, and held-out test splits per replication.",
            "- Trains one continuous network per replication and snapshots checkpoints without re-initialization.",
            "",
            "## Output-rescaling note (no new estimator added)",
            "The existing transferred control coefficient beta already provides scalar output rescaling.",
            "If H~(Z)=aH(Z), then H~(Z)-E[H~(Z)]=a(H(Z)-E[H(Z)]), so multiplying output weights by a is equivalent to beta=a.",
            "No separate physical output-weight rescaling estimator is implemented to avoid duplicating beta-transfer behavior.",
            "",
            "## Future work (not implemented here)",
            "- Retraining individual output weights",
            "- Last-layer fine-tuning",
            "- Parameter-conditioned networks",
            "- Adjusting first-layer weights/biases across strikes/volatility/maturity",
            "",
        ]
    )


def _plot_summary_figure(output_dir: Path, checkpoints: list[int], per_rep_rows: list[dict[str, Any]], gcv_rows: list[dict[str, Any]], target_se: float) -> None:
    import matplotlib.pyplot as plt

    def _series(metric: str, split: str = "validation"):
        by_cp: dict[int, list[float]] = {cp: [] for cp in checkpoints}
        for r in per_rep_rows:
            if r["split"] != split:
                continue
            cp = int(r["checkpoint"])
            v = r.get(metric)
            if isinstance(v, (int, float)) and math.isfinite(float(v)):
                by_cp[cp].append(float(v))
        means = [statistics.mean(by_cp[cp]) if by_cp[cp] else float("nan") for cp in checkpoints]
        bands = []
        for cp in checkpoints:
            vals = by_cp[cp]
            if len(vals) >= 2:
                m = statistics.mean(vals)
                s = statistics.stdev(vals)
                h = 1.96 * s / math.sqrt(len(vals))
                bands.append((m - h, m + h))
            elif len(vals) == 1:
                bands.append((vals[0], vals[0]))
            else:
                bands.append((float("nan"), float("nan")))
        lows = [b[0] for b in bands]
        highs = [b[1] for b in bands]
        return means, lows, highs

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    train_mean, train_lo, train_hi = _series("train_loss", split="validation")
    val_mean, val_lo, val_hi = _series("validation_loss", split="validation")
    axes[0].plot(checkpoints, train_mean, label="Train loss")
    axes[0].plot(checkpoints, val_mean, label="Validation loss")
    axes[0].fill_between(checkpoints, train_lo, train_hi, alpha=0.2)
    axes[0].fill_between(checkpoints, val_lo, val_hi, alpha=0.2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_yscale("log")
    axes[0].legend()

    vrr_mean, vrr_lo, vrr_hi = _series("vrr_ncv_vs_mc", split="validation")
    axes[1].plot(checkpoints, vrr_mean, label="NCV validation VRR")
    axes[1].fill_between(checkpoints, vrr_lo, vrr_hi, alpha=0.2)
    gcv_val = [r for r in gcv_rows if r["split"] == "validation"]
    if gcv_val:
        gcv_mean = statistics.mean([float(r["gcv_vrr_vs_mc"]) for r in gcv_val if math.isfinite(float(r["gcv_vrr_vs_mc"]))])
        axes[1].axhline(gcv_mean, linestyle="--", label="GCV VRR")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("VRR")
    axes[1].set_yscale("log")
    axes[1].legend()

    for q in (1, 10, 100, 1000):
        ys = []
        for cp in checkpoints:
            vals = []
            for r in per_rep_rows:
                if r["split"] != "validation" or int(r["checkpoint"]) != cp:
                    continue
                n_req = compute_required_paths(float(r["residual_variance"]), target_se)
                vals.append(
                    compute_total_cost(
                        float(r["cumulative_training_runtime_s"]),
                        n_req,
                        float(r["inference_runtime_per_observation_median_s"]),
                        q,
                    )
                )
            ys.append(statistics.mean(vals) if vals else float("nan"))
        axes[2].plot(checkpoints, ys, label=f"Q={q}")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Projected total cost")
    axes[2].set_yscale("log")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "ncv_training_curve_summary.png", dpi=200)
    plt.close(fig)


def run_training_curve_experiment(config: TrainingCurveConfig) -> Path:
    validate_checkpoints(config.checkpoints)
    seed_everything(config.base_seed)
    torch = __import__("torch")

    facts = ncv_training_facts()
    cp_max = max(config.checkpoints)
    if cp_max > config.default_epochs:
        checkpoint_conflict = {
            "requested_max_checkpoint": cp_max,
            "existing_default_epoch_count": config.default_epochs,
            "conflict": False,
            "note": "No optimizer/scheduler conflict; experiment extends training horizon beyond current default.",
        }
    else:
        checkpoint_conflict = {
            "requested_max_checkpoint": cp_max,
            "existing_default_epoch_count": config.default_epochs,
            "conflict": False,
            "note": "Checkpoint grid within existing default epoch count.",
        }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(config.output_dir) / f"ncv_training_curve_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_manifest = build_seed_manifest(config.base_seed, config.replications)
    per_replication_rows: list[dict[str, Any]] = []
    diminishing_rows: list[dict[str, Any]] = []
    optimal_rows: list[dict[str, Any]] = []
    gcv_rows: list[dict[str, Any]] = []

    # Warm-up (unrecorded)
    _ = np.random.default_rng(777).standard_normal((64, 64)) @ np.random.default_rng(778).standard_normal((64, 64)).T
    _ = torch.randn(64, 64) @ torch.randn(64, 64).T

    for rep in range(config.replications):
        seeds = replication_seeds(config.base_seed, rep)
        train_split = simulate_split_dataset(config.monitoring_dates, config.train_paths, seeds["train"])
        val_split = simulate_split_dataset(config.monitoring_dates, config.validation_paths, seeds["validation"])
        test_split = simulate_split_dataset(config.monitoring_dates, config.test_paths, seeds["test"])
        pilot_split = simulate_split_dataset(config.monitoring_dates, config.validation_paths, seeds["gcv_pilot"])

        gcv_bench = compute_gcv_benchmark(val_split, test_split, pilot_split, n_reporting=config.pricing_observations_for_reporting)
        for gr in gcv_bench:
            gcv_rows.append({"replication": rep, **gr})

        cfg_train = train_split["cfg"]
        network = build_network(cfg_train, hidden_width=config.hidden_width)
        model = _torch_model_from_initial_network(torch, network)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        loss_fn = torch.nn.MSELoss()

        x_train = torch.tensor(train_split["Z"], dtype=torch.float64)
        y_train = torch.tensor(train_split["payoff_arithmetic"], dtype=torch.float64)
        x_val = torch.tensor(val_split["Z"], dtype=torch.float64)
        y_val = torch.tensor(val_split["payoff_arithmetic"], dtype=torch.float64)
        x_test = torch.tensor(test_split["Z"], dtype=torch.float64)
        y_test = torch.tensor(test_split["payoff_arithmetic"], dtype=torch.float64)

        batch_size = min(config.train_batch_size, len(x_train))
        train_loss_by_epoch: list[float] = []
        eval_no_grad_used = True
        snapshots: dict[int, dict[str, Any]] = {}
        checkpoints = list(config.checkpoints)
        cumulative_training_s = 0.0
        previous_checkpoint_s = 0.0

        def evaluate_checkpoint(epoch: int, cumulative_runtime_s: float) -> None:
            nonlocal eval_no_grad_used
            model.eval()
            with torch.no_grad():
                eval_no_grad_used = eval_no_grad_used and (not torch.is_grad_enabled())
                train_loss = float(loss_fn(model(x_train), y_train).item())
                val_loss = float(loss_fn(model(x_val), y_val).item())
                test_loss = float(loss_fn(model(x_test), y_test).item())

            net_snapshot = _network_from_torch_model(model)
            e_h = analytical_network_expectation(net_snapshot)

            h_val = net_snapshot.forward(val_split["Z"])
            h_test = net_snapshot.forward(test_split["Z"])

            val_diag = compute_ncv_split_diagnostics(
                payoff=val_split["payoff_arithmetic"],
                h_vals=h_val,
                e_h=e_h,
                n_reporting=config.pricing_observations_for_reporting,
            )
            test_diag = compute_ncv_split_diagnostics(
                payoff=test_split["payoff_arithmetic"],
                h_vals=h_test,
                e_h=e_h,
                n_reporting=config.pricing_observations_for_reporting,
            )
            inference = measure_inference_runtime(net_snapshot, test_split["Z"], config.runtime_repeats)

            snapshots[epoch] = {
                "epoch": epoch,
                "network": net_snapshot,
                "cumulative_training_runtime_s": cumulative_runtime_s,
                "train_loss": train_loss,
                "validation_loss": val_loss,
                "test_loss": test_loss,
                "val_diag": val_diag,
                "test_diag": test_diag,
                "inference": inference,
                "e_h": e_h,
            }

        evaluate_checkpoint(0, 0.0)

        t_train_start = time.perf_counter()
        checkpoint_set = set(checkpoints)
        for epoch in range(1, max(checkpoints) + 1):
            model.train()
            perm = torch.randperm(len(x_train))
            batch_losses: list[float] = []
            for start in range(0, len(x_train), batch_size):
                idx = perm[start : start + batch_size]
                xb = x_train[idx]
                yb = y_train[idx]
                pred = model(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.item()))
            train_loss_by_epoch.append(float(statistics.mean(batch_losses)))
            cumulative_training_s = time.perf_counter() - t_train_start
            if epoch in checkpoint_set:
                evaluate_checkpoint(epoch, cumulative_training_s)

        for cp in checkpoints:
            snap = snapshots[cp]
            cumulative = float(snap["cumulative_training_runtime_s"])
            incremental = cumulative - previous_checkpoint_s
            previous_checkpoint_s = cumulative

            for split_name, diag in (("validation", snap["val_diag"]), ("test", snap["test_diag"])):
                row = {
                    "replication": rep,
                    "checkpoint": cp,
                    "split": split_name,
                    "train_seed": seeds["train"],
                    "validation_seed": seeds["validation"],
                    "test_seed": seeds["test"],
                    "gcv_pilot_seed": seeds["gcv_pilot"],
                    "train_paths": config.train_paths,
                    "validation_paths": config.validation_paths,
                    "test_paths": config.test_paths,
                    "train_loss": snap["train_loss"],
                    "validation_loss": snap["validation_loss"],
                    "test_loss": snap["test_loss"],
                    "cumulative_training_runtime_s": cumulative,
                    "incremental_training_runtime_s": 0.0 if cp == 0 else incremental,
                    "objective_name": "MSE",
                    "objective_mean_loss": snap["train_loss"] if split_name == "validation" else snap["test_loss"],
                    "centered_residual_variance": diag["residual_variance"],
                    "timed_repeats": snap["inference"]["timed_repeats"],
                    "inference_runtime_median_s": snap["inference"]["inference_runtime_median_s"],
                    "inference_runtime_mean_s": snap["inference"]["inference_runtime_mean_s"],
                    "inference_runtime_std_s": snap["inference"]["inference_runtime_std_s"],
                    "inference_runtime_per_observation_median_s": snap["inference"]["inference_runtime_per_observation_median_s"],
                    "evaluation_no_grad": eval_no_grad_used,
                    "epoch_zero_training_runtime_zero": snapshots[0]["epoch"] == 0,
                    **diag,
                }
                per_replication_rows.append(row)

        val_rows = [r for r in per_replication_rows if r["replication"] == rep and r["split"] == "validation"]
        val_rows = sorted(val_rows, key=lambda r: int(r["checkpoint"]))
        for i in range(1, len(val_rows)):
            prev_row = val_rows[i - 1]
            row = val_rows[i]
            prev_var = float(prev_row["residual_variance"])
            cur_var = float(row["residual_variance"])
            prev_vrr = float(prev_row["vrr_ncv_vs_mc"])
            cur_vrr = float(row["vrr_ncv_vs_mc"])
            delta_t = float(row["incremental_training_runtime_s"])
            var_pct = float("nan")
            if math.isfinite(prev_var) and prev_var != 0.0 and math.isfinite(cur_var):
                var_pct = 100.0 * (prev_var - cur_var) / prev_var
            vrr_pct = float("nan")
            if math.isfinite(prev_vrr) and prev_vrr != 0.0 and math.isfinite(cur_vrr):
                vrr_pct = 100.0 * (cur_vrr - prev_vrr) / prev_vrr
            gain_per_s = float("nan")
            if math.isfinite(delta_t) and delta_t > 0 and math.isfinite(prev_var) and math.isfinite(cur_var):
                gain_per_s = (prev_var - cur_var) / delta_t
            diminishing_rows.append(
                {
                    "row_type": "step_change",
                    "replication": rep,
                    "from_checkpoint": int(prev_row["checkpoint"]),
                    "to_checkpoint": int(row["checkpoint"]),
                    "validation_residual_variance_reduction_pct": var_pct,
                    "validation_vrr_improvement_pct": vrr_pct,
                    "incremental_training_time_s": delta_t,
                    "variance_reduction_per_training_second": gain_per_s,
                }
            )

        finite_val = [r for r in val_rows if math.isfinite(float(r["residual_variance"]))]
        if finite_val:
            min_row = min(finite_val, key=lambda r: float(r["residual_variance"]))
            min_var = float(min_row["residual_variance"])
            within_rows = [r for r in finite_val if float(r["residual_variance"]) <= 1.01 * min_var]
            first_within = min(within_rows, key=lambda r: int(r["checkpoint"]))
            min_cp = int(min_row["checkpoint"])
            deterioration = any(
                float(r["residual_variance"]) > min_var
                for r in finite_val
                if int(r["checkpoint"]) > min_cp
            )
            diminishing_rows.append(
                {
                    "row_type": "landmark",
                    "replication": rep,
                    "min_validation_residual_checkpoint": min_cp,
                    "first_within_1pct_checkpoint": int(first_within["checkpoint"]),
                    "validation_deteriorates_after_minimum": bool(deterioration),
                }
            )

        validation_rows = [r for r in per_replication_rows if r["replication"] == rep and r["split"] == "validation"]
        test_rows = [r for r in per_replication_rows if r["replication"] == rep and r["split"] == "test"]
        gcv_validation = next(gr for gr in gcv_rows if gr["replication"] == rep and gr["split"] == "validation")
        gcv_test = next(gr for gr in gcv_rows if gr["replication"] == rep and gr["split"] == "test")
        se_targets = [gcv_validation["gcv_standard_error_at_reporting_n"], 0.001]
        for target_se in se_targets:
            for q in config.q_values:
                per_checkpoint_costs = []
                for r in validation_rows:
                    req_n = compute_required_paths(float(r["residual_variance"]), float(target_se))
                    per_obs = float(r["inference_runtime_per_observation_median_s"])
                    cost = compute_total_cost(float(r["cumulative_training_runtime_s"]), req_n, per_obs, int(q))
                    per_checkpoint_costs.append((r, req_n, cost))
                best = min(per_checkpoint_costs, key=lambda x: x[2])
                best_row, best_n, best_cost = best
                matched_test_row = next(rr for rr in test_rows if int(rr["checkpoint"]) == int(best_row["checkpoint"]))

                test_required_n = compute_required_paths(float(matched_test_row["residual_variance"]), float(target_se))
                test_total_cost = compute_total_cost(
                    float(matched_test_row["cumulative_training_runtime_s"]),
                    test_required_n,
                    float(matched_test_row["inference_runtime_per_observation_median_s"]),
                    int(q),
                )

                gcv_req = compute_required_paths(float(gcv_validation["gcv_residual_variance"]), float(target_se))
                gcv_cost = compute_total_cost(0.0, gcv_req, float(gcv_validation["gcv_per_observation_runtime_s"]), int(q))
                gcv_test_req = compute_required_paths(float(gcv_test["gcv_residual_variance"]), float(target_se))
                gcv_test_cost = compute_total_cost(0.0, gcv_test_req, float(gcv_test["gcv_per_observation_runtime_s"]), int(q))
                optimal_rows.append(
                    {
                        "replication": rep,
                        "selection_split": "validation",
                        "target_se": float(target_se),
                        "Q": int(q),
                        "checkpoint": int(best_row["checkpoint"]),
                        "cumulative_training_runtime_s": float(best_row["cumulative_training_runtime_s"]),
                        "validation_residual_variance": float(best_row["residual_variance"]),
                        "test_residual_variance": float(matched_test_row["residual_variance"]),
                        "required_pricing_observations": int(best_n),
                        "projected_marginal_pricing_cost": float(best_n * float(best_row["inference_runtime_per_observation_median_s"])),
                        "projected_total_cost_validation": float(best_cost),
                        "projected_total_cost_test": float(test_total_cost),
                        "required_pricing_observations_test": int(test_required_n),
                        "gcv_projected_total_cost_validation": float(gcv_cost),
                        "gcv_projected_total_cost_test": float(gcv_test_cost),
                        "optimal_ncv_beats_gcv_validation": bool(best_cost < gcv_cost),
                        "optimal_ncv_beats_gcv_test": bool(test_total_cost < gcv_test_cost),
                    }
                )

    summary_rows = summarize_rows(
        rows=per_replication_rows,
        group_keys=["checkpoint", "split"],
        metrics=[
            "train_loss",
            "validation_loss",
            "test_loss",
            "residual_variance",
            "vrr_ncv_vs_mc",
            "cumulative_training_runtime_s",
            "inference_runtime_per_observation_median_s",
        ],
    )

    config_payload = dataclasses.asdict(config)
    config_payload["ncv_training_facts"] = facts
    config_payload["checkpoint_conflict_check"] = checkpoint_conflict

    env = collect_environment_metadata()
    env["seed"] = config.base_seed

    _write_json(out_dir / "training_curve_config.json", config_payload)
    _write_json(out_dir / "training_curve_environment.json", env)
    _write_csv(out_dir / "training_curve_seed_manifest.csv", seed_manifest)
    _write_csv(out_dir / "training_curve_per_replication.csv", per_replication_rows)
    _write_csv(out_dir / "training_curve_summary.csv", summary_rows)
    _write_csv(out_dir / "training_curve_diminishing_returns.csv", diminishing_rows)
    _write_csv(out_dir / "training_curve_optimal_checkpoints.csv", optimal_rows)
    _write_csv(out_dir / "training_curve_gcv_benchmark.csv", gcv_rows)

    _plot_summary_figure(out_dir, list(config.checkpoints), per_replication_rows, gcv_rows, target_se=0.001)

    (out_dir / "TRAINING_CURVE_HANDOVER.md").write_text(build_handover_text(config, facts), encoding="utf-8")

    validation_report = validate_output_schema(out_dir)
    validation_report.update(
        {
            "checkpoint_grid": list(config.checkpoints),
            "checkpoint_grid_starts_at_zero": config.checkpoints[0] == 0,
            "checkpoint_grid_strictly_increasing": all(config.checkpoints[i] < config.checkpoints[i + 1] for i in range(len(config.checkpoints) - 1)),
            "dissertation_monitoring_dates_252": config.monitoring_dates == 252 if config.profile == "dissertation" else True,
        }
    )
    _write_json(out_dir / "training_curve_validation_report.json", validation_report)

    return out_dir
