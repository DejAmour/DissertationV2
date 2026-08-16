from __future__ import annotations

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
from asian_options.estimators import antithetic_variates, standard_monte_carlo
from asian_options.neural_cv import _ShallowNet, analytical_network_expectation, build_network
from asian_options.payoffs import arithmetic_asian_call_payoff, geometric_asian_call_payoff
from asian_options.simulate_gbm import simulate_paths


SEED_OFFSETS = {
    "train": 1_000,
    "validation": 2_000,
    "test": 3_000,
    "gcv_pilot": 4_000,
}
TRAINING_CURVE_SELECTED_NCV_EPOCH = 25


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
    pilot_paths: int
    timing_path_counts: tuple[int, ...]
    timing_repeats: int
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
            pilot_paths=50,
            timing_path_counts=(100, 500),
            timing_repeats=1,
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
            pilot_paths=1_000,
            timing_path_counts=(1_000, 5_000, 10_000),
            timing_repeats=3,
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
    for split_name, split in (("validation", validation_split), ("test", test_split)):
        eg = geometric_asian_call_price(split["cfg"])
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
                "gcv_control_only_pricing_runtime_s": pricing_runtime,
                "gcv_control_only_per_observation_runtime_s": pricing_runtime / max(1, x.size),
                "gcv_per_observation_runtime_s": pricing_runtime / max(1, x.size),
                "path_and_payoff_runtime_s": pricing_runtime,
                "control_evaluation_runtime_s": 0.0,
                "estimator_reduction_runtime_s": 0.0,
                "end_to_end_pricing_runtime_s": pricing_runtime,
                "end_to_end_runtime_per_observation_s": pricing_runtime / max(1, x.size),
                "gcv_price_estimate": float(np.mean(corrected)),
                "gcv_analytical_eg": float(eg),
            }
        )
    return out


def measure_inference_runtime(network: _ShallowNet, z_inputs: np.ndarray, repeats: int) -> dict[str, float]:
    times: list[float] = []
    for _repeat in range(max(1, repeats)):
        t0 = time.perf_counter()
        _result = network.forward(z_inputs)
        times.append(time.perf_counter() - t0)
    return {
        "timed_repeats": int(max(1, repeats)),
        "inference_runtime_median_s": float(statistics.median(times)),
        "inference_runtime_mean_s": float(statistics.mean(times)),
        "inference_runtime_std_s": float(statistics.stdev(times)) if len(times) > 1 else 0.0,
        "inference_runtime_per_observation_median_s": float(statistics.median(times) / max(1, z_inputs.shape[0])),
    }


def _maybe_cuda_sync(torch_mod) -> None:
    if torch_mod is None:
        return
    cuda = getattr(torch_mod, "cuda", None)
    if cuda is None or not getattr(cuda, "is_available", lambda: False)():
        return
    cuda.synchronize()


def _fit_gcv_pilot_once(cfg: ModelConfig, n_pilot: int) -> dict[str, float]:
    if n_pilot < 2:
        raise ValueError(f"n_pilot must be >= 2, got {n_pilot}")
    t0 = time.perf_counter()
    pilot_cfg = dataclasses.replace(cfg, n_paths=n_pilot, seed=cfg.seed)
    pilot_paths = simulate_paths(pilot_cfg)
    x_pilot = arithmetic_asian_call_payoff(pilot_paths, pilot_cfg)
    g_pilot = geometric_asian_call_payoff(pilot_paths, pilot_cfg)
    var_g = _sample_var(g_pilot)
    cov_xg = float(np.cov(x_pilot, g_pilot, ddof=1)[0, 1]) if x_pilot.size > 1 else float("nan")
    beta = cov_xg / var_g if math.isfinite(var_g) and var_g > 0 else float("nan")
    return {
        "beta": beta,
        "eg": float(geometric_asian_call_price(cfg)),
        "pilot_runtime_s": time.perf_counter() - t0,
    }


def _timed_mc_av_gcv(
    method: str,
    cfg: ModelConfig,
    n_pilot: int = 0,
    repeats: int = 1,
    gcv_pilot_fit: dict[str, float] | None = None,
) -> dict[str, float]:
    times_path_and_payoff: list[float] = []
    times_control: list[float] = []
    times_reduce: list[float] = []
    times_end_to_end: list[float] = []
    times_setup: list[float] = []

    for rep_idx in range(max(1, repeats)):
        if method == "MC":
            res = standard_monte_carlo(cfg)
            times_end_to_end.append(float(res.end_to_end_runtime_seconds))
            times_path_and_payoff.append(float(res.end_to_end_runtime_seconds))
            times_control.append(0.0)
            times_reduce.append(0.0)
            times_setup.append(0.0)
            continue
        if method == "AV":
            res = antithetic_variates(cfg)
            times_end_to_end.append(float(res.end_to_end_runtime_seconds))
            times_path_and_payoff.append(float(res.end_to_end_runtime_seconds))
            times_control.append(0.0)
            times_reduce.append(0.0)
            times_setup.append(0.0)
            continue
        if method == "GCV":
            if gcv_pilot_fit is None:
                gcv_pilot_fit = _fit_gcv_pilot_once(cfg, n_pilot)
            rng = np.random.default_rng(cfg.seed + 1 + rep_idx)
            t0 = time.perf_counter()
            z = rng.standard_normal((cfg.n_paths, cfg.m))
            paths = simulate_paths(cfg, shocks=z)
            x = arithmetic_asian_call_payoff(paths, cfg)
            g = geometric_asian_call_payoff(paths, cfg)
            beta = float(gcv_pilot_fit["beta"])
            eg = float(gcv_pilot_fit["eg"])
            corrected = x - beta * (g - eg) if math.isfinite(beta) else x
            _price = float(np.mean(corrected))
            end_to_end = time.perf_counter() - t0
            times_end_to_end.append(end_to_end)
            times_path_and_payoff.append(end_to_end)
            times_control.append(0.0)
            times_reduce.append(0.0)
            times_setup.append(float(gcv_pilot_fit["pilot_runtime_s"]))
            continue
        raise ValueError(f"Unsupported method: {method}")

    end_to_end = statistics.median(times_end_to_end) if times_end_to_end else float("nan")
    path_payoff = statistics.median(times_path_and_payoff) if times_path_and_payoff else float("nan")
    control_eval = statistics.median(times_control) if times_control else float("nan")
    reduction = statistics.median(times_reduce) if times_reduce else float("nan")
    setup = statistics.median(times_setup) if times_setup else float("nan")
    return {
        "path_and_payoff_runtime_s": float(path_payoff),
        "control_evaluation_runtime_s": float(control_eval),
        "estimator_reduction_runtime_s": float(reduction),
        "end_to_end_pricing_runtime_s": float(end_to_end),
        "setup_runtime_s": float(setup),
    }


def build_runtime_profiles(
    *,
    methods: tuple[str, ...],
    monitoring_dates: int,
    seed_base: int,
    timing_path_counts: tuple[int, ...],
    timing_repeats: int,
    n_pilot: int,
    gcv_pilot_fit: dict[str, float] | None = None,
    progress_context: str,
) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for method in methods:
        print(f"[timing-method] {progress_context} method={method}", flush=True)
        method_offset = {"MC": 11, "AV": 17, "GCV": 23}.get(method, 31)
        for idx, n_paths in enumerate(timing_path_counts):
            print(f"[timing-path-count] {progress_context} method={method} n_paths={int(n_paths)}", flush=True)
            cfg = _make_reference_cfg(
                monitoring_dates=monitoring_dates,
                n_paths=int(n_paths),
                seed=seed_base + (idx * 1_000) + method_offset,
            )
            timed = _timed_mc_av_gcv(
                method,
                cfg,
                n_pilot=n_pilot,
                repeats=timing_repeats,
                gcv_pilot_fit=gcv_pilot_fit if method == "GCV" else None,
            )
            profiles.append(
                {
                    "method": method,
                    "n_paths": int(n_paths),
                    "timing_repeats": int(max(1, timing_repeats)),
                    "timing_path_counts": "|".join(str(x) for x in timing_path_counts),
                    **timed,
                    "end_to_end_runtime_per_observation_s": timed["end_to_end_pricing_runtime_s"] / max(1, int(n_paths)),
                }
            )
    return profiles


def _timed_ncv_end_to_end(network: _ShallowNet, cfg: ModelConfig, torch_mod, repeats: int) -> dict[str, float]:
    model = _torch_model_from_initial_network(torch_mod, network).eval()
    e_h = analytical_network_expectation(network)
    times_path_and_payoff: list[float] = []
    times_control: list[float] = []
    times_reduce: list[float] = []
    times_end_to_end: list[float] = []

    for _ in range(max(1, repeats)):
        rng = np.random.default_rng(cfg.seed)
        _maybe_cuda_sync(torch_mod)
        t0 = time.perf_counter()
        z = rng.standard_normal((cfg.n_paths, cfg.m))
        paths = simulate_paths(cfg, shocks=z)
        payoff = arithmetic_asian_call_payoff(paths, cfg)
        t_path = time.perf_counter()
        with torch_mod.no_grad():
            z_t = torch_mod.as_tensor(z, dtype=torch_mod.float64)
            h = model(z_t).detach().cpu().numpy()
        t_control = time.perf_counter()
        corrected = payoff - h + e_h
        _price = float(np.mean(corrected) * cfg.discount_factor)
        _maybe_cuda_sync(torch_mod)
        t1 = time.perf_counter()
        times_path_and_payoff.append(t_path - t0)
        times_control.append(t_control - t_path)
        times_reduce.append(t1 - t_control)
        times_end_to_end.append(t1 - t0)

    return {
        "path_and_payoff_runtime_s": float(statistics.median(times_path_and_payoff)),
        "control_evaluation_runtime_s": float(statistics.median(times_control)),
        "estimator_reduction_runtime_s": float(statistics.median(times_reduce)),
        "end_to_end_pricing_runtime_s": float(statistics.median(times_end_to_end)),
        "torch_tensor_conversion_inside_pricing_timing": True,
    }


def measure_end_to_end_pricing_runtime_profile(
    *,
    network: _ShallowNet,
    monitoring_dates: int,
    pricing_seed: int,
    n_paths: int,
    timing_repeats: int,
    torch_mod,
    progress_context: str,
) -> dict[str, Any]:
    print(f"[timing-method] {progress_context} method=NCV", flush=True)
    print(f"[timing-path-count] {progress_context} method=NCV n_paths={int(n_paths)}", flush=True)
    cfg = _make_reference_cfg(monitoring_dates=monitoring_dates, n_paths=n_paths, seed=pricing_seed)
    ncv_timed = _timed_ncv_end_to_end(network, cfg, torch_mod, repeats=timing_repeats)
    return {
        "method": "NCV",
        "n_paths": int(n_paths),
        "timing_repeats": int(max(1, timing_repeats)),
        "timing_path_counts": str(int(n_paths)),
        **ncv_timed,
        "end_to_end_runtime_per_observation_s": ncv_timed["end_to_end_pricing_runtime_s"] / max(1, cfg.n_paths),
    }


def assess_runtime_scaling(profiles: list[dict[str, Any]], method: str) -> dict[str, Any]:
    points = [
        (int(p["n_paths"]), float(p["end_to_end_pricing_runtime_s"]))
        for p in profiles
        if p.get("method") == method
    ]
    points = [pt for pt in points if pt[0] > 0 and math.isfinite(pt[1])]
    points.sort(key=lambda x: x[0])
    per_obs = [t / n for n, t in points]
    linearity_ratio = (max(per_obs) / min(per_obs)) if per_obs and min(per_obs) > 0 else float("inf")
    is_linear = bool(per_obs) and math.isfinite(linearity_ratio) and linearity_ratio <= 1.15
    basis_n = [n for n, _ in points]
    return {
        "runtime_projection_basis_n": basis_n,
        "runtime_projection_method": "linear_per_observation_rate" if is_linear else "piecewise_nearest_neighbor",
        "runtime_projection_is_empirical_or_projected": "projected_from_empirical_multi_n",
        "runtime_projection_linearity_ratio": float(linearity_ratio) if math.isfinite(linearity_ratio) else "NA",
        "runtime_projection_is_sufficiently_linear": is_linear,
    }


def project_runtime_at_n(profiles: list[dict[str, Any]], method: str, n_required: int) -> dict[str, Any]:
    scaling = assess_runtime_scaling(profiles, method)
    method_rows = [p for p in profiles if p.get("method") == method]
    points = [(int(p["n_paths"]), float(p["end_to_end_pricing_runtime_s"])) for p in method_rows]
    points = [pt for pt in points if pt[0] > 0 and math.isfinite(pt[1])]
    points.sort(key=lambda x: x[0])
    if not points:
        return {
            **scaling,
            "projected_runtime_s": float("nan"),
            "projected_per_observation_s": float("nan"),
            "runtime_at_required_n_is_empirical_or_projected": "projected",
        }

    empirical = {n for n, _ in points}
    if int(n_required) in empirical:
        runtime = next(t for n, t in points if n == int(n_required))
        projection_kind = "empirical"
    elif scaling["runtime_projection_is_sufficiently_linear"]:
        per_obs = statistics.median([t / n for n, t in points])
        runtime = per_obs * int(n_required)
        projection_kind = "projected"
    else:
        nearest_n, nearest_t = min(points, key=lambda x: abs(x[0] - int(n_required)))
        runtime = nearest_t * (int(n_required) / nearest_n)
        projection_kind = "projected"

    return {
        **scaling,
        "projected_runtime_s": float(runtime),
        "projected_per_observation_s": float(runtime) / max(1, int(n_required)),
        "runtime_at_required_n_is_empirical_or_projected": projection_kind,
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


def required_output_files() -> list[str]:
    return [
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


def validate_output_schema(output_dir: Path, *, include_report_in_presence_check: bool) -> dict[str, Any]:
    required = required_output_files()
    checked = required if include_report_in_presence_check else [x for x in required if x != "training_curve_validation_report.json"]
    exists = {name: (output_dir / name).exists() for name in required}
    errors = [f"missing required output file: {name}" for name in checked if not exists.get(name, False)]
    warnings: list[str] = []
    report_present_post_write = bool((output_dir / "training_curve_validation_report.json").exists())
    return {
        "required_files": required,
        "exists": exists,
        "all_present": all(exists.get(name, False) for name in required),
        "errors": errors,
        "warnings": warnings,
        "report_present_post_write": report_present_post_write,
        "passed": len(errors) == 0 and all(exists.get(name, False) for name in checked),
    }


def validate_numeric_content(
    per_replication_rows: list[dict[str, Any]],
    gcv_rows: list[dict[str, Any]],
    optimal_rows: list[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    required_non_negative = (
        "path_and_payoff_runtime_s",
        "control_evaluation_runtime_s",
        "estimator_reduction_runtime_s",
        "end_to_end_pricing_runtime_s",
        "ncv_end_to_end_runtime_per_observation_s",
        "cumulative_training_runtime_s",
        "training_data_generation_runtime_s",
        "optimizer_training_runtime_s",
        "validation_generation_and_evaluation_runtime_s",
    )
    for row in per_replication_rows:
        rid = f"rep={row.get('replication')},cp={row.get('checkpoint')},split={row.get('split')}"
        rv = row.get("residual_variance")
        if not isinstance(rv, (int, float)) or not math.isfinite(float(rv)):
            errors.append(f"{rid}: non-finite residual_variance")
        for name in required_non_negative:
            val = row.get(name)
            if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                errors.append(f"{rid}: non-finite {name}")
            elif float(val) < 0.0:
                errors.append(f"{rid}: negative {name}")
        if row.get("runtime_projection_is_sufficiently_linear") is False:
            warnings.append(f"{rid}: runtime projection not sufficiently linear; piecewise projection used")
    for row in gcv_rows:
        rid = f"rep={row.get('replication')},split={row.get('split')}"
        for name in ("gcv_residual_variance", "end_to_end_pricing_runtime_s", "end_to_end_runtime_per_observation_s"):
            val = row.get(name)
            if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                errors.append(f"{rid}: non-finite {name}")
    for row in optimal_rows:
        rid = f"rep={row.get('replication')},cp={row.get('checkpoint')},Q={row.get('Q')}"
        for name in ("required_pricing_observations", "setup_cost_s", "marginal_pricing_cost_s", "projected_total_cost_s"):
            val = row.get(name)
            if not isinstance(val, (int, float)) or not math.isfinite(float(val)):
                errors.append(f"{rid}: non-finite {name}")
    return errors, warnings


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


def _plot_summary_figure(
    output_dir: Path,
    checkpoints: list[int],
    per_rep_rows: list[dict[str, Any]],
    gcv_rows: list[dict[str, Any]],
    target_se: float,
    q_values: tuple[int, ...],
    fixed_checkpoint: int,
) -> None:
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

    gcv_val = [r for r in gcv_rows if r["split"] == "validation"]
    gcv_var = statistics.mean([float(r["gcv_residual_variance"]) for r in gcv_val if math.isfinite(float(r["gcv_residual_variance"]))]) if gcv_val else float("nan")
    gcv_rate = statistics.mean([float(r.get("end_to_end_runtime_per_observation_s", float("nan"))) for r in gcv_val if math.isfinite(float(r.get("end_to_end_runtime_per_observation_s", float("nan"))))]) if gcv_val else float("nan")
    for q in q_values:
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
                        float(r.get("ncv_end_to_end_runtime_per_observation_s", r["inference_runtime_per_observation_median_s"])),
                        q,
                    )
                )
            ys.append(statistics.mean(vals) if vals else float("nan"))
        axes[2].plot(checkpoints, ys, label=f"Q={q}")
        if math.isfinite(gcv_var) and math.isfinite(gcv_rate):
            gcv_n = compute_required_paths(gcv_var, target_se)
            gcv_cost = compute_total_cost(0.0, gcv_n, gcv_rate, q)
            axes[2].axhline(gcv_cost, linestyle="--", linewidth=0.8, alpha=0.6)
    axes[2].axvline(fixed_checkpoint, linestyle=":", color="black", linewidth=1.0, label=f"fixed={fixed_checkpoint}")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Projected total cost (end-to-end)")
    axes[2].set_yscale("log")
    axes[2].legend()

    fig.tight_layout()
    fig.savefig(output_dir / "ncv_training_curve_summary.png", dpi=200)
    plt.close(fig)


def run_training_curve_experiment(config: TrainingCurveConfig) -> Path:
    validate_checkpoints(config.checkpoints)
    if not config.timing_path_counts:
        raise ValueError("timing_path_counts cannot be empty")
    if any(int(n) <= 0 for n in config.timing_path_counts):
        raise ValueError("timing_path_counts must be strictly positive")
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
        print(f"[replication] {rep + 1}/{config.replications}", flush=True)
        seeds = replication_seeds(config.base_seed, rep)
        t_data = time.perf_counter()
        train_split = simulate_split_dataset(config.monitoring_dates, config.train_paths, seeds["train"])
        val_split = simulate_split_dataset(config.monitoring_dates, config.validation_paths, seeds["validation"])
        test_split = simulate_split_dataset(config.monitoring_dates, config.test_paths, seeds["test"])
        pilot_split = simulate_split_dataset(config.monitoring_dates, config.pilot_paths, seeds["gcv_pilot"])
        data_generation_runtime_s = time.perf_counter() - t_data

        gcv_bench = compute_gcv_benchmark(val_split, test_split, pilot_split, n_reporting=config.pricing_observations_for_reporting)
        gcv_timing_cfg = _make_reference_cfg(
            monitoring_dates=config.monitoring_dates,
            n_paths=max(2, int(config.pilot_paths)),
            seed=seeds["gcv_pilot"] + 991,
        )
        gcv_pilot_fit = _fit_gcv_pilot_once(gcv_timing_cfg, config.pilot_paths)
        baseline_runtime_profiles = build_runtime_profiles(
            methods=("MC", "AV", "GCV"),
            monitoring_dates=config.monitoring_dates,
            seed_base=seeds["test"] + 90_000,
            timing_path_counts=config.timing_path_counts,
            timing_repeats=config.timing_repeats,
            n_pilot=config.pilot_paths,
            gcv_pilot_fit=gcv_pilot_fit,
            progress_context=f"rep={rep}",
        )
        gcv_reporting_projection = project_runtime_at_n(
            baseline_runtime_profiles,
            "GCV",
            config.pricing_observations_for_reporting,
        )
        for gr in gcv_bench:
            gcv_rows.append(
                {
                    "replication": rep,
                    **gr,
                    "path_and_payoff_runtime_s": gcv_reporting_projection["projected_runtime_s"],
                    "control_evaluation_runtime_s": 0.0,
                    "estimator_reduction_runtime_s": 0.0,
                    "end_to_end_pricing_runtime_s": gcv_reporting_projection["projected_runtime_s"],
                    "end_to_end_runtime_per_observation_s": gcv_reporting_projection["projected_per_observation_s"],
                    "gcv_pilot_runtime_s_end_to_end": gcv_pilot_fit["pilot_runtime_s"],
                    "gcv_pilot_runtime_s": gcv_pilot_fit["pilot_runtime_s"],
                    "timing_path_counts": "|".join(str(x) for x in config.timing_path_counts),
                    "timing_repeats": int(config.timing_repeats),
                    "runtime_projection_basis_n": "|".join(str(x) for x in gcv_reporting_projection["runtime_projection_basis_n"]),
                    "runtime_projection_method": gcv_reporting_projection["runtime_projection_method"],
                    "runtime_projection_linearity_ratio": gcv_reporting_projection["runtime_projection_linearity_ratio"],
                    "runtime_projection_is_sufficiently_linear": gcv_reporting_projection["runtime_projection_is_sufficiently_linear"],
                    "runtime_projection_is_empirical_or_projected": gcv_reporting_projection["runtime_projection_is_empirical_or_projected"],
                    "runtime_at_required_n_is_empirical_or_projected": gcv_reporting_projection["runtime_at_required_n_is_empirical_or_projected"],
                }
            )

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
        eval_no_grad_used = True
        snapshots: dict[int, dict[str, Any]] = {}
        checkpoints = list(config.checkpoints)
        cumulative_training_s = 0.0
        previous_checkpoint_s = 0.0
        validation_eval_runtime_cumulative_s = 0.0

        def evaluate_checkpoint(epoch: int, cumulative_runtime_s: float) -> None:
            nonlocal eval_no_grad_used
            nonlocal validation_eval_runtime_cumulative_s
            print(f"[checkpoint] rep={rep} checkpoint={epoch}", flush=True)
            t_eval = time.perf_counter()
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
            validation_eval_runtime_cumulative_s += time.perf_counter() - t_eval

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
                "validation_generation_and_evaluation_runtime_s": validation_eval_runtime_cumulative_s,
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
            cumulative_training_s = time.perf_counter() - t_train_start
            if epoch in checkpoint_set:
                evaluate_checkpoint(epoch, cumulative_training_s)

        missing_checkpoints = [cp for cp in checkpoints if cp not in snapshots]
        if missing_checkpoints:
            raise RuntimeError(f"Missing checkpoint snapshots for epochs: {missing_checkpoints}")

        val_rows_for_selection = [
            {"checkpoint": cp, "residual_variance": snapshots[cp]["val_diag"]["residual_variance"]}
            for cp in checkpoints
            if math.isfinite(float(snapshots[cp]["val_diag"]["residual_variance"]))
        ]
        if val_rows_for_selection:
            selected_runtime_checkpoint = int(min(val_rows_for_selection, key=lambda x: float(x["residual_variance"]))["checkpoint"])
        else:
            selected_runtime_checkpoint = int(checkpoints[0])
        selected_snapshot = snapshots[selected_runtime_checkpoint]
        ncv_runtime_profiles = [
            measure_end_to_end_pricing_runtime_profile(
                network=selected_snapshot["network"],
                monitoring_dates=config.monitoring_dates,
                pricing_seed=seeds["test"] + selected_runtime_checkpoint * 10_000 + 101 + i,
                n_paths=int(n_paths),
                timing_repeats=config.timing_repeats,
                torch_mod=torch,
                progress_context=f"rep={rep},checkpoint={selected_runtime_checkpoint}",
            )
            for i, n_paths in enumerate(config.timing_path_counts)
        ]
        for profile in ncv_runtime_profiles:
            profile["timing_path_counts"] = "|".join(str(x) for x in config.timing_path_counts)
        runtime_tensor_conversion_flag = bool(
            any(bool(p.get("torch_tensor_conversion_inside_pricing_timing", False)) for p in ncv_runtime_profiles)
        )
        ncv_reporting_projection = project_runtime_at_n(
            ncv_runtime_profiles,
            "NCV",
            config.pricing_observations_for_reporting,
        )

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
                    "objective_mean_loss": snap["validation_loss"] if split_name == "validation" else snap["test_loss"],
                    "checkpoint_selection_metric": "validation_residual_variance",
                    "centered_residual_variance": diag["residual_variance"],
                    "timed_repeats": snap["inference"]["timed_repeats"],
                    "inference_runtime_median_s": snap["inference"]["inference_runtime_median_s"],
                    "inference_runtime_mean_s": snap["inference"]["inference_runtime_mean_s"],
                    "inference_runtime_std_s": snap["inference"]["inference_runtime_std_s"],
                    "inference_runtime_per_observation_median_s": snap["inference"]["inference_runtime_per_observation_median_s"],
                    "evaluation_no_grad": eval_no_grad_used,
                    "epoch_zero_training_runtime_zero": snapshots[0]["epoch"] == 0,
                    "training_data_generation_runtime_s": data_generation_runtime_s,
                    "optimizer_training_runtime_s": 0.0 if cp == 0 else incremental,
                    "validation_generation_and_evaluation_runtime_s": snap["validation_generation_and_evaluation_runtime_s"],
                    "path_and_payoff_runtime_s": ncv_reporting_projection["projected_runtime_s"],
                    "control_evaluation_runtime_s": 0.0,
                    "estimator_reduction_runtime_s": 0.0,
                    "end_to_end_pricing_runtime_s": ncv_reporting_projection["projected_runtime_s"],
                    "ncv_end_to_end_runtime_per_observation_s": ncv_reporting_projection["projected_per_observation_s"],
                    "timing_path_counts": "|".join(str(x) for x in config.timing_path_counts),
                    "timing_repeats": int(config.timing_repeats),
                    "runtime_checkpoint_for_pricing_timing": selected_runtime_checkpoint,
                    "runtime_projection_basis_n": "|".join(str(x) for x in ncv_reporting_projection["runtime_projection_basis_n"]),
                    "runtime_projection_method": ncv_reporting_projection["runtime_projection_method"],
                    "runtime_projection_is_empirical_or_projected": ncv_reporting_projection["runtime_projection_is_empirical_or_projected"],
                    "runtime_projection_linearity_ratio": ncv_reporting_projection["runtime_projection_linearity_ratio"],
                    "runtime_projection_is_sufficiently_linear": ncv_reporting_projection["runtime_projection_is_sufficiently_linear"],
                    "runtime_at_required_n_is_empirical_or_projected": ncv_reporting_projection["runtime_at_required_n_is_empirical_or_projected"],
                    "torch_tensor_conversion_inside_pricing_timing": runtime_tensor_conversion_flag,
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
        ncv_runtime_projection_cache: dict[int, dict[str, Any]] = {}
        gcv_runtime_projection_cache: dict[int, dict[str, Any]] = {}

        def _project_cached(cache: dict[int, dict[str, Any]], profiles: list[dict[str, Any]], method: str, n_required: int) -> dict[str, Any]:
            n_key = int(n_required)
            if n_key not in cache:
                cache[n_key] = project_runtime_at_n(profiles, method, n_key)
            return cache[n_key]

        se_targets = [
            ("gcv_matched_at_reporting_n", float(gcv_validation["gcv_standard_error_at_reporting_n"])),
            ("fixed_se", 0.001),
        ]
        for target_definition, target_se in se_targets:
            for q in config.q_values:
                per_checkpoint_costs = []
                for r in validation_rows:
                    req_n = compute_required_paths(float(r["residual_variance"]), float(target_se))
                    runtime_req = _project_cached(ncv_runtime_projection_cache, ncv_runtime_profiles, "NCV", req_n)
                    setup_cost = float(r["cumulative_training_runtime_s"])
                    marginal_pricing = float(runtime_req["projected_runtime_s"])
                    total_cost = setup_cost + int(q) * marginal_pricing
                    per_checkpoint_costs.append((r, req_n, setup_cost, marginal_pricing, total_cost, runtime_req))
                best = min(per_checkpoint_costs, key=lambda x: x[4])
                best_row, best_n, best_setup, best_marginal, best_cost, best_runtime_req = best
                matched_test_row = next(rr for rr in test_rows if int(rr["checkpoint"]) == int(best_row["checkpoint"]))

                test_required_n = compute_required_paths(float(matched_test_row["residual_variance"]), float(target_se))
                test_runtime_req = _project_cached(ncv_runtime_projection_cache, ncv_runtime_profiles, "NCV", test_required_n)
                test_setup = float(matched_test_row["cumulative_training_runtime_s"])
                test_marginal = float(test_runtime_req["projected_runtime_s"])
                test_total_cost = test_setup + int(q) * test_marginal

                gcv_req = compute_required_paths(float(gcv_validation["gcv_residual_variance"]), float(target_se))
                gcv_runtime_req = _project_cached(gcv_runtime_projection_cache, baseline_runtime_profiles, "GCV", gcv_req)
                gcv_setup = float(gcv_validation["gcv_pilot_runtime_s"])
                gcv_marginal = float(gcv_runtime_req["projected_runtime_s"])
                gcv_cost = gcv_setup + int(q) * gcv_marginal
                gcv_test_req = compute_required_paths(float(gcv_test["gcv_residual_variance"]), float(target_se))
                gcv_test_runtime_req = _project_cached(gcv_runtime_projection_cache, baseline_runtime_profiles, "GCV", gcv_test_req)
                gcv_test_setup = float(gcv_test["gcv_pilot_runtime_s"])
                gcv_test_marginal = float(gcv_test_runtime_req["projected_runtime_s"])
                gcv_test_cost = gcv_test_setup + int(q) * gcv_test_marginal
                optimal_rows.append(
                    {
                        "replication": rep,
                        "selection_split": "validation",
                        "cost_scope": "end_to_end",
                        "target_definition": target_definition,
                        "target_se": float(target_se),
                        "Q": int(q),
                        "checkpoint": int(best_row["checkpoint"]),
                        "ncv_epoch_source": "training_curve_validation_tuning",
                        "fixed_ncv_epoch_for_stage8": TRAINING_CURVE_SELECTED_NCV_EPOCH,
                        "setup_cost_s": best_setup,
                        "setup_reuse_assumption": "NCV training counted once then reused Q times",
                        "runtime_projection_basis_n": best_row["runtime_projection_basis_n"],
                        "runtime_projection_method": best_row["runtime_projection_method"],
                        "runtime_projection_is_empirical_or_projected": best_runtime_req["runtime_projection_is_empirical_or_projected"],
                        "runtime_at_required_n_is_empirical_or_projected": best_runtime_req["runtime_at_required_n_is_empirical_or_projected"],
                        "runtime_projection_linearity_ratio": best_runtime_req["runtime_projection_linearity_ratio"],
                        "runtime_projection_is_sufficiently_linear": best_runtime_req["runtime_projection_is_sufficiently_linear"],
                        "timing_path_counts": "|".join(str(x) for x in config.timing_path_counts),
                        "timing_repeats": int(config.timing_repeats),
                        "cumulative_training_runtime_s": best_setup,
                        "validation_residual_variance": float(best_row["residual_variance"]),
                        "test_residual_variance": float(matched_test_row["residual_variance"]),
                        "required_pricing_observations": int(best_n),
                        "marginal_pricing_cost_s": float(best_marginal),
                        "projected_total_cost_s": float(best_cost),
                        "projected_total_cost_validation": float(best_cost),
                        "projected_total_cost_test": float(test_total_cost),
                        "required_pricing_observations_test": int(test_required_n),
                        "marginal_pricing_cost_test_s": float(test_marginal),
                        "gcv_setup_cost_s": gcv_setup,
                        "gcv_setup_reuse_assumption": "GCV pilot counted once and reused across Q for same target",
                        "gcv_required_pricing_observations": int(gcv_req),
                        "gcv_marginal_pricing_cost_s": float(gcv_marginal),
                        "gcv_runtime_at_required_n_is_empirical_or_projected": gcv_runtime_req["runtime_at_required_n_is_empirical_or_projected"],
                        "gcv_projected_total_cost_validation": float(gcv_cost),
                        "gcv_required_pricing_observations_test": int(gcv_test_req),
                        "gcv_marginal_pricing_cost_test_s": float(gcv_test_marginal),
                        "gcv_runtime_at_required_n_test_is_empirical_or_projected": gcv_test_runtime_req["runtime_at_required_n_is_empirical_or_projected"],
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

    _plot_summary_figure(
        out_dir,
        list(config.checkpoints),
        per_replication_rows,
        gcv_rows,
        target_se=float(config.se_targets[0]) if config.se_targets else 0.001,
        q_values=config.q_values,
        fixed_checkpoint=TRAINING_CURVE_SELECTED_NCV_EPOCH if TRAINING_CURVE_SELECTED_NCV_EPOCH in config.checkpoints else int(config.checkpoints[0]),
    )

    (out_dir / "TRAINING_CURVE_HANDOVER.md").write_text(build_handover_text(config, facts), encoding="utf-8")

    provisional_report = validate_output_schema(out_dir, include_report_in_presence_check=False)
    provisional_report["report_write_phase"] = "preliminary_without_self_check"
    _write_json(out_dir / "training_curve_validation_report.json", provisional_report)

    validation_report = validate_output_schema(out_dir, include_report_in_presence_check=True)
    numeric_errors, numeric_warnings = validate_numeric_content(per_replication_rows, gcv_rows, optimal_rows)
    validation_report["errors"].extend(numeric_errors)
    validation_report["warnings"].extend(numeric_warnings)
    validation_report["all_present"] = all(validation_report["exists"].values())
    validation_report["passed"] = validation_report["all_present"] and len(validation_report["errors"]) == 0
    validation_report.update(
        {
            "report_write_phase": "final_with_post_write_self_check",
            "checkpoint_grid": list(config.checkpoints),
            "checkpoint_grid_starts_at_zero": config.checkpoints[0] == 0,
            "checkpoint_grid_strictly_increasing": all(config.checkpoints[i] < config.checkpoints[i + 1] for i in range(len(config.checkpoints) - 1)),
            "dissertation_monitoring_dates_252": config.monitoring_dates == 252 if config.profile == "dissertation" else True,
            "n_errors": len(validation_report["errors"]),
            "n_warnings": len(validation_report["warnings"]),
        }
    )
    _write_json(out_dir / "training_curve_validation_report.json", validation_report)

    return out_dir
