from __future__ import annotations

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

from asian_options.analytical import geometric_asian_call_price
from asian_options.config import ModelConfig, collect_environment_metadata, seed_everything
from asian_options.contracts import CONTRACT_GRID, CONTRACT_IDS, REFERENCE_ID
from asian_options.frozen_transfer import compute_network_hash, train_reference_network
from asian_options.neural_cv import (
    _ShallowNet,
    analytical_network_expectation,
    analytical_network_expectation_conditional,
    build_network,
)
from asian_options.ncv_training_curve import _network_from_torch_model, _torch_model_from_initial_network
from asian_options.payoffs import arithmetic_asian_call_payoff, geometric_asian_call_payoff
from asian_options.simulate_gbm import simulate_paths


PARAM_BOUNDS = {
    "log_k_ratio_min": math.log(0.75),
    "log_k_ratio_max": math.log(1.25),
    "sigma_min": 0.08,
    "sigma_max": 0.55,
    "t_min": 0.4,
    "t_max": 2.2,
}

SHOCK_DIM = 12
PARAM_DIM = 3
INPUT_DIM = SHOCK_DIM + PARAM_DIM
HIDDEN_WIDTH = 32
PCNCV_PARAM_COUNT = HIDDEN_WIDTH * INPUT_DIM + HIDDEN_WIDTH + HIDDEN_WIDTH + 1

SEED_OFFSETS = {
    "training_parameters": 1_000,
    "training_shocks": 2_000,
    "validation_contracts": 3_000,
    "validation_shocks": 4_000,
    "gcv_pilot": 5_000,
    "conditional_ncv_pilot": 6_000,
    "frozen_ncv_pilot": 7_000,
    "final_pricing_shocks": 8_000,
    "reference_training": 9_000,
}


@dataclass(frozen=True)
class PCNCVConfig:
    profile: str
    base_seed: int
    replications: int
    n_training: int
    n_validation: int
    n_pilot: int
    n_pricing: int
    max_epochs: int
    checkpoints: tuple[int, ...]
    hidden_width: int
    learning_rate: float
    batch_size: int
    monitoring_dates: int
    output_dir: str
    n_aux_validation_contracts: int
    frozen_reference_epochs: int


@dataclass(frozen=True)
class _CheckpointSnapshot:
    epoch: int
    network: _ShallowNet
    train_mse: float
    optimizer_cumulative_runtime_s: float


def profile_config(profile: str, output_dir: str, base_seed: int = 42) -> PCNCVConfig:
    p = profile.lower()
    if p == "smoke":
        return PCNCVConfig(
            profile="smoke",
            base_seed=base_seed,
            replications=2,
            n_training=100,
            n_validation=200,
            n_pilot=50,
            n_pricing=200,
            max_epochs=10,
            checkpoints=(0, 2, 5, 10),
            hidden_width=HIDDEN_WIDTH,
            learning_rate=1e-2,
            batch_size=64,
            monitoring_dates=12,
            output_dir=output_dir,
            n_aux_validation_contracts=3,
            frozen_reference_epochs=10,
        )
    if p == "dissertation":
        return PCNCVConfig(
            profile="dissertation",
            base_seed=base_seed,
            replications=10,
            n_training=20_000,
            n_validation=12_000,
            n_pilot=1_000,
            n_pricing=50_000,
            max_epochs=1_000,
            checkpoints=(0, 10, 25, 50, 100, 200, 500, 1000),
            hidden_width=HIDDEN_WIDTH,
            learning_rate=1e-2,
            batch_size=256,
            monitoring_dates=12,
            output_dir=output_dir,
            n_aux_validation_contracts=8,
            frozen_reference_epochs=25,
        )
    raise ValueError(f"Unknown profile: {profile}")


def transform_contract_parameters(K: float, sigma: float, T: float, S0: float = 100.0) -> np.ndarray:
    log_k = math.log(float(K) / float(S0))
    u_k = 2.0 * (log_k - PARAM_BOUNDS["log_k_ratio_min"]) / (
        PARAM_BOUNDS["log_k_ratio_max"] - PARAM_BOUNDS["log_k_ratio_min"]
    ) - 1.0
    u_sigma = 2.0 * (float(sigma) - PARAM_BOUNDS["sigma_min"]) / (
        PARAM_BOUNDS["sigma_max"] - PARAM_BOUNDS["sigma_min"]
    ) - 1.0
    u_t = 2.0 * (float(T) - PARAM_BOUNDS["t_min"]) / (
        PARAM_BOUNDS["t_max"] - PARAM_BOUNDS["t_min"]
    ) - 1.0
    return np.asarray([u_k, u_sigma, u_t], dtype=np.float64)


def build_parameter_inputs(z: np.ndarray, u_params: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    u_params = np.asarray(u_params, dtype=np.float64)
    if z.ndim != 2 or z.shape[1] != SHOCK_DIM:
        raise ValueError(f"z must be shape (n, {SHOCK_DIM}), got {z.shape}")
    if u_params.ndim != 2 or u_params.shape[1] != PARAM_DIM or u_params.shape[0] != z.shape[0]:
        raise ValueError(f"u_params must be shape ({z.shape[0]}, {PARAM_DIM}), got {u_params.shape}")
    return np.hstack([z, u_params])


def trainable_parameter_count(input_dim: int = INPUT_DIM, hidden_width: int = HIDDEN_WIDTH) -> int:
    return hidden_width * input_dim + hidden_width + hidden_width + 1


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


def _student_t_log_ci(values: list[float]) -> tuple[float | None, float | None, float | None]:
    vals = [v for v in values if math.isfinite(v) and v > 0.0]
    if not vals:
        return None, None, None
    logs = [math.log(v) for v in vals]
    m = statistics.mean(logs)
    if len(logs) == 1:
        g = math.exp(m)
        return g, g, g
    s = statistics.stdev(logs)
    t_crit = float(_t_dist.ppf(0.975, df=len(logs) - 1))
    half = t_crit * s / math.sqrt(len(logs))
    return math.exp(m), math.exp(m - half), math.exp(m + half)


def _make_cfg(contract_id: str, n_paths: int, seed: int, monitoring_dates: int = 12) -> ModelConfig:
    cfg = make_contract_cfg(contract_id, n_paths=n_paths, seed=seed)
    if cfg.m != monitoring_dates:
        cfg = dataclasses.replace(cfg, m=monitoring_dates)
    return cfg


def _simulate_contract_payoffs_from_z(
    *,
    K: float,
    sigma: float,
    T: float,
    z: np.ndarray,
    S0: float = 100.0,
    r: float = 0.05,
) -> np.ndarray:
    cfg = ModelConfig(
        S0=S0,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
        m=SHOCK_DIM,
        n_paths=z.shape[0],
        seed=0,
    )
    paths = simulate_paths(cfg, shocks=z)
    return arithmetic_asian_call_payoff(paths, cfg)


def _sample_training_parameters(n: int, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    log_k = rng.uniform(PARAM_BOUNDS["log_k_ratio_min"], PARAM_BOUNDS["log_k_ratio_max"], size=n)
    sigma = rng.uniform(PARAM_BOUNDS["sigma_min"], PARAM_BOUNDS["sigma_max"], size=n)
    T = rng.uniform(PARAM_BOUNDS["t_min"], PARAM_BOUNDS["t_max"], size=n)
    K = 100.0 * np.exp(log_k)
    return K.astype(np.float64), sigma.astype(np.float64), T.astype(np.float64)


def generate_parameter_conditioned_training_data(
    *,
    n_training: int,
    training_parameter_seed: int,
    training_shock_seed: int,
) -> dict[str, Any]:
    K, sigma, T = _sample_training_parameters(n_training, training_parameter_seed)
    rng = np.random.default_rng(training_shock_seed)
    z = rng.standard_normal((n_training, SHOCK_DIM))

    payoffs = np.empty(n_training, dtype=np.float64)
    u_params = np.empty((n_training, PARAM_DIM), dtype=np.float64)
    for i in range(n_training):
        payoffs[i] = _simulate_contract_payoffs_from_z(
            K=float(K[i]),
            sigma=float(sigma[i]),
            T=float(T[i]),
            z=z[i : i + 1, :],
        )[0]
        u_params[i, :] = transform_contract_parameters(float(K[i]), float(sigma[i]), float(T[i]))

    X = build_parameter_inputs(z, u_params)
    return {
        "X_train": X,
        "y_train": payoffs,
        "K": K,
        "sigma": sigma,
        "T": T,
        "u_params": u_params,
        "z": z,
    }


def _sample_aux_validation_contracts(
    *,
    n_contracts: int,
    seed: int,
) -> list[dict[str, float]]:
    final_grid = {(float(K), float(s), float(T)) for (K, s, T) in CONTRACT_GRID.values()}
    K, sigma, T = _sample_training_parameters(max(n_contracts * 3, n_contracts), seed)
    out: list[dict[str, float]] = []
    for k, s, t in zip(K, sigma, T):
        key = (round(float(k), 10), round(float(s), 10), round(float(t), 10))
        grid_keys = {(round(a, 10), round(b, 10), round(c, 10)) for (a, b, c) in final_grid}
        if key in grid_keys:
            continue
        out.append({"K": float(k), "sigma": float(s), "T": float(t)})
        if len(out) >= n_contracts:
            break
    if len(out) < n_contracts:
        raise RuntimeError("failed to sample enough auxiliary validation contracts")
    return out


def _replication_seed_streams(base_seed: int, replication: int) -> dict[str, int]:
    rep_offset = replication * 100_000
    base = int(base_seed) + rep_offset
    return {k: base + v for k, v in SEED_OFFSETS.items()}


def build_seed_manifest(base_seed: int, replications: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rep in range(replications):
        seeds = _replication_seed_streams(base_seed, rep)
        for stream, seed in seeds.items():
            rows.append({"replication": rep, "stream": stream, "seed": int(seed)})
        for idx, cid in enumerate(CONTRACT_IDS):
            rows.append({
                "replication": rep,
                "stream": "gcv_pilot_contract",
                "contract_id": cid,
                "seed": int(seeds["gcv_pilot"] + idx * 100),
            })
            rows.append({
                "replication": rep,
                "stream": "conditional_ncv_pilot_contract",
                "contract_id": cid,
                "seed": int(seeds["conditional_ncv_pilot"] + idx * 100),
            })
            rows.append({
                "replication": rep,
                "stream": "frozen_ncv_pilot_contract",
                "contract_id": cid,
                "seed": int(seeds["frozen_ncv_pilot"] + idx * 100),
            })
            rows.append({
                "replication": rep,
                "stream": "final_pricing_shocks_contract",
                "contract_id": cid,
                "seed": int(seeds["final_pricing_shocks"] + idx * 100),
            })
    return rows


def _train_parameter_conditioned_checkpoints(
    *,
    dataset: dict[str, Any],
    seed: int,
    checkpoints: tuple[int, ...],
    hidden_width: int,
    learning_rate: float,
    batch_size: int,
) -> tuple[dict[int, _CheckpointSnapshot], float]:
    torch = __import__("torch")
    seed_everything(seed)

    cfg_seed = ModelConfig(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, m=SHOCK_DIM, n_paths=2, seed=seed)
    network = build_network(cfg_seed, hidden_width=hidden_width, input_dim=INPUT_DIM)
    model = _torch_model_from_initial_network(torch, network)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    X = torch.tensor(dataset["X_train"], dtype=torch.float64)
    y = torch.tensor(dataset["y_train"], dtype=torch.float64)

    checkpoints_set = {int(cp) for cp in checkpoints}
    max_cp = int(max(checkpoints))

    snapshots: dict[int, _CheckpointSnapshot] = {}

    def _capture(epoch: int, cumulative: float) -> None:
        model.eval()
        with torch.no_grad():
            train_mse = float(loss_fn(model(X), y).item())
        snapshots[int(epoch)] = _CheckpointSnapshot(
            epoch=int(epoch),
            network=_network_from_torch_model(model),
            train_mse=train_mse,
            optimizer_cumulative_runtime_s=float(cumulative),
        )

    _capture(0, 0.0)
    bs = min(int(batch_size), int(len(X)))
    t0 = time.perf_counter()
    for epoch in range(1, max_cp + 1):
        model.train()
        perm = torch.randperm(len(X))
        for start in range(0, len(X), bs):
            idx = perm[start : start + bs]
            xb = X[idx]
            yb = y[idx]
            pred = model(xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if epoch in checkpoints_set:
            _capture(epoch, time.perf_counter() - t0)

    missing = [cp for cp in checkpoints if int(cp) not in snapshots]
    if missing:
        raise RuntimeError(f"missing checkpoints: {missing}")
    return snapshots, float(time.perf_counter() - t0)


def _checkpoint_selection(
    *,
    checkpoints: tuple[int, ...],
    snapshots: dict[int, _CheckpointSnapshot],
    aux_contracts: list[dict[str, float]],
    validation_n: int,
    validation_shock_seed: int,
) -> tuple[int, list[dict[str, Any]]]:
    rng = np.random.default_rng(validation_shock_seed)
    history: list[dict[str, Any]] = []

    by_checkpoint: dict[int, list[float]] = {int(cp): [] for cp in checkpoints}

    for cidx, c in enumerate(aux_contracts):
        z_val = rng.standard_normal((validation_n, SHOCK_DIM))
        y_val = _simulate_contract_payoffs_from_z(K=c["K"], sigma=c["sigma"], T=c["T"], z=z_val)
        var_y = _sample_var(y_val)
        p = transform_contract_parameters(c["K"], c["sigma"], c["T"])
        x_val = build_parameter_inputs(z_val, np.repeat(p.reshape(1, -1), validation_n, axis=0))

        for cp in checkpoints:
            snap = snapshots[int(cp)]
            h = snap.network.forward(x_val)
            e_h = analytical_network_expectation_conditional(snap.network, p, shock_dim=SHOCK_DIM)
            residual_var = _sample_var(y_val - h + e_h)
            ratio = residual_var / var_y if math.isfinite(var_y) and var_y > 0.0 else float("nan")
            if math.isfinite(ratio) and ratio > 0.0:
                by_checkpoint[int(cp)].append(float(ratio))
            history.append(
                {
                    "checkpoint": int(cp),
                    "aux_contract_index": int(cidx),
                    "K": c["K"],
                    "sigma": c["sigma"],
                    "T": c["T"],
                    "validation_payoff_variance": var_y,
                    "validation_residual_variance": residual_var,
                    "normalized_residual_variance": ratio,
                    "train_mse": snap.train_mse,
                    "optimizer_cumulative_training_runtime_s": snap.optimizer_cumulative_runtime_s,
                }
            )

    scores: list[tuple[int, float]] = []
    for cp in checkpoints:
        vals = by_checkpoint[int(cp)]
        if not vals:
            continue
        score = float(math.exp(statistics.mean([math.log(v) for v in vals])))
        scores.append((int(cp), score))
    if not scores:
        return int(checkpoints[0]), history
    selected = min(scores, key=lambda x: x[1])[0]
    return selected, history


def _mc_row(payoffs: np.ndarray, runtime_s: float) -> dict[str, Any]:
    obs_var = _sample_var(payoffs)
    n = int(payoffs.shape[0])
    est_var = obs_var / n if math.isfinite(obs_var) else float("nan")
    se = math.sqrt(est_var) if math.isfinite(est_var) and est_var >= 0.0 else float("nan")
    m = float(np.mean(payoffs))
    return {
        "price": m,
        "observation_variance": obs_var,
        "estimator_variance": est_var,
        "std_error": se,
        "ci_lower": m - 1.96 * se if math.isfinite(se) else float("nan"),
        "ci_upper": m + 1.96 * se if math.isfinite(se) else float("nan"),
        "pricing_observations": n,
        "pricing_simulated_paths": n,
        "pricing_runtime_s": runtime_s,
    }


def _gcv_fit(cfg: ModelConfig, z_pilot: np.ndarray) -> tuple[float, float, float, float]:
    paths = simulate_paths(cfg, shocks=z_pilot)
    x = arithmetic_asian_call_payoff(paths, cfg)
    g = geometric_asian_call_payoff(paths, cfg)
    var_g = _sample_var(g)
    cov_xg = float(np.cov(x, g, ddof=1)[0, 1]) if x.size > 1 else float("nan")
    beta = cov_xg / var_g if math.isfinite(var_g) and var_g > 0.0 else 0.0
    corr = float(np.corrcoef(x, g)[0, 1]) if _sample_var(x) > 0 and var_g > 0 else float("nan")
    return float(beta), float(geometric_asian_call_price(cfg)), float(corr), float(var_g)


def _evaluate_frozen_methods(
    *,
    frozen_network: _ShallowNet,
    frozen_hash: str,
    e_h0: float,
    cfg: ModelConfig,
    z_pricing: np.ndarray,
    z_pilot: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    hash_before = compute_network_hash(frozen_network)
    if hash_before != frozen_hash:
        raise RuntimeError("frozen network hash mismatch before evaluation")

    t0 = time.perf_counter()
    paths_price = simulate_paths(cfg, shocks=z_pricing)
    y_price = arithmetic_asian_call_payoff(paths_price, cfg)
    h_price = frozen_network.forward(z_pricing)
    c0_price = h_price - e_h0
    corrected_b1 = y_price - c0_price
    beta1_runtime = time.perf_counter() - t0

    beta1_row = _mc_row(corrected_b1, beta1_runtime)
    beta1_row.update(
        {
            "beta": 1.0,
            "payoff_control_correlation": float(np.corrcoef(y_price, c0_price)[0, 1]) if _sample_var(c0_price) > 0 else float("nan"),
            "pilot_paths": 0,
            "pilot_runtime_s": 0.0,
        }
    )

    t1 = time.perf_counter()
    paths_pilot = simulate_paths(dataclasses.replace(cfg, n_paths=z_pilot.shape[0]), shocks=z_pilot)
    y_pilot = arithmetic_asian_call_payoff(paths_pilot, cfg)
    h_pilot = frozen_network.forward(z_pilot)
    c0_pilot = h_pilot - e_h0
    var_c0 = _sample_var(c0_pilot)
    cov = float(np.cov(y_pilot, c0_pilot, ddof=1)[0, 1]) if y_pilot.size > 1 else float("nan")
    beta = cov / var_c0 if math.isfinite(var_c0) and var_c0 > 1e-12 else 1.0
    corrected_b = y_price - beta * c0_price
    beta_runtime = time.perf_counter() - t1

    beta_row = _mc_row(corrected_b, beta_runtime)
    beta_row.update(
        {
            "beta": float(beta),
            "payoff_control_correlation": float(np.corrcoef(y_pilot, c0_pilot)[0, 1]) if _sample_var(c0_pilot) > 0 else float("nan"),
            "pilot_paths": int(z_pilot.shape[0]),
            "pilot_runtime_s": beta_runtime,
        }
    )

    hash_after = compute_network_hash(frozen_network)
    if hash_after != frozen_hash:
        raise RuntimeError("frozen network hash mismatch after evaluation")
    return beta1_row, beta_row


def _evaluate_pcncv_methods(
    *,
    pcncv_network: _ShallowNet,
    pcncv_hash: str,
    cfg: ModelConfig,
    z_pricing: np.ndarray,
    z_pilot: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    hash_before = compute_network_hash(pcncv_network)
    if hash_before != pcncv_hash:
        raise RuntimeError("pcncv network hash mismatch before evaluation")

    p = transform_contract_parameters(cfg.K, cfg.sigma, cfg.T)
    X_price = build_parameter_inputs(z_pricing, np.repeat(p.reshape(1, -1), z_pricing.shape[0], axis=0))
    paths_price = simulate_paths(cfg, shocks=z_pricing)
    y_price = arithmetic_asian_call_payoff(paths_price, cfg)
    h_price = pcncv_network.forward(X_price)
    e_h = analytical_network_expectation_conditional(pcncv_network, p, shock_dim=SHOCK_DIM)
    c_price = h_price - e_h

    t0 = time.perf_counter()
    corrected_b1 = y_price - c_price
    beta1_runtime = time.perf_counter() - t0
    beta1_row = _mc_row(corrected_b1, beta1_runtime)
    beta1_row.update(
        {
            "beta": 1.0,
            "payoff_control_correlation": float(np.corrcoef(y_price, c_price)[0, 1]) if _sample_var(c_price) > 0 else float("nan"),
            "pilot_paths": 0,
            "pilot_runtime_s": 0.0,
            "analytical_e_h_conditional": e_h,
        }
    )

    t1 = time.perf_counter()
    X_pilot = build_parameter_inputs(z_pilot, np.repeat(p.reshape(1, -1), z_pilot.shape[0], axis=0))
    paths_pilot = simulate_paths(dataclasses.replace(cfg, n_paths=z_pilot.shape[0]), shocks=z_pilot)
    y_pilot = arithmetic_asian_call_payoff(paths_pilot, cfg)
    h_pilot = pcncv_network.forward(X_pilot)
    c_pilot = h_pilot - e_h
    var_c = _sample_var(c_pilot)
    cov = float(np.cov(y_pilot, c_pilot, ddof=1)[0, 1]) if y_pilot.size > 1 else float("nan")
    beta = cov / var_c if math.isfinite(var_c) and var_c > 1e-12 else 1.0
    corrected_b = y_price - beta * c_price
    beta_runtime = time.perf_counter() - t1
    beta_row = _mc_row(corrected_b, beta_runtime)
    beta_row.update(
        {
            "beta": float(beta),
            "payoff_control_correlation": float(np.corrcoef(y_pilot, c_pilot)[0, 1]) if _sample_var(c_pilot) > 0 else float("nan"),
            "pilot_paths": int(z_pilot.shape[0]),
            "pilot_runtime_s": beta_runtime,
            "analytical_e_h_conditional": e_h,
        }
    )

    hash_after = compute_network_hash(pcncv_network)
    if hash_after != pcncv_hash:
        raise RuntimeError("pcncv network hash mismatch after evaluation")
    return beta1_row, beta_row


def _evaluate_replication(config: PCNCVConfig, replication: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    seeds = _replication_seed_streams(config.base_seed, replication)

    t_data = time.perf_counter()
    train_data = generate_parameter_conditioned_training_data(
        n_training=config.n_training,
        training_parameter_seed=seeds["training_parameters"],
        training_shock_seed=seeds["training_shocks"],
    )
    training_data_runtime = time.perf_counter() - t_data

    snapshots, training_runtime = _train_parameter_conditioned_checkpoints(
        dataset=train_data,
        seed=seeds["training_shocks"],
        checkpoints=config.checkpoints,
        hidden_width=config.hidden_width,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
    )

    aux_contracts = _sample_aux_validation_contracts(
        n_contracts=config.n_aux_validation_contracts,
        seed=seeds["validation_contracts"],
    )
    selected_checkpoint, checkpoint_history = _checkpoint_selection(
        checkpoints=config.checkpoints,
        snapshots=snapshots,
        aux_contracts=aux_contracts,
        validation_n=config.n_validation,
        validation_shock_seed=seeds["validation_shocks"],
    )

    selected_network = snapshots[selected_checkpoint].network
    selected_hash = compute_network_hash(selected_network)

    ref_cfg = _make_cfg(REFERENCE_ID, n_paths=config.n_pricing, seed=seeds["reference_training"], monitoring_dates=config.monitoring_dates)
    frozen_network, e_h0, frozen_hash, frozen_training_runtime = train_reference_network(
        ref_cfg,
        n_training=config.n_training,
        train_seed=seeds["reference_training"],
        hidden_width=config.hidden_width,
        n_epochs=config.frozen_reference_epochs,
    )

    per_rep_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []

    for cidx, cid in enumerate(CONTRACT_IDS):
        cfg = _make_cfg(cid, n_paths=config.n_pricing, seed=0, monitoring_dates=config.monitoring_dates)
        pricing_seed = seeds["final_pricing_shocks"] + cidx * 100
        gcv_pilot_seed = seeds["gcv_pilot"] + cidx * 100
        pcncv_pilot_seed = seeds["conditional_ncv_pilot"] + cidx * 100
        frozen_pilot_seed = seeds["frozen_ncv_pilot"] + cidx * 100

        z_pricing = np.random.default_rng(pricing_seed).standard_normal((config.n_pricing, SHOCK_DIM))

        t_mc = time.perf_counter()
        y_pricing = _simulate_contract_payoffs_from_z(K=cfg.K, sigma=cfg.sigma, T=cfg.T, z=z_pricing)
        mc_row = _mc_row(y_pricing, time.perf_counter() - t_mc)

        z_gcv_pilot = np.random.default_rng(gcv_pilot_seed).standard_normal((config.n_pilot, SHOCK_DIM))
        t_gcv_p = time.perf_counter()
        beta_gcv, eg_gcv, corr_gcv, _ = _gcv_fit(dataclasses.replace(cfg, n_paths=config.n_pilot), z_gcv_pilot)
        gcv_pilot_runtime = time.perf_counter() - t_gcv_p

        t_gcv = time.perf_counter()
        paths_g = simulate_paths(cfg, shocks=z_pricing)
        g_pricing = geometric_asian_call_payoff(paths_g, cfg)
        corrected_gcv = y_pricing - beta_gcv * (g_pricing - eg_gcv)
        gcv_row = _mc_row(corrected_gcv, time.perf_counter() - t_gcv)
        gcv_row.update({"beta": beta_gcv, "payoff_control_correlation": corr_gcv, "pilot_paths": config.n_pilot, "pilot_runtime_s": gcv_pilot_runtime})

        z_pcncv_pilot = np.random.default_rng(pcncv_pilot_seed).standard_normal((config.n_pilot, SHOCK_DIM))
        pcncv_b1, pcncv_b = _evaluate_pcncv_methods(
            pcncv_network=selected_network,
            pcncv_hash=selected_hash,
            cfg=cfg,
            z_pricing=z_pricing,
            z_pilot=z_pcncv_pilot,
        )

        z_frozen_pilot = np.random.default_rng(frozen_pilot_seed).standard_normal((config.n_pilot, SHOCK_DIM))
        frozen_b1, frozen_b = _evaluate_frozen_methods(
            frozen_network=frozen_network,
            frozen_hash=frozen_hash,
            e_h0=e_h0,
            cfg=cfg,
            z_pricing=z_pricing,
            z_pilot=z_frozen_pilot,
        )

        method_rows = {
            "MC": {**mc_row, "beta": float("nan"), "payoff_control_correlation": float("nan"), "pilot_paths": 0, "pilot_runtime_s": 0.0, "training_paths": 0, "training_runtime_s": 0.0, "standalone_runtime_s": mc_row["pricing_runtime_s"]},
            "GCV": {**gcv_row, "training_paths": 0, "training_runtime_s": 0.0, "standalone_runtime_s": gcv_row["pricing_runtime_s"] + gcv_pilot_runtime},
            "NCV_TRANSFER_BETA1": {**frozen_b1, "training_paths": 0, "training_runtime_s": frozen_training_runtime, "standalone_runtime_s": frozen_training_runtime + frozen_b1["pricing_runtime_s"]},
            "NCV_TRANSFER_BETA": {**frozen_b, "training_paths": 0, "training_runtime_s": frozen_training_runtime, "standalone_runtime_s": frozen_training_runtime + frozen_b["pricing_runtime_s"] + frozen_b["pilot_runtime_s"]},
            "PCNCV_BETA1": {**pcncv_b1, "training_paths": config.n_training, "training_runtime_s": training_data_runtime + training_runtime, "standalone_runtime_s": training_data_runtime + training_runtime + pcncv_b1["pricing_runtime_s"]},
            "PCNCV_BETA": {**pcncv_b, "training_paths": config.n_training, "training_runtime_s": training_data_runtime + training_runtime, "standalone_runtime_s": training_data_runtime + training_runtime + pcncv_b["pricing_runtime_s"] + pcncv_b["pilot_runtime_s"]},
        }

        for method, stats in method_rows.items():
            row = {
                "replication": replication,
                "contract_id": cid,
                "method": method,
                "K": cfg.K,
                "sigma": cfg.sigma,
                "T": cfg.T,
                "monitoring_dates": config.monitoring_dates,
                "price": stats["price"],
                "observation_variance": stats["observation_variance"],
                "estimator_variance": stats["estimator_variance"],
                "std_error": stats["std_error"],
                "ci_lower": stats["ci_lower"],
                "ci_upper": stats["ci_upper"],
                "beta": stats.get("beta", float("nan")),
                "payoff_control_correlation": stats.get("payoff_control_correlation", float("nan")),
                "training_paths": stats.get("training_paths", 0),
                "pilot_paths": stats.get("pilot_paths", 0),
                "pricing_observations": stats["pricing_observations"],
                "pricing_simulated_paths": stats["pricing_simulated_paths"],
                "training_runtime_s": stats.get("training_runtime_s", 0.0),
                "pilot_runtime_s": stats.get("pilot_runtime_s", 0.0),
                "marginal_pricing_runtime_s": stats["pricing_runtime_s"],
                "standalone_runtime_s": stats["standalone_runtime_s"],
                "selected_checkpoint": selected_checkpoint,
                "parameter_checkpoint_hash": selected_hash if method.startswith("PCNCV") else (frozen_hash if method.startswith("NCV_TRANSFER") else ""),
                "failure_status": "",
                "failure_message": "",
            }
            per_rep_rows.append(row)
            runtime_rows.append(
                {
                    "replication": replication,
                    "contract_id": cid,
                    "method": method,
                    "training_runtime_s": row["training_runtime_s"],
                    "pilot_runtime_s": row["pilot_runtime_s"],
                    "marginal_pricing_runtime_s": row["marginal_pricing_runtime_s"],
                    "standalone_runtime_s": row["standalone_runtime_s"],
                    "training_paths": row["training_paths"],
                    "pilot_paths": row["pilot_paths"],
                    "pricing_simulated_paths": row["pricing_simulated_paths"],
                }
            )

    training_meta = {
        "replication": replication,
        "selected_checkpoint": selected_checkpoint,
        "selected_checkpoint_hash": selected_hash,
        "frozen_reference_hash": frozen_hash,
        "training_data_generation_runtime_s": training_data_runtime,
        "optimizer_training_runtime_s": training_runtime,
        "frozen_training_runtime_s": frozen_training_runtime,
    }

    return per_rep_rows, checkpoint_history, {"runtime_rows": runtime_rows, "training_meta": training_meta}


def _build_variance_ratio_rows(per_rep_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_key = {(r["replication"], r["contract_id"], r["method"]): r for r in per_rep_rows}
    ratio_rows: list[dict[str, Any]] = []

    comparisons = [
        ("PCNCV_BETA1", "GCV"),
        ("PCNCV_BETA", "GCV"),
        ("PCNCV_BETA1", "NCV_TRANSFER_BETA1"),
        ("PCNCV_BETA", "NCV_TRANSFER_BETA"),
    ]

    for rep, cid, _ in sorted({(r["replication"], r["contract_id"], r["method"]) for r in per_rep_rows}):
        for method, comp in comparisons:
            m_row = by_key.get((rep, cid, method))
            c_row = by_key.get((rep, cid, comp))
            if m_row is None or c_row is None:
                continue
            mv = float(m_row["observation_variance"])
            cv = float(c_row["observation_variance"])
            valid = math.isfinite(mv) and math.isfinite(cv) and mv > 0.0 and cv > 0.0
            ratio_rows.append(
                {
                    "replication": rep,
                    "contract_id": cid,
                    "method": method,
                    "comparator": comp,
                    "variance_ratio": (cv / mv) if valid else "NA",
                    "is_valid": bool(valid),
                    "beats_comparator": bool(valid and (cv / mv) > 1.0),
                }
            )

    summary: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[float]] = {}
    by_method_contract: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in ratio_rows:
        key = (r["method"], r["comparator"])
        by_method_contract.setdefault((r["contract_id"], str(key)), []).append(r)
        if r["is_valid"]:
            grouped.setdefault(key, []).append(float(r["variance_ratio"]))

    for (method, comp), vals in sorted(grouped.items()):
        gm, lo, hi = _student_t_log_ci(vals)
        summary.append(
            {
                "method": method,
                "comparator": comp,
                "geometric_mean": gm if gm is not None else "NA",
                "ci95_lower": lo if lo is not None else "NA",
                "ci95_upper": hi if hi is not None else "NA",
                "n_valid": len(vals),
                "beats_count": sum(1 for v in vals if v > 1.0),
                "beats_percentage": 100.0 * sum(1 for v in vals if v > 1.0) / len(vals) if vals else "NA",
            }
        )

    return ratio_rows, summary


def _runtime_summary(runtime_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in runtime_rows:
        grouped.setdefault((r["contract_id"], r["method"]), []).append(r)

    out: list[dict[str, Any]] = []
    for (cid, method), rows in sorted(grouped.items()):
        out.append(
            {
                "contract_id": cid,
                "method": method,
                "mean_training_runtime_s": statistics.mean([float(r["training_runtime_s"]) for r in rows]),
                "mean_pilot_runtime_s": statistics.mean([float(r["pilot_runtime_s"]) for r in rows]),
                "mean_marginal_pricing_runtime_s": statistics.mean([float(r["marginal_pricing_runtime_s"]) for r in rows]),
                "mean_standalone_runtime_s": statistics.mean([float(r["standalone_runtime_s"]) for r in rows]),
                "n": len(rows),
            }
        )
    return out


def _portfolio_break_even(runtime_summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # Portfolio = price all seven contracts once.
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in runtime_summary:
        by_method.setdefault(str(row["method"]), []).append(row)

    def _cycle(method: str) -> tuple[float, float] | None:
        rows = by_method.get(method, [])
        if len(rows) != len(CONTRACT_IDS):
            return None
        setup = statistics.mean([float(r["mean_training_runtime_s"]) for r in rows])
        marginal = float(sum(float(r["mean_marginal_pricing_runtime_s"]) for r in rows))
        return setup, marginal

    baseline = _cycle("GCV")
    out: list[dict[str, Any]] = []
    if baseline is None:
        return out

    for method in ("PCNCV_BETA1", "PCNCV_BETA", "NCV_TRANSFER_BETA1", "NCV_TRANSFER_BETA"):
        cyc = _cycle(method)
        if cyc is None:
            continue
        b0, bm = baseline
        p0, pm = cyc
        if pm >= bm:
            q = "NA"
            reason = "proposed_marginal_above_baseline"
        else:
            q_star = max(1, int(math.ceil((p0 - b0) / (bm - pm))))
            q = q_star
            reason = ""
        out.append(
            {
                "method": method,
                "baseline_method": "GCV",
                "baseline_setup_cost_s": b0,
                "baseline_marginal_cost_s": bm,
                "proposed_setup_cost_s": p0,
                "proposed_marginal_cost_s": pm,
                "break_even_q": q,
                "failure_reason": reason,
            }
        )
    return out


def run_parameter_conditioned_stage8(config: PCNCVConfig) -> Path:
    if config.monitoring_dates != 12:
        raise ValueError("Parameter-conditioned experiment is restricted to m=12.")
    if trainable_parameter_count(INPUT_DIM, config.hidden_width) != PCNCV_PARAM_COUNT:
        raise RuntimeError("Unexpected PCNCV parameter count.")

    seed_everything(config.base_seed)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = _ensure_unique_run_dir(Path(config.output_dir), f"stage8_pcncv_{config.profile}_{ts}")

    _write_json(
        run_dir / "config_snapshot.json",
        {
            **dataclasses.asdict(config),
            "parameter_bounds": PARAM_BOUNDS,
            "input_dimension": INPUT_DIM,
            "shock_dimension": SHOCK_DIM,
            "parameter_dimension": PARAM_DIM,
            "trainable_parameters": trainable_parameter_count(INPUT_DIM, config.hidden_width),
            "contract_grid": {cid: {"K": k, "sigma": s, "T": t} for cid, (k, s, t) in CONTRACT_GRID.items()},
            "seed_offsets": SEED_OFFSETS,
            "created_at_utc": ts,
        },
    )
    _write_json(run_dir / "environment.json", collect_environment_metadata())

    seed_manifest = build_seed_manifest(config.base_seed, config.replications)
    _write_csv(run_dir / "seed_manifest.csv", seed_manifest)

    per_rep_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    runtime_rows: list[dict[str, Any]] = []
    training_meta_rows: list[dict[str, Any]] = []

    t_main0 = time.perf_counter()
    for rep in range(config.replications):
        rep_rows, rep_checkpoint, rep_meta = _evaluate_replication(config, rep)
        per_rep_rows.extend(rep_rows)
        checkpoint_rows.extend(rep_checkpoint)
        runtime_rows.extend(rep_meta["runtime_rows"])
        training_meta_rows.append(rep_meta["training_meta"])

    main_runtime_s = time.perf_counter() - t_main0

    ratio_rows, ratio_summary = _build_variance_ratio_rows(per_rep_rows)
    runtime_summary = _runtime_summary(runtime_rows)
    portfolio_be = _portfolio_break_even(runtime_summary)

    _write_csv(run_dir / "training_checkpoint_results.csv", checkpoint_rows)
    _write_csv(run_dir / "per_replication_results.csv", per_rep_rows)
    _write_csv(run_dir / "per_replication_variance_ratios.csv", ratio_rows)
    _write_csv(run_dir / "variance_ratio_summary.csv", ratio_summary)
    _write_csv(run_dir / "runtime_raw.csv", runtime_rows)
    _write_csv(run_dir / "runtime_summary.csv", runtime_summary)
    _write_csv(run_dir / "portfolio_break_even.csv", portfolio_be)
    _write_csv(run_dir / "training_meta.csv", training_meta_rows)

    validation_report = {
        "passed": True,
        "m": config.monitoring_dates,
        "trainable_parameters": trainable_parameter_count(INPUT_DIM, config.hidden_width),
        "expected_parameters": PCNCV_PARAM_COUNT,
        "replications": config.replications,
        "main_runtime_s": main_runtime_s,
        "selected_checkpoints": [int(r["selected_checkpoint"]) for r in training_meta_rows],
        "n_rows_per_replication_results": len(per_rep_rows),
        "n_rows_checkpoint_history": len(checkpoint_rows),
        "notes": [
            "Final-pricing shocks are shared within each contract/replication across compatible methods.",
            "PCNCV checkpoint selection uses auxiliary validation contracts and excludes final seven-contract pricing rows.",
            "No m=252 variant executed in this experiment.",
        ],
    }
    _write_json(run_dir / "validation_report.json", validation_report)

    handover = [
        "# Parameter-Conditioned Stage 8 Handover",
        "",
        f"- profile: {config.profile}",
        f"- replications: {config.replications}",
        f"- monitoring_dates: {config.monitoring_dates}",
        f"- selected checkpoints: {[int(r['selected_checkpoint']) for r in training_meta_rows]}",
        f"- main_runtime_s: {main_runtime_s}",
        f"- output_path: {run_dir}",
        "",
        "## Output files",
    ]
    handover.extend([f"- {p.name}" for p in sorted(run_dir.iterdir()) if p.is_file()])
    (run_dir / "handover.md").write_text("\n".join(handover) + "\n", encoding="utf-8")

    return run_dir
