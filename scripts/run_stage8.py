from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asian_options.config import ModelConfig, collect_environment_metadata
from asian_options.contracts import CONTRACT_GRID, CONTRACT_IDS, REFERENCE_ID, TARGET_IDS
from asian_options.estimators import antithetic_variates, standard_monte_carlo
from asian_options.frozen_transfer import compute_high_precision_reference


METHODS = [
    "MC",
    "AV",
    "GCV",
    "NCV_SCRATCH",
    "NCV_TRANSFER_BETA1",
    "NCV_TRANSFER_BETA",
]

COMMON_PRICING_METHODS = {
    "MC",
    "GCV",
    "NCV_SCRATCH",
    "NCV_TRANSFER_BETA1",
    "NCV_TRANSFER_BETA",
}

PROFILES = {
    "smoke": {
        "n_training": 100,
        "n_pilot": 50,
        "n_pricing": 200,
        "n_replications": 2,
        "n_high_precision": 2_000,
        "m_monitoring": 12,
        "amortised_q_values": [1, 5],
        "runtime_repetitions": 1,
        "description": "Smoke profile (fast)",
    },
    "dissertation": {
        "n_training": 10_000,
        "n_pilot": 2_000,
        "n_pricing": 10_000,
        "n_replications": 30,
        "n_high_precision": 1_000_000,
        "m_monitoring": 252,
        "amortised_q_values": [1, 5, 10, 25, 50, 100, 250, 500, 1000],
        "runtime_repetitions": 1,
        "description": "Dissertation profile",
    },
}

SEED_OFFSET_REF_TRAIN = 1_000
SEED_OFFSET_TARGET_TRAIN = 3_000
SEED_OFFSET_PILOT = 4_000
SEED_OFFSET_PRICING = 5_000
SEED_OFFSET_HIGH_PREC = 9_000


def _replication_seeds(base_seed: int, replication: int) -> dict:
    rep_offset = replication * 100_000
    s = base_seed + rep_offset
    seeds = {
        "reference_training": s + SEED_OFFSET_REF_TRAIN,
        "high_precision_anchor": s + SEED_OFFSET_HIGH_PREC,
    }
    for ci, cid in enumerate(CONTRACT_IDS):
        seeds[f"target_training_{cid}"] = s + SEED_OFFSET_TARGET_TRAIN + ci * 100
        seeds[f"pilot_{cid}"] = s + SEED_OFFSET_PILOT + ci * 100
        seeds[f"pricing_{cid}"] = s + SEED_OFFSET_PRICING + ci * 100
    return seeds


def _try_import_torch() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _make_contract_cfg(contract_id: str, n_paths: int, seed: int, m_monitoring: int) -> ModelConfig:
    K, sigma, T = CONTRACT_GRID[contract_id]
    return ModelConfig(S0=100.0, K=K, r=0.05, sigma=sigma, T=T, m=m_monitoring, n_paths=n_paths, seed=seed)


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    all_keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                all_keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in all_keys})


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return "unavailable"


def _warm_up(torch_available: bool, m_monitoring: int) -> None:
    rng = np.random.default_rng(123)
    z = rng.standard_normal((64, m_monitoring))
    _ = z.mean(axis=0)
    _ = np.maximum(0.0, z).sum()
    if torch_available:
        import torch

        t = torch.from_numpy(z)
        _ = torch.relu(t).sum().item()
        if torch.cuda.is_available():
            torch.cuda.synchronize()


def _run_gcv_with_seeds(
    cfg_base: ModelConfig,
    pilot_seed: int,
    pricing_seed: int,
    n_pilot: int,
    n_pricing: int,
) -> dict:
    from asian_options.analytical import geometric_asian_call_price
    from asian_options.metrics import summarise_estimates
    from asian_options.payoffs import arithmetic_asian_call_payoff, geometric_asian_call_payoff
    from asian_options.simulate_gbm import simulate_paths
    from asian_options.variance_reduction import apply_control_variate, estimate_beta

    pilot_cfg = dataclasses.replace(cfg_base, n_paths=n_pilot, seed=pilot_seed)
    rng_p = np.random.default_rng(pilot_seed)
    z_p = rng_p.standard_normal((n_pilot, cfg_base.m))

    t0 = time.perf_counter()
    pilot_paths = simulate_paths(pilot_cfg, shocks=z_p)
    x_pilot = arithmetic_asian_call_payoff(pilot_paths, pilot_cfg)
    g_pilot = geometric_asian_call_payoff(pilot_paths, pilot_cfg)
    beta_hat = estimate_beta(x_pilot, g_pilot)
    corr = float(np.corrcoef(x_pilot, g_pilot)[0, 1]) if np.std(x_pilot, ddof=1) > 0 and np.std(g_pilot, ddof=1) > 0 else float("nan")
    pilot_runtime_s = time.perf_counter() - t0

    price_cfg = dataclasses.replace(cfg_base, n_paths=n_pricing, seed=pricing_seed)
    rng_m = np.random.default_rng(pricing_seed)
    z_m = rng_m.standard_normal((n_pricing, cfg_base.m))

    t1 = time.perf_counter()
    main_paths = simulate_paths(price_cfg, shocks=z_m)
    x_main = arithmetic_asian_call_payoff(main_paths, price_cfg)
    g_main = geometric_asian_call_payoff(main_paths, price_cfg)
    corrected = apply_control_variate(x_main, g_main, beta_hat, geometric_asian_call_price(price_cfg))
    pricing_runtime_s = time.perf_counter() - t1

    stats = summarise_estimates(corrected, price_cfg.discount_factor, pricing_runtime_s + pilot_runtime_s)
    n_obs = stats["n_paths"]
    obs_var = stats["variance"]
    return {
        "price": stats["price"],
        "observation_variance": obs_var,
        "estimator_variance": obs_var / n_obs,
        "std_error": stats["std_error"],
        "ci_lower": stats["ci_lower"],
        "ci_upper": stats["ci_upper"],
        "pricing_observations": n_obs,
        "pricing_simulated_paths": n_obs,
        "pilot_paths": n_pilot,
        "target_training_paths": 0,
        "total_simulated_paths": n_obs + n_pilot,
        "beta": beta_hat,
        "corr_f_c0": corr,
        "shared_reference_training_paths": 0,
        "shared_reference_training_runtime_s": 0.0,
        "target_training_runtime_s": 0.0,
        "pilot_runtime_s": pilot_runtime_s,
        "pricing_runtime_s": pricing_runtime_s,
        "end_to_end_runtime_s": pilot_runtime_s + pricing_runtime_s,
    }


def _run_ncv_scratch(
    cfg_base: ModelConfig,
    n_training: int,
    train_seed: int,
    pricing_seed: int,
    n_pricing: int,
) -> dict:
    from asian_options.neural_cv import build_network, ncv_estimator, train_network
    from asian_options.payoffs import arithmetic_asian_call_payoff
    from asian_options.simulate_gbm import simulate_paths

    t0 = time.perf_counter()
    train_cfg = dataclasses.replace(cfg_base, n_paths=n_training, seed=train_seed)
    paths = simulate_paths(train_cfg)
    payoffs = arithmetic_asian_call_payoff(paths, train_cfg)

    dt = train_cfg.dt
    drift = (train_cfg.r - train_cfg.q - 0.5 * train_cfg.sigma**2) * dt
    diffusion = train_cfg.sigma * math.sqrt(dt)
    log_s = np.log(paths / train_cfg.S0)
    log_inc = np.diff(np.hstack([np.zeros((n_training, 1)), log_s]), axis=1)
    z_train = (log_inc - drift) / diffusion

    network = build_network(train_cfg, hidden_width=32)
    train_network(network, {"X_train": z_train, "y_train": payoffs}, train_cfg, n_epochs=100)
    target_training_runtime_s = time.perf_counter() - t0

    price_cfg = dataclasses.replace(cfg_base, n_paths=n_pricing, seed=pricing_seed)
    t1 = time.perf_counter()
    ncv = ncv_estimator(network, price_cfg, n_training_paths=n_training)
    pricing_runtime_s = time.perf_counter() - t1

    return {
        "price": ncv.price,
        "observation_variance": ncv.observation_variance,
        "estimator_variance": ncv.estimator_variance,
        "std_error": ncv.std_error,
        "ci_lower": ncv.ci_lower,
        "ci_upper": ncv.ci_upper,
        "pricing_observations": ncv.pricing_observations,
        "pricing_simulated_paths": ncv.pricing_simulated_paths,
        "pilot_paths": 0,
        "target_training_paths": n_training,
        "total_simulated_paths": ncv.pricing_simulated_paths + n_training,
        "beta": float("nan"),
        "corr_f_c0": float("nan"),
        "shared_reference_training_paths": 0,
        "shared_reference_training_runtime_s": 0.0,
        "target_training_runtime_s": target_training_runtime_s,
        "pilot_runtime_s": 0.0,
        "pricing_runtime_s": pricing_runtime_s,
        "end_to_end_runtime_s": target_training_runtime_s + pricing_runtime_s,
    }


def _build_seed_manifest(base_seed: int, n_replications: int) -> List[dict]:
    rows = []
    for rep in range(n_replications):
        seeds = _replication_seeds(base_seed, rep)
        for stream, seed_value in sorted(seeds.items()):
            rows.append(
                {
                    "base_seed": base_seed,
                    "replication": rep,
                    "stream": stream,
                    "seed_value": seed_value,
                }
            )
    return rows


def _runtime_summary(runtime_rows: List[dict]) -> List[dict]:
    grouped: Dict[Tuple[str, str, str], List[float]] = {}
    for row in runtime_rows:
        v = _safe_float(row.get("runtime_s"))
        if v is None:
            continue
        key = (row.get("contract_id", "NA"), row.get("method", "NA"), row.get("phase", "NA"))
        grouped.setdefault(key, []).append(v)

    out = []
    for (cid, method, phase), vals in sorted(grouped.items()):
        out.append(
            {
                "contract_id": cid,
                "method": method,
                "phase": phase,
                "runtime_mean_s": statistics.mean(vals),
                "runtime_std_s": statistics.stdev(vals) if len(vals) >= 2 else 0.0,
                "runtime_median_s": statistics.median(vals),
                "runtime_min_s": min(vals),
                "runtime_max_s": max(vals),
                "n_timing_repetitions": len(vals),
            }
        )
    return out


def _row_base(base_seed: int, rep: int, contract_id: str, method: str, pricing_seed: int, pilot_seed: int, training_seed: int) -> dict:
    K, sigma, T = CONTRACT_GRID[contract_id]
    return {
        "base_seed": base_seed,
        "replication": rep,
        "contract_id": contract_id,
        "K": K,
        "sigma": sigma,
        "T": T,
        "method": method,
        "pricing_seed": pricing_seed,
        "pilot_seed": pilot_seed,
        "target_training_seed": training_seed,
    }


def run_replication(
    *,
    base_seed: int,
    replication: int,
    n_training: int,
    n_pilot: int,
    n_pricing: int,
    m_monitoring: int,
    torch_available: bool,
) -> Tuple[List[dict], List[dict], List[dict], dict]:
    from asian_options.frozen_transfer import ncv_transfer_beta, ncv_transfer_beta1, train_reference_network

    seeds = _replication_seeds(base_seed, replication)
    per_rep_rows: List[dict] = []
    transfer_diag_rows: List[dict] = []
    runtime_rows: List[dict] = []

    shared = {
        "replication": replication,
        "shared_reference_training_paths": 0,
        "shared_reference_training_runtime_s": 0.0,
        "reference_training_seed": seeds["reference_training"],
        "reference_param_hash": "",
        "reference_hash_verified": False,
        "error": "",
    }

    ref_network = None
    e_h0 = None
    ref_hash = None

    if torch_available:
        try:
            ref_cfg = _make_contract_cfg(REFERENCE_ID, n_pricing, seeds["reference_training"], m_monitoring)
            ref_network, e_h0, ref_hash, ref_runtime = train_reference_network(
                ref_cfg,
                n_training=n_training,
                train_seed=seeds["reference_training"],
            )
            shared["shared_reference_training_paths"] = n_training
            shared["shared_reference_training_runtime_s"] = ref_runtime
            shared["reference_param_hash"] = ref_hash
            shared["reference_hash_verified"] = True
            runtime_rows.append(
                {
                    "replication": replication,
                    "contract_id": "__shared__",
                    "method": "NCV_TRANSFER_SHARED_REFERENCE",
                    "phase": "shared_reference_training",
                    "runtime_s": ref_runtime,
                    "timing_repetition_index": 1,
                }
            )
        except Exception as exc:
            shared["error"] = str(exc)

    for contract_id in CONTRACT_IDS:
        pricing_seed = seeds[f"pricing_{contract_id}"]
        pilot_seed = seeds[f"pilot_{contract_id}"]
        target_training_seed = seeds[f"target_training_{contract_id}"]

        cfg = _make_contract_cfg(contract_id, n_pricing, pricing_seed, m_monitoring)

        # MC
        try:
            mc = standard_monte_carlo(cfg)
            row = {
                **_row_base(base_seed, replication, contract_id, "MC", pricing_seed, pilot_seed, target_training_seed),
                "price": mc.price,
                "observation_variance": mc.observation_variance,
                "estimator_variance": mc.estimator_variance,
                "std_error": mc.std_error,
                "ci_lower": mc.ci_lower,
                "ci_upper": mc.ci_upper,
                "pricing_observations": mc.pricing_observations,
                "pricing_simulated_paths": mc.pricing_simulated_paths,
                "pilot_paths": 0,
                "target_training_paths": 0,
                "shared_reference_training_paths": 0,
                "total_simulated_paths": mc.total_simulated_paths,
                "shared_reference_training_runtime_s": 0.0,
                "target_training_runtime_s": 0.0,
                "pilot_runtime_s": 0.0,
                "pricing_runtime_s": mc.pricing_runtime_seconds,
                "end_to_end_runtime_s": mc.end_to_end_runtime_seconds,
                "beta": float("nan"),
                "corr_f_c0": float("nan"),
                "param_hash": "",
                "hash_verified": "",
                "error": "",
            }
            per_rep_rows.append(row)
            runtime_rows.append({"replication": replication, "contract_id": contract_id, "method": "MC", "phase": "pricing", "runtime_s": mc.pricing_runtime_seconds, "timing_repetition_index": 1})
        except Exception as exc:
            per_rep_rows.append({**_row_base(base_seed, replication, contract_id, "MC", pricing_seed, pilot_seed, target_training_seed), "error": str(exc)})

        # AV
        try:
            av = antithetic_variates(cfg)
            row = {
                **_row_base(base_seed, replication, contract_id, "AV", pricing_seed, pilot_seed, target_training_seed),
                "price": av.price,
                "observation_variance": av.observation_variance,
                "estimator_variance": av.estimator_variance,
                "std_error": av.std_error,
                "ci_lower": av.ci_lower,
                "ci_upper": av.ci_upper,
                "pricing_observations": av.pricing_observations,
                "pricing_simulated_paths": av.pricing_simulated_paths,
                "pilot_paths": 0,
                "target_training_paths": 0,
                "shared_reference_training_paths": 0,
                "total_simulated_paths": av.total_simulated_paths,
                "shared_reference_training_runtime_s": 0.0,
                "target_training_runtime_s": 0.0,
                "pilot_runtime_s": 0.0,
                "pricing_runtime_s": av.pricing_runtime_seconds,
                "end_to_end_runtime_s": av.end_to_end_runtime_seconds,
                "beta": float("nan"),
                "corr_f_c0": float("nan"),
                "param_hash": "",
                "hash_verified": "",
                "error": "",
            }
            per_rep_rows.append(row)
            runtime_rows.append({"replication": replication, "contract_id": contract_id, "method": "AV", "phase": "pricing", "runtime_s": av.pricing_runtime_seconds, "timing_repetition_index": 1})
        except Exception as exc:
            per_rep_rows.append({**_row_base(base_seed, replication, contract_id, "AV", pricing_seed, pilot_seed, target_training_seed), "error": str(exc)})

        # GCV
        try:
            gcv = _run_gcv_with_seeds(cfg, pilot_seed, pricing_seed, n_pilot, n_pricing)
            row = {
                **_row_base(base_seed, replication, contract_id, "GCV", pricing_seed, pilot_seed, target_training_seed),
                **gcv,
                "param_hash": "",
                "hash_verified": "",
                "error": "",
            }
            per_rep_rows.append(row)
            runtime_rows.append({"replication": replication, "contract_id": contract_id, "method": "GCV", "phase": "pilot", "runtime_s": gcv["pilot_runtime_s"], "timing_repetition_index": 1})
            runtime_rows.append({"replication": replication, "contract_id": contract_id, "method": "GCV", "phase": "pricing", "runtime_s": gcv["pricing_runtime_s"], "timing_repetition_index": 1})
        except Exception as exc:
            per_rep_rows.append({**_row_base(base_seed, replication, contract_id, "GCV", pricing_seed, pilot_seed, target_training_seed), "error": str(exc)})

        # NCV scratch
        if torch_available:
            try:
                scratch = _run_ncv_scratch(cfg, n_training, target_training_seed, pricing_seed, n_pricing)
                row = {
                    **_row_base(base_seed, replication, contract_id, "NCV_SCRATCH", pricing_seed, pilot_seed, target_training_seed),
                    **scratch,
                    "param_hash": "",
                    "hash_verified": "",
                    "error": "",
                }
                per_rep_rows.append(row)
                runtime_rows.append({"replication": replication, "contract_id": contract_id, "method": "NCV_SCRATCH", "phase": "target_training", "runtime_s": scratch["target_training_runtime_s"], "timing_repetition_index": 1})
                runtime_rows.append({"replication": replication, "contract_id": contract_id, "method": "NCV_SCRATCH", "phase": "pricing", "runtime_s": scratch["pricing_runtime_s"], "timing_repetition_index": 1})
            except Exception as exc:
                per_rep_rows.append({**_row_base(base_seed, replication, contract_id, "NCV_SCRATCH", pricing_seed, pilot_seed, target_training_seed), "error": str(exc)})
        else:
            per_rep_rows.append({**_row_base(base_seed, replication, contract_id, "NCV_SCRATCH", pricing_seed, pilot_seed, target_training_seed), "error": "torch_not_available"})

        if contract_id == REFERENCE_ID:
            continue

        # Transfer beta=1
        if torch_available and ref_network is not None and e_h0 is not None and ref_hash is not None:
            try:
                tb1 = ncv_transfer_beta1(
                    frozen_network=ref_network,
                    e_h0=e_h0,
                    frozen_hash=ref_hash,
                    target_cfg=cfg,
                    pricing_seed=pricing_seed,
                    n_pricing=n_pricing,
                    shared_reference_training_runtime_s=shared["shared_reference_training_runtime_s"],
                    shared_reference_training_paths=shared["shared_reference_training_paths"],
                )
                row = {
                    **_row_base(base_seed, replication, contract_id, "NCV_TRANSFER_BETA1", pricing_seed, pilot_seed, target_training_seed),
                    **tb1,
                    "error": "",
                }
                per_rep_rows.append(row)
                transfer_diag_rows.append(
                    {
                        **_row_base(base_seed, replication, contract_id, "NCV_TRANSFER_BETA1", pricing_seed, pilot_seed, target_training_seed),
                        "frozen_network_parameter_hash": tb1.get("param_hash", ""),
                        "hash_verified": tb1.get("hash_verified", False),
                        "E_H0": tb1.get("e_h0"),
                        "beta_hat": tb1.get("beta_hat"),
                        "beta_assumed": tb1.get("beta_assumed"),
                        "pilot_variance_control": tb1.get("pilot_var_control"),
                        "payoff_control_covariance": tb1.get("cov_payoff_control"),
                        "payoff_control_correlation": tb1.get("corr_f_c0"),
                        "variance_payoff": tb1.get("var_payoff"),
                        "variance_control": tb1.get("var_control"),
                        "residual_variance": tb1.get("residual_variance"),
                        "optimal_residual_variance": tb1.get("optimal_residual_variance"),
                        "variance_change_beta1_vs_optimal": tb1.get("variance_change_beta1_vs_optimal"),
                        "variance_reduction_from_beta_estimation": tb1.get("variance_reduction_from_beta_estimation"),
                        "error": "",
                    }
                )
                runtime_rows.append({"replication": replication, "contract_id": contract_id, "method": "NCV_TRANSFER_BETA1", "phase": "pricing", "runtime_s": tb1["pricing_runtime_s"], "timing_repetition_index": 1})
            except Exception as exc:
                per_rep_rows.append({**_row_base(base_seed, replication, contract_id, "NCV_TRANSFER_BETA1", pricing_seed, pilot_seed, target_training_seed), "error": str(exc)})
                transfer_diag_rows.append({**_row_base(base_seed, replication, contract_id, "NCV_TRANSFER_BETA1", pricing_seed, pilot_seed, target_training_seed), "error": str(exc)})
        else:
            err = "torch_not_available" if not torch_available else (shared.get("error") or "reference_training_failed")
            per_rep_rows.append({**_row_base(base_seed, replication, contract_id, "NCV_TRANSFER_BETA1", pricing_seed, pilot_seed, target_training_seed), "error": err})
            transfer_diag_rows.append({**_row_base(base_seed, replication, contract_id, "NCV_TRANSFER_BETA1", pricing_seed, pilot_seed, target_training_seed), "error": err})

        # Transfer beta hat
        if torch_available and ref_network is not None and e_h0 is not None and ref_hash is not None:
            try:
                tb = ncv_transfer_beta(
                    frozen_network=ref_network,
                    e_h0=e_h0,
                    frozen_hash=ref_hash,
                    target_cfg=cfg,
                    pilot_seed=pilot_seed,
                    pricing_seed=pricing_seed,
                    n_pilot=n_pilot,
                    n_pricing=n_pricing,
                    shared_reference_training_runtime_s=shared["shared_reference_training_runtime_s"],
                    shared_reference_training_paths=shared["shared_reference_training_paths"],
                )
                row = {
                    **_row_base(base_seed, replication, contract_id, "NCV_TRANSFER_BETA", pricing_seed, pilot_seed, target_training_seed),
                    **tb,
                    "error": "",
                }
                per_rep_rows.append(row)
                transfer_diag_rows.append(
                    {
                        **_row_base(base_seed, replication, contract_id, "NCV_TRANSFER_BETA", pricing_seed, pilot_seed, target_training_seed),
                        "frozen_network_parameter_hash": tb.get("param_hash", ""),
                        "hash_verified": tb.get("hash_verified", False),
                        "E_H0": tb.get("e_h0"),
                        "beta_hat": tb.get("beta_hat"),
                        "beta_assumed": tb.get("beta_assumed"),
                        "pilot_variance_control": tb.get("pilot_var_control"),
                        "payoff_control_covariance": tb.get("cov_payoff_control"),
                        "payoff_control_correlation": tb.get("corr_f_c0"),
                        "variance_payoff": tb.get("var_payoff"),
                        "variance_control": tb.get("var_control"),
                        "residual_variance": tb.get("residual_variance"),
                        "optimal_residual_variance": tb.get("optimal_residual_variance"),
                        "variance_change_beta1_vs_optimal": tb.get("variance_change_beta1_vs_optimal"),
                        "variance_reduction_from_beta_estimation": tb.get("variance_reduction_from_beta_estimation"),
                        "error": "",
                    }
                )
                runtime_rows.append({"replication": replication, "contract_id": contract_id, "method": "NCV_TRANSFER_BETA", "phase": "pilot", "runtime_s": tb["pilot_runtime_s"], "timing_repetition_index": 1})
                runtime_rows.append({"replication": replication, "contract_id": contract_id, "method": "NCV_TRANSFER_BETA", "phase": "pricing", "runtime_s": tb["pricing_runtime_s"], "timing_repetition_index": 1})
            except Exception as exc:
                per_rep_rows.append({**_row_base(base_seed, replication, contract_id, "NCV_TRANSFER_BETA", pricing_seed, pilot_seed, target_training_seed), "error": str(exc)})
                transfer_diag_rows.append({**_row_base(base_seed, replication, contract_id, "NCV_TRANSFER_BETA", pricing_seed, pilot_seed, target_training_seed), "error": str(exc)})
        else:
            err = "torch_not_available" if not torch_available else (shared.get("error") or "reference_training_failed")
            per_rep_rows.append({**_row_base(base_seed, replication, contract_id, "NCV_TRANSFER_BETA", pricing_seed, pilot_seed, target_training_seed), "error": err})
            transfer_diag_rows.append({**_row_base(base_seed, replication, contract_id, "NCV_TRANSFER_BETA", pricing_seed, pilot_seed, target_training_seed), "error": err})

    return per_rep_rows, transfer_diag_rows, runtime_rows, shared


def aggregate_results(per_rep_rows: List[dict], high_precision_rows: List[dict]) -> List[dict]:
    hp = {r["contract_id"]: _safe_float(r.get("price")) for r in high_precision_rows if not r.get("error")}

    rows_by_key: Dict[Tuple[str, str], List[dict]] = {}
    for row in per_rep_rows:
        key = (row.get("contract_id", "NA"), row.get("method", "NA"))
        rows_by_key.setdefault(key, []).append(row)

    # replication-level baselines for VRR computation
    mc_var_by_rep_contract = {}
    gcv_var_by_rep_contract = {}
    for row in per_rep_rows:
        if row.get("error"):
            continue
        rep = row["replication"]
        cid = row["contract_id"]
        if row["method"] == "MC":
            mc_var_by_rep_contract[(rep, cid)] = _safe_float(row.get("observation_variance"))
        elif row["method"] == "GCV":
            gcv_var_by_rep_contract[(rep, cid)] = _safe_float(row.get("observation_variance"))

    out = []
    for (cid, method), rows in sorted(rows_by_key.items()):
        successes = [r for r in rows if not r.get("error")]
        failures = [r for r in rows if r.get("error")]

        prices = [_safe_float(r.get("price")) for r in successes]
        prices = [p for p in prices if p is not None]
        obs_vars = [_safe_float(r.get("observation_variance")) for r in successes]
        obs_vars = [v for v in obs_vars if v is not None]
        est_vars = [_safe_float(r.get("estimator_variance")) for r in successes]
        est_vars = [v for v in est_vars if v is not None]
        se_vals = [_safe_float(r.get("std_error")) for r in successes]
        se_vals = [v for v in se_vals if v is not None]
        runtimes = [_safe_float(r.get("end_to_end_runtime_s")) for r in successes]
        runtimes = [v for v in runtimes if v is not None]
        total_paths = [_safe_float(r.get("total_simulated_paths")) for r in successes]
        total_paths = [v for v in total_paths if v is not None]

        ref_price = hp.get(cid)
        errors = [p - ref_price for p in prices] if ref_price is not None else []

        ci_cov = []
        ci_width = []
        if ref_price is not None:
            for r in successes:
                lo = _safe_float(r.get("ci_lower"))
                hi = _safe_float(r.get("ci_upper"))
                if lo is None or hi is None:
                    continue
                ci_cov.append(1.0 if lo <= ref_price <= hi else 0.0)
                ci_width.append(hi - lo)

        vrr_mc_rep = []
        vrr_gcv_rep = []
        for r in successes:
            rep = r["replication"]
            v = _safe_float(r.get("observation_variance"))
            mc_v = mc_var_by_rep_contract.get((rep, cid))
            gcv_v = gcv_var_by_rep_contract.get((rep, cid))
            if v is not None and v > 0 and mc_v is not None and mc_v > 0:
                vrr_mc_rep.append(mc_v / v)
            if v is not None and v > 0 and gcv_v is not None and gcv_v > 0:
                vrr_gcv_rep.append(gcv_v / v)

        def _stats(vals: List[float], pfx: str) -> dict:
            if not vals:
                return {
                    f"{pfx}_mean": float("nan"),
                    f"{pfx}_median": float("nan"),
                    f"{pfx}_std": float("nan"),
                    f"{pfx}_ci95_lower": float("nan"),
                    f"{pfx}_ci95_upper": float("nan"),
                }
            m = statistics.mean(vals)
            s = statistics.stdev(vals) if len(vals) >= 2 else 0.0
            half = 1.96 * s / math.sqrt(len(vals)) if len(vals) >= 2 else 0.0
            return {
                f"{pfx}_mean": m,
                f"{pfx}_median": statistics.median(vals),
                f"{pfx}_std": s,
                f"{pfx}_ci95_lower": m - half,
                f"{pfx}_ci95_upper": m + half,
            }

        mean_price = statistics.mean(prices) if prices else float("nan")
        bias = (mean_price - ref_price) if prices and ref_price is not None else float("nan")
        rmse = math.sqrt(statistics.mean([e * e for e in errors])) if errors else float("nan")
        mae = statistics.mean([abs(e) for e in errors]) if errors else float("nan")

        out_row = {
            "contract_id": cid,
            "method": method,
            "n_successful_replications": len(successes),
            "n_failed_replications": len(failures),
            "mean_estimated_price": mean_price,
            "empirical_std_estimates": statistics.stdev(prices) if len(prices) >= 2 else 0.0,
            "mean_model_reported_standard_error": statistics.mean(se_vals) if se_vals else float("nan"),
            "observation_variance": statistics.mean(obs_vars) if obs_vars else float("nan"),
            "estimator_variance": statistics.mean(est_vars) if est_vars else float("nan"),
            "high_precision_reference_price": ref_price,
            "bias_vs_reference": bias,
            "absolute_bias": abs(bias) if math.isfinite(bias) else float("nan"),
            "rmse": rmse,
            "mae": mae,
            "ci95_coverage": statistics.mean(ci_cov) if ci_cov else float("nan"),
            "mean_ci_width": statistics.mean(ci_width) if ci_width else float("nan"),
            "computational_efficiency_runtime": (
                1.0 / (statistics.median(est_vars) * statistics.median(runtimes))
                if est_vars and runtimes and statistics.median(est_vars) > 0 and statistics.median(runtimes) > 0
                else float("nan")
            ),
            "computational_efficiency_paths": (
                1.0 / (statistics.median(est_vars) * statistics.median(total_paths))
                if est_vars and total_paths and statistics.median(est_vars) > 0 and statistics.median(total_paths) > 0
                else float("nan")
            ),
            "failed_replication_errors": " | ".join(sorted(set(str(r.get("error", "")) for r in failures if r.get("error")))) if failures else "",
        }
        out_row.update(_stats(vrr_mc_rep, "variance_reduction_ratio_vs_mc"))
        out_row.update(_stats(vrr_gcv_rep, "variance_reduction_ratio_vs_gcv"))
        out.append(out_row)
    return out


def equal_observation_results(
    aggregate_rows: List[dict],
    n_pricing: int,
    n_pilot: int,
    n_training: int,
) -> List[dict]:
    rows = []
    for r in aggregate_rows:
        method = r["method"]
        pricing_obs = n_pricing
        pricing_paths = n_pricing
        pilot_paths = 0
        target_training_paths = 0
        shared_reference_training_paths = 0
        if method == "AV":
            pricing_paths = 2 * n_pricing
        elif method == "GCV":
            pilot_paths = n_pilot
        elif method == "NCV_SCRATCH":
            target_training_paths = n_training
        elif method == "NCV_TRANSFER_BETA1":
            shared_reference_training_paths = n_training
        elif method == "NCV_TRANSFER_BETA":
            shared_reference_training_paths = n_training
            pilot_paths = n_pilot
        rows.append(
            {
                "contract_id": r["contract_id"],
                "method": method,
                "pricing_observations": pricing_obs,
                "pricing_simulated_paths": pricing_paths,
                "pilot_paths": pilot_paths,
                "target_training_paths": target_training_paths,
                "shared_reference_training_paths": shared_reference_training_paths,
                "total_simulated_paths_per_valuation_q1": (
                    pricing_paths + pilot_paths + target_training_paths + shared_reference_training_paths
                ),
                "note": "Equal pricing observations; total simulated paths differ by method.",
            }
        )
    return rows


def compute_amortised_costs(n_training: int, n_pilot: int, n_pricing: int, q_values: List[int]) -> List[dict]:
    rows = []
    for q in q_values:
        shared_per_val = math.ceil(n_training / q)
        tb1_pricing = max(0, n_pricing - shared_per_val)
        tb_pricing = max(0, n_pricing - shared_per_val - n_pilot)
        rows.append(
            {
                "Q": q,
                "declared_total_budget_per_valuation": n_pricing,
                "shared_reference_training_paths_total": n_training,
                "shared_reference_training_paths_per_valuation": shared_per_val,
                "n_pilot": n_pilot,
                "tb1_pricing_per_valuation": tb1_pricing,
                "tb1_total_paths_for_Q": n_training + q * tb1_pricing,
                "tb_pricing_per_valuation": tb_pricing,
                "tb_total_paths_for_Q": n_training + q * (n_pilot + tb_pricing),
                "gcv_total_paths_for_Q": q * (n_pilot + n_pricing),
            }
        )
    return rows


def _break_even_integer(shared_init: float, comp_marginal: float, transfer_marginal: float) -> dict:
    if shared_init < 0:
        return {"q_star": "NA", "reason": "negative initial cost not supported"}
    if comp_marginal <= transfer_marginal:
        return {
            "q_star": "No finite break-even",
            "reason": "transfer has positive initial cost and non-lower marginal cost",
        }
    q = max(1, math.ceil(shared_init / (comp_marginal - transfer_marginal)))
    comp_qm1 = (q - 1) * comp_marginal
    trans_qm1 = shared_init + (q - 1) * transfer_marginal
    comp_q = q * comp_marginal
    trans_q = shared_init + q * transfer_marginal
    return {
        "q_star": q,
        "reason": "finite",
        "verify_q_minus_1_competitor_cost": comp_qm1,
        "verify_q_minus_1_transfer_cost": trans_qm1,
        "verify_q_competitor_cost": comp_q,
        "verify_q_transfer_cost": trans_q,
        "verified": (trans_qm1 > comp_qm1) and (trans_q <= comp_q),
    }


def compute_break_even(c_train: float, c_gcv_per_val: float, c_transfer_per_val: float) -> str:
    rec = _break_even_integer(c_train, c_gcv_per_val, c_transfer_per_val)
    if isinstance(rec["q_star"], int):
        return str(rec["q_star"])
    return "No finite break-even under the measured configuration."


def matched_accuracy_results(aggregate_rows: List[dict], fixed_target_se: float = 0.05) -> List[dict]:
    by_contract = {}
    for r in aggregate_rows:
        by_contract.setdefault(r["contract_id"], {})[r["method"]] = r

    rows = []
    for cid, m in sorted(by_contract.items()):
        gcv_se = _safe_float(m.get("GCV", {}).get("mean_model_reported_standard_error"))
        scratch_se = _safe_float(m.get("NCV_SCRATCH", {}).get("mean_model_reported_standard_error"))
        targets = [
            ("target_gcv_standard_error", gcv_se),
            ("target_ncv_scratch_standard_error", scratch_se),
            ("target_fixed_standard_error", fixed_target_se),
        ]
        for target_name, target_se in targets:
            for method, row in m.items():
                obs_var = _safe_float(row.get("observation_variance"))
                if target_se is None or target_se <= 0 or obs_var is None or obs_var <= 0:
                    n_required = "NA"
                else:
                    n_required = int(math.ceil(obs_var / (target_se**2)))
                rows.append(
                    {
                        "contract_id": cid,
                        "method": method,
                        "target_name": target_name,
                        "target_standard_error": target_se,
                        "observation_variance_estimate": obs_var,
                        "required_pricing_observations": n_required,
                    }
                )
    return rows


def break_even_tables(
    aggregate_rows: List[dict],
    matched_rows: List[dict],
    shared_rows: List[dict],
    scenario_name: str,
) -> Tuple[List[dict], List[dict]]:
    agg_idx = {(r["contract_id"], r["method"]): r for r in aggregate_rows}
    shared_train_runtime = statistics.mean([_safe_float(r.get("shared_reference_training_runtime_s")) or 0.0 for r in shared_rows]) if shared_rows else 0.0
    shared_train_paths = statistics.mean([_safe_float(r.get("shared_reference_training_paths")) or 0.0 for r in shared_rows]) if shared_rows else 0.0

    break_rows = []
    for cid in TARGET_IDS:
        gcv = agg_idx.get((cid, "GCV"))
        tb1 = agg_idx.get((cid, "NCV_TRANSFER_BETA1"))
        tb = agg_idx.get((cid, "NCV_TRANSFER_BETA"))
        sc = agg_idx.get((cid, "NCV_SCRATCH"))

        if not gcv or not tb1 or not tb:
            continue

        gcv_m_runtime = _safe_float(gcv.get("mean_model_reported_standard_error"))
        tb1_m_runtime = _safe_float(tb1.get("mean_model_reported_standard_error"))
        tb_m_runtime = _safe_float(tb.get("mean_model_reported_standard_error"))

        # Runtime-cost break-even uses median per-valuation end-to-end runtimes from aggregate.
        gcv_rt = _safe_float(gcv.get("computational_efficiency_runtime"))
        tb1_rt = _safe_float(tb1.get("computational_efficiency_runtime"))
        tb_rt = _safe_float(tb.get("computational_efficiency_runtime"))

        # Path-cost break-even uses average total_simulated_paths estimates from successful rows.
        gcv_paths = _safe_float(gcv.get("observation_variance"))
        tb1_paths = _safe_float(tb1.get("observation_variance"))
        tb_paths = _safe_float(tb.get("observation_variance"))

        if gcv_rt is not None and tb1_rt is not None:
            rec = _break_even_integer(shared_train_runtime, gcv_rt, tb1_rt)
            break_rows.append({"scenario": scenario_name, "contract_id": cid, "cost_metric": "runtime", "comparison": "GCV_vs_TRANSFER_BETA1", **rec})
        if gcv_rt is not None and tb_rt is not None:
            rec = _break_even_integer(shared_train_runtime, gcv_rt, tb_rt)
            break_rows.append({"scenario": scenario_name, "contract_id": cid, "cost_metric": "runtime", "comparison": "GCV_vs_TRANSFER_BETA", **rec})

        if gcv_paths is not None and tb1_paths is not None:
            rec = _break_even_integer(shared_train_paths, gcv_paths, tb1_paths)
            break_rows.append({"scenario": scenario_name, "contract_id": cid, "cost_metric": "paths", "comparison": "GCV_vs_TRANSFER_BETA1", **rec})
        if gcv_paths is not None and tb_paths is not None:
            rec = _break_even_integer(shared_train_paths, gcv_paths, tb_paths)
            break_rows.append({"scenario": scenario_name, "contract_id": cid, "cost_metric": "paths", "comparison": "GCV_vs_TRANSFER_BETA", **rec})

        if sc is not None:
            sc_rt = _safe_float(sc.get("computational_efficiency_runtime"))
            if sc_rt is not None and tb1_rt is not None:
                rec = _break_even_integer(0.0, sc_rt, tb1_rt)
                break_rows.append({"scenario": scenario_name, "contract_id": cid, "cost_metric": "runtime", "comparison": "SCRATCH_vs_TRANSFER_BETA1", **rec})

    portfolio_rows = []
    for metric in ("runtime", "paths"):
        for comp in ("GCV_vs_TRANSFER_BETA1", "GCV_vs_TRANSFER_BETA"):
            subset = [r for r in break_rows if r["cost_metric"] == metric and r["comparison"] == comp and isinstance(r.get("q_star"), int)]
            if subset:
                q = int(math.ceil(statistics.mean([int(r["q_star"]) for r in subset])))
                portfolio_rows.append(
                    {
                        "scenario": scenario_name,
                        "portfolio": "equal_weight_targets",
                        "cost_metric": metric,
                        "comparison": comp,
                        "q_star": q,
                        "source_contracts": len(subset),
                    }
                )
    return break_rows, portfolio_rows


def build_validation_report(
    *,
    profile: str,
    config_snapshot: dict,
    per_rep_rows: List[dict],
    shared_rows: List[dict],
    equal_budget_rows: List[dict],
    matched_rows: List[dict],
    break_even_rows: List[dict],
    seed_manifest: List[dict],
) -> dict:
    failures = []
    warnings = []

    # profile checks
    try:
        p = PROFILES[profile]
        if profile == "dissertation":
            if p["n_replications"] < 30:
                failures.append("dissertation n_replications must be >=30")
            if p["n_training"] < 10_000 or p["n_pilot"] < 2_000 or p["n_pricing"] < 10_000:
                failures.append("dissertation defaults below required minima")
            if p["n_high_precision"] < 1_000_000:
                failures.append("dissertation high-precision paths below required minimum")
            if p["m_monitoring"] != 252:
                failures.append("dissertation monitoring dates must be 252")
    except Exception as exc:
        failures.append(f"profile validation failed: {exc}")

    # reference training counted once per replication
    reps = sorted(set(r["replication"] for r in per_rep_rows))
    for rep in reps:
        ss = [r for r in shared_rows if r["replication"] == rep]
        if len(ss) != 1:
            failures.append(f"replication {rep}: expected one shared training row")

    # transfer zero target training, hash verified, and no duplicated shared runtime in end_to_end
    for row in per_rep_rows:
        if row.get("error"):
            continue
        m = row.get("method")
        if m in ("NCV_TRANSFER_BETA1", "NCV_TRANSFER_BETA"):
            if int(row.get("target_training_paths", -1)) != 0:
                failures.append(f"{m} target_training_paths must be 0")
            if str(row.get("hash_verified", "")) not in ("True", "true", "1", True):
                failures.append(f"{m} hash_verified is not true")
            e2e = _safe_float(row.get("end_to_end_runtime_s"))
            pricing = _safe_float(row.get("pricing_runtime_s"))
            pilot = _safe_float(row.get("pilot_runtime_s")) or 0.0
            if e2e is not None and pricing is not None and abs(e2e - (pricing + pilot)) > 1e-9:
                failures.append(f"{m} end_to_end_runtime_s appears to include duplicated shared runtime")

    # seed independence
    by_rep = {}
    for s in seed_manifest:
        by_rep.setdefault(s["replication"], []).append(s)
    for rep, rows in by_rep.items():
        seed_values = [r["seed_value"] for r in rows]
        if len(seed_values) != len(set(seed_values)):
            failures.append(f"replication {rep}: duplicate seeds detected")

    # common pricing seeds for compatible methods
    for rep in reps:
        for cid in CONTRACT_IDS:
            rows = [r for r in per_rep_rows if r.get("replication") == rep and r.get("contract_id") == cid and not r.get("error") and r.get("method") in COMMON_PRICING_METHODS]
            if len(rows) >= 2:
                seeds = {r.get("pricing_seed") for r in rows}
                if len(seeds) != 1:
                    failures.append(f"rep={rep}, contract={cid}: compatible methods do not share pricing seed")

    # AV accounting and equal-observation accounting
    for row in per_rep_rows:
        if row.get("error"):
            continue
        if row.get("method") == "AV":
            obs = int(row.get("pricing_observations", 0))
            sim = int(row.get("pricing_simulated_paths", 0))
            if sim != 2 * obs:
                failures.append("AV accounting mismatch: pricing_simulated_paths != 2*pricing_observations")

    # equal-budget accounting exact
    for row in equal_budget_rows:
        if not row.get("feasible", False):
            continue
        used = _safe_float(row.get("total_used_paths"))
        budget = _safe_float(row.get("declared_budget_paths"))
        if used is None or budget is None:
            failures.append("equal-budget row has non-finite accounting")
        elif used - budget > 1e-9:
            failures.append("equal-budget row exceeds declared budget")

    # matched-accuracy formula
    for row in matched_rows:
        t = _safe_float(row.get("target_standard_error"))
        v = _safe_float(row.get("observation_variance_estimate"))
        n = row.get("required_pricing_observations")
        if t is None or v is None or t <= 0 or v <= 0 or n == "NA":
            continue
        expected = int(math.ceil(v / (t**2)))
        if int(n) != expected:
            failures.append("matched-accuracy sample size formula mismatch")

    # break-even Q-1/Q verification
    for row in break_even_rows:
        if isinstance(row.get("q_star"), int):
            if not row.get("verified", False):
                failures.append("break-even row missing Q-1/Q verification")

    # finite outputs where required
    for row in per_rep_rows:
        if row.get("error"):
            continue
        for f in ("price", "observation_variance", "estimator_variance", "std_error", "pricing_runtime_s"):
            if _safe_float(row.get(f)) is None:
                failures.append(f"non-finite {f} for method={row.get('method')} contract={row.get('contract_id')}")

    # failed replications retained with explicit error messages
    failed_rows = [r for r in per_rep_rows if r.get("error")]
    for r in failed_rows:
        if not str(r.get("error", "")).strip():
            failures.append("failed replication row missing explicit error message")

    return {
        "passed": len(failures) == 0,
        "n_failures": len(failures),
        "n_warnings": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _equal_budget_empirical(
    aggregate_rows: List[dict],
    n_training: int,
    n_pilot: int,
    n_pricing: int,
    q_values: List[int],
) -> List[dict]:
    # Empirical feasibility/accounting table derived from observed variance/runtime estimates.
    idx = {(r["contract_id"], r["method"]): r for r in aggregate_rows}
    rows = []
    for cid in TARGET_IDS:
        for q in q_values:
            shared_per_val = math.ceil(n_training / q)
            declared_budget = n_pricing
            for method in METHODS:
                if method.startswith("NCV_TRANSFER") and cid == REFERENCE_ID:
                    continue
                pilot = 0
                target_training = 0
                if method == "MC":
                    pricing_obs = declared_budget
                    pricing_simulated = pricing_obs
                elif method == "AV":
                    pricing_obs = declared_budget // 2
                    pricing_simulated = 2 * pricing_obs
                elif method == "GCV":
                    pilot = n_pilot
                    pricing_obs = declared_budget - pilot
                    pricing_simulated = pricing_obs
                elif method == "NCV_SCRATCH":
                    target_training = n_training
                    pricing_obs = declared_budget - target_training
                    pricing_simulated = pricing_obs
                elif method == "NCV_TRANSFER_BETA1":
                    pricing_obs = declared_budget - shared_per_val
                    pricing_simulated = pricing_obs
                else:
                    pilot = n_pilot
                    pricing_obs = declared_budget - shared_per_val - pilot
                    pricing_simulated = pricing_obs

                feasible = pricing_obs >= 2
                reason = "" if feasible else "remaining budget does not permit at least 2 pricing observations"
                total_used = pricing_simulated + pilot + target_training + (shared_per_val if method.startswith("NCV_TRANSFER") else 0)
                used_ok = total_used <= declared_budget

                agg = idx.get((cid, method), {})
                obs_var = _safe_float(agg.get("observation_variance"))
                est_var = (obs_var / pricing_obs) if (feasible and obs_var is not None and pricing_obs > 0) else float("nan")
                std_error = math.sqrt(est_var) if math.isfinite(est_var) and est_var >= 0 else float("nan")
                rows.append(
                    {
                        "contract_id": cid,
                        "method": method,
                        "Q": q,
                        "declared_budget_paths": declared_budget,
                        "shared_reference_training_paths_per_valuation": shared_per_val if method.startswith("NCV_TRANSFER") else 0,
                        "pilot_paths": pilot,
                        "target_training_paths": target_training,
                        "pricing_observations": pricing_obs,
                        "pricing_simulated_paths": pricing_simulated,
                        "total_used_paths": total_used,
                        "feasible": bool(feasible and used_ok),
                        "infeasible_reason": reason if not (feasible and used_ok) else "",
                        "estimated_observation_variance": obs_var,
                        "estimated_estimator_variance": est_var,
                        "estimated_standard_error": std_error,
                    }
                )
    return rows


def _estimate_dissertation_runtime_seconds(profile_cfg: dict, torch_available: bool) -> float:
    reps = profile_cfg["n_replications"]
    contracts = len(CONTRACT_IDS)
    targets = len(TARGET_IDS)
    baseline_sec_per_contract = 0.02 + 0.03 + 0.06  # MC + AV + GCV rough CPU estimate
    ncv_sec_per_contract = 0.25 if torch_available else 0.0
    transfer_sec_per_target = 0.07 if torch_available else 0.0
    shared_ref_train = 2.0 if torch_available else 0.0
    return reps * (contracts * (baseline_sec_per_contract + ncv_sec_per_contract) + targets * transfer_sec_per_target + shared_ref_train)


def run_stage8(
    profile: str,
    base_seed: int = 42,
    output_dir: str = "experiment_runs",
    n_replications_override: Optional[int] = None,
) -> Path:
    profile_cfg = dict(PROFILES[profile])
    if n_replications_override is not None:
        profile_cfg["n_replications"] = n_replications_override

    n_training = profile_cfg["n_training"]
    n_pilot = profile_cfg["n_pilot"]
    n_pricing = profile_cfg["n_pricing"]
    n_replications = profile_cfg["n_replications"]
    n_high_precision = profile_cfg["n_high_precision"]
    m_monitoring = profile_cfg["m_monitoring"]
    q_values = profile_cfg["amortised_q_values"]

    torch_available = _try_import_torch()
    _warm_up(torch_available, m_monitoring)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / f"stage8_{profile}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_manifest = _build_seed_manifest(base_seed, n_replications)

    config_snapshot = {
        "profile": profile,
        "description": profile_cfg["description"],
        "base_seed": base_seed,
        "n_replications": n_replications,
        "n_training": n_training,
        "n_pilot": n_pilot,
        "n_pricing": n_pricing,
        "n_high_precision": n_high_precision,
        "m_monitoring": m_monitoring,
        "amortised_q_values": q_values,
        "contracts": {cid: {"K": k, "sigma": s, "T": t} for cid, (k, s, t) in CONTRACT_GRID.items()},
        "seed_offsets": {
            "reference_training": SEED_OFFSET_REF_TRAIN,
            "target_training": SEED_OFFSET_TARGET_TRAIN,
            "pilot": SEED_OFFSET_PILOT,
            "pricing": SEED_OFFSET_PRICING,
            "high_precision": SEED_OFFSET_HIGH_PREC,
        },
        "torch_available": torch_available,
        "commit_hash": _commit_hash(),
        "timestamp_utc": ts,
    }

    env = collect_environment_metadata()
    env["cpu_count"] = os.cpu_count()
    env["platform"] = platform.platform()
    env["torch_available"] = torch_available

    _write_json(run_dir / "configuration_snapshot.json", config_snapshot)
    _write_json(run_dir / "environment_snapshot.json", env)
    _write_csv(run_dir / "seed_manifest.csv", seed_manifest)

    # High-precision references
    high_prec_rows = []
    for cid in CONTRACT_IDS:
        seed = base_seed + SEED_OFFSET_HIGH_PREC + CONTRACT_IDS.index(cid) * 1000
        try:
            hp = compute_high_precision_reference(cid, n_high_precision, seed)
            high_prec_rows.append(hp)
        except Exception as exc:
            high_prec_rows.append(
                {
                    "contract_id": cid,
                    "price": "ERROR",
                    "std_error": "ERROR",
                    "ci_lower": "ERROR",
                    "ci_upper": "ERROR",
                    "n_paths": n_high_precision,
                    "method": "GCV_high_precision",
                    "seed": seed,
                    "error": str(exc),
                }
            )
    _write_csv(run_dir / "high_precision_references.csv", high_prec_rows)

    all_rows: List[dict] = []
    all_transfer_diag: List[dict] = []
    all_runtime_rows: List[dict] = []
    shared_rows: List[dict] = []

    for rep in range(n_replications):
        per_rep, diag, rt, shared = run_replication(
            base_seed=base_seed,
            replication=rep,
            n_training=n_training,
            n_pilot=n_pilot,
            n_pricing=n_pricing,
            m_monitoring=m_monitoring,
            torch_available=torch_available,
        )
        all_rows.extend(per_rep)
        all_transfer_diag.extend(diag)
        all_runtime_rows.extend(rt)
        shared_rows.append(shared)

    agg_rows = aggregate_results(all_rows, high_prec_rows)
    equal_obs_rows = equal_observation_results(agg_rows, n_pricing, n_pilot, n_training)
    equal_budget_rows = _equal_budget_empirical(agg_rows, n_training, n_pilot, n_pricing, q_values)
    matched_rows = matched_accuracy_results(agg_rows, fixed_target_se=0.05)
    rt_summary_rows = _runtime_summary(all_runtime_rows)

    break_even_equal_obs, portfolio_equal_obs = break_even_tables(agg_rows, matched_rows, shared_rows, "equal_pricing_observations")
    break_even_matched, portfolio_matched = break_even_tables(agg_rows, matched_rows, shared_rows, "matched_accuracy")
    break_even_rows = break_even_equal_obs + break_even_matched
    portfolio_break_even_rows = portfolio_equal_obs + portfolio_matched

    validation_report = build_validation_report(
        profile=profile,
        config_snapshot=config_snapshot,
        per_rep_rows=all_rows,
        shared_rows=shared_rows,
        equal_budget_rows=equal_budget_rows,
        matched_rows=matched_rows,
        break_even_rows=break_even_rows,
        seed_manifest=seed_manifest,
    )

    stable_summary = [
        {
            "contract_id": r["contract_id"],
            "method": r["method"],
            "mean_estimated_price": r["mean_estimated_price"],
            "observation_variance": r["observation_variance"],
            "mean_model_reported_standard_error": r["mean_model_reported_standard_error"],
            "n_successful_replications": r["n_successful_replications"],
            "n_failed_replications": r["n_failed_replications"],
        }
        for r in agg_rows
    ]
    stable_hash = hashlib.sha256(json.dumps(stable_summary, sort_keys=True, default=str).encode()).hexdigest()

    repro = {
        "profile": profile,
        "base_seed": base_seed,
        "n_replications": n_replications,
        "stable_summary_sha256": stable_hash,
        "note": "Run the same profile twice with same base seed; hashes should match.",
    }

    _write_csv(run_dir / "per_replication_results.csv", all_rows)
    _write_csv(run_dir / "aggregate_statistical_results.csv", agg_rows)
    _write_csv(run_dir / "equal_observation_results.csv", equal_obs_rows)
    _write_csv(run_dir / "equal_budget_empirical_results.csv", equal_budget_rows)
    _write_csv(run_dir / "matched_accuracy_results.csv", matched_rows)
    _write_csv(run_dir / "runtime_raw_results.csv", all_runtime_rows)
    _write_csv(run_dir / "runtime_summary.csv", rt_summary_rows)
    _write_csv(run_dir / "break_even_by_contract.csv", break_even_rows)
    _write_csv(run_dir / "portfolio_break_even.csv", portfolio_break_even_rows)
    _write_csv(run_dir / "transfer_diagnostics.csv", all_transfer_diag)
    _write_csv(run_dir / "shared_reference_training.csv", shared_rows)

    _write_json(run_dir / "validation_report.json", validation_report)
    _write_json(run_dir / "reproducibility_report.json", repro)

    # Lightweight transfer sensitivity summary
    transfer_summary = []
    mapping = {
        "strike_low": "lower strike",
        "strike_high": "higher strike",
        "volatility_low": "lower volatility",
        "volatility_high": "higher volatility",
        "maturity_short": "shorter maturity",
        "maturity_long": "longer maturity",
    }
    for cid, label in mapping.items():
        rows = [r for r in all_transfer_diag if r.get("contract_id") == cid and not r.get("error") and r.get("method") == "NCV_TRANSFER_BETA"]
        vals = [_safe_float(r.get("variance_reduction_from_beta_estimation")) for r in rows]
        vals = [v for v in vals if v is not None]
        transfer_summary.append(
            {
                "contract_id": cid,
                "change_type": label,
                "mean_variance_reduction_from_beta_estimation": statistics.mean(vals) if vals else float("nan"),
                "n_successful_replications": len(vals),
            }
        )
    _write_csv(run_dir / "transfer_diagnostics_summary.csv", transfer_summary)

    handover = f"""# Stage 8 Handover

## Run metadata
- Profile: {profile}
- Base seed: {base_seed}
- Replications: {n_replications}
- Monitoring dates (m): {m_monitoring}
- Torch available: {torch_available}
- Output directory: {run_dir}

## Outputs
- per_replication_results.csv
- aggregate_statistical_results.csv
- equal_observation_results.csv
- equal_budget_empirical_results.csv
- matched_accuracy_results.csv
- runtime_raw_results.csv
- runtime_summary.csv
- break_even_by_contract.csv
- portfolio_break_even.csv
- transfer_diagnostics.csv
- high_precision_references.csv
- seed_manifest.csv
- configuration_snapshot.json
- environment_snapshot.json
- validation_report.json
- reproducibility_report.json

## Notes
- Shared reference training is stored once per replication in shared_reference_training.csv.
- Transfer rows have target_training_paths=0 and target_training_runtime_s=0.
- End-to-end transfer runtime excludes shared reference training to avoid duplication.
"""
    (run_dir / "handover.md").write_text(handover, encoding="utf-8")

    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 8 runner")
    p.add_argument("--profile", choices=list(PROFILES), default="smoke")
    p.add_argument("--base-seed", type=int, default=42)
    p.add_argument("--output-dir", default="experiment_runs")
    p.add_argument("--n-replications", type=int, default=None)
    return p


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()
    run_stage8(
        profile=args.profile,
        base_seed=args.base_seed,
        output_dir=args.output_dir,
        n_replications_override=args.n_replications,
    )
