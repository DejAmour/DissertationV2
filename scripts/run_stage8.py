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
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asian_options.config import ModelConfig, collect_environment_metadata
from asian_options.contracts import CONTRACT_GRID, CONTRACT_IDS, REFERENCE_ID, TARGET_IDS, make_contract_cfg
from asian_options.estimators import antithetic_variates, geometric_control_variate, standard_monte_carlo

PROFILES = {
    "smoke": {
        "n_training": 100,
        "n_pilot": 50,
        "n_pricing": 200,
        "n_replications": 2,
        "n_high_precision": 2_000,
        "amortised_q_values": [1, 5],
        "description": "Smoke test: tiny sizes, 2 reps, all contracts, all methods",
    },
    "dissertation": {
        "n_training": 5_000,
        "n_pilot": 1_000,
        "n_pricing": 50_000,
        "n_replications": 30,
        "n_high_precision": 500_000,
        "amortised_q_values": [1, 5, 10, 25, 50, 100, 250, 500, 1000],
        "description": "Dissertation: standard sizes, 30 reps, full analyses",
    },
}

SEED_OFFSET_REF_TRAIN = 1_000
SEED_OFFSET_REF_VAL = 2_000
SEED_OFFSET_TARGET_TRAIN = 3_000
SEED_OFFSET_PILOT = 4_000
SEED_OFFSET_PRICING = 5_000
SEED_OFFSET_HIGH_PREC = 9_000

BASE_METHODS = ("MC", "AV", "GCV", "NCV_SCRATCH")
TRANSFER_METHODS = ("NCV_TRANSFER_BETA1", "NCV_TRANSFER_BETA")


def _replication_seeds(base_seed: int, replication: int) -> dict:
    rep_offset = replication * 100_000
    s = base_seed + rep_offset
    out = {
        "ref_train": s + SEED_OFFSET_REF_TRAIN,
        "ref_val": s + SEED_OFFSET_REF_VAL,
        "high_prec": s + SEED_OFFSET_HIGH_PREC,
    }
    for ci, cid in enumerate(CONTRACT_IDS):
        out[f"target_train_{cid}"] = s + SEED_OFFSET_TARGET_TRAIN + ci * 100
        out[f"pilot_{cid}"] = s + SEED_OFFSET_PILOT + ci * 100
        out[f"pricing_{cid}"] = s + SEED_OFFSET_PRICING + ci * 100
    return out


def _try_import_torch() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except ImportError:
        return False


def _torch_version() -> str:
    try:
        import torch

        return str(getattr(torch, "__version__", "unknown"))
    except ImportError:
        return "not-installed"


def _warmup_numpy_and_torch(torch_available: bool) -> None:
    import numpy as np

    x = np.random.default_rng(123).standard_normal((256, 64))
    _ = x @ x.T
    if torch_available:
        import torch

        t = torch.randn(256, 64)
        _ = t @ t.T


def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _mean(vals: Iterable[Optional[float]]) -> Optional[float]:
    clean = [v for v in vals if v is not None]
    return statistics.mean(clean) if clean else None


def _std(vals: Iterable[Optional[float]]) -> Optional[float]:
    clean = [v for v in vals if v is not None]
    return statistics.stdev(clean) if len(clean) >= 2 else None


def _median(vals: Iterable[Optional[float]]) -> Optional[float]:
    clean = [v for v in vals if v is not None]
    return statistics.median(clean) if clean else None


def _quantile(vals: List[float], q: float) -> Optional[float]:
    if not vals:
        return None
    xs = sorted(vals)
    if len(xs) == 1:
        return xs[0]
    pos = q * (len(xs) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return xs[lo]
    w = pos - lo
    return (1 - w) * xs[lo] + w * xs[hi]


def _write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        seen: set[str] = set()
        ordered: List[str] = []
        for row in rows:
            for k in row:
                if k not in seen:
                    seen.add(k)
                    ordered.append(k)
        fieldnames = ordered
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return "unavailable"


def _run_ncv_scratch(
    contract_cfg_base: ModelConfig,
    n_training: int,
    train_seed: int,
    pricing_seed: int,
    n_pricing: int,
) -> dict:
    from asian_options.frozen_transfer import _extract_z_shocks
    from asian_options.neural_cv import build_network, ncv_estimator, train_network
    from asian_options.payoffs import arithmetic_asian_call_payoff
    from asian_options.simulate_gbm import simulate_paths

    t_train = time.perf_counter()
    train_cfg = dataclasses.replace(contract_cfg_base, n_paths=n_training, seed=train_seed)
    train_paths = simulate_paths(train_cfg)
    train_payoffs = arithmetic_asian_call_payoff(train_paths, train_cfg)
    z_train = _extract_z_shocks(train_paths, train_cfg)
    dataset = {"X_train": z_train, "y_train": train_payoffs}
    network = build_network(train_cfg, hidden_width=32)
    train_network(network, dataset, train_cfg, n_epochs=100)
    train_runtime = time.perf_counter() - t_train

    price_cfg = dataclasses.replace(contract_cfg_base, n_paths=n_pricing, seed=pricing_seed)
    result = ncv_estimator(network, price_cfg, n_training_paths=n_training)
    pricing_runtime = float(result.pricing_runtime_seconds)
    return {
        "method": "NCV_SCRATCH",
        "price": result.price,
        "observation_variance": result.observation_variance,
        "estimator_variance": result.estimator_variance,
        "std_error": result.std_error,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "pricing_observations": result.pricing_observations,
        "pricing_simulated_paths": result.pricing_simulated_paths,
        "pilot_paths": 0,
        "pilot_runtime_s": 0.0,
        "target_training_paths": n_training,
        "target_training_runtime_s": train_runtime,
        "shared_reference_training_paths": 0,
        "shared_reference_training_runtime_s": 0.0,
        "training_paths": n_training,
        "total_simulated_paths": result.total_simulated_paths,
        "beta": float("nan"),
        "corr_f_c0": float("nan"),
        "pricing_runtime_s": pricing_runtime,
        "marginal_runtime_s": pricing_runtime,
        "standalone_runtime_s": train_runtime + pricing_runtime,
        "hash_verified": "",
        "param_hash": "",
    }


def _expected_method_rows_per_replication() -> List[Tuple[str, str]]:
    rows: List[Tuple[str, str]] = []
    for cid in CONTRACT_IDS:
        for m in BASE_METHODS:
            rows.append((cid, m))
    for cid in TARGET_IDS:
        for m in TRANSFER_METHODS:
            rows.append((cid, m))
    return rows


def _replication_base_row(base_seed: int, replication: int, contract_id: str, method: str) -> dict:
    K, sigma, T = CONTRACT_GRID[contract_id]
    return {
        "base_seed": base_seed,
        "replication": replication,
        "contract_id": contract_id,
        "K": K,
        "sigma": sigma,
        "T": T,
        "method": method,
    }


def run_replication(
    base_seed: int,
    replication: int,
    n_training: int,
    n_pilot: int,
    n_pricing: int,
    torch_available: bool,
) -> Tuple[List[dict], List[dict], List[dict], dict]:
    seeds = _replication_seeds(base_seed, replication)

    per_rep_rows: List[dict] = []
    beta_rows: List[dict] = []
    runtime_rows: List[dict] = []

    shared_training_row = {
        "base_seed": base_seed,
        "replication": replication,
        "training_paths": n_training,
        "training_runtime_s": 0.0,
        "analytical_e_h0": "",
        "frozen_parameter_hash": "",
        "hash_verification_result": False,
        "error": "",
    }

    ref_network = None
    e_h0 = None
    ref_hash = None
    ref_train_runtime_s = 0.0

    if torch_available:
        try:
            from asian_options.frozen_transfer import compute_network_hash, train_reference_network

            ref_cfg = make_contract_cfg(REFERENCE_ID, n_paths=n_pricing, seed=seeds["ref_train"])
            ref_network, e_h0, ref_hash, ref_train_runtime_s = train_reference_network(
                ref_cfg,
                n_training=n_training,
                train_seed=seeds["ref_train"],
            )
            shared_training_row.update(
                {
                    "training_runtime_s": ref_train_runtime_s,
                    "analytical_e_h0": e_h0,
                    "frozen_parameter_hash": ref_hash,
                    "hash_verification_result": compute_network_hash(ref_network) == ref_hash,
                }
            )
        except Exception as exc:
            shared_training_row["error"] = str(exc)
    else:
        shared_training_row["error"] = "torch_not_available"

    for contract_id in CONTRACT_IDS:
        base_cfg = make_contract_cfg(contract_id, n_paths=n_pricing, seed=seeds[f"pricing_{contract_id}"])
        pricing_seed = seeds[f"pricing_{contract_id}"]
        pilot_seed = seeds[f"pilot_{contract_id}"]
        train_seed = seeds[f"target_train_{contract_id}"]

        # MC
        try:
            mc = standard_monte_carlo(dataclasses.replace(base_cfg, seed=pricing_seed))
            per_rep_rows.append(
                {
                    **_replication_base_row(base_seed, replication, contract_id, "MC"),
                    "price": mc.price,
                    "observation_variance": mc.observation_variance,
                    "estimator_variance": mc.estimator_variance,
                    "std_error": mc.std_error,
                    "ci_lower": mc.ci_lower,
                    "ci_upper": mc.ci_upper,
                    "pricing_observations": mc.pricing_observations,
                    "pricing_simulated_paths": mc.pricing_simulated_paths,
                    "pilot_paths": 0,
                    "pilot_runtime_s": 0.0,
                    "target_training_paths": 0,
                    "target_training_runtime_s": 0.0,
                    "shared_reference_training_paths": 0,
                    "shared_reference_training_runtime_s": 0.0,
                    "training_paths": 0,
                    "total_simulated_paths": mc.total_simulated_paths,
                    "pricing_runtime_s": mc.pricing_runtime_seconds,
                    "marginal_runtime_s": mc.pricing_runtime_seconds,
                    "standalone_runtime_s": mc.pricing_runtime_seconds,
                    "beta": float("nan"),
                    "corr_f_c0": float("nan"),
                    "param_hash": "",
                    "hash_verified": "",
                    "error": "",
                }
            )
        except Exception as exc:
            per_rep_rows.append({**_replication_base_row(base_seed, replication, contract_id, "MC"), "error": str(exc)})

        # AV
        try:
            av = antithetic_variates(dataclasses.replace(base_cfg, seed=pricing_seed))
            per_rep_rows.append(
                {
                    **_replication_base_row(base_seed, replication, contract_id, "AV"),
                    "price": av.price,
                    "observation_variance": av.observation_variance,
                    "estimator_variance": av.estimator_variance,
                    "std_error": av.std_error,
                    "ci_lower": av.ci_lower,
                    "ci_upper": av.ci_upper,
                    "pricing_observations": av.pricing_observations,
                    "pricing_simulated_paths": av.pricing_simulated_paths,
                    "pilot_paths": 0,
                    "pilot_runtime_s": 0.0,
                    "target_training_paths": 0,
                    "target_training_runtime_s": 0.0,
                    "shared_reference_training_paths": 0,
                    "shared_reference_training_runtime_s": 0.0,
                    "training_paths": 0,
                    "total_simulated_paths": av.total_simulated_paths,
                    "pricing_runtime_s": av.pricing_runtime_seconds,
                    "marginal_runtime_s": av.pricing_runtime_seconds,
                    "standalone_runtime_s": av.pricing_runtime_seconds,
                    "beta": float("nan"),
                    "corr_f_c0": float("nan"),
                    "param_hash": "",
                    "hash_verified": "",
                    "error": "",
                }
            )
        except Exception as exc:
            per_rep_rows.append({**_replication_base_row(base_seed, replication, contract_id, "AV"), "error": str(exc)})

        # GCV
        try:
            gcv = geometric_control_variate(
                dataclasses.replace(base_cfg, n_paths=n_pricing, seed=pricing_seed), n_pilot=n_pilot
            )
            pilot_runtime = float(gcv.training_runtime_seconds)
            pricing_runtime = float(gcv.pricing_runtime_seconds)
            per_rep_rows.append(
                {
                    **_replication_base_row(base_seed, replication, contract_id, "GCV"),
                    "price": gcv.price,
                    "observation_variance": gcv.observation_variance,
                    "estimator_variance": gcv.estimator_variance,
                    "std_error": gcv.std_error,
                    "ci_lower": gcv.ci_lower,
                    "ci_upper": gcv.ci_upper,
                    "pricing_observations": gcv.pricing_observations,
                    "pricing_simulated_paths": gcv.pricing_simulated_paths,
                    "pilot_paths": gcv.pilot_paths,
                    "pilot_runtime_s": pilot_runtime,
                    "target_training_paths": 0,
                    "target_training_runtime_s": 0.0,
                    "shared_reference_training_paths": 0,
                    "shared_reference_training_runtime_s": 0.0,
                    "training_paths": 0,
                    "total_simulated_paths": gcv.total_simulated_paths,
                    "pricing_runtime_s": pricing_runtime,
                    "marginal_runtime_s": pilot_runtime + pricing_runtime,
                    "standalone_runtime_s": pilot_runtime + pricing_runtime,
                    "beta": gcv.beta_hat,
                    "corr_f_c0": gcv.corr_estimate,
                    "param_hash": "",
                    "hash_verified": "",
                    "error": "",
                }
            )
        except Exception as exc:
            per_rep_rows.append({**_replication_base_row(base_seed, replication, contract_id, "GCV"), "error": str(exc)})

        # Scratch NCV
        if torch_available:
            try:
                scratch = _run_ncv_scratch(
                    base_cfg,
                    n_training=n_training,
                    train_seed=train_seed,
                    pricing_seed=pricing_seed,
                    n_pricing=n_pricing,
                )
                per_rep_rows.append({**_replication_base_row(base_seed, replication, contract_id, "NCV_SCRATCH"), **scratch, "error": ""})
            except Exception as exc:
                per_rep_rows.append({**_replication_base_row(base_seed, replication, contract_id, "NCV_SCRATCH"), "error": str(exc)})
        else:
            per_rep_rows.append(
                {
                    **_replication_base_row(base_seed, replication, contract_id, "NCV_SCRATCH"),
                    "error": "torch_not_available",
                }
            )

        if contract_id == REFERENCE_ID:
            continue

        # Transfer beta=1
        if torch_available and ref_network is not None and not shared_training_row.get("error"):
            try:
                from asian_options.frozen_transfer import ncv_transfer_beta1

                tb1 = ncv_transfer_beta1(
                    frozen_network=ref_network,
                    e_h0=e_h0,
                    frozen_hash=ref_hash,
                    target_cfg=base_cfg,
                    pricing_seed=pricing_seed,
                    n_pricing=n_pricing,
                    training_runtime_s=ref_train_runtime_s,
                )
                marginal = float(tb1.get("pricing_runtime_s", 0.0))
                per_rep_rows.append(
                    {
                        **_replication_base_row(base_seed, replication, contract_id, "NCV_TRANSFER_BETA1"),
                        **tb1,
                        "pilot_paths": 0,
                        "pilot_runtime_s": 0.0,
                        "target_training_paths": 0,
                        "target_training_runtime_s": 0.0,
                        "shared_reference_training_paths": n_training,
                        "shared_reference_training_runtime_s": ref_train_runtime_s,
                        "training_paths": 0,
                        "total_simulated_paths": tb1.get("pricing_simulated_paths", n_pricing),
                        "marginal_runtime_s": marginal,
                        "standalone_runtime_s": ref_train_runtime_s + marginal,
                        "error": "",
                    }
                )
                beta_rows.append(
                    {
                        **_replication_base_row(base_seed, replication, contract_id, "NCV_TRANSFER_BETA1"),
                        "estimated_beta": 1.0,
                        "payoff_control_correlation": tb1.get("corr_f_c0"),
                        "payoff_variance": tb1.get("payoff_variance"),
                        "control_variance": tb1.get("control_variance"),
                        "payoff_control_covariance": tb1.get("payoff_control_covariance"),
                        "optimal_residual_variance": tb1.get("optimal_residual_variance"),
                        "observed_residual_variance": tb1.get("observation_variance"),
                        "residual_variance_beta_one": tb1.get("observation_variance"),
                        "variance_improvement_from_estimating_beta": 1.0,
                        "parameter_hash": tb1.get("param_hash"),
                        "hash_verification": tb1.get("hash_verified"),
                        "target_training_paths": 0,
                        "pilot_paths": 0,
                        "pricing_paths": tb1.get("pricing_simulated_paths"),
                        "marginal_runtime_s": marginal,
                        "error": "",
                    }
                )
            except Exception as exc:
                per_rep_rows.append(
                    {**_replication_base_row(base_seed, replication, contract_id, "NCV_TRANSFER_BETA1"), "error": str(exc)}
                )
                beta_rows.append(
                    {**_replication_base_row(base_seed, replication, contract_id, "NCV_TRANSFER_BETA1"), "error": str(exc)}
                )
        else:
            reason = "torch_not_available" if not torch_available else "ref_network_failed"
            per_rep_rows.append({**_replication_base_row(base_seed, replication, contract_id, "NCV_TRANSFER_BETA1"), "error": reason})
            beta_rows.append({**_replication_base_row(base_seed, replication, contract_id, "NCV_TRANSFER_BETA1"), "error": reason})

        # Transfer beta estimated
        if torch_available and ref_network is not None and not shared_training_row.get("error"):
            try:
                from asian_options.frozen_transfer import ncv_transfer_beta

                tb = ncv_transfer_beta(
                    frozen_network=ref_network,
                    e_h0=e_h0,
                    frozen_hash=ref_hash,
                    target_cfg=base_cfg,
                    pilot_seed=pilot_seed,
                    pricing_seed=pricing_seed,
                    n_pilot=n_pilot,
                    n_pricing=n_pricing,
                    training_runtime_s=ref_train_runtime_s,
                )
                pilot_runtime = float(tb.get("pilot_runtime_s", 0.0))
                pricing_runtime = float(tb.get("pricing_runtime_s", 0.0))
                marginal = pilot_runtime + pricing_runtime
                per_rep_rows.append(
                    {
                        **_replication_base_row(base_seed, replication, contract_id, "NCV_TRANSFER_BETA"),
                        **tb,
                        "target_training_paths": 0,
                        "target_training_runtime_s": 0.0,
                        "shared_reference_training_paths": n_training,
                        "shared_reference_training_runtime_s": ref_train_runtime_s,
                        "training_paths": 0,
                        "marginal_runtime_s": marginal,
                        "standalone_runtime_s": ref_train_runtime_s + marginal,
                        "error": "",
                    }
                )
                beta_rows.append(
                    {
                        **_replication_base_row(base_seed, replication, contract_id, "NCV_TRANSFER_BETA"),
                        "estimated_beta": tb.get("beta"),
                        "payoff_control_correlation": tb.get("corr_f_c0"),
                        "payoff_variance": tb.get("payoff_variance"),
                        "control_variance": tb.get("control_variance"),
                        "payoff_control_covariance": tb.get("payoff_control_covariance"),
                        "optimal_residual_variance": tb.get("optimal_residual_variance"),
                        "observed_residual_variance": tb.get("observation_variance"),
                        "residual_variance_beta_one": tb.get("residual_variance_beta_one"),
                        "variance_improvement_from_estimating_beta": tb.get("variance_improvement_from_estimating_beta"),
                        "parameter_hash": tb.get("param_hash"),
                        "hash_verification": tb.get("hash_verified"),
                        "target_training_paths": 0,
                        "pilot_paths": n_pilot,
                        "pricing_paths": tb.get("pricing_simulated_paths"),
                        "marginal_runtime_s": marginal,
                        "error": "",
                    }
                )
            except Exception as exc:
                per_rep_rows.append(
                    {**_replication_base_row(base_seed, replication, contract_id, "NCV_TRANSFER_BETA"), "error": str(exc)}
                )
                beta_rows.append(
                    {**_replication_base_row(base_seed, replication, contract_id, "NCV_TRANSFER_BETA"), "error": str(exc)}
                )
        else:
            reason = "torch_not_available" if not torch_available else "ref_network_failed"
            per_rep_rows.append({**_replication_base_row(base_seed, replication, contract_id, "NCV_TRANSFER_BETA"), "error": reason})
            beta_rows.append({**_replication_base_row(base_seed, replication, contract_id, "NCV_TRANSFER_BETA"), "error": reason})

    for row in per_rep_rows:
        if row.get("error"):
            continue
        runtime_rows.append(
            {
                "base_seed": row.get("base_seed"),
                "replication": row.get("replication"),
                "contract_id": row.get("contract_id"),
                "method": row.get("method"),
                "target_training_runtime_s": row.get("target_training_runtime_s", 0.0),
                "pilot_runtime_s": row.get("pilot_runtime_s", 0.0),
                "pricing_runtime_s": row.get("pricing_runtime_s", 0.0),
                "marginal_runtime_s": row.get("marginal_runtime_s", 0.0),
                "standalone_runtime_s": row.get("standalone_runtime_s", 0.0),
                "pricing_observations": row.get("pricing_observations", 0),
            }
        )

    return per_rep_rows, beta_rows, runtime_rows, shared_training_row


def _build_seed_manifest(base_seed: int, n_replications: int) -> List[dict]:
    rows: List[dict] = []
    for rep in range(n_replications):
        seeds = _replication_seeds(base_seed, rep)
        for stream, value in seeds.items():
            rows.append({"base_seed": base_seed, "replication": rep, "stream": stream, "seed_value": value})
    return rows


def compute_all_high_precision_references(n_paths: int, base_seed: int) -> List[dict]:
    from asian_options.frozen_transfer import compute_high_precision_reference

    rows: List[dict] = []
    for i, cid in enumerate(CONTRACT_IDS):
        seed = base_seed + SEED_OFFSET_HIGH_PREC + i * 1000
        try:
            rows.append(compute_high_precision_reference(cid, n_paths=n_paths, seed=seed))
        except Exception as exc:
            rows.append(
                {
                    "contract_id": cid,
                    "price": "ERROR",
                    "std_error": "ERROR",
                    "ci_lower": "ERROR",
                    "ci_upper": "ERROR",
                    "n_paths": n_paths,
                    "method": "GCV_high_precision",
                    "seed": seed,
                    "error": str(exc),
                }
            )
    return rows


def _all_required_rows_present(per_rep_rows: List[dict], n_replications: int) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    lookup = defaultdict(list)
    for row in per_rep_rows:
        lookup[(row.get("replication"), row.get("contract_id"), row.get("method"))].append(row)
    for rep in range(n_replications):
        for cid, method in _expected_method_rows_per_replication():
            k = (rep, cid, method)
            rows = lookup.get(k, [])
            if len(rows) != 1:
                failures.append(f"expected exactly one row for rep={rep}, contract={cid}, method={method}, got {len(rows)}")
    return len(failures) == 0, failures


def _check_required_success(per_rep_rows: List[dict], n_replications: int) -> Tuple[bool, List[str]]:
    failures: List[str] = []
    ok_count, present_failures = _all_required_rows_present(per_rep_rows, n_replications)
    failures.extend(present_failures)

    for row in per_rep_rows:
        if row.get("error"):
            failures.append(
                f"failed row: rep={row.get('replication')}, contract={row.get('contract_id')}, method={row.get('method')}, error={row.get('error')}"
            )
            continue

        for field in (
            "price",
            "observation_variance",
            "estimator_variance",
            "std_error",
            "pricing_runtime_s",
            "marginal_runtime_s",
            "standalone_runtime_s",
        ):
            if _safe_float(row.get(field)) is None:
                failures.append(
                    f"non-finite {field}: rep={row.get('replication')}, contract={row.get('contract_id')}, method={row.get('method')}"
                )

        if row.get("method") in TRANSFER_METHODS and row.get("hash_verified") is not True:
            failures.append(
                f"hash verification failed: rep={row.get('replication')}, contract={row.get('contract_id')}, method={row.get('method')}"
            )

    return ok_count and not failures, failures


def compute_variance_ratio_summary(per_rep_rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    rep_rows = [r for r in per_rep_rows if not r.get("error")]
    by_key = {(r["replication"], r["contract_id"], r["method"]): r for r in rep_rows}

    per_rep_vrr: List[dict] = []
    for r in rep_rows:
        rep = r["replication"]
        cid = r["contract_id"]
        m = r["method"]
        method_var = _safe_float(r.get("observation_variance"))
        mc_var = _safe_float(by_key.get((rep, cid, "MC"), {}).get("observation_variance"))
        gcv_var = _safe_float(by_key.get((rep, cid, "GCV"), {}).get("observation_variance"))

        for comp_name, comp_var in (("MC", mc_var), ("GCV", gcv_var)):
            valid = True
            reason = ""
            ratio = None
            if method_var is None or comp_var is None:
                valid = False
                reason = "non_finite_variance"
            elif method_var <= 0.0 or comp_var <= 0.0:
                valid = False
                reason = "non_positive_variance"
            else:
                ratio = comp_var / method_var
                if not math.isfinite(ratio):
                    valid = False
                    reason = "non_finite_ratio"
            per_rep_vrr.append(
                {
                    "contract_id": cid,
                    "replication": rep,
                    "method": m,
                    "comparator": comp_name,
                    "variance_ratio": ratio if valid else "NA",
                    "is_valid": valid,
                    "invalid_reason": reason,
                    "beats_comparator": bool(valid and ratio is not None and ratio > 1.0),
                }
            )

    summary_rows: List[dict] = []
    grouped = defaultdict(list)
    for r in per_rep_vrr:
        grouped[(r["contract_id"], r["method"], r["comparator"])].append(r)

    for (cid, method, comp), rows in sorted(grouped.items()):
        valid_vals = [float(r["variance_ratio"]) for r in rows if r["is_valid"]]
        logs = [math.log(v) for v in valid_vals if v > 0 and math.isfinite(v)]
        n_valid = len(valid_vals)
        n_total = len(rows)
        beat_count = sum(1 for r in rows if r.get("beats_comparator"))

        log_mean = statistics.mean(logs) if logs else None
        log_sd = statistics.stdev(logs) if len(logs) >= 2 else None
        if log_mean is not None and len(logs) >= 2 and log_sd is not None:
            half = 1.96 * log_sd / math.sqrt(len(logs))
            ci_low = math.exp(log_mean - half)
            ci_high = math.exp(log_mean + half)
        elif log_mean is not None and len(logs) == 1:
            ci_low = math.exp(log_mean)
            ci_high = math.exp(log_mean)
        else:
            ci_low = None
            ci_high = None

        geom = math.exp(log_mean) if log_mean is not None else None

        summary_rows.append(
            {
                "contract_id": cid,
                "method": method,
                "comparator": comp,
                "arithmetic_mean": _mean(valid_vals),
                "median": _median(valid_vals),
                "geometric_mean": geom,
                "std_dev": _std(valid_vals),
                "minimum": min(valid_vals) if valid_vals else "NA",
                "maximum": max(valid_vals) if valid_vals else "NA",
                "log_vrr_ci95_lower": ci_low if ci_low is not None else "NA",
                "log_vrr_ci95_upper": ci_high if ci_high is not None else "NA",
                "n_valid_replications": n_valid,
                "n_invalid_replications": n_total - n_valid,
                "beats_comparator_count": beat_count,
                "beats_comparator_percentage": (100.0 * beat_count / n_total) if n_total else "NA",
            }
        )

    return per_rep_vrr, summary_rows


def aggregate_statistical_results(
    per_rep_rows: List[dict],
    high_prec_rows: List[dict],
    variance_ratio_summary: List[dict],
    n_replications: int,
) -> List[dict]:
    refs = {r["contract_id"]: r for r in high_prec_rows if not r.get("error")}

    grouped = defaultdict(list)
    errors = defaultdict(list)
    for r in per_rep_rows:
        key = (r.get("contract_id"), r.get("method"))
        if r.get("error"):
            errors[key].append(str(r.get("error")))
        else:
            grouped[key].append(r)

    vrr_lookup = {(r["contract_id"], r["method"], r["comparator"]): r for r in variance_ratio_summary}

    rows: List[dict] = []
    all_pairs = {(cid, m) for cid in CONTRACT_IDS for m in BASE_METHODS} | {(cid, m) for cid in TARGET_IDS for m in TRANSFER_METHODS}

    for cid, method in sorted(all_pairs):
        vals = grouped.get((cid, method), [])
        succ = len(vals)
        fail = n_replications - succ

        prices = [_safe_float(v.get("price")) for v in vals]
        prices = [p for p in prices if p is not None]
        obs_vars = [_safe_float(v.get("observation_variance")) for v in vals]
        obs_vars = [v for v in obs_vars if v is not None]
        est_vars = [_safe_float(v.get("estimator_variance")) for v in vals]
        est_vars = [v for v in est_vars if v is not None]
        std_errs = [_safe_float(v.get("std_error")) for v in vals]
        std_errs = [v for v in std_errs if v is not None]
        pricing_obs = [_safe_float(v.get("pricing_observations")) for v in vals]
        pricing_obs = [v for v in pricing_obs if v is not None]
        pricing_sim = [_safe_float(v.get("pricing_simulated_paths")) for v in vals]
        pricing_sim = [v for v in pricing_sim if v is not None]
        pilot_paths = [_safe_float(v.get("pilot_paths")) for v in vals]
        pilot_paths = [v for v in pilot_paths if v is not None]
        target_train_paths = [_safe_float(v.get("target_training_paths")) for v in vals]
        target_train_paths = [v for v in target_train_paths if v is not None]
        ci_widths = [(_safe_float(v.get("ci_upper")), _safe_float(v.get("ci_lower"))) for v in vals]
        ci_widths = [u - l for u, l in ci_widths if u is not None and l is not None]

        ref = refs.get(cid)
        ref_price = _safe_float(ref.get("price")) if ref else None
        abs_errors = [abs(p - ref_price) for p in prices] if ref_price is not None else []
        sq_errors = [(p - ref_price) ** 2 for p in prices] if ref_price is not None else []

        coverage_count = 0
        coverage_denom = 0
        if ref_price is not None:
            for v in vals:
                lo, hi = _safe_float(v.get("ci_lower")), _safe_float(v.get("ci_upper"))
                if lo is not None and hi is not None:
                    coverage_denom += 1
                    if lo <= ref_price <= hi:
                        coverage_count += 1

        mean_price = _mean(prices)
        empirical_std = _std(prices)
        empirical_se = (empirical_std / math.sqrt(succ)) if (empirical_std is not None and succ > 0) else None

        mean_total_paths = _mean([_safe_float(v.get("total_simulated_paths")) for v in vals])
        mean_standalone_runtime = _mean([_safe_float(v.get("standalone_runtime_s")) for v in vals])

        mc_group = grouped.get((cid, "MC"), [])
        mc_paths = _mean([_safe_float(v.get("total_simulated_paths")) for v in mc_group])
        mc_runtime = _mean([_safe_float(v.get("standalone_runtime_s")) for v in mc_group])

        path_eff = (mc_paths / mean_total_paths) if (mc_paths and mean_total_paths) else "NA"
        runtime_eff = (mc_runtime / mean_standalone_runtime) if (mc_runtime and mean_standalone_runtime) else "NA"

        row = {
            "contract_id": cid,
            "method": method,
            "successful_replications": succ,
            "failed_replications": fail,
            "failure_messages": " | ".join(sorted(set(errors.get((cid, method), [])))) if errors.get((cid, method)) else "",
            "mean_price": mean_price if mean_price is not None else "NA",
            "median_price": _median(prices) if prices else "NA",
            "empirical_std_across_replications": empirical_std if empirical_std is not None else "NA",
            "empirical_standard_error_of_mean": empirical_se if empirical_se is not None else "NA",
            "mean_reported_estimator_standard_error": _mean(std_errs) if std_errs else "NA",
            "pricing_observations_mean": _mean(pricing_obs) if pricing_obs else "NA",
            "pricing_simulated_paths_mean": _mean(pricing_sim) if pricing_sim else "NA",
            "pilot_paths_mean": _mean(pilot_paths) if pilot_paths else "NA",
            "target_training_paths_mean": _mean(target_train_paths) if target_train_paths else "NA",
            "mean_observation_variance": _mean(obs_vars) if obs_vars else "NA",
            "mean_estimator_variance": _mean(est_vars) if est_vars else "NA",
            "bias_against_reference_estimate": (mean_price - ref_price) if (mean_price is not None and ref_price is not None) else "NA",
            "absolute_bias": abs(mean_price - ref_price) if (mean_price is not None and ref_price is not None) else "NA",
            "mae": _mean(abs_errors) if abs_errors else "NA",
            "rmse": math.sqrt(_mean(sq_errors)) if sq_errors else "NA",
            "mean_confidence_interval_width": _mean(ci_widths) if ci_widths else "NA",
            "naive_coverage_against_estimated_reference": (
                coverage_count / coverage_denom if coverage_denom else "NA"
            ),
            "naive_coverage_label": "naive_coverage_vs_estimated_reference_not_true_coverage",
            "variance_reduction_vs_mc_arithmetic_mean": (
                vrr_lookup.get((cid, method, "MC"), {}).get("arithmetic_mean", "NA")
            ),
            "variance_reduction_vs_gcv_arithmetic_mean": (
                vrr_lookup.get((cid, method, "GCV"), {}).get("arithmetic_mean", "NA")
            ),
            "path_efficiency_vs_mc": path_eff,
            "runtime_efficiency_vs_mc": runtime_eff,
        }
        rows.append(row)

    return rows


def compute_equal_pricing_observations_summary(
    aggregate_rows: List[dict],
    variance_ratio_summary: List[dict],
    n_training: int,
) -> List[dict]:
    agg = {(r["contract_id"], r["method"]): r for r in aggregate_rows}
    vrr = {(r["contract_id"], r["method"], r["comparator"]): r for r in variance_ratio_summary}

    rows: List[dict] = []
    for cid in CONTRACT_IDS:
        methods = list(BASE_METHODS) + (list(TRANSFER_METHODS) if cid in TARGET_IDS else [])
        for method in methods:
            r = agg.get((cid, method), {})
            pricing_obs = r.get("pricing_observations_mean", r.get("pricing_observations", "NA"))
            pricing_sim = r.get("pricing_simulated_paths_mean", r.get("pricing_simulated_paths", "NA"))
            pilot_paths = r.get("pilot_paths_mean", 0)
            target_training_paths = r.get("target_training_paths_mean", 0)
            shared_paths = n_training if method in TRANSFER_METHODS else 0

            if method == "MC":
                pricing_obs = pricing_obs if pricing_obs != "NA" else 50_000
                pricing_sim = pricing_sim if pricing_sim != "NA" else pricing_obs
                pilot_paths = 0
                target_training_paths = 0
            elif method == "AV":
                pricing_obs = pricing_obs if pricing_obs != "NA" else 50_000
                pricing_sim = pricing_sim if pricing_sim != "NA" else 2 * pricing_obs
                pilot_paths = 0
                target_training_paths = 0
            elif method == "GCV":
                pilot_paths = pilot_paths if pilot_paths else 1000
                pricing_obs = pricing_obs if pricing_obs != "NA" else 50_000
                pricing_sim = pricing_sim if pricing_sim != "NA" else pricing_obs
                target_training_paths = 0
            elif method == "NCV_SCRATCH":
                target_training_paths = target_training_paths if target_training_paths else n_training
                pricing_obs = pricing_obs if pricing_obs != "NA" else 50_000
                pricing_sim = pricing_sim if pricing_sim != "NA" else pricing_obs
                pilot_paths = 0
            elif method == "NCV_TRANSFER_BETA1":
                target_training_paths = 0
                pilot_paths = 0
                pricing_obs = pricing_obs if pricing_obs != "NA" else 50_000
                pricing_sim = pricing_sim if pricing_sim != "NA" else pricing_obs
            elif method == "NCV_TRANSFER_BETA":
                target_training_paths = 0
                pilot_paths = pilot_paths if pilot_paths else 1000
                pricing_obs = pricing_obs if pricing_obs != "NA" else 50_000
                pricing_sim = pricing_sim if pricing_sim != "NA" else pricing_obs

            total_standalone = (pricing_sim if isinstance(pricing_sim, (int, float)) else 0) + (
                pilot_paths if isinstance(pilot_paths, (int, float)) else 0
            ) + (
                target_training_paths if isinstance(target_training_paths, (int, float)) else 0
            ) + shared_paths

            rows.append(
                {
                    "contract_id": cid,
                    "method": method,
                    "pricing_observations": pricing_obs,
                    "pricing_simulated_paths": pricing_sim,
                    "training_paths": target_training_paths,
                    "pilot_paths": pilot_paths,
                    "one_time_shared_paths": shared_paths,
                    "total_paths_one_standalone_valuation": total_standalone,
                    "observation_variance": r.get("mean_observation_variance", "NA"),
                    "estimator_variance": r.get("mean_estimator_variance", "NA"),
                    "standard_error": r.get("mean_reported_estimator_standard_error", "NA"),
                    "pricing_runtime_s": r.get("pricing_runtime_s_median", r.get("pricing_runtime_s_mean", "NA")),
                    "marginal_runtime_s": r.get("marginal_runtime_s_median", r.get("standalone_runtime_s_mean", "NA")),
                    "standalone_runtime_s": r.get("standalone_runtime_s_median", r.get("standalone_runtime_s_mean", "NA")),
                    "vrr_against_mc": vrr.get((cid, method, "MC"), {}).get("arithmetic_mean", "NA"),
                    "vrr_against_gcv": vrr.get((cid, method, "GCV"), {}).get("arithmetic_mean", "NA"),
                }
            )
    return rows


def _required_observations(obs_variance: Optional[float], target_se: Optional[float]) -> Tuple[Optional[int], str]:
    if obs_variance is None or target_se is None or not math.isfinite(obs_variance) or not math.isfinite(target_se):
        return None, "missing_or_non_finite_input"
    if obs_variance <= 0 or target_se <= 0:
        return None, "non_positive_variance_or_target_se"
    return max(2, int(math.ceil(obs_variance / (target_se ** 2)))), ""


def _runtime_per_observation(rows: List[dict], contract_id: str, method: str) -> Tuple[Optional[float], Optional[float]]:
    vals = []
    for r in rows:
        if r.get("contract_id") == contract_id and r.get("method") == method and not r.get("error"):
            rt = _safe_float(r.get("pricing_runtime_s"))
            obs = _safe_float(r.get("pricing_observations"))
            if rt is not None and obs is not None and obs > 0:
                vals.append(rt / obs)
    return _median(vals), _mean(vals)


def compute_matched_accuracy_results(
    aggregate_rows: List[dict],
    per_rep_rows: List[dict],
    n_training: int,
    n_pilot: int,
    shared_train_runtime_median: Optional[float],
) -> List[dict]:
    agg = {(r["contract_id"], r["method"]): r for r in aggregate_rows}
    rows: List[dict] = []

    for cid in CONTRACT_IDS:
        gcv_ref_se = _safe_float(agg.get((cid, "GCV"), {}).get("mean_reported_estimator_standard_error"))
        scratch_ref_se = _safe_float(agg.get((cid, "NCV_SCRATCH"), {}).get("mean_reported_estimator_standard_error"))

        targets = [
            ("match_gcv_50000_se", gcv_ref_se),
            ("match_scratch_50000_se", scratch_ref_se),
            ("fixed_se_0.001", 0.001),
        ]

        methods = list(BASE_METHODS) + (list(TRANSFER_METHODS) if cid in TARGET_IDS else [])
        for method in methods:
            obs_var = _safe_float(agg.get((cid, method), {}).get("mean_observation_variance"))
            rt_per_obs_median, rt_per_obs_mean = _runtime_per_observation(per_rep_rows, cid, method)

            for target_definition, target_se in targets:
                n_required, reason = _required_observations(obs_var, target_se)
                feasible = n_required is not None

                if method == "AV":
                    required_sim_paths = (2 * n_required) if n_required is not None else "NA"
                else:
                    required_sim_paths = n_required if n_required is not None else "NA"

                training_paths = n_training if method == "NCV_SCRATCH" else 0
                pilot_paths = n_pilot if method in ("GCV", "NCV_TRANSFER_BETA") else 0
                shared_paths = n_training if method in TRANSFER_METHODS else 0

                one_time_runtime = 0.0
                if method == "NCV_SCRATCH":
                    one_time_runtime = _safe_float(agg.get((cid, method), {}).get("target_training_runtime_s_median")) or 0.0
                if method in TRANSFER_METHODS:
                    one_time_runtime = shared_train_runtime_median or 0.0

                projected_pricing_runtime = (
                    (rt_per_obs_median * n_required) if (feasible and rt_per_obs_median is not None) else None
                )
                projected_pricing_runtime_mean = (
                    (rt_per_obs_mean * n_required) if (feasible and rt_per_obs_mean is not None) else None
                )

                marginal_runtime = projected_pricing_runtime
                if marginal_runtime is not None and method in ("GCV", "NCV_TRANSFER_BETA"):
                    pilot_rt = _safe_float(agg.get((cid, method), {}).get("pilot_runtime_s_median"))
                    if pilot_rt is not None:
                        marginal_runtime += pilot_rt

                standalone_runtime = (one_time_runtime + marginal_runtime) if marginal_runtime is not None else None

                rows.append(
                    {
                        "contract_id": cid,
                        "method": method,
                        "target_definition": target_definition,
                        "target_standard_error": target_se if target_se is not None else "NA",
                        "required_pricing_observations": n_required if feasible else "NA",
                        "required_simulated_paths": required_sim_paths,
                        "training_paths": training_paths,
                        "pilot_paths": pilot_paths,
                        "shared_paths": shared_paths,
                        "projected_pricing_runtime_s_median": projected_pricing_runtime if projected_pricing_runtime is not None else "NA",
                        "projected_pricing_runtime_s_mean_sensitivity": (
                            projected_pricing_runtime_mean if projected_pricing_runtime_mean is not None else "NA"
                        ),
                        "marginal_runtime_s": marginal_runtime if marginal_runtime is not None else "NA",
                        "one_time_runtime_s": one_time_runtime,
                        "standalone_runtime_s": standalone_runtime if standalone_runtime is not None else "NA",
                        "feasible": feasible,
                        "failure_reason": reason,
                    }
                )

    return rows


def _equal_budget_allocation(method: str, B: int, Q: int, n_training: int, n_pilot: int) -> dict:
    if method == "MC":
        n = B
        sim = n
        total = Q * sim
        return {
            "pricing_observations": n,
            "pricing_simulated_paths": sim,
            "training_paths": 0,
            "pilot_paths": 0,
            "shared_paths": 0,
            "total_paths_used": total,
            "feasible": n >= 2,
            "failure_reason": "" if n >= 2 else "insufficient_budget_for_minimum_observations",
        }

    if method == "AV":
        n = B // 2
        sim = 2 * n
        total = Q * sim
        return {
            "pricing_observations": n,
            "pricing_simulated_paths": sim,
            "training_paths": 0,
            "pilot_paths": 0,
            "shared_paths": 0,
            "total_paths_used": total,
            "feasible": n >= 2,
            "failure_reason": "" if n >= 2 else "insufficient_budget_for_antithetic_pairs",
        }

    if method == "GCV":
        n = B - n_pilot
        sim = n
        total = Q * (n_pilot + sim)
        feasible = n >= 2
        return {
            "pricing_observations": n if feasible else "NA",
            "pricing_simulated_paths": sim if feasible else "NA",
            "training_paths": 0,
            "pilot_paths": n_pilot,
            "shared_paths": 0,
            "total_paths_used": total,
            "feasible": feasible,
            "failure_reason": "" if feasible else "pilot_inside_budget_leaves_insufficient_pricing_paths",
        }

    if method == "NCV_SCRATCH":
        n = math.floor((Q * B - n_training) / Q)
        sim = n
        total = n_training + Q * sim
        feasible = n >= 2
        return {
            "pricing_observations": n if feasible else "NA",
            "pricing_simulated_paths": sim if feasible else "NA",
            "training_paths": n_training,
            "pilot_paths": 0,
            "shared_paths": 0,
            "total_paths_used": total,
            "feasible": feasible,
            "failure_reason": "" if feasible else "training_cost_exceeds_budget",
        }

    if method == "NCV_TRANSFER_BETA1":
        n = math.floor((Q * B - n_training) / Q)
        sim = n
        total = n_training + Q * sim
        feasible = n >= 2
        return {
            "pricing_observations": n if feasible else "NA",
            "pricing_simulated_paths": sim if feasible else "NA",
            "training_paths": 0,
            "pilot_paths": 0,
            "shared_paths": n_training,
            "total_paths_used": total,
            "feasible": feasible,
            "failure_reason": "" if feasible else "shared_reference_training_cost_exceeds_budget",
        }

    if method == "NCV_TRANSFER_BETA":
        n = math.floor((Q * B - n_training - Q * n_pilot) / Q)
        sim = n
        total = n_training + Q * (n_pilot + sim)
        feasible = n >= 2
        return {
            "pricing_observations": n if feasible else "NA",
            "pricing_simulated_paths": sim if feasible else "NA",
            "training_paths": 0,
            "pilot_paths": n_pilot,
            "shared_paths": n_training,
            "total_paths_used": total,
            "feasible": feasible,
            "failure_reason": "" if feasible else "shared_training_plus_pilot_cost_exceeds_budget",
        }

    return {
        "pricing_observations": "NA",
        "pricing_simulated_paths": "NA",
        "training_paths": 0,
        "pilot_paths": 0,
        "shared_paths": 0,
        "total_paths_used": "NA",
        "feasible": False,
        "failure_reason": "unknown_method",
    }


def compute_equal_budget_projected_results(
    aggregate_rows: List[dict],
    n_training: int,
    n_pilot: int,
    budget: int,
    q_values: List[int],
) -> List[dict]:
    agg = {(r["contract_id"], r["method"]): r for r in aggregate_rows}
    rows: List[dict] = []

    for Q in q_values:
        for cid in CONTRACT_IDS:
            methods = list(BASE_METHODS) + (list(TRANSFER_METHODS) if cid in TARGET_IDS else [])
            for method in methods:
                alloc = _equal_budget_allocation(method, budget, Q, n_training, n_pilot)
                obs_var = _safe_float(agg.get((cid, method), {}).get("mean_observation_variance"))
                n_obs = alloc.get("pricing_observations")
                if isinstance(n_obs, (int, float)) and n_obs > 0 and obs_var is not None:
                    projected_se = math.sqrt(obs_var / n_obs)
                else:
                    projected_se = "NA"

                rows.append(
                    {
                        "result_type": "projected",
                        "contract_id": cid,
                        "method": method,
                        "Q": Q,
                        "budget_paths_per_valuation": budget,
                        **alloc,
                        "total_budget_paths": Q * budget,
                        "budget_constraint_satisfied": (
                            bool(alloc.get("total_paths_used", 0) <= Q * budget)
                            if isinstance(alloc.get("total_paths_used"), (int, float))
                            else False
                        ),
                        "projected_estimator_standard_error": projected_se,
                    }
                )

    return rows


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def compute_reference_precision_diagnostics(
    aggregate_rows: List[dict],
    high_prec_rows: List[dict],
) -> List[dict]:
    refs = {r["contract_id"]: r for r in high_prec_rows if not r.get("error")}
    rows: List[dict] = []
    for r in aggregate_rows:
        cid = r["contract_id"]
        ref = refs.get(cid)
        ref_price = _safe_float(ref.get("price")) if ref else None
        ref_se = _safe_float(ref.get("std_error")) if ref else None

        mean_price = _safe_float(r.get("mean_price"))
        empirical_std = _safe_float(r.get("empirical_std_across_replications"))
        n = int(r.get("successful_replications", 0) or 0)
        method_se_mean = _safe_float(r.get("mean_reported_estimator_standard_error"))

        if mean_price is None or ref_price is None or ref_se is None or empirical_std is None or n <= 0:
            rows.append(
                {
                    "contract_id": cid,
                    "method": r["method"],
                    "se_combined": "NA",
                    "z_combined": "NA",
                    "p_value_two_sided": "NA",
                    "difference_ci95_lower": "NA",
                    "difference_ci95_upper": "NA",
                    "reference_se_divided_by_method_reported_se": "NA",
                    "reference_se_below_10pct_of_method_se": "NA",
                    "failure_reason": "missing_input",
                }
            )
            continue

        se_combined = math.sqrt((empirical_std ** 2) / n + ref_se ** 2)
        diff = mean_price - ref_price
        z = diff / se_combined if se_combined > 0 else float("nan")
        p = 2.0 * (1.0 - _normal_cdf(abs(z))) if math.isfinite(z) else float("nan")

        ratio = (ref_se / method_se_mean) if (method_se_mean is not None and method_se_mean > 0) else None

        rows.append(
            {
                "contract_id": cid,
                "method": r["method"],
                "se_combined": se_combined,
                "z_combined": z,
                "p_value_two_sided": p,
                "difference_ci95_lower": diff - 1.96 * se_combined,
                "difference_ci95_upper": diff + 1.96 * se_combined,
                "reference_se_divided_by_method_reported_se": ratio if ratio is not None else "NA",
                "reference_se_below_10pct_of_method_se": bool(ratio < 0.1) if ratio is not None else "NA",
                "failure_reason": "",
            }
        )

    return rows


def _solve_break_even(initial_cost: Optional[float], baseline_marginal: Optional[float], proposed_marginal: Optional[float]) -> dict:
    if initial_cost is None or baseline_marginal is None or proposed_marginal is None:
        return {
            "break_even_q": "NA",
            "failure_reason": "missing_runtime_input",
            "verified_q_minus_1": False,
            "verified_q": False,
        }

    if initial_cost < 0:
        return {
            "break_even_q": "NA",
            "failure_reason": "negative_initial_cost",
            "verified_q_minus_1": False,
            "verified_q": False,
        }

    denom = baseline_marginal - proposed_marginal
    if denom == 0:
        return {
            "break_even_q": "NA",
            "failure_reason": "positive_initial_cost_and_equal_marginal_cost" if initial_cost > 0 else "equal_marginal_cost",
            "verified_q_minus_1": False,
            "verified_q": False,
        }
    if denom < 0:
        return {
            "break_even_q": "NA",
            "failure_reason": "positive_initial_cost_and_higher_marginal_cost",
            "verified_q_minus_1": False,
            "verified_q": False,
        }

    q = max(1, int(math.ceil(initial_cost / denom)))

    def c_base(Q: int) -> float:
        return Q * baseline_marginal

    def c_prop(Q: int) -> float:
        return initial_cost + Q * proposed_marginal

    q_minus_1 = max(1, q - 1)
    verify_prev = c_prop(q_minus_1) > c_base(q_minus_1)
    verify_now = c_prop(q) <= c_base(q)
    if not (verify_prev and verify_now):
        return {
            "break_even_q": "NA",
            "failure_reason": "verification_failed_at_q_minus_1_or_q",
            "verified_q_minus_1": verify_prev,
            "verified_q": verify_now,
        }

    return {
        "break_even_q": q,
        "failure_reason": "",
        "verified_q_minus_1": verify_prev,
        "verified_q": verify_now,
        "cost_baseline_q_minus_1": c_base(q_minus_1),
        "cost_proposed_q_minus_1": c_prop(q_minus_1),
        "cost_baseline_q": c_base(q),
        "cost_proposed_q": c_prop(q),
    }


def compute_break_even_tables(
    aggregate_rows: List[dict],
    matched_accuracy_rows: List[dict],
    shared_training_rows: List[dict],
) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
    agg = {(r["contract_id"], r["method"]): r for r in aggregate_rows}

    shared_train_median = _median([_safe_float(r.get("training_runtime_s")) for r in shared_training_rows if not r.get("error")]) or 0.0

    equal_obs_rows: List[dict] = []
    for cid in TARGET_IDS:
        gcv = _safe_float(agg.get((cid, "GCV"), {}).get("marginal_runtime_s_median"))
        gcv_mean = _safe_float(agg.get((cid, "GCV"), {}).get("marginal_runtime_s_mean"))
        scratch_marg = _safe_float(agg.get((cid, "NCV_SCRATCH"), {}).get("marginal_runtime_s_median"))
        scratch_marg_mean = _safe_float(agg.get((cid, "NCV_SCRATCH"), {}).get("marginal_runtime_s_mean"))
        scratch_init = _safe_float(agg.get((cid, "NCV_SCRATCH"), {}).get("target_training_runtime_s_median"))
        scratch_init_mean = _safe_float(agg.get((cid, "NCV_SCRATCH"), {}).get("target_training_runtime_s_mean"))
        tb1_marg = _safe_float(agg.get((cid, "NCV_TRANSFER_BETA1"), {}).get("marginal_runtime_s_median"))
        tb1_marg_mean = _safe_float(agg.get((cid, "NCV_TRANSFER_BETA1"), {}).get("marginal_runtime_s_mean"))
        tb_marg = _safe_float(agg.get((cid, "NCV_TRANSFER_BETA"), {}).get("marginal_runtime_s_median"))
        tb_marg_mean = _safe_float(agg.get((cid, "NCV_TRANSFER_BETA"), {}).get("marginal_runtime_s_mean"))

        comparisons = [
            ("NCV_SCRATCH", scratch_init, scratch_marg, scratch_init_mean, scratch_marg_mean),
            ("NCV_TRANSFER_BETA1", shared_train_median, tb1_marg, shared_train_median, tb1_marg_mean),
            ("NCV_TRANSFER_BETA", shared_train_median, tb_marg, shared_train_median, tb_marg_mean),
        ]
        for method, init_c, marg, init_c_mean, marg_mean in comparisons:
            be = _solve_break_even(init_c, gcv, marg)
            be_mean = _solve_break_even(init_c_mean, gcv_mean, marg_mean)
            equal_obs_rows.append(
                {
                    "contract_id": cid,
                    "method": method,
                    "baseline_method": "GCV",
                    "initial_cost_s": init_c,
                    "baseline_marginal_cost_s": gcv,
                    "proposed_marginal_cost_s": marg,
                    **be,
                    "break_even_q_mean_runtime_sensitivity": be_mean.get("break_even_q", "NA"),
                    "break_even_q_mean_runtime_sensitivity_reason": be_mean.get("failure_reason", ""),
                }
            )

    by_target = defaultdict(list)
    for r in matched_accuracy_rows:
        by_target[r.get("target_definition")].append(r)

    def _break_even_from_target(target_definition: str) -> List[dict]:
        rows: List[dict] = []
        for r in by_target.get(target_definition, []):
            cid = r["contract_id"]
            method = r["method"]
            if method == "GCV":
                continue
            gcv_row = next(
                (
                    x
                    for x in by_target.get(target_definition, [])
                    if x.get("contract_id") == cid and x.get("method") == "GCV"
                ),
                None,
            )
            if gcv_row is None:
                rows.append(
                    {
                        "contract_id": cid,
                        "method": method,
                        "target_definition": target_definition,
                        "break_even_q": "NA",
                        "failure_reason": "missing_runtime_input",
                        "verified_q_minus_1": False,
                        "verified_q": False,
                    }
                )
                continue
            baseline = _safe_float(gcv_row.get("marginal_runtime_s"))
            proposed = _safe_float(r.get("marginal_runtime_s"))
            init_c = _safe_float(r.get("one_time_runtime_s"))
            baseline_mean = _safe_float(gcv_row.get("projected_pricing_runtime_s_mean_sensitivity"))
            proposed_mean = _safe_float(r.get("projected_pricing_runtime_s_mean_sensitivity"))
            be = _solve_break_even(init_c, baseline, proposed)
            be_mean = _solve_break_even(init_c, baseline_mean, proposed_mean)
            rows.append(
                {
                    "contract_id": cid,
                    "method": method,
                    "target_definition": target_definition,
                    "initial_cost_s": init_c,
                    "baseline_marginal_cost_s": baseline,
                    "proposed_marginal_cost_s": proposed,
                    **be,
                    "break_even_q_mean_runtime_sensitivity": be_mean.get("break_even_q", "NA"),
                    "break_even_q_mean_runtime_sensitivity_reason": be_mean.get("failure_reason", ""),
                }
            )
        return rows

    matched_rows = _break_even_from_target("match_gcv_50000_se")
    fixed_rows = _break_even_from_target("fixed_se_0.001")

    portfolio_rows: List[dict] = []
    gcv_cycle = 0.0
    tb1_cycle = 0.0
    tb_cycle = 0.0
    have = True
    for cid in TARGET_IDS:
        g = _safe_float(agg.get((cid, "GCV"), {}).get("marginal_runtime_s_median"))
        b1 = _safe_float(agg.get((cid, "NCV_TRANSFER_BETA1"), {}).get("marginal_runtime_s_median"))
        b = _safe_float(agg.get((cid, "NCV_TRANSFER_BETA"), {}).get("marginal_runtime_s_median"))
        if g is None or b1 is None or b is None:
            have = False
            break
        gcv_cycle += g
        tb1_cycle += b1
        tb_cycle += b

    if have:
        be1 = _solve_break_even(shared_train_median, gcv_cycle, tb1_cycle)
        be2 = _solve_break_even(shared_train_median, gcv_cycle, tb_cycle)
        portfolio_rows.append(
            {
                "portfolio": "six_target_contract_cycle",
                "shared_reference_training_counted_once": True,
                "initial_cost_s": shared_train_median,
                "gcv_cycle_marginal_cost_s": gcv_cycle,
                "transfer_beta1_cycle_marginal_cost_s": tb1_cycle,
                "transfer_beta_cycle_marginal_cost_s": tb_cycle,
                "break_even_q_beta1": be1.get("break_even_q", "NA"),
                "break_even_q_beta1_failure_reason": be1.get("failure_reason", ""),
                "break_even_q_beta": be2.get("break_even_q", "NA"),
                "break_even_q_beta_failure_reason": be2.get("failure_reason", ""),
            }
        )
    else:
        portfolio_rows.append(
            {
                "portfolio": "six_target_contract_cycle",
                "shared_reference_training_counted_once": True,
                "break_even_q_beta1": "NA",
                "break_even_q_beta1_failure_reason": "missing_runtime_input",
                "break_even_q_beta": "NA",
                "break_even_q_beta_failure_reason": "missing_runtime_input",
            }
        )

    return equal_obs_rows, matched_rows, fixed_rows, portfolio_rows


def _runtime_summary(aggregate_rows: List[dict], shared_rows: List[dict], per_rep_rows: List[dict]) -> Tuple[List[dict], List[dict]]:
    grouped = defaultdict(list)

    for r in per_rep_rows:
        if r.get("error"):
            continue
        for phase, field in (
            ("target_training", "target_training_runtime_s"),
            ("pilot", "pilot_runtime_s"),
            ("pricing", "pricing_runtime_s"),
            ("marginal", "marginal_runtime_s"),
            ("standalone", "standalone_runtime_s"),
        ):
            v = _safe_float(r.get(field))
            if v is not None:
                grouped[(r["contract_id"], r["method"], phase)].append(v)

    for r in shared_rows:
        if r.get("error"):
            continue
        v = _safe_float(r.get("training_runtime_s"))
        if v is not None:
            grouped[("reference", "SHARED_REFERENCE_TRAINING", "shared_reference_training")].append(v)

    summary_rows = []
    for (cid, method, phase), vals in sorted(grouped.items()):
        summary_rows.append(
            {
                "contract_id": cid,
                "method": method,
                "phase": phase,
                "count": len(vals),
                "mean": statistics.mean(vals),
                "std_dev": statistics.stdev(vals) if len(vals) >= 2 else 0.0,
                "median": statistics.median(vals),
                "minimum": min(vals),
                "maximum": max(vals),
                "first_quartile": _quantile(vals, 0.25),
                "third_quartile": _quantile(vals, 0.75),
            }
        )

    # Attach medians/means per phase back into aggregate for downstream cost projections.
    agg_out = []
    for r in aggregate_rows:
        cid = r["contract_id"]
        method = r["method"]
        rr = dict(r)
        for phase, col in (
            ("pricing", "pricing_runtime_s"),
            ("pilot", "pilot_runtime_s"),
            ("target_training", "target_training_runtime_s"),
            ("marginal", "marginal_runtime_s"),
            ("standalone", "standalone_runtime_s"),
        ):
            vals = grouped.get((cid, method, phase), [])
            rr[f"{col}_median"] = statistics.median(vals) if vals else "NA"
            rr[f"{col}_mean"] = statistics.mean(vals) if vals else "NA"
            rr[f"{col}_std"] = statistics.stdev(vals) if len(vals) >= 2 else "NA"
        agg_out.append(rr)

    return summary_rows, agg_out


def _build_validation_report(
    per_rep_rows: List[dict],
    seed_manifest: List[dict],
    n_replications: int,
) -> dict:
    failures: List[str] = []
    warnings: List[str] = []

    expected, row_failures = _all_required_rows_present(per_rep_rows, n_replications)
    if not expected:
        failures.extend(row_failures)

    for cid in CONTRACT_IDS:
        cfg = make_contract_cfg(cid, n_paths=10, seed=0)
        _, _, T = CONTRACT_GRID[cid]
        if abs(cfg.dt - (T / cfg.m)) > 1e-12:
            failures.append(f"dt mismatch for {cid}")

    for r in per_rep_rows:
        if r.get("error"):
            continue
        rid = f"rep={r.get('replication')},contract={r.get('contract_id')},method={r.get('method')}"
        if r.get("method") == "AV":
            po = _safe_float(r.get("pricing_observations"))
            sp = _safe_float(r.get("pricing_simulated_paths"))
            if po is not None and sp is not None and int(sp) != 2 * int(po):
                failures.append(f"{rid}: AV simulated path accounting mismatch")
        ev = _safe_float(r.get("estimator_variance"))
        ov = _safe_float(r.get("observation_variance"))
        po = _safe_float(r.get("pricing_observations"))
        if ev is not None and ov is not None and po is not None and po > 0:
            if abs(ev - ov / po) > 1e-8 * max(1.0, abs(ov / po)):
                failures.append(f"{rid}: estimator variance identity mismatch")

    by_rep = defaultdict(list)
    for s in seed_manifest:
        if "pricing_" in s.get("stream", ""):
            by_rep[s["replication"]].append(s["seed_value"])
    for rep, vals in by_rep.items():
        if len(vals) != len(set(vals)):
            failures.append(f"duplicate pricing seeds in replication {rep}")

    return {
        "passed": len(failures) == 0,
        "n_failures": len(failures),
        "n_warnings": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


def _run_empirical_equal_budget(
    enabled: bool,
    base_seed: int,
    n_replications: int,
    n_training: int,
    n_pilot: int,
    budget: int,
) -> List[dict]:
    if not enabled:
        return []

    q_values = [1, 10, 100, 1000]
    rows: List[dict] = []

    for Q in q_values:
        for rep in range(n_replications):
            seeds = _replication_seeds(base_seed + 77_777, rep)
            for cid in CONTRACT_IDS:
                methods = ("MC", "AV", "GCV")
                for method in methods:
                    alloc = _equal_budget_allocation(method, budget, Q, n_training, n_pilot)
                    if not alloc["feasible"]:
                        rows.append(
                            {
                                "result_type": "empirical",
                                "Q": Q,
                                "replication": rep,
                                "contract_id": cid,
                                "method": method,
                                "feasible": False,
                                "failure_reason": alloc.get("failure_reason"),
                            }
                        )
                        continue
                    n_obs = int(alloc["pricing_observations"])
                    pricing_seed = seeds[f"pricing_{cid}"] + Q * 1_000_000
                    cfg = make_contract_cfg(cid, n_paths=n_obs, seed=pricing_seed)

                    if method == "MC":
                        res = standard_monte_carlo(cfg)
                    elif method == "AV":
                        res = antithetic_variates(cfg)
                    else:
                        res = geometric_control_variate(cfg, n_pilot=n_pilot)

                    rows.append(
                        {
                            "result_type": "empirical",
                            "Q": Q,
                            "replication": rep,
                            "contract_id": cid,
                            "method": method,
                            "pricing_observations": res.pricing_observations,
                            "pricing_simulated_paths": res.pricing_simulated_paths,
                            "price": res.price,
                            "observation_variance": res.observation_variance,
                            "std_error": res.std_error,
                            "simulation_note": "one valuation per Q per replication using amortised per-valuation allocation",
                            "feasible": True,
                            "failure_reason": "",
                        }
                    )

    return rows


def run_stage8(
    profile: str,
    base_seed: int = 42,
    output_dir: str = "experiment_runs",
    n_replications_override: Optional[int] = None,
    empirical_equal_budget: bool = False,
) -> Path:
    profile_cfg = PROFILES[profile]
    n_training = profile_cfg["n_training"]
    n_pilot = profile_cfg["n_pilot"]
    n_pricing = profile_cfg["n_pricing"]
    n_replications = n_replications_override or profile_cfg["n_replications"]
    n_high_prec = profile_cfg["n_high_precision"]
    amortised_q_values = profile_cfg["amortised_q_values"]

    monitoring_dates = make_contract_cfg(REFERENCE_ID, n_paths=2, seed=0).m

    torch_available = _try_import_torch()
    if profile == "dissertation" and not torch_available:
        raise RuntimeError("Dissertation profile requires PyTorch; torch is unavailable. Exiting with non-zero status.")

    _warmup_numpy_and_torch(torch_available)

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / f"stage8_{profile}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Stage 8: {profile_cfg['description']} ===")
    print(f"Output dir: {run_dir}")
    print(f"Profile={profile}, base_seed={base_seed}, replications={n_replications}, torch_available={torch_available}")

    config_snapshot = {
        "profile": profile,
        "base_seed": base_seed,
        "n_replications": n_replications,
        "n_training": n_training,
        "n_pilot": n_pilot,
        "n_pricing": n_pricing,
        "total_path_budget": n_pricing,
        "amortised_q_values": amortised_q_values,
        "n_high_precision": n_high_prec,
        "monitoring_dates": monitoring_dates,
        "torch_available": torch_available,
        "contracts": {cid: {"K": K, "sigma": s, "T": T} for cid, (K, s, T) in CONTRACT_GRID.items()},
        "seed_offsets": {
            "ref_train": SEED_OFFSET_REF_TRAIN,
            "ref_val": SEED_OFFSET_REF_VAL,
            "target_train": SEED_OFFSET_TARGET_TRAIN,
            "pilot": SEED_OFFSET_PILOT,
            "pricing": SEED_OFFSET_PRICING,
            "high_prec": SEED_OFFSET_HIGH_PREC,
        },
        "timestamp_utc": ts,
        "commit_hash": _commit_hash(),
        "empirical_equal_budget_enabled": empirical_equal_budget,
    }
    _write_json(run_dir / "config_snapshot.json", config_snapshot)

    env_meta = collect_environment_metadata()
    env_meta["torch_available"] = torch_available
    env_meta["torch_version"] = _torch_version()
    env_meta["cpu_count"] = os.cpu_count()
    env_meta["platform"] = platform.platform()
    _write_json(run_dir / "environment.json", env_meta)

    seed_manifest = _build_seed_manifest(base_seed, n_replications)
    _write_csv(run_dir / "seed_manifest.csv", seed_manifest)

    high_prec_rows = compute_all_high_precision_references(n_high_prec, base_seed)
    _write_csv(run_dir / "high_precision_references.csv", high_prec_rows)

    all_per_rep: List[dict] = []
    all_beta: List[dict] = []
    all_runtime: List[dict] = []
    shared_training_rows: List[dict] = []

    for rep in range(n_replications):
        print(f"[Stage 8] replication {rep+1}/{n_replications}")
        per_rep, beta, runtime, shared = run_replication(
            base_seed=base_seed,
            replication=rep,
            n_training=n_training,
            n_pilot=n_pilot,
            n_pricing=n_pricing,
            torch_available=torch_available,
        )
        all_per_rep.extend(per_rep)
        all_beta.extend(beta)
        all_runtime.extend(runtime)
        shared_training_rows.append(shared)

    _write_csv(run_dir / "shared_reference_training.csv", shared_training_rows)
    _write_csv(run_dir / "per_replication_results.csv", all_per_rep)
    _write_csv(run_dir / "transfer_diagnostics.csv", all_beta)
    _write_csv(run_dir / "runtime_raw.csv", all_runtime)

    per_rep_vrr, variance_ratio_summary = compute_variance_ratio_summary(all_per_rep)
    _write_csv(run_dir / "per_replication_variance_ratios.csv", per_rep_vrr)
    _write_csv(run_dir / "variance_ratio_summary.csv", variance_ratio_summary)

    aggregate_rows = aggregate_statistical_results(
        per_rep_rows=all_per_rep,
        high_prec_rows=high_prec_rows,
        variance_ratio_summary=variance_ratio_summary,
        n_replications=n_replications,
    )

    runtime_summary_rows, aggregate_rows_with_runtime = _runtime_summary(
        aggregate_rows=aggregate_rows,
        shared_rows=shared_training_rows,
        per_rep_rows=all_per_rep,
    )

    _write_csv(run_dir / "runtime_summary.csv", runtime_summary_rows)
    _write_csv(run_dir / "aggregate_statistical_results.csv", aggregate_rows_with_runtime)
    _write_csv(run_dir / "aggregate_results.csv", aggregate_rows_with_runtime)

    equal_obs_rows = compute_equal_pricing_observations_summary(
        aggregate_rows_with_runtime,
        variance_ratio_summary,
        n_training=n_training,
    )
    _write_csv(run_dir / "equal_pricing_observations_summary.csv", equal_obs_rows)

    equal_budget_rows = compute_equal_budget_projected_results(
        aggregate_rows=aggregate_rows_with_runtime,
        n_training=n_training,
        n_pilot=n_pilot,
        budget=n_pricing,
        q_values=amortised_q_values,
    )
    _write_csv(run_dir / "equal_budget_projected_results.csv", equal_budget_rows)

    shared_train_runtime_median = _median([_safe_float(r.get("training_runtime_s")) for r in shared_training_rows if not r.get("error")])
    matched_rows = compute_matched_accuracy_results(
        aggregate_rows=aggregate_rows_with_runtime,
        per_rep_rows=all_per_rep,
        n_training=n_training,
        n_pilot=n_pilot,
        shared_train_runtime_median=shared_train_runtime_median,
    )
    _write_csv(run_dir / "matched_accuracy_results.csv", matched_rows)

    break_even_equal_obs, break_even_matched, break_even_fixed, portfolio_break_even = compute_break_even_tables(
        aggregate_rows=aggregate_rows_with_runtime,
        matched_accuracy_rows=matched_rows,
        shared_training_rows=shared_training_rows,
    )
    _write_csv(run_dir / "break_even_equal_observations.csv", break_even_equal_obs)
    _write_csv(run_dir / "break_even_matched_accuracy.csv", break_even_matched)
    _write_csv(run_dir / "break_even_fixed_accuracy.csv", break_even_fixed)
    _write_csv(run_dir / "portfolio_break_even.csv", portfolio_break_even)

    ref_precision_rows = compute_reference_precision_diagnostics(aggregate_rows_with_runtime, high_prec_rows)
    _write_csv(run_dir / "reference_precision_diagnostics.csv", ref_precision_rows)

    transfer_summary_rows = []
    transfer_rows_valid = [r for r in all_beta if not r.get("error")]
    for method in TRANSFER_METHODS:
        subset = [r for r in transfer_rows_valid if r.get("method") == method]
        betas = [_safe_float(r.get("estimated_beta")) for r in subset]
        betas = [b for b in betas if b is not None]
        cors = [_safe_float(r.get("payoff_control_correlation")) for r in subset]
        cors = [c for c in cors if c is not None]
        improvements = [_safe_float(r.get("variance_improvement_from_estimating_beta")) for r in subset]
        improvements = [i for i in improvements if i is not None and i > 0]
        transfer_summary_rows.append(
            {
                "method": method,
                "mean_beta": _mean(betas) if betas else "NA",
                "std_beta": _std(betas) if betas else "NA",
                "mean_correlation": _mean(cors) if cors else "NA",
                "std_correlation": _std(cors) if cors else "NA",
                "geometric_mean_variance_improvement_from_estimating_beta": (
                    math.exp(_mean([math.log(i) for i in improvements])) if improvements else "NA"
                ),
            }
        )
    _write_csv(run_dir / "transfer_diagnostics_summary.csv", transfer_summary_rows)

    empirical_rows = _run_empirical_equal_budget(
        enabled=empirical_equal_budget,
        base_seed=base_seed,
        n_replications=n_replications,
        n_training=n_training,
        n_pilot=n_pilot,
        budget=n_pricing,
    )
    _write_csv(run_dir / "equal_budget_empirical_results.csv", empirical_rows)

    validation_report = _build_validation_report(all_per_rep, seed_manifest, n_replications)
    _write_json(run_dir / "validation_report.json", validation_report)

    if profile == "dissertation":
        ok, failures = _check_required_success(all_per_rep, n_replications)
        if not ok:
            _write_json(run_dir / "dissertation_completion_failures.json", {"failures": failures})
            raise RuntimeError(
                "Dissertation run failed strict completion requirements (expected methods/combinations/replications, zero failed rows, finite metrics, verified hashes)."
            )

    stable_rows = [
        {
            "contract_id": r["contract_id"],
            "method": r["method"],
            "successful_replications": r["successful_replications"],
            "mean_price": r["mean_price"],
            "mean_estimator_variance": r["mean_estimator_variance"],
            "mean_reported_estimator_standard_error": r["mean_reported_estimator_standard_error"],
        }
        for r in aggregate_rows_with_runtime
    ]
    _write_csv(run_dir / "summary_stable.csv", stable_rows)

    repro_report = {
        "profile": profile,
        "base_seed": base_seed,
        "n_replications": n_replications,
        "n_rows_per_replication_results": len(all_per_rep),
        "n_rows_aggregate_results": len(aggregate_rows_with_runtime),
        "validation_passed": validation_report["passed"],
        "stable_summary_sha256": hashlib.sha256(
            json.dumps(stable_rows, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "note": "Stable summary hash is available for comparison across a second identical run.",
        "second_identical_run_match_confirmed": False,
    }
    _write_json(run_dir / "reproducibility_report.json", repro_report)

    n_failed = sum(1 for r in all_per_rep if r.get("error"))
    n_success = len(all_per_rep) - n_failed
    output_files = sorted([p.name for p in run_dir.iterdir() if p.is_file()])

    torch_line = f"available ({env_meta.get('torch_version', 'unknown')})" if torch_available else "unavailable"
    handover = f"""# Stage 8 Handover

- profile: {profile}
- base seed: {base_seed}
- replications: {n_replications}
- monitoring dates: {monitoring_dates}
- Torch: {torch_line}
- failed-row count: {n_failed}
- successful-row count: {n_success}
- empirical equal-budget mode run: {empirical_equal_budget}
- reproducibility tested through second identical run: False

## output files
{os.linesep.join(f"- {name}" for name in output_files)}

## reproducibility note
- stable summary hash is available for run-to-run comparison; reproducibility is only confirmed after a second identical run matches.
"""
    (run_dir / "handover.md").write_text(handover, encoding="utf-8")

    print(f"[Stage 8] done. validation_passed={validation_report['passed']} output={run_dir}")
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stage 8: Frozen NCV Transfer, Calibration and Amortisation")
    p.add_argument("--profile", choices=list(PROFILES), default="smoke", help="Execution profile (smoke | dissertation)")
    p.add_argument("--base-seed", type=int, default=42, help="Base random seed")
    p.add_argument("--output-dir", default="experiment_runs", help="Base directory for timestamped output bundles")
    p.add_argument("--n-replications", type=int, default=None, help="Override profile default number of replications")
    p.add_argument(
        "--empirical-equal-budget",
        action="store_true",
        help="Run optional empirical equal-budget reruns at Q=[1,10,100,1000].",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_stage8(
        profile=args.profile,
        base_seed=args.base_seed,
        output_dir=args.output_dir,
        n_replications_override=args.n_replications,
        empirical_equal_budget=args.empirical_equal_budget,
    )
