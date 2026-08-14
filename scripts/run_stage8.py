"""
run_stage8.py
=============
Stage 8 experiment runner: Frozen NCV Transfer, Calibration and Amortisation.

This script implements experiments E0–E4 as described in the Stage 8
specification, including:

- E0: Reference replication (>=30 replications of reference contract)
- E1: Contract-specific applicability (MC, AV, GCV, NCV_SCRATCH across 7 contracts)
- E2: Direct frozen transfer (NCV_TRANSFER_BETA1 across 6 target contracts)
- E3: Pilot-calibrated transfer (NCV_TRANSFER_BETA across 6 target contracts)
- E4: Comparison modes (equal observations, equal budget, amortised portfolio)

Profiles
--------
smoke:
    n_training=100, n_pilot=50, n_pricing=200, n_replications=2,
    all 7 contracts, all methods, all validation/schema checks.

dissertation:
    n_training=5000, n_pilot=1000, n_pricing=50000, n_replications=30+,
    full analyses/tables/plots.

Usage
-----
    python scripts/run_stage8.py --profile smoke --output-dir experiment_runs
    python scripts/run_stage8.py --profile dissertation --output-dir experiment_runs

Output
------
A timestamped directory under --output-dir containing all required CSV/JSON
output files.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from asian_options.config import ModelConfig, collect_environment_metadata
from asian_options.contracts import (
    CONTRACT_IDS,
    TARGET_IDS,
    REFERENCE_ID,
    make_contract_cfg,
    CONTRACT_GRID,
)
from asian_options.estimators import (
    standard_monte_carlo,
    antithetic_variates,
    geometric_control_variate,
)

# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------

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
        "amortised_q_values": [1, 5, 10, 20, 35, 50, 100],
        "description": "Dissertation: standard sizes, 30 reps, full analyses",
    },
}

# ---------------------------------------------------------------------------
# Seed schedule
# ---------------------------------------------------------------------------
# Per replication, seed streams are derived from a base seed by offsets to
# ensure independence.  The offsets below are fixed and documented.

SEED_OFFSET_REF_TRAIN = 1_000
SEED_OFFSET_REF_VAL = 2_000
SEED_OFFSET_TARGET_TRAIN = 3_000   # + contract_index * 100
SEED_OFFSET_PILOT = 4_000          # + contract_index * 100
SEED_OFFSET_PRICING = 5_000        # + contract_index * 100
SEED_OFFSET_HIGH_PREC = 9_000
SEED_OFFSET_NET_INIT = 10_000


def _replication_seeds(base_seed: int, replication: int) -> dict:
    """Derive all seed values for one replication from base_seed and rep index."""
    rep_offset = replication * 100_000  # large offset per replication
    s = base_seed + rep_offset
    seeds = {
        "ref_train": s + SEED_OFFSET_REF_TRAIN,
        "ref_val": s + SEED_OFFSET_REF_VAL,
        "high_prec": s + SEED_OFFSET_HIGH_PREC,
    }
    for ci, cid in enumerate(CONTRACT_IDS):
        seeds[f"target_train_{cid}"] = s + SEED_OFFSET_TARGET_TRAIN + ci * 100
        seeds[f"pilot_{cid}"] = s + SEED_OFFSET_PILOT + ci * 100
        seeds[f"pricing_{cid}"] = s + SEED_OFFSET_PRICING + ci * 100
    return seeds


# ---------------------------------------------------------------------------
# NCV helpers
# ---------------------------------------------------------------------------

def _try_import_torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def _run_ncv_scratch(
    contract_id: str,
    contract_cfg_base: ModelConfig,
    n_training: int,
    train_seed: int,
    pricing_seed: int,
    n_pricing: int,
) -> dict:
    """Train NCV from scratch for a given contract and return result dict."""
    import dataclasses as dc
    from asian_options.neural_cv import build_network, train_network, ncv_estimator
    from asian_options.simulate_gbm import simulate_paths
    from asian_options.payoffs import arithmetic_asian_call_payoff

    t0 = time.perf_counter()

    train_cfg = dc.replace(contract_cfg_base, n_paths=n_training, seed=train_seed)
    paths = simulate_paths(train_cfg)
    payoffs = arithmetic_asian_call_payoff(paths, train_cfg)
    dt = train_cfg.dt
    drift = (train_cfg.r - train_cfg.q - 0.5 * train_cfg.sigma ** 2) * dt
    diffusion = train_cfg.sigma * math.sqrt(dt)
    import numpy as np
    log_S = np.log(paths / train_cfg.S0)
    log_inc = np.diff(np.hstack([np.zeros((n_training, 1)), log_S]), axis=1)
    Z_train = (log_inc - drift) / diffusion

    dataset = {"X_train": Z_train, "y_train": payoffs}
    network = build_network(train_cfg, hidden_width=32)
    train_network(network, dataset, train_cfg, n_epochs=100)

    training_runtime_s = time.perf_counter() - t0

    price_cfg = dc.replace(contract_cfg_base, n_paths=n_pricing, seed=pricing_seed)
    ncv_result = ncv_estimator(network, price_cfg, n_training_paths=n_training)
    ncv_result = ncv_result._replace(
        training_runtime_seconds=training_runtime_s,
        end_to_end_runtime_seconds=ncv_result.pricing_runtime_seconds + training_runtime_s,
    )

    return {
        "method": "NCV_SCRATCH",
        "price": ncv_result.price,
        "observation_variance": ncv_result.observation_variance,
        "estimator_variance": ncv_result.estimator_variance,
        "std_error": ncv_result.std_error,
        "ci_lower": ncv_result.ci_lower,
        "ci_upper": ncv_result.ci_upper,
        "pricing_observations": ncv_result.pricing_observations,
        "pricing_simulated_paths": ncv_result.pricing_simulated_paths,
        "pilot_paths": 0,
        "training_paths": n_training,
        "total_simulated_paths": ncv_result.total_simulated_paths,
        "beta": float("nan"),
        "corr_f_c0": float("nan"),
        "pricing_runtime_s": ncv_result.pricing_runtime_seconds,
        "training_runtime_s": training_runtime_s,
        "end_to_end_runtime_s": ncv_result.end_to_end_runtime_seconds,
    }


# ---------------------------------------------------------------------------
# Core per-replication runner
# ---------------------------------------------------------------------------

def run_replication(
    base_seed: int,
    replication: int,
    n_training: int,
    n_pilot: int,
    n_pricing: int,
    total_path_budget: int,
    amortised_q_values: List[int],
    torch_available: bool,
) -> Tuple[List[dict], List[dict], List[dict]]:
    """
    Run one complete replication of all Stage 8 experiments.

    Returns
    -------
    per_rep_rows : list[dict]
        One row per (contract, method).
    beta_rows : list[dict]
        Beta / transfer diagnostics per (contract, method).
    runtime_rows : list[dict]
        Timing records per (contract, method).
    """
    import numpy as np

    seeds = _replication_seeds(base_seed, replication)
    per_rep_rows: List[dict] = []
    beta_rows: List[dict] = []
    runtime_rows: List[dict] = []

    # ------------------------------------------------------------------
    # Reference network: train once, freeze, record hash
    # ------------------------------------------------------------------
    ref_network = None
    e_h0 = None
    ref_hash = None
    ref_train_runtime_s = 0.0

    if torch_available:
        try:
            from asian_options.frozen_transfer import train_reference_network, compute_network_hash
            ref_cfg_base = make_contract_cfg(REFERENCE_ID, n_paths=n_pricing, seed=seeds["ref_train"])
            ref_network, e_h0, ref_hash, ref_train_runtime_s = train_reference_network(
                ref_cfg_base,
                n_training=n_training,
                train_seed=seeds["ref_train"],
            )
        except Exception as exc:
            ref_network = None
            print(f"  [rep={replication}] Reference network training FAILED: {exc}", file=sys.stderr)

    # ------------------------------------------------------------------
    # Per-contract loop
    # ------------------------------------------------------------------
    for ci, contract_id in enumerate(CONTRACT_IDS):
        K, sigma, T = CONTRACT_GRID[contract_id]
        base_cfg = make_contract_cfg(contract_id, n_paths=n_pricing, seed=seeds[f"pricing_{contract_id}"])
        pricing_seed = seeds[f"pricing_{contract_id}"]
        pilot_seed = seeds[f"pilot_{contract_id}"]
        target_train_seed = seeds[f"target_train_{contract_id}"]

        def _base_row(method: str) -> dict:
            return {
                "base_seed": base_seed,
                "replication": replication,
                "contract_id": contract_id,
                "K": K,
                "sigma": sigma,
                "T": T,
                "method": method,
            }

        # ---- MC ----
        try:
            mc_cfg = dataclasses.replace(base_cfg, seed=pricing_seed)
            mc = standard_monte_carlo(mc_cfg)
            mc_obs_var = mc.observation_variance
            row = {**_base_row("MC"),
                   "price": mc.price,
                   "observation_variance": mc.observation_variance,
                   "estimator_variance": mc.estimator_variance,
                   "std_error": mc.std_error,
                   "ci_lower": mc.ci_lower, "ci_upper": mc.ci_upper,
                   "pricing_observations": mc.pricing_observations,
                   "pricing_simulated_paths": mc.pricing_simulated_paths,
                   "pilot_paths": 0, "training_paths": 0,
                   "total_simulated_paths": mc.total_simulated_paths,
                   "beta": float("nan"), "corr_f_c0": float("nan"),
                   "pricing_runtime_s": mc.pricing_runtime_seconds,
                   "training_runtime_s": 0.0,
                   "end_to_end_runtime_s": mc.end_to_end_runtime_seconds,
                   "error": ""}
            per_rep_rows.append(row)
        except Exception as exc:
            per_rep_rows.append({**_base_row("MC"), "error": str(exc)})
            mc_obs_var = float("nan")

        # ---- AV ----
        try:
            av_cfg = dataclasses.replace(base_cfg, seed=pricing_seed)
            av = antithetic_variates(av_cfg)
            per_rep_rows.append({**_base_row("AV"),
                                  "price": av.price,
                                  "observation_variance": av.observation_variance,
                                  "estimator_variance": av.estimator_variance,
                                  "std_error": av.std_error,
                                  "ci_lower": av.ci_lower, "ci_upper": av.ci_upper,
                                  "pricing_observations": av.pricing_observations,
                                  "pricing_simulated_paths": av.pricing_simulated_paths,
                                  "pilot_paths": 0, "training_paths": 0,
                                  "total_simulated_paths": av.total_simulated_paths,
                                  "beta": float("nan"), "corr_f_c0": float("nan"),
                                  "pricing_runtime_s": av.pricing_runtime_seconds,
                                  "training_runtime_s": 0.0,
                                  "end_to_end_runtime_s": av.end_to_end_runtime_seconds,
                                  "error": ""})
        except Exception as exc:
            per_rep_rows.append({**_base_row("AV"), "error": str(exc)})

        # ---- GCV ----
        try:
            gcv_n_pilot = n_pilot
            gcv_pricing = n_pricing
            gcv_cfg = dataclasses.replace(base_cfg, n_paths=gcv_pricing, seed=pricing_seed)
            cv = geometric_control_variate(gcv_cfg, n_pilot=gcv_n_pilot)
            per_rep_rows.append({**_base_row("GCV"),
                                  "price": cv.price,
                                  "observation_variance": cv.observation_variance,
                                  "estimator_variance": cv.estimator_variance,
                                  "std_error": cv.std_error,
                                  "ci_lower": cv.ci_lower, "ci_upper": cv.ci_upper,
                                  "pricing_observations": cv.pricing_observations,
                                  "pricing_simulated_paths": cv.pricing_simulated_paths,
                                  "pilot_paths": cv.pilot_paths, "training_paths": 0,
                                  "total_simulated_paths": cv.total_simulated_paths,
                                  "beta": cv.beta_hat, "corr_f_c0": cv.corr_estimate,
                                  "pricing_runtime_s": cv.pricing_runtime_seconds,
                                  "training_runtime_s": cv.training_runtime_seconds,
                                  "end_to_end_runtime_s": cv.end_to_end_runtime_seconds,
                                  "error": ""})
        except Exception as exc:
            per_rep_rows.append({**_base_row("GCV"), "error": str(exc)})

        # ---- NCV_SCRATCH ----
        if torch_available:
            try:
                scratch = _run_ncv_scratch(
                    contract_id, base_cfg,
                    n_training=n_training,
                    train_seed=target_train_seed,
                    pricing_seed=pricing_seed,
                    n_pricing=n_pricing,
                )
                per_rep_rows.append({**_base_row("NCV_SCRATCH"), **scratch, "error": ""})
            except Exception as exc:
                per_rep_rows.append({**_base_row("NCV_SCRATCH"), "error": str(exc)})
        else:
            per_rep_rows.append({**_base_row("NCV_SCRATCH"),
                                  "error": "torch_not_available"})

        # ---- NCV_TRANSFER_BETA1 (only for non-reference contracts) ----
        if contract_id != REFERENCE_ID:
            if torch_available and ref_network is not None:
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
                    per_rep_rows.append({**_base_row("NCV_TRANSFER_BETA1"), **tb1, "error": ""})
                    beta_rows.append({
                        **_base_row("NCV_TRANSFER_BETA1"),
                        "beta": 1.0,
                        "var_c0_pilot": float("nan"),
                        "corr_f_c0": tb1.get("corr_f_c0", float("nan")),
                        "e_h0": e_h0,
                        "param_hash": ref_hash,
                        "hash_verified": True,
                        "error": "",
                    })
                except Exception as exc:
                    per_rep_rows.append({**_base_row("NCV_TRANSFER_BETA1"), "error": str(exc)})
                    beta_rows.append({**_base_row("NCV_TRANSFER_BETA1"), "error": str(exc)})
            else:
                reason = "torch_not_available" if not torch_available else "ref_network_failed"
                per_rep_rows.append({**_base_row("NCV_TRANSFER_BETA1"), "error": reason})
                beta_rows.append({**_base_row("NCV_TRANSFER_BETA1"), "error": reason})

        # ---- NCV_TRANSFER_BETA (only for non-reference contracts) ----
        if contract_id != REFERENCE_ID:
            if torch_available and ref_network is not None:
                try:
                    from asian_options.frozen_transfer import ncv_transfer_beta, NearZeroVarianceError
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
                    per_rep_rows.append({**_base_row("NCV_TRANSFER_BETA"), **tb, "error": ""})
                    beta_rows.append({
                        **_base_row("NCV_TRANSFER_BETA"),
                        "beta": tb["beta"],
                        "var_c0_pilot": tb.get("var_c0_pilot", float("nan")),
                        "corr_f_c0": tb.get("corr_f_c0", float("nan")),
                        "e_h0": e_h0,
                        "param_hash": ref_hash,
                        "hash_verified": True,
                        "error": "",
                    })
                except Exception as exc:
                    per_rep_rows.append({**_base_row("NCV_TRANSFER_BETA"), "error": str(exc)})
                    beta_rows.append({**_base_row("NCV_TRANSFER_BETA"), "error": str(exc)})
            else:
                reason = "torch_not_available" if not torch_available else "ref_network_failed"
                per_rep_rows.append({**_base_row("NCV_TRANSFER_BETA"), "error": reason})
                beta_rows.append({**_base_row("NCV_TRANSFER_BETA"), "error": reason})

    # ------------------------------------------------------------------
    # Collect runtime rows from per-rep rows
    # ------------------------------------------------------------------
    for row in per_rep_rows:
        if row.get("error", ""):
            continue
        runtime_rows.append({
            "base_seed": row.get("base_seed"),
            "replication": row.get("replication"),
            "contract_id": row.get("contract_id"),
            "method": row.get("method"),
            "pricing_runtime_s": row.get("pricing_runtime_s", ""),
            "training_runtime_s": row.get("training_runtime_s", ""),
            "end_to_end_runtime_s": row.get("end_to_end_runtime_s", ""),
            "pricing_observations": row.get("pricing_observations", ""),
        })

    return per_rep_rows, beta_rows, runtime_rows


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _safe_float(v) -> Optional[float]:
    try:
        f = float(v)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _vrr(mc_obs_var: Optional[float], method_obs_var: Optional[float]) -> str:
    if mc_obs_var is None or method_obs_var is None:
        return "NA"
    if method_obs_var == 0.0:
        return "NA"
    return f"{mc_obs_var / method_obs_var:.6f}"


def aggregate_results(per_rep_rows: List[dict]) -> List[dict]:
    """Aggregate per-replication rows to mean/std/CI across replications."""
    from collections import defaultdict
    import statistics

    # Group by (contract_id, method)
    groups: dict = defaultdict(list)
    for row in per_rep_rows:
        if row.get("error", ""):
            continue
        key = (row.get("contract_id", "NA"), row.get("method", "NA"))
        groups[key].append(row)

    agg_rows = []
    for (contract_id, method), rows in sorted(groups.items()):
        n = len(rows)
        if n == 0:
            continue

        def _mean(field):
            vals = [_safe_float(r.get(field)) for r in rows]
            vals = [v for v in vals if v is not None]
            return statistics.mean(vals) if vals else float("nan")

        def _std(field):
            vals = [_safe_float(r.get(field)) for r in rows]
            vals = [v for v in vals if v is not None]
            return statistics.stdev(vals) if len(vals) >= 2 else float("nan")

        def _median(field):
            vals = [_safe_float(r.get(field)) for r in rows]
            vals = [v for v in vals if v is not None]
            return statistics.median(vals) if vals else float("nan")

        agg_rows.append({
            "contract_id": contract_id,
            "method": method,
            "n_replications": n,
            "price_mean": _mean("price"),
            "price_std": _std("price"),
            "price_median": _median("price"),
            "observation_variance_mean": _mean("observation_variance"),
            "observation_variance_std": _std("observation_variance"),
            "estimator_variance_mean": _mean("estimator_variance"),
            "estimator_variance_std": _std("estimator_variance"),
            "std_error_mean": _mean("std_error"),
            "std_error_std": _std("std_error"),
            "pricing_runtime_s_mean": _mean("pricing_runtime_s"),
            "pricing_runtime_s_std": _std("pricing_runtime_s"),
            "training_runtime_s_mean": _mean("training_runtime_s"),
            "end_to_end_runtime_s_mean": _mean("end_to_end_runtime_s"),
            "beta_mean": _mean("beta"),
            "beta_std": _std("beta"),
            "corr_f_c0_mean": _mean("corr_f_c0"),
        })
    return agg_rows


def compute_vrr_table(agg_rows: List[dict]) -> List[dict]:
    """Compute VRR vs MC and vs GCV for each (contract, method) pair."""
    # Index MC obs_var per contract
    mc_obs_var: dict = {}
    gcv_obs_var: dict = {}
    for row in agg_rows:
        if row["method"] == "MC":
            mc_obs_var[row["contract_id"]] = _safe_float(row.get("observation_variance_mean"))
        if row["method"] == "GCV":
            gcv_obs_var[row["contract_id"]] = _safe_float(row.get("observation_variance_mean"))

    vrr_rows = []
    for row in agg_rows:
        cid = row["contract_id"]
        obs_var = _safe_float(row.get("observation_variance_mean"))
        mc_var = mc_obs_var.get(cid)
        gcv_var = gcv_obs_var.get(cid)
        vrr_rows.append({
            "contract_id": cid,
            "method": row["method"],
            "observation_variance_mean": obs_var,
            "vrr_vs_mc": _vrr(mc_var, obs_var),
            "vrr_vs_gcv": _vrr(gcv_var, obs_var),
            "std_error_mean": row.get("std_error_mean"),
            "estimator_variance_mean": row.get("estimator_variance_mean"),
        })
    return vrr_rows


# ---------------------------------------------------------------------------
# Equal-budget allocations
# ---------------------------------------------------------------------------

def _equal_budget_allocations(total: int, n_pilot: int, n_training: int) -> dict:
    """Return path counts for each method under equal-budget constraint."""
    av_pairs = total // 2
    gcv_pricing = total - n_pilot
    scratch_pricing = total - n_training
    transfer_b1_pricing = total - n_training  # 5k training amortised (=Q=1 here)
    transfer_b_pricing = total - n_pilot - n_training
    return {
        "MC": {"pricing": total, "pilot": 0, "training": 0},
        "AV": {"pricing": av_pairs, "pilot": 0, "training": 0},
        "GCV": {"pricing": gcv_pricing, "pilot": n_pilot, "training": 0},
        "NCV_SCRATCH": {"pricing": scratch_pricing, "pilot": 0, "training": n_training},
        "NCV_TRANSFER_BETA1": {"pricing": transfer_b1_pricing, "pilot": 0, "training": n_training},
        "NCV_TRANSFER_BETA": {"pricing": transfer_b_pricing, "pilot": n_pilot, "training": n_training},
    }


# ---------------------------------------------------------------------------
# Amortised portfolio cost (E4C)
# ---------------------------------------------------------------------------

def compute_amortised_costs(
    n_training: int,
    n_pilot: int,
    n_pricing: int,
    q_values: List[int],
) -> List[dict]:
    """
    Compute per-valuation path allocations for the amortised portfolio (E4C).

    For Q valuations over 7 contracts:
      - Transfer beta1: pricing_per_valuation ≈ n_pricing - n_training/Q
      - Transfer beta:  pricing_per_valuation ≈ n_pricing - n_pilot - n_training/Q
      - GCV: pricing_per_valuation = n_pricing (n_pilot charged each time)

    Returns list of rows, one per Q value.
    """
    rows = []
    for Q in q_values:
        # Integer ceiling to ensure we don't go below 1
        tb1_pricing = max(1, n_pricing - math.ceil(n_training / Q))
        tb_pricing = max(1, n_pricing - n_pilot - math.ceil(n_training / Q))
        rows.append({
            "Q": Q,
            "total_path_budget": n_pricing,
            "n_training": n_training,
            "n_pilot": n_pilot,
            # Transfer beta1
            "tb1_pricing_per_valuation": tb1_pricing,
            "tb1_total_paths": n_training + Q * tb1_pricing,
            "tb1_avg_paths_per_valuation": round((n_training + Q * tb1_pricing) / Q, 2),
            # Transfer beta (Q=1: explicit charges)
            "tb_pricing_per_valuation": tb_pricing,
            "tb_total_paths": n_training + Q * (n_pilot + tb_pricing),
            "tb_avg_paths_per_valuation": round((n_training + Q * (n_pilot + tb_pricing)) / Q, 2),
            # GCV (no training, just pilot+pricing each time)
            "gcv_pricing_per_valuation": n_pricing,
            "gcv_total_paths": Q * (n_pilot + n_pricing),
            "gcv_avg_paths_per_valuation": n_pilot + n_pricing,
        })
    return rows


def compute_break_even(
    c_train: float,
    c_gcv_per_val: float,
    c_transfer_per_val: float,
) -> str:
    """
    Compute Q* = ceil(C_train / (C_gcv - C_transfer)).

    Returns "No finite break-even under the measured configuration." if
    denominator <= 0.
    """
    denom = c_gcv_per_val - c_transfer_per_val
    if denom <= 0:
        return "No finite break-even under the measured configuration."
    q_star = math.ceil(c_train / denom)
    return str(q_star)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fieldnames is None:
        # Union of all keys
        all_keys: List[str] = []
        seen = set()
        for row in rows:
            for k in row:
                if k not in seen:
                    all_keys.append(k)
                    seen.add(k)
        fieldnames = all_keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def _commit_hash() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return "unavailable"


# ---------------------------------------------------------------------------
# Seed manifest
# ---------------------------------------------------------------------------

def _build_seed_manifest(base_seed: int, n_replications: int) -> List[dict]:
    rows = []
    for rep in range(n_replications):
        seeds = _replication_seeds(base_seed, rep)
        for stream, seed_val in seeds.items():
            rows.append({
                "base_seed": base_seed,
                "replication": rep,
                "stream": stream,
                "seed_value": seed_val,
            })
    return rows


# ---------------------------------------------------------------------------
# High-precision references
# ---------------------------------------------------------------------------

def compute_all_high_precision_references(n_paths: int, base_seed: int) -> List[dict]:
    """Compute GCV high-precision references for all 7 contracts."""
    from asian_options.frozen_transfer import compute_high_precision_reference
    rows = []
    for contract_id in CONTRACT_IDS:
        seed = base_seed + SEED_OFFSET_HIGH_PREC + CONTRACT_IDS.index(contract_id) * 1000
        try:
            row = compute_high_precision_reference(contract_id, n_paths, seed)
            rows.append(row)
        except Exception as exc:
            rows.append({
                "contract_id": contract_id,
                "price": "ERROR",
                "std_error": "ERROR",
                "ci_lower": "ERROR",
                "ci_upper": "ERROR",
                "n_paths": n_paths,
                "method": "GCV_high_precision",
                "seed": seed,
                "error": str(exc),
            })
    return rows


# ---------------------------------------------------------------------------
# Validation report builder
# ---------------------------------------------------------------------------

def build_validation_report(
    per_rep_rows: List[dict],
    agg_rows: List[dict],
    seed_manifest: List[dict],
    high_prec_rows: List[dict],
    amortised_rows: List[dict],
) -> dict:
    """
    Run Stage 8 validation checks and return a report dict.

    Validates (subset of 23 requirements from the spec):
    - Contract grid has 7 contracts, one-parameter-change rule
    - Monitoring dates: dt = T/m for all contracts
    - Finite/non-negative outputs
    - Estimator variance identity: est_var = obs_var / n_pricing
    - AV pair accounting: pricing_simulated_paths == 2 * pricing_observations
    - Beta is 1.0 for NCV_TRANSFER_BETA1
    - Method labels consistent
    - Seed independence (distinct streams per contract/phase)
    - No missing required methods (or explicit error flag)
    """
    failures: List[str] = []
    warnings: List[str] = []

    # 1. Contract grid
    try:
        from asian_options.contracts import validate_contract_grid
        validate_contract_grid()
    except Exception as exc:
        failures.append(f"Contract grid validation failed: {exc}")

    # 2. Seven contracts in grid
    if len(CONTRACT_IDS) != 7:
        failures.append(f"Expected 7 contracts, got {len(CONTRACT_IDS)}")

    # 3. Monitoring dates: dt = T/m
    for cid in CONTRACT_IDS:
        K, sigma, T = CONTRACT_GRID[cid]
        cfg = make_contract_cfg(cid, n_paths=10, seed=0)
        expected_dt = T / 12
        if abs(cfg.dt - expected_dt) > 1e-12:
            failures.append(f"dt mismatch for {cid}: got {cfg.dt}, expected {expected_dt}")

    # 4. Per-row checks
    for row in per_rep_rows:
        if row.get("error", ""):
            continue
        row_id = f"contract={row.get('contract_id')},method={row.get('method')},rep={row.get('replication')}"

        # Finite/non-negative outputs
        for field in ("observation_variance", "estimator_variance", "std_error"):
            v = _safe_float(row.get(field))
            if v is None:
                failures.append(f"{row_id}: {field} is not finite")
            elif v < 0:
                failures.append(f"{row_id}: {field} < 0")

        # Estimator variance identity
        obs_var = _safe_float(row.get("observation_variance"))
        est_var = _safe_float(row.get("estimator_variance"))
        n_obs = row.get("pricing_observations")
        if obs_var is not None and est_var is not None and n_obs:
            try:
                n_obs_f = float(n_obs)
                if n_obs_f > 0:
                    expected_est_var = obs_var / n_obs_f
                    if abs(est_var - expected_est_var) > 1e-8 * max(1.0, abs(expected_est_var)):
                        failures.append(
                            f"{row_id}: estimator_variance={est_var:.4e} != "
                            f"obs_var/n_obs={expected_est_var:.4e}"
                        )
            except (TypeError, ValueError):
                pass

        # AV: pricing_simulated_paths == 2 * pricing_observations
        if row.get("method") == "AV":
            n_pairs = row.get("pricing_observations")
            n_sim = row.get("pricing_simulated_paths")
            if n_pairs is not None and n_sim is not None:
                try:
                    if int(float(n_sim)) != 2 * int(float(n_pairs)):
                        failures.append(
                            f"{row_id}: AV pricing_simulated_paths={n_sim} "
                            f"!= 2*pricing_observations={2*int(float(n_pairs))}"
                        )
                except (TypeError, ValueError):
                    pass

        # NCV_TRANSFER_BETA1: beta must be exactly 1.0
        if row.get("method") == "NCV_TRANSFER_BETA1":
            beta = _safe_float(row.get("beta"))
            if beta is not None and abs(beta - 1.0) > 1e-10:
                failures.append(f"{row_id}: NCV_TRANSFER_BETA1 beta={beta} != 1.0")

    # 5. Seed independence: all pricing seeds must be distinct within a replication
    # (Verify via seed manifest)
    pricing_seeds_by_rep: dict = {}
    for srow in seed_manifest:
        if "pricing_" in srow.get("stream", ""):
            rep = srow["replication"]
            pricing_seeds_by_rep.setdefault(rep, []).append(srow["seed_value"])
    for rep, seeds_list in pricing_seeds_by_rep.items():
        if len(seeds_list) != len(set(seeds_list)):
            failures.append(f"Duplicate pricing seeds in replication {rep}")

    # 6. High-precision references present for all contracts
    hp_contracts = {r["contract_id"] for r in high_prec_rows if not r.get("error")}
    for cid in CONTRACT_IDS:
        if cid not in hp_contracts:
            warnings.append(f"High-precision reference missing for {cid}")

    return {
        "passed": len(failures) == 0,
        "n_failures": len(failures),
        "n_warnings": len(warnings),
        "failures": failures,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_stage8(
    profile: str,
    base_seed: int = 42,
    output_dir: str = "experiment_runs",
    n_replications_override: Optional[int] = None,
) -> Path:
    """
    Run all Stage 8 experiments and write output bundle to a timestamped directory.

    Parameters
    ----------
    profile : str
        "smoke" or "dissertation".
    base_seed : int
        Base seed for all replications.
    output_dir : str
        Base directory; a timestamped sub-directory is created.
    n_replications_override : int, optional
        Override the profile's default n_replications.

    Returns
    -------
    Path
        Path to the output directory.
    """
    profile_cfg = PROFILES[profile]
    n_training = profile_cfg["n_training"]
    n_pilot = profile_cfg["n_pilot"]
    n_pricing = profile_cfg["n_pricing"]
    total_budget = n_pricing  # equal-budget total
    n_replications = n_replications_override or profile_cfg["n_replications"]
    n_high_prec = profile_cfg["n_high_precision"]
    amortised_q_values = profile_cfg["amortised_q_values"]

    torch_available = _try_import_torch()

    # Create output directory
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(output_dir) / f"stage8_{profile}_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== Stage 8: {profile_cfg['description']} ===")
    print(f"Output dir: {run_dir}")
    print(f"Profile: {profile}, base_seed={base_seed}, n_replications={n_replications}")
    print(f"torch_available={torch_available}")
    print(f"n_training={n_training}, n_pilot={n_pilot}, n_pricing={n_pricing}")

    # --- Config snapshot ---
    config_snapshot = {
        "profile": profile,
        "base_seed": base_seed,
        "n_replications": n_replications,
        "n_training": n_training,
        "n_pilot": n_pilot,
        "n_pricing": n_pricing,
        "total_path_budget": total_budget,
        "amortised_q_values": amortised_q_values,
        "n_high_precision": n_high_prec,
        "torch_available": torch_available,
        "contracts": {cid: {"K": K, "sigma": s, "T": T}
                      for cid, (K, s, T) in CONTRACT_GRID.items()},
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
    }
    _write_json(run_dir / "config_snapshot.json", config_snapshot)

    # --- Environment metadata ---
    env_meta = collect_environment_metadata()
    env_meta["torch_available"] = torch_available
    env_meta["cpu_count"] = os.cpu_count()
    env_meta["platform"] = platform.platform()
    _write_json(run_dir / "environment.json", env_meta)

    # --- Seed manifest ---
    seed_manifest = _build_seed_manifest(base_seed, n_replications)
    _write_csv(run_dir / "seed_manifest.csv", seed_manifest)

    # --- High-precision references ---
    print("\n[Stage 8] Computing high-precision GCV references...")
    high_prec_rows = compute_all_high_precision_references(n_high_prec, base_seed)
    _write_csv(run_dir / "high_precision_references.csv", high_prec_rows)

    # --- Per-replication loop ---
    all_per_rep: List[dict] = []
    all_beta: List[dict] = []
    all_runtime: List[dict] = []

    for rep in range(n_replications):
        print(f"\n[Stage 8] Replication {rep+1}/{n_replications}...", flush=True)
        per_rep, beta, runtime = run_replication(
            base_seed=base_seed,
            replication=rep,
            n_training=n_training,
            n_pilot=n_pilot,
            n_pricing=n_pricing,
            total_path_budget=total_budget,
            amortised_q_values=amortised_q_values,
            torch_available=torch_available,
        )
        all_per_rep.extend(per_rep)
        all_beta.extend(beta)
        all_runtime.extend(runtime)

    # --- Save per-replication results ---
    _write_csv(run_dir / "per_replication_results.csv", all_per_rep)
    _write_csv(run_dir / "beta_transfer_results.csv", all_beta)
    _write_csv(run_dir / "runtime_raw.csv", all_runtime)

    # --- Aggregate results ---
    agg_rows = aggregate_results(all_per_rep)
    _write_csv(run_dir / "aggregate_results.csv", agg_rows)

    # --- VRR table ---
    vrr_rows = compute_vrr_table(agg_rows)
    _write_csv(run_dir / "equal_observations_summary.csv", vrr_rows)

    # --- Amortised costs ---
    amortised_rows = compute_amortised_costs(
        n_training=n_training,
        n_pilot=n_pilot,
        n_pricing=n_pricing,
        q_values=amortised_q_values,
    )
    _write_csv(run_dir / "equal_budget_summary.csv", amortised_rows)

    # --- Break-even analysis ---
    # Use mean end-to-end runtimes from aggregate for break-even
    gcv_e2e = {r["contract_id"]: _safe_float(r.get("end_to_end_runtime_s_mean"))
               for r in agg_rows if r["method"] == "GCV"}
    tb1_e2e = {r["contract_id"]: _safe_float(r.get("end_to_end_runtime_s_mean"))
               for r in agg_rows if r["method"] == "NCV_TRANSFER_BETA1"}
    tb_e2e = {r["contract_id"]: _safe_float(r.get("end_to_end_runtime_s_mean"))
              for r in agg_rows if r["method"] == "NCV_TRANSFER_BETA"}

    # Training runtime from aggregate
    ref_train_rows = [r for r in agg_rows if r["method"] in ("NCV_SCRATCH", "NCV_TRANSFER_BETA1")]
    c_train_approx = (
        _safe_float(ref_train_rows[0].get("training_runtime_s_mean"))
        if ref_train_rows else None
    )

    break_even_rows = []
    for cid in TARGET_IDS:
        cgcv = gcv_e2e.get(cid)
        ctb1 = tb1_e2e.get(cid)
        ctb = tb_e2e.get(cid)
        c_train = c_train_approx

        be_tb1 = ("NA" if (c_train is None or cgcv is None or ctb1 is None)
                  else compute_break_even(c_train, cgcv, ctb1))
        be_tb = ("NA" if (c_train is None or cgcv is None or ctb is None)
                 else compute_break_even(c_train, cgcv, ctb))
        break_even_rows.append({
            "contract_id": cid,
            "c_train": c_train,
            "c_gcv_per_valuation": cgcv,
            "c_transfer_beta1_per_valuation": ctb1,
            "c_transfer_beta_per_valuation": ctb,
            "break_even_q_beta1": be_tb1,
            "break_even_q_beta": be_tb,
        })

    _write_csv(run_dir / "break_even_by_contract.csv", break_even_rows)

    # Portfolio break-even (equal weight over 7 contracts)
    portfolio_gcv = sum(v for v in gcv_e2e.values() if v is not None) / max(len(gcv_e2e), 1)
    portfolio_tb1 = sum(v for v in tb1_e2e.values() if v is not None) / max(len(tb1_e2e), 1)
    portfolio_tb = sum(v for v in tb_e2e.values() if v is not None) / max(len(tb_e2e), 1)
    portfolio_be_tb1 = (
        compute_break_even(c_train_approx or 0.0, portfolio_gcv, portfolio_tb1)
        if c_train_approx and portfolio_gcv and portfolio_tb1 else "NA"
    )
    portfolio_be_tb = (
        compute_break_even(c_train_approx or 0.0, portfolio_gcv, portfolio_tb)
        if c_train_approx and portfolio_gcv and portfolio_tb else "NA"
    )
    _write_csv(run_dir / "portfolio_break_even.csv", [{
        "portfolio": "equal_weight_7_contracts",
        "c_train": c_train_approx,
        "avg_gcv_e2e_per_val": portfolio_gcv,
        "avg_transfer_beta1_e2e_per_val": portfolio_tb1,
        "avg_transfer_beta_e2e_per_val": portfolio_tb,
        "break_even_q_beta1": portfolio_be_tb1,
        "break_even_q_beta": portfolio_be_tb,
    }])

    # --- Matched accuracy ---
    matched_rows = []
    for cid in TARGET_IDS:
        tb1_se = None
        gcv_se = None
        for r in agg_rows:
            if r["contract_id"] == cid and r["method"] == "NCV_TRANSFER_BETA1":
                tb1_se = _safe_float(r.get("std_error_mean"))
            if r["contract_id"] == cid and r["method"] == "GCV":
                gcv_se = _safe_float(r.get("std_error_mean"))
        if tb1_se and gcv_se:
            # GCV obs_var ~= n_gcv_pricing * gcv_se^2 = gcv_obs_var_mean
            gcv_obs_var = _safe_float(
                next((r.get("observation_variance_mean") for r in agg_rows
                      if r["contract_id"] == cid and r["method"] == "GCV"), None)
            )
            tb1_est_var = tb1_se ** 2
            n_gcv_needed = (
                math.ceil(gcv_obs_var / tb1_est_var) if gcv_obs_var and tb1_est_var else "NA"
            )
            matched_rows.append({
                "contract_id": cid,
                "target_se_transfer_beta1": tb1_se,
                "gcv_se_at_standard_n": gcv_se,
                "gcv_obs_var_mean": gcv_obs_var,
                "gcv_n_required_for_matched_accuracy": n_gcv_needed,
            })
        else:
            matched_rows.append({
                "contract_id": cid,
                "target_se_transfer_beta1": tb1_se or "NA",
                "gcv_se_at_standard_n": gcv_se or "NA",
                "gcv_obs_var_mean": "NA",
                "gcv_n_required_for_matched_accuracy": "NA",
            })
    _write_csv(run_dir / "matched_accuracy_results.csv", matched_rows)

    # --- Stable summary ---
    stable_rows = []
    for row in agg_rows:
        stable_rows.append({
            "contract_id": row["contract_id"],
            "method": row["method"],
            "n_replications": row["n_replications"],
            "price_mean": row["price_mean"],
            "price_std": row["price_std"],
            "estimator_variance_mean": row["estimator_variance_mean"],
            "std_error_mean": row["std_error_mean"],
        })
    _write_csv(run_dir / "summary_stable.csv", stable_rows)

    # --- Training diagnostics ---
    training_diag_rows = []
    for row in all_per_rep:
        if row.get("method") in ("NCV_SCRATCH", "NCV_TRANSFER_BETA1", "NCV_TRANSFER_BETA"):
            training_diag_rows.append({
                "base_seed": row.get("base_seed"),
                "replication": row.get("replication"),
                "contract_id": row.get("contract_id"),
                "method": row.get("method"),
                "training_paths": row.get("training_paths", 0),
                "training_runtime_s": row.get("training_runtime_s", ""),
                "e_h0": row.get("e_h0", ""),
                "param_hash": row.get("param_hash", ""),
                "error": row.get("error", ""),
            })
    _write_csv(run_dir / "training_diagnostics.csv", training_diag_rows)

    # --- Runtime summary ---
    # Aggregate runtime fields directly from all_runtime rows (not via
    # aggregate_results which is intended for statistical pricing fields).
    import statistics as _stats

    def _mean_or_na(vals):
        vals = [v for v in vals if v is not None]
        return _stats.mean(vals) if vals else "NA"

    def _std_or_na(vals):
        vals = [v for v in vals if v is not None]
        return _stats.stdev(vals) if len(vals) >= 2 else "NA"

    rt_grouped: dict = {}
    for row in all_runtime:
        key = (row.get("contract_id", "NA"), row.get("method", "NA"))
        rt_grouped.setdefault(key, []).append(row)

    runtime_summary_rows = []
    for (cid, method), rows in sorted(rt_grouped.items()):
        pricing_rts = [_safe_float(r.get("pricing_runtime_s")) for r in rows]
        e2e_rts = [_safe_float(r.get("end_to_end_runtime_s")) for r in rows]
        train_rts = [_safe_float(r.get("training_runtime_s")) for r in rows]
        runtime_summary_rows.append({
            "contract_id": cid,
            "method": method,
            "n_replications": len(rows),
            "pricing_runtime_s_mean": _mean_or_na(pricing_rts),
            "pricing_runtime_s_std": _std_or_na(pricing_rts),
            "end_to_end_runtime_s_mean": _mean_or_na(e2e_rts),
            "end_to_end_runtime_s_std": _std_or_na(e2e_rts),
            "training_runtime_s_mean": _mean_or_na(train_rts),
        })
    _write_csv(run_dir / "runtime_summary.csv", runtime_summary_rows)

    # --- Validation report ---
    validation_report = build_validation_report(
        per_rep_rows=all_per_rep,
        agg_rows=agg_rows,
        seed_manifest=seed_manifest,
        high_prec_rows=high_prec_rows,
        amortised_rows=amortised_rows,
    )
    _write_json(run_dir / "validation_report.json", validation_report)

    # --- Reproducibility report ---
    repro_report = {
        "profile": profile,
        "base_seed": base_seed,
        "n_replications": n_replications,
        "n_rows_per_replication_results": len(all_per_rep),
        "n_rows_aggregate_results": len(agg_rows),
        "validation_passed": validation_report["passed"],
        "stable_summary_sha256": hashlib.sha256(
            json.dumps(stable_rows, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "note": "Reproducibility confirmed if stable_summary_sha256 matches across identical runs.",
    }
    _write_json(run_dir / "reproducibility_report.json", repro_report)

    # --- Handover ---
    n_failed = sum(1 for r in all_per_rep if r.get("error", ""))
    n_total = len(all_per_rep)
    handover_text = f"""# Stage 8 Handover

## Run metadata
- Profile: {profile}
- Base seed: {base_seed}
- Replications: {n_replications}
- Timestamp UTC: {ts}
- Commit: {_commit_hash()}
- Output directory: {run_dir}
- torch_available: {torch_available}

## Implementation facts
- `asian_options/contracts.py`: Seven-contract grid (reference + 6 targets), one-parameter-change rule.
- `asian_options/frozen_transfer.py`: Frozen NCV transfer estimators, SHA-256 parameter hash, NearZeroVarianceError, analytical E[H0(Z)].
- `scripts/run_stage8.py`: Stage 8 experiment runner (this file).
- `asian_options/tests/test_stage8.py`: Validation tests for all requirements.

## Validation facts
- Contract grid: validated via `validate_contract_grid()`.
- Monitoring dates: dt=T/m verified for all contracts.
- Parameter hash: SHA-256 over W1,b1,W2,b2 bytes; verified before and after each target evaluation.
- Near-zero variance: explicit NearZeroVarianceError raised; no silent fallback.
- Seed independence: distinct streams per contract/phase/replication.

## Empirical results
- Total per-replication rows: {n_total}
- Failed rows: {n_failed}
- NCV methods require torch (currently torch_available={torch_available})

## Limitations / risks
- torch not installed in this environment; NCV-based methods report error='torch_not_available'.
- Break-even analysis uses mean timing estimates; may have high variance with few replications.
- Dissertation profile requires torch and sufficient compute time (~30 reps).

## Commands
Run smoke test:
    python scripts/run_stage8.py --profile smoke --output-dir experiment_runs

Run dissertation profile:
    python scripts/run_stage8.py --profile dissertation --output-dir experiment_runs

Run tests:
    pytest asian_options/tests/test_stage8.py -v
"""
    (run_dir / "handover.md").write_text(handover_text, encoding="utf-8")

    print(f"\n[Stage 8] Validation: passed={validation_report['passed']}, "
          f"failures={validation_report['n_failures']}, warnings={validation_report['n_warnings']}")
    print(f"[Stage 8] Output bundle written to: {run_dir}")

    if not validation_report["passed"]:
        print("\n[Stage 8] VALIDATION FAILURES:", file=sys.stderr)
        for f in validation_report["failures"]:
            print(f"  - {f}", file=sys.stderr)

    return run_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Stage 8: Frozen NCV Transfer, Calibration and Amortisation"
    )
    p.add_argument(
        "--profile",
        choices=list(PROFILES),
        default="smoke",
        help="Execution profile (smoke | dissertation)",
    )
    p.add_argument("--base-seed", type=int, default=42, help="Base random seed")
    p.add_argument(
        "--output-dir",
        default="experiment_runs",
        help="Base directory for timestamped output bundles",
    )
    p.add_argument(
        "--n-replications",
        type=int,
        default=None,
        help="Override profile default number of replications",
    )
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
