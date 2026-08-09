"""
run_method_comparison.py
========================
End-to-end comparison of MC variance-reduction methods for arithmetic Asian
call pricing.

Methods compared
----------------
1. Standard Monte Carlo (MC)
2. Antithetic Variates (AV)
3. Geometric Control Variate (CV)
4. Neural Control Variate (NCV)

Comparison modes (Stage 4)
--------------------------
A) Equal pricing-observation comparison
   Every method receives the same number of independent estimator observations
   (cfg.n_paths).  Reports raw observation variance and variance-reduction
   ratio.  AV uses 2 * n_paths simulated paths (2 per pair observation).

B) Equal total-path-budget comparison
   Every method receives the same total simulated-path allowance
   (TOTAL_PATH_BUDGET).  AV gets half as many pair observations (budget/2).
   CV deducts pilot paths from pricing allowance (budget - n_pilot).
   NCV deducts training paths from pricing allowance (budget - n_training).
   Reports resulting standard error and estimator variance.

Output
------
Two CSV files:
  asian_options_equal_obs_comparison.csv
  asian_options_equal_budget_comparison.csv
"""

from __future__ import annotations

import math
import sys
import traceback
import dataclasses

import numpy as np

from asian_options.config import ModelConfig, seed_everything
from asian_options.estimators import (
    standard_monte_carlo,
    antithetic_variates,
    geometric_control_variate,
)
from asian_options.results import save_results_csv, print_comparison_table

# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------
CFG = ModelConfig(
    S0=100.0,
    K=100.0,
    r=0.05,
    sigma=0.2,
    T=1.0,
    m=12,
    n_paths=50_000,
    seed=42,
)

# Pilot and training sizes (fixed across both comparison modes)
N_PILOT = 1_000
N_TRAINING = 5_000
TOTAL_PATH_BUDGET = 50_000   # equal-budget mode

OUTPUT_EQUAL_OBS = "asian_options_equal_obs_comparison.csv"
OUTPUT_EQUAL_BUDGET = "asian_options_equal_budget_comparison.csv"

CSV_FIELDNAMES = [
    "comparison_mode",
    "method",
    "pricing_observations",
    "pricing_simulated_paths",
    "pilot_paths",
    "training_paths",
    "total_simulated_paths",
    "price",
    "observation_variance",
    "estimator_variance",
    "variance_reduction_ratio",
    "std_error",
    "ci_lower",
    "ci_upper",
    "runtime_s",
    "notes",
]


def _vrr(mc_obs_var: float, method_obs_var: float) -> str:
    """Variance-reduction ratio = MC obs variance / method obs variance."""
    if method_obs_var > 0:
        return f"{mc_obs_var / method_obs_var:.4f}"
    return "nan"


def _result_to_row(
    mode: str,
    method: str,
    result,
    mc_obs_var: float,
    notes: str = "",
) -> dict:
    """Convert an EstimateResult/CVEstimateResult to a CSV row dict."""
    obs_var = getattr(result, "observation_variance", result.variance)
    est_var = getattr(result, "estimator_variance", result.variance / result.n_paths)
    return {
        "comparison_mode": mode,
        "method": method,
        "pricing_observations": getattr(result, "pricing_observations", result.n_paths),
        "pricing_simulated_paths": getattr(result, "pricing_simulated_paths", result.n_paths),
        "pilot_paths": getattr(result, "pilot_paths", 0),
        "training_paths": getattr(result, "training_paths", 0),
        "total_simulated_paths": getattr(result, "total_simulated_paths", result.n_paths),
        "price": f"{result.price:.6f}",
        "observation_variance": f"{obs_var:.8e}",
        "estimator_variance": f"{est_var:.8e}",
        "variance_reduction_ratio": _vrr(mc_obs_var, obs_var),
        "std_error": f"{result.std_error:.6f}",
        "ci_lower": f"{result.ci_lower:.6f}",
        "ci_upper": f"{result.ci_upper:.6f}",
        "runtime_s": f"{result.runtime_s:.4f}",
        "notes": notes,
    }


def _error_row(mode: str, method: str, error: str) -> dict:
    """Return a placeholder CSV row for a failed method."""
    return {k: ("ERROR" if k not in ("comparison_mode", "method", "notes") else v)
            for k, v in [("comparison_mode", mode), ("method", method), ("notes", error)]
            + [(f, "ERROR") for f in CSV_FIELDNAMES
               if f not in ("comparison_mode", "method", "notes")]}


def _run_ncv(cfg: ModelConfig, n_train: int, seed_offset_train: int = 100, seed_offset_price: int = 200):
    """Run NCV, returning (result, n_training_paths) or raise."""
    from asian_options.neural_cv import build_network, train_network, ncv_estimator
    from asian_options.simulate_gbm import simulate_paths
    from asian_options.payoffs import arithmetic_asian_call_payoff

    train_cfg = dataclasses.replace(cfg, n_paths=n_train, seed=cfg.seed + seed_offset_train)
    train_paths = simulate_paths(train_cfg)
    train_payoffs = arithmetic_asian_call_payoff(train_paths, train_cfg)

    dt = train_cfg.dt
    drift = (train_cfg.r - train_cfg.q - 0.5 * train_cfg.sigma ** 2) * dt
    diffusion = train_cfg.sigma * math.sqrt(dt)
    log_S = np.log(train_paths / train_cfg.S0)
    log_inc = np.diff(
        np.hstack([np.zeros((train_cfg.n_paths, 1)), log_S]), axis=1
    )
    Z_train = (log_inc - drift) / diffusion

    dataset = {"X_train": Z_train, "y_train": train_payoffs}
    network = build_network(train_cfg, hidden_width=32)
    train_network(network, dataset, train_cfg, n_epochs=100)

    price_cfg = dataclasses.replace(cfg, seed=cfg.seed + seed_offset_price)
    ncv_result = ncv_estimator(network, price_cfg, n_training_paths=n_train)
    return ncv_result


# ---------------------------------------------------------------------------
# Mode A — equal pricing-observation comparison
# ---------------------------------------------------------------------------

def run_equal_obs_comparison() -> list[dict]:
    """
    Every method uses cfg.n_paths independent estimator observations.

    AV note: each of the n_paths pair-observations requires 2 simulated paths,
    so AV total_simulated_paths = 2 * n_paths (disclosed in output).
    """
    seed_everything(CFG.seed)
    rows = []

    # --- MC ---
    print("  [A] Running MC...", flush=True)
    mc = standard_monte_carlo(CFG)
    mc_obs_var = mc.observation_variance
    rows.append(_result_to_row("A_equal_obs", "MC", mc, mc_obs_var))
    print(f"      MC: price={mc.price:.4f}  obs_var={mc.observation_variance:.4e}")

    # --- AV (n_paths pair observations => 2*n_paths simulated paths) ---
    print("  [A] Running AV (note: 2 paths per pair observation)...", flush=True)
    av = antithetic_variates(CFG)
    rows.append(_result_to_row("A_equal_obs", "AV", av, mc_obs_var,
                               notes=f"2 paths per pair; total_simulated_paths={av.total_simulated_paths}"))
    print(f"      AV: price={av.price:.4f}  obs_var={av.observation_variance:.4e}  "
          f"vrr={mc_obs_var / av.observation_variance:.3f}")

    # --- CV ---
    print("  [A] Running CV...", flush=True)
    cv = geometric_control_variate(CFG, n_pilot=N_PILOT)
    rows.append(_result_to_row("A_equal_obs", "CV", cv, mc_obs_var,
                               notes=f"pilot_paths={N_PILOT}"))
    print(f"      CV: price={cv.price:.4f}  obs_var={cv.observation_variance:.4e}  "
          f"vrr={mc_obs_var / cv.observation_variance:.3f}")

    # --- NCV ---
    print("  [A] Running NCV...", flush=True)
    try:
        ncv = _run_ncv(CFG, n_train=N_TRAINING)
        rows.append(_result_to_row("A_equal_obs", "NCV", ncv, mc_obs_var,
                                   notes=f"training_paths={N_TRAINING}"))
        print(f"      NCV: price={ncv.price:.4f}  obs_var={ncv.observation_variance:.4e}")
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        print(f"      NCV FAILED: {err}", file=sys.stderr)
        rows.append(_error_row("A_equal_obs", "NCV", err))

    return rows


# ---------------------------------------------------------------------------
# Mode B — equal total-path-budget comparison
# ---------------------------------------------------------------------------

def run_equal_budget_comparison() -> list[dict]:
    """
    Every method receives TOTAL_PATH_BUDGET simulated paths in total.

    Allocation:
    - MC  : pricing_observations = TOTAL_PATH_BUDGET
    - AV  : pricing_observations = TOTAL_PATH_BUDGET // 2 (2 paths per pair)
    - CV  : pricing_observations = TOTAL_PATH_BUDGET - N_PILOT
    - NCV : pricing_observations = TOTAL_PATH_BUDGET - N_TRAINING
    """
    seed_everything(CFG.seed)
    rows = []

    # --- MC ---
    mc_n = TOTAL_PATH_BUDGET
    print(f"  [B] Running MC (n_paths={mc_n})...", flush=True)
    mc_cfg = dataclasses.replace(CFG, n_paths=mc_n)
    mc = standard_monte_carlo(mc_cfg)
    mc_obs_var = mc.observation_variance
    rows.append(_result_to_row("B_equal_budget", "MC", mc, mc_obs_var))
    print(f"      MC: price={mc.price:.4f}  SE={mc.std_error:.5f}  "
          f"est_var={mc.estimator_variance:.4e}")

    # --- AV (half the pairs to stay within budget) ---
    av_pairs = TOTAL_PATH_BUDGET // 2
    print(f"  [B] Running AV (n_pairs={av_pairs}, total_paths={2*av_pairs})...", flush=True)
    av_cfg = dataclasses.replace(CFG, n_paths=av_pairs)
    av = antithetic_variates(av_cfg)
    rows.append(_result_to_row("B_equal_budget", "AV", av, mc_obs_var,
                               notes=f"AV uses 2 paths per pair; pricing_obs={av_pairs}"))
    print(f"      AV: price={av.price:.4f}  SE={av.std_error:.5f}  "
          f"est_var={av.estimator_variance:.4e}")

    # --- CV (deduct pilot from budget) ---
    cv_pricing = TOTAL_PATH_BUDGET - N_PILOT
    print(f"  [B] Running CV (pilot={N_PILOT}, pricing={cv_pricing})...", flush=True)
    cv_cfg = dataclasses.replace(CFG, n_paths=cv_pricing)
    cv = geometric_control_variate(cv_cfg, n_pilot=N_PILOT)
    rows.append(_result_to_row("B_equal_budget", "CV", cv, mc_obs_var,
                               notes=f"pilot_paths={N_PILOT} deducted from budget"))
    print(f"      CV: price={cv.price:.4f}  SE={cv.std_error:.5f}  "
          f"est_var={cv.estimator_variance:.4e}")

    # --- NCV (deduct training from budget) ---
    ncv_pricing = TOTAL_PATH_BUDGET - N_TRAINING
    print(f"  [B] Running NCV (training={N_TRAINING}, pricing={ncv_pricing})...", flush=True)
    try:
        ncv_price_cfg = dataclasses.replace(CFG, n_paths=ncv_pricing)
        ncv = _run_ncv(ncv_price_cfg, n_train=N_TRAINING)
        rows.append(_result_to_row("B_equal_budget", "NCV", ncv, mc_obs_var,
                                   notes=f"training_paths={N_TRAINING} deducted from budget"))
        print(f"      NCV: price={ncv.price:.4f}  SE={ncv.std_error:.5f}  "
              f"est_var={ncv.estimator_variance:.4e}")
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        print(f"      NCV FAILED: {err}", file=sys.stderr)
        rows.append(_error_row("B_equal_budget", "NCV", err))

    return rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_comparison():
    print("\n=== Mode A: Equal pricing-observation comparison ===")
    rows_a = run_equal_obs_comparison()
    save_results_csv(rows_a, OUTPUT_EQUAL_OBS, fieldnames=CSV_FIELDNAMES)
    print(f"\nResults written to {OUTPUT_EQUAL_OBS}")
    print_comparison_table(rows_a)

    print("\n=== Mode B: Equal total-path-budget comparison ===")
    rows_b = run_equal_budget_comparison()
    save_results_csv(rows_b, OUTPUT_EQUAL_BUDGET, fieldnames=CSV_FIELDNAMES)
    print(f"\nResults written to {OUTPUT_EQUAL_BUDGET}")
    print_comparison_table(rows_b)


if __name__ == "__main__":
    run_comparison()
