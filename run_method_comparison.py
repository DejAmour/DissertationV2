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

Comparison modes
----------------
A) Equal pricing-observation comparison (Stage 4)
   Every method receives the same number of independent estimator observations
   (cfg.n_paths).  Reports raw observation variance and variance-reduction
   ratio.  AV uses 2 * n_paths simulated paths (2 per pair observation).

B) Equal total-path-budget comparison (Stage 4)
   Every method receives the same total simulated-path allowance
   (TOTAL_PATH_BUDGET).  AV gets half as many pair observations (budget/2).
   CV deducts pilot paths from pricing allowance (budget - n_pilot).
   NCV deducts training paths from pricing allowance (budget - n_training).
   Reports resulting standard error and estimator variance.

C) Runtime/efficiency comparison (Stage 5)
   Measures wall-clock time for each method in equal-pricing-observation
   mode.  Reports time_per_observation, time_per_simulated_path, and
   efficiency_gain_vs_mc.

   Timing scope: pricing step only (simulation + payoff + correction).
   NCV training is excluded from pricing timing and noted separately.

   Efficiency formula:
       efficiency_gain = (MC_estimator_variance * MC_runtime_s)
                       / (method_estimator_variance * method_runtime_s)

Output
------
Three CSV files:
  asian_options_equal_obs_comparison.csv
  asian_options_equal_budget_comparison.csv
  asian_options_runtime_comparison.csv
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
from asian_options.results import save_results_csv, print_comparison_table, make_runtime_row, print_runtime_table

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
OUTPUT_RUNTIME = "asian_options_runtime_comparison.csv"

RUNTIME_CSV_FIELDNAMES = [
    "comparison_mode",
    "method",
    "runtime_seconds",
    "pricing_observations",
    "pricing_simulated_paths",
    "time_per_observation",
    "time_per_simulated_path",
    "estimator_variance",
    "efficiency_gain_vs_mc",
    "timing_scope",
]

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
# Mode C — runtime/efficiency comparison (Stage 5)
# ---------------------------------------------------------------------------

def run_runtime_comparison() -> list[dict]:
    """
    Measure wall-clock time for each method under equal pricing-observation
    mode (cfg.n_paths observations each).

    Timing scope
    ------------
    The timed region covers: path simulation + payoff computation + any
    control-variate correction.  It does **not** include NCV network training
    (which is a one-off setup cost); training time is reported separately in
    the ``timing_scope`` field.

    Efficiency formula
    ------------------
    efficiency_gain_vs_mc = (MC_estimator_variance * MC_runtime_s)
                           / (method_estimator_variance * method_runtime_s)

    Values > 1 mean the method delivers the same precision as MC in less
    compute time.  This is *not* the same as the variance-reduction ratio.
    """
    import time as _time
    seed_everything(CFG.seed)
    rows = []

    # --- MC ---
    print("  [C] Timing MC...", flush=True)
    t0 = _time.perf_counter()
    mc = standard_monte_carlo(CFG)
    mc_runtime = _time.perf_counter() - t0
    mc_est_var = mc.estimator_variance
    mc_row = make_runtime_row(
        "C_runtime", "MC", mc, mc_runtime, mc_est_var, mc_runtime,
        timing_scope="pricing only (simulation + payoff)",
    )
    rows.append(mc_row)
    print(f"      MC: runtime={mc_runtime:.4f}s  est_var={mc_est_var:.4e}")

    # --- AV ---
    print("  [C] Timing AV...", flush=True)
    t0 = _time.perf_counter()
    av = antithetic_variates(CFG)
    av_runtime = _time.perf_counter() - t0
    rows.append(make_runtime_row(
        "C_runtime", "AV", av, av_runtime, mc_est_var, mc_runtime,
        timing_scope="pricing only (antithetic pair simulation + payoff)",
    ))
    print(f"      AV: runtime={av_runtime:.4f}s  est_var={av.estimator_variance:.4e}")

    # --- CV ---
    print("  [C] Timing CV (includes pilot)...", flush=True)
    t0 = _time.perf_counter()
    cv = geometric_control_variate(CFG, n_pilot=N_PILOT)
    cv_runtime = _time.perf_counter() - t0
    rows.append(make_runtime_row(
        "C_runtime", "CV", cv, cv_runtime, mc_est_var, mc_runtime,
        timing_scope=f"pricing + pilot ({N_PILOT} pilot paths included)",
    ))
    print(f"      CV: runtime={cv_runtime:.4f}s  est_var={cv.estimator_variance:.4e}")

    # --- NCV ---
    print("  [C] Timing NCV (pricing only; training excluded)...", flush=True)
    try:
        # Train first (not timed as pricing runtime)
        from asian_options.neural_cv import build_network, train_network
        from asian_options.simulate_gbm import simulate_paths
        from asian_options.payoffs import arithmetic_asian_call_payoff
        import time as _t2

        train_cfg = dataclasses.replace(CFG, n_paths=N_TRAINING, seed=CFG.seed + 100)
        train_paths = simulate_paths(train_cfg)
        train_payoffs = arithmetic_asian_call_payoff(train_paths, train_cfg)
        dt = train_cfg.dt
        drift = (train_cfg.r - train_cfg.q - 0.5 * train_cfg.sigma ** 2) * dt
        diffusion = math.sqrt(dt) * train_cfg.sigma
        log_S = np.log(train_paths / train_cfg.S0)
        log_inc = np.diff(np.hstack([np.zeros((N_TRAINING, 1)), log_S]), axis=1)
        Z_train = (log_inc - drift) / diffusion
        dataset = {"X_train": Z_train, "y_train": train_payoffs}
        network = build_network(train_cfg, hidden_width=32)
        t_train_start = _t2.perf_counter()
        train_network(network, dataset, train_cfg, n_epochs=100)
        training_time = _t2.perf_counter() - t_train_start

        # Timed pricing step
        from asian_options.neural_cv import ncv_estimator
        price_cfg = dataclasses.replace(CFG, seed=CFG.seed + 200)
        t0 = _t2.perf_counter()
        ncv = ncv_estimator(network, price_cfg, n_training_paths=N_TRAINING)
        ncv_pricing_runtime = _t2.perf_counter() - t0

        rows.append(make_runtime_row(
            "C_runtime", "NCV", ncv, ncv_pricing_runtime,
            mc_est_var, mc_runtime,
            timing_scope=(
                f"pricing only ({N_TRAINING} training paths excluded; "
                f"training_time={training_time:.4f}s noted separately)"
            ),
        ))
        print(f"      NCV: pricing_runtime={ncv_pricing_runtime:.4f}s  "
              f"training_time={training_time:.4f}s  "
              f"est_var={ncv.estimator_variance:.4e}")
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        print(f"      NCV FAILED: {err}", file=sys.stderr)
        rows.append({
            "comparison_mode": "C_runtime",
            "method": "NCV",
            "runtime_seconds": "ERROR",
            "pricing_observations": "ERROR",
            "pricing_simulated_paths": "ERROR",
            "time_per_observation": "ERROR",
            "time_per_simulated_path": "ERROR",
            "estimator_variance": "ERROR",
            "efficiency_gain_vs_mc": "ERROR",
            "timing_scope": err,
        })

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

    print("\n=== Mode C: Runtime/efficiency comparison (Stage 5) ===")
    rows_c = run_runtime_comparison()
    save_results_csv(rows_c, OUTPUT_RUNTIME, fieldnames=RUNTIME_CSV_FIELDNAMES)
    print(f"\nResults written to {OUTPUT_RUNTIME}")
    print_runtime_table(rows_c)


if __name__ == "__main__":
    run_comparison()
