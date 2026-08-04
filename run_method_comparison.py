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

Output
------
CSV file: asian_options_method_comparison.csv
Columns: method, price, variance, std_error, ci_lower, ci_upper,
         n_paths, runtime_s, speed_ratio_vs_mc, notes
"""

from __future__ import annotations

import csv
import sys
import traceback

from asian_options.config import ModelConfig
from asian_options.estimators import (
    standard_monte_carlo,
    antithetic_variates,
    geometric_control_variate,
)

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

OUTPUT_CSV = "asian_options_method_comparison.csv"
CSV_FIELDNAMES = [
    "method",
    "price",
    "variance",
    "std_error",
    "ci_lower",
    "ci_upper",
    "n_paths",
    "runtime_s",
    "speed_ratio_vs_mc",
    "notes",
]


def _result_to_row(method: str, result, mc_variance: float, notes: str = "") -> dict:
    """Convert an EstimateResult/CVEstimateResult to a CSV row dict."""
    speed_ratio = (
        mc_variance / result.variance if result.variance > 0 else float("nan")
    )
    return {
        "method": method,
        "price": f"{result.price:.6f}",
        "variance": f"{result.variance:.8e}",
        "std_error": f"{result.std_error:.6f}",
        "ci_lower": f"{result.ci_lower:.6f}",
        "ci_upper": f"{result.ci_upper:.6f}",
        "n_paths": result.n_paths,
        "runtime_s": f"{result.runtime_s:.4f}",
        "speed_ratio_vs_mc": f"{speed_ratio:.4f}",
        "notes": notes,
    }


def _error_row(method: str, error: str) -> dict:
    """Return a placeholder CSV row for a failed method."""
    return {
        "method": method,
        "price": "ERROR",
        "variance": "ERROR",
        "std_error": "ERROR",
        "ci_lower": "ERROR",
        "ci_upper": "ERROR",
        "n_paths": "ERROR",
        "runtime_s": "ERROR",
        "speed_ratio_vs_mc": "ERROR",
        "notes": error,
    }


def run_comparison():
    rows = []

    # ------------------------------------------------------------------
    # 1. Standard Monte Carlo (baseline)
    # ------------------------------------------------------------------
    print("Running Standard Monte Carlo...", flush=True)
    mc_result = standard_monte_carlo(CFG)
    mc_variance = mc_result.variance
    rows.append(_result_to_row("MC", mc_result, mc_variance))
    print(f"  MC  price={mc_result.price:.4f}  variance={mc_result.variance:.6e}")

    # ------------------------------------------------------------------
    # 2. Antithetic Variates
    # ------------------------------------------------------------------
    print("Running Antithetic Variates...", flush=True)
    av_result = antithetic_variates(CFG)
    rows.append(_result_to_row("AV", av_result, mc_variance))
    print(f"  AV  price={av_result.price:.4f}  variance={av_result.variance:.6e}")

    # ------------------------------------------------------------------
    # 3. Geometric Control Variate
    # ------------------------------------------------------------------
    print("Running Geometric Control Variate...", flush=True)
    cv_result = geometric_control_variate(CFG, n_pilot=1000)
    rows.append(_result_to_row("CV", cv_result, mc_variance))
    print(f"  CV  price={cv_result.price:.4f}  variance={cv_result.variance:.6e}")

    # ------------------------------------------------------------------
    # 4. Neural Control Variate (failure must not crash full run)
    # ------------------------------------------------------------------
    print("Running Neural Control Variate...", flush=True)
    try:
        from asian_options.neural_cv import build_network, train_network, ncv_estimator
        from asian_options.simulate_gbm import simulate_paths
        from asian_options.payoffs import arithmetic_asian_call_payoff
        import dataclasses

        # Build training dataset (separate seed from pricing to avoid bias)
        train_cfg = dataclasses.replace(CFG, n_paths=5_000, seed=CFG.seed + 100)
        train_paths = simulate_paths(train_cfg)
        train_payoffs = arithmetic_asian_call_payoff(train_paths, train_cfg)
        # Use GBM log-increments as inputs: shape (n_paths, m)
        import numpy as np
        import math
        dt = train_cfg.dt
        drift = (train_cfg.r - train_cfg.q - 0.5 * train_cfg.sigma ** 2) * dt
        diffusion = train_cfg.sigma * math.sqrt(dt)
        # Recover Z from paths (log-increments normalised)
        log_S = np.log(train_paths / train_cfg.S0)
        log_inc = np.diff(
            np.hstack([np.zeros((train_cfg.n_paths, 1)), log_S]), axis=1
        )
        Z_train = (log_inc - drift) / diffusion  # approx standard normal inputs

        dataset = {"X_train": Z_train, "y_train": train_payoffs}

        network = build_network(train_cfg, hidden_width=32)
        train_network(network, dataset, train_cfg, n_epochs=100)

        # Price using independent seed
        price_cfg = dataclasses.replace(CFG, seed=CFG.seed + 200)
        ncv_result = ncv_estimator(network, price_cfg)
        rows.append(_result_to_row("NCV", ncv_result, mc_variance))
        print(
            f"  NCV price={ncv_result.price:.4f}  variance={ncv_result.variance:.6e}"
        )
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"
        print(f"  NCV FAILED: {err_msg}", file=sys.stderr)
        rows.append(_error_row("NCV", err_msg))

    # ------------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------------
    with open(OUTPUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults written to {OUTPUT_CSV}")

    # Print a quick summary table
    print("\n{:<6} {:>10} {:>14} {:>18}".format(
        "Method", "Price", "Variance", "SpeedRatioVsMC"
    ))
    print("-" * 52)
    for row in rows:
        print("{:<6} {:>10} {:>14} {:>18}".format(
            row["method"],
            row["price"],
            row["variance"],
            row["speed_ratio_vs_mc"],
        ))


if __name__ == "__main__":
    run_comparison()
