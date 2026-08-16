"""
frozen_transfer.py
==================
Stage 8: Frozen NCV transfer estimators.

Implements NCV_TRANSFER_BETA1 and NCV_TRANSFER_BETA for transferring a
reference network trained at the reference contract to six non-reference
target contracts.

Reference network protocol
--------------------------
1. Train one reference network at the reference contract.
2. Freeze parameters post-training (numpy arrays, no further updates).
3. Compute E[H0(Z)] analytically (Z ~ N(0, I_m)).
4. Record a deterministic SHA-256 hash of the frozen parameters.
5. Reuse exactly that frozen network across all six target contracts.
6. Verify hash unchanged after each target evaluation (call verify_frozen_hash).

Centered control definition
----------------------------
    C0(Z) = H0(Z) - E[H0(Z)]

Transfer estimators
-------------------
NCV_TRANSFER_BETA1 (beta = 1):
    Y = f_theta(Z) - C0(Z) = f_theta(Z) - H0(Z) + E[H0(Z)]

NCV_TRANSFER_BETA (pilot-estimated beta):
    beta_hat = Cov(f_theta, C0) / Var(C0)   [from independent pilot]
    Y = f_theta(Z) - beta_hat * C0(Z)

Pilot requirements
------------------
- Pilot is independent of training, validation, and final pricing.
- If Var(C0) is near-zero or non-finite, a NearZeroVarianceError is raised
  (no silent fallback).
- Consistent covariance/variance conventions with the rest of the project
  (ddof=1 throughout).

Dependencies
------------
Requires PyTorch for training the reference network.  All functions that
require torch raise ImportError with a clear message if torch is unavailable.
"""
from __future__ import annotations

import hashlib
import math
import time
from typing import Optional, Tuple

import numpy as np

from asian_options.config import ModelConfig
from asian_options.neural_cv import (
    _ShallowNet,
    build_network,
    train_network,
    analytical_network_expectation,
)


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class NearZeroVarianceError(RuntimeError):
    """
    Raised when Var(C0) is near-zero or non-finite during pilot-beta estimation.

    Callers should record this as an explicit failure (NA) with diagnostics
    rather than silently falling back to beta=1 or beta=0.
    """


# ---------------------------------------------------------------------------
# Parameter hashing
# ---------------------------------------------------------------------------

def compute_network_hash(network: _ShallowNet) -> str:
    """
    Compute a deterministic SHA-256 hash of all network parameters.

    The hash is derived from the concatenated raw bytes of W1, b1, W2, b2
    in a canonical (C-order) byte layout.  This provides a lightweight
    integrity check that the frozen network has not been modified between
    the reference training phase and any subsequent target evaluation.

    Parameters
    ----------
    network : _ShallowNet
        Network whose parameters are to be hashed.

    Returns
    -------
    str
        64-character lowercase hexadecimal SHA-256 digest.
    """
    h = hashlib.sha256()
    for arr in (network.W1, network.b1, network.W2, network.b2):
        h.update(np.ascontiguousarray(arr, dtype=np.float64).tobytes())
    return h.hexdigest()


def verify_frozen_hash(network: _ShallowNet, expected_hash: str) -> None:
    """
    Verify that the network parameters have not changed since freezing.

    Parameters
    ----------
    network : _ShallowNet
        Network to verify.
    expected_hash : str
        SHA-256 digest recorded immediately after training/freezing.

    Raises
    ------
    RuntimeError
        If the current hash does not match the expected hash.
    """
    current = compute_network_hash(network)
    if current != expected_hash:
        raise RuntimeError(
            f"Frozen network hash mismatch: expected {expected_hash!r}, "
            f"got {current!r}.  The reference network parameters have been "
            f"modified after freezing."
        )


# ---------------------------------------------------------------------------
# Reference network training
# ---------------------------------------------------------------------------

def _extract_z_shocks(
    paths: np.ndarray,
    cfg: ModelConfig,
) -> np.ndarray:
    """
    Recover the standard-normal shock matrix Z from simulated GBM paths.

    Under risk-neutral GBM::

        log(S_{j}/S_{j-1}) = (r - q - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z_j

    So Z_j = (log_increment - drift) / diffusion.

    Parameters
    ----------
    paths : np.ndarray, shape (n_paths, m)
        Spot prices at monitoring dates (S0 not included).
    cfg : ModelConfig
        Configuration used to generate the paths.

    Returns
    -------
    np.ndarray, shape (n_paths, m)
        Recovered standard-normal shocks.
    """
    dt = cfg.dt
    drift = (cfg.r - cfg.q - 0.5 * cfg.sigma ** 2) * dt
    diffusion = cfg.sigma * math.sqrt(dt)
    log_S = np.log(paths / cfg.S0)
    log_inc = np.diff(
        np.hstack([np.zeros((cfg.n_paths, 1)), log_S]),
        axis=1,
    )
    return (log_inc - drift) / diffusion


def train_reference_network(
    ref_cfg: ModelConfig,
    n_training: int,
    train_seed: int,
    hidden_width: int = 32,
    n_epochs: int = 100,
    lr: float = 1e-2,
) -> Tuple[_ShallowNet, float, str, float]:
    """
    Train the reference network on the reference contract and freeze it.

    Parameters
    ----------
    ref_cfg : ModelConfig
        Configuration for the reference contract (K=100, sigma=0.20, T=1.0).
        n_paths is overridden with n_training inside this function.
    n_training : int
        Number of training paths.
    train_seed : int
        Seed for training data generation and network initialisation.
    hidden_width : int
        Hidden layer width (default 32, consistent with existing architecture).
    n_epochs : int
        Training epochs.
    lr : float
        Learning rate.

    Returns
    -------
    network : _ShallowNet
        Frozen (numpy-only) network.
    e_h0 : float
        Analytical E[H0(Z)] for Z ~ N(0, I_m).
    param_hash : str
        SHA-256 hash of frozen network parameters.
    training_runtime_s : float
        Wall-clock time for training (excluding data generation).

    Raises
    ------
    ImportError
        If PyTorch is not available.
    """
    import dataclasses
    from asian_options.simulate_gbm import simulate_paths
    from asian_options.payoffs import arithmetic_asian_call_payoff

    t0 = time.perf_counter()

    train_cfg = dataclasses.replace(ref_cfg, n_paths=n_training, seed=train_seed)
    paths = simulate_paths(train_cfg)
    payoffs = arithmetic_asian_call_payoff(paths, train_cfg)
    Z_train = _extract_z_shocks(paths, train_cfg)

    dataset = {"X_train": Z_train, "y_train": payoffs}
    network = build_network(train_cfg, hidden_width=hidden_width)
    train_network(network, dataset, train_cfg, n_epochs=n_epochs, lr=lr)

    training_runtime_s = time.perf_counter() - t0

    e_h0 = analytical_network_expectation(network)
    param_hash = compute_network_hash(network)

    return network, e_h0, param_hash, training_runtime_s


# ---------------------------------------------------------------------------
# Transfer estimators
# ---------------------------------------------------------------------------

def ncv_transfer_beta1(
    frozen_network: _ShallowNet,
    e_h0: float,
    frozen_hash: str,
    target_cfg: ModelConfig,
    pricing_seed: int,
    n_pricing: int,
    training_runtime_s: float = 0.0,
) -> dict:
    """
    NCV_TRANSFER_BETA1: Frozen reference network with beta=1.

    Corrected observations::

        Y = f_theta(Z) - H0(Z) + E[H0(Z)]

    where f_theta is the discounted arithmetic Asian payoff and H0 is the
    frozen reference network.

    Parameters
    ----------
    frozen_network : _ShallowNet
        Frozen reference network (parameters must not change after freezing).
    e_h0 : float
        Analytical E[H0(Z)] from the reference contract training phase.
    frozen_hash : str
        Hash recorded at freeze time; used to verify network integrity.
    target_cfg : ModelConfig
        Configuration for the target contract (seed/n_paths overridden).
    pricing_seed : int
        Seed for pricing shocks.
    n_pricing : int
        Number of pricing paths.
    training_runtime_s : float
        Reference training wall-clock time (for end-to-end accounting).

    Returns
    -------
    dict
        Full result record including price, variance, SE, CI, budget, timing,
        beta (=1.0), and control-correlation fields.
    """
    import dataclasses
    from asian_options.simulate_gbm import simulate_paths
    from asian_options.payoffs import arithmetic_asian_call_payoff
    from asian_options.metrics import summarise_estimates

    verify_frozen_hash(frozen_network, frozen_hash)

    t_pricing_start = time.perf_counter()

    price_cfg = dataclasses.replace(target_cfg, n_paths=n_pricing, seed=pricing_seed)
    # Generate Z explicitly so that the same shocks are fed to both
    # simulate_paths (which uses them to build GBM paths) and network.forward
    # (which evaluates H0 on Z).  simulate_paths uses shocks exclusively when
    # the shocks argument is supplied and ignores price_cfg.seed in that path.
    rng = np.random.default_rng(pricing_seed)
    Z = rng.standard_normal((n_pricing, target_cfg.m))

    paths = simulate_paths(price_cfg, shocks=Z)
    payoffs = arithmetic_asian_call_payoff(paths, price_cfg)

    h0_vals = frozen_network.forward(Z)
    corrected = payoffs - h0_vals + e_h0

    pricing_runtime_s = time.perf_counter() - t_pricing_start

    stats = summarise_estimates(corrected, price_cfg.discount_factor, pricing_runtime_s)
    n_obs = stats["n_paths"]
    obs_var = stats["variance"]

    # Control correlation: corr(f_theta, C0) where C0 = H0 - E[H0]
    c0_vals = h0_vals - e_h0
    payoff_var = float(np.var(payoffs, ddof=1))
    control_var = float(np.var(c0_vals, ddof=1))
    cov_fc0 = float(np.cov(payoffs, c0_vals, ddof=1)[0, 1])
    f_std = float(np.std(payoffs, ddof=1))
    c0_std = float(np.std(c0_vals, ddof=1))
    if f_std > 0 and c0_std > 0:
        corr_f_c0 = float(np.corrcoef(payoffs, c0_vals)[0, 1])
    else:
        corr_f_c0 = float("nan")

    # Verify hash again after evaluation (integrity check)
    verify_frozen_hash(frozen_network, frozen_hash)

    end_to_end_s = training_runtime_s + pricing_runtime_s

    return {
        "method": "NCV_TRANSFER_BETA1",
        "price": stats["price"],
        "observation_variance": obs_var,
        "estimator_variance": obs_var / n_obs,
        "std_error": stats["std_error"],
        "ci_lower": stats["ci_lower"],
        "ci_upper": stats["ci_upper"],
        "n_pricing": n_obs,
        "pricing_observations": n_obs,
        "pricing_simulated_paths": n_obs,
        "pilot_paths": 0,
        "training_paths": 0,  # reference training charged once at portfolio level
        "total_simulated_paths": n_obs,
        "beta": 1.0,
        "corr_f_c0": corr_f_c0,
        "payoff_variance": payoff_var,
        "control_variance": control_var,
        "payoff_control_covariance": cov_fc0,
        "optimal_residual_variance": float(np.var(payoffs - c0_vals, ddof=1)),
        "e_h0": e_h0,
        "pricing_runtime_s": pricing_runtime_s,
        "training_runtime_s": training_runtime_s,
        "end_to_end_runtime_s": end_to_end_s,
        "param_hash": frozen_hash,
        "hash_verified": True,
    }


def ncv_transfer_beta(
    frozen_network: _ShallowNet,
    e_h0: float,
    frozen_hash: str,
    target_cfg: ModelConfig,
    pilot_seed: int,
    pricing_seed: int,
    n_pilot: int,
    n_pricing: int,
    training_runtime_s: float = 0.0,
    near_zero_threshold: float = 1e-12,
) -> dict:
    """
    NCV_TRANSFER_BETA: Frozen reference network with pilot-estimated beta.

    Algorithm
    ---------
    1. Draw n_pilot paths with pilot_seed (independent of training and pricing).
    2. Evaluate f_theta (discounted payoffs) and C0 = H0 - E[H0] on pilot.
    3. Estimate beta_hat = Cov(f_theta, C0) / Var(C0) [ddof=1 throughout].
       If Var(C0) < near_zero_threshold or is non-finite, raise NearZeroVarianceError.
    4. Draw n_pricing independent pricing paths with pricing_seed.
    5. Corrected observations: Y = f_theta - beta_hat * C0.

    Parameters
    ----------
    frozen_network : _ShallowNet
        Frozen reference network.
    e_h0 : float
        Analytical E[H0(Z)].
    frozen_hash : str
        Hash recorded at freeze time.
    target_cfg : ModelConfig
        Configuration for the target contract.
    pilot_seed : int
        Seed for pilot shocks (distinct from training and pricing seeds).
    pricing_seed : int
        Seed for pricing shocks.
    n_pilot : int
        Number of pilot paths.
    n_pricing : int
        Number of pricing paths.
    training_runtime_s : float
        Reference training time for end-to-end accounting.
    near_zero_threshold : float
        Var(C0) below this value triggers NearZeroVarianceError.

    Returns
    -------
    dict
        Full result record.

    Raises
    ------
    NearZeroVarianceError
        If Var(C0) on the pilot is near-zero or non-finite.
    ImportError
        If PyTorch is not available (required only for network training,
        not for this estimator which uses a pre-trained frozen network).
    """
    import dataclasses
    from asian_options.simulate_gbm import simulate_paths
    from asian_options.payoffs import arithmetic_asian_call_payoff
    from asian_options.metrics import summarise_estimates

    verify_frozen_hash(frozen_network, frozen_hash)

    t_pilot_start = time.perf_counter()

    # --- Pilot phase: estimate beta_hat ---
    pilot_cfg = dataclasses.replace(target_cfg, n_paths=n_pilot, seed=pilot_seed)
    # Generate Z_pilot explicitly so that the same shocks are fed to both
    # simulate_paths and network.forward (consistent with ncv_transfer_beta1).
    rng_pilot = np.random.default_rng(pilot_seed)
    Z_pilot = rng_pilot.standard_normal((n_pilot, target_cfg.m))

    paths_pilot = simulate_paths(pilot_cfg, shocks=Z_pilot)
    payoffs_pilot = arithmetic_asian_call_payoff(paths_pilot, pilot_cfg)
    h0_pilot = frozen_network.forward(Z_pilot)
    c0_pilot = h0_pilot - e_h0

    var_c0 = float(np.var(c0_pilot, ddof=1))

    if not math.isfinite(var_c0) or var_c0 < near_zero_threshold:
        raise NearZeroVarianceError(
            f"Var(C0) on pilot is {var_c0:.4e} (threshold={near_zero_threshold:.4e}). "
            f"Cannot estimate beta; this is an explicit failure. "
            f"contract={target_cfg.K}/{target_cfg.sigma}/{target_cfg.T}, "
            f"n_pilot={n_pilot}."
        )

    payoff_var = float(np.var(payoffs_pilot, ddof=1))
    cov_fc0 = float(np.cov(payoffs_pilot, c0_pilot, ddof=1)[0, 1])
    beta_hat = cov_fc0 / var_c0
    optimal_residual_variance = payoff_var - (cov_fc0**2 / var_c0)

    pilot_runtime_s = time.perf_counter() - t_pilot_start

    # Control correlation on pilot
    f_std = float(np.std(payoffs_pilot, ddof=1))
    c0_std = float(np.sqrt(var_c0))
    if f_std > 0 and c0_std > 0:
        corr_f_c0 = float(np.corrcoef(payoffs_pilot, c0_pilot)[0, 1])
    else:
        corr_f_c0 = float("nan")

    # --- Pricing phase ---
    t_pricing_start = time.perf_counter()

    price_cfg = dataclasses.replace(target_cfg, n_paths=n_pricing, seed=pricing_seed)
    # Generate Z_price explicitly so that the same shocks are fed to both
    # simulate_paths and network.forward (same pattern as ncv_transfer_beta1).
    rng_price = np.random.default_rng(pricing_seed)
    Z_price = rng_price.standard_normal((n_pricing, target_cfg.m))

    paths_price = simulate_paths(price_cfg, shocks=Z_price)
    payoffs_price = arithmetic_asian_call_payoff(paths_price, price_cfg)
    h0_price = frozen_network.forward(Z_price)
    c0_price = h0_price - e_h0
    corrected = payoffs_price - beta_hat * c0_price
    beta1_residual = payoffs_price - c0_price

    pricing_runtime_s = time.perf_counter() - t_pricing_start

    stats = summarise_estimates(corrected, price_cfg.discount_factor, pricing_runtime_s)
    n_obs = stats["n_paths"]
    obs_var = stats["variance"]

    # Verify hash unchanged after evaluation
    verify_frozen_hash(frozen_network, frozen_hash)

    end_to_end_s = training_runtime_s + pilot_runtime_s + pricing_runtime_s

    return {
        "method": "NCV_TRANSFER_BETA",
        "price": stats["price"],
        "observation_variance": obs_var,
        "estimator_variance": obs_var / n_obs,
        "std_error": stats["std_error"],
        "ci_lower": stats["ci_lower"],
        "ci_upper": stats["ci_upper"],
        "n_pricing": n_obs,
        "pricing_observations": n_obs,
        "pricing_simulated_paths": n_obs,
        "pilot_paths": n_pilot,
        "training_paths": 0,  # charged once at portfolio level
        "total_simulated_paths": n_pilot + n_pricing,
        "beta": beta_hat,
        "var_c0_pilot": var_c0,
        "payoff_variance": payoff_var,
        "control_variance": var_c0,
        "payoff_control_covariance": cov_fc0,
        "optimal_residual_variance": optimal_residual_variance,
        "corr_f_c0": corr_f_c0,
        "e_h0": e_h0,
        "pricing_runtime_s": pricing_runtime_s,
        "pilot_runtime_s": pilot_runtime_s,
        "training_runtime_s": training_runtime_s,
        "end_to_end_runtime_s": end_to_end_s,
        "param_hash": frozen_hash,
        "hash_verified": True,
        "residual_variance_beta_one": float(np.var(beta1_residual, ddof=1)),
        "variance_improvement_from_estimating_beta": (
            float(np.var(beta1_residual, ddof=1)) / obs_var if obs_var > 0 else float("nan")
        ),
    }


# ---------------------------------------------------------------------------
# High-precision reference prices
# ---------------------------------------------------------------------------

def compute_high_precision_reference(
    contract_id: str,
    n_paths: int,
    seed: int,
) -> dict:
    """
    Compute a high-precision GCV reference price for a contract.

    Uses geometric control variate with a large independent sample.  The
    result is used only for empirical bias, RMSE, and CI-coverage computations
    across replications.

    Parameters
    ----------
    contract_id : str
        Contract identifier from the Stage 8 grid.
    n_paths : int
        Large sample size for high precision.
    seed : int
        Independent seed (must not overlap with experimental pricing seeds).

    Returns
    -------
    dict
        Keys: contract_id, price, std_error, ci_lower, ci_upper, n_paths,
        method, seed.
    """
    from asian_options.contracts import make_contract_cfg
    from asian_options.estimators import geometric_control_variate

    cfg = make_contract_cfg(contract_id, n_paths=n_paths, seed=seed)
    result = geometric_control_variate(cfg, n_pilot=max(1000, n_paths // 50))
    return {
        "contract_id": contract_id,
        "price": result.price,
        "std_error": result.std_error,
        "ci_lower": result.ci_lower,
        "ci_upper": result.ci_upper,
        "n_paths": result.n_paths,
        "method": "GCV_high_precision",
        "seed": seed,
    }
