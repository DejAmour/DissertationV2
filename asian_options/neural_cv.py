"""
neural_cv.py
============
Neural control variate (NCV) based on El Filali Ech-Chafiq, Lelong & Reghai.

Architecture (one hidden ReLU layer)::

    H_theta(Z) = W2 * ReLU(W1 @ Z + b1) + b2

The corrected observation is::

    Y^(k) = f(Z^(k)) - H_theta(Z^(k)) + E[H_theta(Z)]

where E[H_theta(Z)] is computed analytically from the frozen network weights.

Implementation notes
--------------------
* Uses PyTorch for network construction and training.  If torch is not
  installed, all public functions raise ``ImportError`` with a clear message.
* The network is lightweight (one hidden ReLU layer) so that the analytical
  expectation E[H_theta(Z)] can be computed in closed form using the
  Gaussian expectation of a ReLU (half-normal formula).
* Training is deterministic via ``cfg.seed``.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from asian_options.config import ModelConfig, seed_everything


def _require_torch():
    try:
        import torch
        return torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch is required for the NCV estimator. "
            "Install it with: pip install torch"
        ) from exc


class _ShallowNet:
    """
    Lightweight one-hidden-layer ReLU network stored as plain numpy arrays.

    This avoids a hard torch dependency at import time while still allowing
    the NCV estimator to function when torch is available for training.

    Attributes
    ----------
    W1 : np.ndarray, shape (hidden_width, m)
    b1 : np.ndarray, shape (hidden_width,)
    W2 : np.ndarray, shape (1, hidden_width)
    b2 : np.ndarray, shape (1,)
    """

    def __init__(self, W1, b1, W2, b2):
        self.W1 = np.asarray(W1, dtype=np.float64)
        self.b1 = np.asarray(b1, dtype=np.float64)
        self.W2 = np.asarray(W2, dtype=np.float64)
        self.b2 = np.asarray(b2, dtype=np.float64)

    def forward(self, Z: np.ndarray) -> np.ndarray:
        """
        Evaluate H_theta(Z) for a batch of inputs.

        Parameters
        ----------
        Z : np.ndarray, shape (n_paths, m)

        Returns
        -------
        np.ndarray, shape (n_paths,)
        """
        hidden = np.maximum(0.0, Z @ self.W1.T + self.b1)   # (n, hidden)
        out = hidden @ self.W2.T + self.b2                   # (n, 1)
        return out.ravel()


def build_network(cfg: ModelConfig, hidden_width: int = 32) -> _ShallowNet:
    """
    Construct the shallow integrable neural network H_theta.

    The architecture is fixed to one hidden ReLU layer to preserve the
    analytical-expectation property used by the NCV estimator.

    Parameters
    ----------
    cfg : ModelConfig
        Configuration providing the input dimension (cfg.m).
    hidden_width : int
        Number of hidden neurons.

    Returns
    -------
    _ShallowNet
        Untrained network with Xavier-uniform initialisation.
    """
    rng = np.random.default_rng(cfg.seed)
    fan_in = cfg.m
    fan_out = hidden_width
    limit1 = math.sqrt(6.0 / (fan_in + fan_out))
    W1 = rng.uniform(-limit1, limit1, size=(hidden_width, fan_in))
    b1 = np.zeros(hidden_width)
    limit2 = math.sqrt(6.0 / (hidden_width + 1))
    W2 = rng.uniform(-limit2, limit2, size=(1, hidden_width))
    b2 = np.zeros(1)
    return _ShallowNet(W1, b1, W2, b2)


def train_network(
    network: _ShallowNet,
    dataset: dict,
    cfg: ModelConfig,
    n_epochs: int = 200,
    lr: float = 1e-2,
) -> dict:
    """
    Train the network on a dedicated training dataset using mini-batch SGD.

    Uses PyTorch for automatic differentiation.  The network weights are
    synchronised back to the ``_ShallowNet`` numpy arrays after training so
    that ``analytical_network_expectation`` can use them without torch.

    Parameters
    ----------
    network : _ShallowNet
        Network returned by ``build_network``.
    dataset : dict
        Must contain keys ``"X_train"`` (inputs, shape (n, m)) and
        ``"y_train"`` (targets, shape (n,)).
    cfg : ModelConfig
        Experiment configuration (seed used for reproducibility).

    Returns
    -------
    dict
        Training history with key ``"train_loss"`` (list of per-epoch losses).
    """
    torch = _require_torch()
    seed_everything(cfg.seed)

    X = torch.tensor(dataset["X_train"], dtype=torch.float64)
    y = torch.tensor(dataset["y_train"], dtype=torch.float64)

    W1 = torch.nn.Parameter(torch.tensor(network.W1.copy(), dtype=torch.float64))
    b1 = torch.nn.Parameter(torch.tensor(network.b1.copy(), dtype=torch.float64))
    W2 = torch.nn.Parameter(torch.tensor(network.W2.copy(), dtype=torch.float64))
    b2 = torch.nn.Parameter(torch.tensor(network.b2.copy(), dtype=torch.float64))

    params = [W1, b1, W2, b2]
    optimizer = torch.optim.Adam(params, lr=lr)
    loss_fn = torch.nn.MSELoss()

    history = {"train_loss": []}
    batch_size = min(256, len(X))

    for epoch in range(n_epochs):
        perm = torch.randperm(len(X))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, len(X), batch_size):
            idx = perm[start:start + batch_size]
            Xb, yb = X[idx], y[idx]
            hidden = torch.relu(Xb @ W1.T + b1)
            pred = (hidden @ W2.T + b2).squeeze(1)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        history["train_loss"].append(epoch_loss / max(n_batches, 1))

    # Sync weights back to numpy arrays
    network.W1 = W1.detach().numpy().copy()
    network.b1 = b1.detach().numpy().copy()
    network.W2 = W2.detach().numpy().copy()
    network.b2 = b2.detach().numpy().copy()
    return history


def analytical_network_expectation(network: _ShallowNet) -> float:
    """
    Compute E[H_theta(Z)] analytically from frozen network weights.

    For hidden neuron i with pre-activation a_i ~ N(mu_i, sigma_i^2)::

        E[a_i^+] = sigma_i * phi(mu_i/sigma_i) + mu_i * Phi(mu_i/sigma_i)

    When sigma_i == 0: E[a_i^+] = max(0, mu_i).

    The full expectation is::

        E[H_theta(Z)] = W2 @ (E[a_1^+], ..., E[a_n^+])^T + b2

    The input Z ~ N(0, I_m), so each pre-activation
    a_i = W1[i,:] @ Z + b1[i] ~ N(b1[i], ||W1[i,:]||^2).

    Parameters
    ----------
    network : _ShallowNet
        Frozen network.

    Returns
    -------
    float
        Analytical expectation of H_theta(Z).
    """
    from asian_options.analytical import relu_expected_value

    W1, b1, W2, b2 = network.W1, network.b1, network.W2, network.b2

    # Shape validation
    hidden_width, m = W1.shape
    if b1.shape != (hidden_width,):
        raise ValueError(
            f"b1 shape {b1.shape} inconsistent with W1 hidden_width {hidden_width}."
        )
    if W2.shape != (1, hidden_width):
        raise ValueError(
            f"W2 shape {W2.shape} must be (1, {hidden_width})."
        )
    if b2.shape != (1,):
        raise ValueError(f"b2 shape {b2.shape} must be (1,).")

    # Finite-value checks
    for name, arr in [("W1", W1), ("b1", b1), ("W2", W2), ("b2", b2)]:
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"Network parameter {name} contains non-finite values.")

    sigma_vec = np.linalg.norm(W1, axis=1)   # shape (hidden_width,)
    e_relu = relu_expected_value(b1, sigma_vec)

    result = float(W2.reshape(-1) @ e_relu + float(b2.reshape(-1)[0]))
    return result


def ncv_estimator(network: _ShallowNet, cfg: ModelConfig):
    """
    Neural control-variate price estimator.

    For each path k in the independent pricing sample::

        Y^(k) = f(Z^(k)) - H_theta(Z^(k)) + E[H_theta(Z)]

    The NCV price is the sample mean of Y^(k).

    Parameters
    ----------
    network : _ShallowNet
        Frozen network.
    cfg : ModelConfig
        Experiment configuration.

    Returns
    -------
    EstimateResult
        Full pricing summary including variance-reduction ratio and runtimes.
    """
    import time
    from asian_options.simulate_gbm import simulate_paths
    from asian_options.payoffs import arithmetic_asian_call_payoff
    from asian_options.metrics import summarise_estimates
    from asian_options.estimators import EstimateResult

    t0 = time.perf_counter()

    rng = np.random.default_rng(cfg.seed)
    Z = rng.standard_normal((cfg.n_paths, cfg.m))

    if Z.shape[1] != network.W1.shape[1]:
        raise ValueError(
            f"Z has {Z.shape[1]} columns but network expects input dim "
            f"{network.W1.shape[1]} (cfg.m={cfg.m})."
        )

    paths = simulate_paths(cfg, shocks=Z)
    payoffs = arithmetic_asian_call_payoff(paths, cfg)

    # Control variate correction
    h_vals = network.forward(Z)
    e_h = analytical_network_expectation(network)
    corrected = payoffs - h_vals + e_h

    runtime_s = time.perf_counter() - t0
    stats = summarise_estimates(corrected, cfg.discount_factor, runtime_s)

    return EstimateResult(
        price=stats["price"],
        variance=stats["variance"],
        std_dev=stats["std_dev"],
        std_error=stats["std_error"],
        ci_lower=stats["ci_lower"],
        ci_upper=stats["ci_upper"],
        n_paths=stats["n_paths"],
        runtime_s=stats["runtime_s"],
    )
