"""
config.py
=========
Validated configuration dataclass for Asian option pricing experiments.

All financial and simulation parameters live here so that every module
receives an identical, already-validated snapshot of the experiment setup.
"""

from __future__ import annotations

import os
import platform
import random
import sys
from dataclasses import dataclass, field
from importlib import import_module

import numpy as np


@dataclass
class ModelConfig:
    """
    Parameters for arithmetic Asian option pricing under risk-neutral GBM.

    Attributes
    ----------
    S0 : float
        Initial spot price (must be strictly positive).
    K : float
        Strike price (must be strictly positive).
    r : float
        Continuously-compounded risk-free rate.  May be zero or negative.
    q : float
        Continuous dividend / convenience-yield rate.  Defaults to zero.
    sigma : float
        Annualised volatility (must be strictly positive).
    T : float
        Option maturity in years (must be strictly positive).
    m : int
        Number of equally-spaced monitoring dates (must be >= 1).
        The time step is dt = T / m.
    n_paths : int
        Number of Monte Carlo sample paths (must be >= 1).
    seed : int
        Non-negative integer seed for reproducible random-number generation.
    """

    S0: float
    K: float
    r: float
    sigma: float
    T: float
    m: int
    n_paths: int
    q: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        """Validate all parameters immediately after construction."""
        if self.S0 <= 0:
            raise ValueError(f"S0 must be strictly positive, got {self.S0}")
        if self.K <= 0:
            raise ValueError(f"K must be strictly positive, got {self.K}")
        if self.sigma <= 0:
            raise ValueError(f"sigma must be strictly positive, got {self.sigma}")
        if self.T <= 0:
            raise ValueError(f"T must be strictly positive, got {self.T}")
        if self.m < 1:
            raise ValueError(f"m must be >= 1, got {self.m}")
        if self.n_paths < 1:
            raise ValueError(f"n_paths must be >= 1, got {self.n_paths}")
        if self.seed < 0:
            raise ValueError(f"seed must be non-negative, got {self.seed}")

    @property
    def dt(self) -> float:
        """Uniform time step between consecutive monitoring dates (years)."""
        return self.T / self.m

    @property
    def discount_factor(self) -> float:
        """Risk-neutral discount factor e^{-rT}."""
        import math
        return math.exp(-self.r * self.T)


def seed_everything(seed: int, deterministic_torch: bool = True) -> dict[str, object]:
    """
    Seed Python, NumPy, and PyTorch RNGs for reproducible experiments.

    Parameters
    ----------
    seed : int
        Non-negative integer seed shared across supported RNG backends.
    deterministic_torch : bool, default=True
        When PyTorch is available, request deterministic algorithms where
        practical and disable cuDNN benchmarking.

    Returns
    -------
    dict[str, object]
        Snapshot of the seeding and determinism state that can be persisted for
        audit purposes.
    """
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")

    random.seed(seed)
    np.random.seed(seed)

    metadata: dict[str, object] = {
        "seed": seed,
        "python_random_seeded": True,
        "numpy_seeded": True,
        "torch_available": False,
        "cuda_available": False,
        "cuda_seeded": False,
        "torch_deterministic_requested": deterministic_torch,
        "torch_deterministic_enabled": False,
        "torch_deterministic_warn_only": False,
    }

    try:
        torch = import_module("torch")
    except ImportError:
        return metadata

    metadata["torch_available"] = True
    torch.manual_seed(seed)

    cuda_available = bool(torch.cuda.is_available())
    metadata["cuda_available"] = cuda_available
    if cuda_available:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        metadata["cuda_seeded"] = True

    if deterministic_torch:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(torch, "use_deterministic_algorithms"):
            torch.use_deterministic_algorithms(True, warn_only=True)
            metadata["torch_deterministic_warn_only"] = True
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if hasattr(torch, "are_deterministic_algorithms_enabled"):
        metadata["torch_deterministic_enabled"] = bool(
            torch.are_deterministic_algorithms_enabled()
        )
    else:
        metadata["torch_deterministic_enabled"] = deterministic_torch

    return metadata


def collect_environment_metadata(
    seed: int = 0,
    deterministic_torch: bool = True,
) -> dict[str, object]:
    """
    Collect runtime metadata needed to reproduce Stage 1 Asian-option results.
    """
    metadata = seed_everything(seed=seed, deterministic_torch=deterministic_torch)
    metadata.update(
        {
            "platform": platform.platform(),
            "python_version": sys.version.split()[0],
        }
    )

    for package_name in ("numpy", "scipy", "torch", "pytest"):
        try:
            module = import_module(package_name)
        except ImportError:
            metadata[f"{package_name}_version"] = "not-installed"
        else:
            metadata[f"{package_name}_version"] = getattr(
                module, "__version__", "unknown"
            )

    return metadata
