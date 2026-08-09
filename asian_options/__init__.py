"""
asian_options
=============
Modular library for pricing arithmetic Asian options via Monte Carlo,
classical variance reduction, and neural control variates.

Stage 1: project structure and validated configuration only.
Estimator implementations follow in later stages.
"""

from asian_options.config import (
    ModelConfig,
    collect_environment_metadata,
    seed_everything,
)
from asian_options.estimators import (
    CVEstimateResult,
    EstimateResult,
    antithetic_variates,
    geometric_control_variate,
    standard_monte_carlo,
)
from asian_options.neural_cv import build_network, ncv_estimator, train_network

__all__ = [
    "ModelConfig",
    "EstimateResult",
    "CVEstimateResult",
    "seed_everything",
    "collect_environment_metadata",
    "standard_monte_carlo",
    "antithetic_variates",
    "geometric_control_variate",
    "build_network",
    "train_network",
    "ncv_estimator",
]
