"""
test_structure.py
=================
Stage 1 structural tests: verify that every module in the asian_options
package can be imported without error.
"""

import importlib
import pytest


MODULES = [
    "asian_options",
    "asian_options.config",
    "asian_options.simulate_gbm",
    "asian_options.payoffs",
    "asian_options.analytical",
    "asian_options.estimators",
    "asian_options.variance_reduction",
    "asian_options.neural_cv",
    "asian_options.metrics",
    "asian_options.results",
]


@pytest.mark.parametrize("module_name", MODULES)
def test_module_importable(module_name: str) -> None:
    """Each module must be importable without raising any exception."""
    mod = importlib.import_module(module_name)
    assert mod is not None


def test_estimate_result_namedtuple() -> None:
    """EstimateResult can be constructed with all required fields."""
    from asian_options.estimators import EstimateResult

    result = EstimateResult(
        price=5.0,
        variance=0.1,
        std_dev=0.316,
        std_error=0.01,
        ci_lower=4.98,
        ci_upper=5.02,
        n_paths=10_000,
        runtime_s=0.5,
    )
    assert result.price == 5.0
    assert result.n_paths == 10_000
