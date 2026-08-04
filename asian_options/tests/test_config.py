"""
test_config.py
==============
Stage 1 configuration validation tests.

Tests cover:
- Successful construction with valid parameters.
- Rejection of invalid inputs (non-positive spot, strike, sigma, T; m < 1;
  n_paths < 1; negative seed).
- Derived properties: dt and discount_factor.
"""

import math
import pytest
from asian_options.config import ModelConfig


# ---------------------------------------------------------------------------
# Valid configuration
# ---------------------------------------------------------------------------

VALID_KWARGS = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0, m=12, n_paths=10_000, seed=42)


def test_valid_config_constructs() -> None:
    """A fully valid set of parameters must not raise."""
    cfg = ModelConfig(**VALID_KWARGS)
    assert cfg.S0 == 100.0
    assert cfg.K == 100.0
    assert cfg.r == 0.05
    assert cfg.q == 0.0          # default
    assert cfg.sigma == 0.2
    assert cfg.T == 1.0
    assert cfg.m == 12
    assert cfg.n_paths == 10_000
    assert cfg.seed == 42


def test_valid_config_with_dividend() -> None:
    """Non-zero dividend yield q is accepted."""
    cfg = ModelConfig(**{**VALID_KWARGS, "q": 0.02})
    assert cfg.q == 0.02


def test_dt_property() -> None:
    """dt = T / m."""
    cfg = ModelConfig(**VALID_KWARGS)
    assert math.isclose(cfg.dt, 1.0 / 12)


def test_discount_factor_property() -> None:
    """discount_factor = exp(-r * T)."""
    cfg = ModelConfig(**VALID_KWARGS)
    expected = math.exp(-0.05 * 1.0)
    assert math.isclose(cfg.discount_factor, expected)


def test_zero_rate_accepted() -> None:
    """r = 0 is financially valid."""
    cfg = ModelConfig(**{**VALID_KWARGS, "r": 0.0})
    assert cfg.r == 0.0


def test_negative_rate_accepted() -> None:
    """Negative risk-free rates are financially possible."""
    cfg = ModelConfig(**{**VALID_KWARGS, "r": -0.01})
    assert cfg.r == -0.01


def test_zero_seed_accepted() -> None:
    """seed = 0 is valid (boundary)."""
    cfg = ModelConfig(**{**VALID_KWARGS, "seed": 0})
    assert cfg.seed == 0


# ---------------------------------------------------------------------------
# Invalid inputs — each must raise ValueError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field,value", [
    ("S0",      0.0),
    ("S0",     -1.0),
    ("K",       0.0),
    ("K",      -50.0),
    ("sigma",   0.0),
    ("sigma",  -0.1),
    ("T",       0.0),
    ("T",      -1.0),
    ("m",       0),
    ("m",      -5),
    ("n_paths", 0),
    ("n_paths", -100),
    ("seed",   -1),
])
def test_invalid_input_raises(field: str, value) -> None:
    """Every invalid parameter must raise ValueError."""
    kwargs = {**VALID_KWARGS, field: value}
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)
