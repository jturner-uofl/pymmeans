"""Tests for the VanderWeele-Ding (2017) E-value sensitivity module."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pymmeans import EValueResult, e_value
from pymmeans.sensitivity import _e_from_rr

# ---------------------------------------------------------------------- closed form

def test_e_from_rr_published_examples():
    """Closed-form E-value reproduces VanderWeele-Ding 2017 published values."""
    # VanderWeele & Ding (2017) Annals Intern Med, canonical worked example:
    # smoking → lung cancer with RR = 10.73 has E ≈ 20.95.
    assert _e_from_rr(10.73) == pytest.approx(20.95, abs=0.01)
    # Closed form: E = RR + sqrt(RR (RR-1)).
    assert _e_from_rr(2.0) == pytest.approx(2.0 + math.sqrt(2.0), abs=1e-12)
    assert _e_from_rr(1.5) == pytest.approx(1.5 + math.sqrt(1.5 * 0.5), abs=1e-12)
    # Boundary: RR = 1 (null) → E = 1.
    assert _e_from_rr(1.0) == pytest.approx(1.0, abs=1e-12)


def test_e_from_rr_symmetric_below_one():
    """RR and 1/RR yield the same E-value (symmetry property)."""
    for rr in (0.3, 0.5, 0.7, 0.9):
        assert _e_from_rr(rr) == pytest.approx(_e_from_rr(1.0 / rr), abs=1e-12)


def test_e_value_invalid_rr_raises():
    """Negative / zero / non-finite RR raises ValueError."""
    with pytest.raises(ValueError):
        e_value(0.0)
    with pytest.raises(ValueError):
        e_value(-1.5)
    with pytest.raises(ValueError):
        e_value(float("nan"))
    with pytest.raises(ValueError):
        e_value(float("inf"))


# ---------------------------------------------------------------------- CI handling

def test_e_value_ci_lower_when_estimate_above_one():
    """When CI is above 1, the lower CI bound drives the CI E-value."""
    e = e_value(2.0, ci_lo=1.5, ci_hi=2.7, kind="rr")
    assert e.e_point == pytest.approx(_e_from_rr(2.0), abs=1e-12)
    assert e.e_ci == pytest.approx(_e_from_rr(1.5), abs=1e-12)


def test_e_value_ci_upper_when_estimate_below_one():
    """When CI is below 1, the upper CI bound (closer to 1) drives the CI E-value."""
    e = e_value(0.5, ci_lo=0.4, ci_hi=0.7, kind="rr")
    # E-value of point estimate is symmetric: same as RR=2.
    assert e.e_point == pytest.approx(_e_from_rr(2.0), abs=1e-12)
    # CI bound closer to 1 is 0.7. Its E-value = E(1/0.7).
    assert e.e_ci == pytest.approx(_e_from_rr(1.0 / 0.7), abs=1e-12)


def test_e_value_ci_crossing_null_returns_one():
    """If the CI crosses 1, e_ci is 1 by convention (no robustness)."""
    e = e_value(1.2, ci_lo=0.8, ci_hi=1.7, kind="rr")
    assert e.e_ci == pytest.approx(1.0, abs=1e-12)


def test_e_value_no_ci_yields_none():
    """Omitting both CI bounds leaves e_ci as None."""
    e = e_value(2.0)
    assert e.e_ci is None


# ---------------------------------------------------------------------- scale conversion

def test_e_value_or_rare_outcome_acts_like_rr():
    """Odds ratio with rare outcome (default) is treated as RR."""
    e_rr = e_value(3.0, kind="rr")
    e_or = e_value(3.0, kind="or")
    assert e_or.e_point == pytest.approx(e_rr.e_point, abs=1e-12)


def test_e_value_or_common_outcome_uses_sqrt_conversion():
    """Odds ratio with common outcome (prevalence >= 0.15) → RR ≈ sqrt(OR)."""
    e_or_common = e_value(4.0, kind="or", prevalence=0.30)
    # Expected: RR ≈ sqrt(4) = 2.0, then E(2.0) = 2 + sqrt(2).
    assert e_or_common.estimate == pytest.approx(2.0, abs=1e-12)
    assert e_or_common.e_point == pytest.approx(_e_from_rr(2.0), abs=1e-12)


def test_e_value_hr_rare_event_acts_like_rr():
    """Hazard ratio with rare event acts as RR (rare-event approximation)."""
    e_rr = e_value(2.5, kind="rr")
    e_hr = e_value(2.5, kind="hr")
    assert e_hr.e_point == pytest.approx(e_rr.e_point, abs=1e-12)


def test_e_value_smd_uses_exp_091_conversion():
    """Standardized mean difference is converted via RR ≈ exp(0.91 * d)."""
    d = 0.5
    e_smd = e_value(d, kind="smd")
    # Expected RR.
    assert e_smd.estimate == pytest.approx(math.exp(0.91 * d), abs=1e-12)
    # And E-value follows from that RR.
    assert e_smd.e_point == pytest.approx(_e_from_rr(math.exp(0.91 * d)), abs=1e-12)


def test_e_value_rejects_unknown_kind():
    """Unknown ``kind`` raises ValueError."""
    with pytest.raises(ValueError, match="kind must be one of"):
        e_value(2.0, kind="nonsense")  # type: ignore[arg-type]


def test_e_value_rejects_invalid_prevalence():
    """Prevalence outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError, match="prevalence"):
        e_value(2.0, kind="or", prevalence=-0.1)
    with pytest.raises(ValueError, match="prevalence"):
        e_value(2.0, kind="or", prevalence=1.5)


# ---------------------------------------------------------------------- return shape

def test_e_value_result_is_evalueresult_frozen():
    """Returned object is the documented dataclass + frozen."""
    e = e_value(2.0)
    assert isinstance(e, EValueResult)
    with pytest.raises((AttributeError, Exception)):
        e.e_point = 999.0  # frozen → cannot set


def test_e_value_result_repr_includes_e_point():
    """repr is informative and includes the E-value."""
    r = repr(e_value(2.0))
    assert "e_point" in r and "EValueResult" in r


# ---------------------------------------------------------------------- monotonicity

def test_e_value_monotonic_in_rr():
    """E-value is monotonically non-decreasing in |log RR|."""
    rrs = np.linspace(1.0, 10.0, 50)
    es = [_e_from_rr(r) for r in rrs]
    diffs = np.diff(es)
    assert np.all(diffs >= -1e-12), (
        "E-value should be monotonically non-decreasing in RR for RR > 1"
    )
