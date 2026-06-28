"""Tests for the parametric response transforms power / sympower /
yj.power / bcnPower against R `emmeans::make.tran` and `car`.

Reference values are committed constants produced by R (car 3.1, emmeans
1.10) so the suite runs without an R toolchain.
"""

from __future__ import annotations

import numpy as np
import pytest

from pymmeans.transforms import make_tran

_Z = np.array([0.3, 0.8, 1.5, 2.2])


def _num_deriv(t, z, h=1e-6):
    return (t.inverse(z + h) - t.inverse(z - h)) / (2.0 * h)


# ---------------------------------------------------------------------- power


def test_power_matches_R_make_tran():
    """emmeans make.tran('power', 0.5): linkinv(z) = z**2, mu.eta = 2z."""
    t = make_tran("power", lambda_=0.5)
    np.testing.assert_allclose(t.inverse(_Z), _Z**2, atol=1e-12)
    np.testing.assert_allclose(t.inverse_deriv(_Z), 2.0 * _Z, atol=1e-12)


def test_power_inverse_deriv_self_consistent():
    t = make_tran("power", lambda_=0.5)
    np.testing.assert_allclose(t.inverse_deriv(_Z), _num_deriv(t, _Z), atol=1e-7)


# ---------------------------------------------------------------------- sympower


def test_sympower_inverse_matches_R():
    """sympower(0.5) linkinv = sign(z)|z|**2; matches R make.tran exactly."""
    t = make_tran("sympower", lambda_=0.5)
    np.testing.assert_allclose(t.inverse(_Z), np.sign(_Z) * np.abs(_Z) ** 2, atol=1e-12)


def test_sympower_deriv_is_mathematically_correct():
    """pymmeans uses the correct derivative (1/lambda)|z|**(1/lambda - 1) =
    2z; R emmeans' sympower mu.eta drops the 1/lambda factor (returns z),
    which is an emmeans bug. pymmeans intentionally does NOT replicate it."""
    t = make_tran("sympower", lambda_=0.5)
    # correct derivative == 2z, and self-consistent with the numerical one
    np.testing.assert_allclose(t.inverse_deriv(_Z), 2.0 * _Z, atol=1e-12)
    np.testing.assert_allclose(t.inverse_deriv(_Z), _num_deriv(t, _Z), atol=1e-7)
    # explicitly NOT R's buggy `z`
    assert not np.allclose(t.inverse_deriv(_Z), _Z)


# ---------------------------------------------------------------------- yj.power


def test_yj_power_round_trips_through_car_forward():
    """car::yjPower(y, 0.5) forward, then pymmeans inverse, recovers y."""
    y = np.array([-1.5, -0.2, 0.0, 0.7, 2.0])
    car_forward = np.array([
        -1.9685647168, -0.2096894253, 0.0, 0.6076809621, 1.4641016151,
    ])
    t = make_tran("yj.power", lambda_=0.5)
    np.testing.assert_allclose(t.inverse(car_forward), y, atol=1e-9)


def test_yj_power_alias_and_deriv():
    t = make_tran("yjPower", lambda_=0.5)
    np.testing.assert_allclose(t.inverse_deriv(_Z), _num_deriv(t, _Z), atol=1e-7)


# ---------------------------------------------------------------------- bcnPower


def test_bcn_power_inverse_matches_car():
    """car::bcnPowerInverse(z, lambda=0.5, gamma=1): Hawkins-Weisberg
    smoothed Box-Cox, y = s - gamma^2/(4 s), s = (lambda z + 1)^(1/lambda)."""
    z = np.array([0.3, 0.8, 1.5])
    car_inverse = np.array([1.1334640832, 1.8324489796, 2.9808673469])
    t = make_tran("bcnPower", lambda_=0.5, gamma=1.0)
    np.testing.assert_allclose(t.inverse(z), car_inverse, atol=1e-9)


def test_bcn_power_round_trips_through_car_forward():
    y = np.array([-1.5, -0.2, 0.0, 0.7, 2.0])
    car_forward = np.array([
        -1.2218282481, -0.7195282879, -0.5857864376, -0.0400736945, 0.9106933805,
    ])
    t = make_tran("bcnPower", lambda_=0.5, gamma=1.0)
    np.testing.assert_allclose(t.inverse(car_forward), y, atol=1e-9)


def test_bcn_power_deriv_self_consistent():
    """The fixed derivative carries the gamma smoothing term (it is NOT a
    plain Box-Cox derivative)."""
    t = make_tran("bcnPower", lambda_=0.5, gamma=1.0)
    np.testing.assert_allclose(t.inverse_deriv(_Z), _num_deriv(t, _Z), atol=1e-7)


def test_bcn_power_requires_both_params():
    with pytest.raises(ValueError, match="lambda"):
        make_tran("bcnPower", lambda_=0.5)
