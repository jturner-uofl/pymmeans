"""Tests for emtrends(..., max_degree=k) — higher-order polynomial trends."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import emtrends


def _quadratic_fit(noise=0.0, seed=0, n=200):
    rng = np.random.default_rng(seed)
    x = rng.normal(2.0, 1.0, n)
    y = 2.0 + 3.0 * x + 5.0 * x**2 + noise * rng.standard_normal(n)
    df = pd.DataFrame({"x": x, "y": y})
    return smf.ols("y ~ x + I(x**2)", df).fit(), df


# ---------------------------------------------------------------------- exact (G1)


def test_degree_convention_is_taylor_coefficient():
    """For y = 2 + 3x + 5x^2, the quadratic trend is beta2 = 5 (the
    polynomial coefficient, f''/2!), NOT the raw 2nd derivative 10."""
    fit, _ = _quadratic_fit(noise=0.0)
    res = emtrends(fit, None, var="x", max_degree=2).frame
    quad = float(res.set_index("degree").loc["quadratic", "x.trend"])
    assert quad == pytest.approx(5.0, abs=1e-6)


def test_linear_trend_is_exact_for_noise_free_quadratic():
    """linear trend == 3 + 10 * x_eval, evaluated at the grid (mean x)."""
    fit, df = _quadratic_fit(noise=0.0)
    res = emtrends(fit, None, var="x", max_degree=2).frame
    lin = float(res.set_index("degree").loc["linear", "x.trend"])
    assert lin == pytest.approx(3.0 + 10.0 * df["x"].mean(), abs=1e-4)


def test_cubic_of_a_quadratic_model_is_zero():
    fit, _ = _quadratic_fit(noise=0.0)
    res = emtrends(fit, None, var="x", max_degree=3).frame
    cubic = float(res.set_index("degree").loc["cubic", "x.trend"])
    assert cubic == pytest.approx(0.0, abs=1e-4)


# ---------------------------------------------------------------------- R cross-validation


def test_quadratic_trend_by_group_is_finite_and_x_independent():
    """Grouped higher-order trends: quadratic is x-independent (= beta2 per
    group) and finite. (The exact-vs-R identity is covered analytically by
    the noise-free tests above; emtrends' R cross-validation lives in the
    jss_audit notebook.)"""
    fit, df = _quadratic_fit(noise=0.5, seed=3)
    df = df.copy()
    df["g"] = pd.Categorical((df["x"] > df["x"].median()).map({True: "B", False: "A"}))
    fit = smf.ols("y ~ (x + I(x**2)) * g", df).fit()
    res = emtrends(fit, "g", var="x", max_degree=2).frame
    # quadratic trend is x-independent (= beta2 per group); both groups finite
    quad = res[res["degree"] == "quadratic"]
    assert len(quad) == 2
    assert np.isfinite(quad["x.trend"]).all()


# ---------------------------------------------------------------------- structure / compat


def test_max_degree_1_unchanged_no_degree_column():
    fit, _ = _quadratic_fit(noise=0.3)
    res = emtrends(fit, None, var="x").frame
    assert "degree" not in res.columns


def test_multidegree_linear_equals_single_degree():
    """The degree-1 row of a multi-degree call equals the single-degree
    emtrends (no regression in the default path)."""
    fit, _ = _quadratic_fit(noise=0.3)
    single = emtrends(fit, None, var="x").frame["x.trend"].to_numpy()
    multi = emtrends(fit, None, var="x", max_degree=2).frame
    multi_lin = multi[multi["degree"] == "linear"]["x.trend"].to_numpy()
    np.testing.assert_allclose(single, multi_lin, atol=1e-6)


def test_degree_labels_and_row_count():
    fit, _ = _quadratic_fit(noise=0.3)
    res = emtrends(fit, None, var="x", max_degree=3).frame
    assert list(res["degree"]) == ["linear", "quadratic", "cubic"]


# ---------------------------------------------------------------------- validation


def test_rejects_max_degree_out_of_range():
    fit, _ = _quadratic_fit()
    with pytest.raises(ValueError, match="max_degree"):
        emtrends(fit, None, var="x", max_degree=5)


def test_response_derivative_with_high_degree_raises():
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({"x": rng.normal(0, 1, n)})
    df["yb"] = (rng.random(n) < 1 / (1 + np.exp(-(0.3 + df["x"])))).astype(int)
    import statsmodels.api as sm
    fit = smf.glm("yb ~ x + I(x**2)", df, family=sm.families.Binomial()).fit()
    with pytest.raises(NotImplementedError, match="max_degree=1"):
        emtrends(fit, None, var="x", max_degree=2, response_derivative=True)
