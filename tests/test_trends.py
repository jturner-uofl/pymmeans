"""Tests for emtrends()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymmeans import emtrends


@pytest.fixture
def linear_no_interaction():
    rng = np.random.default_rng(0)
    n = 200
    x = rng.normal(size=n)
    g = rng.choice(["a", "b", "c"], n)
    y = 1.0 + 0.5 * x + np.where(g == "b", 0.3, 0.0) + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"y": y, "x": x, "g": pd.Categorical(g)})
    return smf.ols("y ~ g + x", data=df).fit()


@pytest.fixture
def linear_with_interaction():
    rng = np.random.default_rng(1)
    n = 300
    x = rng.normal(size=n)
    g = rng.choice(["a", "b"], n)
    # True slopes: a -> 0.2, b -> 1.0
    slope = np.where(g == "a", 0.2, 1.0)
    y = slope * x + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({"y": y, "x": x, "g": pd.Categorical(g)})
    return smf.ols("y ~ g * x", data=df).fit()


def test_overall_slope_when_no_interaction(linear_no_interaction):
    # Without specs, all g share the same slope (coef of x ≈ 0.5)
    res = emtrends(linear_no_interaction, None, "x")
    assert res.n_rows == 1
    slope = res.frame["x.trend"].iloc[0]
    assert slope == pytest.approx(0.5, abs=0.05)


def test_per_group_slope_when_interaction(linear_with_interaction):
    res = emtrends(linear_with_interaction, "g", "x")
    assert res.n_rows == 2
    slopes = dict(
        zip(res.frame["g"].astype(str), res.frame["x.trend"], strict=True)
    )
    assert slopes["a"] == pytest.approx(0.2, abs=0.05)
    assert slopes["b"] == pytest.approx(1.0, abs=0.05)


def test_slope_matches_coefficient(linear_no_interaction):
    res = emtrends(linear_no_interaction, "g", "x")
    # All three groups should report the same slope = the x coefficient
    slope_values = res.frame["x.trend"].to_numpy()
    np.testing.assert_array_almost_equal(slope_values, [slope_values[0]] * 3)
    coef = linear_no_interaction.params["x"]
    assert slope_values[0] == pytest.approx(coef, abs=1e-6)


def test_se_positive(linear_with_interaction):
    res = emtrends(linear_with_interaction, "g", "x")
    assert (res.frame["SE"] > 0).all()


def test_rejects_unknown_var(linear_no_interaction):
    with pytest.raises(ValueError, match="not a numeric"):
        emtrends(linear_no_interaction, "g", "z")


def test_rejects_factor_as_var(linear_no_interaction):
    with pytest.raises(ValueError, match="not a numeric"):
        emtrends(linear_no_interaction, "g", "g")


def test_response_scale_trend_for_binomial():
    rng = np.random.default_rng(5)
    n = 500
    x = rng.normal(size=n)
    eta = 0.1 + 0.8 * x
    p = 1.0 / (1.0 + np.exp(-eta))
    y = rng.binomial(1, p)
    df = pd.DataFrame({"y": y, "x": x, "g": pd.Categorical(rng.choice(["a", "b"], n))})
    fit = smf.glm("y ~ g + x", data=df, family=sm.families.Binomial()).fit()

    link_res = emtrends(fit, None, "x")
    # c: `type='response'` is now ignored by R parity;
    # the chain-rule response-scale slope is opt-in via
    # `response_derivative=True`.
    resp_res = emtrends(fit, None, "x", response_derivative=True)
    # Response-scale slope at the mean grid point should be smaller in
    # magnitude than the link-scale slope because the logistic derivative
    # h'(eta) = p(1-p) is bounded by 0.25.
    link_slope = link_res.frame["x.trend"].iloc[0]
    resp_slope = resp_res.frame["x.trend"].iloc[0]
    assert abs(resp_slope) <= 0.25 * abs(link_slope) + 1e-6
    # Bare `type='response'` (R-parity default) returns link-scale slopes:
    same_as_link = emtrends(fit, None, "x", type="response")
    assert same_as_link.frame["x.trend"].iloc[0] == pytest.approx(link_slope)
    assert same_as_link.type == "link"
