"""Tests for joint_tests() — Type III joint Wald tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymmeans import joint_tests


def test_joint_tests_one_way_ols():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "y": np.r_[
                rng.normal(0.0, 0.5, 30),
                rng.normal(1.0, 0.5, 30),
                rng.normal(2.0, 0.5, 30),
            ],
            "g": pd.Categorical(np.repeat(["a", "b", "c"], 30)),
        }
    )
    fit = smf.ols("y ~ g", data=df).fit()
    out = joint_tests(fit)
    # Only one non-intercept term: g
    assert list(out["term"]) == ["g"]
    assert out["df_num"].iloc[0] == 2
    # Large F because the means truly differ
    assert out["statistic"].iloc[0] > 10
    assert out["p_value"].iloc[0] < 1e-10


def test_joint_tests_interaction_present():
    # Note: Type III tests are conditional on reference levels. We design
    # the cell means so a, b, and a:b are all non-null at the reference.
    rng = np.random.default_rng(1)
    n_per = 50
    cell_means = {
        ("lo", "x"): 0.0, ("lo", "y"): 0.5, ("lo", "z"): 1.0,
        ("hi", "x"): 1.5, ("hi", "y"): 2.5, ("hi", "z"): 4.0,
    }
    rows = [
        {"y": rng.normal(m, 0.3), "a": a, "b": b}
        for (a, b), m in cell_means.items()
        for _ in range(n_per)
    ]
    df = pd.DataFrame(rows)
    df["a"] = pd.Categorical(df["a"], categories=["lo", "hi"])
    df["b"] = pd.Categorical(df["b"], categories=["x", "y", "z"])
    fit = smf.ols("y ~ a * b", data=df).fit()
    out = joint_tests(fit)
    assert list(out["term"]) == ["a", "b", "a:b"]
    assert (out["p_value"] < 1e-3).all()


def test_joint_tests_glm_uses_chi_squared():
    rng = np.random.default_rng(2)
    n = 200
    df = pd.DataFrame(
        {
            "y": rng.integers(0, 2, n),
            "g": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        }
    )
    fit = smf.glm("y ~ g", data=df, family=sm.families.Binomial()).fit()
    out = joint_tests(fit)
    # df_denom is infinity for GLM
    assert np.isinf(out["df_denom"]).all()


def test_joint_tests_omits_intercept():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"y": rng.normal(size=50), "x": rng.normal(size=50)})
    fit = smf.ols("y ~ x", data=df).fit()
    out = joint_tests(fit)
    assert "Intercept" not in list(out["term"])
    assert "x" in list(out["term"])


def test_joint_tests_matches_t_squared_for_single_coef():
    # For a single-coefficient term, the joint Wald F equals the squared t-stat
    rng = np.random.default_rng(4)
    df = pd.DataFrame(
        {"y": rng.normal(size=200), "x": rng.normal(size=200)}
    )
    fit = smf.ols("y ~ x", data=df).fit()
    out = joint_tests(fit)
    f = out.loc[out["term"] == "x", "statistic"].iloc[0]
    t_x = fit.tvalues["x"]
    assert f == pytest.approx(t_x**2, rel=1e-6)
