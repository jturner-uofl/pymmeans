"""Tests for ml_avg_slopes / ml_avg_comparisons — bootstrap marginal
effects for black-box predictive models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

pytest.importorskip("sklearn")
from sklearn.linear_model import LinearRegression

from pymmeans import (
    MLMarginalResult,
    from_predict,
    ml_avg_comparisons,
    ml_avg_slopes,
)

_FEATS = ["x", "z"]


def _ols_refit(d):
    m = LinearRegression().fit(d[_FEATS], d["y"])
    return lambda nd: m.predict(nd[_FEATS])


def _info(n=500, seed=0, with_refit=True):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.standard_normal(n), "z": rng.standard_normal(n)})
    df["y"] = 1.5 * df["x"] - 0.7 * df["z"] + rng.standard_normal(n)
    return from_predict(
        predict_fn=_ols_refit(df),
        data=df,
        factors={},
        numerics=_FEATS,
        response="y",
        refit_fn=_ols_refit if with_refit else None,
    ), df


# ---------------------------------------------------------------------- G1: point identities


def test_ml_slope_point_equals_ols_coefficient():
    """For a linear predict_fn, the ML AME equals the OLS coefficient."""
    info, df = _info()
    res = ml_avg_slopes(info, "x", n_boot=200, seed=1)
    ols = smf.ols("y ~ x + z", df).fit()
    assert float(res.frame["slope"].iloc[0]) == pytest.approx(
        float(ols.params["x"]), abs=1e-7
    )


def test_ml_comparison_difference_equals_coefficient():
    info, df = _info()
    res = ml_avg_comparisons(info, "x", n_boot=200, seed=1)
    ols = smf.ols("y ~ x + z", df).fit()
    assert float(res.frame["estimate"].iloc[0]) == pytest.approx(
        float(ols.params["x"]), abs=1e-7
    )


# ------------------------------------------------------- G2: bootstrap SE recovers analytic


def test_bootstrap_se_recovers_analytic_ols_se():
    """The pairs-bootstrap SE matches the analytic OLS coefficient SE
    within Monte-Carlo tolerance."""
    info, df = _info(n=600, seed=2)
    res = ml_avg_slopes(info, "x", n_boot=600, seed=3)
    ols = smf.ols("y ~ x + z", df).fit()
    boot_se = float(res.frame["SE"].iloc[0])
    analytic = float(ols.bse["x"])
    # MC noise on a bootstrap SD with B=600 is ~SE/sqrt(2B) ~ 3%; allow 12%.
    assert abs(boot_se - analytic) / analytic < 0.12


# ---------------------------------------------------------------------- point-only / refit_fn


def test_point_only_when_no_refit_fn():
    info, _ = _info(with_refit=False)
    res = ml_avg_slopes(info, "x", n_boot=10)
    assert np.isfinite(float(res.frame["slope"].iloc[0]))
    assert np.isnan(float(res.frame["SE"].iloc[0]))
    assert np.isnan(float(res.frame["lower_cl"].iloc[0]))


def test_reproducible_with_seed():
    info, _ = _info()
    a = ml_avg_slopes(info, "x", n_boot=150, seed=42).frame["SE"].iloc[0]
    b = ml_avg_slopes(info, "x", n_boot=150, seed=42).frame["SE"].iloc[0]
    assert float(a) == float(b)


# ---------------------------------------------------------------------- grouping / categorical


def test_by_grouping():
    rng = np.random.default_rng(5)
    n = 500
    df = pd.DataFrame({
        "x": rng.standard_normal(n),
        "z": rng.standard_normal(n),
        "g": pd.Categorical(rng.choice(["A", "B"], n)),
    })
    gshift = df["g"].map({"A": 0.0, "B": 1.0}).astype(float)
    df["y"] = 1.5 * df["x"] + gshift + rng.standard_normal(n)
    feats = ["x", "z"]

    def refit(d):
        # one-hot g for the linear learner
        dd = d.assign(gB=(d["g"] == "B").astype(float))
        m = LinearRegression().fit(dd[["x", "z", "gB"]], dd["y"])
        return lambda nd: m.predict(nd.assign(gB=(nd["g"] == "B").astype(float))[["x", "z", "gB"]])

    info = from_predict(predict_fn=refit(df), data=df, factors={"g": ["A", "B"]},
                        numerics=feats, response="y", refit_fn=refit)
    res = ml_avg_slopes(info, "x", by="g", n_boot=120, seed=1)
    assert set(res.frame["g"]) == {"A", "B"}
    # additive x slope is the same in both groups
    s = res.frame.set_index("g")["slope"]
    assert float(s["A"]) == pytest.approx(float(s["B"]), abs=0.05)


def test_categorical_comparison_levels_vs_reference():
    rng = np.random.default_rng(6)
    n = 500
    df = pd.DataFrame({
        "x": rng.standard_normal(n),
        "g": pd.Categorical(rng.choice(["A", "B", "C"], n)),
    })
    df["y"] = df["g"].map({"A": 0.0, "B": 1.0, "C": 2.0}).astype(float) + rng.standard_normal(n)

    def refit(d):
        dd = pd.get_dummies(d["g"], prefix="g").astype(float)
        feat = list(dd.columns)
        full = pd.concat([d[["x"]], dd], axis=1)
        m = LinearRegression().fit(full[["x", *feat]], d["y"])

        def pf(nd):
            ndd = (
                pd.get_dummies(nd["g"], prefix="g")
                .reindex(columns=feat, fill_value=0.0)
                .astype(float)
                .reset_index(drop=True)
            )
            full = pd.concat([nd[["x"]].reset_index(drop=True), ndd], axis=1)
            return m.predict(full[["x", *feat]])
        return pf

    info = from_predict(predict_fn=refit(df), data=df, factors={"g": ["A", "B", "C"]},
                        numerics=["x"], response="y", refit_fn=refit)
    res = ml_avg_comparisons(info, "g", n_boot=120, seed=1).frame.set_index("contrast")
    assert list(res.index) == ["B - A", "C - A"]
    assert float(res.loc["B - A", "estimate"]) == pytest.approx(1.0, abs=0.2)
    assert float(res.loc["C - A", "estimate"]) == pytest.approx(2.0, abs=0.2)


# ---------------------------------------------------------------------- validation


def test_slopes_rejects_categorical_var():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "g": pd.Categorical(rng.choice(["A", "B"], 100)),
        "y": rng.standard_normal(100),
    })
    info = from_predict(predict_fn=lambda d: np.zeros(len(d)), data=df,
                        factors={"g": ["A", "B"]}, numerics=[], response="y")
    with pytest.raises(ValueError, match="numeric"):
        ml_avg_slopes(info, "g")


def test_rejects_unknown_comparison():
    info, _ = _info()
    with pytest.raises(ValueError, match="comparison"):
        ml_avg_comparisons(info, "x", comparison="nope")


def test_returns_documented_type_and_columns():
    info, _ = _info()
    res = ml_avg_slopes(info, "x", n_boot=80)
    assert isinstance(res, MLMarginalResult)
    for col in ("var", "slope", "SE", "lower_cl", "upper_cl"):
        assert col in res.frame.columns
