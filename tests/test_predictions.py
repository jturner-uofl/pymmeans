"""Tests for avg_predictions() / predictions() — average adjusted predictions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import patsy
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymmeans import PredictionsResult, avg_predictions, predictions


def _ols(n=500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.standard_normal(n), "z": rng.standard_normal(n)})
    df["y"] = 2.0 + 1.5 * df["x"] - 0.7 * df["z"] + rng.standard_normal(n)
    return smf.ols("y ~ x + z", df).fit(), df


def _logit(n=500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x": rng.standard_normal(n),
        "z": rng.standard_normal(n),
        "g": pd.Categorical(rng.choice(["A", "B", "C"], n)),
    })
    eta = 0.3 + 0.8 * df["x"] - 0.5 * df["z"]
    df["yb"] = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    return smf.glm("yb ~ x + z + g", df, family=sm.families.Binomial()).fit(), df


# ---------------------------------------------------------------------- G1: closed form


def test_ols_link_prediction_equals_xbar_beta():
    """avg_predictions on the link scale == Xbar @ beta, exactly."""
    fit, df = _ols()
    X = np.asarray(
        patsy.build_design_matrices(
            [fit.model.data.design_info], df, return_type="matrix"
        )[0]
    )
    xbar = X.mean(0)
    res = avg_predictions(fit, type="link")
    assert float(res.frame["estimate"].iloc[0]) == pytest.approx(
        float(xbar @ fit.params), abs=1e-10
    )


def test_ols_link_prediction_se_is_exact():
    """And its SE == sqrt(Xbar V Xbar^T), exactly (linear)."""
    fit, df = _ols()
    X = np.asarray(
        patsy.build_design_matrices(
            [fit.model.data.design_info], df, return_type="matrix"
        )[0]
    )
    xbar = X.mean(0)
    V = np.asarray(fit.cov_params())
    res = avg_predictions(fit, type="link")
    assert float(res.frame["SE"].iloc[0]) == pytest.approx(
        float(np.sqrt(xbar @ V @ xbar)), abs=1e-9
    )


def test_logit_calibration_identity():
    """For a logit MLE fit, mean response prediction == observed mean."""
    fit, df = _logit()
    res = avg_predictions(fit, type="response")
    assert float(res.frame["estimate"].iloc[0]) == pytest.approx(
        float(df["yb"].mean()), abs=1e-8
    )


def test_predictions_mean_equals_avg_predictions():
    """Mean of per-row predictions == avg_predictions."""
    fit, _ = _logit()
    per = predictions(fit, type="response")
    avg = avg_predictions(fit, type="response")
    assert per.frame["estimate"].mean() == pytest.approx(
        float(avg.frame["estimate"].iloc[0]), abs=1e-9
    )


# ---------------------------------------------------------------------- by-grouping


def test_by_grouping_returns_row_per_level():
    fit, _ = _logit()
    res = avg_predictions(fit, by="g")
    assert len(res.frame) == 3
    assert set(res.frame["g"]) == {"A", "B", "C"}


# ---------------------------------------------------------------------- G2: marginaleffects


def test_cross_validate_against_marginaleffects():
    me = pytest.importorskip("marginaleffects")
    fit, _ = _logit()
    pm = avg_predictions(fit).frame.iloc[0]
    ref = me.avg_predictions(fit).to_pandas().iloc[0]
    assert float(pm["estimate"]) == pytest.approx(float(ref["estimate"]), abs=1e-9)
    assert float(pm["SE"]) == pytest.approx(float(ref["std_error"]), abs=1e-5)
    # by group
    pmg = avg_predictions(fit, by="g").frame.set_index("g")
    refg = me.avg_predictions(fit, by="g").to_pandas().set_index("g")
    for lvl in ("A", "B", "C"):
        assert float(pmg.loc[lvl, "estimate"]) == pytest.approx(
            float(refg.loc[lvl, "estimate"]), abs=1e-9
        )
        assert float(pmg.loc[lvl, "SE"]) == pytest.approx(
            float(refg.loc[lvl, "std_error"]), abs=1e-5
        )


# ---------------------------------------------------------------------- validation


def test_rejects_invalid_level():
    fit, _ = _ols()
    with pytest.raises(ValueError, match="level"):
        avg_predictions(fit, level=0.0)


def test_response_on_linear_model_raises():
    fit, _ = _ols()
    with pytest.raises(ValueError, match="response"):
        avg_predictions(fit, type="response")


def test_returns_documented_columns():
    fit, _ = _ols()
    res = avg_predictions(fit, type="link")
    assert isinstance(res, PredictionsResult)
    for col in (
        "estimate", "SE", "df", "t_ratio", "p_value", "lower_cl", "upper_cl",
    ):
        assert col in res.frame.columns
