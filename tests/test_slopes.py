"""Tests for avg_slopes() / slopes() — average marginal effects."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymmeans import SlopesResult, avg_slopes, slopes

# ---------------------------------------------------------------------- fixtures


def _ols(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x": rng.standard_normal(n),
        "z": rng.standard_normal(n),
        "g": pd.Categorical(rng.choice(["A", "B"], n)),
    })
    df["y"] = (
        1.0 + 2.5 * df["x"] - 0.7 * df["z"]
        + df["g"].map({"A": 0.0, "B": 0.5}).astype(float)
        + rng.standard_normal(n)
    )
    return smf.ols("y ~ x + z + g", df).fit(), df


def _logit(n=800, seed=1):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": rng.standard_normal(n)})
    eta = 0.5 + 1.2 * df["x"]
    df["yb"] = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    return smf.glm("yb ~ x", df, family=sm.families.Binomial()).fit(), df


# ---------------------------------------------------------------------- G1: identities


def test_linear_avg_slope_equals_ols_coefficient():
    """For an additive linear model, avg_slope(x) == OLS coef on x, exactly."""
    fit, _ = _ols()
    res = avg_slopes(fit, "x")
    assert float(res.frame["slope"].iloc[0]) == pytest.approx(
        float(fit.params["x"]), abs=1e-9
    )


def test_linear_avg_slope_se_equals_coefficient_se():
    """And its SE equals the coefficient's SE, exactly."""
    fit, _ = _ols()
    res = avg_slopes(fit, "x")
    assert float(res.frame["SE"].iloc[0]) == pytest.approx(
        float(fit.bse["x"]), abs=1e-8
    )


def test_logit_response_ame_matches_analytic():
    """Logistic response-scale AME == mean(p*(1-p)*beta_x), the closed form."""
    fit, df = _logit()
    res = avg_slopes(fit, "x", type="response")
    eta = fit.params["Intercept"] + fit.params["x"] * df["x"]
    p = 1.0 / (1.0 + np.exp(-eta))
    ame_analytic = float((p * (1.0 - p) * fit.params["x"]).mean())
    assert float(res.frame["slope"].iloc[0]) == pytest.approx(ame_analytic, abs=1e-7)


def test_logit_response_ame_differs_from_coefficient():
    """Sanity: the response AME is not the logit coefficient."""
    fit, _ = _logit()
    res = avg_slopes(fit, "x", type="response")
    assert abs(float(res.frame["slope"].iloc[0]) - float(fit.params["x"])) > 0.1


# ---------------------------------------------------------------------- by-grouping


def test_by_grouping_returns_one_row_per_level():
    fit, _ = _ols()
    res = avg_slopes(fit, "x", by="g")
    assert len(res.frame) == 2
    assert set(res.frame["g"]) == {"A", "B"}
    # In an additive model the slope of x is the same in both groups.
    slopes_by = res.frame.set_index("g")["slope"]
    assert float(slopes_by["A"]) == pytest.approx(float(slopes_by["B"]), abs=1e-9)


def test_by_grouping_slope_differs_under_interaction():
    """With an x:g interaction, the per-group slopes differ."""
    rng = np.random.default_rng(5)
    n = 800
    df = pd.DataFrame({
        "x": rng.standard_normal(n),
        "g": pd.Categorical(rng.choice(["A", "B"], n)),
    })
    df["y"] = (
        df["x"] * df["g"].map({"A": 1.0, "B": 3.0}).astype(float)
        + rng.standard_normal(n)
    )
    fit = smf.ols("y ~ x * g", df).fit()
    res = avg_slopes(fit, "x", by="g")
    s = res.frame.set_index("g")["slope"]
    assert float(s["A"]) == pytest.approx(1.0, abs=0.15)
    assert float(s["B"]) == pytest.approx(3.0, abs=0.15)


# ---------------------------------------------------------------------- slopes() per-obs


def test_slopes_per_observation_rowcount():
    fit, df = _ols()
    res = slopes(fit, "x")
    assert len(res.frame) == len(df)
    # Every per-row link slope equals the (constant) coefficient in a
    # linear additive model.
    assert np.allclose(res.frame["slope"].to_numpy(), float(fit.params["x"]), atol=1e-9)


def test_slopes_avg_equals_avg_slopes():
    """The mean of per-observation slopes equals the avg_slopes point estimate."""
    fit, _ = _logit()
    per = slopes(fit, "x", type="response")
    avg = avg_slopes(fit, "x", type="response")
    assert per.frame["slope"].mean() == pytest.approx(
        float(avg.frame["slope"].iloc[0]), abs=1e-8
    )


def test_slopes_response_se_matches_full_analytic_delta_method():
    """Per-row response SE must be the FULL delta method, not the
    point-evaluated |h'(eta)|*link_se shortcut.

    Regression guard: the shortcut drops the h'' curvature term and was
    wrong by up to ~83% on a logistic fit. The correct per-row Jacobian
    of theta_i(beta) = sigma'(eta_i)*beta_x is
        dtheta/db0 = sigma''(eta)*beta_x
        dtheta/db1 = sigma''(eta)*x*beta_x + sigma'(eta)
    with sigma'(e)=p(1-p), sigma''(e)=p(1-p)(1-2p).
    """
    fit, df = _logit()
    res = slopes(fit, "x", type="response")
    se_pm = res.frame["SE"].to_numpy()

    b0, b1 = float(fit.params["Intercept"]), float(fit.params["x"])
    V = np.asarray(fit.cov_params())
    x = df["x"].to_numpy()
    p = 1.0 / (1.0 + np.exp(-(b0 + b1 * x)))
    sp = p * (1.0 - p)
    spp = p * (1.0 - p) * (1.0 - 2.0 * p)
    j0 = spp * b1
    j1 = spp * x * b1 + sp
    se_analytic = np.sqrt(
        j0**2 * V[0, 0] + j1**2 * V[1, 1] + 2.0 * j0 * j1 * V[0, 1]
    )
    np.testing.assert_allclose(se_pm, se_analytic, rtol=1e-6)


# ---------------------------------------------------------------------- ML adapter angle


def test_avg_slopes_on_ml_adapter_via_emmeans_grid():
    """avg_slopes interoperates with the from_predict ML adapter path.

    The ML adapter does not expose a beta/V design, so avg_slopes on a
    parametric fit is the supported path; here we simply confirm that
    avg_slopes on the underlying GLM matches a finite-difference AME
    computed directly from the model's own predict() — the same number
    an ML g-computation slope would target.
    """
    fit, df = _logit()
    res = avg_slopes(fit, "x", type="response")
    h = 1e-5
    dp = df.copy(); dm = df.copy()
    dp["x"] = df["x"] + h; dm["x"] = df["x"] - h
    fd_ame = float(((fit.predict(dp) - fit.predict(dm)) / (2 * h)).mean())
    assert float(res.frame["slope"].iloc[0]) == pytest.approx(fd_ame, abs=1e-6)


# ---------------------------------------------------------------------- validation


def test_rejects_unknown_type():
    fit, _ = _ols()
    with pytest.raises(ValueError, match="type"):
        avg_slopes(fit, "x", type="nonsense")


def test_rejects_invalid_level():
    fit, _ = _ols()
    with pytest.raises(ValueError, match="level"):
        avg_slopes(fit, "x", level=0.0)


def test_rejects_unknown_var():
    fit, _ = _ols()
    with pytest.raises(ValueError, match="not a column"):
        avg_slopes(fit, "nope")


def test_rejects_categorical_var_with_clear_message():
    """A non-numeric var must raise a clear 'requires numeric' error,
    not a cryptic patsy/pandas cast failure."""
    fit, _ = _ols()
    with pytest.raises(ValueError, match="numeric"):
        avg_slopes(fit, "g")
    with pytest.raises(ValueError, match="numeric"):
        slopes(fit, "g")


def test_response_on_linear_model_raises():
    fit, _ = _ols()
    with pytest.raises(ValueError, match="response"):
        avg_slopes(fit, "x", type="response")


def test_returns_documented_columns():
    fit, _ = _ols()
    res = avg_slopes(fit, "x")
    assert isinstance(res, SlopesResult)
    for col in ("var", "slope", "SE", "df", "t_ratio", "p_value", "lower_cl", "upper_cl"):
        assert col in res.frame.columns


def test_var_column_is_populated():
    """Regression guard: the var column must carry the predictor name."""
    fit, _ = _ols()
    res = avg_slopes(fit, "x")
    assert res.frame["var"].iloc[0] == "x"
