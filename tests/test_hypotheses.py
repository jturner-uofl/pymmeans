"""Tests for hypotheses() — nonlinear delta-method tests on coefficients."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import HypothesisResult, contrast, emmeans, hypotheses

# ---------------------------------------------------------------------- helpers


def _ols_fit(n=500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
    })
    df["y"] = 2.0 + 1.5 * df["x1"] + 0.8 * df["x2"] + rng.standard_normal(n)
    return smf.ols("y ~ x1 + x2", df).fit()


# ---------------------------------------------------------------------- G1: closed form


def test_ratio_se_matches_closed_form():
    """Numerical-Jacobian SE of b1/b2 matches the closed-form ratio SE."""
    fit = _ols_fit()
    beta = np.asarray(fit.params)
    V = np.asarray(fit.cov_params())
    # param order: Intercept(0), x1(1), x2(2)
    res = hypotheses(fit, lambda b: b[1] / b[2], labels=["x1/x2"])

    b1, b2 = beta[1], beta[2]
    r = b1 / b2
    var_cf = r**2 * (V[1, 1] / b1**2 + V[2, 2] / b2**2 - 2 * V[1, 2] / (b1 * b2))
    se_cf = np.sqrt(var_cf)

    assert float(res.estimate[0]) == pytest.approx(r, abs=1e-12)
    assert float(res.se[0]) == pytest.approx(se_cf, abs=1e-8)


def test_linear_g_reduces_to_exact_contrast():
    """When g is linear, the delta-method SE equals the exact L V L^T SE."""
    fit = _ols_fit()
    V = np.asarray(fit.cov_params())
    # g(b) = b1 - b2  (a linear contrast)
    res = hypotheses(fit, lambda b: b[1] - b[2], labels=["x1 - x2"])
    L = np.zeros(len(fit.params))
    L[1] = 1.0
    L[2] = -1.0
    se_exact = np.sqrt(L @ V @ L)
    assert float(res.se[0]) == pytest.approx(se_exact, abs=1e-9)


def test_identity_g_recovers_coefficient_se():
    """g(b) = b_k returns the coefficient and its SE exactly."""
    fit = _ols_fit()
    res = hypotheses(fit, lambda b: b[1], labels=["x1"])
    assert float(res.estimate[0]) == pytest.approx(float(fit.params.iloc[1]), abs=1e-10)
    assert float(res.se[0]) == pytest.approx(float(fit.bse.iloc[1]), abs=1e-8)


# ---------------------------------------------------------------------- df / inference


def test_df_uses_model_resid_df():
    """The t-test uses the model's residual df by default."""
    fit = _ols_fit(n=200)
    res = hypotheses(fit, lambda b: b[1] / b[2])
    assert float(res.df) == pytest.approx(float(fit.df_resid), abs=1e-9)


def test_df_override():
    """An explicit df override is honoured."""
    fit = _ols_fit()
    res = hypotheses(fit, lambda b: b[1], df=10.0)
    assert float(res.df) == pytest.approx(10.0)


def test_ci_consistent_with_t_critical():
    """CI bounds equal estimate +/- t_crit * SE at the model df."""
    import scipy.stats as stats
    fit = _ols_fit()
    res = hypotheses(fit, lambda b: b[1] / b[2], level=0.95)
    est = float(res.estimate[0])
    se = float(res.se[0])
    tcrit = stats.t.isf(0.025, res.df)
    assert float(res.frame["lower_cl"].iloc[0]) == pytest.approx(est - tcrit * se, abs=1e-10)
    assert float(res.frame["upper_cl"].iloc[0]) == pytest.approx(est + tcrit * se, abs=1e-10)


# ---------------------------------------------------------------------- vector g


def test_vector_valued_g():
    """g may return multiple quantities; each gets its own row."""
    fit = _ols_fit()
    res = hypotheses(
        fit,
        lambda b: np.array([b[1] / b[2], b[1] + b[2]]),
        labels=["ratio", "sum"],
    )
    assert len(res.frame) == 2
    assert res.jacobian.shape == (2, len(fit.params))
    # The sum row's SE must match the exact linear-combination SE.
    V = np.asarray(fit.cov_params())
    L = np.zeros(len(fit.params)); L[1] = 1.0; L[2] = 1.0
    assert float(res.se[1]) == pytest.approx(np.sqrt(L @ V @ L), abs=1e-9)


# ---------------------------------------------------------------------- accepts results


def test_accepts_emmresult_model_info():
    """hypotheses() works on a pymmeans EMMResult (via its model_info)."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "g": pd.Categorical(rng.choice(["A", "B", "C"], 200)),
        "x": rng.standard_normal(200),
    })
    df["y"] = df["g"].map({"A": 0.0, "B": 0.5, "C": 1.0}).astype(float) + rng.standard_normal(200)
    fit = smf.ols("y ~ g + x", df).fit()
    em = emmeans(fit, "g")
    # A nonlinear function of the underlying betas still works because the
    # EMMResult carries model_info.
    res = hypotheses(em, lambda b: b[1] / b[2])
    assert isinstance(res, HypothesisResult)
    assert np.isfinite(res.se[0])


def test_accepts_contrastresult_model_info():
    fit = _ols_fit()
    em = emmeans(fit, "x1", at={"x1": [0.0, 1.0]})
    ct = contrast(em, method="consec")
    res = hypotheses(ct, lambda b: b[1])
    assert isinstance(res, HypothesisResult)


# ---------------------------------------------------------------------- validation


def test_rejects_non_callable_g():
    fit = _ols_fit()
    with pytest.raises(TypeError, match="callable"):
        hypotheses(fit, "not callable")  # type: ignore[arg-type]


def test_rejects_invalid_level():
    fit = _ols_fit()
    with pytest.raises(ValueError, match="level"):
        hypotheses(fit, lambda b: b[1], level=1.5)


def test_rejects_mismatched_labels():
    fit = _ols_fit()
    with pytest.raises(ValueError, match="labels"):
        hypotheses(fit, lambda b: b[1], labels=["a", "b"])


def test_rejects_2d_g_output():
    fit = _ols_fit()
    with pytest.raises(ValueError, match="1-D"):
        hypotheses(fit, lambda b: np.eye(2))


def test_returns_documented_frame_columns():
    fit = _ols_fit()
    res = hypotheses(fit, lambda b: b[1] / b[2])
    for col in ("hypothesis", "estimate", "SE", "df", "t_ratio", "p_value", "lower_cl", "upper_cl"):
        assert col in res.frame.columns
