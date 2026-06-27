"""Tests for the pyfixest adapter (coefficient-level surface).

pyfixest absorbs fixed effects and parses with formulaic, so pymmeans
supports the coefficient-level operations (hypotheses / delta method)
on pyfixest fits but not reference-grid operations (emmeans /
avg_slopes), which require patsy. These tests validate the supported
surface against a dummy-encoded statsmodels OLS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

pf = pytest.importorskip("pyfixest")

from pymmeans import (  # noqa: E402
    avg_slopes,
    emmeans,
    from_pyfixest,
    hypotheses,
    ref_grid,
)
from pymmeans.utils import from_fitted  # noqa: E402


def _data(n=600, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x1": rng.standard_normal(n),
        "x2": rng.standard_normal(n),
        "fe": rng.integers(0, 10, n),
        "fe2": rng.integers(0, 5, n),
    })
    df["y"] = (
        1.5 * df["x1"] - 0.8 * df["x2"] + df["fe"] * 0.4
        + rng.standard_normal(n)
    )
    df["yc"] = (
        rng.random(n) < 1.0 / (1.0 + np.exp(-(0.2 + 0.3 * df["x1"])))
    ).astype(int)
    return df


# ---------------------------------------------------------------------- extraction


def test_from_pyfixest_extracts_within_fe_coefficients():
    """beta / vcov / names match a dummy-encoded statsmodels OLS."""
    df = _data()
    m = pf.feols("y ~ x1 + x2 | fe", data=df)
    fit = smf.ols("y ~ x1 + x2 + C(fe)", df).fit()
    info = from_pyfixest(m)

    assert list(info.param_names) == ["x1", "x2"]
    # Coefficients match the dummy-OLS within-FE estimates.
    np.testing.assert_allclose(
        info.beta,
        [float(fit.params["x1"]), float(fit.params["x2"])],
        atol=1e-9,
    )
    # design_info is None — reference-grid ops unsupported, by design.
    assert info.design_info is None


def test_from_fitted_auto_detects_pyfixest():
    df = _data()
    m = pf.feols("y ~ x1 + x2 | fe", data=df)
    info = from_fitted(m)
    assert list(info.param_names) == ["x1", "x2"]


# ---------------------------------------------------------------------- G1 / cross-val


def test_hypotheses_ratio_matches_dummy_ols():
    """hypotheses(b1/b2) on pyfixest matches the dummy-OLS closed form."""
    df = _data()
    m = pf.feols("y ~ x1 + x2 | fe", data=df)
    fit = smf.ols("y ~ x1 + x2 + C(fe)", df).fit()

    res = hypotheses(m, lambda b: b[0] / b[1], labels=["x1/x2"])

    beta = np.asarray(fit.params)
    V = np.asarray(fit.cov_params())
    names = list(fit.params.index)
    i, j = names.index("x1"), names.index("x2")
    b1, b2 = beta[i], beta[j]
    r = b1 / b2
    se_cf = np.sqrt(
        r**2 * (V[i, i] / b1**2 + V[j, j] / b2**2 - 2 * V[i, j] / (b1 * b2))
    )

    assert float(res.estimate[0]) == pytest.approx(r, abs=1e-6)
    assert float(res.se[0]) == pytest.approx(se_cf, abs=1e-6)


def test_hypotheses_linear_combo_matches_dummy_ols():
    """A linear g(b) = b1 - b2 matches the exact dummy-OLS contrast SE."""
    df = _data()
    m = pf.feols("y ~ x1 + x2 | fe", data=df)
    fit = smf.ols("y ~ x1 + x2 + C(fe)", df).fit()
    res = hypotheses(m, lambda b: b[0] - b[1])
    V = np.asarray(fit.cov_params())
    names = list(fit.params.index)
    i, j = names.index("x1"), names.index("x2")
    L = np.zeros(len(names)); L[i] = 1.0; L[j] = -1.0
    assert float(res.se[0]) == pytest.approx(np.sqrt(L @ V @ L), abs=1e-6)


# ---------------------------------------------------------------------- df / inference


def test_feols_df_matches_dummy_encoded_ols():
    """feols residual df must equal the dummy-encoded OLS df, accounting
    for the absorbed fixed-effect dimensions (single and multiple FE).

    Regression guard: an earlier version used N - k, ignoring the
    absorbed FE dimensions, overstating df and understating p-values.
    """
    df = _data()
    for spec, dummy in (
        ("y ~ x1 + x2 | fe", "y ~ x1 + x2 + C(fe)"),
        ("y ~ x1 + x2 | fe + fe2", "y ~ x1 + x2 + C(fe) + C(fe2)"),
    ):
        m = pf.feols(spec, data=df)
        info = from_pyfixest(m)
        dummy_df = float(smf.ols(dummy, df).fit().df_resid)
        assert float(info.df_resid) == pytest.approx(dummy_df, abs=1e-9), spec


def test_feols_hypotheses_pvalue_matches_pyfixest_own():
    """hypotheses() inference on a feols fit must reproduce pyfixest's
    own (t-based) p-value, not just its point estimate."""
    df = _data()
    m = pf.feols("y ~ x1 + x2 | fe + fe2", data=df)
    res = hypotheses(m, lambda b: b[0])
    pf_p = float(m.tidy()["Pr(>|t|)"].iloc[0])
    assert float(res.frame["p_value"].iloc[0]) == pytest.approx(pf_p, abs=1e-7)


def test_fepois_uses_asymptotic_z_inference():
    """Fepois (Poisson GLM) is asymptotic: df must be inf (z-test), and
    hypotheses() must reproduce pyfixest's own p-value.

    Regression guard: an earlier version applied a finite-df t-test to
    the GLM, producing materially wrong p-values.
    """
    df = _data()
    mp = pf.fepois("yc ~ x1 | fe", data=df)
    info = from_pyfixest(mp)
    assert info.df_resid == np.inf
    res = hypotheses(mp, lambda b: b[0])
    pf_p = float(mp.tidy()["Pr(>|t|)"].iloc[0])  # pyfixest labels it t but uses z
    assert float(res.frame["p_value"].iloc[0]) == pytest.approx(pf_p, abs=1e-7)


# ---------------------------------------------------------------------- errors


def test_from_pyfixest_rejects_non_pyfixest():
    with pytest.raises(TypeError, match="pyfixest"):
        from_pyfixest(object())


@pytest.mark.parametrize("fn", [emmeans, ref_grid, avg_slopes])
def test_reference_grid_ops_give_clean_error_on_pyfixest(fn):
    """emmeans / ref_grid / avg_slopes on a pyfixest fit must raise a
    clear, steering ValueError (not a cryptic AttributeError or a false
    'this was pickled' message)."""
    df = _data()
    m = pf.feols("y ~ x1 | fe", data=df)
    with pytest.raises(ValueError, match="pyfixest"):
        fn(m, "x1") if fn is not ref_grid else fn(m)
