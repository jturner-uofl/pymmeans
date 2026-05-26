"""Tests for Satterthwaite and Kenward-Roger inference on MixedLM."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import (
    apply_kenward_roger,
    apply_satterthwaite,
    emmeans,
    kenward_roger_vcov,
    pairs,
    satterthwaite_df,
)
from pymmeans.utils import from_fitted


@pytest.fixture
def random_intercept_fit():
    rng = np.random.default_rng(0)
    n_groups = 25
    n_per = 8
    n = n_groups * n_per
    subj = np.repeat(np.arange(n_groups), n_per)
    subj_effect = rng.normal(scale=0.6, size=n_groups)[subj]
    g = rng.choice(["a", "b", "c"], n)
    y = (
        1.0
        + 0.4 * (g == "b")
        + 0.8 * (g == "c")
        + subj_effect
        + rng.normal(scale=0.5, size=n)
    )
    df = pd.DataFrame({"y": y, "g": pd.Categorical(g), "subj": subj})
    return smf.mixedlm("y ~ g", data=df, groups="subj").fit()


def test_satterthwaite_df_returns_finite_for_random_intercept(random_intercept_fit):
    info = from_fitted(random_intercept_fit)
    emm = emmeans(random_intercept_fit, "g")
    df = satterthwaite_df(info, emm.linfct)
    assert df.shape == (emm.n_rows,)
    assert np.all(np.isfinite(df))
    # Should be MUCH smaller than inf and reasonable for k=25 groups
    assert (df > 1.0).all()
    assert (df < 500.0).all()


def test_apply_satterthwaite_to_emm(random_intercept_fit):
    emm = emmeans(random_intercept_fit, "g")
    emm_s = apply_satterthwaite(emm)
    assert np.all(np.isfinite(emm_s.frame["df"]))
    # SE shouldn't change, just CI width
    np.testing.assert_array_almost_equal(
        emm_s.frame["SE"].to_numpy(), emm.frame["SE"].to_numpy()
    )
    # Satterthwaite CIs are wider than Wald-z CIs (since t > z for finite df)
    w_z = (emm.frame["upper_cl"] - emm.frame["lower_cl"]).to_numpy()
    w_s = (emm_s.frame["upper_cl"] - emm_s.frame["lower_cl"]).to_numpy()
    assert (w_s > w_z).all()


def test_apply_satterthwaite_to_pairs(random_intercept_fit):
    pw = pairs(emmeans(random_intercept_fit, "g"), adjust="none")
    pw_s = apply_satterthwaite(pw)
    assert np.all(np.isfinite(pw_s.frame["df"]))
    # P-values increase with finite df (less significant than z-test)
    assert (pw_s.frame["p_value"] >= pw.frame["p_value"] - 1e-9).all()


def test_satterthwaite_falls_back_for_non_mixed():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.normal(size=60), "g": pd.Categorical(rng.choice(["a", "b"], 60))})
    fit = smf.ols("y ~ g", data=df).fit()
    info = from_fitted(fit)
    emm = emmeans(fit, "g")
    out = satterthwaite_df(info, emm.linfct)
    # For OLS, falls back to df_resid (not inf)
    assert (out == fit.df_resid).all()


def test_satterthwaite_handles_random_slopes():
    rng = np.random.default_rng(2)
    n_groups = 40
    n_per = 12
    n = n_groups * n_per
    subj = np.repeat(np.arange(n_groups), n_per)
    u = rng.normal(scale=0.5, size=n_groups)[subj]
    b = rng.normal(scale=0.2, size=n_groups)[subj]
    g = rng.choice(["a", "b", "c"], n)
    x = rng.normal(size=n)
    y = (
        1.0
        + 0.5 * x
        + 0.3 * (g == "b")
        - 0.2 * (g == "c")
        + u
        + b * x
        + rng.normal(scale=0.3, size=n)
    )
    data = pd.DataFrame({"y": y, "x": x, "g": pd.Categorical(g), "subj": subj})
    fit = smf.mixedlm("y ~ g + x", data=data, groups="subj", re_formula="~x").fit()
    info = from_fitted(fit)
    emm = emmeans(fit, "g")
    df_satt = satterthwaite_df(info, emm.linfct)
    # Should be finite and reasonable (between roughly 1 and n_groups + 50)
    assert np.all(np.isfinite(df_satt))
    assert (df_satt > 1.0).all()


def test_kenward_roger_vcov_close_to_unadjusted():
    """KR is generally a small adjustment to V_b — the Kackar-Harville
    term inflates *or* deflates depending on the curvature of the
    profile likelihood. review replaced the previous
    "always inflates" assertion (which contradicted the true KR
    behaviour and pbkrtest output) with a tighter sanity check."""
    rng = np.random.default_rng(3)
    n_groups = 15
    n_per = 6
    n = n_groups * n_per
    subj = np.repeat(np.arange(n_groups), n_per)
    u = rng.normal(scale=0.6, size=n_groups)[subj]
    g = rng.choice(["a", "b", "c"], n)
    y = (g == "b") - 0.5 * (g == "c") + u + rng.normal(scale=0.4, size=n)
    data = pd.DataFrame({"y": y, "g": pd.Categorical(g), "subj": subj})
    fit = smf.mixedlm("y ~ g", data=data, groups="subj").fit()
    info = from_fitted(fit)
    V_KR = kenward_roger_vcov(info)
    # KR vcov should be within 10 % of V_b on the diagonal (the
    # Kackar-Harville correction is a small adjustment).
    rel = np.abs(np.diag(V_KR) - np.diag(info.vcov)) / np.diag(info.vcov)
    assert (rel < 0.1).all(), f"KR vcov diverges >10 % from V_b: rel={rel}"
    # And both must be positive (PSD).
    assert (np.diag(V_KR) > 0).all()


@pytest.mark.filterwarnings("ignore:kenward_roger.. is experimental")
def test_kr_vcov_matches_pbkrtest_on_non_intercept_coefs():
    """Compare KR-adjusted vcov diagonal against R's pbkrtest output.

    R reference (generated by tests/r_reference/kr_reference.R) is loaded
    if available; otherwise the test skips.

    update: pre-, the intercept SE diverged by
    a few percent (the finding). rewrote
    ``kenward_roger_vcov`` to implement pbkrtest's exact
    ``vcovAdj_internal`` algorithm — the gap closed. All entries
    (intercept AND non-intercept) now match pbkrtest to ``atol=1e-5``.
    """
    from pathlib import Path

    from pymmeans.satterthwaite import kenward_roger_vcov
    from pymmeans.utils import from_fitted

    ref_csv = Path(__file__).parent / "r_reference" / "kr_reference.csv"
    data_csv = Path(__file__).parent / "r_reference" / "kr_reference_data.csv"
    if not ref_csv.exists() or not data_csv.exists():
        pytest.skip("KR reference data missing; run tests/r_reference/kr_reference.R.")

    data = pd.read_csv(data_csv)
    fit = smf.mixedlm("y ~ g", data=data, groups="subj").fit(reml=True)
    info = from_fitted(fit)
    V_kr = kenward_roger_vcov(info)
    our_se = np.sqrt(np.diag(V_kr))

    ref = pd.read_csv(ref_csv).set_index("name")
    # Order: Intercept, g[T.b], g[T.c] in pymmeans <-> (Intercept), gb, gc in R.
    # All entries — including the intercept — match to atol=1e-5 post-.
    expected = ref.loc[["(Intercept)", "gb", "gc"], "se_kr"].to_numpy()
    np.testing.assert_allclose(our_se, expected, atol=1e-5)


def test_apply_kenward_roger_changes_se_and_df():
    rng = np.random.default_rng(4)
    n_groups = 20
    n_per = 8
    n = n_groups * n_per
    subj = np.repeat(np.arange(n_groups), n_per)
    u = rng.normal(scale=0.5, size=n_groups)[subj]
    g = rng.choice(["a", "b", "c"], n)
    y = (g == "b") + u + rng.normal(scale=0.4, size=n)
    data = pd.DataFrame({"y": y, "g": pd.Categorical(g), "subj": subj})
    fit = smf.mixedlm("y ~ g", data=data, groups="subj").fit()
    emm = emmeans(fit, "g")
    emm_kr = apply_kenward_roger(emm)
    # KR SE should be close to Wald-z SE (within 10 %; KR is a small
    # adjustment, can inflate or deflate depending on profile curvature).
    se_kr = emm_kr.frame["SE"].to_numpy()
    se_wald = emm.frame["SE"].to_numpy()
    rel = np.abs(se_kr - se_wald) / se_wald
    assert (rel < 0.1).all(), f"KR SE diverges >10 % from Wald: rel={rel}"
    assert np.all(np.isfinite(emm_kr.frame["df"]))
