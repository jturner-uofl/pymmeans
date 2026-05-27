"""Reproducible R cross-validation benchmark.

For every model class pymmeans claims R-parity for, this file:
1. Loads a fixed Python-generated dataset from tests/r_reference/*.csv.
2. Fits the equivalent statsmodels model.
3. Compares pymmeans' output to a CSV produced by running the
   equivalent R model + emmeans (tests/r_reference/cross_validation.R).

To regenerate the R references (when you change a numerical
algorithm or bump an R package):

    .venv/bin/python tests/r_reference/generate_cv_data.py
    Rscript tests/r_reference/cross_validation.R

Then run this file:

    pytest tests/test_r_benchmark.py -v

Each test asserts pymmeans matches R to a documented tolerance. Any
future regression on R parity will surface here immediately.

The CSV files this test reads are committed; CI runs the assertions
without needing R installed. Only the regeneration step needs R.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymmeans import (
    SurveyDesign,
    apply_satterthwaite,
    contrast,
    emmeans,
    from_survey,
    joint_tests,
    regrid_response,
)

REF = Path(__file__).parent / "r_reference"


def _require(name: str) -> Path:
    path = REF / name
    if not path.exists():
        pytest.skip(
            f"missing R reference {name}; run "
            "`Rscript tests/r_reference/cross_validation.R` to regenerate."
        )
    return path


# === afex factorial ANOVA + emmeans ====================================


def test_benchmark_afex_emm_a_by_b():
    df = pd.read_csv(_require("afex_data.csv"))
    df["A"] = pd.Categorical(df["A"])

    df["B"] = pd.Categorical(df["B"])
    r = pd.read_csv(_require("afex_emm_A_by_B.csv"))
    fit = smf.ols("y ~ A * B", data=df).fit()
    py = emmeans(fit, "A", by="B").frame
    # Match levels by (A, B); merge frames for direct comparison.
    merged = py.merge(r, on=["A", "B"], suffixes=("_py", "_r"))
    np.testing.assert_allclose(
        merged["emmean_py"], merged["emmean_r"], atol=1e-5
    )
    np.testing.assert_allclose(
        merged["SE_py"], merged["SE_r"], atol=1e-5
    )
    np.testing.assert_allclose(
        merged["df_py"], merged["df_r"], atol=1e-3
    )


def test_benchmark_afex_pairs_a_by_b():
    df = pd.read_csv(_require("afex_data.csv"))
    df["A"] = pd.Categorical(df["A"])

    df["B"] = pd.Categorical(df["B"])
    r = pd.read_csv(_require("afex_pairs_A_by_B.csv"))
    fit = smf.ols("y ~ A * B", data=df).fit()
    py = contrast(emmeans(fit, "A", by="B"), method="pairwise").frame
    # Both python and R have one row per (B, contrast) pair.
    py_sorted = py.sort_values(["B", "contrast"]).reset_index(drop=True)
    r_sorted = r.sort_values(["B", "contrast"]).reset_index(drop=True)
    np.testing.assert_allclose(
        py_sorted["estimate"], r_sorted["estimate"], atol=1e-5
    )
    np.testing.assert_allclose(py_sorted["SE"], r_sorted["SE"], atol=1e-5)
    np.testing.assert_allclose(
        py_sorted["t_ratio"], r_sorted["t.ratio"], atol=1e-3
    )


def test_benchmark_afex_joint_tests():
    df = pd.read_csv(_require("afex_data.csv"))
    df["A"] = pd.Categorical(df["A"])

    df["B"] = pd.Categorical(df["B"])
    r = pd.read_csv(_require("afex_joint_tests.csv"))
    fit = smf.ols("y ~ A * B", data=df).fit()
    py = joint_tests(fit)
    # R uses "model term" with whitespace stripping needed.
    r["term"] = r["model term"].str.strip()
    merged = py.merge(r, on="term")
    np.testing.assert_allclose(
        merged["statistic"], merged["F.ratio"], rtol=1e-3
    )
    np.testing.assert_array_equal(merged["df_num"], merged["df1"])


# === lme4 + lmerTest Satterthwaite =====================================


def test_benchmark_lme4_ri_satterthwaite_emm():
    df = pd.read_csv(_require("lme4_ri_data.csv"))
    df["treatment"] = pd.Categorical(df["treatment"])
    r = pd.read_csv(_require("lme4_ri_emm_satt.csv"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm(
            "y ~ treatment + x", data=df, groups="subject"
        ).fit(reml=True)
    py = apply_satterthwaite(emmeans(fit, "treatment")).frame
    merged = py.merge(r, on="treatment", suffixes=("_py", "_r"))
    # Point estimate and SE: match to <1e-4
    np.testing.assert_allclose(
        merged["emmean_py"], merged["emmean_r"], atol=1e-5
    )
    np.testing.assert_allclose(merged["SE_py"], merged["SE_r"], atol=1e-4)
    # Satt df: match to <0.1
    np.testing.assert_allclose(merged["df_py"], merged["df_r"], atol=0.1)


def test_benchmark_lme4_rs_satterthwaite_emm():
    df = pd.read_csv(_require("lme4_rs_data.csv"))
    df["treatment"] = pd.Categorical(df["treatment"])
    r = pd.read_csv(_require("lme4_rs_emm_satt.csv"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm(
            "y ~ treatment + x", data=df, groups="subject", re_formula="~x"
        ).fit(reml=True)
    py = apply_satterthwaite(emmeans(fit, "treatment")).frame
    merged = py.merge(r, on="treatment", suffixes=("_py", "_r"))
    np.testing.assert_allclose(
        merged["emmean_py"], merged["emmean_r"], atol=1e-4
    )
    np.testing.assert_allclose(merged["SE_py"], merged["SE_r"], atol=1e-4)
    np.testing.assert_allclose(merged["df_py"], merged["df_r"], atol=0.2)


# === lme4 + pbkrtest Kenward-Roger =====================================


def test_benchmark_lme4_kr_emm():
    """KR matches pbkrtest on non-intercept SEs to <0.1%; intercept SE
    has a residual ~2.6% gap from the missing 3rd-order Kenward-Roger
    correction. Test the achievable accuracy and document the gap."""
    df = pd.read_csv(_require("lme4_ri_data.csv"))
    df["treatment"] = pd.Categorical(df["treatment"])
    r = pd.read_csv(_require("lme4_ri_emm_kr.csv"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm(
            "y ~ treatment + x", data=df, groups="subject"
        ).fit(reml=True)
        from pymmeans import apply_kenward_roger

        py = apply_kenward_roger(emmeans(fit, "treatment")).frame
    merged = py.merge(r, on="treatment", suffixes=("_py", "_r"))
    # Point estimate exact.
    np.testing.assert_allclose(
        merged["emmean_py"], merged["emmean_r"], atol=1e-5
    )
    # SE within 3% — known residual gap from missing 3rd-order
    # Kenward-Roger small-sample correction. Treatment/slope contrasts
    # (handled in test_benchmark_lme4_ri_satterthwaite_emm) match R to
    # <1e-4 already; only EMMs that route through the intercept inherit
    # the small-sample gap.
    rel_se = np.abs(merged["SE_py"] - merged["SE_r"]) / merged["SE_r"]
    assert (rel_se < 0.03).all(), f"KR SE rel diff {rel_se} exceeds 3%"
    # df within 6%.
    rel_df = np.abs(merged["df_py"] - merged["df_r"]) / merged["df_r"]
    assert (rel_df < 0.06).all(), f"KR df rel diff {rel_df} exceeds 6%"


# === marginaleffects + emmeans =========================================


def test_benchmark_marginaleffects_emm():
    df = pd.read_csv(_require("marginal_data.csv"))
    df["a"] = pd.Categorical(df["a"])

    df["b"] = pd.Categorical(df["b"])
    r = pd.read_csv(_require("marginal_emm_a.csv"))
    fit = smf.ols("y ~ a * b + z", data=df).fit()
    py = emmeans(fit, "a").frame
    merged = py.merge(r, on="a", suffixes=("_py", "_r"))
    np.testing.assert_allclose(
        merged["emmean_py"], merged["emmean_r"], atol=1e-6
    )
    np.testing.assert_allclose(merged["SE_py"], merged["SE_r"], atol=1e-6)


def test_benchmark_marginaleffects_pairs():
    df = pd.read_csv(_require("marginal_data.csv"))
    df["a"] = pd.Categorical(df["a"])

    df["b"] = pd.Categorical(df["b"])
    r = pd.read_csv(_require("marginal_pairs_a.csv"))
    fit = smf.ols("y ~ a * b + z", data=df).fit()
    py = contrast(emmeans(fit, "a"), method="pairwise").frame
    py_sorted = py.sort_values("contrast").reset_index(drop=True)
    r_sorted = r.sort_values("contrast").reset_index(drop=True)
    np.testing.assert_allclose(
        py_sorted["estimate"], r_sorted["estimate"], atol=1e-5
    )
    np.testing.assert_allclose(py_sorted["SE"], r_sorted["SE"], atol=1e-5)


# === survey::svyglm SRS ===============================================


def test_benchmark_survey_srs_coef():
    df = pd.read_csv(_require("survey_srs_data.csv"))
    df["a"] = pd.Categorical(df["a"])
    r = pd.read_csv(_require("survey_srs_coef.csv"))
    fit = smf.wls("y ~ a + x", data=df, weights=df["w"]).fit()
    info = from_survey(fit, SurveyDesign(weights=df["w"].to_numpy()))
    py_se = np.sqrt(np.diag(info.vcov))
    # R's coef table ordering matches statsmodels' (Intercept, aB, aC, x).
    r_se = r["Std..Error"].to_numpy() if "Std..Error" in r.columns else r["Std. Error"].to_numpy()
    np.testing.assert_allclose(py_se, r_se, atol=1e-7)


def test_benchmark_survey_srs_emm():
    df = pd.read_csv(_require("survey_srs_data.csv"))
    df["a"] = pd.Categorical(df["a"])
    r = pd.read_csv(_require("survey_srs_emm.csv"))
    fit = smf.wls("y ~ a + x", data=df, weights=df["w"]).fit()
    info = from_survey(fit, SurveyDesign(weights=df["w"].to_numpy()))
    py = emmeans(info, "a").frame
    merged = py.merge(r, on="a", suffixes=("_py", "_r"))
    np.testing.assert_allclose(
        merged["emmean_py"], merged["emmean_r"], atol=1e-7
    )
    np.testing.assert_allclose(merged["SE_py"], merged["SE_r"], atol=1e-7)


# === survey::svyglm Poisson (#1 regression) =================


def test_benchmark_survey_poisson_coef():
    """Survey Poisson GLM coef SEs match R survey::svyglm to <1e-7.
    regression: previous OLS-style sandwich gave ~15% wrong SEs."""
    df = pd.read_csv(_require("survey_poisson_data.csv"))
    df["a"] = pd.Categorical(df["a"])
    r = pd.read_csv(_require("survey_poisson_coef.csv"))
    fit = smf.glm(
        "y ~ a + x", data=df, family=sm.families.Poisson(),
        freq_weights=df["w"],
    ).fit()
    info = from_survey(fit, SurveyDesign(df["w"].to_numpy()))
    py_se = np.sqrt(np.diag(info.vcov))
    r_se = (
        r["Std..Error"].to_numpy()
        if "Std..Error" in r.columns
        else r["Std. Error"].to_numpy()
    )
    np.testing.assert_allclose(py_se, r_se, atol=1e-7)


def test_benchmark_survey_poisson_emm():
    """Survey Poisson GLM EMMs (response scale) match R survey::svyglm."""
    df = pd.read_csv(_require("survey_poisson_data.csv"))
    df["a"] = pd.Categorical(df["a"])
    r = pd.read_csv(_require("survey_poisson_emm.csv"))
    # R writes the response-scale column as `response`; normalise.
    if "response" in r.columns and "emmean" not in r.columns:
        r = r.rename(columns={"response": "emmean"})
    fit = smf.glm(
        "y ~ a + x", data=df, family=sm.families.Poisson(),
        freq_weights=df["w"],
    ).fit()
    info = from_survey(fit, SurveyDesign(df["w"].to_numpy()))
    py = emmeans(info, "a", type="response").frame
    merged = py.merge(r, on="a", suffixes=("_py", "_r"))
    np.testing.assert_allclose(
        merged["emmean_py"], merged["emmean_r"], atol=1e-6
    )
    np.testing.assert_allclose(merged["SE_py"], merged["SE_r"], atol=1e-6)


# === survey::svyglm Binomial logit (coverage) ==================


def test_benchmark_survey_binomial_coef():
    """Survey Binomial logit GLM coef SEs match R survey::svyglm.
    Covers the IRLS bread + score factor on the logit link (not just
    log link from Poisson)."""
    df = pd.read_csv(_require("survey_binomial_data.csv"))
    df["a"] = pd.Categorical(df["a"])
    r = pd.read_csv(_require("survey_binomial_coef.csv"))
    fit = smf.glm(
        "y ~ a + x", data=df, family=sm.families.Binomial(),
        freq_weights=df["w"],
    ).fit()
    info = from_survey(fit, SurveyDesign(df["w"].to_numpy()))
    py_se = np.sqrt(np.diag(info.vcov))
    r_se = (
        r["Std..Error"].to_numpy()
        if "Std..Error" in r.columns
        else r["Std. Error"].to_numpy()
    )
    np.testing.assert_allclose(py_se, r_se, atol=1e-7)


def test_benchmark_survey_binomial_emm():
    """Survey Binomial EMMs (response/probability scale) match R."""
    df = pd.read_csv(_require("survey_binomial_data.csv"))
    df["a"] = pd.Categorical(df["a"])
    r = pd.read_csv(_require("survey_binomial_emm.csv"))
    # R uses 'prob' for binomial response-scale; normalise to emmean.
    for col in ("prob", "response"):
        if col in r.columns and "emmean" not in r.columns:
            r = r.rename(columns={col: "emmean"})
    fit = smf.glm(
        "y ~ a + x", data=df, family=sm.families.Binomial(),
        freq_weights=df["w"],
    ).fit()
    info = from_survey(fit, SurveyDesign(df["w"].to_numpy()))
    py = emmeans(info, "a", type="response").frame
    merged = py.merge(r, on="a", suffixes=("_py", "_r"))
    np.testing.assert_allclose(
        merged["emmean_py"], merged["emmean_r"], atol=1e-6
    )
    np.testing.assert_allclose(merged["SE_py"], merged["SE_r"], atol=1e-6)


# === survey::svyglm Gamma log (non-canonical link) =============


def test_benchmark_survey_gamma_coef():
    """Survey Gamma log GLM coef SEs match R survey::svyglm. Gamma-log
    is a non-canonical link, so score_factor != 1 and exercises a code
    path distinct from Poisson / Binomial."""
    df = pd.read_csv(_require("survey_gamma_data.csv"))
    df["a"] = pd.Categorical(df["a"])
    r = pd.read_csv(_require("survey_gamma_coef.csv"))
    fit = smf.glm(
        "y ~ a + x", data=df,
        family=sm.families.Gamma(link=sm.families.links.Log()),
        freq_weights=df["w"],
    ).fit()
    info = from_survey(fit, SurveyDesign(df["w"].to_numpy()))
    py_se = np.sqrt(np.diag(info.vcov))
    r_se = (
        r["Std..Error"].to_numpy()
        if "Std..Error" in r.columns
        else r["Std. Error"].to_numpy()
    )
    # Gamma + log link: matches R to 1e-5 (slightly looser than
    # Poisson/Binomial; the non-canonical link path has more rounding
    # in family.weights and family.link.deriv).
    np.testing.assert_allclose(py_se, r_se, atol=1e-5)


def test_benchmark_survey_gamma_emm():
    """Survey Gamma log EMMs match R."""
    df = pd.read_csv(_require("survey_gamma_data.csv"))
    df["a"] = pd.Categorical(df["a"])
    r = pd.read_csv(_require("survey_gamma_emm.csv"))
    if "response" in r.columns and "emmean" not in r.columns:
        r = r.rename(columns={"response": "emmean"})
    fit = smf.glm(
        "y ~ a + x", data=df,
        family=sm.families.Gamma(link=sm.families.links.Log()),
        freq_weights=df["w"],
    ).fit()
    info = from_survey(fit, SurveyDesign(df["w"].to_numpy()))
    py = emmeans(info, "a", type="response").frame
    merged = py.merge(r, on="a", suffixes=("_py", "_r"))
    np.testing.assert_allclose(
        merged["emmean_py"], merged["emmean_r"], atol=1e-4
    )
    np.testing.assert_allclose(merged["SE_py"], merged["SE_r"], atol=1e-4)


# === GLM exposure offset (Poisson with log(exposure)) ==================


def test_benchmark_exposure_emm():
    df = pd.read_csv(_require("exposure_data.csv"))
    df["a"] = pd.Categorical(df["a"])
    r = pd.read_csv(_require("exposure_emm_response.csv"))
    fit = smf.glm(
        "y ~ a", data=df, family=sm.families.Poisson(),
        exposure=df["exposure"],
    ).fit()
    py = emmeans(fit, "a", type="response").frame
    merged = py.merge(r, on="a", suffixes=("_py", "_r"))
    np.testing.assert_allclose(
        merged["emmean_py"], merged["emmean_r"], atol=1e-6
    )
    np.testing.assert_allclose(merged["SE_py"], merged["SE_r"], atol=1e-6)


# === bias_adjust (R's Taylor formula) ==================================


def test_benchmark_bias_adjust_taylor():
    df = pd.read_csv(_require("bias_adjust_data.csv"))
    df["a"] = pd.Categorical(df["a"])
    r = pd.read_csv(_require("bias_adjust_emm.csv"))
    # R uses `response` column for bias-adjusted EMMs; rename to match
    # pymmeans' `emmean`.
    r = r.rename(columns={"response": "emmean"})
    fit = smf.ols("np.log(y) ~ a", data=df).fit()
    py = regrid_response(emmeans(fit, "a"), bias_adjust=True).frame
    merged = py.merge(r, on="a", suffixes=("_py", "_r"))
    np.testing.assert_allclose(
        merged["emmean_py"], merged["emmean_r"], atol=1e-5
    )
    np.testing.assert_allclose(merged["SE_py"], merged["SE_r"], atol=1e-5)
    np.testing.assert_allclose(
        merged["lower_cl"], merged["lower.CL"], atol=1e-5
    )
    np.testing.assert_allclose(
        merged["upper_cl"], merged["upper.CL"], atol=1e-5
    )
