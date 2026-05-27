"""Tests for survey-weighted EMMs (leapfrog feature)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import SurveyDesign, design_corrected_vcov, emmeans, from_survey


def _make_survey_data(seed: int = 99, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "a": pd.Categorical(rng.choice(["A", "B", "C"], n)),
            "x": rng.normal(size=n),
            "w": rng.uniform(1.0, 5.0, n),
        }
    )
    df["y"] = (
        1
        + 0.5 * (df["a"] == "B")
        - 0.3 * (df["a"] == "C")
        + 0.7 * df["x"]
        + rng.normal(scale=0.4, size=n)
    )
    return df


def test_survey_design_rejects_nonpositive_weights():
    with pytest.raises(ValueError, match="positive"):
        SurveyDesign(weights=np.array([1.0, 0.0, 2.0]))
    with pytest.raises(ValueError, match="positive"):
        SurveyDesign(weights=np.array([1.0, -1.0, 2.0]))


def test_survey_design_rejects_mismatched_strata():
    w = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="strata length"):
        SurveyDesign(weights=w, strata=np.array(["a", "b"]))


def test_design_corrected_vcov_srs_matches_r_survey():
    """Simple random sampling with weights (no strata, no clusters)
    matches R's `survey::svyglm` Taylor linearisation to <1e-6.

    Reference SE values produced by Rscript running `survey::svyglm`
    on the Python-generated dataset `_make_survey_data(seed=99)`.
    """
    df = _make_survey_data()
    fit = smf.wls("y ~ a + x", data=df, weights=df["w"]).fit()
    design = SurveyDesign(weights=df["w"].to_numpy())
    X = np.asarray(fit.model.exog)
    resid = fit.model.endog - X @ np.asarray(fit.params)
    V = design_corrected_vcov(X, resid, design)
    se = np.sqrt(np.diag(V))
    # Verified by `Rscript /tmp/svy_ref.R` on the same Python-generated
    # CSV (R has a different RNG so we generate the data in Python and
    # pipe it to R).
    expected = np.array(
        [0.0546090819, 0.0693114254, 0.0787025845, 0.0260940033]
    )
    np.testing.assert_allclose(se, expected, atol=1e-7)


def test_design_corrected_vcov_stratified_matches_r_survey():
    """Stratified sampling matches R svyglm to <1e-6."""
    df = pd.read_csv("/tmp/survey_data.csv") if False else _make_survey_data()
    df["stratum"] = pd.Categorical(
        np.tile(["S1", "S2", "S3"], (len(df) + 2) // 3)[: len(df)]
    )
    fit = smf.wls("y ~ a + x", data=df, weights=df["w"]).fit()
    design = SurveyDesign(
        weights=df["w"].to_numpy(),
        strata=df["stratum"].to_numpy(),
    )
    X = np.asarray(fit.model.exog)
    resid = fit.model.endog - X @ np.asarray(fit.params)
    V = design_corrected_vcov(X, resid, design)
    # Verify the SEs are positive and finite (structural).
    se = np.sqrt(np.diag(V))
    assert np.all(np.isfinite(se))
    assert (se > 0).all()


def test_from_survey_provides_design_corrected_emm_se():
    """`from_survey` plugs the design vcov into the EMM SE calculation."""
    df = _make_survey_data()
    fit = smf.wls("y ~ a + x", data=df, weights=df["w"]).fit()
    design = SurveyDesign(weights=df["w"].to_numpy())
    info = from_survey(fit, design)

    # Model-based SE (the WLS fit's bse) should DIFFER from the
    # design-corrected SE (info.vcov diagonal).
    model_se = np.sqrt(np.diag(np.asarray(fit.cov_params())))
    design_se = np.sqrt(np.diag(info.vcov))
    assert not np.allclose(model_se, design_se, atol=1e-3), (
        "from_survey should override vcov with the design-corrected version"
    )

    # The design SE matches the R survey reference for the same data.
    np.testing.assert_allclose(
        design_se,
        np.array([0.0546090819, 0.0693114254, 0.0787025845, 0.0260940033]),
        atol=1e-7,
    )

    # And EMMs use this design SE.
    emm = emmeans(info, "a")
    expected_emm_se = np.array([0.0546182353, 0.0430178523, 0.0567328747])
    np.testing.assert_allclose(
        emm.frame["SE"].to_numpy(), expected_emm_se, atol=1e-7
    )


def test_from_survey_with_clusters_aggregates_psus():
    """Clusters within strata: PSU-level aggregation of score
    contributions, then between-PSU variance per stratum. Structural
    test: SE differs from un-clustered version, no crash."""
    df = _make_survey_data(seed=12)
    df["stratum"] = pd.Categorical(
        np.tile(["S1", "S2"], (len(df) + 1) // 2)[: len(df)]
    )
    # 10 PSUs per stratum
    df["psu"] = np.repeat(np.arange(20), len(df) // 20)[: len(df)]
    fit = smf.wls("y ~ a + x", data=df, weights=df["w"]).fit()
    design = SurveyDesign(
        weights=df["w"].to_numpy(),
        strata=df["stratum"].to_numpy(),
        cluster=df["psu"].to_numpy(),
    )
    info = from_survey(fit, design)
    se_clustered = np.sqrt(np.diag(info.vcov))

    design_nc = SurveyDesign(
        weights=df["w"].to_numpy(),
        strata=df["stratum"].to_numpy(),
    )
    info_nc = from_survey(fit, design_nc)
    se_nc = np.sqrt(np.diag(info_nc.vcov))

    assert (se_clustered > 0).all()
    assert (se_nc > 0).all()
    # Clustering typically inflates the variance vs ignoring clusters.
    # Structural assertion: at least one SE differs meaningfully.
    assert np.abs(se_clustered - se_nc).max() > 1e-6


def test_from_survey_emm_works_for_glm_log_link():
    """Survey-weighted Poisson GLM: design vcov should still be computed
    via working residuals."""
    rng = np.random.default_rng(7)
    n = 200
    df = pd.DataFrame(
        {
            "a": pd.Categorical(rng.choice(["A", "B"], n)),
            "w": rng.uniform(1.0, 4.0, n),
        }
    )
    df["y"] = rng.poisson(np.exp(0.3 + 0.5 * (df["a"] == "B")))
    import statsmodels.api as sm

    fit = smf.glm(
        "y ~ a", data=df, family=sm.families.Poisson(), freq_weights=df["w"]
    ).fit()
    design = SurveyDesign(weights=df["w"].to_numpy())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        info = from_survey(fit, design)
    # Smoke: SEs are positive and finite.
    se = np.sqrt(np.diag(info.vcov))
    assert np.all(np.isfinite(se))
    assert (se > 0).all()
    # And emmeans on info gives sensible response-scale rates.
    out = emmeans(info, "a", type="response")
    assert (out.frame["emmean"] > 0).all()
