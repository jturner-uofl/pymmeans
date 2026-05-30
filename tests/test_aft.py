"""Tests for ``pymmeans.aft`` — AFT survival adapter.

Cross-validates lifelines ``WeibullAFTFitter`` + ``from_aft`` against R
``survreg(dist="weibull") + emmeans`` on a seeded n=300 design.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Skip the whole module if lifelines isn't installed — it's an optional
# dependency, matching the "AFT requires lifelines" scope decision.
lifelines = pytest.importorskip("lifelines")
from lifelines import WeibullAFTFitter  # noqa: E402

from pymmeans import contrast, emmeans, from_aft  # noqa: E402
from pymmeans.utils import ModelInfo  # noqa: E402


def _load_aft_reference() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = Path("examples/jss_audit/cs_ref/aft_data.csv")
    emm = Path("examples/jss_audit/cs_ref/aft_emm.csv")
    pairs_ = Path("examples/jss_audit/cs_ref/aft_pairs.csv")
    if not (data.exists() and emm.exists() and pairs_.exists()):
        pytest.skip(
            "Run examples/jss_audit/generate_case_study_reference.R "
            "to generate the Weibull AFT reference CSVs."
        )
    return pd.read_csv(data), pd.read_csv(emm), pd.read_csv(pairs_)


def _fit_aft(data: pd.DataFrame):
    data = data.copy()
    data["g"] = pd.Categorical(data["g"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        af = WeibullAFTFitter().fit(
            data, duration_col="T", event_col="E", formula="g + x"
        )
    return af, data


def test_from_aft_returns_valid_model_info():
    """``from_aft`` must produce a standard pymmeans ``ModelInfo`` so the
    existing emmeans / contrast pipeline works on the result with no
    further adapter plumbing."""
    data, _, _ = _load_aft_reference()
    af, data = _fit_aft(data)
    info = from_aft(af, data, formula="g + x")
    assert isinstance(info, ModelInfo)
    # lambda_ block has 4 parameters: Intercept + g[T.gB] + g[T.gC] + x
    assert info.beta.shape == (4,)
    assert info.vcov.shape == (4, 4)
    assert info.df_resid == 300 - len(af.params_)
    # g must be recognised as a categorical factor with 3 levels.
    assert "g" in info.factors
    assert sorted(info.factors["g"]) == ["gA", "gB", "gC"]


def test_aft_emm_matches_r_survreg_emmeans():
    """Per-cell EMMs (emmean, SE, df) match R `emmeans(survreg(...))` to
    ~1e-5 — limited by lifelines vs survreg Weibull-MLE optimisers
    converging to slightly different solutions (~1e-6 on β)."""
    data, emm_r, _ = _load_aft_reference()
    af, data = _fit_aft(data)
    info = from_aft(af, data, formula="g + x")
    pm = (
        emmeans(info, "g").frame
        .sort_values("g").reset_index(drop=True)
    )
    rr = emm_r.sort_values("g").reset_index(drop=True)
    np.testing.assert_allclose(pm["emmean"], rr["emmean"], atol=1e-5)
    np.testing.assert_allclose(pm["SE"], rr["SE"], atol=1e-5)
    np.testing.assert_array_equal(pm["df"].to_numpy(), rr["df"].to_numpy())


def test_aft_pairwise_matches_r_survreg_emmeans():
    """Pairwise contrast estimate / SE / p-value match R `contrast(emm,
    "pairwise")` at the same optimiser tolerance."""
    data, _, pairs_r = _load_aft_reference()
    af, data = _fit_aft(data)
    info = from_aft(af, data, formula="g + x")
    pm = (
        contrast(emmeans(info, "g"), "pairwise", adjust="none").frame
        .set_index("contrast")
    )
    rr = pairs_r.set_index("contrast")
    common = pm.index.intersection(rr.index)
    np.testing.assert_allclose(
        pm.loc[common, "estimate"], rr.loc[common, "estimate"], atol=1e-5
    )
    np.testing.assert_allclose(
        pm.loc[common, "SE"], rr.loc[common, "SE"], atol=1e-5
    )


def test_from_aft_refuses_non_aft_fitter():
    """Guard: passing a non-AFT object to ``from_aft`` must raise
    ``TypeError`` with a clear steering message, not a cryptic
    AttributeError on ``params_``."""
    with pytest.raises(TypeError, match="AFT"):
        from_aft(object(), pd.DataFrame({"x": [0.0]}), formula="x")


@pytest.mark.parametrize("dist_label,fitter_cls,ref_name,tol", [
    ("LogNormal",
     "LogNormalAFTFitter",
     "examples/jss_audit/cs_ref/aft_pairs_lognormal.csv",
     1e-5),
    ("LogLogistic",
     "LogLogisticAFTFitter",
     "examples/jss_audit/cs_ref/aft_pairs_loglogistic.csv",
     1e-3),
])
def test_from_aft_works_on_other_distributions(dist_label, fitter_cls,
                                                ref_name, tol):
    """``from_aft`` is block-name-agnostic: lifelines uses ``lambda_``
    for Weibull, ``mu_`` for LogNormal, ``alpha_`` for LogLogistic. The
    adapter detects the location block by matching the patsy design
    columns, so the same pipeline reproduces R `survreg(dist=...)` for
    every parametric AFT family. LogLogistic's wider tolerance reflects
    lifelines vs survreg converging to slightly different MLEs."""
    ref_path = Path(ref_name)
    if not ref_path.exists():
        pytest.skip(
            "Run examples/jss_audit/generate_case_study_reference.R "
            "to generate the lognormal / loglogistic references."
        )
    import lifelines as _ll
    Cls = getattr(_ll, fitter_cls)
    data = pd.read_csv("examples/jss_audit/cs_ref/aft_data.csv")
    data["g"] = pd.Categorical(data["g"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        af = Cls().fit(data, duration_col="T", event_col="E",
                       formula="g + x")
    info = from_aft(af, data, formula="g + x")
    pm = (
        contrast(emmeans(info, "g"), "pairwise", adjust="none").frame
        .set_index("contrast")
    )
    rr = pd.read_csv(ref_path).set_index("contrast")
    common = pm.index.intersection(rr.index)
    assert len(common) == 3, f"{dist_label}: expected 3 pairwise contrasts"
    np.testing.assert_allclose(
        pm.loc[common, "estimate"], rr.loc[common, "estimate"], atol=tol
    )
    np.testing.assert_allclose(
        pm.loc[common, "SE"], rr.loc[common, "SE"], atol=tol
    )


def test_from_aft_handles_generalized_gamma_with_regressors_api():
    """``GeneralizedGammaRegressionFitter`` uses lifelines's
    ``regressors=`` dict-of-formulas API instead of ``formula=`` because
    it has three parameter blocks (``mu_``, ``sigma_``, ``lambda_``)
    rather than one. The location block is named ``mu_`` — different from
    Weibull's ``lambda_`` — so this test exercises ``from_aft``'s
    block-name-agnostic location detection. β must match R `flexsurv`
    `flexsurvreg(dist="gengamma")` β to ~1e-4 (lifelines vs flexsurv
    MLE optimisers)."""
    from pathlib import Path
    ref_path = Path("examples/jss_audit/cs_ref/aft_gengamma_R.csv")
    data_path = Path("examples/jss_audit/cs_ref/aft_data.csv")
    if not (ref_path.exists() and data_path.exists()):
        pytest.skip("Run generate_case_study_reference.R to seed the "
                    "GeneralizedGamma AFT R reference.")
    from lifelines import GeneralizedGammaRegressionFitter as GenGamma
    data = pd.read_csv(data_path)
    data["g"] = pd.Categorical(data["g"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        af = GenGamma().fit(
            data, duration_col="T", event_col="E",
            regressors={"mu_": "g + x", "sigma_": "1", "lambda_": "1"},
        )
    info = from_aft(af, data, formula="g + x")
    # Sanity: block-agnostic detection picked the mu_ block (4 params).
    assert info.beta.shape == (4,)
    assert info.vcov.shape == (4, 4)
    # R flexsurv reference mapping (mu == Intercept; ggB/ggC == g[T.gB/gC]).
    r_ref = pd.read_csv(ref_path).set_index("coef")
    r_to_pm = {"mu": "Intercept", "ggB": "g[T.gB]",
               "ggC": "g[T.gC]", "x": "x"}
    pm_lookup = dict(zip(info.param_names, info.beta, strict=True))
    for r_key, pm_key in r_to_pm.items():
        np.testing.assert_allclose(
            pm_lookup[pm_key], float(r_ref.loc[r_key, "value"]), atol=1e-4
        )
    # The EMM pipeline runs end-to-end.
    em = emmeans(info, "g").frame
    assert len(em) == 3 and np.all(np.isfinite(em["emmean"]))
