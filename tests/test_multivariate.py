"""Tests for ``pymmeans.multivariate`` — multivariate-OLS EMMs +
``mvcontrast``.

Cross-validates per-cell × per-response EMMs and the Hotelling-T² /
F multivariate test against R `emmeans` + `mvcontrast` on a seeded
3-response × 3-group fixture. Reference values come from
``tests/r_reference/multivariate_reference.R`` (run that script to
regenerate the CSVs).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pymmeans import (
    MultivariateEMM,
    MultivariateInfo,
    from_multivariate,
    multivariate_emmeans,
    mvcontrast,
)


def _load_reference() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = Path("tests/r_reference/multivariate_data.csv")
    emm = Path("tests/r_reference/multivariate_emm.csv")
    mvc = Path("tests/r_reference/multivariate_mvc.csv")
    if not (data.exists() and emm.exists() and mvc.exists()):
        pytest.skip(
            "Run tests/r_reference/multivariate_reference.R to "
            "generate the multivariate EMM + mvcontrast references."
        )
    return pd.read_csv(data), pd.read_csv(emm), pd.read_csv(mvc)


def _fit_mv(data: pd.DataFrame):
    from statsmodels.multivariate.multivariate_ols import _MultivariateOLS
    data = data.copy()
    data["g"] = pd.Categorical(data["g"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _MultivariateOLS.from_formula(
            "y1 + y2 + y3 ~ g + x", data=data
        ).fit(), data


def test_from_multivariate_extracts_B_inv_cov_sigma():
    """``from_multivariate`` must pull B (k×p), (X'X)⁻¹ (k×k), Σ̂ (p×p),
    and df_resid out of statsmodels' private ``_fittedmod`` tuple
    correctly, in absolute units. Σ̂ = sscpr / df_resid."""
    data, _, _ = _load_reference()
    fit, _ = _fit_mv(data)
    info = from_multivariate(fit)
    assert isinstance(info, MultivariateInfo)
    p = len(info.endog_names); k = len(info.exog_names)
    assert info.B.shape == (k, p)
    assert info.inv_cov.shape == (k, k)
    assert info.Sigma_hat.shape == (p, p)
    assert info.df_resid == fit._fittedmod[1]
    # Σ̂ must equal sscpr / df_resid (the construction the rest of the
    # module assumes).
    np.testing.assert_allclose(
        info.Sigma_hat * info.df_resid, fit._fittedmod[3], atol=1e-15
    )


def test_multivariate_emmeans_matches_r_per_cell():
    """Per-cell × per-response EMMs (emmean, SE, df, CI) must match
    ``emmeans(mlm_fit, ~ g | rep.meas)`` to machine precision —
    deterministic linear algebra, no solver/REML/QMC noise."""
    data, emm_r, _ = _load_reference()
    fit, data = _fit_mv(data)
    em = multivariate_emmeans(fit, data, "g")
    assert isinstance(em, MultivariateEMM)
    pm = em.frame.assign(
        g=lambda d: d["g"].astype(str),
        rep_meas=lambda d: d["rep_meas"].astype(str),
    ).sort_values(["g", "rep_meas"]).reset_index(drop=True)
    rr = emm_r.assign(
        g=lambda d: d["g"].astype(str),
        rep_meas=lambda d: d["rep_meas"].astype(str),
    ).sort_values(["g", "rep_meas"]).reset_index(drop=True)
    np.testing.assert_allclose(pm["emmean"], rr["emmean"], atol=1e-12)
    np.testing.assert_allclose(pm["SE"], rr["SE"], atol=1e-12)
    np.testing.assert_array_equal(pm["df"].to_numpy(), rr["df"].to_numpy())
    np.testing.assert_allclose(pm["lower_cl"], rr["lower_cl"], atol=1e-12)
    np.testing.assert_allclose(pm["upper_cl"], rr["upper_cl"], atol=1e-12)


def test_mvcontrast_matches_r_hotelling_pairwise():
    """Hotelling T² / F-ratio / df / Sidak-adjusted p-value per
    between-contrast must match ``mvcontrast(emm, "pairwise",
    mult.name="rep.meas")`` to machine precision."""
    data, _, mvc_r = _load_reference()
    fit, data = _fit_mv(data)
    em = multivariate_emmeans(fit, data, "g")
    pm = mvcontrast(em, "pairwise").assign(
        contrast=lambda d: d["contrast"].astype(str)
    ).sort_values("contrast").reset_index(drop=True)
    rr = mvc_r.assign(
        contrast=lambda d: d["contrast"].astype(str)
    ).sort_values("contrast").reset_index(drop=True)
    np.testing.assert_array_equal(pm["df1"].to_numpy(), rr["df1"].to_numpy())
    np.testing.assert_array_equal(pm["df2"].to_numpy(), rr["df2"].to_numpy())
    np.testing.assert_allclose(pm["T_square"], rr["T_square"], atol=1e-11)
    np.testing.assert_allclose(pm["F_ratio"], rr["F_ratio"], atol=1e-12)
    np.testing.assert_allclose(pm["p_value"], rr["p_value"], atol=1e-13)


def test_mvcontrast_bonferroni_and_none_match_formula():
    """The non-default adjustments must apply the standard formulas
    over the family of between-contrasts (3 for pairwise on 3 cells)."""
    data, _, _ = _load_reference()
    fit, data = _fit_mv(data)
    em = multivariate_emmeans(fit, data, "g")
    raw = mvcontrast(em, "pairwise", adjust="none")
    bon = mvcontrast(em, "pairwise", adjust="bonferroni")
    sid = mvcontrast(em, "pairwise", adjust="sidak")
    k = len(raw)
    np.testing.assert_allclose(
        bon["p_value"], np.minimum(1.0, raw["p_value"] * k), atol=1e-15
    )
    np.testing.assert_allclose(
        sid["p_value"], 1.0 - (1.0 - raw["p_value"]) ** k, atol=1e-15
    )


def test_mvcontrast_trt_vs_ctrl():
    """``trt.vs.ctrl`` family must produce (k − 1) contrasts each vs the
    first cell. Sanity: the first contrast equals "b - a" given level
    order a/b/c, and the T² statistics are positive."""
    data, _, _ = _load_reference()
    fit, data = _fit_mv(data)
    em = multivariate_emmeans(fit, data, "g")
    out = mvcontrast(em, "trt.vs.ctrl", adjust="none")
    assert len(out) == 2
    assert list(out["contrast"]) == ["b - a", "c - a"]
    assert np.all(out["T_square"] > 0)
    assert np.all((out["p_value"] >= 0) & (out["p_value"] <= 1))


def test_from_multivariate_refuses_non_multivariate_result():
    """Guard against a user accidentally passing a univariate OLS fit
    to ``from_multivariate`` — they must get a clear TypeError rather
    than an AttributeError on ``_fittedmod``."""
    import statsmodels.formula.api as smf
    df_ = pd.DataFrame({
        "g": pd.Categorical(["a", "b", "a", "b"] * 5),
        "y": np.arange(20.0),
    })
    ols = smf.ols("y ~ g", df_).fit()
    with pytest.raises(TypeError, match="_MultivariateOLSResults"):
        from_multivariate(ols)
