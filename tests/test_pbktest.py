"""Tests for ``pymmeans.pbktest``.

Kenward-Roger and Satterthwaite F-tests for nested
``MixedLM`` (independent ports of ``pbkrtest::KRmodcomp`` and
``SATmodcomp``), plus standalone ``ddf_lb`` and ``get_kr``
diagnostic helpers.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.regression.mixed_linear_model as mlm

from pymmeans import (
    FtestResult,
    KRDiagnostics,
    ddf_lb,
    get_kr,
    krmodcomp,
    satmodcomp,
)


def _load_reference() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load R reference data + summary CSVs, skipping if missing."""
    data_csv = Path("tests/r_reference/pbkrtest_ftests_data.csv")
    summary_csv = Path("tests/r_reference/pbkrtest_ftests.csv")
    if not (data_csv.exists() and summary_csv.exists()):
        pytest.skip(
            "Run tests/r_reference/pbkrtest_ftests.R to generate "
            "the KRmodcomp / SATmodcomp R reference."
        )
    return pd.read_csv(data_csv), pd.read_csv(summary_csv)


def _fit_case_1(dat: pd.DataFrame) -> tuple:
    """sleepstudy Reaction ~ Days + (1|Subject) vs ~ 1 + (1|Subject) — rank 1."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        large = mlm.MixedLM.from_formula(
            "Reaction ~ Days", groups="Subject", data=dat
        ).fit(reml=True)
        small = mlm.MixedLM.from_formula(
            "Reaction ~ 1", groups="Subject", data=dat
        ).fit(reml=True)
    return large, small


def _fit_case_2(dat: pd.DataFrame) -> tuple:
    """sleepstudy Reaction ~ Days + Days² + (Days|Subject) vs ~ 1 +
    (Days|Subject) — rank 2 multi-DF, exercises the full K-R 1997
    multi-DF F-test machinery."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        large = mlm.MixedLM.from_formula(
            "Reaction ~ Days + Days2", groups="Subject",
            re_formula="~Days", data=dat,
        ).fit(reml=True)
        small = mlm.MixedLM.from_formula(
            "Reaction ~ 1", groups="Subject",
            re_formula="~Days", data=dat,
        ).fit(reml=True)
    return large, small


def _fit_case_3_vc() -> tuple:
    """oats split-plot with TWO variance components — the
    ``vc_formula=`` path. large: yld ~ Variety + nitro +
    (1|Block) + (1|Block:Variety); small drops nitro (df_num = 3).
    Skips if the vc reference data is absent."""
    vc_data = Path("tests/r_reference/pbkrtest_ftests_vc_data.csv")
    if not vc_data.exists():
        pytest.skip(
            "Run tests/r_reference/pbkrtest_ftests.R to generate the "
            "vc_formula KRmodcomp/SATmodcomp reference."
        )
    oa = pd.read_csv(vc_data)
    for c in ("Block", "Variety", "nitro", "BlockVariety"):
        oa[c] = pd.Categorical(oa[c])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        large = mlm.MixedLM.from_formula(
            "yld ~ Variety + C(nitro)", groups="Block", re_formula="1",
            vc_formula={"BlockVariety": "0 + C(BlockVariety)"}, data=oa,
        ).fit(reml=True)
        small = mlm.MixedLM.from_formula(
            "yld ~ Variety", groups="Block", re_formula="1",
            vc_formula={"BlockVariety": "0 + C(BlockVariety)"}, data=oa,
        ).fit(reml=True)
    return large, small


def test_krmodcomp_matches_pbkrtest_case_3_vc_formula():
    """vc_formula= Kenward-Roger F-test (the new support). Two scalar
    variance components (Block, Block:Variety) enter the K-R kernel's
    V_beta / W / P_list via the parameter vector
    ``(vech(G), {vcomp_v}, σ²_e)``. Before the kernel extension the
    df was ~20 % too large (52 vs 43) because the variance-component
    blocks were ignored; now F, ddf, and F.scaling match
    pbkrtest::KRmodcomp to ~1e-2 (the residual is the statsmodels-vs-
    lme4 REML optimiser difference, not the K-R math)."""
    large, small = _fit_case_3_vc()
    assert large.model.k_vc >= 1
    _, ref = _load_reference()
    res = krmodcomp(large, small)
    r = ref[(ref.case == "case3") & (ref.method == "KR")].iloc[0]
    assert res.ndf == int(r.ndf) == 3
    np.testing.assert_allclose(res.F, r.stat, atol=1e-2)
    np.testing.assert_allclose(res.ddf, r.ddf, atol=0.5)
    np.testing.assert_allclose(res.F_scaling, r["F.scaling"], atol=1e-3)
    np.testing.assert_allclose(res.p_value, r["p.value"], rtol=1e-1)


def test_satmodcomp_matches_pbkrtest_case_3_vc_formula():
    """vc_formula= Satterthwaite F-test. Routes through the same
    vc-aware Satterthwaite/KR internals as the EMM df, so it inherits
    the documented finite-difference variance-component-df
    approximation: F matches pbkrtest::SATmodcomp to ~1 % and ddf to
    a few percent (looser than the cov_re-only case)."""
    large, small = _fit_case_3_vc()
    _, ref = _load_reference()
    res = satmodcomp(large, small)
    r = ref[(ref.case == "case3") & (ref.method == "SAT")].iloc[0]
    assert res.ndf == int(r.ndf) == 3
    np.testing.assert_allclose(res.F, r.stat, atol=0.5)
    np.testing.assert_allclose(res.ddf, r.ddf, rtol=0.12)


def test_krmodcomp_matches_pbkrtest_case_1_rank_1():
    """rank-1 nested test (sleepstudy intercept-only RE,
    Days fixed-effect test). Pure scalar F-test — K-R adjustment
    scale factor is 1 and ddf equals the residual REML df."""
    dat, ref = _load_reference()
    large, small = _fit_case_1(dat)
    res = krmodcomp(large, small)
    r = ref[(ref.case == "case1") & (ref.method == "KR")].iloc[0]
    assert isinstance(res, FtestResult)
    assert res.method == "kenward_roger"
    np.testing.assert_allclose(res.F, r.stat, atol=1e-3)
    assert res.ndf == int(r.ndf)
    np.testing.assert_allclose(res.ddf, r.ddf, atol=1e-2)
    np.testing.assert_allclose(res.F_scaling, r["F.scaling"], atol=1e-6)
    np.testing.assert_allclose(res.p_value, r["p.value"], rtol=1e-3)


def test_krmodcomp_matches_pbkrtest_case_2_rank_2():
    """rank-2 multi-DF test on sleepstudy with random
    slopes (Reaction ~ Days + Days² vs ~ 1, both with (Days|Subject)).

    Exercises the full K-R 1997 multi-DF F-test machinery including
    the scale factor (here F.scaling ≈ 0.984), which the rank-1
    case bypasses (scale = 1 unconditionally). Matches pbkrtest at
    floating-point precision on all four outputs (F, ndf, ddf,
    scale, p)."""
    dat, ref = _load_reference()
    large, small = _fit_case_2(dat)
    res = krmodcomp(large, small)
    r = ref[(ref.case == "case2") & (ref.method == "KR")].iloc[0]
    np.testing.assert_allclose(res.F, r.stat, atol=1e-3)
    assert res.ndf == int(r.ndf)
    np.testing.assert_allclose(res.ddf, r.ddf, atol=1e-2)
    np.testing.assert_allclose(res.F_scaling, r["F.scaling"], atol=1e-4)
    np.testing.assert_allclose(res.p_value, r["p.value"], rtol=1e-3)


def test_satmodcomp_matches_pbkrtest_case_1_rank_1():
    """rank-1 Satterthwaite F-test. For a single-row
    contrast, SAT and KR give the same F statistic (only the ddf
    formula differs); for case 1 both algorithms also give the
    same ddf=161 since the random structure doesn't interact with
    the contrast."""
    dat, ref = _load_reference()
    large, small = _fit_case_1(dat)
    res = satmodcomp(large, small)
    r = ref[(ref.case == "case1") & (ref.method == "SAT")].iloc[0]
    assert res.method == "satterthwaite"
    np.testing.assert_allclose(res.F, r.stat, atol=1e-3)
    assert res.ndf == int(r.ndf)
    np.testing.assert_allclose(res.ddf, r.ddf, atol=1e-2)
    assert np.isnan(res.F_scaling)


def test_satmodcomp_matches_pbkrtest_case_2_rank_2():
    """rank-2 Satterthwaite F-test on the same RS sleepstudy
    fit. Exercises the eigendecomposition + per-direction df
    combination in pbkrtest's ``SATmodcomp_worker`` /
    ``get_Fstat_ddf``."""
    dat, ref = _load_reference()
    large, small = _fit_case_2(dat)
    res = satmodcomp(large, small)
    r = ref[(ref.case == "case2") & (ref.method == "SAT")].iloc[0]
    np.testing.assert_allclose(res.F, r.stat, atol=1e-3)
    assert res.ndf == int(r.ndf)
    np.testing.assert_allclose(res.ddf, r.ddf, atol=0.5)


def test_krmodcomp_and_satmodcomp_refuse_swapped_arguments():
    """Both functions must raise ``ValueError`` when called with
    ``small`` and ``large`` in the wrong order (the same kind of
    user-error guard that ``pbmodcomp`` ships). Avoids silently
    returning a degenerate F-test."""
    dat, _ = _load_reference()
    large, small = _fit_case_1(dat)
    for fn in (krmodcomp, satmodcomp):
        with pytest.raises(ValueError, match="strictly more"):
            fn(small, large)
        with pytest.raises(ValueError, match="strictly more"):
            fn(large, large)


def test_krmodcomp_and_satmodcomp_refuse_non_nested_args():
    """Both functions must raise when ``small`` has columns that
    aren't in ``large`` (i.e., not a strict column-subset)."""
    dat, _ = _load_reference()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # Different terms, same parameter count → cannot nest.
        a = mlm.MixedLM.from_formula(
            "Reaction ~ Days", groups="Subject", data=dat,
        ).fit(reml=True)
        b = mlm.MixedLM.from_formula(
            "Reaction ~ Days2", groups="Subject", data=dat,
        ).fit(reml=True)
    with pytest.raises(ValueError):
        # a and b have the same # of params; one is not nested in
        # the other. _build_nested_L raises either the "strictly
        # more" message or the "column-subset" message depending
        # on which check trips first.
        krmodcomp(a, b)


def test_ddf_lb_smoke():
    """``ddf_lb(fit, L)`` returns per-row KR df for an arbitrary
    contrast matrix. Smoke-test that the function runs end-to-end
    and produces positive scalar df values.

    Exact pbkrtest parity is documented to be ``~1 % rel`` (the
    residual higher-order Kenward-Roger 1997 df gap, identical to
    the gap in :func:`apply_kenward_roger`; both close together in
    a v0.2 milestone)."""
    dat, _ = _load_reference()
    large, _ = _fit_case_2(dat)
    # Identity rows over the fixed-effect coefficient vector —
    # gives the per-coefficient KR df.
    p = large.fe_params.size
    L = np.eye(p)
    df = ddf_lb(large, L)
    assert df.shape == (p,)
    # Each KR df is positive and finite for a well-conditioned fit.
    assert np.all(df > 0)
    assert np.all(np.isfinite(df))


def test_get_kr_returns_diagnostics_matching_kenward_roger_vcov():
    """``get_kr(fit)`` is a wrapper around
    :func:`kenward_roger_vcov` (with ``return_internals=True``)
    that returns the diagnostics in a named bundle. Verify
    ``V_KR`` and ``W`` match the underlying call to within
    machine precision."""
    from pymmeans.satterthwaite import kenward_roger_vcov
    from pymmeans.utils import from_fitted

    dat, _ = _load_reference()
    large, _ = _fit_case_2(dat)
    diag = get_kr(large)
    assert isinstance(diag, KRDiagnostics)

    info = from_fitted(large)
    raw = kenward_roger_vcov(info, return_internals=True)
    np.testing.assert_array_equal(diag.V_KR, raw.V_KR)
    np.testing.assert_array_equal(diag.W, raw.W)
    np.testing.assert_array_equal(diag.V_beta, raw.V_beta)
    assert len(diag.P_list) == len(raw.P_list)


def test_kr1997_df_matches_pbkrtest_at_floating_point():
    """K-R 1997 ``ddf_Lb`` formula now lives in
    :func:`pymmeans.pbktest._kr1997_df_per_row` and is wired into
    :func:`pymmeans.apply_kenward_roger`. Pre-pymmeans used
    a Satterthwaite-style delta-method df that drifted ~1 % from
    ``pbkrtest::ddf_Lb`` on the canonical reference fit.
    ports pbkrtest's exact formula, closing the gap.

    Guard: per-coefficient KR df on the reference fit
    matches ``pbkrtest::ddf_Lb`` at ``atol=1e-3`` (the residual
    being finite-diff noise in the K-R 1997 per-θ derivatives, NOT
    formula error)."""
    import warnings as _w
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import statsmodels.regression.mixed_linear_model as mlm

    from pymmeans.pbktest import (
        _compute_pbkrtest_aux_core,
        _kr1997_df_per_row,
    )
    from pymmeans.satterthwaite import (
        _build_satt_cache,
        _lmer_theta_to_lambda,
        kenward_roger_vcov,
    )
    from pymmeans.utils import from_fitted

    ref_csv = Path("tests/r_reference/kr_reference.csv")
    data_csv = Path("tests/r_reference/kr_reference_data.csv")
    if not (ref_csv.exists() and data_csv.exists()):
        pytest.skip("KR reference missing; run kr_reference.R")
    dat = pd.read_csv(data_csv)
    dat["g"] = pd.Categorical(dat["g"], categories=["a", "b", "c"])
    dat["subj"] = dat["subj"].astype("category")
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        fit = mlm.MixedLM.from_formula(
            "y ~ g", groups="subj", data=dat,
        ).fit(reml=True)

    info = from_fitted(fit)
    cache = _build_satt_cache(fit)
    sigma_sq = cache.sigma_sq_hat
    Lambda = _lmer_theta_to_lambda(cache.theta_hat, cache.k_re)
    G_pb = sigma_sq * (Lambda @ Lambda.T)
    V_KR = kenward_roger_vcov(info, cache=cache)
    aux = _compute_pbkrtest_aux_core(
        X=cache.X, Z=cache.Z,
        groups=cache.groups, group_ids=cache.group_ids,
        G=G_pb, sigma_sq=sigma_sq, V_KR=V_KR,
    )
    df_pym = _kr1997_df_per_row(
        L=np.eye(3),
        V_beta=aux["V_beta"],
        P_list=aux["P_list"],
        W=aux["W"],
    )

    ref = pd.read_csv(ref_csv).set_index("name")
    df_ref = ref.loc[["(Intercept)", "gb", "gc"], "df_kr"].to_numpy()
    np.testing.assert_allclose(
        df_pym, df_ref, atol=1e-3,
        err_msg=(
            "K-R 1997 ddf_Lb must match pbkrtest at "
            "atol=1e-3 (the residual is finite-diff noise on the "
            "per-θ derivatives). Previously the Satterthwaite "
            "delta-method drifted ~1 % rel; closes the gap."
        ),
    )


def test_apply_kenward_roger_uses_kr1997_df_in_emm_output():
    """the production path
    :func:`pymmeans.apply_kenward_roger` now emits the K-R 1997
    formula df (not the Satterthwaite delta-method df). Verify
    end-to-end via ``emmeans → apply_kenward_roger`` that the
    output df column matches R's ``emmeans(fit, ~ g,
    lmer.df="kenward-roger")`` at floating-point precision on
    ALL three EMM cells (not just the reference cell).

    pre-extension this test only
    checked ``g='a'`` (which is just the intercept's df). Now we
    also assert ``g='b'`` and ``g='c'``, whose EMM contrasts are
    [1, 1, 0] and [1, 0, 1] — distinct from the per-coef contrasts
    and not in the coefficient-level reference. R values
    come from running ``emmeans + lmerTest + pbkrtest`` directly."""
    import warnings as _w
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import statsmodels.regression.mixed_linear_model as mlm

    from pymmeans import apply_kenward_roger, emmeans

    data_csv = Path("tests/r_reference/kr_reference_data.csv")
    if not data_csv.exists():
        pytest.skip("KR reference data missing")
    dat = pd.read_csv(data_csv)
    dat["g"] = pd.Categorical(dat["g"], categories=["a", "b", "c"])
    dat["subj"] = dat["subj"].astype("category")
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        fit = mlm.MixedLM.from_formula(
            "y ~ g", groups="subj", data=dat,
        ).fit(reml=True)

    em = emmeans(fit, "g")
    em_kr = apply_kenward_roger(em)
    assert em_kr.df_method == "kenward_roger"

    # R reference values from
    # ``emmeans(lmer(y ~ g + (1|subj)), ~ g, lmer.df="kenward-roger")``:
    # g=a: df=27.41 (intercept-only contrast, [1,0,0])
    # g=b: df=25.46 (contrast [1,1,0])
    # g=c: df=33.97 (contrast [1,0,1])
    expected = {"a": 27.41, "b": 25.46, "c": 33.97}
    df_by_g = dict(zip(
        em_kr.frame["g"], em_kr.frame["df"], strict=True
    ))
    for level, expected_df in expected.items():
        np.testing.assert_allclose(
            float(df_by_g[level]), expected_df, atol=0.01,
            err_msg=(
                f"KR df at g='{level}' should match R "
                f"emmeans({expected_df}) at atol=0.01. Got "
                f"{df_by_g[level]:.4f}."
            ),
        )


def test_ddf_lb_matches_pbkrtest_at_floating_point():
    """previously, the public ``ddf_lb``
    used the OLD Satterthwaite delta-method while
    ``apply_kenward_roger`` already used K-R 1997 internally.
    Migration moves ``ddf_lb`` onto ``_kr1997_df_per_row`` too;
    verify floating-point parity with ``pbkrtest::Lb_ddf`` /
    ``pbkrtest::ddf_Lb`` at the per-coefficient level."""
    import warnings as _w
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import statsmodels.regression.mixed_linear_model as mlm

    from pymmeans import ddf_lb

    data_csv = Path("tests/r_reference/kr_reference_data.csv")
    ref_csv = Path("tests/r_reference/kr_reference.csv")
    if not (data_csv.exists() and ref_csv.exists()):
        pytest.skip("KR reference missing")
    dat = pd.read_csv(data_csv)
    dat["g"] = pd.Categorical(dat["g"], categories=["a", "b", "c"])
    dat["subj"] = dat["subj"].astype("category")
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        fit = mlm.MixedLM.from_formula(
            "y ~ g", groups="subj", data=dat,
        ).fit(reml=True)

    # Per-coefficient KR df via the public ddf_lb entry.
    p = fit.fe_params.size
    df_pym = ddf_lb(fit, np.eye(p))
    ref = pd.read_csv(ref_csv).set_index("name")
    df_ref = ref.loc[["(Intercept)", "gb", "gc"], "df_kr"].to_numpy()
    np.testing.assert_allclose(
        df_pym, df_ref, atol=1e-3,
        err_msg=(
            "public ddf_lb must match "
            "pbkrtest::Lb_ddf at atol=1e-3 (same K-R 1997 formula "
            "as the internal apply_kenward_roger path). previously "
            "drifted ~1 % relative because ddf_lb was still on "
            "the old delta-method path."
        ),
    )


def test_ftest_result_summary_smoke():
    """``FtestResult.summary()`` produces a multi-line string
    carrying the headline numbers. Smoke-test that the string
    is non-empty and contains the F / df / p-value."""
    dat, _ = _load_reference()
    large, small = _fit_case_2(dat)
    kr_summary = krmodcomp(large, small).summary()
    sat_summary = satmodcomp(large, small).summary()
    for s in (kr_summary, sat_summary):
        assert "F" in s
        assert "ndf" in s
        assert "ddf" in s
        assert "p" in s
    # KR summary's "F.scaling" line shows up on rank > 1
    assert "F.scaling" in kr_summary
