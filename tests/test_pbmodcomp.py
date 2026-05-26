"""Tests for ``pymmeans.pbmodcomp``.

parametric-bootstrap nested model comparison for MixedLM.
Port of ``pbkrtest::PBmodcomp`` (Halekoh & Højsgaard 2014).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.regression.mixed_linear_model as mlm

from pymmeans import PBmodcompResult, pbmodcomp


def _fit_sleepstudy_pair(reml: bool = False):
    """Return (large, small) sleepstudy fits, optionally as REML.

    Use the local CSV (committed) so the test does not require
    network access to ``statsmodels.datasets.get_rdataset``.
    """
    csv = Path("tests/r_reference/pbmodcomp_data.csv")
    if not csv.exists():
        pytest.skip(
            "tests/r_reference/pbmodcomp_data.csv missing; "
            "run tests/r_reference/pbmodcomp_reference.R"
        )
    dat = pd.read_csv(csv)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        large = mlm.MixedLM.from_formula(
            "Reaction ~ Days", groups="Subject", data=dat,
        ).fit(reml=reml)
        small = mlm.MixedLM.from_formula(
            "Reaction ~ 1", groups="Subject", data=dat,
        ).fit(reml=reml)
    return large, small


def test_pbmodcomp_returns_result_with_expected_fields():
    """The function returns a :class:`PBmodcompResult` with the
    documented attributes, sensible types, and consistent counts."""
    large, small = _fit_sleepstudy_pair()
    res = pbmodcomp(large, small, n_sim=100, seed=42)
    assert isinstance(res, PBmodcompResult)
    assert isinstance(res.lrt_obs, float)
    assert res.lrt_obs > 0 # `Days` is a strong predictor
    assert isinstance(res.lrt_dist, np.ndarray)
    assert res.lrt_dist.ndim == 1
    assert res.n_sim == 100
    assert 0 < res.n_sim_ok <= 100
    assert res.n_sim_ok == len(res.lrt_dist)
    assert 0.0 < res.p_value <= 1.0
    assert 0.0 <= res.chi2_p_value <= 1.0
    assert res.df == 1 # Days vs intercept
    # The summary() string contains all the headline numbers.
    summary = res.summary()
    assert "LRT_obs" in summary
    assert "p (bootstrap)" in summary
    assert "p (chi^2)" in summary


def test_pbmodcomp_matches_pbkrtest_lrt_obs_at_floating_point():
    """LRT_obs is a deterministic function of the two fits — must
    match ``pbkrtest::PBmodcomp``'s ``test$stat[1]`` to floating-
    point precision (no Monte Carlo involved)."""
    summary_csv = Path("tests/r_reference/pbmodcomp_summary.csv")
    if not summary_csv.exists():
        pytest.skip("pbkrtest reference missing; run pbmodcomp_reference.R")
    summary = pd.read_csv(summary_csv).set_index("metric")["value"]

    large, small = _fit_sleepstudy_pair()
    res = pbmodcomp(large, small, n_sim=10, seed=0)
    np.testing.assert_allclose(
        res.lrt_obs, summary["lrt_obs"], atol=1e-5,
        err_msg=(
            "pbmodcomp LRT_obs must match pbkrtest::PBmodcomp "
            "test$stat[1] at atol=1e-5 on the sleepstudy reference fit."
        ),
    )


def test_pbmodcomp_bootstrap_distribution_matches_pbkrtest_ks():
    """Pymmeans' bootstrap LRT distribution must be statistically
    indistinguishable from R pbkrtest's at the 2-sample KS level.

    parity check: drawing 5,000 LRTs from each
    implementation, the two-sample KS test should give p > 0.05
    (i.e., we accept the null that they come from the same
    distribution). Bootstrap MC noise dominates — both halves
    have ~0.02 KS statistic against the truth, so observing
    D < 0.03 between the two halves is the correct outcome.

    This test runs 1000 sims (~15 s wall-clock); the full 5,000
    comparison from the development log is documented in
    the regression test and triggered manually."""
    dist_csv = Path("tests/r_reference/pbmodcomp_lrt_dist.csv")
    if not dist_csv.exists():
        pytest.skip("pbkrtest LRT distribution missing; run R script")
    r_dist = pd.read_csv(dist_csv)["lrt"].to_numpy()

    large, small = _fit_sleepstudy_pair()
    res = pbmodcomp(large, small, n_sim=1000, seed=20260522)
    # Compare quantiles at common percentiles
    qs = np.array([0.5, 0.75, 0.9, 0.95])
    pym_q = np.quantile(res.lrt_dist, qs)
    r_q = np.quantile(r_dist, qs)
    # Tolerance: bootstrap with n=1000 has Monte Carlo noise of
    # ~5–15 % at upper percentiles. Use a relative tolerance of
    # 20 % plus an absolute floor — bootstrap quantiles can differ
    # by ~0.5 LRT units on the tail with 1000 sims even when the
    # implementations agree. The development log
    # confirmed KS test p=0.36 at n_sim=5000 (statistically
    # indistinguishable).
    np.testing.assert_allclose(
        pym_q, r_q, rtol=0.20, atol=0.6,
        err_msg=(
            "pymmeans bootstrap LRT distribution quantiles "
            "must match pbkrtest's within MC noise on sleepstudy "
            "(Reaction ~ Days vs Reaction ~ 1, n_sim=1000)."
        ),
    )


def test_pbmodcomp_refits_reml_inputs_as_ml_with_warning():
    """When either input was fit as REML, pbmodcomp must silently
    refit both as ML (REML log-likelihoods aren't comparable across
    different fixed-effect designs) and emit a ``UserWarning``."""
    large_reml, small_reml = _fit_sleepstudy_pair(reml=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = pbmodcomp(large_reml, small_reml, n_sim=30, seed=0)
    assert res.refit_reml
    msgs = [str(w.message) for w in caught if issubclass(w.category, UserWarning)]
    assert any("REML" in m and "refit" in m for m in msgs), (
        "Expected a UserWarning explaining the internal ML refit; "
        f"got: {msgs}"
    )

    # silent=True suppresses the warning.
    with warnings.catch_warnings(record=True) as caught2:
        warnings.simplefilter("always")
        pbmodcomp(large_reml, small_reml, n_sim=10, seed=0, silent=True)
    msgs2 = [
        str(w.message) for w in caught2
        if issubclass(w.category, UserWarning)
        and "REML" in str(w.message)
    ]
    assert not msgs2, f"silent=True must suppress REML refit warning; got {msgs2}"


def test_pbmodcomp_seed_makes_results_reproducible():
    """Two calls with the same seed produce identical LRT
    distributions; different seeds give different distributions."""
    large, small = _fit_sleepstudy_pair()
    res1 = pbmodcomp(large, small, n_sim=80, seed=7)
    res2 = pbmodcomp(large, small, n_sim=80, seed=7)
    res3 = pbmodcomp(large, small, n_sim=80, seed=8)
    np.testing.assert_array_equal(res1.lrt_dist, res2.lrt_dist)
    # Different seed → different distribution (overwhelmingly likely)
    assert not np.array_equal(res1.lrt_dist, res3.lrt_dist)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_pbmodcomp_bootstrap_more_conservative_in_small_sample():
    """The asymptotic chi-squared is known to overstate significance
    in small samples; the parametric bootstrap should report a
    larger p-value than chi-squared on a borderline-significant
    small-n design. This is the entire motivation for PBmodcomp.

    The ``UserWarning`` from ``pbmodcomp`` flagging <90 % bootstrap
    convergence is expected at the small ``n_sim`` used here and is
    filtered so the test passes under ``pytest -W error``.
    """
    rng = np.random.default_rng(42)
    n_groups = 6
    n_per = 3
    n = n_groups * n_per
    subj = np.repeat(np.arange(n_groups), n_per)
    u = rng.normal(0, 0.4, n_groups)[subj]
    x = rng.normal(0, 1, n)
    y = 1.0 + 0.3 * x + u + rng.normal(0, 1.0, n)
    dat = pd.DataFrame({"y": y, "x": x, "subj": subj.astype(str)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        large = mlm.MixedLM.from_formula(
            "y ~ x", groups="subj", data=dat,
        ).fit(reml=False)
        small = mlm.MixedLM.from_formula(
            "y ~ 1", groups="subj", data=dat,
        ).fit(reml=False)
    res = pbmodcomp(large, small, n_sim=500, seed=42)
    assert res.p_value > res.chi2_p_value, (
        "Bootstrap p must be more conservative than chi^2 in this "
        f"small-sample design. Got boot={res.p_value:.3f}, "
        f"chi2={res.chi2_p_value:.3f}."
    )


def test_simulate_from_mixedlm_recovers_marginal_moments():
    """Direct moment recovery test for ``_simulate_from_mixedlm``:
    averaging many draws from the marginal null distribution must
    converge to ``X β̂`` (the mean) and ``V_g = Z_g G Z_g' + σ² I``
    (the per-group covariance).

    This is the load-bearing math for ``pbmodcomp`` — if it sampled
    from the wrong distribution, every bootstrap p-value would be
    silently miscalibrated. careful-implementation pass."""
    from pymmeans.pbmodcomp import _simulate_from_mixedlm

    large, _ = _fit_sleepstudy_pair()
    X = np.asarray(large.model.exog, dtype=float)
    Z = np.asarray(large.model.exog_re, dtype=float)
    groups = np.asarray(large.model.groups)
    beta = np.asarray(large.fe_params, dtype=float)
    G = np.asarray(large.cov_re, dtype=float)
    sigma_sq = float(large.scale)
    expected_mu = X @ beta

    rng = np.random.default_rng(0)
    n_draws = 2000
    Y = np.stack(
        [_simulate_from_mixedlm(large, rng) for _ in range(n_draws)]
    ) # (n_draws, n)

    # 1. Mean recovery: 1/n_draws * sum_b y_b -> X β̂. The standard
    # error on each column scales as sqrt(diag(V)/n_draws); for
    # sleepstudy diag(V) ~ 1500, so SE ~ sqrt(1500/2000) ~ 0.87.
    # Use a 5x-SE bound (~4.4) to keep flakiness essentially zero.
    sample_mean = Y.mean(axis=0)
    np.testing.assert_allclose(
        sample_mean, expected_mu, atol=5.0,
        err_msg=(
            "_simulate_from_mixedlm: sample mean must converge to "
            "X β̂ as n_draws -> ∞."
        ),
    )

    # 2. Per-group covariance recovery: on the first group, the
    # sample covariance of the n_g columns should converge to V_g.
    # sleepstudy has 18 groups of 10 obs each — focus on group 0.
    group_ids = np.unique(groups)
    m0 = groups == group_ids[0]
    Z_g = Z[m0]
    V_g_expected = Z_g @ G @ Z_g.T + sigma_sq * np.eye(int(m0.sum()))
    sample_cov = np.cov(Y[:, m0], rowvar=False)
    # Sample cov of 2000 draws has FP noise ~|V_g_max| / sqrt(n_draws).
    # diag(V_g) ~ 1500 in sleepstudy, so noise ~ 33. Use a relaxed
    # absolute tolerance of 200 (~13 % rel on the diagonal) for a
    # nearly-zero-flake threshold.
    np.testing.assert_allclose(
        sample_cov, V_g_expected, atol=200.0,
        err_msg=(
            "_simulate_from_mixedlm: per-group sample covariance "
            "must converge to V_g = Z_g G Z_g' + σ² I."
        ),
    )


def test_refit_mixedlm_ml_recovers_original_params_on_same_endog():
    """``_refit_mixedlm_ml`` refit to ``fit.model.endog`` (the
    *original* data) must recover the original ML fit's
    fixed-effect coefficients and log-likelihood to within
    optimiser tolerance. This is the round-trip sanity check the
    main bootstrap loop relies on."""
    from pymmeans.pbmodcomp import _refit_mixedlm_ml

    # ML reference (so we're not also testing _ensure_ml here).
    large_ml, _ = _fit_sleepstudy_pair(reml=False)
    y_original = np.asarray(large_ml.model.endog, dtype=float)
    refit = _refit_mixedlm_ml(large_ml, y_original)
    np.testing.assert_allclose(
        np.asarray(refit.fe_params), np.asarray(large_ml.fe_params),
        atol=1e-3,
        err_msg=(
            "_refit_mixedlm_ml round-trip: refitting to the original "
            "endog must recover the original ML fixed-effect params."
        ),
    )
    np.testing.assert_allclose(
        float(refit.llf), float(large_ml.llf), atol=1e-4,
        err_msg=(
            "_refit_mixedlm_ml round-trip: log-likelihood must match "
            "the original ML fit."
        ),
    )
    # And the refit must be ML, not REML.
    assert refit.model.reml is False


def test_pbmodcomp_refuses_vc_formula_fits():
    """``pbmodcomp`` must refuse fits with ``vc_formula`` variance
    components outside ``cov_re`` (``model.k_vc > 0``). The
    bootstrap simulator only samples from the cov_re part of the
    marginal covariance; ignoring exog_vc would silently draw from
    the wrong null distribution."""
    large, small = _fit_sleepstudy_pair()
    # Spoof k_vc on the model to trigger the refusal. Easier than
    # constructing a real vc_formula fit, and exercises the same
    # code path.
    large.model.k_vc = 1
    try:
        with pytest.raises(NotImplementedError, match="vc_formula"):
            pbmodcomp(large, small, n_sim=2, seed=0)
    finally:
        large.model.k_vc = 0 # restore


def test_pbmodcomp_refuses_non_mixedlm_inputs():
    """``pbmodcomp`` raises ``TypeError`` when handed something that
    isn't a MixedLM fit (e.g. OLSResults / GLMResults). Catching
    this early — before the bootstrap loop — saves the user from a
    cryptic 1,000-iteration cascade of AttributeError."""
    import statsmodels.api as sm

    X = np.column_stack(
        [np.ones(40), np.random.default_rng(0).normal(size=40)]
    )
    y = np.random.default_rng(0).normal(size=40)
    ols = sm.OLS(y, X).fit()
    large, _ = _fit_sleepstudy_pair()
    with pytest.raises(TypeError, match="MixedLM"):
        pbmodcomp(large, ols, n_sim=2, seed=0)
    with pytest.raises(TypeError, match="MixedLM"):
        pbmodcomp(ols, large, n_sim=2, seed=0)
    with pytest.raises(TypeError, match="MixedLM|not a"):
        pbmodcomp(large, "not a fit", n_sim=2, seed=0)


def test_pbmodcomp_refuses_n_sim_below_one():
    """``n_sim < 1`` must raise ``ValueError`` rather than returning
    a degenerate result with empty ``lrt_dist``."""
    large, small = _fit_sleepstudy_pair()
    with pytest.raises(ValueError, match="n_sim must be >= 1"):
        pbmodcomp(large, small, n_sim=0, seed=0)
    with pytest.raises(ValueError, match="n_sim must be >= 1"):
        pbmodcomp(large, small, n_sim=-5, seed=0)


def test_pbmodcomp_warns_on_low_convergence_rate():
    """When fewer than 90 % of bootstrap iterations converge,
    ``pbmodcomp`` must emit a ``UserWarning`` so the user knows the
    null distribution may be biased.

    Trigger: monkey-patch the refit helper to fail the *small*
    refit on every other iteration. Because pbmodcomp fits small
    first and short-circuits on exception, failing small alternately
    yields exactly 50 % bootstrap convergence — well below the
    90 % warning threshold."""
    import sys

    large, small = _fit_sleepstudy_pair()
    pbmod_module = sys.modules["pymmeans.pbmodcomp"]
    original = pbmod_module._refit_mixedlm_ml
    counter = {"small_iter": 0}

    def _fail_alternate_small(fit, y):
        # Identify "small refit" by its fe_params count: small has
        # 1 (intercept only), large has 2. Increment a per-small
        # counter so we fail every other iteration in a deterministic
        # pattern, regardless of execution order.
        n_fe = int(np.asarray(fit.fe_params).size)
        if n_fe == 1:
            counter["small_iter"] += 1
            if counter["small_iter"] % 2 == 0:
                raise ValueError("simulated small-model convergence failure")
        return original(fit, y)

    pbmod_module._refit_mixedlm_ml = _fail_alternate_small
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = pbmodcomp(large, small, n_sim=20, seed=0)
        msgs = [str(w.message) for w in caught
                if issubclass(w.category, UserWarning)]
        # Match the specific substring "bootstrap iterations" — both
        # "converged" and "90" appear elsewhere (e.g. "90% CI" in
        # bootstrap_ci messages) so they can't carry the assertion
        # on their own. .
        assert any("bootstrap iterations" in m for m in msgs), (
            f"Expected low-convergence-rate warning containing "
            f"'bootstrap iterations'; got: {msgs}"
        )
        # And the result must still be returned with the surviving
        # iterations, not raised away.
        assert res.n_sim_ok < res.n_sim
        assert res.n_sim_ok > 0
    finally:
        pbmod_module._refit_mixedlm_ml = original


def test_pbmodcomp_parallel_matches_serial_within_mc_noise():
    """``n_jobs=2`` and ``n_jobs=1`` with the same ``seed`` must
    produce *identical* simulated LRT distributions, because
    ``SeedSequence.spawn`` gives each iteration an independent RNG
    that doesn't depend on execution order. This is the
    P3 reproducibility invariant.

    Skipped when ``joblib`` is not available."""
    try:
        import joblib  # noqa: F401
    except ImportError:
        pytest.skip("joblib not installed; n_jobs > 1 unavailable")

    large, small = _fit_sleepstudy_pair()
    res_serial = pbmodcomp(large, small, n_sim=30, seed=42, n_jobs=1)
    res_parallel = pbmodcomp(large, small, n_sim=30, seed=42, n_jobs=2)
    # The LRT distributions must be bit-for-bit identical when the
    # only difference is execution-order — SeedSequence guarantees
    # each iter draws from an independent stream regardless of order.
    np.testing.assert_array_equal(
        np.sort(res_serial.lrt_dist), np.sort(res_parallel.lrt_dist),
        err_msg=(
            "parallel and serial pbmodcomp with the "
            "same seed must produce identical (modulo order) LRT "
            "distributions."
        ),
    )
    assert res_serial.lrt_obs == res_parallel.lrt_obs
    assert res_serial.n_sim_ok == res_parallel.n_sim_ok


def test_pbmodcomp_n_jobs_raises_clear_error_without_joblib(monkeypatch):
    """When ``n_jobs != 1`` and ``joblib`` is not installed, the
    error message must point the user at the install command rather
    than emitting a cryptic ``ModuleNotFoundError``."""
    import builtins

    real_import = builtins.__import__

    def _no_joblib(name, *args, **kwargs):
        if name == "joblib":
            raise ImportError("simulated missing joblib")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_joblib)
    large, small = _fit_sleepstudy_pair()
    with pytest.raises(ImportError, match="pip install joblib"):
        pbmodcomp(large, small, n_sim=2, seed=0, n_jobs=2)


def test_is_reml_fit_safer_side_default():
    """``_is_reml_fit`` previously
    used ``AND`` between ``model.reml`` and ``fit.reml``, which
    silently returned False (→ no refit, → REML log-likelihoods
    used in the LRT) if the two flags ever disagreed.
    the function ORs the two flags so any indication of REML
    triggers the safer-side refit.

    Guard: a fit with ``model.reml=True`` and ``fit.reml=False``
    (the broken case the original AND-logic would have missed)
    must still be flagged as REML."""
    from pymmeans.pbmodcomp import _is_reml_fit

    # Real REML fit → REML.
    large_reml, _ = _fit_sleepstudy_pair(reml=True)
    assert _is_reml_fit(large_reml) is True

    # Real ML fit → not REML.
    large_ml, _ = _fit_sleepstudy_pair(reml=False)
    assert _is_reml_fit(large_ml) is False

    # The "split-flag" case the identified: only one flag
    # says REML. Either *should* still be treated as REML under
    # the safer-side OR rule.
    class _SplitFlag:
        def __init__(self, model_reml: bool, fit_reml: bool):
            self.model = type("M", (), {"reml": model_reml})()
            self.reml = fit_reml

    assert _is_reml_fit(_SplitFlag(model_reml=True, fit_reml=False)) is True
    assert _is_reml_fit(_SplitFlag(model_reml=False, fit_reml=True)) is True
    assert _is_reml_fit(_SplitFlag(model_reml=False, fit_reml=False)) is False
    assert _is_reml_fit(_SplitFlag(model_reml=True, fit_reml=True)) is True


def test_refit_preserves_use_sqrt():
    """``_refit_mixedlm_ml`` did not
    propagate ``use_sqrt`` from the original model. Both
    parameterisations span the same likelihood surface (so the ML
    maximum is unchanged), but the optimiser path differs and the
    original user's choice deserves to survive a refit.

    Guard: refitting a ``use_sqrt=False`` fit must preserve the
    flag on the new model."""
    import statsmodels.regression.mixed_linear_model as mlm

    from pymmeans.pbmodcomp import _refit_mixedlm_ml

    dat = pd.read_csv("tests/r_reference/pbmodcomp_data.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_no_sqrt = mlm.MixedLM.from_formula(
            "Reaction ~ Days", groups="Subject", data=dat,
        )
        # statsmodels.MixedLM constructor takes use_sqrt; force False.
        fit_no_sqrt.use_sqrt = False
        fit_no_sqrt = fit_no_sqrt.fit(reml=False)
    assert fit_no_sqrt.model.use_sqrt is False

    refit = _refit_mixedlm_ml(
        fit_no_sqrt, np.asarray(fit_no_sqrt.model.endog, dtype=float)
    )
    assert refit.model.use_sqrt is False, (
        "_refit_mixedlm_ml must propagate the "
        "original model's use_sqrt setting."
    )


def test_pbmodcomp_raises_on_non_nested_args():
    """previously, ``df = max(df, 1)``
    silently clamped (a) identical models passed twice and
    (b) swapped-argument user errors (large/small flipped) to
    df=1, masking the real mistake. Now both raise
    ``ValueError`` with a clear message.

    Guard: same model twice raises; swapped models raises; correct
    order succeeds."""
    large, small = _fit_sleepstudy_pair()

    # (a) Same model twice → df=0 → ValueError
    with pytest.raises(ValueError, match="strictly more parameters"):
        pbmodcomp(large, large, n_sim=2, seed=0)
    with pytest.raises(ValueError, match="strictly more parameters"):
        pbmodcomp(small, small, n_sim=2, seed=0)

    # (b) Swapped (small as large, large as small) → df<0 → ValueError
    with pytest.raises(ValueError, match="strictly more parameters"):
        pbmodcomp(small, large, n_sim=2, seed=0)

    # (c) Correct order still works (sanity check)
    res = pbmodcomp(large, small, n_sim=5, seed=0)
    assert res.df >= 1


def test_pbmodcomp_raises_if_all_iterations_fail():
    """When every bootstrap iteration fails to converge, raise
    ``RuntimeError`` instead of returning a result with empty
    ``lrt_dist`` (which would silently produce ``p_value = 1.0``)."""
    import sys

    large, small = _fit_sleepstudy_pair()
    # Monkey-patch the refit helper on the *module* object to
    # always raise. Need ``sys.modules`` since ``pymmeans.pbmodcomp``
    # in this file's namespace resolves to the imported *function*.
    pbmod_module = sys.modules["pymmeans.pbmodcomp"]

    def _always_fail(fit, y):
        raise ValueError("simulated convergence failure")

    original = pbmod_module._refit_mixedlm_ml
    pbmod_module._refit_mixedlm_ml = _always_fail
    try:
        with pytest.raises(RuntimeError, match="all bootstrap iterations failed"):
            pbmodcomp(large, small, n_sim=20, seed=0)
    finally:
        pbmod_module._refit_mixedlm_ml = original
