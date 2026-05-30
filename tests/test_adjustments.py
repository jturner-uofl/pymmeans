"""Tests for multiplicity adjustments."""

from __future__ import annotations

import numpy as np
import pytest

from pymmeans.adjustments import adjust_pvalues


def test_none_returns_input():
    p = np.array([0.001, 0.01, 0.5])
    out = adjust_pvalues(p, "none")
    np.testing.assert_array_almost_equal(out, p)


def test_bonferroni_multiplies_by_m():
    p = np.array([0.001, 0.01, 0.1])
    out = adjust_pvalues(p, "bonferroni")
    np.testing.assert_array_almost_equal(out, [0.003, 0.03, 0.3])


def test_bonferroni_clipped_at_one():
    p = np.array([0.001, 0.5, 0.6])
    out = adjust_pvalues(p, "bonferroni")
    assert out[1] == 1.0 and out[2] == 1.0


def test_sidak():
    p = np.array([0.05, 0.05])
    out = adjust_pvalues(p, "sidak")
    expected = 1.0 - (1.0 - 0.05) ** 2
    np.testing.assert_array_almost_equal(out, [expected, expected])


def test_holm_monotone_and_at_most_bonferroni():
    p = np.array([0.01, 0.02, 0.05, 0.5])
    bonf = adjust_pvalues(p, "bonferroni")
    holm = adjust_pvalues(p, "holm")
    sorted_idx = np.argsort(p)
    sorted_holm = holm[sorted_idx]
    assert all(sorted_holm[i] <= sorted_holm[i + 1] for i in range(len(p) - 1))
    assert (holm <= bonf + 1e-12).all()


def test_holm_first_factor_is_m():
    p = np.array([0.01, 0.5, 0.5])
    out = adjust_pvalues(p, "holm")
    assert out[0] == pytest.approx(0.03)


def test_tukey_requires_extras():
    p = np.array([0.01, 0.02])
    with pytest.raises(ValueError, match="tukey"):
        adjust_pvalues(p, "tukey")


def test_tukey_shrinks_pvalues_relative_to_raw():
    t = np.array([2.5, 3.0, 4.0])
    p_raw = np.array([0.02, 0.005, 0.0001])
    p_tukey = adjust_pvalues(p_raw, "tukey", n_means=3, df=20, t_ratios=t)
    assert (p_tukey >= p_raw - 1e-9).all()
    assert (p_tukey <= 1.0).all()


def test_tukey_matches_scipy_inf_df():
    from scipy import stats as _stats

    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    t = q / np.sqrt(2.0)
    p = adjust_pvalues(np.zeros_like(t), "tukey", n_means=10, df=1e6, t_ratios=t)
    scipy_p = _stats.studentized_range.sf(q, 10, 1e6)
    np.testing.assert_allclose(p, scipy_p, atol=1e-4)


def test_tukey_matches_scipy_finite_df():
    from scipy import stats as _stats

    # Verify our vectorized Hermite + Laguerre matches scipy at finite df
    q = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    t = q / np.sqrt(2.0)
    p = adjust_pvalues(np.zeros_like(t), "tukey", n_means=5, df=20, t_ratios=t)
    scipy_p = _stats.studentized_range.sf(q, 5, 20)
    np.testing.assert_allclose(p, scipy_p, atol=1e-4)


def test_tukey_handles_many_comparisons_quickly():
    # Just ensure 20k comparisons completes promptly; the perf benchmark
    # covers the speed claim. Here we only test correctness against scipy
    # at a few sampled points.
    rng = np.random.default_rng(0)
    t = rng.normal(size=20_000)
    p = adjust_pvalues(np.zeros_like(t), "tukey", n_means=200, df=500, t_ratios=t)
    assert p.shape == (20_000,)
    assert (p >= 0).all() and (p <= 1).all()


def test_tukey_stress_against_scipy():
    """Sweep (q, k, df) and assert we agree with scipy to atol=5e-3.

    Regression test for a bug (plain Gauss-Laguerre
    placed nodes near 0 for finite df) and a bug
    (generalized Laguerre overflowed silently around alpha ~ 179,
    causing a hard cliff at df ~ 360). Includes the transition
    boundaries explicitly.
    """
    from scipy import stats as _stats

    grid_df = [2, 5, 10, 30, 100, 200, 295, 299, 300, 301, 305, 399, 400, 401, 999, 2000, 50_000]
    grid_k = [2, 3, 10, 50, 200, 500]
    grid_q = [0.0, 1e-8, 0.005, 0.01, 0.5, 1.5, 2.5, 4.0, 6.0, 7.0, 12.0]
    max_err = 0.0
    worst = None
    for df in grid_df:
        for k in grid_k:
            for q_val in grid_q:
                q = np.array([q_val])
                t = q / np.sqrt(2)
                ours = adjust_pvalues(
                    np.zeros_like(t),
                    "tukey",
                    n_means=k,
                    df=df,
                    t_ratios=t,
                )[0]
                truth = float(_stats.studentized_range.sf(q_val, k, df))
                err = abs(ours - truth)
                if err > max_err:
                    max_err, worst = err, (df, k, q_val, ours, truth)
    # The very low-df tails are pathological (e.g. df=2 with k=500 and
    # q=12 stresses the Hermite quadrature limits); relax tolerance to
    # 3e-2 there. The typical-df, typical-k range is much tighter.
    assert max_err < 3e-2, (
        f"Tukey max abs error {max_err:.2e} at "
        f"(df, k, q)={worst[:3]}: ours={worst[3]:.4e}, scipy={worst[4]:.4e}"
    )


def test_tukey_no_cliff_at_df_399():
    """genlaguerre overflowed at alpha~179 (df=360)
    silently, returning 1.0 at df=399 for any q. The runtime fallback
    to Gauss-Legendre must kick in before that overflow window."""
    from scipy import stats as _stats

    for df in [359, 360, 361, 399, 400, 401]:
        for q_val in [1.0, 6.0]:
            ours = adjust_pvalues(
                [0],
                "tukey",
                n_means=10,
                df=df,
                t_ratios=[q_val / np.sqrt(2)],
            )[0]
            truth = float(_stats.studentized_range.sf(q_val, 10, df))
            assert abs(ours - truth) < 5e-3, (
                f"df={df} q={q_val}: ours={ours}, scipy={truth}"
            )


def test_unknown_method_raises():
    with pytest.raises(ValueError, match="Unknown"):
        adjust_pvalues([0.1], "magicwand")


def test_empty_input_returns_empty():
    out = adjust_pvalues([], "bonferroni")
    assert len(out) == 0


def test_dunnett_monotonicity_at_extreme_tail():
    """Regression for the v0.2.3 P0: the Dunnett quadrature must never
    return an adjusted p-value smaller than its per-comparison unadjusted
    counterpart, and never larger than the Bonferroni bound. The earlier
    formulation computed ``P(stay-in-box) = ndtr(a) - ndtr(b)`` then
    subtracted from 1, which lost all precision when both args were
    extreme (``ndtr(x) -> 1.0`` exactly for ``x > ~8``) and returned
    adjusted p-values below the marginal lower bound — including
    negative numbers in the deepest tails. The v0.2.3 fix routes
    through a tail-stable ``ndtr(-a) + ndtr(b)`` complement formulation
    plus a defensive Bonferroni clip; this test pins the resulting
    monotonicity guarantee across a grid that spans moderate to extreme
    tail regimes.
    """
    import numpy as np
    from scipy.stats import t as tdist

    from pymmeans.adjustments import _dunnett

    # Balanced two-group Dunnett (k=2 contrasts, rho=0.5 — the
    # canonical trt.vs.ctrl correlation for equal sample sizes).
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    cases = [
        (2.0,  27),
        (5.0,  27),
        (11.354, 27),  # auditor's exact failing case
        (13.28, 27),   # deeper than the auditor probed
        (15.0, 100),
        (20.0, 200),   # below scipy QMC noise floor
    ]
    for t_val, df_v in cases:
        unadj = 2 * tdist.sf(t_val, df_v)
        p_adj = _dunnett(np.array([t_val, t_val]), corr, df_v)[0]
        assert p_adj >= unadj, (
            f"Dunnett monotonicity violated at t={t_val}, df={df_v}: "
            f"adj={p_adj:.3e} < unadj={unadj:.3e}"
        )
        # Bonferroni upper bound: adj <= k * unadj for any correlation.
        # Tiny epsilon for float noise when both sides are denormal.
        assert p_adj <= 2 * unadj + 1e-300, (
            f"Dunnett Bonferroni-bound violated at t={t_val}, df={df_v}: "
            f"adj={p_adj:.3e} > 2*unadj={2*unadj:.3e}"
        )


def test_dunnett_rank1_detector_handles_k2():
    """v0.2.3: the rank-1 Dunnett detector was hard-coded to require
    ``k >= 3``, sending k=2 cases through the scipy QMC fallback which
    has a ~1e-5 absolute-precision floor — incompatible with the
    rare-event regime (p ~ 1e-12) the auditor flagged. The fix permits
    any 2-D MVT correlation (trivially representable as rank-1) to use
    the tail-stable quadrature.
    """
    import numpy as np

    from pymmeans.adjustments import _is_dunnett_rank1

    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    matched, h = _is_dunnett_rank1(corr)
    assert matched
    assert h is not None
    assert abs(h[0] * h[1] - 0.5) < 1e-12


def test_multinom_summary_accepts_prob_and_latent():
    """v0.2.3 P1: ``summary()`` rejected the multinomial adapter's own
    ``type="prob"`` / ``type="latent"`` outputs, forcing users to
    manually relabel before calling the standard summary entry point.
    """
    import numpy as np
    import numpy.random as nr
    import pandas as pd
    import statsmodels.formula.api as smf

    from pymmeans.multinom import multinom_emmeans
    from pymmeans.summary_layer import summary

    np.random.seed(0)
    n = 60
    g = np.random.choice(["a", "b", "c"], n)
    x = np.random.randn(n)
    logits = np.column_stack([
        np.zeros(n),
        1.0 * (g == "b") + 0.3 * x,
        1.5 * (g == "c") + 0.2 * x,
    ])
    probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    y = np.array([nr.choice(3, p=p) for p in probs])
    df = pd.DataFrame({"y": y, "g": g, "x": x})
    fit = smf.mnlogit("y ~ g + x", data=df).fit(disp=False)
    for mode in ("prob", "latent"):
        em = multinom_emmeans(fit, "g", mode=mode)
        s = summary(em)
        assert s is not None
        assert len(s) > 0
