"""Tests for Rubin's-rules pooling of multiply-imputed pymmeans results."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import (
    PooledImputationResult,
    contrast,
    emmeans,
    pool_imputed,
)

# ---------------------------------------------------------------------- helpers

def _make_simple_imputations(n=200, M=5, true_ate=1.5, seed=0):
    """Build M synthetic imputed datasets with a known ATE.

    Each dataset is a fresh draw from the same DGP — so this is a
    *fake* multiple imputation (all M "imputations" are independent
    samples). Useful for verifying closed-form properties; for a
    realistic MAR-with-imputation scenario, see the notebook §XIX.
    """
    rng = np.random.default_rng(seed)
    imputed_results = []
    for _ in range(M):
        df = pd.DataFrame({
            "treat": pd.Categorical(rng.integers(0, 2, n)),
            "x": rng.standard_normal(n),
        })
        df["y"] = (
            df["treat"].astype(int) * true_ate
            + 0.3 * df["x"]
            + rng.standard_normal(n)
        )
        fit = smf.ols("y ~ treat + x", df).fit()
        em = emmeans(fit, "treat")
        ct = contrast(em, method="trt.vs.ctrl", ref=0)
        imputed_results.append(ct)
    return imputed_results


# ---------------------------------------------------------------------- validation

def test_pool_imputed_empty_list_raises():
    """Empty input list raises."""
    with pytest.raises(ValueError, match="non-empty"):
        pool_imputed([])


def test_pool_imputed_single_result_raises():
    """M=1 is not a multiple imputation."""
    ct_list = _make_simple_imputations(n=100, M=1)
    with pytest.raises(ValueError, match="M >= 2"):
        pool_imputed(ct_list)


def test_pool_imputed_mixed_types_raises():
    """Mixed EMMResult / ContrastResult inputs raises TypeError."""
    cts = _make_simple_imputations(n=100, M=3)
    # Build an EMM-result-typed object alongside the contrasts.
    fit = smf.ols("y ~ treat + x", pd.DataFrame({
        "treat": pd.Categorical([0, 1, 0, 1, 0, 1] * 20),
        "x": np.random.default_rng(0).standard_normal(120),
        "y": np.random.default_rng(0).standard_normal(120),
    })).fit()
    em = emmeans(fit, "treat")
    with pytest.raises(TypeError, match="same type"):
        pool_imputed([cts[0], em, cts[2]])


def test_pool_imputed_mismatched_rowcounts_raises():
    """Row-count mismatch across imputations raises."""
    cts = _make_simple_imputations(n=80, M=3)
    # Corrupt: trim one frame to have fewer rows.
    from dataclasses import replace as _dc_replace
    bad = _dc_replace(cts[1], frame=cts[1].frame.iloc[:0])
    with pytest.raises(ValueError, match="identical row counts"):
        pool_imputed([cts[0], bad, cts[2]])


# ---------------------------------------------------------------------- closed form

def test_pool_imputed_recovers_truth_on_iid_replicates():
    """On iid replicates of the same DGP, pooled estimate ≈ truth."""
    true_ate = 1.5
    cts = _make_simple_imputations(n=400, M=5, true_ate=true_ate, seed=42)
    pooled = pool_imputed(cts)
    est = float(pooled.frame["estimate"].iloc[0])
    se = float(pooled.frame["SE"].iloc[0])
    # On iid replicates the pooled SE should bracket the true value.
    assert abs(est - true_ate) < 4.0 * se, (
        f"pooled estimate {est:.4f} ± {se:.4f} should be near truth {true_ate}"
    )


def test_pool_imputed_within_plus_between_identity():
    """Total variance T = Ū + (1+1/M) · B holds row-wise."""
    cts = _make_simple_imputations(n=300, M=5, seed=7)
    pooled = pool_imputed(cts)

    # Recompute Rubin's rules by hand.
    pts = np.stack([c.frame["estimate"].to_numpy() for c in cts])
    ses = np.stack([c.frame["SE"].to_numpy() for c in cts])
    M = len(cts)
    theta_bar = pts.mean(axis=0)
    U_bar = (ses ** 2).mean(axis=0)
    B = ((pts - theta_bar) ** 2).sum(axis=0) / (M - 1)
    T = U_bar + (1.0 + 1.0 / M) * B
    SE_man = np.sqrt(T)

    np.testing.assert_allclose(
        pooled.frame["estimate"].to_numpy(), theta_bar, atol=1e-12
    )
    np.testing.assert_allclose(
        pooled.frame["SE"].to_numpy(), SE_man, atol=1e-12
    )


def test_pool_imputed_zero_between_variance_collapses_to_within():
    """If every imputation gives the IDENTICAL estimate, B=0 and SE = SE_within.

    This is a structural identity: when between-imputation variance is
    exactly zero, the pooled SE equals the (common) within-imputation SE.
    """
    cts_list = _make_simple_imputations(n=200, M=3, seed=11)
    # Force IDENTICAL frames across all "imputations" — this is
    # impossible in practice but tests the boundary correctly.
    from dataclasses import replace as _dc_replace
    forced = [
        _dc_replace(c, frame=cts_list[0].frame.copy()) for c in cts_list
    ]
    pooled = pool_imputed(forced)
    expected_se = float(cts_list[0].frame["SE"].iloc[0])
    actual_se = float(pooled.frame["SE"].iloc[0])
    assert actual_se == pytest.approx(expected_se, abs=1e-12)


def test_pool_imputed_fmi_in_unit_interval():
    """Fraction of missing information should lie in [0, 1] per row."""
    cts = _make_simple_imputations(n=300, M=5)
    pooled = pool_imputed(cts)
    assert np.all(pooled.fmi >= 0.0 - 1e-12)
    assert np.all(pooled.fmi <= 1.0 + 1e-12)


def test_pool_imputed_relative_increase_nonneg():
    """Relative increase r = (1+1/M)·B/Ū is non-negative."""
    cts = _make_simple_imputations(n=300, M=5)
    pooled = pool_imputed(cts)
    assert np.all(pooled.relative_increase >= 0.0 - 1e-12)


def test_pool_imputed_returns_dataclass_with_expected_fields():
    """Return type is the documented dataclass."""
    cts = _make_simple_imputations(n=200, M=4)
    pooled = pool_imputed(cts)
    assert isinstance(pooled, PooledImputationResult)
    assert pooled.M == 4
    for col in ("estimate", "SE", "df", "t_ratio", "p_value", "lower_cl", "upper_cl"):
        assert col in pooled.frame.columns


def test_pool_imputed_ci_bounds_consistent_with_t_critical():
    """CI bounds use SE_pooled × t-critical at Barnard-Rubin df."""
    import scipy.stats as stats
    cts = _make_simple_imputations(n=300, M=5)
    pooled = pool_imputed(cts)
    est = pooled.frame["estimate"].to_numpy()
    se = pooled.frame["SE"].to_numpy()
    df = pooled.frame["df"].to_numpy()
    tcrit = stats.t.isf(0.025, df)  # 95%
    np.testing.assert_allclose(
        pooled.frame["lower_cl"].to_numpy(), est - tcrit * se, atol=1e-10
    )
    np.testing.assert_allclose(
        pooled.frame["upper_cl"].to_numpy(), est + tcrit * se, atol=1e-10
    )
