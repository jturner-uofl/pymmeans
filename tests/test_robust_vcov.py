"""Robust / cluster-robust covariance on EMMs and contrasts.

pymmeans propagates a robust (HC) or cluster-robust covariance to marginal
means and their contrasts: automatically when the statsmodels fit carries
one (``cov_type="cluster"`` / ``"HC3"``), and explicitly via ``vcov=`` as a
matrix or an R-style ``vcov.=fn`` callable. The marginal SE is exactly the
sandwich identity ``sqrt(diag(L V_robust Lᵀ))``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import emmeans, pairs, summary


def _data(n=300, n_clusters=20, seed=0):
    rng = np.random.default_rng(seed)
    cl = rng.integers(0, n_clusters, n)
    d = pd.DataFrame({
        "g": pd.Categorical(rng.choice(["A", "B", "C"], n)),
        "cl": cl,
    })
    u = rng.standard_normal(n_clusters)[cl]
    d["y"] = d["g"].map({"A": 1.0, "B": 2.0, "C": 3.0}).astype(float) + u + rng.standard_normal(n)
    return d


def test_cluster_robust_fit_autopropagates_to_emm():
    """emmeans(cluster_fit) uses the fit's cluster covariance — the marginal
    SE equals the sandwich identity sqrt(diag(L V_cluster Lᵀ))."""
    d = _data()
    fit = smf.ols("y ~ g", d).fit(cov_type="cluster", cov_kwds={"groups": d["cl"]})
    em = emmeans(fit, "g")
    V = np.asarray(fit.cov_params(), dtype=float)
    expected = np.sqrt(np.diag(em.linfct @ V @ em.linfct.T))
    np.testing.assert_allclose(em.frame["SE"].to_numpy(), expected, atol=1e-12)


def test_hc3_vcov_matrix_matches_sandwich_identity():
    d = _data()
    base = smf.ols("y ~ g", d).fit()
    V = np.asarray(smf.ols("y ~ g", d).fit(cov_type="HC3").cov_params(), dtype=float)
    em = emmeans(base, "g", vcov=V)
    expected = np.sqrt(np.diag(em.linfct @ V @ em.linfct.T))
    np.testing.assert_allclose(em.frame["SE"].to_numpy(), expected, atol=1e-12)


def test_vcov_callable_equals_matrix():
    d = _data()
    base = smf.ols("y ~ g", d).fit()
    V = np.asarray(smf.ols("y ~ g", d).fit(cov_type="HC3").cov_params(), dtype=float)
    by_matrix = emmeans(base, "g", vcov=V).frame["SE"].to_numpy()
    by_thunk = emmeans(base, "g", vcov=lambda m: V).frame["SE"].to_numpy()
    np.testing.assert_allclose(by_matrix, by_thunk, atol=0)


def test_vcov_callable_receives_the_model():
    """R-style vcov.=fn: the callable is handed the fitted model and can
    compute the robust covariance from it."""
    d = _data()
    base = smf.ols("y ~ g", d).fit()
    V = np.asarray(smf.ols("y ~ g", d).fit(cov_type="HC3").cov_params(), dtype=float)

    def robust(model):
        return np.asarray(model.get_robustcov_results("HC3").cov_params(), dtype=float)

    by_fn = emmeans(base, "g", vcov=robust).frame["SE"].to_numpy()
    by_matrix = emmeans(base, "g", vcov=V).frame["SE"].to_numpy()
    np.testing.assert_allclose(by_fn, by_matrix, atol=1e-12)


def test_contrasts_inherit_robust_vcov():
    d = _data()
    base = smf.ols("y ~ g", d).fit()
    V = np.asarray(smf.ols("y ~ g", d).fit(cov_type="HC3").cov_params(), dtype=float)
    em = emmeans(base, "g", vcov=V)
    ct = summary(pairs(em))
    L = emmeans(base, "g").linfct
    d01 = L[0] - L[1]  # A - B
    expected = float(np.sqrt(d01 @ V @ d01))
    assert float(ct["SE"].iloc[0]) == pytest.approx(expected, abs=1e-10)


def test_vcov_callable_bad_shape_is_validated():
    d = _data()
    base = smf.ols("y ~ g", d).fit()
    with pytest.raises(ValueError, match="vcov="):
        emmeans(base, "g", vcov=lambda m: np.eye(2))
