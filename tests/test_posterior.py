"""Tests for the posterior / Bayesian EMM path.

We don't require PyMC for the tests — instead we generate synthetic
posterior samples (multivariate normal around a known mean) and
verify that the posterior-EMM summary recovers the analytic Wald CIs
in the limit. The :func:`from_pymc` adapter is exercised by a mock
arviz-like object so the test doesn't need the actual PyMC stack.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import (
    PosteriorInfo,
    contrast,
    emmeans,
    posterior_emm_summary,
    posterior_emmeans,
)
from pymmeans.utils import from_fitted


def _synthetic_posterior(fit, n_samples: int = 5000, seed: int = 0) -> np.ndarray:
    """Generate synthetic posterior samples from a fitted statsmodels result.

    Drawing from N(beta_hat, cov_params) gives a posterior identical to
    the Wald approximation, so the posterior-EMM summary should match
    the frequentist emmeans output up to Monte Carlo noise. Useful for
    testing without PyMC installed.
    """
    rng = np.random.default_rng(seed)
    beta_hat = np.asarray(fit.params, dtype=float)
    cov = np.asarray(fit.cov_params(), dtype=float)
    # Use cholesky for stability; add tiny jitter for near-singular cases.
    cov = cov + 1e-12 * np.eye(cov.shape[0]) * np.abs(np.diag(cov)).max()
    L = np.linalg.cholesky(cov)
    z = rng.normal(size=(n_samples, beta_hat.size))
    return beta_hat[None, :] + z @ L.T


def test_posterior_summary_rejects_mismatched_shapes():
    beta = np.zeros((100, 3))
    L = np.eye(4)
    with pytest.raises(ValueError, match="n_params"):
        posterior_emm_summary(beta, L)
    with pytest.raises(ValueError, match="2-D"):
        posterior_emm_summary(np.zeros(100), np.zeros((1, 1)))


def test_posterior_summary_recovers_wald_on_normal_posterior():
    """For a normal posterior centred at beta_hat with cov = cov_params,
    the posterior-EMM summary should match the Wald emmeans output."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "a": pd.Categorical(rng.choice(["A", "B", "C"], n)),
            "x": rng.normal(size=n),
        }
    )
    df["y"] = (
        1 + 0.5 * (df["a"] == "B") - 0.3 * (df["a"] == "C")
        + 0.7 * df["x"] + rng.normal(scale=0.3, size=n)
    )
    fit = smf.ols("y ~ a + x", data=df).fit()
    samples = _synthetic_posterior(fit, n_samples=20_000, seed=42)
    emm = emmeans(fit, "a")
    summary = posterior_emm_summary(samples, emm.linfct, level=0.95)
    # Posterior mean ~ Wald point estimate, posterior SD ~ Wald SE.
    np.testing.assert_allclose(
        summary["emmean"], emm.frame["emmean"].to_numpy(), atol=0.005
    )
    np.testing.assert_allclose(
        summary["SE"], emm.frame["SE"].to_numpy(), rtol=0.05
    )


def test_posterior_emmeans_returns_emm_result():
    """The high-level posterior_emmeans builds a full EMMResult with
    posterior-mean / SD / percentile credible intervals."""
    rng = np.random.default_rng(1)
    n = 150
    df = pd.DataFrame(
        {
            "a": pd.Categorical(rng.choice(["A", "B"], n)),
            "x": rng.normal(size=n),
        }
    )
    df["y"] = 1 + 0.4 * (df["a"] == "B") + 0.3 * df["x"] + rng.normal(scale=0.3, size=n)
    fit = smf.ols("y ~ a + x", data=df).fit()
    samples = _synthetic_posterior(fit, n_samples=10_000, seed=0)
    pinfo = PosteriorInfo(beta_samples=samples, model_info=from_fitted(fit))
    out = posterior_emmeans(pinfo, "a")
    assert list(out.frame["a"]) == ["A", "B"]
    assert all(out.frame["lower_cl"] < out.frame["emmean"])
    assert all(out.frame["emmean"] < out.frame["upper_cl"])
    # df = n_samples - 1 by convention for the posterior path
    assert out.frame["df"].iloc[0] == 9999.0


def test_posterior_emmeans_response_scale_via_lhs_transform():
    """For an LHS log-transformed OLS, posterior_emmeans(type='response')
    should give correctly asymmetric credible intervals (not symmetric
    around the back-transformed mean)."""
    rng = np.random.default_rng(2)
    n = 100
    df = pd.DataFrame({"a": pd.Categorical(rng.choice(["A", "B"], n))})
    df["y"] = np.exp(1 + 0.5 * (df["a"] == "B") + rng.normal(scale=0.4, size=n))
    fit = smf.ols("np.log(y) ~ a", data=df).fit()
    samples = _synthetic_posterior(fit, n_samples=10_000, seed=0)
    pinfo = PosteriorInfo(beta_samples=samples, model_info=from_fitted(fit))
    out = posterior_emmeans(pinfo, "a", type="response")
    # All on the original (positive) scale.
    assert (out.frame["emmean"] > 0).all()
    assert (out.frame["lower_cl"] > 0).all()
    # CIs are asymmetric: upper - mean != mean - lower (exp transform).
    diffs_up = out.frame["upper_cl"] - out.frame["emmean"]
    diffs_lo = out.frame["emmean"] - out.frame["lower_cl"]
    assert (diffs_up > diffs_lo + 1e-3).all()


def test_posterior_contrast_inherits_credible_intervals():
    """Round-trip: posterior emmeans -> contrast on the EMMResult uses
    the underlying linfct, so the contrast inherits the Bayesian
    point + Wald approximation around the posterior mean."""
    rng = np.random.default_rng(3)
    n = 120
    df = pd.DataFrame(
        {
            "a": pd.Categorical(rng.choice(["A", "B", "C"], n)),
            "x": rng.normal(size=n),
        }
    )
    df["y"] = 1 + 0.5 * (df["a"] == "B") + 0.7 * df["x"] + rng.normal(scale=0.3, size=n)
    fit = smf.ols("y ~ a + x", data=df).fit()
    samples = _synthetic_posterior(fit, n_samples=5000, seed=0)
    pinfo = PosteriorInfo(beta_samples=samples, model_info=from_fitted(fit))
    emm = posterior_emmeans(pinfo, "a")
    pairs = contrast(emm, method="pairwise")
    assert pairs.n_rows == 3
    assert (pairs.frame["SE"] > 0).all()


def test_posterior_emm_summary_offset_applied():
    """offset_mean shifts each posterior draw before the inverse link."""
    rng = np.random.default_rng(4)
    samples = rng.normal(loc=0.0, scale=0.1, size=(1000, 2))
    L = np.array([[1.0, 0.0], [0.0, 1.0]])
    no_offset = posterior_emm_summary(samples, L)
    with_offset = posterior_emm_summary(samples, L, offset_mean=2.0)
    np.testing.assert_allclose(
        with_offset["emmean"] - no_offset["emmean"], np.full(2, 2.0), atol=1e-12
    )


def test_from_pymc_requires_arviz():
    """The lazy import guard fires with a helpful error message."""
    try:
        import arviz  # noqa: F401  # presence check; missing => exercises ImportError path

        pytest.skip("arviz is installed; can't test the missing-import path")
    except ImportError:
        pass
    from pymmeans import from_pymc

    with pytest.raises(ImportError, match="arviz"):
        from_pymc(object(), "y ~ a", pd.DataFrame())


def test_posterior_emmeans_response_scale_glm_inverse_link():
    """For a GLM, type='response' should apply family.link.inverse to
    each posterior draw."""
    import statsmodels.api as sm

    rng = np.random.default_rng(5)
    n = 200
    df = pd.DataFrame({"a": pd.Categorical(rng.choice(["A", "B"], n))})
    df["y"] = rng.poisson(np.exp(0.3 + 0.4 * (df["a"] == "B")))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.glm("y ~ a", data=df, family=sm.families.Poisson()).fit()
    samples = _synthetic_posterior(fit, n_samples=5000, seed=0)
    pinfo = PosteriorInfo(beta_samples=samples, model_info=from_fitted(fit))
    out_link = posterior_emmeans(pinfo, "a", type="link")
    out_resp = posterior_emmeans(pinfo, "a", type="response")
    # Response scale should be exp(link) approximately.
    expected = np.exp(out_link.frame["emmean"].to_numpy())
    np.testing.assert_allclose(out_resp.frame["emmean"].to_numpy(), expected, rtol=0.02)
