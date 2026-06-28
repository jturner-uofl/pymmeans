"""HPD (highest-posterior-density) credible intervals for posterior EMMs —
R emmeans' hpd.summary. Validated against arviz.hdi (the reference HDI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import PosteriorInfo, posterior_emm_summary, posterior_emmeans
from pymmeans.posterior import _hpd_intervals
from pymmeans.utils import from_fitted

az = pytest.importorskip("arviz")


def test_hpd_intervals_match_arviz_hdi_exactly():
    """The Chen-Shao HPD matches arviz.hdi to the bit on skewed draws."""
    rng = np.random.default_rng(7)
    n = 15000
    samples = np.column_stack([
        rng.lognormal(0.0, 0.6, n),
        rng.lognormal(0.5, 0.9, n),
        rng.gamma(2.0, 2.0, n),
    ])
    for level in (0.95, 0.9, 0.8):
        lo, hi = _hpd_intervals(samples, level)
        for r in range(samples.shape[1]):
            a = az.hdi(samples[:, r], hdi_prob=level)
            assert lo[r] == pytest.approx(a[0], abs=1e-12)
            assert hi[r] == pytest.approx(a[1], abs=1e-12)


def test_posterior_emm_summary_hpd_matches_arviz():
    """hpd=True on the public summary returns arviz HDI endpoints."""
    rng = np.random.default_rng(11)
    n = 12000
    beta = np.column_stack([rng.lognormal(0.0, 0.7, n), rng.lognormal(0.3, 0.5, n)])
    out = posterior_emm_summary(beta, np.eye(2), level=0.95, hpd=True)
    for r in range(2):
        a = az.hdi(beta[:, r], hdi_prob=0.95)
        assert out["lower_cl"][r] == pytest.approx(a[0], abs=1e-12)
        assert out["upper_cl"][r] == pytest.approx(a[1], abs=1e-12)


def test_hpd_is_narrower_than_equal_tailed_for_skewed():
    rng = np.random.default_rng(3)
    beta = rng.lognormal(0.0, 0.8, (20000, 1))
    hpd = posterior_emm_summary(beta, np.eye(1), hpd=True)
    eti = posterior_emm_summary(beta, np.eye(1), hpd=False)
    w_hpd = float(hpd["upper_cl"][0] - hpd["lower_cl"][0])
    w_eti = float(eti["upper_cl"][0] - eti["lower_cl"][0])
    assert w_hpd < w_eti  # strictly narrower for a skewed posterior


def test_hpd_approximates_equal_tailed_for_symmetric():
    rng = np.random.default_rng(5)
    beta = rng.standard_normal((40000, 1))
    hpd = posterior_emm_summary(beta, np.eye(1), hpd=True)
    eti = posterior_emm_summary(beta, np.eye(1), hpd=False)
    # For a symmetric posterior HPD and equal-tailed coincide up to MC noise.
    assert hpd["lower_cl"][0] == pytest.approx(eti["lower_cl"][0], abs=0.05)
    assert hpd["upper_cl"][0] == pytest.approx(eti["upper_cl"][0], abs=0.05)


def test_posterior_emmeans_hpd_end_to_end():
    """posterior_emmeans(hpd=True) threads HPD into the EMMResult frame."""
    rng = np.random.default_rng(1)
    n = 150
    df = pd.DataFrame({
        "a": pd.Categorical(rng.choice(["A", "B"], n)),
        "x": rng.normal(size=n),
    })
    df["y"] = 1 + 0.4 * (df["a"] == "B") + 0.3 * df["x"] + rng.normal(scale=0.3, size=n)
    fit = smf.ols("y ~ a + x", data=df).fit()
    rng2 = np.random.default_rng(0)
    beta_hat = np.asarray(fit.params, dtype=float)
    cov = np.asarray(fit.cov_params(), dtype=float)
    chol = np.linalg.cholesky(cov + 1e-12 * np.eye(cov.shape[0]))
    samples = beta_hat[None, :] + rng2.normal(size=(10000, beta_hat.size)) @ chol.T
    pinfo = PosteriorInfo(beta_samples=samples, model_info=from_fitted(fit))
    out = posterior_emmeans(pinfo, "a", hpd=True)
    # the frame's intervals equal the direct-summary HPD on the same linfct
    direct = posterior_emm_summary(samples, out.linfct, level=0.95, hpd=True)
    np.testing.assert_allclose(out.frame["lower_cl"].to_numpy(), direct["lower_cl"], atol=1e-12)
    np.testing.assert_allclose(out.frame["upper_cl"].to_numpy(), direct["upper_cl"], atol=1e-12)
    assert (out.frame["lower_cl"] < out.frame["emmean"]).all()
    assert (out.frame["emmean"] < out.frame["upper_cl"]).all()
