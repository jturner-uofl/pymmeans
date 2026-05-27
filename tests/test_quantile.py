"""Tests for the P^2 streaming quantile estimator (leapfrog feature)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import bootstrap_ci, emmeans
from pymmeans.quantile import P2Batch, P2Estimator


def test_p2_estimator_converges_to_true_quantile_normal():
    """P² on a stream from N(0, 1) converges to the standard normal
    quantile to within ~0.5% at 50k samples."""
    rng = np.random.default_rng(0)
    n = 50_000
    samples = rng.normal(size=n)
    for p in [0.025, 0.5, 0.975]:
        est = P2Estimator(p)
        for x in samples:
            est.update(float(x))
        truth = float(np.quantile(samples, p))
        approx = est.value()
        assert abs(approx - truth) < 0.05, (
            f"p={p}: P^2={approx:.4f} vs np.quantile={truth:.4f}"
        )


def test_p2_estimator_burn_in_returns_nan():
    """Before 5 observations the estimator hasn't bootstrapped yet."""
    est = P2Estimator(0.5)
    assert np.isnan(est.value())
    for x in [1.0, 2.0, 3.0, 4.0]:
        est.update(x)
        assert np.isnan(est.value())
    est.update(5.0)
    assert np.isfinite(est.value())


def test_p2_estimator_rejects_invalid_percentile():
    with pytest.raises(ValueError):
        P2Estimator(0.0)
    with pytest.raises(ValueError):
        P2Estimator(1.0)
    with pytest.raises(ValueError):
        P2Estimator(-0.1)


def test_p2_batch_vectorised_matches_per_stream():
    """The vectorised P2Batch must give the same answers as running
    independent P2Estimator instances per (quantile, stream)."""
    rng = np.random.default_rng(1)
    n, n_streams = 10_000, 5
    batch = rng.normal(size=(n, n_streams))
    percentiles = [0.025, 0.5, 0.975]

    pb = P2Batch(percentiles, n_streams)
    pb.update_batch(batch)

    singles = [
        [P2Estimator(p) for _ in range(n_streams)] for p in percentiles
    ]
    for row in batch:
        for q_idx in range(len(percentiles)):
            for s_idx, est in enumerate(singles[q_idx]):
                est.update(float(row[s_idx]))
    expected = np.array(
        [[est.value() for est in row_ests] for row_ests in singles]
    )
    np.testing.assert_allclose(pb.values(), expected, atol=1e-12)


def test_bootstrap_ci_streaming_close_to_exact():
    """The streaming bootstrap should give CIs within bootstrap noise of
    the exact percentile bootstrap at the same seed and n_samples."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "g": pd.Categorical(rng.choice(["a", "b", "c", "d"], n)),
        }
    )
    fit = smf.ols("y ~ g", data=df).fit()
    emm = emmeans(fit, "g")
    exact = bootstrap_ci(emm, n_samples=50_000, seed=0, method="exact")
    stream = bootstrap_ci(emm, n_samples=50_000, seed=0, method="streaming")
    # At 50k samples P² is well-converged; absolute error should be < 0.01
    # on this scale (SD-1 normal data).
    np.testing.assert_allclose(
        stream.frame["lower_cl"].to_numpy(),
        exact.frame["lower_cl"].to_numpy(),
        atol=0.01,
    )
    np.testing.assert_allclose(
        stream.frame["upper_cl"].to_numpy(),
        exact.frame["upper_cl"].to_numpy(),
        atol=0.01,
    )


def test_bootstrap_ci_streaming_chunk_invariance():
    """P² is sequential per stream so the result must be invariant to
    chunk_size — different chunk sizes only batch the RNG, never the
    order in which observations are fed to P²."""
    rng = np.random.default_rng(0)
    n = 100
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "g": pd.Categorical(rng.choice(["a", "b"], n)),
        }
    )
    fit = smf.ols("y ~ g", data=df).fit()
    emm = emmeans(fit, "g")
    a = bootstrap_ci(emm, n_samples=20_000, seed=7, method="streaming", chunk_size=500)
    b = bootstrap_ci(emm, n_samples=20_000, seed=7, method="streaming", chunk_size=5000)
    np.testing.assert_allclose(
        a.frame["lower_cl"].to_numpy(),
        b.frame["lower_cl"].to_numpy(),
        atol=1e-12,
    )


def test_bootstrap_ci_method_validation():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=60),
            "g": pd.Categorical(rng.choice(["a", "b"], 60)),
        }
    )
    fit = smf.ols("y ~ g", data=df).fit()
    with pytest.raises(ValueError, match="method"):
        bootstrap_ci(emmeans(fit, "g"), n_samples=200, method="quantile")


def test_p2_batch_handles_extreme_values():
    """Skewed input (lognormal) — P² should still converge."""
    rng = np.random.default_rng(2)
    n, n_streams = 20_000, 3
    batch = np.exp(rng.normal(size=(n, n_streams)))
    pb = P2Batch([0.025, 0.975], n_streams)
    pb.update_batch(batch)
    for s in range(n_streams):
        truth_lo = float(np.quantile(batch[:, s], 0.025))
        truth_hi = float(np.quantile(batch[:, s], 0.975))
        approx_lo, approx_hi = pb.values()[:, s]
        # Lognormal has fat right tail; allow a slightly looser tol.
        assert abs(approx_lo - truth_lo) < 0.05 * truth_lo + 0.01
        assert abs(approx_hi - truth_hi) < 0.05 * truth_hi + 0.01
