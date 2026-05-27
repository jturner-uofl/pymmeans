"""Tests for bootstrap_ci()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymmeans import bootstrap_ci, emmeans


@pytest.fixture
def ols_fit():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=90) + np.repeat([0.0, 1.0, 2.0], 30),
            "g": pd.Categorical(np.repeat(["a", "b", "c"], 30)),
        }
    )
    return smf.ols("y ~ g", data=df).fit()


@pytest.fixture
def binomial_fit():
    rng = np.random.default_rng(2)
    n = 400
    df = pd.DataFrame(
        {
            "y": rng.integers(0, 2, n),
            "g": pd.Categorical(rng.choice(["a", "b", "c"], n)),
            "x": rng.normal(size=n),
        }
    )
    return smf.glm("y ~ g + x", data=df, family=sm.families.Binomial()).fit()


def test_bootstrap_returns_emmresult(ols_fit):
    emm = emmeans(ols_fit, "g")
    boot = bootstrap_ci(emm, n_samples=2000, seed=0)
    assert boot.n_rows == emm.n_rows
    assert set(boot.frame.columns) == set(emm.frame.columns)


def test_bootstrap_preserves_point_estimates(ols_fit):
    emm = emmeans(ols_fit, "g")
    boot = bootstrap_ci(emm, n_samples=2000, seed=0)
    np.testing.assert_array_almost_equal(
        boot.frame["emmean"].to_numpy(), emm.frame["emmean"].to_numpy()
    )
    np.testing.assert_array_almost_equal(
        boot.frame["SE"].to_numpy(), emm.frame["SE"].to_numpy()
    )


def test_bootstrap_ci_close_to_t_ci_for_well_behaved(ols_fit):
    # For a large well-behaved OLS, parametric bootstrap percentile CIs
    # should approximately match t-CIs at 95%.
    emm = emmeans(ols_fit, "g")
    boot = bootstrap_ci(emm, n_samples=10_000, seed=42)
    np.testing.assert_allclose(
        boot.frame["lower_cl"].to_numpy(), emm.frame["lower_cl"].to_numpy(), atol=0.1
    )
    np.testing.assert_allclose(
        boot.frame["upper_cl"].to_numpy(), emm.frame["upper_cl"].to_numpy(), atol=0.1
    )


def test_bootstrap_seed_reproducible(ols_fit):
    emm = emmeans(ols_fit, "g")
    a = bootstrap_ci(emm, n_samples=2000, seed=123)
    b = bootstrap_ci(emm, n_samples=2000, seed=123)
    np.testing.assert_array_equal(
        a.frame["lower_cl"].to_numpy(), b.frame["lower_cl"].to_numpy()
    )


def test_bootstrap_binomial_response_stays_in_unit_interval(binomial_fit):
    emm = emmeans(binomial_fit, "g", type="response")
    boot = bootstrap_ci(emm, n_samples=2000, seed=7)
    assert (boot.frame["lower_cl"] > 0).all()
    assert (boot.frame["upper_cl"] < 1).all()
    assert (boot.frame["lower_cl"] <= boot.frame["emmean"]).all()
    assert (boot.frame["emmean"] <= boot.frame["upper_cl"]).all()


def test_bootstrap_level_changes_width(ols_fit):
    emm = emmeans(ols_fit, "g")
    boot95 = bootstrap_ci(emm, n_samples=5000, seed=1)
    boot99 = bootstrap_ci(emm, n_samples=5000, seed=1, level=0.99)
    w95 = (boot95.frame["upper_cl"] - boot95.frame["lower_cl"]).to_numpy()
    w99 = (boot99.frame["upper_cl"] - boot99.frame["lower_cl"]).to_numpy()
    assert (w99 > w95).all()


def test_bootstrap_rejects_tiny_n_samples(ols_fit):
    emm = emmeans(ols_fit, "g")
    with pytest.raises(ValueError, match="n_samples"):
        bootstrap_ci(emm, n_samples=10)
