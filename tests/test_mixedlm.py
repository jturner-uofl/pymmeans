"""Tests for MixedLM support."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import emmeans, pairs


@pytest.fixture
def mixedlm_fit():
    rng = np.random.default_rng(0)
    n_groups = 25
    n_per = 20
    n = n_groups * n_per
    subj = np.repeat(np.arange(n_groups), n_per)
    subj_effect = rng.normal(scale=0.5, size=n_groups)[subj]
    g = rng.choice(["a", "b", "c"], n)
    x = rng.normal(size=n)
    y = (
        1.0
        + 0.5 * (g == "b")
        - 0.3 * (g == "c")
        + 0.4 * x
        + subj_effect
        + rng.normal(scale=0.5, size=n)
    )
    df = pd.DataFrame({"y": y, "g": pd.Categorical(g), "x": x, "subj": subj})
    return smf.mixedlm("y ~ g + x", data=df, groups="subj").fit()


def test_emmeans_for_mixedlm(mixedlm_fit):
    emm = emmeans(mixedlm_fit, "g")
    assert emm.n_rows == 3
    # MixedLM uses Wald z-tests (df = inf)
    assert np.isinf(emm.frame["df"]).all()


def test_pairs_for_mixedlm(mixedlm_fit):
    pw = pairs(emmeans(mixedlm_fit, "g"))
    assert pw.n_rows == 3
    assert np.isinf(pw.frame["df"]).all()
    # p-values should be reasonable for the simulated effect structure
    assert (pw.frame["p_value"] >= 0).all()
    assert (pw.frame["p_value"] <= 1).all()


def test_mixedlm_emm_matches_predict_at_grand_mean(mixedlm_fit):
    # For MixedLM with formula 'y ~ g + x', emmeans for g should be the
    # population-average prediction at each level with x at its mean and
    # without the random effect — i.e. exactly what model.predict gives
    # on a small synthetic frame.
    emm = emmeans(mixedlm_fit, "g")
    info = emm.model_info
    expected = info.beta[0] + np.array(
        [
            0,  # reference level "a"
            info.beta[info.param_names.index("g[T.b]")],
            info.beta[info.param_names.index("g[T.c]")],
        ]
    ) + info.beta[info.param_names.index("x")] * info.numeric_means["x"]
    np.testing.assert_array_almost_equal(
        emm.frame["emmean"].to_numpy(), expected, decimal=8
    )
