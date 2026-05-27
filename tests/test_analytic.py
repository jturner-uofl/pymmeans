"""Tests for analytic marginalization.

The analytic path computes L_marg from patsy's term/factor structure
without materializing the grid. It must produce numerically identical
output to the eager/streamed paths for every supported model.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymmeans.analytic import analytic_marginalize
from pymmeans.emmeans import _marginalize_streamed
from pymmeans.ref_grid import build_grid_spec
from pymmeans.utils import from_statsmodels


def _info(fit):
    return from_statsmodels(fit)


@pytest.mark.parametrize(
    "formula,target,by",
    [
        ("y ~ g", ["g"], []),
        ("y ~ g + x", ["g"], []),
        ("y ~ g + h", ["g"], []),
        ("y ~ g + h", ["h"], []),
        ("y ~ g + h", ["g"], ["h"]),
        ("y ~ g * h", ["g"], []),
        ("y ~ g * h", ["g"], ["h"]),
        ("y ~ g * h + x", ["g"], []),
        ("y ~ g * h + x", ["g", "h"], []),
    ],
)
def test_analytic_matches_streamed(formula, target, by):
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "g": pd.Categorical(rng.choice(["a", "b", "c"], n)),
            "h": pd.Categorical(rng.choice(["x", "y", "z", "w"], n)),
            "x": rng.normal(size=n),
        }
    )
    fit = smf.ols(formula, data=df).fit()
    info = _info(fit)

    group_cols = list(target) + list(by)
    spec = build_grid_spec(info, None)

    L_an, keys_an = analytic_marginalize(info, spec, group_cols)
    L_st, keys_st = _marginalize_streamed(info, spec, group_cols, chunk_size=50)

    assert keys_an == keys_st
    np.testing.assert_allclose(L_an, L_st, atol=1e-10)


def test_analytic_matches_streamed_glm_logit():
    rng = np.random.default_rng(1)
    n = 400
    df = pd.DataFrame(
        {
            "y": rng.integers(0, 2, n),
            "g": pd.Categorical(rng.choice(["a", "b", "c"], n)),
            "h": pd.Categorical(rng.choice(["lo", "hi"], n)),
            "x": rng.normal(size=n),
        }
    )
    fit = smf.glm(
        "y ~ g * h + x", data=df, family=sm.families.Binomial()
    ).fit()
    info = _info(fit)
    spec = build_grid_spec(info, None)
    L_an, _ = analytic_marginalize(info, spec, ["g"])
    L_st, _ = _marginalize_streamed(info, spec, ["g"], chunk_size=50)
    np.testing.assert_allclose(L_an, L_st, atol=1e-10)


def test_analytic_with_at_override():
    rng = np.random.default_rng(2)
    n = 100
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "g": pd.Categorical(rng.choice(["a", "b"], n)),
            "x": rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ g + x", data=df).fit()
    info = _info(fit)

    # at sets x to a specific value; averaging is over that single value.
    spec = build_grid_spec(info, at={"x": 1.5})
    L_an, _ = analytic_marginalize(info, spec, ["g"])
    L_st, _ = _marginalize_streamed(info, spec, ["g"], chunk_size=50)
    np.testing.assert_allclose(L_an, L_st, atol=1e-10)


def test_analytic_at_multi_value_numeric_averages():
    rng = np.random.default_rng(3)
    n = 150
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "g": pd.Categorical(rng.choice(["a", "b", "c"], n)),
            "x": rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ g + x", data=df).fit()
    info = _info(fit)

    spec = build_grid_spec(info, at={"x": [-1.0, 0.0, 1.0]})
    L_an, _ = analytic_marginalize(info, spec, ["g"])
    L_st, _ = _marginalize_streamed(info, spec, ["g"], chunk_size=50)
    np.testing.assert_allclose(L_an, L_st, atol=1e-10)


def test_analytic_handles_intercept_only_term():
    rng = np.random.default_rng(4)
    n = 80
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "g": pd.Categorical(rng.choice(["a", "b"], n)),
        }
    )
    fit = smf.ols("y ~ g", data=df).fit()
    info = _info(fit)
    spec = build_grid_spec(info, None)
    L_an, _ = analytic_marginalize(info, spec, ["g"])
    assert L_an.shape == (2, info.n_params)
    # Intercept column should be 1.0 for both rows
    assert (L_an[:, 0] == 1.0).all()
