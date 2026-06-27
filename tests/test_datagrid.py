"""Tests for datagrid() + newdata= on predictions / slopes / comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymmeans import (
    avg_comparisons,
    avg_predictions,
    avg_slopes,
    datagrid,
)


def _logit(n=500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x": rng.standard_normal(n) * 2 + 1,
        "z": rng.standard_normal(n),
        "g": pd.Categorical(rng.choice(["A", "B", "C"], n)),
    })
    eta = 0.3 + 0.5 * df["x"] - 0.5 * df["z"]
    df["yb"] = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    return smf.glm("yb ~ x + z + g", df, family=sm.families.Binomial()).fit(), df


# ---------------------------------------------------------------------- datagrid


def test_datagrid_crosses_specified_holds_rest_typical():
    fit, df = _logit()
    grid = datagrid(fit, x=[0, 1, 2])
    assert grid.shape[0] == 3
    # unspecified numeric z -> mean; unspecified factor g -> mode
    assert np.allclose(grid["z"].to_numpy(), df["z"].mean())
    assert (grid["g"] == df["g"].mode().iloc[0]).all()
    # specified values present
    assert list(grid["x"]) == [0, 1, 2]


def test_datagrid_cartesian_product():
    fit, _ = _logit()
    grid = datagrid(fit, x=[0, 1], g=["A", "B", "C"])
    assert grid.shape[0] == 6  # 2 x 3


def test_datagrid_preserves_categorical_dtype():
    fit, df = _logit()
    grid = datagrid(fit, x=[0, 1])
    assert hasattr(grid["g"], "cat")
    assert list(grid["g"].cat.categories) == list(df["g"].cat.categories)


def test_datagrid_rejects_unknown_variable():
    fit, _ = _logit()
    with pytest.raises(ValueError, match="non-predictor"):
        datagrid(fit, nope=[1, 2])


# ---------------------------------------------------------------------- newdata


def test_avg_predictions_newdata_equals_manual_grid_mean():
    fit, _ = _logit()
    grid = datagrid(fit, x=[0, 1, 2])
    res = avg_predictions(fit, newdata=grid)
    manual = float(fit.predict(grid).mean())
    assert float(res.frame["estimate"].iloc[0]) == pytest.approx(manual, abs=1e-9)


def test_avg_predictions_newdata_matches_marginaleffects():
    me = pytest.importorskip("marginaleffects")
    fit, _ = _logit()
    grid = datagrid(fit, x=[0, 1, 2])
    res = avg_predictions(fit, newdata=grid)
    ref = me.predictions(fit, newdata=me.datagrid(model=fit, x=[0, 1, 2])).to_pandas()
    assert float(res.frame["estimate"].iloc[0]) == pytest.approx(
        float(ref["estimate"].mean()), abs=1e-8
    )


def test_avg_slopes_newdata_runs_and_differs_from_sample():
    fit, _ = _logit()
    grid = datagrid(fit, x=[-2, 0, 2])
    at_grid = avg_slopes(fit, "x", type="response", newdata=grid)
    over_sample = avg_slopes(fit, "x", type="response")
    assert np.isfinite(float(at_grid.frame["slope"].iloc[0]))
    # The grid average is a different estimand than the sample average.
    assert float(at_grid.frame["slope"].iloc[0]) != float(
        over_sample.frame["slope"].iloc[0]
    )


def test_avg_comparisons_newdata_runs():
    fit, _ = _logit()
    grid = datagrid(fit, x=[0, 1])
    res = avg_comparisons(fit, "x", newdata=grid)
    assert np.isfinite(float(res.frame["estimate"].iloc[0]))
    assert np.isfinite(float(res.frame["SE"].iloc[0]))
