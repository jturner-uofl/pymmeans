"""Tests for plot_predictions / plot_slopes / plot_comparisons."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

pytest.importorskip("matplotlib")
import matplotlib

matplotlib.use("Agg")
from matplotlib.axes import Axes

from pymmeans import (
    avg_comparisons,
    avg_slopes,
    plot_comparisons,
    plot_predictions,
    plot_slopes,
)


def _logit(n=400, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x": rng.standard_normal(n),
        "z": rng.standard_normal(n),
        "g": pd.Categorical(rng.choice(["A", "B", "C"], n)),
    })
    eta = 0.3 + 0.8 * df["x"] - 0.5 * df["z"]
    df["yb"] = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    return smf.glm("yb ~ x + z + g", df, family=sm.families.Binomial()).fit(), df


# ---------------------------------------------------------------------- structure


def test_plot_predictions_numeric_is_line_with_band():
    fit, _ = _logit()
    ax = plot_predictions(fit, "x")
    assert isinstance(ax, Axes)
    assert len(ax.get_lines()) >= 1          # the prediction curve
    assert len(ax.collections) >= 1          # the CI band (fill_between)
    ydata = ax.get_lines()[0].get_ydata()
    assert ydata.min() >= 0.0 and ydata.max() <= 1.0  # valid probabilities


def test_plot_predictions_categorical_is_points():
    fit, _ = _logit()
    ax = plot_predictions(fit, "g")
    assert isinstance(ax, Axes)
    assert len(ax.containers) >= 1           # errorbar container
    assert [t.get_text() for t in ax.get_xticklabels()] == ["A", "B", "C"]


def test_plot_slopes_returns_forest():
    fit, _ = _logit()
    ax = plot_slopes(fit, "x", by="g")
    assert isinstance(ax, Axes)
    assert set(t.get_text() for t in ax.get_yticklabels()) == {"A", "B", "C"}


def test_plot_comparisons_ratio_reference_line_at_one():
    fit, _ = _logit()
    ax = plot_comparisons(fit, "x", comparison="ratio")
    assert isinstance(ax, Axes)
    dashed_x = [
        ln.get_xdata()[0]
        for ln in ax.get_lines()
        if ln.get_linestyle() == "--" and len(ln.get_xdata())
    ]
    assert any(abs(x - 1.0) < 1e-9 for x in dashed_x)


# ---------------------------------------------------------------------- correctness


def test_plot_slopes_value_matches_avg_slopes():
    """The plotted point equals the avg_slopes estimate it visualises."""
    fit, _ = _logit()
    ax = plot_slopes(fit, "x")
    # errorbar's central markers live in the container's data line.
    plotted = float(ax.containers[0][0].get_xdata()[0])
    expected = float(avg_slopes(fit, "x", type="response").frame["slope"].iloc[0])
    assert plotted == pytest.approx(expected, abs=1e-9)


def test_plot_predictions_endpoint_matches_gcomputation():
    """A point on the prediction curve equals the g-computation average
    prediction with the condition set to that value for all rows."""
    fit, df = _logit()
    ax = plot_predictions(fit, "x", grid_n=11)
    line = ax.get_lines()[0]
    xs, ys = line.get_xdata(), line.get_ydata()
    # check the middle grid point
    v = float(xs[len(xs) // 2])
    cf = df.copy(); cf["x"] = v
    manual = float(fit.predict(cf).mean())
    assert float(ys[len(ys) // 2]) == pytest.approx(manual, abs=1e-9)


# ---------------------------------------------------------------------- validation


def test_plot_predictions_rejects_unknown_condition():
    fit, _ = _logit()
    with pytest.raises(ValueError, match="not a column"):
        plot_predictions(fit, "nope")


def test_plot_comparisons_estimate_matches_avg_comparisons():
    fit, _ = _logit()
    ax = plot_comparisons(fit, "x", comparison="difference")
    plotted = float(ax.containers[0][0].get_xdata()[0])
    expected = float(avg_comparisons(fit, "x").frame["estimate"].iloc[0])
    assert plotted == pytest.approx(expected, abs=1e-9)
