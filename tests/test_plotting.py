"""Tests for plot() and emmip(). Uses matplotlib's Agg backend (no GUI)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from matplotlib.axes import Axes

from pymmeans import emmeans
from pymmeans.plotting import emmip, plot


@pytest.fixture
def simple_ols():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=120) + np.repeat([0, 1, 2], 40),
            "g": pd.Categorical(np.tile(["a", "b", "c"], 40)),
            "h": pd.Categorical(np.repeat(["x", "y"], 60)),
        }
    )
    return smf.ols("y ~ g * h", data=df).fit()


def test_plot_returns_axes(simple_ols):
    emm = emmeans(simple_ols, "g")
    ax = plot(emm)
    assert isinstance(ax, Axes)


def test_plot_has_one_marker_per_row(simple_ols):
    emm = emmeans(simple_ols, "g")
    ax = plot(emm)
    # errorbar -> one Line2D for the markers + bars
    assert len(ax.get_yticklabels()) == 3


def test_plot_by_grouped_labels(simple_ols):
    emm = emmeans(simple_ols, "g", by="h")
    ax = plot(emm)
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert len(labels) == 6
    assert "g=a" in labels[0] and "h=" in labels[0]


def test_plot_ref_line_drawn(simple_ols):
    emm = emmeans(simple_ols, "g")
    ax = plot(emm, ref_line=0.0)
    # axvline adds a Line2D with vertical=True (xdata constant across the 2 endpoints)
    verticals = [
        ln
        for ln in ax.get_lines()
        if len(ln.get_xdata()) == 2 and ln.get_xdata()[0] == ln.get_xdata()[1]
    ]
    assert len(verticals) >= 1


def test_plot_response_scale_label():
    rng = np.random.default_rng(3)
    n = 200
    df = pd.DataFrame(
        {
            "y": rng.integers(0, 2, n),
            "g": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        }
    )
    fit = smf.glm("y ~ g", data=df, family=sm.families.Binomial()).fit()
    emm = emmeans(fit, "g", type="response")
    ax = plot(emm)
    assert "response" in ax.get_xlabel()


def test_emmip_single_factor_returns_axes(simple_ols):
    ax = emmip(simple_ols, x="g")
    assert isinstance(ax, Axes)
    # one line drawn
    assert len(ax.get_lines()) == 1


def test_emmip_with_by_creates_multiple_lines(simple_ols):
    ax = emmip(simple_ols, x="g", by="h")
    assert len(ax.get_lines()) == 2  # one per h level
    legend = ax.get_legend()
    assert legend is not None
    legend_labels = [t.get_text() for t in legend.get_texts()]
    assert any("h=x" in s for s in legend_labels)
    assert any("h=y" in s for s in legend_labels)


def test_emmip_response_scale_label():
    rng = np.random.default_rng(5)
    n = 200
    df = pd.DataFrame(
        {
            "y": rng.integers(0, 2, n),
            "g": pd.Categorical(rng.choice(["a", "b", "c"], n)),
            "h": pd.Categorical(rng.choice(["x", "y"], n)),
        }
    )
    fit = smf.glm("y ~ g * h", data=df, family=sm.families.Binomial()).fit()
    ax = emmip(fit, x="g", by="h", type="response")
    assert "response" in ax.get_ylabel()
