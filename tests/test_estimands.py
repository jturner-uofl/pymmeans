"""estimands() — the "which average am I taking?" decision aid (wish-list #4).

Computes the EMM under each marginalisation scheme side by side so the
estimand choice (balanced/experimental vs population-marginal vs
sample-weighted) is explicit. See docs/wishlist.md.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import emmeans, estimands


def _imbalanced_fit(seed=0, n=600):
    """g↔h imbalance differs across g, so the schemes diverge."""
    rng = np.random.default_rng(seed)
    g = rng.choice(["A", "B"], n)
    h = np.where(
        g == "A", rng.choice(["p", "q"], n, p=[0.9, 0.1]),
        rng.choice(["p", "q"], n, p=[0.2, 0.8]),
    )
    d = pd.DataFrame({"g": pd.Categorical(g), "h": pd.Categorical(h)})
    base = d["g"].map({"A": 0.0, "B": 1.0}).astype(float)
    d["y"] = base + 1.5 * (d["h"] == "q") + rng.standard_normal(n)
    return smf.ols("y ~ g*h", d).fit()


def _balanced_fit(seed=0, reps=80):
    rows = list(itertools.product(["A", "B"], ["p", "q"])) * reps
    d = pd.DataFrame(rows, columns=["g", "h"])
    d["g"] = pd.Categorical(d["g"])
    d["h"] = pd.Categorical(d["h"])
    rng = np.random.default_rng(seed)
    base = d["g"].map({"A": 0.0, "B": 1.0}).astype(float)
    d["y"] = base + 1.5 * (d["h"] == "q") + rng.standard_normal(len(d))
    return smf.ols("y ~ g*h", d).fit()


def test_each_column_matches_the_individual_emmeans_call():
    fit = _imbalanced_fit()
    tab = estimands(fit, "g")
    for w in ("equal", "proportional", "cells"):
        individual = emmeans(fit, "g", weights=w).frame["emmean"].to_numpy()
        np.testing.assert_allclose(tab[f"emmean[{w}]"].to_numpy(), individual, atol=1e-12)


def test_schemes_diverge_on_imbalanced_design():
    tab = estimands(_imbalanced_fit(), "g")
    assert not np.allclose(tab["emmean[equal]"], tab["emmean[cells]"])


def test_schemes_coincide_on_balanced_design():
    tab = estimands(_balanced_fit(), "g")
    np.testing.assert_allclose(tab["emmean[equal]"], tab["emmean[proportional]"], atol=1e-9)
    np.testing.assert_allclose(tab["emmean[equal]"], tab["emmean[cells]"], atol=1e-9)


def test_factor_columns_and_shape():
    tab = estimands(_imbalanced_fit(), "g")
    assert "g" in tab.columns
    assert list(tab.columns) == ["g", "emmean[equal]", "emmean[proportional]", "emmean[cells]"]
    assert len(tab) == 2  # one row per level of g


def test_se_option_adds_se_columns():
    tab = estimands(_imbalanced_fit(), "g", se=True)
    for w in ("equal", "proportional", "cells"):
        assert f"SE[{w}]" in tab.columns
        assert (tab[f"SE[{w}]"] > 0).all()


def test_single_scheme_string():
    tab = estimands(_imbalanced_fit(), "g", schemes="proportional")
    assert list(tab.columns) == ["g", "emmean[proportional]"]


def test_by_groups_expand_rows():
    fit = _imbalanced_fit()
    tab = estimands(fit, "g", by="h")
    assert {"g", "h"}.issubset(tab.columns)
    assert len(tab) == 4  # g (2) x h (2)


def test_empty_schemes_rejected():
    with pytest.raises(ValueError, match="at least one"):
        estimands(_imbalanced_fit(), "g", schemes=())


def test_describe_points_to_estimands_for_equal_weights():
    txt = emmeans(_imbalanced_fit(), "g").describe()
    assert "balanced / experimental estimand" in txt
    assert "estimands()" in txt
