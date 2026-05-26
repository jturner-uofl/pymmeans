"""Tests for the `weights` argument on emmeans()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import emmeans


def _unbalanced_two_way():
    """g has 3 levels; h has 2 levels; cells are unbalanced."""
    rng = np.random.default_rng(0)
    # Heavy on g=a, balanced on h
    rows = (
        [("a", "x"), ("a", "y")] * 50
        + [("b", "x"), ("b", "y")] * 20
        + [("c", "x"), ("c", "y")] * 5
    )
    df = pd.DataFrame(rows, columns=["g", "h"])
    df["g"] = pd.Categorical(df["g"], categories=["a", "b", "c"])
    df["h"] = pd.Categorical(df["h"], categories=["x", "y"])
    df["y"] = rng.normal(size=len(df))
    return df


def test_equal_weights_is_default(_unbalanced_two_way_fixture=None):
    df = _unbalanced_two_way()
    fit = smf.ols("y ~ g + h", data=df).fit()
    emm_default = emmeans(fit, "g")
    emm_equal = emmeans(fit, "g", weights="equal")
    np.testing.assert_array_almost_equal(
        emm_default.frame["emmean"].to_numpy(),
        emm_equal.frame["emmean"].to_numpy(),
    )


def test_proportional_differs_from_equal_on_unbalanced():
    df = _unbalanced_two_way()
    fit = smf.ols("y ~ g + h", data=df).fit()
    emm_eq = emmeans(fit, "g", weights="equal")
    emm_prop = emmeans(fit, "g", weights="proportional")
    # h has equal counts in this fixture so g EMMs shouldn't differ much,
    # but the API path should still produce a result.
    assert emm_eq.n_rows == emm_prop.n_rows == 3
    # Now flip: g EMMs against an asymmetric h. Build new fixture.
    rng = np.random.default_rng(1)
    rows = (
        [("a", "x")] * 80
        + [("a", "y")] * 20
        + [("b", "x")] * 30
        + [("b", "y")] * 70
    )
    df2 = pd.DataFrame(rows, columns=["g", "h"])
    df2["g"] = pd.Categorical(df2["g"], categories=["a", "b"])
    df2["h"] = pd.Categorical(df2["h"], categories=["x", "y"])
    df2["y"] = (
        1.0
        + 0.5 * (df2["g"] == "b")
        + 1.0 * (df2["h"] == "y")
        + rng.normal(scale=0.3, size=len(df2))
    )
    fit2 = smf.ols("y ~ g + h", data=df2).fit()
    emm2_eq = emmeans(fit2, "g", weights="equal")
    emm2_prop = emmeans(fit2, "g", weights="proportional")
    # h is much more "x" in g=a (80%) and much more "y" in g=b (70%); for an
    # additive model, "proportional" weights tilt the g EMM toward h's
    # marginal mean (which is 50/50 in this fixture) and the equal weights
    # give a different number. Allow the comparison to be order-1 different.
    assert not np.allclose(
        emm2_eq.frame["emmean"].to_numpy(),
        emm2_prop.frame["emmean"].to_numpy(),
        atol=1e-6,
    )


def test_outer_equals_proportional_in_additive_models():
    """For additive (no-interaction) models, weights='outer' and
    weights='proportional' coincide because the interaction terms that
    would differentiate them are absent."""
    df = _unbalanced_two_way()
    fit = smf.ols("y ~ g + h", data=df).fit()
    emm_prop = emmeans(fit, "g", weights="proportional")
    emm_outer = emmeans(fit, "g", weights="outer")
    np.testing.assert_array_almost_equal(
        emm_prop.frame["emmean"].to_numpy(),
        emm_outer.frame["emmean"].to_numpy(),
    )


def test_proportional_differs_from_outer_with_two_correlated_nontargets():
    """When TWO non-target factors are correlated AND a term in the model
    involves their interaction, R's 'proportional' (joint cell counts)
    and 'outer' (product of marginals) differ. With a single non-target,
    they coincide, which is why our 2-factor test was misleading."""
    rng = np.random.default_rng(42)
    # Three factors; b and c strongly correlated
    n = 600
    a = rng.choice(["a1", "a2", "a3"], n)
    b = rng.choice(["b1", "b2"], n)
    # c follows b with high probability
    c_prob = np.where(b == "b1", 0.8, 0.2)
    c = np.where(rng.uniform(size=n) < c_prob, "c1", "c2")
    df = pd.DataFrame({"a": a, "b": b, "c": c})
    df["a"] = pd.Categorical(df["a"])
    df["b"] = pd.Categorical(df["b"])
    df["c"] = pd.Categorical(df["c"])
    df["y"] = (
        (df["a"] == "a2") * 0.6
        - (df["a"] == "a3") * 0.3
        + (df["b"] == "b2") * 1.0
        + (df["c"] == "c2") * 0.7
        + (df["b"] == "b2") * (df["c"] == "c2") * 1.5 # b:c interaction
        + rng.normal(scale=0.3, size=n)
    )
    fit = smf.ols("y ~ a + b * c", data=df).fit()
    prop = emmeans(fit, "a", weights="proportional").frame["emmean"].to_numpy()
    out = emmeans(fit, "a", weights="outer").frame["emmean"].to_numpy()
    # The b:c interaction term sees correlated b and c, so joint vs outer
    # weighting gives different marginal means for a.
    assert not np.allclose(prop, out, atol=1e-3), (
        f"proportional and outer should differ but got prop={prop} out={out}"
    )


def test_cells_matches_observed_cell_means_on_unbalanced():
    """`weights='cells'` should reproduce the observed
    cell means per target level — the classic unweighted-ANOVA-on-
    unbalanced-data "marginal mean of the data" reading. Manual
    `df.groupby('g')['y'].mean()` is the reference."""
    df = _unbalanced_two_way()
    fit = smf.ols("y ~ g * h", data=df).fit()
    em = emmeans(fit, "g", weights="cells")
    manual = df.groupby("g", observed=False)["y"].mean()
    np.testing.assert_allclose(
        em.frame["emmean"].to_numpy(),
        manual.reindex(em.frame["g"]).to_numpy(),
        atol=1e-12,
    )


def test_invalid_weights_raises():
    df = _unbalanced_two_way()
    fit = smf.ols("y ~ g + h", data=df).fit()
    with pytest.raises(ValueError, match="weights must be"):
        emmeans(fit, "g", weights="nonexistent")
