"""Tests for estimability handling under rank-deficient designs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import emmeans
from pymmeans.estimability import estimable_mask, null_space_basis


def test_estimable_mask_full_rank():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 5))
    L = rng.normal(size=(10, 5))
    mask = estimable_mask(L, X)
    assert mask.all()


def test_estimable_mask_rank_deficient():
    """An L row pointing along a null-space direction must be flagged."""
    rng = np.random.default_rng(1)
    base = rng.normal(size=(50, 4))
    # Make column 4 a duplicate of column 0 — explicit rank deficiency
    X = np.column_stack([base, base[:, 0]])
    null = null_space_basis(X)
    assert null.shape[1] == 1
    # A contrast aligned with the null direction is not estimable
    L = np.vstack([np.eye(5)[0], null.T[0], np.ones(5)])
    mask = estimable_mask(L, X)
    # First row (1,0,0,0,0) projects onto duplicated dim -> not estimable
    # Second row exactly null -> not estimable
    # Third row (1,1,1,1,1) — also has component in null space? Check:
    # null direction is e_0 - e_4 (or similar). 1·1 + 1·(-1) = 0 — orthogonal.
    assert not mask[1]


def test_emmeans_warns_for_rank_deficient_design():
    """this test used to only verify that emmeans didn't
    crash, not that it surfaced a warning or NaN. Now we assert both."""
    n = 60
    rng = np.random.default_rng(2)
    g1 = rng.choice(["a", "b"], n)
    g2 = np.where(g1 == "a", "x", "y") # g2 perfectly co-varies with g1
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "g1": pd.Categorical(g1),
            "g2": pd.Categorical(g2),
        }
    )
    fit = smf.ols("y ~ g1 + g2", data=df).fit()
    with pytest.warns(UserWarning, match="not estimable"):
        emm = emmeans(fit, "g1")
    # The non-estimable rows should be marked NaN
    assert emm.frame["emmean"].isna().any()


def test_estimable_mask_flags_missing_interaction_cell():
    """y ~ a*b with a missing (a=hi, b=z) cell makes the corresponding
    EMM non-estimable — interaction-cell smoke test."""
    rng = np.random.default_rng(42)
    # Build a clean grid with one cell missing
    rows = [("lo", "x"), ("lo", "y"), ("lo", "z"), ("hi", "x"), ("hi", "y")]
    df = pd.DataFrame(
        [(a, b) for (a, b) in rows for _ in range(15)], columns=["a", "b"]
    )
    df["a"] = pd.Categorical(df["a"], categories=["lo", "hi"])
    df["b"] = pd.Categorical(df["b"], categories=["x", "y", "z"])
    df["y"] = rng.normal(size=len(df))
    fit = smf.ols("y ~ a * b", data=df).fit()
    with pytest.warns(UserWarning, match="not estimable"):
        emm = emmeans(fit, "a", by="b")
    # 6 rows total (2 a x 3 b); one should be NaN (the missing (hi, z) cell)
    assert emm.frame["emmean"].isna().sum() >= 1
