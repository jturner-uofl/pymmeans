"""Tests for linearmodels (PanelOLS) support."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

linearmodels = pytest.importorskip("linearmodels")

from pymmeans import (  # noqa: E402  # import after importorskip guard
    emmeans,
    from_linearmodels,
    pairs,
)


def _panel():
    rng = np.random.default_rng(0)
    n_entity = 40
    n_time = 5
    n = n_entity * n_time
    df = pd.DataFrame(
        {
            "entity": np.repeat(np.arange(n_entity), n_time),
            "time": np.tile(np.arange(n_time), n_entity),
            "x": rng.normal(size=n),
            "g": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        }
    )
    df["y"] = (
        0.5 * df["x"]
        + (df["g"] == "b")
        - 0.5 * (df["g"] == "c")
        + rng.normal(scale=0.3, size=n)
    )
    return df.set_index(["entity", "time"])


def test_emmeans_works_on_panelols():
    from linearmodels import PanelOLS

    df = _panel()
    fit = PanelOLS.from_formula("y ~ 1 + x + g", df).fit()
    info = from_linearmodels(fit, data=df)
    emm = emmeans(info, "g")
    assert emm.n_rows == 3
    g_levels = list(emm.frame["g"].astype(str))
    means = dict(zip(g_levels, emm.frame["emmean"].to_numpy(), strict=True))
    assert means["b"] > means["a"]
    assert means["c"] < means["a"]


def test_pairs_on_panelols():
    from linearmodels import PanelOLS

    df = _panel()
    fit = PanelOLS.from_formula("y ~ 1 + x + g", df).fit()
    info = from_linearmodels(fit, data=df)
    pw = pairs(emmeans(info, "g"))
    assert pw.n_rows == 3
    assert (pw.frame["p_value"] >= 0).all()


def test_linearmodels_rejects_no_intercept_formula():
    from linearmodels import PanelOLS

    df = _panel()
    fit = PanelOLS.from_formula("y ~ x + g", df).fit()
    with pytest.raises(ValueError, match="explicit '1' intercept"):
        from_linearmodels(fit, data=df)
