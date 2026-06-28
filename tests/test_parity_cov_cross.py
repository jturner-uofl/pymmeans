"""Parity closers: bare-callable cov_reduce and cross_adjust (cross-group
multiplicity), validated against R emmeans' rules."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import emmeans, pairs, ref_grid, summary

# ---------------------------------------------------------------------- cov_reduce


def _cov_fit(n=200, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "g": pd.Categorical(rng.choice(["A", "B"], n)),
        "z": rng.standard_normal(n) * 3 + 5,
    })
    base = df["g"].map({"A": 0.0, "B": 1.0}).astype(float)
    df["y"] = base + 0.5 * df["z"] + rng.standard_normal(n)
    return smf.ols("y ~ g + z", df).fit(), df


def test_cov_reduce_bare_callable_applies_to_all_numerics():
    """R-style ``cov_reduce=np.median`` evaluates every numeric covariate
    at its median; the EMM shifts by coef * (median - mean)."""
    fit, df = _cov_fit()
    at_mean = float(emmeans(fit, "g").frame.set_index("g").loc["A", "emmean"])
    at_med = float(emmeans(fit, "g", cov_reduce=np.median).frame.set_index("g").loc["A", "emmean"])
    expected = float(fit.params["z"]) * (df["z"].median() - df["z"].mean())
    assert (at_med - at_mean) == pytest.approx(expected, abs=1e-9)


def test_cov_reduce_bare_callable_equals_dict_form():
    fit, _ = _cov_fit()
    bare = emmeans(fit, "g", cov_reduce=np.median).frame["emmean"].to_numpy()
    asdict = emmeans(fit, "g", cov_reduce={"z": np.median}).frame["emmean"].to_numpy()
    np.testing.assert_allclose(bare, asdict, atol=1e-12)


def test_ref_grid_accepts_bare_callable():
    fit, _ = _cov_fit()
    assert ref_grid(fit, cov_reduce=np.median) is not None


def test_cov_reduce_rejects_non_callable_non_dict():
    fit, _ = _cov_fit()
    with pytest.raises(TypeError, match="cov_reduce"):
        emmeans(fit, "g", cov_reduce=3.0)


# ---------------------------------------------------------------------- cross_adjust


def _cross_fit(n=120, seed=3):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "t": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        "g": pd.Categorical(rng.choice(["G1", "G2"], n)),
    })
    tmap = df["t"].map({"a": 0.6, "b": 1.2, "c": 1.8}).astype(float)
    df["y"] = tmap + (df["g"] == "G2") * 0.4 + rng.standard_normal(n)
    return smf.ols("y ~ t * g", df).fit()


def test_cross_adjust_bonferroni_multiplies_by_n_groups():
    """R rule: cross.adjust='bonferroni' multiplies each within-by-adjusted
    p by the number of by-groups G (capped at 1)."""
    fit = _cross_fit()
    ct = pairs(emmeans(fit, "t", by="g"))
    within = summary(ct, adjust="tukey")
    crossed = summary(ct, adjust="tukey", cross_adjust="bonferroni")
    g = within["g"].nunique()
    key = ["contrast", "g"]
    w = within.set_index(key)["p_value"]
    c = crossed.set_index(key)["p_value"]
    expected = np.minimum(1.0, w * g)
    np.testing.assert_allclose(c.to_numpy(), expected.loc[c.index].to_numpy(), atol=1e-9)


def test_cross_adjust_sidak_is_one_minus_complement_power_g():
    """R rule: cross.adjust='sidak' gives 1 - (1 - p)^G."""
    fit = _cross_fit()
    ct = pairs(emmeans(fit, "t", by="g"))
    within = summary(ct, adjust="tukey")
    crossed = summary(ct, adjust="tukey", cross_adjust="sidak")
    g = within["g"].nunique()
    key = ["contrast", "g"]
    w = within.set_index(key)["p_value"]
    c = crossed.set_index(key)["p_value"]
    expected = 1.0 - (1.0 - w) ** g
    np.testing.assert_allclose(c.to_numpy(), expected.loc[c.index].to_numpy(), atol=1e-9)


def test_cross_adjust_none_is_noop():
    fit = _cross_fit()
    ct = pairs(emmeans(fit, "t", by="g"))
    a = summary(ct, adjust="tukey")["p_value"].to_numpy()
    b = summary(ct, adjust="tukey", cross_adjust=None)["p_value"].to_numpy()
    np.testing.assert_allclose(a, b, atol=1e-12)
