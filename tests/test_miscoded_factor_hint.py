"""Mistake-catching guidance (wish-list sweetener #2): when a focal variable
is a numeric covariate that looks like a miscoded factor (few distinct
values), the numeric-target error leads with the "make it a factor" fix.

R emmeans gives a silent one-row answer here and will not warn
(rvlenth/emmeans #523, wontfix); pymmeans names the most probable cause.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import emmeans


def _num_focal_fit(levels, n=200, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"grp": rng.choice(levels, n)})
    d["y"] = d["grp"] * 0.7 + rng.standard_normal(n)
    return smf.ols("y ~ grp", d).fit()


def test_miscoded_factor_error_names_the_factor_fix():
    fit = _num_focal_fit([1, 2, 3])
    with pytest.raises(ValueError) as exc:
        emmeans(fit, "grp")
    msg = str(exc.value)
    assert "C(grp)" in msg
    assert "3 distinct values" in msg
    assert "categorical" in msg


def test_continuous_covariate_gets_generic_message_not_factor_hint():
    """A genuine continuous covariate (many distinct values) keeps the
    at=/emtrends guidance and does NOT get the spurious 'make it a factor'
    hint."""
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"x": rng.standard_normal(200)})
    d["y"] = 2 * d["x"] + rng.standard_normal(200)
    fit = smf.ols("y ~ x", d).fit()
    with pytest.raises(ValueError) as exc:
        emmeans(fit, "x")
    msg = str(exc.value)
    assert "emtrends" in msg
    assert "C(x)" not in msg
    assert "distinct values" not in msg


def test_threshold_many_levels_is_treated_as_continuous():
    """A numeric covariate with > _CATEGORICAL_NUMERIC_MAX_LEVELS distinct
    values is continuous-like → generic message, no factor hint."""
    fit = _num_focal_fit(list(range(40)))  # 40 distinct values
    with pytest.raises(ValueError) as exc:
        emmeans(fit, "grp")
    assert "C(grp)" not in str(exc.value)


def test_proper_factor_focal_still_works():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"grp": pd.Categorical(rng.choice([1, 2, 3], 200))})
    d["y"] = d["grp"].astype(int) * 0.7 + rng.standard_normal(200)
    fit = smf.ols("y ~ grp", d).fit()
    assert len(emmeans(fit, "grp").frame) == 3


def test_at_sweep_on_numeric_focal_still_works():
    fit = _num_focal_fit([1, 2, 3])
    assert len(emmeans(fit, "grp", at={"grp": [1, 2, 3]}).frame) == 3
