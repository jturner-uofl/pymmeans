"""Tests for formula-expression support (C(), np.log(), etc.)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import emmeans, pairs
from pymmeans.utils import from_statsmodels


def test_emmeans_with_C_expression():
    rng = np.random.default_rng(0)
    n = 90
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n) + np.repeat([0, 1, 2], n // 3),
            "percent": np.repeat([9, 12, 15], n // 3),
            "source": pd.Categorical(rng.choice(["fish", "soy"], n)),
        }
    )
    fit = smf.ols("y ~ C(percent) + source", data=df).fit()

    # User uses the underlying column name; aliases resolve it.
    emm = emmeans(fit, "percent")
    assert emm.n_rows == 3

    # User can also use the patsy-canonical name.
    emm2 = emmeans(fit, "C(percent)")
    np.testing.assert_array_almost_equal(
        emm.frame["emmean"].to_numpy(), emm2.frame["emmean"].to_numpy()
    )


def test_emmeans_with_C_matches_pre_converted():
    rng = np.random.default_rng(1)
    n = 60
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "dose": np.repeat([0.5, 1.0, 2.0], n // 3),
            "supp": pd.Categorical(rng.choice(["OJ", "VC"], n)),
        }
    )
    fit_expr = smf.ols("y ~ C(dose) * supp", data=df).fit()

    df2 = df.copy()
    df2["dose_cat"] = pd.Categorical(df2["dose"])
    fit_plain = smf.ols("y ~ dose_cat * supp", data=df2).fit()

    emm_expr = emmeans(fit_expr, "dose").frame["emmean"].to_numpy()
    emm_plain = emmeans(fit_plain, "dose_cat").frame["emmean"].to_numpy()
    np.testing.assert_array_almost_equal(emm_expr, emm_plain)


def test_aliases_populated_for_C_expression():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=60),
            "p": np.repeat([10, 20, 30], 20),
        }
    )
    fit = smf.ols("y ~ C(p)", data=df).fit()
    info = from_statsmodels(fit)
    assert info.aliases == {"p": "C(p)"}
    assert "C(p)" in info.factors


def test_emmeans_with_np_log_numeric():
    rng = np.random.default_rng(3)
    n = 90
    x = rng.uniform(1.0, 5.0, n)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n) + np.repeat([0, 1, 2], n // 3),
            "g": pd.Categorical(np.repeat(["a", "b", "c"], n // 3)),
            "x": x,
        }
    )
    fit = smf.ols("y ~ g + np.log(x)", data=df).fit()

    emm = emmeans(fit, "g")
    assert emm.n_rows == 3
    # Compare to manually-computed prediction at mean(log(x))
    mean_log_x = float(np.log(df["x"]).mean())
    info = from_statsmodels(fit)
    expected = info.beta[0] + np.array(
        [
            0,
            info.beta[info.param_names.index("g[T.b]")],
            info.beta[info.param_names.index("g[T.c]")],
        ]
    ) + info.beta[info.param_names.index("np.log(x)")] * mean_log_x
    np.testing.assert_array_almost_equal(
        emm.frame["emmean"].to_numpy(), expected
    )


def test_pigs_canonical_formula_works_directly():
    # Without pre-conversion: smf.ols('log(conc) ~ source + factor(percent)'...)
    # patsy doesn't accept R's factor(); we use C() which is patsy's idiom.
    rng = np.random.default_rng(4)
    n = 60
    df = pd.DataFrame(
        {
            "conc": np.exp(rng.normal(size=n)),
            "source": pd.Categorical(rng.choice(["fish", "soy", "skim"], n)),
            "percent": np.repeat([9, 12, 15, 18], n // 4),
        }
    )
    # Canonical Python form: log on the LHS, C() for the categorical on RHS
    fit = smf.ols("np.log(conc) ~ source + C(percent)", data=df).fit()
    pw = pairs(emmeans(fit, "source"))
    assert pw.n_rows == 3


def test_streaming_path_still_rejects_expressions():
    # Explicit chunk_size= forces the streaming path which can't handle
    # expression factors yet; should raise NotImplementedError.
    rng = np.random.default_rng(5)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=40),
            "p": np.repeat([1, 2, 3, 4], 10),
        }
    )
    fit = smf.ols("y ~ C(p)", data=df).fit()
    with pytest.raises(NotImplementedError, match="plain"):
        emmeans(fit, "p", chunk_size=10)
