"""Tests for avg_comparisons() / comparisons() — counterfactual contrasts."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymmeans import ComparisonsResult, avg_comparisons, comparisons

# ---------------------------------------------------------------------- fixtures


def _ols(n=500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x": rng.standard_normal(n),
        "z": rng.standard_normal(n),
    })
    df["y"] = 2.0 + 1.5 * df["x"] - 0.7 * df["z"] + rng.standard_normal(n)
    return smf.ols("y ~ x + z", df).fit(), df


def _logit(n=500, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x": rng.standard_normal(n),
        "z": rng.standard_normal(n),
        "g": pd.Categorical(rng.choice(["A", "B", "C"], n)),
    })
    eta = 0.3 + 0.8 * df["x"] - 0.5 * df["z"]
    df["yb"] = (rng.random(n) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    return smf.glm("yb ~ x + z + g", df, family=sm.families.Binomial()).fit(), df


# ---------------------------------------------------------------------- G1: closed form


def test_linear_difference_equals_coefficient():
    """OLS difference of a numeric var over a unit step == its coefficient."""
    fit, _ = _ols()
    res = avg_comparisons(fit, "x", type="link")
    assert float(res.frame["estimate"].iloc[0]) == pytest.approx(
        float(fit.params["x"]), abs=1e-9
    )


def test_linear_difference_scales_with_step():
    """Difference over step s == s * coefficient, exactly (linear model)."""
    fit, _ = _ols()
    res = avg_comparisons(fit, "x", type="link", step=3.0)
    assert float(res.frame["estimate"].iloc[0]) == pytest.approx(
        3.0 * float(fit.params["x"]), abs=1e-9
    )


def test_linear_difference_se_equals_step_times_coef_se():
    """And its SE == step * coefficient SE, exactly."""
    fit, _ = _ols()
    res = avg_comparisons(fit, "x", type="link", step=2.0)
    assert float(res.frame["SE"].iloc[0]) == pytest.approx(
        2.0 * float(fit.bse["x"]), abs=1e-8
    )


def test_logit_difference_matches_gcomputation():
    """Logit response difference == centred g-computation prediction diff."""
    fit, df = _logit()
    res = avg_comparisons(fit, "x", comparison="difference")
    dhi = df.copy(); dhi["x"] = df["x"] + 0.5
    dlo = df.copy(); dlo["x"] = df["x"] - 0.5
    manual = float((fit.predict(dhi) - fit.predict(dlo)).mean())
    assert float(res.frame["estimate"].iloc[0]) == pytest.approx(manual, abs=1e-9)


def test_categorical_difference_matches_counterfactual():
    """Categorical B-A == counterfactual all-rows level prediction diff."""
    fit, df = _logit()
    res = avg_comparisons(fit, "g").frame.set_index("contrast")
    n = len(df)
    dB = df.copy(); dB["g"] = pd.Categorical(["B"] * n, categories=["A", "B", "C"])
    dA = df.copy(); dA["g"] = pd.Categorical(["A"] * n, categories=["A", "B", "C"])
    manual = float((fit.predict(dB) - fit.predict(dA)).mean())
    assert float(res.loc["B - A", "estimate"]) == pytest.approx(manual, abs=1e-9)


def test_ratio_is_ratio_of_means():
    """comparison='ratio' on avg_comparisons is mean(hi)/mean(lo)."""
    fit, df = _logit()
    res = avg_comparisons(fit, "x", comparison="ratio")
    dhi = df.copy(); dhi["x"] = df["x"] + 0.5
    dlo = df.copy(); dlo["x"] = df["x"] - 0.5
    manual = float(fit.predict(dhi).mean() / fit.predict(dlo).mean())
    assert float(res.frame["estimate"].iloc[0]) == pytest.approx(manual, abs=1e-9)


def test_comparisons_perrow_mean_difference_equals_avg():
    """For 'difference', mean of per-row comparisons == avg_comparisons."""
    fit, _ = _logit()
    per = comparisons(fit, "x", comparison="difference")
    avg = avg_comparisons(fit, "x", comparison="difference")
    assert per.frame["estimate"].mean() == pytest.approx(
        float(avg.frame["estimate"].iloc[0]), abs=1e-9
    )


# ---------------------------------------------------------------------- G2: marginaleffects


def test_cross_validate_against_marginaleffects():
    """avg_comparisons estimate + SE match marginaleffects-py across every
    comparison function (numeric and categorical)."""
    me = pytest.importorskip("marginaleffects")
    fit, _ = _logit()
    for cmp in ("difference", "ratio", "lnratio", "lnor", "lift"):
        pm = avg_comparisons(fit, "x", comparison=cmp).frame.iloc[0]
        ref = me.avg_comparisons(fit, variables="x", comparison=cmp).to_pandas().iloc[0]
        assert float(pm["estimate"]) == pytest.approx(float(ref["estimate"]), abs=1e-7), cmp
        assert float(pm["SE"]) == pytest.approx(float(ref["std_error"]), abs=1e-5), cmp
    # categorical, each level vs reference
    pmg = avg_comparisons(fit, "g").frame.set_index("contrast")
    refg = me.avg_comparisons(fit, variables="g").to_pandas().set_index("contrast")
    for c in ("B - A", "C - A"):
        assert float(pmg.loc[c, "estimate"]) == pytest.approx(
            float(refg.loc[c, "estimate"]), abs=1e-7
        )
        assert float(pmg.loc[c, "SE"]) == pytest.approx(
            float(refg.loc[c, "std_error"]), abs=1e-5
        )


def test_cross_validate_change_specs_against_marginaleffects():
    """The sd / 2sd / iqr / minmax / (lo,hi) change specs match
    marginaleffects-py (estimate and standard error)."""
    me = pytest.importorskip("marginaleffects")
    fit, _ = _logit()
    for spec in ("sd", "2sd", "iqr", "minmax", [0.0, 1.0]):
        pm = avg_comparisons(fit, variables={"x": spec}).frame.iloc[0]
        ref = me.avg_comparisons(fit, variables={"x": spec}).to_pandas().iloc[0]
        assert float(pm["estimate"]) == pytest.approx(
            float(ref["estimate"]), abs=1e-7
        ), spec
        assert float(pm["SE"]) == pytest.approx(float(ref["std_error"]), abs=1e-5), spec


def test_change_specs_closed_form():
    """sd/2sd are centred per-row SD steps; iqr/minmax/(lo,hi) are absolute."""
    fit, df = _logit()
    x = df["x"].to_numpy()
    sd = float(np.std(x, ddof=1))
    cases = {
        "sd": (x - sd / 2, x + sd / 2),
        "2sd": (x - sd, x + sd),
        "iqr": (np.quantile(x, 0.25), np.quantile(x, 0.75)),
        "minmax": (x.min(), x.max()),
    }
    for spec, (lo, hi) in cases.items():
        dl = df.copy(); dl["x"] = lo
        dh = df.copy(); dh["x"] = hi
        manual = float((fit.predict(dh) - fit.predict(dl)).mean())
        pm = float(avg_comparisons(fit, variables={"x": spec}).frame["estimate"].iloc[0])
        assert pm == pytest.approx(manual, abs=1e-9), spec


def test_callable_comparison_matches_named():
    fit, _ = _logit()
    named = float(avg_comparisons(fit, "x", comparison="difference").frame["estimate"].iloc[0])
    custom = float(
        avg_comparisons(fit, "x", comparison=lambda hi, lo: hi - lo).frame["estimate"].iloc[0]
    )
    assert custom == pytest.approx(named, abs=1e-12)


def test_rejects_unknown_change_spec():
    fit, _ = _logit()
    with pytest.raises(ValueError, match="change spec"):
        avg_comparisons(fit, variables={"x": "nonsense"})


# ---------------------------------------------------------------------- by-grouping


def test_by_grouping_returns_row_per_level():
    fit, _ = _logit()
    res = avg_comparisons(fit, "x", by="g")
    assert len(res.frame) == 3
    assert set(res.frame["g"]) == {"A", "B", "C"}


# ---------------------------------------------------------------------- validation


def test_rejects_unknown_comparison():
    fit, _ = _ols()
    with pytest.raises(ValueError, match="comparison"):
        avg_comparisons(fit, "x", comparison="nope")


def test_rejects_invalid_level():
    fit, _ = _ols()
    with pytest.raises(ValueError, match="level"):
        avg_comparisons(fit, "x", level=1.0)


def test_rejects_unknown_var():
    fit, _ = _logit()
    with pytest.raises(ValueError, match="not a column"):
        avg_comparisons(fit, "nope")


def test_response_on_linear_model_raises():
    fit, _ = _ols()
    with pytest.raises(ValueError, match="response"):
        avg_comparisons(fit, "x", type="response")


def test_default_variables_covers_all_predictors():
    """variables=None reports every predictor (numeric +1, categorical levels)."""
    fit, _ = _logit()
    res = avg_comparisons(fit)
    terms = set(res.frame["term"])
    assert {"x", "z", "g"} <= terms


def test_returns_documented_columns():
    fit, _ = _ols()
    res = avg_comparisons(fit, "x", type="link")
    assert isinstance(res, ComparisonsResult)
    for col in (
        "term", "contrast", "estimate", "SE", "df",
        "t_ratio", "p_value", "lower_cl", "upper_cl",
    ):
        assert col in res.frame.columns
