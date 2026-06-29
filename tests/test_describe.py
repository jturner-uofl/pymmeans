"""EMMResult.describe() — the "self-describing result" sweetener cluster.

Covers wish-list items #1 (active scale, averaged-over factors + weights,
held covariates), #3 (named non-estimable cells), #5 (interaction -> by=
hint), and #6 (comparison guidance). See docs/wishlist.md.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymmeans import emmeans


def _ols_interaction():
    rng = np.random.default_rng(0)
    n = 400
    d = pd.DataFrame({
        "g": pd.Categorical(rng.choice(["A", "B"], n)),
        "block": pd.Categorical(rng.choice(["x", "y", "z"], n)),
        "age": rng.normal(40, 8, n),
    })
    d["y"] = (d["g"].map({"A": 0, "B": 1}).astype(float)
              + 0.3 * (d["block"] == "y") + 0.05 * d["age"]
              + rng.standard_normal(n))
    return smf.ols("y ~ g*block + age", d).fit()


def test_describe_reports_scale_averaging_and_held_covariate():
    txt = emmeans(_ols_interaction(), "g").describe()
    assert "Estimated marginal means of g" in txt
    assert "Scale: response scale" in txt
    assert "Averaged over: block" in txt and "weights: equal" in txt
    assert "Covariates held fixed: age =" in txt


def test_describe_flags_interaction_with_by_suggestion():
    txt = emmeans(_ols_interaction(), "g").describe()
    assert "block interacts with the target" in txt
    assert "by='block'" in txt


def test_describe_no_interaction_note_when_additive():
    rng = np.random.default_rng(0)
    n = 300
    d = pd.DataFrame({
        "g": pd.Categorical(rng.choice(["A", "B"], n)),
        "block": pd.Categorical(rng.choice(["x", "y"], n)),
    })
    d["y"] = d["g"].map({"A": 0, "B": 1}).astype(float) + rng.standard_normal(n)
    fit = smf.ols("y ~ g + block", d).fit()  # additive, no interaction
    txt = emmeans(fit, "g").describe()
    assert "interacts with the target" not in txt
    assert "Averaged over: block" in txt


def test_describe_link_vs_response_scale():
    rng = np.random.default_rng(2)
    n = 300
    d = pd.DataFrame({"g": pd.Categorical(rng.choice(["A", "B"], n))})
    d["yb"] = (rng.random(n) < 0.5).astype(int)
    fit = smf.glm("yb ~ g", d, family=sm.families.Binomial()).fit()
    link_txt = emmeans(fit, "g").describe()
    resp_txt = emmeans(fit, "g", type="response").describe()
    assert "link) scale [logit]" in link_txt and "NOT the response" in link_txt
    assert "response scale (back-transformed" in resp_txt


def test_describe_names_non_estimable_cells():
    rng = np.random.default_rng(1)
    n = 300
    d = pd.DataFrame({
        "g": pd.Categorical(rng.choice(["A", "B", "C"], n)),
        "blk": pd.Categorical(rng.choice(["x", "y"], n)),
    })
    d = d[~((d["g"] == "C") & (d["blk"] == "y"))].copy()  # empty cell
    d["y"] = d["g"].cat.codes * 0.5 + (d["blk"] == "y") * 0.3 + rng.standard_normal(len(d))
    fit = smf.ols("y ~ g*blk", d).fit()
    txt = emmeans(fit, ["g", "blk"]).describe()
    assert "Non-estimable cells" in txt
    assert "g=C, blk=y" in txt


def test_describe_always_gives_comparison_guidance():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"g": pd.Categorical(rng.choice(["A", "B", "C"], 150))})
    d["y"] = d["g"].cat.codes + rng.standard_normal(150)
    fit = smf.ols("y ~ g", d).fit()
    txt = emmeans(fit, "g").describe()
    assert "use pairs()/contrast()" in txt
    assert "overlapping confidence intervals" in txt
    # minimal model: no covariates / no averaging lines
    assert "Covariates held fixed" not in txt
    assert "Averaged over" not in txt


def test_describe_reports_satterthwaite_df():
    import statsmodels.regression.mixed_linear_model as mlm

    rng = np.random.default_rng(3)
    n, ng = 200, 12
    grp = rng.integers(0, ng, n)
    d = pd.DataFrame({
        "g": pd.Categorical(rng.choice(["A", "B"], n)),
        "subj": grp,
    })
    d["y"] = (d["g"].map({"A": 0.0, "B": 1.0}).astype(float)
              + rng.standard_normal(ng)[grp] + rng.standard_normal(n))
    fit = mlm.MixedLM.from_formula("y ~ g", groups="subj", data=d).fit()
    from pymmeans import apply_satterthwaite
    txt = apply_satterthwaite(emmeans(fit, "g")).describe()
    assert "Satterthwaite" in txt
