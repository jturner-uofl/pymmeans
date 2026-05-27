"""Tests for pairs() and contrast()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from pymmeans.contrasts import ContrastResult, contrast, pairs
from pymmeans.emmeans import emmeans


@pytest.fixture
def one_way():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=90) + np.repeat([0.0, 0.5, 1.0], 30),
            "g": pd.Categorical(np.repeat(["a", "b", "c"], 30)),
        }
    )
    return smf.ols("y ~ g", data=df).fit()


@pytest.fixture
def by_grouped():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=120),
            "a": pd.Categorical(rng.choice(["lo", "hi"], 120)),
            "b": pd.Categorical(rng.choice(["x", "y", "z"], 120)),
        }
    )
    return smf.ols("y ~ a * b", data=df).fit()


def test_returns_contrastresult(one_way):
    emm = emmeans(one_way, "g")
    pw = pairs(emm)
    assert isinstance(pw, ContrastResult)


def test_pairwise_count(one_way):
    emm = emmeans(one_way, "g")
    pw = pairs(emm)
    assert pw.n_rows == 3 # 3 choose 2


def test_pairwise_labels(one_way):
    emm = emmeans(one_way, "g")
    pw = pairs(emm)
    assert list(pw.frame["contrast"]) == ["a - b", "a - c", "b - c"]


def test_pairwise_estimate_matches_emm_diff(one_way):
    emm = emmeans(one_way, "g")
    pw = pairs(emm)
    mu = emm.frame["emmean"].to_numpy()
    expected = [mu[0] - mu[1], mu[0] - mu[2], mu[1] - mu[2]]
    np.testing.assert_array_almost_equal(pw.frame["estimate"].to_numpy(), expected)


def test_reverse_flips_sign(one_way):
    emm = emmeans(one_way, "g")
    pw = pairs(emm)
    pw_rev = pairs(emm, reverse=True)
    np.testing.assert_array_almost_equal(
        pw_rev.frame["estimate"].to_numpy(), -pw.frame["estimate"].to_numpy()
    )


def test_t_ratio_and_p_value(one_way):
    emm = emmeans(one_way, "g")
    pw = pairs(emm, adjust="none")
    t_calc = pw.frame["estimate"] / pw.frame["SE"]
    np.testing.assert_array_almost_equal(pw.frame["t_ratio"], t_calc)
    p_calc = 2.0 * stats.t.sf(np.abs(t_calc), one_way.df_resid)
    np.testing.assert_array_almost_equal(pw.frame["p_value"], p_calc)


def test_by_group_pairs(by_grouped):
    emm = emmeans(by_grouped, "a", by="b")
    pw = pairs(emm)
    # 1 pair per b-level, 3 b-levels
    assert pw.n_rows == 3
    assert set(pw.frame["b"].astype(str)) == {"x", "y", "z"}
    assert list(pw.frame["contrast"]) == ["hi - lo", "hi - lo", "hi - lo"]


def test_tukey_is_default_and_differs_from_none(one_way):
    emm = emmeans(one_way, "g")
    pw_def = pairs(emm)
    pw_none = pairs(emm, adjust="none")
    assert pw_def.adjust == "tukey"
    assert not np.allclose(pw_def.frame["p_value"], pw_none.frame["p_value"])


def test_contrast_pairwise_alias(one_way):
    emm = emmeans(one_way, "g")
    a = pairs(emm)
    b = contrast(emm, method="pairwise")
    np.testing.assert_array_almost_equal(
        a.frame["estimate"].to_numpy(), b.frame["estimate"].to_numpy()
    )


def test_contrast_revpairwise(one_way):
    emm = emmeans(one_way, "g")
    fwd = pairs(emm)
    rev = contrast(emm, method="revpairwise")
    np.testing.assert_array_almost_equal(
        rev.frame["estimate"].to_numpy(), -fwd.frame["estimate"].to_numpy()
    )


def test_trt_vs_ctrl_default_ref(one_way):
    emm = emmeans(one_way, "g")
    tc = contrast(emm, method="trt.vs.ctrl")
    assert tc.n_rows == 2
    assert list(tc.frame["contrast"]) == ["b - a", "c - a"]


def test_trt_vs_ctrl_with_named_ref(one_way):
    emm = emmeans(one_way, "g")
    tc = contrast(emm, method="trt.vs.ctrl", ref="b")
    assert list(tc.frame["contrast"]) == ["a - b", "c - b"]


def test_trt_vs_ctrl_defaults_to_dunnettx(one_way):
    """trt.vs.ctrl default adjustment changed from
    `dunnett` to `dunnettx` to match R `emmeans` defaults. `dunnettx`
    is aliased to the same exact-multivariate-t Genz QMC code path
    (R uses an approximation; ours is strictly more accurate)."""
    emm = emmeans(one_way, "g")
    tc = contrast(emm, method="trt.vs.ctrl")
    assert tc.adjust == "dunnettx"


def test_dunnett_p_between_raw_and_bonferroni(one_way):
    emm = emmeans(one_way, "g")
    raw = contrast(emm, method="trt.vs.ctrl", adjust="none")
    dun = contrast(emm, method="trt.vs.ctrl", adjust="dunnett")
    bon = contrast(emm, method="trt.vs.ctrl", adjust="bonferroni")
    # Dunnett accounts for correlation; should be between raw and Bonferroni
    raw_p = raw.frame["p_value"].to_numpy()
    dun_p = dun.frame["p_value"].to_numpy()
    bon_p = bon.frame["p_value"].to_numpy()
    assert (dun_p >= raw_p - 1e-9).all()
    assert (dun_p <= bon_p + 1e-9).all()


def test_dunnett_glm():
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame(
        {
            "y": rng.integers(0, 2, n),
            "g": pd.Categorical(rng.choice(["ctrl", "a", "b"], n)),
        }
    )
    fit = smf.glm("y ~ g", data=df, family=sm.families.Binomial()).fit()
    tc = contrast(emmeans(fit, "g"), method="trt.vs.ctrl", ref="ctrl")
    # trt.vs.ctrl default is now "dunnettx" (R parity)
    assert tc.adjust == "dunnettx"
    assert tc.n_rows == 2
    assert (tc.frame["p_value"] >= 0).all() and (tc.frame["p_value"] <= 1).all()


def test_unknown_method_raises(one_way):
    emm = emmeans(one_way, "g")
    with pytest.raises(ValueError, match="Unknown contrast"):
        contrast(emm, method="magicwand")


def test_glm_pairs_uses_inf_df():
    rng = np.random.default_rng(7)
    n = 200
    df = pd.DataFrame(
        {
            "y": rng.integers(0, 2, n),
            "g": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        }
    )
    fit = smf.glm("y ~ g", data=df, family=sm.families.Binomial()).fit()
    pw = pairs(emmeans(fit, "g"), adjust="none")
    assert np.isinf(pw.frame["df"]).all()


def test_linfct_at_beta_reproduces_estimates(by_grouped):
    emm = emmeans(by_grouped, "a", by="b")
    pw = pairs(emm)
    est = pw.linfct @ emm.model_info.beta
    np.testing.assert_array_almost_equal(est, pw.frame["estimate"].to_numpy())


# --- Custom contrasts ----------------------------------------------------


def test_custom_contrast_dict(one_way):
    emm = emmeans(one_way, "g")
    c = contrast(emm, method={"a_minus_avg_bc": [1.0, -0.5, -0.5]})
    assert list(c.frame["contrast"]) == ["a_minus_avg_bc"]
    mu = emm.frame["emmean"].to_numpy()
    expected = mu[0] - 0.5 * mu[1] - 0.5 * mu[2]
    assert c.frame["estimate"].iloc[0] == pytest.approx(expected)


def test_custom_contrast_matrix(one_way):
    emm = emmeans(one_way, "g")
    D = np.array([[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    c = contrast(emm, method=D)
    assert list(c.frame["contrast"]) == ["c1", "c2"]
    mu = emm.frame["emmean"].to_numpy()
    np.testing.assert_array_almost_equal(
        c.frame["estimate"].to_numpy(), [mu[0] - mu[1], mu[1] - mu[2]]
    )


def test_custom_contrast_wrong_length_raises(one_way):
    emm = emmeans(one_way, "g") # 3 levels
    with pytest.raises(ValueError, match="does not match"):
        contrast(emm, method={"bad": [1.0, -1.0]}) # length 2


def test_custom_contrast_within_by_groups(by_grouped):
    emm = emmeans(by_grouped, "a", by="b") # 2 a-levels per b-group
    c = contrast(emm, method={"hi_vs_lo": [1.0, -1.0]})
    assert c.n_rows == 3 # one per b level
    assert (c.frame["contrast"] == "hi_vs_lo").all()


# --- Polynomial / consecutive contrasts ---------------------------------


def test_poly_contrast_labels_and_count(one_way):
    emm = emmeans(one_way, "g") # 3 levels
    c = contrast(emm, method="poly")
    assert list(c.frame["contrast"]) == ["linear", "quadratic"]


def test_poly_contrast_linear_is_endpoint_diff(one_way):
    """poly now uses R `emmeans::poly.emmc`
    integer-scaled contrasts. For k=3 the linear contrast is
    `[-1, 0, 1]`, so the estimate is `mu[2] - mu[0]` directly (not
    divided by sqrt(2) as the orthonormal `contr.poly` version was)."""
    emm = emmeans(one_way, "g")
    c = contrast(emm, method="poly", adjust="none")
    mu = emm.frame["emmean"].to_numpy()
    expected_linear = mu[2] - mu[0] # integer coefficients [-1, 0, 1]
    assert c.frame["estimate"].iloc[0] == pytest.approx(expected_linear, abs=1e-6)


def test_consec_contrast_count_and_labels(one_way):
    emm = emmeans(one_way, "g") # levels a, b, c
    c = contrast(emm, method="consec")
    assert list(c.frame["contrast"]) == ["b - a", "c - b"]
    mu = emm.frame["emmean"].to_numpy()
    np.testing.assert_array_almost_equal(
        c.frame["estimate"].to_numpy(), [mu[1] - mu[0], mu[2] - mu[1]]
    )


# --- Effect sizes -------------------------------------------------------


def test_effect_size_cohen_d_and_hedges_g(one_way):
    from pymmeans import effect_size

    emm = emmeans(one_way, "g")
    pw = pairs(emm)
    out = effect_size(pw)
    assert "cohen_d" in out.columns
    assert "hedges_g" in out.columns
    # Sigma = sqrt(residual variance). cohen_d = estimate / sigma.
    sigma = float(np.sqrt(one_way.scale))
    np.testing.assert_array_almost_equal(
        out["cohen_d"].to_numpy(), pw.frame["estimate"].to_numpy() / sigma
    )
    # Hedge's J factor is < 1, so |g| <= |d|
    assert (np.abs(out["hedges_g"]) <= np.abs(out["cohen_d"]) + 1e-12).all()


def test_effect_size_with_custom_sigma(one_way):
    from pymmeans import effect_size

    emm = emmeans(one_way, "g")
    pw = pairs(emm)
    out = effect_size(pw, sigma=2.0)
    np.testing.assert_array_almost_equal(
        out["cohen_d"].to_numpy(), pw.frame["estimate"].to_numpy() / 2.0
    )


def test_effect_size_rejects_nonpositive_sigma(one_way):
    from pymmeans import effect_size

    emm = emmeans(one_way, "g")
    pw = pairs(emm)
    with pytest.raises(ValueError, match="positive"):
        effect_size(pw, sigma=0.0)
