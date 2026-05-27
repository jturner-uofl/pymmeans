"""Tests for emmeans() core computation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

from pymmeans.emmeans import EMMResult, emmeans


@pytest.fixture
def balanced_one_way():
    df = pd.DataFrame(
        {
            "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            "g": pd.Categorical(["a"] * 3 + ["b"] * 3 + ["c"] * 3),
        }
    )
    return smf.ols("y ~ g", data=df).fit(), df


@pytest.fixture
def two_way_interaction():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=120),
            "a": pd.Categorical(rng.choice(["lo", "hi"], 120)),
            "b": pd.Categorical(rng.choice(["x", "y", "z"], 120)),
        }
    )
    return smf.ols("y ~ a * b", data=df).fit(), df


def test_returns_emmresult(balanced_one_way):
    fit, _ = balanced_one_way
    emm = emmeans(fit, "g")
    assert isinstance(emm, EMMResult)


def test_balanced_one_way_matches_cell_means(balanced_one_way):
    fit, df = balanced_one_way
    emm = emmeans(fit, "g")
    expected = df.groupby("g", observed=True)["y"].mean().sort_index().to_numpy()
    np.testing.assert_array_almost_equal(emm.frame["emmean"].to_numpy(), expected)


def test_output_columns(balanced_one_way):
    fit, _ = balanced_one_way
    emm = emmeans(fit, "g")
    assert list(emm.frame.columns) == [
        "g", "emmean", "SE", "df", "lower_cl", "upper_cl",
    ]


def test_one_row_per_target_level(balanced_one_way):
    fit, _ = balanced_one_way
    emm = emmeans(fit, "g")
    assert emm.n_rows == 3
    assert list(emm.frame["g"].astype(str)) == ["a", "b", "c"]


def test_two_way_marginal_averages_other_factor(two_way_interaction):
    fit, _ = two_way_interaction
    emm_a = emmeans(fit, "a")
    assert emm_a.n_rows == 2

    # Compare against manual marginalization: average predictions over b levels
    expected = []
    for level in ["hi", "lo"]:
        grid = pd.DataFrame(
            {
                "a": pd.Categorical([level] * 3, categories=["hi", "lo"]),
                "b": pd.Categorical(["x", "y", "z"], categories=["x", "y", "z"]),
            }
        )
        preds = fit.predict(grid).to_numpy()
        expected.append(preds.mean())
    np.testing.assert_array_almost_equal(emm_a.frame["emmean"].to_numpy(), expected)


def test_by_grouping_produces_one_row_per_combo(two_way_interaction):
    fit, _ = two_way_interaction
    emm = emmeans(fit, "a", by="b")
    assert emm.n_rows == 2 * 3
    assert set(zip(emm.frame["a"].astype(str), emm.frame["b"].astype(str), strict=True)) == {
        ("hi", "x"), ("hi", "y"), ("hi", "z"),
        ("lo", "x"), ("lo", "y"), ("lo", "z"),
    }


def test_by_emm_equals_grid_prediction(two_way_interaction):
    fit, _ = two_way_interaction
    emm = emmeans(fit, "a", by="b")
    # With both factors specified (target=a, by=b), no marginalization happens —
    # values should equal direct predictions at each (a, b) combo.
    grid = emm.frame[["a", "b"]].copy()
    grid["a"] = pd.Categorical(grid["a"], categories=["hi", "lo"])
    grid["b"] = pd.Categorical(grid["b"], categories=["x", "y", "z"])
    expected = fit.predict(grid).to_numpy()
    np.testing.assert_array_almost_equal(emm.frame["emmean"].to_numpy(), expected)


def test_multi_target_cross(two_way_interaction):
    fit, _ = two_way_interaction
    emm = emmeans(fit, ["a", "b"])
    assert emm.n_rows == 6


def test_ci_uses_t_distribution(balanced_one_way):
    fit, _ = balanced_one_way
    emm = emmeans(fit, "g", level=0.95)
    crit = stats.t.ppf(0.975, fit.df_resid)
    half_width = crit * emm.frame["SE"].to_numpy()
    np.testing.assert_array_almost_equal(
        emm.frame["upper_cl"].to_numpy() - emm.frame["emmean"].to_numpy(),
        half_width,
    )


def test_ci_level_changes_width(balanced_one_way):
    fit, _ = balanced_one_way
    emm95 = emmeans(fit, "g", level=0.95)
    emm99 = emmeans(fit, "g", level=0.99)
    w95 = (emm95.frame["upper_cl"] - emm95.frame["lower_cl"]).to_numpy()
    w99 = (emm99.frame["upper_cl"] - emm99.frame["lower_cl"]).to_numpy()
    assert (w99 > w95).all()


def test_df_resid_for_ols(balanced_one_way):
    fit, _ = balanced_one_way
    emm = emmeans(fit, "g")
    assert (emm.frame["df"] == fit.df_resid).all()


def test_df_inf_for_glm():
    rng = np.random.default_rng(7)
    n = 200
    df = pd.DataFrame(
        {
            "y": rng.integers(0, 2, n),
            "g": pd.Categorical(rng.choice(["a", "b", "c"], n)),
            "x": rng.normal(size=n),
        }
    )
    fit = smf.glm("y ~ g + x", data=df, family=sm.families.Binomial()).fit()
    emm = emmeans(fit, "g")
    assert np.isinf(emm.frame["df"]).all()


def test_linfct_shape_matches_n_rows_and_params(two_way_interaction):
    fit, _ = two_way_interaction
    emm = emmeans(fit, "a")
    assert emm.linfct.shape == (emm.n_rows, emm.model_info.n_params)


def test_linfct_at_beta_reproduces_emmean(two_way_interaction):
    fit, _ = two_way_interaction
    emm = emmeans(fit, "a")
    mu = emm.linfct @ emm.model_info.beta
    np.testing.assert_array_almost_equal(mu, emm.frame["emmean"].to_numpy())


def test_rejects_unknown_factor(balanced_one_way):
    fit, _ = balanced_one_way
    with pytest.raises(
        ValueError, match="not a factor or numeric covariate"
    ):
        emmeans(fit, "nonexistent")


def test_rejects_overlap_between_specs_and_by(two_way_interaction):
    fit, _ = two_way_interaction
    with pytest.raises(ValueError, match="both specs and by"):
        emmeans(fit, "a", by="a")


def test_rejects_empty_specs(balanced_one_way):
    fit, _ = balanced_one_way
    with pytest.raises(ValueError, match="at least one"):
        emmeans(fit, [])


# --- Response-scale back-transformation -----------------------------------


@pytest.fixture
def binomial_glm():
    rng = np.random.default_rng(11)
    n = 300
    df = pd.DataFrame(
        {
            "y": rng.integers(0, 2, n),
            "g": pd.Categorical(rng.choice(["a", "b", "c"], n)),
            "x": rng.normal(size=n),
        }
    )
    return smf.glm("y ~ g + x", data=df, family=sm.families.Binomial()).fit()


@pytest.mark.filterwarnings("ignore::statsmodels.tools.sm_exceptions.PerfectSeparationWarning")
def test_response_scale_binomial_in_unit_interval(binomial_glm):
    emm = emmeans(binomial_glm, "g", type="response")
    assert (emm.frame["emmean"] > 0).all()
    assert (emm.frame["emmean"] < 1).all()


def test_response_emmean_equals_sigmoid_of_link_emmean(binomial_glm):
    emm_link = emmeans(binomial_glm, "g")
    emm_resp = emmeans(binomial_glm, "g", type="response")
    expected = 1.0 / (1.0 + np.exp(-emm_link.frame["emmean"].to_numpy()))
    np.testing.assert_array_almost_equal(emm_resp.frame["emmean"].to_numpy(), expected)


def test_response_ci_bounds_match_inverse_link_of_link_ci(binomial_glm):
    emm_link = emmeans(binomial_glm, "g")
    emm_resp = emmeans(binomial_glm, "g", type="response")
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    np.testing.assert_array_almost_equal(
        emm_resp.frame["lower_cl"].to_numpy(),
        sigmoid(emm_link.frame["lower_cl"].to_numpy()),
    )
    np.testing.assert_array_almost_equal(
        emm_resp.frame["upper_cl"].to_numpy(),
        sigmoid(emm_link.frame["upper_cl"].to_numpy()),
    )


def test_response_se_via_delta_method(binomial_glm):
    emm_link = emmeans(binomial_glm, "g")
    emm_resp = emmeans(binomial_glm, "g", type="response")
    eta = emm_link.frame["emmean"].to_numpy()
    p = 1.0 / (1.0 + np.exp(-eta))
    deriv = p * (1.0 - p)  # d/d_eta of sigmoid
    expected_se = np.abs(deriv) * emm_link.frame["SE"].to_numpy()
    np.testing.assert_array_almost_equal(emm_resp.frame["SE"].to_numpy(), expected_se)


def test_response_preserves_linfct_link_scale(binomial_glm):
    emm_link = emmeans(binomial_glm, "g")
    emm_resp = emmeans(binomial_glm, "g", type="response")
    np.testing.assert_array_almost_equal(emm_link.linfct, emm_resp.linfct)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_response_for_ols_is_noop(balanced_one_way):
    fit, _ = balanced_one_way
    a = emmeans(fit, "g")
    b = emmeans(fit, "g", type="response")
    np.testing.assert_array_almost_equal(
        a.frame["emmean"].to_numpy(), b.frame["emmean"].to_numpy()
    )
    np.testing.assert_array_almost_equal(
        a.frame["SE"].to_numpy(), b.frame["SE"].to_numpy()
    )


def test_invalid_type_raises(balanced_one_way):
    fit, _ = balanced_one_way
    with pytest.raises(ValueError, match="'link' or 'response'"):
        emmeans(fit, "g", type="probit")


# --- Chunked marginalization ---------------------------------------------


def test_chunked_matches_inmemory(two_way_interaction):
    fit, _ = two_way_interaction
    eager = emmeans(fit, "a", by="b")
    streamed = emmeans(fit, "a", by="b", chunk_size=2)  # forces multi-chunk
    np.testing.assert_array_almost_equal(
        eager.frame["emmean"].to_numpy(), streamed.frame["emmean"].to_numpy()
    )
    np.testing.assert_array_almost_equal(
        eager.frame["SE"].to_numpy(), streamed.frame["SE"].to_numpy()
    )
    np.testing.assert_array_almost_equal(eager.linfct, streamed.linfct)


def test_chunked_handles_many_factor_interaction():
    rng = np.random.default_rng(13)
    n = 1500
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "a": pd.Categorical(rng.choice(["a0", "a1"], n)),
            "b": pd.Categorical(rng.choice(["b0", "b1"], n)),
            "c": pd.Categorical(rng.choice(["c0", "c1", "c2"], n)),
            "d": pd.Categorical(rng.choice(["d0", "d1", "d2"], n)),
            "e": pd.Categorical(rng.choice([f"e{i}" for i in range(8)], n)),
        }
    )
    fit = smf.ols("y ~ a + b + c + d + e", data=df).fit()
    # Grid size = 2*2*3*3*8 = 288 — small, but force streaming to exercise path
    emm = emmeans(fit, "e", chunk_size=50)
    assert emm.n_rows == 8
    # Sanity: emmeans for e averages over a/b/c/d. With balanced random
    # data, the 8 means should be close to the grand mean
    assert emm.frame["emmean"].std() < 1.0
