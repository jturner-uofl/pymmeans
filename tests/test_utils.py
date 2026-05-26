"""Tests for utils.from_statsmodels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymmeans.utils import ModelInfo, from_statsmodels


@pytest.fixture
def simple_ols():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=60),
            "g": pd.Categorical(rng.choice(["a", "b", "c"], 60)),
            "x": rng.normal(size=60),
        }
    )
    return smf.ols("y ~ g + x", data=df).fit()


def test_returns_modelinfo(simple_ols):
    info = from_statsmodels(simple_ols)
    assert isinstance(info, ModelInfo)


def test_beta_and_vcov_shapes(simple_ols):
    info = from_statsmodels(simple_ols)
    np.testing.assert_array_almost_equal(info.beta, np.asarray(simple_ols.params))
    assert info.vcov.shape == (info.n_params, info.n_params)
    np.testing.assert_array_almost_equal(
        info.vcov, np.asarray(simple_ols.cov_params())
    )


def test_df_resid_matches(simple_ols):
    info = from_statsmodels(simple_ols)
    assert info.df_resid == pytest.approx(simple_ols.df_resid)


def test_categorical_factor_levels(simple_ols):
    info = from_statsmodels(simple_ols)
    assert "g" in info.factors
    assert info.factors["g"] == ["a", "b", "c"]


def test_numeric_held_at_mean(simple_ols):
    info = from_statsmodels(simple_ols)
    assert "x" in info.numeric_means
    assert info.numeric_means["x"] == pytest.approx(info.data["x"].mean())


def test_param_names_align_with_beta(simple_ols):
    info = from_statsmodels(simple_ols)
    assert len(info.param_names) == info.n_params
    assert info.param_names[0] == "Intercept"


def test_response_name_captured(simple_ols):
    info = from_statsmodels(simple_ols)
    assert info.response_name == "y"


def test_ols_has_no_family(simple_ols):
    info = from_statsmodels(simple_ols)
    assert info.family is None


def test_rejects_non_formula_model():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(30, 2))
    y = rng.normal(size=30)
    result = sm.OLS(y, sm.add_constant(X)).fit()
    with pytest.raises(ValueError, match="design_info"):
        from_statsmodels(result)


def test_rejects_non_results_object():
    with pytest.raises(TypeError, match="statsmodels Results"):
        from_statsmodels("not a model")


def test_glm_binomial_family_captured():
    rng = np.random.default_rng(2)
    n = 200
    df = pd.DataFrame(
        {
            "y": rng.integers(0, 2, n),
            "g": pd.Categorical(rng.choice(["a", "b"], n)),
            "x": rng.normal(size=n),
        }
    )
    result = smf.glm("y ~ g + x", data=df, family=sm.families.Binomial()).fit()
    info = from_statsmodels(result)
    assert info.family is not None
    assert type(info.family).__name__ == "Binomial"


def test_interaction_term_preserves_factor_info():
    rng = np.random.default_rng(3)
    n = 100
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "a": pd.Categorical(rng.choice(["lo", "hi"], n)),
            "b": pd.Categorical(rng.choice(["x", "y", "z"], n)),
        }
    )
    result = smf.ols("y ~ a * b", data=df).fit()
    info = from_statsmodels(result)
    assert info.factors == {"a": ["hi", "lo"], "b": ["x", "y", "z"]}
    # 1 intercept + 1 (a) + 2 (b) + 2 (a:b) = 6 params
    assert info.n_params == 6
