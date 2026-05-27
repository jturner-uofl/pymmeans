"""Tests for ref_grid construction."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans.ref_grid import RefGrid, ref_grid


@pytest.fixture
def one_factor_one_numeric():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=60),
            "g": pd.Categorical(rng.choice(["a", "b", "c"], 60)),
            "x": rng.normal(size=60),
        }
    )
    return smf.ols("y ~ g + x", data=df).fit(), df


@pytest.fixture
def two_factors():
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=80),
            "a": pd.Categorical(rng.choice(["lo", "hi"], 80)),
            "b": pd.Categorical(rng.choice(["x", "y", "z"], 80)),
        }
    )
    return smf.ols("y ~ a * b", data=df).fit(), df


def test_returns_refgrid(one_factor_one_numeric):
    fit, _ = one_factor_one_numeric
    rg = ref_grid(fit)
    assert isinstance(rg, RefGrid)


def test_one_factor_grid_has_one_row_per_level(one_factor_one_numeric):
    fit, _ = one_factor_one_numeric
    rg = ref_grid(fit)
    assert rg.n_rows == 3
    assert list(rg.grid["g"]) == ["a", "b", "c"]


def test_numeric_held_at_training_mean(one_factor_one_numeric):
    fit, df = one_factor_one_numeric
    rg = ref_grid(fit)
    assert (rg.grid["x"] == df["x"].mean()).all()


def test_two_factors_cross_produces_all_combos(two_factors):
    fit, _ = two_factors
    rg = ref_grid(fit)
    assert rg.n_rows == 2 * 3
    assert set(zip(rg.grid["a"], rg.grid["b"], strict=True)) == {
        ("hi", "x"), ("hi", "y"), ("hi", "z"),
        ("lo", "x"), ("lo", "y"), ("lo", "z"),
    }


def test_linfct_shape_matches_params(two_factors):
    fit, _ = two_factors
    rg = ref_grid(fit)
    assert rg.linfct.shape == (rg.n_rows, rg.model_info.n_params)


def test_linfct_at_beta_matches_predict(one_factor_one_numeric):
    fit, _ = one_factor_one_numeric
    rg = ref_grid(fit)
    mu = rg.linfct @ rg.model_info.beta
    expected = fit.predict(rg.grid).to_numpy()
    np.testing.assert_array_almost_equal(mu, expected)


def test_at_overrides_numeric(one_factor_one_numeric):
    fit, _ = one_factor_one_numeric
    rg = ref_grid(fit, at={"x": [-1.0, 0.0, 1.0]})
    assert rg.n_rows == 3 * 3
    assert sorted(rg.grid["x"].unique()) == [-1.0, 0.0, 1.0]


def test_at_subsets_factor_levels(two_factors):
    fit, _ = two_factors
    rg = ref_grid(fit, at={"b": ["x", "z"]})
    assert rg.n_rows == 2 * 2
    assert set(rg.grid["b"].astype(str)) == {"x", "z"}


def test_at_scalar_accepted(one_factor_one_numeric):
    fit, _ = one_factor_one_numeric
    rg = ref_grid(fit, at={"x": 0.0})
    assert rg.n_rows == 3
    assert (rg.grid["x"] == 0.0).all()


def test_at_rejects_unknown_variable(one_factor_one_numeric):
    fit, _ = one_factor_one_numeric
    with pytest.raises(ValueError, match="unknown variables"):
        ref_grid(fit, at={"nonexistent": 0.0})


def test_at_rejects_unknown_factor_level(one_factor_one_numeric):
    fit, _ = one_factor_one_numeric
    with pytest.raises(ValueError, match="unknown levels"):
        ref_grid(fit, at={"g": ["a", "z"]})


def test_rejects_categorical_expression_in_formula():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=40),
            "p": rng.choice([10, 20, 30], 40),
        }
    )
    fit = smf.ols("y ~ C(p)", data=df).fit()
    with pytest.raises(NotImplementedError, match="plain column"):
        ref_grid(fit)


def test_rejects_numeric_transformation_in_formula():
    rng = np.random.default_rng(3)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=40),
            "x": rng.uniform(0.1, 10.0, 40),
        }
    )
    fit = smf.ols("y ~ np.log(x)", data=df).fit()
    with pytest.raises(NotImplementedError, match="plain"):
        ref_grid(fit)
