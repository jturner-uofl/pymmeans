"""Smoke-cover the public API surface — auditor V12-A2 F1.

The maintainer's strict-correctness test suite covers contracts named
in the JSS scorecard; sixteen public ``__all__`` exports had zero
public-tracked test references prior to v0.2.4. This file does not
prove correctness — it proves the documented entry point can be
imported, called with a minimal valid argument set, and returns the
documented type without raising. The intent is to make a future
refactor that silently breaks one of these names surface as a CI
failure rather than as a downstream user bug.

Correctness contracts for each name live in the strict suite; this
file is a *guard rail*, not a substitute for that coverage. New names
added to ``__all__`` should get a matching entry here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf


# Shared lightweight fixture: a balanced 3-group OLS the smoke
# tests can re-fit cheaply.
@pytest.fixture(scope="module")
def _ols_fit():
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame({
        "g": pd.Categorical(np.repeat(["A", "B", "C"], n // 3)),
        "x": rng.normal(size=n),
        "y": rng.normal(size=n),
    })
    fit = smf.ols("y ~ C(g) + x", df).fit()
    return fit, df


# ---------------------------------------------------------------------
# Grid-ops surface
# ---------------------------------------------------------------------


def test_split_fac_smoke():
    from pymmeans import comb_facs, emmeans, split_fac

    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "a": pd.Categorical(np.repeat(["x", "y"], 15)),
        "b": pd.Categorical(np.tile(["p", "q", "r"], 10)),
        "y": rng.normal(size=30),
    })
    fit = smf.ols("y ~ a + b", df).fit()
    em = emmeans(fit, ["a", "b"])
    em_comb = comb_facs(em, ["a", "b"], new_name="ab", sep=":")
    em_split = split_fac(em_comb, "ab", new_names=["a", "b"], sep=":")
    assert "a" in em_split.frame.columns
    assert "b" in em_split.frame.columns
    assert len(em_split.frame) == len(em.frame)


def test_comb_facs_smoke():
    from pymmeans import comb_facs, emmeans

    rng = np.random.default_rng(2)
    df = pd.DataFrame({
        "a": pd.Categorical(np.repeat(["x", "y"], 15)),
        "b": pd.Categorical(np.tile(["p", "q", "r"], 10)),
        "y": rng.normal(size=30),
    })
    fit = smf.ols("y ~ a + b", df).fit()
    em = emmeans(fit, ["a", "b"])
    em_c = comb_facs(em, ["a", "b"], new_name="ab", sep=":")
    assert "ab" in em_c.frame.columns
    assert len(em_c.frame) == len(em.frame)


def test_add_grouping_smoke(_ols_fit):
    from pymmeans import add_grouping, emmeans

    fit, _ = _ols_fit
    em = emmeans(fit, "g")
    em_grp = add_grouping(em, "treatment", "g",
                          {"A": "control", "B": "active", "C": "active"})
    assert "treatment" in em_grp.frame.columns
    assert set(em_grp.frame["treatment"]) == {"control", "active"}


def test_permute_levels_smoke(_ols_fit):
    from pymmeans import emmeans, permute_levels

    fit, _ = _ols_fit
    em = emmeans(fit, "g")
    em_perm = permute_levels(em, "g", ["C", "A", "B"])
    assert list(em_perm.frame["g"]) == ["C", "A", "B"]
    # EMMs themselves are unchanged; only ordering moved.
    assert set(em_perm.frame["emmean"].round(10)) == set(em.frame["emmean"].round(10))


def test_force_regular_smoke(_ols_fit):
    from pymmeans import emmeans, force_regular

    fit, _ = _ols_fit
    em = emmeans(fit, "g")
    em_fr = force_regular(em)
    assert len(em_fr.frame) == len(em.frame)


# ---------------------------------------------------------------------
# Transforms surface
# ---------------------------------------------------------------------


def test_make_tran_power_smoke():
    from pymmeans import make_tran

    tr = make_tran("power", lambda_=2.0)
    # Forward y -> y^2; inverse eta -> sqrt(eta). Round-trip on
    # positive reals must match to float precision.
    eta = np.array([1.0, 4.0, 9.0, 16.0])
    np.testing.assert_allclose(tr.inverse(eta), np.sqrt(eta), rtol=1e-12)


def test_register_transform_smoke():
    from pymmeans import TRANSFORMS, Transform, register_transform

    name = "smoke_neg_log"
    tr = Transform(
        name=name,
        inverse=lambda eta: np.exp(-eta),
        inverse_deriv=lambda eta: -np.exp(-eta),
    )
    register_transform(
        name, tr, overwrite=True,
        forward=lambda y: -np.log(y),
        forward_deriv=lambda y: -1.0 / y,
    )
    assert name in TRANSFORMS
    # The Transform NamedTuple stores the inverse; the forward is
    # registered in a separate ``_FORWARD`` lookup. Exercise the
    # round-trip via the inverse leg of the registered transform.
    tr_reg = TRANSFORMS[name]
    eta = np.array([1.0])
    np.testing.assert_allclose(tr_reg.inverse(eta), np.exp(-eta), rtol=1e-12)


def test_register_contrast_method_smoke():
    import numpy as np

    from pymmeans import CONTRAST_METHODS, register_contrast_method

    name = "smoke_first_vs_last"

    def _builder(k, labels=None, **_kw):
        # one contrast: first level vs last
        L = np.zeros((1, k))
        L[0, 0], L[0, -1] = 1.0, -1.0
        return L, ["first - last"]

    register_contrast_method(name, builder=_builder, default_adjust="bonferroni",
                              overwrite=True)
    assert name in CONTRAST_METHODS


# ---------------------------------------------------------------------
# Contrast / poly surface
# ---------------------------------------------------------------------


def test_opoly_smoke():
    from pymmeans import opoly

    # k=4 levels, default max_degree gives k-1 = 3 polynomial columns.
    L, labels = opoly(4)
    assert L.shape == (3, 4)
    assert len(labels) == 3
    # Orthogonality: rows should be mutually orthogonal.
    G = L @ L.T
    off_diag_max = float(np.max(np.abs(G - np.diag(np.diag(G)))))
    assert off_diag_max < 1e-10


# ---------------------------------------------------------------------
# Joint / effect size
# ---------------------------------------------------------------------


def test_eta_squared_smoke(_ols_fit):
    from pymmeans import eta_squared

    fit, _ = _ols_fit
    # eta_squared operates on the raw model fit, not on a joint_tests
    # frame; it returns a DataFrame of effect-size estimates.
    out = eta_squared(fit)
    assert hasattr(out, "columns")
    # At least one of the standard columns must be present.
    cols = set(out.columns)
    assert cols & {"eta_sq", "partial_eta_sq", "omega_sq", "cohens_f"}


# ---------------------------------------------------------------------
# Resampling surface
# ---------------------------------------------------------------------


def test_permutation_test_smoke(_ols_fit):
    from pymmeans import contrast, emmeans, permutation_test

    fit, _ = _ols_fit
    em = emmeans(fit, "g")
    ct = contrast(em, method="pairwise")
    out = permutation_test(ct, n_permutations=99, seed=0)
    # Returns a DataFrame with at least one permutation-based column.
    assert hasattr(out, "columns")
    assert len(out) >= 1


# ---------------------------------------------------------------------
# Plotting surface (matplotlib optional)
# ---------------------------------------------------------------------


def test_pwpp_smoke(_ols_fit):
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    from pymmeans import emmeans, pwpp

    fit, _ = _ols_fit
    em = emmeans(fit, "g")
    ax = pwpp(em)
    assert ax is not None


# ---------------------------------------------------------------------
# ML surface (scikit-learn optional)
# ---------------------------------------------------------------------


def test_ml_emmeans_and_contrast_and_pairs_smoke():
    pytest.importorskip("sklearn")
    from sklearn.ensemble import RandomForestRegressor

    from pymmeans import from_predict, ml_contrast, ml_emmeans, ml_pairs

    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "g": pd.Categorical(np.repeat(["A", "B", "C", "D"], n // 4)),
        "x": rng.normal(size=n),
        "y": rng.normal(size=n),
    })
    X = pd.get_dummies(df[["g", "x"]], drop_first=False).astype(float)
    y = df["y"].values
    model = RandomForestRegressor(n_estimators=20, random_state=0).fit(X, y)

    # from_predict needs a callable that accepts a DataFrame keyed
    # by the original factor / numeric columns. Wrap the rf to do
    # the dummy-encoding inside.
    feature_cols = list(X.columns)

    def _predict(df_new: pd.DataFrame) -> np.ndarray:
        Xn = pd.get_dummies(df_new[["g", "x"]], drop_first=False).astype(float)
        Xn = Xn.reindex(columns=feature_cols, fill_value=0.0)
        return model.predict(Xn)

    info = from_predict(
        predict_fn=_predict,
        data=df,
        factors=["g"],
        numerics=["x"],
        response="y",
    )
    em = ml_emmeans(info, "g")
    assert len(em.frame) == 4

    # ml_contrast / ml_pairs return plain DataFrames, not EMMResult
    # wrappers (the ML adapter does not carry a covariance to support
    # the full inference layer).
    ct = ml_contrast(em, method="pairwise")
    assert len(ct) >= 1

    pr = ml_pairs(em)
    assert len(pr) >= 1
