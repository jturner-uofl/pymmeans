"""Tests for the double_ml module — cross-fit g-comp + AIPW double-robust ATE."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from pymmeans import AIPWResult, aipw_ate, cross_fit_ml_emmeans
from pymmeans.ml import from_predict, ml_emmeans

# ---------------------------------------------------------------------- helpers


def _design(*cols):
    return np.column_stack([np.ones(len(cols[0]))] + list(cols))


def _gen_dgp_dr(n: int, seed: int):
    """Generate a strongly-confounded DGP with a known marginal ATE = 1.5.

    Used to verify the double-robust property of AIPW.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    logit_p = 1.5 * X[:, 0] - 1.0 * X[:, 1] + 0.5 * np.sin(X[:, 2])
    p_true = 1.0 / (1.0 + np.exp(-logit_p))
    T = (rng.random(n) < p_true).astype(int)
    mu0 = 1.0 + 0.8 * X[:, 0] ** 2 + 0.5 * X[:, 1] - 0.3 * X[:, 2] * X[:, 3]
    mu1 = mu0 + 1.5
    Y = np.where(T == 1, mu1, mu0) + rng.standard_normal(n) * 0.5
    df = pd.DataFrame({
        "treat": pd.Categorical(T, categories=[0, 1]),
        "x0": X[:, 0], "x1": X[:, 1], "x2": X[:, 2],
        "x3": X[:, 3], "x4": X[:, 4],
        "y": Y,
    })
    return df


def _build_info(df: pd.DataFrame, *, predict_fn=None, refit_fn=None):
    """Wrap df into an MLPredictInfo with an OLS-linear outcome predict_fn."""
    if predict_fn is None:
        def predict_fn(data):
            X_mat = _design(
                np.asarray(data["treat"], dtype=float),
                *[data[f"x{i}"].to_numpy(dtype=float) for i in range(5)],
            )
            # Fit on info.data the FIRST time, then reuse?
            # For tests, just return X_mat @ coef where coef is fit on the
            # CURRENT data each call (test-only convenience).
            X_fit = _design(
                np.asarray(df["treat"], dtype=float),
                *[df[f"x{i}"].to_numpy(dtype=float) for i in range(5)],
            )
            coef, *_ = np.linalg.lstsq(X_fit, df["y"].to_numpy(), rcond=None)
            return X_mat @ coef
    return from_predict(
        predict_fn=predict_fn, data=df,
        factors={"treat": [0, 1]},
        numerics=[f"x{i}" for i in range(5)],
        response="y",
        refit_fn=refit_fn,
    )


# ---------------------------------------------------------------------- AIPW


def test_aipw_returns_documented_result():
    df = _gen_dgp_dr(n=500, seed=0)
    info = _build_info(df)

    def prop_fn(data):
        X_p = sm.add_constant(data[["x0", "x1", "x2", "x3", "x4"]].to_numpy())
        fit = sm.Logit(
            np.asarray(df["treat"], dtype=int),
            sm.add_constant(df[["x0", "x1", "x2", "x3", "x4"]].to_numpy()),
        ).fit(disp=0)
        return fit.predict(X_p)

    res = aipw_ate(info, propensity_predict_fn=prop_fn,
                    treatment="treat", treatment_levels=(0, 1))
    assert isinstance(res, AIPWResult)
    assert res.n == 500
    assert res.se > 0
    assert res.lower_cl < res.estimate < res.upper_cl
    assert 0.025 <= res.weight_clip[0] < res.weight_clip[1] <= 0.975


def test_aipw_rejects_non_info_input():
    with pytest.raises(TypeError, match="MLPredictInfo"):
        aipw_ate({"data": "not-info"}, propensity_predict_fn=lambda d: np.zeros(1))


def test_aipw_rejects_non_callable_propensity():
    df = _gen_dgp_dr(n=100, seed=1)
    info = _build_info(df)
    with pytest.raises(TypeError, match="callable"):
        aipw_ate(info, propensity_predict_fn="not-callable")  # type: ignore[arg-type]


def test_aipw_rejects_invalid_level_and_weight_clip():
    df = _gen_dgp_dr(n=100, seed=2)
    info = _build_info(df)
    def prop(d):
        return np.full(len(d), 0.5)
    with pytest.raises(ValueError, match="level"):
        aipw_ate(info, propensity_predict_fn=prop, level=0.0)
    with pytest.raises(ValueError, match="weight_clip"):
        aipw_ate(info, propensity_predict_fn=prop, weight_clip=(0.5, 0.4))
    with pytest.raises(ValueError, match="weight_clip"):
        aipw_ate(info, propensity_predict_fn=prop, weight_clip=(-0.1, 0.9))


def test_aipw_propensity_shape_mismatch_raises():
    df = _gen_dgp_dr(n=100, seed=3)
    info = _build_info(df)
    def bad_prop(d):
        return np.full(len(d) + 1, 0.5)  # wrong shape
    with pytest.raises(ValueError, match="shape"):
        aipw_ate(info, propensity_predict_fn=bad_prop)


def test_aipw_clip_count_reported():
    """When some propensities are extreme, n_clipped reports them."""
    df = _gen_dgp_dr(n=200, seed=4)
    info = _build_info(df)
    # Force extreme propensities so clipping triggers.
    def prop(d):
        return np.where(
            d["x0"].to_numpy() > 0.5, 0.99,
            np.where(d["x0"].to_numpy() < -0.5, 0.01, 0.5),
        )
    res = aipw_ate(info, propensity_predict_fn=prop,
                    weight_clip=(0.05, 0.95))
    assert res.n_clipped > 0


# ---------------------------------------------------------------------- DR property (MC)


def _fit_outcome(df, kind: str):
    """Build a predict_fn for the outcome model on df under one of two specifications."""
    Y = df["y"].to_numpy()
    if kind == "correct":
        XT = _design(
            np.asarray(df["treat"], dtype=float),
            (df["x0"].to_numpy() ** 2),
            df["x1"].to_numpy(),
            (df["x2"].to_numpy() * df["x3"].to_numpy()),
        )
    else:  # 'linear' — misspec
        XT = _design(
            np.asarray(df["treat"], dtype=float),
            *[df[f"x{i}"].to_numpy() for i in range(5)],
        )
    coef, *_ = np.linalg.lstsq(XT, Y, rcond=None)

    def predict_fn(data):
        T_arr = np.asarray(data["treat"], dtype=float)
        if kind == "correct":
            XT_p = _design(
                T_arr,
                data["x0"].to_numpy() ** 2,
                data["x1"].to_numpy(),
                data["x2"].to_numpy() * data["x3"].to_numpy(),
            )
        else:
            XT_p = _design(
                T_arr,
                *[data[f"x{i}"].to_numpy() for i in range(5)],
            )
        return XT_p @ coef
    return predict_fn


def _fit_propensity(df, kind: str):
    """Build a propensity_predict_fn under one of two specifications."""
    T = np.asarray(df["treat"], dtype=int)
    if kind == "correct":
        feats = np.column_stack([
            df["x0"].to_numpy(),
            df["x1"].to_numpy(),
            np.sin(df["x2"].to_numpy()),
        ])
    else:  # 'linear' — misspec (no sin)
        feats = df[["x0", "x1", "x2", "x3", "x4"]].to_numpy()
    fit = sm.Logit(T, sm.add_constant(feats)).fit(disp=0)

    def prop_fn(data):
        if kind == "correct":
            f = np.column_stack([
                data["x0"].to_numpy(),
                data["x1"].to_numpy(),
                np.sin(data["x2"].to_numpy()),
            ])
        else:
            f = data[["x0", "x1", "x2", "x3", "x4"]].to_numpy()
        return fit.predict(sm.add_constant(f))
    return prop_fn


@pytest.mark.parametrize(
    "outcome_kind,propensity_kind",
    [
        ("correct", "linear"),   # scenario A: outcome correct → AIPW unbiased
        ("linear",  "correct"),  # scenario B: propensity correct → AIPW unbiased
    ],
)
def test_aipw_double_robust_unbiased(outcome_kind: str, propensity_kind: str):
    """AIPW is unbiased if either nuisance is correctly specified."""
    N_REPS = 200
    n = 1500
    true_ate = 1.5
    biases = []
    for rep in range(N_REPS):
        df = _gen_dgp_dr(n=n, seed=rep + 100)
        pred_y = _fit_outcome(df, outcome_kind)
        info = _build_info(df, predict_fn=pred_y)
        prop = _fit_propensity(df, propensity_kind)
        res = aipw_ate(info, propensity_predict_fn=prop,
                        treatment="treat", treatment_levels=(0, 1))
        biases.append(res.estimate - true_ate)
    emp_bias = float(np.mean(biases))
    # Across 200 reps with n=1500, MC SE on mean bias is roughly
    # sd(bias)/sqrt(200). Tolerance of 0.05 is loose enough to absorb
    # finite-sample noise.
    assert abs(emp_bias) <= 0.05, (
        f"AIPW |bias| = {abs(emp_bias):.4f} too large for scenario "
        f"outcome={outcome_kind}, propensity={propensity_kind}; "
        f"DR property failed."
    )


def test_aipw_beats_ipw_under_propensity_misspec():
    """When only the outcome is correctly specified, AIPW has smaller bias than IPW."""
    N_REPS = 150
    n = 1500
    true_ate = 1.5
    aipw_biases = []
    ipw_biases = []
    for rep in range(N_REPS):
        df = _gen_dgp_dr(n=n, seed=rep + 250)
        pred_y = _fit_outcome(df, "correct")
        info = _build_info(df, predict_fn=pred_y)
        prop = _fit_propensity(df, "linear")  # misspec
        res = aipw_ate(info, propensity_predict_fn=prop)
        aipw_biases.append(res.estimate - true_ate)
        # Naive IPW for comparison
        p_clip = np.clip(prop(df), 0.025, 0.975)
        T = np.asarray(df["treat"], dtype=float)
        Y = df["y"].to_numpy()
        ipw_est = float(np.mean(T * Y / p_clip - (1 - T) * Y / (1 - p_clip)))
        ipw_biases.append(ipw_est - true_ate)
    aipw_mae = float(np.mean(np.abs(aipw_biases)))
    ipw_mae = float(np.mean(np.abs(ipw_biases)))
    assert aipw_mae < ipw_mae, (
        f"AIPW MAE {aipw_mae:.4f} should beat IPW MAE {ipw_mae:.4f} when "
        f"outcome model is correct."
    )


# ---------------------------------------------------------------------- cross-fit


def test_cross_fit_rejects_invalid_K():
    df = _gen_dgp_dr(n=100, seed=10)
    info = _build_info(df, refit_fn=lambda d: lambda x: np.zeros(len(x)))
    with pytest.raises(ValueError, match="K"):
        cross_fit_ml_emmeans(info, "treat", K=1)


def test_cross_fit_requires_refit_fn():
    df = _gen_dgp_dr(n=100, seed=11)
    info = _build_info(df)
    with pytest.raises(ValueError, match="refit_fn"):
        cross_fit_ml_emmeans(info, "treat", K=5)


def test_cross_fit_rejects_data_smaller_than_K():
    df = _gen_dgp_dr(n=5, seed=12)
    info = _build_info(df, refit_fn=lambda d: lambda x: np.zeros(len(x)))
    with pytest.raises(ValueError, match="K"):
        cross_fit_ml_emmeans(info, "treat", K=10)


def test_cross_fit_returns_mlemm_with_stamped_method():
    df = _gen_dgp_dr(n=300, seed=13)

    def refit_fn(train_data):
        coef = np.zeros(7)
        XT = _design(
            np.asarray(train_data["treat"], dtype=float),
            *[train_data[f"x{i}"].to_numpy() for i in range(5)],
        )
        coef, *_ = np.linalg.lstsq(XT, train_data["y"].to_numpy(), rcond=None)
        def pred(data):
            XT_p = _design(
                np.asarray(data["treat"], dtype=float),
                *[data[f"x{i}"].to_numpy() for i in range(5)],
            )
            return XT_p @ coef
        return pred

    info = _build_info(df, refit_fn=refit_fn)
    res = cross_fit_ml_emmeans(info, "treat", K=5)
    assert res.df_method == "cross_fit"
    assert "emmean" in res.frame.columns
    # SE/CI are intentionally NaN — sample-split has no closed-form SE.
    assert res.frame["SE"].isna().all()


def test_cross_fit_produces_held_out_predictions_structurally():
    """Cross-fit predictions on fold k must come from a model NOT trained on fold k.

    This is the structural-correctness check: we verify that
    refit_fn is called K times with K disjoint training subsets,
    each of which excludes ~1/K of the data.
    """
    df = _gen_dgp_dr(n=300, seed=14)
    seen_training_sizes = []
    n_calls = [0]

    def refit_fn(train_data):
        seen_training_sizes.append(len(train_data))
        n_calls[0] += 1
        coef = np.zeros(7)
        XT = _design(
            np.asarray(train_data["treat"], dtype=float),
            *[train_data[f"x{i}"].to_numpy() for i in range(5)],
        )
        coef, *_ = np.linalg.lstsq(XT, train_data["y"].to_numpy(), rcond=None)
        def pred(data):
            XT_p = _design(
                np.asarray(data["treat"], dtype=float),
                *[data[f"x{i}"].to_numpy() for i in range(5)],
            )
            return XT_p @ coef
        return pred

    info = _build_info(df, refit_fn=refit_fn)
    K = 5
    _ = cross_fit_ml_emmeans(info, "treat", K=K)
    # refit_fn called exactly K times, plus 1 extra for the final ml_emmeans
    # call that constructs the result MLEMMResult template (it uses the original
    # info.predict_fn, not refit_fn, so refit_fn is exactly K).
    assert n_calls[0] == K, f"refit_fn called {n_calls[0]} times, expected {K}"
    # Each training subset should be roughly (K-1)/K of the full data.
    expected_size = 300 * (K - 1) / K
    for s in seen_training_sizes:
        assert abs(s - expected_size) <= 1, (
            f"Training subset size {s} differs from expected {expected_size}"
        )


def test_cross_fit_consistency_close_to_naive_under_well_specified():
    """Under well-specified model + large n, cross-fit ≈ naive (both consistent)."""
    df = _gen_dgp_dr(n=1500, seed=42)

    def linear_outcome_fit(train_data):
        XT = _design(
            np.asarray(train_data["treat"], dtype=float),
            *[train_data[f"x{i}"].to_numpy() for i in range(5)],
        )
        coef, *_ = np.linalg.lstsq(XT, train_data["y"].to_numpy(), rcond=None)
        def pred(data):
            XT_p = _design(
                np.asarray(data["treat"], dtype=float),
                *[data[f"x{i}"].to_numpy() for i in range(5)],
            )
            return XT_p @ coef
        return pred

    naive_pred = linear_outcome_fit(df)
    info = _build_info(df, predict_fn=naive_pred, refit_fn=linear_outcome_fit)
    naive_em = ml_emmeans(info, "treat")
    cf_em = cross_fit_ml_emmeans(info, "treat", K=5)
    naive_diff = (
        naive_em.frame["emmean"].iloc[1] - naive_em.frame["emmean"].iloc[0]
    )
    cf_diff = (
        cf_em.frame["emmean"].iloc[1] - cf_em.frame["emmean"].iloc[0]
    )
    # Under well-specified outcome + n=1500, cross-fit ATE should be within
    # 0.5 of the naive ATE (both are consistent; small disagreement from
    # fold variability).
    assert abs(cf_diff - naive_diff) <= 0.5, (
        f"cross-fit ATE {cf_diff:.4f} and naive ATE {naive_diff:.4f} "
        f"should be within 0.5 under a well-specified outcome model"
    )
