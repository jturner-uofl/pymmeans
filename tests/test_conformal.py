"""Tests for the conformal module — split-conformal and Lei-Candès counterfactual PIs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression

from pymmeans import (
    ConformalCounterfactualResult,
    ConformalPIResult,
    conformal_counterfactual_pi,
    split_conformal_pi,
)
from pymmeans.conformal import _conformal_half_width, _weighted_quantile
from pymmeans.ml import from_predict, ml_emmeans

# ---------------------------------------------------------------------- helpers


def _fit_ml_em(
    *,
    n_tr=400,
    n_cal=300,
    seed=0,
    error_distro="gaussian",
    response="y",
):
    """Build (train + calibration) data, fit a GBM, wrap via from_predict.

    Returns (em_ml, calibration_df, training_df).
    """
    rng = np.random.default_rng(seed)
    n_total = n_tr + n_cal
    df = pd.DataFrame({
        "treat": pd.Categorical(rng.integers(0, 2, n_total)),
        "x1":    rng.standard_normal(n_total),
        "x2":    rng.standard_normal(n_total),
    })
    mu_true = df["treat"].astype(int) * 1.5 + 0.5 * df["x1"] + 0.2 * df["x2"]
    if error_distro == "gaussian":
        eps = rng.standard_normal(n_total)
    elif error_distro == "t3":
        eps = rng.standard_t(df=3, size=n_total)
    elif error_distro == "contaminated":
        eps = rng.standard_normal(n_total)
        eps[rng.random(n_total) < 0.10] *= 5
    else:
        raise ValueError(error_distro)
    df[response] = mu_true.to_numpy() + eps

    tr = df.iloc[:n_tr].reset_index(drop=True)
    cal = df.iloc[n_tr:].reset_index(drop=True)

    X_tr = tr[["treat", "x1", "x2"]].astype(float).to_numpy()
    y_tr = tr[response].to_numpy()
    model = GradientBoostingRegressor(
        n_estimators=50, max_depth=3, random_state=seed,
    ).fit(X_tr, y_tr)

    def predict_fn(data):
        return model.predict(
            data[["treat", "x1", "x2"]].astype(float).to_numpy()
        )

    info = from_predict(
        predict_fn=predict_fn, data=tr,
        factors={"treat": [0, 1]}, numerics=["x1", "x2"],
        response=response,
    )
    em = ml_emmeans(info, "treat")
    return em, cal, tr


# ---------------------------------------------------------------------- closed form


def test_conformal_half_width_closed_form_index():
    """The ⌈(n+1)·level⌉-th sorted score is the conformal half-width."""
    scores = np.array([0.1, 0.5, 0.3, 0.9, 0.7])  # sorted: [0.1,0.3,0.5,0.7,0.9]
    # n = 5; level=0.95: k = ceil(6 * 0.95) = ceil(5.7) = 6, clip to 5 → 0.9
    assert _conformal_half_width(scores, 0.95) == pytest.approx(0.9, abs=1e-12)
    # level=0.80: k = ceil(6 * 0.80) = ceil(4.8) = 5 → 0.9
    assert _conformal_half_width(scores, 0.80) == pytest.approx(0.9, abs=1e-12)
    # level=0.50: k = ceil(6 * 0.50) = 3 → 0.5
    assert _conformal_half_width(scores, 0.50) == pytest.approx(0.5, abs=1e-12)


def test_conformal_half_width_requires_minimum_two_observations():
    """A single observation is not enough."""
    with pytest.raises(ValueError, match="at least 2"):
        _conformal_half_width(np.array([0.5]), 0.95)


def test_weighted_quantile_uniform_weights_matches_quantile():
    """Weighted quantile with equal weights matches an unweighted quantile."""
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    w_uniform = np.ones_like(scores)
    # cumulative: [0.2, 0.4, 0.6, 0.8, 1.0]; level=0.6 → idx where cum >= 0.6 → 0.3
    assert _weighted_quantile(scores, w_uniform, 0.6) == pytest.approx(0.3, abs=1e-12)


def test_weighted_quantile_rejects_negative_weights():
    with pytest.raises(ValueError, match="non-negative"):
        _weighted_quantile(np.array([0.1, 0.2]), np.array([1.0, -0.5]), 0.5)


def test_weighted_quantile_rejects_zero_total_weight():
    with pytest.raises(ValueError, match="positive"):
        _weighted_quantile(np.array([0.1, 0.2]), np.array([0.0, 0.0]), 0.5)


# ---------------------------------------------------------------------- API validation


def test_split_conformal_pi_rejects_non_mlemm():
    """Anything that isn't an MLEMMResult raises TypeError."""
    df = pd.DataFrame({"y": [1.0, 2.0, 3.0]})
    with pytest.raises(TypeError, match="MLEMMResult"):
        split_conformal_pi(df, df, level=0.95)  # type: ignore[arg-type]


def test_split_conformal_pi_rejects_invalid_level():
    em, cal, _ = _fit_ml_em(n_tr=200, n_cal=100)
    with pytest.raises(ValueError, match="level"):
        split_conformal_pi(em, cal, level=0.0)
    with pytest.raises(ValueError, match="level"):
        split_conformal_pi(em, cal, level=1.0)
    with pytest.raises(ValueError, match="level"):
        split_conformal_pi(em, cal, level=1.5)


def test_split_conformal_pi_missing_response_column():
    em, cal, _ = _fit_ml_em(n_tr=200, n_cal=100)
    cal_bad = cal.drop(columns=["y"])
    with pytest.raises(ValueError, match="no column"):
        split_conformal_pi(em, cal_bad, level=0.95)


def test_split_conformal_pi_non_finite_y_raises():
    em, cal, _ = _fit_ml_em(n_tr=200, n_cal=100)
    cal_nan = cal.copy()
    cal_nan.loc[0, "y"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        split_conformal_pi(em, cal_nan, level=0.95)


# ---------------------------------------------------------------------- output shape


def test_split_conformal_pi_returns_documented_result():
    em, cal, _ = _fit_ml_em(n_tr=300, n_cal=200, seed=1)
    res = split_conformal_pi(em, cal, level=0.95)
    assert isinstance(res, ConformalPIResult)
    assert res.level == pytest.approx(0.95)
    assert res.n_calibration == 200
    assert "lower_pi" in res.frame.columns
    assert "upper_pi" in res.frame.columns
    # PI is symmetric around emmean.
    emmean = res.frame["emmean"].to_numpy()
    np.testing.assert_allclose(
        res.frame["upper_pi"].to_numpy() - emmean,
        emmean - res.frame["lower_pi"].to_numpy(),
        atol=1e-12,
    )


def test_split_conformal_pi_widens_at_lower_level():
    """Higher coverage level → wider PI."""
    em, cal, _ = _fit_ml_em(n_tr=300, n_cal=200, seed=2)
    res_80 = split_conformal_pi(em, cal, level=0.80)
    res_95 = split_conformal_pi(em, cal, level=0.95)
    assert res_95.q_hat >= res_80.q_hat


# ---------------------------------------------------------------------- coverage


@pytest.mark.parametrize(
    "level,distro",
    [
        (0.80, "gaussian"),
        (0.90, "gaussian"),
        (0.95, "gaussian"),
        (0.90, "t3"),
        (0.90, "contaminated"),
    ],
)
def test_split_conformal_pi_empirical_coverage(level: float, distro: str):
    """Empirical coverage across 50 reps × 200 test points ≥ level - 0.03.

    Across 50 reps × 200 test points = 10,000 test points, the Monte-Carlo
    SE on empirical coverage at level=0.90 is sqrt(0.9*0.1/10000) ≈ 0.003.
    A tolerance of 0.03 (10x SE) is comfortably loose.
    """
    rng = np.random.default_rng(20260601)
    N_REPS = 50
    N_TEST = 200
    covered = []
    for rep in range(N_REPS):
        em, cal, _ = _fit_ml_em(n_tr=400, n_cal=300, seed=rep, error_distro=distro)
        res = split_conformal_pi(em, cal, level=level)
        # Build a fresh test set from the same DGP.
        n = N_TEST
        df_te = pd.DataFrame({
            "treat": pd.Categorical(rng.integers(0, 2, n)),
            "x1":    rng.standard_normal(n),
            "x2":    rng.standard_normal(n),
        })
        mu_te = df_te["treat"].astype(int) * 1.5 + 0.5 * df_te["x1"] + 0.2 * df_te["x2"]
        if distro == "gaussian":
            eps = rng.standard_normal(n)
        elif distro == "t3":
            eps = rng.standard_t(df=3, size=n)
        else:
            eps = rng.standard_normal(n)
            eps[rng.random(n) < 0.10] *= 5
        y_te = mu_te.to_numpy() + eps
        # PI is around the individual prediction mu_hat(X_test).
        # The cell-level bounds in res.frame are not what we want for an
        # individual-outcome coverage test; use the per-individual prediction
        # plus q_hat instead.
        em_pred = em.ml_info.predict_fn(df_te)
        lo = em_pred - res.q_hat
        hi = em_pred + res.q_hat
        covered.append(((lo <= y_te) & (y_te <= hi)).mean())
    emp = float(np.mean(covered))
    assert emp >= level - 0.03, (
        f"Empirical coverage {emp:.4f} undercovers nominal {level:.4f} "
        f"(distro={distro})"
    )


# ---------------------------------------------------------------------- conformal counterfactual


def _make_cf_em(*, n_tr=400, n_cal=300, seed=0):
    """Build a counterfactual-PI scenario with known Y(0), Y(1)."""
    rng = np.random.default_rng(seed)
    n_total = n_tr + n_cal + 200  # +200 test
    X = rng.standard_normal((n_total, 3))
    logit = 0.5 * X[:, 0] - 0.3 * X[:, 1]
    p1 = 1.0 / (1.0 + np.exp(-logit))
    T = (rng.random(n_total) < p1).astype(int)
    Y0 = 1.0 + 0.5 * X[:, 0] + 0.2 * X[:, 1] + rng.standard_normal(n_total)
    Y1 = 3.0 + 0.7 * X[:, 0] - 0.1 * X[:, 1] + rng.standard_normal(n_total)
    Y_obs = np.where(T == 1, Y1, Y0)

    df = pd.DataFrame({
        "treat": pd.Categorical(T),
        "x1": X[:, 0], "x2": X[:, 1], "x3": X[:, 2],
        "y": Y_obs,
        "_y1_true": Y1, "_y0_true": Y0,
    })
    tr = df.iloc[:n_tr].reset_index(drop=True)
    cal = df.iloc[n_tr:n_tr + n_cal].reset_index(drop=True)
    te = df.iloc[n_tr + n_cal:].reset_index(drop=True)

    # Outcome model on T=1 subset of train.
    tr_t1 = tr[tr["treat"] == 1]
    X_tr = tr_t1[["treat", "x1", "x2", "x3"]].astype(float).to_numpy()
    y_tr = tr_t1["y"].to_numpy()
    outcome = GradientBoostingRegressor(
        n_estimators=50, max_depth=3, random_state=seed,
    ).fit(X_tr, y_tr)

    def outcome_predict(data):
        return outcome.predict(
            data[["treat", "x1", "x2", "x3"]].astype(float).to_numpy()
        )

    # Propensity model on the full train set.
    Xp_tr = tr[["x1", "x2", "x3"]].to_numpy()
    Tp_tr = tr["treat"].astype(int).to_numpy()
    prop = LogisticRegression(max_iter=1000).fit(Xp_tr, Tp_tr)

    def propensity_predict(data):
        return prop.predict_proba(data[["x1", "x2", "x3"]].to_numpy())[:, 1]

    info = from_predict(
        predict_fn=outcome_predict, data=tr,
        factors={"treat": [0, 1]},
        numerics=["x1", "x2", "x3"], response="y",
    )
    em = ml_emmeans(info, "treat")
    return em, cal, te, propensity_predict


def test_conformal_counterfactual_pi_returns_documented_result():
    em, cal, _, prop_fn = _make_cf_em(n_tr=300, n_cal=200, seed=0)
    res = conformal_counterfactual_pi(em, cal, prop_fn, treatment_value=1, level=0.95)
    assert isinstance(res, ConformalCounterfactualResult)
    assert res.level == pytest.approx(0.95)
    assert res.treatment_value == 1
    assert "lower_pi" in res.frame.columns
    assert "upper_pi" in res.frame.columns


def test_conformal_counterfactual_pi_rejects_invalid_weight_clip():
    em, cal, _, prop_fn = _make_cf_em(n_tr=200, n_cal=200, seed=1)
    with pytest.raises(ValueError, match="weight_clip"):
        conformal_counterfactual_pi(em, cal, prop_fn, weight_clip=(0.5, 0.4))
    with pytest.raises(ValueError, match="weight_clip"):
        conformal_counterfactual_pi(em, cal, prop_fn, weight_clip=(-0.1, 0.9))


def test_conformal_counterfactual_pi_rejects_multinomial_for_now():
    em, cal, _, prop_fn = _make_cf_em(n_tr=200, n_cal=200, seed=2)
    with pytest.raises(NotImplementedError, match="binary"):
        conformal_counterfactual_pi(em, cal, prop_fn, treatment_value=2)


def test_conformal_counterfactual_pi_rejects_non_callable_propensity():
    em, cal, _, _ = _make_cf_em(n_tr=200, n_cal=200, seed=3)
    with pytest.raises(TypeError, match="callable"):
        conformal_counterfactual_pi(em, cal, "not-callable")  # type: ignore[arg-type]


def test_conformal_counterfactual_pi_empirical_coverage():
    """Across reps, weighted-conformal PI covers the TRUE Y(1) at ≥ nominal level.

    Verified specifically at T=0 test units — the missing-counterfactual case.
    """
    N_REPS = 40
    level = 0.90
    covered_all = []
    covered_t0 = []
    for rep in range(N_REPS):
        em, cal, te, prop_fn = _make_cf_em(n_tr=400, n_cal=300, seed=rep)
        res = conformal_counterfactual_pi(em, cal, prop_fn, treatment_value=1, level=level)
        # Apply per-individual: predict on test, then ± q_hat.
        mu_te = em.ml_info.predict_fn(te)
        lo = mu_te - res.q_hat
        hi = mu_te + res.q_hat
        y1_true = te["_y1_true"].to_numpy()
        covered_all.append(((lo <= y1_true) & (y1_true <= hi)).mean())
        mask_t0 = (te["treat"].astype(int) == 0).to_numpy()
        if mask_t0.sum() >= 5:
            covered_t0.append(
                ((lo[mask_t0] <= y1_true[mask_t0])
                 & (y1_true[mask_t0] <= hi[mask_t0])).mean()
            )
    emp_all = float(np.mean(covered_all))
    emp_t0 = float(np.mean(covered_t0))
    # 40 reps × ~200 test units = 8000 test points → MC SE ~ 0.003.
    # Allow 0.04 tolerance (~13× SE) for safety against rep-level noise.
    assert emp_all >= level - 0.04, (
        f"All-units empirical coverage {emp_all:.4f} undercovers {level:.2f}"
    )
    assert emp_t0 >= level - 0.05, (
        f"T=0 empirical coverage {emp_t0:.4f} undercovers {level:.2f} "
        f"(the missing-counterfactual case)"
    )


def test_conformal_counterfactual_pi_q_hat_increases_with_level():
    em, cal, _, prop_fn = _make_cf_em(n_tr=300, n_cal=300, seed=4)
    r80 = conformal_counterfactual_pi(em, cal, prop_fn, level=0.80)
    r95 = conformal_counterfactual_pi(em, cal, prop_fn, level=0.95)
    assert r95.q_hat >= r80.q_hat
