"""Tests for hypothesis= (row contrasts) and transform= on the
predictions / slopes / comparisons family."""

from __future__ import annotations

import numpy as np
import pandas as pd
import patsy
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymmeans import avg_comparisons, avg_predictions, avg_slopes


def _logit_interaction(n=600, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "x": rng.standard_normal(n),
        "g": pd.Categorical(rng.choice(["A", "B", "C"], n)),
    })
    slope = df["g"].map({"A": 0.5, "B": 1.5, "C": 2.5}).astype(float)
    df["yb"] = (rng.random(n) < 1.0 / (1.0 + np.exp(-(0.2 + df["x"] * slope)))).astype(int)
    return smf.glm("yb ~ x * g", df, family=sm.families.Binomial()).fit(), df


# ---------------------------------------------------------------------- ordering


def test_by_groups_are_in_factor_level_order():
    fit, _ = _logit_interaction()
    frame = avg_slopes(fit, "x", by="g", type="response").frame
    assert list(frame["g"]) == ["A", "B", "C"]


# ---------------------------------------------------------------------- hypothesis labels + point


def test_pairwise_labels_and_point_estimates():
    fit, _ = _logit_interaction()
    base = avg_slopes(fit, "x", by="g", type="response").frame.set_index("g")["slope"]
    pw = avg_slopes(fit, "x", by="g", type="response", hypothesis="pairwise").frame
    assert list(pw["hypothesis"]) == ["A - B", "A - C", "B - C"]
    assert float(pw.set_index("hypothesis").loc["A - B", "slope"]) == pytest.approx(
        float(base["A"] - base["B"]), abs=1e-9
    )
    assert float(pw.set_index("hypothesis").loc["B - C", "slope"]) == pytest.approx(
        float(base["B"] - base["C"]), abs=1e-9
    )


def test_reference_and_sequential_labels():
    fit, _ = _logit_interaction()
    ref = avg_slopes(fit, "x", by="g", type="response", hypothesis="reference").frame
    assert list(ref["hypothesis"]) == ["B - A", "C - A"]
    seq = avg_slopes(fit, "x", by="g", type="response", hypothesis="sequential").frame
    assert list(seq["hypothesis"]) == ["B - A", "C - B"]


# ------------------------------------------------------------- hypothesis SE (ground truth)


def test_pairwise_se_matches_independent_analytic_delta():
    """The hypothesis contrast SE equals sqrt(L J V J^T L^T), verified
    against an independently constructed Jacobian."""
    fit, df = _logit_interaction()
    pw = avg_slopes(fit, "x", by="g", type="response", hypothesis="pairwise").frame
    pw = pw.set_index("hypothesis")

    b = np.asarray(fit.params)
    V = np.asarray(fit.cov_params())
    di = fit.model.data.design_info
    def _design(frame):
        return np.asarray(
            patsy.build_design_matrices([di], frame, return_type="matrix")[0]
        )

    X = _design(df)
    h = 1e-6
    ls = (_design(df.assign(x=df["x"] + h)) - _design(df.assign(x=df["x"] - h))) / (2 * h)

    def theta(bb):
        p = 1.0 / (1.0 + np.exp(-(X @ bb)))
        rs = p * (1 - p) * (ls @ bb)
        return np.array([rs[(df["g"] == lv).to_numpy()].mean() for lv in ["A", "B", "C"]])

    jac = np.zeros((3, len(b)))
    for k in range(len(b)):
        s = 1e-6 * max(1.0, abs(b[k]))
        bp = b.copy(); bp[k] += s
        bm = b.copy(); bm[k] -= s
        jac[:, k] = (theta(bp) - theta(bm)) / (2 * s)
    cov = jac @ V @ jac.T

    for (i, j, lab) in [(0, 1, "A - B"), (0, 2, "A - C"), (1, 2, "B - C")]:
        ell = np.zeros(3); ell[i] = 1.0; ell[j] = -1.0
        truth_se = float(np.sqrt(ell @ cov @ ell.T))
        assert float(pw.loc[lab, "SE"]) == pytest.approx(truth_se, abs=1e-7), lab


def test_custom_contrast_matrix():
    """A numeric L matrix is applied directly."""
    fit, _ = _logit_interaction()
    base = avg_slopes(fit, "x", by="g", type="response").frame["slope"].to_numpy()
    # average of A and B minus C
    L = np.array([[0.5, 0.5, -1.0]])
    res = avg_slopes(fit, "x", by="g", type="response", hypothesis=L).frame
    assert len(res) == 1
    assert float(res["slope"].iloc[0]) == pytest.approx(
        0.5 * base[0] + 0.5 * base[1] - base[2], abs=1e-9
    )


# ---------------------------------------------------------------------- predictions / comparisons


def test_avg_predictions_hypothesis():
    fit, _ = _logit_interaction()
    ph = avg_predictions(fit, by="g", hypothesis="reference").frame
    assert list(ph["hypothesis"]) == ["B - A", "C - A"]


def test_avg_comparisons_hypothesis():
    fit, _ = _logit_interaction()
    h = avg_comparisons(fit, "x", by="g", hypothesis="pairwise").frame
    # 3 pairwise contrasts among the A, B, C group comparisons
    assert len(h) == 3
    assert np.isfinite(h["SE"]).all()


# ---------------------------------------------------------------------- transform


def test_transform_applies_to_estimate_and_ci():
    fit, _ = _logit_interaction()
    nt = avg_comparisons(fit, "x", comparison="lnratio").frame.iloc[0]
    tr = avg_comparisons(fit, "x", comparison="lnratio", transform=np.exp).frame.iloc[0]
    assert float(tr["estimate"]) == pytest.approx(float(np.exp(nt["estimate"])), abs=1e-9)
    assert float(tr["lower_cl"]) == pytest.approx(float(np.exp(nt["lower_cl"])), abs=1e-9)
    assert float(tr["upper_cl"]) == pytest.approx(float(np.exp(nt["upper_cl"])), abs=1e-9)
    assert np.isnan(tr["SE"])  # SE is ill-defined on the transformed scale


def test_transform_with_hypothesis():
    fit, _ = _logit_interaction()
    # exp of a difference contrast -> ratio-of-effects style readout
    res = avg_slopes(
        fit, "x", by="g", type="response", hypothesis="reference", transform=np.exp
    ).frame
    assert (res["slope"] > 0).all()
    assert res["SE"].isna().all()


# ---------------------------------------------------------------------- validation


def test_rejects_unknown_hypothesis():
    fit, _ = _logit_interaction()
    with pytest.raises(ValueError, match="hypothesis"):
        avg_slopes(fit, "x", by="g", hypothesis="nonsense")


def test_rejects_wrong_shape_contrast_matrix():
    fit, _ = _logit_interaction()
    with pytest.raises(ValueError, match="columns"):
        avg_slopes(fit, "x", by="g", hypothesis=np.array([[1.0, -1.0]]))  # 2 != 3
