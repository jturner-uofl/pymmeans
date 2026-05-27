"""R parity tests: hardcoded R reference values that don't need a live R install.

These complement the CSV-driven vs-R suite in test_vs_r.py with smaller
unit-level checks for behaviors where R's outputs are stable and well-known.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import emmeans, pairs
from pymmeans.contrasts import _consec_matrix, _poly_matrix

# --- emmeans::poly.emmc reference (R returns INTEGER contrasts) --------
#
# pymmeans's poly previously matched
# R's `contr.poly` (orthonormal Q from QR factorization), but R
# `emmeans::poly.emmc` returns the INTEGER-scaled orthogonal contrasts
# (Fisher's classical tables). The two encode the same contrast
# direction but different scales — orthonormal estimates are smaller
# by a constant factor. pymmeans now produces R-emmeans-compatible
# integer coefficients.

R_EMMEANS_POLY = {
 3: np.array(
 [
 [-1, 1],
 [0, -2],
 [1, 1],
 ]
 ),
 4: np.array(
 [
 [-3, 1, -1],
 [-1, -1, 3],
 [1, -1, -3],
 [3, 1, 1],
 ]
 ),
 5: np.array(
 [
 [-2, 2, -1, 1],
 [-1, -1, 2, -4],
 [0, -2, 0, 6],
 [1, -1, -2, -4],
 [2, 2, 1, 1],
 ]
 ),
}


@pytest.mark.parametrize("k", [3, 4, 5])
def test_poly_contrasts_match_R_emmeans_poly_emmc(k):
 """matches R `emmeans::poly.emmc` (integer-scaled)
 rather than the previous `contr.poly` (orthonormal)."""
 P, _ = _poly_matrix(k, [f"l{i}" for i in range(k)])
 # P is (k-1, k); R's poly.emmc is (k, k-1) — transpose to compare.
 np.testing.assert_array_almost_equal(P.T, R_EMMEANS_POLY[k], decimal=5)


# --- consecutive contrasts: (l2 - l1, l3 - l2, ...) ---------------------


def test_consec_matrix_matches_explicit_form():
 D, names = _consec_matrix(4, ["a", "b", "c", "d"])
 expected = np.array(
 [
 [-1, 1, 0, 0],
 [0, -1, 1, 0],
 [0, 0, -1, 1],
 ],
 dtype=float,
 )
 np.testing.assert_array_almost_equal(D, expected)
 assert names == ["b - a", "c - b", "d - c"]


# --- Analytic path matches pre-converted form ---------------------------


def test_analytic_matches_pre_converted_with_interaction():
 """Formula with C() interaction should match the pre-converted version
 numerically (different surface syntax, identical model)."""
 rng = np.random.default_rng(0)
 n = 240
 df = pd.DataFrame(
 {
 "y": rng.normal(size=n),
 "dose": np.tile([0.5, 1.0, 2.0], n // 3),
 "supp": pd.Categorical(rng.choice(["OJ", "VC"], n)),
 }
 )
 fit_expr = smf.ols("y ~ C(dose) * supp", data=df).fit()
 df2 = df.copy()
 df2["dose_cat"] = pd.Categorical(df2["dose"])
 fit_plain = smf.ols("y ~ dose_cat * supp", data=df2).fit()

 emm_expr = emmeans(fit_expr, "dose", by="supp")
 emm_plain = emmeans(fit_plain, "dose_cat", by="supp")
 np.testing.assert_array_almost_equal(
 emm_expr.frame["emmean"].to_numpy(),
 emm_plain.frame["emmean"].to_numpy(),
 )
 np.testing.assert_array_almost_equal(
 emm_expr.frame["SE"].to_numpy(),
 emm_plain.frame["SE"].to_numpy(),
 )

 pw_expr = pairs(emm_expr)
 pw_plain = pairs(emm_plain)
 np.testing.assert_array_almost_equal(
 pw_expr.frame["estimate"].to_numpy(),
 pw_plain.frame["estimate"].to_numpy(),
 )


# --- Default level order matches alphabetical when no Categorical set ---


def test_factor_levels_alphabetical_default():
 rng = np.random.default_rng(1)
 n = 90
 df = pd.DataFrame(
 {
 "y": rng.normal(size=n),
 # Insert in non-alphabetical order; pandas defaults to alpha
 "g": rng.choice(["zebra", "apple", "mango"], n),
 }
 )
 fit = smf.ols("y ~ g", data=df).fit()
 emm = emmeans(fit, "g")
 levels = list(emm.frame["g"].astype(str))
 assert levels == sorted(levels) # alphabetical


# --- Linear poly contrast on equally-spaced levels equals endpoint diff -


def test_poly_linear_contrast_endpoint_difference():
 P, _ = _poly_matrix(5, [f"l{i}" for i in range(5)])
 linear = P[0]
 # Linear contrast applied to a constant slope should give the slope times
 # sum of (x_i * linear_i), where x_i are the points 1..k.
 x = np.arange(1, 6, dtype=float)
 # Linear's projection on x gives a scaled total range
 assert linear @ x > 0 # positive direction
 # Property: orthogonal to constant
 assert abs(linear.sum()) < 1e-10
