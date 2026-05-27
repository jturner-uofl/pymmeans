"""Estimability checks for rank-deficient designs.

A contrast ``c'β`` is **estimable** iff ``c`` lies in the row space of
the design matrix X. For full-rank X (the common case) every contrast
is estimable. For rank-deficient X — missing factor combinations,
perfect collinearity between predictors, dummy-variable traps, etc. —
some contrasts depend on which generalised inverse statsmodels used to
compute β̂, and their numerical values are an artefact of the solver
rather than a fact about the population. R ``emmeans`` flags these as
``NA``; we mirror that with ``NaN`` plus an explicit ``UserWarning``.

The detection uses SVD: a contrast ``c`` is estimable iff it is
orthogonal to every column of the null-space basis of X. We compute
that basis once via :func:`numpy.linalg.svd` (full matrices, then keep
the right-singular vectors corresponding to singular values below a
relative tolerance) and project each row of L onto it.

Edge cases:

- Full-rank X short-circuits to an all-True mask without computing the
  projection.
- The tolerance scales with the largest singular value, so badly
  conditioned but technically full-rank designs (e.g. Hilbert-style)
  are treated as full rank. this is the
  defensible behaviour; expose ``rtol=`` here if a future user needs
  to tune it.

References
----------
- Searle, S. R., Speed, F. M., & Milliken, G. A. (1980). "Population
  Marginal Means in the Linear Model: An Alternative to Least Squares
  Means." *The American Statistician* 34(4), 216-221.
- Searle, S. R. (1971). *Linear Models*. Wiley. Chapter on estimability.
"""

from __future__ import annotations

import numpy as np


def null_space_basis(X: np.ndarray, rtol: float | None = None) -> np.ndarray:
    """Orthonormal basis for the null space of X.

    Columns that lie in null(X) correspond to linear dependencies among
    X's columns. A contrast c is estimable iff c is orthogonal to every
    column of this basis.

    Parameters
    ----------
    X
        Design matrix of shape (n_obs, n_params).
    rtol
        Relative tolerance for declaring a singular value zero. Default
        uses ``max(X.shape) * eps * sigma_max``.

    Returns
    -------
    ndarray of shape (n_params, n_null), or (n_params, 0) if X has full
    column rank.
    """
    X = np.asarray(X, dtype=float)
    if X.size == 0:
        return np.zeros((X.shape[1], 0))
    # #7: full_matrices=True materialises an (n, n) U
    # block which dominates runtime on tall designs (n >> p). For
    # n >= p, full_matrices=False gives the same (p, p) Vt at much
    # lower cost. For wide designs (n < p) we still need the full
    # (p, p) Vt to recover null-space directions in rows n..p, so
    # keep full_matrices=True there.
    full = X.shape[0] < X.shape[1]
    _U, s, Vt = np.linalg.svd(X, full_matrices=full)
    if rtol is None:
        rtol = max(X.shape) * np.finfo(s.dtype).eps
    tol = rtol * (s.max() if s.size else 1.0)
    rank = int(np.sum(s > tol))
    if rank == X.shape[1]:
        return np.zeros((X.shape[1], 0))
    return Vt[rank:, :].T # shape (n_params, n_null)


def estimable_mask(
    L: np.ndarray, X: np.ndarray, rtol: float | None = None
) -> np.ndarray:
    """Per-row boolean mask: True if the row of L is estimable under X.

    For full-rank X returns an all-True mask in O(rank computation only).
    """
    if L.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    # an EMM row with all-NaN entries (e.g.
    # an empty observed cell under `weights='cells'`) used to poison
    # the global tolerance via `np.abs(L).max()` returning NaN, which
    # made `np.abs(proj) < NaN` evaluate to False for EVERY row.
    # That marked all rows non-estimable even when the model was full
    # rank and the OTHER L rows were perfectly fine. Mask non-finite
    # rows separately: they get `False` (correctly non-estimable) and
    # are excluded from the tolerance calculation.
    finite_rows = np.all(np.isfinite(L), axis=1)
    out = np.zeros(L.shape[0], dtype=bool)
    if not finite_rows.any():
        return out
    Lf = L[finite_rows]
    null = null_space_basis(X, rtol=rtol)
    if null.shape[1] == 0:
        out[finite_rows] = True
        return out
    # c is estimable iff c · n = 0 for every null-basis vector n
    proj = Lf @ null # shape (n_finite_L, n_null)
    if rtol is None:
        rtol = max(X.shape) * np.finfo(float).eps
    # Use the finite subset's max for the tolerance, not the full L's
    # (otherwise a NaN would propagate).
    threshold = rtol * (np.abs(Lf).max() if Lf.size else 1.0) * 100.0
    out[finite_rows] = np.all(np.abs(proj) < threshold, axis=1)
    return out


def estimable_mask_from_basis(
    L: np.ndarray, row_space_basis: np.ndarray, rtol: float | None = None
) -> np.ndarray:
    """Estimability check using a precomputed orthonormal row-space basis.

    Used when the original design matrix X isn't available — e.g. after a
    pickle round-trip that dropped ``raw_result`` — but ``ModelInfo``
    carries the right-singular vectors of X corresponding to non-trivial
    singular values (built once at adapter time via
    :func:`pymmeans.utils._build_estimability_basis`).

    A contrast ``c`` is estimable iff it lies in row(X), so
    ``||c - V_r V_r^T c|| / ||c||`` should be ~0 for estimable rows.
    """
    if L.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    if row_space_basis is None or row_space_basis.size == 0:
        return np.ones(L.shape[0], dtype=bool)
    V = np.asarray(row_space_basis, dtype=float)
    proj = L @ V.T @ V # rows projected back into row(X)
    resid = L - proj
    if rtol is None:
        rtol = np.finfo(float).eps
    row_norm = np.linalg.norm(L, axis=1)
    resid_norm = np.linalg.norm(resid, axis=1)
    threshold = np.maximum(rtol * (row_norm * 100.0), rtol * 100.0)
    return resid_norm < threshold
