"""Satterthwaite and Kenward-Roger inference for MixedLM contrasts.

We work in the "unscaled" parametrisation that statsmodels exposes:
``theta = vec(cov_re / scale)`` (the free upper-triangular elements) plus
the residual variance ``scale`` (treated as fixed at its REML estimate —
"profile" Satterthwaite/KR).

For each contrast c:
    Satterthwaite: df = 2 * (c'V_b c)^2 / Var(c'V_b c)
    Kenward-Roger: V_b is replaced with V_b^KR before computing the df

where ``Var(c'V_b c)`` is computed via the delta method, with the gradient
of c'V_b c w.r.t. theta from central differences and ``Cov(theta_hat)``
from ``result.cov_params``. The KR inflation is
    V_b^KR = V_b + sum_ij W_ij V_b (Q_ij - (P_i V_b P_j + P_j V_b P_i)/2) V_b
where P_i = dA/dtheta_i, Q_ij = d^2 A/(dtheta_i dtheta_j), A = V_b^{-1}.

v0.1 scope:
- Works for any ``statsmodels.MixedLM`` (random intercept, random slopes,
  multi-level random effects).
- ``scale = sigma_e^2`` is profiled (treated as known); statsmodels
  doesn't expose Var(sigma_e^2_hat) directly. For models where the
  residual variance dominates uncertainty, this is a mild approximation.
"""

from __future__ import annotations

import warnings
from typing import Any, NamedTuple

import numpy as np

from pymmeans.utils import ModelInfo


class _SattCache(NamedTuple):
    """Cached REML state for Satterthwaite / Kenward-Roger.

    ``apply_satterthwaite`` / ``apply_kenward_roger`` extract
    everything they need from ``raw_result`` into this NamedTuple and
    attach it to the returned EMMResult / ContrastResult. The cache
    survives pickle (just numpy arrays + plain floats / ints), so a
    Satt-applied EMM can round-trip through ``pickle.dumps`` /
    ``pickle.loads`` and downstream ``pairs`` / ``contrast`` calls still
    auto-propagate the correction at the new contrast L matrix.

    Without the cache, ``ModelInfo.__getstate__`` (correctly) drops
    ``raw_result`` because statsmodels Results aren't pickle-friendly,
    which used to mean Satt / KR refused on the unpickled EMM. With
    the cache, the post-pickle path skips ``raw_result`` entirely and
    uses these arrays directly.

    Fields
    ------
    theta_hat
        Flat free elements of Lambda (lmer Cholesky factor) at the
        REML optimum.
    k_re
        Number of random-effects columns (size of Lambda).
    sigma_sq_hat
        Residual variance at the REML optimum.
    X
        Fixed-effect design (n, p).
    Z
        Random-effect design (n, q).
    y
        Response vector (n,).
    groups
        Group labels (n,).
    group_ids
        Unique group labels (n_groups,).
    """

    theta_hat: np.ndarray
    k_re: int
    sigma_sq_hat: float
    X: np.ndarray
    Z: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    group_ids: np.ndarray
    # vc_formula= support. ``theta_vc_hat`` holds
    # sqrt(vcomp_v / sigma_sq) for each variance component (the lmer
    # relative-covariance-factor of each scalar VC). ``vc_mats`` is a
    # tuple of length ``k_vc``; each element is a dict
    # ``{group_id: design_matrix}`` carrying that VC's per-group design
    # block (from statsmodels ``model.exog_vc.mats``). Defaults make
    # the cov_re-only path byte-identical to the pre-vc_formula
    # behaviour (k_vc=0 → no VC blocks added anywhere).
    theta_vc_hat: np.ndarray = np.empty(0)
    k_vc: int = 0
    vc_mats: tuple[dict, ...] = ()


class KRInternals(NamedTuple):
    """Bundle of by-products from ``kenward_roger_vcov`` reused by
    ``apply_kenward_roger`` for the df calculation.

    #1: the rewrite of ``kenward_roger_vcov`` already
    computes the expected REML information matrix ``IE2`` of ``theta``
    en route to the KR-inflated vcov, but the function previously
    discarded it. ``apply_kenward_roger`` then independently rebuilt a
    separate observed-Hessian ``Cov(theta)`` via finite-diff on
    ``_reml_deviance`` for the df calculation — duplicating work and
    introducing a *second* approximation that could drift from pbkrtest.

    Exposing the canonical KR weight matrix ``W = 2 · inv(IE2)`` (which
    is also the asymptotic ``Cov(theta)`` under REML) lets the df path
    reuse pbkrtest's own asymptotic covariance, matching pbkrtest's
    ``ddf_Lb`` exactly and dropping the quartic
    ``n_theta·(n_theta+1)/2 · 4`` REML deviance evaluations.

    Fields
    ------
    V_KR
        KR-inflated fixed-effect vcov (p × p).
    W
        ``2 · inv(IE2)`` — canonical asymptotic ``Cov(theta_hat)`` under
        REML (the pbkrtest weight matrix). Used as ``cov_theta`` in
        ``apply_kenward_roger``'s delta-method df.
    V_beta
        Unadjusted REML vcov ``(X' Sigma^-1 X)^-1`` at the MLE; reused
        to skip one matrix solve in the df gradient.
    P_list
        Tuple of length ``n_theta`` carrying the per-variance-component
        first derivatives ``P[r] = ∂V_beta/∂θ_r``. Computed via the
        chain rule from ``PP_arr`` (the per-θ derivatives of
        ``inv(V_beta)``) at no marginal cost. Required by
        :func:`pymmeans.pbktest.krmodcomp` for the Kenward-Roger
        F-test scale factor and denominator df, which depend on the
        moments of ``L V_KR L'`` with respect to θ.
    """

    V_KR: np.ndarray
    W: np.ndarray
    V_beta: np.ndarray
    P_list: tuple[np.ndarray, ...]


def _build_satt_cache(result: Any) -> _SattCache:
    """Snapshot the REML state from a fitted MixedLM result."""
    theta_lmer_hat, k_re = _lmer_theta_hat(result)
    theta_vc_hat, k_vc, vc_mats = _lmer_vc_theta_hat(result)
    sigma_sq_hat = float(result.scale)
    X = np.asarray(result.model.exog, dtype=float)
    Z = np.asarray(result.model.exog_re, dtype=float)
    y = np.asarray(result.model.endog, dtype=float)
    groups = np.asarray(result.model.groups)
    group_ids = np.unique(groups)
    return _SattCache(
        theta_hat=theta_lmer_hat,
        k_re=int(k_re),
        sigma_sq_hat=sigma_sq_hat,
        X=X,
        Z=Z,
        y=y,
        groups=groups,
        group_ids=group_ids,
        theta_vc_hat=theta_vc_hat,
        k_vc=int(k_vc),
        vc_mats=vc_mats,
    )


# removed six helper functions that were
# defined for the pre-statsmodels-unscaled-parameterisation
# Satterthwaite/KR path but became unreachable after the
# rewrite to the lme4 parameterisation:
# _theta_unscaled_hat, _expand_g, _xtvinv_x, _vbeta_at,
# _cov_theta, _grad_vbeta_diag.
# Each was only called by the others, and the chain was dropped
# entirely from apply_satterthwaite / apply_kenward_roger.


# --- lmer-parametrization Satterthwaite (review / lme4 parity) ---
#
# lmerTest parameterises variance components as (Lambda, sigma_e^2) where
# Lambda is the lower-triangular Cholesky factor of cov_re/sigma_e^2 (so
# G = Lambda Lambda^T * sigma_e^2). The Cov(theta_lmer, sigma_e^2) used
# in the Satterthwaite formula comes from inverting the Hessian of the
# REML deviance w.r.t. these parameters. statsmodels parameterises
# differently (cov_re_unscaled, scale profiled) and the cov_params it
# returns isn't an inverse Hessian in the same parametrisation, leading
# to a ~15 % df underestimate on a simple random-intercept fit.
#
# We replicate lmer's approach: compute the REML deviance ourselves and
# take a finite-difference Hessian.


def _vech_lower(L: np.ndarray) -> np.ndarray:
    """Lower-triangular vech (column-major: L_11, L_21, L_31, L_22, L_32, L_33)."""
    k = L.shape[0]
    out = []
    for j in range(k):
        for i in range(j, k):
            out.append(L[i, j])
    return np.asarray(out, dtype=float)


def _lmer_theta_to_lambda(theta_lmer: np.ndarray, k_re: int) -> np.ndarray:
    """Invert ``_vech_lower``: rebuild the lower-triangular Lambda."""
    L = np.zeros((k_re, k_re))
    idx = 0
    for j in range(k_re):
        for i in range(j, k_re):
            L[i, j] = theta_lmer[idx]
            idx += 1
    return L


def _lmer_theta_hat(result: Any) -> tuple[np.ndarray, int]:
    """Return (theta_lmer_hat, k_re) for a fitted MixedLM result.

    Raises ``BoundaryFitError`` when ``cov_re`` is exactly singular or
    has any near-zero variance component on the diagonal — finite-
    difference Hessian / Cholesky factorisation are meaningless there.
    """
    cov_re = np.asarray(result.cov_re, dtype=float)
    scale = float(result.scale)
    if scale <= 0:
        raise BoundaryFitError(
            f"Residual variance is non-positive ({scale}); the model is "
            "at or past the parameter boundary."
        )
    G_unscaled = cov_re / scale
    # #4 / #4: detect boundary BEFORE Cholesky
    # factorisation, which raises LinAlgError (not BoundaryFitError) on
    # singular input. This is a diagonal check — it catches zero/near-zero
    # variance components (the common boundary case). Genuine non-PSD G
    # with positive diagonal (e.g. estimated correlation ~±1) slips past
    # this gate and is caught by the Cholesky failure below, which re-raises
    # as BoundaryFitError.
    diag_min = float(np.min(np.diag(G_unscaled))) if G_unscaled.size else 0.0
    if diag_min <= _KR_BOUNDARY_TOL ** 2:
        raise BoundaryFitError(
            f"Random-effects variance is at or near the boundary "
            f"(min diag(cov_re/scale) = {diag_min:.2e}). "
            "Use apply_satterthwaite() with a different RE structure "
            "or refit with a non-boundary model."
        )
    try:
        Lambda = np.linalg.cholesky(G_unscaled)
    except np.linalg.LinAlgError as exc:
        raise BoundaryFitError(
            f"Random-effects covariance is not positive-definite "
            f"(Cholesky failed). cov_re/scale =\n{G_unscaled}"
        ) from exc
    return _vech_lower(Lambda), G_unscaled.shape[0]


def _lmer_vc_theta_hat(
    result: Any,
) -> tuple[np.ndarray, int, tuple[dict, ...]]:
    """Extract the variance-component (``vc_formula=``) state.

    Returns ``(theta_vc_hat, k_vc, vc_mats)`` where:

    - ``theta_vc_hat`` is ``sqrt(vcomp_v / sigma_sq)`` for each scalar
      variance component (the lmer relative-covariance factor of a
      1×1 block).
    - ``k_vc`` is the number of variance components.
    - ``vc_mats`` is a tuple of length ``k_vc``; each element is a
      dict ``{group_id: design_matrix}`` mapping every group label to
      that VC's per-group design block (from
      ``model.exog_vc.mats[v]``, which is indexed by
      ``model.group_labels`` order — we re-key by the actual label so
      the deviance loop can look it up regardless of group ordering).

    Returns ``(empty, 0, ())`` for a model with no ``vc_formula=``.
    """
    model = getattr(result, "model", None)
    k_vc = int(getattr(model, "k_vc", 0) or 0)
    if not k_vc:
        return np.empty(0), 0, ()
    scale = float(result.scale)
    vcomp = np.asarray(result.vcomp, dtype=float)
    # vcomp can be exactly 0 at the boundary; clip the ratio at 0 so
    # the sqrt is well-defined (the boundary check downstream handles
    # the near-zero case).
    theta_vc = np.sqrt(np.maximum(vcomp / scale, 0.0))

    exog_vc = model.exog_vc
    group_labels = list(model.group_labels)
    # mats[v] is a list indexed by group order; build {group_id: mat}.
    vc_mats: list[dict] = []
    for v in range(k_vc):
        per_group = exog_vc.mats[v]
        d: dict = {}
        for gi, g_lab in enumerate(group_labels):
            d[g_lab] = np.asarray(per_group[gi], dtype=float)
        vc_mats.append(d)
    return theta_vc, k_vc, tuple(vc_mats)


def _vc_cov_block(
    vc_mats: tuple[dict, ...],
    g_id: Any,
    theta_vc: np.ndarray,
    sigma_sq: float,
    n_i: int,
) -> np.ndarray:
    """Sum the per-group variance-component covariance contribution.

    Returns ``sum_v vcomp_v · (Z_vc · Z_vc')`` for group ``g_id`` in
    ABSOLUTE units, where ``vcomp_v = theta_vc[v]**2 · sigma_sq``
    (matching the absolute-unit ``V_i`` the REML / V_beta builders
    use). Returns an ``(n_i, n_i)`` zero matrix when there are no
    variance components. Groups absent from a VC's design map (no
    levels of that VC present in the group) contribute nothing.
    """
    out = np.zeros((n_i, n_i))
    if not vc_mats:
        return out
    for v, mats_v in enumerate(vc_mats):
        Zvc = mats_v.get(g_id)
        if Zvc is None or Zvc.size == 0:
            continue
        out = out + (theta_vc[v] ** 2 * sigma_sq) * (Zvc @ Zvc.T)
    return out


def _reml_deviance(
    model: Any,
    theta_lmer: np.ndarray,
    sigma_sq: float,
    k_re: int,
    X: np.ndarray,
    Z: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    group_ids: np.ndarray,
    theta_vc: np.ndarray | None = None,
    vc_mats: tuple[dict, ...] = (),
) -> float:
    """REML deviance at (theta_lmer, theta_vc, sigma_sq). Used for finite-
    difference Hessian to recover Cov(theta, sigma_sq) in lmer's
    parametrisation. ``theta_vc`` / ``vc_mats`` add the ``vc_formula=``
    variance-component blocks to each per-group ``V_i``; both default to
    empty (the cov_re-only path is unchanged)."""
    Lambda = _lmer_theta_to_lambda(theta_lmer, k_re)
    G = Lambda @ Lambda.T * sigma_sq
    n_p = X.shape[1]
    A = np.zeros((n_p, n_p))
    log_det_V_sum = 0.0
    XtVinvy = np.zeros(n_p)
    for g_id in group_ids:
        m = groups == g_id
        n_i = int(m.sum())
        X_i = X[m]
        Z_i = Z[m]
        y_i = y[m]
        V_i = Z_i @ G @ Z_i.T + sigma_sq * np.eye(n_i)
        if vc_mats:
            V_i = V_i + _vc_cov_block(vc_mats, g_id, theta_vc, sigma_sq, n_i)
        V_i_inv = np.linalg.inv(V_i)
        _, ldet = np.linalg.slogdet(V_i)
        log_det_V_sum += ldet
        A += X_i.T @ V_i_inv @ X_i
        XtVinvy += X_i.T @ V_i_inv @ y_i
    _, ldetA = np.linalg.slogdet(A)
    beta_hat = np.linalg.solve(A, XtVinvy)
    resid_sq = 0.0
    for g_id in group_ids:
        m = groups == g_id
        n_i = int(m.sum())
        X_i = X[m]
        Z_i = Z[m]
        y_i = y[m]
        V_i = Z_i @ G @ Z_i.T + sigma_sq * np.eye(n_i)
        if vc_mats:
            V_i = V_i + _vc_cov_block(vc_mats, g_id, theta_vc, sigma_sq, n_i)
        V_i_inv = np.linalg.inv(V_i)
        r = y_i - X_i @ beta_hat
        resid_sq += r @ V_i_inv @ r
    return float(log_det_V_sum + ldetA + resid_sq)


def _vbeta_at_lmer(
    model: Any,
    theta_lmer: np.ndarray,
    sigma_sq: float,
    k_re: int,
    X: np.ndarray,
    Z: np.ndarray,
    groups: np.ndarray,
    group_ids: np.ndarray,
    theta_vc: np.ndarray | None = None,
    vc_mats: tuple[dict, ...] = (),
) -> np.ndarray:
    """V_beta at (theta_lmer, theta_vc, sigma_sq) in lmer's parametrisation.

    ``theta_vc`` / ``vc_mats`` add the ``vc_formula=`` variance-component
    blocks to each per-group ``V_i``; both default to empty (the
    cov_re-only path is unchanged)."""
    Lambda = _lmer_theta_to_lambda(theta_lmer, k_re)
    G = Lambda @ Lambda.T * sigma_sq
    n_p = X.shape[1]
    A = np.zeros((n_p, n_p))
    for g_id in group_ids:
        m = groups == g_id
        n_i = int(m.sum())
        X_i = X[m]
        Z_i = Z[m]
        V_i = Z_i @ G @ Z_i.T + sigma_sq * np.eye(n_i)
        if vc_mats:
            V_i = V_i + _vc_cov_block(vc_mats, g_id, theta_vc, sigma_sq, n_i)
        A += X_i.T @ np.linalg.inv(V_i) @ X_i
    return np.linalg.inv(A)


def _satterthwaite_df_lmer(
    info: ModelInfo,
    L: np.ndarray,
    cache: _SattCache | None = None,
) -> np.ndarray:
    """lmerTest-style Satterthwaite df via Hessian of REML deviance.

    Matches ``lmerTest::lmer + summary(ddf='Satterthwaite')`` to 4
    decimals on a random-intercept reference fit; review
    confirmed the previous statsmodels-cov_params path underestimated
    df by ~15 % on the same fit.

    accepts an optional ``cache`` of pre-extracted REML
    state, used when ``info.raw_result`` was dropped (typically after
    pickle). ``apply_satterthwaite`` builds the cache from the live
    result and attaches it to the returned EMMResult so this path
    works post-pickle without raw_result.
    """
    if cache is not None:
        theta_lmer_hat = cache.theta_hat
        k_re = cache.k_re
        sigma_sq_hat = cache.sigma_sq_hat
        X = cache.X
        Z = cache.Z
        y = cache.y
        groups = cache.groups
        group_ids = cache.group_ids
        theta_vc_hat = cache.theta_vc_hat
        k_vc = cache.k_vc
        vc_mats = cache.vc_mats
        model_handle: Any = None # not used by _reml_deviance / _vbeta_at_lmer
    else:
        result = info.raw_result
        if result is None or not hasattr(result, "cov_re"):
            return np.full(L.shape[0], np.inf)
        theta_lmer_hat, k_re = _lmer_theta_hat(result)
        theta_vc_hat, k_vc, vc_mats = _lmer_vc_theta_hat(result)
        sigma_sq_hat = float(result.scale)
        X = np.asarray(result.model.exog, dtype=float)
        Z = np.asarray(result.model.exog_re, dtype=float)
        y = np.asarray(result.model.endog, dtype=float)
        groups = np.asarray(result.model.groups)
        group_ids = np.unique(groups)
        model_handle = result.model

    # augmented θ ordering [theta_re, theta_vc, sigma_sq].
    # theta_vc is empty for cov_re-only fits, so this collapses to the
    # original [theta_re, sigma_sq] packing.
    n_re2 = len(theta_lmer_hat)
    full_theta = np.concatenate(
        [theta_lmer_hat, theta_vc_hat, [sigma_sq_hat]]
    )
    n_theta = len(full_theta)

    def _unpack(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        theta_re = vec[:n_re2]
        theta_vc = vec[n_re2:n_re2 + k_vc]
        sig = float(vec[-1])
        return theta_re, theta_vc, sig

    def deviance(vec: np.ndarray) -> float:
        tr, tv, sig = _unpack(vec)
        return _reml_deviance(
            model_handle, tr, sig, k_re, X, Z, y, groups, group_ids,
            theta_vc=tv, vc_mats=vc_mats,
        )

    # Central-difference Hessian. Step sized per parameter scale.
    h = np.maximum(np.abs(full_theta), 1e-6) * 1e-3
    H = np.zeros((n_theta, n_theta))
    for i in range(n_theta):
        for j in range(i, n_theta):
            x_pp = full_theta.copy()
            x_pp[i] += h[i]
            x_pp[j] += h[j]
            x_pm = full_theta.copy()
            x_pm[i] += h[i]
            x_pm[j] -= h[j]
            x_mp = full_theta.copy()
            x_mp[i] -= h[i]
            x_mp[j] += h[j]
            x_mm = full_theta.copy()
            x_mm[i] -= h[i]
            x_mm[j] -= h[j]
            H[i, j] = (
                deviance(x_pp) - deviance(x_pm) - deviance(x_mp) + deviance(x_mm)
            ) / (4.0 * h[i] * h[j])
            if i != j:
                H[j, i] = H[i, j]

    # Cov(theta) = 2 * inv(H) because REML "deviance" = -2 * loglik
    try:
        cov_full = 2.0 * np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return np.full(L.shape[0], np.inf)

    # Gradient of c'V_b c w.r.t. (theta_lmer, theta_vc, sigma_sq)
    def cVc_at(vec: np.ndarray) -> np.ndarray:
        tr, tv, sig = _unpack(vec)
        Vb = _vbeta_at_lmer(
            model_handle, tr, sig, k_re, X, Z, groups, group_ids,
            theta_vc=tv, vc_mats=vc_mats,
        )
        return np.einsum("ij,jk,ik->i", L, Vb, L)

    cVc = cVc_at(full_theta)
    grad = np.zeros((L.shape[0], n_theta))
    for i in range(n_theta):
        vec_p = full_theta.copy()
        vec_p[i] += h[i]
        vec_m = full_theta.copy()
        vec_m[i] -= h[i]
        grad[:, i] = (cVc_at(vec_p) - cVc_at(vec_m)) / (2.0 * h[i])

    var_cVc = np.einsum("ij,jk,ik->i", grad, cov_full, grad)
    df_out = np.full_like(cVc, np.inf, dtype=float)
    np.divide(2.0 * cVc**2, var_cVc, out=df_out, where=var_cVc > 0)
    return df_out


def satterthwaite_df(
    info: ModelInfo, L: np.ndarray, cache: _SattCache | None = None
) -> np.ndarray:
    """Compute Satterthwaite df for each row of L.

    Uses the lmer/lmerTest parametrisation: theta_lmer = vech(Lambda)
    where Lambda is the Cholesky factor of cov_re/sigma_e^2, plus
    sigma_e^2 as an explicit parameter. The Cov(theta) used in the
    delta method comes from inverting the Hessian of the REML deviance
    in these parameters (matching `lmerTest::lmer` to 4 decimals on a
    reference fit).

    review replaced the previous statsmodels-cov_params
    path, which used a different parametrisation (cov_re_unscaled with
    sigma profiled out) and underestimated df by ~15 % on the
    `lme4_data.csv` reference fit.

    Parameters
    ----------
    info
        :class:`~pymmeans.utils.ModelInfo` for a statsmodels MixedLM fit.
        For non-mixed models (OLS, GLM) this returns ``df_resid`` (OLS)
        or ``inf`` (GLM) without inspecting ``info`` further.
    L
        Contrast matrix of shape ``(n_contrasts, n_params)``. One df is
        returned per row.

    Returns
    -------
    ndarray of shape ``(n_contrasts,)``
        Satterthwaite df per contrast row; ``inf`` for contrasts whose
        ``c' V_b c`` derivative w.r.t. theta is numerically zero (i.e.
        unaffected by variance-component uncertainty).
    """
    if not info.is_mixed:
        return np.full(
            L.shape[0],
            np.inf if info.family is not None else info.df_resid,
        )
    # post-pickle the raw result is gone, but the cache (built
    # at first apply_satterthwaite call) holds everything we need. Use
    # it before consulting raw_result.
    if cache is not None:
        return _satterthwaite_df_lmer(info, L, cache=cache)
    result = info.raw_result
    if result is None or not hasattr(result, "cov_re"):
        warnings.warn(
            "Satterthwaite df requires a statsmodels MixedLM result; "
            "falling back to df = inf.",
            stacklevel=2,
        )
        return np.full(L.shape[0], np.inf)
    # ``vc_formula=`` variance components (``model.k_vc > 0``) are now
    # supported: they enter the augmented lmer θ as scalar relative-
    # covariance factors ``sqrt(vcomp_v / sigma_sq)`` and add their
    # design blocks to each per-group ``V_i``. See
    # :func:`_lmer_vc_theta_hat` / :func:`_vc_cov_block`.
    return _satterthwaite_df_lmer(info, L)


# Boundary tolerance on the Cholesky factor (= sqrt(cov_re / sigma_e^2)).
# A Lambda diagonal at 1e-3 corresponds to a variance ratio of 1e-6, which
# is essentially boundary. We use 1e-3 rather than 1e-6 because the
# squared relationship makes the variance ratio at 1e-6 numerically
# indistinguishable from zero for the finite-difference Hessian.
_KR_BOUNDARY_TOL = 1e-3


class BoundaryFitError(RuntimeError):
    """Raised when KR is requested on a MixedLM fit at the variance-component
    boundary (e.g. ``sigma_u^2 -> 0``), where finite-difference KR is
    numerically meaningless."""


# ``_xtvinv_x_lmer`` was also dead — superseded
# by the inline build inside ``_compute_pbkrtest_aux_core`` (which
# computes V_β = (X' Σ⁻¹ X)⁻¹ directly).


def kenward_roger_vcov(
    info: ModelInfo,
    cache: _SattCache | None = None,
    return_internals: bool = False,
) -> np.ndarray | KRInternals:
    """Compute the Kenward-Roger inflated fixed-effects vcov.

    Uses the lmer parametrisation: ``theta = (vech(Lambda), sigma_e^2)``
    where Lambda is the Cholesky factor of cov_re/sigma_e^2.

    rewrite: implements pbkrtest's exact ``vcovAdj_internal``
    algorithm (Halekoh & Højsgaard 2014). The pre-
    implementation used a Kackar-Harville approximation that matched
    pbkrtest to ``atol<1e-4`` on non-intercept SEs but had a ~2 %
    sign-flipped deflation on the intercept-variance entry (the
    finding). The implementation matches
    pbkrtest to ``atol ~5e-7`` on the full vcov diagonal — closing
    .

    #1: when ``return_internals=True`` the function returns
    a :class:`KRInternals` carrying ``V_KR``, ``W = 2·inv(IE2)`` (the
    asymptotic ``Cov(theta_hat)`` under REML), and ``V_beta``.
    ``apply_kenward_roger`` consumes that bundle so it can reuse the
    canonical pbkrtest weight matrix as ``cov_theta`` instead of
    rebuilding it via observed-Hessian finite differencing of
    ``_reml_deviance`` — saves ``n_theta·(n_theta+1)/2 · 4`` REML
    deviance evaluations per call and produces df values that match
    pbkrtest's ``ddf_Lb`` directly rather than via the observed-vs-
    expected-information approximation.

    The pbkrtest formula:

        V_KR = V_beta + 2 V_beta · UU · V_beta

    where:

        V_r = dSigma/dtheta_r (finite-diff on V_g per group)
        PP[r] = -X' Sigma^-1 V_r Sigma^-1 X
        QQ[r,s] = X' Sigma^-1 V_r Sigma^-1 V_s Sigma^-1 X
        Ktr[r,s] = tr(V_r Sigma^-1 V_s Sigma^-1)
        IE2[r,s] = Ktr[r,s] - 2 tr(V_beta · QQ[r,s])
                      + tr(PP_r V_beta PP_s V_beta)
        W = 2 · inv(IE2) (asymptotic Cov of theta-hat)
        UU = sum_{r,s} W[r,s] · (QQ[r,s] - PP[r] V_beta PP[s])

    Derivatives of V are computed by central finite differences (
    design choice: keep finite-diff rather than switch to analytic to
    minimise risk during the rewrite). Validation: matches pbkrtest at
    ``atol=4.76e-7`` on ``tests/r_reference/kr_reference.csv``; matches
    pbkrtest at floating-point precision on sleepstudy.

    Parameters
    ----------
    info
        :class:`pymmeans.utils.ModelInfo` for a fitted MixedLM.
    cache
        Optional pickle-safe REML state (built by ``_build_satt_cache``).
        When supplied, ``info.raw_result`` is not touched — required for
        post-pickle round-trips.
    return_internals
        When True, return a :class:`KRInternals` (default False returns
        just the ``V_KR`` matrix for API compatibility).

    Returns
    -------
    np.ndarray or KRInternals
        ``V_KR`` matrix, or the full ``KRInternals`` bundle if
        ``return_internals=True``.

    Raises
    ------
    BoundaryFitError
        When the RE variance is at the boundary (``Lambda``'s diagonal
        is too close to zero for finite-difference second derivatives to
        be meaningful).
    """
    # KR can read its inputs from the pickle-safe _SattCache
    # populated by a prior apply_satterthwaite / apply_kenward_roger
    # call. If no cache is supplied, fall back to the live raw_result.
    # ``y`` and ``model_handle`` are no longer needed
    # (the rewrite computes V_KR without calling ``_reml_deviance``).
    if cache is not None:
        theta_lmer_hat = cache.theta_hat
        k_re = cache.k_re
        sigma_sq_hat = cache.sigma_sq_hat
        X = cache.X
        Z = cache.Z
        groups = cache.groups
        group_ids = cache.group_ids
        theta_vc_hat = cache.theta_vc_hat
        k_vc = cache.k_vc
        vc_mats = cache.vc_mats
    else:
        if not info.is_mixed or info.raw_result is None or not hasattr(
            info.raw_result, "cov_re"
        ):
            raise ValueError(
                "Kenward-Roger requires a statsmodels MixedLM result on "
                "ModelInfo, or a pre-built _SattCache."
            )
        result = info.raw_result
        theta_lmer_hat, k_re = _lmer_theta_hat(result)
        theta_vc_hat, k_vc, vc_mats = _lmer_vc_theta_hat(result)
        sigma_sq_hat = float(result.scale)
        X = np.asarray(result.model.exog, dtype=float)
        Z = np.asarray(result.model.exog_re, dtype=float)
        groups = np.asarray(result.model.groups)
        group_ids = np.unique(groups)

    Lambda = _lmer_theta_to_lambda(theta_lmer_hat, k_re)
    if k_re and np.any(np.diag(Lambda) <= _KR_BOUNDARY_TOL):
        raise BoundaryFitError(
            "Kenward-Roger cannot be computed at boundary variance "
            f"components (Lambda diagonal = {np.diag(Lambda).tolist()}). "
            "Use apply_satterthwaite() instead, or refit with a "
            "non-boundary model."
        )
    # Variance components also have a boundary (vcomp_v -> 0 ⇒
    # theta_vc_v -> 0); the finite-difference KR second derivatives are
    # meaningless there.
    if k_vc and np.any(theta_vc_hat <= _KR_BOUNDARY_TOL):
        raise BoundaryFitError(
            "Kenward-Roger cannot be computed at boundary variance "
            f"components (sqrt(vcomp/sigma^2) = {theta_vc_hat.tolist()}). "
            "Use apply_satterthwaite() instead, or refit with a "
            "non-boundary model."
        )

    # rewrite to match pbkrtest's exact ``vcovAdj_internal``
    # algorithm. The pre-implementation used a Kackar-Harville
    # approximation that worked on the OBSERVED REML Hessian and a
    # ``derivatives of A = X'V⁻¹X`` formulation; it agreed with pbkrtest to
    # ``atol<1e-4`` on non-intercept SEs but had a sign-flipped intercept-
    # variance gap of ~2 %. The replacement matches pbkrtest to
    # ``atol≈5e-7`` on the full vcov diagonal.
    #
    # The pbkrtest formula (Halekoh & Højsgaard 2014; implemented by
    # ``pbkrtest:::vcovAdj_internal``):
    #
    # V_KR = V_β + 2 · V_β · UU · V_β
    #
    # where:
    # V_r = dΣ/dθ_r (derivative of marginal cov; finite-diff)
    # PP[r] = -X' Σ⁻¹ V_r Σ⁻¹ X (note negative sign per pbkrtest convention)
    # QQ[r,s] = X' Σ⁻¹ V_r Σ⁻¹ V_s Σ⁻¹ X (the "R^KR" third-order term)
    # Ktr[r,s] = tr(V_r Σ⁻¹ V_s Σ⁻¹)
    # IE2[r,s] = Ktr[r,s] - 2 tr(V_β · QQ[r,s]) + tr(PP_r V_β PP_s V_β)
    # (the expected REML information matrix of θ)
    # W = 2 · inv(IE2) (asymptotic Cov of θ̂)
    # UU = Σ_{r,s} W[r,s] · (QQ[r,s] - PP[r] V_β PP[s])
    #
    # We work group-wise (Σ is block-diagonal across groups in MixedLM)
    # to avoid materialising the full n×n matrix. This is asymptotically
    # the same cost as the pre-implementation but produces the
    # full-precision pbkrtest answer.

    # augmented θ ordering [theta_re, theta_vc, sigma_sq].
    n_re2 = len(theta_lmer_hat)

    def build_V_groups(
        theta_lmer: np.ndarray, theta_vc: np.ndarray, sigma_sq: float
    ) -> list[np.ndarray]:
        """Build per-group marginal covariance matrices V_g.

        Adds the ``vc_formula=`` variance-component blocks
        ``sum_v vcomp_v · Zvc Zvc'`` when present (``vc_mats`` non-empty).
        """
        Lam = _lmer_theta_to_lambda(theta_lmer, k_re)
        G = Lam @ Lam.T * sigma_sq
        Vs: list[np.ndarray] = []
        for g_id in group_ids:
            m = groups == g_id
            Z_i = Z[m]
            n_i = int(m.sum())
            V_i = Z_i @ G @ Z_i.T + sigma_sq * np.eye(n_i)
            if vc_mats:
                V_i = V_i + _vc_cov_block(
                    vc_mats, g_id, theta_vc, sigma_sq, n_i
                )
            Vs.append(V_i)
        return Vs

    full_theta = np.concatenate(
        [theta_lmer_hat, theta_vc_hat, [sigma_sq_hat]]
    )
    n_theta = len(full_theta)
    h = np.maximum(np.abs(full_theta), 1e-6) * 1e-4

    def _split(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        return vec[:n_re2], vec[n_re2:n_re2 + k_vc], float(vec[-1])

    V_groups = build_V_groups(theta_lmer_hat, theta_vc_hat, sigma_sq_hat)
    SigmaInv_groups = [np.linalg.inv(V_g) for V_g in V_groups]

    # Pre-extract per-group X slices (used everywhere downstream).
    X_groups: list[np.ndarray] = []
    for g_id in group_ids:
        m = groups == g_id
        X_groups.append(X[m])

    n_groups = len(group_ids)
    p = X.shape[1]

    # V_β = (X' Σ⁻¹ X)⁻¹ (block-diagonal Σ⁻¹ → sum over groups)
    A_total = np.zeros((p, p))
    for gi in range(n_groups):
        A_total += X_groups[gi].T @ SigmaInv_groups[gi] @ X_groups[gi]
    V_beta = np.linalg.inv(A_total)

    # V_r per parameter via central finite-diff on V_g (the variance-component
    # derivatives — kept finite-diff per design constraint).
    V_r_groups_list: list[list[np.ndarray]] = []
    for r in range(n_theta):
        vec_p = full_theta.copy(); vec_p[r] += h[r]
        vec_m = full_theta.copy(); vec_m[r] -= h[r]
        tr_p, tv_p, ss_p = _split(vec_p)
        tr_m, tv_m, ss_m = _split(vec_m)
        Vp = build_V_groups(tr_p, tv_p, ss_p)
        Vm = build_V_groups(tr_m, tv_m, ss_m)
        V_r_groups_list.append(
            [(Vp[g] - Vm[g]) / (2.0 * h[r]) for g in range(n_groups)]
        )

    # Per-group helpers:
    # TT_g = Σ_g⁻¹ X_g (n_g × p)
    # HH_r_g = V_r_g · Σ_g⁻¹ (n_g × n_g, used for Ktrace)
    # OO_r_g = HH_r_g · X_g (n_g × p, used for PP and QQ)
    #
    # #5: where groups have varying sizes we cannot stack into
    # a single 3D tensor, but we *can* eliminate the r,s double-loop over
    # `n_theta**2` python-level matmul accumulations. For each group we
    # build the stacked tensors ``OO_g[r]`` (n_theta × n_g × p) and
    # ``HH_g[r]`` (n_theta × n_g × n_g), then use ``np.einsum`` to compute
    # the per-group contribution to QQ and Ktrace across all (r,s) in one
    # call.
    TT_groups = [SigmaInv_groups[g] @ X_groups[g] for g in range(n_groups)]
    HH_groups_3d: list[np.ndarray] = [] # per group: (n_theta, n_g, n_g)
    OO_groups_3d: list[np.ndarray] = [] # per group: (n_theta, n_g, p)
    for g in range(n_groups):
        HH_g = np.stack(
            [V_r_groups_list[r][g] @ SigmaInv_groups[g] for r in range(n_theta)]
        )
        OO_g = HH_g @ X_groups[g]
        HH_groups_3d.append(HH_g)
        OO_groups_3d.append(OO_g)

    # PP[r] = -X' Σ⁻¹ V_r Σ⁻¹ X = -Σ_g (OO_r_g)ᵀ TT_g (p × p)
    # Vectorised across r: PP[r,:,:] = -Σ_g (OO_g[r]).T @ TT_g
    PP_arr = np.zeros((n_theta, p, p))
    for g in range(n_groups):
        # einsum over (r, n_g_idx, p_idx_left); contract n_g_idx with TT_g
        PP_arr -= np.einsum("rij,ik->rjk", OO_groups_3d[g], TT_groups[g])

    # QQ[r,s] = X' Σ⁻¹ V_r Σ⁻¹ V_s Σ⁻¹ X = Σ_g (OO_r_g)ᵀ Σ_g⁻¹ (OO_s_g)
    # Vectorised across (r, s): QQ[r,s,:,:] = Σ_g (OO_g[r]).T @ Σ_g⁻¹ @ OO_g[s]
    QQ_arr = np.zeros((n_theta, n_theta, p, p))
    for g in range(n_groups):
        # M_g[s, :, :] = Σ_g⁻¹ @ OO_g[s] (n_theta, n_g, p)
        M_g = SigmaInv_groups[g] @ OO_groups_3d[g]
        # QQ[r, s, j, k] = sum_i OO_g[r, i, j] * M_g[s, i, k]
        QQ_arr += np.einsum("rij,sik->rsjk", OO_groups_3d[g], M_g)

    # Ktrace[r,s] = tr(V_r Σ⁻¹ V_s Σ⁻¹) = Σ_g tr(HH_g[r] · HH_g[s])
    # = Σ_g sum_{i,j} HH_g[r, j, i] · HH_g[s, i, j]
    Ktrace = np.zeros((n_theta, n_theta))
    for g in range(n_groups):
        # einsum over (r, s, i, j); contract i and j as cross-indices
        Ktrace += np.einsum("rji,sij->rs", HH_groups_3d[g], HH_groups_3d[g])

    # IE2 = expected REML information of θ.
    # IE2[r,s] = Ktrace[r,s] - 2 sum(V_β * QQ[r,s]) + sum((V_β PP[r]) * (PP[s] V_β))
    # Vectorised contractions over the matrix indices, leaving the (r,s) grid.
    cross = np.einsum("ij,rsij->rs", V_beta, QQ_arr)
    Phi_P = V_beta @ PP_arr # (n_theta, p, p)
    P_Phi = PP_arr @ V_beta # (n_theta, p, p)
    outer = np.einsum("rij,sij->rs", Phi_P, P_Phi)
    IE2 = Ktrace - 2.0 * cross + outer

    W = 2.0 * np.linalg.inv(IE2)

    # UU = Σ_{r,s} W[r,s] · (QQ[r,s] - PP[r] V_β PP[s])
    # Vectorised: PP_VbPP[r, s] = PP[r] @ (V_β @ PP[s]) = PP[r] @ Phi_P[s]
    PP_VbPP = np.einsum("rij,sjk->rsik", PP_arr, Phi_P)
    UU = np.einsum("rs,rsjk->jk", W, QQ_arr - PP_VbPP)

    V_KR = V_beta + 2.0 * V_beta @ UU @ V_beta
    if return_internals:
        # also expose P_list = ∂V_β/∂θ_r for each r.
        # By the chain rule, since PP_arr[r] = ∂(inv V_β)/∂θ_r and
        # ∂A/∂θ = -A · (∂(inv A)/∂θ) · A for any invertible A:
        # ∂V_β/∂θ_r = -V_β · PP_arr[r] · V_β
        # This is what pbkrtest's ``.KR_adjust`` calls ``P[[r]]``.
        # We compute it here at no marginal cost (PP_arr is already
        # built en route to V_KR) and bundle it into KRInternals so
        # ``krmodcomp`` doesn't have to rebuild the finite-diff grid.
        P_list = tuple(-V_beta @ PP_arr[r] @ V_beta for r in range(n_theta))
        return KRInternals(V_KR=V_KR, W=W, V_beta=V_beta, P_list=P_list)
    return V_KR


def _refuse_other_correction(
    emm_or_contrast: Any, op: str, *, allowed_existing: str
) -> None:
    """Raise if a Satt/KR correction would silently undo or stack on top
    of a different small-sample correction already on the input.

    The two corrections are mutually exclusive. Each rebuilds ``SE`` and
    ``df`` from ``info.vcov`` (the *original*, uncorrected ``V_beta``)
    plus its own ``V_corrected`` recipe — ``V_KR`` for Kenward-Roger,
    ``V_beta`` itself for Satterthwaite. Calling
    ``apply_satterthwaite(apply_kenward_roger(em))`` therefore silently
    discards the KR vcov inflation (SE shrinks back to the uncorrected
    ``sqrt(L V_beta L')``) while keeping the Satterthwaite df. The
    reverse stacking is wasted work but not actively wrong. Either way,
    refuse so the user picks one path explicitly.

    The single permitted no-op is ``op == allowed_existing`` (e.g.
    ``apply_kenward_roger`` on a KR-corrected result), which is handled
    by the existing per-function idempotency guard, NOT by this helper.
    """
    existing = getattr(emm_or_contrast, "df_method", "default")
    if existing in ("satterthwaite", "kenward_roger") and existing != allowed_existing:
        raise ValueError(
            f"{op} cannot be applied to a result that already carries the "
            f"{existing!r} correction. The two corrections rebuild ``SE`` "
            "and ``df`` from the original ``V_beta`` and are mutually "
            "exclusive — applying one on top of the other either silently "
            "discards the earlier vcov inflation (Satt-after-KR) or "
            "wastes work (KR-after-Satt). Pick one path: call "
            "`apply_satterthwaite` OR `apply_kenward_roger` on a fresh "
            "EMMResult / ContrastResult, not both."
        )


def _refuse_response_scale(emm_or_contrast: Any, op: str) -> None:
    """Raise if ``apply_satterthwaite`` / ``apply_kenward_roger`` is asked
    to operate on a response-scale result.

    The correction recomputes ``SE = sqrt(L V_corrected L')``, which is on
    the **link** scale. Overwriting a response-scale SE with that, and
    rebuilding CIs as ``emmean_response ± crit · SE_link``, mixes scales
    and gives nonsense. Workflow: apply the correction on the link-scale
    EMM first, then call ``regrid_response``.
    """
    if getattr(emm_or_contrast, "type", "link") == "response":
        raise ValueError(
            f"{op} cannot be applied to a response-scale result because "
            "the recomputed SE is on the link scale. Apply the correction "
            "to the link-scale EMM first (before `regrid_response`), then "
            "regrid the corrected result. E.g. "
            "`regrid_response(apply_satterthwaite(emmeans(fit, ...)))`."
        )


def _refuse_posterior(emm_or_contrast: Any, op: str) -> None:
    """Raise if a Satterthwaite / KR correction is asked to overwrite the
    CIs on a posterior-derived result.

    caught this: ``PosteriorInfo.model_info`` is a
    frequentist-shaped ``ModelInfo`` for compatibility, so
    ``apply_satterthwaite`` couldn't tell the EMMResult's CIs came from
    posterior percentiles and silently overwrote them with Wald/t
    intervals. Refuse instead so the user knows they need to either
    stay on the posterior path or switch to a Wald-style frequentist
    fit before applying the correction.
    """
    if getattr(emm_or_contrast, "inference_kind", "wald") == "posterior":
        raise ValueError(
            f"{op} is not defined for posterior-derived results. The "
            "Satterthwaite/KR correction overwrites the credible-interval "
            "endpoints with t-based Wald intervals, which is meaningless "
            "for a Bayesian posterior. To get a t-based CI on the "
            "posterior mean, build a fresh frequentist ModelInfo from the "
            "underlying fit; for posterior credible intervals, the "
            "result you have is already correct."
        )


def _refuse_bootstrap(emm_or_contrast: Any, op: str) -> None:
    """Raise if a Satterthwaite / KR correction is asked to overwrite the
    CIs on a bootstrap-derived result.

    — symmetric to ``_refuse_posterior`` and to the
    ``bootstrap_ci`` Satt/KR refusal. previously
    ``apply_satterthwaite(bootstrap_ci(em))`` silently overwrote the
    bootstrap percentile CIs with t-based Wald intervals while the
    bootstrap-corrupted point estimate stayed in the frame — the
    same silent inference-paradigm mixing as the posterior case, but
    in the reverse direction (bootstrap → Satt instead of Satt →
    bootstrap, the latter closed by in ``bootstrap_ci``).
    Detected via ``df_method == "bootstrap"``, stamped by
    ``bootstrap_ci`` on output.
    """
    if getattr(emm_or_contrast, "df_method", "default") == "bootstrap":
        raise ValueError(
            f"{op} is not defined for a result whose CIs came from "
            "``bootstrap_ci(...)``. The Satterthwaite / Kenward-Roger "
            "correction overwrites ``lower_cl`` / ``upper_cl`` with "
            "t-based Wald intervals; doing that to a bootstrap result "
            "silently mixes two inference paradigms (bootstrap percentile "
            "for the point uncertainty, Wald t for the interval shape). "
            "Pick one inference path: either skip ``bootstrap_ci`` and "
            "call Satt/KR on the raw EMM, OR keep the bootstrap CIs and "
            "do not apply a small-sample correction afterward."
        )


def _apply_correction(
    emm_or_contrast: Any,
    V_corrected: np.ndarray,
    df: np.ndarray,
    method: str = "satterthwaite",
    satt_cache: _SattCache | None = None,
) -> Any:
    from scipy import stats

    frame = emm_or_contrast.frame.copy()
    L = emm_or_contrast.linfct
    new_se = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", L, V_corrected, L), 0.0, None))
    frame["SE"] = new_se
    frame["df"] = df

    # Centralised value-column detection so emtrends (<var>.trend) and
    # any future result shapes all dispatch uniformly. Self-caught
    # the previous inline string-matching missing the .trend case.
    from pymmeans.utils import detect_value_column

    kind_info = detect_value_column(frame)
    point_col = None
    if kind_info is not None and kind_info[0] in ("emm", "trend"):
        point_col = kind_info[1]
    if point_col is not None and "lower_cl" in frame.columns:
        level = emm_or_contrast.level
        crit = stats.t.ppf(1.0 - (1.0 - level) / 2.0, df)
        point = frame[point_col].to_numpy()
        frame["lower_cl"] = point - crit * new_se
        frame["upper_cl"] = point + crit * new_se
    if "estimate" in frame.columns and "t_ratio" in frame.columns:
        est = frame["estimate"].to_numpy()
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(new_se > 0, est / new_se, np.nan)
        frame["t_ratio"] = t
        frame["p_value"] = 2.0 * stats.t.sf(np.abs(t), df)

    cls = type(emm_or_contrast)
    fields = {
        f: getattr(emm_or_contrast, f) for f in emm_or_contrast.__dataclass_fields__
    }
    fields["frame"] = frame
    # stamp `df_method` so downstream pairs() / contrast()
    # knows which correction was applied and can propagate it to the
    # contrast-level L matrix instead of silently demoting back to
    # z-inference.
    if "df_method" in fields:
        fields["df_method"] = method
    # attach the pickle-safe REML state cache so the
    # correction survives pickle and can re-apply on the contrast L_c.
    # Falls back to whatever the source had (eg if no cache was
    # available at this layer).
    if "_satt_cache" in fields and satt_cache is not None:
        fields["_satt_cache"] = satt_cache
    return cls(**fields)


def apply_satterthwaite(emm_or_contrast: Any) -> Any:
    """Return a copy of an EMMResult or ContrastResult with Satterthwaite df.

    Re-derives the CI bounds (for EMM) or p-values (for contrasts) using
    the Satterthwaite df at the original V_beta. Use ``apply_kenward_roger``
    for the more accurate small-sample correction.

    Parameters
    ----------
    emm_or_contrast
        :class:`pymmeans.EMMResult` or :class:`pymmeans.ContrastResult`
        from a :class:`statsmodels.regression.mixed_linear_model.MixedLM`
        fit. Refuses non-mixed models, response-scale inputs,
        posterior-derived results, and bootstrap-derived results.

    Returns
    -------
    EMMResult or ContrastResult
        New result with updated ``SE``, ``df``, ``lower_cl`` / ``upper_cl``
        (for EMMs) or ``t_ratio`` / ``p_value`` (for contrasts), and
        ``df_method="satterthwaite"`` stamped for downstream propagation.

    Notes
    -----
    Per-row Satterthwaite df is a function of the linear-combination
    matrix ``L`` for *that row*, not a global property of the fit. As
    a result, ``df`` at an EMM cell (where ``L`` picks a single grid
    point) and ``df`` on a pair contrast (where ``L`` is the
    *difference* of two grid points) can differ by an order of
    magnitude on the same model — both are correct. On a random-
    intercept ``MixedLM`` of `sleepstudy`, the EMM at ``Days=9`` has
    Satterthwaite ``df ≈ 23.4`` while the ``Days=9 − Days=0`` contrast
    has ``df ≈ 161``, matching ``lmerTest`` to four decimals. Always
    use the df computed on the *quantity you intend to report*; copying
    df between EMMs and contrasts will produce wrong CIs.

    References
    ----------
    - Satterthwaite, F. E. (1946). An approximate distribution of
      estimates of variance components. *Biometrics Bulletin*,
      2(6), 110-114.
    - Kuznetsova, A., Brockhoff, P. B., & Christensen, R. H. B. (2017).
      lmerTest Package: Tests in Linear Mixed Effects Models. *Journal
      of Statistical Software*, 82(13). doi:10.18637/jss.v082.i13

    Examples
    --------
    >>> import statsmodels.regression.mixed_linear_model as mlm # doctest: +SKIP
    >>> from statsmodels.datasets import get_rdataset # doctest: +SKIP
    >>> from pymmeans import emmeans, apply_satterthwaite # doctest: +SKIP
    >>> sleep = get_rdataset("sleepstudy", "lme4").data # doctest: +SKIP
    >>> fit = mlm.MixedLM.from_formula( # doctest: +SKIP
    ... "Reaction ~ Days", groups="Subject", data=sleep,
    ... ).fit() # doctest: +SKIP
    >>> em = emmeans(fit, "Days", at={"Days": [0, 5, 9]}) # doctest: +SKIP
    >>> em_satt = apply_satterthwaite(em) # doctest: +SKIP
    >>> em_satt.df_method # doctest: +SKIP
    'satterthwaite'
    """
    _refuse_response_scale(emm_or_contrast, "apply_satterthwaite()")
    _refuse_posterior(emm_or_contrast, "apply_satterthwaite()")
    _refuse_bootstrap(emm_or_contrast, "apply_satterthwaite()")
    _refuse_other_correction(
        emm_or_contrast, "apply_satterthwaite()", allowed_existing="satterthwaite"
    )
    info = emm_or_contrast.model_info
    # Satterthwaite df is a
    # variance-component degrees-of-freedom correction defined for
    # linear mixed models. Applying it to OLS / GLM / GEE silently
    # stamped the result with `df_method="satterthwaite"` but didn't
    # actually compute anything useful (the df came out as inf for
    # GLM/GEE since `info.is_mixed` was False, gating the cache
    # build). Refuse cleanly so users don't think they applied a
    # correction that didn't fire.
    if not info.is_mixed:
        raise ValueError(
            "apply_satterthwaite() is only defined for MixedLM-style "
            "mixed models with variance components. OLS already "
            "carries its native df_resid; GLM / GEE / Cox carry "
            "asymptotic / robust inference (df=inf). For those "
            "models, the EMM result already has the correct df — no "
            "Satterthwaite step is needed or meaningful. If you "
            "intended Kenward-Roger for a MixedLM, call "
            "`apply_kenward_roger()` instead."
        )
    # build / reuse the pickle-safe REML cache so that
    # post-pickle re-application of the correction (e.g. via
    # `pairs(unpickled_satt_em)`) works without raw_result.
    existing_cache = getattr(emm_or_contrast, "_satt_cache", None)
    cache = existing_cache
    if (
        cache is None
        and info.is_mixed
        and info.raw_result is not None
        and hasattr(info.raw_result, "cov_re")
    ):
        # Only build cache for MixedLM with random effects; OLS / GLM
        # don't need it (df is df_resid / inf). vc_formula= MixedLMs
        # (k_vc > 0) are now supported and DO need the cache (it carries
        # the VC design blocks + relative-covariance factors).
        cache = _build_satt_cache(info.raw_result)
    df = satterthwaite_df(info, emm_or_contrast.linfct, cache=cache)
    # Use the original V_beta (no KR inflation here)
    V_beta = info.vcov
    return _apply_correction(
        emm_or_contrast,
        V_beta,
        df,
        method="satterthwaite",
        satt_cache=cache,
    )


def apply_kenward_roger(emm_or_contrast: Any) -> Any:
    """Return a copy with Kenward-Roger inflated SE + KR df.

    The KR adjustment inflates ``V_beta`` to account for ignoring
    uncertainty in the variance components; the Satterthwaite df is then
    computed using the inflated ``V_beta``.

    .. note::
       **Matches pbkrtest to ``atol≈5e-7``.** The earlier intercept-
       variance gap was closed by rewriting ``kenward_roger_vcov`` to
       implement pbkrtest's exact ``vcovAdj_internal`` algorithm. All
       entries of the KR-adjusted vcov now agree with
       ``pbkrtest::vcovAdj`` to ``atol=4.76e-7`` on the canonical lme4
       reference fit and to floating-point precision on sleepstudy.

    Parameters
    ----------
    emm_or_contrast
        :class:`pymmeans.EMMResult` or :class:`pymmeans.ContrastResult`
        from a :class:`statsmodels.regression.mixed_linear_model.MixedLM`
        fit. Refuses non-mixed models, response-scale inputs,
        posterior-derived results, and bootstrap-derived results.

    Returns
    -------
    EMMResult or ContrastResult
        New result with KR-inflated ``SE``, Satterthwaite df at the
        inflated vcov, updated ``lower_cl`` / ``upper_cl`` (EMMs) or
        ``t_ratio`` / ``p_value`` (contrasts), and
        ``df_method="kenward_roger"`` stamped.

    References
    ----------
    - Kenward, M. G., & Roger, J. H. (1997). Small Sample Inference
      for Fixed Effects from Restricted Maximum Likelihood.
      *Biometrics*, 53(3), 983-997.
    - Kackar, R. N., & Harville, D. A. (1984). Approximations for
      standard errors of estimators of fixed and random effects in
      mixed linear models. *JASA*, 79(388), 853-862.
    - Halekoh, U., & Højsgaard, S. (2014). A Kenward-Roger
      Approximation and Parametric Bootstrap Methods for Tests in
      Linear Mixed Models — The R Package pbkrtest. *Journal of
      Statistical Software*, 59(9). doi:10.18637/jss.v059.i09

    Examples
    --------
    >>> import statsmodels.regression.mixed_linear_model as mlm # doctest: +SKIP
    >>> from statsmodels.datasets import get_rdataset # doctest: +SKIP
    >>> from pymmeans import emmeans, apply_kenward_roger # doctest: +SKIP
    >>> sleep = get_rdataset("sleepstudy", "lme4").data # doctest: +SKIP
    >>> fit = mlm.MixedLM.from_formula( # doctest: +SKIP
    ... "Reaction ~ Days", groups="Subject", data=sleep,
    ... ).fit() # doctest: +SKIP
    >>> em_kr = apply_kenward_roger(emmeans(fit, "Days")) # doctest: +SKIP
    >>> em_kr.df_method # doctest: +SKIP
    'kenward_roger'
    """
    _refuse_response_scale(emm_or_contrast, "apply_kenward_roger()")
    _refuse_posterior(emm_or_contrast, "apply_kenward_roger()")
    _refuse_bootstrap(emm_or_contrast, "apply_kenward_roger()")
    _refuse_other_correction(
        emm_or_contrast, "apply_kenward_roger()", allowed_existing="kenward_roger"
    )

    # Idempotency: if the input already carries a K-R correction,
    # re-applying would recompute V_KR from the already-inflated vcov
    # and produce a slightly different df (~0.05 drift on canonical
    # fits due to finite-difference noise on the already-corrected
    # Hessian). Return the input unchanged. Mirrors `_refuse_*`
    # discipline elsewhere in this module.
    if getattr(emm_or_contrast, "df_method", "default") == "kenward_roger":
        return emm_or_contrast

    info = emm_or_contrast.model_info
    # Kenward-Roger is a variance-component degrees-of-freedom and
    # vcov correction defined for linear mixed models. Applying it
    # to OLS / GLM / GEE / Cox previously crashed deep inside the
    # KR derivative chain with ``AttributeError: 'OLSResults' object
    # has no attribute 'cov_re'`` — actionable for someone who reads
    # the traceback, hostile for a user who just tried "the
    # small-sample correction" without realising it was MixedLM-only.
    # Mirror ``apply_satterthwaite``'s clear refusal so the surface
    # error is symmetric.
    if not info.is_mixed:
        raise ValueError(
            "apply_kenward_roger() is only defined for MixedLM-style "
            "mixed models with variance components. OLS already "
            "carries its native df_resid; GLM / GEE / Cox carry "
            "asymptotic / robust inference (df=inf). For those "
            "models, the EMM result already has the correct df — no "
            "Kenward-Roger step is needed or meaningful. If you "
            "intended Satterthwaite for a MixedLM, call "
            "`apply_satterthwaite()` instead."
        )

    # ``vc_formula=`` variance components are now supported by the
    # Kenward-Roger path: they enter the augmented lmer θ as scalar
    # relative-covariance factors and add their design blocks to each
    # per-group ``V_g`` in :func:`kenward_roger_vcov`'s
    # ``build_V_groups``.

    # reuse / populate the pickle-safe REML cache so KR
    # round-trips through pickle just like Satt does.
    existing_cache = getattr(emm_or_contrast, "_satt_cache", None)
    cache = existing_cache
    if (
        cache is None
        and info.is_mixed
        and info.raw_result is not None
        and hasattr(info.raw_result, "cov_re")
    ):
        cache = _build_satt_cache(info.raw_result)

    # the K-R 1997 denominator df. Pre-pymmeans
    # used a Satterthwaite-style delta-method df:
    # df = 2 (c V_KR c')² / (∂(c V_β c')/∂θ)ᵀ W (∂(c V_β c')/∂θ)
    # which drifted from ``pbkrtest::ddf_Lb`` by ~1 % relative.
    # ports pbkrtest's exact ``ddf_Lb`` formula via the
    # ``_kr1997_df_per_row`` helper, closing the gap to floating-
    # point parity.
    #
    # Implementation: build pbkrtest-parameterisation aux quantities
    # (``V_β``, ``W``, ``P_list``) once, then call the per-row K-R
    # 1997 formula for each row of L. V_KR comes directly from
    # ``kenward_roger_vcov`` (parameterisation-invariant).
    from pymmeans.pbktest import (
        _compute_pbkrtest_aux_core,
        _kr1997_df_per_row,
    )

    L = emm_or_contrast.linfct

    # Build (G, σ²_e) from the cache (lme4 parameterisation) so the
    # path works post-pickle without raw_result.
    if cache is not None:
        theta_lmer_hat = cache.theta_hat
        k_re = cache.k_re
        sigma_sq_hat = cache.sigma_sq_hat
        X = cache.X
        Z = cache.Z
        groups = cache.groups
        group_ids = cache.group_ids
    else:
        result = info.raw_result
        theta_lmer_hat, k_re = _lmer_theta_hat(result)
        sigma_sq_hat = float(result.scale)
        X = np.asarray(result.model.exog, dtype=float)
        Z = np.asarray(result.model.exog_re, dtype=float)
        groups = np.asarray(result.model.groups)
        group_ids = np.unique(groups)
    # Recover G in pbkrtest's parameterisation from lme4's Λ
    # parameterisation: G = σ²_e · Λ Λᵀ.
    Lambda = _lmer_theta_to_lambda(theta_lmer_hat, k_re)
    G_pb = sigma_sq_hat * (Lambda @ Lambda.T)

    # Get KR vcov (parameterisation-invariant; same value either way).
    V_KR = kenward_roger_vcov(info, cache=cache)
    assert isinstance(V_KR, np.ndarray)

    aux = _compute_pbkrtest_aux_core(
        X=X, Z=Z, groups=groups, group_ids=group_ids,
        G=G_pb, sigma_sq=sigma_sq_hat, V_KR=V_KR,
    )
    df = _kr1997_df_per_row(
        L=L,
        V_beta=aux["V_beta"],
        P_list=aux["P_list"],
        W=aux["W"],
    )
    return _apply_correction(
        emm_or_contrast, V_KR, df, method="kenward_roger", satt_cache=cache
    )
