"""Kenward-Roger and Satterthwaite F-tests for nested ``MixedLM``.

Python ports of the user-facing ``pbkrtest`` API in R (Halekoh &
Højsgaard 2014, *J. Stat. Softw.* 59(9)) that complement the
parametric-bootstrap ``pbmodcomp`` already shipped in

- :func:`krmodcomp` — Kenward-Roger small-sample F-test for two
  nested ``MixedLM`` fits. Asymptotic counterpart to
  :func:`pymmeans.pbmodcomp.pbmodcomp`; use it when ``n_sim``
  bootstrap iterations are too expensive.
- :func:`satmodcomp` — Satterthwaite F-test for the same. Faster
  but less accurate at small ``n``; matches lmerTest's
  ``anova(model, ddf="Satterthwaite")``.
- :func:`ddf_lb` — standalone Kenward-Roger denominator df for an
  arbitrary contrast matrix L. The same machinery that powers
  :func:`pymmeans.satterthwaite.apply_kenward_roger`'s per-row df,
  exposed as a public function for users building their own
  contrasts.
- :func:`get_kr` — extract the KR adjustment's internal
  diagnostics (V_KR, W, V_beta, P_list). Useful for users
  reproducing the algorithm by hand or feeding the components
  into downstream code.

ships these as the second pbkrtest-equivalence batch.
After ``pymmeans`` covers **all four** user-facing
``pbkrtest`` functions (``vcovAdj``, ``PBmodcomp``, ``KRmodcomp``,
``SATmodcomp``) plus standalone ``ddf_Lb`` and ``getKR``. The
only remaining gap is the higher-order Kenward-Roger 1997 df
corrections (residual ~1% in ``ddf_Lb`` at the coefficient level);
those are a v0.2 milestone.

Algorithm
---------

The math here is ported line-for-line from pbkrtest's R source
(``.KR_adjust`` for KR; ``SATmodcomp_worker`` + ``get_Fstat_ddf``
for SAT). Validated against pbkrtest's printed output on
two reference cases (sleepstudy rank-1 + rank-2 nested tests) at
``atol=1e-4`` on the F statistic, df, and p-value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymmeans.satterthwaite import (
    KRInternals,
    _build_satt_cache,
    kenward_roger_vcov,
)
from pymmeans.utils import from_fitted


@dataclass
class FtestResult:
    """Result of :func:`krmodcomp` or :func:`satmodcomp`.

    Attributes
    ----------
    method
        Either ``"kenward_roger"`` or ``"satterthwaite"``.
    F
        Observed F statistic. For KR this is the **scaled** F
        (``F_scaling * F_unscaled``).
    ndf
        Numerator degrees of freedom = ``rank(L)``.
    ddf
        Denominator degrees of freedom (KR or Satterthwaite,
        depending on ``method``).
    p_value
        ``scipy.stats.f.sf(F, ndf, ddf)``.
    F_unscaled
        Unscaled Wald F = ``(Lβ)' (L V L')^{-1} (Lβ) / q``. For
        KR, also reported alongside the scaled F (matches pbkrtest's
        ``FtestU`` row). For SAT, equal to ``F``.
    F_scaling
        Kenward-Roger scale factor. ``1.0`` for ``method="kenward_roger"``
        with rank-1 contrasts; ``np.nan`` for ``method="satterthwaite"``.
    L
        The restriction matrix used.
    """

    method: str
    F: float
    ndf: int
    ddf: float
    p_value: float
    F_unscaled: float
    F_scaling: float
    L: np.ndarray

    def summary(self) -> str:
        """One-block text summary in pbkrtest style."""
        lines = [
            f"{self.method.replace('_', '-').title()} F-test "
            "(pymmeans port of pbkrtest)",
            "",
            f" F = {self.F:.4f}",
            f" ndf = {self.ndf}",
            f" ddf = {self.ddf:.4f}",
            f" p = {self.p_value:.4g}",
        ]
        if self.method == "kenward_roger" and self.ndf > 1:
            lines.append(
                f" F.scaling = {self.F_scaling:.4f} (F_unscaled = "
                f"{self.F_unscaled:.4f})"
            )
        return "\n".join(lines)


@dataclass
class KRDiagnostics:
    """Return type of :func:`get_kr`.

    Bundle of the KR adjustment's internal diagnostics. Useful for
    reproducing the algorithm by hand or feeding the components into
    downstream code (e.g., custom multi-DF F-tests beyond the ones
    pbkrtest itself implements).

    Attributes
    ----------
    V_KR
        Kenward-Roger inflated fixed-effect vcov (p × p).
    W
        Asymptotic ``Cov(θ̂)`` under REML, ``2 · inv(IE2)`` (n_θ × n_θ).
    V_beta
        Unadjusted REML vcov ``(X' Σ⁻¹ X)⁻¹`` (p × p).
    P_list
        Tuple of length n_θ; ``P_list[r] = ∂V_β/∂θ_r`` (each p × p).
    """

    V_KR: np.ndarray
    W: np.ndarray
    V_beta: np.ndarray
    P_list: tuple[np.ndarray, ...]


def _compute_pbkrtest_aux_core(
    X: np.ndarray,
    Z: np.ndarray,
    groups: np.ndarray,
    group_ids: np.ndarray,
    G: np.ndarray,
    sigma_sq: float,
    V_KR: np.ndarray,
) -> dict:
    """Core math kernel: build V_β, W, P_list in pbkrtest's
    parameterisation ``φ = (vech(G), σ²_e)`` from raw inputs.

    Decoupled from any specific fit-object API so that callers
    holding only a pickle-safe ``_SattCache`` can reach it. The
    fit-based wrapper :func:`_compute_pbkrtest_aux` just unpacks
    the fit and delegates here.
    """
    k_re = int(G.shape[0])
    vech_idx = [(i, j) for i in range(k_re) for j in range(i + 1)]
    n_g = len(vech_idx)
    n_phi = n_g + 1

    def build_Sigma_groups(G_in: np.ndarray, sigma_sq_in: float) -> list[np.ndarray]:
        Vs = []
        for g_id in group_ids:
            m = groups == g_id
            Z_g = Z[m]
            n_obs_g = int(m.sum())
            V_g = Z_g @ G_in @ Z_g.T + sigma_sq_in * np.eye(n_obs_g)
            Vs.append(V_g)
        return Vs

    def Vbeta_at(G_in: np.ndarray, sigma_sq_in: float) -> np.ndarray:
        A = np.zeros((X.shape[1], X.shape[1]))
        for g_id in group_ids:
            m = groups == g_id
            Z_g = Z[m]
            X_g = X[m]
            n_obs_g = int(m.sum())
            V_g = Z_g @ G_in @ Z_g.T + sigma_sq_in * np.eye(n_obs_g)
            A += X_g.T @ np.linalg.solve(V_g, X_g)
        return np.linalg.inv(A)

    V_beta = Vbeta_at(G, sigma_sq)

    n_groups = len(group_ids)
    X_groups = [X[groups == g_id] for g_id in group_ids]
    Sigma_groups = build_Sigma_groups(G, sigma_sq)
    SigmaInv_groups = [np.linalg.inv(S) for S in Sigma_groups]

    def Vp_g_for_phi(r: int, g_idx: int) -> np.ndarray:
        Z_g = Z[groups == group_ids[g_idx]]
        n_obs_g = Z_g.shape[0]
        if r < n_g:
            i, j = vech_idx[r]
            E_ij = np.zeros((k_re, k_re))
            if i == j:
                E_ij[i, i] = 1.0
            else:
                E_ij[i, j] = 1.0
                E_ij[j, i] = 1.0
            return Z_g @ E_ij @ Z_g.T
        return np.eye(n_obs_g)

    Vp_groups = [
        [Vp_g_for_phi(r, g) for g in range(n_groups)]
        for r in range(n_phi)
    ]
    HH = [
        [Vp_groups[r][g] @ SigmaInv_groups[g] for g in range(n_groups)]
        for r in range(n_phi)
    ]
    OO = [
        [HH[r][g] @ X_groups[g] for g in range(n_groups)]
        for r in range(n_phi)
    ]
    TT = [SigmaInv_groups[g] @ X_groups[g] for g in range(n_groups)]

    Ktrace = np.zeros((n_phi, n_phi))
    for r in range(n_phi):
        for s in range(r, n_phi):
            tr_acc = 0.0
            for g in range(n_groups):
                tr_acc += float(np.sum(HH[r][g].T * HH[s][g]))
            Ktrace[r, s] = Ktrace[s, r] = tr_acc

    PP_pb: list[np.ndarray] = []
    for r in range(n_phi):
        PP_r = np.zeros((X.shape[1], X.shape[1]))
        for g in range(n_groups):
            PP_r -= OO[r][g].T @ TT[g]
        PP_pb.append(PP_r)

    QQ_pb: dict[tuple[int, int], np.ndarray] = {}
    for r in range(n_phi):
        for s in range(r, n_phi):
            Q_rs = np.zeros((X.shape[1], X.shape[1]))
            for g in range(n_groups):
                Q_rs += OO[r][g].T @ SigmaInv_groups[g] @ OO[s][g]
            QQ_pb[(r, s)] = Q_rs
            QQ_pb[(s, r)] = Q_rs.T

    IE2 = np.zeros((n_phi, n_phi))
    for r in range(n_phi):
        Phi_Pr = V_beta @ PP_pb[r]
        for s in range(r, n_phi):
            cross = float(np.sum(V_beta * QQ_pb[(r, s)]))
            Ps_Phi = PP_pb[s] @ V_beta
            outer = float(np.sum(Phi_Pr * Ps_Phi))
            IE2[r, s] = IE2[s, r] = Ktrace[r, s] - 2.0 * cross + outer

    W = 2.0 * np.linalg.inv(IE2)

    return dict(
        V_beta=V_beta,
        V_KR=V_KR,
        P_list=tuple(PP_pb),
        W=W,
    )


def _compute_pbkrtest_aux(fit: Any) -> dict:
    """Build the KR auxiliary quantities in **pbkrtest's parameterisation**.

    pymmeans's :class:`KRInternals` is in lme4's
    parameterisation ``(vech(Λ), σ²_e)``; pbkrtest's ``.KR_adjust``
    expects ``(vech(G), σ²_e)``. The two are related by
    ``G = σ²_e · Λ Λ'`` (a non-linear transformation), so the
    ``P_list`` and ``W`` matrices in the two parameterisations
    differ — and the ``A1``/``A2`` sums in ``.KR_adjust`` are NOT
    invariant under that transformation (the formula assumes
    pbkrtest's parameterisation specifically).

    This helper computes ``V_β``, ``V_KR``, ``P_list``, and ``W``
    fresh in pbkrtest's parameterisation:

    - ``P_list[r] = ∂V_β/∂φ_r`` for ``φ = (vech(G), σ²_e)``.
      Finite-difference on V_β at perturbed ``(G, σ²_e)``.
    - ``W = 2 · I_E^{-1}``, where ``I_E[r, s] = (1/2) tr(Σ⁻¹ V'_r Σ⁻¹ V'_s)``
      with ``V'_r = ∂Σ/∂φ_r``: for ``G_jk`` it's
      ``Z · ½(e_j e_k' + e_k e_j') · Z'``; for ``σ²_e`` it's the
      identity.

    Returns a dict with keys ``V_beta``, ``V_KR``, ``P_list``,
    ``W``, ``beta``. V_KR is taken from
    :func:`kenward_roger_vcov` (parameterisation-invariant, so the
    lme4 path's V_KR is the correct answer to use here).
    """
    info = from_fitted(fit)
    cache = _build_satt_cache(fit)
    internals = kenward_roger_vcov(info, cache=cache, return_internals=True)
    assert isinstance(internals, KRInternals)
    V_KR = internals.V_KR
    beta = np.asarray(fit.fe_params, dtype=float)
    G = np.asarray(fit.cov_re, dtype=float)
    sigma_sq = float(fit.scale)

    aux = _compute_pbkrtest_aux_core(
        X=cache.X,
        Z=cache.Z,
        groups=cache.groups,
        group_ids=cache.group_ids,
        G=G,
        sigma_sq=sigma_sq,
        V_KR=V_KR,
    )
    return dict(
        V_beta=aux["V_beta"],
        V_KR=aux["V_KR"],
        P_list=aux["P_list"],
        W=aux["W"],
        beta=beta,
    )


def _build_nested_L(large: Any, small: Any) -> np.ndarray:
    """Return the contrast matrix L that captures the fixed-effect
    difference between ``small`` and ``large``.

    Approach: match by column names in ``model.exog_names``. If
    ``small`` is a strict column-subset of ``large`` (the
    common-case nested-formula pattern), build L as the identity
    rows of ``large``'s coefficient vector corresponding to the
    columns added in ``large``.

    Raises ``ValueError`` if (a) ``small``'s columns are not a
    subset of ``large``'s, (b) the difference is empty (models are
    identical), or (c) the difference is negative (arguments
    swapped).
    """
    large_cols = list(getattr(large.model, "exog_names", [])) or [
        f"x{i}" for i in range(large.model.exog.shape[1])
    ]
    small_cols = list(getattr(small.model, "exog_names", [])) or [
        f"x{i}" for i in range(small.model.exog.shape[1])
    ]

    if len(small_cols) >= len(large_cols):
        raise ValueError(
            f"krmodcomp/satmodcomp: `large` must have strictly more "
            f"fixed-effect parameters than `small` "
            f"(large has {len(large_cols)}, small has {len(small_cols)}). "
            "If you swapped the arguments, retry with `large` first."
        )

    missing = [c for c in small_cols if c not in large_cols]
    if missing:
        raise ValueError(
            "krmodcomp/satmodcomp: `small` must be a strict "
            "column-subset of `large` (column-name matching). "
            f"Columns in `small` but not `large`: {missing}. "
            "Refit both with formulas that share the common terms."
        )

    added = [c for c in large_cols if c not in small_cols]
    p_large = len(large_cols)
    rows = [large_cols.index(c) for c in added]
    L = np.zeros((len(rows), p_large))
    for i, j in enumerate(rows):
        L[i, j] = 1.0
    return L


def _spur(A: np.ndarray) -> float:
    """Trace shorthand (matches pbkrtest's `.spur`)."""
    return float(np.trace(A))


def _kr_adjust(
    Phi_A: np.ndarray,
    Phi: np.ndarray,
    L: np.ndarray,
    beta: np.ndarray,
    P_list: tuple[np.ndarray, ...],
    W: np.ndarray,
) -> tuple[float, float, float, float, float]:
    """Pymmeans port of pbkrtest's ``.KR_adjust`` math kernel.

    Inputs are the KR-adjusted vcov ``Phi_A = V_KR``, the
    unadjusted vcov ``Phi = V_beta``, the contrast matrix ``L``,
    the fixed-effect coefficient vector ``beta``, the per-θ
    derivatives ``P_list[r] = ∂V_β/∂θ_r``, and the variance-
    component covariance ``W = 2 · inv(IE2)``.

    Returns a tuple ``(F, F_unscaled, ndf, ddf, F_scaling)``
    matching pbkrtest's ``Fstat``, ``FstatU``, ``ndf``, ``ddf``,
    ``F.scaling``.

    See the inline comments for which line in pbkrtest's R source
    maps to which Python expression.
    """
    # Theta = L' (L Φ L')^{-1} L — p × p (pbkrtest's `Theta`)
    LPhiLt = L @ Phi @ L.T
    Theta = L.T @ np.linalg.solve(LPhiLt, L)

    # ThetaPhi = Theta · Φ — p × p
    ThetaPhi = Theta @ Phi

    # A1 and A2 sums over the variance-component pairs.
    n_theta = len(P_list)
    A1 = 0.0
    A2 = 0.0
    for ii in range(n_theta):
        for jj in range(ii, n_theta):
            e = 1.0 if ii == jj else 2.0
            ui = ThetaPhi @ P_list[ii] @ Phi
            uj = ThetaPhi @ P_list[jj] @ Phi
            A1 += e * float(W[ii, jj]) * _spur(ui) * _spur(uj)
            # spur(ui · uj^T) = sum elementwise (ui · uj^T) =
            # sum elementwise (ui ⊙ uj.T) = sum(ui * uj.T).
            A2 += e * float(W[ii, jj]) * float(np.sum(ui * uj.T))

    # q = rank(L). For the nested-model case L is a row-selector
    # of unique rows, so rank = number of rows. Use SVD for the
    # rare ill-conditioned case.
    q = int(np.linalg.matrix_rank(L))

    B = (1.0 / (2.0 * q)) * (A1 + 6.0 * A2)
    g = ((q + 1.0) * A1 - (q + 4.0) * A2) / ((q + 2.0) * A2) if A2 != 0 else 0.0
    denom_c = 3.0 * q + 2.0 * (1.0 - g)
    c1 = g / denom_c
    c2 = (q - g) / denom_c
    c3 = (q + 2.0 - g) / denom_c

    V0 = 1.0 + c1 * B
    V1 = 1.0 - c2 * B
    V2 = 1.0 - c3 * B
    if abs(V0) < 1e-10:
        V0 = 0.0

    # rho = (1/q) · ((1 - A2/q) / V1)^2 · V0 / V2 (with safe div)
    inner = (1.0 - A2 / q) / V1 if V1 != 0 else 0.0
    rho = (1.0 / q) * (inner ** 2) * V0 / V2 if V2 != 0 else 0.0

    df2 = 4.0 + (q + 2.0) / (q * rho - 1.0) if (q * rho - 1.0) != 0 else np.inf

    F_scaling = (
        1.0
        if abs(df2 - 2.0) < 0.01
        else df2 * (1.0 - A2 / q) / (df2 - 2.0)
    )

    # Wald (using PhiA = V_KR) — pbkrtest names this `Wald`.
    LPhiALt = L @ Phi_A @ L.T
    Lb = L @ beta
    Wald = float(Lb @ np.linalg.solve(LPhiALt, Lb))

    # Unscaled F = Wald / q.
    F_unscaled = Wald / q
    F = F_scaling * F_unscaled

    return F, F_unscaled, q, df2, F_scaling


def _kr1997_df_per_row(
    L: np.ndarray,
    V_beta: np.ndarray,
    P_list: tuple[np.ndarray, ...] | list[np.ndarray],
    W: np.ndarray,
) -> np.ndarray:
    """Kenward-Roger 1997 denominator df per row of L.

    Direct port of ``pbkrtest:::ddf_Lb`` (single-row, q=1 special
    case of ``.KR_adjust``). For each row ``L_i`` of ``L``:

    1. ``vlb = L_i' V_β L_i`` (scalar variance)
    2. ``Theta = (L_i L_i') / vlb`` (p × p rank-1 outer product)
    3. ``A1, A2`` via the same double-sum over variance-component
       pairs that drives the F-test scale factor:
       ``A1 = Σ_ij e · W[i,j] · tr(u_i) · tr(u_j)``
       ``A2 = Σ_ij e · W[i,j] · Σ(u_i ⊙ u_j^T)``
       where ``u_i = Theta · V_β · P_i · V_β`` and ``e = 1`` if
       ``i == j`` else ``2``.
    4. ``df = 4 + 3 / (ρ - 1)`` where ``ρ`` and the helper
       constants come from the K-R 1997 second-moment matching
       (see :func:`_kr_adjust` for the multi-DF parent formula
       with ``q = 1`` substituted throughout).

    Returns a length-``q`` array of dfs, one per row of L. All
    inputs must be in pbkrtest's parameterisation (use
    :func:`_compute_pbkrtest_aux` or
    :func:`_compute_pbkrtest_aux_core` to build them).

    pre-this-function pymmeans used a Satterthwaite-style
    delta-method df that drifted from pbkrtest's ``ddf_Lb`` by
    ~1 % relative on the canonical reference fit. This function
    closes the gap to floating-point parity by porting pbkrtest's
    exact formula.
    """
    L = np.atleast_2d(np.asarray(L, dtype=float))
    n_rows = L.shape[0]
    n_theta = len(P_list)
    df_out = np.zeros(n_rows)
    for r_idx in range(n_rows):
        L_row = L[r_idx]
        vlb = float(L_row @ V_beta @ L_row)
        if vlb <= 0:
            df_out[r_idx] = np.inf
            continue
        # Theta = (L_row outer L_row) / vlb (p × p rank-1)
        Theta = np.outer(L_row, L_row) / vlb
        ThetaVb = Theta @ V_beta
        # A1, A2 accumulation (q = 1 specialisation of _kr_adjust).
        A1 = 0.0
        A2 = 0.0
        for ii in range(n_theta):
            for jj in range(ii, n_theta):
                e = 1.0 if ii == jj else 2.0
                ui = ThetaVb @ P_list[ii] @ V_beta
                uj = ThetaVb @ P_list[jj] @ V_beta
                tr_ui = float(np.trace(ui))
                tr_uj = float(np.trace(uj))
                A1 += e * float(W[ii, jj]) * tr_ui * tr_uj
                A2 += e * float(W[ii, jj]) * float(np.sum(ui * uj.T))
        # B, g, c1, c2, c3, V0, V1, V2, rho, df2 — exactly the
        # q=1 substitution of _kr_adjust's formulas, with the same
        # ``.divZero`` edge-case convention pbkrtest uses (returns
        # 1 when both numerator and denominator are within tol of
        # zero; otherwise plain division allowing inf/nan to
        # propagate). a divergence
        # where pymmeans previously returned 0 on degenerate
        # branches while pbkrtest returned 1 or Inf.
        B = (A1 + 6.0 * A2) / 2.0
        g = _div_zero(2.0 * A1 - 5.0 * A2, 3.0 * A2)
        denom_c = 3.0 + 2.0 * (1.0 - g)
        c1 = g / denom_c
        c2 = (1.0 - g) / denom_c
        c3 = (3.0 - g) / denom_c
        V0 = 1.0 + c1 * B
        V1 = 1.0 - c2 * B
        V2 = 1.0 - c3 * B
        if abs(V0) < 1e-10:
            V0 = 0.0
        inner = _div_zero(1.0 - A2, V1)
        rho = (inner ** 2) * V0 / V2 if V2 != 0 else np.inf
        if rho == 1.0:
            df_out[r_idx] = np.inf
        else:
            df_out[r_idx] = 4.0 + 3.0 / (rho - 1.0)
    return df_out


def _div_zero(x: float, y: float, tol: float = 1e-14) -> float:
    """Python port of pbkrtest's ``.divZero(x, y, tol)``.

    Returns ``1`` when both ``|x|`` and ``|y|`` are below ``tol``;
    otherwise plain ``x / y`` (allowing ``Inf`` / ``NaN`` to
    propagate as in R). previously pymmeans
    silently returned 0 on degenerate branches, diverging from
    pbkrtest's ``.divZero`` behaviour. On the canonical reference
    fits these branches never fire (the standard validation lands
    at rel ~5e-7), but a pathological model can hit them and the
    the previous version would have produced a different df than R.
    """
    if abs(x) < tol and abs(y) < tol:
        return 1.0
    return x / y


def krmodcomp(large: Any, small: Any) -> FtestResult:
    """Kenward-Roger F-test for two nested ``MixedLM`` fits.

    Python port of ``pbkrtest::KRmodcomp``. Asymptotic counterpart
    to :func:`pymmeans.pbmodcomp.pbmodcomp`; use this when the
    bootstrap is too expensive. At small sample sizes the
    bootstrap is generally more accurate; at moderate n the F-test
    converges to the bootstrap and is much faster.

    Parameters
    ----------
    large, small
        Two ``statsmodels.MixedLMResults`` fits with ``small``
        nested in ``large`` via fixed-effect formula restriction.
        Both fits should be REML (the algorithm uses REML
        log-likelihoods; passing ML fits will produce a warning
        but still run).

    Returns
    -------
    FtestResult
        ``method="kenward_roger"``.

    Validation
    ----------
    Matches ``pbkrtest::KRmodcomp`` at ``atol=1e-4`` on the F
    statistic, df, and p-value on two reference cases:
    sleepstudy rank-1 (``Reaction ~ Days + (1|Subject)`` vs ``~ 1
    + (1|Subject)``) and sleepstudy rank-2 (``Reaction ~ Days +
    Days^2 + (Days|Subject)`` vs ``~ 1 + (Days|Subject)``). See
    ``tests/r_reference/pbkrtest_ftests.R``.

    References
    ----------
    - Kenward, M. G., & Roger, J. H. (1997). Small Sample Inference
      for Fixed Effects from Restricted Maximum Likelihood.
      *Biometrics*, 53(3), 983-997.
    - Halekoh, U., & Højsgaard, S. (2014). A Kenward-Roger
      Approximation and Parametric Bootstrap Methods for Tests in
      Linear Mixed Models — The R Package pbkrtest.
      *Journal of Statistical Software*, 59(9). doi:10.18637/jss.v059.i09
    """
    L = _build_nested_L(large, small)
    aux = _compute_pbkrtest_aux(large)
    F, F_unscaled, ndf, ddf, F_scaling = _kr_adjust(
        Phi_A=aux["V_KR"],
        Phi=aux["V_beta"],
        L=L,
        beta=aux["beta"],
        P_list=aux["P_list"],
        W=aux["W"],
    )

    from scipy import stats
    p_value = float(stats.f.sf(F, ndf, ddf))

    return FtestResult(
        method="kenward_roger",
        F=F,
        ndf=ndf,
        ddf=ddf,
        p_value=p_value,
        F_unscaled=F_unscaled,
        F_scaling=F_scaling,
        L=L,
    )


def satmodcomp(large: Any, small: Any) -> FtestResult:
    """Satterthwaite F-test for two nested ``MixedLM`` fits.

    Python port of ``pbkrtest::SATmodcomp``. Faster and less
    accurate than :func:`krmodcomp` at small ``n``; equivalent
    asymptotically. At moderate-to-large samples both converge
    to the same F distribution.

    Algorithm (pbkrtest's ``SATmodcomp_worker``):

    1. Compute ``vcov_Lβ = L V_β L'`` and its eigendecomposition
       ``vcov_Lβ = P D P'``.
    2. The eigenvectors give a rotated contrast basis;
       ``q = rank(L)`` is the number of non-zero eigenvalues.
    3. For each rotated direction ``m = 1..q``:
       ``t_m^2 = (P' L β)_m^2 / D_m``.
    4. ``F = (Σ t_m^2) / q``.
    5. ``df_den`` per direction =
       ``2 D_m^2 / Var(P' L V_β L' P)_m``, then combined across
       directions via Satterthwaite's harmonic-like formula
       (``get_Fstat_ddf`` in pbkrtest).

    Parameters
    ----------
    large, small
        See :func:`krmodcomp`.

    Returns
    -------
    FtestResult
        ``method="satterthwaite"``. ``F_unscaled == F``,
        ``F_scaling = NaN``.

    Validation
    ----------
    Matches ``pbkrtest::SATmodcomp`` at ``atol=1e-3`` on the F
    statistic and ``atol=0.5`` on ``ddf`` (the multi-DF
    Satterthwaite df has substantial finite-difference noise from
    the V_β gradient). See ``tests/r_reference/pbkrtest_ftests.R``.
    """
    L = _build_nested_L(large, small)
    info = from_fitted(large)
    V_beta = np.asarray(info.vcov, dtype=float)
    beta = np.asarray(large.fe_params, dtype=float)

    vcov_LB = L @ V_beta @ L.T
    eigvals, eigvecs = np.linalg.eigh(vcov_LB)
    # eigh returns ascending; reverse for pbkrtest's "leading first"
    order = np.argsort(-eigvals)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    tol = max(np.sqrt(np.finfo(float).eps) * eigvals[0], 0.0)
    pos = eigvals > tol
    q = int(np.sum(pos))
    if q == 0:
        raise ValueError(
            "satmodcomp: contrast matrix is degenerate "
            "(no positive eigenvalues of L V_β L')."
        )

    # Rotated contrast basis: PtL has shape (q, p) = (q rotated
    # directions, p fixed-effect parameters).
    PtL = (eigvecs.T @ L)[:q, :]
    Lbeta_rotated = PtL @ beta
    t2 = Lbeta_rotated ** 2 / eigvals[:q]
    F = float(np.sum(t2) / q)

    # Per-direction Satterthwaite df.
    # ddf_m = 2 D_m^2 / (g_m' W g_m)
    # where g_m[r] = ∂(PtL_m · V_β · PtL_m')/∂θ_r =
    # PtL_m · (∂V_β/∂θ_r) · PtL_m' (a scalar in m)
    cache = _build_satt_cache(large)
    internals = kenward_roger_vcov(info, cache=cache, return_internals=True)
    assert isinstance(internals, KRInternals)
    P_list = internals.P_list
    W = internals.W
    n_theta = len(P_list)

    nu_m = np.zeros(q)
    for m in range(q):
        v = PtL[m, :]
        grad = np.array(
            [float(v @ P_list[r] @ v) for r in range(n_theta)]
        )
        var_d = float(grad @ W @ grad)
        nu_m[m] = 2.0 * (eigvals[m] ** 2) / var_d if var_d > 0 else np.inf

    ddf = _combine_satterthwaite_df(nu_m)

    from scipy import stats
    p_value = float(stats.f.sf(F, q, ddf))

    return FtestResult(
        method="satterthwaite",
        F=F,
        ndf=q,
        ddf=ddf,
        p_value=p_value,
        F_unscaled=F,
        F_scaling=float("nan"),
        L=L,
    )


def _combine_satterthwaite_df(nu: np.ndarray, tol: float = 1e-8) -> float:
    """Pymmeans port of pbkrtest's ``get_Fstat_ddf``.

    Combines per-direction Satterthwaite df values into a single
    F-distribution denominator df. When ``len(nu) == 1`` returns
    ``nu[0]``; when all ``nu`` agree within ``tol`` returns
    ``mean(nu)``; otherwise uses pbkrtest's pooled formula.
    """
    if len(nu) == 1:
        return float(nu[0])
    if np.all(np.abs(np.diff(nu)) < tol):
        return float(np.mean(nu))
    if np.any(nu <= 2):
        return 2.0
    E = float(np.sum(nu / (nu - 2.0)))
    return 2.0 * E / (E - len(nu))


def ddf_lb(fit: Any, L: np.ndarray) -> np.ndarray:
    """Kenward-Roger denominator df for an arbitrary contrast matrix L.

    Python equivalent of ``pbkrtest::Lb_ddf`` / ``ddf_Lb``. Each
    row of ``L`` is treated as a separate scalar contrast
    returned array has one df per row.

    Parameters
    ----------
    fit
        Fitted ``statsmodels.MixedLMResults``.
    L
        ``(q, p)`` contrast matrix, each row a linear combination
        of the fixed-effect coefficients.

    Returns
    -------
    np.ndarray
        Length-``q`` vector of KR denominator df values (one per
        row of L).

    Notes
    -----
    For scalar contrasts (single-row L) this matches the per-row
    df produced internally by
    :func:`pymmeans.satterthwaite.apply_kenward_roger`. Exposing
    it as a standalone function lets users build their own
    contrasts (e.g., bespoke linear combinations not expressible
    as ``pairs()``) without going through the EMM pipeline.

    previously, ``ddf_lb`` used a Satterthwaite-
    style delta-method df that drifted ~1 % rel from
    ``pbkrtest::Lb_ddf``. had already closed that gap in
    ``apply_kenward_roger`` via the K-R 1997 formula;
    caught that ``ddf_lb`` (the public, doc-discoverable entry
    point named after ``Lb_ddf``) was still on the old path.
    Now both paths use :func:`_kr1997_df_per_row`, the exact port
    of pbkrtest's ``ddf_Lb`` math kernel. Matches
    ``pbkrtest::Lb_ddf`` at ``atol=1e-3`` on the canonical
    reference fit (residual dominated by finite-diff noise on the
    per-θ derivatives, not formula error).
    """
    L = np.atleast_2d(np.asarray(L, dtype=float))
    info = from_fitted(fit)
    cache = _build_satt_cache(fit)
    # Recover pbkrtest's parameterisation (G = σ²_e · Λ Λᵀ).
    from pymmeans.satterthwaite import _lmer_theta_to_lambda
    sigma_sq = cache.sigma_sq_hat
    Lambda = _lmer_theta_to_lambda(cache.theta_hat, cache.k_re)
    G_pb = sigma_sq * (Lambda @ Lambda.T)
    V_KR_raw = kenward_roger_vcov(info, cache=cache)
    assert isinstance(V_KR_raw, np.ndarray)
    aux = _compute_pbkrtest_aux_core(
        X=cache.X, Z=cache.Z,
        groups=cache.groups, group_ids=cache.group_ids,
        G=G_pb, sigma_sq=sigma_sq, V_KR=V_KR_raw,
    )
    return _kr1997_df_per_row(
        L=L,
        V_beta=aux["V_beta"],
        P_list=aux["P_list"],
        W=aux["W"],
    )


def get_kr(fit: Any) -> KRDiagnostics:
    """Extract Kenward-Roger adjustment diagnostics from a MixedLM fit.

    Python equivalent of ``pbkrtest::getKR``. Returns the four
    matrices that pbkrtest exposes via attributes on
    ``vcovAdj(model)``: the KR-adjusted vcov, the variance-
    component covariance ``W = 2 · inv(IE2)``, the unadjusted REML
    vcov ``V_beta``, and the list of per-θ derivatives of ``V_β``.

    Parameters
    ----------
    fit
        Fitted ``statsmodels.MixedLMResults``.

    Returns
    -------
    KRDiagnostics
        ``V_KR``, ``W``, ``V_beta``, ``P_list``.

    Notes
    -----
    This is a thin wrapper over
    :func:`pymmeans.satterthwaite.kenward_roger_vcov` with
    ``return_internals=True``; the motivation is to give
    R users a function with a familiar name.
    """
    info = from_fitted(fit)
    internals = kenward_roger_vcov(info, return_internals=True)
    assert isinstance(internals, KRInternals)
    return KRDiagnostics(
        V_KR=internals.V_KR,
        W=internals.W,
        V_beta=internals.V_beta,
        P_list=internals.P_list,
    )
