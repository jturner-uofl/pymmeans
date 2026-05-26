"""Cumulative-link ordinal regression EMMs (``statsmodels.OrderedModel``).

bringing pymmeans up to R `emmeans`'s ordinal-model
coverage. Adds response-scale modes for ordinal fits:

- ``mode="latent"`` (default for :func:`emmeans` on OrderedModel) —
  the linear predictor ``η = X β`` at the reference grid. This
  mode is just the standard EMM call; it works through
  :func:`emmeans` directly with no special path.
- ``mode="prob"`` — per-category probabilities ``P(Y = j | x)``
  for each category ``j``. Multi-row output (one row per
  target-level × category).
- ``mode="mean.class"`` — expected category ``E[Y | x] =
  Σ_j j · P(Y = j | x)``. Single-row output.
- ``mode="cum.prob"`` / ``mode="exc.prob"`` — cumulative
  probabilities ``P(Y ≤ j | x)`` and exceedance probabilities
  ``P(Y > j | x)``.

For the response-scale modes we apply the delta method through
the cumulative-link transformation, accounting for the
covariance between the structural ``β`` and the cumulative
thresholds ``τ``. R `emmeans` does exactly the same thing under
the hood.

Validation: against R `MASS::polr` and `ordinal::clm` on the
classic ``housing`` dataset (Venables & Ripley) at
``atol=1e-3`` on point estimates and ``atol=1e-2`` on SEs.

References
----------
- McCullagh, P. (1980). Regression Models for Ordinal Data.
  *Journal of the Royal Statistical Society. Series B
  (Methodological)*, 42(2), 109-142.
- Christensen, R. H. B. (2019). ``ordinal`` — Regression Models
  for Ordinal Data. R package version 2019.12-10.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pymmeans.emmeans import EMMResult, emmeans
from pymmeans.utils import ModelInfo, from_statsmodels


def _ordinal_thresholds_and_jacobian(
    result: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Recover the untransformed cumulative thresholds and the
    Jacobian from statsmodels' transformed parameterisation to the
    actual thresholds.

    statsmodels' OrderedModel parameterises the J-1 thresholds as

        ξ_0 = τ_0 (raw first threshold)
        ξ_j = log(τ_j - τ_{j-1}) j ≥ 1 (log-spacing)

    so that τ_j = τ_{j-1} + exp(ξ_j) is guaranteed monotonically
    increasing. This helper returns ``(τ, J)`` where ``τ`` is the
    actual threshold vector (length J-1) and ``J`` is the
    Jacobian ``∂τ/∂ξ`` (a lower-triangular (J-1) × (J-1) matrix)
    used by the delta method to map ``cov(ξ)`` into ``cov(τ)``.
    """
    k_vars = int(result.model.k_vars)
    k_levels = int(result.model.k_levels)
    n_thresh = k_levels - 1
    xi = np.asarray(result.params[k_vars:], dtype=float)
    if len(xi) != n_thresh:
        raise ValueError(
            f"OrderedModel param vector size mismatch: expected "
            f"{n_thresh} threshold parameters after the {k_vars} "
            f"structural beta(s), got {len(xi)}."
        )
    tau = np.zeros(n_thresh)
    tau[0] = xi[0]
    for j in range(1, n_thresh):
        tau[j] = tau[j - 1] + np.exp(xi[j])
    # Jacobian ∂τ/∂ξ:
    # ∂τ_j/∂ξ_0 = 1 (every τ depends on the base)
    # ∂τ_j/∂ξ_i = exp(ξ_i) (for i in [1, j])
    # ∂τ_j/∂ξ_i = 0 (for i > j)
    Jac = np.zeros((n_thresh, n_thresh))
    Jac[:, 0] = 1.0
    for i in range(1, n_thresh):
        Jac[i:, i] = np.exp(xi[i])
    return tau, Jac


def _link_cdf_pdf(link: str) -> tuple[Any, Any]:
    """Return the (CDF, PDF) pair for the cumulative-link family.

    an earlier version had a hand-coded
    ``cloglog`` branch using the Gumbel-min CDF ``1 - exp(-exp(x))``.
    But statsmodels' ``OrderedModel`` does NOT natively expose
    cloglog as a built-in distr name — to get it the user passes a
    custom scipy distribution. Different scipy choices yield
    different conventions (``gumbel_r`` is Gumbel-max with CDF
    ``exp(-exp(-x))``; the right cloglog is Gumbel-min). A user
    copying R's cloglog convention with the wrong scipy class would
    have gotten silently wrong SEs and probabilities.

    refuse anything other than the two
    statsmodels-native distrs (logit, probit). Users who need
    cloglog can fit it externally and feed β + V through
    :func:`pymmeans.qdrg`, where the link/transform is explicit.
    """
    from scipy import stats
    link = link.lower()
    if link in ("logit", "logistic"):
        return stats.logistic.cdf, stats.logistic.pdf
    if link == "probit":
        return stats.norm.cdf, stats.norm.pdf
    raise NotImplementedError(
        f"ordinal_emmeans: cumulative-link family `{link}` is not "
        "supported. statsmodels' ``OrderedModel`` natively "
        "supports `distr='logit'` and `distr='probit'`; other "
        "distrs (cloglog / Gumbel) require careful sign-convention "
        "matching that pymmeans does not yet handle. For a "
        "cloglog ordinal fit, build the design matrix and "
        "coefficient vector manually and pass them through "
        "``pymmeans.qdrg(formula, data, coef, vcov, df, family=...)``."
    )


def _extract_ordinal_state(result: Any) -> dict[str, Any]:
    """Pull the structural beta, the threshold vector, the full
    parameter covariance, and the link from a fitted OrderedModel.

    Returns a dict with keys:

    - ``beta``: structural coefficients (length p, no intercept)
    - ``tau``: cumulative thresholds (length J - 1, in
      original-scale order)
    - ``cov_full``: full ``(p + J - 1) × (p + J - 1)`` parameter
      covariance from the fit, **reparameterised to (β, τ)**
      via the Jacobian helper above
    - ``link``: ``'logit'`` / ``'probit'`` / ``'cloglog'``
    - ``categories``: ordered list of J category labels (as
      ``pd.Categorical``-style codes if the original endog was
      categorical, else simple integer codes)
    """
    model = result.model
    k_vars = int(model.k_vars)
    n_thresh = int(model.k_levels) - 1
    beta = np.asarray(result.params[:k_vars], dtype=float)
    tau, Jac = _ordinal_thresholds_and_jacobian(result)

    cov_full_raw = np.asarray(result.cov_params(), dtype=float)
    # Reparameterise threshold block to (τ, τ, ...) scale via the
    # block-diagonal Jacobian [[I, 0], [0, J_ξ→τ]].
    p = k_vars
    M = np.eye(p + n_thresh)
    M[p:, p:] = Jac
    cov_full = M @ cov_full_raw @ M.T

    # Map link name from statsmodels distr to scipy. ``distr``
    # can be the user-passed string OR (after fitting) a scipy
    # frozen distribution object — dispatch on either form.
    #
    # previously used substring matching on the
    # class name (e.g. ``"logistic" in name.lower()``). That's
    # brittle — a future scipy rename would silently break it. We
    # now compare against the actual `stats.logistic` /
    # `stats.norm` references by `.cdf` identity, falling back to
    # the string-name path only when the object isn't recognised
    # (in which case the downstream ``_link_cdf_pdf`` refusal
    # fires with a clear error).
    # Scipy ``rv_continuous`` instances expose a stable ``.name``
    # string ('logistic', 'norm', 'gumbel_r', etc.) that survives
    # bound-method identity issues.
    distr_attr = getattr(model, "distr", "logit")
    if isinstance(distr_attr, str):
        link = distr_attr
    else:
        distr_name = str(getattr(distr_attr, "name", "")).lower()
        if distr_name == "logistic":
            link = "logit"
        elif distr_name == "norm":
            link = "probit"
        else:
            # Unknown distr; pass it through to the refuser so the
            # user gets a clear "not supported" error mentioning
            # qdrg as the workaround.
            link = distr_name or str(type(distr_attr).__name__)
    link = {"logistic": "logit"}.get(link, link)

    # Categories: prefer pandas Categorical metadata if available.
    categories: list[Any]
    endog = getattr(model, "endog", None)
    labels = getattr(model, "labels", None)
    if labels is not None and len(labels) == n_thresh + 1:
        categories = list(labels)
    elif endog is not None:
        try:
            categories = sorted(set(np.asarray(endog).tolist()))
        except Exception:
            categories = list(range(n_thresh + 1))
    else:
        categories = list(range(n_thresh + 1))

    return {
        "beta": beta,
        "tau": tau,
        "cov_full": cov_full,
        "link": link,
        "categories": categories,
    }


def _broadcast_to_reference_grid(
    fit: Any, specs: Any, **emm_kwargs: Any
) -> tuple[EMMResult, ModelInfo]:
    """Run a standard latent-mode :func:`emmeans` to get the
    reference grid + the linear predictor ``L β`` at each row.

    Returns the EMMResult plus the ModelInfo (for re-use with the
    full covariance below).
    """
    info = from_statsmodels(fit)
    em = emmeans(info, specs, **emm_kwargs)
    return em, info


def ordinal_emmeans(
    fit: Any,
    specs: Any,
    mode: str = "prob",
    by: Any = None,
    at: Any = None,
    level: float = 0.95,
    **kwargs: Any,
) -> EMMResult:
    """Response-scale EMMs for ``statsmodels.OrderedModel``.

    Parameters
    ----------
    fit
        A fitted ``statsmodels.miscmodels.ordinal_model.OrderedModelResults``.
    specs
        Factors / numeric predictors to marginalise over — same
        semantics as :func:`emmeans`.
    mode
        Response-scale mode. One of:

        - ``"latent"`` — linear predictor ``η = X β`` (delegates to
          :func:`emmeans`; ``ordinal_emmeans`` is a no-op wrapper
          in this case).
        - ``"cum.prob"`` — cumulative probabilities ``P(Y ≤ j | x)``
          for ``j = 0, ..., J - 2``. Multi-row output.
        - ``"exc.prob"`` — exceedance probabilities
          ``P(Y > j | x) = 1 - P(Y ≤ j | x)``. Multi-row.
        - ``"prob"`` (default) — per-category probabilities
          ``P(Y = j | x)`` for all J categories. Multi-row.
        - ``"mean.class"`` — expected category index
          ``E[Y | x] = Σ_j j · P(Y = j | x)``. Single-row.

    by, at, level, **kwargs
        Forwarded to :func:`emmeans` for the latent-mode call that
        builds the reference grid.

    Returns
    -------
    EMMResult
        With a tidy ``frame`` containing the response-scale point
        estimates, delta-method SEs, df (from the underlying
        latent EMM), and Wald confidence intervals. Multi-row
        modes (``prob``, ``cum.prob``, ``exc.prob``) add a
        ``category`` column.

    Notes
    -----
    - The SE is computed via the delta method through the
      cumulative-link transformation, accounting for the
      covariance between the structural ``β`` and the cumulative
      thresholds ``τ``.
    - For ``mode="prob"``, the per-category probabilities sum to
      1.0 across categories at each reference cell — verify
      empirically with ``frame.groupby(group_cols).emmean.sum()``.
    """
    if mode == "latent":
        # No transformation — just defer to standard emmeans.
        return emmeans(fit, specs, by=by, at=at, level=level, **kwargs)
    if mode not in ("cum.prob", "exc.prob", "prob", "mean.class"):
        raise ValueError(
            f"ordinal_emmeans: unknown mode {mode!r}. Valid modes: "
            "'latent', 'cum.prob', 'exc.prob', 'prob', 'mean.class'."
        )

    # 1) Latent EMM gives us the reference grid + L matrix.
    em_latent, info = _broadcast_to_reference_grid(
        fit, specs, by=by, at=at, level=level, **kwargs
    )
    L = em_latent.linfct # (n_cells, p)
    eta = L @ info.beta # (n_cells,)

    # 2) Ordinal-specific state from the fit.
    state = _extract_ordinal_state(fit)
    beta = state["beta"]
    tau = state["tau"]
    cov_full = state["cov_full"]
    cdf, pdf = _link_cdf_pdf(state["link"])
    categories = state["categories"]
    n_cells = L.shape[0]
    n_thresh = len(tau)
    J = n_thresh + 1
    p = len(beta)

    # removed a runtime
    # ``np.testing.assert_allclose(eta, L @ beta, atol=1e-10)``
    # that fired on every call — the two values are identical by
    # construction (both come from the same fit's params slice),
    # so the check was test-only waste on the hot path.

    # 3) Per-cell cumulative probabilities and link densities.
    # P_cum[c, j] = F(τ_j - η_c)
    # pdf[c, j] = f(τ_j - η_c)
    eta_c = eta.reshape(-1, 1) # (n_cells, 1)
    tau_j = tau.reshape(1, -1) # (1, J - 1)
    arg = tau_j - eta_c # (n_cells, J - 1)
    P_cum = cdf(arg) # (n_cells, J - 1)
    f_arg = pdf(arg) # (n_cells, J - 1)

    # 4) Per-mode point estimates.
    if mode == "cum.prob":
        # Output: n_cells × n_thresh rows, one per (cell, j).
        point = P_cum # (n_cells, J - 1)
    elif mode == "exc.prob":
        point = 1.0 - P_cum
    elif mode == "prob":
        # Differences of adjacent cum-probs, with sentinels at 0/1.
        # P_y_eq_0 = P_cum[:, 0]
        # P_y_eq_j = P_cum[:, j] - P_cum[:, j-1] for 1 <= j < J-1
        # P_y_eq_{J-1} = 1 - P_cum[:, -1]
        P0 = P_cum[:, [0]]
        Pmid = np.diff(P_cum, axis=1)
        Plast = 1.0 - P_cum[:, [-1]]
        point = np.concatenate([P0, Pmid, Plast], axis=1) # (n_cells, J)
    elif mode == "mean.class":
        # R `emmeans` convention: categories are RANKED 1..J, so
        # E[Y_rank] = Σ_{j=1}^{J} j · P(Y_rank = j)
        # = J - Σ_{j=1}^{J-1} P(Y_rank ≤ j)
        # = J - Σ_j P_cum[:, j]
        # (This is the R-emmeans default for ``mode="mean.class"``;
        # pymmeans matches the 1-indexed convention so direct
        # comparisons to R don't drift by a constant offset.)
        point = J - P_cum.sum(axis=1) # (n_cells,)

    # 5) Delta-method SEs.
    # For each output element g(β, τ), Var(g) = grad' cov_full grad
    # where grad is the (p + J - 1)-vector of partial derivatives.
    #
    # ∂P_cum[c, j]/∂β = -f_arg[c, j] · L_c (sign: arg = τ - L β)
    # ∂P_cum[c, j]/∂τ_j = f_arg[c, j]
    # ∂P_cum[c, j]/∂τ_i = 0 (i ≠ j)
    se: np.ndarray
    if mode in ("cum.prob", "exc.prob"):
        # cum and exc differ only by sign; SE is identical.
        var = np.zeros((n_cells, n_thresh))
        for c in range(n_cells):
            for j in range(n_thresh):
                grad = np.zeros(p + n_thresh)
                grad[:p] = -f_arg[c, j] * L[c]
                grad[p + j] = f_arg[c, j]
                var[c, j] = float(grad @ cov_full @ grad)
        se = np.sqrt(np.clip(var, 0.0, None))
        point_out = point
    elif mode == "prob":
        var = np.zeros((n_cells, J))
        for c in range(n_cells):
            for k in range(J):
                grad = np.zeros(p + n_thresh)
                # k=0: P = F(τ_0 - η) → only τ_0 and β contribute
                # k=J-1: P = 1 - F(τ_{J-2} - η)
                # 1≤k≤J-2: P = F(τ_k - η) - F(τ_{k-1} - η)
                if k == 0:
                    grad[:p] = -f_arg[c, 0] * L[c]
                    grad[p + 0] = f_arg[c, 0]
                elif k == J - 1:
                    grad[:p] = f_arg[c, -1] * L[c]
                    grad[p + (J - 2)] = -f_arg[c, -1]
                else:
                    grad[:p] = (f_arg[c, k] - f_arg[c, k - 1]) * (-L[c])
                    grad[p + k] = f_arg[c, k]
                    grad[p + (k - 1)] = -f_arg[c, k - 1]
                var[c, k] = float(grad @ cov_full @ grad)
        se = np.sqrt(np.clip(var, 0.0, None))
        point_out = point
    elif mode == "mean.class":
        # mean.class = J - Σ_j P_cum[:, j] (R 1-indexed rank convention;
        # see the point-estimate block above). the
        # old comment said ``(J - 1) - Σ ...`` which was the *0-indexed*
        # formula — the code itself uses the R convention correctly.
        # ∂/∂β = +Σ_j f_arg[c, j] · L_c (sign from arg=τ-Lβ)
        # ∂/∂τ_j = -f_arg[c, j]
        var = np.zeros(n_cells)
        for c in range(n_cells):
            grad = np.zeros(p + n_thresh)
            grad[:p] = f_arg[c, :].sum() * L[c]
            grad[p:] = -f_arg[c, :]
            var[c] = float(grad @ cov_full @ grad)
        se = np.sqrt(np.clip(var, 0.0, None))
        point_out = point

    # 6) Assemble the output frame.
    df_resid = em_latent.frame["df"].to_numpy()
    from scipy import stats
    if mode in ("cum.prob", "exc.prob", "prob"):
        # Wide → long: replicate the latent grid rows, add a category col.
        groups_frame = em_latent.frame.drop(
            columns=[c for c in ("emmean", "SE", "df", "lower_cl", "upper_cl")
                     if c in em_latent.frame.columns]
        )
        n_out_per_cell = point_out.shape[1]
        rep_groups = groups_frame.loc[
            groups_frame.index.repeat(n_out_per_cell)
        ].reset_index(drop=True)
        flat_point = point_out.reshape(-1)
        flat_se = se.reshape(-1)
        rep_df = np.repeat(df_resid, n_out_per_cell)
        # Categories list: for prob we have J entries; for cum/exc we
        # have J-1 (the threshold positions, labelled with the
        # category at or below).
        if mode == "prob":
            cat_labels = np.tile(categories, n_cells)
        else:
            # Use string labels "P(Y ≤ <category>)" for cum, mirror for exc
            cat_labels = np.tile(categories[:-1], n_cells)
        crit = stats.t.ppf(1.0 - (1.0 - level) / 2.0, rep_df)
        out = pd.DataFrame(
            {
                **{
                    col: rep_groups[col].to_numpy()
                    for col in rep_groups.columns
                },
                "category": cat_labels,
                "emmean": flat_point,
                "SE": flat_se,
                "df": rep_df,
                "lower_cl": flat_point - crit * flat_se,
                "upper_cl": flat_point + crit * flat_se,
            }
        )
    else: # mean.class
        crit = stats.t.ppf(1.0 - (1.0 - level) / 2.0, df_resid)
        out = em_latent.frame.copy()
        out["emmean"] = point_out
        out["SE"] = se
        out["lower_cl"] = point_out - crit * se
        out["upper_cl"] = point_out + crit * se

    # 7) Wrap in an EMMResult that downstream pairs/contrast can
    # consume. Reuse the latent EMM's linfct (the categories
    # column is informational only — pairs() of probabilities
    # is well-defined per category).
    import dataclasses
    new_em = dataclasses.replace(
        em_latent,
        frame=out,
        type=mode,
    )
    return new_em
