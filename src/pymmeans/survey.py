"""Survey-weighted inference for EMMs.

Complex sample designs (probability weights, stratification, clustered
PSUs) produce **design-corrected** variance estimates that differ from
the model-based variance returned by an ordinary OLS / WLS / GLM fit.
R's ``survey::svyglm`` is the canonical reference; this module
implements the equivalent Taylor linearisation in Python.

For a (W)LS fit ``y ~ X`` with sampling weights ``w_i``, the
design-corrected vcov is

    V_design = (X' W X)^-1 * Omega * (X' W X)^-1

where ``W = diag(w_i)`` and ``Omega`` is the variance of the weighted
score contributions ``s_i = w_i * x_i * resid_i``:

- **Simple random sampling (no strata, no clusters):**
  ``Omega = (n / (n - 1)) * sum_i (s_i - bar_s)(s_i - bar_s)'``.
  R `survey` treats SRS as one stratum with ``n`` PSUs and applies
  the ``n / (n - 1)`` finite-sample correction; the centred form
  here matches that (Lumley 2010 eq. 2.7). At the MLE, ``bar_s = 0``
  by the first-order condition so the centred and uncentred forms
  agree numerically; the centring is robust to imperfect
  optimiser convergence.
- **Stratified sampling, no clusters:**
  ``Omega = sum_h (n_h / (n_h - 1)) * sum_{i in h} (s_i - bar_s_h)(s_i - bar_s_h)'``
- **Stratified, clustered:**
  The PSU (primary sampling unit) is the aggregation level for the
  outer-product sum. Score contributions are first aggregated to the
  PSU level, then the between-PSU variance is computed (per stratum).

This MVP supports the first two cases (no clusters or clusters
within strata). FPC (finite population correction) is not yet wired
in. For features beyond, fit with ``svyglm`` in R and pass the
design-corrected vcov directly.

References
----------
- Binder, D. A. (1983). "On the variances of asymptotically normal
  estimators from complex surveys." *International Statistical Review*
  51(3), 279-292.
- Lumley, T. (2010). *Complex Surveys: A Guide to Analysis Using R*.
  Wiley.
- R ``survey`` package documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pymmeans.utils import (
    ModelInfo,
    _build_estimability_basis,
    _extract_offset,
)


@dataclass(frozen=True)
class SurveyDesign:
    """A complex-sample design specification.

    Parameters
    ----------
    weights
        Per-observation sampling weights (probability of inclusion).
        Inverse-probability weights; ``w_i = 1 / pi_i`` where ``pi_i`` is
        the probability that unit ``i`` was included in the sample.
    strata
        Optional stratum labels (any hashable type). Stratification
        produces independent within-stratum variance contributions.
    cluster
        Optional PSU (primary sampling unit) labels within each stratum.
        Score contributions are aggregated to the PSU level before the
        outer-product sum.
    """

    weights: np.ndarray
    strata: np.ndarray | None = None
    cluster: np.ndarray | None = None

    def __post_init__(self) -> None:
        # cast errors (pd.NA, complex types, strings) used
        # to surface as raw TypeError before the finiteness check ran.
        # Wrap the cast so every invalid-input shape goes through one
        # ValueError with a consistent message.
        try:
            w = np.asarray(self.weights, dtype=float).ravel()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "SurveyDesign.weights must be a sequence of finite real "
                f"numbers; got {type(self.weights).__name__} that does "
                f"not cast cleanly to float64 ({exc})."
            ) from exc
        if w.size == 0:
            raise ValueError("SurveyDesign.weights must be non-empty.")
        # #5: NaN comparisons return False so (w<=0) misses
        # both NaN and +inf; reject them explicitly.
        if not np.isfinite(w).all():
            raise ValueError(
                "SurveyDesign.weights must be finite (no NaN or inf); "
                f"got n_invalid = {int((~np.isfinite(w)).sum())}."
            )
        if (w <= 0).any():
            raise ValueError(
                "SurveyDesign.weights must be strictly positive (got "
                f"{w.min():g})."
            )
        object.__setattr__(self, "weights", w)
        if self.strata is not None:
            s = np.asarray(self.strata).ravel()
            if s.size != w.size:
                raise ValueError(
                    f"strata length {s.size} != weights length {w.size}."
                )
            object.__setattr__(self, "strata", s)
        if self.cluster is not None:
            c = np.asarray(self.cluster).ravel()
            if c.size != w.size:
                raise ValueError(
                    f"cluster length {c.size} != weights length {w.size}."
                )
            object.__setattr__(self, "cluster", c)


def design_corrected_vcov(
    X: np.ndarray,
    residuals: np.ndarray,
    design: SurveyDesign,
    irls_weights: np.ndarray | None = None,
    score_factor: np.ndarray | None = None,
) -> np.ndarray:
    """Taylor-linearisation design-corrected vcov for a (W)LS / GLM fit.

    Parameters
    ----------
    X
        Design matrix, shape ``(n, p)``.
    residuals
        Response residuals ``y - mu``, shape ``(n,)``. For OLS this is
        ``y - X @ beta``; for GLM this is ``y - mu`` (NOT
        ``resid_working``). #1 caught the previous code
        using working residuals + an OLS bread, which gave wrong GLM
        SEs.
    design
        The :class:`SurveyDesign` describing weights / strata / clusters.
    irls_weights
        Per-observation IRLS weights ``w_i = (dmu_i/deta_i)^2 / V(mu_i)``
        for GLM. ``None`` (default) means OLS / identity link
        (``irls_weights = 1`` everywhere).
    score_factor
        Per-observation score factor ``f_i = (dmu_i/deta_i) / V(mu_i)``
        for GLM; the score is ``f_i * (y_i - mu_i) * x_i``. For
        canonical-link families (Poisson-log, Binomial-logit,
        Gaussian-identity) this collapses to 1. For non-canonical
        links (e.g. Gamma-log) it is ``1/mu``.

    Returns
    -------
    ndarray
        The design-corrected vcov matrix, shape ``(p, p)``.

    Notes
    -----
    The full GLM sandwich is

        V = I_w^-1 * Cov(score) * I_w^-1

    where ``I_w = X' diag(sw * irls) X`` is the design-weighted GLM
    information and ``score_i = sw_i * (y_i - mu_i) * score_factor_i *
    x_i``. For OLS / identity link both ``irls`` and ``score_factor``
    are 1, so the formula reduces to the OLS sandwich. Matches R
    ``survey::svyglm`` at the per-family tolerances asserted in
    ``tests/test_r_benchmark.py`` (1e-7 Gaussian, 1e-6 Poisson /
    Binomial logit, 1e-5 Gamma log; the Gamma-log path has slightly
    more rounding because of the non-canonical link's score factor).
    """
    X = np.asarray(X, dtype=float)
    resid = np.asarray(residuals, dtype=float).ravel()
    w = design.weights
    if X.shape[0] != resid.size or resid.size != w.size:
        raise ValueError(
            f"X rows ({X.shape[0]}), residuals ({resid.size}), and "
            f"weights ({w.size}) must agree."
        )
    n, p = X.shape

    # #5: validate finite-ness on every numeric input.
    # SurveyDesign.weights are already validated; X, residuals, and the
    # optional IRLS / score-factor arrays were not. Non-finite values
    # used to produce all-NaN vcov silently.
    def _require_finite(name: str, arr: np.ndarray) -> None:
        if not np.isfinite(arr).all():
            raise ValueError(
                f"design_corrected_vcov: {name} must contain only finite "
                f"values; n_invalid = {int((~np.isfinite(arr)).sum())}."
            )
    _require_finite("X", X)
    _require_finite("residuals", resid)

    # Bread for the GLM information sandwich: I_w = X' diag(sw * irls) X
    irls = (
        np.ones(n) if irls_weights is None
        else np.asarray(irls_weights, dtype=float).ravel()
    )
    if irls.size != n:
        raise ValueError(
            f"irls_weights length ({irls.size}) must equal n ({n})."
        )
    _require_finite("irls_weights", irls)
    bread = np.linalg.inv(X.T @ ((w * irls)[:, None] * X))

    # Score factor for non-canonical-link GLMs; 1 elsewhere.
    sf = (
        np.ones(n) if score_factor is None
        else np.asarray(score_factor, dtype=float).ravel()
    )
    if sf.size != n:
        raise ValueError(
            f"score_factor length ({sf.size}) must equal n ({n})."
        )
    _require_finite("score_factor", sf)

    # Score contributions s_i = sw_i * (y - mu) * sf_i * x_i (shape n x p)
    S = (w * resid * sf)[:, None] * X

    # #9: np.unique sorts heterogeneous object arrays, which
    # fails on strata = ["1", 2, "1", 2]. pd.factorize handles mixed
    # types (and is also faster on string labels).
    def _label_codes(arr: np.ndarray) -> tuple[np.ndarray, int]:
        codes, _ = pd.factorize(arr, sort=False)
        return codes, codes.max() + 1 if codes.size else 0

    if design.cluster is not None:
        # Aggregate scores to PSU level. Each PSU contributes one
        # vector to the outer-product sum.
        if design.strata is None:
            # Treat all clusters as a single stratum.
            psu_codes, _ = _label_codes(design.cluster)
            stratum_codes = np.zeros(n, dtype=int)
            n_strata = 1
        else:
            psu_codes, _ = _label_codes(design.cluster)
            stratum_codes, n_strata = _label_codes(design.strata)
        omega = np.zeros((p, p))
        for h in range(n_strata):
            h_mask = stratum_codes == h
            psus_in_h = np.unique(psu_codes[h_mask])
            n_h = len(psus_in_h)
            if n_h < 2:
                continue # Single PSU stratum contributes 0 by convention.
            psu_sums = np.zeros((n_h, p))
            for i, c in enumerate(psus_in_h):
                c_mask = h_mask & (psu_codes == c)
                psu_sums[i] = S[c_mask].sum(axis=0)
            mean_psu = psu_sums.mean(axis=0, keepdims=True)
            dev = psu_sums - mean_psu
            omega += (n_h / (n_h - 1.0)) * (dev.T @ dev)
    elif design.strata is not None:
        # Stratified, no clusters: each observation is its own PSU.
        stratum_codes, n_strata = _label_codes(design.strata)
        omega = np.zeros((p, p))
        for h in range(n_strata):
            h_mask = stratum_codes == h
            n_h = int(h_mask.sum())
            if n_h < 2:
                continue
            S_h = S[h_mask]
            mean_h = S_h.mean(axis=0, keepdims=True)
            dev = S_h - mean_h
            omega += (n_h / (n_h - 1.0)) * (dev.T @ dev)
    else:
        # Simple random sampling with weights: R's survey package treats
        # this as 1 stratum with n PSUs and applies the n/(n-1) finite-
        # sample correction (Lumley 2010, eqn 2.7).
        mean_s = S.mean(axis=0, keepdims=True)
        dev = S - mean_s
        n_eff = n
        omega = np.zeros((p, p)) if n_eff < 2 else n_eff / (n_eff - 1.0) * (dev.T @ dev)

    return bread @ omega @ bread


def from_survey(
    result: Any,
    design: SurveyDesign,
    data: pd.DataFrame | None = None,
) -> ModelInfo:
    """Build a survey-weighted ModelInfo from a fitted statsmodels result.

    Wraps an ordinary statsmodels OLS / WLS / GLM fit so that emmeans
    and downstream contrasts use the **design-corrected** vcov from
    :func:`design_corrected_vcov` in place of the model-based vcov.
    All other ModelInfo plumbing (offsets, factor metadata,
    estimability basis, etc.) is preserved.

    Parameters
    ----------
    result
        Fitted statsmodels result (OLS, WLS, GLM). For best results,
        fit with the same `weights=` you pass to the design; pymmeans
        will refit-correct internally when needed.
    design
        The :class:`SurveyDesign`.
    data
        Optional original DataFrame. Used by linearmodels-style fits;
        statsmodels usually has it on `result.model.data.frame`.

    Returns
    -------
    ModelInfo
        With ``vcov`` replaced by the design-corrected version and
        ``fit_weights`` set to ``design.weights`` so the
        ``weights="outer"`` / ``"proportional"`` paths use the design
        weights for marginal-mean averaging.
    """
    model = getattr(result, "model", None)
    if model is None:
        raise TypeError(
            "from_survey requires a statsmodels result with .model. Got "
            f"{type(result).__name__}."
        )

    design_info = getattr(getattr(model, "data", None), "design_info", None)
    if design_info is None:
        raise ValueError(
            "from_survey requires a model fit via the formula API "
            "(smf.ols / smf.wls / smf.glm)."
        )
    frame = getattr(model.data, "frame", None)
    if frame is None and data is None:
        raise ValueError(
            "from_survey could not recover the training DataFrame. Pass "
            "`data=` explicitly."
        )
    if frame is None:
        frame = data

    beta = np.asarray(result.params, dtype=float)
    param_names = (
        list(result.params.index)
        if hasattr(result.params, "index")
        else list(model.exog_names)
    )

    X = np.asarray(model.exog, dtype=float)
    y = np.asarray(model.endog, dtype=float)
    if X.shape[0] != design.weights.size:
        raise ValueError(
            f"design.weights length ({design.weights.size}) does not "
            f"match the fit's n ({X.shape[0]})."
        )

    # #1: For GLM fits, the design-corrected sandwich needs
    # the GLM information matrix (NOT the OLS bread X'WX) and response
    # residuals (NOT resid_working). The previous code used resid_working
    # + OLS bread which gave ~15% wrong SEs vs R survey::svyglm.
    #
    # GLM sandwich:
    # I_w = sum_i sw_i * irls_i * x_i x_i' where irls = (dmu/deta)^2/V(mu)
    # s_i = sw_i * (y_i - mu_i) * f_i * x_i where f = (dmu/deta)/V(mu)
    # V = I_w^-1 @ Cov(score) @ I_w^-1
    family = getattr(model, "family", None)
    if family is not None:
        mu = np.asarray(result.predict(), dtype=float).ravel()
        resid = (y - mu).ravel()
        # IRLS weight per obs = 1/(link.deriv^2 * V(mu)) — exposed by
        # statsmodels family.weights(mu) for canonical and non-canonical
        # links alike.
        irls = np.asarray(family.weights(mu), dtype=float).ravel()
        # Score factor per obs = (dmu/deta) / V(mu) = irls * link.deriv(mu).
        # (For canonical links this collapses to 1.)
        link_deriv = np.asarray(family.link.deriv(mu), dtype=float).ravel()
        score_factor = irls * link_deriv
        vcov = design_corrected_vcov(
            X, resid, design,
            irls_weights=irls,
            score_factor=score_factor,
        )
    else:
        resid = (y - X @ beta).ravel()
        vcov = design_corrected_vcov(X, resid, design)

    # Build factor metadata (mirror from_statsmodels)
    factors: dict[str, list[str]] = {}
    numeric_means: dict[str, float] = {}
    multi_col_factors_dict: dict[str, list[str]] = {}
    frame_cols = set(frame.columns)
    aliases: dict[str, str] = {}
    from pymmeans.utils import _underlying_columns
    for factor, fi in design_info.factor_infos.items():
        name = factor.name()
        if fi.type == "categorical":
            factors[name] = list(fi.categories)
        elif name in frame.columns:
            numeric_means[name] = float(frame[name].mean())
        else:
            # support multi-col basis expressions
            # (``bs(x, df=3)``, ``cr(x, df=4)``, etc.) in the survey
            # adapter — matches ``from_statsmodels`` handling.
            standalone_width = getattr(fi, "num_columns", None) or 0
            handled_via_main = False
            for term in design_info.terms:
                if len(term.factors) == 1 and term.factors[0] is factor:
                    tslice = design_info.term_slices[term]
                    width = tslice.stop - tslice.start
                    if width == 1:
                        numeric_means[name] = float(X[:, tslice].mean())
                    elif width > 1:
                        underlying_cols = _underlying_columns(
                            factor.code, frame_cols
                        )
                        if underlying_cols:
                            multi_col_factors_dict[name] = underlying_cols
                            for col in underlying_cols:
                                if col in frame.columns:
                                    numeric_means.setdefault(
                                        col, float(frame[col].mean())
                                    )
                    handled_via_main = True
                    break
            if not handled_via_main and standalone_width > 1:
                underlying_cols = _underlying_columns(
                    factor.code, frame_cols
                )
                if underlying_cols:
                    multi_col_factors_dict[name] = underlying_cols
                    for col in underlying_cols:
                        if col in frame.columns:
                            numeric_means.setdefault(
                                col, float(frame[col].mean())
                            )
        if name not in frame_cols:
            underlying = _underlying_columns(factor.code, frame_cols)
            if len(underlying) == 1 and underlying[0] != name:
                aliases.setdefault(underlying[0], name)

    return ModelInfo(
        beta=beta,
        vcov=vcov,
        param_names=param_names,
        factors=factors,
        numeric_means=numeric_means,
        df_resid=float(result.df_resid),
        design_info=design_info,
        data=frame,
        response_name=model.endog_names,
        family=family,
        scale=float(getattr(result, "scale", 1.0) or 1.0),
        is_mixed=False,
        aliases=aliases,
        raw_result=result,
        offset_mean=_extract_offset(result, model),
        # Survey weights flow into the emmeans weighting machinery as
        # frequency-style weights — same machinery the WLS fit_weights
        # path uses, so `weights="proportional"` / `"outer"` average
        # the EMM over the survey-weighted training distribution.
        fit_weights=design.weights.copy(),
        estimability_basis=_build_estimability_basis(X),
        multi_col_factors=multi_col_factors_dict,
    )
