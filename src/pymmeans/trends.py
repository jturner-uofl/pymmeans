"""Estimated marginal trends — slopes of the linear predictor w.r.t. a numeric.

The slope at a grid point r is

    d eta / d x = (L(x + h, r) - L(x - h, r)) / (2 h) @ beta

Because v0.1 only supports plain identifier numerics on the RHS, the model
matrix is linear in ``x`` and central differences are exact (to
floating-point precision) for any reasonable ``h``. Standard errors follow
the same ``L_slope @ V @ L_slope.T`` machinery as ``emmeans``.

For response-scale trends (``type="response"``), the slope is transformed
via the inverse link's derivative: ``(d mu / d x) = h'(eta) * (d eta / d x)``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from patsy import build_design_matrices
from scipy import stats

from pymmeans.emmeans import EMMResult, _as_list
from pymmeans.ref_grid import build_grid_spec
from pymmeans.utils import ModelInfo, from_fitted

_DEFAULT_H = 1e-5


def emtrends(
    model: Any,
    specs: str | list[str] | None,
    var: str,
    by: str | list[str] | None = None,
    at: dict[str, Any] | None = None,
    level: float = 0.95,
    type: str = "link",
    response_derivative: bool = False,
    h: float = _DEFAULT_H,
    delta_var: float | None = None,
    max_degree: int = 1,
) -> EMMResult:
    """Estimated marginal trends: slopes of ``var`` at each level combo of ``specs``.

    Parameters
    ----------
    model
        Fitted statsmodels OLS/GLM result or a ``ModelInfo``.
    specs
        Factor name(s) whose levels stratify the slopes. ``None`` returns a
        single overall slope.
    var
        Name of the numeric covariate whose slope to estimate. Must be a
        plain column reference (v0.1 limitation).
    by, at, level, type
        Same as ``emmeans()``. **c**: for
        trends, ``type='response'`` is IGNORED by default — matching
        R `emtrends`, which always returns link-scale slopes. The
        chain-rule response-scale slope (``h'(eta) * (d eta / d x)``)
        is opt-in via ``response_derivative=True`` (see below).
    response_derivative
        If True, apply the inverse-link chain rule and return the
        response-scale slope ``h'(eta) * (d eta / d x)``. Default
        False to match R `emtrends`. This is a deliberate pymmeans
        extension — useful when you genuinely want the slope on the
        response scale (e.g. probability per unit change for a
        Binomial logit). R users porting code should leave this False.
    h
        Finite-difference step. Default 1e-5 is exact-ish for plain linear
        models; reduce if you see noise at very large grids.
    delta_var
        R-parity alias for ``h`` (R `emtrends` uses ``delta.var``). If
        supplied, overrides ``h``. When ``var`` has only one observed
        value, ``h`` defaulting silently to ``1e-5`` produces a
        spurious slope. R raises "Provide a nonzero value of
        'delta.var'" in that case; we mirror R and refuse with the
        same message unless ``delta_var`` (or ``h``) was explicitly
        supplied.

    Returns
    -------
    EMMResult
        With ``<var>.trend`` reinterpreted as the slope of ``var``.
    """
    from pymmeans.emmeans import _validate_level

    level = _validate_level(level)
    info = model if isinstance(model, ModelInfo) else from_fitted(model)

    if var not in info.numeric_means:
        raise ValueError(
            f"'{var}' is not a numeric covariate in the model. "
            f"Known numerics: {sorted(info.numeric_means)}."
        )

    # Refuse a degenerate numeric predictor (zero range) unless the
    # user explicitly supplied ``delta_var=`` (or a custom ``h``).
    # R raises the exact text ``"Provide a nonzero value of
    # 'delta.var'"`` in this case (see emmeans/R/emtrends.R), and
    # silently returning a slope from a vacuous finite-difference
    # would mislead users. We use a structurally separated try-block
    # (rather than a string-match catch). Numeric coercion errors are
    # caught and ignored (let downstream code raise its own error);
    # the deliberately-raised "Provide a nonzero value" ValueError is
    # raised AFTER the try-block, so it can never be conflated with
    # a coercion error and the catch never has to inspect message
    # text.
    user_specified_h = delta_var is not None or h != _DEFAULT_H
    if not user_specified_h and getattr(info, "data", None) is not None \
            and var in info.data.columns:
        finite = None
        try:
            vals = np.asarray(info.data[var], dtype=float)
            finite = vals[np.isfinite(vals)]
        except (TypeError, ValueError):
            # Non-numeric coercion — let downstream code raise.
            finite = None
        if finite is not None and finite.size > 0 \
                and float(finite.max()) == float(finite.min()):
            raise ValueError("Provide a nonzero value of 'delta_var'")
    if delta_var is not None:
        h = float(delta_var)
    type = type.lower()
    if type not in ("link", "response"):
        raise ValueError(f"'type' must be 'link' or 'response', got {type!r}.")

    target = _as_list(specs)
    by_list = _as_list(by)

    for name in target + by_list:
        if name not in info.factors:
            raise ValueError(
                f"'{name}' is not a categorical factor in the model. "
                f"Known factors: {sorted(info.factors)}."
            )

    spec = build_grid_spec(info, at)
    cols = list(spec.keys())

    # Build the centered grid (eager, since trend grids are typically small)
    import itertools

    combos = list(itertools.product(*[spec[c] for c in cols]))
    grid_center = pd.DataFrame(combos, columns=cols)
    for name, levels in info.factors.items():
        if name in grid_center.columns:
            grid_center[name] = pd.Categorical(grid_center[name], categories=levels)

    grid_plus = grid_center.copy()
    grid_minus = grid_center.copy()
    grid_plus[var] = grid_plus[var].astype(float) + h
    grid_minus[var] = grid_minus[var].astype(float) - h

    [L_plus] = build_design_matrices([info.design_info], grid_plus, return_type="matrix")
    [L_minus] = build_design_matrices(
        [info.design_info], grid_minus, return_type="matrix"
    )
    L_slope = (np.asarray(L_plus) - np.asarray(L_minus)) / (2.0 * h)

    # Marginalize L_slope over non-(target+by) variables, like emmeans
    group_cols = target + by_list
    if group_cols:
        keys = pd.MultiIndex.from_frame(grid_center[group_cols].astype(object))
        unique_keys = keys.unique()
        L_marg = np.empty((len(unique_keys), info.n_params))
        for i, key in enumerate(unique_keys):
            mask = keys == key
            L_marg[i] = L_slope[mask].mean(axis=0)
    else:
        keys = None
        unique_keys = pd.MultiIndex.from_tuples([("overall",)], names=["term"])
        L_marg = L_slope.mean(axis=0, keepdims=True)
        group_cols = ["term"]

    # Higher-order polynomial trends (R `emtrends(..., max.degree=k)`):
    # the degree-d trend is the d-th derivative of the linear predictor
    # divided by d! (the Taylor / polynomial-coefficient convention, so a
    # raw y = b2 x^2 fit reports quadratic = b2, not 2 b2). Isolated in a
    # helper so the default (single-degree) path is untouched.
    if max_degree != 1:
        if max_degree < 1 or max_degree > 4:
            raise ValueError(f"max_degree must be in 1..4; got {max_degree}.")
        if response_derivative and info.family is not None:
            raise NotImplementedError(
                "response_derivative=True is supported only with max_degree=1; "
                "higher-order response-scale trends need the inverse link's "
                "higher derivatives."
            )
        return _emtrends_multidegree(
            info, grid_center, var, group_cols, keys, unique_keys,
            target, by_list, level, max_degree,
        )

    mu = L_marg @ info.beta
    var_mu = np.einsum("ij,jk,ik->i", L_marg, info.vcov, L_marg)
    se = np.sqrt(np.clip(var_mu, 0.0, None))

    # Even when the user supplies ``delta_var=`` to bypass the
    # zero-range guard, the linfct slope
    # row may still be non-estimable (e.g. ``y ~ g * x`` with x
    # constant — the g:x interaction columns are in the model but
    # carry no data, so statsmodels' pseudoinverse returns the
    # garbage least-norm coefficient). Apply the standard
    # estimability check and NaN the trend / SE for any row not in
    # the row-space of the design matrix. R returns ``nonEst`` in
    # exactly this case (R/estim.R / R/emtrends.R).
    from pymmeans.estimability import (
        estimable_mask,
        estimable_mask_from_basis,
    )
    est_mask = None
    X_design = None
    if info.raw_result is not None and hasattr(info.raw_result, "model"):
        X_design = np.asarray(getattr(info.raw_result.model, "exog", None))
    if (
        X_design is not None
        and X_design.ndim == 2
        and X_design.shape[1] == info.n_params
    ):
        est_mask = estimable_mask(L_marg, X_design)
    elif info.estimability_basis is not None:
        est_mask = estimable_mask_from_basis(L_marg, info.estimability_basis)
    if est_mask is not None and not est_mask.all():
        mu = np.where(est_mask, mu, np.nan)
        se = np.where(est_mask, se, np.nan)

    df_value: float = (
        np.inf if (info.family is not None or info.is_mixed) else info.df_resid
    )
    alpha = 1.0 - level
    crit = float(stats.t.ppf(1.0 - alpha / 2.0, df_value))
    lower = mu - crit * se
    upper = mu + crit * se

    # c: R `emtrends` ignores `type='response'` and always
    # returns link-scale slopes. The chain-rule response-scale slope
    # `h'(eta) * (d eta / d x)` is a pymmeans extension, opt-in via
    # `response_derivative=True`. Default off so the R workflow ports
    # unchanged.
    if response_derivative and info.family is not None:
        # Response-scale trend = h'(eta) * (d eta / d x), evaluated at the
        # corresponding emmeans eta. Use eta from the centered grid.
        [L_center] = build_design_matrices(
            [info.design_info], grid_center, return_type="matrix"
        )
        L_center_arr = np.asarray(L_center)
        # Marginalize L_center too
        if group_cols != ["term"]:
            L_eta = np.empty((len(unique_keys), info.n_params))
            for i, key in enumerate(unique_keys):
                mask = keys == key
                L_eta[i] = L_center_arr[mask].mean(axis=0)
        else:
            L_eta = L_center_arr.mean(axis=0, keepdims=True)
        eta = L_eta @ info.beta
        link = info.family.link
        d_h = link.inverse_deriv(eta)
        mu = d_h * mu
        se = np.abs(d_h) * se
        lower = mu - crit * se # symmetric Wald on response-scale trend
        upper = mu + crit * se

    frame_data: dict[str, Any] = {}
    for i, col in enumerate(group_cols):
        frame_data[col] = [k[i] for k in unique_keys]
    frame = pd.DataFrame(frame_data)
    frame[f"{var}.trend"] = mu
    frame["SE"] = se
    frame["df"] = np.full(len(unique_keys), df_value)
    frame["lower_cl"] = lower
    frame["upper_cl"] = upper

    # c: stamp the result type based on whether the
    # response-scale chain rule actually fired. `type='response' +
    # response_derivative=False` is the R-parity case (link slope is
    # returned despite the user's `type=` request) — labelling it as
    # response would mislead downstream summary calls. Mirror R's
    # convention and report `type='link'` whenever the slope is on
    # the link scale.
    result_type = "response" if (response_derivative and info.family is not None) else "link"
    return EMMResult(
        frame=frame,
        linfct=L_marg,
        model_info=info,
        target=target if target else ["term"],
        by=by_list,
        level=level,
        type=result_type,
    )


# Central finite-difference stencils for the d-th derivative (offset ->
# weight), and degree labels matching R `emtrends`.
_FD_STENCILS: dict[int, dict[int, float]] = {
    1: {-1: -0.5, 0: 0.0, 1: 0.5},
    2: {-1: 1.0, 0: -2.0, 1: 1.0},
    3: {-2: -0.5, -1: 1.0, 0: 0.0, 1: -1.0, 2: 0.5},
    4: {-2: 1.0, -1: -4.0, 0: 6.0, 1: -4.0, 2: 1.0},
}
_DEGREE_LABELS = {1: "linear", 2: "quadratic", 3: "cubic", 4: "quartic"}


def _emtrends_multidegree(
    info: ModelInfo,
    grid_center: pd.DataFrame,
    var: str,
    group_cols: list,
    keys: Any,
    unique_keys: Any,
    target: list,
    by_list: list,
    level: float,
    max_degree: int,
) -> EMMResult:
    """Stack the degree-1..k trends for ``emtrends(..., max_degree=k)``.

    The degree-d trend is ``(d-th derivative of X beta) / d!``. Derivatives
    use a central finite-difference stencil with a moderate step (exact for
    a polynomial design of degree <= the stencil order, regardless of the
    step); the result carries an extra ``degree`` column and matches R's
    polynomial-coefficient convention.
    """
    import math

    from pymmeans.estimability import estimable_mask, estimable_mask_from_basis

    step = 1e-3  # robust for higher-order FD; exact on polynomial designs
    offsets = sorted({j for d in range(1, max_degree + 1) for j in _FD_STENCILS[d]})
    designs: dict[int, np.ndarray] = {}
    for j in offsets:
        g = grid_center.copy()
        g[var] = g[var].astype(float) + j * step
        [des] = build_design_matrices([info.design_info], g, return_type="matrix")
        designs[j] = np.asarray(des)

    df_value: float = (
        np.inf if (info.family is not None or info.is_mixed) else info.df_resid
    )
    crit = float(stats.t.ppf(1.0 - (1.0 - level) / 2.0, df_value))

    X_design = None
    if info.raw_result is not None and hasattr(info.raw_result, "model"):
        X_design = np.asarray(getattr(info.raw_result.model, "exog", None))

    frames: list[pd.DataFrame] = []
    linfcts: list[np.ndarray] = []
    for d in range(1, max_degree + 1):
        sten = _FD_STENCILS[d]
        L_d = sum(w * designs[j] for j, w in sten.items()) / (step**d)
        L_d = L_d / math.factorial(d)
        if keys is not None:
            L_marg = np.empty((len(unique_keys), info.n_params))
            for i, key in enumerate(unique_keys):
                L_marg[i] = L_d[keys == key].mean(axis=0)
        else:
            L_marg = L_d.mean(axis=0, keepdims=True)

        mu = L_marg @ info.beta
        se = np.sqrt(
            np.clip(np.einsum("ij,jk,ik->i", L_marg, info.vcov, L_marg), 0.0, None)
        )
        est_mask = None
        if X_design is not None and X_design.ndim == 2 and X_design.shape[1] == info.n_params:
            est_mask = estimable_mask(L_marg, X_design)
        elif info.estimability_basis is not None:
            est_mask = estimable_mask_from_basis(L_marg, info.estimability_basis)
        if est_mask is not None and not est_mask.all():
            mu = np.where(est_mask, mu, np.nan)
            se = np.where(est_mask, se, np.nan)

        fd: dict[str, Any] = {
            "degree": [_DEGREE_LABELS[d]] * len(unique_keys),
        }
        for i, col in enumerate(group_cols):
            fd[col] = [k[i] for k in unique_keys]
        fr = pd.DataFrame(fd)
        fr[f"{var}.trend"] = mu
        fr["SE"] = se
        fr["df"] = np.full(len(unique_keys), df_value)
        fr["lower_cl"] = mu - crit * se
        fr["upper_cl"] = mu + crit * se
        frames.append(fr)
        linfcts.append(L_marg)

    frame = pd.concat(frames, ignore_index=True)
    return EMMResult(
        frame=frame,
        linfct=np.vstack(linfcts),
        model_info=info,
        target=target if target else ["term"],
        by=["degree", *by_list],
        level=level,
        type="link",
    )
