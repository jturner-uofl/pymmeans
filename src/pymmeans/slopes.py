"""Average marginal effects (slopes) over the observed sample.

:func:`emtrends` reports the slope of a numeric predictor at the cells
of a *balanced* reference grid. The econometrics / `marginaleffects`
tradition instead reports the **average marginal effect** (AME): the
slope evaluated at each *observed* row and then averaged over the
sample. The two coincide for an additive linear model but differ
whenever the slope depends on other covariates (interactions,
non-linear links).

This module provides that average-over-the-sample estimand.

* :func:`avg_slopes` returns the sample-averaged marginal effect of a
  numeric predictor, optionally within levels of a ``by`` factor, with
  a delta-method standard error.
* :func:`slopes` returns the per-observation marginal effects (one row
  per observation), each with its own delta-method standard error.

Both reuse the same finite-difference design-derivative machinery as
:func:`emtrends`: the derivative of the model's design row with
respect to ``var`` is

    L_slope(x) = ( X(x + h) - X(x - h) ) / (2 h),

so the link-scale marginal effect at a row is ``L_slope @ beta`` and
its variance is ``L_slope V L_slope^T`` --- exact, no approximation.
On the **link** scale the sample-averaged slope is therefore
``mean_i(L_slope_i) @ beta`` with an exact standard error. On the
**response** scale the per-row slope is ``h'(eta_i) * (L_slope_i @
beta)`` (chain rule through the inverse link), which is non-linear in
``beta``; its standard error is obtained by the delta method with a
finite-difference Jacobian.

A useful identity: for an additive linear model, ``avg_slopes(fit,
x)`` returns the OLS coefficient on ``x`` together with its standard
error, exactly.

References
----------
- Arel-Bundock, V. (2024). `marginaleffects`: predictions,
  comparisons, slopes, and hypothesis tests.
- Bartus, T. (2005). Estimation of marginal effects using margeff.
  *The Stata Journal*, 5(3), 309-329.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as _stats
from patsy import build_design_matrices

__all__ = ["SlopesResult", "avg_slopes", "slopes"]

_DEFAULT_H = 1e-5


@dataclass(frozen=True)
class SlopesResult:
    """Result of :func:`avg_slopes` or :func:`slopes`.

    Attributes
    ----------
    frame
        One row per reported slope (a single row for an overall AME,
        one per ``by`` level when grouped, or one per observation for
        :func:`slopes`). Columns: the ``by`` factor(s) when present,
        ``var``, ``slope``, ``SE``, ``df``, ``t_ratio``, ``p_value``,
        ``lower_cl``, ``upper_cl``.
    var
        Name of the numeric predictor.
    type
        ``"link"`` or ``"response"``.
    level
        Confidence level used for the interval.
    """

    frame: pd.DataFrame
    var: str
    type: str
    level: float


def _get_info(obj: Any) -> Any:
    info = getattr(obj, "model_info", None)
    if info is None:
        from pymmeans.utils import from_fitted

        info = from_fitted(obj)
    return info


def _require_reference_data(info: Any, fn: str) -> pd.DataFrame:
    """Return the model data, or raise a clear (pyfixest-aware) error."""
    data = getattr(info, "data", None)
    if data is not None and len(data) > 0:
        return data
    raw = getattr(info, "raw_result", None)
    if raw is not None and raw.__class__.__module__.split(".")[0] == "pyfixest":
        raise ValueError(
            f"{fn} is a reference-grid operation and is not supported on "
            "pyfixest fits: they absorb fixed effects and carry no patsy "
            "design/data. Use hypotheses(fit, g) for coefficient-level "
            "delta-method tests, or refit with statsmodels for marginal "
            "effects."
        )
    raise ValueError(f"{fn} requires the model's data on info.data.")


def _require_numeric_var(data: pd.DataFrame, var: str) -> None:
    """Reject a non-numeric slope variable with a clear message.

    Marginal effects are derivatives, so ``var`` must be continuous.
    Without this guard a categorical/string column fails deep inside
    patsy/pandas with an opaque cast error.
    """
    if not pd.api.types.is_numeric_dtype(data[var]):
        raise ValueError(
            f"slopes/avg_slopes require a numeric (continuous) variable; "
            f"{var!r} has dtype {data[var].dtype}. For the effect of a "
            f"categorical predictor, use emmeans(...) with contrast(...)."
        )


def _slope_design(info: Any, data: pd.DataFrame, var: str, h: float) -> np.ndarray:
    """Per-row derivative of the design matrix w.r.t. ``var`` (central diff)."""
    dp = data.copy()
    dm = data.copy()
    dp[var] = data[var].astype(float) + h
    dm[var] = data[var].astype(float) - h
    try:
        [Lp] = build_design_matrices([info.design_info], dp, return_type="matrix")
        [Lm] = build_design_matrices([info.design_info], dm, return_type="matrix")
    except Exception as exc:  # patsy raises bespoke, non-public error types
        raise ValueError(
            f"Could not build the perturbed design for {var!r}. This happens "
            f"when {var!r} enters through a term that is not differentiable by "
            f"a small numeric shift (e.g. a spline bs()/poly() basis whose "
            f"knots the perturbation crosses, or a categorical encoding). "
            f"Original error: {exc}"
        ) from exc
    return (np.asarray(Lp) - np.asarray(Lm)) / (2.0 * h)


def _center_design(info: Any, data: pd.DataFrame) -> np.ndarray:
    [Lc] = build_design_matrices([info.design_info], data, return_type="matrix")
    return np.asarray(Lc)


def _wald_frame(
    est: np.ndarray, se: np.ndarray, df_value: float, level: float
) -> dict[str, np.ndarray]:
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ratio = np.where(se > 0, est / se, np.nan)
    if np.isfinite(df_value):
        p_value = 2.0 * _stats.t.sf(np.abs(t_ratio), df_value)
        crit = _stats.t.isf((1.0 - level) / 2.0, df_value)
    else:
        p_value = 2.0 * _stats.norm.sf(np.abs(t_ratio))
        crit = _stats.norm.isf((1.0 - level) / 2.0)
    return {
        "slope": est,
        "SE": se,
        "df": np.full(est.shape[0], df_value),
        "t_ratio": t_ratio,
        "p_value": p_value,
        "lower_cl": est - crit * se,
        "upper_cl": est + crit * se,
    }


def _df_value(info: Any) -> float:
    # Mixed / GLM models: use a z-test (inf df), matching emtrends.
    return float(
        np.inf if (info.family is not None or getattr(info, "is_mixed", False))
        else getattr(info, "df_resid", np.inf)
    )


def _beta_jacobian(
    theta: Any, beta: np.ndarray, *, rel_step: float = 1e-6
) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference Jacobian of a vector map ``theta(beta)``.

    Returns ``(theta(beta), J)`` where ``J`` has one row per output of
    ``theta`` and one column per parameter. Used for the response-scale
    delta method, where the per-row response slope
    ``h'(eta_i)(L_slope_i beta)`` is non-linear in ``beta`` and the full
    Jacobian (including the ``h''`` curvature term) is required for a
    correct standard error. Both the per-observation and the
    sample-averaged paths route through this so they cannot drift apart.
    """
    g0 = np.atleast_1d(np.asarray(theta(beta), dtype=float))
    p = beta.shape[0]
    jac = np.empty((g0.shape[0], p))
    for k in range(p):
        step = rel_step * max(1.0, abs(beta[k]))
        bp = beta.copy(); bp[k] += step
        bm = beta.copy(); bm[k] -= step
        jac[:, k] = (
            np.asarray(theta(bp), dtype=float)
            - np.asarray(theta(bm), dtype=float)
        ) / (2.0 * step)
    return g0, jac


def avg_slopes(
    obj: Any,
    var: str,
    *,
    by: str | list[str] | None = None,
    type: str = "link",
    level: float = 0.95,
    h: float = _DEFAULT_H,
) -> SlopesResult:
    """Average marginal effect of ``var`` over the observed sample.

    The slope of the model w.r.t. ``var`` is evaluated at every observed
    row and averaged, optionally within levels of a ``by`` factor.

    Parameters
    ----------
    obj
        A fitted model (statsmodels / linearmodels / any registered
        adapter) or a pymmeans result carrying ``model_info``.
    var
        Name of the numeric predictor whose slope is averaged.
    by
        Optional factor(s) to group the average within. ``None`` gives
        a single overall AME.
    type
        ``"link"`` (default) for the linear-predictor-scale slope, or
        ``"response"`` for the inverse-link chain-rule slope
        ``h'(eta) * d eta / d x`` (for GLMs).
    level
        Confidence level.
    h
        Finite-difference step for the design derivative.

    Returns
    -------
    SlopesResult

    Examples
    --------
    For an additive linear model, the average slope of ``x`` equals the
    OLS coefficient on ``x`` (with its standard error)::

        >>> from pymmeans import avg_slopes  # doctest: +SKIP
        >>> avg_slopes(fit, "x").frame  # doctest: +SKIP
    """
    if type not in ("link", "response"):
        raise ValueError(f"type must be 'link' or 'response'; got {type!r}.")
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1); got {level!r}.")

    info = _get_info(obj)
    data = _require_reference_data(info, "avg_slopes")
    if var not in data.columns:
        raise ValueError(f"{var!r} is not a column of the model data.")
    _require_numeric_var(data, var)

    by_list = [by] if isinstance(by, str) else (list(by) if by else [])
    beta = np.asarray(info.beta, dtype=float)
    vcov = np.asarray(info.vcov, dtype=float)

    L_slope = _slope_design(info, data, var, h)  # (n, p)

    # Build row groups: overall, or partitioned by the `by` factor(s).
    if by_list:
        keys = pd.MultiIndex.from_frame(data[by_list].astype(object))
        unique_keys = list(dict.fromkeys(keys))
        groups = [(k, np.asarray(keys == k)) for k in unique_keys]
    else:
        groups = [((), np.ones(len(data), dtype=bool))]

    df_value = _df_value(info)

    if type == "link":
        # Exact: averaged slope-design row, L_avg @ beta, sqrt(L_avg V L_avg^T).
        ests = np.empty(len(groups))
        ses = np.empty(len(groups))
        for gi, (_k, mask) in enumerate(groups):
            L_avg = L_slope[mask].mean(axis=0, keepdims=True)
            ests[gi] = float((L_avg @ beta)[0])
            ses[gi] = float(np.sqrt((L_avg @ vcov @ L_avg.T)[0, 0]))
    else:
        # Response scale: theta(beta) = mean_i h'(X_i beta) (L_slope_i beta).
        # Non-linear in beta -> delta method with a finite-difference Jacobian.
        if info.family is None:
            raise ValueError(
                "type='response' requires a GLM family on the model "
                "(info.family is None for a linear model; use type='link')."
            )
        X_center = _center_design(info, data)
        link = info.family.link

        def theta(b: np.ndarray) -> np.ndarray:
            eta = X_center @ b
            dh = np.asarray(link.inverse_deriv(eta), dtype=float)
            row_slope = dh * (L_slope @ b)
            return np.array([row_slope[mask].mean() for _k, mask in groups])

        ests, J = _beta_jacobian(theta, beta)
        cov = J @ vcov @ J.T
        ses = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    cols = _wald_frame(ests, ses, df_value, level)
    # Assemble the by-factor columns first so they define the row index,
    # then the var label, then the Wald columns.
    frame_data: dict[str, Any] = {}
    for i, name in enumerate(by_list):
        frame_data[name] = [k[i] for k, _m in groups]
    frame_data["var"] = [var] * len(groups)
    frame_data.update(cols)
    frame = pd.DataFrame(frame_data)

    return SlopesResult(frame=frame, var=var, type=type, level=level)


def slopes(
    obj: Any,
    var: str,
    *,
    type: str = "link",
    level: float = 0.95,
    h: float = _DEFAULT_H,
) -> SlopesResult:
    """Per-observation marginal effects of ``var``.

    Returns one row per observation, each carrying the marginal effect
    of ``var`` at that row and its delta-method standard error. For the
    sample-averaged effect, use :func:`avg_slopes`.

    Parameters are as in :func:`avg_slopes` (without ``by``). On the
    response scale the per-row standard error is a full delta-method
    standard error: the per-observation response slope
    ``h'(eta_i)(L_slope_i beta)`` is non-linear in ``beta``, so its
    Jacobian (carrying both the ``h''`` curvature term and the linear
    slope term) is taken numerically and propagated through ``V``.
    """
    if type not in ("link", "response"):
        raise ValueError(f"type must be 'link' or 'response'; got {type!r}.")
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1); got {level!r}.")

    info = _get_info(obj)
    data = _require_reference_data(info, "slopes")
    if var not in data.columns:
        raise ValueError(f"{var!r} is not a column of the model data.")
    _require_numeric_var(data, var)

    beta = np.asarray(info.beta, dtype=float)
    vcov = np.asarray(info.vcov, dtype=float)
    L_slope = _slope_design(info, data, var, h)  # (n, p)

    if type == "response":
        if info.family is None:
            raise ValueError(
                "type='response' requires a GLM family (got a linear model; "
                "use type='link')."
            )
        X_center = _center_design(info, data)
        link = info.family.link

        def theta(b: np.ndarray) -> np.ndarray:
            # Per-observation response slope, length n; non-linear in b.
            eta = X_center @ b
            dh = np.asarray(link.inverse_deriv(eta), dtype=float)
            return dh * (L_slope @ b)

        est, J = _beta_jacobian(theta, beta)  # J: (n, p)
        row_var = np.einsum("ij,jk,ik->i", J, vcov, J)
        se = np.sqrt(np.clip(row_var, 0.0, None))
    else:
        est = L_slope @ beta
        link_var = np.einsum("ij,jk,ik->i", L_slope, vcov, L_slope)
        se = np.sqrt(np.clip(link_var, 0.0, None))

    df_value = _df_value(info)
    cols = _wald_frame(est, se, df_value, level)
    frame = pd.DataFrame({"row": np.arange(len(data)), "var": var})
    for c, v in cols.items():
        frame[c] = v
    return SlopesResult(frame=frame, var=var, type=type, level=level)
