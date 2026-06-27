"""Nonlinear hypothesis tests on model coefficients via the delta method.

R `emmeans` and `multcomp` handle *linear* combinations of the
coefficient vector (``L @ beta``) directly. Many quantities of
interest, however, are *nonlinear* functions ``g(beta)`` --- a ratio
of two coefficients, a relative change, a back-transformed product,
the value of a fitted curve at a point, and so on. The R package
`car` exposes :func:`car::deltaMethod` for exactly this case, and the
`marginaleffects` package exposes ``hypotheses()``.

:func:`hypotheses` is the pymmeans analogue. Given a fitted model (or
any pymmeans result carrying a ``model_info``) and a callable
``g(beta) -> ndarray``, it returns the point estimate ``g(beta_hat)``
together with a standard error obtained by the delta method,

    SE(g) = sqrt( J V J^T ),    J = d g / d beta  evaluated at beta_hat,

where ``V`` is the coefficient covariance. The Jacobian ``J`` is
computed by central finite differences, so ``g`` may be any
black-box callable; no symbolic derivative is required. When ``g``
happens to be linear the finite-difference Jacobian recovers the
exact linear-combination row, so :func:`hypotheses` reduces to the
ordinary ``L @ beta`` contrast at machine precision.

References
----------
- Fox, J., & Weisberg, S. (2019). *An R Companion to Applied
  Regression* (3rd ed.), Sage. (`car::deltaMethod`.)
- Oehlert, G. W. (1992). A note on the delta method.
  *The American Statistician*, 46(1), 27-29.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as _stats

__all__ = ["HypothesisResult", "hypotheses"]


@dataclass(frozen=True)
class HypothesisResult:
    """Result of a nonlinear-hypothesis delta-method test.

    Attributes
    ----------
    frame
        A pandas DataFrame with one row per element of ``g(beta)`` and
        columns ``estimate``, ``SE``, ``df``, ``t_ratio``, ``p_value``,
        ``lower_cl``, ``upper_cl``.
    estimate
        The point estimate vector ``g(beta_hat)``.
    se
        The delta-method standard errors.
    jacobian
        The finite-difference Jacobian ``J = dg/dbeta`` at ``beta_hat``
        (shape ``(len(g), len(beta))``), retained for inspection.
    df
        Degrees of freedom used for the t / CI (``inf`` for a z-test
        when the model carries no residual df).
    level
        The confidence level used for ``lower_cl`` / ``upper_cl``.
    """

    frame: pd.DataFrame
    estimate: np.ndarray
    se: np.ndarray
    jacobian: np.ndarray
    df: float
    level: float


def _extract_beta_vcov(obj: Any) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    """Pull (beta, vcov, param_names, df) from a model, result, or ModelInfo."""
    # Accept (in order): an object already exposing the coefficient
    # surface directly (a ModelInfo, or the output of from_pyfixest /
    # from_fitted); a pymmeans result carrying a model_info; or a raw
    # fitted model routed through the adapter registry.
    if (
        getattr(obj, "model_info", None) is None
        and hasattr(obj, "beta")
        and hasattr(obj, "vcov")
        and hasattr(obj, "param_names")
    ):
        info = obj
    else:
        info = getattr(obj, "model_info", None)
        if info is None:
            from pymmeans.utils import from_fitted

            info = from_fitted(obj)
    beta = np.asarray(info.beta, dtype=float)
    vcov = np.asarray(info.vcov, dtype=float)
    names = list(info.param_names)
    # Residual df, if the adapter recorded one; else z-test (inf df).
    df = float(getattr(info, "df_resid", np.inf) or np.inf)
    return beta, vcov, names, df


def _numerical_jacobian(
    g: Callable[[np.ndarray], np.ndarray],
    beta: np.ndarray,
    *,
    rel_step: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference Jacobian of ``g`` at ``beta``.

    Returns ``(g0, J)`` where ``g0 = g(beta)`` (1-D) and ``J`` has
    shape ``(len(g0), len(beta))``. The per-coordinate step is
    ``rel_step * max(1, |beta_k|)`` so the differencing is well scaled
    for both small and large coefficients.
    """
    beta = np.asarray(beta, dtype=float)
    m = beta.shape[0]
    g0 = np.atleast_1d(np.asarray(g(beta), dtype=float))
    if g0.ndim != 1:
        raise ValueError(
            f"g(beta) must return a scalar or 1-D array; got shape {g0.shape}."
        )
    J = np.empty((g0.shape[0], m), dtype=float)
    for k in range(m):
        step = rel_step * max(1.0, abs(beta[k]))
        bp = beta.copy(); bp[k] += step
        bm = beta.copy(); bm[k] -= step
        gp = np.atleast_1d(np.asarray(g(bp), dtype=float))
        gm = np.atleast_1d(np.asarray(g(bm), dtype=float))
        if gp.shape != g0.shape or gm.shape != g0.shape:
            raise ValueError(
                "g(beta) returned inconsistent output shapes across the "
                "finite-difference grid; g must return a fixed-length vector."
            )
        J[:, k] = (gp - gm) / (2.0 * step)
    return g0, J


def hypotheses(
    obj: Any,
    g: Callable[[np.ndarray], np.ndarray],
    *,
    labels: Sequence[str] | None = None,
    level: float = 0.95,
    df: float | None = None,
    rel_step: float = 1e-6,
) -> HypothesisResult:
    """Test a nonlinear function ``g(beta)`` of model coefficients.

    Computes the point estimate ``g(beta_hat)`` and a delta-method
    standard error ``sqrt(J V J^T)`` with a finite-difference Jacobian
    ``J``. Works on a fitted model directly or on any pymmeans result
    object that carries a ``model_info`` (an :class:`EMMResult`,
    :class:`ContrastResult`, etc.).

    Parameters
    ----------
    obj
        A fitted model (``statsmodels`` / ``linearmodels`` / any
        registered adapter) or a pymmeans result carrying
        ``model_info``.
    g
        Callable mapping the coefficient vector ``beta`` (a 1-D NumPy
        array, ordered as ``model_info.param_names``) to a scalar or
        1-D array of derived quantities. For example,
        ``lambda b: b[1] / b[2]`` for a coefficient ratio.
    labels
        Optional row labels for the output frame (one per element of
        ``g(beta)``). Defaults to ``g[0], g[1], ...``.
    level
        Confidence level for the returned interval.
    df
        Override the degrees of freedom for the t / CI. By default the
        residual df recorded on the model is used; if none is
        available a z-test (``df = inf``) is used.
    rel_step
        Relative step for the central-difference Jacobian. The
        per-coordinate step is ``rel_step * max(1, |beta_k|)``.

    Returns
    -------
    HypothesisResult

    Examples
    --------
    Ratio of two regression coefficients with a delta-method SE::

        >>> import statsmodels.formula.api as smf  # doctest: +SKIP
        >>> from pymmeans import hypotheses  # doctest: +SKIP
        >>> fit = smf.ols("y ~ x1 + x2", data).fit()  # doctest: +SKIP
        >>> # param order is [Intercept, x1, x2]
        >>> hypotheses(fit, lambda b: b[1] / b[2],  # doctest: +SKIP
        ...            labels=["x1/x2"]).frame

    Notes
    -----
    When ``g`` is linear in ``beta`` the finite-difference Jacobian is
    exact and :func:`hypotheses` returns the same SE as the ordinary
    ``L @ beta`` contrast at machine precision. This makes it a strict
    generalisation of the linear-contrast machinery.
    """
    if not callable(g):
        raise TypeError("g must be callable: g(beta) -> scalar or 1-D array.")
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1); got {level!r}.")

    beta, vcov, _names, model_df = _extract_beta_vcov(obj)
    if vcov.shape != (beta.shape[0], beta.shape[0]):
        raise ValueError(
            f"coefficient covariance shape {vcov.shape} is inconsistent with "
            f"beta length {beta.shape[0]}."
        )

    g0, J = _numerical_jacobian(g, beta, rel_step=rel_step)
    cov_g = J @ vcov @ J.T
    var_g = np.clip(np.diag(cov_g), 0.0, None)
    se = np.sqrt(var_g)

    use_df = float(df) if df is not None else model_df
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ratio = np.where(se > 0, g0 / se, np.nan)
    if np.isfinite(use_df):
        p_value = 2.0 * _stats.t.sf(np.abs(t_ratio), use_df)
        tcrit = _stats.t.isf((1.0 - level) / 2.0, use_df)
    else:
        p_value = 2.0 * _stats.norm.sf(np.abs(t_ratio))
        tcrit = _stats.norm.isf((1.0 - level) / 2.0)

    lower = g0 - tcrit * se
    upper = g0 + tcrit * se

    if labels is None:
        labels = [f"g[{k}]" for k in range(g0.shape[0])]
    elif len(labels) != g0.shape[0]:
        raise ValueError(
            f"labels has length {len(labels)} but g(beta) has "
            f"{g0.shape[0]} element(s)."
        )

    frame = pd.DataFrame({
        "hypothesis": list(labels),
        "estimate": g0,
        "SE": se,
        "df": use_df,
        "t_ratio": t_ratio,
        "p_value": p_value,
        "lower_cl": lower,
        "upper_cl": upper,
    })

    return HypothesisResult(
        frame=frame,
        estimate=g0,
        se=se,
        jacobian=J,
        df=use_df,
        level=level,
    )
