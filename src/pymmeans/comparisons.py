"""Counterfactual comparisons (``avg_comparisons`` / ``comparisons``).

The ``marginaleffects`` ``comparisons()`` family reports the change in a
model's predicted outcome induced by a *counterfactual* change in a focal
predictor, averaged over the observed sample (g-computation). This is the
discrete-change companion to :func:`pymmeans.avg_slopes` (an instantaneous
derivative): where ``avg_slopes`` answers "what is the slope?",
``avg_comparisons`` answers "what happens to the prediction if I move this
variable from here to there?".

* For a **numeric** predictor the default is a *centred* one-unit change,

      mean_i[ g( h(X(x_i + s/2) beta), h(X(x_i - s/2) beta) ) ],

  with step ``s = 1`` and ``g`` the comparison function below.
* For a **categorical** predictor it is each non-reference level versus
  the reference,

      mean_i[ g( h(X(var=level) beta), h(X(var=ref) beta) ) ].

The ``comparison`` argument selects ``g`` applied to the two averaged
predictions ``(hi, lo) = (mean h(X_hi beta), mean h(X_lo beta))``:

==============  ====================================
``comparison``  g(hi, lo)
==============  ====================================
``difference``  ``hi - lo``        (default)
``ratio``       ``hi / lo``
``lnratio``     ``log(hi / lo)``
``lnor``        ``log( (hi/(1-hi)) / (lo/(1-lo)) )``
``lift``        ``(hi - lo) / lo``
==============  ====================================

Each estimand is a smooth function of ``beta``, so its standard error is
the delta method with a finite-difference Jacobian -- the same machinery
as :func:`pymmeans.avg_slopes` (see
:func:`pymmeans.slopes._beta_jacobian`). On an identity-link linear model
the difference of a numeric predictor over a unit step equals the OLS
coefficient (times the step) exactly.

References
----------
- Arel-Bundock, V. (2024). `marginaleffects`: predictions, comparisons,
  slopes, and hypothesis tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as _stats
from patsy import build_design_matrices

from pymmeans.slopes import (
    _assemble_marginal,
    _beta_jacobian,
    _df_value,
    _get_info,
    _groups_for,
    _require_reference_data,
)

__all__ = ["ComparisonsResult", "avg_comparisons", "comparisons"]

# Comparison functions operate on the two (group-averaged, for
# ``avg_comparisons``; per-row, for ``comparisons``) predictions.
_COMPARISONS = {
    "difference": lambda hi, lo: hi - lo,
    "ratio": lambda hi, lo: hi / lo,
    "lnratio": lambda hi, lo: np.log(hi / lo),
    "lnor": lambda hi, lo: np.log((hi / (1.0 - hi)) / (lo / (1.0 - lo))),
    "lift": lambda hi, lo: (hi - lo) / lo,
}


@dataclass(frozen=True)
class ComparisonsResult:
    """Result of :func:`avg_comparisons` or :func:`comparisons`.

    Attributes
    ----------
    frame
        One row per (focal variable, contrast) -- and per ``by`` level or
        per observation, depending on the call. Columns: the ``by``
        factor(s) when present, ``term``, ``contrast``, ``estimate``,
        ``SE``, ``df``, ``t_ratio``, ``p_value``, ``lower_cl``,
        ``upper_cl``.
    comparison
        The comparison function applied (``"difference"`` etc.).
    type
        ``"link"`` or ``"response"``.
    level
        Confidence level used for the interval.
    """

    frame: pd.DataFrame
    comparison: str
    type: str
    level: float


def _wald_cols(
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
        "estimate": est,
        "SE": se,
        "df": np.full(est.shape[0], df_value),
        "t_ratio": t_ratio,
        "p_value": p_value,
        "lower_cl": est - crit * se,
        "upper_cl": est + crit * se,
    }


def _cf_design(info: Any, data: pd.DataFrame, var: str, newval: Any) -> np.ndarray:
    """Design matrix with ``var`` set to ``newval`` for every row.

    ``newval`` is either a scalar (a categorical level) or an array (the
    perturbed numeric values). The dtype of the focal column is preserved
    so patsy encodes the counterfactual exactly as the original fit.
    """
    cf = data.copy()
    if pd.api.types.is_numeric_dtype(data[var]):
        cf[var] = np.asarray(newval, dtype=float)
    elif hasattr(data[var], "cat"):
        cf[var] = pd.Categorical(
            [newval] * len(data), categories=data[var].cat.categories
        )
    else:
        cf[var] = [newval] * len(data)
    try:
        [des] = build_design_matrices(
            [info.design_info], cf, return_type="matrix"
        )
    except Exception as exc:  # patsy raises bespoke, non-public error types
        raise ValueError(
            f"Could not build the counterfactual design for {var!r}. This "
            f"happens when {var!r} enters through a term whose value cannot "
            f"be overridden cleanly (e.g. a spline bs()/poly() basis whose "
            f"knots the counterfactual crosses). Original error: {exc}"
        ) from exc
    return np.asarray(des)


def _reference_level(col: pd.Series) -> Any:
    """The baseline level for a categorical predictor (first category)."""
    if hasattr(col, "cat"):
        return col.cat.categories[0]
    return sorted(pd.unique(col))[0]


def _resolve_var_specs(
    info: Any, data: pd.DataFrame, variables: Any
) -> list[tuple[str, Any]]:
    """Resolve ``variables`` to a list of ``(name, spec)`` pairs.

    ``None`` -> every predictor with a default spec; a string / list ->
    those variables with default specs; a dict -> per-variable specs.
    """
    if variables is None:
        resp = getattr(info, "response_name", None)
        return [(c, None) for c in data.columns if c != resp]
    if isinstance(variables, str):
        return [(variables, None)]
    if isinstance(variables, dict):
        return list(variables.items())
    return [(v, None) for v in variables]


def _pred_fn(info: Any, type: str) -> Any:
    """Return the prediction map applied to the linear predictor eta."""
    if type not in ("link", "response"):
        raise ValueError(f"type must be 'link' or 'response'; got {type!r}.")
    if type == "response":
        if info.family is None:
            raise ValueError(
                "type='response' requires a GLM family (got a linear model; "
                "use type='link')."
            )
        link = info.family.link
        return lambda eta: np.asarray(link.inverse(eta), dtype=float)
    return lambda eta: np.asarray(eta, dtype=float)


def _numeric_change(
    col: pd.Series, spec: Any, default_step: float
) -> tuple[str, Any, Any]:
    """Resolve a numeric change spec to ``(label, lo, hi)``.

    ``lo`` / ``hi`` are per-row arrays for *centred* specs (a unit/SD step
    around each observed value) or scalars for *absolute* specs (the same
    endpoints for every row), matching ``marginaleffects``:

    ===============  ====================================================
    spec             change
    ===============  ====================================================
    ``None`` / num   centred ``x +- step/2`` (step = ``default_step``/num)
    ``"sd"``         centred ``x +- sd/2``
    ``"2sd"``        centred ``x +- sd``
    ``"iqr"``        absolute Q1 -> Q3
    ``"minmax"``     absolute min -> max
    ``(lo, hi)``     absolute lo -> hi
    ===============  ====================================================
    """
    x = col.to_numpy(dtype=float)
    if spec is None or (isinstance(spec, (int, float)) and not isinstance(spec, bool)):
        step = default_step if spec is None else float(spec)
        return f"+{step:g}", x - step / 2.0, x + step / 2.0
    if isinstance(spec, str):
        key = spec.lower()
        if key == "sd":
            sd = float(np.std(x, ddof=1))
            return "+sd", x - sd / 2.0, x + sd / 2.0
        if key == "2sd":
            sd = float(np.std(x, ddof=1))
            return "+2sd", x - sd, x + sd
        if key == "iqr":
            q1, q3 = float(np.quantile(x, 0.25)), float(np.quantile(x, 0.75))
            return "Q3 - Q1", q1, q3
        if key == "minmax":
            return "max - min", float(np.min(x)), float(np.max(x))
        raise ValueError(
            f"unknown numeric change spec {spec!r}; expected a number, "
            "'sd', '2sd', 'iqr', 'minmax', or a (lo, hi) pair."
        )
    # (lo, hi) absolute pair.
    seq = list(spec)
    if len(seq) != 2:
        raise ValueError(
            f"a numeric (lo, hi) change spec must have length 2; got {spec!r}."
        )
    lo, hi = float(seq[0]), float(seq[1])
    return f"{hi:g} - {lo:g}", lo, hi


def _contrasts_for(
    info: Any, data: pd.DataFrame, var: str, spec: Any, default_step: float
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Build (label, X_hi, X_lo) design pairs for a focal variable.

    Numeric -> one contrast from the resolved change spec. Categorical ->
    one contrast per non-reference level, each versus the reference level
    (a categorical ``spec`` may give an explicit list of levels to use).
    """
    if var not in data.columns:
        raise ValueError(f"{var!r} is not a column of the model data.")
    col = data[var]
    if pd.api.types.is_numeric_dtype(col):
        label, lo, hi = _numeric_change(col, spec, default_step)
        x_hi = _cf_design(info, data, var, hi)
        x_lo = _cf_design(info, data, var, lo)
        return [(label, x_hi, x_lo)]
    ref = _reference_level(col)
    if spec is not None:
        levels = list(spec)
    else:
        levels = (
            list(col.cat.categories) if hasattr(col, "cat") else sorted(pd.unique(col))
        )
    x_lo = _cf_design(info, data, var, ref)
    out = []
    for lvl in levels:
        if lvl == ref:
            continue
        x_hi = _cf_design(info, data, var, lvl)
        out.append((f"{lvl} - {ref}", x_hi, x_lo))
    return out


def _resolve_comparison(comparison: Any) -> Any:
    """A comparison function from the registry name or a user callable."""
    if callable(comparison):
        return comparison
    if comparison not in _COMPARISONS:
        raise ValueError(
            f"comparison must be one of {sorted(_COMPARISONS)} or a callable "
            f"(hi, lo) -> value; got {comparison!r}."
        )
    return _COMPARISONS[comparison]


def avg_comparisons(
    obj: Any,
    variables: str | list[str] | dict[str, Any] | None = None,
    *,
    comparison: Any = "difference",
    by: str | list[str] | None = None,
    type: str = "response",
    step: float = 1.0,
    newdata: pd.DataFrame | None = None,
    hypothesis: Any = None,
    transform: Any = None,
    level: float = 0.95,
) -> ComparisonsResult:
    """Average counterfactual comparison(s) over a sample or grid.

    Parameters
    ----------
    obj
        A fitted model (statsmodels / linearmodels / ...) or a pymmeans
        result carrying ``model_info``.
    variables
        Focal predictor(s). ``None`` (default) uses every predictor; a
        string or list restricts to those (default change spec); a dict
        ``{name: spec}`` sets a per-variable change spec. A numeric ``spec``
        is a number (centred step), ``"sd"`` / ``"2sd"`` (centred SD step),
        ``"iqr"`` (Q1->Q3), ``"minmax"`` (min->max), or a ``(lo, hi)`` pair;
        a categorical ``spec`` is the explicit list of levels to contrast
        against the reference.
    comparison
        ``"difference"`` (default), ``"ratio"``, ``"lnratio"``, ``"lnor"``,
        ``"lift"``, or a callable ``(hi, lo) -> value`` applied to the two
        averaged predictions.
    by
        Optional factor(s) to report the average within.
    type
        ``"response"`` (default) applies the inverse link; ``"link"`` works
        on the linear-predictor scale.
    step
        Default centred change for numeric predictors without an explicit
        spec. Default 1.0.
    newdata
        Optional DataFrame (e.g. from :func:`pymmeans.datagrid`) to average
        over instead of the model's observed sample.
    level
        Confidence level for the interval.

    Returns
    -------
    ComparisonsResult
    """
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1); got {level!r}.")
    cmp = _resolve_comparison(comparison)

    info = _get_info(obj)
    data = (
        newdata if newdata is not None else _require_reference_data(info, "avg_comparisons")
    )
    pred = _pred_fn(info, type)
    beta = np.asarray(info.beta, dtype=float)
    vcov = np.asarray(info.vcov, dtype=float)
    df_value = _df_value(info)

    by_list = [by] if isinstance(by, str) else (list(by) if by else [])
    groups = _groups_for(data, by_list)

    # Collect every (variable, contrast, by-group) row's estimate and its
    # Jacobian, so a `hypothesis` contrast spans the full result set.
    all_est: list[float] = []
    all_jac: list[np.ndarray] = []
    id_rows: list[dict[str, Any]] = []
    labels: list[str] = []
    for var, spec in _resolve_var_specs(info, data, variables):
        for label, x_hi, x_lo in _contrasts_for(info, data, var, spec, step):

            def theta(b: np.ndarray, _hi=x_hi, _lo=x_lo) -> np.ndarray:
                phi_hi = pred(_hi @ b)
                phi_lo = pred(_lo @ b)
                return np.array([
                    cmp(phi_hi[m].mean(), phi_lo[m].mean()) for _k, m in groups
                ])

            est_v, jac_v = _beta_jacobian(theta, beta)
            for gi, (k, _m) in enumerate(groups):
                all_est.append(float(est_v[gi]))
                all_jac.append(jac_v[gi])
                idr: dict[str, Any] = {
                    name: k[bi] for bi, name in enumerate(by_list)
                }
                idr["term"] = var
                idr["contrast"] = label
                id_rows.append(idr)
                grp = (" " + ", ".join(str(x) for x in k)) if by_list else ""
                labels.append(f"{var} {label}{grp}")

    est = np.asarray(all_est, dtype=float)
    jac = np.asarray(all_jac, dtype=float)
    id_frame = pd.DataFrame(id_rows)
    frame = _assemble_marginal(
        est, jac, vcov, id_frame=id_frame, labels=labels, value_name="estimate",
        df_value=df_value, level=level, hypothesis=hypothesis, transform=transform,
    )
    return ComparisonsResult(
        frame=frame, comparison=comparison, type=type, level=level
    )


def comparisons(
    obj: Any,
    variables: str | list[str] | dict[str, Any] | None = None,
    *,
    comparison: Any = "difference",
    type: str = "response",
    step: float = 1.0,
    newdata: pd.DataFrame | None = None,
    level: float = 0.95,
) -> ComparisonsResult:
    """Per-observation counterfactual comparison(s).

    Like :func:`avg_comparisons` but returns one row per observation per
    contrast (the comparison function is applied per row, not to the
    sample-averaged predictions), each with its own delta-method standard
    error. Accepts the same ``variables`` change specs, callable
    ``comparison``, and ``newdata``. For the sample-averaged effect use
    :func:`avg_comparisons`.
    """
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1); got {level!r}.")
    cmp = _resolve_comparison(comparison)

    info = _get_info(obj)
    data = (
        newdata if newdata is not None else _require_reference_data(info, "comparisons")
    )
    pred = _pred_fn(info, type)
    beta = np.asarray(info.beta, dtype=float)
    vcov = np.asarray(info.vcov, dtype=float)
    df_value = _df_value(info)
    n = len(data)

    frames: list[pd.DataFrame] = []
    for var, spec in _resolve_var_specs(info, data, variables):
        for label, x_hi, x_lo in _contrasts_for(info, data, var, spec, step):

            def theta(b: np.ndarray, _hi=x_hi, _lo=x_lo) -> np.ndarray:
                return cmp(pred(_hi @ b), pred(_lo @ b))

            est, jac = _beta_jacobian(theta, beta)  # jac: (n, p)
            row_var = np.einsum("ij,jk,ik->i", jac, vcov, jac)
            se = np.sqrt(np.clip(row_var, 0.0, None))
            cols = _wald_cols(est, se, df_value, level)
            fr = pd.DataFrame({
                "row": np.arange(n),
                "term": var,
                "contrast": label,
            })
            for c, v in cols.items():
                fr[c] = v
            frames.append(fr)

    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return ComparisonsResult(
        frame=frame, comparison=comparison, type=type, level=level
    )
