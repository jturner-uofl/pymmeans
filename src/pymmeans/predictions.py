"""Average adjusted predictions (``avg_predictions`` / ``predictions``).

The third leg of the ``marginaleffects`` triad (alongside
:func:`pymmeans.avg_slopes` and :func:`pymmeans.avg_comparisons`):
the model's fitted value, on the response or link scale, averaged over
the observed sample (or within levels of a ``by`` factor).

* :func:`avg_predictions` returns ``mean_i h(X_i beta)`` (response) or
  ``mean_i X_i beta`` (link), optionally within ``by`` groups, with a
  delta-method standard error.
* :func:`predictions` returns the per-observation fitted value and its
  standard error.

On the link scale the average prediction is exact -- ``Xbar beta`` with
standard error ``sqrt(Xbar V Xbar^T)``. On the response scale it is a
smooth non-linear function of ``beta`` whose standard error is the delta
method with a finite-difference Jacobian (the shared
:func:`pymmeans.slopes._beta_jacobian`).

A useful identity: for a logistic GLM fit by maximum likelihood,
``avg_predictions`` on the response scale equals the observed outcome
mean (score-equation calibration).

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

from pymmeans.comparisons import _pred_fn, _wald_cols
from pymmeans.slopes import (
    _assemble_marginal,
    _beta_jacobian,
    _center_design,
    _df_value,
    _get_info,
    _groups_for,
    _require_reference_data,
)

__all__ = ["PredictionsResult", "avg_predictions", "predictions"]


@dataclass(frozen=True)
class PredictionsResult:
    """Result of :func:`avg_predictions` or :func:`predictions`.

    Attributes
    ----------
    frame
        One row per ``by`` group (a single row for the overall average),
        or one row per observation for :func:`predictions`. Columns: the
        ``by`` factor(s) when present, ``estimate``, ``SE``, ``df``,
        ``t_ratio``, ``p_value``, ``lower_cl``, ``upper_cl``.
    type
        ``"link"`` or ``"response"``.
    level
        Confidence level used for the interval.
    """

    frame: pd.DataFrame
    type: str
    level: float


def avg_predictions(
    obj: Any,
    *,
    by: str | list[str] | None = None,
    type: str = "response",
    newdata: pd.DataFrame | None = None,
    hypothesis: Any = None,
    transform: Any = None,
    level: float = 0.95,
) -> PredictionsResult:
    """Average adjusted prediction over a sample or grid.

    Parameters
    ----------
    obj
        A fitted model or a pymmeans result carrying ``model_info``.
    by
        Optional factor(s) to average within, rather than over the whole
        sample.
    type
        ``"response"`` (default) applies the inverse link; ``"link"``
        averages on the linear-predictor scale.
    level
        Confidence level for the interval.

    Returns
    -------
    PredictionsResult
    """
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1); got {level!r}.")

    info = _get_info(obj)
    data = (
        newdata if newdata is not None
        else _require_reference_data(info, "avg_predictions")
    )
    pred = _pred_fn(info, type)
    beta = np.asarray(info.beta, dtype=float)
    vcov = np.asarray(info.vcov, dtype=float)
    design = _center_design(info, data)
    df_value = _df_value(info)

    by_list = [by] if isinstance(by, str) else (list(by) if by else [])
    groups = _groups_for(data, by_list)

    def theta(b: np.ndarray) -> np.ndarray:
        phi = pred(design @ b)
        return np.array([phi[m].mean() for _k, m in groups])

    est, jac = _beta_jacobian(theta, beta)

    id_frame = pd.DataFrame(
        {name: [k[bi] for k, _m in groups] for bi, name in enumerate(by_list)}
    )
    labels = (
        [", ".join(str(x) for x in k) for k, _m in groups]
        if by_list
        else ["prediction"]
    )
    frame = _assemble_marginal(
        est, jac, vcov, id_frame=id_frame, labels=labels, value_name="estimate",
        df_value=df_value, level=level, hypothesis=hypothesis, transform=transform,
    )
    return PredictionsResult(frame=frame, type=type, level=level)


def predictions(
    obj: Any,
    *,
    type: str = "response",
    newdata: pd.DataFrame | None = None,
    level: float = 0.95,
) -> PredictionsResult:
    """Per-observation adjusted prediction with a delta-method SE.

    Like :func:`avg_predictions` but returns one row per observation. For
    the sample-averaged value use :func:`avg_predictions`.
    """
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1); got {level!r}.")

    info = _get_info(obj)
    data = (
        newdata if newdata is not None
        else _require_reference_data(info, "predictions")
    )
    pred = _pred_fn(info, type)
    beta = np.asarray(info.beta, dtype=float)
    vcov = np.asarray(info.vcov, dtype=float)
    design = _center_design(info, data)
    df_value = _df_value(info)

    def theta(b: np.ndarray) -> np.ndarray:
        return pred(design @ b)

    est, jac = _beta_jacobian(theta, beta)  # jac: (n, p)
    row_var = np.einsum("ij,jk,ik->i", jac, vcov, jac)
    se = np.sqrt(np.clip(row_var, 0.0, None))
    cols = _wald_cols(est, se, df_value, level)
    frame = pd.DataFrame({"row": np.arange(len(data))})
    for c, v in cols.items():
        frame[c] = v
    return PredictionsResult(frame=frame, type=type, level=level)
