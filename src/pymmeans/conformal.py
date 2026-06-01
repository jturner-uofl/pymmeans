"""Conformal prediction intervals for pymmeans's ML adapter.

This module implements two **distribution-free, finite-sample-valid**
prediction-interval constructors that integrate with the pymmeans
``ml_emmeans`` / ``from_predict`` API:

* :func:`split_conformal_pi` — the **split-conformal predictor** of
  Vovk et al. (2005), Lei et al. (2018). Given a fitted outcome
  model and a held-out calibration set, returns per-cell prediction
  intervals on individual outcomes with the marginal coverage
  guarantee ``P(Y_new in PI(X_new)) >= 1 - alpha`` under exchangeability.
* :func:`conformal_counterfactual_pi` — the **weighted split-
  conformal counterfactual predictor** of Lei & Candès (2021),
  which extends conformal to the missing-counterfactual setting.
  Given an outcome model for the treatment-of-interest group, a
  propensity-score model, and a held-out calibration set, returns
  prediction intervals on the **counterfactual outcome** ``Y(t*) | X``
  with valid coverage even at points where the counterfactual was
  not observed.

Both functions return distribution-free coverage guarantees that
hold for any (well-defined) outcome and propensity model — the
guarantee does NOT depend on the model being correctly specified.
This is the key conceptual contribution of conformal prediction
that Wald / bootstrap / posterior intervals do not provide.

R `emmeans` does not implement either of these — sensitivity to
unmeasured confounding (E-value) and multiple-imputation pooling
were the v0.3.0 contributions. Conformal prediction is the v0.4.0
contribution.

References
----------
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic
  Learning in a Random World*. Springer.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., & Wasserman,
  L. (2018). Distribution-free predictive inference for regression.
  *Journal of the American Statistical Association*, 113(523),
  1094-1111.
- Lei, L., & Candès, E. J. (2021). Conformal inference of
  counterfactuals and individual treatment effects.
  *Journal of the Royal Statistical Society Series B*, 83(5),
  911-938.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "ConformalCounterfactualResult",
    "ConformalPIResult",
    "conformal_counterfactual_pi",
    "split_conformal_pi",
]


@dataclass(frozen=True)
class ConformalPIResult:
    """Result of :func:`split_conformal_pi`.

    Attributes
    ----------
    frame
        A copy of the input ``MLEMMResult.frame`` with two new
        columns: ``lower_pi`` and ``upper_pi``, the conformal
        prediction-interval bounds at each cell.
    q_hat
        The conformal quantile of nonconformity scores at level
        ``1 - level`` (the half-width of every cell's PI).
    level
        The nominal coverage level supplied to
        :func:`split_conformal_pi`.
    n_calibration
        Number of calibration observations used to compute ``q_hat``.

    Notes
    -----
    The PI at each cell is the same width: ``[emmean - q_hat,
    emmean + q_hat]``. This bounds the **individual outcome**
    distribution, NOT the cell-mean uncertainty (which would be a
    confidence interval). For CIs on the cell mean, use
    :func:`pymmeans.bootstrap_ci`.
    """

    frame: pd.DataFrame
    q_hat: float
    level: float
    n_calibration: int


@dataclass(frozen=True)
class ConformalCounterfactualResult:
    """Result of :func:`conformal_counterfactual_pi`.

    Attributes
    ----------
    frame
        A copy of the input ``MLEMMResult.frame`` with two new
        columns: ``lower_pi`` and ``upper_pi``, the weighted-
        conformal counterfactual-PI bounds at each cell, **valid
        even for cells outside the observed treatment-of-interest
        subset**.
    q_hat
        The weighted-quantile of nonconformity scores at level
        ``1 - level`` (the half-width of every cell's PI).
    level, n_calibration
        Same semantics as :class:`ConformalPIResult`.
    treatment_value
        The counterfactual treatment level the PI covers.
    weight_clip
        The ``(low, high)`` propensity-score clipping range applied
        before inverting to weights; protects against extreme weights
        when ``π̂(x) → 0`` or ``1``.
    """

    frame: pd.DataFrame
    q_hat: float
    level: float
    n_calibration: int
    treatment_value: Any
    weight_clip: tuple[float, float]


def _validate_level(level: float) -> None:
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1); got {level!r}.")


def _validate_em_ml(em_ml: Any) -> None:
    """Confirm input is an MLEMMResult with a populated predict_fn + data."""
    from pymmeans.ml import MLEMMResult

    if not isinstance(em_ml, MLEMMResult):
        raise TypeError(
            f"split_conformal_pi expects an MLEMMResult (from ml_emmeans); "
            f"got {type(em_ml).__name__}."
        )
    if em_ml.ml_info is None:
        raise ValueError(
            "split_conformal_pi: MLEMMResult.ml_info is None — cannot "
            "extract the predict_fn or response name."
        )
    if em_ml.ml_info.predict_fn is None:
        raise ValueError(
            "split_conformal_pi: MLEMMResult.ml_info.predict_fn is None — "
            "the conformal PI requires a callable to predict on new data."
        )
    if em_ml.ml_info.response is None or em_ml.ml_info.response == "":
        raise ValueError(
            "split_conformal_pi: MLEMMResult.ml_info.response is empty — "
            "from_predict(...) must have been called with response='...'."
        )


def _conformal_half_width(scores: np.ndarray, level: float) -> float:
    """Compute the ⌈(n+1)(1-α)⌉ / n empirical quantile of nonconformity scores.

    This is the finite-sample-corrected quantile in Vovk et al.
    (2005), Lei et al. (2018) — it guarantees marginal coverage of
    at least ``level`` under exchangeability of (training,
    calibration, test) draws.
    """
    n = int(scores.shape[0])
    if n < 2:
        raise ValueError(
            f"Conformal PI requires at least 2 calibration observations; "
            f"got {n}."
        )
    # Order statistic index (1-based) at the ⌈(n+1)·level⌉ position;
    # clipped to [1, n]. This is the finite-sample-corrected quantile.
    k = int(np.ceil((n + 1.0) * level))
    k = max(1, min(k, n))
    sorted_scores = np.sort(scores)
    return float(sorted_scores[k - 1])


def split_conformal_pi(
    em_ml: Any,
    calibration_data: pd.DataFrame,
    *,
    level: float = 0.95,
) -> ConformalPIResult:
    """Split-conformal prediction intervals for an MLEMMResult.

    Given an :class:`pymmeans.MLEMMResult` (produced by
    :func:`pymmeans.ml_emmeans`) and a **held-out calibration set**
    (not used in fitting), compute per-cell prediction intervals
    with the finite-sample marginal-coverage guarantee::

        P( Y_new in [emmean(cell) - q̂, emmean(cell) + q̂] ) >= level

    under exchangeability of (training, calibration, test) draws.
    The PI is the same width for every cell.

    Parameters
    ----------
    em_ml
        An :class:`MLEMMResult` whose ``ml_info.predict_fn`` is the
        outcome model trained on data *disjoint from* ``calibration_data``.
    calibration_data
        A DataFrame containing the same columns as the training data,
        with the response column named by ``em_ml.ml_info.response``.
        Used to compute nonconformity scores. Must be disjoint from
        the training data — otherwise the marginal-coverage guarantee
        does not hold.
    level
        Desired marginal coverage (e.g. 0.95 for 95% PIs).

    Returns
    -------
    ConformalPIResult
        ``.frame`` is a copy of ``em_ml.frame`` with two extra
        columns: ``lower_pi`` and ``upper_pi``.

    Notes
    -----
    * Coverage is **marginal**, not conditional: the PI has the same
      width at every cell, even where the model is less accurate.
      Conditional-coverage variants (CQR, Romano-Patterson-Candès
      2019) are a future extension.
    * Coverage is **distribution-free**: it holds for any outcome
      distribution under exchangeability, including heavy-tailed
      and contaminated residuals where Wald intervals undercover.
    * The PI bounds an individual outcome, NOT the cell-mean. For
      CIs on the cell mean, use :func:`pymmeans.bootstrap_ci`.

    Examples
    --------
    >>> import pandas as pd  # doctest: +SKIP
    >>> from sklearn.ensemble import GradientBoostingRegressor  # doctest: +SKIP
    >>> from pymmeans.ml import from_predict, ml_emmeans  # doctest: +SKIP
    >>> from pymmeans import split_conformal_pi  # doctest: +SKIP
    >>>
    >>> # Train + calibration split.
    >>> train_df = ...  # 70% of data  # doctest: +SKIP
    >>> cal_df   = ...  # 30% of data  # doctest: +SKIP
    >>>
    >>> model = GradientBoostingRegressor().fit(  # doctest: +SKIP
    ...     train_df[['x1', 'x2']], train_df['y']
    ... )
    >>> info = from_predict(  # doctest: +SKIP
    ...     predict_fn=lambda d: model.predict(d[['x1', 'x2']]),
    ...     data=train_df, factors={"treat": [0, 1]},
    ...     numerics=["x1", "x2"], response="y",
    ... )
    >>> em = ml_emmeans(info, "treat")  # doctest: +SKIP
    >>> pi = split_conformal_pi(em, cal_df, level=0.95)  # doctest: +SKIP
    >>> pi.frame  # has lower_pi, upper_pi columns  # doctest: +SKIP
    """
    _validate_level(level)
    _validate_em_ml(em_ml)

    response = em_ml.ml_info.response
    predict_fn = em_ml.ml_info.predict_fn

    if response not in calibration_data.columns:
        raise ValueError(
            f"split_conformal_pi: calibration_data has no column "
            f"{response!r} (the MLEMMResult.ml_info.response). "
            f"Available columns: {list(calibration_data.columns)}"
        )
    y_cal = calibration_data[response].to_numpy(dtype=float)
    if not np.all(np.isfinite(y_cal)):
        raise ValueError(
            "split_conformal_pi: calibration_data response column "
            "contains non-finite values; conformal scores cannot be "
            "computed on these rows."
        )

    mu_cal = np.asarray(predict_fn(calibration_data), dtype=float)
    if mu_cal.shape != y_cal.shape:
        raise ValueError(
            f"split_conformal_pi: predict_fn returned shape "
            f"{mu_cal.shape}, expected {y_cal.shape}."
        )
    scores = np.abs(y_cal - mu_cal)
    q_hat = _conformal_half_width(scores, level)

    new_frame = em_ml.frame.copy()
    emmean = new_frame["emmean"].to_numpy(dtype=float)
    new_frame["lower_pi"] = emmean - q_hat
    new_frame["upper_pi"] = emmean + q_hat
    return ConformalPIResult(
        frame=new_frame,
        q_hat=float(q_hat),
        level=float(level),
        n_calibration=len(y_cal),
    )


# ---------------------------------------------------------------------- Lei-Candès


def _weighted_quantile(
    scores: np.ndarray, weights: np.ndarray, level: float
) -> float:
    """Smallest score s such that the cumulative normalised weight at s is >= level.

    Used by :func:`conformal_counterfactual_pi` for the weighted
    quantile in Lei-Candès (2021) Algorithm 1.
    """
    if scores.shape != weights.shape or scores.ndim != 1:
        raise ValueError("scores and weights must be 1-D arrays of equal length.")
    if scores.size == 0:
        raise ValueError("weighted_quantile requires at least one score.")
    if np.any(weights < 0):
        raise ValueError("All weights must be non-negative.")
    if weights.sum() <= 0:
        raise ValueError("Sum of weights must be positive.")

    order = np.argsort(scores)
    s_sorted = scores[order]
    w_sorted = weights[order]
    cum = np.cumsum(w_sorted) / w_sorted.sum()
    idx = int(np.searchsorted(cum, level, side="left"))
    idx = min(idx, scores.size - 1)
    return float(s_sorted[idx])


def conformal_counterfactual_pi(
    em_ml: Any,
    calibration_data: pd.DataFrame,
    propensity_predict_fn: Callable[[pd.DataFrame], np.ndarray],
    *,
    treatment_value: Any = 1,
    level: float = 0.95,
    weight_clip: tuple[float, float] = (0.05, 0.95),
) -> ConformalCounterfactualResult:
    """Weighted split-conformal PI for the counterfactual outcome ``Y(t*) | x``.

    Implements Lei & Candès (2021) Algorithm 1 (Weighted Split CP).
    Given an outcome model trained on the subset of observations
    with treatment level ``t* = treatment_value``, a propensity-
    score model, and a held-out calibration set, returns
    prediction intervals on the counterfactual outcome
    ``Y(t*) | X = x*`` for any ``x*``, **including ``x*`` from the
    OTHER treatment group** (the missing-counterfactual case).

    Parameters
    ----------
    em_ml
        :class:`MLEMMResult` whose ``ml_info.predict_fn`` was
        trained ONLY on the subset of training data with
        ``treatment == treatment_value``. Its ``ml_info.factors``
        must include the treatment column (so the EMM grid includes
        cells at the relevant levels).
    calibration_data
        Held-out DataFrame containing the response column
        (named by ``em_ml.ml_info.response``) and the treatment
        column. **Only rows with ``treatment == treatment_value``
        contribute nonconformity scores**; other rows are ignored
        but used in deriving propensity weights.
    propensity_predict_fn
        A callable accepting a DataFrame and returning
        ``P(treatment == 1 | X)`` per row (a 1-D array of
        probabilities in (0, 1)).
    treatment_value
        The treatment level whose counterfactual the PI covers
        (typically ``1`` for "treated").
    level
        Desired marginal coverage.
    weight_clip
        ``(low, high)`` clipping range applied to the propensity
        score before computing IPW weights ``w = 1 / π̂(x)`` (or
        ``1 / (1 - π̂(x))`` for ``treatment_value=0``). Protects
        against extreme weights when ``π̂(x)`` is near 0 or 1.

    Returns
    -------
    ConformalCounterfactualResult
        ``.frame`` is a copy of ``em_ml.frame`` with extra columns
        ``lower_pi`` and ``upper_pi``.

    Notes
    -----
    The marginal-coverage guarantee
    ``P(Y(t*)_new in PI(x*_new)) >= level`` holds under the standard
    causal-inference assumptions of **unconfoundedness** and
    **overlap**, plus exchangeability of (training, calibration,
    test) draws. The coverage is **robust** to misspecification of
    either the outcome model OR the propensity model individually
    — but not both.

    The propensity-score clipping is a deliberate trade-off:
    extreme weights at small propensity scores produce unstable
    quantile estimates. Default clipping is ``(0.05, 0.95)``;
    users analysing data with severe overlap violations should
    pre-trim their sample and pass ``weight_clip=(0.01, 0.99)``
    or similar.

    References
    ----------
    Lei, L., & Candès, E. J. (2021). Conformal inference of
    counterfactuals and individual treatment effects.
    *Journal of the Royal Statistical Society Series B*, 83(5),
    911-938.
    """
    _validate_level(level)
    _validate_em_ml(em_ml)
    if not (0.0 < weight_clip[0] < weight_clip[1] < 1.0):
        raise ValueError(
            f"weight_clip must be (low, high) with 0 < low < high < 1; "
            f"got {weight_clip!r}."
        )
    if not callable(propensity_predict_fn):
        raise TypeError(
            "propensity_predict_fn must be callable; got "
            f"{type(propensity_predict_fn).__name__}."
        )
    # Binary-treatment restriction is enforced UP-FRONT, before any
    # factor lookup. Multinomial support is on the roadmap but the
    # IPW-weight derivation below assumes binary {0, 1} coding.
    if treatment_value not in (0, 1):
        raise NotImplementedError(
            "conformal_counterfactual_pi currently supports only binary "
            "treatments coded {0, 1}. For multinomial treatments, pass "
            "a propensity_predict_fn that returns the probability of the "
            "chosen treatment level and call this function once per level."
        )

    response = em_ml.ml_info.response
    predict_fn = em_ml.ml_info.predict_fn

    # Identify the treatment column. The MLEMMResult tracks one or
    # more factor columns; we look for one whose levels include
    # ``treatment_value``.
    factor_levels = em_ml.ml_info.factor_levels
    treatment_col = None
    for col, levels in factor_levels.items():
        if treatment_value in levels:
            treatment_col = col
            break
    if treatment_col is None:
        raise ValueError(
            f"conformal_counterfactual_pi: could not find a factor "
            f"column whose levels include treatment_value={treatment_value!r}. "
            f"factor_levels: {factor_levels}"
        )

    if response not in calibration_data.columns:
        raise ValueError(
            f"calibration_data has no response column {response!r}."
        )
    if treatment_col not in calibration_data.columns:
        raise ValueError(
            f"calibration_data has no treatment column {treatment_col!r}."
        )

    # Calibration subset: only rows with treatment == treatment_value.
    mask = (calibration_data[treatment_col] == treatment_value).to_numpy()
    cal_t = calibration_data.loc[mask].reset_index(drop=True)
    if len(cal_t) < 2:
        raise ValueError(
            f"conformal_counterfactual_pi: too few calibration rows "
            f"with {treatment_col} == {treatment_value!r} "
            f"(found {len(cal_t)}); need at least 2."
        )

    # Nonconformity scores on the t* subset.
    y_t = cal_t[response].to_numpy(dtype=float)
    mu_t = np.asarray(predict_fn(cal_t), dtype=float)
    if mu_t.shape != y_t.shape:
        raise ValueError(
            f"predict_fn returned shape {mu_t.shape}, expected {y_t.shape}."
        )
    scores = np.abs(y_t - mu_t)

    # IPW weights derived from the propensity model on the t* subset.
    p1 = np.asarray(propensity_predict_fn(cal_t), dtype=float)
    if p1.shape != y_t.shape:
        raise ValueError(
            f"propensity_predict_fn returned shape {p1.shape}, "
            f"expected {y_t.shape}."
        )
    p1 = np.clip(p1, weight_clip[0], weight_clip[1])
    # By construction (validated up-front) treatment_value is 0 or 1.
    if treatment_value == 1:
        weights = 1.0 / p1
    else:
        weights = 1.0 / (1.0 - p1)

    q_hat = _weighted_quantile(scores, weights, level)

    new_frame = em_ml.frame.copy()
    emmean = new_frame["emmean"].to_numpy(dtype=float)
    new_frame["lower_pi"] = emmean - q_hat
    new_frame["upper_pi"] = emmean + q_hat

    return ConformalCounterfactualResult(
        frame=new_frame,
        q_hat=float(q_hat),
        level=float(level),
        n_calibration=len(y_t),
        treatment_value=treatment_value,
        weight_clip=weight_clip,
    )
