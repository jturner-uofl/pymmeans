"""Double machine learning estimators for pymmeans's ML adapter.

This module implements two estimators that together form the
**double / debiased machine learning (DML)** framework of
Chernozhukov et al. (2018), integrated with pymmeans's
``ml_emmeans`` / ``from_predict`` API surface:

* :func:`cross_fit_ml_emmeans` — **K-fold cross-fitted g-computation**.
  Splits the training data into K folds; for each fold k, refits
  the outcome model on the OTHER K-1 folds, then predicts cell-
  level marginal means on fold k. The K cell-mean estimates are
  averaged. This guarantees that every prediction used in the
  marginal average is **out-of-sample** with respect to the model
  that produced it — the structural property that DML relies on.
* :func:`aipw_ate` — **Augmented inverse-probability weighting**
  (Robins, Rotnitzky & Zhao 1994) for the average treatment effect.
  Combines an outcome model with a propensity-score model in the
  classic doubly-robust estimating equation:

      ψ_i  =  μ̂₁(X_i) − μ̂₀(X_i)
              + T_i · (Y_i − μ̂₁(X_i)) / π̂(X_i)
              − (1 − T_i) · (Y_i − μ̂₀(X_i)) / (1 − π̂(X_i))

  ``τ̂_AIPW = mean(ψ)`` is consistent for the ATE if EITHER the
  outcome model OR the propensity model is correctly specified
  (the **double-robust** property). The function reports the
  point estimate plus an influence-function-based standard error.

Composing the two — running ``aipw_ate`` with cross-fitted
outcome and propensity nuisances — gives the Chernozhukov-style
DML ATE.

R `emmeans` does not provide either of these.

References
----------
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E.,
  Hansen, C., Newey, W., & Robins, J. (2018). Double/debiased
  machine learning for treatment and structural parameters.
  *The Econometrics Journal*, 21(1), C1-C68.
- Robins, J. M., Rotnitzky, A., & Zhao, L. P. (1994). Estimation
  of regression coefficients when some regressors are not always
  observed. *Journal of the American Statistical Association*,
  89(427), 846-866.
- Bang, H., & Robins, J. M. (2005). Doubly robust estimation in
  missing data and causal inference models. *Biometrics*, 61(4),
  962-973.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from dataclasses import replace as _dc_replace
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as _stats

__all__ = [
    "AIPWResult",
    "aipw_ate",
    "cross_fit_ml_emmeans",
]


@dataclass(frozen=True)
class AIPWResult:
    """Result of :func:`aipw_ate`.

    Attributes
    ----------
    estimate
        AIPW point estimate of the ATE ``E[Y(1) - Y(0)]``.
    se
        Influence-function-based standard error.
        ``SE = sqrt(Var(ψ) / n)``.
    lower_cl, upper_cl
        Confidence-interval bounds at level ``level``.
    df, level
        Wald-test degrees of freedom (set to ``n - 1`` per the
        influence-function asymptotics) and the nominal CI level.
    n
        Number of observations.
    n_clipped
        Number of propensity-score observations clipped to the
        ``weight_clip`` range.
    weight_clip
        The ``(low, high)`` clipping applied to propensity scores
        before inversion.
    """

    estimate: float
    se: float
    lower_cl: float
    upper_cl: float
    df: float
    level: float
    n: int
    n_clipped: int
    weight_clip: tuple[float, float]


def _validate_em_ml(em_ml: Any) -> None:
    from pymmeans.ml import MLEMMResult

    if not isinstance(em_ml, MLEMMResult):
        raise TypeError(
            f"expected an MLEMMResult (from ml_emmeans); got "
            f"{type(em_ml).__name__}."
        )


def aipw_ate(
    info: Any,
    *,
    propensity_predict_fn: Callable[[pd.DataFrame], np.ndarray],
    treatment: str = "treat",
    treatment_levels: tuple[Any, Any] = (0, 1),
    level: float = 0.95,
    weight_clip: tuple[float, float] = (0.025, 0.975),
) -> AIPWResult:
    """Doubly-robust AIPW estimate of the binary-treatment ATE.

    Combines an outcome model (supplied via ``info`` from
    :func:`pymmeans.from_predict`) with a propensity-score model
    in the augmented inverse-probability-weighting estimator
    (Robins, Rotnitzky & Zhao 1994). The estimator is **doubly
    robust**: consistent for the ATE if EITHER the outcome model
    OR the propensity model is correctly specified.

    Parameters
    ----------
    info
        An :class:`MLPredictInfo` from :func:`from_predict`. Its
        ``predict_fn`` must accept a DataFrame with the treatment
        column overridden to either level value and return
        per-row outcome predictions.
    propensity_predict_fn
        Callable ``data → P(T == treatment_levels[1] | X)``
        returning a 1-D array of probabilities.
    treatment
        Name of the treatment column in ``info.data``.
    treatment_levels
        ``(t0, t1)`` — the two treatment-level values to contrast.
        The estimand is ``E[Y(t1) - Y(t0)]``.
    level
        Nominal confidence level for the returned CI.
    weight_clip
        ``(low, high)`` clipping for the propensity score before
        inversion. Protects against extreme weights when
        ``π̂(x)`` is near 0 or 1. Default ``(0.025, 0.975)``.

    Returns
    -------
    AIPWResult
        Point estimate, SE, CI bounds, and diagnostics.

    Notes
    -----
    The standard error is computed from the empirical variance
    of the per-observation influence function::

        SE = sqrt( Var(ψ_i) / n )

    which is the standard sandwich-style SE for an M-estimator.
    The Wald CI uses ``df = n - 1`` (informative for n moderate;
    converges to the z-interval for large n).

    Both the outcome and propensity models must be fit on the
    same data as ``info.data``. For sample-splitting variants
    (Chernozhukov et al. 2018 DML), compose this function with
    :func:`cross_fit_ml_emmeans` (and refit the propensity in
    held-out folds in user code).
    """
    from pymmeans.ml import MLPredictInfo

    if not isinstance(info, MLPredictInfo):
        raise TypeError(
            f"aipw_ate expects an MLPredictInfo (from from_predict); "
            f"got {type(info).__name__}."
        )
    if not callable(propensity_predict_fn):
        raise TypeError(
            "propensity_predict_fn must be callable."
        )
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1); got {level!r}.")
    if not (0.0 < weight_clip[0] < weight_clip[1] < 1.0):
        raise ValueError(
            f"weight_clip must be (low, high) with 0 < low < high < 1; "
            f"got {weight_clip!r}."
        )
    if treatment not in info.data.columns:
        raise ValueError(
            f"treatment column {treatment!r} not in info.data; "
            f"available: {list(info.data.columns)}"
        )
    if info.response not in info.data.columns:
        raise ValueError(
            f"response column {info.response!r} not in info.data."
        )

    t0_val, t1_val = treatment_levels
    df = info.data
    n = len(df)
    if n < 2:
        raise ValueError("aipw_ate requires at least 2 observations.")

    # Predict under both counterfactuals.
    d0 = df.copy(); d0[treatment] = t0_val
    d1 = df.copy(); d1[treatment] = t1_val
    mu0_hat = np.asarray(info.predict_fn(d0), dtype=float)
    mu1_hat = np.asarray(info.predict_fn(d1), dtype=float)
    if mu0_hat.shape != (n,) or mu1_hat.shape != (n,):
        raise ValueError(
            f"predict_fn returned wrong shape; expected ({n},), got "
            f"mu0={mu0_hat.shape}, mu1={mu1_hat.shape}."
        )

    # Propensity score on observed data.
    p1_hat = np.asarray(propensity_predict_fn(df), dtype=float)
    if p1_hat.shape != (n,):
        raise ValueError(
            f"propensity_predict_fn returned shape {p1_hat.shape}, "
            f"expected ({n},)."
        )
    # Clip + count clipped.
    p_clipped = np.clip(p1_hat, weight_clip[0], weight_clip[1])
    n_clipped = int(np.sum(p1_hat != p_clipped))

    # Build T_i indicator and Y vector.
    T = np.asarray(df[treatment] == t1_val, dtype=float)
    Y = np.asarray(df[info.response], dtype=float)

    # Influence function.
    psi = (
        mu1_hat - mu0_hat
        + T * (Y - mu1_hat) / p_clipped
        - (1.0 - T) * (Y - mu0_hat) / (1.0 - p_clipped)
    )
    estimate = float(psi.mean())
    se = float(psi.std(ddof=1) / np.sqrt(n))

    alpha = 1.0 - level
    df_w = float(n - 1)
    tcrit = float(_stats.t.isf(alpha / 2.0, df_w))
    lower = estimate - tcrit * se
    upper = estimate + tcrit * se

    return AIPWResult(
        estimate=estimate,
        se=se,
        lower_cl=lower,
        upper_cl=upper,
        df=df_w,
        level=level,
        n=n,
        n_clipped=n_clipped,
        weight_clip=weight_clip,
    )


def cross_fit_ml_emmeans(
    info: Any,
    specs: str | list[str],
    *,
    refit_fn: Callable[[pd.DataFrame], Callable] | None = None,
    K: int = 5,
    seed: int = 0,
    by: str | list[str] | None = None,
    at: dict[str, Any] | None = None,
    level: float = 0.95,
) -> Any:
    """K-fold cross-fitted ml_emmeans.

    For each of K folds, refits the outcome model on the OTHER
    K-1 folds using ``refit_fn`` (or ``info.refit_fn``), then
    computes the cell-level marginal mean by averaging the
    held-out fold's predictions over the cell's covariate
    distribution. The K per-fold cell-mean estimates are
    averaged (weighted by fold size).

    This is the **sample-splitting** building block of
    Chernozhukov et al. (2018) double machine learning: every
    prediction used in the marginal mean is produced by a model
    that did NOT see the data point being predicted. It is the
    structural fix to in-sample overfit bias in nonparametric
    g-computation.

    Parameters
    ----------
    info
        :class:`MLPredictInfo` from :func:`from_predict`.
    specs
        Target factor name(s) for the marginal-means grid.
    refit_fn
        Callable ``refit_fn(data) -> new_predict_fn`` that
        refits the outcome model on a subset of training data
        and returns a fresh ``predict_fn``. If ``None``,
        ``info.refit_fn`` is used.
    K
        Number of cross-validation folds. Default 5.
    seed
        Random seed for the fold-shuffle.
    by, at, level
        Forwarded to :func:`pymmeans.ml_emmeans` for each
        per-fold call.

    Returns
    -------
    MLEMMResult
        With ``.frame`` carrying the **cross-fitted** cell-level
        marginal means. The dataclass field ``df_method`` is
        stamped ``"cross_fit"`` so downstream operations can
        detect this is a sample-split estimate.

    Notes
    -----
    * Cross-fitting reduces in-sample overfit bias and provides
      the *honest residual variance* that downstream sandwich
      / influence-function SEs need to be valid.
    * For an ATE through this API, follow up with
      :func:`pymmeans.ml_contrast`, or feed the per-fold-refit
      outcome and a held-out propensity into :func:`aipw_ate`
      for the full double-machine-learning ATE.

    Raises
    ------
    ValueError
        If ``K < 2``, if no ``refit_fn`` is available, or if any
        fold's refit fails (no silent fallback to naive
        prediction).
    """
    from sklearn.model_selection import KFold

    from pymmeans.ml import MLPredictInfo, ml_emmeans

    if not isinstance(info, MLPredictInfo):
        raise TypeError(
            f"expected an MLPredictInfo (from from_predict); got "
            f"{type(info).__name__}."
        )
    if K < 2:
        raise ValueError(f"K must be >= 2; got {K}.")
    if refit_fn is None:
        refit_fn = info.refit_fn
    if refit_fn is None:
        raise ValueError(
            "cross_fit_ml_emmeans requires a refit_fn (either passed "
            "directly or stored on info.refit_fn) — sample-splitting "
            "cannot reuse the original predict_fn."
        )
    if len(info.data) < K:
        raise ValueError(
            f"info.data has {len(info.data)} rows but K={K}; need "
            f"len(data) >= K."
        )

    kf = KFold(n_splits=K, shuffle=True, random_state=seed)
    fold_frames: list[pd.DataFrame] = []
    fold_sizes: list[int] = []
    idx = np.arange(len(info.data))

    for tr_idx, te_idx in kf.split(idx):
        # Keep the original row indices on the slices so users can
        # audit (in tests and downstream tools) which rows ended up
        # in which fold. This is the structural-correctness handle
        # that DML's "out-of-sample prediction" guarantee depends on.
        train_sub = info.data.iloc[tr_idx]
        test_sub = info.data.iloc[te_idx]
        new_predict = refit_fn(train_sub)
        # Accept either a callable predict_fn or a fitted model
        # exposing a ``.predict()`` method (matches the bootstrap_ci
        # convention).
        if hasattr(new_predict, "predict") and not callable(new_predict):
            new_predict_fn = new_predict.predict
        elif callable(new_predict) and hasattr(new_predict, "predict"):
            # Prefer the `.predict()` method over `__call__` for
            # sklearn-style fitted-model objects (matches the
            # bootstrap_ci ordering — see ml.py docstring).
            new_predict_fn = new_predict.predict
        elif callable(new_predict):
            new_predict_fn = new_predict
        else:
            raise TypeError(
                "refit_fn must return a callable or an object with "
                f".predict(); got {type(new_predict).__name__}."
            )
        # Build a fresh MLPredictInfo whose .data is the held-out
        # fold (so ml_emmeans's population average is over the
        # held-out distribution) but whose predict_fn was fit on
        # the OTHER folds.
        fold_info = _dc_replace(
            info,
            predict_fn=new_predict_fn,
            data=test_sub,
            factors=info.factor_levels,  # freeze level dict
        )
        fold_em = ml_emmeans(
            fold_info, specs,
            by=by, at=at, level=level,
        )
        fold_frames.append(fold_em.frame.copy())
        fold_sizes.append(len(test_sub))

    # Aggregate per-fold cell estimates → weighted average.
    base = fold_frames[0].copy()
    # Detect target columns (factors) by exclusion.
    non_target = {"emmean", "SE", "df", "lower_cl", "upper_cl"}
    target_cols = [c for c in base.columns if c not in non_target]
    # Align all fold frames on the same target-row ordering.
    def _sort_key(f):
        return f.sort_values(target_cols).reset_index(drop=True)
    aligned = [_sort_key(f) for f in fold_frames]

    weights = np.asarray(fold_sizes, dtype=float)
    weights = weights / weights.sum()
    emm_stack = np.stack([a["emmean"].to_numpy(dtype=float) for a in aligned])
    pooled_emm = (weights[:, None] * emm_stack).sum(axis=0)

    pooled = aligned[0].copy()
    pooled["emmean"] = pooled_emm
    # SE / CI: cross-fit doesn't have a closed-form per-cell SE the
    # way OLS+emmeans does. Leave SE/CI as NaN and let downstream
    # bootstrap or AIPW provide inference. Stamp df_method so this
    # is auditable.
    pooled["SE"] = np.nan
    pooled["df"] = np.inf
    pooled["lower_cl"] = np.nan
    pooled["upper_cl"] = np.nan

    # Build the output MLEMMResult by replacing the frame.
    result = _dc_replace(
        ml_emmeans(info, specs, by=by, at=at, level=level),
        frame=pooled,
        df_method="cross_fit",
    )
    return result
