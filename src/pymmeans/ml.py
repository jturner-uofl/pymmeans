"""ML-model adapter — marginal means for scikit-learn / xgboost / any
``.predict()``-compatible model. **marquee differentiator.**

R ``emmeans`` is tied to R's S4 model-class system and only supports
models that expose a linear-predictor + vcov pair. ML models (random
forests, gradient boosting, neural nets, ...) don't have a tractable
β / V representation, so they're entirely outside R `emmeans`'s scope.

pymmeans bridges this via **prediction-surface averaging**
(g-computation): the EMM at level ``X`` of factor ``g`` is the
average of ``predict_fn(D')`` over the training data ``D`` with the
``g`` column overridden to ``X``. This is the same population-level
"marginal effect" concept; only the variance-quantification path
differs (case bootstrap rather than the linear-predictor delta
method).

Workflow (schematic — see the README "Beyond R parity — ML
adapter example" section for a self-contained, runnable version
with fixed random seed and the exact expected output)::

    from sklearn.ensemble import RandomForestRegressor
    from pymmeans import from_predict, ml_emmeans, bootstrap_ci

    # 1) Train any model with a `.predict()` method.
    rf = RandomForestRegressor(random_state=0).fit(X_train, y_train)

    # 2) Wrap the predict callable + the training-data fixture.
    info = from_predict(
        predict_fn=rf.predict,
        data=df,
        factors=["treatment", "site"],
        numerics=["age", "dose"],
        refit_fn=lambda sample: ..., # see README for a full example
    )

    # 3) Marginal means via prediction-surface averaging.
    em = ml_emmeans(info, "treatment")

    # 4) Bootstrap CIs (kind="case" refits per resample).
    em_ci = bootstrap_ci(em, kind="case", n_samples=500, seed=0)

References
----------
- Robins (1986). G-computation framework for causal inference.
- Hernán & Robins (2020). *Causal Inference: What If*, ch. 13
  (g-formula) — the prediction-surface averaging used here.
- Greenland & Pearce (2015). Statistical Foundations for Model-Based
  Adjustments. *Annual Review of Public Health*, 36.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MLPredictInfo:
    """Lightweight model-info shim for any ``.predict()``-compatible model.

    Holds the prediction callable plus the original data used for
    population-average EMM computation. Designed to be passed to
    :func:`ml_emmeans` in place of a fitted statsmodels result.

    Attributes
    ----------
    predict_fn
        Callable ``predict_fn(data: pd.DataFrame) -> ndarray`` (or
        anything that takes ``data[feature_cols]`` and returns a
        prediction vector). The callable is used at every grid cell
        to compute population-average predictions.
    data
        Training data the model was fit on. EMMs are computed by
        averaging predictions over the rows of this DataFrame with
        the target columns overridden to each level value. Must
        include all columns the model uses.
    factors
        Names of categorical factor columns. ``ml_emmeans`` treats
        these as the universe of possible levels. Either a list of
        names (levels inferred from ``data[col].unique()``) or a
        dict mapping name → ordered list of levels.
    numerics
        Names of numeric columns the model uses. Optional; default
        is all non-factor columns of ``data`` other than ``response``.
    response
        Name of the response column (used for documentation only;
        ``predict_fn`` is the actual source of truth).
    refit_fn
        Optional callable ``refit_fn(data) -> new_predict_fn`` for
        case-bootstrap support. If supplied,
        ``bootstrap_ci(em, kind='case')`` will use it to recompute
        EMMs on each resample. If ``None``, bootstrap support
        requires passing ``refit_fn=`` directly to ``bootstrap_ci``.
    """

    predict_fn: Callable
    data: pd.DataFrame
    factors: list[str] | dict[str, list]
    numerics: list[str] = field(default_factory=list)
    response: str = "y"
    refit_fn: Callable | None = None

    @property
    def factor_levels(self) -> dict[str, list]:
        """Per-factor ordered level list. If ``factors`` was a list,
        infer levels from ``data[col]``.

        for ``pd.Categorical`` columns, honour the
        declared categories (including levels that aren't observed
        in the data — matches R's ``factor(x, levels=...)``
        semantics). pymmeans previously called ``unique()`` and
        silently dropped declared-but-unobserved levels.
        """
        if isinstance(self.factors, dict):
            return dict(self.factors)
        out: dict[str, list] = {}
        for f in self.factors:
            s = self.data[f]
            if isinstance(s.dtype, pd.CategoricalDtype):
                # Categorical: preserve the declared category order
                # AND keep unobserved categories.
                out[f] = list(s.cat.categories)
            else:
                out[f] = sorted(s.dropna().unique().tolist())
        return out

    @property
    def factor_names(self) -> list[str]:
        return (
            list(self.factors.keys())
            if isinstance(self.factors, dict)
            else list(self.factors)
        )


def from_predict(
    predict_fn: Callable,
    data: pd.DataFrame,
    factors: list[str] | dict[str, list],
    numerics: list[str] | None = None,
    response: str = "y",
    refit_fn: Callable | None = None,
) -> MLPredictInfo:
    """Build an :class:`MLPredictInfo` for use with
    :func:`ml_emmeans`, :func:`ml_pairs`, and :func:`bootstrap_ci`.

    Parameters
    ----------
    predict_fn
        Any callable accepting a DataFrame (or ndarray) and returning
        a 1-D prediction vector. Most sklearn / xgboost / lightgbm
        models satisfy this via ``model.predict``.
    data
        The training data (or any representative DataFrame) the
        model was fit on. EMMs marginalize predictions over its rows
        with target columns overridden.
    factors
        Categorical predictor names. List form: levels inferred from
        unique values. Dict form: explicit ordered levels.
    numerics
        Numeric predictor names. Default: all non-factor columns
        except ``response``.
    response
        Response column name (informational; ``predict_fn`` is the
        actual source).
    refit_fn
        Optional. Callable ``refit_fn(resampled_data) -> new_predict_fn``
        for bootstrap support. Make sure to use a fresh model
        instance inside (don't reuse a global one).

    Returns
    -------
    MLPredictInfo

    Examples
    --------
    The doctest is ``+SKIP``-marked: it depends on an actual sklearn
    fit (``GradientBoostingRegressor``) and on a function definition
    with body — both of which the doctest collector can't construct
    in-place without the heavier setup. The snippet below is the
    canonical usage recipe.

    >>> from sklearn.ensemble import GradientBoostingRegressor  # doctest: +SKIP
    >>> import pandas as pd, numpy as np  # doctest: +SKIP
    >>> rng = np.random.default_rng(0)  # doctest: +SKIP
    >>> df = pd.DataFrame({  # doctest: +SKIP
    ...     "treatment": pd.Categorical(["A", "B", "C"] * 50),
    ...     "age": rng.normal(size=150),
    ...     "y": rng.normal(size=150),
    ... })  # doctest: +SKIP
    >>> X = pd.get_dummies(df[["treatment", "age"]], drop_first=True)  # doctest: +SKIP
    >>> gbm = GradientBoostingRegressor().fit(X, df["y"])  # doctest: +SKIP
    >>> def predict(d):  # doctest: +SKIP
    ...     X_new = pd.get_dummies(d[["treatment", "age"]], drop_first=True)
    ...     return gbm.predict(X_new[X.columns])
    >>> info = from_predict(  # doctest: +SKIP
    ...     predict, df, factors=["treatment"], numerics=["age"], response="y",
    ... )
    """
    if numerics is None:
        factor_names = (
            list(factors.keys()) if isinstance(factors, dict) else list(factors)
        )
        numerics = [
            c for c in data.columns
            if c not in factor_names and c != response
            and pd.api.types.is_numeric_dtype(data[c])
        ]
    return MLPredictInfo(
        predict_fn=predict_fn,
        data=data,
        factors=factors,
        numerics=numerics,
        response=response,
        refit_fn=refit_fn,
    )


def ml_emmeans(
    info: MLPredictInfo,
    specs: str | list[str],
    by: str | list[str] | None = None,
    at: dict[str, Any] | None = None,
    level: float = 0.95,
) -> MLEMMResult:
    """Marginal means for ML models via prediction-surface averaging.

    For each cell of the target × by grid, this overrides the
    target / by columns of ``info.data`` to the cell values, calls
    ``info.predict_fn`` on the modified DataFrame, and reports the
    mean prediction as the EMM at that cell. Numeric columns named
    in ``at=`` are also overridden; non-target numerics are
    marginalized by leaving them at their observed values
    (population-average / g-computation semantics).

    Returns an :class:`MLEMMResult` (a thin pandas-first wrapper)
    rather than the full ``EMMResult`` — ML EMMs don't have a
    linear-predictor / vcov pair, so the contrast machinery that
    needs ``linfct @ vcov @ linfct.T`` doesn't apply directly. Use
    :func:`bootstrap_ci` (``kind="case"``) for SEs and CIs.

    Parameters
    ----------
    info
        ``MLPredictInfo`` from :func:`from_predict`.
    specs
        Target factor name or list (cross product). Must be in
        ``info.factor_names``.
    by
        Optional by-grouping factor(s). One EMM per (by-cell, target-cell).
    at
        Optional ``{column: [value, ...]}`` to override numerics
        or restrict factor levels. Single-element lists pin the
        column; multi-element lists cross-product with the target.
    level
        Stored for downstream CI computation (no effect on point
        estimates here).

    Returns
    -------
    MLEMMResult
    """
    target = [specs] if isinstance(specs, str) else list(specs)
    by_list = (
        [by] if isinstance(by, str) else (list(by) if by else [])
    )

    # refuse empty data — marginal means over zero
    # rows is undefined.
    if len(info.data) == 0:
        raise ValueError(
            "ml_emmeans requires info.data to have at least one row; "
            "got an empty DataFrame."
        )

    factor_levels = info.factor_levels
    for f in target + by_list:
        if f not in factor_levels:
            raise ValueError(
                f"{f!r} is not in info.factor_names "
                f"({list(factor_levels)}). For numeric grid points "
                "pass them via at={'name': [values...]}."
            )

    # Build cell grid: cross product of (target levels) × (by levels) ×
    # (at-overrides for any columns the user pinned).
    at = at or {}
    grid_cols = list(target) + list(by_list)
    grid_lvls = [factor_levels[c] for c in grid_cols]
    at_extra = [(col, vals) for col, vals in at.items() if col not in grid_cols]

    # validate at= column names match info.data columns
    # (otherwise pins silently no-op and users wonder why their override
    # didn't take effect).
    for col, _vals in at.items():
        if col not in info.data.columns:
            raise ValueError(
                f"at={{{col!r}: ...}} references a column not in "
                f"info.data; known columns: {list(info.data.columns)}."
            )

    import itertools as _it

    # Numerics in `at` with single values pin; with multiple values
    # cross-product into the grid (each value becomes a separate cell).
    multi_at = [(col, vals) for col, vals in at_extra if len(vals) > 1]
    pin_at = {col: vals[0] for col, vals in at_extra if len(vals) == 1}
    multi_at_cols = [c for c, _ in multi_at]
    multi_at_lvls = [v for _, v in multi_at]

    all_combos = list(
        _it.product(*grid_lvls, *multi_at_lvls)
    ) if (grid_lvls or multi_at_lvls) else [()]

    all_cols = list(grid_cols) + list(multi_at_cols)
    cells_df = pd.DataFrame(all_combos, columns=all_cols)

    emmeans_arr = np.empty(len(cells_df), dtype=float)
    base_data = info.data.copy()
    # Apply pin_at globally (all rows of base_data).
    for col, val in pin_at.items():
        if col in base_data.columns:
            base_data[col] = val

    for i, row in cells_df.iterrows():
        df_cell = base_data.copy()
        for col in all_cols:
            df_cell[col] = row[col]
        preds = info.predict_fn(df_cell)
        preds_arr = np.asarray(preds, dtype=float)
        # refuse 2-D / multi-output predict returns.
        # The mean across a (n, k) array silently averages every
        # output column too — almost always not what the user wants.
        # Common gotcha: passing ``classifier.predict_proba`` instead
        # of ``lambda d: classifier.predict_proba(d)[:, 1]``.
        if preds_arr.ndim > 1:
            raise ValueError(
                f"predict_fn returned a {preds_arr.ndim}-D array of "
                f"shape {preds_arr.shape}; expected a 1-D vector. "
                "For multi-output models, wrap the callable to select "
                "the column you want, e.g. "
                "``lambda d: model.predict_proba(d)[:, 1]``."
            )
        emmeans_arr[i] = float(np.mean(preds_arr))

    out_frame = cells_df.copy()
    out_frame["emmean"] = emmeans_arr
    out_frame["SE"] = np.nan
    out_frame["df"] = np.inf
    out_frame["lower_cl"] = np.nan
    out_frame["upper_cl"] = np.nan

    return MLEMMResult(
        frame=out_frame,
        ml_info=info,
        target=target,
        by=by_list,
        level=level,
        at=dict(at),
    )


@dataclass(frozen=True)
class MLEMMResult:
    """ML-model marginal means result.

    Tidy DataFrame-first design. Point estimates come from
    prediction-surface averaging; SEs and CIs are NaN until
    :func:`bootstrap_ci` populates them.

    Pickling caveat
    ---------------
    ``MLEMMResult`` holds a reference to ``MLPredictInfo`` which
    holds the user-supplied ``predict_fn`` callable. Lambdas and
    closures are **not picklable**. To persist results to disk,
    write ``.frame`` (a pandas DataFrame) directly via
    ``frame.to_csv(...)`` / ``frame.to_parquet(...)``. To re-run
    bootstrap inference later, either pickle the trained sklearn
    model separately and rebuild the predict_fn on load, or pass a
    module-level (non-lambda) ``predict_fn`` so the wrapper picks
    up the right reference under ``pickle.dumps``.
    """

    frame: pd.DataFrame
    ml_info: MLPredictInfo
    target: list[str]
    by: list[str]
    level: float
    at: dict[str, Any]
    df_method: str = "default"
    """``"default"`` for fresh ML EMMs;
    ``"bootstrap"`` after :func:`pymmeans.bootstrap_ci` so the
    summary / confint / test layer can recognise percentile-CI
    objects and refuse / preserve appropriately. Mirrors
    :attr:`EMMResult.df_method`. Without this stamp, the
    bootstrap-recompute lockdown bypassed the ML path entirely:
    ``summary(ml_em_b)`` silently overwrote percentile CIs with
    asymptotic Wald CIs and emitted z-tests."""

    def __repr__(self) -> str: # pragma: no cover - cosmetic
        return f"MLEMMResult({len(self.frame)} rows, target={self.target})\n{self.frame!r}"


def ml_pairs(
    em: MLEMMResult,
    reverse: bool = False,
) -> pd.DataFrame:
    """All pairwise differences of an :class:`MLEMMResult`.

    For each by-group, produces ``k*(k-1)/2`` contrasts of the EMM
    levels. Point estimates are simple differences of the EMM column;
    SEs / CIs are NaN until populated by
    :func:`bootstrap_ci` ``(kind="case")`` on the EMM (then the same
    function on the pair frame, or equivalently re-derive pairs from
    bootstrap-CI-populated EMMs).
    """
    if not isinstance(em, MLEMMResult):
        raise TypeError(
            f"ml_pairs requires MLEMMResult; got {type(em).__name__}."
        )
    # refuse a bootstrap-stamped ML EMM — sibling
    # of the refusal on the statsmodels ``pairs(em_b)``
    # path. The stored ``frame['lower_cl']`` / ``frame['upper_cl']``
    # are percentile bootstrap intervals on the EMM grid; ``ml_pairs``
    # would silently compute contrast point estimates as raw
    # differences and report NaN SE/CIs, which the user might confuse
    # with "bootstrap CIs on the contrast" (they're not — they're
    # uninitialised). The correct workflow is
    # ``bootstrap_ci(ml_pairs(raw_em))`` to bootstrap the contrasts.
    if getattr(em, "df_method", "default") == "bootstrap":
        raise ValueError(
            "ml_pairs() is not defined for a bootstrap-derived "
            "MLEMMResult (df_method='bootstrap'). The stored CIs are "
            "percentile bootstrap intervals on the EMM grid; deriving "
            "pairwise contrasts here would emit raw differences with "
            "NaN SEs that the user could confuse with bootstrap "
            "contrast inference. Correct workflow:\n"
            " ml_pairs(em) # build the pairwise frame first\n"
            " bootstrap_ci(<bootstrap on EMM, then pairs>) — case "
            "bootstrap of contrasts is not yet implemented for ML; "
            "compose ``ml_pairs(em)`` after bootstrap-CI-populating "
            "the EMM and accept the diff-of-percentile point estimates."
        )

    pieces = []
    by_cols = em.by
    target_cols = em.target
    frame = em.frame

    if by_cols:
        for by_vals, grp in frame.groupby(by_cols, sort=False):
            if not isinstance(by_vals, tuple):
                by_vals = (by_vals,)
            sub = grp.reset_index(drop=True)
            pieces.append(_pairs_within(sub, target_cols, by_cols, by_vals, reverse))
    else:
        pieces.append(_pairs_within(frame.reset_index(drop=True),
                                     target_cols, [], (), reverse))

    return pd.concat(pieces, ignore_index=True)


def _pairs_within(sub, target_cols, by_cols, by_vals, reverse):
    rows = []
    k = len(sub)
    for i in range(k):
        for j in range(i + 1, k):
            i_lab = " ".join(str(sub.iloc[i][c]) for c in target_cols)
            j_lab = " ".join(str(sub.iloc[j][c]) for c in target_cols)
            if reverse:
                lab = f"{j_lab} - {i_lab}"
                est = sub.iloc[j]["emmean"] - sub.iloc[i]["emmean"]
            else:
                lab = f"{i_lab} - {j_lab}"
                est = sub.iloc[i]["emmean"] - sub.iloc[j]["emmean"]
            row = {"contrast": lab, "estimate": est}
            for col, val in zip(by_cols, by_vals, strict=True):
                row[col] = val
            row["SE"] = np.nan
            row["df"] = np.inf
            row["lower_cl"] = np.nan
            row["upper_cl"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def ml_contrast(
    em: MLEMMResult,
    method: str | dict[str, list[float]] | np.ndarray = "pairwise",
    ref: int | str | None = None,
) -> pd.DataFrame:
    """Generic contrasts on an :class:`MLEMMResult`.

    (beyond R `emmeans` parity, ML-flavoured). Mirrors
    the linear-model :func:`pymmeans.contrast` API for ML EMMs. The
    contrast is applied to the **point estimates** (no SE / CI —
    those require :func:`bootstrap_ci` ``kind="case"``).

    Parameters
    ----------
    em
        ML EMM result from :func:`ml_emmeans`.
    method
        Either a string method name (``"pairwise"`` (default),
        ``"revpairwise"``, ``"trt.vs.ctrl"`` / ``"trt.vs.ctrl1"`` /
        ``"trt.vs.ctrlk"``, ``"poly"``, ``"consec"``, ``"eff"``,
        ``"del.eff"``, ``"mean_chg"``, ``"identity"``, ``"helmert"``)
        OR a dict ``{name: [coefs]}`` OR an ndarray of shape
        ``(n_contrasts, n_levels)`` for custom contrasts.
    ref
        Reference level for ``trt.vs.ctrl`` (name or index).

    Returns
    -------
    pandas.DataFrame
        Per-contrast frame with ``contrast``, ``estimate``, plus
        the by-columns when ``em.by`` is non-empty. SE / df /
        lower_cl / upper_cl are NaN; pass the EMM through
        ``bootstrap_ci(..., kind="case")`` first for proper CIs.

    Notes
    -----
    Pairwise contrasts go through :func:`ml_pairs` (which has the
    same semantic). All other methods rebuild a coefficient matrix
    from the per-factor builders in :mod:`pymmeans.contrasts` and
    apply it row-wise within each by-group of the EMM frame.
    """
    if isinstance(em, pd.DataFrame):
        raise TypeError(
            "ml_contrast expects an MLEMMResult; got a DataFrame "
            "(did you mean to pass the ML EMM, not its .frame?)."
        )
    if not isinstance(em, MLEMMResult):
        raise TypeError(
            f"ml_contrast expects MLEMMResult; got {type(em).__name__}."
        )
    # refuse a bootstrap-stamped ML EMM — sibling
    # of the refusal on the statsmodels ``contrast(em_b)``
    # path. The stored CIs are percentile bootstrap intervals on the
    # EMM grid; deriving a contrast here emits raw differences with
    # NaN SEs.
    if getattr(em, "df_method", "default") == "bootstrap":
        raise ValueError(
            "ml_contrast() is not defined for a bootstrap-derived "
            "MLEMMResult (df_method='bootstrap'). Same refusal logic "
            "as ``ml_pairs(em_b)``; the stored CIs are percentile "
            "bootstrap intervals on the EMM grid, not on contrasts. "
            "Compute contrasts on the raw (un-bootstrapped) EMM first."
        )

    # Shortcut: pairwise → reuse ml_pairs.
    # guard with `isinstance(method, str)` first —
    # ``method == "pairwise"`` on a numpy ndarray raises
    # "truth value of an array with more than one element is
    # ambiguous". The same gotcha bit `contrast()` historically and
    # is now guarded for in the ML adapter too.
    if isinstance(method, str) and method == "pairwise":
        return ml_pairs(em)
    if isinstance(method, str) and method == "revpairwise":
        return ml_pairs(em, reverse=True)

    # Build the coefficient matrix.
    from pymmeans.contrasts import (
        _consec_matrix,
        _del_eff_matrix,
        _eff_matrix,
        _helmert_matrix,
        _identity_matrix,
        _mean_chg_matrix,
        _poly_matrix,
        _trt_vs_ctrl_matrix,
    )

    # Per-by-group iteration: get rows for each by-cell.
    by_cols = em.by
    target_cols = em.target
    frame = em.frame
    rows_out: list[dict] = []

    if by_cols:
        groups = frame.groupby(by_cols, sort=False)
    else:
        groups = [((), frame)]

    for by_vals, grp in groups:
        if not isinstance(by_vals, tuple):
            by_vals = (by_vals,) if by_cols else ()
        sub = grp.reset_index(drop=True)
        # Level labels = concat of target columns
        labels = [
            " ".join(str(sub.iloc[i][c]) for c in target_cols)
            for i in range(len(sub))
        ]
        k = len(sub)
        if isinstance(method, str):
            m_lc = method.lower()
            if m_lc in ("trt.vs.ctrl", "trt.vs.ctrl1"):
                ref_idx = labels.index(ref) if isinstance(ref, str) else (ref or 0)
                D, names = _trt_vs_ctrl_matrix(k, labels, ref_idx)
            elif m_lc == "trt.vs.ctrlk":
                D, names = _trt_vs_ctrl_matrix(k, labels, k - 1)
            elif m_lc == "poly":
                D, names = _poly_matrix(k, labels)
            elif m_lc == "consec":
                D, names = _consec_matrix(k, labels)
            elif m_lc == "eff":
                D, names = _eff_matrix(k, labels)
            elif m_lc == "del.eff":
                D, names = _del_eff_matrix(k, labels)
            elif m_lc == "mean_chg":
                D, names = _mean_chg_matrix(k, labels)
            elif m_lc == "identity":
                D, names = _identity_matrix(k, labels)
            elif m_lc == "helmert":
                D, names = _helmert_matrix(k, labels)
            else:
                raise ValueError(
                    f"Unknown method {method!r} for ml_contrast. "
                    "Supported: pairwise, revpairwise, trt.vs.ctrl{,1,k}, "
                    "poly, consec, eff, del.eff, mean_chg, identity, "
                    "helmert, or a dict/ndarray of custom coefficients."
                )
        elif isinstance(method, dict):
            names = list(method.keys())
            D = np.asarray([method[n] for n in names], dtype=float)
            if D.shape[1] != k:
                raise ValueError(
                    f"Custom contrast length {D.shape[1]} does not match "
                    f"number of EMM rows per by-group ({k})."
                )
        elif isinstance(method, np.ndarray):
            D = np.asarray(method, dtype=float)
            if D.ndim == 1:
                D = D[None, :]
            if D.shape[1] != k:
                raise ValueError(
                    f"Custom contrast ndarray shape {D.shape} does not "
                    f"match k={k} EMM rows per by-group."
                )
            names = [f"c{i+1}" for i in range(D.shape[0])]
        else:
            raise TypeError(
                f"method must be a string, dict, or ndarray; got "
                f"{type(method).__name__}."
            )

        emm_vals = sub["emmean"].to_numpy(dtype=float)
        estimates = D @ emm_vals
        for nm, est in zip(names, estimates, strict=True):
            row = {"contrast": nm, "estimate": float(est)}
            for col, val in zip(by_cols, by_vals, strict=True):
                row[col] = val
            row["SE"] = np.nan
            row["df"] = np.inf
            row["lower_cl"] = np.nan
            row["upper_cl"] = np.nan
            rows_out.append(row)

    return pd.DataFrame(rows_out)


# ---------------------------------------------------------------------------
# Bootstrap marginal effects for black-box predictive models.
#
# ``avg_slopes`` / ``avg_comparisons`` on a parametric fit use the exact
# delta method via the coefficient covariance. A black-box ``predict_fn``
# (scikit-learn, gradient boosting, a neural net) exposes no coefficient
# covariance, so the only principled standard error is the *bootstrap*:
# resample the data, refit the model via ``refit_fn``, recompute the
# estimand, and take the spread across replicates. The point estimate uses
# numerical g-computation on ``predict_fn`` alone; a valid standard error
# additionally requires ``refit_fn`` (without it the standard error is NaN,
# matching the point-only behaviour of other black-box tools).
#
# Note on smoothness: ``ml_avg_slopes`` is a numerical derivative and is
# meaningful only for a *smooth* ``predict_fn`` (linear models, kernel
# methods, neural nets). Tree ensembles are piecewise constant, so their
# finite-difference derivative is zero almost everywhere and spikes at
# splits -- use ``ml_avg_comparisons`` (a discrete change) for those.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MLMarginalResult:
    """Result of :func:`ml_avg_slopes` or :func:`ml_avg_comparisons`.

    ``frame`` columns: the ``by`` factor(s) when present, ``var`` (and
    ``contrast`` for comparisons), the point estimate (``slope`` or
    ``estimate``), bootstrap ``SE``, asymptotic ``t_ratio`` / ``p_value``,
    and percentile-bootstrap ``lower_cl`` / ``upper_cl``. ``SE`` and the
    interval are ``NaN`` when no ``refit_fn`` was supplied.
    """

    frame: pd.DataFrame
    kind: str
    var: str
    n_boot: int
    level: float


def _ml_ordered_keys(data: pd.DataFrame, by_list: list) -> list:
    """By-group keys in factor-level order (categories / sorted unique)."""
    if not by_list:
        return [()]
    import itertools as _it

    levels = [
        list(data[c].cat.categories) if hasattr(data[c], "cat") else sorted(pd.unique(data[c]))
        for c in by_list
    ]
    present = set(
        map(tuple, data[by_list].astype(object).itertuples(index=False, name=None))
    )
    return [k for k in _it.product(*levels) if k in present]


def _ml_mask(data: pd.DataFrame, by_list: list, key: tuple) -> np.ndarray:
    if not by_list:
        return np.ones(len(data), dtype=bool)
    mask = np.ones(len(data), dtype=bool)
    for c, v in zip(by_list, key, strict=True):
        mask &= data[c].to_numpy() == v
    return mask


def _ml_bootstrap(
    info: MLPredictInfo, point: Callable, n_boot: int, seed: int
) -> np.ndarray | None:
    """Pairs bootstrap: resample rows, refit, recompute the estimand.

    Returns an ``(n_boot, K)`` array of replicate estimates, or ``None``
    when no ``refit_fn`` is available (point estimate only).
    """
    if info.refit_fn is None:
        return None
    n = len(info.data)
    rng = np.random.default_rng(seed)
    out = np.empty((n_boot, len(point(info.predict_fn, info.data))))
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        db = info.data.iloc[idx].reset_index(drop=True)
        out[b] = point(info.refit_fn(db), db)
    return out


def _ml_frame(
    est: np.ndarray,
    boot: np.ndarray | None,
    id_rows: list[dict],
    value_name: str,
    level: float,
) -> pd.DataFrame:
    import scipy.stats as _st

    k = len(est)
    if boot is not None:
        se = np.nanstd(boot, axis=0, ddof=1)
        lo = np.nanpercentile(boot, 100.0 * (1.0 - level) / 2.0, axis=0)
        hi = np.nanpercentile(boot, 100.0 * (1.0 + level) / 2.0, axis=0)
    else:
        se = lo = hi = np.full(k, np.nan)
    rows = []
    for gi in range(k):
        row = dict(id_rows[gi])
        row[value_name] = float(est[gi])
        row["SE"] = float(se[gi])
        with np.errstate(divide="ignore", invalid="ignore"):
            t = est[gi] / se[gi] if se[gi] > 0 else np.nan
        row["t_ratio"] = float(t)
        row["p_value"] = float(2.0 * _st.norm.sf(abs(t))) if np.isfinite(t) else np.nan
        row["lower_cl"] = float(lo[gi])
        row["upper_cl"] = float(hi[gi])
        rows.append(row)
    return pd.DataFrame(rows)


def ml_avg_slopes(
    info: MLPredictInfo,
    var: str,
    *,
    by: str | list | None = None,
    h: float | None = None,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> MLMarginalResult:
    """Average marginal effect of a numeric ``var`` for a black-box model.

    The point estimate is a numerical central-difference derivative of
    ``info.predict_fn`` averaged over the sample (optionally within ``by``
    groups). The standard error and percentile confidence interval come
    from a pairs bootstrap using ``info.refit_fn`` (``NaN`` if none).

    Only meaningful for a *smooth* ``predict_fn``; for tree ensembles use
    :func:`ml_avg_comparisons`.
    """
    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1); got {level!r}.")
    if var not in info.data.columns:
        raise ValueError(f"{var!r} is not a column of the model data.")
    if not pd.api.types.is_numeric_dtype(info.data[var]):
        raise ValueError(
            f"ml_avg_slopes requires a numeric var; {var!r} is "
            f"{info.data[var].dtype}. Use ml_avg_comparisons for a factor."
        )
    by_list = [by] if isinstance(by, str) else (list(by) if by else [])
    if h is None:
        h = max(1e-6, 1e-4 * float(np.std(info.data[var].to_numpy(dtype=float))))
    keys = _ml_ordered_keys(info.data, by_list)

    def point(pf: Callable, data: pd.DataFrame) -> np.ndarray:
        dp = data.copy(); dp[var] = data[var].to_numpy(dtype=float) + h
        dm = data.copy(); dm[var] = data[var].to_numpy(dtype=float) - h
        deriv = (np.asarray(pf(dp), dtype=float) - np.asarray(pf(dm), dtype=float)) / (2.0 * h)
        out = np.empty(len(keys))
        for gi, key in enumerate(keys):
            mask = _ml_mask(data, by_list, key)
            out[gi] = deriv[mask].mean() if mask.any() else np.nan
        return out

    est = point(info.predict_fn, info.data)
    boot = _ml_bootstrap(info, point, n_boot, seed)
    id_rows = [
        {**{name: key[bi] for bi, name in enumerate(by_list)}, "var": var}
        for key in keys
    ]
    frame = _ml_frame(est, boot, id_rows, "slope", level)
    return MLMarginalResult(frame=frame, kind="slope", var=var, n_boot=n_boot, level=level)


def ml_avg_comparisons(
    info: MLPredictInfo,
    var: str,
    *,
    comparison: str = "difference",
    step: float = 1.0,
    by: str | list | None = None,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> MLMarginalResult:
    """Average counterfactual comparison of ``var`` for a black-box model.

    g-computation point estimate (numeric: a centred ``step`` change;
    categorical: each level versus the reference) with a pairs-bootstrap
    standard error / percentile interval via ``info.refit_fn``. The
    ``comparison`` function is one of ``"difference"`` / ``"ratio"`` /
    ``"lnratio"`` / ``"lnor"`` / ``"lift"``.
    """
    from pymmeans.comparisons import _COMPARISONS, _reference_level

    if not (0.0 < level < 1.0):
        raise ValueError(f"level must be in (0, 1); got {level!r}.")
    if comparison not in _COMPARISONS:
        raise ValueError(
            f"comparison must be one of {sorted(_COMPARISONS)}; got {comparison!r}."
        )
    if var not in info.data.columns:
        raise ValueError(f"{var!r} is not a column of the model data.")
    cmp = _COMPARISONS[comparison]
    by_list = [by] if isinstance(by, str) else (list(by) if by else [])
    keys = _ml_ordered_keys(info.data, by_list)
    col = info.data[var]

    # Build the contrast list: (label, hi_assign, lo_assign), where each
    # *_assign maps a data frame to the counterfactual value of `var`.
    if pd.api.types.is_numeric_dtype(col):
        contrasts = [(
            f"+{step:g}",
            lambda d: d[var].to_numpy(dtype=float) + step / 2.0,
            lambda d: d[var].to_numpy(dtype=float) - step / 2.0,
        )]
    else:
        ref = _reference_level(col)
        levels = list(col.cat.categories) if hasattr(col, "cat") else sorted(pd.unique(col))
        cats = col.cat.categories if hasattr(col, "cat") else None

        def _fill(value, n):
            return pd.Categorical([value] * n, categories=cats) if cats is not None else [value] * n

        contrasts = [
            (
                f"{lvl} - {ref}",
                (lambda d, _l=lvl: _fill(_l, len(d))),
                (lambda d, _r=ref: _fill(_r, len(d))),
            )
            for lvl in levels
            if lvl != ref
        ]

    def point(pf: Callable, data: pd.DataFrame) -> np.ndarray:
        out = []
        for _label, hi_a, lo_a in contrasts:
            dh = data.copy(); dh[var] = hi_a(data)
            dl = data.copy(); dl[var] = lo_a(data)
            phi_hi = np.asarray(pf(dh), dtype=float)
            phi_lo = np.asarray(pf(dl), dtype=float)
            for key in keys:
                mask = _ml_mask(data, by_list, key)
                if mask.any():
                    out.append(cmp(phi_hi[mask].mean(), phi_lo[mask].mean()))
                else:
                    out.append(np.nan)
        return np.asarray(out)

    est = point(info.predict_fn, info.data)
    boot = _ml_bootstrap(info, point, n_boot, seed)
    id_rows = []
    for label, _hi, _lo in contrasts:
        for key in keys:
            row = {name: key[bi] for bi, name in enumerate(by_list)}
            row["term"] = var
            row["contrast"] = label
            id_rows.append(row)
    frame = _ml_frame(est, boot, id_rows, "estimate", level)
    return MLMarginalResult(frame=frame, kind="comparison", var=var, n_boot=n_boot, level=level)
