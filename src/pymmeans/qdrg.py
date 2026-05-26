"""``qdrg`` / ``emmobj`` — hand-built reference grids without a fit.

Python equivalents of R `emmeans`'s ``qdrg()`` and
``emmobj()`` constructors. These let users compute EMMs on a
model that pymmeans doesn't otherwise know how to introspect — by
supplying the formula, data, coefficient vector, and vcov
directly.

Typical use cases:

- A custom-fitted model from a non-statsmodels framework where
  the user has ``β̂`` and ``V̂`` but no adapter.
- A published model where the user wants to compute EMMs from
  the printed coefficient table without refitting.
- A Bayesian posterior summary where the user wants Wald-style
  EMMs on the posterior mean (the more common path is
  :func:`pymmeans.posterior.posterior_emmeans` which uses
  percentile CIs from draws).

Notes
-----
- ``qdrg`` and R `emmeans`'s ``qdrg`` both take ``(formula, data,
  coef, vcov, df)`` arguments and build a "ModelInfo" stand-in.
  ``emmobj`` is a lower-level constructor that R uses internally;
  in pymmeans the same role is filled by directly constructing a
  :class:`pymmeans.utils.ModelInfo`.
- The returned object can be passed to any function that accepts
  a :class:`ModelInfo`: :func:`emmeans`, :func:`emtrends`,
  :func:`pairs`, :func:`contrast`, etc.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pymmeans.utils import ModelInfo


def qdrg(
    formula: str,
    data: pd.DataFrame,
    coef: np.ndarray | pd.Series,
    vcov: np.ndarray | pd.DataFrame,
    df: float = np.inf,
    *,
    response_name: str = "y",
    family: Any | None = None,
    scale: float = 1.0,
) -> ModelInfo:
    """Build a :class:`ModelInfo` from raw coefficient + vcov inputs.

    Python equivalent of R's ``qdrg(formula, data, coef, vcov, df)``.

    Parameters
    ----------
    formula
        Patsy-style RHS-only formula (e.g. ``"x1 + g"``) or a full
        ``"y ~ x1 + g"`` formula. Used to build ``design_info`` via
        :func:`patsy.dmatrices`. The LHS, if present, must be a
        column in ``data`` (any numeric value is fine — the LHS is
        only used to drive patsy's design construction). If the
        formula is RHS-only, an ``Intercept`` is added by default.
    data
        The DataFrame used to evaluate the design. Factor levels
        and numeric means are computed from this frame.
    coef
        Coefficient vector ``β̂`` of length ``p`` matching the design
        matrix produced by the formula (in column order — see
        ``design_info.column_names``). ``pd.Series`` is accepted
        for self-documentation.
    vcov
        ``p × p`` covariance matrix of ``β̂``.
    df
        Residual degrees of freedom. Defaults to ``inf`` (Wald-z
        tests). Pass an integer for t-tests at that df.
    response_name
        Display label for the response column on EMM frames.
        Defaults to ``"y"``.
    family
        Optional GLM family object (e.g. ``sm.families.Binomial()``)
        if the user wants ``emmeans(..., type="response")`` to apply
        the inverse-link transformation. Defaults to ``None``
        (Gaussian / identity).
    scale
        Residual variance for OLS-style fits, dispersion for GLM.
        Defaults to ``1.0``.

    Returns
    -------
    ModelInfo
        Carries the supplied ``β̂``, ``V̂``, ``df``, design_info, and
        data. Pass to any ``pymmeans`` entry point.

    Examples
    --------
    >>> import numpy as np # doctest: +SKIP
    >>> import pandas as pd # doctest: +SKIP
    >>> from pymmeans import qdrg, emmeans # doctest: +SKIP
    >>> df = pd.DataFrame({ # doctest: +SKIP
    ... "y": [0.0] * 6,
    ... "g": ["a", "b", "c"] * 2,
    ... })
    >>> info = qdrg( # doctest: +SKIP
    ... "g", df, coef=np.array([1.0, 0.5, -0.3]),
    ... vcov=np.diag([0.04, 0.04, 0.04]), df=20,
    ... )
    >>> emmeans(info, "g").frame # doctest: +SKIP
    """
    # Parse formula via patsy. Accept either RHS-only or
    # full y ~ ... form. We need a placeholder LHS so dmatrices
    # produces a design.
    from patsy import ModelDesc, dmatrices

    formula_s = formula.strip()
    if "~" not in formula_s:
        # RHS-only: synthesize a dummy LHS using the first column
        # of data (any numeric column works).
        if response_name in data.columns:
            lhs_col = response_name
        elif len(data.columns) > 0:
            lhs_col = str(data.columns[0])
        else:
            raise ValueError(
                "qdrg: cannot synthesize an LHS — `data` has no "
                "columns. Pass a full formula like `'y ~ ...'`."
            )
        full_formula = f"{lhs_col} ~ {formula_s}"
    else:
        full_formula = formula_s

    _ = ModelDesc.from_formula(full_formula) # validate syntax
    _, X_design = dmatrices(full_formula, data, return_type="dataframe")
    design_info = X_design.design_info

    coef_arr = np.asarray(coef, dtype=float).ravel()
    vcov_arr = np.asarray(vcov, dtype=float)
    if vcov_arr.ndim != 2 or vcov_arr.shape[0] != vcov_arr.shape[1]:
        raise ValueError(
            f"qdrg: vcov must be a square matrix, got shape {vcov_arr.shape}."
        )
    if vcov_arr.shape[0] != coef_arr.size:
        raise ValueError(
            f"qdrg: coef length ({coef_arr.size}) and vcov size "
            f"({vcov_arr.shape[0]}) must match."
        )
    # validate symmetry. A non-symmetric
    # vcov (e.g. from a hand-pasted half-symmetric matrix or a
    # published correlation table with the lower-triangular cells
    # omitted) would silently produce wrong SEs. Symmetrise within
    # a small tolerance, reject otherwise.
    if not np.allclose(vcov_arr, vcov_arr.T, atol=1e-10, rtol=1e-8):
        raise ValueError(
            "qdrg: vcov must be symmetric "
            f"(max asymmetry |V - V'| = "
            f"{float(np.max(np.abs(vcov_arr - vcov_arr.T))):.3e}). "
            "Check that you passed a full covariance matrix, not a "
            "lower- or upper-triangular slice with the other half "
            "implicit."
        )
    # Symmetrise to clean up FP noise before downstream operations.
    vcov_arr = 0.5 * (vcov_arr + vcov_arr.T)
    if len(design_info.column_names) != coef_arr.size:
        raise ValueError(
            f"qdrg: formula produces {len(design_info.column_names)} "
            f"design columns ({design_info.column_names}) but coef "
            f"has {coef_arr.size} entries. Make sure the formula "
            "matches the coef vector's order."
        )

    # Build factors / numeric_means from the design.
    factors: dict[str, list[str]] = {}
    numeric_means: dict[str, float] = {}
    for factor, fi in design_info.factor_infos.items():
        name = factor.name()
        if fi.type == "categorical":
            factors[name] = list(fi.categories)
        elif name in data.columns:
            numeric_means[name] = float(data[name].mean())

    return ModelInfo(
        beta=coef_arr,
        vcov=vcov_arr,
        param_names=list(design_info.column_names),
        factors=factors,
        numeric_means=numeric_means,
        df_resid=float(df),
        design_info=design_info,
        data=data,
        response_name=response_name,
        family=family,
        scale=float(scale),
        raw_result=None,
    )


def emmobj(
    bhat: np.ndarray,
    V: np.ndarray,
    levels: dict[str, list[Any]] | None = None,
    *,
    linfct: np.ndarray | None = None,
    df: float = np.inf,
    response_name: str = "y",
    family: Any | None = None,
) -> ModelInfo:
    """Low-level constructor — Python equivalent of R's ``emmobj``.

    Builds a :class:`ModelInfo` from ``bhat`` + ``V`` + a dict of
    factor levels (no formula needed). Useful when you have a
    coefficient vector but no formula API — typical case is when
    porting a published table.

    Parameters
    ----------
    bhat
        Coefficient vector ``β̂`` of length ``p``.
    V
        ``p × p`` covariance matrix.
    levels
        ``{factor_name: [level0, level1, ...]}`` for each
        categorical predictor. Required if you want
        :func:`pymmeans.emmeans` to be able to enumerate factor
        levels at the reference grid. Pass ``None`` to leave the
        factors dict empty (numeric-only design).
    linfct
        Optional pre-built linfct matrix. Currently unused
        (pymmeans builds the marginal design from ``design_info``
        on demand); kept for R compatibility.
    df, response_name, family
        Same as :func:`qdrg`.

    Notes
    -----
    ``emmobj`` does NOT supply a ``design_info`` — many pymmeans
    paths require one to construct the reference grid. Users
    wanting full functionality should call :func:`qdrg` instead
    (which accepts a formula and builds ``design_info`` internally
    via patsy).
    """
    bhat_arr = np.asarray(bhat, dtype=float).ravel()
    V_arr = np.asarray(V, dtype=float)
    if V_arr.shape != (bhat_arr.size, bhat_arr.size):
        raise ValueError(
            f"emmobj: V shape {V_arr.shape} must match "
            f"(len(bhat), len(bhat)) = ({bhat_arr.size}, {bhat_arr.size})."
        )
    factors = dict(levels) if levels else {}

    # Without a formula we don't have a design_info; many emmeans
    # paths will fail without one. Use a sentinel "_emmobj_origin"
    # column on the data DataFrame so the emmeans() error path can
    # distinguish emmobj-origin (no design_info ever existed) from
    # pickle-origin (design_info was dropped during pickling).
    return ModelInfo(
        beta=bhat_arr,
        vcov=V_arr,
        param_names=[f"x{i}" for i in range(bhat_arr.size)],
        factors=factors,
        numeric_means={},
        df_resid=float(df),
        design_info=None,
        data=pd.DataFrame({"_emmobj_origin": []}),
        response_name=response_name,
        family=family,
        scale=1.0,
        raw_result=None,
    )
