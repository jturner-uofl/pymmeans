"""Multivariate-response linear model EMMs + ``mvcontrast``.

Multivariate / repeated-measures support for marginal means. Mirrors R
``emmeans``'s handling of multivariate ``lm`` (a.k.a. ``mlm``: an ``lm``
fit with a matrix response ``cbind(y1, y2, ...) ~ x``).

What it does
------------

Given a statsmodels ``_MultivariateOLSResults`` fit
``Y (n × p) = X (n × k) B (k × p) + E`` with rows of ``E`` i.i.d.
``N(0, Σ)``:

* :func:`multivariate_emmeans` builds the **per-cell × per-response**
  EMM table — one row for every combination of the reference-grid cell
  (from ``specs``) and the response column (the ``rep_meas``
  pseudo-factor R introduces automatically). At cell ``x`` and response
  ``j`` the estimand is ``x' B[:, j]`` and the SE is
  ``√((x' (X'X)⁻¹ x) · Σ̂[j, j])`` (the conditional standard error,
  treating Σ as estimated).
* :func:`mvcontrast` applies a between-cell contrast (e.g. ``pairwise``
  over ``g``) and returns a **Hotelling T² / F test** *jointly across
  the p responses* per between-contrast:

  * ``T² = (L B) Σ̂⁻¹ (L B)' / (L (X'X)⁻¹ L')`` (scalar; ``L`` is the
    1 × k between-contrast row).
  * ``F = T² · df₂ / (df₁ · df_resid)``, with ``df₁ = p``,
    ``df₂ = df_resid − p + 1``.
  * Sidak-adjusted by default, matching R ``mvcontrast``.

The math is the small-sample Hotelling–Lawley one-row form (which here
coincides with Pillai / Wilks because the between-rank is 1).

Scope
-----

MVP slice: statsmodels :class:`_MultivariateOLS` only. Out of scope:
``AnovaRM`` long-format repeated measures, multivariate GLMs, mixed
multivariate-response models, brms multivariate posteriors — these
remain roadmap items.

References
----------
- Hotelling, H. (1931). The generalization of Student's ratio.
  *Annals of Mathematical Statistics*, 2(3), 360–378.
- Lenth, R. V. (2024). ``emmeans::mvcontrast`` reference.
- Searle, S. R. (1971). *Linear Models*, §5.7 (multivariate linear
  model; per-response EMMs match the marginal expectations).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from patsy import build_design_matrices
from scipy import stats


@dataclass
class MultivariateInfo:
    """Multivariate-OLS model metadata for marginal-means.

    Built by :func:`from_multivariate` from a fitted
    statsmodels ``_MultivariateOLSResults``. The four core quantities
    fully determine both the per-response EMMs and the multivariate
    ``mvcontrast`` Hotelling test:
    """

    B: np.ndarray            #: (k, p) coefficient matrix
    inv_cov: np.ndarray      #: (k, k) ``(X'X)^{-1}``
    Sigma_hat: np.ndarray    #: (p, p) residual covariance = sscpr / df_resid
    df_resid: int            #: residual df (n − k)
    endog_names: list[str]   #: response column names, length p
    exog_names: list[str]    #: fixed-effect parameter names, length k
    design_info: Any         #: patsy DesignInfo (for building L rows)


def from_multivariate(result: Any) -> MultivariateInfo:
    """Wrap a fitted ``_MultivariateOLSResults`` as a
    :class:`MultivariateInfo`.

    Parameters
    ----------
    result
        Output of
        ``statsmodels.multivariate.multivariate_ols._MultivariateOLS
        .from_formula(formula, data).fit()``. The fixed-effect
        parameters, ``(X'X)^{-1}``, residual SS-CP, and residual df
        are extracted from the result's ``_fittedmod`` tuple
        (statsmodels does not expose these via public attributes —
        only the ``mv_test`` MANOVA helper — but the four arrays are
        stable across statsmodels 0.13+).
    """
    if not hasattr(result, "_fittedmod"):
        raise TypeError(
            "from_multivariate: result must be a fitted "
            "statsmodels._MultivariateOLSResults; got "
            f"{type(result).__name__}."
        )
    B, df_resid, inv_cov, sscpr = result._fittedmod
    df_resid = int(df_resid)
    return MultivariateInfo(
        B=np.asarray(B, dtype=float),
        inv_cov=np.asarray(inv_cov, dtype=float),
        Sigma_hat=np.asarray(sscpr, dtype=float) / df_resid,
        df_resid=df_resid,
        endog_names=list(result.endog_names),
        exog_names=list(result.exog_names),
        design_info=result.design_info,
    )


@dataclass
class MultivariateEMM:
    """Long-format multivariate EMM result.

    The ``frame`` is a tidy DataFrame with one row per
    ``(cell × response)`` combination; columns are the specs factor(s),
    the ``rep_meas`` pseudo-factor naming the response, and
    ``emmean / SE / df / lower_cl / upper_cl``. The auxiliary fields
    expose the construction so :func:`mvcontrast` can run on it.
    """

    frame: pd.DataFrame
    info: MultivariateInfo
    specs: list[str]
    mult_name: str
    L_cells: np.ndarray  #: (n_cells, k) design matrix at the ref-grid cells
    cell_labels: pd.DataFrame  #: (n_cells, len(specs)) factor-level table


def _build_cells_grid(
    info: MultivariateInfo,
    data: pd.DataFrame,
    specs: list[str],
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build the cells DataFrame (factor combos in specs, numerics at
    sample mean) and the patsy design matrix evaluated at those cells.

    MVP simplification: every categorical factor in the design must
    appear in ``specs``. Non-specs columns must be numeric (covariates
    held at their sample mean from ``data``).
    """
    factor_infos = getattr(info.design_info, "factor_infos", {}) or {}
    # patsy factor names (the strings under each EvalFactor)
    cat_factors: dict[str, list] = {}
    for factor, fi in factor_infos.items():
        if getattr(fi, "type", None) == "categorical":
            cat_factors[factor.name()] = list(fi.categories)

    missing_cat = [c for c in cat_factors if c not in specs]
    if missing_cat:
        raise NotImplementedError(
            "multivariate_emmeans MVP: every categorical factor in the "
            f"design must be in `specs`. Missing: {missing_cat}. "
            "(non-specs categorical averaging is on the v0.3 roadmap.)"
        )

    # Build the long grid: cartesian product of the specs categorical levels.
    cell_rows: list[dict[str, Any]] = [{}]
    for s in specs:
        levels = cat_factors.get(s)
        if levels is None:
            # numeric spec — not supported in MVP
            raise NotImplementedError(
                "multivariate_emmeans MVP: numeric `specs` "
                f"({s!r}) are not yet supported."
            )
        cell_rows = [{**r, s: lev} for r in cell_rows for lev in levels]
    cells_df = pd.DataFrame(cell_rows)

    # Pad with numeric-at-mean columns the patsy formula needs.
    numeric_cols = [
        c for c in data.columns
        if c not in cells_df.columns and pd.api.types.is_numeric_dtype(data[c])
    ]
    for c in numeric_cols:
        cells_df[c] = float(data[c].mean())

    L_cells = np.asarray(
        build_design_matrices([info.design_info], cells_df)[0], dtype=float
    )
    return cells_df.loc[:, specs].reset_index(drop=True), L_cells


def multivariate_emmeans(
    result: Any,
    data: pd.DataFrame,
    specs: str | list[str],
    *,
    mult_name: str = "rep_meas",
    level: float = 0.95,
) -> MultivariateEMM:
    """Build per-cell × per-response marginal means for a multivariate-
    OLS fit.

    Parameters
    ----------
    result
        Fitted statsmodels ``_MultivariateOLSResults``.
    data
        The DataFrame used to fit ``result``. Required so non-specs
        numeric covariates can be held at their sample means.
    specs
        Factor name (or list) to compute EMMs over. Every categorical
        factor in the design must be listed in MVP.
    mult_name
        Name of the pseudo-factor that indexes the p response columns.
        R uses ``"rep.meas"``; we default to ``"rep_meas"`` for
        Pythonic naming (the user can override to match R).
    level
        Confidence-interval level. Default 0.95.

    Returns
    -------
    MultivariateEMM
        Long-format result with one row per ``(cell × response)`` and
        the auxiliary :class:`MultivariateInfo` + design matrix
        :func:`mvcontrast` needs.
    """
    if isinstance(specs, str):
        specs = [specs]
    info = from_multivariate(result)
    cell_labels, L_cells = _build_cells_grid(info, data, specs)
    p = len(info.endog_names)
    n_cells = L_cells.shape[0]

    emmean = L_cells @ info.B                   # (n_cells, p)
    cell_quad = np.einsum("ij,jk,ik->i", L_cells, info.inv_cov, L_cells)
    se = np.sqrt(np.outer(cell_quad, np.diag(info.Sigma_hat)))  # (n_cells, p)
    df = float(info.df_resid)
    tcrit = float(stats.t.ppf(0.5 + level / 2.0, df))

    # Long format: stack cells × responses.
    rows = []
    for i in range(n_cells):
        for j in range(p):
            row = {s: cell_labels.iloc[i][s] for s in specs}
            row[mult_name] = info.endog_names[j]
            row["emmean"] = float(emmean[i, j])
            row["SE"] = float(se[i, j])
            row["df"] = df
            row["lower_cl"] = row["emmean"] - tcrit * row["SE"]
            row["upper_cl"] = row["emmean"] + tcrit * row["SE"]
            rows.append(row)
    frame = pd.DataFrame(rows)
    return MultivariateEMM(
        frame=frame, info=info, specs=specs, mult_name=mult_name,
        L_cells=L_cells, cell_labels=cell_labels,
    )


def mvcontrast(
    emm: MultivariateEMM,
    method: str = "pairwise",
    adjust: str = "sidak",
) -> pd.DataFrame:
    """Multivariate (Hotelling) test of a between-cell contrast jointly
    across all responses.

    Parameters
    ----------
    emm
        Output of :func:`multivariate_emmeans`.
    method
        Between-contrast family applied to the cells of the specs
        factor. ``"pairwise"`` is the default (all unordered pairs).
        Currently supported: ``"pairwise"``, ``"trt.vs.ctrl"``,
        ``"trt.vs.ctrlk"``.
    adjust
        Multiplicity adjustment over the contrast family.
        ``"sidak"`` (R default), ``"bonferroni"``, or ``"none"``.

    Returns
    -------
    DataFrame
        One row per between-contrast with columns
        ``contrast / T_square / df1 / df2 / F_ratio / p_value``.
        ``df1 = p`` (responses), ``df2 = df_resid − p + 1``.

    Notes
    -----
    For a single-row (rank-1) between-contrast L:

    .. math::

        T^2 = \\frac{(L B) \\hat{\\Sigma}^{-1} (L B)^\\top}
                   {L (X^\\top X)^{-1} L^\\top}, \\quad
        F = T^2 \\cdot \\frac{df_2}{df_1 \\cdot df_{resid}}.

    This is the standard small-sample Hotelling form (Searle 1971,
    §5.7); for rank-1 contrasts it coincides with the Pillai / Wilks /
    Lawley-Hotelling F because all four MANOVA criteria collapse to
    the same statistic at q = 1.
    """
    if len(emm.specs) != 1:
        raise NotImplementedError(
            "mvcontrast MVP: single-factor `specs` only "
            f"(got {emm.specs}). Multi-factor between-contrasts are on "
            "the v0.3 roadmap."
        )
    spec = emm.specs[0]
    cells = emm.cell_labels[spec].astype(str).tolist()
    n_cells = len(cells)

    if method == "pairwise":
        pairs = [(i, j) for i in range(n_cells) for j in range(i + 1, n_cells)]
        C = np.array([
            [(1.0 if k == i else (-1.0 if k == j else 0.0)) for k in range(n_cells)]
            for (i, j) in pairs
        ])
        labels = [f"{cells[i]} - {cells[j]}" for (i, j) in pairs]
    elif method in ("trt.vs.ctrl", "trt.vs.ctrl1"):
        # First level = control; each remaining vs control.
        ctrl = 0
        others = list(range(1, n_cells))
        C = np.array([
            [(1.0 if k == t else (-1.0 if k == ctrl else 0.0)) for k in range(n_cells)]
            for t in others
        ])
        labels = [f"{cells[t]} - {cells[ctrl]}" for t in others]
    elif method == "trt.vs.ctrlk":
        # Last level = control.
        ctrl = n_cells - 1
        others = list(range(0, ctrl))
        C = np.array([
            [(1.0 if k == t else (-1.0 if k == ctrl else 0.0)) for k in range(n_cells)]
            for t in others
        ])
        labels = [f"{cells[t]} - {cells[ctrl]}" for t in others]
    else:
        raise NotImplementedError(
            f"mvcontrast method={method!r} not yet implemented "
            "(supported: 'pairwise', 'trt.vs.ctrl', 'trt.vs.ctrlk')."
        )

    info = emm.info
    L_b = C @ emm.L_cells                            # (n_contrasts, k)
    LB = L_b @ info.B                                # (n_contrasts, p)
    Sigma_inv = np.linalg.inv(info.Sigma_hat)
    # T² per row: (LB_r Σ̂⁻¹ LB_r') / (L_b,r inv_cov L_b,r')
    num = np.einsum("ij,jk,ik->i", LB, Sigma_inv, LB)
    denom = np.einsum("ij,jk,ik->i", L_b, info.inv_cov, L_b)
    if np.any(denom <= 0):
        raise ValueError(
            "mvcontrast: between-contrast row gave a non-positive "
            "L (X'X)⁻¹ L'; the contrast may be degenerate."
        )
    T_square = num / denom
    p = len(info.endog_names)
    df1 = p
    df2 = info.df_resid - p + 1
    if df2 <= 0:
        raise ValueError(
            f"mvcontrast: df2 = df_resid − p + 1 = {df2} ≤ 0; the "
            f"model has too few residual df ({info.df_resid}) for "
            f"a joint test on p={p} responses."
        )
    F = T_square * df2 / (df1 * info.df_resid)
    p_raw = stats.f.sf(F, df1, df2)

    k_fam = len(labels)
    if adjust == "none":
        p_adj = p_raw
    elif adjust == "bonferroni":
        p_adj = np.minimum(1.0, p_raw * k_fam)
    elif adjust == "sidak":
        p_adj = 1.0 - (1.0 - p_raw) ** k_fam
    else:
        raise NotImplementedError(
            f"mvcontrast adjust={adjust!r} not yet implemented "
            "(supported: 'sidak', 'bonferroni', 'none')."
        )

    return pd.DataFrame({
        "contrast": labels,
        "T_square": T_square,
        "df1": np.full(k_fam, df1, dtype=int),
        "df2": np.full(k_fam, float(df2)),
        "F_ratio": F,
        "p_value": p_adj,
    })
