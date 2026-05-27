"""Contrast methods over an EMM result.

Supported contrast methods (matching R ``emmeans`` semantics):

- ``pairwise`` / ``revpairwise`` — all ``k(k-1)/2`` differences with
  Tukey HSD adjustment by default (Tukey, 1953).
- ``trt.vs.ctrl`` — one-vs-control differences with Dunnett adjustment
  (Dunnett, 1955) via the multivariate-t CDF, which respects the
  correlation between contrasts that share a control.
- ``poly`` — orthogonal polynomial contrasts on equally spaced levels,
  built from the QR factorization of the Vandermonde matrix; matches
  R's ``contr.poly`` row-by-row up to a sign flip that we enforce to
  make each contrast's last element positive.
- ``consec`` — successive-difference contrasts (``i+1 - i``).
- ``eff`` — each level vs. the grand mean (effect coding). Coefficient
  matrix is ``I - 1/k``; matches R ``emmeans::eff.emmc``.
- ``del.eff`` — each level vs. the average of the OTHER levels.
  Coefficient is ``+1`` on the focal level and ``-1/(k-1)`` elsewhere;
  matches R ``emmeans::del.eff.emmc``.
- ``mean_chg`` — mean of higher-indexed levels vs. mean of lower-indexed
  levels at every split point (``k-1`` rows); matches R
  ``emmeans::mean_chg.emmc``. Useful for ordered factors when you want
  to detect a shift somewhere in the sequence.
- Custom: ``method=`` accepts a coefficient dict ``{name: vector}`` or
  a ``(n_contrasts, k)`` ndarray.

A contrast is a linear combination of EMMs:

    estimate = D @ L_marg @ beta
    var = diag(D @ L_marg @ V @ L_marg.T @ D.T)

By-grouped contrasts: when the source EMM was conditioned on ``by``,
comparisons are computed *within* each by-group, and multiplicity
adjustments apply per by-group (per "family").

``effect_size(contrast_result)`` reports Cohen's d (Cohen, 1988) and
Hedge's g (Hedges, 1981; small-sample correction factor
``J = 1 - 3/(4·df_resid - 1)``) using the model's residual SD as the
pooled standard deviation.

References
----------
- Searle, Speed & Milliken (1980). "Population Marginal Means in the
  Linear Model: An Alternative to Least Squares Means." *The American
  Statistician*, 34(4).
- Lenth (2024). *emmeans: Estimated Marginal Means, aka Least-Squares
  Means*. R package documentation.
- Tukey (1953). "The Problem of Multiple Comparisons." Unpublished
  Princeton manuscript; collected in Tukey's papers.
- Dunnett (1955). "A Multiple Comparison Procedure for Comparing
  Several Treatments with a Control." *JASA* 50(272), 1096-1121.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from pymmeans.adjustments import adjust_pvalues
from pymmeans.emmeans import EMMResult
from pymmeans.utils import ModelInfo


@dataclass(frozen=True)
class ContrastResult:
    """Pairwise / custom contrasts with adjusted p-values.

    .. warning::
       Do **not** mutate this dataclass directly via
       ``dataclasses.replace``. ``frame`` / ``linfct`` / ``method`` /
       ``method_args`` / ``at`` / ``weights`` / ``bias_adjust`` /
       ``inference_kind`` / ``df_method`` are coupled: changing one
       in isolation produces a split-brain object whose displayed
       numbers silently disagree with its metadata, and downstream
       rebuild-aware ops (``bootstrap_ci(kind="case")``,
       ``permutation_test``, ``contrast(simple=)``) will read the
       stamped fields and produce wrong answers. Use
       :func:`pymmeans.update` for *display* fields (``level`` /
       ``adjust``); recompute via :func:`pymmeans.contrast` /
       :func:`pymmeans.pairs` for *reconstruction* fields.
       ``update()`` enforces this distinction by refusing
       reconstruction-control kwargs.
    """

    frame: pd.DataFrame
    linfct: np.ndarray
    model_info: ModelInfo
    adjust: str
    type: str = "link"
    """``"link"`` if the contrast estimates are on the linear-predictor
    scale, ``"response"`` if :func:`regrid_response` has been applied.
    Used to refuse subsequent contrast operations that would silently
    return wrong-scale numbers."""
    bias_adjust: bool = False
    """True iff the response-scale values came from
    ``regrid_response(..., bias_adjust=True)``. Mirrors
    ``EMMResult.bias_adjust``."""
    bias_sigma: Any = None
    """the ``sigma=`` override (if any) supplied to
    ``regrid_response(..., bias_adjust=True, sigma=...)``. Mirrors
    ``EMMResult.bias_sigma``."""
    inference_kind: str = "wald"
    """``"wald"`` (default) or ``"posterior"``. Mirrors
    ``EMMResult.inference_kind`` and propagates through contrast
    operations on a posterior EMM."""
    df_method: str = "default"
    """``"default"`` / ``"satterthwaite"`` / ``"kenward_roger"``;
    mirrors :attr:`EMMResult.df_method` so a corrected EMM's df / vcov
    correction round-trips to ``pairs`` / ``contrast``."""
    _satt_cache: Any = None
    """pickle-safe REML state cache; see
    :attr:`EMMResult._satt_cache`."""
    level: float = 0.95
    """the confidence level the contrast
    was built at. Carried so `summary(ct, level=)` knows what level to
    recompute against (the EMM has `level`, but the contrast didn't
    carry it forward — `summary(ct)` crashed with AttributeError).
    Propagated from `emm.level` in `_contrast_result_from_emm`."""
    _adjust_meta: dict[str, Any] | None = None
    """per-by-group adjustment metadata
    (n_means, df, correlation matrix) populated by
    `_contrast_one_family`. Used by `summary(ct, adjust=)` and
    `update(ct, adjust=)` to recompute p-values correctly without
    re-deriving the family structure from `model_info.param_names`
    (which is the OUTER param vector, not the family size).

    Keys: 'families' (list of per-by-group dicts with start/stop/
    n_means/df/correlation). None when the contrast wasn't built
    via `_contrast_one_family` (e.g. hand-constructed)."""
    _pair_indices: tuple | None = None
    """structural EMM-row index pairs
    for pairwise contrast results, as a tuple of (i, j) tuples in
    the SAME order as ``frame.iterrows()``. Populated by ``pairs()``
    only — other contrast methods (`poly`, `consec`, custom) leave
    this ``None``. Used by ``cld()`` / ``pwpp()`` for structural row
    lookup that avoids parsing the `" - "` delimiter in level names.

    originally stored these as `_pair_i` / `_pair_j` columns
    on ``frame``, but that leaked through ``effect_size`` and any
    other consumer that walks `frame.columns`. The dataclass field
    is the proper home for internal metadata."""
    target: list[str] = field(default_factory=list)
    """(beyond-R-parity): the source EMM's ``target``
    factors. Carried so case-bootstrap and permutation-test paths
    can rebuild the contrast under a refit. Empty list for
    hand-constructed contrasts."""
    by: list[str] = field(default_factory=list)
    """the source EMM's ``by`` factors. Same rationale
    as ``target``."""
    method: str | None = None
    """the contrast-builder method name (``"pairwise"``,
    ``"trt.vs.ctrl"``, ``"consec"``, ``"poly"``, ``"eff"``, ...) or
    ``"custom"`` for user-supplied coefficient matrices /
    callables, or ``"interaction"`` for Kronecker-product calls.
    Carried so case-bootstrap and permutation-test paths can rebuild
    the same contrast structure under a refit instead of guessing
    from labels (the heuristic mis-classified
    ``trt.vs.ctrl`` etc. and produced silently-NaN CIs).
    ``None`` for hand-built / pre-results."""
    method_args: dict[str, Any] = field(default_factory=dict)
    """extra arguments to the contrast builder needed for
    a faithful rebuild (e.g. ``{"ref": 0}`` for ``trt.vs.ctrl``,
    ``{"reverse": True}`` for ``revpairwise``, or
    ``{"coefs": {...}}`` for custom dicts). Used together with
    :attr:`method` by case-bootstrap / permutation_test."""
    at: dict[str, Any] | None = None
    """``at=`` overrides from the source EMM. Carried so
    case-bootstrap / permutation_test rebuild the contrast with the
    same grid restrictions."""
    weights: str = "equal"
    """``weights=`` mode from the source EMM."""

    def __repr__(self) -> str:
        header = f"{len(self.frame)} contrasts (adjust={self.adjust})"
        return f"{header}\n{self.frame!r}"

    @property
    def n_rows(self) -> int:
        """Number of contrast rows in the summary frame."""
        return len(self.frame)


def _row_labels(frame: pd.DataFrame, cols: list[str]) -> list[str]:
    if len(cols) == 1:
        return [str(v) for v in frame[cols[0]]]
    return [
        ",".join(str(v) for v in row)
        for row in frame[cols].itertuples(index=False, name=None)
    ]


def _pairwise_matrix(
    k: int, labels: list[str], reverse: bool
) -> tuple[np.ndarray, list[str], list[tuple[int, int]]]:
    """Build the pairwise contrast matrix.

    also returns the (positive, negative)
    EMM-row index pairs alongside the labels, so downstream consumers
    (`cld()`, `pwpp()`) can do STRUCTURAL row lookup instead of parsing
    the " - " in the label string — which silently mishandles level
    names that themselves contain " - " (e.g. ``"A - B"`` as a level).
    Indices are within-by-group; the caller maps them to original
    EMM-frame positions via `indices`.
    """
    n_pairs = k * (k - 1) // 2
    D = np.zeros((n_pairs, k))
    names: list[str] = []
    idx_pairs: list[tuple[int, int]] = []
    row = 0
    for i in range(k):
        for j in range(i + 1, k):
            if reverse:
                D[row, j] = 1.0
                D[row, i] = -1.0
                names.append(f"{labels[j]} - {labels[i]}")
                idx_pairs.append((j, i)) # (positive_idx, negative_idx)
            else:
                D[row, i] = 1.0
                D[row, j] = -1.0
                names.append(f"{labels[i]} - {labels[j]}")
                idx_pairs.append((i, j))
            row += 1
    return D, names, idx_pairs


def _trt_vs_ctrl_matrix(
    k: int, labels: list[str], ref_idx: int
) -> tuple[np.ndarray, list[str]]:
    if k < 2:
        raise ValueError(
            f"trt.vs.ctrl requires at least 2 levels (got k={k}). With one "
            "treatment level there are no non-control comparisons to make."
        )
    if not 0 <= ref_idx < k:
        raise ValueError(f"ref index {ref_idx} out of range for k={k}.")
    rows = []
    names = []
    for i in range(k):
        if i == ref_idx:
            continue
        row = np.zeros(k)
        row[i] = 1.0
        row[ref_idx] = -1.0
        rows.append(row)
        names.append(f"{labels[i]} - {labels[ref_idx]}")
    return np.asarray(rows), names


_POLY_NAMES = ["linear", "quadratic", "cubic", "quartic"]


def _poly_matrix(
    k: int, _labels: list[str], max_degree: int | None = None
) -> tuple[np.ndarray, list[str]]:
    """Orthogonal polynomial contrasts at equally spaced points.

    previously matched R's `contr.poly`
    (orthonormal Q from QR factorization), not R `emmeans::poly.emmc`
    (integer-scaled orthogonal contrasts).

    the fix matched R for
    k=3..7 but DIVERGED for k > 7 because the integer-multiplier
    search capped at 24 — and high-degree contrasts at large k need
    larger multipliers (cubic for k=15 needs multiplier ~250). The
    Gram-Schmidt-on-raw-powers approach was also numerically
    fragile.

    New algorithm (R `emmeans::poly.emmc` direct port):

    1. Build orthonormal `contr.poly` via Vandermonde-QR (stable).
    2. R's two-step integer scaling loop:
       a. Divide each column by the smallest positive entry.
       b. While `max(|col - round(col)|) > 0.05`, divide by that
          maximum (iterative refinement that converges fast).
       c. Round to integer.
    3. Cap output at `max_degree = min(6, k-1)` by default (R's
       default), accept an explicit override.
    4. Names: "linear", "quadratic", "cubic", "quartic", then
       "degree 5", "degree 6", ... (R uses space, not underscore).
    """
    if k < 2:
        raise ValueError(f"polynomial contrasts require k >= 2 (got {k}).")
    max_d = min(6, k - 1) if max_degree is None else min(max_degree, k - 1)
    if max_d < 1:
        raise ValueError(
            f"polynomial contrasts need max_degree >= 1 (got {max_d})."
        )
    # Step 1: orthonormal contr.poly via QR of centred Vandermonde.
    x = np.arange(1, k + 1, dtype=float)
    V = np.vander(x, k, increasing=True)
    Q, _ = np.linalg.qr(V)
    # Drop the constant column; transpose so each row is one contrast.
    P_orth = Q[:, 1:max_d + 1].T # shape (max_d, k)
    # Sign-normalise: last entry positive (R convention)
    for d in range(P_orth.shape[0]):
        if P_orth[d, -1] < 0:
            P_orth[d] = -P_orth[d]
    # Step 2: integer scaling via Fractions. 's "divide by max
    # deviation" iterative refinement worked for k <= 7 but exploded
    # to 1e15-scale values for k=15 high-degree contrasts because the
    # iteration kept dividing by float-noise-tiny `dev`s. The
    # Fractions approach finds the smallest integer multiplier
    # exactly: convert each entry to a Fraction with a bounded
    # denominator (drops float noise), then multiply by the LCM of
    # all denominators in the row.
    from fractions import Fraction
    from math import lcm

    P = P_orth.astype(float, copy=True)
    for j in range(P.shape[0]):
        con = P[j].copy()
        # 2a: divide by smallest positive entry (centres the scale
        # around 1.0 so the bounded-denominator step has a stable
        # reference).
        pos_mask = con > 0.01
        if pos_mask.any():
            con = con / float(np.min(con[pos_mask]))
        # 2b: rationalise each entry with a bounded denominator and
        # find the row's LCM-of-denominators. `limit_denominator(2000)`
        # is conservative — R's tabulated poly.emmc values for k <= 20
        # all have denominators well under 1000.
        fracs = [Fraction(float(v)).limit_denominator(2000) for v in con]
        denom = 1
        for f in fracs:
            denom = lcm(denom, f.denominator)
        # Cap the resulting magnitude — if the LCM blows up (rare
        # high-k pathological case), fall back to the orthonormal
        # output rather than emitting astronomical integers.
        scaled = np.array([float(f * denom) for f in fracs])
        if np.max(np.abs(scaled)) > 1e6:
            P[j] = P_orth[j]
        else:
            P[j] = scaled
    names = [
        _POLY_NAMES[d] if d < len(_POLY_NAMES) else f"degree {d + 1}"
        for d in range(max_d)
    ]
    return P, names


def _consec_matrix(k: int, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """Consecutive contrasts: each level minus the previous (i+1 vs i)."""
    if k < 2:
        raise ValueError(f"consecutive contrasts require k >= 2 (got {k}).")
    D = np.zeros((k - 1, k))
    names = []
    for i in range(k - 1):
        D[i, i + 1] = 1.0
        D[i, i] = -1.0
        names.append(f"{labels[i + 1]} - {labels[i]}")
    return D, names


def _eff_matrix(k: int, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """`eff` contrasts: each level vs. the grand mean.

    Matches R ``emmeans::eff.emmc``. Coefficient matrix is ``I - 1/k``
    (rows sum to zero). Useful for ANOVA-style "effect coding" reports
    where each row is the deviation of one level from the average.
    """
    if k < 2:
        raise ValueError(f"eff contrasts require k >= 2 (got {k}).")
    D = np.eye(k) - 1.0 / k
    names = [f"{labels[i]} effect" for i in range(k)]
    return D, names


def _del_eff_matrix(k: int, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """`del.eff` contrasts: each level vs. the average of the others.

    Matches R ``emmeans::del.eff.emmc``. Row i has ``+1`` on the
    focal level and ``-1/(k-1)`` on every other level (rows sum to
    zero). Distinct from ``eff``: the reference is the mean of the
    OTHER levels, not the grand mean — so the contrast value scales
    differently when k changes.
    """
    if k < 2:
        raise ValueError(f"del.eff contrasts require k >= 2 (got {k}).")
    D = np.full((k, k), -1.0 / (k - 1))
    np.fill_diagonal(D, 1.0)
    names = [f"{labels[i]} effect" for i in range(k)]
    return D, names


def _mean_chg_matrix(k: int, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """`mean_chg` contrasts: mean of higher-indexed levels vs. mean of
    lower-indexed levels, at every split point.

    Matches R ``emmeans::mean_chg.emmc``. For ``k`` levels there are
    ``k-1`` rows; row ``i`` (1-indexed split at position ``i``)
    contrasts the mean of levels ``i+1..k`` against the mean of levels
    ``1..i``. Useful for ordered-factor analyses where you want to
    detect a shift somewhere in the sequence rather than a specific
    polynomial trend.

    C3 labels follow R's compact pipe
    format ``levs[i-1]|levs[i]`` (e.g. ``A|B``, ``B|C``, ``C|D``)
    rather than pymmeans's older ``{A,B} - {C,D}`` set notation.
    R uses pipe because the actual contrast is the mean-change at
    the split point, not literally "set diff".
    """
    if k < 2:
        raise ValueError(f"mean_chg contrasts require k >= 2 (got {k}).")
    D = np.zeros((k - 1, k))
    names = []
    for split in range(1, k):
        D[split - 1, :split] = -1.0 / split
        D[split - 1, split:] = 1.0 / (k - split)
        names.append(f"{labels[split - 1]}|{labels[split]}")
    return D, names


def _identity_matrix(k: int, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """R `emmeans::identity.emmc` — each row is one EMM
    tested against zero. Matrix is the k×k identity."""
    if k < 1:
        raise ValueError(f"identity contrasts require k >= 1 (got {k}).")
    return np.eye(k), [str(lv) for lv in labels]


def _helmert_matrix(k: int, labels: list[str]) -> tuple[np.ndarray, list[str]]:
    """R `emmeans::helmert.emmc` — each level vs. mean of
    PREVIOUS (earlier) levels.

    For level i (i >= 1): row coefficients are +1 on level i and
    -1/i on each earlier level (0..i-1). The conventional R scaling
    multiplies the row through by `i + 1` so the rightmost positive
    coefficient becomes integer; e.g. for k=4:
        H[0] = [-1, 1, 0, 0] (B vs A)
        H[1] = [-1, -1, 2, 0] (C vs avg(A, B))
        H[2] = [-1, -1, -1, 3] (D vs avg(A, B, C))
    Useful for ordered factors with monotone effects.
    """
    if k < 2:
        raise ValueError(f"helmert contrasts require k >= 2 (got {k}).")
    D = np.zeros((k - 1, k))
    names = []
    for i in range(1, k):
        D[i - 1, :i] = -1.0
        D[i - 1, i] = float(i)
        if i == 1:
            lo = labels[0]
        else:
            lo = "{" + ",".join(labels[:i]) + "}"
        names.append(f"{labels[i]} - {lo}")
    return D, names


def _iter_by_groups(emm: EMMResult):
    if not emm.by:
        yield (), np.arange(emm.n_rows)
        return
    indices = emm.frame.groupby(emm.by, observed=True, sort=False).indices
    for key, idx in indices.items():
        key_tuple = key if isinstance(key, tuple) else (key,)
        yield key_tuple, np.asarray(idx)


def _contrast_one_family(
    emm: EMMResult,
    indices: np.ndarray,
    by_key: tuple,
    D: np.ndarray,
    names: list[str],
    adjust: str,
) -> tuple[pd.DataFrame, np.ndarray, dict]:
    info = emm.model_info
    sub_L = emm.linfct[indices]
    k = len(indices)
    L_c = D @ sub_L

    from pymmeans.estimability import estimable_mask, estimable_mask_from_basis

    # Estimability check sources, in priority order:
    # 1. Live design matrix from `raw_result.model.exog` (fastest, most
    # precise; pre-pickle).
    # 2. Picklable row-space basis stored on ModelInfo (post-pickle, or
    # when the raw result was never attached). #5 fixed the
    # silent NaN→0 corruption that used to happen here.
    # 3. No basis available -> assume estimable (full-rank designs hit
    # this path harmlessly).
    X_design = None
    if info.raw_result is not None and hasattr(info.raw_result, "model"):
        X_design = np.asarray(getattr(info.raw_result.model, "exog", None))
    if (
        X_design is not None
        and X_design.ndim == 2
        and X_design.shape[1] == info.n_params
    ):
        est_mask = estimable_mask(L_c, X_design)
    elif info.estimability_basis is not None:
        est_mask = estimable_mask_from_basis(L_c, info.estimability_basis)
    else:
        est_mask = np.ones(L_c.shape[0], dtype=bool)
    if not est_mask.all():
        import warnings as _warn

        _warn.warn(
            f"{int((~est_mask).sum())} of {len(est_mask)} contrasts "
            "are not estimable under the design; marking NaN.",
            UserWarning,
            stacklevel=3,
        )

    est = L_c @ info.beta

    # ``mvt`` is the documented alias for ``dunnett``;
    # ``dunnettx`` (R `.pdunnx`) is correlation-free and uses its own
    # separate distribution. Only the dunnett / mvt path needs the
    # full contrast-correlation matrix (off-diagonals feed the MVT
    # CDF); every other adjustment method (tukey, sidak,
    # bonferroni, holm, scheffe, bh, by, hochberg, hommel, none,
    # dunnettx, ...) needs only the diagonal SEs.
    _adj_lower = adjust.lower()
    adj_canon = "dunnett" if _adj_lower == "mvt" else _adj_lower
    needs_full_cov = adj_canon == "dunnett" and len(est) > 1

    if needs_full_cov:
        # Dunnett / mvt branch: need the full ``L_c @ V @ L_c.T`` for
        # the off-diagonal correlation. This is O(m^2) where
        # m = #contrasts; for k=200 group levels via pairwise that
        # is ~7.7 GB on float64. Caller is on the hook for that
        # cost; the dunnett-max-k guard in ``_dunnett`` already
        # bounds it before the QMC integrator hits an
        # effectively-unbounded runtime.
        cov_c = L_c @ info.vcov @ L_c.T
        var = np.diag(cov_c)
    else:
        # All other adjustment paths: compute only the diagonal of
        # ``L_c @ V @ L_c.T`` via einsum. Memory stays O(m * p) where
        # p = #params (vs O(m^2) for the full matrix). At k=250 group
        # levels via pairwise (m = 31,125 contrasts, p ~= 250) this
        # is ~62 MB instead of ~7.7 GB on float64.
        cov_c = None # signal "full matrix not built"
        var = np.einsum("ij,jk,ik->i", L_c, info.vcov, L_c)
    se = np.sqrt(np.clip(var, 0.0, None))
    if not est_mask.all():
        est = np.where(est_mask, est, np.nan)
        se = np.where(est_mask, se, np.nan)
    df_val: float = (
        np.inf if (info.family is not None or info.is_mixed) else info.df_resid
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        t_ratio = np.where(se > 0, est / se, np.nan)
    p_raw = 2.0 * stats.t.sf(np.abs(t_ratio), df_val)

    correlation = None
    if needs_full_cov:
        outer = np.outer(se, se)
        with np.errstate(divide="ignore", invalid="ignore"):
            correlation = np.where(outer > 0, cov_c / outer, 0.0)
        np.fill_diagonal(correlation, 1.0)

    # use the COUNT-OF-ESTIMABLE-MEANS as ``n_means``
    # in the adjustment so unused / non-estimable EMM levels don't
    # silently inflate the family for Tukey / Scheffé. R `emmeans`
    # drops non-estimable means from the family before computing the
    # studentized-range / Scheffé quantile; pymmeans previously counted
    # every patsy category, so an unused ``pd.Categorical`` level
    # (or an empty cell with ``weights="cells"``) produced a wider
    # adjusted p-value than R. The summary-layer ``adjust=`` path
    # has used ``n_means_valid`` since ; this brings the
    # initial adjustment in ``_contrast_one_family`` in line.
    valid_rows = np.isfinite(est) & np.isfinite(se)
    n_rows_valid = int(valid_rows.sum())
    # `n_means_valid`: how many of the underlying EMM levels are
    # actually referenced by an estimable contrast row. Used by
    # Tukey/Scheffé where the studentized-range / Scheffé F-quantile
    # parameter is the count of means, not the count of contrasts.
    if n_rows_valid > 0:
        # A column j is referenced by any valid row that has nonzero
        # coefficient at j.
        referenced = np.any(D[valid_rows] != 0, axis=0)
        n_means_valid = int(referenced.sum())
    else:
        n_means_valid = 0
    # If every row is estimable, ``n_means_valid == k`` and there's no
    # behavioral change; only the unused-level / empty-cell paths
    # differ.
    _n_means_for_adjust = n_means_valid if n_means_valid > 0 else k

    p_adj = adjust_pvalues(
        p_raw,
        adjust,
        n_means=_n_means_for_adjust,
        df=df_val,
        t_ratios=t_ratio,
        correlation=correlation,
    )

    piece = pd.DataFrame({"contrast": names})
    for col, val in zip(emm.by, by_key, strict=True):
        piece[col] = val
    piece["estimate"] = est
    piece["SE"] = se
    piece["df"] = df_val
    piece["t_ratio"] = t_ratio
    piece["p_value"] = p_adj

    # emit per-family adjustment metadata
    # so `summary(ct, adjust=)` and `update(ct, adjust=)` can recompute
    # p-values correctly without falsely treating
    # `len(model_info.param_names)` as the family size and without
    # losing the precomputed Dunnett correlation matrix. Slice
    # boundaries are filled in by the caller once the family piece is
    # concatenated.
    # also stash the
    # estimable-row count and the estimable-means count so the
    # `summary` adjustment path can use the R-correct family `n` on
    # rank-deficient designs.
    fam_meta = {
        "n_rows": int(D.shape[0]),
        "n_rows_valid": n_rows_valid,
        "n_means": int(k),
        "n_means_valid": n_means_valid,
        "df": float(df_val),
        "correlation": correlation,
        "by_key": tuple(by_key) if by_key else (),
    }
    return piece, L_c, fam_meta


def _contrast_result_from_emm(
    emm: EMMResult,
    frame: pd.DataFrame,
    linfct: np.ndarray,
    adjust: str,
) -> ContrastResult:
    """Construct a ContrastResult and propagate every metadata field
    from the source EMMResult.

    #3: the three contrast-construction sites used to pass
    only ``frame / linfct / model_info / adjust``, dropping ``type``,
    ``bias_adjust``, and ``inference_kind``. The dropped
    ``inference_kind`` meant ``apply_satterthwaite(pairs(posterior_emm))``
    silently overwrote posterior credible intervals with Wald t-intervals
    -- the refusal couldn't fire because the metadata had been
    lost during the contrast step.

    when the source EMM had Satterthwaite df applied
    (``info.is_mixed`` and every ``frame["df"]`` is finite while the
    mixed-model default is inf), propagate the correction to the new
    contrast result. Without this, the canonical lmerTest workflow
    ``apply_satterthwaite(emm) -> pairs(emm)`` silently demotes the
    contrast df back to inf and reports z-based p-values labelled as
    t. We recompute Satt at the contrast-level L_c matrix because each
    contrast is a different linear combination of beta with its own
    Satt df -- can't just copy the EMM-row df.
    """
    # IMPORTANT: do NOT propagate the EMM's df_method into the freshly
    # built ContrastResult. The contrast's SE here was computed from
    # ``L_c @ info.vcov @ L_c.T`` (uncorrected V_beta) by
    # ``_contrast_one_family``, NOT from V_KR. Stamping
    # df_method="kenward_roger" on construction would trip the
    # idempotency guard in ``apply_kenward_roger`` (which short-circuits
    # when ``df_method == "kenward_roger"``) and silently leave the
    # uncorrected V_beta SE in place.
    #
    # The correct stamp happens below: if the source EMM carried a Satt
    # or KR correction, we call ``apply_satterthwaite`` /
    # ``apply_kenward_roger`` on the contrast, which (a) recomputes
    # SE / df at the contrast L_c using the corrected vcov and (b)
    # stamps ``df_method`` on the output via ``_apply_correction``.
    # If the source EMM had ``df_method="default"``, the contrast keeps
    # ``df_method="default"`` which is also correct.
    result = ContrastResult(
        frame=frame,
        linfct=linfct,
        model_info=emm.model_info,
        adjust=adjust,
        type=getattr(emm, "type", "link"),
        bias_adjust=getattr(emm, "bias_adjust", False),
        inference_kind=getattr(emm, "inference_kind", "wald"),
        df_method="default",
        _satt_cache=getattr(emm, "_satt_cache", None),
        level=getattr(emm, "level", 0.95),
        # propagate source EMM's target / by for case-
        # bootstrap and permutation_test rebuild paths.
        target=list(getattr(emm, "target", []) or []),
        by=list(getattr(emm, "by", []) or []),
        # propagate at / weights too so the rebuild
        # matches the original's grid restrictions exactly.
        at=getattr(emm, "at", None),
        weights=getattr(emm, "weights", "equal") or "equal",
    )
    df_method = getattr(emm, "df_method", "default")
    # refuse contrasts on a bootstrap-derived EMM.
    # The stored ``frame['lower_cl']`` / ``frame['upper_cl']`` are
    # percentile bootstrap intervals (not Wald), but
    # ``contrast()`` / ``pairs()`` derive new intervals via
    # ``L_c @ vcov @ L_c.T`` — Wald math on whatever ``vcov`` was
    # used to *seed* the original bootstrap. Mixing percentile point-
    # uncertainty with Wald contrast-uncertainty is silently wrong.
    # The correct workflow is ``bootstrap_ci(pairs(raw_em))`` —
    # bootstrap the contrasts, not the EMMs.
    if df_method == "bootstrap":
        raise ValueError(
            "pairs() / contrast() are not defined for a bootstrap-"
            "derived EMMResult (df_method='bootstrap'). The stored CIs "
            "are percentile bootstrap intervals; deriving a new "
            "contrast via Wald math would silently mix two inference "
            "paradigms. Correct workflows:\n"
            " - bootstrap_ci(pairs(em)) # bootstrap the contrasts\n"
            " - bootstrap_ci(contrast(em, ...))\n"
            "i.e. apply pairs/contrast FIRST, then bootstrap the result."
        )
    if df_method in ("satterthwaite", "kenward_roger"):
        # both Satt and KR carry a pickle-safe ``_satt_cache``
        # populated by the original apply_* call, so the propagation
        # works post-pickle without `raw_result`. The refusal below
        # only fires if BOTH raw_result and the cache are missing —
        # usually means the EMM was hand-built or built
        # before introduced the cache.
        if emm.model_info.raw_result is None and \
                getattr(emm, "_satt_cache", None) is None:
            raise ValueError(
                f"pairs / contrast cannot propagate the {df_method!r} "
                "correction because the source EMM lost both its "
                "raw_result AND its _satt_cache. Re-run apply_* on a "
                "fresh emmeans() result to populate the cache, then "
                "pickle / re-use as needed."
            )
        # Lazy import to avoid a satterthwaite <-> contrasts cycle at
        # module-load time.
        from pymmeans.satterthwaite import (
            apply_kenward_roger,
            apply_satterthwaite,
        )

        # Recompute the correction at the contrast-level L_c matrix. Each
        # contrast row is a different linear combination of beta with its
        # own Satt / KR df, so we can't just copy from the EMM frame.
        if df_method == "satterthwaite":
            result = apply_satterthwaite(result)
        else: # kenward_roger
            result = apply_kenward_roger(result)
    return result


def _refuse_response_scale_contrast(emm: EMMResult, op: str) -> None:
    """Raise if ``emm`` is already on the response scale.

    A linear contrast on response-scale EMMs (e.g. ``A_resp - B_resp``)
    is **not** the same as the response-scale version of the link-scale
    contrast (which would be a *ratio* under a log link). Computing
    ``D @ L_marg @ beta`` after the EMMs have been back-transformed
    silently produces a link-scale difference labelled as a
    response-scale estimate. #7 caught us doing exactly this.

    The fix: refuse, and steer the user to the correct workflow:
    contrast on the link scale, then ``regrid_response`` the contrast
    (which produces the right scale and the right column name —
    ``ratio`` for log families).
    """
    if getattr(emm, "type", "link") == "response":
        raise ValueError(
            f"{op} on a response-scale EMMResult is not supported because "
            "a linear contrast of back-transformed means is not the same "
            "as the back-transformed contrast (e.g. for a log link, A - B "
            "is not exp(A_link - B_link) = A/B). Either compute the "
            "contrast on the link-scale EMM and then call "
            "`regrid_response(contrast)`, or pass `type='link'` to "
            "`emmeans(...)` before contrasting."
        )


def pairs(
    emm: EMMResult,
    adjust: str | None = None,
    reverse: bool = False,
    simple: str | list[str] | None = None,
    max_contrasts: int | None = 50,
) -> ContrastResult | EmmList:
    """All pairwise comparisons of EMMs.

    With ``k`` target levels, produces ``k * (k-1) / 2`` contrasts. If the EMM
    is by-grouped, comparisons are computed within each by-group as a
    separate family.

    Parameters
    ----------
    emm
        Result from ``emmeans(...)`` on the **link** scale. Response-scale
        EMMs are rejected — see Notes.
    adjust
        Multiplicity correction; default ``tukey``.
    reverse
        If True, use ``j - i`` instead of ``i - j``.
    simple
        pymmeans-only feature: when ``emm.target`` lists
        multiple factors, ``simple=`` controls which factor's
        pairwise comparisons to compute — treating the *other*
        factors as by-groups. Accepts:

        - ``str``: name of one target factor → returns a single
          ``ContrastResult`` of pairs within that factor, by-grouped
          over the others.
        - ``list[str]``: returns an ``EmmList`` keyed by factor name,
          one entry per requested factor.
        - ``"each"``: shorthand for the full list ``emm.target``.

        Mirrors R ``pairs(emm, simple = "...")``. Saves users from
        the "k1·k2·k3 pairwise explosion" footgun when they really
        only want per-factor comparisons.
    max_contrasts
        pymmeans-only guard: refuse to compute when the
        projected contrast count exceeds this limit. Default ``50``.
        Set ``None`` to disable. The error message suggests
        ``by=`` / ``simple=`` decomposition workflows.

        Rationale: ``pairs(emmeans(fit, ["a","b","c"]))`` on factors
        with (5, 4, 3) levels produces 60 × 59 / 2 = 1770 contrasts,
        which is rarely what the user wants. R `emmeans` does this
        silently. pymmeans surfaces the cost upfront so users opt in
        explicitly.

    Returns
    -------
    ContrastResult (or EmmList when ``simple=`` is a list / "each")

    Notes
    -----
    A linear contrast of back-transformed means is not the back-transform
    of the contrast. For log-family transforms, ``A - B`` on the
    link scale corresponds to ``A / B`` on the response scale.
    ``pairs(regrid_response(emm))`` therefore raises; use
    ``regrid_response(pairs(emm))`` instead so the ratio interpretation
    falls out automatically.
    """
    # consult emm_options for default adjust
    from pymmeans.options import get_emm_option as _opt
    if adjust is None:
        adjust = _opt("adjust", "tukey")
    _refuse_response_scale_contrast(emm, "pairs()")

    # Non-estimable rows (rank-deficient designs) carry NaN in
    # ``emmean`` / ``SE``. Differencing NaN propagates into any
    # contrast that touches such a row; the resulting rows in the
    # ContrastResult are NaN but look syntactically valid. Warn
    # explicitly so the user is not surprised by ``estimate=NaN`` in
    # the output frame — R `emmeans` marks these as ``nonEst``.
    _frame = emm.frame
    if "emmean" in _frame.columns and _frame["emmean"].isna().any():
        n_bad = int(_frame["emmean"].isna().sum())
        n_tot = len(_frame)
        import warnings as _w
        _w.warn(
            f"pairs(): the input EMM has {n_bad} of {n_tot} non-estimable "
            f"rows (``emmean`` is NaN, typically from a rank-deficient "
            f"design). Contrasts touching those rows will carry NaN through "
            f"to ``estimate`` / ``SE`` / ``p_value``. Inspect the result "
            f"frame for NaN; consider subsetting to estimable rows or "
            f"fixing the rank deficiency at the model level.",
            UserWarning,
            stacklevel=2,
        )

    # ``simple=`` dispatch. Defer to the helper which
    # rebuilds an EMM per simple-target with the other factors
    # marginalised into by-groups, then recurses into pairs.
    if simple is not None:
        return _pairs_simple(
            emm, simple, adjust=adjust, reverse=reverse,
            max_contrasts=max_contrasts,
        )

    # project the contrast count for the friendly-footgun
    # guard. For an EMMResult with N target rows and by-groups of
    # size k_i, the pair count is sum_i k_i*(k_i-1)/2. When all
    # by-groups share the same size, k*(k-1)/2 per group.
    if max_contrasts is not None:
        projected = 0
        for _by_key, indices in _iter_by_groups(emm):
            k = len(indices)
            projected += k * (k - 1) // 2
        if projected > max_contrasts:
            target_factors = emm.target
            tips = []
            if len(target_factors) > 1:
                tips.append(
                    f"reduce to per-factor pairs via "
                    f"``pairs(emm, simple={target_factors!r})``"
                )
                tips.append(
                    f"or re-call emmeans with one target + the others "
                    f"as by-factors: "
                    f"``emmeans(fit, {target_factors[0]!r}, "
                    f"by={target_factors[1:]!r})``"
                )
            tips.append(
                f"or opt in to the full set with "
                f"``pairs(emm, max_contrasts={projected})`` (or "
                f"``max_contrasts=None``)"
            )
            tip_str = "\n - ".join(tips)
            raise ValueError(
                f"pairs() would produce {projected} contrasts, "
                f"exceeding the max_contrasts={max_contrasts} guard. "
                "This is usually the multi-factor-pairwise footgun "
                "(k_total = k_a · k_b · ...). To proceed:\n - "
                + tip_str
            )
    pieces: list[pd.DataFrame] = []
    all_L: list[np.ndarray] = []

    # collect structural row indices
    # alongside the by-group pieces; we'll attach them as the
    # `_pair_indices` dataclass field on the result, NOT as
    # frame columns (leaked through `effect_size`).
    all_pair_indices: list[tuple[int, int]] = []
    family_meta: list[dict] = []
    cursor = 0
    for by_key, indices in _iter_by_groups(emm):
        sub_frame = emm.frame.iloc[indices].reset_index(drop=True)
        labels = _row_labels(sub_frame, emm.target)
        D, names, idx_pairs = _pairwise_matrix(len(indices), labels, reverse=reverse)
        piece, L_c, fam = _contrast_one_family(emm, indices, by_key, D, names, adjust)
        all_pair_indices.extend(idx_pairs)
        pieces.append(piece)
        all_L.append(L_c)
        fam["start"] = cursor
        cursor += fam["n_rows"]
        fam["stop"] = cursor
        family_meta.append(fam)

    frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    linfct = np.vstack(all_L) if all_L else np.empty((0, emm.model_info.n_params))
    result = _contrast_result_from_emm(emm, frame, linfct, adjust)
    # Stamp the pair-indices via dataclasses.replace (frozen dataclass).
    from dataclasses import replace as _dc_replace
    return _dc_replace(
        result,
        _pair_indices=tuple(all_pair_indices),
        _adjust_meta={"families": family_meta},
        method="revpairwise" if reverse else "pairwise",
        method_args={"reverse": reverse},
    )


def contrast(
    emm: EMMResult,
    method: (
        str
        | dict[str, list[float]]
        | np.ndarray
        | Callable[[list[str]], Any]
    ) = "eff",
    ref: str | int | None = None,
    adjust: str | None = None,
    interaction: str | list[str] | None = None,
    simple: str | list[str] | None = None,
    combine: bool = False,
) -> ContrastResult | EmmList:
    """Compute contrasts of EMMs using a named method or custom coefficients.

    default `method` changed from
    ``"pairwise"`` to ``"eff"`` to match R `emmeans::contrast.emmGrid`,
    where the documented default is ``"eff"`` (each level vs. grand
    mean). Use :func:`pairs` for pairwise comparisons — it always
    defaulted to pairwise + Tukey, which is the more common workflow
    most users want.

    Parameters
    ----------
    emm
        Result from ``emmeans(...)``.
    method
        Either a named method (``"eff"`` (default), ``"pairwise"``,
        ``"revpairwise"``, ``"tukey"`` (alias for pairwise + Tukey
        adjust), ``"trt.vs.ctrl"`` / ``"trt.vs.ctrl1"`` (first level
        as ref) / ``"trt.vs.ctrlk"`` (last level as ref), ``"poly"``,
        ``"consec"``, ``"del.eff"``, ``"mean_chg"``, ``"identity"``,
        ``"helmert"``) **or** custom contrast coefficients:

        - ``dict[str, list[float]]``: mapping from contrast name to a vector
          of coefficients of length ``emm.n_rows`` (or, with ``by``, length
          equal to rows per by-group).
        - ``np.ndarray`` of shape ``(n_contrasts, k)``: an unnamed
          coefficient matrix; contrasts are labeled ``"c1", "c2", ...``.
        - ``Callable[[list[str]], dict | DataFrame | ndarray]``: an
          ``.emmc_*``-style function. Called once with the target's
          level labels; must return one of the three forms above
          (DataFrame columns are read as R-style contrast columns,
          and an ndarray of shape ``(n_levels, n_contrasts)`` is
          auto-transposed).

        Custom contrasts apply within each by-group.
    ref
        Reference level for ``trt.vs.ctrl``. Either a level name (str) or a
        0-based index. Defaults to the first level.
    adjust
        Multiplicity correction. ``None`` selects R's per-method default:
        ``tukey`` for ``pairwise`` / ``revpairwise`` / ``tukey``;
        ``dunnettx`` for ``trt.vs.ctrl{,1,k}``; ``mvt`` for ``consec`` /
        ``mean_chg``; ``fdr`` for ``eff`` / ``del.eff``; ``none`` for
        ``poly`` / ``identity`` / ``helmert``; and ``bonferroni`` for
        custom coefficient matrices.

    Examples
    --------
    >>> # Drug vs placebo, then drug A vs drug B
    >>> custom = {"drug_vs_placebo": [-1, 0.5, 0.5], "A_vs_B": [0, 1, -1]} # doctest: +SKIP
    >>> contrast(emm, method=custom) # doctest: +SKIP
    """
    _refuse_response_scale_contrast(emm, "contrast()")

    # R's ``contrast(em, interaction = c("pairwise", "consec"))``
    # builds the Kronecker product of per-factor contrast matrices and
    # applies it to the EMM. Each entry of ``interaction`` is the
    # contrast method for the corresponding target factor (in the
    # order specs were given to ``emmeans()``). The combined family
    # uses the named methods' joint adjustment default (R falls back
    # to ``"sidak"`` for the multi-factor product).
    if interaction is not None:
        return _interaction_contrast(emm, interaction, adjust=adjust)

    # ``simple=`` / ``combine=`` for R parity. Mirrors
    # ``contrast(emm, method, simple="a")`` / ``combine=TRUE``.
    # Refactors the ``_pairs_simple`` dispatcher into a
    # generic per-method version so any contrast method (not just
    # pairwise) can be decomposed by factor.
    if simple is not None:
        return _simple_contrast(
            emm, method=method, simple=simple, combine=combine,
            adjust=adjust, ref=ref,
        )

    # ``emm_options(adjust=...)`` propagates into contrast()'s default
    # selection: the named-method dispatch below consults the option,
    # and the custom-contrast branch falls through to bonferroni
    # only if neither caller nor option supplied anything.
    from pymmeans.options import get_emm_option as _opt
    opt_adjust = _opt("adjust", None)

    # 80%-parity push: accept a *callable* method, mirroring
    # R's ``.emmc_*`` mechanism. The callable receives the target's
    # level labels (as a list of str) and returns one of:
    #
    # - ``dict[str, sequence[float]]`` — Pythonic; keys are contrast
    # labels, values are coefficient vectors of length ``n_levels``.
    # - ``pd.DataFrame`` — R convention: each column is
    # one contrast (column name = label, values across rows = the
    # coefficient vector for that contrast).
    # - ``np.ndarray`` of shape ``(n_levels, n_contrasts)`` — same
    # layout as R's data.frame return; transposed to the standard
    # ``(n_contrasts, n_levels)`` matrix before dispatch.
    #
    # The callable is invoked once with the levels of the first by-
    # group; pymmeans (like R) assumes target levels are constant
    # across by-cells, and ``_custom_contrast`` re-applies the same
    # coefficients to each by-group.
    if callable(method):
        # Pull the target levels from the EMM result (per by-group,
        # they are identical — checked by _iter_by_groups).
        first_indices = next(iter(_iter_by_groups(emm)))[1]
        first_frame = emm.frame.iloc[first_indices]
        levels = [str(v) for v in _row_labels(first_frame, emm.target)]
        coefs = method(list(levels))
        # R's ``.emmc_*`` default adjustment is "none". The dict /
        # ndarray paths (pymmeans-native syntax) keep "bonferroni",
        # but for the R-style callable path we match R exactly so
        # ``contrast(em, method=my_emmc)`` reproduces R's output.
        # Explicit ``adjust=`` and ``emm_options`` still override.
        cal_default = "none"
        chosen_adjust = adjust or opt_adjust or cal_default
        # Normalise the return value into the dict / ndarray form that
        # _custom_contrast already accepts.
        if isinstance(coefs, pd.DataFrame):
            coefs_dict = {
                str(col): list(coefs[col].astype(float))
                for col in coefs.columns
            }
            return _custom_contrast(emm, coefs_dict, chosen_adjust)
        if isinstance(coefs, np.ndarray):
            arr = np.asarray(coefs, dtype=float)
            if arr.ndim == 2 and arr.shape[0] == len(levels):
                # R convention: rows = levels, cols = contrasts → T.
                arr = arr.T
            return _custom_contrast(emm, arr, chosen_adjust)
        if isinstance(coefs, dict):
            return _custom_contrast(emm, coefs, chosen_adjust)
        raise TypeError(
            "Callable method= must return a dict, pandas DataFrame, "
            f"or numpy ndarray of coefficients; got {type(coefs).__name__}."
        )

    if isinstance(method, (dict, np.ndarray, list)):
        return _custom_contrast(
            emm, method, adjust or opt_adjust or "bonferroni"
        )

    method = method.lower()
    # (, #2): align method aliases and
    # default adjustments with R `emmeans`. The R defaults table is:
    # pairwise / revpairwise / tukey -> tukey
    # trt.vs.ctrl / trt.vs.ctrl1 / trt.vs.ctrlk -> dunnettx
    # poly -> none
    # consec, mean_chg -> mvt
    # eff, del.eff -> fdr
    # identity -> none
    # helmert -> none (R `.emmc.defaults`;
    # had this as
    # bonferroni — wrong)
    if method == "pairwise":
        return pairs(emm, adjust=adjust or opt_adjust or "tukey")
    if method == "revpairwise":
        return pairs(
            emm, adjust=adjust or opt_adjust or "tukey", reverse=True
        )
    if method == "tukey":
        # R alias: `tukey` is pairwise with Tukey adjustment.
        return pairs(emm, adjust=adjust or opt_adjust or "tukey")
    # trt.vs.ctrl1 (first level as ref), trt.vs.ctrlk (last level as
    # ref), and bare trt.vs.ctrl (ref=first by default) all map to
    # the same builder with different ref_idx.
    SUPPORTED = (
        "trt.vs.ctrl", "trt.vs.ctrl1", "trt.vs.ctrlk",
        "poly", "consec",
        "eff", "del.eff", "mean_chg",
        "identity", "helmert",
    )
    if method in SUPPORTED:
        if len(emm.target) != 1:
            raise NotImplementedError(
                f"'{method}' in v0.1 supports a single target factor only."
            )
        pieces: list[pd.DataFrame] = []
        all_L: list[np.ndarray] = []
        if adjust is None:
            # option overrides the per-method default
            # (option < explicit kwarg; per-method default < option).
            adj = opt_adjust or {
                "poly": "none",
                "trt.vs.ctrl": "dunnettx",
                "trt.vs.ctrl1": "dunnettx",
                "trt.vs.ctrlk": "dunnettx",
                "consec": "mvt",
                "mean_chg": "mvt",
                "eff": "fdr",
                "del.eff": "fdr",
                "identity": "none",
                "helmert": "none",
            }.get(method, "bonferroni")
        else:
            adj = adjust
        family_meta: list[dict] = []
        cursor = 0
        for by_key, indices in _iter_by_groups(emm):
            sub_frame = emm.frame.iloc[indices].reset_index(drop=True)
            labels = _row_labels(sub_frame, emm.target)
            k = len(indices)
            if method == "trt.vs.ctrl":
                ref_idx = (
                    labels.index(ref) if isinstance(ref, str) else (ref or 0)
                )
                D, names = _trt_vs_ctrl_matrix(k, labels, ref_idx)
            elif method == "trt.vs.ctrl1":
                # First level as control (matches R)
                D, names = _trt_vs_ctrl_matrix(k, labels, 0)
            elif method == "trt.vs.ctrlk":
                # Last level as control (matches R)
                D, names = _trt_vs_ctrl_matrix(k, labels, k - 1)
            elif method == "poly":
                D, names = _poly_matrix(k, labels)
            elif method == "consec":
                D, names = _consec_matrix(k, labels)
            elif method == "eff":
                D, names = _eff_matrix(k, labels)
            elif method == "del.eff":
                D, names = _del_eff_matrix(k, labels)
            elif method == "mean_chg":
                D, names = _mean_chg_matrix(k, labels)
            elif method == "identity":
                D, names = _identity_matrix(k, labels)
            else: # helmert
                D, names = _helmert_matrix(k, labels)
            piece, L_c, fam = _contrast_one_family(emm, indices, by_key, D, names, adj)
            pieces.append(piece)
            all_L.append(L_c)
            fam["start"] = cursor
            cursor += fam["n_rows"]
            fam["stop"] = cursor
            family_meta.append(fam)
        frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
        linfct = (
            np.vstack(all_L) if all_L else np.empty((0, emm.model_info.n_params))
        )
        result = _contrast_result_from_emm(emm, frame, linfct, adj)
        from dataclasses import replace as _dc_replace
        return _dc_replace(
            result,
            _adjust_meta={"families": family_meta},
            method=method,
            method_args={"ref": ref} if ref is not None else {},
        )
    raise ValueError(
        f"Unknown contrast method '{method}'. "
        "Supported: pairwise, revpairwise, tukey, trt.vs.ctrl, "
        "trt.vs.ctrl1, trt.vs.ctrlk, poly, consec, eff, del.eff, "
        "mean_chg, identity, helmert, or a dict/ndarray of custom "
        "coefficients."
    )


def effect_size(
    c: ContrastResult | EMMResult,
    sigma: float | None = None,
    edf: float | None = None,
    method: str = "pairwise",
    measure: str = "cohens_d",
) -> pd.DataFrame:
    """Cohen's d / Hedges' g plus R-style standardised effect sizes.

    R's ``emmeans::eff_size(emm, sigma, edf)`` accepts an EMM (not just a
    contrast) plus the user's pooled SD and its effective df. pymmeans
    accepts the same EMM input path, exposes an ``edf`` argument, and
    emits the R-style ``SE`` / ``lower_cl`` / ``upper_cl`` columns
    derived from R's log-sigma delta-method approximation
    ``SE(d) = sqrt((SE_estimate / sigma)^2 + (d^2) / (2 * edf))``.

    Parameters
    ----------
    c
        ``EMMResult`` (builds contrasts internally using
        ``method``) or ``ContrastResult`` from ``pairs()`` / ``contrast()``.
    sigma
        Pooled SD. If ``None``, uses ``sqrt(model_info.scale)``. R errors
        when not supplied for a `t` model; pymmeans defaults to the OLS
        residual SD for convenience but the user should supply sigma
        explicitly to match R exactly.
    edf
        Effective df for the sigma estimate (R's `df.residual(fit)`).
        Defaults to the model's residual df. added the kwarg
        to thread a user-supplied edf into the Hedges correction.
    method
        Contrast method used when ``c`` is an ``EMMResult``. Default
        ``"pairwise"`` matches R `eff_size`.
    measure
        pymmeans-only extension: which standardized effect
        size to report. Default ``"cohens_d"`` (current behavior;
        backwards-compatible). Other options:

        - ``"odds_ratio"``: for **binomial / logit** contrasts —
          exponentiates the link-scale contrast to produce odds
          ratios with the delta-method SE and CIs. Adds columns
          ``odds_ratio``, ``odds_ratio_SE``,
          ``odds_ratio_lower_cl``, ``odds_ratio_upper_cl``.
        - ``"risk_ratio"`` / ``"rate_ratio"``: for **log-link
          (Poisson, log-LHS OLS)** contrasts — same delta-method
          pipeline as ``odds_ratio`` but the underlying logarithm
          interprets as relative risk or rate ratio. Adds the
          ``risk_ratio`` / ``rate_ratio`` columns.
        - ``"hazard_ratio"``: for **Cox PH (PHReg)** contrasts —
          ratio = exp(log-hazard difference). Adds ``hazard_ratio``
          columns.

        Each ratio measure uses the standard delta-method SE
        ``SE(exp(x)) = exp(x) · SE(x)`` and exponentiated link-scale
        CI endpoints (which is also what R `confint` reports on
        ``regrid_response`` ratios — symmetric Wald on link scale,
        then exp). The legacy ``cohen_d`` / ``hedges_g`` columns
        remain in the output regardless of ``measure``.

    Returns
    -------
    pd.DataFrame
        The contrast frame plus new ``effect_size`` (=Cohen's d),
        ``effect_size_SE``, ``effect_size_lower_cl``,
        ``effect_size_upper_cl``, ``cohen_d``, and ``hedges_g`` columns.
        The legacy ``cohen_d`` / ``hedges_g`` columns are retained for
        backwards compatibility with pre-callers.
    """
    # also accept an ``EmmList`` — recurse into
    # each member and return a dict keyed by member name. Matches the
    # / recursion in ``summary`` / ``confint`` /
    # ``as_r_frame``. Without this, ``effect_size(emm_list)`` raised
    # ``AttributeError: 'EmmList' object has no attribute 'model_info'``
    # the moment the function tried to read ``c.model_info``.
    if isinstance(c, EmmList):
        return {
            name: effect_size(member, sigma=sigma, edf=edf, method=method)
            for name, member in zip(c.names, c, strict=True)
        }

    # refuse a bootstrap-derived contrast — the
    # stored CIs are percentile bootstrap intervals but
    # ``effect_size`` builds analytic delta-method CIs around the
    # standardised contrast. Silently mixing percentile point
    # uncertainty with delta-method effect-size uncertainty is the
    # same composition class closed for ``test()``,
    # closed for ``summary(..., infer=...)``, and round
    # 54 F5 closed for ``pairs(em_b)``. The bootstrap path is now
    # uniformly refused on all "analytic-uncertainty consumer" wrappers.
    if getattr(c, "df_method", "default") == "bootstrap":
        # distinguish EMM vs ContrastResult inputs
        # in the error message — both can hit this guard (EMM via the
        # ``effect_size(em)`` shortcut below), but a user
        # passing an EMM gets the wrong workflow hint if we always
        # say "bootstrap-derived contrast".
        from pymmeans.emmeans import EMMResult as _EMMResult
        _src_label = (
            "EMMResult" if isinstance(c, _EMMResult) else "contrast"
        )
        raise ValueError(
            f"effect_size() is not defined for a bootstrap-derived "
            f"{_src_label} (df_method='bootstrap'). The stored CIs "
            "are percentile bootstrap intervals; ``effect_size`` "
            "builds analytic delta-method CIs around the standardised "
            "contrast, which would silently mix two inference "
            "paradigms. Compute effect sizes BEFORE bootstrap:\n"
            " d = effect_size(pairs(raw_em)) # raw, not bootstrap\n"
            " # then inspect d directly — or bootstrap a user-\n"
            " # defined standardised contrast on the link-scale EMM."
        )

    # accept an EMMResult by computing the contrast first.
    from pymmeans.emmeans import EMMResult as _EMMResult
    if isinstance(c, _EMMResult):
        c = pairs(c) if method == "pairwise" else contrast(c, method=method)
    from pymmeans.utils import detect_value_column

    info = c.model_info
    kind_info = detect_value_column(c.frame)
    if kind_info is None or kind_info[0] != "contrast":
        # Anything that's not a link-scale contrast (e.g. a regridded
        # log-family contrast with `ratio`, or an EMM with `emmean`)
        # doesn't have a meaningful Cohen's d interpretation.
        if kind_info is not None and kind_info[0] == "ratio":
            raise ValueError(
                "effect_size() is not defined for a ratio-scale contrast "
                "(this looks like the result of `regrid_response` on a "
                "log-family contrast). Compute effect sizes on the "
                "link-scale ContrastResult instead, e.g. "
                "`effect_size(contrast(emm))` before regridding."
            )
        raise ValueError(
            f"effect_size() requires a link-scale contrast with an "
            f"'estimate' column; got {list(c.frame.columns)}."
        )
    # #4: posterior contrasts do not have a meaningful
    # residual SD (`info.scale` defaults to 1.0 on PosteriorInfo).
    # Refuse unless the user supplied sigma explicitly.
    if (
        getattr(c, "inference_kind", "wald") == "posterior"
        and sigma is None
    ):
        raise ValueError(
            "effect_size() on a posterior contrast requires an explicit "
            "sigma= argument: posterior fixed-effect draws do not define "
            "a residual standard deviation. Pass the residual SD from a "
            "matching frequentist fit, or compute Cohen's d manually."
        )
    if sigma is None:
        # statsmodels exposes "scale" on the fit; for OLS this is sigma^2
        scale = getattr(info, "scale", None)
        if scale is None:
            raise ValueError(
                "Could not determine residual SD automatically. Pass "
                "sigma= explicitly."
            )
        sigma = float(np.sqrt(scale))
    if sigma <= 0:
        raise ValueError(f"sigma must be positive (got {sigma}).")

    out = c.frame.copy()
    out["cohen_d"] = out["estimate"].to_numpy() / sigma
    # #4: the Hedges small-sample correction uses the
    # *contrast row's* df (which apply_satterthwaite / apply_kenward_roger
    # update), not the model's residual df. Falling back to df_resid
    # only when the frame has no df column (e.g. a raw custom contrast).
    if "df" in out.columns:
        df_arr = out["df"].to_numpy(dtype=float)
    else:
        df_arr = np.full(len(out), float(info.df_resid))
    j = np.ones(len(out), dtype=float)
    ok = np.isfinite(df_arr) & (df_arr > 1)
    j[ok] = 1.0 - 3.0 / (4.0 * df_arr[ok] - 1.0)
    out["hedges_g"] = j * out["cohen_d"].to_numpy()

    # R-style effect_size SE + CI columns. R's eff_size.emmGrid uses
    # the log-sigma delta approximation:
    #
    # SE(d) = sqrt((SE_estimate / sigma)^2 + (d^2) / (2 * edf))
    #
    # CIs come from ``d ± t_crit * SE_d`` at the contrast row's df.
    # ``edf`` defaults to the model's residual df (R: df.residual(fit))
    # unless the user provided one explicitly.
    edf_use = float(edf) if edf is not None else float(info.df_resid)
    if not np.isfinite(edf_use) or edf_use <= 0:
        edf_use = np.inf # no small-sample SE inflation
    d_arr = out["cohen_d"].to_numpy()
    se_est = out["SE"].to_numpy() if "SE" in out.columns else None
    if se_est is not None:
        with np.errstate(invalid="ignore"):
            se_d = np.sqrt(
                (se_est / sigma) ** 2 + d_arr ** 2 / (2.0 * edf_use)
            )
        out["effect_size"] = d_arr
        out["effect_size_SE"] = se_d
        level = float(getattr(c, "level", 0.95) or 0.95)
        # Two-sided t CI at the contrast row's df.
        crit = stats.t.ppf(0.5 + level / 2.0, df_arr)
        out["effect_size_lower_cl"] = d_arr - crit * se_d
        out["effect_size_upper_cl"] = d_arr + crit * se_d

    # pymmeans-only: ratio measures for log / logit / cloglog
    # contrast scales. Exponentiate the link-scale contrast and apply
    # the delta-method SE + symmetric-Wald-then-exp CI endpoints.
    if measure != "cohens_d":
        ratio_aliases = {
            "odds_ratio": ("odds_ratio", ("logit",)),
            "log_odds_ratio": ("odds_ratio", ("logit",)),
            "or": ("odds_ratio", ("logit",)),
            "risk_ratio": ("risk_ratio", ("log",)),
            "rr": ("risk_ratio", ("log",)),
            "rate_ratio": ("rate_ratio", ("log",)),
            "hazard_ratio": ("hazard_ratio", ("log",)),
            "hr": ("hazard_ratio", ("log",)),
        }
        if measure not in ratio_aliases:
            raise ValueError(
                f"Unknown measure {measure!r}. Supported: "
                f"'cohens_d' (default), {sorted(ratio_aliases)}."
            )
        col_name, expected_link_families = ratio_aliases[measure]
        # Sanity check: the contrast should be on the link scale of
        # the right family. Warn (not raise) so users can override.
        # also warn when the model is OLS without a
        # log-LHS response, since `exp(contrast)` of a raw mean
        # difference is meaningless. Cox PH and log-LHS OLS are
        # handled via response_name (e.g. "np.log(y)") even when
        # family is None.
        info = c.model_info
        link_name = None
        if info.family is not None:
            link_name = type(info.family.link).__name__.lower()
        elif info.response_name and any(
            tk in info.response_name.lower() for tk in ("log", "logit")
        ):
            # Log-LHS OLS or Cox PH (whose response_name is set to
            # "np.log(<endog>)" by _cox_response_name_override).
            link_name = "log"
        if link_name is None or not any(
            link_name.startswith(lf) for lf in expected_link_families
        ):
            import warnings as _w
            _model_desc = (
                "OLS with response " + (info.response_name or "?")
                if info.family is None
                else "GLM with link " + str(link_name)
            )
            _w.warn(
                f"effect_size(measure={measure!r}) expects a "
                f"{expected_link_families[0]}-link model (binomial "
                "logit, log-LHS OLS, Cox PH, ...). The model here "
                f"is {_model_desc} — ``exp(contrast)`` may not have "
                "the intended interpretation.",
                UserWarning,
                stacklevel=2,
            )
        est_link = out["estimate"].to_numpy()
        se_link = out["SE"].to_numpy()
        ratio = np.exp(est_link)
        ratio_se = ratio * se_link # delta method: |d/dx exp(x)| = exp(x)
        level = float(getattr(c, "level", 0.95) or 0.95)
        df_for_crit = out["df"].to_numpy(dtype=float)
        crit = stats.t.ppf(0.5 + level / 2.0, df_for_crit)
        lo_link = est_link - crit * se_link
        hi_link = est_link + crit * se_link
        out[col_name] = ratio
        out[f"{col_name}_SE"] = ratio_se
        out[f"{col_name}_lower_cl"] = np.exp(lo_link)
        out[f"{col_name}_upper_cl"] = np.exp(hi_link)

    return out


def _pairs_simple(
    emm: EMMResult,
    simple: str | list[str],
    adjust: str | None,
    reverse: bool,
    max_contrasts: int | None,
) -> ContrastResult | EmmList:
    """pymmeans-only ``pairs(emm, simple=...)`` dispatch.

    Re-runs ``emmeans()`` for each requested simple-target with the
    other target factors moved into by-grouping, then runs ``pairs``
    on each rebuilt EMM. Avoids the k1·k2·...·(k1·k2·...−1)/2
    pairwise explosion when the user really only wants pairs within
    each factor.

    Returns a single ``ContrastResult`` when ``simple`` is a string,
    or an ``EmmList`` keyed by factor name when ``simple`` is a list
    or ``"each"``.
    """
    from pymmeans.emmeans import emmeans as _emmeans

    if simple == "each":
        if len(emm.target) <= 1:
            raise ValueError(
                "simple='each' requires a multi-factor EMM "
                f"(emm.target={emm.target}); use pairs(emm) directly."
            )
        simple_list = list(emm.target)
    elif isinstance(simple, str):
        if simple not in emm.target:
            raise ValueError(
                f"simple={simple!r} is not in emm.target={emm.target}."
            )
        simple_list = [simple]
    else:
        simple_list = list(simple)
        for s in simple_list:
            if s not in emm.target:
                raise ValueError(
                    f"simple={s!r} is not in emm.target={emm.target}."
                )

    # Locate the original fit so we can rebuild EMMs with different
    # target / by grouping.
    info = emm.model_info
    raw = info.raw_result
    if raw is None:
        # surface the actionable workaround. Picked EMMs
        # carry their `frame` / `linfct` / `at` / `weights` faithfully
        # (verified), but `simple=` decomposition needs to
        # call `emmeans()` again with the by-grouping restructured —
        # which needs the original fit. The user has two options:
        # (a) re-load the fit and call ``pairs`` from scratch, or
        # (b) skip ``simple=`` and call ``pairs(em)`` directly,
        # then filter the result by hand to the simple effects
        # they want.
        raise ValueError(
            "pairs(simple=...) needs the original fitted model on "
            "``emm.model_info.raw_result``, but it is None (this "
            "EMM appears to have been pickled and re-loaded; pickle "
            "deliberately drops the raw fit). Options:\n"
            " 1. Re-fit the model and call ``pairs(emmeans(fit, "
            "target, by=...), simple=...)`` again from scratch.\n"
            " 2. Skip ``simple=`` and call ``pairs(em)`` directly, "
            "then filter the resulting frame by hand to the simple-"
            "effects rows you want (the EMM's preserved ``at`` and "
            "``weights`` metadata is honored on this path).\n"
            " 3. Pickle the original fit alongside the EMM and re-"
            "attach via ``info.raw_result = fit`` after loading."
        )

    # forward the source EMM's ``at=`` and ``weights=``
    # so that the rebuilt EMMs use the same grid restrictions /
    # marginalisation scheme. Without this, ``pairs(em, simple="a")``
    # on an EMM built with ``at={"x": [10]}`` silently rebuilds at
    # ``x = mean(x)`` and gives WRONG simple-effect estimates.
    # also forward ``max_contrasts`` so a user-set
    # guard still fires within each simple family (the
    # bypass-on-decomposition was too permissive).
    orig_at = getattr(emm, "at", None)
    orig_weights = getattr(emm, "weights", "equal")
    results = {}
    for target_name in simple_list:
        other_factors = [f for f in emm.target if f != target_name]
        new_em = _emmeans(
            raw,
            target_name,
            by=(emm.by + other_factors) if emm.by else other_factors,
            level=getattr(emm, "level", 0.95),
            type=getattr(emm, "type", "link"),
            at=orig_at,
            weights=orig_weights,
        )
        results[target_name] = pairs(
            new_em, adjust=adjust, reverse=reverse,
            max_contrasts=max_contrasts,
        )

    if len(results) == 1:
        return next(iter(results.values()))
    return EmmList(**results)


def _simple_contrast(
    emm: EMMResult,
    method: str | dict | np.ndarray | Callable,
    simple: str | list[str],
    combine: bool,
    adjust: str | None,
    ref: str | int | None,
) -> ContrastResult | EmmList:
    """generic ``simple=`` dispatcher for any contrast method.

    Mirrors R `emmeans::contrast(emm, method, simple="a")` /
    `combine=TRUE`. For each requested simple-target, re-runs
    ``emmeans()`` with the other target factors moved into
    by-grouping, then applies `method` to the rebuilt EMM.

    - ``simple=str`` → single ContrastResult.
    - ``simple=list`` or ``"each"`` → ``EmmList`` of ContrastResults
      (one per requested factor) when ``combine=False`` (default);
      a single ``rbind``'d ContrastResult when ``combine=True``.
    """
    from pymmeans.emmeans import emmeans as _emmeans

    if simple == "each":
        if len(emm.target) <= 1:
            raise ValueError(
                "simple='each' requires a multi-factor EMM "
                f"(emm.target={emm.target}); use contrast(emm) directly."
            )
        simple_list = list(emm.target)
    elif isinstance(simple, str):
        if simple not in emm.target:
            raise ValueError(
                f"simple={simple!r} is not in emm.target={emm.target}."
            )
        simple_list = [simple]
    else:
        simple_list = list(simple)
        for s in simple_list:
            if s not in emm.target:
                raise ValueError(
                    f"simple={s!r} is not in emm.target={emm.target}."
                )

    info = emm.model_info
    raw = info.raw_result
    if raw is None:
        raise ValueError(
            "contrast(simple=...) needs the original fitted model on "
            "``emm.model_info.raw_result`` — same caveat as "
            "pairs(simple=...)."
        )

    orig_at = getattr(emm, "at", None)
    orig_weights = getattr(emm, "weights", "equal") or "equal"

    results: dict[str, ContrastResult] = {}
    for target_name in simple_list:
        other_factors = [f for f in emm.target if f != target_name]
        new_em = _emmeans(
            raw,
            target_name,
            by=(emm.by + other_factors) if emm.by else other_factors,
            level=getattr(emm, "level", 0.95),
            type=getattr(emm, "type", "link"),
            at=orig_at,
            weights=orig_weights,
        )
        # Recursive contrast call: pass through method / ref / adjust.
        # The recursive call must NOT re-enter simple= dispatch.
        results[target_name] = contrast(
            new_em, method=method, ref=ref, adjust=adjust,
        )

    if combine:
        # combine via rbind so the joint multiplicity
        # adjustment fires across the union of all simple families
        # (matches R `contrast(..., combine=TRUE)`).
        #
        # consult ``emm_options(adjust=...)`` before
        # falling back to ``"bonferroni"``. R's call-time defaults
        # propagate through ``contrast(..., simple=, combine=TRUE)``,
        # but the implementation hard-coded the fallback
        # and silently overrode user-set Holm / Šidák / FDR / etc.
        # adjustments. The explicit ``adjust=`` kwarg still wins
        # over the option (same precedence as everywhere else).
        from pymmeans.contrasts import rbind as _rbind
        from pymmeans.options import get_emm_option as _get_emm_option
        joint_adjust = adjust
        if joint_adjust is None:
            joint_adjust = _get_emm_option("adjust", None)
        if joint_adjust is None:
            joint_adjust = "bonferroni"
        return _rbind(*results.values(), adjust=joint_adjust)

    if len(results) == 1:
        return next(iter(results.values()))
    return EmmList(**results)


def _interaction_contrast(
    emm: EMMResult,
    interaction: str | list[str],
    adjust: str | None,
) -> ContrastResult:
    """(): Kronecker-product contrast.

    For ``contrast(emm, interaction=c(m1, m2, ...))`` build the
    per-factor contrast matrix using each method m_i and Kronecker
    them in the order ``emm.target`` lists factors. Apply the
    combined matrix to the EMM, then run the standard adjustment.

    Mirrors R `contrast.emmGrid(..., interaction=...)`. For the
    canonical 2x3 ``wool * tension`` warpbreaks fit with
    ``interaction=c("pairwise","consec")``, this produces the
    2-row "interaction contrast" frame R reports.
    """
    if isinstance(interaction, str):
        interaction = [interaction]
    if len(interaction) != len(emm.target):
        raise ValueError(
            f"interaction= length ({len(interaction)}) must match the "
            f"number of target factors ({len(emm.target)}): "
            f"{emm.target}."
        )

    # Build per-factor contrast matrices using the existing helpers.
    # Some builders return a 3-tuple (D, names, idx_pairs); drop the
    # third element here since interaction-contrast row labels are
    # computed via the Kronecker product, not row-index lookup.
    def _pw(k, labs, rev=False):
        D, names, _ = _pairwise_matrix(k, labs, rev)
        return D, names
    def _trt(k, labs, ref_idx):
        return _trt_vs_ctrl_matrix(k, labs, ref_idx)
    method_builders = {
        "pairwise": lambda k, labs: _pw(k, labs, False),
        "revpairwise": lambda k, labs: _pw(k, labs, True),
        "tukey": lambda k, labs: _pw(k, labs, False),
        "trt.vs.ctrl": lambda k, labs: _trt(k, labs, 0),
        "trt.vs.ctrl1": lambda k, labs: _trt(k, labs, 0),
        "trt.vs.ctrlk": lambda k, labs: _trt(k, labs, k - 1),
        "poly": lambda k, labs: _poly_matrix(k, labs),
        "consec": lambda k, labs: _consec_matrix(k, labs),
        "eff": lambda k, labs: _eff_matrix(k, labs),
        "del.eff": lambda k, labs: _del_eff_matrix(k, labs),
        "mean_chg": lambda k, labs: _mean_chg_matrix(k, labs),
        "identity": lambda k, labs: _identity_matrix(k, labs),
        "helmert": lambda k, labs: _helmert_matrix(k, labs),
    }

    Ds: list[np.ndarray] = []
    name_lists: list[list[str]] = []
    factor_pretty: list[str] = []
    for factor, m in zip(emm.target, interaction, strict=True):
        m_lc = m.lower()
        if m_lc not in method_builders:
            raise ValueError(
                f"Unsupported interaction method {m!r}; supported: "
                f"{sorted(method_builders)}."
            )
        labs = list(emm.model_info.factors.get(factor, []))
        D_i, names_i = method_builders[m_lc](len(labs), labs)
        Ds.append(D_i)
        name_lists.append(names_i)
        factor_pretty.append(f"{factor}_{m_lc}")

    # Kronecker product. emm.frame rows are typically laid out with
    # the LAST factor in emm.target varying fastest (analytic
    # marginalize's product order). np.kron(A, B) makes B vary
    # fastest in the resulting matrix; that's compatible with our
    # row layout if we kron from left to right.
    D_full = Ds[0]
    for D_next in Ds[1:]:
        D_full = np.kron(D_full, D_next)

    # Build the combined contrast labels: Cartesian product of
    # the per-factor labels in the same Kronecker order.
    import itertools as _it
    combined_labels = []
    for tup in _it.product(*name_lists):
        combined_labels.append(tuple(tup))

    # R `contrast(em, interaction=...)` defaults adjust
    # to ``"none"`` (per emmeans/R/contrast.emmGrid.R — the
    # interaction Kronecker contrast doesn't have a standard
    # adjustment family, so R lets the user pick explicitly). Match.
    from pymmeans.options import get_emm_option as _opt
    chosen_adjust = (
        adjust or _opt("adjust", None) or "none"
    )

    pieces: list[pd.DataFrame] = []
    all_L: list[np.ndarray] = []
    family_meta: list[dict] = []
    cursor = 0
    for by_key, indices in _iter_by_groups(emm):
        # Use the kron contrast matrix directly (it's defined on the
        # EMM rows of this by-group).
        names = [" : ".join(t) for t in combined_labels]
        piece, L_c, fam = _contrast_one_family(
            emm, indices, by_key, D_full, names, chosen_adjust,
        )
        # Replace the single "contrast" column with one column per
        # factor_method (matches R's printed output).
        piece = piece.drop(columns=["contrast"])
        for i, factor_col in enumerate(factor_pretty):
            piece.insert(i, factor_col, [t[i] for t in combined_labels])
        pieces.append(piece)
        all_L.append(L_c)
        fam["start"] = cursor
        cursor += fam["n_rows"]
        fam["stop"] = cursor
        family_meta.append(fam)
    frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    linfct = (
        np.vstack(all_L) if all_L else np.empty((0, emm.model_info.n_params))
    )
    result = _contrast_result_from_emm(emm, frame, linfct, chosen_adjust)
    from dataclasses import replace as _dc_replace
    return _dc_replace(
        result,
        _adjust_meta={"families": family_meta},
        method="interaction",
        method_args={"interaction": list(interaction)},
    )


def _custom_contrast(
    emm: EMMResult,
    coefs: dict[str, list[float]] | np.ndarray | list,
    adjust: str,
) -> ContrastResult:
    """Apply user-supplied contrast coefficients (per by-group)."""
    if isinstance(coefs, dict):
        names = list(coefs.keys())
        if not names:
            # empty dict used to cascade into an IndexError
            # from `D.shape[1]` on a (0, 0) array. Raise cleanly.
            raise ValueError(
                "Custom contrast dict is empty; pass at least one "
                "{name: [coef1, ...]} entry."
            )
        D = np.asarray([coefs[n] for n in names], dtype=float)
    else:
        D = np.asarray(coefs, dtype=float)
        if D.ndim == 1:
            D = D[None, :]
        if D.size == 0 or D.shape[0] == 0:
            # empty ndarray / list yielded the same IndexError.
            raise ValueError(
                "Custom contrast matrix is empty; pass at least one "
                "row of coefficients."
            )
        names = [f"c{i + 1}" for i in range(D.shape[0])]

    pieces: list[pd.DataFrame] = []
    all_L: list[np.ndarray] = []
    family_meta: list[dict] = []
    cursor = 0
    for by_key, indices in _iter_by_groups(emm):
        k = len(indices)
        if D.shape[1] != k:
            raise ValueError(
                f"Contrast coefficient length {D.shape[1]} does not match "
                f"number of EMM rows per by-group ({k})."
            )
        piece, L_c, fam = _contrast_one_family(emm, indices, by_key, D, names, adjust)
        pieces.append(piece)
        all_L.append(L_c)
        fam["start"] = cursor
        cursor += fam["n_rows"]
        fam["stop"] = cursor
        family_meta.append(fam)
    frame = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame()
    linfct = (
        np.vstack(all_L) if all_L else np.empty((0, emm.model_info.n_params))
    )
    result = _contrast_result_from_emm(emm, frame, linfct, adjust)
    from dataclasses import replace as _dc_replace
    # store enough metadata to rebuild this custom contrast
    # under a case-bootstrap refit. Coefs as a dict / ndarray preserve
    # the per-row design; coefs as a Callable can't be pickled across
    # processes so we just store the method tag.
    if isinstance(coefs, dict):
        ma = {"coefs": dict(coefs)}
    elif callable(coefs):
        ma = {"coefs": None, "callable": True}
    else:
        # ndarray / list
        ma = {"coefs": np.asarray(coefs).tolist()}
    return _dc_replace(
        result,
        _adjust_meta={"families": family_meta},
        method="custom",
        method_args=ma,
    )


# ---------------------------------------------------------------------------
# rbind / emm_list (80%-parity push)
# ---------------------------------------------------------------------------


class EmmList(tuple):
    """Ordered, optionally-named container of ``EMMResult`` /
    ``ContrastResult`` objects.

    Mirrors R's ``emm_list`` — the list-of-emm-objects produced by
    e.g. ``emmeans(model, pairwise ~ x)``. In pymmeans we keep
    ``emmeans()`` and ``pairs()`` as separate calls, so ``EmmList``
    is rarely produced directly. It exists mainly so users can write
    R-style code like::

        result = EmmList(emmeans=emm, contrasts=ct)
        result["emmeans"], result["contrasts"]
        result[0], result[1] # positional access also works
        list(result) # iteration

    The container is a plain tuple subclass — pickling, ``len()``,
    iteration, slicing, and equality all work out of the box.
    """

    def __new__(cls, *args, **kwargs):
        if args and kwargs:
            raise TypeError(
                "EmmList accepts positional OR keyword members, not both."
            )
        if kwargs:
            members = tuple(kwargs.values())
            names = tuple(kwargs.keys())
        else:
            members = tuple(args)
            names = tuple(f"_{i}" for i in range(len(members)))
        obj = super().__new__(cls, members)
        obj._names = names # type: ignore[attr-defined]
        return obj

    @property
    def names(self) -> tuple[str, ...]:
        return self._names # type: ignore[attr-defined]

    def __getitem__(self, key):
        if isinstance(key, str):
            try:
                idx = self._names.index(key) # type: ignore[attr-defined]
            except ValueError as e:
                raise KeyError(
                    f"EmmList has no member named {key!r}; "
                    f"available: {self._names!r}" # type: ignore[attr-defined]
                ) from e
            return tuple.__getitem__(self, idx)
        return tuple.__getitem__(self, key)

    def __repr__(self) -> str: # pragma: no cover - cosmetic
        body = ", ".join(
            f"{n}={type(v).__name__}" for n, v in zip(self._names, self, strict=False) # type: ignore[attr-defined]
        )
        return f"EmmList({body})"

    def __reduce__(self):
        """Pickle support: tuple's default __reduce__ collapses to
        ``EmmList(tuple_of_items)`` which our ``__new__`` then mis-
        interprets as a single positional arg. Custom reducer keeps
        both the per-member objects and the names tuple intact."""
        return (_emm_list_unpickle, (self._names, tuple(self))) # type: ignore[attr-defined]


def _emm_list_unpickle(names: tuple[str, ...], members: tuple) -> EmmList:
    """Module-level helper used by ``EmmList.__reduce__`` to rebuild
    the container with both names and members restored. Lives at
    module scope so pickle can resolve it by qualified name."""
    obj = tuple.__new__(EmmList, members)
    obj._names = tuple(names) # type: ignore[attr-defined]
    return obj


def rbind(
    *results: ContrastResult,
    adjust: str | None = None,
) -> ContrastResult:
    """Concatenate ``ContrastResult`` objects and apply a joint
    multiplicity adjustment.

    Mirrors R ``rbind.emm_list`` / ``rbind.emmGrid``. Use this when
    you want to control the family-wise error rate across contrasts
    drawn from *different* contrast families (e.g. pairwise + a
    trt.vs.ctrl block from the same EMM, or contrasts from two
    different EMMs of the same model).

    Parameters
    ----------
    *results
        Two or more ``ContrastResult`` objects. All must share the
        same underlying ``model_info`` (i.e. be derived from the
        same fitted model) — pymmeans rejects rbind across models
        for safety, matching R's ``check.gdvars()`` guard.
    adjust
        Joint multiplicity adjustment applied to the *combined* set
        of rows. ``None`` (default) consults ``emm_options("adjust")``
        before falling back to ``"bonferroni"`` (R's documented rbind
        default). added the option-consult to bring
        ``rbind`` in line with the fix for
        ``contrast(simple=, combine=)``. Any adjustment accepted by
        :func:`pymmeans.adjust.adjust_pvalues` works (``"sidak"``,
        ``"holm"``, ``"hochberg"``, ``"fdr"``, ``"none"``, etc).

    Returns
    -------
    ContrastResult
        Combined frame and ``linfct``; ``p_value`` column re-computed
        under the joint ``adjust``. ``_adjust_meta["families"]`` is
        reset to a single family covering all rows (since rbind by
        construction defines one family).

    Raises
    ------
    TypeError
        If fewer than two results, or if any input is not a
        ContrastResult.
    ValueError
        If inputs reference different ``model_info`` instances (i.e.
        different fitted models), or if the by-column schemas differ
        in a way that prevents safe concatenation.
    """
    from dataclasses import replace as _dc_replace

    from pymmeans.adjustments import adjust_pvalues

    # adopt the precedence —
    # explicit kwarg > emm_options("adjust") > documented default
    # ("bonferroni" for rbind, R parity). Previously the kwarg
    # defaulted to "bonferroni" and silently overrode user-set
    # Holm / Šidák / FDR adjustments inside ``with emm_options(
    # adjust=...):`` blocks. Symmetric with ``_simple_contrast``.
    if adjust is None:
        from pymmeans.options import get_emm_option as _get_emm_option
        adjust = _get_emm_option("adjust", None) or "bonferroni"

    if len(results) < 2:
        raise TypeError(
            f"rbind requires at least two ContrastResults; got {len(results)}."
        )
    for i, r in enumerate(results):
        if not isinstance(r, ContrastResult):
            raise TypeError(
                f"rbind input #{i} is {type(r).__name__}; only "
                "ContrastResult is supported in v0.1."
            )

    base = results[0]
    info = base.model_info

    def _same_model(a, b) -> bool:
        """Two ModelInfo instances refer to the same fit if they
        share the same fitted result object, OR if their beta and
        vcov arrays are numerically identical. Identity comparison
        alone is too strict — two ``emmeans(m, ...)`` calls produce
        fresh ``ModelInfo`` instances even from the same fit."""
        if a is b:
            return True
        if (
            a.raw_result is not None
            and b.raw_result is not None
            and a.raw_result is b.raw_result
        ):
            return True
        if a.n_params != b.n_params:
            return False
        if a.beta.shape != b.beta.shape or a.vcov.shape != b.vcov.shape:
            return False
        return bool(
            np.array_equal(a.beta, b.beta) and np.array_equal(a.vcov, b.vcov)
        )

    for i, r in enumerate(results[1:], start=1):
        if not _same_model(r.model_info, info):
            raise ValueError(
                f"rbind input #{i} comes from a different model; "
                "cross-model concatenation is unsafe (different beta, "
                "vcov, and df). R's rbind also refuses this case."
            )

    # Concatenate frames. pandas does a union-of-columns concat by
    # default, filling missing cells with NaN — that's the right
    # behaviour when families have different by-column sets.
    new_frame = pd.concat(
        [r.frame for r in results], ignore_index=True, sort=False
    )
    new_linfct = np.vstack([r.linfct for r in results])

    # Recompute raw two-sided p-values from t-ratios / df already in
    # the frames (cheaper than re-doing the L_c @ beta path and avoids
    # re-running estimability checks that the inputs already passed).
    t_ratios = new_frame["t_ratio"].to_numpy(dtype=float)
    df_col = new_frame["df"].to_numpy(dtype=float)
    # When df is a single shared value (most cases), use it scalar;
    # when df varies (rare: rbind across families with different
    # adjustments + Satt), apply per-row sf.
    with np.errstate(divide="ignore", invalid="ignore"):
        p_raw = 2.0 * stats.t.sf(np.abs(t_ratios), df_col)

    n_total = int(new_frame.shape[0])
    p_adj = adjust_pvalues(
        p_raw,
        adjust,
        n_means=n_total,
        df=float(np.nanmin(df_col)) if n_total > 0 else np.inf,
        t_ratios=t_ratios,
        correlation=None, # no analytic correlation across rbind families
    )
    new_frame = new_frame.copy()
    new_frame["p_value"] = p_adj

    # Build a single family-meta covering all rows. This is what R
    # effectively does after rbind — joint adjustment, one family.
    fam_meta = {
        "n_rows": n_total,
        "n_rows_valid": int(np.isfinite(t_ratios).sum()),
        "n_means": n_total,
        "n_means_valid": n_total,
        "df": float(np.nanmin(df_col)) if n_total > 0 else np.inf,
        "correlation": None,
        "by_key": (),
        "start": 0,
        "stop": n_total,
    }

    # preserve metadata when all inputs share
    # the same value; otherwise stamp a "rbind" sentinel.
    # ALWAYS stamp ``method="rbind"`` and store per-child
    # metadata so case-bootstrap / permutation_test can rebuild each
    # child separately and concatenate draws. The homogeneous-preserve
    # branch was wrong because even homogeneous rbind has a different
    # row count than a single child — bootstrap's row-count check
    # would silently drop every draw without the per-child recipe.
    def _all_same(attr: str):
        first = getattr(results[0], attr, None)
        return all(getattr(r, attr, None) == first for r in results[1:])

    # Per-child rebuild recipe: tuple of (method, method_args,
    # target, by, at, weights, bias_sigma, n_rows) for each input.
    # also record each child's ``bias_sigma`` so
    # case-bootstrap and permutation_test rebuilds can re-apply the
    # response back-transform at the same sigma each child was built
    # with. Previously the recipe dropped sigma and rebuilds reverted to
    # ``info.scale`` (or no bias adjust at all).
    children_meta = [
        {
            "method": getattr(r, "method", None),
            "method_args": dict(getattr(r, "method_args", {}) or {}),
            "target": list(getattr(r, "target", []) or []),
            "by": list(getattr(r, "by", []) or []),
            "at": getattr(r, "at", None),
            "weights": getattr(r, "weights", "equal") or "equal",
            "bias_sigma": getattr(r, "bias_sigma", None),
            "n_rows": int(r.frame.shape[0]),
        }
        for r in results
    ]

    merged_method = "rbind"
    merged_method_args = {"children": children_meta}
    # Target / by / at / weights: keep base's value only when ALL
    # children agree; otherwise leave defaults. (The children_meta
    # carries per-child values for rebuild.)
    merged_target = (
        list(getattr(base, "target", []) or [])
        if _all_same("target") else []
    )
    merged_by = (
        list(getattr(base, "by", []) or [])
        if _all_same("by") else []
    )
    merged_at = (
        getattr(base, "at", None)
        if _all_same("at") else None
    )
    merged_weights = (
        getattr(base, "weights", "equal") or "equal"
    ) if _all_same("weights") else "equal"

    # preserve ``bias_sigma`` on the combined
    # result when every child shares the same value. Heterogeneous
    # ``bias_sigma`` is rare in practice (rbind across contrasts
    # from different sigma overrides) and falls back to None — the
    # per-child recipe still carries each child's sigma for the
    # case-bootstrap / permutation_test rebuild paths.
    merged_bias_sigma = (
        getattr(base, "bias_sigma", None)
        if _all_same("bias_sigma") else None
    )

    combined = ContrastResult(
        frame=new_frame,
        linfct=new_linfct,
        model_info=info,
        adjust=adjust,
        type=getattr(base, "type", "link"),
        bias_adjust=getattr(base, "bias_adjust", False),
        bias_sigma=merged_bias_sigma,
        inference_kind=getattr(base, "inference_kind", "wald"),
        df_method=getattr(base, "df_method", "default"),
        _satt_cache=getattr(base, "_satt_cache", None),
        level=getattr(base, "level", 0.95),
        target=merged_target,
        by=merged_by,
        method=merged_method,
        method_args=merged_method_args,
        at=merged_at,
        weights=merged_weights,
    )
    return _dc_replace(combined, _adjust_meta={"families": [fam_meta]})
