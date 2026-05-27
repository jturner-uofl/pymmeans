"""Pairwise p-value matrix (R `emmeans::pwpm`).

 R `pwpm(emm)` produces a square
matrix DataFrame with:
- EMM point estimates on the DIAGONAL
- Pairwise differences (or ratios on response scale) BELOW the diagonal
- Adjusted p-values ABOVE the diagonal

It's a compact alternative to the `pairs()` long-table format, and
a common reporting convention in pharma / agronomy. The R version
returns a printable DataFrame; pymmeans returns a plain
`pandas.DataFrame` indexed by level labels.

Example output for 3 levels A, B, C:

           A B C
    A 1.50 0.12 0.005
    B -2.30 3.80 0.45
    C -5.10 -2.80 6.60

Reading: diagonal = EMMs; below-diagonal = differences (A-B is at
[A, B]); above-diagonal = adjusted p-values.
"""

from __future__ import annotations

from dataclasses import replace as _dc_replace
from typing import Any

import numpy as np
import pandas as pd


def _link_emm_to_response_view(emm: Any) -> Any:
    """Build a response-scale view of a link-scale EMM (diagonal-only).

    when the user calls ``pwpm(link_emm, type='response')``
    we need response-scale means for the matrix diagonal. We reuse
    :func:`summary_layer._apply_response_inverse` (the same path that
    ``emmeans(..., type='response')`` runs) so the back-transform
    rules — GLM family inverse-link or LHS-detected transform —
    are identical to a direct response-scale construction.
    """
    from pymmeans.summary_layer import _apply_response_inverse

    frame = emm.frame.copy()
    info = emm.model_info
    mu = frame["emmean"].to_numpy(dtype=float)
    se = frame["SE"].to_numpy(dtype=float)
    lower = frame.get("lower_cl", pd.Series(np.full(len(mu), np.nan))).to_numpy()
    upper = frame.get("upper_cl", pd.Series(np.full(len(mu), np.nan))).to_numpy()
    mu_r, se_r, lo_r, up_r = _apply_response_inverse(info, mu, se, lower, upper)
    frame["emmean"] = mu_r
    frame["SE"] = se_r
    if "lower_cl" in frame.columns:
        frame["lower_cl"] = lo_r
        frame["upper_cl"] = up_r
    fields = {"frame": frame}
    if "type" in emm.__dataclass_fields__:
        fields["type"] = "response"
    return _dc_replace(emm, **fields)


def _response_to_link_emm(emm: Any) -> Any:
    """Rebuild a link-scale EMMResult from a response-scale one.

    Thin wrapper that delegates to the canonical
    :func:`summary_layer._response_to_link_result` (which handles both
    EMM and contrast frames). The private name is kept stable for
    callers inside `pwpm.py`.
    """
    from pymmeans.summary_layer import _response_to_link_result
    return _response_to_link_result(emm)


def pwpm(
    emm: Any,
    adjust: str | None = None,
    by: str | list[str] | None = None,
    means: bool = True,
    flip: bool = False,
    type: str | None = None,
) -> pd.DataFrame | dict[tuple, pd.DataFrame]:
    """Pairwise p-value matrix from an EMMResult.

    Parameters
    ----------
    emm
        Result from :func:`pymmeans.emmeans`.
    adjust
        Multiplicity correction passed to :func:`pymmeans.pairs`.
        ``None`` uses the default ("tukey", or `emm_options('adjust')`).
    by
        No-op for now (the EMMResult's own ``by`` structure governs
        which families are formed). Accepted for R API parity.
    means
        If True (default), put EMMs on the diagonal. If False, leave
        diagonal as NaN.
    flip
        If True, swap the triangles — differences ABOVE, p-values
        BELOW. R default is differences below, p-values above.
    type
        ``'link'`` or ``'response'``. ``None`` inherits ``emm.type``.
        On ``'response'`` for log-family transforms, the diagonal holds
        response means and the off-diagonal "difference" cells hold
        **ratios** (matching R `emmeans::pwpm(em, type='response')`).
        Internally this branch reconstructs a link-scale EMM, runs
        :func:`pymmeans.pairs`, regrids the contrast, and stitches the
        ratio matrix together (an earlier implementation refused this
        path through the `pairs()` response-scale guard).

    Returns
    -------
    pandas.DataFrame (one EMM family) OR dict mapping by-key tuple
    to DataFrame (one per by-group).
    """
    from pymmeans.contrasts import _iter_by_groups, _row_labels, pairs

    # refuse a bootstrap-derived EMM at the
    # ``pwpm()`` wrapper level. Previously the internal ``pairs(emm)``
    # call surfaced the refusal with a generic
    # "pairs() / contrast() are not defined..." message — confusing
    # for users who didn't call ``pairs`` themselves. ``pwpm`` is
    # fundamentally a function of pairwise p-values; bootstrap
    # results don't carry the per-pair analytic uncertainty needed
    # for the matrix.
    if getattr(emm, "df_method", "default") == "bootstrap":
        raise ValueError(
            "pwpm() is not defined for a bootstrap-derived EMMResult "
            "(df_method='bootstrap'). The pairwise p-value matrix "
            "requires analytic-Wald per-pair p-values, which would "
            "silently mix with the stored percentile bootstrap "
            "uncertainty. Compute the matrix on the raw EMM:\n"
            " pwpm(emmeans(model, ...)) # raw EMM\n"
        )

    # route response-scale requests through link.
    emm_type = getattr(emm, "type", "link")
    target_type = type if type is not None else emm_type
    if target_type == "response" and emm_type == "link":
        # User passed type='response' on a link EMM. Run pairs on link,
        # regrid the contrast back to response.
        link_emm = emm
    elif target_type == "response" and emm_type == "response":
        # Response-scale input: rebuild a link-scale view from `linfct`
        # so `pairs()` can run, then regrid the contrast.
        link_emm = _response_to_link_emm(emm)
    elif target_type == "link" and emm_type == "response":
        link_emm = _response_to_link_emm(emm)
    else: # link -> link
        link_emm = emm

    pr = pairs(link_emm, adjust=adjust) if adjust is not None else pairs(link_emm)
    if target_type == "response":
        from pymmeans.transforms import regrid_response as _regrid
        pr = _regrid(pr)
    pair_indices = getattr(pr, "_pair_indices", None)
    use_indices = pair_indices is not None and len(pair_indices) == len(pr.frame)

    # Build one matrix per by-group
    matrices: dict[tuple, pd.DataFrame] = {}
    # We need to know which contrast rows belong to which by-group.
    # Re-derive via the same by-iteration the pairs() builder used.
    # iterate against the link-scale EMM that backed
    # `pairs()`, not the original (response) `emm` — `_iter_by_groups`
    # uses `frame.groupby(by)` which is the same on either, but we
    # also need `linfct` row alignment downstream.
    by_groups: list[tuple[tuple, list[int]]] = list(_iter_by_groups(link_emm))

    # Slice pr.frame by accumulated row counts per by-group
    cursor = 0
    for by_key, indices in by_groups:
        k = len(indices)
        n_pairs = k * (k - 1) // 2
        slice_end = cursor + n_pairs
        family = pr.frame.iloc[cursor:slice_end].reset_index(drop=True)
        family_indices = (
            pair_indices[cursor:slice_end]
            if use_indices
            else None
        )

        # Level labels for this by-group's rows
        sub = emm.frame.iloc[indices].reset_index(drop=True)
        labels = _row_labels(sub, emm.target)
        # Use string labels (categoricals don't index nicely)
        label_strs = [str(lv) for lv in labels]

        # Build empty square matrix
        mat = pd.DataFrame(
            np.full((k, k), np.nan),
            index=label_strs,
            columns=label_strs,
        )

        # Diagonal: EMM point estimates.
        # - If the caller passed a response-scale EMM, `emm.frame`
        # already carries response means → use those directly.
        # - If the caller passed a link-scale EMM but asked for
        # `type='response'` display, the response means aren't on
        # `emm.frame`; back-transform via the model's link / LHS
        # transform (matching `emmeans(..., type='response')`).
        # - Otherwise (link → link), the link-scale EMM frame is the
        # right source.
        if means:
            from pymmeans.summary_layer import _value_col
            if target_type == "response":
                if emm_type == "response":
                    diag_source = emm
                else:
                    diag_source = _link_emm_to_response_view(emm)
                emm_value = _value_col(diag_source)
            else:
                diag_source = link_emm
                emm_value = _value_col(diag_source)
            diag_sub = diag_source.frame.iloc[indices].reset_index(drop=True)
            for i, lab in enumerate(label_strs):
                mat.iloc[i, i] = float(diag_sub[emm_value].iloc[i])

        # Off-diagonal: differences (or ratios) below, p-values above
        # (or swapped under `flip=True`).
        for row_idx in range(n_pairs):
            row = family.iloc[row_idx]
            if family_indices is not None:
                i, j = family_indices[row_idx]
            else:
                # Parse label as fallback
                label = row["contrast"]
                if " - " not in label:
                    continue
                a, b = label.split(" - ", 1)
                if a not in label_strs or b not in label_strs:
                    continue
                i, j = label_strs.index(a), label_strs.index(b)

            est = (row["estimate"] if "estimate" in family.columns
                   else row.get("ratio", np.nan))
            p = float(row["p_value"])

            # i is the positive index in the contrast, j is the negative.
            # For i < j, R puts the i-j difference (estimate) BELOW
            # diagonal at [j, i]; p-value ABOVE at [i, j].
            # We invert sign so [row, col] = row_value - col_value.
            row_a, col_a = i, j
            if flip:
                # Differences above, p-values below
                mat.iloc[row_a, col_a] = float(est)
                mat.iloc[col_a, row_a] = p
            else:
                # R default: differences below, p-values above
                mat.iloc[col_a, row_a] = float(est)
                mat.iloc[row_a, col_a] = p

        matrices[by_key] = mat
        cursor = slice_end

    # If no by, return the single matrix directly (matches R)
    if not emm.by:
        return matrices[()]
    return matrices
