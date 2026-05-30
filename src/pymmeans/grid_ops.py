"""Reference-grid manipulation utilities.

Mirrors R ``emmeans``'s grid-manipulation helpers — ``add_grouping``,
``comb_facs``, ``split_fac``, ``permute_levels`` — that operate on
:class:`~pymmeans.EMMResult` objects after they're built (rather
than during construction). All four functions return a NEW EMMResult;
the input is never mutated.

These are pure DataFrame operations on ``EMMResult.frame`` (and, for
:func:`permute_levels`, a parallel row reorder on
``EMMResult.linfct``). They do not change the underlying ``model_info``
or recompute any marginalization — they only relabel / re-shape the
already-computed EMMs for downstream contrast / display ergonomics.

Examples
--------
>>> from pymmeans import add_grouping, comb_facs, split_fac, permute_levels  # doctest: +SKIP
"""

from __future__ import annotations

from dataclasses import replace as _dc_replace
from typing import Any

import numpy as np
import pandas as pd

from pymmeans.emmeans import EMMResult


def _ensure_emm(obj: Any, op: str) -> EMMResult:
    """Refuse anything that isn't a bare :class:`EMMResult`.

    Reusing the metadata-rewrite path on a :class:`ContrastResult`
    would silently corrupt the contrast frame (``estimate`` / ``SE``
    / ``t_ratio`` semantics differ from EMM rows). Refuse cleanly so
    the user sees the API mismatch early.
    """
    if not isinstance(obj, EMMResult):
        raise TypeError(
            f"{op} requires an EMMResult input; got "
            f"{obj.__class__.__name__}. Grid-manipulation utilities "
            "operate on the post-emmeans grid (EMMResult.frame), not "
            "on ContrastResult / EmmList / RefGrid."
        )
    return obj


def add_grouping(
    emm: EMMResult,
    new_factor: str,
    refname: str,
    newlevs: list | dict,
) -> EMMResult:
    """Add a new grouping factor by mapping an existing factor's levels.

    Mirrors R ``emmeans::add_grouping(em, newname, refname, newlevs)``.
    Each level of ``refname`` (an existing factor column in
    ``emm.frame``) is mapped through ``newlevs`` to a new level of
    ``new_factor``. The resulting EMMResult has the new column added
    to its frame; ``linfct`` is unchanged (the underlying predictions
    are unchanged, just relabelled).

    Useful for nested-style aggregation post-hoc: e.g., an EMM over
    ``treatment`` with 4 levels can be augmented with a
    ``drug_class`` column mapping ``placebo -> "control"`` and
    ``drugA / drugB / drugC -> "active"``, enabling downstream
    marginal queries on ``drug_class``.

    Parameters
    ----------
    emm
        Source EMMResult. Must contain ``refname`` as a column.
    new_factor
        Name for the new column. Must not already exist in
        ``emm.frame.columns``.
    refname
        Existing factor column whose levels are being mapped.
    newlevs
        Either:

        - ``dict[level -> new_level]`` (recommended, unambiguous):
          explicit mapping by source level. Every level present in
          ``emm.frame[refname]`` must appear as a key.
        - ``list[new_level]`` (positional): ``newlevs[i]`` is
          assigned to rows where ``refname`` takes its ``i``-th
          *categorical level* (``cat.categories`` order for
          Categorical columns; first-occurrence order otherwise).
          Length must equal the number of distinct levels.

        Tip: prefer the dict form unless you're certain about the
        categorical level order — the positional form silently maps
        ``placebo`` to the last slot if it sorts alphabetically
        last among the four levels.

    Returns
    -------
    EMMResult
        A new EMMResult with ``new_factor`` added to the frame and
        appended to the ``target`` list.

    Raises
    ------
    TypeError
        If ``emm`` is not an :class:`EMMResult`.
    ValueError
        On any of: ``refname`` missing, ``new_factor`` colliding with
        an existing column, ``newlevs`` length mismatch (positional)
        or unknown / missing key (dict).
    """
    em = _ensure_emm(emm, "add_grouping()")
    if not isinstance(new_factor, str) or not new_factor:
        raise ValueError(
            "add_grouping(new_factor=...) must be a non-empty string."
        )
    if not isinstance(refname, str) or not refname:
        raise ValueError(
            "add_grouping(refname=...) must be a non-empty string."
        )
    if refname not in em.frame.columns:
        raise ValueError(
            f"add_grouping: refname={refname!r} is not a column in "
            f"emm.frame. Available columns: {list(em.frame.columns)}."
        )
    if new_factor in em.frame.columns:
        raise ValueError(
            f"add_grouping: new_factor={new_factor!r} already exists "
            "in emm.frame. Choose a different name or drop the column "
            "first."
        )
    # Discover the levels actually present in the column. Use the
    # Categorical level order when available (R's convention), else
    # first-occurrence order via ``pd.unique`` (also stable).
    col = em.frame[refname]
    if isinstance(col.dtype, pd.CategoricalDtype):
        existing_levels = [
            lv for lv in col.cat.categories if lv in set(col)
        ]
    else:
        existing_levels = list(pd.unique(col))

    if isinstance(newlevs, dict):
        # Dict form (unambiguous). Validate that every actually-
        # present level is keyed.
        keys = set(newlevs)
        missing = [lv for lv in existing_levels if lv not in keys]
        if missing:
            raise ValueError(
                f"add_grouping: dict newlevs is missing entries for "
                f"levels {missing!r} of {refname!r}. Every present "
                f"level needs a mapping (existing levels: "
                f"{existing_levels})."
            )
        extras = [k for k in newlevs if k not in set(existing_levels)]
        if extras:
            raise ValueError(
                f"add_grouping: dict newlevs has extra keys "
                f"{extras!r} not in the present levels "
                f"{existing_levels} of {refname!r}."
            )
        mapping: dict[Any, Any] = dict(newlevs)
    else:
        # Positional list form. Length must match the level count.
        new_levs_list = list(newlevs)
        if len(new_levs_list) != len(existing_levels):
            raise ValueError(
                f"add_grouping: newlevs has {len(new_levs_list)} "
                f"entries but {refname!r} has "
                f"{len(existing_levels)} distinct levels "
                f"({existing_levels}). Each existing level needs "
                "exactly one new-level mapping (use the same string "
                "twice to merge levels). Prefer the dict form "
                "{level: new_level} to avoid silent positional "
                "misalignment."
            )
        mapping = dict(zip(existing_levels, new_levs_list, strict=True))

    new_frame = em.frame.copy()
    new_frame[new_factor] = new_frame[refname].map(mapping)
    # Append the new factor to ``target`` (it's now a grid dimension
    # the EMM result advertises). Preserve order so downstream
    # display / pairs / contrast see the existing targets first.
    new_target = [*em.target, new_factor]
    return _dc_replace(em, frame=new_frame, target=new_target)


def comb_facs(
    emm: EMMResult,
    factors: list[str],
    new_name: str,
    sep: str = ":",
) -> EMMResult:
    """Combine two or more factor columns into a single concatenated factor.

    Mirrors R ``emmeans::comb_facs(em, c("a", "b"), sep=":")``: the
    combined column's value for each row is the joined string of the
    component columns' values. The component columns are dropped from
    the frame (matching R), but ``linfct`` is unchanged (only labels
    move; the underlying predictions stay put).

    Parameters
    ----------
    emm
        Source EMMResult.
    factors
        Sequence of existing column names to combine. Must contain at
        least two distinct names, all present in ``emm.frame``.
    new_name
        Name for the new combined column. Must not collide with any
        column that survives the operation.
    sep
        Separator inserted between component values in the combined
        string. Default ``":"`` matches R.

    Returns
    -------
    EMMResult
        A new EMMResult with the component columns dropped and
        ``new_name`` added.

    Raises
    ------
    TypeError
        If ``emm`` is not an :class:`EMMResult`.
    ValueError
        On any of: fewer than 2 factors, duplicate names, unknown
        column, or ``new_name`` collision.
    """
    em = _ensure_emm(emm, "comb_facs()")
    factors_list = list(factors)
    if len(factors_list) < 2:
        raise ValueError(
            f"comb_facs requires at least 2 factors to combine; got "
            f"{factors_list!r}."
        )
    if len(set(factors_list)) != len(factors_list):
        raise ValueError(
            f"comb_facs: factors list has duplicates: {factors_list!r}."
        )
    missing = [f for f in factors_list if f not in em.frame.columns]
    if missing:
        raise ValueError(
            f"comb_facs: unknown columns {missing!r}. Available: "
            f"{list(em.frame.columns)}."
        )
    if not isinstance(new_name, str) or not new_name:
        raise ValueError(
            "comb_facs(new_name=...) must be a non-empty string."
        )
    # ``new_name`` may shadow one of the inputs (e.g. combine ('a',
    # 'b') into 'a' when the user wants to drop 'b' under a renamed
    # combined column). Refuse only when it would collide with a
    # column NOT in the inputs (preserved columns).
    preserved = [c for c in em.frame.columns if c not in factors_list]
    if new_name in preserved:
        raise ValueError(
            f"comb_facs: new_name={new_name!r} collides with an "
            "existing non-component column. Choose a different name."
        )
    new_frame = em.frame.copy()
    # Build the combined column from string-cast components so
    # heterogeneous dtypes (Categorical, str, int) all concatenate
    # cleanly.
    combined = (
        new_frame[factors_list[0]].astype(str).copy()
    )
    for f in factors_list[1:]:
        combined = combined + sep + new_frame[f].astype(str)
    new_frame = new_frame.drop(columns=factors_list)
    new_frame[new_name] = combined.values
    # Update target / by: drop component names, append the new one
    # iff the components were already targets (otherwise just drop).
    components_were_target = any(f in em.target for f in factors_list)
    new_target = [t for t in em.target if t not in factors_list]
    if components_were_target:
        new_target.append(new_name)
    new_by = [b for b in em.by if b not in factors_list]
    return _dc_replace(emm, frame=new_frame, target=new_target, by=new_by)


def split_fac(
    emm: EMMResult,
    factor: str,
    new_names: list[str],
    sep: str = ":",
) -> EMMResult:
    """Split a combined factor column back into multiple component columns.

    Inverse of :func:`comb_facs`. Splits the named column's values
    by ``sep`` into ``len(new_names)`` parts; raises if any row's
    value doesn't split into exactly that many parts.

    Parameters
    ----------
    emm
        Source EMMResult.
    factor
        Existing column to split.
    new_names
        Names for the resulting component columns, in left-to-right
        order. Must contain at least two distinct names and must not
        collide with any other surviving column.
    sep
        Separator to split on. Default ``":"``.

    Returns
    -------
    EMMResult
        A new EMMResult with ``factor`` dropped and the component
        columns added.
    """
    em = _ensure_emm(emm, "split_fac()")
    if not isinstance(factor, str) or not factor:
        raise ValueError(
            "split_fac(factor=...) must be a non-empty string."
        )
    if factor not in em.frame.columns:
        raise ValueError(
            f"split_fac: factor={factor!r} is not a column in "
            f"emm.frame. Available: {list(em.frame.columns)}."
        )
    names_list = list(new_names)
    if len(names_list) < 2:
        raise ValueError(
            f"split_fac requires at least 2 new_names; got "
            f"{names_list!r}."
        )
    if len(set(names_list)) != len(names_list):
        raise ValueError(
            f"split_fac: new_names has duplicates: {names_list!r}."
        )
    # Refuse collisions with non-source columns (the source column
    # itself is dropped, so it doesn't count).
    preserved = [c for c in em.frame.columns if c != factor]
    collisions = [n for n in names_list if n in preserved]
    if collisions:
        raise ValueError(
            f"split_fac: new_names {collisions!r} collide with "
            "existing non-source columns. Rename or drop those "
            "columns first."
        )
    # Materialise the split.
    source = em.frame[factor].astype(str)
    parts = source.str.split(sep, expand=True)
    if parts.shape[1] != len(names_list):
        # Sample the bad row for the message — usually one
        # malformed value is the culprit.
        actual_widths = source.str.count(sep) + 1
        bad_idx = (actual_widths != len(names_list)).idxmax() if (
            (actual_widths != len(names_list)).any()
        ) else None
        bad_sample = em.frame.loc[bad_idx, factor] if bad_idx is not None else "?"
        raise ValueError(
            f"split_fac({factor!r}, sep={sep!r}): expected "
            f"{len(names_list)} parts per row but got "
            f"{parts.shape[1]} on at least one row (first bad "
            f"value: {bad_sample!r}). Verify ``sep`` and the source "
            "column's format."
        )
    new_frame = em.frame.copy().drop(columns=[factor])
    for i, n in enumerate(names_list):
        new_frame[n] = parts.iloc[:, i].values
    # Update target / by.
    factor_was_target = factor in em.target
    new_target = [t for t in em.target if t != factor]
    if factor_was_target:
        new_target.extend(names_list)
    new_by = [b for b in em.by if b != factor]
    return _dc_replace(emm, frame=new_frame, target=new_target, by=new_by)


def permute_levels(
    emm: EMMResult,
    factor: str,
    perm: list,
) -> EMMResult:
    """Reorder the EMM rows so the named factor's levels appear in a new order.

    Mirrors R ``emmeans::permute_levels(em, factor, perm)``. The
    ``perm`` argument can be either:

    - A list of integers (0-based positional permutation): each
      element ``i`` says "place the element currently at position
      ``perm[i]`` at output position ``i``". Length must equal the
      number of distinct levels.
    - A list of level labels (named permutation): the levels are
      reordered to match the supplied sequence directly.

    The function performs a stable sort, so rows that share a level
    of ``factor`` retain their original relative order on every
    other column.

    ``linfct`` rows are reordered in lockstep with the frame so the
    EMM <-> design-matrix correspondence stays intact for downstream
    ``pairs`` / ``contrast`` operations.

    Parameters
    ----------
    emm
        Source EMMResult.
    factor
        Column whose levels are being permuted.
    perm
        Positional permutation (list of unique ints) OR named
        permutation (list of labels matching the current levels).

    Returns
    -------
    EMMResult
        A new EMMResult with reordered rows / linfct.
    """
    em = _ensure_emm(emm, "permute_levels()")
    if not isinstance(factor, str) or not factor:
        raise ValueError(
            "permute_levels(factor=...) must be a non-empty string."
        )
    if factor not in em.frame.columns:
        raise ValueError(
            f"permute_levels: factor={factor!r} is not a column in "
            f"emm.frame. Available: {list(em.frame.columns)}."
        )
    perm_list = list(perm)
    if not perm_list:
        raise ValueError(
            "permute_levels: perm must contain at least one element."
        )
    current_levels = list(pd.unique(em.frame[factor]))
    k = len(current_levels)
    # Branch on positional vs named. ``isinstance`` checks ``bool``
    # too (bool is a subclass of int); exclude it explicitly so
    # ``perm=[True, False]`` doesn't get treated as positional.
    is_positional = all(
        isinstance(p, (int, np.integer)) and not isinstance(p, bool)
        for p in perm_list
    )
    if is_positional:
        if sorted(perm_list) != list(range(k)):
            raise ValueError(
                f"permute_levels: positional perm must be a "
                f"permutation of 0..{k - 1}; got {perm_list!r} for "
                f"{k} levels."
            )
        new_order = [current_levels[i] for i in perm_list]
    else:
        new_order = list(perm_list)
        if len(set(new_order)) != len(new_order):
            raise ValueError(
                f"permute_levels: named perm has duplicate entries: "
                f"{new_order!r}."
            )
        if set(new_order) != set(current_levels):
            extra = sorted(set(new_order) - set(current_levels), key=str)
            missing = sorted(set(current_levels) - set(new_order), key=str)
            raise ValueError(
                f"permute_levels: named perm must be a permutation "
                f"of the current levels {current_levels!r}. "
                f"Extras in perm: {extra!r}. Missing from perm: "
                f"{missing!r}."
            )
    # Build the row-rank map for a stable sort that places rows of
    # the first new-order level first, second next, etc.
    rank = {lv: i for i, lv in enumerate(new_order)}
    new_frame = em.frame.copy()
    sort_keys = new_frame[factor].map(rank).to_numpy()
    new_idx = np.argsort(sort_keys, kind="stable")
    new_frame = new_frame.iloc[new_idx].reset_index(drop=True)
    new_linfct = em.linfct[new_idx]
    return _dc_replace(emm, frame=new_frame, linfct=new_linfct)


def force_regular(emm: EMMResult) -> EMMResult:
    """Expand an irregular EMM grid back to a full Cartesian product.

    Mirrors R ``emmeans::force_regular``: take an
    :class:`~pymmeans.EMMResult` whose ``frame`` is missing some
    factor-level combinations (typically because :func:`add_grouping`
    introduced a partial nesting or because ``emmeans(...,
    nesting=...)`` filtered the grid to observed tuples) and
    reflate it to the full Cartesian product of every factor's
    unique values. Cells that were already present keep their
    estimates verbatim. Cells that need to be added get ``NaN`` in
    ``emmean`` / ``SE`` / ``lower_cl`` / ``upper_cl`` / ``p_value``
    / ``t_ratio`` / ``df`` / ``z_ratio`` (whichever columns the
    source frame carries) and an all-zero row in the ``linfct``
    matrix.

    Parameters
    ----------
    emm
        The :class:`EMMResult` to reflate.

    Returns
    -------
    EMMResult
        A new result whose ``frame`` is the full Cartesian product of
        the factor columns currently in ``frame``. Display-only —
        rows added by reflation are NOT estimable, so downstream
        ``pairs(reflated)`` / ``contrast(reflated)`` will produce
        ``NaN`` for any contrast that touches an added row. Use this
        for tabular display when you want a regular k1 × k2 layout
        even though the underlying model didn't fit every cell.

    Raises
    ------
    TypeError
        If ``emm`` is not an :class:`EMMResult`.
    ValueError
        If the frame has zero factor columns (nothing to reflate).

    Notes
    -----
    R's ``force_regular`` also marks reflated rows with an
    ``estimable=FALSE`` flag. We add an ``estimable`` boolean column
    iff at least one row is reflated; rows that were present in the
    original frame get ``True``, added rows get ``False``. If the
    frame is already regular, ``emm`` is returned unchanged (no
    ``estimable`` column added).
    """
    import itertools

    em = _ensure_emm(emm, "force_regular")
    frame = em.frame

    # Factor columns: everything that is NOT one of the numeric
    # display columns. The display columns are stable across
    # EMMResult / posterior frames so we can enumerate them.
    display_cols = {
        "emmean", "SE", "df", "lower_cl", "upper_cl",
        "t_ratio", "z_ratio", "p_value", "asymp_LCL", "asymp_UCL",
    }
    factor_cols = [c for c in frame.columns if c not in display_cols]
    if not factor_cols:
        raise ValueError(
            "force_regular: no factor columns in the EMM frame "
            f"(columns: {list(frame.columns)}). Reflation requires at "
            "least one factor whose Cartesian product can be expanded."
        )

    # Per-factor unique levels, preserving Categorical order if any.
    per_factor_levels: list[list[Any]] = []
    for col in factor_cols:
        series = frame[col]
        if isinstance(series.dtype, pd.CategoricalDtype):
            levels = list(series.cat.categories)
        else:
            # numeric / object: use first-appearance order
            levels = list(pd.Series(series.unique()))
        per_factor_levels.append(levels)

    full_combos = list(itertools.product(*per_factor_levels))
    expected_n = len(full_combos)
    if expected_n == em.n_rows:
        # already regular — no-op.
        return em

    # Build the expanded frame in Cartesian order, then merge
    # present-row values in.
    full = pd.DataFrame(full_combos, columns=factor_cols)
    # Preserve Categorical dtypes on the expanded frame so the
    # downstream sort order matches the original.
    for col, levels in zip(factor_cols, per_factor_levels, strict=True):
        if isinstance(frame[col].dtype, pd.CategoricalDtype):
            full[col] = pd.Categorical(full[col], categories=levels)

    # Tuple-key lookup of existing rows (handles NaN-safe matching).
    src_keys = list(
        zip(*(frame[c].tolist() for c in factor_cols), strict=True)
    )
    src_index = {k: i for i, k in enumerate(src_keys)}

    # Reflated rows get NaN linfct (not zeros)
    # so any downstream linfct arithmetic — pairs(), contrast(),
    # joint_tests(), etc. — surfaces NaN at the row that touched a
    # non-estimable cell instead of silently producing a finite
    # garbage value. With zeros, `pairs()` would compute
    # ``L_i - L_j = L_i - 0 = L_i`` (a single cell masquerading as a
    # contrast); with NaN, the subtraction poisons the contrast row
    # and the user sees the non-estimability instead of a
    # plausible-looking but wrong number.
    new_linfct = np.full(
        (expected_n, em.linfct.shape[1]),
        np.nan,
        dtype=em.linfct.dtype,
    )
    estimable = np.zeros(expected_n, dtype=bool)
    new_rows = []
    for i, combo in enumerate(full_combos):
        src_i = src_index.get(combo)
        if src_i is None:
            row = {c: combo[j] for j, c in enumerate(factor_cols)}
            for dc in display_cols:
                if dc in frame.columns:
                    row[dc] = np.nan
            new_rows.append(row)
            estimable[i] = False
        else:
            row = frame.iloc[src_i].to_dict()
            new_rows.append(row)
            new_linfct[i] = em.linfct[src_i]
            estimable[i] = True

    new_frame = pd.DataFrame(new_rows, columns=frame.columns)
    new_frame["estimable"] = estimable
    # Preserve Categorical dtypes after DataFrame round-trip.
    for col, levels in zip(factor_cols, per_factor_levels, strict=True):
        if isinstance(frame[col].dtype, pd.CategoricalDtype):
            new_frame[col] = pd.Categorical(
                new_frame[col], categories=levels
            )
    new_frame = new_frame.reset_index(drop=True)
    return _dc_replace(em, frame=new_frame, linfct=new_linfct)
