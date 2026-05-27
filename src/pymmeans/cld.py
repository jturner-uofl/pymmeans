"""Compact letter display (Piepho 2004) for EMM results.

Given a set of pairwise-comparison results, ``cld()`` assigns each EMM
a string of letters such that two EMMs share at least one letter iff
they are NOT significantly different at level ``alpha``. The result
mirrors R ``emmeans::cld`` (which in turn uses the same Piepho /
multcomp::cld algorithm) and is a standard reporting format in
agronomy, pharma, and ANOVA-heavy social-science fields:

    A emmean .group
    a -1.20 a
    b 0.05 ab
    c 0.40 bc
    d 1.85 c

Reading: EMMs sharing a letter are NOT significantly different;
EMMs with disjoint letter sets ARE significantly different.

Algorithm (Piepho 2004, simplified for ordered means)
-----------------------------------------------------

1. Compute pairwise adjusted p-values via :func:`pymmeans.pairs`.
2. Sort EMMs ascending by point estimate.
3. Build the **non-significance graph** ``G`` whose edge ``(i, j)``
   means ``p_adj(i, j) > alpha``.
4. Find all **maximal non-significance intervals** on the sorted
   order: contiguous index ranges ``[lo, hi]`` such that every pair
   inside is non-significant, and extending either endpoint breaks
   the property. Each maximal interval becomes one letter.
5. Each EMM's letter set is the union of letters of intervals
   containing it.

For ordered means with transitive non-significance (the typical ANOVA
pattern), this is the **minimum-letter assignment** — i.e. you cannot
re-letter with fewer cliques. Pathological non-transitive cases (e.g.
mean B differs from A and C but A doesn't differ from C) get a valid
letter display but possibly with more letters than a clique-cover
minimum; ``cld()`` documents this and prints a warning when detected.

By-grouped EMMs: letters are assigned **within** each by-group as a
separate family, matching R `cld()` behavior.

References
----------
- Piepho, H. P. (2004). "An algorithm for a letter-based
  representation of all-pairwise comparisons." *Journal of
  Computational and Graphical Statistics* 13(2), 456-466.
- Lenth, R. (2024). ``cld.emmGrid`` documentation in the ``emmeans``
  R package.
"""

from __future__ import annotations

import string
from typing import Any

import numpy as np
import pandas as pd

from pymmeans.contrasts import _iter_by_groups, _row_labels, pairs
from pymmeans.emmeans import EMMResult


def _letter_for_index(i: int) -> str:
    """Map letter-index 0..25 -> a..z, 26.. -> _27_, _28_, ...

    Beyond 26 maximal cliques we fall back to numeric placeholders so
    the printed string stays parseable. In practice you rarely see
    >10 letters even with k=20 means.
    """
    if 0 <= i < 26:
        return string.ascii_lowercase[i]
    return f"_{i + 1}_"


def _maximal_nonsig_intervals(ns: np.ndarray) -> list[tuple[int, int]]:
    """Find maximal intervals on a sorted index where every pair is
    non-significantly different.

    Parameters
    ----------
    ns
        ``(k, k)`` boolean symmetric matrix of non-significance flags
        in the SORTED order. Diagonal is True (each EMM is non-sig
        with itself).

    Returns
    -------
    list of (lo, hi) pairs
        Each (lo, hi) is a maximal non-significance interval; lo and
        hi are sorted-order indices, both inclusive.
    """
    k = ns.shape[0]
    # Compute the rightmost extent for each starting index `lo`:
    # the largest `hi >= lo` such that all pairs in [lo..hi] are non-sig.
    extents: list[tuple[int, int]] = []
    for lo in range(k):
        hi = lo
        while hi + 1 < k:
            # The new index hi+1 must be non-sig with every i in [lo..hi].
            # Equivalently: row `ns[hi+1, lo..hi]` is all True.
            if np.all(ns[hi + 1, lo:hi + 1]):
                hi += 1
            else:
                break
        extents.append((lo, hi))

    # Keep only the maximal ones (those not strictly contained in another).
    # An interval (lo, hi) is non-maximal iff there's some other interval
    # (lo', hi') with lo' <= lo and hi' >= hi and (lo', hi') != (lo, hi).
    maximal: list[tuple[int, int]] = []
    for iv in extents:
        is_max = True
        for other in extents:
            if other == iv:
                continue
            if other[0] <= iv[0] and other[1] >= iv[1]:
                is_max = False
                break
        if is_max:
            maximal.append(iv)
    # Deduplicate while preserving order (multiple `lo`s may produce the
    # same maximal interval).
    seen: set[tuple[int, int]] = set()
    unique: list[tuple[int, int]] = []
    for iv in maximal:
        if iv not in seen:
            seen.add(iv)
            unique.append(iv)
    return unique


def _check_transitivity(ns: np.ndarray) -> bool:
    """Return True iff the non-significance relation is transitive on
    the sorted order — i.e. if EMMs i < j < k have ns[i,j] and ns[j,k]
    then ns[i,k]. Non-transitive cases get a warning from `cld` because
    the interval algorithm is no longer guaranteed to be minimal."""
    k = ns.shape[0]
    for i in range(k):
        for j in range(i + 1, k):
            if not ns[i, j]:
                continue
            for ell in range(j + 1, k):
                if ns[j, ell] and not ns[i, ell]:
                    return False
    return True


def cld(
    emm: EMMResult,
    alpha: float = 0.05,
    adjust: str = "tukey",
    sort: bool = True,
    reverse: bool = False,
) -> pd.DataFrame:
    """Compact letter display: tag each EMM with a letter group.

    Two EMMs share at least one letter iff they are NOT significantly
    different at level ``alpha`` (after the chosen multiplicity
    adjustment). The format mirrors ``emmeans::cld`` and is the
    standard reporting convention in agronomy / pharma / many
    ANOVA-heavy fields.

    Parameters
    ----------
    emm
        Result from :func:`pymmeans.emmeans` on the **link** scale.
        Response-scale EMMs are refused — letter assignment depends
        on pairwise differences, which on a non-linear link are not
        the same as differences of back-transformed means.
    alpha
        Significance threshold for the letter split. Default 0.05.
    adjust
        Multiplicity correction passed to :func:`pymmeans.pairs`.
        Default ``"tukey"`` (matches R ``cld`` default for balanced
        designs). Pass ``"none"`` for raw pairwise letters.
    sort
        If True (default), sort the output frame ascending by EMM
        estimate within each by-group so letters read in increasing
        order.
    reverse
        If True, sort descending instead.

    Returns
    -------
    pandas.DataFrame
        Copy of ``emm.frame`` with a new ``.group`` column containing
        the letter assignment per row. Non-estimable rows (emmean is
        NaN) get an empty ``.group`` string.

    Raises
    ------
    ValueError
        If ``emm`` is on the response scale, or ``alpha`` is outside
        ``(0, 1)``.

    Examples
    --------
    Doctest runs with ``+SKIP`` because it relies on a fitted OLS; the
    snippet below is the canonical recipe. The random draws are seeded
    (``np.random.default_rng(0)``) so the printed letter assignments
    are reproducible — the unseeded form previously here produced
    different ``.group`` columns on every render and could not be
    verified by ``doctest`` even with ``+SKIP``.

    >>> import statsmodels.formula.api as smf # doctest: +SKIP
    >>> import pandas as pd, numpy as np # doctest: +SKIP
    >>> rng = np.random.default_rng(0) # doctest: +SKIP
    >>> df = pd.DataFrame({ # doctest: +SKIP
    ... "a": rng.choice(list("ABCD"), 100),
    ... }) # doctest: +SKIP
    >>> df["y"] = (df["a"] == "B") * 0.5 + rng.normal(size=100) # doctest: +SKIP
    >>> from pymmeans import emmeans, cld # doctest: +SKIP
    >>> from pymmeans.utils import from_fitted # doctest: +SKIP
    >>> info = from_fitted(smf.ols("y ~ a", df).fit()) # doctest: +SKIP
    >>> cld(emmeans(info, "a")) # doctest: +SKIP
       a emmean SE df lower_cl upper_cl .group
    0 A -0.123 0.143 ... ... ... a
    1 B 0.530 0.144 ... ... ... b
    ...
    """
    # explicit type check. previously passing a
    # ``ContrastResult`` (or an ``rbind`` output, which is a
    # ContrastResult with ``method="rbind"``) crashed deep inside
    # ``pairs()`` with a confusing ``KeyError: '<target>'``.
    # ``cld`` is defined on the EMM grid (it labels means, not
    # contrasts); calling it on a contrast result has no statistical
    # meaning. Refuse cleanly.
    from pymmeans.contrasts import ContrastResult as _ContrastResult
    if isinstance(emm, _ContrastResult):
        raise TypeError(
            "cld() requires an EMMResult (the input to ``pairs``), "
            "not a ContrastResult. Letter displays label the EMM "
            "*means*, not the differences between them. If you have "
            "the contrasts already, recover the EMM via the source "
            f"call. (Got a ContrastResult with method={emm.method!r}.)"
        )
    # refuse a bootstrap-derived EMM at the cld()
    # wrapper level with a clear, cld-specific error. previously the
    # internal ``pairs(emm)`` call surfaced the refusal
    # ("pairs() / contrast() are not defined for a bootstrap-derived
    # EMMResult..."), which confused users who hadn't called
    # ``pairs`` themselves. cld is fundamentally a function of
    # pairwise comparisons, and bootstrap-derived EMMs don't carry
    # the per-pair uncertainty needed for letter assignment; refuse
    # at this layer so the workflow hint is actionable.
    if getattr(emm, "df_method", "default") == "bootstrap":
        raise ValueError(
            "cld() is not defined for a bootstrap-derived EMMResult "
            "(df_method='bootstrap'). Letter displays require "
            "per-pair p-values, which would come from analytic-Wald "
            "math on a result whose stored uncertainty is percentile "
            "bootstrap — silently mixing two inference paradigms. "
            "Compute the letter display on the raw EMM:\n"
            " cld(emmeans(model, ...)) # raw EMM, then optional\n"
            " bootstrap_ci(emmeans(...)) # for the CI columns\n"
        )
    if getattr(emm, "type", "link") == "response":
        raise ValueError(
            "cld() requires a link-scale EMMResult; pairwise letter "
            "assignment depends on link-scale differences. On a "
            "log-family transform `regrid_response`'s ratios aren't "
            "the same comparison; on probit/cloglog the differences "
            "aren't even on a meaningful scale. Call cld() on the "
            "link-scale EMM, then regrid the result yourself if you "
            "need a response-scale display column."
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}.")

    # Compute all pairwise comparisons once for the entire grid; we'll
    # filter to within-by-group rows inside the loop.
    # opt past the `max_contrasts=50` footgun
    # guard — `cld()` legitimately needs all k(k-1)/2 pairs to build
    # the letter display, even for k=20 (190 contrasts) or larger.
    pr = pairs(emm, adjust=adjust, max_contrasts=None)

    out = emm.frame.copy()
    out[".group"] = ""

    # Build a label -> p-value map keyed by the contrast label so we
    # don't have to re-derive contrast indices.
    # use the structural (i, j) row
    # indices that `pairs()` now attaches as the `_pair_indices`
    # dataclass field (used frame columns, which leaked
    # through effect_size). fallback for label-parsing is
    # retained for ContrastResults built outside `pairs()`.
    pair_indices = getattr(pr, "_pair_indices", None)
    has_indices = pair_indices is not None and len(pair_indices) == len(pr.frame)
    pr_lookup_by_index: dict[tuple[Any, ...], dict[tuple[int, int], float]] = {}
    pr_lookup_by_label: dict[tuple[Any, ...], dict[tuple[str, str], float]] = {}
    for row_idx, (_, row) in enumerate(pr.frame.iterrows()):
        if emm.by:
            by_key = tuple(row[c] for c in emm.by)
        else:
            by_key = ()
        p = float(row["p_value"])
        if has_indices:
            i, j = pair_indices[row_idx]
            pr_lookup_by_index.setdefault(by_key, {})[(i, j)] = p
            pr_lookup_by_index[by_key][(j, i)] = p
        else:
            label = row["contrast"]
            if " - " not in label:
                continue
            a, b = label.split(" - ", 1)
            pr_lookup_by_label.setdefault(by_key, {})[(a, b)] = p
            pr_lookup_by_label[by_key][(b, a)] = p

    non_transitive_seen = False

    for by_key, indices in _iter_by_groups(emm):
        sub = emm.frame.iloc[indices].reset_index(drop=True)
        labels = _row_labels(sub, emm.target)
        k = len(labels)
        if k == 0:
            continue
        if k == 1:
            out.loc[indices[0], ".group"] = _letter_for_index(0)
            continue

        # Build the non-significance matrix in ORIGINAL (within-group)
        # order, then sort.
        ns = np.eye(k, dtype=bool)
        if has_indices:
            idx_bucket = pr_lookup_by_index.get(by_key, {})
            for i in range(k):
                for j in range(i + 1, k):
                    p = idx_bucket.get((i, j))
                    if p is None:
                        continue
                    ns[i, j] = ns[j, i] = (p > alpha)
        else:
            label_bucket = pr_lookup_by_label.get(by_key, {})
            for i in range(k):
                for j in range(i + 1, k):
                    p = label_bucket.get((labels[i], labels[j]))
                    if p is None:
                        continue
                    ns[i, j] = ns[j, i] = (p > alpha)

        # Non-estimable rows (NaN emmean) — keep them but assign empty
        # group so they don't pollute the letter assignment.
        emm_values = sub["emmean"].to_numpy(dtype=float)
        valid_mask = np.isfinite(emm_values)
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            continue
        if len(valid_idx) < k:
            # Restrict ns to the valid subset
            ns_valid = ns[np.ix_(valid_idx, valid_idx)]
        else:
            ns_valid = ns
        kv = len(valid_idx)

        # Sort valid EMMs by point estimate
        sort_order = np.argsort(emm_values[valid_idx])
        if reverse:
            sort_order = sort_order[::-1]
        ns_sorted = ns_valid[np.ix_(sort_order, sort_order)]

        # Transitivity check (warn on non-transitive non-sig relation)
        if not non_transitive_seen and not _check_transitivity(ns_sorted):
            non_transitive_seen = True
            import warnings as _w
            _w.warn(
                "cld(): the non-significance relation is not "
                "transitive on the sorted means (means i < j < k where "
                "i~j and j~k but i!=k). The interval-graph letter "
                "assignment is still valid but may use more letters "
                "than a minimum clique cover.",
                UserWarning, stacklevel=2,
            )

        maximal = _maximal_nonsig_intervals(ns_sorted)
        # Assign letters
        letter_sets: list[set[str]] = [set() for _ in range(kv)]
        for li, (lo, hi) in enumerate(maximal):
            letter = _letter_for_index(li)
            for m in range(lo, hi + 1):
                letter_sets[sort_order[m]].add(letter)

        # Format and write back. The within-group order matches the
        # original frame's row order (we didn't sort the frame, only the
        # internal letter computation), so map back via valid_idx.
        for j_in_valid, idx_in_sub in enumerate(valid_idx):
            row_in_orig = indices[idx_in_sub]
            out.loc[row_in_orig, ".group"] = "".join(sorted(letter_sets[j_in_valid]))

    if sort:
        sort_keys = list(emm.by) + ["emmean"] if emm.by else ["emmean"]
        out = out.sort_values(
            sort_keys, ascending=not reverse, kind="stable"
        ).reset_index(drop=True)
    return out
