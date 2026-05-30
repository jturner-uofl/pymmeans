"""Reference grid construction.

The reference grid is the cross of all factor levels with numeric covariates
held at their training-data means. ``mu = L @ beta`` evaluated on this grid
gives predictions at each combination; marginal means come from averaging
rows of ``L`` that share a target factor level.
"""

from __future__ import annotations

import itertools
import keyword
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from patsy import build_design_matrices

from pymmeans.utils import ModelInfo, from_fitted


@dataclass(frozen=True)
class RefGrid:
    """Reference grid: the points at which marginal means are evaluated."""

    grid: pd.DataFrame
    linfct: np.ndarray
    model_info: ModelInfo

    @property
    def n_rows(self) -> int:
        """Number of rows in the reference grid (= rows of ``linfct``)."""
        return len(self.grid)


def _is_plain_identifier(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name)


def _as_list(value: Any) -> list:
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return list(value)
    return [value]


def _validate_design(info: ModelInfo) -> None:
    """Raise NotImplementedError for non-identifier (expression) factors.

    registered multi-column basis
    factors (``bs(x, df=3)``, ``cr(x, df=4)``, etc.) are
    explicitly *allowed* here — the analytic and eager grid paths
    both know how to re-evaluate the basis at user / mean values
    via patsy's stored factor state. Pre-this check
    refused them with a misleading "v0.1 supports only identifier
    factors" message even though the analytic path supported
    them.
    """
    multi_col = getattr(info, "multi_col_factors", {}) or {}
    for factor in info.design_info.factor_infos:
        name = factor.name()
        if _is_plain_identifier(name):
            continue
        if name in multi_col:
            # Registered multi-col basis — analytic + eager grid
            # paths handle this via patsy ``factor.eval(state, ...)``
            # re-evaluation.
            continue
        raise NotImplementedError(
            f"Term '{name}' is an expression, not a plain column "
            "reference. The eager ``ref_grid`` builder requires plain "
            "identifier factors; multi-column basis expressions like "
            "``bs(x, df=3)`` are supported via the analytic "
            "``emmeans(...)`` path (registered in "
            "``info.multi_col_factors``). Workaround for arbitrary "
            "expressions: pre-transform / pre-convert with "
            "pd.Categorical and refit."
        )


def _resolve_cov_reduce(
    info: ModelInfo,
    cov_reduce: dict[str, Callable | float] | None,
) -> dict[str, float]:
    """Resolve a ``cov_reduce={col: callable_or_scalar}`` mapping into
    a flat ``{col: scalar}`` dictionary suitable for substituting into
    the numeric-means lookup inside :func:`build_grid_spec`.

    Validates:
      - every key references a known numeric covariate;
      - callable values get the training-data Series and produce a
        single float (refuses post-pickle inputs where
        ``info.data`` was dropped);
      - scalar values are coerced to ``float`` (rejects NaN/Inf).

    Returns an empty dict when ``cov_reduce`` is ``None`` so the
    caller's downstream loop sees a uniform no-op shape.
    """
    if cov_reduce is None:
        return {}
    if not isinstance(cov_reduce, dict):
        raise TypeError(
            f"cov_reduce must be a dict[name -> callable | scalar]; "
            f"got {type(cov_reduce).__name__}."
        )

    known = set(info.numeric_means)
    unknown = [k for k in cov_reduce if k not in known]
    if unknown:
        raise ValueError(
            f"cov_reduce references unknown numeric covariates: "
            f"{sorted(unknown)}. Known numerics: {sorted(known)}. "
            "(Factor levels are pinned via ``at=`` instead.)"
        )

    resolved: dict[str, float] = {}
    for col, spec in cov_reduce.items():
        if callable(spec):
            # Callable form: apply to the training-data series.
            # Refuse when the training data was dropped (typical post-
            # pickle ModelInfo). The error names the post-pickle
            # cause and steers to the scalar form.
            if info.data is None or (
                hasattr(info.data, "columns") and len(info.data) == 0
            ):
                raise ValueError(
                    f"cov_reduce[{col!r}] is a callable, but the "
                    "ModelInfo has no training data attached "
                    "(typically because the model_info / EMM was "
                    "pickled — pickling drops ``info.data``). Pass "
                    f"``cov_reduce={{{col!r}: <scalar>}}`` with the "
                    "value already computed, or build a fresh "
                    "ModelInfo from a live fit in the current "
                    "process."
                )
            if col not in info.data.columns:
                raise ValueError(
                    f"cov_reduce[{col!r}] is a callable, but "
                    f"info.data has no column {col!r}. Available "
                    f"columns: {sorted(info.data.columns)}."
                )
            try:
                value = float(spec(info.data[col]))
            except Exception as exc:
                raise ValueError(
                    f"cov_reduce[{col!r}] callable raised when "
                    f"applied to info.data[{col!r}]: "
                    f"{type(exc).__name__}: {exc}. The callable "
                    "must accept a pandas Series and return a "
                    "single finite float."
                ) from exc
        elif isinstance(spec, (int, float, np.integer, np.floating)):
            value = float(spec)
        else:
            raise TypeError(
                f"cov_reduce[{col!r}] must be a callable or numeric "
                f"scalar; got {type(spec).__name__}."
            )
        if not np.isfinite(value):
            raise ValueError(
                f"cov_reduce[{col!r}] resolved to a non-finite value "
                f"({value!r}); pass a finite scalar or a callable "
                "that returns one."
            )
        resolved[col] = value
    return resolved


def _resolve_cov_keep(
    info: ModelInfo,
    cov_keep: list[str] | None,
    cov_reduce: dict[str, Callable | float] | None,
) -> dict[str, list[float]]:
    """Resolve ``cov_keep`` into ``{col: [sorted unique training values]}``.

    Mirrors R ``emmeans``'s ``cov.keep``: the named numeric
    covariates are KEPT at their distinct observed values (treated as
    grid factors) rather than reduced to a single summary value, so
    the EMM table carries one row per (target × kept-covariate-value).

    Validates:
      - every name is a known numeric covariate (factors are gridded
        by default — pass them via ``specs`` / ``at=`` instead);
      - no overlap with ``cov_reduce`` (keeping AND reducing the same
        covariate is contradictory);
      - the training data is live (the unique values come from
        ``info.data``; refuses post-pickle where it was dropped).

    Returns an empty dict when ``cov_keep`` is ``None``.
    """
    if cov_keep is None:
        return {}
    if isinstance(cov_keep, str):
        cov_keep = [cov_keep]
    if not isinstance(cov_keep, (list, tuple)):
        raise TypeError(
            f"cov_keep must be a string or list of covariate names; "
            f"got {type(cov_keep).__name__}."
        )
    names = [str(c) for c in cov_keep]
    known = set(info.numeric_means)
    unknown = [c for c in names if c not in known]
    if unknown:
        raise ValueError(
            f"cov_keep references unknown numeric covariates: "
            f"{sorted(unknown)}. Known numerics: {sorted(known)}. "
            "(Categorical factors are gridded over their levels "
            "automatically; pass them via specs / at=.)"
        )
    if cov_reduce:
        overlap = [c for c in names if c in cov_reduce]
        if overlap:
            raise ValueError(
                f"cov_keep and cov_reduce both name {sorted(overlap)}; "
                "a covariate cannot be simultaneously KEPT at its "
                "unique values and REDUCED to a summary. Drop it from "
                "one of the two."
            )
    if info.data is None or (
        hasattr(info.data, "columns") and len(info.data) == 0
    ):
        raise ValueError(
            "cov_keep requires the training data on ``info.data`` to "
            "discover each covariate's unique values, but it is None "
            "or empty (typically because the ModelInfo / EMM was "
            "pickled — pickling drops the training data). Either call "
            "on a fresh fit, or pin the values explicitly via "
            "``at={col: [v1, v2, ...]}``."
        )
    resolved: dict[str, list[float]] = {}
    for col in names:
        if col not in info.data.columns:
            raise ValueError(
                f"cov_keep[{col!r}]: info.data has no column {col!r}. "
                f"Available columns: {sorted(info.data.columns)}."
            )
        uniq = np.unique(np.asarray(info.data[col], dtype=float))
        resolved[col] = [float(v) for v in uniq]
    return resolved


def build_grid_spec(
    info: ModelInfo,
    at: dict[str, Any] | None = None,
    *,
    require_plain_identifiers: bool = False,
    cov_reduce: dict[str, Callable | float] | None = None,
    cov_keep: list[str] | None = None,
) -> dict[str, list]:
    """Build the per-variable value lists that define the reference grid.

    Used by the analytic path (no patsy round-trip needed) and by the eager
    ``ref_grid`` / streaming marginalization (which DO need plain identifier
    factors to feed back to patsy). Set ``require_plain_identifiers=True``
    from the latter callers.

    Parameters
    ----------
    info
        Source ``ModelInfo``. Provides ``factors`` (categoricals with
        levels), ``numeric_means`` (default value per numeric
        covariate), and the training data referenced by callable
        ``cov_reduce`` entries.
    at
        Per-variable overrides. For factors, restricts to the named
        levels; for numerics, replaces the default with the supplied
        list (and the spec column carries every value). ``at=``
        wins over ``cov_reduce`` (an explicit override of an
        override).
    require_plain_identifiers
        When ``True``, enforce the eager-path restriction that
        every factor name is a bare identifier (rejecting patsy
        expressions like ``np.log(x)`` that can't be re-evaluated
        via ``build_design_matrices``). The analytic path leaves
        this ``False`` because it never round-trips through patsy.
    cov_reduce
        Per-numeric-covariate reduction override. Each value is
        either a callable ``f(series) -> scalar`` (applied to the
        training-data Series for that covariate) or a numeric
        scalar (used directly). Replaces ``info.numeric_means[col]``
        for the named covariates only; unnamed covariates retain
        their training-data mean. Useful for sensitivity analyses
        (e.g. predict at the 75th percentile instead of the mean)
        and for post-pickle workflows that need an explicit value
        because the training data was dropped. See
        :func:`pymmeans.ref_grid` for usage examples.
    """
    if require_plain_identifiers:
        _validate_design(info)

    at = at or {}
    # Reject non-string keys up front; otherwise sorting the unknown set
    # below can crash with a UFuncTypeError when keys mix types
    # .
    non_str = [k for k in at if not isinstance(k, str)]
    if non_str:
        raise TypeError(
            f"'at' keys must be strings; got non-string keys: "
            f"{[type(k).__name__ for k in non_str]}."
        )
    known = set(info.factors) | set(info.numeric_means)
    # Raw source columns are also valid `at=` keys when a formula wraps
    # them (e.g. `at={"x": ...}` is allowed when the formula is
    # `y ~ np.log(x)`); the alias map carries raw -> canonical.
    raw_keys = set(info.aliases.keys())
    unknown = set(at) - known - raw_keys
    if unknown:
        raise ValueError(
            f"'at' references unknown variables: {sorted(unknown)}. "
            f"Known variables: {sorted(known | raw_keys)}."
        )

    # Reject non-finite numeric `at=` values — they propagate to NaN/inf
    # in the design matrix and silently give NaN EMMs .
    # replace the previous string-match catch
    # (``if "non-finite" in str(e): raise``) with a structurally
    # separated coercion / validation split. Numeric-conversion
    # errors (``float("abc")``) are swallowed so downstream code
    # produces its own actionable message; the deliberately-raised
    # non-finite ValueError is raised AFTER the conversion succeeds,
    # so it can never be conflated with a coercion error.
    for k, v in at.items():
        if k in info.numeric_means or k in raw_keys:
            vals = _as_list(v)
            for x in vals:
                if x is None:
                    continue
                try:
                    fx = float(x)
                except (TypeError, ValueError):
                    # Non-numeric in a numeric slot — defer to
                    # downstream patsy / numpy to raise.
                    continue
                if not np.isfinite(fx):
                    raise ValueError(
                        f"'at[{k!r}]' contains a non-finite value "
                        f"({x!r}); pass finite numerics only."
                    )

    # `at=` overrides on a raw source column when a transformed factor
    # depends on it: either (a) raise if only the raw key is given (the
    # transformed term would silently stay at its training-data mean), or
    # (b) validate consistency when BOTH raw and canonical are given.
    import warnings as _w

    from pymmeans.transforms import detect_transform

    multi_col = getattr(info, "multi_col_factors", {}) or {}
    for at_name in at:
        for raw_col, canonical in info.aliases.items():
            if at_name != raw_col or canonical == raw_col:
                continue
            if canonical in multi_col:
                # ``at={"x": ...}`` is unambiguously correct
                # when the canonical name is a multi-column basis
                # expression (``bs(x, df=3)``, ``poly(x, 3)``, etc.)
                # — the user can't supply a single number for the
                # canonical expansion, only for the underlying column.
                # Skip the round-trip-consistency check.
                continue
            if canonical not in at:
                raise NotImplementedError(
                    f"'at' override for raw column '{at_name}' is ambiguous "
                    f"because the formula uses '{canonical}'. Either override "
                    f"the canonical name directly (at={{'{canonical}': ...}}) "
                    "or pass both keys with consistent values."
                )
            # Both raw and canonical given — check round-trip via the
            # detected transform after broadcasting both to lists so that
            # mismatched-length or list-vs-scalar inputs are caught.
            raw_list = _as_list(at[raw_col])
            canon_list = _as_list(at[canonical])
            transform = detect_transform(canonical)
            if transform is None:
                continue
            # Broadcast scalar to match the other side's length
            if len(raw_list) == 1 and len(canon_list) > 1:
                raw_list = raw_list * len(canon_list)
            elif len(canon_list) == 1 and len(raw_list) > 1:
                canon_list = canon_list * len(raw_list)
            if len(raw_list) != len(canon_list):
                _w.warn(
                    f"at={{'{raw_col}': {at[raw_col]!r}, '{canonical}': "
                    f"{at[canonical]!r}}} has mismatched lengths "
                    f"({len(raw_list)} vs {len(canon_list)}); ignoring "
                    "the raw key and using the canonical values.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            for rv, cv in zip(raw_list, canon_list, strict=True):
                try:
                    expected_raw = float(transform.inverse(float(cv)))
                except (TypeError, ValueError):
                    continue
                if not np.isclose(float(rv), expected_raw, rtol=1e-3):
                    _w.warn(
                        f"at={{'{raw_col}': {rv}, '{canonical}': {cv}}} "
                        f"is inconsistent: transform.inverse({cv}) = "
                        f"{expected_raw:.4g}. Using the canonical value "
                        "for the design column.",
                        UserWarning,
                        stacklevel=2,
                    )
                    break

    spec: dict[str, list] = {}
    for name, levels in info.factors.items():
        if name in at:
            override = _as_list(at[name])
            # empty `at={'g': []}` used
            # to silently propagate an empty grid through every path,
            # producing an empty-frame EMM that downstream tooling
            # then exploded on (empty linfct, NaN arithmetic, etc.).
            # Refuse up-front with a clear pointer.
            if not override:
                raise ValueError(
                    f"'at' for factor '{name}' must contain at least "
                    "one level; got an empty list. Drop the key from "
                    "`at` to use all levels."
                )
            unknown_levels = [v for v in override if v not in levels]
            if unknown_levels:
                raise ValueError(
                    f"'at' for factor '{name}' references unknown levels: "
                    f"{unknown_levels}. Known levels: {levels}."
                )
            spec[name] = override
        else:
            spec[name] = list(levels)
    # Resolve ``cov_reduce`` once up front so the per-name loop below
    # can do a clean dict lookup; precedence is ``at`` > ``cov_keep``
    # > ``cov_reduce`` > ``info.numeric_means`` (default mean).
    cov_overrides = _resolve_cov_reduce(info, cov_reduce)
    cov_keep_vals = _resolve_cov_keep(info, cov_keep, cov_reduce)
    for name, mean in info.numeric_means.items():
        if name in at:
            override = _as_list(at[name])
            if not override:
                raise ValueError(
                    f"'at' for numeric covariate '{name}' must contain "
                    "at least one value; got an empty list."
                )
            spec[name] = [float(v) for v in override]
        elif name in cov_keep_vals:
            # Keep the covariate at its unique training-data values
            # (treated as a grid factor) instead of reducing to a
            # single summary value.
            spec[name] = cov_keep_vals[name]
        elif name in cov_overrides:
            spec[name] = [cov_overrides[name]]
        else:
            spec[name] = [mean]

    if not spec:
        raise ValueError("Model has no factors or numeric covariates to grid.")
    return spec


def grid_size(spec: dict[str, list]) -> int:
    """Number of rows in the full cartesian product of a grid spec."""
    n = 1
    for vals in spec.values():
        n *= len(vals)
    return n


def _detect_nesting(info: ModelInfo) -> dict[str, list[str]]:
    """Empirically detect nesting structure from the training data.

    For each ordered pair of categorical factors ``(inner, outer)``,
    ``inner`` is nested in ``outer`` when knowing the ``inner`` level
    determines the ``outer`` level — i.e. every level of ``inner``
    co-occurs with exactly one level of ``outer`` in
    ``info.data`` (the many-to-one school→district pattern). The
    enclosing factor must be strictly coarser (fewer levels) than the
    nested one, which breaks the degenerate 1:1-bijection tie (two
    factors that perfectly co-vary aren't a nesting — they're
    redundant codings).

    Mirrors the empirical half of R ``emmeans``'s automatic nesting
    detection. Returns ``{}`` when no training data is available
    (e.g. post-pickle) or no nesting is found.
    """
    data = info.data
    if data is None or not hasattr(data, "columns") or len(data) == 0:
        return {}
    factors = [f for f in info.factors if f in data.columns]
    if len(factors) < 2:
        return {}
    nlev = {f: int(data[f].nunique()) for f in factors}
    nesting: dict[str, list[str]] = {}
    for inner in factors:
        enclosing: list[str] = []
        for outer in factors:
            if inner == outer:
                continue
            # Enclosing must be strictly coarser (fewer levels) to be
            # the "containing" factor; this also rules out the 1:1
            # bijection case where each would nest in the other.
            if nlev[outer] >= nlev[inner]:
                continue
            determined = data.groupby(inner, observed=True)[outer].nunique()
            if len(determined) and int(determined.max()) <= 1:
                enclosing.append(outer)
        if enclosing:
            nesting[inner] = enclosing
    return nesting


def _resolve_auto_nesting(
    info: ModelInfo,
    nesting: Any,
) -> dict[str, str | list[str]] | None:
    """Resolve ``nesting="auto"`` to a detected nesting dict.

    Pass-through for any other value (``None``, an explicit dict).
    When ``"auto"`` is requested, run :func:`_detect_nesting`; if a
    structure is found, emit an R-style informational NOTE naming it
    and return the dict, otherwise return ``None`` (no nesting).

    Opt-in by design: unlike R, pymmeans does NOT auto-detect nesting
    on the default ``nesting=None`` path, because silently changing
    EMM values based on an empirical heuristic (which can false-
    positive on small/collinear samples) is a footgun in a library.
    Users request it explicitly with ``nesting="auto"``.
    """
    if not (isinstance(nesting, str) and nesting.lower() == "auto"):
        return nesting
    detected = _detect_nesting(info)
    if not detected:
        return None
    import warnings as _w
    desc = "; ".join(
        f"{nested} in {enc if len(enc) > 1 else enc[0]}"
        for nested, enc in detected.items()
    )
    _w.warn(
        f"nesting='auto' detected nesting structure from the data: "
        f"{desc}. The reference grid is filtered to the observed "
        "(nested, enclosing) combinations. Pass an explicit "
        "nesting={...} to override, or nesting=None to disable.",
        UserWarning,
        stacklevel=3,
    )
    return detected


def _validate_nesting(
    info: ModelInfo,
    nesting: dict[str, str | list[str]] | None,
) -> dict[str, list[str]]:
    """Validate and normalize a ``nesting`` declaration.

    Mirrors R ``ref_grid(..., nesting = list(b = "a"))`` for the
    common case where one categorical factor's levels are only
    meaningful within another's (e.g. school within district).

    Accepts:
      - ``None`` (no nesting declared; returns empty dict).
      - ``dict[nested -> enclosing]`` where each value is a single
        factor name OR a list of factor names. Multiple separate
        nestings sit in one dict (different keys).

    Validates:
      - Each key and value names a known categorical factor in
        ``info.factors`` (or its alias). Numerics rejected.
      - No key appears in its own list of enclosing factors
        (no self-loops). Multi-key cycles aren't statically
        detectable in the dict form, but the per-key filter
        below tolerates them — it just produces an empty grid.
      - Values are non-empty after normalization.

    Returns
    -------
    dict[str, list[str]]
        Normalized {canonical_nested_name: [canonical_enclosing_names]}.
        Empty dict when ``nesting`` is ``None``.
    """
    if nesting is None:
        return {}
    if not isinstance(nesting, dict):
        raise TypeError(
            f"nesting= must be a dict[nested -> enclosing | list of "
            f"enclosing]; got {nesting.__class__.__name__}."
        )
    if not nesting:
        raise ValueError(
            "nesting= must contain at least one entry "
            "(or be None to opt out); got an empty dict."
        )

    normalized: dict[str, list[str]] = {}
    known_factors = set(info.factors)
    for raw_nested, raw_enclosing in nesting.items():
        if not isinstance(raw_nested, str):
            raise TypeError(
                "nesting= keys must be strings (factor names); got "
                f"non-string key of type {raw_nested.__class__.__name__}."
            )
        nested = info.aliases.get(raw_nested, raw_nested)
        if nested not in known_factors:
            raise ValueError(
                f"nesting= key {raw_nested!r} is not a known "
                f"categorical factor. Known factors: "
                f"{sorted(known_factors)}. Numeric covariates "
                "cannot be nested — pin them via ``at=`` instead."
            )

        # Normalize the enclosing side to a list.
        if isinstance(raw_enclosing, str):
            raw_enclosing_list: list[str] = [raw_enclosing]
        elif isinstance(raw_enclosing, (list, tuple)):
            raw_enclosing_list = list(raw_enclosing)
        else:
            raise TypeError(
                f"nesting=[{raw_nested!r}] value must be a string or "
                "list of strings (enclosing factor names); got "
                f"{raw_enclosing.__class__.__name__}."
            )
        if not raw_enclosing_list:
            raise ValueError(
                f"nesting=[{raw_nested!r}] value must name at least "
                "one enclosing factor; got an empty list."
            )

        enclosing_canon: list[str] = []
        for raw_e in raw_enclosing_list:
            if not isinstance(raw_e, str):
                raise TypeError(
                    f"nesting=[{raw_nested!r}] enclosing entry must "
                    "be a string (factor name); got non-string of "
                    f"type {raw_e.__class__.__name__}."
                )
            canon_e = info.aliases.get(raw_e, raw_e)
            if canon_e not in known_factors:
                raise ValueError(
                    f"nesting=[{raw_nested!r}] enclosing factor "
                    f"{raw_e!r} is not a known categorical factor. "
                    f"Known factors: {sorted(known_factors)}."
                )
            if canon_e == nested:
                raise ValueError(
                    f"nesting=[{raw_nested!r}] lists itself as an "
                    "enclosing factor (self-loop). A factor cannot be "
                    "nested inside itself; drop the self-reference."
                )
            if canon_e in enclosing_canon:
                # Silently dedup — repeated names don't change the
                # filter semantics.
                continue
            enclosing_canon.append(canon_e)
        normalized[nested] = enclosing_canon
    return normalized


def _apply_nesting_filter(
    grid: pd.DataFrame,
    nesting: dict[str, list[str]],
    info: ModelInfo,
) -> pd.DataFrame:
    """Drop reference-grid rows whose ``(nested, *enclosing)`` tuples
    do not appear in the training data.

    Returns a fresh DataFrame with the rows that survive ALL the
    declared nestings; index is reset.

    Caller is responsible for ensuring ``info.data`` is non-empty
    (callers use :func:`_validate_nesting` first and then check
    ``info.data`` to raise a clear post-pickle error before reaching
    here).
    """
    if not nesting:
        return grid
    # Canonical -> raw column-name map (info.factors are canonical
    # patsy expressions; info.data columns are raw user-facing names).
    canon_to_raw: dict[str, str] = {v: k for k, v in info.aliases.items()}

    mask = pd.Series(True, index=grid.index)
    for nested, enclosings in nesting.items():
        nested_raw = canon_to_raw.get(nested, nested)
        enclosings_raw = [canon_to_raw.get(e, e) for e in enclosings]
        cols = [nested_raw, *enclosings_raw]
        # Build the set of valid tuples from the training data. Use
        # ``drop_duplicates`` so the membership check is O(n_unique)
        # per row rather than O(n_data).
        # ``itertuples(index=False, name=None)`` yields raw tuples for
        # hashable membership testing.
        valid_tuples = set(
            info.data[cols]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
        # The reference grid uses CANONICAL column names; lookup the
        # canonical-column tuples on the grid.
        canon_cols = [nested, *enclosings]
        # Coerce values to match the data side's typing — pd.Categorical
        # equality goes through the underlying dtype, so tuples
        # built from grid Categoricals compare correctly against
        # tuples built from data columns even when one side is
        # str-backed and the other category-backed.
        grid_tuples = list(
            grid[canon_cols].itertuples(index=False, name=None)
        )
        row_mask = pd.Series(
            [t in valid_tuples for t in grid_tuples],
            index=grid.index,
        )
        mask &= row_mask
    return grid[mask].reset_index(drop=True)


def ref_grid(
    model: Any,
    at: dict[str, Any] | None = None,
    *,
    cov_reduce: dict[str, Callable | float] | None = None,
    cov_keep: list[str] | None = None,
    nesting: dict[str, str | list[str]] | None = None,
) -> RefGrid:
    """Build a reference grid from a fitted statsmodels model.

    Parameters
    ----------
    model
        Fitted statsmodels OLS or GLM result, or a ``ModelInfo``.
    at
        Optional dict mapping variable name to a scalar or list of values
        that override the default. For factors the default is all levels;
        for numerics it is the training-data mean.
    cov_reduce
        Optional dict ``{numeric_col: callable | scalar}`` overriding
        the default training-data-mean reduction per numeric
        covariate. The callable receives the training-data Series
        for that covariate and must return a single finite float;
        the scalar form is used directly. Covariates not named in
        ``cov_reduce`` keep their training-data mean. ``at=`` wins
        over ``cov_reduce`` when both name the same column.

        Common patterns::

            ref_grid(fit, cov_reduce={"x": np.median})
            ref_grid(fit, cov_reduce={"x": lambda s: s.quantile(0.75)})
            ref_grid(fit, cov_reduce={"x": 0.0})  # constant override

        Callable form requires ``info.data`` to be live (training
        data was dropped by a pickle round-trip — use the scalar
        form there).
    nesting
        Optional ``dict[nested -> enclosing | list-of-enclosing]``
        declaring that a factor's levels are only meaningful within
        another's. Mirrors R
        ``ref_grid(..., nesting = list(school = "district"))``: the
        cartesian-product grid is filtered to only the
        ``(nested, *enclosing)`` tuples that actually appear in
        ``info.data``, so phantom (district, school-not-in-that-
        district) cells are dropped from the EMM result.

        Pass the string ``"auto"`` to empirically detect nesting from
        the training data: any factor whose level determines a
        strictly-coarser factor's level (the many-to-one school→
        district pattern) is treated as nested in it, with an
        informational NOTE naming the detected structure. Unlike R,
        auto-detection is OPT-IN (the default ``nesting=None`` never
        auto-detects) — silently reshaping EMMs from an empirical
        heuristic that can false-positive on small / collinear
        samples is avoided.

        Multiple separate nestings sit in one dict::

            nesting = {"school": "district"}
            nesting = {"school": ["district", "region"]}
            nesting = {"school": "district", "classroom": "school"}

        Refuses post-pickle (the filter needs ``info.data`` to
        discover valid combinations), and refuses self-loops
        (a factor cannot be nested inside itself).

    Returns
    -------
    RefGrid
        Container with the grid DataFrame, the L matrix, and ``ModelInfo``.

    Raises
    ------
    NotImplementedError
        If the formula uses non-identifier factor expressions (e.g.
        ``factor(percent)``, ``np.log(x)``). v0.1 supports plain column
        references only; pre-convert with ``pd.Categorical`` and refit.
    ValueError
        If ``at`` references unknown variables or unknown factor levels,
        if ``cov_reduce`` references an unknown / non-numeric column,
        or if a callable in ``cov_reduce`` returns a non-finite value.

    Examples
    --------
    >>> import pandas as pd
    >>> import statsmodels.formula.api as smf
    >>> df = pd.DataFrame({
    ... "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ... "g": pd.Categorical(["a", "b", "a", "b", "a", "b"]),
    ... "x": [0.0, 1.0, 2.0, 0.5, 1.5, 2.5],
    ... })
    >>> rg = ref_grid(smf.ols("y ~ g + x", data=df).fit())
    >>> rg.grid # doctest: +NORMALIZE_WHITESPACE
       g x
    0 a 1.25
    1 b 1.25
    """
    info = model if isinstance(model, ModelInfo) else from_fitted(model)
    nesting = _resolve_auto_nesting(info, nesting)
    nesting_norm = _validate_nesting(info, nesting)
    if nesting_norm and (info.data is None or len(info.data) == 0):
        raise ValueError(
            "nesting= requires the training data on ``info.data`` "
            "to discover the valid (nested, enclosing) tuples, but "
            "``info.data`` is None or empty (typically because the "
            "ModelInfo / EMM was pickled — pickling drops the "
            "training data). Fixes:\n"
            "  - Call ref_grid(fit, nesting=...) on a fresh fit in "
            "the current process.\n"
            "  - Drop nesting= and use ``at={...}`` to manually pin "
            "the valid (nested, enclosing) combinations."
        )
    spec = build_grid_spec(
        info, at,
        require_plain_identifiers=True,
        cov_reduce=cov_reduce,
        cov_keep=cov_keep,
    )

    names = list(spec)
    combos = list(itertools.product(*[spec[n] for n in names]))
    grid = pd.DataFrame(combos, columns=names)

    for name, levels in info.factors.items():
        grid[name] = pd.Categorical(grid[name], categories=levels)

    # Apply the nesting filter (if any) BEFORE building the design
    # matrices, so ``linfct`` lives on the same row index as the
    # filtered grid and downstream consumers (the eager-path
    # averaging in :func:`emmeans`) average over only valid
    # (nested, enclosing) tuples.
    if nesting_norm:
        grid = _apply_nesting_filter(grid, nesting_norm, info)
        if len(grid) == 0:
            raise ValueError(
                "nesting= produced an empty reference grid — no "
                "(nested, enclosing) tuple from the declared nesting "
                "appears in ``info.data``. Double-check the nesting "
                "spec against the actual factor-level structure in "
                "the training data."
            )

    [linfct] = build_design_matrices(
        [info.design_info], grid, return_type="matrix"
    )
    linfct = np.asarray(linfct, dtype=float)

    # GLMGam smoother append. For a
    # statsmodels GLMGam fit the design is ``[linear | smoother_basis]``
    # and ``design_info`` covers only the linear part; re-evaluate the
    # penalised spline basis at the grid's covariate values and
    # concatenate so the linfct matches the full ``beta`` (linear +
    # smoother coefficients) and EMMs are smoother-correct at every
    # grid point. See :func:`pymmeans.utils.from_glmgam`.
    if getattr(info, "smoother_info", None) is not None:
        smoother, svars = info.smoother_info
        missing = [v for v in svars if v not in grid.columns]
        if missing:
            raise ValueError(
                f"GLMGam smoother needs columns {missing} in the "
                "reference grid, but they're absent. The smoother "
                "variables must be present as covariates — pass them "
                "via at= / cov_keep= or leave them to reduce to their "
                "training mean."
            )
        basis = np.asarray(
            smoother.transform(grid[svars].to_numpy()), dtype=float
        )
        linfct = np.hstack([linfct, basis])

    return RefGrid(
        grid=grid,
        linfct=linfct,
        model_info=info,
    )
