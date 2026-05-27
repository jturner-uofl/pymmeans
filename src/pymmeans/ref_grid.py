"""Reference grid construction.

The reference grid is the cross of all factor levels with numeric covariates
held at their training-data means. ``mu = L @ beta`` evaluated on this grid
gives predictions at each combination; marginal means come from averaging
rows of ``L`` that share a target factor level.
"""

from __future__ import annotations

import itertools
import keyword
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


def build_grid_spec(
    info: ModelInfo,
    at: dict[str, Any] | None = None,
    *,
    require_plain_identifiers: bool = False,
) -> dict[str, list]:
    """Build the per-variable value lists that define the reference grid.

    Used by the analytic path (no patsy round-trip needed) and by the eager
    ``ref_grid`` / streaming marginalization (which DO need plain identifier
    factors to feed back to patsy). Set ``require_plain_identifiers=True``
    from the latter callers.
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
    for name, mean in info.numeric_means.items():
        if name in at:
            override = _as_list(at[name])
            if not override:
                raise ValueError(
                    f"'at' for numeric covariate '{name}' must contain "
                    "at least one value; got an empty list."
                )
            spec[name] = [float(v) for v in override]
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


def ref_grid(
    model: Any,
    at: dict[str, Any] | None = None,
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
        If ``at`` references unknown variables or unknown factor levels.

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
    spec = build_grid_spec(info, at, require_plain_identifiers=True)

    names = list(spec)
    combos = list(itertools.product(*[spec[n] for n in names]))
    grid = pd.DataFrame(combos, columns=names)

    for name, levels in info.factors.items():
        grid[name] = pd.Categorical(grid[name], categories=levels)

    [linfct] = build_design_matrices(
        [info.design_info], grid, return_type="matrix"
    )

    return RefGrid(
        grid=grid,
        linfct=np.asarray(linfct, dtype=float),
        model_info=info,
    )
