"""Counterfactual reference grids (``datagrid``).

``datagrid`` builds a small DataFrame of covariate combinations to feed to
:func:`pymmeans.avg_predictions`, :func:`pymmeans.avg_slopes`, or
:func:`pymmeans.avg_comparisons` via their ``newdata=`` argument, so an
analyst can evaluate predictions / marginal effects at *specific* covariate
values instead of averaging over the observed sample (g-computation).

Matching ``marginaleffects.datagrid``: variables named in ``**kwargs`` are
set to the supplied value(s) and crossed (Cartesian product); every other
predictor is held at a typical value -- the **mean** for a numeric column
and the **mode** (most frequent level) for a categorical column.

>>> from pymmeans import datagrid, avg_predictions       # doctest: +SKIP
>>> grid = datagrid(fit, x=[0, 1, 2])                     # doctest: +SKIP
>>> avg_predictions(fit, newdata=grid)                    # doctest: +SKIP
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd

from pymmeans.slopes import _get_info, _require_reference_data

__all__ = ["datagrid"]


def _typical(col: pd.Series) -> Any:
    """A representative value for a column: mean (numeric) or mode (factor)."""
    if pd.api.types.is_numeric_dtype(col):
        return float(col.mean())
    return col.mode(dropna=True).iloc[0]


def datagrid(obj: Any, **kwargs: Any) -> pd.DataFrame:
    """Build a counterfactual reference grid for ``newdata=``.

    Parameters
    ----------
    obj
        A fitted model or a pymmeans result carrying ``model_info`` -- used
        only to recover the predictor columns, their dtypes, and the
        typical values for unspecified variables.
    **kwargs
        ``variable=value`` or ``variable=[values...]``. Named variables are
        crossed; all others are held at their typical value (mean for
        numeric, mode for categorical).

    Returns
    -------
    pandas.DataFrame
        One row per combination of the supplied values, with the model's
        predictor columns and dtypes preserved. Pass it as ``newdata=`` to
        :func:`pymmeans.avg_predictions` / :func:`pymmeans.avg_slopes` /
        :func:`pymmeans.avg_comparisons`.
    """
    info = _get_info(obj)
    data = _require_reference_data(info, "datagrid")
    resp = getattr(info, "response_name", None)
    predictors = [c for c in data.columns if c != resp]

    unknown = set(kwargs) - set(predictors)
    if unknown:
        raise ValueError(
            f"datagrid got value(s) for non-predictor column(s) {sorted(unknown)}; "
            f"known predictors are {predictors}."
        )

    # Normalise each specified variable to a list of values.
    specified = {
        k: (list(v) if isinstance(v, (list, tuple, np.ndarray, pd.Series)) else [v])
        for k, v in kwargs.items()
    }
    names = list(specified)
    combos = (
        list(itertools.product(*(specified[n] for n in names))) if names else [()]
    )

    rows = []
    for combo in combos:
        row = {c: _typical(data[c]) for c in predictors}
        row.update(dict(zip(names, combo, strict=True)))
        rows.append(row)
    grid = pd.DataFrame(rows, columns=predictors)

    # Preserve categorical dtype / categories so patsy encodes the grid
    # exactly as the original fit.
    for c in predictors:
        if hasattr(data[c], "cat"):
            grid[c] = pd.Categorical(grid[c], categories=data[c].cat.categories)
        else:
            grid[c] = grid[c].astype(data[c].dtype)
    return grid
