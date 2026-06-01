"""Multiple-imputation pooling for pymmeans output (Rubin's rules).

Given a list of EMM / contrast results — one per multiply-imputed
dataset — this module pools them into a single combined-inference
result using **Rubin's (1987) rules** for the point estimate,
within- and between-imputation variance, total variance, and the
**Barnard-Rubin (1999)** degrees-of-freedom correction.

References
----------
- Rubin, D. B. (1987). *Multiple Imputation for Nonresponse in
  Surveys*. New York: Wiley.
- Barnard, J., & Rubin, D. B. (1999). Small-sample degrees of
  freedom with multiple imputation. *Biometrika*, 86(4), 948-955.
- van Buuren, S. (2018). *Flexible Imputation of Missing Data*
  (2nd ed.), Section 2.3 ("Rubin's rules").

Algorithm
---------
Given ``M`` imputations producing per-imputation point estimates
``θ_1, …, θ_M`` and SEs ``SE_1, …, SE_M``:

* Pooled point estimate:  ``θ̄ = (1/M) Σ θ_m``
* Within-imputation variance:  ``Ū = (1/M) Σ SE_m²``
* Between-imputation variance:  ``B = (1/(M-1)) Σ (θ_m - θ̄)²``
* Total variance:  ``T = Ū + (1 + 1/M) · B``
* Pooled SE:  ``sqrt(T)``
* Fraction of missing information:  ``γ = (1 + 1/M) · B / T``
* Relative increase in variance due to missingness:  ``r = (1 + 1/M) · B / Ū``
* Barnard-Rubin (1999) df, given the complete-data df ``ν``::

      ν_old = (M - 1) (1 + 1/r)²
      ν_obs = ((ν + 1) / (ν + 3)) · ν · (1 - γ)
      ν_BR  = (1 / ν_old + 1 / ν_obs)⁻¹

  If complete-data df is infinite (e.g. z-test), ``ν_BR = ν_old``.

This module exposes a single public function, :func:`pool_imputed`,
which accepts a list of pymmeans ``EMMResult`` or ``ContrastResult``
objects (all from the same model specification on M imputed
datasets) and returns a pooled object with updated point estimates,
SEs, dfs, and p-values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats

__all__ = ["PooledImputationResult", "pool_imputed"]


@dataclass(frozen=True)
class PooledImputationResult:
    """Result of Rubin's-rules pooling across M imputations.

    Attributes
    ----------
    frame
        Pooled DataFrame with columns for the pooled point estimate
        (``emmean`` or ``estimate``), pooled SE, Barnard-Rubin df,
        and (for contrast inputs) t-ratio / p-value / CI bounds.
    M
        Number of imputations pooled.
    fmi
        Per-row fraction of missing information (FMI) — the fraction
        of the pooled variance attributable to between-imputation
        variability.
    relative_increase
        Per-row relative increase in variance due to missingness
        (``r = (1 + 1/M) · B / Ū``).
    """

    frame: pd.DataFrame
    M: int
    fmi: np.ndarray
    relative_increase: np.ndarray


def _validate_results(results: list[Any]) -> tuple[str, str]:
    """Confirm the input list is non-empty, type-uniform, and shape-compatible.

    Returns ``(point_col, type_name)`` where ``point_col`` is the
    column in ``.frame`` carrying the point estimate (``"emmean"``
    for EMM, ``"estimate"`` for contrasts).
    """
    if not results:
        raise ValueError("pool_imputed requires a non-empty list of results.")
    if len(results) < 2:
        raise ValueError(
            f"pool_imputed requires M >= 2 imputations; got {len(results)}."
        )
    type_name = type(results[0]).__name__
    for r in results[1:]:
        if type(r).__name__ != type_name:
            raise TypeError(
                f"pool_imputed: all results must be the same type; "
                f"got {type_name} and {type(r).__name__}."
            )
    first = results[0].frame
    for i, r in enumerate(results[1:], start=1):
        if len(r.frame) != len(first):
            raise ValueError(
                f"pool_imputed: results must have identical row counts "
                f"(imputation 0 has {len(first)}, imputation {i} has "
                f"{len(r.frame)})."
            )
    # Detect point-estimate column.
    cols = set(first.columns)
    if "emmean" in cols:
        point_col = "emmean"
    elif "estimate" in cols:
        point_col = "estimate"
    elif "ratio" in cols:
        point_col = "ratio"
    else:
        raise ValueError(
            "pool_imputed: cannot find a point-estimate column "
            "('emmean' / 'estimate' / 'ratio') in the first result's frame."
        )
    if "SE" not in cols:
        raise ValueError(
            "pool_imputed: each result's frame must contain an 'SE' column."
        )
    return point_col, type_name


def _barnard_rubin_df(
    M: int, complete_df: np.ndarray, r: np.ndarray, gamma: np.ndarray
) -> np.ndarray:
    """Barnard-Rubin (1999) small-sample df adjustment.

    Per Barnard-Rubin 1999 Equation 3::

        ν_old = (M - 1) (1 + 1/r)²
        ν_obs = ((ν + 1)/(ν + 3)) · ν · (1 - γ)
        ν_BR  = (1/ν_old + 1/ν_obs)⁻¹
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        nu_old = (M - 1) * (1.0 + 1.0 / r) ** 2
        # If complete_df is infinite (z-test), the observed-df
        # contribution is also infinite and ν_BR collapses to ν_old.
        finite = np.isfinite(complete_df)
        nu_obs = np.where(
            finite,
            ((complete_df + 1.0) / (complete_df + 3.0))
            * complete_df * (1.0 - gamma),
            np.inf,
        )
        nu_br = 1.0 / (1.0 / nu_old + 1.0 / nu_obs)
    return nu_br


def pool_imputed(results: list[Any]) -> PooledImputationResult:
    """Pool ``M`` pymmeans results via Rubin's rules + Barnard-Rubin df.

    Pass a list of :class:`pymmeans.EMMResult` or
    :class:`pymmeans.ContrastResult` objects — one per multiply-
    imputed dataset, all from the same model specification — and
    receive a single pooled result.

    Parameters
    ----------
    results
        List of length ``M >= 2`` of pymmeans results. All elements
        must be the same type and have identical-length frames with
        the same row ordering.

    Returns
    -------
    PooledImputationResult
        With ``.frame`` carrying the pooled point estimate, SE,
        Barnard-Rubin df, t-ratio (for contrast inputs), p-value,
        and CI bounds; plus ``.fmi`` and ``.relative_increase``
        diagnostics per row.

    Notes
    -----
    * The pooled SE uses Rubin's total-variance formula
      ``T = Ū + (1 + 1/M) · B``.
    * The Barnard-Rubin df is the small-sample-aware df that should
      be used for t-statistics and CIs after pooling.
    * Confidence intervals at level ``level`` (default 0.95, taken
      from the first input's ``.level`` attribute when present)
      use the Barnard-Rubin df.

    Examples
    --------
    Given a list ``imputed_dfs`` of ``M`` imputed DataFrames sharing
    the same model specification, the standard usage is::

        fits   = [smf.ols("y ~ treat + x", d).fit() for d in imputed_dfs]
        emms   = [emmeans(f, "treat") for f in fits]
        cts    = [contrast(em, method="trt.vs.ctrl", ref=0) for em in emms]
        pooled = pool_imputed(cts)
        print(pooled.frame)

    The ``pooled.frame`` columns are the same as for a non-pooled
    ``ContrastResult`` (``estimate`` / ``SE`` / ``df`` / ``t_ratio``
    / ``p_value`` / ``lower_cl`` / ``upper_cl``); the SE and df now
    incorporate the between-imputation variance via Rubin's rules
    and the Barnard-Rubin small-sample df correction.
    """
    point_col, type_name = _validate_results(results)
    M = len(results)

    # Stack per-imputation point estimates + SEs into (M, n_rows) arrays.
    pts = np.stack(
        [r.frame[point_col].to_numpy(dtype=float) for r in results]
    )
    ses = np.stack(
        [r.frame["SE"].to_numpy(dtype=float) for r in results]
    )

    # Per-imputation complete-data df (use the first imputation as
    # the reference, since they should all match by construction).
    if "df" in results[0].frame.columns:
        complete_df = results[0].frame["df"].to_numpy(dtype=float)
    else:
        complete_df = np.full(pts.shape[1], np.inf)

    # Rubin's rules.
    theta_bar = pts.mean(axis=0)
    U_bar = (ses ** 2).mean(axis=0)
    if M > 1:
        B = ((pts - theta_bar) ** 2).sum(axis=0) / (M - 1)
    else:
        B = np.zeros_like(theta_bar)
    T = U_bar + (1.0 + 1.0 / M) * B
    SE_pool = np.sqrt(T)

    # Diagnostics: r and γ per row.
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (1.0 + 1.0 / M) * B / np.where(U_bar > 0, U_bar, np.nan)
        # Where there is no between-variance, r → 0 (and γ → 0).
        r = np.nan_to_num(r, nan=0.0, posinf=np.inf, neginf=0.0)
        gamma = (1.0 + 1.0 / M) * B / np.where(T > 0, T, np.nan)
        gamma = np.nan_to_num(gamma, nan=0.0, posinf=1.0, neginf=0.0)

    df_br = _barnard_rubin_df(M, complete_df, r, gamma)

    # Build the pooled DataFrame from the first input's row metadata
    # (factor levels, contrast labels, etc.).
    pooled = results[0].frame.copy()
    pooled[point_col] = theta_bar
    pooled["SE"] = SE_pool
    pooled["df"] = df_br

    # If the input is a ContrastResult, recompute t / p / CI under
    # the pooled SE and Barnard-Rubin df.
    if type_name == "ContrastResult" or "t_ratio" in pooled.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            tstat = np.where(SE_pool > 0, theta_bar / SE_pool, np.nan)
        pooled["t_ratio"] = tstat
        pooled["p_value"] = 2.0 * stats.t.sf(np.abs(tstat), df_br)

    # CI bounds at the input level (or 0.95 if not present).
    level = float(getattr(results[0], "level", 0.95))
    alpha = 1.0 - level
    tcrit = stats.t.isf(alpha / 2.0, df_br)
    pooled["lower_cl"] = theta_bar - tcrit * SE_pool
    pooled["upper_cl"] = theta_bar + tcrit * SE_pool

    return PooledImputationResult(
        frame=pooled,
        M=M,
        fmi=gamma,
        relative_increase=r,
    )
