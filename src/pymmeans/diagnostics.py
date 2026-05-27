"""Health checks for EMM and contrast results.

R's ``emmeans`` leaves the user to assemble diagnostics by hand —
condition numbers from ``solve``, leverage from ``hatvalues``,
boundary checks from ``isSingular``, etc. ``health_check()`` bundles
these into one call so the user gets *one* report:

- **estimability**: any non-estimable rows?
- **conditioning**: vcov condition number; flag at >1e10.
- **rank**: design matrix rank vs n_params; flag rank deficiency.
- **df sanity**: any Satterthwaite df < 3, or df = inf when residual
  df was expected?
- **influence** (OLS only): max leverage; flag points with
  ``h_ii > 3p/n`` (Belsley-Kuh-Welsch threshold).
- **cell counts** (categorical EMMs): minimum cell size per
  target/by combination in the training data; flag cells with n=0
  or n<5.
- **boundary fits** (MixedLM): random-effects variance ~ 0.

The output is a structured dataclass with severity tiers (``ok``,
``warning``, ``critical``) so callers can decide what to surface.
``__repr__`` produces a one-screen summary; the underlying fields
are pandas-friendly for downstream tooling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class Check:
    """A single diagnostic finding.

    ``severity`` is one of ``"ok"``, ``"warning"``, ``"critical"``. The
    repr renders critical ones in red-equivalent prefix so they are
    impossible to miss in a terminal log.
    """

    name: str
    severity: str
    message: str
    value: Any = None

    def __repr__(self) -> str:
        prefix = {"ok": "[ok]", "warning": "[warn]", "critical": "[CRIT]"}.get(
            self.severity, "[?]"
        )
        return f"{prefix} {self.name}: {self.message}"


@dataclass
class HealthReport:
    """Bundle of checks from :func:`health_check`.

    Iterating the report yields :class:`Check` instances in the order
    they were run; filtering by ``severity`` lets the caller surface
    only the actionable findings.
    """

    checks: list[Check] = field(default_factory=list)

    def add(
        self,
        name: str,
        severity: str,
        message: str,
        value: Any = None,
    ) -> None:
        """Record a finding."""
        self.checks.append(Check(name, severity, message, value))

    @property
    def critical(self) -> list[Check]:
        """All critical findings (numerical or interpretive correctness)."""
        return [c for c in self.checks if c.severity == "critical"]

    @property
    def warnings(self) -> list[Check]:
        """All warning-tier findings (worth inspecting, not necessarily wrong)."""
        return [c for c in self.checks if c.severity == "warning"]

    @property
    def ok(self) -> bool:
        """True iff there are no critical findings."""
        return not self.critical

    def __iter__(self):
        return iter(self.checks)

    def __len__(self) -> int:
        return len(self.checks)

    def __repr__(self) -> str:
        if not self.checks:
            return "HealthReport: no checks run"
        n_crit = len(self.critical)
        n_warn = len(self.warnings)
        n_ok = len(self.checks) - n_crit - n_warn
        header = (
            f"HealthReport: {n_ok} ok, {n_warn} warning, "
            f"{n_crit} critical"
        )
        body = "\n".join(repr(c) for c in self.checks)
        return f"{header}\n{body}"

    def to_frame(self) -> pd.DataFrame:
        """Return the checks as a tidy DataFrame for downstream tooling."""
        return pd.DataFrame(
            [
                {
                    "name": c.name,
                    "severity": c.severity,
                    "message": c.message,
                    "value": c.value,
                }
                for c in self.checks
            ]
        )


def _check_estimability(result: Any, report: HealthReport) -> None:
    from pymmeans.utils import detect_value_column

    frame = result.frame
    kind_info = detect_value_column(frame)
    if kind_info is None:
        return
    _kind, value_col = kind_info
    n_nan = int(frame[value_col].isna().sum())
    if n_nan == 0:
        report.add(
            "estimability",
            "ok",
            f"All {len(frame)} rows are estimable.",
            value=0,
        )
    else:
        report.add(
            "estimability",
            "critical",
            f"{n_nan} of {len(frame)} rows are non-estimable (NaN). "
            "These usually indicate the model can't separate the effect "
            "you're asking for; consider adding observations or dropping "
            "a collinear predictor.",
            value=n_nan,
        )


def _check_vcov_conditioning(result: Any, report: HealthReport) -> None:
    vcov = result.model_info.vcov
    if vcov is None or vcov.size == 0:
        return
    try:
        cond = float(np.linalg.cond(vcov))
    except np.linalg.LinAlgError:
        report.add(
            "conditioning",
            "critical",
            "vcov is ill-conditioned (LinAlgError from np.linalg.cond).",
        )
        return
    if not np.isfinite(cond):
        report.add(
            "conditioning",
            "critical",
            "vcov condition number is infinite — singular covariance.",
            value=cond,
        )
    elif cond > 1e12:
        report.add(
            "conditioning",
            "critical",
            f"vcov condition number is {cond:.2e} (>1e12); SEs may be "
            "unreliable. Check for near-collinear predictors.",
            value=cond,
        )
    elif cond > 1e8:
        report.add(
            "conditioning",
            "warning",
            f"vcov condition number is {cond:.2e}; borderline. "
            "Consider centering / scaling continuous predictors.",
            value=cond,
        )
    else:
        report.add(
            "conditioning",
            "ok",
            f"vcov condition number is {cond:.2e}.",
            value=cond,
        )


def _check_rank(result: Any, report: HealthReport) -> None:
    info = result.model_info
    n_params = info.n_params
    if info.raw_result is not None and hasattr(info.raw_result, "model"):
        X = np.asarray(getattr(info.raw_result.model, "exog", None))
    else:
        X = None
    if X is None or X.ndim != 2:
        # Use the precomputed estimability basis when available.
        basis = info.estimability_basis
        if basis is None:
            # Either full rank or no X — nothing to check.
            return
        rank = int(basis.shape[0])
    else:
        rank = int(np.linalg.matrix_rank(X))
    if rank == n_params:
        report.add(
            "rank",
            "ok",
            f"Design matrix is full column rank ({rank} of {n_params}).",
            value=rank,
        )
    else:
        report.add(
            "rank",
            "critical",
            f"Design matrix is rank deficient ({rank} of {n_params} "
            "columns linearly independent). Some EMMs / contrasts may "
            "be non-estimable.",
            value=rank,
        )


def _check_df_sanity(result: Any, report: HealthReport) -> None:
    frame = result.frame
    if "df" not in frame.columns:
        return
    df = frame["df"].to_numpy(dtype=float)
    if np.any(np.isnan(df)):
        report.add(
            "df_sanity",
            "warning",
            "Some df values are NaN — Satterthwaite/KR may have failed.",
        )
    finite = df[np.isfinite(df)]
    if finite.size > 0 and finite.min() < 3.0:
        report.add(
            "df_sanity",
            "warning",
            f"Minimum df is {finite.min():.2f}; below 3 means t-quantiles "
            "are extremely heavy-tailed and CIs may overcover.",
            value=float(finite.min()),
        )
    elif finite.size > 0:
        report.add(
            "df_sanity",
            "ok",
            f"df ranges {finite.min():.1f} to {finite.max():.1f}.",
            value=float(finite.min()),
        )


def _check_cell_counts(result: Any, report: HealthReport) -> None:
    """Min training-data n per cell. Only runs for EMMResult."""
    info = result.model_info
    target = getattr(result, "target", None)
    by = getattr(result, "by", None) or []
    if not target or info.data is None or info.data.empty:
        return
    canonical_to_raw = {v: k for k, v in info.aliases.items()}
    group_cols = target + by
    raw_cols = [canonical_to_raw.get(c, c) for c in group_cols]
    valid = [c for c in raw_cols if c in info.data.columns]
    if not valid:
        return
    # observed=False so zero-count cells (factor level present in the
    # data dtype but never observed) show up as n=0 rather than being
    # silently dropped.
    counts = info.data.groupby(valid, observed=False).size()
    if counts.empty:
        return
    min_n = int(counts.min())
    if min_n == 0:
        report.add(
            "cell_counts",
            "critical",
            "At least one EMM cell has zero training observations. "
            "EMMs for those cells will be non-estimable.",
            value=0,
        )
    elif min_n < 5:
        report.add(
            "cell_counts",
            "warning",
            f"Minimum cell size is {min_n}; small-cell EMMs have inflated "
            "SE and unreliable CIs.",
            value=min_n,
        )
    else:
        report.add(
            "cell_counts",
            "ok",
            f"Minimum cell size is {min_n}.",
            value=min_n,
        )


def _check_leverage(result: Any, report: HealthReport) -> None:
    """Belsley-Kuh-Welsch high-leverage flag (OLS only)."""
    info = result.model_info
    raw = info.raw_result
    if raw is None or info.family is not None or info.is_mixed:
        return
    if not hasattr(raw, "get_influence"):
        return
    try:
        infl = raw.get_influence()
        h = np.asarray(infl.hat_matrix_diag)
    except Exception:
        return
    n, p = h.size, info.n_params
    if n == 0:
        return
    threshold = 3.0 * p / n
    high = int((h > threshold).sum())
    max_h = float(h.max())
    if high == 0:
        report.add(
            "influence",
            "ok",
            f"No high-leverage points (max h = {max_h:.3f}, threshold "
            f"{threshold:.3f}).",
            value=max_h,
        )
    elif high / n < 0.05:
        report.add(
            "influence",
            "warning",
            f"{high} of {n} points exceed h > 3p/n = {threshold:.3f} "
            f"(max h = {max_h:.3f}); EMMs may be sensitive to these rows.",
            value=max_h,
        )
    else:
        report.add(
            "influence",
            "critical",
            f"{high} of {n} points exceed the leverage threshold "
            f"({100 * high / n:.1f}% of the data); EMMs are likely "
            "sensitive to a non-trivial fraction of observations.",
            value=max_h,
        )


def _check_boundary_fit(result: Any, report: HealthReport) -> None:
    """MixedLM random-effects variance on the boundary."""
    info = result.model_info
    if not info.is_mixed or info.raw_result is None:
        return
    cov_re = getattr(info.raw_result, "cov_re", None)
    if cov_re is None:
        return
    try:
        diag = np.asarray(cov_re).diagonal()
    except Exception:
        return
    min_var = float(diag.min()) if diag.size else 0.0
    if min_var < 1e-6:
        report.add(
            "boundary_fit",
            "critical",
            f"Random-effects variance hits the boundary "
            f"(min diag = {min_var:.2e}). Kenward-Roger / Satterthwaite "
            "will be unreliable; fall back to OLS or a simpler RE structure.",
            value=min_var,
        )
    else:
        report.add(
            "boundary_fit",
            "ok",
            f"Random-effects variance is well into the interior "
            f"(min diag = {min_var:.2e}).",
            value=min_var,
        )


def health_check(result: Any) -> HealthReport:
    """Run every diagnostic that applies to ``result`` and return a report.

    Accepts an :class:`~pymmeans.EMMResult` or
    :class:`~pymmeans.ContrastResult`. Each check is best-effort: if it
    can't run (missing attribute, GLM where the diagnostic isn't
    meaningful, post-pickle stripped state, etc.) it is silently
    skipped, so the report only contains findings that were actually
    computed.

    Examples
    --------
    >>> emm = emmeans(fit, "treatment")  # doctest: +SKIP
    >>> rep = health_check(emm)  # doctest: +SKIP
    >>> if not rep.ok:  # doctest: +SKIP
    ...     for c in rep.critical:
    ...         print(c)

    Parameters
    ----------
    result
        An ``EMMResult`` or ``ContrastResult``.

    Returns
    -------
    HealthReport
        A list-like bundle of :class:`Check` findings, each with a
        ``severity`` and a human-readable ``message``.
    """
    report = HealthReport()
    _check_estimability(result, report)
    _check_vcov_conditioning(result, report)
    _check_rank(result, report)
    _check_df_sanity(result, report)
    _check_cell_counts(result, report)
    _check_leverage(result, report)
    _check_boundary_fit(result, report)
    return report
