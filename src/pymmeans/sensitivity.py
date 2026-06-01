"""Sensitivity analysis bundled with pymmeans output.

Currently implements the **VanderWeele-Ding (2017) E-value** — the
minimum strength (on the risk-ratio scale) of association that an
unmeasured confounder would need to have with *both* the treatment
and the outcome to fully explain away an observed treatment-outcome
association.

References
----------
- VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in
  observational research: introducing the E-value. *Annals of
  Internal Medicine*, 167(4), 268-274.
- Mathur, M. B., Ding, P., Riddell, C. A., & VanderWeele, T. J.
  (2018). Web site and R package for computing E-values.
  *Epidemiology*, 29(5), e45-e47.

Background
----------
For a risk ratio ``RR > 1``, the E-value is the unique solution to

    RR  =  E * E / (E + (E - 1))

or equivalently

    E   =  RR  +  sqrt( RR * (RR - 1) )

For ``RR < 1``, replace ``RR`` by ``1 / RR`` (the E-value is
symmetric about the null).

Interpretation: an unmeasured confounder ``U`` would need to be
associated with both treatment and outcome by a risk ratio of at
least ``E`` (on both sides) to fully nullify the observed effect.
Higher E-values → more robust to unmeasured confounding.

The "E-value for the CI" is computed by applying the same formula
to the CI bound *closer to 1* — if that bound's E-value > 1, the
finding is statistically robust to unmeasured confounding of at
least that strength.

This module exposes one public function, :func:`e_value`, with
conversion helpers for non-RR effect-size scales (odds ratio,
hazard ratio, standardized mean difference). See VanderWeele-Ding
2017 Table 1 and Mathur et al. 2018 for the approximation
formulae.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

__all__ = ["EValueResult", "e_value"]


@dataclass(frozen=True)
class EValueResult:
    """Result of an E-value sensitivity calculation.

    Attributes
    ----------
    estimate
        The observed effect on the **risk-ratio scale** (after any
        scale conversion). Always >= 1 (we mirror RR < 1 to 1/RR per
        VanderWeele-Ding 2017's symmetry convention).
    e_point
        The E-value for the point estimate. Equal to
        ``RR + sqrt(RR * (RR - 1))``.
    e_ci
        The E-value for the CI bound closer to 1, or ``None`` if no
        CI was supplied. When > 1, the observed finding is robust to
        an unmeasured confounder of at least this strength on both
        the confounder-treatment and confounder-outcome scales.
    kind
        The original effect-size scale (``"rr"``, ``"or"``, ``"hr"``,
        ``"smd"``).
    """

    estimate: float
    e_point: float
    e_ci: float | None
    kind: Literal["rr", "or", "hr", "smd"]

    def __repr__(self) -> str:
        ci_str = (
            f", e_ci={self.e_ci:.4f}"
            if self.e_ci is not None
            else ""
        )
        return (
            f"EValueResult(kind={self.kind!r}, estimate={self.estimate:.4f}, "
            f"e_point={self.e_point:.4f}{ci_str})"
        )


def _e_from_rr(rr: float) -> float:
    """Closed-form E-value from a risk-ratio-scale estimate.

    Per VanderWeele-Ding 2017 Equation 1::

        E = RR + sqrt(RR * (RR - 1))   for RR >= 1
        E = 1/RR + sqrt(1/RR * (1/RR - 1))  for RR < 1
    """
    if not np.isfinite(rr) or rr <= 0:
        raise ValueError(
            f"Risk ratio must be a finite positive number; got {rr!r}."
        )
    if rr < 1.0:
        rr = 1.0 / rr
    return float(rr + np.sqrt(rr * (rr - 1.0)))


def _convert_to_rr(
    estimate: float, kind: str, prevalence: float | None
) -> float:
    """Convert an effect-size estimate to the risk-ratio scale.

    Per VanderWeele-Ding 2017 Table 1 + Mathur et al. 2018.
    """
    if kind == "rr":
        return estimate
    if kind == "or":
        # Odds ratio approximation. For rare outcomes (prevalence < 15%),
        # OR ≈ RR. For common outcomes, apply VanderWeele-Ding's
        # square-root conversion:  RR ≈ sqrt(OR).
        if prevalence is None or prevalence < 0.15:
            return estimate
        return float(np.sqrt(estimate) if estimate >= 1 else 1.0 / np.sqrt(1.0 / estimate))
    if kind == "hr":
        # Hazard ratio. For rare events, HR ≈ RR. For common events,
        # VanderWeele-Ding 2017 use the approximation
        #   RR ≈ (1 - 0.5^sqrt(HR)) / (1 - 0.5^sqrt(1/HR))
        # but for typical observational analyses with prevalence < 15%
        # the rare-event approximation HR ≈ RR is used.
        if prevalence is None or prevalence < 0.15:
            return estimate
        # Common-event conversion. Returns RR symmetric about 1.
        hr_pos = estimate if estimate >= 1 else 1.0 / estimate
        rr_pos = (
            (1.0 - 0.5 ** np.sqrt(hr_pos))
            / (1.0 - 0.5 ** np.sqrt(1.0 / hr_pos))
        )
        return float(rr_pos if estimate >= 1 else 1.0 / rr_pos)
    if kind == "smd":
        # Standardized mean difference d (Cohen's d): VanderWeele-Ding
        # 2017 recommend RR ≈ exp(0.91 * d) for binary-ish outcomes.
        return float(np.exp(0.91 * estimate))
    raise ValueError(
        f"kind must be one of 'rr', 'or', 'hr', 'smd'; got {kind!r}."
    )


def e_value(
    estimate: float,
    *,
    ci_lo: float | None = None,
    ci_hi: float | None = None,
    kind: Literal["rr", "or", "hr", "smd"] = "rr",
    prevalence: float | None = None,
) -> EValueResult:
    """Compute the VanderWeele-Ding (2017) E-value.

    The E-value is the minimum strength of association (on the
    risk-ratio scale) that an unmeasured confounder would need to
    have with **both** the treatment and the outcome to fully
    nullify the observed effect. Higher E-values → more robust to
    unmeasured confounding.

    Parameters
    ----------
    estimate
        The observed effect-size estimate on the scale named by
        ``kind``.
    ci_lo, ci_hi
        Optional confidence interval bounds. If supplied, the E-value
        for the CI bound *closer to 1* is reported as ``e_ci``. This
        is the standard "is the CI robust to unmeasured confounding?"
        check.
    kind
        Effect-size scale. One of:

        - ``"rr"`` (risk ratio): used directly.
        - ``"or"`` (odds ratio): treated as RR if ``prevalence`` is
          rare (<15%); otherwise converted via ``RR ≈ sqrt(OR)``.
        - ``"hr"`` (hazard ratio): treated as RR if event is rare;
          otherwise converted via the VanderWeele-Ding 2017 formula.
        - ``"smd"`` (Cohen's d): converted via ``RR ≈ exp(0.91 × d)``.
    prevalence
        Outcome prevalence (proportion in 0..1). Only used for the
        ``or`` / ``hr`` rare-vs-common-outcome decision. If ``None``
        the rare-outcome approximation is used.

    Returns
    -------
    EValueResult
        E-value for the point estimate and (if a CI is supplied) for
        the CI bound closer to 1.

    Examples
    --------
    From VanderWeele-Ding (2017) Annals of Internal Medicine, the
    canonical smoking-and-lung-cancer example with observed
    ``RR = 10.73``::

        >>> e = e_value(10.73, kind="rr")
        >>> round(e.e_point, 2)
        20.95

    With a CI bound, the E-value for the lower bound is reported::

        >>> e2 = e_value(2.0, ci_lo=1.5, ci_hi=2.7, kind="rr")
        >>> round(e2.e_point, 4)
        3.4142
        >>> round(e2.e_ci, 4)
        2.366

    References
    ----------
    VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in
    observational research: introducing the E-value.
    *Annals of Internal Medicine*, 167(4), 268-274.
    """
    if kind not in ("rr", "or", "hr", "smd"):
        raise ValueError(
            f"kind must be one of 'rr', 'or', 'hr', 'smd'; got {kind!r}."
        )
    if prevalence is not None and not (0.0 <= prevalence <= 1.0):
        raise ValueError(
            f"prevalence must be in [0, 1]; got {prevalence!r}."
        )

    rr_point = _convert_to_rr(float(estimate), kind, prevalence)
    e_point = _e_from_rr(rr_point)

    e_ci: float | None = None
    if ci_lo is not None and ci_hi is not None:
        rr_lo = _convert_to_rr(float(ci_lo), kind, prevalence)
        rr_hi = _convert_to_rr(float(ci_hi), kind, prevalence)
        # CI bound CLOSER to 1 is the one to test for robustness.
        # On the RR scale (normalised to >=1), that's whichever bound
        # is smaller after mirroring to the >=1 side.
        candidates = []
        for rr in (rr_lo, rr_hi):
            mirrored = rr if rr >= 1.0 else 1.0 / rr
            candidates.append(mirrored)
        # If the CI crosses 1 (i.e. one bound < 1 and the other > 1)
        # the E-value for the CI is 1 by convention — the finding
        # is not robust to ANY unmeasured confounding.
        crosses_null = (rr_lo < 1.0 < rr_hi) or (rr_hi < 1.0 < rr_lo)
        if crosses_null:
            e_ci = 1.0
        else:
            e_ci = _e_from_rr(min(candidates))

    return EValueResult(
        estimate=float(rr_point),
        e_point=float(e_point),
        e_ci=e_ci,
        kind=kind,
    )
