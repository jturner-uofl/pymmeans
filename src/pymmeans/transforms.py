"""Response-scale back-transformations for LHS-transformed OLS models.

For a model ``smf.ols("np.log(y) ~ ...", ...)`` the fit predicts
``E[log(y)]``. To report results on the original ``y`` scale we need the
inverse transformation plus delta-method SEs.

When ``bias_adjust=True`` is passed to :func:`regrid_response`, pymmeans
applies R ``emmeans``'s **second-order Taylor** bias correction
``E[Y] ~ exp(mu) * (1 + sigma^2/2)`` (#5 switched to this
for R parity). The exact lognormal correction ``exp(mu + sigma^2/2)``
differs from R's Taylor formula by ~0.05% at sigma~0.25 and ~1.5% at
sigma~0.6; we chose R parity over the exact lognormal mean.

Transformations recognised by name (matching the patsy-canonical form of
``model.endog_names``):

- ``np.log(x)``, ``log(x)`` -- inverse ``exp``, Taylor bias adjust
- ``np.log10(x)`` -- inverse ``10**x``
- ``np.log1p(x)`` -- inverse ``expm1``
- ``np.log2(x)`` -- inverse ``2**x``
- ``np.sqrt(x)``, ``sqrt(x)`` -- inverse ``square``
- ``np.exp(x)`` -- inverse ``log``
- ``np.reciprocal(x)`` -- inverse ``1 / x`` (self-inverse)
- ``logit(x)`` -- inverse ``expit`` (sigmoid)
- ``probit(x)`` -- inverse ``Phi`` (normal CDF)
- ``cloglog(x)`` -- inverse ``1 - exp(-exp(.))``
- ``arcsin(sqrt(x))`` -- inverse ``sin(.)**2``
- ``np.arctanh(x)`` / ``make_tran("atanh")`` -- inverse ``tanh(.)``

Auto-detection requires the inner expression to be a single identifier.
Composite forms like ``np.log(y + 1)``, ``np.sqrt(y * 2)``, or
``np.log(np.log(y))`` are intentionally refused (silently
applying ``exp`` to ``log(y + 1)`` would yield response-scale
predictions off by exactly the additive constant). Users with composite
LHS expressions should pass an explicit transform via
``regrid_response(emm, tran=make_tran("genlog", base=1))`` for
``log(y + 1)`` and similar parametric families.

Parametric transforms (via :func:`make_tran` only — not auto-detected):

- ``make_tran("boxcox", lambda_=)`` -- forward ``(y**λ - 1)/λ``
- ``make_tran("genlog", base=)`` -- forward ``log(y + base)``
- ``make_tran("sqrt_const", const=)`` -- forward ``sqrt(y + const)``
- ``make_tran("scale", mean=, sd=)`` -- forward ``(y - mean) / sd``
- ``make_tran("power", lambda_=)`` -- forward ``y**λ`` (strict-power; ``y > 0``)
- ``make_tran("sympower", lambda_=)`` -- forward ``sign(y) * |y|**λ`` (signed power)
- ``make_tran("bcnPower", lambda_=, gamma=)`` -- forward
  ``((y+γ)**λ - 1) / λ`` (Box-Cox with negatives via shift)
- ``make_tran("yj.power", lambda_=)`` -- Yeo-Johnson piecewise
  power; handles negative y natively

The four proportion-family transforms (``logit`` / ``probit`` /
``cloglog`` / ``asin_sqrt``) are intended for OLS fits on a (0, 1)-
valued response, e.g. ``ols("logit(y) ~ x")``. They do NOT expose
``bias_mean`` — the second derivative of each inverse is sign-changing
on the link scale, so the Taylor bias-correction R uses for logs
doesn't have a clean closed form. Use ``bootstrap_ci`` for
bias-adjusted response-scale CIs on these transforms.

Users can also pass a ``tran=`` dict ``{"inverse": fn, "deriv": fn}`` to
``regrid_response`` for custom transformations.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, NamedTuple

import numpy as np
from scipy import stats


class NonLogContrastBiasAdjustError(ValueError):
    """structured sentinel raised by
    ``regrid_response(contrast, bias_adjust=True)`` when the transform
    is not log-family.

    ``summary_layer._safe_recompute`` catches this specific subclass
    instead of string-matching the error message (the
    fix used ``"log-family" in str(exc)``, which would silently break
    if a future round reworded the error). Subclassing ``ValueError``
    keeps the public API contract: callers that ``except ValueError``
    still see this; callers that want the targeted catch can
    ``except NonLogContrastBiasAdjustError``."""


class Transform(NamedTuple):
    """A LHS transformation + its inverse for response-scale regridding.

    Attributes
    ----------
    name
        Identifier matching the patsy expression name (``"log"`` for
        ``np.log(y)``, ``"sqrt"`` for ``np.sqrt(y)``, etc.).
    inverse
        Maps link-scale predictions back to response scale. For
        ``np.log(y)`` this is ``np.exp``.
    inverse_deriv
        Derivative of ``inverse`` w.r.t. its argument, used for the
        delta-method SE on the response scale.
    bias_mean
        Optional bias-corrected response-scale mean estimator ``f(mu,
        sigma_sq)`` returning ``E[Y]`` when the link is the transform of
        a normal latent. Built-ins use R ``emmeans``'s **second-order
        Taylor** convention (#5), e.g. log uses
        ``exp(mu) * (1 + sigma_sq/2)`` — NOT the exact lognormal
        ``exp(mu + sigma_sq/2)``. The two agree to ~0.05% at
        ``sigma ≈ 0.25`` and diverge to ~1.5% at ``sigma ≈ 0.6``
        chose R parity over the exact lognormal mean. ``None`` for
        transforms where no closed-form correction exists (the
        proportion family, asin_sqrt, exp).
    is_log
        True for the log family (``log``, ``log10``, ``log2``,
        ``log1p``). Drives the contrast → ratio relabelling in
        :func:`regrid_response`.

    All callables are module-level functions (no closures or lambdas)
    so ``Transform`` instances are picklable for multiprocessing
    bootstrap and joblib caching workflows.
    """

    name: str
    inverse: Callable[[np.ndarray], np.ndarray]
    inverse_deriv: Callable[[np.ndarray], np.ndarray]
    bias_mean: Callable[[np.ndarray, float], np.ndarray] | None = None
    is_log: bool = False
    bias_deriv: Callable[[np.ndarray, float], np.ndarray] | None = None
    """Derivative of ``bias_mean`` w.r.t. its first argument; used by
    :func:`regrid_response` for response-scale SE under bias_adjust.
    ``None`` falls back to ``inverse_deriv`` (the unadjusted gradient)."""
    contrast_inverse: Callable[[np.ndarray], np.ndarray] | None = None
    """separate inverse for the CONTRAST
    back-transform path. For plain log (``log(y)``), the contrast
    ``A - B`` exponentiates to the RATIO ``A/B`` of response-scale
    means — i.e. ``contrast_inverse == inverse == exp``. But for
    SHIFTED logs (genlog, log1p, log_const), the EMMs are on the
    ``log(y + c)`` scale and the EMM inverse is ``exp(z) - c``
    CONTRAST inverse must be ``exp(z)`` alone so that the reported
    "ratio" is ``(A + c) / (B + c)``. Without this separate hook,
    `regrid_response` produced contrasts off by exactly the constant
    on every row.

    ``None`` means "use ``inverse`` for the contrast path too"
    (correct for plain log family). Custom transforms that have a
    different contrast inverse can opt in."""
    contrast_inverse_deriv: Callable[[np.ndarray], np.ndarray] | None = None
    """Derivative of ``contrast_inverse``; falls back to
    ``inverse_deriv`` when ``None``."""


# All bias/inverse/deriv helpers are module-level so the built-in
# Transforms are picklable (lambdas and closures would break
# multiprocessing bootstrap and caching).


# R `emmeans` uses a second-order Taylor expansion for `bias.adjust`,
# not the exact lognormal mean: bias_mean ~ inverse(mu) + 0.5 *
# inverse''(mu) * sigma^2. For log links this is `exp(mu) * (1 + s^2/2)`,
# which differs from the exact lognormal `exp(mu + s^2/2)` by a factor
# of `exp(s^2/2) / (1 + s^2/2)`. The discrepancy is ~0.06% at s=0.26
# (where measured "agreement") and ~1.5% at s=0.6 (where
# it). We adopted R's formula in #5 so
# `regrid_response(..., bias_adjust=True)` matches R `summary(..., bias.adjust=TRUE)`
# to floating-point precision.
def _bias_log(mu: np.ndarray, s2: float) -> np.ndarray:
    return np.exp(mu) * (1.0 + s2 / 2.0)


def _bias_log10(mu: np.ndarray, s2: float) -> np.ndarray:
    return 10.0**mu * (1.0 + (np.log(10.0) ** 2) * s2 / 2.0)


def _bias_log2(mu: np.ndarray, s2: float) -> np.ndarray:
    return 2.0**mu * (1.0 + (np.log(2.0) ** 2) * s2 / 2.0)


def _bias_log1p(mu: np.ndarray, s2: float) -> np.ndarray:
    # inverse(eta) = expm1(eta); inverse''(eta) = exp(eta); Taylor:
    return np.expm1(mu) + np.exp(mu) * s2 / 2.0


def _bias_sqrt(mu: np.ndarray, s2: float) -> np.ndarray:
    # inverse(eta) = eta^2; inverse''(eta) = 2; Taylor gives eta^2 + s2,
    # which also happens to be the exact mean of a squared Gaussian, so
    # this matches both R and the exact closed form.
    return mu**2 + s2


# Derivatives of `bias_mean(mu, s2)` w.r.t. mu. Used by `regrid_response`
# under bias_adjust=True for the response-scale SE, matching R's
# convention of evaluating the delta-method gradient at the
# bias-corrected point.
def _bias_deriv_log(mu: np.ndarray, s2: float) -> np.ndarray:
    return np.exp(mu) * (1.0 + s2 / 2.0)


def _bias_deriv_log10(mu: np.ndarray, s2: float) -> np.ndarray:
    return 10.0**mu * np.log(10.0) * (1.0 + (np.log(10.0) ** 2) * s2 / 2.0)


def _bias_deriv_log2(mu: np.ndarray, s2: float) -> np.ndarray:
    return 2.0**mu * np.log(2.0) * (1.0 + (np.log(2.0) ** 2) * s2 / 2.0)


def _bias_deriv_log1p(mu: np.ndarray, s2: float) -> np.ndarray:
    return np.exp(mu) * (1.0 + s2 / 2.0)


def _bias_deriv_sqrt(mu: np.ndarray, s2: float) -> np.ndarray:
    return 2.0 * mu


def _pow10(x: np.ndarray) -> np.ndarray:
    return 10.0**x


def _pow10_deriv(x: np.ndarray) -> np.ndarray:
    return 10.0**x * np.log(10.0)


def _pow2(x: np.ndarray) -> np.ndarray:
    return 2.0**x


def _pow2_deriv(x: np.ndarray) -> np.ndarray:
    return 2.0**x * np.log(2.0)


def _square(x: np.ndarray) -> np.ndarray:
    return x**2


def _two_x(x: np.ndarray) -> np.ndarray:
    return 2.0 * x


def _reciprocal(x: np.ndarray) -> np.ndarray:
    return 1.0 / x


def _neg_reciprocal_sq(x: np.ndarray) -> np.ndarray:
    """Derivative of ``1/x`` w.r.t. x: ``-1/x²``."""
    return -1.0 / (x * x)


# proportion-family LHS transforms commonly used in pre-GLM-era
# ANOVA workflows. All map a (0, 1)-valued response to the real line so an
# OLS fit makes sense, with the inverse mapping the link back to (0, 1).
#
# Note these are also the canonical GLM link inverses — when the user
# fits `glm(family=Binomial(link=Logit()))` we use the family object's
# inverse, not these transforms. These are for when the user instead
# writes `ols("logit(y) ~ ...")` and wants `regrid_response()` to
# back-transform.


def _expit(x: np.ndarray) -> np.ndarray:
    """Inverse of the logit link: ``1 / (1 + exp(-x))``.

    Stable for large |x|: uses ``scipy.special.expit`` which switches
    branches to avoid overflow.
    """
    from scipy.special import expit
    return expit(x)


def _expit_deriv(x: np.ndarray) -> np.ndarray:
    p = _expit(x)
    return p * (1.0 - p)


def _ndtr(x: np.ndarray) -> np.ndarray:
    """Standard-normal CDF — inverse of the probit link."""
    from scipy.special import ndtr
    return ndtr(x)


def _ndtr_deriv(x: np.ndarray) -> np.ndarray:
    """Standard-normal PDF (derivative of ``ndtr``)."""
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)


def _cloglog_inv(x: np.ndarray) -> np.ndarray:
    """Inverse of the complementary-log-log link: ``1 - exp(-exp(x))``."""
    return -np.expm1(-np.exp(x))


def _cloglog_inv_deriv(x: np.ndarray) -> np.ndarray:
    # d/dx [1 - exp(-exp(x))] = exp(x) * exp(-exp(x)) = exp(x - exp(x))
    return np.exp(x - np.exp(x))


def _sin_sq(x: np.ndarray) -> np.ndarray:
    """Inverse of asin(sqrt(.)) — squared sine."""
    return np.sin(x) ** 2


def _sin_sq_deriv(x: np.ndarray) -> np.ndarray:
    # d/dx sin(x)^2 = 2 sin(x) cos(x) = sin(2x)
    return np.sin(2.0 * x)


def _tanh_inv(x: np.ndarray) -> np.ndarray:
    """Inverse of atanh(.) — hyperbolic tangent.

    Forward (link): ``atanh(y) = 0.5 * log((1 + y) / (1 - y))`` for
    ``y ∈ (-1, 1)``. Common use cases are Fisher's z-transform of
    Pearson correlations and analyses of bounded-on-(-1, 1) outcomes
    like LD parity or normalised differences.
    """
    return np.tanh(x)


def _tanh_inv_deriv(x: np.ndarray) -> np.ndarray:
    # d/dz tanh(z) = 1 - tanh(z)^2 = sech(z)^2. Numerically stable as
    # written (avoids 1/cosh^2 which can underflow at large |z|).
    return 1.0 - np.tanh(x) ** 2


# Box-Cox helpers: module-level so `functools.partial(_boxcox_inv, lam)`
# is picklable (the bound `lam` is a plain float). Lambdas/closures would
# break multiprocessing bootstrap and joblib caching, so we go through
# this dance instead.
def _boxcox_inv(lam: float, x: np.ndarray) -> np.ndarray:
    """Inverse Box-Cox at parameter ``lam``.

    Forward: ``z = (y**lam - 1) / lam`` for ``lam != 0``, else ``log(y)``.
    Inverse: ``y = (lam * z + 1) ** (1/lam)`` for ``lam != 0``,
             else ``y = exp(z)``.

    the previous implementation used
    ``np.sign(base) * np.abs(base) ** (1/lam)``, which returned
    well-defined-looking NEGATIVE responses for ``lam * z + 1 < 0``
    even though the original Box-Cox domain requires ``y > 0`` and
    therefore ``lam * z + 1 >= 0``. That silently produced finite
    out-of-domain values; downstream code never noticed. Now return
    NaN cleanly outside the domain so users can spot pathological
    inputs.
    """
    if lam == 0.0:
        return np.exp(x)
    base = lam * x + 1.0
    return np.where(base >= 0.0, np.power(np.maximum(base, 0.0), 1.0 / lam), np.nan)


def _boxcox_inv_deriv(lam: float, x: np.ndarray) -> np.ndarray:
    """Derivative of ``_boxcox_inv`` w.r.t. ``x`` at parameter ``lam``.

    NaN outside the Box-Cox domain (``lam * z + 1 <= 0``); see
    :func:`_boxcox_inv` for the rationale.
    """
    if lam == 0.0:
        return np.exp(x)
    base = lam * x + 1.0
    return np.where(base > 0.0, np.power(np.maximum(base, 0.0), 1.0 / lam - 1.0), np.nan)


# `genlog(y, base)` is R's `make.tran("genlog", base)` — the
# generalized log for shifted data. Forward: z = log(y + base); inverse:
# y = exp(z) - base. Common when y has zeros and a plain `log(y)` fails.
def _genlog_inv(base: float, x: np.ndarray) -> np.ndarray:
    return np.exp(x) - base


def _genlog_inv_deriv(base: float, x: np.ndarray) -> np.ndarray:
    # d/dx [exp(x) - base] = exp(x)
    return np.exp(x)


# `sqrt_const(y, c)` = sqrt(y + c). Forward z = sqrt(y + c);
# inverse y = z^2 - c. Used in variance-stabilising count transforms
# where `sqrt(y)` is over-stabilised near zero (Anscombe 1948 suggests
# c = 3/8 = 0.375 for Poisson counts).
def _sqrt_const_inv(const: float, x: np.ndarray) -> np.ndarray:
    return x * x - const


def _sqrt_const_inv_deriv(const: float, x: np.ndarray) -> np.ndarray:
    # d/dx [x^2 - const] = 2x
    return 2.0 * x


# `scale(y, mu, sd)` = (y - mu)/sd. The forward transform
# z-scores the response using a fixed (mu, sd) — typically the
# training-data mean/SD — so the fit is on a standardised scale.
# Inverse: y = z*sd + mu. Inverse derivative: sd (constant). Useful
# for Bayesian / posterior workflows that compute on standardised
# coefficients and want to back-transform to the original units.
def _scale_inv(mu: float, sd: float, x: np.ndarray) -> np.ndarray:
    return x * sd + mu


def _scale_inv_deriv(mu: float, sd: float, x: np.ndarray) -> np.ndarray:
    # d/dx [x*sd + mu] = sd; broadcast to x.shape so the SE machinery
    # gets a same-shape array rather than a scalar.
    return np.full_like(np.asarray(x, dtype=float), sd)


# ``power(y, lambda)`` — R ``make.tran("power", lambda)``.
# Forward: z = y^lambda for y > 0 (lambda > 0); a strict-power transform
# with no shift. Distinct from Box-Cox: Box-Cox subtracts 1 and divides
# by lambda to give a smooth lambda -> 0 limit; ``power`` is just the
# raw power, useful when the user wants ``y^0.5`` (square root) or
# ``y^-1`` (reciprocal) without the Box-Cox affine adjustment.
#
# Inverse: y = z^(1/lambda). Domain: z^(1/lambda) is real-valued for
# z > 0 when 1/lambda is non-integer; we return NaN for z <= 0 to
# match the Box-Cox out-of-domain policy and avoid silent
# wrong-numerics on negative link values.
def _power_inv(lam: float, x: np.ndarray) -> np.ndarray:
    """Inverse of ``power(., lambda)`` — i.e. ``z ** (1/lambda)``.

    Returns NaN for ``z <= 0`` when ``1/lambda`` is non-integer to
    match the strict-power domain (the forward transform requires
    ``y > 0``).
    """
    inv_lam = 1.0 / lam
    x_arr = np.asarray(x, dtype=float)
    return np.where(
        x_arr > 0.0, np.power(np.maximum(x_arr, 0.0), inv_lam), np.nan,
    )


def _power_inv_deriv(lam: float, x: np.ndarray) -> np.ndarray:
    """Derivative of ``_power_inv`` w.r.t. ``x`` at parameter ``lam``.

    ``d/dz z^(1/lam) = (1/lam) * z^(1/lam - 1)`` for ``z > 0``.
    NaN for ``z <= 0``; see :func:`_power_inv`.
    """
    inv_lam = 1.0 / lam
    x_arr = np.asarray(x, dtype=float)
    return np.where(
        x_arr > 0.0,
        inv_lam * np.power(np.maximum(x_arr, 0.0), inv_lam - 1.0),
        np.nan,
    )


# ``sympower(y, lambda)`` — R ``make.tran("sympower", lambda)``.
# Forward: z = sign(y) * |y|^lambda, defined for all real y.
# Useful when the response can be negative (e.g. detrended signals)
# and a plain ``power`` is out-of-domain.
#
# Inverse: y = sign(z) * |z|^(1/lambda). Symmetric in sign, so
# negative link values map to negative response values without the
# NaN-at-zero policy ``power`` needs.
def _sympower_inv(lam: float, x: np.ndarray) -> np.ndarray:
    """Inverse of ``sympower(., lambda)`` — sign-preserving power.

    ``y = sign(z) * |z|^(1/lambda)`` for all real z. Continuous and
    differentiable at z = 0 when ``1/lambda > 1`` (i.e. ``lambda <
    1``); when ``1/lambda < 1`` the derivative diverges at zero but
    the inverse itself is still well-defined.
    """
    inv_lam = 1.0 / lam
    x_arr = np.asarray(x, dtype=float)
    return np.sign(x_arr) * np.power(np.abs(x_arr), inv_lam)


def _sympower_inv_deriv(lam: float, x: np.ndarray) -> np.ndarray:
    """Derivative of ``_sympower_inv`` w.r.t. ``x`` at parameter ``lam``.

    ``d/dz [sign(z) * |z|^(1/lam)] = (1/lam) * |z|^(1/lam - 1)``;
    the sign cancels because ``sign(z)^2 = 1`` on its support. At
    ``z = 0`` the derivative is ``+inf`` when ``1/lam < 1`` (i.e.
    ``lam > 1``); we return ``+inf`` rather than NaN so the
    delta-method SE machinery sees a clearly-flagged divergence
    instead of a silent zero.
    """
    inv_lam = 1.0 / lam
    x_arr = np.asarray(x, dtype=float)
    abs_x = np.abs(x_arr)
    if inv_lam == 1.0:
        # Identity case: derivative is 1 everywhere.
        return np.ones_like(x_arr)
    return inv_lam * np.power(abs_x, inv_lam - 1.0)


# ``bcnPower(y, lambda, gamma)`` — "Box-Cox with negatives", from
# Hawkins & Weisberg (2017). Forward: ``z = ((y + gamma)^lambda - 1)
# / lambda`` for lambda != 0, else ``log(y + gamma)``. Implementation
# notes:
#   - Reuses :func:`_boxcox_inv` for the lambda-dependent shape; the
#     additive gamma shift drops out of the derivative (so we reuse
#     :func:`_boxcox_inv_deriv` directly via partial).
#   - Domain: forward requires ``y + gamma > 0``; inverse returns
#     NaN where the Box-Cox base ``lam * z + 1`` is non-positive.
def _bcn_power_inv(
    lam: float, gamma: float, x: np.ndarray,
) -> np.ndarray:
    """Inverse of bcnPower(., lambda, gamma).

    ``y = (lam * z + 1)^(1/lam) - gamma`` for ``lam != 0``;
    ``y = exp(z) - gamma`` for ``lam == 0``. NaN where the
    underlying Box-Cox inverse is NaN.
    """
    return _boxcox_inv(lam, x) - gamma


# Inverse derivative is identical to plain Box-Cox's (gamma is a
# constant shift that vanishes under differentiation). The
# ``make_tran('bcnPower', ...)`` branch reuses
# :func:`_boxcox_inv_deriv` via ``functools.partial`` — no separate
# helper needed.


# ``yj.power(y, lambda)`` — Yeo & Johnson (2000) power transform.
# Defined piecewise so the transform handles negative ``y`` natively
# (unlike Box-Cox, which requires ``y > 0``). Forward:
#   - ``y >= 0``, ``lambda != 0``: ``((y + 1)^lambda - 1) / lambda``
#   - ``y >= 0``, ``lambda == 0``: ``log(y + 1)``
#   - ``y <  0``, ``lambda != 2``: ``-((-y + 1)^(2-lambda) - 1) / (2-lambda)``
#   - ``y <  0``, ``lambda == 2``: ``-log(-y + 1)``
# Forward is monotonic in y; ``sign(z) == sign(y)`` so the inverse
# branches on ``z >= 0`` vs ``z < 0``.
def _yj_power_inv(lam: float, x: np.ndarray) -> np.ndarray:
    """Inverse of yj.power(., lambda)."""
    x_arr = np.asarray(x, dtype=float)
    out = np.empty_like(x_arr)
    pos = x_arr >= 0.0
    neg = ~pos
    # Positive branch: inverse of ((y+1)^lam - 1)/lam (lam != 0)
    # or log(y+1) (lam == 0). Result is y >= 0.
    if lam == 0.0:
        out[pos] = np.expm1(x_arr[pos])  # exp(z) - 1
    else:
        base_pos = lam * x_arr[pos] + 1.0
        out[pos] = np.power(np.maximum(base_pos, 0.0), 1.0 / lam) - 1.0
    # Negative branch: inverse of -((-y+1)^(2-lam) - 1) / (2-lam)
    # (lam != 2) or -log(-y+1) (lam == 2). Result is y < 0.
    if lam == 2.0:
        out[neg] = 1.0 - np.exp(-x_arr[neg])
    else:
        two_m = 2.0 - lam
        base_neg = 1.0 - two_m * x_arr[neg]
        out[neg] = 1.0 - np.power(np.maximum(base_neg, 0.0), 1.0 / two_m)
    return out


def _yj_power_inv_deriv(lam: float, x: np.ndarray) -> np.ndarray:
    """Derivative of yj.power inverse w.r.t. z."""
    x_arr = np.asarray(x, dtype=float)
    out = np.empty_like(x_arr)
    pos = x_arr >= 0.0
    neg = ~pos
    if lam == 0.0:
        out[pos] = np.exp(x_arr[pos])
    else:
        base_pos = lam * x_arr[pos] + 1.0
        out[pos] = np.power(np.maximum(base_pos, 0.0), 1.0 / lam - 1.0)
    if lam == 2.0:
        out[neg] = np.exp(-x_arr[neg])
    else:
        two_m = 2.0 - lam
        base_neg = 1.0 - two_m * x_arr[neg]
        out[neg] = np.power(np.maximum(base_neg, 0.0), 1.0 / two_m - 1.0)
    return out


# the scale transform is LINEAR, so the
# contrast back-transform is well-defined: a difference `A - B` on the
# z-scored scale corresponds to `sd * (A - B)` on the original scale
# (the constant `mu` cancels). Provide `contrast_inverse` so
# `regrid_response` on a contrast works for scale, even though
# `is_log=False` (no ratio interpretation).
def _scale_contrast_inv(sd: float, x: np.ndarray) -> np.ndarray:
    """Linear contrast back-transform: `sd * diff` (mean cancels)."""
    return x * sd


def _scale_contrast_inv_deriv(sd: float, x: np.ndarray) -> np.ndarray:
    return np.full_like(np.asarray(x, dtype=float), sd)


# identity transform — explicit no-op.
# R supports `make.tran("identity")` for explicit pass-through.
def _identity_fn(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)


def _identity_deriv(x: np.ndarray) -> np.ndarray:
    return np.ones_like(np.asarray(x, dtype=float))


# Forward transforms (response -> link). Used by :func:`regrid` for
# the general "switch to a different scale" workflow — apply the
# new transform's forward to the response-scale EMMs, with delta-
# method SE via the chain rule ``forward'(y) = 1 / inverse'(forward(y))``.
#
# Each helper is a module-level function so the resulting synthetic
# Transform stays picklable (regrid composes a temporary Transform
# carrying these as its ``inverse`` and ``inverse_deriv`` for reuse
# of the existing :func:`regrid_response` machinery).
def _log_forward(y: np.ndarray) -> np.ndarray:
    return np.log(y)


def _log_forward_deriv(y: np.ndarray) -> np.ndarray:
    return 1.0 / np.asarray(y, dtype=float)


def _log10_forward(y: np.ndarray) -> np.ndarray:
    return np.log10(y)


def _log10_forward_deriv(y: np.ndarray) -> np.ndarray:
    return 1.0 / (np.asarray(y, dtype=float) * np.log(10.0))


def _log2_forward(y: np.ndarray) -> np.ndarray:
    return np.log2(y)


def _log2_forward_deriv(y: np.ndarray) -> np.ndarray:
    return 1.0 / (np.asarray(y, dtype=float) * np.log(2.0))


def _log1p_forward(y: np.ndarray) -> np.ndarray:
    return np.log1p(y)


def _log1p_forward_deriv(y: np.ndarray) -> np.ndarray:
    return 1.0 / (np.asarray(y, dtype=float) + 1.0)


def _sqrt_forward(y: np.ndarray) -> np.ndarray:
    return np.sqrt(y)


def _sqrt_forward_deriv(y: np.ndarray) -> np.ndarray:
    return 0.5 / np.sqrt(np.asarray(y, dtype=float))


def _exp_forward(y: np.ndarray) -> np.ndarray:
    return np.exp(y)


def _exp_forward_deriv(y: np.ndarray) -> np.ndarray:
    return np.exp(y)


def _logit_forward(y: np.ndarray) -> np.ndarray:
    from scipy.special import logit as _scipy_logit
    return _scipy_logit(y)


def _logit_forward_deriv(y: np.ndarray) -> np.ndarray:
    y_arr = np.asarray(y, dtype=float)
    return 1.0 / (y_arr * (1.0 - y_arr))


def _probit_forward(y: np.ndarray) -> np.ndarray:
    from scipy.stats import norm as _scipy_norm
    return _scipy_norm.ppf(y)


def _probit_forward_deriv(y: np.ndarray) -> np.ndarray:
    # forward'(y) = 1 / pdf(ppf(y)). For the standard normal,
    # pdf(z) = exp(-z^2/2)/sqrt(2*pi); compute directly to avoid
    # the scipy dispatch on the hot path.
    from scipy.stats import norm as _scipy_norm
    z = _scipy_norm.ppf(y)
    return np.sqrt(2.0 * np.pi) * np.exp(0.5 * z * z)


def _cloglog_forward(y: np.ndarray) -> np.ndarray:
    # cloglog(y) = log(-log(1 - y)) for y in (0, 1).
    return np.log(-np.log1p(-np.asarray(y, dtype=float)))


def _cloglog_forward_deriv(y: np.ndarray) -> np.ndarray:
    # d/dy log(-log(1-y))
    # = (1/(-log(1-y))) * (d/dy(-log(1-y)))
    # = (1/(-log(1-y))) * (1/(1-y))
    # = -1 / ((1-y) * log(1-y))
    y_arr = np.asarray(y, dtype=float)
    return -1.0 / ((1.0 - y_arr) * np.log1p(-y_arr))


def _asin_sqrt_forward(y: np.ndarray) -> np.ndarray:
    return np.arcsin(np.sqrt(y))


def _asin_sqrt_forward_deriv(y: np.ndarray) -> np.ndarray:
    # d/dy arcsin(sqrt(y)) = 1 / (2 * sqrt(y * (1 - y)))
    y_arr = np.asarray(y, dtype=float)
    return 1.0 / (2.0 * np.sqrt(y_arr * (1.0 - y_arr)))


def _arctanh_forward(y: np.ndarray) -> np.ndarray:
    return np.arctanh(y)


def _arctanh_forward_deriv(y: np.ndarray) -> np.ndarray:
    # d/dy arctanh(y) = 1 / (1 - y^2)
    y_arr = np.asarray(y, dtype=float)
    return 1.0 / (1.0 - y_arr * y_arr)


def _reciprocal_forward(y: np.ndarray) -> np.ndarray:
    return np.reciprocal(np.asarray(y, dtype=float))


def _reciprocal_forward_deriv(y: np.ndarray) -> np.ndarray:
    y_arr = np.asarray(y, dtype=float)
    return -1.0 / (y_arr * y_arr)


# Registry of forward functions keyed by transform name. The
# :func:`regrid` "switch to a new scale" path looks transforms up
# here; parametric families (boxcox / genlog / ...) and user-
# defined transforms are not in the table and must be requested
# via the explicit Transform-instance form (deferred to 0.2.0).
_FORWARD: dict[str, tuple[
    Callable[[np.ndarray], np.ndarray],
    Callable[[np.ndarray], np.ndarray],
]] = {
    "log": (_log_forward, _log_forward_deriv),
    "log10": (_log10_forward, _log10_forward_deriv),
    "log2": (_log2_forward, _log2_forward_deriv),
    "log1p": (_log1p_forward, _log1p_forward_deriv),
    "sqrt": (_sqrt_forward, _sqrt_forward_deriv),
    "exp": (_exp_forward, _exp_forward_deriv),
    "logit": (_logit_forward, _logit_forward_deriv),
    "probit": (_probit_forward, _probit_forward_deriv),
    "cloglog": (_cloglog_forward, _cloglog_forward_deriv),
    "asin_sqrt": (_asin_sqrt_forward, _asin_sqrt_forward_deriv),
    "arctanh": (_arctanh_forward, _arctanh_forward_deriv),
    "atanh": (_arctanh_forward, _arctanh_forward_deriv),
    "reciprocal": (_reciprocal_forward, _reciprocal_forward_deriv),
    "identity": (_identity_fn, _identity_deriv),
}


_BUILTIN: dict[str, Transform] = {
    "log": Transform("log", np.exp, np.exp, _bias_log, True, _bias_deriv_log),
    "log10": Transform(
        "log10", _pow10, _pow10_deriv, _bias_log10, True, _bias_deriv_log10
    ),
    # log1p is a shifted log
    # (y → log(y + 1)). Contrasts on the link scale exponentiate to
    # (y_a + 1)/(y_b + 1), so contrast_inverse is plain `exp` while
    # the EMM-row inverse `expm1` handles the back-shift.
    "log1p": Transform(
        "log1p", np.expm1, np.exp, _bias_log1p, True, _bias_deriv_log1p,
        contrast_inverse=np.exp, contrast_inverse_deriv=np.exp,
    ),
    "log2": Transform(
        "log2", _pow2, _pow2_deriv, _bias_log2, True, _bias_deriv_log2
    ),
    "sqrt": Transform("sqrt", _square, _two_x, _bias_sqrt, False, _bias_deriv_sqrt),
    "exp": Transform("exp", np.log, _reciprocal, None, False, None),
    # proportion-family transforms. None expose `bias_mean`
    # because there's no clean closed-form Taylor correction for these
    # links (the second derivative of the inverse is sign-changing on
    # the link scale). Users wanting bias-adjusted response-scale means
    # for these transforms should use a parametric bootstrap via
    # bootstrap_ci instead.
    "logit": Transform("logit", _expit, _expit_deriv, None, False, None),
    "probit": Transform("probit", _ndtr, _ndtr_deriv, None, False, None),
    "cloglog": Transform(
        "cloglog", _cloglog_inv, _cloglog_inv_deriv, None, False, None
    ),
    "asin_sqrt": Transform(
        "asin_sqrt", _sin_sq, _sin_sq_deriv, None, False, None
    ),
    # 1/y is self-inverse (1/(1/y) = y). Useful for pharma /
    # rate models where a reciprocal transform stabilises variance.
    # Auto-detection works when the user writes `np.reciprocal(y)` in
    # the formula; for `I(1/y)` patsy wraps in identity and the endog
    # name becomes `"I(1 / y)"`, which detect_transform refuses as a
    # composite inner — the user must pass `tran=make_tran('reciprocal')`.
    "reciprocal": Transform(
        "reciprocal", _reciprocal, _neg_reciprocal_sq, None, False, None
    ),
    # Hyperbolic arctangent. Registered under TWO keys so both
    # ``np.arctanh(y)`` (numpy/patsy idiom; auto-detected) and
    # ``make_tran("atanh")`` (R / math idiom; explicit) resolve to
    # the same Transform. No ``bias_mean`` — like the other bounded-
    # response transforms (logit / probit / cloglog / asin_sqrt),
    # the inverse's second derivative is sign-changing on the link
    # scale, so the Taylor bias correction R uses for logs has no
    # clean closed form. Users wanting bias-adjusted response-scale
    # means on atanh-transformed responses should use bootstrap_ci
    # with kind='parametric'.
    "arctanh": Transform(
        "arctanh", _tanh_inv, _tanh_inv_deriv, None, False, None
    ),
    "atanh": Transform(
        "atanh", _tanh_inv, _tanh_inv_deriv, None, False, None
    ),
}


# Public alias for the built-in registry. The previously-private
# ``_BUILTIN`` mapping is now a documented surface so downstream
# code (and external callers building custom transforms) can read
# the available built-ins by name, iterate the supported transforms,
# and plug new ones in via :func:`register_transform`. ``_BUILTIN``
# is retained as a private alias for backwards compatibility with
# existing internal call sites that already reference it.
TRANSFORMS: dict[str, Transform] = _BUILTIN


def register_transform(
    name: str,
    transform: Transform,
    *,
    overwrite: bool = False,
    forward: Callable[[np.ndarray], np.ndarray] | None = None,
    forward_deriv: Callable[[np.ndarray], np.ndarray] | None = None,
) -> None:
    """Register a custom transform so it's resolvable by name.

    After registration, ``make_tran(name)`` returns the registered
    ``Transform`` and ``detect_transform("np.{name}(y)")`` resolves to
    it when the LHS uses the matching function name.

    Parameters
    ----------
    name
        Identifier the transform should be registered under. Use the
        bare function name (``"power"``, ``"atanh"``) so
        :func:`detect_transform` picks it up from ``"np.power(y)"``-
        style patsy endog strings. Names are lower-cased on lookup
        to match ``detect_transform``'s canonicalisation.
    transform
        The :class:`Transform` to register. Must already be a fully
        built instance (use :func:`make_tran` for parametric families
        or construct ``Transform(...)`` directly for ad-hoc inverses).
    overwrite
        Default ``False``: registering a name that already exists
        raises ``ValueError`` so plugin code doesn't silently shadow
        a built-in. Pass ``True`` to deliberately replace an existing
        entry.
    forward
        Optional callable mapping response-scale values to the new
        display scale (the "forward" direction; ``transform`` stores
        the back-transform). When supplied alongside
        ``forward_deriv``, the entry is ALSO registered in the
        named-forward dispatch table so
        ``regrid(em, transform=name)`` can apply this transform to a
        response-scale EMM. When omitted, only the
        back-transform path is wired up — ``regrid(transform=name)``
        will raise with a clear "register the forward direction"
        message.
    forward_deriv
        Derivative of ``forward`` w.r.t. its argument. Required
        whenever ``forward`` is supplied (used by the delta-method SE
        in the forward-scale display).

    Raises
    ------
    ValueError
        If ``name`` is empty, not a string, or already registered
        (when ``overwrite=False``).
    TypeError
        If ``transform`` is not a :class:`Transform` instance.

    Examples
    --------
    >>> from pymmeans import Transform, register_transform # doctest: +SKIP
    >>> import numpy as np # doctest: +SKIP
    >>> # Custom atanh transform.
    >>> register_transform( # doctest: +SKIP
    ... "atanh",
    ... Transform("atanh", np.tanh, lambda x: 1 - np.tanh(x) ** 2),
    ... )

    Notes
    -----
    The registry is process-local (a plain module-level dict, not a
    :class:`contextvars.ContextVar`). Custom registrations made in
    the parent process do NOT propagate to ``joblib.Parallel`` /
    ``ProcessPoolExecutor`` workers; re-register inside the worker
    function (or use the worker's ``initializer=`` hook).
    """
    if not isinstance(name, str):
        raise TypeError(
            f"register_transform(name=...): name must be a string, got "
            f"{type(name).__name__}."
        )
    if not name:
        raise ValueError(
            "register_transform(name=...): name must be non-empty."
        )
    if not isinstance(transform, Transform):
        raise TypeError(
            "register_transform(transform=...): transform must be a "
            f"pymmeans.Transform instance, got {type(transform).__name__}."
        )
    key = name.lower()
    if key in TRANSFORMS and not overwrite:
        raise ValueError(
            f"register_transform: name {name!r} already registered "
            f"(existing transform: {TRANSFORMS[key].name!r}). Pass "
            "``overwrite=True`` to replace it."
        )
    # Asymmetric forward/forward_deriv: providing one without the
    # other would silently route through ``_FORWARD`` with a missing
    # callable. Refuse cleanly so the bug surfaces at registration
    # rather than at the first ``regrid(transform=name)`` call.
    if (forward is None) != (forward_deriv is None):
        raise ValueError(
            "register_transform: 'forward' and 'forward_deriv' must be "
            "supplied together (or both omitted). Got "
            f"forward={'set' if forward is not None else 'None'}, "
            f"forward_deriv={'set' if forward_deriv is not None else 'None'}."
        )
    TRANSFORMS[key] = transform
    # When the caller supplies both directions, wire up the named-
    # forward dispatch table too so ``regrid(em, transform=name)``
    # can apply this transform to a response-scale EMM. Without the
    # forward pair the back-transform path still works (regrid_response,
    # detect_transform on patsy LHS) — only forward-name dispatch is
    # opt-in.
    if forward is not None and forward_deriv is not None:
        _FORWARD[key] = (forward, forward_deriv)
    elif overwrite and key in _FORWARD:
        # Overwriting a transform that previously HAD a forward pair,
        # with one that omits the forward kwargs, must drop the stale
        # entry — otherwise ``regrid(em, transform=name)`` would
        # silently keep applying the OLD transform's forward direction.
        del _FORWARD[key]


def detect_transform(endog_name: str) -> Transform | None:
    """Identify the LHS transformation in a patsy endog string.

    ``"np.log(conc)" -> Transform('log', ...)``
    ``"log10(y)" -> Transform('log10', ...)``
    ``"logit(y)" -> Transform('logit', ...)``
    ``"np.arcsin(np.sqrt(y))" -> Transform('asin_sqrt', ...)``
    ``"y" -> None``
    ``"np.log(y + 1)" -> None`` (composite inner; user must
                                          pass tran= explicitly)
    """
    if not endog_name:
        return None
    # composite-form detection. `arcsin(sqrt(.))` and its
    # numpy-prefixed cousin are the canonical proportion-family LHS
    # transform; patsy spells it nested rather than as a single
    # identifier, so we pattern-match the composition explicitly before
    # falling back to the simple-unary regex.
    # the Unicode fix landed
    # on the simple-unary regex but missed this composite pattern.
    # Use a permissive inner capture + structural identifier check
    # via `str.isidentifier()` per dotted segment.
    composite_asin_sqrt = re.match(
        r"^(?:np\.)?(?:arcsin|asin)\(\s*(?:np\.)?sqrt\(\s*(.+?)\s*\)\s*\)$",
        endog_name,
        flags=re.UNICODE,
    )
    if composite_asin_sqrt is not None:
        inner = composite_asin_sqrt.group(1)
        if inner and all(seg.isidentifier() for seg in inner.split(".")):
            return _BUILTIN.get("asin_sqrt")
        # Composite-inner asin(sqrt(...)) (e.g. arcsin(sqrt(y + 0.1)))
        # — refuse, matching the composite-inner policy.
        return None
    # previously `^(?:np\.)?(\w+)\(.*\)$` matched ANY inner
    # expression, including `np.log(y + 1)` -> 'log'. That meant
    # `regrid_response` silently used the wrong inverse (exp(L) instead
    # of exp(L)-1), giving response-scale predictions off by exactly
    # the additive constant. Now require the inner expression to be a
    # single identifier (possibly with dotted attribute access like
    # `data.y`); composite expressions are refused so the user must
    # pass an explicit `tran=make_tran('genlog', base=1)` for log(y+c)
    # cases (R's `make.tran("genlog", c)` equivalent).
    # the ASCII-only `[a-zA-Z_]` regex
    # missed valid Python identifiers (Unicode names like `données`).
    # Use a permissive outer regex that captures any callable name +
    # any inner string, then validate the inner string structurally
    # via `str.isidentifier()` to accept Unicode identifiers + dotted
    # attribute access (`data.y`). Composite expressions still refused.
    m = re.match(
        r"^(?:np\.)?([^\W\d]\w*)\(\s*(.+?)\s*\)$",
        endog_name,
        flags=re.UNICODE,
    )
    if m is None:
        return None
    func = m.group(1).lower()
    inner = m.group(2)
    # Inner must be a simple identifier or dotted-attribute access:
    # each `.`-separated segment must be a Python identifier (no
    # operators, no nested calls, no literals).
    if not inner or not all(seg.isidentifier() for seg in inner.split(".")):
        return None
    # `arcsin(.)` alone (not composed with sqrt) is NOT one of the
    # proportion transforms — that would be the inverse sine of a raw
    # proportion, which is mathematically valid but uncommon and not
    # built-in. Refuse rather than misidentify.
    if func in ("arcsin", "asin"):
        return None
    return _BUILTIN.get(func)


def make_tran(
    name: str,
    inverse: Callable[[np.ndarray], np.ndarray] | None = None,
    inverse_deriv: Callable[[np.ndarray], np.ndarray] | None = None,
    *,
    bias_mean: Callable[[np.ndarray, float], np.ndarray] | None = None,
    bias_deriv: Callable[[np.ndarray, float], np.ndarray] | None = None,
    is_log: bool = False,
    lambda_: float | None = None,
    base: float | None = None,
    const: float | None = None,
    mean: float | None = None,
    sd: float | None = None,
    gamma: float | None = None,
    contrast_inverse: Callable[[np.ndarray], np.ndarray] | None = None,
    contrast_inverse_deriv: Callable[[np.ndarray], np.ndarray] | None = None,
) -> Transform:
    """Build a :class:`Transform` by name, mirroring R's ``make.tran()``.

    Three forms:

    1. **Named lookup** — ``make_tran("log")`` returns the same built-in
       used by ``detect_transform("np.log(y)")``. Supported names:
       ``log`` / ``log10`` / ``log2`` / ``log1p`` / ``sqrt`` / ``exp``
       / ``logit`` / ``probit`` / ``cloglog`` / ``asin_sqrt``.
    2. **Parametric** — ``make_tran("boxcox", lambda_=0.5)`` builds the
       Box-Cox transform at a specific power parameter. The forward
       transform is ``z = (y**λ - 1) / λ`` for ``λ != 0`` and
       ``log(y)`` at ``λ == 0``; ``make_tran`` returns the inverse and
       its derivative. Implementation uses ``functools.partial`` over
       module-level helpers so the resulting Transform stays picklable
       (closures would break multiprocessing bootstrap and joblib
       caching).
    3. **Custom build** — pass ``inverse`` and ``inverse_deriv`` to
       construct a one-off ``Transform`` for an LHS expression pymmeans
       doesn't auto-detect (e.g. a user-defined link). Optional
       ``bias_mean`` / ``bias_deriv`` enable
       ``regrid_response(..., bias_adjust=True)`` on the custom
       transform; ``is_log=True`` makes contrast back-transforms emit
       ``ratio`` columns instead of ``estimate``.

    Use the returned ``Transform`` via
    ``regrid_response(emm, tran=transform)`` when the LHS isn't a
    canonical built-in name.

    Parameters
    ----------
    name
        Identifier. For named lookup this must be one of the built-in
        keys; for a custom build it's a free-form label that shows up
        in error messages and downstream column names.
    inverse, inverse_deriv
        Required for custom builds; ignored when ``name`` matches a
        built-in (the built-in's callables win).
    bias_mean, bias_deriv, is_log
        Optional metadata for custom builds, see :class:`Transform`.
    lambda_
        Power parameter for Box-Cox. Required when ``name == "boxcox"``;
        ignored otherwise. ``lambda_=0`` collapses Box-Cox to the log
        transform.
    base
        Additive constant for ``"genlog"`` — forward ``log(y + base)``,
        inverse ``exp(z) - base``. Required when ``name == "genlog"``;
        commonly ``base=1`` for count data with zeros (equivalent to
        the R ``make.tran("genlog", 1)`` workflow).
    const
        Additive constant for ``"sqrt_const"`` — forward
        ``sqrt(y + const)``, inverse ``z² - const``. Required when
        ``name == "sqrt_const"``. Anscombe (1948) suggests
        ``const=0.375`` for Poisson-distributed counts to stabilise
        variance near zero better than plain ``sqrt(y)``.
    mean, sd
        Center and scale for ``"scale"`` — forward
        ``(y - mean) / sd``, inverse ``z * sd + mean``. Both required
        when ``name == "scale"``. Typical workflow: pass the
        training-data mean/SD to back-transform a fit done on
        standardised response values to the original units.

    Returns
    -------
    Transform
        Picklable namedtuple ready to pass as ``tran=`` to
        :func:`regrid_response` or to drop into a user-built
        :class:`~pymmeans.utils.ModelInfo`.

    Raises
    ------
    ValueError
        If ``name`` isn't a built-in and ``inverse`` / ``inverse_deriv``
        weren't supplied — there's nothing to back-transform with.
        Also if a parametric family is named without its required
        parameter (``lambda_`` for boxcox, ``base`` for genlog,
        ``const`` for sqrt_const).
    """
    import functools

    # every parametric family below
    # used to accept NaN / Inf without complaint, producing transforms
    # that silently return NaN on every call. Guard up-front via the
    # helper so the failure is at construction time, not downstream.
    def _require_finite_param(label: str, value: float) -> float:
        v = float(value)
        if not np.isfinite(v):
            raise ValueError(
                f"make_tran({name!r}): {label}={value!r} must be finite."
            )
        return v

    # R name aliases for the
    # transform family. These let users coming from R `make.tran()`
    # use R names (`asin.sqrt`, `log+1`, `sqrt+.5`, `identity`) and
    # get the equivalent pymmeans transform.
    _R_ALIAS_MAP: dict[str, tuple[str, dict[str, float]]] = {
        "asin.sqrt": ("asin_sqrt", {}),
        "log+1": ("genlog", {"base": 1.0}),
        "log+.5": ("genlog", {"base": 0.5}),
        "log+0.5": ("genlog", {"base": 0.5}),
        "sqrt+.5": ("sqrt_const", {"const": 0.5}),
        "sqrt+0.5": ("sqrt_const", {"const": 0.5}),
        "+.5": ("sqrt_const", {"const": 0.5}),
    }
    key = name.lower()
    if key in _R_ALIAS_MAP:
        canonical_name, defaults = _R_ALIAS_MAP[key]
        key = canonical_name
        # Fill in default parameter values from the alias map; explicit
        # kwargs win (matches the `emm_options` precedence).
        if "base" in defaults and base is None:
            base = defaults["base"]
        if "const" in defaults and const is None:
            const = defaults["const"]
    if key == "identity":
        # R `make.tran("identity")` — explicit no-op. Useful for
        # workflows that want to opt out of auto-detection.
        return Transform(
            name="identity",
            inverse=_identity_fn,
            inverse_deriv=_identity_deriv,
            bias_mean=None,
            is_log=False,
            bias_deriv=None,
        )
    if key == "boxcox":
        if lambda_ is None:
            raise ValueError(
                "make_tran('boxcox') requires a power parameter "
                "`lambda_=`. Pass `lambda_=0` for the log transform "
                "or `lambda_=0.5` for the square-root variant."
            )
        lam = _require_finite_param("lambda_", lambda_)
        inv_fn = functools.partial(_boxcox_inv, lam)
        d_fn = functools.partial(_boxcox_inv_deriv, lam)
        return Transform(
            name=f"boxcox(lambda={lam})",
            inverse=inv_fn,
            inverse_deriv=d_fn,
            bias_mean=None,
            is_log=(lam == 0.0),
            bias_deriv=None,
        )
    if key == "genlog":
        if base is None:
            raise ValueError(
                "make_tran('genlog') requires an additive constant "
                "`base=`. R parity: `base=1` for the common log(y+1) "
                "case (counts with zeros)."
            )
        b = _require_finite_param("base", base)
        inv_fn = functools.partial(_genlog_inv, b)
        d_fn = functools.partial(_genlog_inv_deriv, b)
        # genlog IS a log-family transform — back-transformed contrasts
        # become ratios on the (y + base) scale. Mark is_log=True so
        # `regrid_response` relabels the contrast column appropriately.
        #
        # the CONTRAST inverse for genlog
        # must be `exp` (NOT `exp(z) - base`), because:
        # EMM(a) = log(y_a + base), EMM(b) = log(y_b + base)
        # contrast = EMM(a) - EMM(b) = log((y_a + base)/(y_b + base))
        # exp(contrast) = (y_a + base)/(y_b + base) ← the right ratio
        # Without this hook, regrid_response applied `exp(z) - base`
        # to the contrast, producing nonsense `ratio = (y_a+b)/(y_b+b) - b`.
        return Transform(
            name=f"genlog(base={b})",
            inverse=inv_fn,
            inverse_deriv=d_fn,
            bias_mean=None,
            is_log=True,
            bias_deriv=None,
            contrast_inverse=np.exp,
            contrast_inverse_deriv=np.exp,
        )
    if key == "sqrt_const":
        if const is None:
            raise ValueError(
                "make_tran('sqrt_const') requires an additive constant "
                "`const=`. Anscombe (1948) suggests `const=0.375` for "
                "Poisson counts; `const=0.5` is another common choice."
            )
        c = _require_finite_param("const", const)
        inv_fn = functools.partial(_sqrt_const_inv, c)
        d_fn = functools.partial(_sqrt_const_inv_deriv, c)
        return Transform(
            name=f"sqrt_const(const={c})",
            inverse=inv_fn,
            inverse_deriv=d_fn,
            bias_mean=None,
            is_log=False,
            bias_deriv=None,
        )
    if key == "scale":
        if mean is None or sd is None:
            raise ValueError(
                "make_tran('scale') requires BOTH `mean=` and `sd=`. "
                "Typical workflow: pass the training-data mean and SD "
                "used to standardise the response before fitting."
            )
        mu = _require_finite_param("mean", mean)
        s = _require_finite_param("sd", sd)
        if not (s > 0):
            raise ValueError(
                f"make_tran('scale') requires `sd > 0`; got sd={s}."
            )
        inv_fn = functools.partial(_scale_inv, mu, s)
        d_fn = functools.partial(_scale_inv_deriv, mu, s)
        cinv = functools.partial(_scale_contrast_inv, s)
        cderiv = functools.partial(_scale_contrast_inv_deriv, s)
        # supply contrast hooks so
        # `regrid_response` on a scale-transformed contrast returns
        # `sd * link_diff` (the constant `mu` cancels in differences).
        # is_log stays False (no ratio interpretation), but the
        # regrid path now also accepts non-log transforms that
        # opt-in via `contrast_inverse`.
        return Transform(
            name=f"scale(mean={mu}, sd={s})",
            inverse=inv_fn,
            inverse_deriv=d_fn,
            bias_mean=None,
            is_log=False,
            bias_deriv=None,
            contrast_inverse=cinv,
            contrast_inverse_deriv=cderiv,
        )
    if key == "power":
        # R ``make.tran("power", lambda)``. Strict-power forward
        # ``z = y^lambda``; distinct from Box-Cox (no -1/lambda
        # affine adjustment). Domain: ``y > 0``.
        if lambda_ is None:
            raise ValueError(
                "make_tran('power') requires a power parameter "
                "``lambda_=``. Use ``make_tran('boxcox', "
                "lambda_=...)`` if you want the Box-Cox affine "
                "variant (smooth limit as lambda -> 0), or ``"
                "make_tran('log')`` for the lambda = 0 case."
            )
        lam = _require_finite_param("lambda_", lambda_)
        if lam == 0.0:
            raise ValueError(
                "make_tran('power', lambda_=0) is undefined "
                "(``y^0 = 1`` for every y, so the transform is "
                "non-invertible). Use ``make_tran('log')`` for the "
                "logarithmic limit instead."
            )
        inv_fn = functools.partial(_power_inv, lam)
        d_fn = functools.partial(_power_inv_deriv, lam)
        return Transform(
            name=f"power(lambda={lam})",
            inverse=inv_fn,
            inverse_deriv=d_fn,
            bias_mean=None,
            is_log=False,
            bias_deriv=None,
        )
    if key == "sympower":
        # R ``make.tran("sympower", lambda)``. Sign-preserving power
        # ``z = sign(y) * |y|^lambda``, defined on all real y. Useful
        # for detrended / mean-subtracted responses that can be
        # negative.
        if lambda_ is None:
            raise ValueError(
                "make_tran('sympower') requires a power parameter "
                "``lambda_=``. Common choices: lambda_=2 (squared "
                "with sign preserved), lambda_=0.5 (signed sqrt)."
            )
        lam = _require_finite_param("lambda_", lambda_)
        if lam == 0.0:
            raise ValueError(
                "make_tran('sympower', lambda_=0) is undefined "
                "(``sign(y) * |y|^0`` collapses to ``sign(y)`` "
                "which is non-invertible across zero). Use a small "
                "positive lambda (e.g. 0.5) or ``make_tran('log')`` "
                "after taking ``log(|y|+1)`` with ``genlog``."
            )
        inv_fn = functools.partial(_sympower_inv, lam)
        d_fn = functools.partial(_sympower_inv_deriv, lam)
        return Transform(
            name=f"sympower(lambda={lam})",
            inverse=inv_fn,
            inverse_deriv=d_fn,
            bias_mean=None,
            is_log=False,
            bias_deriv=None,
        )
    if key == "bcnpower":
        # R ``make.tran("bcnPower", lambda, gamma)`` — "Box-Cox with
        # Negatives" (Hawkins & Weisberg 2017). Equivalent to a
        # Box-Cox on the shifted response ``y + gamma``; the
        # derivative is identical to Box-Cox's (gamma drops out
        # under differentiation), so we reuse the Box-Cox
        # derivative helper directly via partial.
        if lambda_ is None or gamma is None:
            raise ValueError(
                "make_tran('bcnPower') requires BOTH a power "
                "``lambda_=`` and a shift ``gamma=``. Common "
                "workflow: choose gamma so all observations of "
                "y + gamma are strictly positive, then estimate "
                "lambda via profile likelihood."
            )
        lam = _require_finite_param("lambda_", lambda_)
        gam = _require_finite_param("gamma", gamma)
        inv_fn = functools.partial(_bcn_power_inv, lam, gam)
        d_fn = functools.partial(_boxcox_inv_deriv, lam)
        return Transform(
            name=f"bcnPower(lambda={lam}, gamma={gam})",
            inverse=inv_fn,
            inverse_deriv=d_fn,
            bias_mean=None,
            is_log=(lam == 0.0),
            bias_deriv=None,
        )
    if key in ("yj.power", "yj_power", "yjpower"):
        # R ``make.tran("yj.power", lambda)`` — Yeo & Johnson (2000)
        # power transform. Piecewise definition makes it well-
        # defined for negative ``y`` natively (unlike Box-Cox).
        if lambda_ is None:
            raise ValueError(
                "make_tran('yj.power') requires a power parameter "
                "``lambda_=``. Common defaults: lambda_=0 collapses "
                "to log(y+1) on the positive side; lambda_=2 to "
                "-log(-y+1) on the negative side. R `car::yjPower` "
                "uses lambda estimated via profile likelihood."
            )
        lam = _require_finite_param("lambda_", lambda_)
        inv_fn = functools.partial(_yj_power_inv, lam)
        d_fn = functools.partial(_yj_power_inv_deriv, lam)
        # is_log marks the family in :func:`regrid_response`'s
        # contrast back-transform; Yeo-Johnson at lambda=0 reduces
        # to log(y+1) on the y>=0 side but not on the y<0 side, so
        # the ratio interpretation only partially holds. Conservative
        # choice: is_log=False — users wanting log-family contrast
        # semantics should pass ``tran=make_tran('log1p')`` for the
        # positive-response special case.
        return Transform(
            name=f"yj.power(lambda={lam})",
            inverse=inv_fn,
            inverse_deriv=d_fn,
            bias_mean=None,
            is_log=False,
            bias_deriv=None,
        )
    if inverse is None and inverse_deriv is None:
        if key in _BUILTIN:
            return _BUILTIN[key]
        raise ValueError(
            f"make_tran('{name}') is not a built-in transform "
            f"(known: {sorted(_BUILTIN)}). To build a custom Transform "
            "pass `inverse` and `inverse_deriv` callables, or pass one "
            "of the parametric families: `lambda_=` (boxcox / power / "
            "sympower / yj.power), `lambda_=` + `gamma=` (bcnPower), "
            "`base=` (genlog), `const=` (sqrt_const), `mean=` + `sd=` "
            "(scale)."
        )
    if inverse is None or inverse_deriv is None:
        raise ValueError(
            "Custom make_tran() requires BOTH `inverse` and "
            "`inverse_deriv`; one of them was None."
        )
    # pass through the new
    # contrast hooks so user-built shifted-log transforms can opt in.
    return Transform(
        name=name,
        inverse=inverse,
        inverse_deriv=inverse_deriv,
        bias_mean=bias_mean,
        is_log=is_log,
        bias_deriv=bias_deriv,
        contrast_inverse=contrast_inverse,
        contrast_inverse_deriv=contrast_inverse_deriv,
    )


def _interval_inverse(
    tran: Transform, lo: np.ndarray, hi: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Map a (lo, hi) link-scale interval through ``tran.inverse``,
    respecting non-monotone inverses.

    the sqrt family's inverse is
    ``z²``, which is NON-monotonic across z=0. The previous
    ``min/max(inverse(lo), inverse(hi))`` reduction silently dropped
    zero — e.g. for link CI = [-0.2, +0.2] the response-scale CI
    should be [0, 0.04], but the old code produced [0.04, 0.04]
    (both endpoints square to the same value).

    For sqrt and sqrt_const(const), the response-scale minimum is
    ``inverse(0) = 0`` (or ``-const``) whenever the link CI crosses
    zero. Detect that and use it.

    For monotone inverses (log family, identity, etc.) the
    ``min/max`` reduction is correct and we use it.
    """
    a = tran.inverse(lo)
    b = tran.inverse(hi)
    lower = np.minimum(a, b)
    upper = np.maximum(a, b)
    # Box-Cox and exp-inverse-log
    # transforms have RESTRICTED domains where `tran.inverse` returns
    # NaN outside the valid region. A link CI that straddles the
    # boundary produces NaN endpoints; the response-scale interval
    # should clamp to the boundary value instead. For Box-Cox with
    # lambda>0: the boundary z = -1/lambda has inverse value 0.
    # For exp-LHS (inverse = log): the boundary z = 0 has log(0) = -inf.
    if tran.name.startswith("boxcox("):
        # Extract lambda from name "boxcox(lambda=0.5)"
        try:
            lam = float(tran.name.split("=")[1].rstrip(")"))
        except (IndexError, ValueError):
            lam = None
        if lam is not None and lam != 0:
            boundary = -1.0 / lam
            below = lo < boundary # entire interval below domain
            crosses = (lo < boundary) & (hi >= boundary)
            # Wholly below: response interval is just [0, 0] at the
            # boundary (limit). Crosses: lower endpoint replaced by 0,
            # upper endpoint is inverse(hi).
            if np.any(crosses | below):
                # Crosses: lower = 0 (boundary inverse), upper = inverse(hi)
                lower = np.where(crosses, 0.0, np.where(below, 0.0, lower))
                upper = np.where(below, 0.0, np.where(crosses, b, upper))
    elif tran.name == "exp":
        # exp's inverse is log; log(0) = -inf, log(<0) = NaN. A link
        # CI crossing zero on the response scale means the lower
        # response bound is -inf.
        crosses_zero = (lo <= 0) & (hi >= 0)
        if np.any(crosses_zero):
            lower = np.where(crosses_zero, -np.inf, lower)
    if tran.name == "sqrt" or tran.name.startswith("sqrt_const"):
        # Inverse is z² or z² - const, with global min at z=0.
        crosses_zero = (lo <= 0) & (hi >= 0)
        if np.any(crosses_zero):
            min_at_zero = tran.inverse(np.zeros_like(lo))
            lower = np.where(crosses_zero, min_at_zero, lower)
    elif tran.name == "reciprocal":
        # 1/z is non-monotone AND has a
        # singularity at z=0. A CI that crosses zero has an unbounded
        # response-scale interval — the previous min/max reduction
        # returned a finite (and wrong) interval. Refuse cleanly.
        crosses_zero = (lo <= 0) & (hi >= 0)
        if np.any(crosses_zero):
            raise ValueError(
                "reciprocal response-scale CI crosses the link-scale "
                "singularity at 0; the response interval is "
                "mathematically unbounded. Report link-scale results, "
                "or refit with a monotone link if a bounded response-"
                "scale CI is needed."
            )
    elif tran.name == "asin_sqrt":
        # inverse is sin(z)², with critical
        # points at z = n*pi/2 (minima at z=n*pi, maxima at z=n*pi/2 odd).
        # The sqrt fix didn't extend to asin_sqrt. Loop over each
        # row's link CI and include any critical points strictly inside.
        new_lower = lower.copy()
        new_upper = upper.copy()
        half_pi = np.pi / 2.0
        for idx in range(len(lo)):
            lo_i, hi_i = float(lo[idx]), float(hi[idx])
            if not (np.isfinite(lo_i) and np.isfinite(hi_i)):
                continue
            pts = [lo_i, hi_i]
            # Critical points: n * pi/2 for integer n strictly inside (lo, hi)
            n_lo = int(np.ceil(lo_i / half_pi))
            n_hi = int(np.floor(hi_i / half_pi))
            for n in range(n_lo, n_hi + 1):
                p = n * half_pi
                if lo_i <= p <= hi_i:
                    pts.append(p)
            vals = tran.inverse(np.asarray(pts))
            new_lower[idx] = float(np.nanmin(vals))
            new_upper[idx] = float(np.nanmax(vals))
        lower, upper = new_lower, new_upper
    return lower, upper


def regrid_response(
    emm_or_contrast: Any,
    tran: Transform | dict | None = None,
    bias_adjust: bool = False,
    force: bool = False,
    sigma: float | None = None,
) -> Any:
    """Re-express an EMM or contrast result on the response (untransformed) scale.

    For OLS models with an LHS transformation we apply the inverse to the
    point estimate and use the delta method for SE. CI bounds come from
    transforming the link-scale endpoints (matching R emmeans).

    The ``sigma=`` kwarg mirrors R's
    `summary(em, type="response", bias.adjust=TRUE, sigma=...)`. When
    provided, ``sigma**2`` is used as the residual variance for the
    bias correction in place of ``model_info.scale``. Useful when
    you have a more reliable variance estimate (e.g. a pooled
    cross-study SD) than the within-model residual.

    ``sigma`` may be a scalar OR a 1-D ndarray broadcastable against
    the EMM rows (one element per row, R parity). Negative or
    non-finite values trigger a UserWarning and skip the bias
    adjustment.

    Parameters
    ----------
    emm_or_contrast
        Either an ``EMMResult`` or ``ContrastResult``.
    tran
        Optional override. If ``None`` we try to detect from
        ``model_info.response_name``. Pass a ``Transform`` namedtuple or a
        dict ``{"inverse": fn, "inverse_deriv": fn, "is_log": bool}``.
    bias_adjust
        If True, apply the transform's ``bias_mean(mu, sigma_sq)``
        second-order Taylor bias correction so the response-scale value
        is the mean rather than the median. For the log family the
        formula is ``exp(mu) * (1 + sigma_sq/2)`` (R `emmeans` parity,
        #5); for sqrt it is ``mu**2 + sigma_sq``. Requires
        the transform to expose ``bias_mean``; the built-in ``exp``
        inverse does not and raises if ``bias_adjust=True``.
    force
        If ``True``, bypass the guard that refuses to back-transform an
        already-response-scale EMMResult / ContrastResult. The guard
        exists because applying ``inverse`` twice on a log-LHS model
        produces ``exp(exp(...))``-style nonsense; ``force=True`` is the
        opt-in escape hatch for advanced use.
    """
    info = emm_or_contrast.model_info
    # #1: refuse posterior EMMs. `regrid_response` applies
    # the inverse transform to the SUMMARY (mean of the posterior), not
    # to each draw — that's Jensen-wrong by 6-7% on log-LHS posteriors.
    # The correct workflow is `posterior_emmeans(..., type="response")`,
    # which transforms each draw before percentiles.
    if getattr(emm_or_contrast, "inference_kind", "wald") == "posterior":
        raise ValueError(
            "regrid_response is not defined for posterior-derived "
            "results: it would apply the inverse transform to the "
            "posterior mean rather than to each draw, giving a "
            "Jensen-biased point estimate (E[exp(L beta)] != "
            "exp(E[L beta]) for log-family transforms). Use "
            "`posterior_emmeans(pinfo, ..., type='response')` instead, "
            "which transforms each draw before summarising."
        )
    # Guard against double back-transformation: applying inverse twice
    # silently produces exp(exp(...))-style damage. R's regrid() flips
    # an internal grid-state flag; we mirror that via EMMResult.type.
    if (
        getattr(emm_or_contrast, "type", "link") == "response"
        and not force
    ):
        raise ValueError(
            "regrid_response: input is already on the response scale "
            "(type='response'). Pass force=True to re-transform anyway."
        )
    if tran is None:
        if info.response_name is None:
            raise ValueError(
                "Cannot detect LHS transformation: response_name is missing."
            )
        # Binomial GLMs fit with a 2-column LHS
        # (``(y > c) ~ x`` or ``cbind(succ, fail) ~ x``) expose
        # ``endog_names`` as a *list* (e.g. ``['y > 1.0[False]', 'y >
        # 1.0[True]']``). ``detect_transform`` is regex-based and
        # crashed with ``TypeError: expected string or bytes-like
        # object, got 'list'``. For these LHS shapes there is no
        # parseable inverse transform on the response name; we fall
        # through to the link-based path below (which the binomial
        # GLM does have via the logit / probit / cloglog link).
        if isinstance(info.response_name, list):
            tran = None
        else:
            tran = detect_transform(info.response_name)
        # for GLM families (Poisson, Binomial,
        # NegativeBinomial, Gamma, ...) the LHS variable is the raw
        # response, not the linear predictor — so detect_transform on
        # ``response_name`` returns ``None``. Build the transform from
        # the family's *link* instead so `regrid_response(pairs(em))`
        # on a NB / Poisson model produces ratios (matching R
        # `pairs(emmeans(fit, ~g, type="response"))` workflow).
        if tran is None and info.family is not None:
            link_name = type(info.family.link).__name__.lower()
            link_to_transform = {
                "log": "log",
                "logit": "logit",
                "probit": "probit",
                "cloglog": "cloglog",
                "loglog": None, # not implemented in pymmeans
                "identity": "identity",
                "inverse": None, # 1/mu — not currently supported
                "power": None,
            }
            tran_name = link_to_transform.get(link_name)
            if tran_name is not None:
                # ``detect_transform`` is keyed off the response name's
                # function-call shape (e.g. ``np.log(y)``), so it can't
                # synthesize an identity transform from a plain
                # identifier. Use the explicit ``make_tran("identity")``
                # builtin instead so identity-link GLMs round-trip
                # through ``regrid_response`` as a no-op (matches R
                # `emmeans::regrid(type="response")` on Gaussian-identity).
                tran = (
                    detect_transform(f"np.{tran_name}(y)")
                    if tran_name != "identity"
                    else make_tran("identity")
                )
        if tran is None:
            raise ValueError(
                f"No transform recognised from response "
                f"{info.response_name!r}. Composite inner expressions "
                "like `log(y + 1)` or `sqrt(y * 2)` are intentionally "
                "not auto-detected (the silent-wrong inverse "
                "would be off by exactly the constant). Pass an "
                "explicit `tran=` — e.g. "
                "`tran=make_tran('genlog', base=1)` for `log(y + 1)`, "
                "`tran=make_tran('sqrt_const', const=0.375)` for "
                "`sqrt(y + 0.375)`, or build a custom Transform via "
                "`make_tran(name, inverse=, inverse_deriv=)`."
            )
    if isinstance(tran, dict):
        # the module docstring (and an
        # earlier round's doc example) advertised the dict-form
        # transform with `{"inverse": ..., "deriv": ...}`, but the
        # constructor demanded `"inverse_deriv"`. Accept both spellings
        # so the documented example actually works.
        deriv = tran.get("inverse_deriv", tran.get("deriv"))
        if "inverse" not in tran:
            raise ValueError(
                "tran dict requires 'inverse' (and 'inverse_deriv' or "
                "the alias 'deriv')."
            )
        if deriv is None:
            raise ValueError(
                "tran dict requires 'inverse_deriv' (or its alias "
                "'deriv') alongside 'inverse'."
            )
        # the `contrast_inverse`
        # / `contrast_inverse_deriv` fields were added to `Transform`
        # but the dict-form constructor was never updated — so users
        # passing a custom shifted-log via dict got the wrong contrast
        # back-transform (silently). Pass them through.
        tran = Transform(
            tran.get("name", "custom"),
            tran["inverse"],
            deriv,
            tran.get("bias_mean"),
            tran.get("is_log", False),
            tran.get("bias_deriv"),
            tran.get("contrast_inverse"),
            tran.get("contrast_inverse_deriv"),
        )

    frame = emm_or_contrast.frame.copy()
    # ``sigma=`` kwarg overrides the model's residual SD.
    # R warns + skips bias adjustment on negative or non-finite
    # sigma; a `float(sigma)**2` would silently square the bad value.
    # Also accept a 1-D vector sigma broadcast per-row (R
    # `summary(em, sigma=c(.1,.2,.3))`).
    if sigma is None:
        # Refuse default-sigma bias adjustment on GLMs with a non-identity
        # link: ``info.scale`` is the residual dispersion on the response
        # scale (e.g. 1.0 for Poisson, fit.scale for Gaussian(log)), NOT
        # the variance of the linear predictor that the Jensen correction
        # expects. Using it silently inflates the response-scale EMM by a
        # family-dependent factor (1.5× for canonical Poisson, ~3× for
        # Gaussian(log) with MSE ~ 4 on the response scale). R `emmeans`
        # enforces the same constraint (`summary(..., bias.adjust=TRUE)`
        # requires an explicit ``sigma=`` on non-OLS).
        #
        # Allowed paths without explicit sigma:
        # - OLS with an LHS transform (lm(log(y) ~ ...)): ``info.family``
        #   is ``None`` and ``info.scale`` is the residual variance on the
        #   transformed response, which is the correct Jensen sigma².
        # - GLM with identity link: bias adjustment is identically 1; the
        #   value of sigma² is irrelevant.
        if bias_adjust and info.family is not None:
            link_name = type(getattr(info.family, "link", object())).__name__
            if link_name != "Identity":
                raise ValueError(
                    f"bias_adjust=True on a GLM with non-identity link "
                    f"({link_name!r}) requires an explicit sigma= argument. "
                    f"The model's residual dispersion (fit.scale="
                    f"{info.scale!r}) is the variance of the response, "
                    f"NOT the variance of the linear predictor, so using "
                    f"it as the Jensen-correction sigma² silently inflates "
                    f"the response-scale EMM by a family-dependent factor. "
                    f"R `emmeans::summary(..., bias.adjust=TRUE)` enforces "
                    f"the same constraint. Pass sigma=<residual SD on the "
                    f"link scale> (a scalar or length-n_rows vector) — for "
                    f"a per-row choice use sigma=numpy.sqrt(numpy.diag("
                    f"L @ V_beta @ L.T)) computed from emm.linfct, or set "
                    f"bias_adjust=False to skip the correction."
                )
        sigma_sq = info.scale
    else:
        sig = np.asarray(sigma, dtype=float)
        if sig.ndim > 1:
            raise ValueError(
                "sigma must be a scalar or 1-D array, got "
                f"ndim={sig.ndim}."
            )
        if not np.all(np.isfinite(sig)) or np.any(sig < 0):
            import warnings as _w
            _w.warn(
                "Bias adjustment skipped: No valid 'sigma' provided",
                UserWarning,
                stacklevel=2,
            )
            bias_adjust = False
            sigma_sq = info.scale
        else:
            sigma_sq = sig * sig
            # If a vector was supplied, broadcast it to match the
            # number of rows in the frame. R applies per-row sigma
            # values directly to `bias_mean(mu_i, sigma_i**2)`
            # do the same by feeding a length-n array down the
            # transform machinery (numpy broadcasting takes care
            # of the rest in `bias_mean(mu, sigma_sq)`).
            if np.ndim(sigma_sq) == 1:
                n_rows = len(frame)
                if sigma_sq.shape[0] not in (1, n_rows):
                    raise ValueError(
                        "vector sigma must have length 1 or "
                        f"{n_rows} (the number of EMM rows); got "
                        f"length {sigma_sq.shape[0]}."
                    )
                if sigma_sq.shape[0] == 1:
                    sigma_sq = float(sigma_sq[0])

    from pymmeans.utils import detect_value_column

    kind_info = detect_value_column(frame)
    if kind_info is None:
        raise ValueError(
            "regrid_response: input frame has no recognisable value "
            "column (expected one of 'emmean', 'estimate', 'ratio', or "
            "a '<var>.trend' column)."
        )
    kind, _value_col = kind_info
    if kind == "trend":
        # emtrends results are derivatives, not the back-transformable
        # quantity. regrid_response on a trend
        # result silently returning the unchanged link-scale slopes with
        # type='response' — wrong on inspection. Refuse cleanly.
        raise NotImplementedError(
            "regrid_response is not defined for emtrends results because "
            "the values are slopes (derivatives), not back-transformable "
            "means. Apply regrid_response to the EMM grid first, then "
            "compute trends on the response-scale EMMResult."
        )
    if kind == "ratio":
        # the previous message said
        # "Pass force=True to re-apply" but ratio re-application is
        # mathematically nonsense (exp(exp(.))), so the force escape
        # hatch was never honoured here. Remove the misleading sentence.
        raise ValueError(
            "regrid_response: input is already on the response scale "
            "(ratio column present from a prior log-family back-transform). "
            "Re-applying the inverse to a ratio is mathematically "
            "undefined; refuse absolutely."
        )

    if kind == "emm":
        mu_link = frame["emmean"].to_numpy()
        se_link = frame["SE"].to_numpy()
        if bias_adjust:
            # refuse ``bias_adjust=True`` on a
            # bootstrap-derived EMM. The Taylor bias correction
            # ``E[g(X)] ≈ g(E[X]) + g''(E[X]) * σ²/2`` must be
            # applied to each draw before percentiling (the
            # fix for case bootstrap does exactly that),
            # but a bootstrap-derived result only stores the
            # precomputed percentile bounds — there are no draws
            # left to bias-correct. Previously the path silently
            # computed bias-adjusted symmetric Wald bounds around
            # the bias-adjusted mean, which numerically disagreed
            # with the correct draw-level workflow by ~5% on the
            # the bug reproduction. The correct composition is
            # ``bootstrap_ci(regrid_response(em, bias_adjust=True))``
            # — bias-adjust first, then bootstrap.
            if getattr(emm_or_contrast, "df_method", "default") == "bootstrap":
                raise ValueError(
                    "regrid_response(..., bias_adjust=True) cannot be "
                    "applied to a bootstrap-derived result because the "
                    "individual draws needed for the per-draw bias "
                    "correction are not stored — only the precomputed "
                    "percentile bounds. Rebuild via "
                    "``bootstrap_ci(regrid_response(raw_em, "
                    "bias_adjust=True), ...)``, which bias-corrects each "
                    "draw before percentiling."
                )
            if tran.bias_mean is None:
                raise ValueError(
                    f"bias_adjust=True is not defined for transform {tran.name!r}."
                )
            mu_resp = tran.bias_mean(mu_link, sigma_sq)
            # R `emmeans` evaluates the delta-method gradient at the
            # bias-corrected point, not the link-scale point — that's
            # `bias_deriv`. #6 caught us using the unadjusted
            # gradient, which gave SEs that were ~3% too small on log
            # links. Fall back to inverse_deriv only when a transform
            # doesn't supply a bias_deriv (custom user transforms).
            grad = (
                tran.bias_deriv(mu_link, sigma_sq)
                if tran.bias_deriv is not None
                else tran.inverse_deriv(mu_link)
            )
            se_resp = np.abs(grad) * se_link
            # R `regrid(em, bias.adjust=TRUE)` builds the response-scale CI as
            # ``response ± crit * SE_response`` (standard Wald around
            # the bias-corrected mean), NOT as ``bias_mean`` of the
            # link-scale endpoints. Pre-pymmeans applied the
            # transform to the asymmetric link endpoints, which both
            # mislocated the CI center (away from the bias-adjusted
            # mean) and gave slightly wider widths. Symmetric Wald
            # matches R exactly for log / sqrt / logit / probit / etc.
            #
            # The link-scale half-width is ``crit_link * se_link``;
            # we recover ``crit`` directly from the existing CI to
            # carry through arbitrary critical values (sidak/tukey/
            # scheffe widening), one-sided side=, and Satterthwaite df.
            lo_link = frame["lower_cl"].to_numpy()
            hi_link = frame["upper_cl"].to_numpy()
            with np.errstate(divide="ignore", invalid="ignore"):
                # Use the AVERAGE half-width on the link scale to be
                # robust to one-sided endpoints already at ±Inf.
                half_link = (hi_link - lo_link) / 2.0
                crit = np.where(
                    np.isfinite(half_link) & (se_link > 0),
                    half_link / se_link,
                    stats.t.ppf(0.975, frame["df"].to_numpy(dtype=float))
                    if "df" in frame.columns
                    else 1.96,
                )
            lo = mu_resp - crit * se_resp
            hi = mu_resp + crit * se_resp
            # Preserve one-sided endpoints (±Inf): the link-scale
            # frame may already encode them (caller passed side=).
            if np.any(~np.isfinite(lo_link)):
                lo = np.where(np.isfinite(lo_link), lo, -np.inf)
            if np.any(~np.isfinite(hi_link)):
                hi = np.where(np.isfinite(hi_link), hi, np.inf)
            # Keep the non-monotone sqrt fallback for CIs crossing
            # zero — when the link-scale CI straddles 0 the symmetric
            # Wald above can still produce lo < sigma²; R uses
            # ``lo = sigma²`` (the minimum of the parabola).
            if tran.name == "sqrt" or tran.name.startswith("sqrt_const"):
                crosses_zero = (lo_link <= 0) & (hi_link >= 0)
                if np.any(crosses_zero):
                    min_at_zero = tran.bias_mean(np.zeros_like(lo_link), sigma_sq)
                    lo = np.where(crosses_zero, min_at_zero, lo)
        else:
            mu_resp = tran.inverse(mu_link)
            se_resp = np.abs(tran.inverse_deriv(mu_link)) * se_link
            # use _interval_inverse so non-monotone
            # inverses (sqrt, sqrt_const) handle CI-crossing-zero
            # correctly. For monotone inverses this is a no-op.
            lo, hi = _interval_inverse(
                tran,
                frame["lower_cl"].to_numpy(),
                frame["upper_cl"].to_numpy(),
            )
        frame["emmean"] = mu_resp
        frame["SE"] = se_resp
        frame["lower_cl"] = lo
        frame["upper_cl"] = hi
    elif kind == "contrast":
        # R `emmeans` actually
        # supports `bias.adjust=TRUE` for log-family contrasts —
        # despite the "ratio not mean" objection, R adds the same
        # `+ sigma^2 / 2` shift to the link contrast before
        # exponentiating, giving E[Y_a/Y_b] under the standard
        # log-normal assumption (since both numerator and denominator
        # carry the same correction in the lognormal expectation, the
        # shift survives in the ratio). The refusal was
        # overly cautious; we now implement the bias-adjusted log
        # contrast and refuse only for non-log transforms (where the
        # second-order correction genuinely has no meaning).
        if bias_adjust:
            if not tran.is_log:
                # raise the structured sentinel subclass
                # so ``summary_layer._safe_recompute`` can isinstance-
                # check instead of string-matching the message. The
                # subclass IS a ValueError, so any caller that
                # ``except ValueError`` still catches it.
                raise NonLogContrastBiasAdjustError(
                    "regrid_response(contrast, bias_adjust=True) is "
                    "only defined for log-family transforms (where "
                    f"E[Y] = exp(eta + sigma^2/2)). Got {tran.name!r}; "
                    "use `bootstrap_ci` on the regridded EMM for "
                    "response-scale uncertainty under non-log "
                    "transforms."
                )
        # Back-transforming a contrast only makes sense for the log family
        # because `exp(A - B) = exp(A) / exp(B)` is an interpretable ratio
        # of means. For sqrt / other non-log transforms, `inverse(A - B)`
        # is NOT the response-scale difference `inverse(A) - inverse(B)`
        # (e.g. sqrt: `(A-B)^2 != A^2 - B^2`), so the column would carry
        # mathematically meaningless numbers labelled "estimate". R's
        # workflow for the non-log case is `pairs(regrid(em))` — compute
        # the contrast AFTER regridding the EMMs — which we refuse via
        # the #7 guard because the linear contrast of
        # back-transformed means is also not the back-transform of the
        # contrast. Net: refuse cleanly and tell the user what to use
        # instead. Self-between rounds 6 and 7 caught this.
        # also allow non-log transforms
        # that supply an explicit `contrast_inverse` hook (e.g. `scale`,
        # which is LINEAR — contrast back-transform is `sd * diff`,
        # the mean cancels). For genuinely ill-defined cases (sqrt,
        # exp) the Transform has no contrast_inverse so we still
        # refuse cleanly.
        if not tran.is_log and tran.contrast_inverse is None:
            raise NotImplementedError(
                f"regrid_response on a contrast is only defined for "
                f"log-family transforms or transforms with an explicit "
                f"`contrast_inverse` hook (got {tran.name!r} with "
                "neither). For non-log transforms without a linear "
                "structure (sqrt, exp), the back-transform of the "
                "contrast is not a meaningful quantity. Use "
                "`bootstrap_ci` on the regridded EMM for response-scale "
                "uncertainty, or compute the desired quantity manually."
            )
        est_link = frame["estimate"].to_numpy()
        se_link = frame["SE"].to_numpy()
        # use `contrast_inverse` when the
        # Transform supplies one. For plain log family this is the same
        # as `inverse` (both are `exp`), but for SHIFTED logs (log1p,
        # genlog) the EMM-row inverse is `exp(z) - c` while the
        # CONTRAST inverse must be plain `exp` to give the ratio
        # `(y_a + c)/(y_b + c)`. Without this, contrasts on log1p /
        # genlog were off by exactly the additive constant.
        cinv = tran.contrast_inverse if tran.contrast_inverse is not None else tran.inverse
        cderiv = (
            tran.contrast_inverse_deriv
            if tran.contrast_inverse_deriv is not None
            else tran.inverse_deriv
        )
        # b: bias-adjusted log contrasts use R's
        # second-order Taylor correction
        # `est_adj = cinv(diff) + 0.5 * sigma^2 * cinv''(diff)`
        # NOT the exact lognormal `exp(diff + sigma^2/2)`. R's
        # `emmeans::.adj.fns` performs:
        #
        # link$linkinv(eta) + 0.5 * sigma^2 * inv_second_deriv(eta)
        #
        # For log family the second derivative of `exp` is `exp`, so
        # the Taylor formula gives `exp(diff) * (1 + sigma^2/2)` — a
        # constant multiplicative factor of `(1 + sigma^2/2)` on the
        # ratio. The implementation used `exp(diff + s2/2)`,
        # which agrees to first-order but differs by ~0.014% for
        # sigma=0.18 (and grows with sigma).
        # pinned the R behaviour via a 3-group fit:
        # bias.adjust=FALSE ratio A/B = 0.9079588888
        # bias.adjust=TRUE ratio A/B = 0.9227969910
        # R multiplier = 1.0163422622 = 1 + sigma^2/2 ✓
        #
        # For generic contrast_inverse hooks we use a central
        # finite-difference estimate of the second derivative
        # (`cinv''(z) ≈ (cinv(z+h) - 2*cinv(z) + cinv(z-h))/h^2`).
        # For log family the analytic shortcut `cinv(z) * (1+s2/2)`
        # avoids the noise.
        if bias_adjust:
            # accept vector ``sigma_sq`` for the
            # per-contrast bias correction so callers can pass
            # ``regrid_response(ct, bias_adjust=True, sigma=np.array(
            # [...]))`` matching R's
            # ``summary(pairs(em), type="response", bias.adjust=TRUE,
            # sigma=c(...))``. Previously the contrast branch coerced
            # ``sigma_sq`` to a Python float, which raised
            # ``TypeError: only 0-dimensional arrays can be converted
            # to Python scalars`` on array input.
            half_s2 = 0.5 * np.asarray(sigma_sq, dtype=float)
            if half_s2.ndim > 0:
                # Validate length matches contrast count, otherwise the
                # broadcast against ``est_link`` would silently
                # mis-pair sigma_i with contrast j.
                if half_s2.shape != (len(est_link),):
                    raise ValueError(
                        f"sigma= length ({half_s2.shape[0]}) must match "
                        f"the contrast count ({len(est_link)}) when "
                        "passing an array."
                    )
            if tran.is_log:
                # Analytic shortcut: bias-adjusted ratio is the
                # un-adjusted ratio scaled by (1 + sigma^2/2).
                factor = 1.0 + half_s2
                est_resp = cinv(est_link) * factor
                se_resp = factor * np.abs(cderiv(est_link)) * se_link
            else:
                # Generic Taylor via central finite differences.
                # Only reached when tran.contrast_inverse is set;
                # enabled this for `scale`.
                h = 5e-4
                second = (cinv(est_link + h) - 2.0 * cinv(est_link)
                          + cinv(est_link - h)) / (h * h)
                est_resp = cinv(est_link) + half_s2 * second
                # Derivative of `est_adj` w.r.t. eta:
                # cderiv(eta) + 0.5 * sigma^2 * cinv'''(eta)
                # Approximate cinv''' via central diff of cderiv.
                third = (cderiv(est_link + h) - 2.0 * cderiv(est_link)
                         + cderiv(est_link - h)) / (h * h)
                se_resp = np.abs(cderiv(est_link) + half_s2 * third) * se_link
        else:
            est_resp = cinv(est_link)
            se_resp = np.abs(cderiv(est_link)) * se_link
        # the contrast branch
        # historically didn't transform `lower_cl` / `upper_cl`,
        # leaving them at link scale on a response-scale frame. R
        # transforms them via `contrast_inverse` (for log families
        # that's `exp`; for `scale` it's a linear rescale). Do the
        # same so a log contrast yields a proper ratio CI like
        # `[exp(L), exp(U)]`.
        if "lower_cl" in frame.columns and "upper_cl" in frame.columns:
            lo_link_ci = frame["lower_cl"].to_numpy(dtype=float)
            hi_link_ci = frame["upper_cl"].to_numpy(dtype=float)
            # b: R applies the same Taylor correction to
            # the CI endpoints — `cinv(lo) + 0.5 * sigma^2 * cinv''(lo)`
            # — so the ratio CI bounds inherit the `(1 + sigma^2/2)`
            # factor on log family.
            if bias_adjust:
                # cont.: vector sigma also flows
                # into the CI back-transform.
                half_s2 = 0.5 * np.asarray(sigma_sq, dtype=float)
                if tran.is_log:
                    factor = 1.0 + half_s2
                    lo_resp_ci = cinv(lo_link_ci) * factor
                    hi_resp_ci = cinv(hi_link_ci) * factor
                else:
                    h = 5e-4
                    second_lo = (cinv(lo_link_ci + h) - 2.0 * cinv(lo_link_ci)
                                 + cinv(lo_link_ci - h)) / (h * h)
                    second_hi = (cinv(hi_link_ci + h) - 2.0 * cinv(hi_link_ci)
                                 + cinv(hi_link_ci - h)) / (h * h)
                    lo_resp_ci = cinv(lo_link_ci) + half_s2 * second_lo
                    hi_resp_ci = cinv(hi_link_ci) + half_s2 * second_hi
            else:
                lo_resp_ci = cinv(lo_link_ci)
                hi_resp_ci = cinv(hi_link_ci)
            # `contrast_inverse` is monotone for all R-supported
            # transforms (log -> exp; scale -> linear); guard via
            # min/max just in case a user supplies an exotic
            # contrast_inverse that flips order.
            frame["lower_cl"] = np.minimum(lo_resp_ci, hi_resp_ci)
            frame["upper_cl"] = np.maximum(lo_resp_ci, hi_resp_ci)
        if tran.is_log and "contrast" in frame.columns:
            # Log-family back-transforms turn DIFFERENCES into RATIOS.
            # Always rename `estimate` -> `ratio` so downstream tooling
            # sees the correct interpretation. We optionally also rewrite
            # "a - b" -> "a / b" labels, but ONLY when it's safe: a naive
            # string replace would corrupt labels for valid factor levels
            # that themselves contain " - " .
            levels_unsafe = any(
                isinstance(lv, str) and (" - " in lv or " / " in lv)
                for fs in info.factors.values()
                for lv in fs
            )
            frame = frame.rename(columns={"estimate": "ratio"})
            frame["ratio"] = est_resp
            if not levels_unsafe:
                frame["contrast"] = (
                    frame["contrast"]
                    .astype(str)
                    .str.replace(" - ", " / ", regex=False)
                )
        else:
            frame["estimate"] = est_resp
        frame["SE"] = se_resp

    cls = type(emm_or_contrast)
    fields = {
        f: getattr(emm_or_contrast, f) for f in emm_or_contrast.__dataclass_fields__
    }
    fields["frame"] = frame
    if "type" in fields:
        fields["type"] = "response"
    if "bias_adjust" in fields:
        # mark bias_adjust=True when the Taylor correction
        # was actually applied. extended this to the
        # contrast branch (log-family only); EMM branch is unchanged.
        # Other kinds (trend, ratio) still refuse bias_adjust up-front.
        fields["bias_adjust"] = bool(bias_adjust and kind in ("emm", "contrast"))
    if "bias_sigma" in fields:
        # persist the user-supplied ``sigma=``
        # override so later ``summary(em, level=...)`` /
        # case-bootstrap recomputes don't silently revert to
        # ``info.scale``. ``None`` (the default) means "use
        # info.scale" — same semantics as before , just
        # explicitly stored. Only set when bias_adjust=True actually
        # fired; otherwise the field is meaningless and stays None.
        fields["bias_sigma"] = (
            sigma if (bias_adjust and kind in ("emm", "contrast")) else None
        )
    return cls(**fields)


# ---------------------------------------------------------------------------
# regrid (80%-parity push)
# ---------------------------------------------------------------------------

# Aliases for ``regrid(transform=...)`` mirroring R ``emmeans::regrid``.
# All three of "response" / "mu" / "unlink" mean "back-transform to the
# untransformed response scale". R also accepts "log", "log10", etc., to
# *re-transform* the response-scale EMMs onto a different link scale —
# we mark those as not-yet-implemented and route users to the manual
# workflow.
_REGRID_RESPONSE_ALIASES = {"response", "mu", "unlink"}


def regrid(
    emm_or_contrast: Any,
    transform: str | None = "response",
    bias_adjust: bool = False,
    force: bool = False,
) -> Any:
    """R-style ``regrid(object, transform=...)`` wrapper.

    Mirrors R ``emmeans::regrid``:

    - ``transform="response"`` (or ``"mu"`` / ``"unlink"``, all R
      aliases): back-transform to the response scale. Equivalent to
      :func:`regrid_response`. ``bias_adjust`` and ``force`` are
      forwarded.
    - ``transform="pass"`` / ``"none"`` / ``None``: no-op. Returns
      the input unchanged. The ``None`` value matches R's
      ``transform = NULL`` idiom; the string forms match R's
      documented aliases.
    - ``transform="log" | "log10" | "log2" | "log1p" | "sqrt" |
      "exp" | "logit" | "probit" | "cloglog" | "asin_sqrt" |
      "arctanh" / "atanh" | "reciprocal" | "identity"``:
      re-express the EMMs on a different scale by applying the
      named transform's **forward** to the response-scale
      predictions, with SEs propagated via the delta method
      (chain rule using the transform's existing inverse
      derivative: ``forward'(y) = 1 / inverse'(forward(y))``). CI
      endpoints are likewise transformed (matching R's regrid
      convention; symmetric Wald is rebuilt on the new scale
      downstream via :func:`summary`). When the input is on the
      link scale, it is first back-transformed to response via
      the model's own inverse-link before the new forward is
      applied.

    Parametric families (boxcox / genlog / sqrt_const / scale /
    power / sympower / bcnPower / yj.power) and user-built
    Transform instances are not yet supported as the ``transform=``
    target here — the v0.1 forward registry covers a finite set of
    named scales. Refused with a clear pointer.

    Parameters
    ----------
    emm_or_contrast
        An ``EMMResult`` or ``ContrastResult``.
    transform
        See above. Default ``"response"``. ``None`` is accepted as
        an alias for ``"pass"`` (R's NULL convention).
    bias_adjust, force
        Passed through to :func:`regrid_response` when applicable.
        ``bias_adjust`` is honoured only on the response-back-
        transformation step (R's behaviour), not when applying a
        subsequent forward.

    Returns
    -------
    Same type as the input. The returned EMM's ``type`` is
    stamped ``"response"`` (R's convention: anything that is no
    longer on the model's native link is treated as a response-
    scale display).
    """
    # R idiomatically uses ``transform = NULL`` to mean "leave the
    # scale alone". Accept Python ``None`` as the equivalent so users
    # porting R code don't hit a NotImplementedError on a documented
    # no-op.
    if transform is None:
        return emm_or_contrast
    t = str(transform).lower()
    if t in _REGRID_RESPONSE_ALIASES:
        return regrid_response(
            emm_or_contrast, bias_adjust=bias_adjust, force=force,
        )
    if t in ("pass", "none"):
        return emm_or_contrast
    if t in _FORWARD:
        return _regrid_to_named_scale(
            emm_or_contrast, t, bias_adjust=bias_adjust, force=force,
        )
    # The name is registered as a back-transform (via
    # register_transform) but no forward direction was supplied. Give
    # the user a specific actionable message rather than the generic
    # NotImplementedError — the user did everything right except
    # declare the forward direction.
    if t in TRANSFORMS:
        raise NotImplementedError(
            f"regrid(transform={transform!r}): the transform is "
            "registered in pymmeans.TRANSFORMS (for back-transform / "
            "detect_transform lookup) but no FORWARD direction was "
            "supplied — ``regrid(em, transform=name)`` needs to apply "
            "the transform to response-scale predictions, which "
            "requires the forward callable plus its derivative.\n\n"
            "Fix: re-register with both directions:\n\n"
            "  from pymmeans import register_transform, Transform\n"
            "  register_transform(\n"
            f"      {transform!r},\n"
            f"      TRANSFORMS[{transform!r}],   # existing back-transform\n"
            "      forward=...,                 # response -> new scale\n"
            "      forward_deriv=...,           # d(forward)/dx\n"
            "      overwrite=True,\n"
            "  )"
        )
    # Anything else (parametric families, bare callable): explicit
    # not-yet-implemented to avoid silently returning an EMM with the
    # wrong .type tag.
    raise NotImplementedError(
        f"regrid(transform={transform!r}) is not currently implemented. "
        f"Supported aliases:\n"
        f"  - 'response' / 'mu' / 'unlink' — back-transform to "
        f"response scale (calls regrid_response).\n"
        f"  - 'pass' / 'none' / None — no-op.\n"
        f"  - named forwards: {sorted(_FORWARD)} — apply that "
        f"transform's forward to the response-scale predictions.\n"
        f"Parametric families (boxcox / genlog / power / yj.power "
        f"/ ...) and user-built Transform instances as the target "
        f"are deferred to 0.2.0. Workaround: regrid to response "
        f"first, then build the forward output manually via "
        f"``Transform.inverse`` evaluated on the response-scale "
        f"predictions, with a delta-method SE update."
    )


def _regrid_to_named_scale(
    emm_or_contrast: Any,
    target_name: str,
    *,
    bias_adjust: bool,
    force: bool,
) -> Any:
    """Apply the named transform's forward to a (possibly link-scale)
    EMM, returning an EMM on the new scale.

    Implementation strategy: reuse :func:`regrid_response` by passing
    a synthetic ``Transform`` whose ``inverse`` field carries our
    NEW transform's *forward* and whose ``inverse_deriv`` carries
    the *forward* derivative. From regrid_response's point of view
    it is "applying ``inverse`` to a link-scale value to produce a
    response-scale value" — the math is identical regardless of
    which direction the user thinks of as "link" vs "response".
    """
    if target_name not in _FORWARD:
        # Should never reach here — caller checks; defensive.
        raise NotImplementedError(
            f"_regrid_to_named_scale: {target_name!r} is not in the "
            f"forward registry."
        )
    fwd, fwd_deriv = _FORWARD[target_name]
    # If the input is on the link scale, normalise to response first
    # so the new forward is applied to the predicted response values
    # (matching R `regrid(em, transform="log")` semantics).
    #
    # Only call ``regrid_response`` when the model genuinely has
    # something to back-transform — a non-identity GLM link OR a LHS
    # transform on the response. For a plain identity-link OLS
    # (``family=None``, untransformed ``y``) the link scale IS the
    # response scale, and ``regrid_response`` would raise "No
    # transform recognised from response 'y'" instead of no-opping.
    # In that case we apply the forward directly to the link-scale
    # values.
    em = emm_or_contrast
    if getattr(em, "type", "link") == "link":
        info = getattr(em, "model_info", None)
        needs_response_normalize = False
        if info is not None:
            rname = getattr(info, "response_name", None)
            if isinstance(rname, str) and detect_transform(rname) is not None:
                needs_response_normalize = True
            fam = getattr(info, "family", None)
            if fam is not None:
                link_name = type(
                    getattr(fam, "link", None)
                ).__name__.lower()
                # NoneType (no link attr) and identity both mean
                # "response scale == link scale".
                if link_name not in ("identity", "nonetype"):
                    needs_response_normalize = True
        if needs_response_normalize:
            em = regrid_response(em, bias_adjust=bias_adjust, force=force)
    # Synthetic transform: ``inverse`` = our new forward;
    # ``inverse_deriv`` = derivative of the new forward.
    # ``bias_mean`` / ``bias_deriv`` left None — applying a
    # bias-adjust on top of an already-bias-adjusted response would
    # double-correct.
    synthetic = Transform(
        name=target_name,
        inverse=fwd,
        inverse_deriv=fwd_deriv,
        bias_mean=None,
        is_log=False,
        bias_deriv=None,
    )
    # ``force=True`` because ``em`` is now type="response" and
    # regrid_response would otherwise refuse the second pass; the
    # safety guard exists for the wrong-direction case (calling
    # regrid_response twice with the model's own inverse), which
    # isn't what's happening here.
    return regrid_response(em, tran=synthetic, force=True)
