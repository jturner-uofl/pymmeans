"""Posterior-based EMMs for Bayesian models.

For a Bayesian fit with posterior draws ``beta_samples`` (chains x draws
x n_params), the EMM at each row of ``L_marg`` is the marginal posterior
of ``L_marg @ beta``. We report the posterior mean, posterior SD, and
percentile credible intervals -- analogous to ``emmeans`` on a frequentist
fit but using the posterior distribution directly rather than the
Wald approximation.

The standalone helper :func:`posterior_emm_summary` works on raw
``beta_samples`` arrays, so it doesn't require PyMC / NumPyro / arviz
to be installed. The :func:`from_pymc` adapter (lazy-imports arviz)
wraps a PyMC ``idata`` (arviz ``InferenceData``) into a pymmeans
``ModelInfo`` so all the usual emmeans / contrast / pairs / regrid_response
machinery flows through.

For models pymmeans already supports (OLS, GLM, MixedLM), this gives
a Bayesian path that:

- Reports posterior credible intervals instead of Wald CIs on the EMM
  rows (correctly asymmetric on the response scale for non-linear
  links).
- Naturally handles arbitrary priors and shrinkage estimators.
- Composes with ``contrast()`` but with a caveat: ``contrast()`` is a
  Wald approximation that uses the posterior *covariance* of beta as
  if it were the frequentist vcov. For a true posterior-of-contrast
  credible interval (which captures the full posterior distribution of
  ``D @ L @ beta``), build the contrast's L matrix from
  ``contrast(...).linfct`` and call
  ``posterior_emm_summary(pinfo.beta_samples, L_c)`` directly.

Pairs well with ``brms``- and ``rstanarm``-style workflows: fit in R,
write idata to disk, load with arviz, pass to pymmeans for the
post-processing (R `emmeans` has the same flow but pymmeans gives
faster bulk percentile computations + the streaming bootstrap
interface).

This is a leapfrog feature R ``emmeans`` supports via its
``emm_basis.brmsfit`` method; pymmeans now offers the equivalent in
Python without needing R + brms + rstan.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from pymmeans.emmeans import EMMResult, _validate_level
from pymmeans.utils import ModelInfo


@dataclass(frozen=True)
class PosteriorInfo:
    """Model metadata + posterior samples for a Bayesian fit.

    Mirrors ``ModelInfo`` but stores ``beta_samples`` (the posterior
    draws of the fixed-effect coefficients) instead of a single point
    estimate + vcov. The point estimate becomes the posterior mean and
    ``vcov`` becomes the posterior covariance for Wald-style contrast
    summaries via :func:`~pymmeans.contrast`. Small-sample corrections
    (``apply_satterthwaite``, ``apply_kenward_roger``) and
    :func:`~pymmeans.bootstrap_ci` are intentionally **refused** on
    posterior-derived EMMs and contrasts (rounds 9-11); see the
    refusals in ``satterthwaite.py`` and ``summary.py``. EMMResult
    arithmetic (``+ - * /``) is also refused on posterior summaries
    because it would replace percentile credible intervals with Wald
    intervals.
    """

    beta_samples: np.ndarray # shape (n_samples, n_params)
    model_info: ModelInfo

    @property
    def n_samples(self) -> int:
        return self.beta_samples.shape[0]

    @property
    def n_params(self) -> int:
        return self.beta_samples.shape[1]


def posterior_emm_summary(
    beta_samples: np.ndarray,
    linfct: np.ndarray,
    level: float = 0.95,
    response_transform: Any = None,
    family: Any = None,
    bias_adjust: bool = False,
    sigma_sq: float | None = None,
    offset_mean: float = 0.0,
) -> dict[str, np.ndarray]:
    """Summarise the posterior of ``L_marg @ beta`` at each row of ``L_marg``.

    Parameters
    ----------
    beta_samples
        Posterior samples of beta, shape ``(n_samples, n_params)``.
    linfct
        EMM contrast / linear-functional matrix, shape
        ``(n_rows, n_params)``.
    level
        Credible-interval level (0, 1).
    response_transform
        Optional :class:`~pymmeans.transforms.Transform`. When set,
        the inverse is applied to each posterior draw before
        summarising -- gives the correct asymmetric credible interval
        for non-linear links (R ``emmeans``'s ``type="response"`` on
        ``brms`` fits does the same).
    family
        Optional statsmodels GLM family; used for the inverse-link
        path when ``response_transform`` is None.
    bias_adjust
        Apply :meth:`~pymmeans.transforms.Transform.bias_mean` instead
        of ``inverse``. Requires ``sigma_sq``.
    sigma_sq
        Residual variance for the bias adjustment.
    offset_mean
        Constant added to each draw before applying the link / transform.

    Returns
    -------
    dict
        Keys ``emmean`` (posterior mean), ``SE`` (posterior SD),
        ``lower_cl``, ``upper_cl`` (posterior percentiles).
    """
    level = _validate_level(level)
    beta_samples = np.asarray(beta_samples, dtype=float)
    if beta_samples.ndim != 2:
        raise ValueError(
            f"beta_samples must be 2-D (n_samples, n_params); got shape "
            f"{beta_samples.shape}."
        )
    # #6: std(ddof=1) on n=1 returns NaN; require >= 2.
    if beta_samples.shape[0] < 2:
        raise ValueError(
            f"posterior_emm_summary requires at least 2 posterior samples "
            f"(got {beta_samples.shape[0]}); the SE = std(ddof=1) is "
            "undefined on a single draw."
        )
    # NaN / Inf draws (e.g. divergent MCMC chains) used to
    # propagate silently into mean/SE/percentiles, returning all-NaN or
    # mixed-Inf summaries. Reject up front and report where the
    # invalid values are so the user can debug their sampler.
    if not np.isfinite(beta_samples).all():
        n_bad = int((~np.isfinite(beta_samples)).sum())
        bad_rows = np.unique(
            np.argwhere(~np.isfinite(beta_samples))[:, 0]
        )
        sample_excerpt = bad_rows[:5].tolist()
        raise ValueError(
            f"beta_samples contains {n_bad} non-finite value(s) "
            f"(NaN or Inf). First affected sample indices: "
            f"{sample_excerpt}{'...' if bad_rows.size > 5 else ''}. "
            "Filter divergent draws before calling pymmeans, or pass "
            "only the finite subset."
        )
    linfct = np.asarray(linfct, dtype=float)
    if not np.isfinite(linfct).all():
        raise ValueError(
            f"linfct contains non-finite values; n_invalid = "
            f"{int((~np.isfinite(linfct)).sum())}."
        )
    if linfct.shape[1] != beta_samples.shape[1]:
        raise ValueError(
            f"linfct n_params ({linfct.shape[1]}) doesn't match "
            f"beta_samples ({beta_samples.shape[1]})."
        )
    mu_samples = beta_samples @ linfct.T # (n_samples, n_rows)
    if offset_mean:
        mu_samples = mu_samples + offset_mean
    if response_transform is not None:
        if bias_adjust:
            if sigma_sq is None:
                raise ValueError(
                    "bias_adjust=True requires sigma_sq."
                )
            # the frequentist
            # `regrid_response` has this guard; the posterior twin
            # didn't, so transforms without a `bias_mean` (logit,
            # probit, cloglog, asin_sqrt, etc.) crashed with
            # `TypeError: 'NoneType' object is not callable`.
            if getattr(response_transform, "bias_mean", None) is None:
                raise ValueError(
                    "bias_adjust=True is not defined for transform "
                    f"{getattr(response_transform, 'name', response_transform)!r} "
                    "(no `bias_mean` closed-form correction). For the "
                    "proportion family / asin_sqrt / exp, use a "
                    "parametric bootstrap on the link-scale posterior "
                    "instead."
                )
            mu_samples = response_transform.bias_mean(mu_samples, sigma_sq)
        else:
            mu_samples = response_transform.inverse(mu_samples)
    elif family is not None:
        mu_samples = family.link.inverse(mu_samples)

    alpha = 1.0 - level
    return {
        "emmean": mu_samples.mean(axis=0),
        "SE": mu_samples.std(axis=0, ddof=1),
        "lower_cl": np.percentile(mu_samples, 100.0 * alpha / 2.0, axis=0),
        "upper_cl": np.percentile(mu_samples, 100.0 * (1.0 - alpha / 2.0), axis=0),
    }


def posterior_emmeans(
    pinfo: PosteriorInfo,
    specs: str | list[str],
    by: str | list[str] | None = None,
    at: dict[str, Any] | None = None,
    level: float = 0.95,
    type: str = "link",
    tran: Any = None,
    weights: str | None = None,
    *,
    mode: str | None = None,
) -> EMMResult:
    """Posterior-based emmeans on a Bayesian fit.

    Same API as :func:`pymmeans.emmeans` but uses the posterior
    distribution instead of the Wald approximation. The returned
    ``EMMResult`` has ``emmean`` = posterior mean, ``SE`` = posterior SD,
    and ``lower_cl`` / ``upper_cl`` = posterior credible-interval
    endpoints from percentiles.

    The ``linfct`` matrix attached to the result is identical to the
    frequentist path, so downstream :func:`~pymmeans.contrast` and
    :func:`~pymmeans.pairs` work -- they will compute contrasts under
    the Wald approximation (using ``vcov = posterior_cov(beta)``) by
    default. For posterior-distribution contrasts, the cleanest
    workflow is: compute the contrast L_c matrix via ``pairs(...)``
    on the posterior result, then call
    :func:`posterior_emm_summary(pinfo.beta_samples, L_c)` to get
    the posterior-of-contrast credible intervals directly.

    Parameters
    ----------
    pinfo
        :class:`PosteriorInfo` holding posterior draws + model metadata.
    specs
        Target factor name(s); same semantics as
        :func:`pymmeans.emmeans` ``specs``.
    by
        Optional factor name(s) to condition on; one EMM per by-level.
    at
        Optional dict mapping variable name to a scalar or list of
        values that overrides the default. Same semantics as
        :func:`pymmeans.emmeans`.
    level
        Credible-interval level in ``(0, 1)``. Default 0.95.
    type
        ``"link"`` (default) for linear-predictor-scale EMMs or
        ``"response"`` to apply the inverse link / LHS-transform
        inverse to each posterior draw before summarising. The latter
        gives the *correct* posterior of ``E[inverse(L beta)]``, not
        ``inverse(E[L beta])`` (which is what
        :func:`regrid_response` would give if it weren't refused on
        posterior inputs).
    mode
        Optional R-style alias for ``type=``. Mirrors brms / rstanarm
        ``emmeans(..., mode = ...)``. Mutually exclusive with
        ``type=``. Supported values:

        - ``"latent"`` / ``"linear.predictor"`` / ``"link"``
          → equivalent to ``type="link"``.
        - ``"response"`` / ``"prob"`` / ``"mean"``
          → equivalent to ``type="response"``.
        - ``"mean.class"`` is refused on the generic path because it
          requires per-category posterior summaries; use
          :func:`pymmeans.ordinal_emmeans` / :func:`pymmeans.multinom_emmeans`
          instead.
    """
    from pymmeans.emmeans import emmeans as _emmeans
    from pymmeans.transforms import detect_transform

    # R-style ``mode=`` aliases.
    # brms / rstanarm use ``mode=`` for the EMM scale on Bayesian fits.
    # We accept R's vocabulary and route it to our existing ``type=``
    # semantics so users transferring code from R don't have to learn
    # a second name. ``mode=`` and ``type=`` are mutually exclusive
    # (refused with a clear message) so the resolved scale is never
    # ambiguous.
    #
    # Maps:
    #   - latent / linear.predictor / link  → type='link'
    #   - response / prob / mean             → type='response'
    #   - mean.class                         → not supported on the
    #     generic posterior path (steers to ordinal_emmeans/multinom_emmeans)
    if mode is not None:
        # Explicit type= AND explicit mode= is ambiguous; refuse.
        if type != "link":
            raise ValueError(
                "Pass either `mode=` (R-style) OR `type=` (pymmeans-"
                "style), not both. `mode=` and `type=` are aliases "
                f"for the same display scale; got mode={mode!r}, "
                f"type={type!r}."
            )
        m = mode.lower().strip()
        if m in ("latent", "linear.predictor", "linear_predictor", "link"):
            type = "link"
        elif m in ("response", "prob", "mean"):
            type = "response"
        elif m == "mean.class":
            raise NotImplementedError(
                "mode='mean.class' is only meaningful for ordinal / "
                "multinomial Bayesian fits, where the posterior is "
                "over category probabilities. Use the dedicated "
                "entry points instead:\n"
                "  - Ordinal:    `pymmeans.ordinal_emmeans(fit, ..., "
                "mode='mean.class')`\n"
                "  - Multinomial: `pymmeans.multinom_emmeans(fit, ..., "
                "mode='prob')` then summarise the per-category posterior."
            )
        else:
            raise ValueError(
                f"Unknown mode={mode!r}. Supported R-style names: "
                "'latent' / 'linear.predictor' / 'link' (= type='link'); "
                "'response' / 'prob' / 'mean' (= type='response'); "
                "'mean.class' (use ordinal_emmeans / multinom_emmeans)."
            )

    # #7: validate type just like emmeans() does — otherwise
    # `type='foo'` flows through and downstream code sees an invalid tag.
    type = type.lower() if isinstance(type, str) else type
    if type not in ("link", "response"):
        raise ValueError(
            f"'type' must be 'link' or 'response', got {type!r}."
        )

    # Build the linfct via the standard emmeans path, using the
    # posterior-mean ModelInfo. We discard the frequentist frame and
    # rebuild it from posterior summaries.
    # forward ``weights=`` to the link-scale base EMM so
    # users can request `weights="proportional"` / `"outer"` / etc.
    # on a posterior fit (advertised propagation of the
    # `weights` field but the kwarg wasn't accepted).
    base = _emmeans(
        pinfo.model_info, specs, by=by, at=at, level=level, type="link",
        weights=weights,
    )
    L = base.linfct

    info = pinfo.model_info
    response_transform = None
    if type == "response" and info.family is None:
        # accept an explicit `tran=`
        # so users with composite LHS expressions (e.g. `log(y + 1)`)
        # have the same ergonomic shape the frequentist `regrid_response`
        # offers, without dropping to the lower-level `posterior_emm_
        # summary`. When `tran=` is supplied, it wins.
        if tran is not None:
            # Accept a Transform instance OR the dict-form spec.
            if isinstance(tran, dict):
                from pymmeans.transforms import Transform
                deriv = tran.get("inverse_deriv", tran.get("deriv"))
                if "inverse" not in tran or deriv is None:
                    raise ValueError(
                        "posterior_emmeans(..., tran=dict) requires "
                        "both 'inverse' and 'inverse_deriv' (or alias "
                        "'deriv')."
                    )
                response_transform = Transform(
                    tran.get("name", "custom"),
                    tran["inverse"],
                    deriv,
                    tran.get("bias_mean"),
                    tran.get("is_log", False),
                    tran.get("bias_deriv"),
                    tran.get("contrast_inverse"),
                    tran.get("contrast_inverse_deriv"),
                )
            else:
                response_transform = tran
        else:
            response_name = info.response_name or ""
            t = detect_transform(response_name)
            if t is not None:
                response_transform = t
            elif "(" in response_name:
                # Composite-LHS refusal (/ ).
                raise ValueError(
                    f"posterior_emmeans(..., type='response') with "
                    f"response_name={response_name!r} cannot be auto-back-"
                    "transformed because the inner expression is composite. "
                    "Either (a) pass `tran=make_tran('genlog', base=1)` "
                    "to posterior_emmeans directly, or (b) call "
                    "`posterior_emm_summary(pinfo.beta_samples, L, "
                    "response_transform=...)` for full control."
                )
    summary = posterior_emm_summary(
        pinfo.beta_samples,
        L,
        level=level,
        response_transform=response_transform,
        family=info.family if type == "response" else None,
        offset_mean=info.offset_mean,
    )
    frame = base.frame.copy()
    frame["emmean"] = summary["emmean"]
    frame["SE"] = summary["SE"]
    frame["lower_cl"] = summary["lower_cl"]
    frame["upper_cl"] = summary["upper_cl"]
    # df is not meaningful for a posterior; convention: use n_samples - 1.
    frame["df"] = float(pinfo.n_samples - 1)

    return EMMResult(
        frame=frame,
        linfct=L,
        model_info=info,
        target=base.target,
        by=base.by,
        level=level,
        type=type,
        inference_kind="posterior",
        # propagate the ``at`` / ``weights``
        # metadata from the link-scale base EMM so downstream
        # operations (``pairs(simple=)``, ``regrid_response``, etc.)
        # see the same grid restrictions used when building this
        # posterior EMM.
        at=getattr(base, "at", None),
        weights=getattr(base, "weights", "equal") or "equal",
    )


def from_pymc(
    idata: Any,
    formula: str,
    data: pd.DataFrame,
    var_name: str = "beta",
) -> PosteriorInfo:
    """Build a :class:`PosteriorInfo` from an arviz :class:`InferenceData`.

    Despite the historical name, this function is framework-agnostic:
    it accepts any arviz ``InferenceData`` object with a ``posterior``
    group, regardless of which MCMC engine produced it. PyMC, numpyro,
    blackjax, cmdstanpy / Stan, TFP, and pyro all expose arviz
    converters (``arviz.from_pymc``, ``arviz.from_numpyro``,
    ``arviz.from_cmdstanpy``, ``arviz.from_pyro``, etc.) — convert
    your fit to InferenceData via the appropriate converter, then
    pass it here.

    See :func:`from_arviz` for the framework-neutral alias.

    Lazy-imports arviz; raises ``ImportError`` if arviz isn't available.

    Parameters
    ----------
    idata
        An arviz :class:`InferenceData` object. Must contain a
        ``posterior`` group with the fixed-effect coefficients under
        ``var_name``. Sources include (non-exhaustive):

        - PyMC: ``pm.sample(..., return_inferencedata=True)``
        - numpyro: ``arviz.from_numpyro(mcmc)``
        - blackjax: ``arviz.from_blackjax(states, ...)``
        - cmdstanpy / Stan: ``arviz.from_cmdstanpy(fit)``
        - pyro: ``arviz.from_pyro(mcmc)``

    formula
        The patsy formula used to build the design matrix. pymmeans
        rebuilds the design via patsy on ``data`` so we get factor
        levels, numeric means, and the rest of ``ModelInfo``.
    data
        The DataFrame the model was fit on.
    var_name
        Name of the fixed-effects coefficient variable in
        ``idata.posterior``. Default ``"beta"`` matches the convention
        used in most PyMC examples (``coords={"feature": ...}``).

    Returns
    -------
    PosteriorInfo
        With ``beta_samples`` extracted from the posterior and
        ``model_info`` carrying the design metadata + posterior-mean
        coefficients + posterior covariance as ``vcov``.

    Examples
    --------
    >>> import pymc as pm # doctest: +SKIP
    >>> with pm.Model(coords={"feature": X.columns}) as model: # doctest: +SKIP
    ... beta = pm.Normal("beta", 0, 5, dims="feature")
    ... sigma = pm.HalfNormal("sigma", 1)
    ... pm.Normal("y_obs", mu=X @ beta, sigma=sigma, observed=y)
    ... idata = pm.sample(1000, chains=4)
    >>> pinfo = from_pymc(idata, "y ~ a + b", df, var_name="beta") # doctest: +SKIP
    >>> emm = posterior_emmeans(pinfo, "a") # doctest: +SKIP
    """
    # Availability check. The unused ``arviz`` name is silenced via a
    # ``F401`` noqa rather than ``importlib.util.find_spec`` because
    # tests that mock arviz with ``monkeypatch.setitem(sys.modules,
    # "arviz", types.SimpleNamespace())`` do not set ``__spec__`` on
    # the mock — ``find_spec`` then raises ``ValueError`` and the
    # missing-arviz error message never fires.
    try:
        import arviz  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "from_pymc requires arviz. Install with: pip install arviz"
        ) from exc

    if not hasattr(idata, "posterior"):
        raise ValueError(
            "from_pymc: idata has no 'posterior' group. Pass an arviz "
            "InferenceData object produced by pm.sample(..., return_inferencedata=True)."
        )

    posterior = idata.posterior[var_name]
    # Build patsy design first so we know how many coefficients to expect.
    from patsy import dmatrices

    _, X_design = dmatrices(formula, data, return_type="dataframe")
    design_info = X_design.design_info
    param_names = list(design_info.column_names)
    n_params_expected = len(param_names)

    # Stack chain + draw -> single 'sample' dim. After stacking, the
    # array has dims (other_dims..., 'sample'). We transpose to put
    # 'sample' FIRST so the resulting numpy array is always
    # (n_samples, ...remaining...) — #2 fixed the previous
    # shape-comparison guess that silently transposed square shapes.
    stacked = posterior.stack(sample=("chain", "draw"))
    if "sample" not in stacked.dims:
        raise ValueError(
            f"Posterior['{var_name}'] does not have ('chain', 'draw') "
            "dimensions; cannot extract samples deterministically. "
            "Reshape your idata or pass beta_samples directly via "
            "PosteriorInfo(beta_samples=..., model_info=...)."
        )
    # Place 'sample' first; '...' keeps the remaining dim order.
    stacked = stacked.transpose("sample", ...)
    # if the coefficient dim has named coordinates that
    # match patsy's column names, REORDER the samples to patsy order.
    # fix only ensured 'sample' was first, but left the
    # coefficient order as-given — silently permuting beta when the
    # user's PyMC coords were in a different order than patsy.
    remaining_dims = [d for d in stacked.dims if d != "sample"]
    coords = getattr(stacked, "coords", None)
    if len(remaining_dims) >= 2:
        # #2: a posterior var with >= 2 non-sample dims
        # (e.g. PyMC `beta[row, col]`) used to be silently reshaped to
        # `(n_samples, -1)`, producing an arbitrary flattened order
        # that almost never matches patsy. Refuse cleanly and tell
        # the user how to fix it.
        raise ValueError(
            f"Posterior['{var_name}'] has {len(remaining_dims)} "
            f"coefficient dimensions {tuple(remaining_dims)}; pymmeans "
            "cannot infer the patsy column order from a multi-dim "
            "coefficient array. Either (a) flatten in PyMC so the "
            "posterior has a single coordinate matching patsy column "
            f"names {param_names}, or (b) build the beta_samples "
            "array yourself and call "
            "PosteriorInfo(beta_samples=..., model_info=...) directly."
        )
    if len(remaining_dims) == 1 and coords is not None:
        coef_dim = remaining_dims[0]
        if coef_dim in coords:
            coord_labels = list(coords[coef_dim].values)
            if set(coord_labels) == set(param_names):
                # Coords match patsy columns (modulo order) — align.
                stacked = stacked.sel({coef_dim: param_names})
            elif set(coord_labels) != set(param_names):
                raise ValueError(
                    f"Posterior['{var_name}'] coordinate labels along "
                    f"'{coef_dim}' = {coord_labels} don't match patsy's "
                    f"column names {param_names}. Either match the "
                    "names exactly or pass beta_samples directly via "
                    "PosteriorInfo(beta_samples=..., model_info=...)."
                )
        else:
            # No coords — positional only. Warn the user so they don't
            # silently send beta in the wrong order.
            import warnings as _w
            _w.warn(
                f"Posterior dim {coef_dim!r} has no coordinate labels; "
                "using positional ordering. If your PyMC model's "
                f"coefficient order doesn't match patsy "
                f"({param_names}), the resulting beta will be permuted. "
                "Add `coords={...: param_names}` to your PyMC model to "
                "silence this warning.",
                UserWarning, stacklevel=2,
            )
    samples = np.asarray(stacked.values, dtype=float)
    if samples.ndim == 1:
        # Single-coefficient model: posterior had no 'feature' dim.
        if n_params_expected != 1:
            raise ValueError(
                f"Posterior['{var_name}'] is 1-D (single coefficient) but "
                f"the formula expects {n_params_expected} coefficients: "
                f"{param_names}."
            )
        samples = samples[:, None]
    elif samples.ndim > 2:
        # Flatten any extra dims into the coefficient axis (rare; some
        # PyMC models use multi-dim coords for hierarchical effects).
        samples = samples.reshape(samples.shape[0], -1)

    if samples.shape[1] != n_params_expected:
        raise ValueError(
            f"Posterior n_params ({samples.shape[1]}) doesn't match the "
            f"formula's design ({n_params_expected} columns: {param_names}). "
            "If your PyMC model uses coords with named dimensions, ensure "
            "the coordinate order matches patsy's column order."
        )

    beta_mean = samples.mean(axis=0)
    beta_cov = np.cov(samples, rowvar=False)

    # Build ModelInfo from the design + posterior summaries.
    from pymmeans.utils import _build_estimability_basis, _underlying_columns

    X = np.asarray(X_design, dtype=float)
    factors: dict[str, list[str]] = {}
    numeric_means: dict[str, float] = {}
    frame_cols = set(data.columns)
    aliases: dict[str, str] = {}
    for factor, fi in design_info.factor_infos.items():
        name = factor.name()
        if fi.type == "categorical":
            factors[name] = list(fi.categories)
        elif name in data.columns:
            numeric_means[name] = float(data[name].mean())
        if name not in frame_cols:
            underlying = _underlying_columns(factor.code, frame_cols)
            if len(underlying) == 1 and underlying[0] != name:
                aliases.setdefault(underlying[0], name)

    info = ModelInfo(
        beta=beta_mean,
        vcov=beta_cov,
        param_names=param_names,
        factors=factors,
        numeric_means=numeric_means,
        df_resid=float(len(data) - len(param_names)),
        design_info=design_info,
        data=data,
        response_name=formula.split("~")[0].strip(),
        family=None,
        scale=1.0,
        is_mixed=False,
        aliases=aliases,
        raw_result=None,
        offset_mean=0.0,
        fit_weights=None,
        estimability_basis=_build_estimability_basis(X),
    )
    return PosteriorInfo(beta_samples=samples, model_info=info)


# Framework-neutral alias.
# ``from_pymc`` is purely arviz-based — it never touches PyMC's API. The
# function name is historical: it was added when PyMC was the only
# supported source. Any MCMC engine that ships an arviz converter
# (numpyro, blackjax, cmdstanpy / Stan, pyro, TFP, ...) produces an
# InferenceData that flows through unchanged. ``from_arviz`` is the
# name that reflects what the function actually does; ``from_pymc``
# is retained as an alias so existing user code keeps working.
from_arviz = from_pymc
