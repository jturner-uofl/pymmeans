"""Summary helpers and bootstrap inference for EMM results.

The bootstrap path here is **parametric** — samples ``beta* ~ N(beta_hat,
vcov)`` rather than resampling rows of the data and refitting. That's
fast (no model refits), natural for our linear-combination framework
(``mu* = L_marg @ beta*``). R ``emmeans`` does not ship a bootstrap
helper of its own; R users typically compose ``emmeans`` with the
``boot`` package manually, or roll their own ``mvtnorm::rmvnorm`` draw
loop. The pymmeans ``bootstrap_ci`` is therefore a beyond-R-parity
convenience — see also ``bootstrap_ci(kind="case")`` for the true
non-parametric resample.

For response-scale EMMs the inverse link is applied to each sample
*before* computing percentiles, so asymmetric response-scale CIs (e.g.
probabilities near 0 or 1, or rates near zero) come out correctly
without the delta-method approximation that the default
``apply_satterthwaite`` / ``regrid_response`` path uses.

Edge cases worth knowing:

- Rank-deficient ``vcov`` is allowed via ``scipy.stats.multivariate_normal(
  ..., allow_singular=True)``.
- The ``seed`` argument is deterministic across runs and Python versions
  via ``numpy.random.default_rng``.

References
----------
- Efron & Tibshirani (1993). *An Introduction to the Bootstrap*. The
  parametric variant; ours is "model-based parametric bootstrap."
- Lenth (2024). ``emmeans::confint`` documentation on the bootstrap CI
  option.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from pymmeans.emmeans import EMMResult

_DEFAULT_CHUNK = 50_000


def bootstrap_ci(
    emm: EMMResult | Any,
    n_samples: int = 5000,
    level: float | None = None,
    seed: int | None = None,
    chunk_size: int | None = None,
    regularize: bool = True,
    method: str = "exact",
    kind: str = "parametric",
    refit_fn: Callable | None = None,
) -> EMMResult | Any:
    """Replace an EMMResult's CIs with parametric-bootstrap percentile CIs.

    Samples ``beta* ~ N(beta_hat, vcov)``, propagates through ``L_marg``
    (and the inverse link, if ``type="response"``), and reports the
    percentile interval at ``level``.

    Parameters
    ----------
    emm
        Source EMMResult.
    n_samples
        Number of bootstrap draws. Default 5000.
    level
        Confidence level. If ``None``, reuses ``emm.level``.
    seed
        Optional seed for reproducibility.
    chunk_size
        Draws to materialise *per chunk*. Defaults to ``min(n_samples,
        50_000)``. Under ``method="exact"`` (default) only controls the
        per-iteration ``beta_chunk`` allocation; the full
        ``(n_samples, n_emm_rows)`` matrix is still kept because exact
        percentiles need the whole distribution. Under
        ``method="streaming"`` chunk_size only affects RNG batching;
        peak memory is bounded by the P² estimators (~80 bytes per
        (percentile x EMM row)).
    regularize
        If ``True`` (default), nudge a near-singular or numerically
        non-PSD ``vcov`` toward PSD by adding ``1e-12 * I`` so
        ``multivariate_normal`` doesn't return nonsense draws. Disable
        only when you've already verified the vcov is well-conditioned.
    method
        Percentile estimator. ``"exact"`` (default) materialises every
        draw and calls :func:`numpy.percentile`; memory is
        ``O(n_samples · n_emm_rows)``. ``"streaming"`` uses Jain &
        Chlamtac's P² algorithm (1985) to estimate percentiles online
        in **constant** memory (~80 bytes per percentile x EMM row),
        making n_samples up to ~10^7 practical on a 200-row EMMResult.
        Tradeoff: P² is accurate to ~0.05% after 50k samples and
        ~0.01% at 1M (better than the Monte Carlo noise floor for any
        reasonable n_samples), but the per-draw update is sequential —
        for very small ``n_emm_rows`` and small ``n_samples`` the
        ``"exact"`` path is faster. Use ``"streaming"`` when the
        materialised ``n_samples x n_emm_rows`` matrix would not fit
        in RAM. This is a leapfrog feature R ``emmeans`` does not
        offer.
    kind
        (beyond-R-parity): which bootstrap scheme to use.

        - ``"parametric"`` (default; backwards-compatible) — samples
          ``β* ~ N(β̂, V̂)`` then propagates through ``linfct``.
          Fast (no refits), but assumes the asymptotic Gaussian
          approximation of the sampling distribution is correct.
        - ``"case"`` — true **non-parametric** case-resampling
          bootstrap. For each draw, resamples the original data rows
          with replacement, refits the model on the resample,
          rebuilds the EMM, and stores the resulting point estimate.
          Slower (one model refit per draw) but makes weaker
          assumptions; recommended for small samples, heavy-tailed
          residuals, or robust-SE consistency checks. Requires
          either ``obj.model_info.raw_result`` to be a statsmodels
          formula-API result, OR an explicit ``refit_fn`` callable.
    refit_fn
        Optional callable ``refit_fn(data) -> fitted_result`` used
        by ``kind="case"`` to rebuild the model from a resampled
        DataFrame. Default: reconstruct via the original model's
        ``model.formula`` and class. Useful when the model has
        custom data preprocessing (e.g. weights / clustering) that
        the default reconstruction would lose.

    Returns
    -------
    Same type as ``emm``. returns

    - ``EMMResult`` for an EMMResult input (parametric or case).
    - ``ContrastResult`` for a ContrastResult input (;
        both parametric and case dispatch). also handles
        ``method="rbind"`` ContrastResults by rebuilding each child
        and concatenating draws.
    - ``MLEMMResult`` for an ML EMM.
    - ``EmmList`` for an EmmList (— recurses
        member-wise).

    Raises ``TypeError`` for a raw ``pandas.DataFrame`` input
    (e.g. from ``ml_contrast``) — bootstrap CIs for ML contrasts
    are not yet implemented; bootstrap the EMM first.

    References
    ----------
    - Efron, B. (1979). Bootstrap Methods: Another Look at the
      Jackknife. *Annals of Statistics*, 7(1), 1-26.
    - Davison, A. C., & Hinkley, D. V. (1997). *Bootstrap Methods
      and Their Application*. Cambridge University Press.
    - Jain, R., & Chlamtac, I. (1985). The P² Algorithm for Dynamic
      Calculation of Quantiles and Histograms Without Storing
      Observations. *Communications of the ACM*, 28(10), 1076-1085.

    Examples
    --------
    >>> import statsmodels.formula.api as smf # doctest: +SKIP
    >>> from statsmodels.datasets import get_rdataset # doctest: +SKIP
    >>> from pymmeans import emmeans, bootstrap_ci # doctest: +SKIP
    >>> wb = get_rdataset("warpbreaks").data # doctest: +SKIP
    >>> fit = smf.ols("breaks ~ wool * tension", wb).fit() # doctest: +SKIP
    >>> em = emmeans(fit, "tension") # doctest: +SKIP
    >>> em_b = bootstrap_ci(em, n_samples=2000, seed=0) # doctest: +SKIP
    >>> em_b.df_method # doctest: +SKIP
    'bootstrap'

    Case-resampling bootstrap (refits the model per draw)::

        em_case = bootstrap_ci(em, n_samples=500, kind="case", seed=0)
    """
    if n_samples < 100:
        raise ValueError(f"n_samples must be >= 100 for stable CIs, got {n_samples}.")
    method = method.lower()
    if method not in ("exact", "streaming"):
        raise ValueError(
            f"method must be 'exact' or 'streaming'; got {method!r}."
        )
    kind = kind.lower()
    if kind not in ("parametric", "case"):
        raise ValueError(
            f"kind must be 'parametric' or 'case'; got {kind!r}."
        )

    # dispatch on input type. ContrastResult goes through
    # the same machinery — the contrast's linfct stores D @ L_marg.
    from pymmeans.contrasts import ContrastResult
    is_contrast = isinstance(emm, ContrastResult)

    # ML adapter dispatch — totally separate code path.
    from pymmeans.ml import MLEMMResult
    if isinstance(emm, MLEMMResult):
        return _bootstrap_ci_ml(
            emm, n_samples=n_samples, level=level, seed=seed,
            refit_fn=refit_fn,
        )

    # posterior / Satt / KR refusals MUST run before the
    # case-bootstrap dispatch. The ordering placed the
    # `kind=='case'` branch first, so ``bootstrap_ci(posterior_emm,
    # kind='case')`` silently replaced percentile credible intervals
    # with case-bootstrap Wald-style CIs — the exact silent-inference-
    # corruption the guard was designed to prevent.

    # #1: refuse posterior EMMs. They already carry
    # posterior percentile credible intervals; re-bootstrapping their
    # posterior covariance would replace them with a Wald-style
    # Gaussian approximation -- silently wrong inference type.
    if getattr(emm, "inference_kind", "wald") == "posterior":
        raise ValueError(
            "bootstrap_ci() is not defined for posterior-derived EMMs. "
            "posterior_emmeans() already reports posterior percentile "
            "credible intervals from beta_samples; re-bootstrapping the "
            "posterior covariance would replace them with a Wald-style "
            "Gaussian approximation."
        )

    # refuse a multiplicity-adjusted EMM / contrast.
    # made ``update(em, adjust='bonferroni')`` widen the
    # frame's ``lower_cl`` / ``upper_cl`` to the family-wise critical
    # value. ``bootstrap_ci`` then OVERWROTE those columns with raw
    # percentile (i.e. unwidened) bootstrap CIs while leaving the
    # dataclass field ``adjust='bonferroni'`` stamped — the same
    # split-brain class closed for ``at`` / ``weights``,
    # but for the adjust field. Bootstrap CIs are NOT an adjusted-
    # inference method (they reflect sampling uncertainty, not
    # multiplicity), so the two are incompatible. Force the user to
    # bootstrap the raw EMM and apply the multiplicity correction
    # separately.
    _adjust = getattr(emm, "adjust", None)
    if _adjust is not None and _adjust not in (
        "none", "default",
    ):
        # Distinguish "user set adjust on a ContrastResult (always
        # present)" from "user called update(em, adjust=...) on an
        # EMMResult". On ContrastResult, ``adjust`` is the contrast's
        # native family adjustment — bootstrap then reapplies it to
        # the new t-ratios via the existing summary path. Only refuse
        # on EMMResult, where ``adjust`` widens CIs that bootstrap
        # would then silently replace.
        if not is_contrast:
            raise ValueError(
                "bootstrap_ci() is not defined for an EMMResult whose "
                f"``adjust`` field is non-trivial (here: {_adjust!r}). "
                "widens the frame's CI columns to the family-"
                "wise critical value when ``adjust`` is set; bootstrap "
                "percentile CIs would overwrite those widened columns "
                "with raw (unwidened) percentiles while leaving the "
                "``adjust`` field stamped — silently mixing two "
                "inference paradigms. Either:\n"
                " - bootstrap the un-adjusted EMM first, then apply "
                "the adjustment to the result via ``update(adjust=...)`` "
                "(R parity);\n"
                " - or stay with parametric inference and skip the "
                "bootstrap."
            )

    # EmmList → recurse member-wise (so users can
    # ``bootstrap_ci(pairs(em, simple='each'))`` cleanly).
    from pymmeans.contrasts import EmmList
    if isinstance(emm, EmmList):
        from pymmeans.contrasts import EmmList as _EmmList
        return _EmmList(**{
            name: bootstrap_ci(
                member, n_samples=n_samples, level=level, seed=seed,
                chunk_size=chunk_size, regularize=regularize,
                method=method, kind=kind, refit_fn=refit_fn,
            )
            for name, member in zip(emm.names, emm, strict=True)
        })

    # refuse a raw DataFrame (e.g. result of
    # ``ml_contrast``). ``ml_contrast`` returns a DataFrame, not an
    # MLEMMResult, so bootstrap_ci(ml_contrast(...)) would crash on
    # `.model_info`. Surface a clear error.
    if isinstance(emm, pd.DataFrame):
        raise TypeError(
            "bootstrap_ci() got a raw DataFrame (likely from "
            "ml_contrast()). Pass the MLEMMResult to bootstrap_ci "
            "FIRST to populate CIs, then compute contrasts; bootstrap "
            "CIs for ml_contrast outputs are not yet implemented."
        )

    # move the Satt/KR refusal block ABOVE the case-
    # bootstrap dispatch — the ordering had ``kind=='case'``
    # firing first, which meant ``bootstrap_ci(apply_satterthwaite(em),
    # kind='case')`` silently tried a model refit and raised the
    # generic "unstable" RuntimeError instead of the
    # refusal. Same fix pattern as for posterior.

    # refuse Satt / KR EMMs. The bootstrap samples beta
    # ~ N(beta_hat, info.vcov), which is V_beta. For KR that's
    # inconsistent with the EMM's SE (computed from V_KR) — the CIs
    # would ignore the KR variance-component-uncertainty inflation
    # entirely. For Satt the SE is still V_beta-based so the SE +
    # bootstrap CIs are internally consistent, but the percentile
    # bootstrap CIs supersede the t-based Satt CIs while the frame
    # would still carry Satt df, silently mixing two inference
    # paradigms. Force the user to pick: parametric (Satt/KR) or
    # bootstrap.
    _df_method = getattr(emm, "df_method", "default")
    if _df_method != "default":
        raise ValueError(
            f"bootstrap_ci() is not defined for EMMs with the "
            f"{_df_method!r} small-sample correction applied. "
            "Pick one inference path: either parametric Satt / KR "
            "(call apply_satterthwaite / apply_kenward_roger and stop), "
            "or percentile bootstrap (call bootstrap_ci on the plain "
            "EMM before applying any correction). Mixing the two would "
            "silently combine V_beta-based bootstrap draws with "
            "V_KR-based SEs (KR case) or replace t-based Satt CIs "
            "while leaving Satt df on the frame (Satt case)."
        )

    # case bootstrap dispatch (separate code path, totally
    # different from parametric since we have to refit).
    if kind == "case":
        # ``method="streaming"`` is implemented only for
        # the parametric path (P² online quantile estimator). Case
        # bootstrap requires materialising the n_samples × n_rows
        # draw matrix; reject the combination explicitly instead of
        # silently allocating the full matrix (which would mislead
        # users who passed ``streaming`` for memory reasons).
        if method == "streaming":
            raise ValueError(
                "method='streaming' is supported only for "
                "kind='parametric'. The case bootstrap rebuilds the "
                "EMM at every draw and stores the full per-draw "
                "estimates; pass method='exact' (default) for "
                "kind='case', or kind='parametric' for streaming."
            )
        return _bootstrap_ci_case(
            emm, n_samples=n_samples, level=level, seed=seed,
            refit_fn=refit_fn,
        )

    rng = np.random.default_rng(seed)
    if level is None:
        level = emm.level
    else:
        from pymmeans.emmeans import _validate_level

        level = _validate_level(level)
    chunk = min(n_samples, chunk_size or _DEFAULT_CHUNK)

    info = emm.model_info
    vcov = info.vcov
    if regularize:
        # Nudge any tiny negative eigenvalue (numerical noise) to a small
        # positive value. Cheaper than full eigendecomposition: add a
        # symmetric jitter, then rely on multivariate_normal's
        # allow_singular path for genuinely rank-deficient cases.
        vcov = np.asarray(vcov, dtype=float).copy()
        diag_scale = float(np.abs(np.diag(vcov)).max()) if vcov.size else 1.0
        vcov.flat[:: vcov.shape[0] + 1] += diag_scale * 1e-12

    import scipy.stats as _stats

    rv = _stats.multivariate_normal(
        info.beta, vcov, allow_singular=True, seed=rng
    )

    # When emm.type == "response" we have to invert whatever took us to the
    # link scale in the first place. For GLMs that's the family's link
    # function. For OLS with an LHS transform (e.g. `np.log(y) ~ ...`) the
    # "link" is the LHS transform; detect it from info.response_name.
    # us applying neither when family is None, silently writing
    # link-scale draws into a response-scale frame.
    #
    # #4: when ``emm`` was produced via
    # ``regrid_response(..., bias_adjust=True)`` the response-scale
    # values are bias-corrected. To keep bootstrap CIs on the same
    # scale we apply ``tran.bias_mean(mu, sigma^2)`` to each draw
    # instead of the plain inverse.
    bias_adjust = bool(getattr(emm, "bias_adjust", False))
    response_transform = None
    if emm.type == "response" and info.family is None:
        from pymmeans.transforms import detect_transform

        lhs_tran = detect_transform(info.response_name or "")
        if lhs_tran is not None:
            response_transform = lhs_tran
    # when the source is a bias-adjusted
    # response-scale GLM EMM (``info.family is not None`` and
    # ``bias_adjust=True``), build the response_transform from the
    # GLM family's link function. Previously the path only knew how to
    # detect LHS transforms (OLS with ``np.log(y) ~ ...``), so a
    # Poisson / Gamma / Binomial bias-adjusted EMM was a valid
    # ``regrid_response`` output but the parametric bootstrap
    # refused it. Limited to log link in this round (the only GLM
    # link where ``bias_mean`` is well-defined as
    # ``exp(eta + sigma^2/2)``); other links need their own bias-
    # correction formulas (logit Taylor expansion isn't shipped).
    if bias_adjust and response_transform is None and info.family is not None:
        from pymmeans.transforms import make_tran as _make_tran
        link_name = type(info.family.link).__name__.lower()
        if link_name == "log":
            response_transform = _make_tran("log")
        else:
            raise ValueError(
                "bootstrap_ci on a bias-adjusted GLM response EMM is "
                "currently implemented only for the log link (Poisson, "
                "Gamma with log link, NegBinomial); got link "
                f"{link_name!r}. Bootstrap the link-scale EMM and "
                "call ``regrid_response(bootstrap_result)`` for "
                "non-log links."
            )
    if bias_adjust and response_transform is None:
        raise ValueError(
            "bootstrap_ci on a bias-adjusted response EMM requires the "
            "originating transform to be detectable from "
            "model_info.response_name. If you applied a custom transform, "
            "either bootstrap the link-scale EMM and regrid the result, "
            "or skip bias_adjust."
        )
    # honor stored ``bias_sigma`` set by
    # ``regrid_response(em, bias_adjust=True, sigma=...)`` (
    # F4). Previously the parametric bootstrap always used
    # ``info.scale``, so percentile CIs on a bias-adjusted response
    # EMM were computed at the model default sigma while the point
    # estimate was bias-adjusted at the user's sigma — split-brain.
    # The case-bootstrap path already read ``bias_sigma``; this
    # brings the parametric path in line.
    _bias_sigma_override = getattr(emm, "bias_sigma", None)
    if _bias_sigma_override is not None:
        _sig = np.asarray(_bias_sigma_override, dtype=float)
        sigma_sq = float(_sig * _sig) if _sig.ndim == 0 else _sig * _sig
    else:
        sigma_sq = float(info.scale) if info.scale is not None else 1.0

    offset_mean = float(getattr(info, "offset_mean", 0.0) or 0.0)
    # #2: skip non-estimable rows. The input EMM already
    # marks them NaN in `emmean` / `SE` (estimability path);
    # bootstrap was projecting all linfct rows blindly and writing
    # finite percentile intervals on top, producing a NaN point with
    # a finite CI -- silently wrong. Mask valid rows in, write NaN
    # back out for the rest.
    # ContrastResult uses 'estimate' or 'ratio'.
    if "emmean" in emm.frame.columns:
        point_col = "emmean"
    elif "estimate" in emm.frame.columns:
        point_col = "estimate"
    elif "ratio" in emm.frame.columns:
        point_col = "ratio"
    else:
        point_col = next(
            (
                c
                for c in emm.frame.columns
                if isinstance(c, str) and c.endswith(".trend")
            ),
            None,
        )
    if point_col is not None:
        valid_rows = (
            np.isfinite(emm.frame[point_col].to_numpy(dtype=float))
            & np.isfinite(emm.frame["SE"].to_numpy(dtype=float))
            & np.all(np.isfinite(emm.linfct), axis=1)
        )
    else:
        valid_rows = np.all(np.isfinite(emm.linfct), axis=1)
    L_boot = emm.linfct[valid_rows]
    n_rows = L_boot.shape[0]
    alpha = 1.0 - level
    p_lo = alpha / 2.0
    p_hi = 1.0 - alpha / 2.0

    if n_rows == 0:
        # Every row was non-estimable; nothing to bootstrap. Leave the
        # frame's CIs as they were (typically NaN already).
        lower = np.empty(0)
        upper = np.empty(0)
    elif method == "streaming":
        # P² online quantile: constant memory regardless of n_samples.
        from pymmeans.quantile import P2Batch

        batch = P2Batch([p_lo, p_hi], n_rows)
        drawn = 0
        while drawn < n_samples:
            m = min(chunk, n_samples - drawn)
            beta_chunk = rv.rvs(size=m)
            mu_chunk = beta_chunk @ L_boot.T
            if offset_mean:
                mu_chunk = mu_chunk + offset_mean
            if (
                emm.type == "response"
                and info.family is not None
                and not bias_adjust
            ):
                # GLM response, plain inverse-link (no bias adjust).
                mu_chunk = info.family.link.inverse(mu_chunk)
            elif response_transform is not None:
                # the GLM bias-adjusted log-link
                # path now flows here too, via the new
                # ``response_transform = make_tran("log")`` branch
                # added above when bias_adjust=True and info.family
                # is not None.
                if bias_adjust:
                    mu_chunk = response_transform.bias_mean(mu_chunk, sigma_sq)
                else:
                    mu_chunk = response_transform.inverse(mu_chunk)
            batch.update_batch(mu_chunk)
            drawn += m
        quantiles = batch.values()
        lower = quantiles[0]
        upper = quantiles[1]
    else:
        all_mu = np.empty((n_samples, n_rows))
        drawn = 0
        while drawn < n_samples:
            m = min(chunk, n_samples - drawn)
            beta_chunk = rv.rvs(size=m)
            mu_chunk = beta_chunk @ L_boot.T
            if offset_mean:
                mu_chunk = mu_chunk + offset_mean
            if (
                emm.type == "response"
                and info.family is not None
                and not bias_adjust
            ):
                # GLM response, plain inverse-link (no bias adjust).
                mu_chunk = info.family.link.inverse(mu_chunk)
            elif response_transform is not None:
                # the GLM bias-adjusted log-link
                # path now flows here too, via the new
                # ``response_transform = make_tran("log")`` branch
                # added above when bias_adjust=True and info.family
                # is not None.
                if bias_adjust:
                    mu_chunk = response_transform.bias_mean(mu_chunk, sigma_sq)
                else:
                    mu_chunk = response_transform.inverse(mu_chunk)
            all_mu[drawn : drawn + m] = mu_chunk
            drawn += m
        lower = np.percentile(all_mu, 100.0 * p_lo, axis=0)
        upper = np.percentile(all_mu, 100.0 * p_hi, axis=0)

    new_frame = emm.frame.copy()
    # Scatter the valid-row percentile results back; non-estimable rows
    # stay NaN.
    lower_full = np.full(len(new_frame), np.nan, dtype=float)
    upper_full = np.full(len(new_frame), np.nan, dtype=float)
    lower_full[valid_rows] = lower
    upper_full[valid_rows] = upper
    new_frame["lower_cl"] = lower_full
    new_frame["upper_cl"] = upper_full

    # when the input was a ContrastResult, return a
    # ContrastResult with refreshed CI columns (preserve every other
    # dataclass field via dataclasses.replace).
    #
    # use the SAME ``dataclasses.replace`` path for
    # EMMResult so the metadata fields (``at`` / ``weights``
    # / ``adjust`` / ``df_method`` / ``_satt_cache``) survive the
    # parametric bootstrap. Previously the EMM branch manually
    # reconstructed ``EMMResult(...)`` with only a hand-picked
    # subset of fields and silently dropped ``at`` / ``weights``,
    # which then poisoned any downstream ``pairs(simple=)`` /
    # ``contrast(simple=)`` that consults those fields to rebuild
    # the grid. (The Satt/KR refusal earlier in this function already
    # prevents the ``df_method != "default"`` case from reaching
    # here, but ``_satt_cache`` and ``adjust`` are still meaningful
    # to preserve.)
    #
    # stamp ``df_method="bootstrap"`` so downstream
    # ``apply_satterthwaite`` / ``apply_kenward_roger`` can refuse a
    # post-bootstrap result. Without this, ``apply_satterthwaite(
    # bootstrap_ci(em))`` silently OVERWROTE the percentile CIs with
    # Satterthwaite t-CIs while keeping the bootstrap-corrupted point
    # estimate — the exact "silent inference-paradigm mixing" the
    # / / refusal symmetry was designed to
    # close, but in the reverse direction (bootstrap → Satt). previously
    # ``df_method`` was inherited as ``"default"`` through
    # ``dataclasses.replace``, so the Satt path saw nothing
    # unusual. Symmetric refusal now lives in ``_refuse_bootstrap`` in
    # ``satterthwaite.py``.
    from dataclasses import replace as _dc_replace
    return _dc_replace(
        emm, frame=new_frame, level=level, df_method="bootstrap",
    )


# ---------------------------------------------------------------------------
# (beyond-R-parity): non-parametric case bootstrap +
# permutation test for contrasts. R `emmeans` has no built-in for
# either; users typically reach for the separate `boot` package and
# stitch results back manually.
# ---------------------------------------------------------------------------


def _default_refit_fn(raw_result: Any) -> Callable:
    """Build a default case-bootstrap refit callable from a fitted
    statsmodels formula-API result.

    Extracts the formula and model class, then on each call recreates
    ``cls.from_formula(formula, data=new_data, **kwargs).fit()``. The
    family / link is preserved for GLMs. Status / weights / etc.
    fields that aren't recoverable from `from_formula` will need a
    custom `refit_fn`.
    """
    model = getattr(raw_result, "model", None)
    if model is None:
        raise ValueError(
            "Could not infer the original model from raw_result; "
            "supply a refit_fn=lambda df: ... callable explicitly."
        )
    cls = type(model)
    formula = getattr(model, "formula", None)
    if formula is None:
        raise ValueError(
            f"Could not extract formula from {cls.__name__}; supply "
            "a refit_fn=lambda df: ... callable explicitly."
        )
    # Preserve GLM family.
    family = getattr(model, "family", None)
    fit_kwargs = {}
    init_kwargs = {}
    if family is not None:
        init_kwargs["family"] = family
    # refuse to silently reconstruct a model whose
    # weights / offset / exposure were non-trivial — the default
    # refit path drops them and produces a bootstrap CI from an
    # UNWEIGHTED model (silently wrong inference). The user must
    # supply ``refit_fn=`` that preserves those options. We allow
    # constant (== first value) arrays because they're effectively
    # no-ops; only non-constant arrays trigger the refusal.
    import numpy as _np
    for attr in (
        "weights", "var_weights", "freq_weights", "offset", "exposure",
    ):
        val = getattr(model, attr, None)
        if val is None:
            continue
        try:
            arr = _np.asarray(val)
        except Exception:
            continue
        if arr.size == 0 or arr.ndim == 0:
            continue
        # All-same-value arrays are no-ops.
        try:
            same = _np.allclose(arr, arr.flat[0])
        except (TypeError, ValueError):
            same = False
        if not same:
            raise ValueError(
                f"bootstrap_ci(kind='case') cannot safely use the "
                f"default refit path on this {cls.__name__} fit: the "
                f"model carries a non-trivial '{attr}' array that "
                "``from_formula`` does not round-trip cleanly. "
                "Pass an explicit refit_fn= callable that rebuilds "
                f"the fit with the original {attr}, e.g.:\n\n"
                f" refit_fn=lambda d: smf.wls(formula, d, "
                f"weights=d['w']).fit()\n\n"
                "Without this, the bootstrap silently produces CIs "
                f"from a model that ignores '{attr}'."
            )

    def refit(new_data: Any) -> Any:
        new_fit = cls.from_formula(
            formula, data=new_data, **init_kwargs,
        ).fit(**fit_kwargs)
        return new_fit

    return refit


def _bootstrap_ci_case(
    obj: Any,
    n_samples: int,
    level: float | None,
    seed: int | None,
    refit_fn: Callable | None,
) -> Any:
    """Non-parametric case bootstrap: resample data with replacement,
    refit, recompute EMM/contrast at each draw.

    (beyond-R-parity). R `emmeans` lacks this — users do it
    manually via the `boot` package. pymmeans bakes it in.
    """
    from dataclasses import replace as _dc_replace

    from pymmeans.contrasts import (
        ContrastResult,
    )
    from pymmeans.contrasts import (
        contrast as _contrast,
    )
    from pymmeans.contrasts import (
        pairs as _pairs,
    )
    from pymmeans.emmeans import emmeans as _emmeans

    info = obj.model_info
    if info.data is None:
        raise ValueError(
            "case bootstrap requires the original data on "
            "model_info.data — not available (pickled or hand-built "
            "ModelInfo). Pass refit_fn= to override."
        )
    data = info.data
    n_obs = len(data)
    if n_obs == 0:
        raise ValueError("Cannot bootstrap a zero-row dataset.")

    if refit_fn is None:
        if info.raw_result is None:
            raise ValueError(
                "case bootstrap needs either raw_result on the "
                "model_info OR an explicit refit_fn= callable."
            )
        refit_fn = _default_refit_fn(info.raw_result)

    is_contrast = isinstance(obj, ContrastResult)
    level = level if level is not None else float(getattr(obj, "level", 0.95))

    # Figure out point-column and target / by / at for re-construction.
    if "emmean" in obj.frame.columns:
        point_col = "emmean"
    elif "estimate" in obj.frame.columns:
        point_col = "estimate"
    elif "ratio" in obj.frame.columns:
        point_col = "ratio"
    else:
        point_col = next(
            (
                c
                for c in obj.frame.columns
                if isinstance(c, str) and c.endswith(".trend")
            ),
            None,
        )
    if point_col is None:
        raise ValueError(
            f"Could not detect the value column on {type(obj).__name__}."
        )

    # read the stored ``method`` / ``method_args`` from
    # the ContrastResult. 's label-heuristic
    # (``" - "`` in every label → "pairwise" else "eff") silently
    # mis-classified ``trt.vs.ctrl`` / ``consec`` / ``poly`` /
    # ``interaction=`` / custom-callable results and produced
    # all-NaN CIs. With the metadata present, the bootstrap rebuilds
    # via the exact same contrast call as the original.
    contrast_method = None
    contrast_method_args: dict = {}
    if is_contrast:
        contrast_method = getattr(obj, "method", None)
        contrast_method_args = dict(getattr(obj, "method_args", {}) or {})
        if contrast_method is None:
            raise ValueError(
                "bootstrap_ci(kind='case') on a ContrastResult "
                "requires the ``method`` / ``method_args`` "
                "metadata on the result. The contrast appears to "
                "have been built before (or constructed by "
                "hand). Recompute the contrast with the current "
                "pymmeans, or pass refit_fn= that rebuilds the EMM "
                "+ contrast manually."
            )
        if contrast_method == "custom" and contrast_method_args.get("callable"):
            raise ValueError(
                "case bootstrap cannot rebuild a custom contrast "
                "from a callable ``method=`` — the callable closure "
                "may not be picklable / reproducible. Pass refit_fn= "
                "or rebuild the contrast with explicit coefs."
            )

    n_rows = len(obj.frame)
    draws = np.full((n_samples, n_rows), np.nan, dtype=float)
    rng = np.random.default_rng(seed)
    n_failed = 0
    # capture first failure (sibling of
    # which fixed the same anti-pattern in ``_bootstrap_ci_ml`` only).
    # Without this, ``_bootstrap_ci_case`` emits "Case bootstrap dropped
    # 100/100 unstable resamples" with no actionable hint about what
    # broke — and the user has to add their own logging to debug a
    # misbehaving refit_fn. (Symmetric with permutation_test, fixed
    # below.)
    first_failure: tuple[int, str] | None = None
    for i in range(n_samples):
        idx = rng.integers(0, n_obs, size=n_obs)
        sample = data.iloc[idx].reset_index(drop=True)
        try:
            new_fit = refit_fn(sample)

            # rbind dispatch. For a ContrastResult with
            # ``method="rbind"``, the rbind output stacks per-child
            # contrast results — each child has its own target /
            # by / at / weights / method. Rebuild each child
            # independently on the resample and concatenate.
            if is_contrast and contrast_method == "rbind":
                children = contrast_method_args.get("children", [])
                if not children:
                    n_failed += 1
                    continue
                draws_for_row: list[np.ndarray] = []
                child_failed = False
                for child in children:
                    c_target = child["target"] or None
                    c_by = child["by"] or None
                    c_at = child["at"]
                    c_weights = child["weights"] or "equal"
                    c_method = child["method"]
                    c_args = child["method_args"]
                    # read the child's stored
                    # ``bias_sigma`` (added it to the
                    # recipe). When the parent rbind result was
                    # built from bias-adjusted response contrasts,
                    # rebuild each child on LINK scale and then
                    # ``regrid_response(bias_adjust=True,
                    # sigma=child['bias_sigma'])`` — same pattern as
                    # for the non-rbind case-bootstrap
                    # path. Without this, the rbind rebuild called
                    # ``_emmeans(type='response')`` then
                    # ``_pairs(response_em)``, which refuses
                    # response-scale contrasts and blew up every draw.
                    c_bias_sigma = child.get("bias_sigma")
                    c_bias_adjust = bool(
                        getattr(obj, "bias_adjust", False)
                        and (c_bias_sigma is not None
                             or getattr(obj, "type", "link") == "response")
                    )
                    if c_method is None or c_target is None:
                        child_failed = True
                        break
                    try:
                        c_em = _emmeans(
                            new_fit, c_target,
                            by=c_by, level=level,
                            type=(
                                "link" if c_bias_adjust
                                else getattr(obj, "type", "link")
                            ),
                            at=c_at, weights=c_weights,
                        )
                        if c_method in ("pairwise", "revpairwise"):
                            c_obj = _pairs(
                                c_em, max_contrasts=None,
                                reverse=bool(c_args.get("reverse", False)),
                            )
                        elif c_method == "custom":
                            coefs = c_args.get("coefs")
                            if coefs is None:
                                child_failed = True
                                break
                            c_obj = _contrast(c_em, method=coefs)
                        elif c_method == "interaction":
                            c_obj = _contrast(
                                c_em,
                                interaction=c_args.get("interaction", []),
                            )
                        else:
                            ckw = {
                                k: v for k, v in c_args.items()
                                if k in ("ref",)
                            }
                            c_obj = _contrast(c_em, method=c_method, **ckw)
                        # apply the per-draw
                        # bias-corrected back-transform AFTER the
                        # link-scale contrast is built, using the
                        # child's stored sigma.
                        if c_bias_adjust:
                            from pymmeans.transforms import (
                                regrid_response as _regrid_response,
                            )
                            if c_bias_sigma is not None:
                                c_obj = _regrid_response(
                                    c_obj, bias_adjust=True,
                                    sigma=c_bias_sigma,
                                )
                            else:
                                c_obj = _regrid_response(
                                    c_obj, bias_adjust=True,
                                )
                        if len(c_obj.frame) != child["n_rows"]:
                            child_failed = True
                            break
                        draws_for_row.append(
                            c_obj.frame[point_col].to_numpy(dtype=float)
                        )
                    except Exception:
                        child_failed = True
                        break
                if child_failed:
                    n_failed += 1
                    continue
                stacked = np.concatenate(draws_for_row)
                if len(stacked) != n_rows:
                    n_failed += 1
                    continue
                draws[i] = stacked
                continue

            # Non-rbind path: rebuild single EMM with stored at/weights.
            # Rebuild EMM at same target / by.
            target = obj.target if not is_contrast else getattr(obj, "target", None)
            by = obj.by if not is_contrast else getattr(obj, "by", None)
            # forward stored `at` and `weights` so the
            # rebuilt EMM uses the SAME grid restrictions /
            # marginalisation scheme as the original. previously,
            # `bootstrap_ci(pairs(em with at={x: [10]}), kind="case")`
            # rebuilt at `x = mean(x)` and produced CIs centered ~30
            # units away from the point estimate — the exact silent-
            # metadata-regression class was meant to close.
            orig_at = getattr(obj, "at", None)
            orig_weights = getattr(obj, "weights", "equal") or "equal"
            # if the source EMM is a bias-adjusted
            # response-scale result (``type="response"``,
            # ``bias_adjust=True``), the rebuild must apply the same
            # Taylor-corrected back-transform on each draw. previously the
            # rebuild forwarded ``type="response"`` but ``emmeans()`` has
            # no ``bias_adjust=`` kwarg, so each draw produced unadjusted
            # response means — silently mixing bias-adjusted point
            # estimates with unadjusted bootstrap CIs. Rebuild on link
            # scale, then ``regrid_response(bias_adjust=True)``.
            _src_bias = bool(getattr(obj, "bias_adjust", False))
            _src_type = getattr(obj, "type", "link")
            _rebuild_type = "link" if _src_bias else _src_type
            new_em = _emmeans(
                new_fit,
                target if target else [c for c in obj.frame.columns
                                       if c not in (point_col, "SE", "df",
                                                    "lower_cl", "upper_cl",
                                                    "t_ratio", "p_value",
                                                    "contrast")][:1],
                by=by if by else None,
                level=level,
                type=_rebuild_type,
                at=orig_at,
                weights=orig_weights,
            )
            if _src_bias:
                # Re-apply the bias-corrected back-transform on each
                # draw at the same sigma (taken from the source's
                # ``bias_sigma`` if set, else ``info.scale``, which
                # ``regrid_response`` picks up by default).
                from pymmeans.transforms import regrid_response as _regrid_response
                _bs = getattr(obj, "bias_sigma", None)
                if _bs is not None:
                    new_em = _regrid_response(
                        new_em, bias_adjust=True, sigma=_bs,
                    )
                else:
                    new_em = _regrid_response(new_em, bias_adjust=True)
            if is_contrast:
                # dispatch on stored method metadata.
                if contrast_method in ("pairwise", "revpairwise"):
                    new_obj = _pairs(
                        new_em, max_contrasts=None,
                        reverse=bool(contrast_method_args.get("reverse", False)),
                    )
                elif contrast_method == "custom":
                    coefs = contrast_method_args.get("coefs")
                    if coefs is None:
                        n_failed += 1
                        continue
                    new_obj = _contrast(new_em, method=coefs)
                elif contrast_method == "interaction":
                    new_obj = _contrast(
                        new_em,
                        interaction=contrast_method_args.get("interaction", []),
                    )
                else:
                    # All named methods: pass through.
                    kwargs = {
                        k: v for k, v in contrast_method_args.items()
                        if k in ("ref",)
                    }
                    new_obj = _contrast(new_em, method=contrast_method, **kwargs)
                if len(new_obj.frame) != n_rows:
                    if first_failure is None:
                        first_failure = (
                            i,
                            f"row-count mismatch: rebuilt contrast has "
                            f"{len(new_obj.frame)} rows, expected "
                            f"{n_rows} (resample likely dropped a "
                            "factor level)",
                        )
                    n_failed += 1
                    continue
                draws[i] = new_obj.frame[point_col].to_numpy(dtype=float)
            else:
                if len(new_em.frame) != n_rows:
                    if first_failure is None:
                        first_failure = (
                            i,
                            f"row-count mismatch: rebuilt EMM has "
                            f"{len(new_em.frame)} rows, expected "
                            f"{n_rows} (resample likely dropped a "
                            "factor level)",
                        )
                    n_failed += 1
                    continue
                draws[i] = new_em.frame[point_col].to_numpy(dtype=float)
        except Exception as exc:
            if first_failure is None:
                first_failure = (i, repr(exc))
            n_failed += 1
            continue

    failure_hint = (
        f" First failure: resample #{first_failure[0]} → "
        f"{first_failure[1]}."
        if first_failure is not None
        else ""
    )
    if n_failed > n_samples // 2:
        raise RuntimeError(
            f"Case bootstrap failed on {n_failed}/{n_samples} draws — "
            "the refit_fn / data combination is unstable. Try a larger "
            "dataset, a simpler model, or pass refit_fn= explicitly."
            + failure_hint
        )
    if n_failed:
        import warnings as _w
        _w.warn(
            f"Case bootstrap dropped {n_failed}/{n_samples} unstable "
            "resamples (refit failures or rank-deficient draws). "
            "Percentiles computed over the remaining "
            f"{n_samples - n_failed} draws."
            + failure_hint,
            UserWarning,
            stacklevel=2,
        )

    alpha = 1.0 - level
    lower = np.nanpercentile(draws, 100.0 * (alpha / 2), axis=0)
    upper = np.nanpercentile(draws, 100.0 * (1.0 - alpha / 2), axis=0)
    new_frame = obj.frame.copy()
    new_frame["lower_cl"] = lower
    new_frame["upper_cl"] = upper

    return _dc_replace(obj, frame=new_frame, level=level)


def permutation_test(
    contrast_result: Any,
    n_permutations: int = 1000,
    seed: int | None = None,
    by_factor: str | None = None,
    refit_fn: Callable | None = None,
) -> pd.DataFrame:
    """Permutation-based p-values for a ``ContrastResult``.

    (beyond-R-parity). For each permutation:

      1. Shuffle the labels of the target factor (or the factor named
         in ``by_factor``) in the original data;
      2. Refit the model on the permuted data;
      3. Recompute the contrast statistics (t-ratios).

    The two-sided p-value for each contrast row is the fraction of
    permutation |t|s ≥ the observed |t|, with the Phipson & Smyth
    (2010) add-one correction ``p = (n_extreme + 1) / (n_valid + 1)``
    to guarantee a strictly positive estimate. Robust to mis-
    specified residual distributions and to small-sample bias in
    t-distribution p-values.

    Parameters
    ----------
    contrast_result
        ``ContrastResult`` from ``pairs`` / ``contrast``.
    n_permutations
        Number of label-shuffle permutations (default 1000).
    seed
        Optional ``numpy.random.default_rng`` seed for reproducibility.
    by_factor
        Permute the labels of this factor specifically. By default
        permutes the first factor in ``contrast_result.target``.
    refit_fn
        Same as in :func:`bootstrap_ci`. Default: extract from
        ``raw_result``.

    Returns
    -------
    pd.DataFrame
        Original ``contrast_result.frame`` with a new
        ``p_permutation`` column.
    """

    from pymmeans.contrasts import (
        ContrastResult,
    )
    from pymmeans.contrasts import (
        contrast as _contrast,
    )
    from pymmeans.contrasts import (
        pairs as _pairs,
    )
    from pymmeans.emmeans import emmeans as _emmeans

    if not isinstance(contrast_result, ContrastResult):
        raise TypeError(
            "permutation_test requires a ContrastResult (from "
            f"pairs/contrast); got {type(contrast_result).__name__}."
        )
    info = contrast_result.model_info
    if info.data is None:
        raise ValueError(
            "permutation_test requires model_info.data — not available "
            "(pickled or hand-built ModelInfo)."
        )
    data = info.data
    target = contrast_result.target
    if not target:
        raise ValueError(
            "permutation_test needs a target factor (the contrast's "
            "linfct must come from a target). Hand-built contrasts "
            "are not supported."
        )
    factor_to_permute = by_factor or target[0]
    if factor_to_permute not in data.columns:
        raise ValueError(
            f"by_factor={factor_to_permute!r} is not a column in "
            f"the model data ({list(data.columns)})."
        )

    if refit_fn is None:
        if info.raw_result is None:
            raise ValueError(
                "permutation_test needs either raw_result on the "
                "model_info OR an explicit refit_fn= callable."
            )
        refit_fn = _default_refit_fn(info.raw_result)

    # read stored contrast method metadata instead of
    # guessing from labels. 's `" - " in label` heuristic
    # mis-classified ``trt.vs.ctrl`` / ``consec`` / ``interaction=``
    # / custom contrasts and produced "all permutations failed"
    # RuntimeErrors.
    contrast_method = getattr(contrast_result, "method", None)
    contrast_method_args = dict(getattr(contrast_result, "method_args", {}) or {})
    if contrast_method is None:
        raise ValueError(
            "permutation_test requires the ``method`` / "
            "``method_args`` metadata on the ContrastResult. The "
            "contrast appears to have been built before or "
            "constructed by hand. Recompute via pairs / contrast on "
            "current pymmeans, or pass refit_fn= that rebuilds the "
            "contrast manually."
        )
    if contrast_method == "custom" and contrast_method_args.get("callable"):
        raise ValueError(
            "permutation_test cannot rebuild a callable custom "
            "contrast. Pass refit_fn= or rebuild with explicit coefs."
        )

    observed_t = np.abs(
        contrast_result.frame["t_ratio"].to_numpy(dtype=float)
    )
    n_rows = len(contrast_result.frame)
    perm_extreme = np.zeros(n_rows, dtype=int)
    rng = np.random.default_rng(seed)
    n_failed = 0
    # capture first failure (sibling of
    # which fixed the ML case-bootstrap version of this anti-pattern,
    # and of which fixed the statsmodels case-bootstrap
    # version). Without this, a misbehaving refit_fn or a degenerate
    # permutation produces "permutation_test failed on N/N
    # permutations" with no actionable hint.
    first_failure: tuple[int, str] | None = None
    for _perm_idx in range(n_permutations):
        permuted_data = data.copy()
        permuted_data[factor_to_permute] = rng.permutation(
            permuted_data[factor_to_permute].to_numpy()
        )
        try:
            new_fit = refit_fn(permuted_data)

            # (sibling): rbind dispatch for permutation
            # tests. Each child contributes its own |t| stack.
            if contrast_method == "rbind":
                children = contrast_method_args.get("children", [])
                if not children:
                    n_failed += 1
                    continue
                t_stacks: list[np.ndarray] = []
                ch_failed = False
                for child in children:
                    c_target = child["target"] or None
                    c_by = child["by"] or None
                    c_at = child["at"]
                    c_weights = child["weights"] or "equal"
                    c_method = child["method"]
                    c_args = child["method_args"]
                    # sibling: read child's
                    # bias_sigma + rebuild on link scale + regrid
                    # per-child. Mirrors the case-bootstrap fix
                    # above. Without this, permutation_test on a
                    # bias-adjusted response rbind blew up every
                    # permutation.
                    c_bias_sigma = child.get("bias_sigma")
                    c_bias_adjust = bool(
                        getattr(contrast_result, "bias_adjust", False)
                        and (c_bias_sigma is not None
                             or getattr(contrast_result, "type", "link") == "response")
                    )
                    if c_method is None or c_target is None:
                        ch_failed = True
                        break
                    try:
                        c_em = _emmeans(
                            new_fit, c_target,
                            by=c_by,
                            level=getattr(contrast_result, "level", 0.95),
                            type=(
                                "link" if c_bias_adjust
                                else getattr(contrast_result, "type", "link")
                            ),
                            at=c_at, weights=c_weights,
                        )
                        if c_method in ("pairwise", "revpairwise"):
                            c_obj = _pairs(
                                c_em, max_contrasts=None,
                                reverse=bool(c_args.get("reverse", False)),
                            )
                        elif c_method == "custom":
                            coefs = c_args.get("coefs")
                            if coefs is None:
                                ch_failed = True
                                break
                            c_obj = _contrast(c_em, method=coefs)
                        elif c_method == "interaction":
                            c_obj = _contrast(
                                c_em,
                                interaction=c_args.get("interaction", []),
                            )
                        else:
                            ckw = {
                                k: v for k, v in c_args.items()
                                if k in ("ref",)
                            }
                            c_obj = _contrast(
                                c_em, method=c_method, **ckw
                            )
                        if c_bias_adjust:
                            from pymmeans.transforms import (
                                regrid_response as _regrid_response,
                            )
                            if c_bias_sigma is not None:
                                c_obj = _regrid_response(
                                    c_obj, bias_adjust=True,
                                    sigma=c_bias_sigma,
                                )
                            else:
                                c_obj = _regrid_response(
                                    c_obj, bias_adjust=True,
                                )
                        if len(c_obj.frame) != child["n_rows"]:
                            ch_failed = True
                            break
                        t_stacks.append(
                            np.abs(c_obj.frame["t_ratio"].to_numpy(dtype=float))
                        )
                    except Exception:
                        ch_failed = True
                        break
                if ch_failed:
                    n_failed += 1
                    continue
                perm_t = np.concatenate(t_stacks)
                if len(perm_t) != n_rows:
                    n_failed += 1
                    continue
                with np.errstate(invalid="ignore"):
                    perm_extreme += (perm_t >= observed_t).astype(int)
                continue

            # Non-rbind path: rebuild single EMM with at/weights.
            orig_at = getattr(contrast_result, "at", None) or None
            orig_weights = getattr(contrast_result, "weights", None) or "equal"
            new_em = _emmeans(
                new_fit, target,
                by=contrast_result.by if contrast_result.by else None,
                level=getattr(contrast_result, "level", 0.95),
                type=getattr(contrast_result, "type", "link"),
                at=orig_at,
                weights=orig_weights,
            )
            # Rebuild via stored method.
            if contrast_method in ("pairwise", "revpairwise"):
                new_ct = _pairs(
                    new_em, max_contrasts=None,
                    reverse=bool(contrast_method_args.get("reverse", False)),
                )
            elif contrast_method == "custom":
                coefs = contrast_method_args.get("coefs")
                if coefs is None:
                    n_failed += 1
                    continue
                new_ct = _contrast(new_em, method=coefs)
            elif contrast_method == "interaction":
                new_ct = _contrast(
                    new_em,
                    interaction=contrast_method_args.get("interaction", []),
                )
            else:
                kwargs = {
                    k: v for k, v in contrast_method_args.items()
                    if k in ("ref",)
                }
                new_ct = _contrast(new_em, method=contrast_method, **kwargs)
            if len(new_ct.frame) != n_rows:
                if first_failure is None:
                    first_failure = (
                        _perm_idx,
                        f"row-count mismatch: rebuilt contrast has "
                        f"{len(new_ct.frame)} rows, expected {n_rows} "
                        "(permutation likely dropped a factor level)",
                    )
                n_failed += 1
                continue
            perm_t = np.abs(
                new_ct.frame["t_ratio"].to_numpy(dtype=float)
            )
            with np.errstate(invalid="ignore"):
                perm_extreme += (perm_t >= observed_t).astype(int)
        except Exception as exc:
            if first_failure is None:
                first_failure = (_perm_idx, repr(exc))
            n_failed += 1
            continue

    n_valid = n_permutations - n_failed
    failure_hint = (
        f" First failure: permutation #{first_failure[0]} → "
        f"{first_failure[1]}."
        if first_failure is not None
        else ""
    )
    if n_valid < n_permutations // 2:
        raise RuntimeError(
            f"permutation_test failed on {n_failed}/{n_permutations} "
            "permutations — refit_fn / data combination unstable."
            + failure_hint
        )
    # Add-one correction (Phipson & Smyth 2010): (n_extreme + 1) / (n_valid + 1)
    p_perm = (perm_extreme + 1) / (n_valid + 1)
    out = contrast_result.frame.copy()
    out["p_permutation"] = p_perm
    return out


def _bootstrap_ci_ml(
    em_ml: Any,
    n_samples: int,
    level: float | None,
    seed: int | None,
    refit_fn: Callable | None,
) -> Any:
    """Case-bootstrap CIs for an :class:`MLEMMResult`.

    (marquee beyond-R-parity feature). For each resample:

      1. Resample rows of ``info.data`` with replacement.
      2. Either:
         - call ``refit_fn(resampled_data)`` to get a fresh predict_fn
           (recommended — train a NEW model on the resample), OR
         - reuse the original ``predict_fn`` (assumes the model is
           fixed and we're bootstrapping the population, not the
           model — useful for pre-trained models where you can't
           refit).
      3. Recompute the prediction-averaged EMM at every grid cell.
      4. Aggregate percentiles into the CI columns.

    Returns a new :class:`MLEMMResult` with refreshed
    ``lower_cl`` / ``upper_cl`` (and a populated ``SE`` from the
    bootstrap-standard-deviation).
    """
    from dataclasses import replace as _dc_replace

    from pymmeans.ml import ml_emmeans

    info = em_ml.ml_info
    if info.data is None or len(info.data) == 0:
        raise ValueError(
            "ML case bootstrap requires non-empty info.data."
        )
    data = info.data
    n_obs = len(data)
    level = level if level is not None else em_ml.level

    # Resolve refit_fn: caller > info.refit_fn > fixed-model fallback
    if refit_fn is None:
        refit_fn = info.refit_fn

    n_rows = len(em_ml.frame)
    draws = np.full((n_samples, n_rows), np.nan, dtype=float)
    rng = np.random.default_rng(seed)
    n_failed = 0
    # capture the FIRST failing resample (index +
    # repr of the raised exception) so the eventual
    # ``RuntimeError`` / ``UserWarning`` points the user at a
    # concrete failure rather than the generic "all dropped" message.
    # Without this, debugging a misbehaving ``refit_fn`` required
    # the user to add their own logging — the previous error
    # swallowed every exception silently in the ``except: continue``.
    first_failure: tuple[int, str] | None = None
    for i in range(n_samples):
        idx = rng.integers(0, n_obs, size=n_obs)
        sample = data.iloc[idx].reset_index(drop=True)
        try:
            if refit_fn is not None:
                new_predict = refit_fn(sample)
                # PREFER ``.predict()`` over ``__call__``
                # when both are present. Sklearn-style fit results
                # almost always have a ``predict`` method, and some
                # objects also implement ``__call__`` (custom wrappers,
                # functools.partial-wrapped objects). The
                # ``callable() first`` ordering would pick __call__ →
                # wrong predictions. Detect "is fitted model" via the
                # presence of ``predict`` and fall back to ``callable``
                # only when no ``predict`` attribute exists.
                if (
                    not callable(new_predict)
                    or hasattr(new_predict, "predict")
                ):
                    if hasattr(new_predict, "predict"):
                        new_predict = new_predict.predict
                    else:
                        raise TypeError(
                            "refit_fn must return a callable "
                            "predict_fn or a model with .predict()."
                        )
                # Build a fresh MLPredictInfo on the resampled data
                # so the EMM averaging marginalizes correctly.
                from pymmeans.ml import MLPredictInfo
                new_info = MLPredictInfo(
                    predict_fn=new_predict,
                    data=sample,
                    # pass the FROZEN level dict
                    # (info.factor_levels), not the raw `factors`
                    # spec. Without this, every resample that misses
                    # a rare level shrank the EMM grid → row-count
                    # mismatch → silently dropped draws (66/500 on
                    # a 10/18/2 three-level rare factor).
                    factors=info.factor_levels,
                    numerics=info.numerics,
                    response=info.response,
                )
            else:
                # No refit: just use the resampled data for averaging,
                # keeping the same predict_fn. This is "fixed-model
                # population bootstrap" — variance reflects sampling
                # variability of the average over the data
                # distribution but NOT the model's training noise.
                from pymmeans.ml import MLPredictInfo
                new_info = MLPredictInfo(
                    predict_fn=info.predict_fn,
                    data=sample,
                    # pass the FROZEN level dict
                    # (info.factor_levels), not the raw `factors`
                    # spec. Without this, every resample that misses
                    # a rare level shrank the EMM grid → row-count
                    # mismatch → silently dropped draws (66/500 on
                    # a 10/18/2 three-level rare factor).
                    factors=info.factor_levels,
                    numerics=info.numerics,
                    response=info.response,
                )
            new_em = ml_emmeans(
                new_info, em_ml.target,
                by=em_ml.by if em_ml.by else None,
                at=em_ml.at, level=level,
            )
            if len(new_em.frame) != n_rows:
                if first_failure is None:
                    first_failure = (
                        i,
                        f"row-count mismatch: rebuilt EMM has "
                        f"{len(new_em.frame)} rows, expected {n_rows} "
                        "(usually means the resample dropped a factor "
                        "level)",
                    )
                n_failed += 1
                continue
            draws[i] = new_em.frame["emmean"].to_numpy(dtype=float)
        except Exception as exc:
            if first_failure is None:
                first_failure = (i, repr(exc))
            n_failed += 1
            continue

    failure_hint = (
        f" First failure: resample #{first_failure[0]} → "
        f"{first_failure[1]}."
        if first_failure is not None
        else ""
    )
    if n_failed > n_samples // 2:
        raise RuntimeError(
            f"ML case bootstrap failed on {n_failed}/{n_samples} "
            "resamples. Check refit_fn (it should return a fresh "
            "predict_fn or a model with .predict()) and that the "
            "resampled data still covers all factor levels."
            + failure_hint
        )
    if n_failed:
        import warnings as _w
        _w.warn(
            f"ML case bootstrap dropped {n_failed}/{n_samples} "
            "unstable resamples. Percentiles computed over the "
            f"remaining {n_samples - n_failed} draws."
            + failure_hint,
            UserWarning,
            stacklevel=2,
        )

    alpha = 1.0 - level
    lower = np.nanpercentile(draws, 100.0 * (alpha / 2), axis=0)
    upper = np.nanpercentile(draws, 100.0 * (1.0 - alpha / 2), axis=0)
    se = np.nanstd(draws, axis=0, ddof=1)

    new_frame = em_ml.frame.copy()
    new_frame["SE"] = se
    new_frame["lower_cl"] = lower
    new_frame["upper_cl"] = upper
    # stamp ``df_method="bootstrap"`` so summary /
    # confint / test recognise the result as bootstrap-derived and
    # apply the /55 preservation / refusal logic. previously
    # ``summary(ml_em_b)`` silently overwrote percentile bounds with
    # asymptotic Wald CIs and emitted z-tests.
    return _dc_replace(
        em_ml, frame=new_frame, level=level, df_method="bootstrap",
    )
