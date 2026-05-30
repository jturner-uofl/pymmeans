"""Parametric-bootstrap model comparison for nested MixedLM fits.

Python port of ``pbkrtest::PBmodcomp`` (Halekoh & Højsgaard 2014,
59(9), doi:10.18637/jss.v059.i09). Tests the null hypothesis
that the smaller (nested) of two linear mixed models is sufficient
against the alternative that the larger model is needed, via the
parametric bootstrap of the likelihood-ratio statistic under H_0.

Algorithm
---------

Given two nested ``statsmodels`` ``MixedLM`` fits ``large`` and
``small`` (with ``small`` strictly nested in ``large``):

1. Refit both as **maximum likelihood** (REML log-likelihoods are not
   comparable across different fixed-effect designs, so the LRT for
   fixed-effect tests requires ML). ``pbmodcomp`` does this
   internally when either fit was REML.
2. Compute the observed likelihood-ratio statistic
   ``LRT_obs = -2 (logL(small) - logL(large))``.
3. For ``b = 1, ..., n_sim``:

   * Simulate ``y_b`` from the fitted ``small`` model (parametric
     bootstrap of the null distribution),
     ``y_b | g ~ N(X_g β̂_small, Z_g G_small Z_g' + σ²_small I)``,
     drawing each group independently.
   * Refit both ``small`` and ``large`` to ``y_b`` as ML and compute
     ``LRT_b = -2 (logL(small_b) - logL(large_b))``.
   * Failed-to-converge iterations are skipped and counted.

4. Report

   ``p_value = (1 + #{ LRT_b ≥ LRT_obs }) / (1 + n_sim_ok)``

   the standard one-sided bootstrap p-value with the
   :math:`+1` continuity correction (Davison & Hinkley 1997).

The result also carries the simulated LRT distribution so the user
can inspect histograms, quantiles, or recompute p-values at
alternative reference statistics (e.g. KR-corrected F).

Scope and relationship to pbkrtest
----------------------------------

This is the second pbkrtest leaf to be ported to ``pymmeans`` (after
``vcovAdj`` / Kenward-Roger; see :mod:`pymmeans.satterthwaite`). The
third leaf — pbkrtest's higher-order ``ddf_Lb`` Kenward-Roger
degrees-of-freedom corrections — remains a v0.2 milestone (the
``apply_kenward_roger`` df currently matches ``ddf_Lb`` at ~1 %
relative error from the leading-order delta-method approximation
alone).

Examples
--------

>>> import warnings # doctest: +SKIP
>>> import statsmodels.regression.mixed_linear_model as mlm # doctest: +SKIP
>>> from statsmodels.datasets import get_rdataset # doctest: +SKIP
>>> from pymmeans import pbmodcomp # doctest: +SKIP
>>> sl = get_rdataset("sleepstudy", "lme4").data # doctest: +SKIP
>>> with warnings.catch_warnings(): # doctest: +SKIP
... warnings.simplefilter("ignore")
... large = mlm.MixedLM.from_formula( # doctest: +SKIP
... "Reaction ~ Days", groups="Subject", data=sl,
... ).fit(reml=False)
... small = mlm.MixedLM.from_formula( # doctest: +SKIP
... "Reaction ~ 1", groups="Subject", data=sl,
... ).fit(reml=False)
>>> res = pbmodcomp(large, small, n_sim=200, seed=42) # doctest: +SKIP
>>> res.p_value # doctest: +SKIP
0.0049...
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PBmodcompResult:
    """Result of :func:`pbmodcomp`.

    Attributes
    ----------
    lrt_obs
        Observed likelihood-ratio statistic
        ``-2 (logL(small) - logL(large))`` on the ML-refit pair.
        Positive when the larger model fits better.
    lrt_dist
        Simulated LRT statistics from the bootstrap (length
        ``n_sim_ok``). Already filtered for converged iterations;
        failed iterations are dropped, not silently treated as zero.
    p_value
        Parametric-bootstrap p-value with the +1 continuity
        correction (Davison & Hinkley 1997, eq. 4.10):
        ``(1 + sum(lrt_dist >= lrt_obs)) / (1 + n_sim_ok)``.
    n_sim
        Number of bootstrap iterations requested.
    n_sim_ok
        Number of iterations where both refits converged
        (``≤ n_sim``).
    chi2_p_value
        Asymptotic chi-squared p-value for the same LRT, with
        degrees of freedom ``n_params_large - n_params_small``.
        Provided for direct comparison to the asymptotic test
        (which the parametric bootstrap exists to improve on at
        small sample sizes).
    df
        ``n_params_large - n_params_small`` — the parameter-count
        difference used for the chi-squared comparison.
    refit_reml
        ``True`` when either input fit was REML and the function
        internally refit both as ML before computing the LRT
        (printed as a note in :meth:`summary`).
    """

    lrt_obs: float
    lrt_dist: np.ndarray
    p_value: float
    n_sim: int
    n_sim_ok: int
    chi2_p_value: float
    df: int
    refit_reml: bool = False
    _failed_iters: list[str] = field(default_factory=list, repr=False)

    def summary(self) -> str:
        """Return a one-block text summary, R-pbkrtest-style."""
        lines = [
            "Parametric-bootstrap model comparison "
            "(pymmeans port of pbkrtest::PBmodcomp)",
            "",
            f" LRT_obs = {self.lrt_obs:.4f} (df = {self.df})",
            f" p (bootstrap) = {self.p_value:.4f} "
            f"({self.n_sim_ok}/{self.n_sim} sims converged)",
            f" p (chi^2) = {self.chi2_p_value:.4f} "
            "(asymptotic; bootstrap is preferred at small n)",
        ]
        if self.refit_reml:
            lines.append(
                " Note: both fits were silently refit as ML "
                "(REML log-likelihoods are not comparable)."
            )
        return "\n".join(lines)


def _check_pbmodcomp_inputs(large: Any, small: Any) -> None:
    """Validate that two fits are eligible for ``pbmodcomp``.

    (careful-implementation pass): refuse fits with
    ``vc_formula``-style variance components (``model.k_vc > 0``).
    ``_simulate_from_mixedlm`` only samples from the ``cov_re`` part
    of the marginal covariance; ignoring ``exog_vc`` would silently
    draw from the wrong null distribution. This mirrors the
    ``kenward_roger_vcov`` refusal for the same reason.

    Also raises ``TypeError`` on inputs that aren't ``MixedLM`` fits
    at all.
    """
    for label, fit in [("large", large), ("small", small)]:
        if not hasattr(fit, "model") or not hasattr(fit, "fe_params"):
            raise TypeError(
                f"pbmodcomp(): the `{label}` argument is not a "
                "statsmodels MixedLM fit (expected a "
                "MixedLMResults-like object with .model and "
                "fe_params attributes)."
            )
        model = fit.model
        if not hasattr(model, "exog_re") or not hasattr(model, "groups"):
            raise TypeError(
                f"pbmodcomp(): the `{label}` argument's model has no "
                "exog_re / groups — it does not look like a MixedLM. "
                "pbmodcomp is defined only for statsmodels.MixedLM "
                "fits."
            )
        k_vc = int(getattr(model, "k_vc", 0))
        if k_vc:
            raise NotImplementedError(
                f"pbmodcomp(): the `{label}` fit has {k_vc} "
                "vc_formula variance component(s) outside cov_re; "
                "this is not yet supported because the bootstrap "
                "would only sample from the cov_re part and ignore "
                "the additional variance components. Refit with all "
                "random effects in re_formula=, or use a different "
                "test."
            )


def _simulate_from_mixedlm(
    fit: Any, rng: np.random.Generator
) -> np.ndarray:
    """Sample y from the marginal distribution implied by ``fit``.

    Per group g, draws
        y_g ~ N(X_g β̂, Z_g G Z_g' + σ̂² I_{n_g})
    independently of the other groups. This is the *marginal*
    parametric bootstrap of the null distribution (the random
    effects ``u_g`` are integrated out into the marginal covariance
    ``V_g``). Mathematically equivalent to the *joint* sample
    (``u_g ~ N(0, G); ε_g ~ N(0, σ²I); y_g = X_g β + Z_g u_g + ε_g``);
    we use the marginal version because the per-group Cholesky of
    ``V_g`` is what the rest of the KR / Satterthwaite code already
    builds.

    Boundary fallback: when ``V_g`` is exactly singular (e.g.
    ``cov_re`` and ``sigma_sq`` both estimated at zero) the Cholesky
    raises ``LinAlgError``. We fall back to an eigen-decomposition
    ``V_g = Q Λ Qᵀ`` and use ``L = Q · diag(sqrt(max(Λ, 0)))`` as
    the sampling factor — ``L Lᵀ = V_g``, so the distribution of
    ``L z`` for ``z ~ N(0, I_{n_g})`` is exactly ``N(0, V_g)``.
    Tiny negative eigenvalues from FP noise are clipped to zero.

    Notes
    -----
    Sums each group's draws into the appropriate slots of ``y_sim``
    according to the original ``groups`` ordering — so the returned
    vector has the same row order as ``fit.model.endog``.
    """
    model = fit.model
    X = np.asarray(model.exog, dtype=float)
    Z = np.asarray(model.exog_re, dtype=float)
    groups = np.asarray(model.groups)
    group_ids = np.unique(groups)
    beta = np.asarray(fit.fe_params, dtype=float)
    G = np.asarray(fit.cov_re, dtype=float)
    sigma_sq = float(fit.scale)

    mu = X @ beta
    y_sim = np.zeros(X.shape[0], dtype=float)
    for g_id in group_ids:
        m = groups == g_id
        Z_g = Z[m]
        n_g = int(m.sum())
        # V_g = Z_g G Z_g' + σ² I. PSD by construction; can be
        # rank-deficient only at the parameter-space boundary.
        V_g = Z_g @ G @ Z_g.T + sigma_sq * np.eye(n_g)
        try:
            L = np.linalg.cholesky(V_g)
        except np.linalg.LinAlgError:
            # Boundary: random-effect variance estimated as exactly
            # zero (or numerical-noise singular). Eigen-decompose
            # and zero-out any negative eigenvalues from FP slop.
            # The result L satisfies L L^T = V_g exactly when V_g
            # is PSD, so y_g = μ_g + L z has Cov = V_g for
            # z ~ N(0, I_{n_g}).
            w, V = np.linalg.eigh(V_g)
            w = np.clip(w, 0.0, None)
            L = V * np.sqrt(w)
        eps = L @ rng.standard_normal(n_g)
        y_sim[m] = mu[m] + eps
    return y_sim


def _is_reml_fit(fit: Any) -> bool:
    """Return True if ``fit`` looks like a REML fit (safer-side default).

    statsmodels stores the REML flag on both ``fit.model.reml`` and
    ``fit.reml`` after fitting (verified empirically on statsmodels
    0.14, where the two are always set in lockstep). Both are
    consulted here with a "True if *either* says REML" rule and a
    True default for missing attributes. The motivation is
    correctness-safe behaviour under future statsmodels API drift:

    - If a future statsmodels release stores only one of the two
      flags (or names it differently), the safer-side OR returns
      True → we refit as ML, paying a small extra-compute cost
      rather than silently using REML log-likelihoods in the LRT.
    - If a user-constructed wrapper drops one of the two flags
      while keeping the other, same story: we refit, no silent
      REML LRT.
    - When both flags are explicitly False (a normal
      ``fit(reml=False)`` call), we correctly skip the refit.

    a previous version used
    AND between the two flags, which would *miss* the case where
    one flag indicated REML and the other indicated ML — exactly
    the silent-REML-LRT bug the cleanup was meant to
    close. The current OR-with-True-default is the safer-side
    choice the docstring now promises.
    """
    model = getattr(fit, "model", None)
    model_reml = getattr(model, "reml", True) if model is not None else True
    fit_reml = getattr(fit, "reml", True)
    return bool(model_reml) or bool(fit_reml)


def _refit_mixedlm_ml(fit: Any, y_new: np.ndarray) -> Any:
    """Refit a ``MixedLM`` to ``y_new`` as maximum likelihood.

    Bypasses the formula API by reconstructing the model from the
    fitted ``model.exog / exog_re / groups`` arrays — avoids the
    pickle / re-parsing cost of patsy and works for fits where the
    original ``data`` frame is no longer in scope. Always uses
    ``reml=False`` regardless of the input fit's REML flag (the
    caller is responsible for passing an ML-eligible fit; see
    ``_ensure_ml``).

    ``use_sqrt`` was previously
    not propagated from the original model. Both parameterisations
    span the same likelihood surface so the ML *maximum* is
    unchanged, but the optimiser path differs and the original
    user's choice deserves to survive a refit. ``exog_vc`` is also
    propagated here for future-proofing, though the public
    :func:`pbmodcomp` entry point still refuses fits with
    ``k_vc > 0`` (see :func:`_check_pbmodcomp_inputs`) since the
    bootstrap simulator does not yet sample from
    ``vc_formula``-style variance components.
    """
    import statsmodels.regression.mixed_linear_model as mlm

    model = fit.model
    kwargs: dict[str, Any] = {
        "endog": np.asarray(y_new, dtype=float),
        "exog": np.asarray(model.exog, dtype=float),
        "groups": np.asarray(model.groups),
        "exog_re": np.asarray(model.exog_re, dtype=float),
        "use_sqrt": bool(getattr(model, "use_sqrt", True)),
    }
    exog_vc = getattr(model, "exog_vc", None)
    if exog_vc is not None:
        kwargs["exog_vc"] = exog_vc
    new_model = mlm.MixedLM(**kwargs)
    return new_model.fit(reml=False, method="lbfgs", disp=False)


def _ensure_ml(fit: Any) -> tuple[Any, bool]:
    """Return ``(ml_fit, was_reml)`` so the LRT uses ML log-likelihoods.

    The likelihood-ratio test for nested mixed models with *different*
    fixed-effect designs is only valid using ML (REML log-likelihoods
    drop the determinant of X' V⁻¹ X, which differs across nestings).
    ``pbmodcomp`` therefore refits both inputs as ML if either was
    REML, matching pbkrtest's silent-refit behaviour.

    (careful-implementation pass): cleaned up the
    detection logic. Pre-the code AND-combined model.reml
    and fit.reml then OR-ed in an obscure ``_use_reml`` flag — that
    short-circuited to ML in the rare case where only one of the
    two flags was True, which would have silently used the REML
    log-likelihoods for the LRT. We now ask one focused question
    via :func:`_is_reml_fit`: is the fit REML? If yes, refit as ML.
    """
    if not _is_reml_fit(fit):
        return fit, False
    refit = _refit_mixedlm_ml(fit, np.asarray(fit.model.endog, dtype=float))
    return refit, True


def _n_fixed_params(fit: Any) -> int:
    """Return the number of fixed-effect parameters (rank of X)."""
    return int(np.asarray(fit.fe_params).size)


def _n_params_total(fit: Any) -> int:
    """Return the total number of free parameters in a MixedLM fit.

    For ``MixedLM`` this is ``p + k_re(k_re+1)/2 + 1`` (fixed effects
    + unique elements of cov_re + residual variance). Used as the
    chi-squared degrees of freedom for the LRT.
    """
    p = _n_fixed_params(fit)
    k_re = int(np.asarray(fit.cov_re).shape[0])
    return p + k_re * (k_re + 1) // 2 + 1


def _single_iteration(
    iter_seed: np.random.SeedSequence,
    small_ml: Any,
    large_ml: Any,
) -> tuple[float | None, str | None]:
    """Run a single bootstrap iteration.

    Returns
    -------
    (lrt_b, error) tuple
        ``(lrt_b, None)`` on success; ``(None, error_message)`` on
        any kind of refit failure. The tuple form lets a parallel
        backend collect both successes and failure messages without
        a separate sync barrier.
    """
    rng = np.random.default_rng(iter_seed)
    try:
        y_b = _simulate_from_mixedlm(small_ml, rng)
    except (np.linalg.LinAlgError, ValueError, RuntimeError) as exc:
        return None, f"simulate: {type(exc).__name__}: {exc}"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            small_b = _refit_mixedlm_ml(small_ml, y_b)
            large_b = _refit_mixedlm_ml(large_ml, y_b)
    except (np.linalg.LinAlgError, ValueError, RuntimeError) as exc:
        return None, f"refit: {type(exc).__name__}: {exc}"
    if not (
        getattr(small_b, "converged", True)
        and getattr(large_b, "converged", True)
    ):
        return None, "optimiser did not report converged"
    # Guard against pathological negative LRT from optimiser noise.
    lrt_b = max(-2.0 * (float(small_b.llf) - float(large_b.llf)), 0.0)
    return lrt_b, None


def pbmodcomp(
    large: Any,
    small: Any,
    n_sim: int = 1000,
    seed: int | None = None,
    silent: bool = False,
    n_jobs: int = 1,
) -> PBmodcompResult:
    """Parametric-bootstrap nested-model comparison for ``MixedLM``.

    Python port of ``pbkrtest::PBmodcomp``. Tests
    ``H_0`` (``small`` is sufficient) against
    ``H_1`` (``large`` is needed) by drawing ``n_sim`` parametric
    bootstrap replicates from the fitted ``small`` model, refitting
    both models to each replicate, and comparing the observed LRT
    statistic to the bootstrap distribution under ``H_0``.

    Parameters
    ----------
    large, small
        Two ``statsmodels.regression.mixed_linear_model.MixedLMResults``
        fits with ``small`` strictly nested in ``large``. Nesting is
        not verified automatically — the user is responsible for
        passing genuinely nested models (matching pbkrtest's
        behaviour). If either fit was REML, both are silently refit
        as ML internally (REML log-likelihoods are not comparable
        across different fixed-effect designs). Fits with
        ``vc_formula`` variance components outside ``cov_re``
        (``model.k_vc > 0``) are refused — the bootstrap simulator
        would ignore them, drawing from the wrong null.
    n_sim
        Number of parametric bootstrap iterations (default 1000;
        must be ``>= 1``). ``pbkrtest::PBmodcomp`` defaults to
        1000 as well.
    seed
        Optional seed for the per-iteration RNG via
        ``numpy.random.SeedSequence``. Each iteration draws from
        an independent child sequence, so results are reproducible
        regardless of parallel execution order (``n_jobs > 1``).
    silent
        When ``True``, suppress the REML-refit and low-convergence
        ``UserWarning``\\ s.
    n_jobs
        Number of parallel workers (default 1, serial). Values
        ``> 1`` use ``joblib.Parallel`` over a process pool.
        ``-1`` means "all available cores". When ``joblib`` is not
        installed, anything other than ``1`` raises ``ImportError``
        with installation instructions. Each worker refits both
        models per iteration — speedup is roughly linear in
        ``n_jobs`` until the per-iteration cost is dominated by
        process-pool overhead (typically below 50 ms per fit).

    Returns
    -------
    PBmodcompResult
        Object carrying the observed LRT, the simulated null
        distribution, the bootstrap p-value, and (for comparison)
        the asymptotic chi-squared p-value at the same df.

    Raises
    ------
    ValueError
        When ``n_sim < 1``.
    TypeError
        When ``large`` / ``small`` are not MixedLM fits.
    NotImplementedError
        When either fit uses ``vc_formula`` variance components.
    ImportError
        When ``n_jobs != 1`` and ``joblib`` is not installed.
    RuntimeError
        When every bootstrap iteration fails to converge.

    Warns
    -----
    UserWarning
        - When either input was REML and pbmodcomp internally
          refits as ML.
        - When the converged-iterations rate drops below 90 % —
          a signal that the bootstrap may not be sampling the null
          distribution faithfully.

    Notes
    -----
    - Failed bootstrap iterations (where either refit does not
      converge) are dropped, not silently treated as zero LRT —
      see :attr:`PBmodcompResult.n_sim_ok`.
    - The cost is ``O(n_sim · (fit_cost(small) + fit_cost(large)))``,
      which can be minutes on moderately-sized fits with
      ``n_sim=1000`` and ``n_jobs=1``. Set ``n_jobs=-1`` to use
      all cores.
    - This is the second of pbkrtest's three R functions ported to
      pymmeans (after :func:`pymmeans.satterthwaite.kenward_roger_vcov`).
      The third — higher-order Kenward-Roger ``ddf_Lb`` corrections —
      remains a v0.2 milestone.

    References
    ----------
    - Halekoh, U., & Højsgaard, S. (2014). A Kenward-Roger
      Approximation and Parametric Bootstrap Methods for Tests in
      Linear Mixed Models — The R Package pbkrtest. 59(9).
      doi:10.18637/jss.v059.i09
    - Davison, A. C., & Hinkley, D. V. (1997).
      *Bootstrap Methods and Their Application*. Cambridge UP.

    Examples
    --------
    >>> import warnings # doctest: +SKIP
    >>> import statsmodels.regression.mixed_linear_model as mlm # doctest: +SKIP
    >>> from statsmodels.datasets import get_rdataset # doctest: +SKIP
    >>> from pymmeans import pbmodcomp # doctest: +SKIP
    >>> sl = get_rdataset("sleepstudy", "lme4").data # doctest: +SKIP
    >>> large = mlm.MixedLM.from_formula( # doctest: +SKIP
    ... "Reaction ~ Days", groups="Subject", data=sl).fit(reml=False)
    >>> small = mlm.MixedLM.from_formula( # doctest: +SKIP
    ... "Reaction ~ 1", groups="Subject", data=sl).fit(reml=False)
    >>> res = pbmodcomp(large, small, n_sim=500, seed=42, n_jobs=-1) # doctest: +SKIP
    >>> print(res.summary()) # doctest: +SKIP
    """
    # ------------------------------------------------------------------
    # careful-implementation pass: input validation up front.
    # ------------------------------------------------------------------
    if n_sim < 1:
        raise ValueError(
            f"pbmodcomp(): n_sim must be >= 1, got {n_sim}. "
            "For the chi-squared test alone (no bootstrap), compute "
            "the LRT manually and call scipy.stats.chi2.sf directly."
        )
    _check_pbmodcomp_inputs(large, small)

    # Refit as ML if either model was REML (matches pbkrtest's
    # silent-refit behaviour, but we emit a UserWarning by default so
    # the user knows what happened).
    large_ml, l_was_reml = _ensure_ml(large)
    small_ml, s_was_reml = _ensure_ml(small)
    refit_reml = l_was_reml or s_was_reml
    if refit_reml and not silent:
        warnings.warn(
            "pbmodcomp(): one or both inputs were REML fits; refit "
            "internally as ML because the likelihood-ratio test "
            "requires comparable log-likelihoods across the two "
            "fixed-effect designs. Pass `silent=True` to suppress.",
            UserWarning,
            stacklevel=2,
        )

    # Observed LRT. ML logL is monotonic in the data, so the LRT is
    # nonnegative up to optimiser noise (clip in case of FP slop).
    lrt_obs = -2.0 * (float(small_ml.llf) - float(large_ml.llf))
    lrt_obs = max(lrt_obs, 0.0)

    # Total parameter-count difference: fixed-effect rank +
    # unique elements of cov_re + residual variance. When testing on
    # a random-effect boundary (variance estimated at zero) the
    # asymptotic chi-squared is mis-calibrated; the parametric
    # bootstrap is the recommended fix and is what we report as the
    # primary p-value.
    #
    # the code clamped
    # ``df = max(df, 1)``, which masked two real user errors:
    # (a) passing the same fit as both ``large`` and ``small``
    # (df=0, but reported as df=1) and (b) passing the models in
    # the wrong order (df < 0, but silently flipped to df=1). Both
    # are now caught with a clear ``ValueError``.
    df = _n_params_total(large_ml) - _n_params_total(small_ml)
    if df < 1:
        raise ValueError(
            "pbmodcomp(): `large` must have strictly more "
            "parameters than `small` (got "
            f"n_params(large)={_n_params_total(large_ml)}, "
            f"n_params(small)={_n_params_total(small_ml)}, "
            f"difference={df}). If you passed the same model "
            "twice, the test is vacuous (LRT_obs = 0). If you "
            "swapped the arguments, retry with `large` first."
        )

    from scipy import stats
    chi2_p_value = float(stats.chi2.sf(lrt_obs, df))

    # ------------------------------------------------------------------
    # Per-iteration RNG via SeedSequence — gives reproducible per-iter
    # streams regardless of parallel ordering.
    # ------------------------------------------------------------------
    parent_seq = np.random.SeedSequence(seed)
    child_seeds = parent_seq.spawn(n_sim)

    # ------------------------------------------------------------------
    # Dispatch: serial (no joblib needed) or joblib.Parallel.
    # ------------------------------------------------------------------
    if n_jobs == 1:
        iter_results = [
            _single_iteration(child_seeds[b], small_ml, large_ml)
            for b in range(n_sim)
        ]
    else:
        try:
            from joblib import Parallel, delayed
        except ImportError as exc: # pragma: no cover - tested separately
            raise ImportError(
                "pbmodcomp(n_jobs != 1) requires joblib. Install via "
                "`pip install joblib` (or add `pymmeans[parallel]` to "
                "your install extras)."
            ) from exc
        iter_results = Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_single_iteration)(child_seeds[b], small_ml, large_ml)
            for b in range(n_sim)
        )

    lrt_dist: list[float] = []
    failed: list[str] = []
    for b, (lrt_b, err) in enumerate(iter_results):
        if err is None:
            assert lrt_b is not None # narrow for type-checker
            lrt_dist.append(lrt_b)
        else:
            failed.append(f"iter {b}: {err}")

    lrt_arr = np.asarray(lrt_dist, dtype=float)
    n_ok = len(lrt_arr)
    if n_ok == 0:
        raise RuntimeError(
            "pbmodcomp(): all bootstrap iterations failed to "
            "converge. Inspect the fits or reduce n_sim. "
            f"First failure: {failed[0] if failed else 'unknown'}"
        )
    # P4: warn when convergence rate is alarmingly low.
    # Below ~90 % means the parametric bootstrap may not be sampling
    # the null faithfully — the user should inspect the surviving
    # draws (and probably the model spec).
    convergence_rate = n_ok / n_sim
    if convergence_rate < 0.90 and not silent:
        warnings.warn(
            f"pbmodcomp(): only {n_ok}/{n_sim} bootstrap iterations "
            f"({100 * convergence_rate:.1f} %) converged. Below ~90 % "
            "the bootstrap distribution may be biased toward the "
            "well-behaved corners of parameter space. Inspect "
            "`result._failed_iters` for the failure modes and "
            "consider reformulating or reducing the random-effects "
            "structure.",
            UserWarning,
            stacklevel=2,
        )
    # Standard +1 continuity correction (Davison & Hinkley 4.10).
    p_value = float(
        (1.0 + int(np.sum(lrt_arr >= lrt_obs))) / (1.0 + n_ok)
    )

    return PBmodcompResult(
        lrt_obs=lrt_obs,
        lrt_dist=lrt_arr,
        p_value=p_value,
        n_sim=n_sim,
        n_sim_ok=n_ok,
        chi2_p_value=chi2_p_value,
        df=df,
        refit_reml=refit_reml,
        _failed_iters=failed,
    )
