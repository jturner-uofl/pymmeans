"""Multiplicity corrections for families of comparisons.

Supported methods (all matching R ``emmeans::adjust`` semantics):

- ``none`` — raw two-sided p-values, no adjustment.
- ``bonferroni`` — ``min(m·p, 1)`` where m is the family size.
- ``holm`` — step-down Bonferroni (Holm, 1979).
- ``sidak`` — ``1 - (1-p)^m``, assumes independence.
- ``tukey`` — Tukey-HSD via the studentized-range survival function.
  Implemented directly via Gauss-Hermite quadrature for the inner
  normal integral and Gauss-generalized-Laguerre / Gauss-Legendre for
  the outer chi-squared scaling (vectorized across all t-statistics in
  one shot, unlike scipy's per-element ``studentized_range.sf``).
- ``dunnett`` / ``mvt`` — multivariate-t CDF (Dunnett, 1955) using
  scipy's Genz QMC implementation, with the correct correlation
  structure for contrasts that share a control. ``mvt`` is R emmeans's
  name for the same math when no shared-control structure exists.
- ``dunnettx`` — R's ``.pdunnx`` approximation: a weighted mixture of
  the studentized-range CDF (Tukey limit, ρ=0.5) and the independence
  limit ``F(q²; 1, df)^k``. Faster than exact MVT and the historical
  R default for ``trt.vs.ctrl``. Doesn't need a correlation matrix.
- ``BH`` / ``fdr`` — Benjamini-Hochberg false discovery rate
  (Benjamini & Hochberg, 1995). Step-up procedure controlling FDR
  under independent or PRDS (positive regression dependence) tests.
- ``BY`` — Benjamini-Yekutieli (Benjamini & Yekutieli, 2001). BH with
  the harmonic-number correction; controls FDR under arbitrary
  dependence.
- ``hochberg`` — Hochberg's step-up Bonferroni (Hochberg, 1988). More
  powerful than Holm under the same FWER guarantee when p-values are
  independent or positively dependent.
- ``hommel`` — Hommel's procedure (Hommel, 1988). Most powerful of the
  closed-testing FWER family under independence; complex but well-
  tested in ``statsmodels.stats.multitest`` which we delegate to.
- ``scheffe`` — Scheffé's method (Scheffé, 1959). F-distribution-based
  simultaneous bound that holds for ALL linear combinations of the k
  means (not just the chosen contrast family). The most conservative
  option; useful when contrasts are decided post-hoc.

Conventions:

- All methods take raw two-sided p-values.
- ``tukey`` and ``scheffe`` additionally need the t-ratios, the number
  of means k, and the residual df.
- ``dunnett`` / ``mvt`` additionally need the t-ratios, the correlation
  matrix of the t-statistics, and df.
- ``BH`` / ``fdr`` / ``BY`` / ``hochberg`` / ``hommel`` need only the
  raw p-values; the family is exactly ``len(p_raw)``.
- Returned p-values are clipped to ``[0, 1]``.
- A "family" is the set of comparisons being adjusted together. For
  by-grouped contrasts, the adjustment applies within each by-group
  (per "family" in R parlance).

References
----------
- Tukey, J. W. (1953). "The Problem of Multiple Comparisons."
- Dunnett, C. W. (1955). "A Multiple Comparison Procedure for Comparing
  Several Treatments with a Control." *JASA* 50(272), 1096-1121.
- Holm, S. (1979). "A Simple Sequentially Rejective Multiple Test
  Procedure." *Scandinavian Journal of Statistics* 6, 65-70.
- Hochberg, Y. (1988). "A Sharper Bonferroni Procedure for Multiple
  Tests of Significance." *Biometrika* 75(4), 800-802.
- Hommel, G. (1988). "A Stagewise Rejective Multiple Test Procedure
  Based on a Modified Bonferroni Test." *Biometrika* 75(2), 383-386.
- Benjamini, Y. & Hochberg, Y. (1995). "Controlling the False
  Discovery Rate." *JRSS-B* 57(1), 289-300.
- Benjamini, Y. & Yekutieli, D. (2001). "The Control of the FDR in
  Multiple Testing Under Dependency." *Annals of Statistics* 29(4).
- Scheffé, H. (1959). *The Analysis of Variance*. Wiley.
- Genz, A. & Bretz, F. (2009). *Computation of Multivariate Normal and
  t Probabilities*. Springer LNS 195. — the QMC algorithm scipy uses
  for the multivariate-t CDF.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

_METHODS = (
    "none",
    "bonferroni",
    "holm",
    "sidak",
    "tukey",
    "dunnett",
    "mvt",
    "dunnettx",
    "bh",
    "by",
    "hochberg",
    "hommel",
    "scheffe",
)

# `mvt` is R emmeans's name for the same multivariate-t CDF
# we use for `dunnett` (Genz QMC under the correlation of the test
# statistics). Different default contexts: emmeans picks `mvt` for
# arbitrary contrast families without a control structure, and `dunnett`
# for trt.vs.ctrl. The math is identical, so we resolve `mvt` to the
# same code path.
#
# R's `fdr` is an alias for Benjamini-Hochberg; we keep both
# names. R uses uppercase 'BH' / 'BY' but we lowercase before lookup so
# users can pass either case.
_ADJUST_ALIASES = {
    "mvt": "dunnett",
    "fdr": "bh",
    # `dunnettx` is now its own method
    # (R's `.pdunnx` ported below), NOT an alias for exact mvt. R's
    # `dunnettx` is a faster studentized-range / F-distribution-based
    # approximation that R uses as the default for `trt.vs.ctrl`; our
    # alias to exact mvt gave more accurate p-values but
    # didn't match R's reference values. Now we port R's actual
    # algorithm so `adjust='dunnettx'` matches `summary(emm,
    # adjust='dunnettx')` from R.
}


def adjust_pvalues(
    p_raw: Iterable[float],
    method: str,
    *,
    n_means: int | None = None,
    df: float | None = None,
    t_ratios: Iterable[float] | None = None,
    correlation: np.ndarray | None = None,
) -> np.ndarray:
    """Apply a multiplicity correction to a family of p-values.

    Parameters
    ----------
    p_raw
        Iterable of raw (un-adjusted) two-sided p-values, one per
        comparison.
    method
        Adjustment name. Supported: ``"none"``, ``"bonferroni"``,
        ``"sidak"``, ``"holm"``, ``"tukey"`` (studentized-range CDF;
        requires ``t_ratios``, ``n_means``, ``df``), ``"dunnett"``
        (multivariate-t CDF with shared-control correlation; requires
        ``t_ratios``, ``df``, ``correlation``).
    n_means
        Number of group means in the family. Required for ``"tukey"``;
        sets the studentized-range CDF parameter ``k``.
    df
        Denominator degrees of freedom for the t / studentized-range /
        multivariate-t distributions. Required for ``"tukey"`` and
        ``"dunnett"``.
    t_ratios
        Per-comparison t statistics, used by ``"tukey"`` and
        ``"dunnett"`` to map raw p back into the joint distribution
        of the test statistic.
    correlation
        ``(k, k)`` correlation matrix among comparisons for the Dunnett
        adjustment; typically derived from
        ``cov_c / (SE_i * SE_j)`` on the contrast matrix.

    Returns
    -------
    ndarray
        Adjusted p-values, same length as ``p_raw``, clipped to
        ``[0, 1]``.
    """
    # preserve the user's original
    # spelling in the error message so they can see exactly what they
    # passed (uppercase/typo) instead of a lowercased version that
    # doesn't match their code.
    method_orig = method
    method = method.lower()
    method = _ADJUST_ALIASES.get(method, method)
    if method not in _METHODS:
        raise ValueError(
            f"Unknown adjustment method '{method_orig}'. "
            f"Supported: {_METHODS}."
        )
    # Don't dispatch on the alias name internally; collapse to canonical.
    if method == "mvt":
        method = "dunnett"

    p = np.asarray(list(p_raw), dtype=float)
    m = len(p)
    if m == 0:
        return p

    if method == "none":
        return np.clip(p, 0.0, 1.0)
    if method == "bonferroni":
        return np.clip(p * m, 0.0, 1.0)
    if method == "sidak":
        return np.clip(1.0 - (1.0 - p) ** m, 0.0, 1.0)
    if method == "holm":
        return _holm(p)
    if method == "tukey":
        if t_ratios is None or n_means is None or df is None:
            raise ValueError(
                "'tukey' adjustment requires t_ratios, n_means, and df."
            )
        return _tukey(np.asarray(list(t_ratios), dtype=float), n_means, df)
    if method == "dunnettx":
        # R's `.pdunnx` approximation.
        # Treats P(max |T_i| <= q) as a weighted mixture of the
        # studentized-range CDF (correlation = 0.5, the "fully shared
        # control" limit, mapped to k_tukey means) and the independent-
        # F^k product (correlation = 0, the "no shared variance" limit).
        # Faster than exact mvt and historically R's default for
        # trt.vs.ctrl. Doesn't need a correlation matrix.
        if t_ratios is None or df is None:
            raise ValueError(
                "'dunnettx' adjustment requires t_ratios and df."
            )
        t_arr = np.asarray(list(t_ratios), dtype=float)
        return _dunnettx(t_arr, df)
    if method == "dunnett":
        if t_ratios is None or df is None:
            raise ValueError(
                "'dunnett' adjustment requires t_ratios and df."
            )
        t_arr = np.asarray(list(t_ratios), dtype=float)
        # silently using identity correlation for k>1
        # quietly degrades Dunnett to a Sidak-style independence
        # approximation (over-conservative by 5-15% at typical shared-
        # control rho=0.5). For k=1 the family is just one two-sided
        # t-test so identity is correct. Require the caller to pass the
        # true correlation matrix when k>1.
        if correlation is None:
            if len(t_arr) > 1:
                raise ValueError(
                    "'dunnett' adjustment requires `correlation` for "
                    f"families of size > 1 (got {len(t_arr)}). Pass the "
                    "correlation of the test statistics — for "
                    "treatment-vs-control contrasts this is "
                    "`cov_c / outer(SE, SE)` on the contrast matrix. "
                    "Without it, the result would silently default to "
                    "identity (independent comparisons), which gives an "
                    "over-conservative Sidak-style approximation rather "
                    "than a true Dunnett adjustment."
                )
            corr = np.eye(len(t_arr))
        else:
            corr = np.asarray(correlation, dtype=float)
        return _dunnett(t_arr, corr, df)
    if method in ("bh", "by", "hochberg", "hommel"):
        # delegate to statsmodels (which is already a hard
        # dep). Their implementations are battle-tested against R's
        # `p.adjust` to floating-point precision for all four methods.
        # The Hommel procedure in particular is recursive and tricky to
        # get right; reusing the statsmodels impl is far safer than
        # rolling our own.
        from statsmodels.stats.multitest import multipletests as _mt

        sm_method = {
            "bh": "fdr_bh",
            "by": "fdr_by",
            "hochberg": "simes-hochberg",
            "hommel": "hommel",
        }[method]
        # multipletests returns (reject_array, adj_p, alpha_sidak, alpha_bonf)
        _, p_adj, _, _ = _mt(p, alpha=0.05, method=sm_method)
        return np.clip(np.asarray(p_adj, dtype=float), 0.0, 1.0)
    if method == "scheffe":
        # Scheffé's simultaneous bound: under H0 the F-statistic
        # `t^2 / (k-1)` is bounded above by F(k-1, df) at the chosen
        # alpha for ANY linear contrast. So `p_adj = F.sf(t^2/(k-1),
        # k-1, df)`. The conservatism comes from the (k-1) divisor —
        # we're paying for the freedom to pick contrasts post-hoc.
        if t_ratios is None or n_means is None or df is None:
            raise ValueError(
                "'scheffe' adjustment requires t_ratios, n_means, and df."
            )
        if n_means < 2:
            raise ValueError(
                f"'scheffe' adjustment requires n_means >= 2 (got "
                f"{n_means}); the F-distribution numerator df = k-1 "
                "must be positive."
            )
        # the original comment claimed
        # scipy handles `df=np.inf` for the denominator. It does not —
        # `f.sf(F, k-1, np.inf)` returns NaN. The limiting distribution
        # of F(k-1, df) as df -> inf is chi-squared(k-1)/(k-1), so the
        # tail p-value at F-stat equals chi2.sf(t^2, k-1). Route to
        # chi2 explicitly when df is non-finite.
        from scipy.stats import chi2 as _chi2
        from scipy.stats import f as _f

        t_arr = np.asarray(list(t_ratios), dtype=float)
        if not np.isfinite(df):
            return np.clip(
                _chi2.sf(t_arr**2, n_means - 1), 0.0, 1.0
            )
        F_stat = (t_arr**2) / (n_means - 1)
        return np.clip(_f.sf(F_stat, n_means - 1, df), 0.0, 1.0)
    raise AssertionError(f"unreachable: {method}")


def _holm(p: np.ndarray) -> np.ndarray:
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, candidate)
        adj[idx] = running_max
    return adj


# Vectorized studentized-range SF — implemented directly via Gauss-Hermite
# quadrature (and Gauss-Legendre over the chi scaling for finite df). Avoids
# scipy.stats.studentized_range.sf's per-element numerical integration, which
# was the v0.1 Tukey-at-scale bottleneck.

# Conservative static limit + runtime overflow detection: scipy's
# roots_genlaguerre actually overflows around alpha~179 for n=60,
# not the 249 originally estimated. Belt-and-suspenders avoids ever
# returning garbage values from overflowed weights.
_TUKEY_GENLAG_DF_LIMIT = 300.0
_TUKEY_INF_DF_THRESHOLD = 50_000.0
_TUKEY_N_HERMITE = 80
_TUKEY_N_LAGUERRE = 60
_TUKEY_N_LEGENDRE = 80

_HERMITE_X, _HERMITE_W = np.polynomial.hermite_e.hermegauss(_TUKEY_N_HERMITE)
_NORM_CONST = 1.0 / np.sqrt(2.0 * np.pi)


def _srange_sf_inf_df(q: np.ndarray, k: int) -> np.ndarray:
    """Studentized-range SF at df=infinity, vectorized over q.

    F(q; k, inf) = k * integral phi(s) * (Phi(s+q) - Phi(s))^(k-1) ds
    Approximated via probabilist's Hermite quadrature.
    """
    from scipy.special import ndtr

    inner = ndtr(_HERMITE_X[None, :] + q[:, None]) - ndtr(_HERMITE_X[None, :])
    np.clip(inner, 0.0, None, out=inner)
    integrand = inner ** (k - 1)
    cdf = (integrand @ _HERMITE_W) * k * _NORM_CONST
    return 1.0 - cdf


def _srange_sf_finite_df(q: np.ndarray, k: int, df: float) -> np.ndarray:
    """Studentized-range SF for finite df, vectorized over q.

    For small-to-moderate df (< ~400) we use Gauss generalized-Laguerre
    quadrature with weight ``u^alpha * exp(-u)`` (alpha = df/2 - 1), which
    matches the chi-squared(df)/2 density exactly.

    For larger df, scipy's ``roots_genlaguerre`` overflows internally
    (weights ~ Gamma(alpha + n) become inf), so we fall back to Gauss-
    Legendre on an adaptive interval around the chi-squared mean.
    """

    if df < _TUKEY_GENLAG_DF_LIMIT:
        return _srange_sf_genlag(q, k, df)
    # Adaptive Legendre on [W_mean - 10*sd, W_mean + 10*sd]
    return _srange_sf_legendre(q, k, df)


def _srange_sf_genlag(q: np.ndarray, k: int, df: float) -> np.ndarray:
    from scipy.special import gammaln, ndtr, roots_genlaguerre

    alpha = df / 2.0 - 1.0
    try:
        nodes, weights = roots_genlaguerre(_TUKEY_N_LAGUERRE, alpha)
    except Exception:
        return _srange_sf_legendre(q, k, df)
    # Runtime detection of scipy overflow: if nodes or weights aren't finite,
    # we'd silently return garbage. Fall through to the Legendre path.
    if not (np.all(np.isfinite(nodes)) and np.all(np.isfinite(weights))):
        return _srange_sf_legendre(q, k, df)

    s_nodes = np.sqrt(2.0 * nodes / df)
    with np.errstate(divide="ignore"):
        log_weights = np.log(np.abs(weights))
    weights_normalized = np.sign(weights) * np.exp(log_weights - gammaln(df / 2.0))

    result = np.zeros(len(q))
    for sj, wj in zip(s_nodes, weights_normalized, strict=True):
        qs = q * sj
        inner = ndtr(_HERMITE_X[None, :] + qs[:, None]) - ndtr(_HERMITE_X[None, :])
        np.clip(inner, 0.0, None, out=inner)
        integrand = inner ** (k - 1)
        cdf_inf = (integrand @ _HERMITE_W) * k * _NORM_CONST
        result += (1.0 - cdf_inf) * wj
    return result


def _srange_sf_legendre(q: np.ndarray, k: int, df: float) -> np.ndarray:
    from scipy.special import gammaln, ndtr

    mean_W = df
    std_W = np.sqrt(2.0 * df)
    W_lo = max(1e-6, mean_W - 10.0 * std_W)
    W_hi = mean_W + 10.0 * std_W
    x, w = np.polynomial.legendre.leggauss(_TUKEY_N_LEGENDRE)
    W_nodes = 0.5 * (x + 1.0) * (W_hi - W_lo) + W_lo
    W_jac = 0.5 * (W_hi - W_lo)

    # log p_W = (df/2 - 1) * log(W) - W/2 - (df/2)*log(2) - gammaln(df/2)
    log_p = (
        (df / 2.0 - 1.0) * np.log(W_nodes)
        - W_nodes / 2.0
        - (df / 2.0) * np.log(2.0)
        - gammaln(df / 2.0)
    )
    p_W = np.exp(log_p)
    weights_normalized = w * W_jac * p_W
    s_nodes = np.sqrt(W_nodes / df)

    result = np.zeros(len(q))
    for sj, wj in zip(s_nodes, weights_normalized, strict=True):
        qs = q * sj
        inner = ndtr(_HERMITE_X[None, :] + qs[:, None]) - ndtr(_HERMITE_X[None, :])
        np.clip(inner, 0.0, None, out=inner)
        integrand = inner ** (k - 1)
        cdf_inf = (integrand @ _HERMITE_W) * k * _NORM_CONST
        result += (1.0 - cdf_inf) * wj
    return result


def _tukey(t_ratios: np.ndarray, n_means: int, df: float) -> np.ndarray:
    if n_means < 2:
        # returning zeros here silently reported
        # every contrast as "highly significant" when called with
        # n_means=1 (e.g. a buggy upstream contrast-builder). Tukey
        # HSD with k<2 has no comparisons to make and is mathematically
        # undefined, so raise instead.
        raise ValueError(
            f"'tukey' adjustment requires n_means >= 2 (got {n_means}); "
            "the studentized-range distribution is undefined for k<2."
        )
    q = np.abs(t_ratios) * np.sqrt(2.0)

    if df < 3.0:
        # Pathological low-df region: Hermite quadrature loses accuracy
        # because the chi-squared(df)/df distribution has very heavy
        # tails. Fall back to scipy's per-element exact integration —
        # slow, but rare in practice (df<3 means n<=p+2).
        from scipy.stats import studentized_range
        return np.clip(studentized_range.sf(q, n_means, df), 0.0, 1.0)
    if df >= _TUKEY_INF_DF_THRESHOLD:
        out = _srange_sf_inf_df(q, n_means)
    else:
        out = _srange_sf_finite_df(q, n_means, df)
    return np.clip(out, 0.0, 1.0)


def _regularize_corr_for_mvt(corr: np.ndarray) -> np.ndarray:
    """ridge-regularize a correlation matrix so the MVT
    CDF is well-defined on it.

    Pairwise contrast correlations on k >= 3 means are SINGULAR — the
    contrasts span a (k-1)-dimensional subspace because they satisfy
    sum-to-zero relationships (e.g. ``(A-B) + (B-C) - (A-C) = 0`` for
    k=3). scipy's ``multivariate_t.cdf(..., allow_singular=True)``
    handles the singular case via Moore-Penrose pseudoinverse, but the
    resulting integral is NOT the MVT orthant probability the user
    expects — it gives systematically wrong values that don't match R
    `mvtnorm::pmvt`.

    The fix here: symmetrize the matrix, detect the minimum eigenvalue,
    and add a tiny ridge so the matrix becomes positive-definite;
    re-normalise the diagonal back to 1 so the result is still a
    valid correlation matrix. R's `mvtnorm::pmvt` uses a different
    strategy — Cholesky pivoting plus dimension reduction down to
    the rank of the contrast matrix — which is mathematically cleaner
    but more invasive to port. The ridge introduces an O(1e-9) bias
    that is invisible at the four-decimal tolerance we validate
    against R.
    """
    corr = np.asarray(corr, dtype=float)
    corr = 0.5 * (corr + corr.T)
    lam_min = float(np.linalg.eigvalsh(corr)[0])
    if lam_min <= 1e-9:
        corr = corr + (-lam_min + 1e-9) * np.eye(corr.shape[0])
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)
    return corr


def _dunnett(t_ratios: np.ndarray, correlation: np.ndarray, df: float) -> np.ndarray:
    """Two-sided Dunnett-adjusted p-values via the multivariate-t CDF.

    For each comparison i with observed |t_i|:
        p_i = 1 - P(|T_j| <= |t_i| for all j) under MVT(R, df)

    scipy's `multivariate_t.cdf` is a
    QMC implementation that advances internal random state across
    calls — two contrasts with IDENTICAL |t| produced DIFFERENT
    p-values (off by ~1e-5). Fix: (a) pass `random_state=` per call
    rather than reusing the `rv` instance, and (b) cache by unique
    `q` so equal thresholds are computed once and identical.
    """
    from scipy import stats

    k = len(t_ratios)
    if k == 0:
        return np.zeros(0)
    if k == 1:
        return np.clip(2.0 * stats.t.sf(np.abs(t_ratios), df), 0.0, 1.0)

    # Exact-Dunnett complexity wall (proper redesign scheduled for
    # 0.2.0). scipy's ``multivariate_t.cdf`` is a QMC integrator whose
    # cost is roughly ``O(maxpts * k)`` per unique |t|, with
    # ``maxpts`` itself scaled as ``50_000 * k`` so the orthant
    # integral has enough samples to converge below the advertised
    # 1e-4 tolerance. At k = 100 the integration is still tractable
    # (~minute on a laptop); at k >= 150 the QMC path becomes
    # effectively unbounded (k=200 ran past 70 s in benchmarks).
    # Guard the user with a clear error rather than letting the call
    # hang. The threshold is overridable via
    # ``set_emm_options(dunnett_max_k=...)`` for the rare case where
    # the caller has the time budget and wants the exact integral.
    from pymmeans.options import get_emm_option as _opt

    max_k = int(_opt("dunnett_max_k", 100))
    if k > max_k:
        raise ValueError(
            f"Exact Dunnett at k={k} ({k} comparisons, "
            f"{k}-dimensional MVT integral) exceeds the configured "
            f"safety cap (dunnett_max_k={max_k}). The QMC integration "
            "cost scales steeply with k and becomes effectively "
            "unbounded for k >= ~150. Pick one:\n"
            f"  - Use the closed-form approximation: ``adjust="
            f"'dunnettx'`` (R `.pdunnx`, finite-time at any k).\n"
            "  - Reduce the contrast set (e.g. only the comparisons "
            "of substantive interest).\n"
            "  - If you genuinely need exact Dunnett at this k, "
            "raise the cap explicitly: "
            "``pymmeans.set_emm_options(dunnett_max_k=...)``. Budget "
            "minutes-to-hours of CPU time."
        )

    # regularize against singular pairwise contrast
    # correlations (k >= 3 means produce a rank-(k-1) contrast
    # correlation; scipy's pseudoinverse path gives wrong probabilities).
    correlation = _regularize_corr_for_mvt(correlation)
    use_normal = not np.isfinite(df)
    # Bumped maxpts so the QMC integration tolerance is well below the
    # ~5e-3 we'd otherwise hit at k=10. Conservative cap.
    maxpts = max(1_000_000, 50_000 * k)

    # Cache by unique q so identical |t| → identical p_value.
    unique_q: dict[float, float] = {}
    for t in t_ratios:
        q = abs(float(t))
        if q in unique_q:
            continue
        # Use the classmethod with explicit random_state so each call
        # is independent and deterministic. R also uses a fixed seed
        # in its mvtnorm CDF call.
        if use_normal:
            inside = float(
                stats.multivariate_normal.cdf(
                    np.full(k, q),
                    mean=np.zeros(k),
                    cov=correlation,
                    allow_singular=True,
                    lower_limit=np.full(k, -q),
                )
            )
        else:
            inside = float(
                stats.multivariate_t.cdf(
                    np.full(k, q),
                    shape=correlation,
                    df=df,
                    allow_singular=True,
                    lower_limit=np.full(k, -q),
                    random_state=0,
                    maxpts=maxpts,
                )
            )
        unique_q[q] = 1.0 - inside

    out = np.array([unique_q[abs(float(t))] for t in t_ratios])
    return np.clip(out, 0.0, 1.0)


def adjust_mvt_tail(
    t_ratios: Iterable[float],
    correlation: np.ndarray,
    df: float,
    *,
    tail: int,
) -> np.ndarray:
    """family-wise one-sided MVT adjustment.

    Port of R `emmeans::.my.pmvt` and the corresponding
    `.adj.p.value` mvt branch (now pinned via R-source consultation):

    * ``tail = +1`` (right tail; noninferiority): for each contrast
      ``i`` with observed t-statistic ``t_i``,
      ``p_adj[i] = 1 - P(all T_j <= t_i | T ~ MVT(corr, df))``.
    * ``tail = -1`` (left tail; nonsuperiority):
      ``p_adj[i] = 1 - P(all T_j >= t_i)``.
    * ``tail = 0`` (two-sided):
      ``p_adj[i] = 1 - P(all |T_j| <= |t_i|)``. Equivalent to the
      existing :func:`_dunnett` two-sided path; provided here for
      uniformity.

    For TOST equivalence (`delta>0` + `side='two-sided'`), the caller
    computes ``p_lo_adj = adjust_mvt_tail(t_lo, corr, df, tail=+1)``
    and ``p_hi_adj = adjust_mvt_tail(t_hi, corr, df, tail=-1)``, then
    returns ``max(p_lo_adj, p_hi_adj)`` per contrast — the
    intersection-union TOST family-wise p-value.
    """
    from scipy import stats

    t_arr = np.asarray(list(t_ratios), dtype=float)
    if t_arr.size == 0:
        return np.zeros(0)
    corr = _regularize_corr_for_mvt(np.asarray(correlation, dtype=float))
    k = corr.shape[0]
    if k == 1:
        # Univariate: fall back to the per-row t / normal.
        if tail == 0:
            return np.clip(
                2.0 * stats.t.sf(np.abs(t_arr), df), 0.0, 1.0
            )
        if tail > 0:
            return np.clip(stats.t.sf(t_arr, df), 0.0, 1.0)
        return np.clip(stats.t.cdf(t_arr, df), 0.0, 1.0)
    use_normal = not np.isfinite(df)
    maxpts = max(1_000_000, 50_000 * k)
    out = np.empty(t_arr.size, dtype=float)
    for i, t in enumerate(t_arr):
        if tail == 0:
            lower = np.full(k, -abs(float(t)))
            upper = np.full(k, abs(float(t)))
        elif tail > 0:
            lower = np.full(k, -np.inf)
            upper = np.full(k, float(t))
        else:
            lower = np.full(k, float(t))
            upper = np.full(k, np.inf)
        if use_normal:
            prob = float(
                stats.multivariate_normal.cdf(
                    upper,
                    mean=np.zeros(k),
                    cov=corr,
                    allow_singular=True,
                    lower_limit=lower,
                )
            )
        else:
            prob = float(
                stats.multivariate_t.cdf(
                    upper,
                    shape=corr,
                    df=df,
                    allow_singular=True,
                    lower_limit=lower,
                    random_state=0,
                    maxpts=maxpts,
                )
            )
        out[i] = 1.0 - prob
    return np.clip(out, 0.0, 1.0)


def _dunnettx(t_ratios: np.ndarray, df: float) -> np.ndarray:
    """R's ``.pdunnx`` — two-sided Dunnett-X approximate p-values.

    Port of R emmeans's ``.pdunnx`` helper. The idea: for k treatments
    vs a control, treat ``P(max_i |T_i| <= q)`` as a convex combination
    of two extremes,

    - the Tukey HSD limit (treatments share the control fully, ρ=0.5
      pairwise) with a re-mapped "k_tukey" group count, and
    - the independence limit (ρ=0) which gives ``F(q² ; 1, df)^k``.

    The weight ``twt = (k-1)/k`` is from R's source. Faster than exact
    MVT (no QMC integration), and historically R's default for
    ``trt.vs.ctrl``. Calibrated to be within ~1e-3 of exact in the
    relevant tail region for moderate k.
    """
    from scipy import stats

    k = len(t_ratios)
    if k == 0:
        return np.zeros(0)
    if k == 1:
        # Single-treatment family is just a two-sided t-test.
        return np.clip(2.0 * stats.t.sf(np.abs(t_ratios), df), 0.0, 1.0)

    x = np.abs(t_ratios)
    twt = (k - 1) / k
    # The "effective k for Tukey" satisfies k_tukey*(k_tukey-1)/2 = k,
    # giving k_tukey = (1 + sqrt(1 + 8k)) / 2. (See R source: the
    # mapping comes from matching the number of correlated pairs.)
    k_tukey = (1.0 + np.sqrt(1.0 + 8.0 * k)) / 2.0
    if not np.isfinite(df):
        # df → ∞: studentized-range still works (chi/df → 1); F → χ².
        cdf = (
            twt * stats.studentized_range.cdf(np.sqrt(2.0) * x, k_tukey, np.inf)
            + (1.0 - twt) * stats.chi2.cdf(x * x, 1) ** k
        )
    else:
        cdf = (
            twt * stats.studentized_range.cdf(np.sqrt(2.0) * x, k_tukey, df)
            + (1.0 - twt) * stats.f.cdf(x * x, 1, df) ** k
        )
    return np.clip(1.0 - cdf, 0.0, 1.0)
