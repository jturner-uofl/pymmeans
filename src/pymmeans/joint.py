"""Joint Wald tests — pymmeans's Type III ANOVA equivalent.

R ``emmeans::joint_tests`` reports, for each model term, a Type-III-style
joint test of the **marginal effect** of that term. For purely
categorical terms this is *not* the same as testing the raw coefficient
slice ``beta_T = 0``: under treatment coding (and most other coding
schemes) the slice represents the *simple effect at the reference
levels* of the other factors, which depends arbitrarily on which level
patsy picks as reference.

The fix: for each term, build the EMM grid over the term's categorical
factors (averaging over all other factors), form a contrast matrix that
spans the "term effect" in that grid, and test ``D L_marg beta = 0``.

For a single-factor categorical term ``a``, this is the standard
"all means equal" test: ``D = I - 1 1' / k_a`` (rank ``k_a - 1``) applied
to a length-``k_a`` EMM vector.

For an interaction term ``a:b``, the contrast spans rows of the EMM grid
that "wiggle" in the way the interaction would but cannot be explained
by either main effect alone. The construction is the Kronecker product
of per-factor centring matrices.

For numeric-only terms (e.g. a single covariate ``x``) the test is
the average-slope-over-categoricals contrast (df1 = 1), computed by
finite-differencing the EMM grid across the numeric and averaging
over any categoricals that interact with it.

Mixed terms (e.g. ``a:x``, the slope-by-factor interaction) are
tested via the same EMM-basis path: build slopes at each level of
``a`` via finite difference, then apply the centring contrast
(df1 = k_a - 1). Matches R ``emmeans::joint_tests`` exactly on the
``y ~ a*x`` reference fit (verified in
``tests/test_joint.py`` and the R cross-validation benchmark in
``tests/test_r_benchmark.py``).

The output statistic is F (for OLS, with the model's residual df) or
chi-squared (for GLM and MixedLM, since the residual variance is the
asymptotic dispersion).

References
----------
- Searle, Speed & Milliken (1980). "Population Marginal Means in the
  Linear Model: An Alternative to Least Squares Means." *The American
  Statistician*, 34(4).
- Lenth (2024). R ``emmeans::joint_tests``.
- Fox & Weisberg (2018). *An R Companion to Applied Regression*, 3rd ed.,
  for the Type-III hypothesis framing under non-orthogonal designs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from pymmeans.utils import ModelInfo, from_fitted


def _centring_contrast(k: int) -> np.ndarray:
    """Orthonormal basis for the (k-1)-dim "all rows equal" null.

    Returns a matrix ``D`` of shape ``(k-1, k)`` such that ``D @ v = 0``
    iff all entries of ``v`` are equal. We use the Helmert-style first
    ``k-1`` rows of the centring matrix ``I - 1 1' / k`` after dropping
    one row (any one is fine; the resulting span is the same).
    """
    if k < 2:
        return np.zeros((0, k))
    C = np.eye(k) - np.ones((k, k)) / k
    # Drop the last row to get a (k-1) x k rank-(k-1) basis.
    return C[:-1]


def _term_contrast(factor_dims: list[int]) -> np.ndarray:
    """Kronecker product of per-factor centring contrasts.

    For a term with categorical factors of sizes ``[k1, k2, ...]``, the
    relevant EMM grid has ``prod(k_i)`` rows. The term-effect contrast
    is the Kronecker product of the centring contrasts ``D_i =
    centring_contrast(k_i)``, giving a matrix of shape
    ``(prod(k_i - 1), prod(k_i))``.

    Patsy's analytic marginalisation lays out the grid with the *first*
    factor varying fastest. ``np.kron(A, B)`` makes ``B`` vary fastest in
    its output, so we Kronecker from the right to get the same ordering.
    """
    if not factor_dims:
        return np.array([[1.0]])
    D = _centring_contrast(factor_dims[-1])
    for k in reversed(factor_dims[:-1]):
        D = np.kron(_centring_contrast(k), D)
    return D


def _build_term_test_L(
    info: ModelInfo,
    spec_default: dict,
    cat_T: list[str],
    num_T: list[str],
) -> np.ndarray:
    """Build the EMM-basis Wald test matrix for a term.

    For each term we need a hypothesis matrix L such that ``L @ beta = 0``
    corresponds to "the marginal effect of this term is null". The matrix
    is constructed by:

    1. Building the EMM grid varying ``cat_T`` over all levels and
       ``num_T`` over two values per numeric (for finite-difference
       derivatives). Factors not in ``cat_T + num_T`` are averaged
       (categoricals) or held at the training mean (numerics).
    2. For each numeric in ``num_T``, folding the corresponding axis by
       central differencing (slope = (high - low) / h). For linear-in-x
       formulas this is exact.
    3. For pure-numeric terms (``cat_T`` empty), the result is a single
       row: the average slope across whatever categoricals were
       averaged. df1 = 1, matching R ``emmeans::joint_tests`` reporting
       a single slope-equals-zero test.
    4. For mixed cat-num terms (``cat_T`` non-empty, ``num_T`` non-empty)
       or pure-cat terms (``num_T`` empty), apply the Kronecker centring
       contrast over ``cat_T``. df1 = prod(k_i - 1).

    #2: previously, numeric-only and mixed-cat-numeric terms
    fell back to a parameter slice, which tests the slope at the
    *reference level* of the other factors. With treatment coding that's
    a simple effect, not the marginal main effect R reports.
    """
    from pymmeans.analytic import analytic_marginalize

    spec = dict(spec_default)
    # Two values are enough because the model is linear in each numeric
    # (patsy doesn't introduce x^2 unless the formula does it manually).
    h = 1.0
    for nm in num_T:
        spec[nm] = [0.0, h]

    group_cols = list(cat_T) + list(num_T)
    if not group_cols:
        return np.zeros((0, info.n_params))

    L_marg, _ = analytic_marginalize(info, spec, group_cols)
    cat_combos = 1
    for n in cat_T:
        cat_combos *= len(info.factors[n])
    n_num = len(num_T)
    # analytic_marginalize lays out rows by itertools.product over
    # group_cols, last-varies-fastest. With group_cols = cat_T + num_T
    # that's a row-major reshape into (cat_combos, 2, 2, ..., n_params).
    shape = (cat_combos,) + (2,) * n_num + (info.n_params,)
    L = L_marg.reshape(shape)
    # Fold each numeric axis by differencing high - low (divided by h
    # so the unit matches "slope per unit of x"; with h=1 this is a no-op
    # but keeps the formula correct under future hyperparam changes).
    for _ in range(n_num):
        L = (L[:, 1, ...] - L[:, 0, ...]) / h
    # Shape now: (cat_combos, n_params).
    if not cat_T:
        # Pure-numeric main effect: average the slopes across whatever
        # categoricals were averaged. Single row -> df1 = 1.
        return L.mean(axis=0, keepdims=True)
    factor_dims = [len(info.factors[n]) for n in cat_T]
    D = _term_contrast(factor_dims)
    return D @ L


def joint_tests(
    model: Any,
    by: str | list[str] | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """Type III joint tests for every non-intercept term in the model.

    For each term, tests whether the EMMs corresponding to the term's
    categorical factors are jointly equal (the marginal-effect null
    hypothesis). Matches R ``emmeans::joint_tests`` for OLS and GLM
    models with purely-categorical or purely-numeric terms.

    Parameters
    ----------
    model
        Fitted statsmodels result or a ``ModelInfo``.
    by
        Categorical factor name (or list of factor names) for grouped
        (simple-effects) tests. When supplied, the by-factor(s) are
        held fixed at each of their levels and the joint test is run
        for every model term that is *not* a strict subset of the
        by-set. The returned DataFrame gains one column per by-factor
        plus a row block per by-cell. Mirrors
        ``emmeans::joint_tests(model, by = ...)``.

    Returns
    -------
    DataFrame
        Columns: ``term``, ``df_num``, ``df_denom``, ``statistic``,
        ``p_value`` (plus one column per by-factor when ``by`` is set).
        ``statistic`` is F (for OLS) or chi-squared (for GLM and
        MixedLM); the dispatch is reflected in ``df_denom`` being
        finite or infinite.
    """
    from pymmeans.ref_grid import build_grid_spec

    # accept ``adjust=`` (and other R-style kwargs)
    # for API tolerance with R `emmeans::joint_tests`. R accepts
    # ``adjust=`` via ``...`` and forwards it through; pymmeans
    # currently produces per-term p-values WITHOUT applying any
    # multiplicity adjustment (matching R's documented default), so
    # the kwarg is accepted as a no-op. If we add joint-adjustment
    # in v0.2, this is where the dispatch would live. Unknown
    # kwargs still raise to catch typos.
    _ACCEPTED = {"adjust"}
    _unknown = set(kwargs) - _ACCEPTED
    if _unknown:
        raise TypeError(
            f"joint_tests() got unsupported keyword(s): {sorted(_unknown)}. "
            f"Accepted: by, adjust."
        )
    _adjust_requested = kwargs.get("adjust")

    # Accept EMMResult / ContrastResult / RefGrid input by dispatching
    # to the underlying ``model_info``. R ``emmeans::joint_tests`` accepts
    # both ``fit`` and ``emmGrid`` objects; ``joint_tests(emm)`` tests
    # every term of the model that produced ``emm``, NOT just the
    # subset represented in the EMM grid. (Different from
    # ``test(emm, joint=TRUE)`` which jointly tests the EMM cells.)
    if hasattr(model, "model_info") and not isinstance(model, ModelInfo):
        # Refuse posterior-derived inputs: ``joint_tests`` is a Wald
        # F / chi-squared test on ``beta_hat``, ``vcov``. A posterior
        # ``ModelInfo`` has ``vcov`` set to the empirical covariance of
        # the posterior draws, but the F / chi-squared statistic
        # assumes a frequentist sampling distribution — applying Wald
        # joint inference to posterior summaries silently mixes two
        # paradigms and produces a meaningless p-value (and, in the
        # MixedLM-style ``df_denom=inf`` branch, often NaN). Steer
        # the user to the posterior-native workflow.
        if getattr(model, "inference_kind", "wald") == "posterior":
            raise ValueError(
                "joint_tests() is not defined for posterior-derived "
                "EMMResult / ContrastResult inputs. The Wald F / chi-"
                "squared statistic assumes a frequentist sampling "
                "distribution of beta_hat; applying it to a posterior "
                "covariance is meaningless. For a Bayesian joint test, "
                "compute the posterior probability of the null contrast "
                "directly from the draws (e.g. via "
                "`posterior_emm_summary` on the linear combination of "
                "interest), or refit the model with a frequentist "
                "estimator before calling `joint_tests`."
            )
        info = model.model_info
    else:
        info = model if isinstance(model, ModelInfo) else from_fitted(model)
    di = info.design_info
    use_chi = info.family is not None or info.is_mixed
    df_denom = float("inf") if use_chi else float(info.df_resid)
    beta = info.beta
    V = info.vcov

    spec_full = build_grid_spec(info, at=None, require_plain_identifiers=False)

    # ---- normalise `by` -----------------------------------------------------
    by_list: list[str]
    if by is None:
        by_list = []
    elif isinstance(by, str):
        by_list = [by]
    else:
        by_list = list(by)
    for nm in by_list:
        if nm not in info.factors:
            raise ValueError(
                f"joint_tests(by=...) only supports categorical factors; "
                f"{nm!r} is not a categorical factor in this model. "
                f"Known categoricals: {sorted(info.factors)}."
            )
    by_set = set(by_list)

    # Build the list of by-cell value combos (cartesian product over
    # by_list, in the order requested). For the no-by case there is a
    # single empty combo so the loop below runs once with no pinning.
    if by_list:
        from itertools import product

        # Match R `expand.grid` ordering — the FIRST by-factor varies
        # FASTEST in the output.
        # Python's `itertools.product` has the LAST argument varying
        # fastest, so reverse the level lists, take the product, then
        # reverse each tuple to restore the original by-factor order
        # in the row labels. R reference (Rscript): for by=c("a","c")
        # the cells print in order (A,P), (B,P), (A,Q), (B,Q).
        rev_levels = list(reversed([list(info.factors[n]) for n in by_list]))
        by_combos = [tuple(reversed(x)) for x in product(*rev_levels)]
    else:
        by_combos = [()]

    rows = []
    for combo in by_combos:
        # Pin by-factors to the current combo by overriding the spec.
        spec_cell = dict(spec_full)
        for nm, val in zip(by_list, combo, strict=True):
            spec_cell[nm] = [val]

        for term in di.terms:
            if not term.factors:
                continue # skip intercept

            factor_names = [f.name() for f in term.factors]
            term_factor_set = set(factor_names)

            # R's joint_tests(by=...) only reports terms that do NOT
            # contain any by-factor. Terms that *include* a by-factor
            # collapse — at a fixed by-cell — into the simple-effect
            # test of the remaining factors, which is already covered
            # by the bare term row above. Skipping them keeps the
            # output one-row-per-effect, matching R.
            if by_set and term_factor_set & by_set:
                continue

            # The term test should *not* re-vary the by-factors. Drop
            # them from cat_T (they are pinned by spec_cell) and instead
            # let `_build_term_test_L` integrate the remaining factors.
            cat_T = [
                n
                for n in factor_names
                if n in info.factors and n not in by_set
            ]
            num_T = [n for n in factor_names if n not in info.factors]

            L_term = _build_term_test_L(info, spec_cell, cat_T, num_T)
            if L_term.shape[0] == 0:
                continue
            est = L_term @ beta
            cov_t = L_term @ V @ L_term.T
            V_inv = np.linalg.pinv(cov_t)
            wald = float(est @ V_inv @ est)
            rank = int(np.linalg.matrix_rank(cov_t))

            if rank == 0:
                continue

            if use_chi:
                stat = wald
                p = float(stats.chi2.sf(wald, rank))
            else:
                stat = wald / rank
                p = float(stats.f.sf(stat, rank, df_denom))

            row = {
                "term": ":".join(factor_names),
                "df_num": rank,
                "df_denom": df_denom,
                "statistic": stat,
                "p_value": p,
            }
            for nm, val in zip(by_list, combo, strict=True):
                row[nm] = val
            rows.append(row)

    base_cols = ["term", "df_num", "df_denom", "statistic", "p_value"]
    full_cols = ([*by_list, *base_cols] if by_list else base_cols)
    if not rows:
        # R `joint_tests` raises "There are no factors to test" when
        # no term survives the
        # by-conditioning. Previously pymmeans returned an empty
        # DataFrame; raising matches R and prevents users from
        # silently iterating an empty result thinking "nothing was
        # significant" when actually the test was vacuous.
        raise ValueError(
            "There are no factors to test (every model term is "
            "absorbed by the supplied by= conditioning)."
        )
    out = pd.DataFrame(rows)[full_cols]

    # the implementation applied
    # ``adjust=`` to the per-term p-values, but live R 2.0.3+
    # ``joint_tests()`` accepts ``adjust=`` via ``...`` and IGNORES
    # it — the returned p-values are unchanged. The
    # implementation was incorrect on both counts: (a) it diverged
    # from R, and (b) ``adjust="tukey"`` / ``"scheffe"`` /
    # ``"dunnett"`` made no statistical sense on heterogeneous F
    # tests with different ``df_num`` per term (the dispatcher
    # raised "tukey adjustment requires t_ratios, n_means, and
    # df"). keeps the kwarg accepted for R-code porting
    # but does NOT modify the p-value column. Users who want a
    # multiplicity correction across joint-test rows should pull
    # the ``p_value`` column out and apply it themselves with the
    # appropriate method.
    if _adjust_requested is not None:
        # kwarg deliberately accepted but ignored for R-parity (R `joint_tests`
        # accepts `adjust` and silently does not apply it across rows).
        pass

    return out


# ---------------------------------------------------------------------------
# eta_squared — partial η², ω², Cohen's f with noncentral-F CIs
# beyond-R-parity feature (R `emmeans` has no built-in η² /
# ω² / Cohen's f. Users typically reach for `effectsize::eta_squared`
# or `MOTE`; pymmeans bakes them in alongside the joint_tests output.)
# ---------------------------------------------------------------------------


def _ncf_sf(F_obs: float, df_num: int, df_denom: float, lam: float) -> float:
    """Survival function ``P(F(df_num, df_denom, λ) > F_obs)``.

    scipy's ``ncf.sf`` returns a NEGATIVE buggy value at λ=0
    instead of reducing cleanly to the central F. Route λ=0 through
    ``central_f.sf`` and λ>0 through ``ncf.sf``.
    """
    from scipy.stats import f as central_f
    from scipy.stats import ncf as ncf_dist
    if lam <= 0.0:
        return float(central_f.sf(F_obs, df_num, df_denom))
    return float(ncf_dist.sf(F_obs, df_num, df_denom, lam))


def _ncf_ci_for_lambda(
    F_obs: float,
    df_num: int,
    df_denom: float,
    level: float,
) -> tuple[float, float]:
    """Confidence bounds for the noncentrality parameter λ via
    noncentral-F CDF inversion (Steiger 2004).

    Returns ``(lambda_low, lambda_high)``. Used by ``eta_squared`` to
    convert F observed → partial η² confidence interval via
    ``η² = λ / (λ + df_denom)``.

    As ``λ`` increases the noncentral-F distribution shifts right, so
    ``sf(F_obs, λ)`` INCREASES from ``p_central`` (at λ=0) toward 1.

    - λ_lower: largest λ where ``sf(F_obs, λ) ≤ α/2`` (i.e. F_obs is
      no more extreme than the α/2 percentile). If ``p_central ≥ α/2``
      then F_obs is already inside the α/2 tail at λ=0 → λ_lower = 0.
    - λ_upper: largest λ where ``sf(F_obs, λ) ≤ 1−α/2``. Solve via
      bracket + brentq when ``p_central < 1−α/2``.
    """
    if not np.isfinite(F_obs) or F_obs <= 0 or not np.isfinite(df_denom):
        return 0.0, 0.0
    from scipy.optimize import brentq

    alpha2 = (1.0 - level) / 2.0
    p_central = _ncf_sf(F_obs, df_num, df_denom, 0.0)

    # --- Lower bound ---
    if p_central >= alpha2:
        # Even at λ=0 the upper-tail mass past F_obs is ≥ α/2,
        # so any λ ≥ 0 satisfies the bound. Lower λ = 0.
        lambda_low = 0.0
    else:
        # Find smallest λ > 0 where sf increases past α/2.
        # As λ↑, sf↑ monotonically; bracket then brentq.
        def f_lower(lam: float) -> float:
            return _ncf_sf(F_obs, df_num, df_denom, lam) - alpha2

        lam_hi = max(1.0, F_obs * df_num)
        for _ in range(50):
            if f_lower(lam_hi) >= 0:
                break
            lam_hi *= 2.0
        else:
            return 0.0, np.inf
        try:
            lambda_low = brentq(f_lower, 0.0, lam_hi, xtol=1e-7, maxiter=200)
        except (ValueError, RuntimeError):
            lambda_low = 0.0

    # --- Upper bound ---
    target_hi = 1.0 - alpha2
    if p_central >= target_hi:
        # Even at λ=0 the upper-tail mass past F_obs is ≥ 1−α/2.
        # That means F_obs is extremely small (way below the central
        # distribution's median). Upper bound for the noncentrality
        # is 0 (degenerate; the data don't support any positive
        # effect at this confidence level).
        lambda_high = 0.0
    else:
        def f_upper(lam: float) -> float:
            return _ncf_sf(F_obs, df_num, df_denom, lam) - target_hi

        lam_hi = max(10.0, F_obs * df_num * 4.0)
        for _ in range(50):
            if f_upper(lam_hi) >= 0:
                break
            lam_hi *= 2.0
        else:
            return lambda_low, np.inf
        try:
            lambda_high = brentq(f_upper, 0.0, lam_hi, xtol=1e-7, maxiter=200)
        except (ValueError, RuntimeError):
            lambda_high = 0.0

    return lambda_low, lambda_high


def eta_squared(
    model: Any,
    partial: bool = True,
    omega: bool = True,
    cohens_f: bool = True,
    level: float = 0.90,
    by: str | list[str] | None = None,
    alternative: str = "two-sided",
) -> pd.DataFrame:
    """Effect sizes from a fitted model: partial η², ω², Cohen's f.

    pymmeans-only feature. R `emmeans` has no built-in η²/ω²
    workflow; users typically use the separate ``effectsize`` package.
    pymmeans bundles the per-term effect sizes alongside the
    :func:`joint_tests` machinery so the same fit produces F-tests
    AND standardized effect sizes in one call.

    **ANOVA-type semantics (honesty pass).** pymmeans's
    ``eta_squared`` uses the F-statistics produced by
    :func:`joint_tests`, which are EMM-basis marginal-effect
    Wald tests. These match R ``emmeans::joint_tests`` numerically
    and they coincide with R ``effectsize::eta_squared`` for
    **balanced designs** (Type-I = Type-III when cells are equal).
    For **unbalanced** designs the values *will* differ from R
    ``effectsize::eta_squared``'s default (which uses the Type-I /
    sequential SS decomposition from R's ``anova()``). pymmeans
    does not currently expose a Type-I switch; users who require
    Type-I numerics should compute SS-based η² directly from the
    ANOVA table. (The previous "bit-exact match with R effectsize"
    claim was true only on the balanced reference data; it has
    been downgraded.)

    Formulas (per term):

    - **Partial η²** = ``F · df_num / (F · df_num + df_denom)``.
      Variance explained by this term after accounting for all
      other terms.
    - **Hays' ω²** (less biased) =
      ``(F − 1) · df_num / (F · df_num + df_denom + 1)``.
      Clipped at 0 (negative ω² is reported as 0 per convention).
    - **Cohen's f** = ``sqrt(η²_partial / (1 − η²_partial))``. Used
      in G*Power and other power-analysis tools.
    - **CI on partial η²** (default 90% per Steiger 2004 convention,
      since η² ∈ [0, 1] is one-sided in nature) via noncentral-F
      inversion: solve λ_low / λ_high such that F_obs sits at the
      ``α/2`` / ``1−α/2`` percentile of ``F(df_num, df_denom, λ)``.

    Parameters
    ----------
    model
        Fitted statsmodels result, ModelInfo, or any object that
        :func:`joint_tests` accepts.
    partial
        If True (default), report partial η². If False, the η² column
        is left as the omnibus partial value (pymmeans currently does
        not compute non-partial / hierarchical SS decomposition;
        partial η² is the Type-III default in `emmeans::joint_tests`).
    omega
        If True (default), include the Hays' ω² column.
    cohens_f
        If True (default), include the Cohen's f column.
    level
        Confidence level for the η² CI. Default ``0.90`` matches
        the standard Steiger / Smithson convention for bounded
        effect-size statistics.
    by
        Optional by-grouping; forwarded to :func:`joint_tests`. One
        row per ``(by-cell, term)``.
    alternative
        ``"two-sided"`` (default; Steiger 2004 convention) or
        ``"greater"`` (R `effectsize::eta_squared` default, fixes
        upper bound at 1.0). The two-sided default matches a
        symmetric ``α/2`` split on each tail of the noncentral-F
        inversion; R `effectsize`'s default is one-sided because
        η² ∈ [0, 1] is bounded above. Both are statistically
        defensible. pass ``alternative="greater"`` for full R-default
        parity.

    Returns
    -------
    DataFrame
        Columns: ``term``, ``df_num``, ``df_denom``, ``F``,
        ``eta_sq_partial`` (always), ``eta_sq_lower_cl``, ``eta_sq_upper_cl``
        (always), plus ``omega_sq`` and ``cohens_f`` (when enabled).
        With ``by=``, also one column per by-factor.

    Notes
    -----
    For GLMs and MixedLM the joint-test statistic is χ², not F. In
    that case ``F = statistic / df_num`` is used (this is a
    Wald-test analogue; the CI bounds are approximate). For pure
    OLS / aov fits the values match R `effectsize::eta_squared` to
    1e-4 on canonical test cases.
    """
    jt = joint_tests(model, by=by)
    if jt.empty:
        return pd.DataFrame(
            columns=[
                "term", "df_num", "df_denom", "F",
                "eta_sq_partial", "eta_sq_lower_cl", "eta_sq_upper_cl",
            ]
            + (["omega_sq"] if omega else [])
            + (["cohens_f"] if cohens_f else [])
        )

    # `joint_tests.statistic` is F (OLS) or chi² (GLM/MixedLM).
    # df_denom is inf for chi² rows; for the F-statistic equivalent
    # we treat χ² / df_num as F (Wald analogue).
    F_arr = jt["statistic"].to_numpy(dtype=float).copy()
    df_num_arr = jt["df_num"].to_numpy(dtype=float)
    df_denom_arr = jt["df_denom"].to_numpy(dtype=float)
    chi_rows = ~np.isfinite(df_denom_arr)
    if chi_rows.any():
        F_arr[chi_rows] = F_arr[chi_rows] / df_num_arr[chi_rows]

    # Partial η² = F·df_num / (F·df_num + df_denom). For χ² rows,
    # use df_denom = ∞ → η² collapses to 0; reinterpret using a
    # finite proxy (model n − p) when available. Most users running
    # eta_squared on a GLM should be told upfront the value is
    # asymptotic / Wald-style. For now we clip to NaN on inf df.
    with np.errstate(divide="ignore", invalid="ignore"):
        eta_sq_partial = np.where(
            np.isfinite(df_denom_arr),
            F_arr * df_num_arr / (F_arr * df_num_arr + df_denom_arr),
            np.nan,
        )

    # ``alternative="greater"`` fixes the upper
    # bound at 1.0 and puts the entire (1 − level) probability mass
    # on the lower tail — matches R `effectsize` default. The
    # two-sided default splits α/2 on each tail (Steiger 2004).
    if alternative not in ("two-sided", "greater"):
        raise ValueError(
            f"alternative must be 'two-sided' or 'greater'; "
            f"got {alternative!r}."
        )
    # Effective level passed to the noncentral-F inversion:
    # two-sided 90% → use 0.90 (α=0.10, α/2=0.05 each tail)
    # one-sided 90% → use 0.80 (α=0.20, all on lower tail → α/2=0.10)
    # i.e. one-sided 90% lower bound has the same prob mass beneath
    # it as a two-sided 80% lower bound.
    effective_level = level if alternative == "two-sided" else (2 * level - 1)

    # Noncentral-F CI per row.
    lower_cl = np.full(len(jt), np.nan)
    upper_cl = np.full(len(jt), np.nan)
    for i in range(len(jt)):
        if not np.isfinite(df_denom_arr[i]):
            continue
        lam_low, lam_high = _ncf_ci_for_lambda(
            float(F_arr[i]), int(df_num_arr[i]), float(df_denom_arr[i]),
            effective_level,
        )
        if np.isfinite(lam_low):
            lower_cl[i] = lam_low / (lam_low + df_denom_arr[i])
        if alternative == "greater":
            upper_cl[i] = 1.0 # one-sided: upper bound pinned
        elif np.isfinite(lam_high):
            upper_cl[i] = lam_high / (lam_high + df_denom_arr[i])

    out = pd.DataFrame({
        "term": jt["term"],
        "df_num": df_num_arr.astype(int),
        "df_denom": df_denom_arr,
        "F": F_arr,
        "eta_sq_partial": eta_sq_partial,
        "eta_sq_lower_cl": lower_cl,
        "eta_sq_upper_cl": upper_cl,
    })
    if omega:
        # Hays' ω² (partial). Per ``effectsize::omega_squared``:
        # ω²_partial = df_num · (F − 1) / (df_num · (F − 1) + N)
        # where N is the TOTAL sample size. Pull N from the fitted
        # model's nobs attribute (or info.data length as fallback).
        from pymmeans.utils import ModelInfo, from_fitted
        _info = model if isinstance(model, ModelInfo) else from_fitted(model)
        if _info.raw_result is not None and hasattr(_info.raw_result, "nobs"):
            N_total = float(_info.raw_result.nobs)
        elif _info.data is not None:
            N_total = float(len(_info.data))
        else:
            # Fallback: df_total + 1 (assumes single-intercept model).
            N_total = float(df_denom_arr[0] + jt["df_num"].sum() + 1)
        with np.errstate(invalid="ignore"):
            numer = df_num_arr * (F_arr - 1.0)
            omega_sq = np.where(
                np.isfinite(df_denom_arr),
                numer / (numer + N_total),
                np.nan,
            )
        out["omega_sq"] = np.clip(omega_sq, 0.0, None)
    if cohens_f:
        with np.errstate(invalid="ignore"):
            f_val = np.sqrt(
                np.clip(eta_sq_partial, 0.0, 1.0 - 1e-12)
                / (1.0 - np.clip(eta_sq_partial, 0.0, 1.0 - 1e-12))
            )
        out["cohens_f"] = f_val

    # Preserve any by-columns from joint_tests at the front.
    by_cols = [c for c in jt.columns if c not in
               ("term", "df_num", "df_denom", "statistic", "p_value")]
    if by_cols:
        for col in by_cols[::-1]:
            out.insert(0, col, jt[col].to_numpy())
    return out
