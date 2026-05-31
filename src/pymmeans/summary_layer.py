"""R `summary.emmGrid` / `confint.emmGrid` / `test.emmGrid` layer.

R `emmeans` separates the EMM
computation from the *display / inference* layer. `summary(emm,
infer=, level=, side=, null=, delta=, ...)` lets users:

- Toggle CI display via ``infer[0]``.
- Toggle hypothesis-test display via ``infer[1]``.
- Recompute CIs at a different ``level=``.
- Re-apply a different ``adjust=`` without rebuilding the contrast.
- Switch ``type='link' <-> 'response'`` post-hoc.
- One-sided tests / CIs via ``side='<'`` or ``'>'``.
- Test against a non-zero ``null=`` (e.g. ratio of 1 on response scale).
- Equivalence / non-inferiority via ``delta=``.

R `emmeans` exposes inference via the post-hoc functions
``summary`` / ``confint`` / ``test`` / ``update`` / ``as_r_frame``
rather than baking CIs and tests into the EMM result itself.
`pymmeans` mirrors that surface here.

The signatures align with R's defaults:

    summary(obj, infer=None, level=None, adjust=None,
            type=None, side='two-sided', null=0.0, delta=0.0)
        # infer=None: EMM -> (True, False), contrast -> (False, True)
        # adjust=None: obj.adjust or emm_options('adjust') or 'none'
        # level=None : emm_options('level') or obj.level or 0.95
        # side: accepts 'two-sided' / '<' / '>' plus R aliases
        # 'noninferiority', 'nonsuperiority', 'equivalence',
        # 'upper', 'lower', 'right', 'left'.
    confint(obj, level=None, side='two-sided', adjust=None, type=None)
        # level=None propagates emm_options('level').
    test(obj, null=0.0, side='two-sided', delta=0.0, adjust=None,
         type=None)

Returns ``pandas.DataFrame`` with the same row order as the input
result's frame. Use :func:`as_r_frame` to rename columns to R's
dot-separated conventions (``lower.CL``, ``t.ratio``, etc.).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _is_emm_result(obj: Any) -> bool:
    """Detect an `EMMResult` (or duck-typed equivalent) without
    importing it at module load (the `emmeans` module imports
    `summary_layer`; circular-import dance).

    c: also recognises ``emtrends()`` frames (value
    column is ``<var>.trend`` rather than ``emmean``). The
    contrast-vs-EMM discriminator is presence of ``emmean`` /
    ``.trend`` (EMM-like) vs ``estimate`` / ``ratio`` (contrast-like).

    also recognise ``MLEMMResult`` — it has
    ``ml_info`` / ``frame`` (with ``emmean``) but no ``linfct`` /
    ``model_info``. Without this branch, the per-object-default
    ``infer`` resolution treats ML EMMs as contrasts (defaulting to
    ``(False, True)`` tests-only), so ``summary(ml_em_b)`` silently
    emitted analytic z-tests instead of preserving the bootstrap
    percentile CIs.
    """
    if hasattr(obj, "ml_info") and "emmean" in getattr(
        obj, "frame", pd.DataFrame()
    ).columns:
        return True
    if not (
        hasattr(obj, "frame")
        and hasattr(obj, "linfct")
        and hasattr(obj, "model_info")
    ):
        return False
    cols = getattr(obj, "frame", pd.DataFrame()).columns
    if "emmean" in cols:
        return True
    # emtrends-style: any `<var>.trend` column
    return any(isinstance(c, str) and c.endswith(".trend") for c in cols)


def _is_contrast_result(obj: Any) -> bool:
    """Detect a `ContrastResult` (or duck-typed equivalent).

 also recognize a contrast
    that has been through `regrid_response()` — its value column is
    `ratio` (not `estimate`). Without this, `summary` / `update` on a
    regridded log-family contrast silently skipped the adjustment
    re-apply, kept p_value unchanged, and gave wrong-scale test stats.

    The richer guard — present `adjust` attribute + present `linfct`
    + value column among (estimate, ratio) — also rules out plain
    EMMResults (no `adjust` field) and hand-built DataFrames.
    """
    if not (hasattr(obj, "frame") and hasattr(obj, "linfct")):
        return False
    if not hasattr(obj, "adjust"):
        return False
    cols = getattr(obj, "frame", pd.DataFrame()).columns
    return any(c in cols for c in ("estimate", "ratio"))


def _apply_response_inverse(
    info: Any,
    mu: np.ndarray,
    se: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(, #3, #4): factor out the response-
    scale transform that ``emmeans(type='response')`` runs. Returns
    (mu_resp, se_resp, lower_resp, upper_resp).

    Mirrors the body of ``emmeans.emmeans()`` at the response-scale
    branch:
    - GLM family: ``family.link.inverse`` + ``inverse_deriv``.
    - LHS transform: ``detect_transform(response_name).inverse`` +
      ``_interval_inverse`` for non-monotone CI handling.

    Used by `summary(type='response')` and `update(type='response')`
    so they don't have to re-call `emmeans()` (which would also rerun
    the whole reference-grid construction).
    """
    from pymmeans.transforms import (
        _interval_inverse as _iv_inv,
    )
    from pymmeans.transforms import (
        detect_transform as _detect,
    )

    if info.family is not None:
        link = info.family.link
        mu_resp = link.inverse(mu)
        se_resp = np.abs(link.inverse_deriv(mu)) * se
        lo_resp = link.inverse(lower)
        up_resp = link.inverse(upper)
        return mu_resp, se_resp, np.minimum(lo_resp, up_resp), np.maximum(lo_resp, up_resp)

    tran = _detect(info.response_name or "")
    if tran is None:
        response_name = info.response_name or ""
        if "(" in response_name:
            raise ValueError(
                f"summary/update(type='response') with response_name="
                f"{response_name!r} cannot be auto-back-transformed "
                "because the inner expression is composite. Pass an "
                "explicit `tran=` via `regrid_response` on the link-"
                "scale result first."
            )
        # plain identity LHS — link == response (no-op)
        return mu, se, lower, upper
    mu_resp = tran.inverse(mu)
    se_resp = np.abs(tran.inverse_deriv(mu)) * se
    lower_resp, upper_resp = _iv_inv(tran, lower, upper)
    return mu_resp, se_resp, lower_resp, upper_resp


def _recompute_response_from_link(
    obj: Any,
    level: float,
    adjust: str | None = None,
    side: str = "two-sided",
    bias_adjust: bool = False,
    sigma: float | None = None,
) -> Any:
    """Take a link-scale result; return a new result on the response
    scale. CIs are recomputed at the requested level on the LINK scale
    first, then transformed (matches emmeans response-scale convention).

    when ``obj`` is a contrast, route through
    :func:`regrid_response`. A contrast on a log-family scale should
    come back as a *ratio* (``A / B``), not as a generic inverse of a
    difference — and `regrid_response` already encodes the
    ``contrast_inverse`` / label-rename semantics (including for
    shifted-log families like ``log1p`` and ``genlog`` where the
    contrast back-transform differs from the EMM back-transform).

    ``adjust=`` / ``side=`` thread the requested
    multiplicity adjustment through to the link-scale CI critical
    value before the inverse transform. Without this, a response-
    scale contrast's CI was computed against the unadjusted Wald
    quantile and the back-transformed bounds were silently too
    narrow vs R's adjusted output.

    ``bias_adjust`` threads the
    Taylor bias correction through to ``regrid_response`` (for the
    contrast branch) and to a dedicated regrid call (for the EMM
    branch). Without this, ``summary(rb, ...)`` /
    ``confint(rb, ...)`` / ``update(rb, ...)`` on a result built via
    ``regrid_response(..., bias_adjust=True)`` silently reverted to
    the unbiased back-transform — matching R `summary.emmGrid`'s
    ``bias.adjust = TRUE`` semantics requires preserving the flag.
    """
    from dataclasses import replace as _dc_replace

    from scipy import stats

    info = obj.model_info
    frame = obj.frame.copy()
    vcol = _value_col(obj)
    mu = frame[vcol].to_numpy(dtype=float)
    se = frame["SE"].to_numpy(dtype=float)
    df_arr = frame["df"].to_numpy(dtype=float)

    # Compute Wald CIs at the requested level ON THE LINK scale, using
    # the family-wise critical value when an adjustment is requested.
    if (
        adjust
        and adjust.lower() != "none"
        and len(mu) > 1
        and _is_contrast_result(obj)
    ):
        crit = _contrast_ci_critical_value(obj, adjust, level, df_arr, side)
    elif (
        adjust
        and adjust.lower() != "none"
        and len(mu) > 1
        and _is_emm_result(obj)
    ):
        crit = _emm_ci_critical_value(adjust, level, df_arr, len(mu), side)
    elif side == "two-sided":
        alpha = 1.0 - level
        crit = stats.t.ppf(1.0 - alpha / 2.0, df_arr)
    else:
        crit = stats.t.ppf(level, df_arr)
    lower = mu - crit * se
    upper = mu + crit * se

    # One-sided ``side=`` should leave the unconstrained endpoint at
    # ±Inf on the response scale.
    # R `summary.emmGrid(..., side='>')` returns ``upper.CL = Inf``;
    # ``side='<'`` returns ``lower.CL = -Inf``. Stamp those after the
    # regrid / inverse-transform step so any back-transform of the
    # one-sided link-scale endpoint doesn't accidentally produce a
    # finite "upper" value that R never reports.
    def _stamp_one_sided(out_obj):
        if side == "two-sided":
            return out_obj
        new_frame = out_obj.frame.copy()
        if side == ">":
            new_frame["upper_cl"] = np.inf
        else: # side == "<"
            new_frame["lower_cl"] = -np.inf
        return _dc_replace(out_obj, frame=new_frame)

    if _is_contrast_result(obj):
        # defer to `regrid_response` for the
        # contrast-back-transform semantics. First stamp link-scale CI
        # endpoints at the requested level so regrid uses them; then
        # regrid will apply `contrast_inverse` to the point, SE, and
        # CI endpoints and rename `estimate` -> `ratio` for log
        # families. Carry the new level on the returned object.
        # thread bias_adjust through.
        # thread sigma through too.
        from pymmeans.transforms import regrid_response as _regrid

        frame["lower_cl"] = lower
        frame["upper_cl"] = upper
        link_obj = _dc_replace(obj, frame=frame)
        regridded = _regrid(
            link_obj, bias_adjust=bias_adjust, sigma=sigma,
        )
        if "level" in regridded.__dataclass_fields__:
            regridded = _dc_replace(regridded, level=level)
        return _stamp_one_sided(regridded)

    # EMM branch. when bias_adjust=True is requested,
    # we cannot use `_apply_response_inverse` (which is a plain
    # delta-method inverse without the Taylor correction). Route
    # through `regrid_response` instead so the Taylor formula
    # ``exp(mu) * (1 + sigma^2/2)`` (log-family) or its sqrt /
    # logit / etc. analogue is applied to the point estimate and
    # the SE is the bias-adjusted derivative. CI endpoints come from
    # transforming the link-scale endpoints; this matches R's
    # `summary.emmGrid(..., type='response', bias.adjust=TRUE)`.
    # pass `sigma=` through too.
    if bias_adjust:
        from pymmeans.transforms import regrid_response as _regrid

        frame["lower_cl"] = lower
        frame["upper_cl"] = upper
        link_obj = _dc_replace(obj, frame=frame)
        regridded = _regrid(link_obj, bias_adjust=True, sigma=sigma)
        if "level" in regridded.__dataclass_fields__:
            regridded = _dc_replace(regridded, level=level)
        return _stamp_one_sided(regridded)

    mu_r, se_r, lo_r, up_r = _apply_response_inverse(info, mu, se, lower, upper)

    frame[vcol] = mu_r
    frame["SE"] = se_r
    frame["lower_cl"] = lo_r
    frame["upper_cl"] = up_r
    fields = {"frame": frame, "type": "response"}
    if "level" in obj.__dataclass_fields__:
        fields["level"] = level
    return _stamp_one_sided(_dc_replace(obj, **fields))


def _recompute_link_from_response(obj: Any) -> Any:
    """Refuse — going response → link without the original linfct is
    ill-defined (the inverse-link transform isn't bijective for all
    families)."""
    raise ValueError(
        "summary/update cannot convert a response-scale result back "
        "to link scale. Re-call emmeans(..., type='link') with the "
        "original model."
    )


def _response_to_link_result(obj: Any) -> Any:
    """Rebuild a link-scale view of a response-scale EMM / contrast.

 `summary(response_emm,
    level=...)` and `update(response_emm, level=...)` used to refuse
    on the grounds that the link-scale SEs aren't on the frame. But
    `linfct @ beta` and `sqrt(diag(linfct V linfct.T))` reconstruct
    them exactly (the `linfct` matrix and `model_info.vcov` are
    preserved through `regrid_response` and `type='response'`
    construction).

    Works for EMMResult (`emmean` column) and ContrastResult (either
    `estimate` or `ratio`). The returned object has the same dataclass
    type as the input, with `type='link'`, the canonical link value
    column, link-scale SE, and link-scale CI endpoints computed at
    the input's stored level (the caller is expected to follow up
    with `_recompute_response_from_link(..., level=new_lvl)` to land
    on response at the requested level).
    """
    from dataclasses import replace as _dc_replace

    from scipy import stats

    info = obj.model_info
    linfct = obj.linfct
    beta = info.beta
    vcov = info.vcov
    eta = linfct @ beta
    cov = linfct @ vcov @ linfct.T
    with np.errstate(invalid="ignore"):
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    frame = obj.frame.copy()
    df_arr = frame["df"].to_numpy(dtype=float)
    level = getattr(obj, "level", 0.95)
    crit = stats.t.ppf(1.0 - (1.0 - level) / 2.0, df_arr)
    lower = eta - crit * se
    upper = eta + crit * se

    # Decide which column to write the link-scale point into.
    if "emmean" in frame.columns:
        link_col = "emmean"
    elif "ratio" in frame.columns:
        # Regridded log-family contrast: rename `ratio` -> `estimate`
        # on the link-scale view, matching what `pairs()` produces.
        frame = frame.rename(columns={"ratio": "estimate"})
        link_col = "estimate"
    elif "estimate" in frame.columns:
        link_col = "estimate"
    else:
        raise ValueError(
            "Cannot rebuild link view: object frame has no recognised "
            f"value column (got {list(frame.columns)})."
        )
    frame[link_col] = eta
    frame["SE"] = se
    frame["lower_cl"] = lower
    frame["upper_cl"] = upper

    # If we renamed ratio -> estimate, also flip any "A / B" labels
    # back to "A - B" so the link-scale frame round-trips.
    if "contrast" in frame.columns:
        frame["contrast"] = (
            frame["contrast"].astype(str).str.replace(" / ", " - ", regex=False)
        )

    fields = {"frame": frame}
    if "type" in obj.__dataclass_fields__:
        fields["type"] = "link"
    # do NOT zero out `bias_adjust` here. The link-scale
    # view is just a numeric reconstruction of L @ beta; whether the
    # original response-scale frame had bias adjustment applied is a
    # *user intent* the caller (summary / update / confint) needs to
    # see so it can re-apply the Taylor correction on the way back.
    # Previously we set `bias_adjust=False`, which silently dropped
    # the flag through every recompute path.
    return _dc_replace(obj, **fields)


def _value_col(obj: Any) -> str:
    """Return the canonical value column for `obj.frame`.

    c: route through `utils.detect_value_column` so
    ``emtrends`` results (with ``<var>.trend`` value columns) are
    recognised. Previously the hard-coded list was emm/estimate/ratio
    only — `summary(emtrends(...))` crashed with
    "cannot find value column".
    """
    from pymmeans.utils import detect_value_column

    found = detect_value_column(obj.frame)
    if found is not None:
        return found[1]
    raise ValueError(
        f"summary_layer: cannot find value column in {list(obj.frame.columns)}"
    )


def _contrast_ci_critical_value(
    obj: Any,
    adj: str,
    level: float,
    df_arr: np.ndarray,
    side: str,
) -> np.ndarray:
    """family-wise critical value for ContrastResult CIs.

    R `confint.emmGrid` on a contrast frame widens the CI by the
    family adjustment — same rule as on EMM rows but with two
    differences for `tukey` / `dunnett`:

    1. The Tukey critical value uses ``n_means`` (the size of the
       UNDERLYING EMM family that produced the pairwise contrast),
       not ``n_rows`` (the number of pairwise comparisons).
       e.g. pairs(em) for k=3 means produces 3 contrasts but the
       Tukey q uses k=3, not 3.
    2. For Dunnett-family contrasts (`trt.vs.ctrl`), R uses the MVT
       quantile under the contrast correlation matrix. v0.1 falls
       back to the bonferroni-equivalent quantile (the same
       fallback `_emm_ci_critical_value` uses) so the CI is at
       least conservative; emits a warning.

    Pulls family meta from ``obj._adjust_meta`` when available
    (populated by `_contrast_one_family`). For by-grouped contrasts
    each family has its own ``n_means`` / ``df`` / ``correlation``;
    we apply the per-family critical value to its slice.

    Returns a critical-value array of shape (n_rows,) so the caller
    can do ``point ± crit * SE`` uniformly across the frame.
    """
    from scipy import stats

    n = len(df_arr)
    crit = np.zeros(n)
    adj_lower = (adj or "none").lower()
    sequential = {"holm", "hommel", "hochberg", "bh", "by", "fdr"}

    # No adjustment / sequential: per-row t-quantile, no widening.
    if adj_lower in sequential or adj_lower == "none":
        if side == "two-sided":
            return stats.t.ppf(1.0 - (1.0 - level) / 2.0, df_arr)
        return stats.t.ppf(level, df_arr)

    meta = getattr(obj, "_adjust_meta", None)
    families = (meta or {}).get("families") if meta else None

    def _crit_for(adj_name: str, n_means: int, n_rows_fam: int,
                  df_fam: float, slice_arr: np.ndarray) -> np.ndarray:
        name = adj_name.lower()
        if name == "bonferroni":
            alpha_per = (1.0 - level) / max(1, n_rows_fam)
        elif name == "sidak":
            alpha_per = 1.0 - level ** (1.0 / max(1, n_rows_fam))
        elif name == "scheffe":
            if side != "two-sided":
                alpha_per = (1.0 - level) / max(1, n_rows_fam)
            else:
                with np.errstate(invalid="ignore"):
                    f_crit = stats.f.ppf(level, max(1, n_means - 1), slice_arr)
                return np.sqrt((n_means - 1) * f_crit)
        elif name == "tukey":
            from scipy.stats import studentized_range as _sr
            # Tukey's q uses the UNDERLYING EMM family size, not the
            # number of contrasts.
            q = _sr.ppf(level, max(2, n_means), slice_arr)
            return q / np.sqrt(2.0)
        elif name in ("dunnett", "dunnettx", "mvt"):
            import warnings as _w
            _w.warn(
                f"confint(ct, adjust={name!r}): pymmeans currently uses a "
                "bonferroni-equivalent critical value for the contrast "
                "CI display; the p-values still use the requested "
                "adjustment. Use bootstrap_ci for the exact MVT "
                "simultaneous CI envelope.",
                UserWarning,
                stacklevel=4,
            )
            alpha_per = (1.0 - level) / max(1, n_rows_fam)
        else:
            alpha_per = 1.0 - level

        if side == "two-sided":
            return stats.t.ppf(1.0 - alpha_per / 2.0, slice_arr)
        return stats.t.ppf(1.0 - alpha_per, slice_arr)

    if families:
        for fam in families:
            sl = slice(fam["start"], fam["stop"])
            df_slice = df_arr[sl]
            # prefer the -added valid-row /
            # valid-means counts (excludes non-estimable rows from
            # the multiplicity calculation). Fall back to the raw
            # counts on pickled-from-earlier-version metadata.
            n_means_use = int(fam.get("n_means_valid", fam["n_means"]) or fam["n_means"])
            n_rows_use = int(fam.get("n_rows_valid", fam["n_rows"]) or fam["n_rows"])
            crit[sl] = _crit_for(
                adj_lower,
                n_means_use,
                n_rows_use,
                float(fam["df"]),
                df_slice,
            )
        return crit
    # No family metadata: treat as one family. Use heuristic
    # n_means = ceil(0.5 + sqrt(0.25 + 2*n_rows)) which inverts the
    # pairwise-count formula k(k-1)/2 = n_rows. Doesn't matter for
    # bonferroni / sidak / dunnett (which only need n_rows) — it
    # only matters for tukey / scheffe.
    n_rows_total = n
    n_means_guess = int(np.ceil(0.5 + np.sqrt(0.25 + 2.0 * n_rows_total)))
    crit[:] = _crit_for(
        adj_lower,
        n_means_guess,
        n_rows_total,
        float(np.nanmean(df_arr)),
        df_arr,
    )
    return crit


def _emm_ci_critical_value(
    adj: str,
    level: float,
    df_arr: np.ndarray,
    n_means: int,
    side: str,
) -> np.ndarray:
    """family-wise critical value for EMM-row CIs.

    R `summary.emmGrid(..., adjust=…, infer=(True, True))` widens
    each EMM's CI to the family-wise critical value matching the
    requested adjustment. We implement the closed-form critical
    values that have one (bonferroni, sidak, scheffe, tukey,
    dunnett); for *sequential* methods (holm / fdr / BY / hochberg /
    hommel) R doesn't adjust CIs at all — the rejection boundary is
    data-dependent so no single per-comparison alpha applies.

    Returns a critical value array (broadcast against df_arr). For
    two-sided CIs at confidence level ``level``, the CI is
    ``[point - crit*SE, point + crit*SE]``.

    For one-sided CIs (side='<' or '>'), the critical value is the
    one-tail t-quantile at the family-wise alpha; the caller wraps
    one bound with ±inf.
    """
    from scipy import stats

    adj_lower = (adj or "none").lower()
    m = int(n_means)

    # Sequential methods: don't adjust the CI critical value. R
    # leaves it at the unadjusted t-quantile.
    sequential = {"holm", "hommel", "hochberg", "bh", "by", "fdr"}
    if adj_lower in sequential or adj_lower == "none" or m <= 1:
        alpha_per = 1.0 - level
    elif adj_lower == "bonferroni":
        alpha_per = (1.0 - level) / m
    elif adj_lower == "sidak":
        alpha_per = 1.0 - level ** (1.0 / m)
    elif adj_lower == "scheffe":
        # Scheffé's simultaneous bound: critical value is
        # sqrt((m-1) * F_{m-1, df, 1-alpha}). Returned as a t-style
        # crit so the caller can use the same `point ± crit*SE`
        # formula.
        if side != "two-sided":
            # R falls back to bonferroni-style alpha for one-sided
            # under Scheffé.
            alpha_per = (1.0 - level) / m
        else:
            with np.errstate(invalid="ignore"):
                f_crit = stats.f.ppf(level, m - 1, df_arr)
            crit = np.sqrt((m - 1) * f_crit)
            return crit
    elif adj_lower == "tukey":
        # Studentized-range critical value at level `level`. The CI
        # half-width is `q / sqrt(2) * SE` (because the pair-difference
        # studentized statistic has the same distribution).
        from scipy.stats import studentized_range as _sr
        q = _sr.ppf(level, m, df_arr)
        crit = q / np.sqrt(2.0)
        return crit
    elif adj_lower in ("dunnett", "dunnettx", "mvt"):
        # These need an MVT (or `.pdunnx`) critical value. Computing
        # it here would require either a Genz QMC root-find (slow
        # and overkill for the CI display) or the dunnettx mixture
        # formula inverted — neither is in v0.1 scope. Fall back to
        # bonferroni-style alpha so the CI is at least conservative,
        # not under-corrected; emit a warning so the user knows.
        import warnings as _w
        _w.warn(
            f"summary(emm, adjust={adj_lower!r}): pymmeans currently uses "
            "a bonferroni-equivalent critical value for the EMM CI "
            "display; the p-values still use the requested "
            "adjustment. Use bootstrap_ci for the exact dunnett / "
            "mvt simultaneous CI envelope.",
            UserWarning,
            stacklevel=3,
        )
        alpha_per = (1.0 - level) / m
    else:
        alpha_per = 1.0 - level

    if side == "two-sided":
        return stats.t.ppf(1.0 - alpha_per / 2.0, df_arr)
    # One-sided: alpha_per spent on a single tail
    return stats.t.ppf(1.0 - alpha_per, df_arr)


_BY_UNSET = object()


def summary(
    obj: Any,
    infer: tuple[bool, bool] | bool | None = None,
    level: float | None = None,
    adjust: str | None = None,
    type: str | None = None,
    side: str = "two-sided",
    null: float = 0.0,
    delta: float = 0.0,
    by: str | list[str] | None | object = _BY_UNSET,
    bias_adjust: bool | None = None,
    sigma: float | None = None,
    *,
    cross_adjust: str | None = None,
) -> pd.DataFrame:
    """Recompute the CI / test columns on an EMMResult or ContrastResult.

    mirrors R `summary.emmGrid`. The
    input result is NOT mutated; this returns a fresh DataFrame with
    columns adjusted to the requested ``infer`` / ``side`` / ``null``
    / ``delta`` / ``adjust`` / ``level``.

    Parameters
    ----------
    obj
        ``EMMResult`` or ``ContrastResult``.
    infer
        Two-tuple ``(show_ci, show_tests)`` — controls whether CI and
        hypothesis-test columns appear in the output. Passing a single
        bool sets both. ``None`` (default) follows R's per-object
        convention EMMs default to
        ``(True, False)`` (CIs only), contrasts default to
        ``(False, True)`` (tests only). An ``emm_options(infer=)``
        setting takes precedence over the per-object default.
    level
        Confidence level for CIs. ``None`` reuses ``obj.level``.
    adjust
        Multiplicity correction. ``None`` reuses ``obj.adjust``.
        For **ContrastResult**, controls the p-value adjustment over
        the contrast family. For **EMMResult**, widens the CI by the
        family-wise critical value (sidak / bonferroni / tukey /
        scheffe / dunnett / mvt). Both branches honor the requested
        adjustment so a single ``summary(obj, adjust=...)`` call
        produces R-consistent intervals and tests regardless of
        whether ``obj`` is an EMM or a contrast.
    type
        ``'link'`` (default) or ``'response'``. If set, applies the
        appropriate inverse transform / link before computing CIs and
        tests. ``None`` reuses ``obj.type``.
    side
        ``'two-sided'`` (default), ``'<'`` (lower-tailed), or
        ``'>'`` (upper-tailed) for hypothesis tests and CIs.
    null
        Null-hypothesis value on the **link / test scale** (default
        ``0.0``). R keeps ``null`` on the link scale and only
        back-transforms the *display* for response-scale summaries.
        For a log-link model the default ``null=0.0`` corresponds
        to response-scale ratio = 1 ("no effect"); pass
        ``null=log(target_ratio)`` if testing against a non-zero
        ratio.
    delta
        Equivalence / non-inferiority margin. ``delta > 0`` switches
        to two-one-sided-tests (TOST) for equivalence; ``null + delta``
        and ``null - delta`` are the equivalence bounds.
    by
        Override the by-group structure (no-op in current
        implementation; reserved for future cross-group adjustment).

    Returns
    -------
    pandas.DataFrame
        New frame with the requested columns. Original ``obj.frame``
        unchanged.

        When ``obj`` is an :class:`~pymmeans.EmmList`
        #4 — mirrors R ``summary.emm_list``), the return type is a
        dict mapping each member name to its summary DataFrame.
    """
    # R `summary.emm_list` recurses into each member and returns a
    # named list. The Python analogue
    # is a dict keyed by `EmmList.names`. Without this branch the
    # internal `_value_col(obj)` access raises AttributeError, since
    # an EmmList is a tuple (no `.frame`).
    from pymmeans.contrasts import EmmList
    if isinstance(obj, EmmList):
        # Forward kwargs verbatim. ``by`` is forwarded only when the
        # caller passed something — else the recursive call inherits
        # the sentinel default.
        kw = dict(
            infer=infer, level=level, adjust=adjust, type=type,
            side=side, null=null, delta=delta,
            bias_adjust=bias_adjust, sigma=sigma,
        )
        if by is not _BY_UNSET:
            kw["by"] = by
        return {
            name: summary(member, **kw)
            for name, member in zip(obj.names, obj, strict=True)
        }

    from scipy import stats

    # Batch C1 + C2 resolve `infer` default per R's
    # per-object convention, with `emm_options(infer=...)` overriding.
    # Explicit user kwarg always wins; None means "use the default".
    from pymmeans.options import get_emm_option as _opt
    # track whether the user explicitly asked for
    # tests so we can distinguish "I want analytic Wald tests on this
    # bootstrap object (refused)" from "I just typed summary(em_b)
    # and got the default contrast (False, True) — show me the stored
    # CIs instead" (suppress tests silently).
    _infer_was_default = infer is None and _opt("infer", None) is None
    if infer is None:
        infer = _opt("infer", None)
    if infer is None:
        infer = (True, False) if _is_emm_result(obj) else (False, True)
    if isinstance(infer, bool):
        infer = (infer, infer)
    show_ci, show_tests = infer

    # accept R-style side aliases. R `summary.emmGrid`
    # accepts `"two-sided"` / `"equivalence"` (TOST when delta>0),
    # `">"` / `"upper"` / `"right"` / `"noninferiority"`, and
    # `"<"` / `"lower"` / `"left"` / `"nonsuperiority"`. We normalize
    # to the three canonical tokens used internally.
    _SIDE_ALIASES = {
        "two-sided": "two-sided",
        "two.sided": "two-sided",
        "equivalence": "two-sided",
        ">": ">",
        "upper": ">",
        "right": ">",
        "noninferiority": ">",
        "non-inferiority": ">",
        "<": "<",
        "lower": "<",
        "left": "<",
        "nonsuperiority": "<",
        "non-superiority": "<",
    }
    if side not in _SIDE_ALIASES:
        raise ValueError(
            f"side must be one of {sorted(_SIDE_ALIASES)}, got {side!r}."
        )
    side = _SIDE_ALIASES[side]
    if delta < 0:
        raise ValueError(f"delta must be >= 0, got {delta!r}.")

    # precedence is
    # explicit kwarg > emm_options('level') > obj.level > 0.95.
    # The wiring had `getattr(obj, 'level', _opt(...))` which
    # only consulted the option when the obj had no level field — that
    # effectively never fired since every result carries `level`. Flip
    # so `with emm_options(level=0.99): summary(em)` actually changes
    # the displayed CIs.
    default_level = _opt("level", getattr(obj, "level", 0.95))
    lvl = default_level if level is None else float(level)
    if not 0.0 < lvl < 1.0:
        raise ValueError(f"level must be in (0, 1), got {lvl!r}.")

    # (, #3, #4): handle scale changes.
    obj_type = getattr(obj, "type", "link")
    requested_type = type if type is not None else obj_type
    # Adapters for non-canonical families (multinomial, ordinal) stamp
    # their own scale labels on the EMMResult (``"prob"``, ``"latent"``,
    # ``"cum.prob"``) and the user should be able to call
    # ``summary(em)`` without manually translating those labels back to
    # ``"link"`` / ``"response"`` (auditor V11 P1: validator rejected
    # the adapter's default ``type="prob"``). Permit any string that
    # the object already carries (pass-through is always safe), and
    # the two canonical scales for explicit user requests.
    _VALID_SCALES = ("link", "response", "prob", "latent", "cum.prob")
    if requested_type not in _VALID_SCALES and requested_type != obj_type:
        raise ValueError(
            f"type must be one of {_VALID_SCALES} or the object's own "
            f"scale ({obj_type!r}), got {requested_type!r}."
        )

    # posterior results carry
    # PERCENTILE credible intervals from `posterior_emmeans` — Wald /
    # t-quantile recomputation here would silently corrupt them
    # (negative lower bounds on lognormal-style posteriors). Refuse
    # the two state-changes that would force a recompute (scale and
    # level); the CI-preservation branch below ensures the existing
    # percentile CIs round-trip through `summary` unchanged.
    is_posterior = getattr(obj, "inference_kind", "wald") == "posterior"
    if is_posterior:
        if requested_type != obj_type:
            raise ValueError(
                "summary(posterior_emm, type=...) cannot regrid posterior "
                "results because the inverse transform must apply to each "
                "draw, not to the summary mean (Jensen's inequality). "
                "Re-run posterior_emmeans(..., type='response') so the "
                "transform fires per-draw before percentile computation."
            )

    # bootstrap-derived results carry PERCENTILE
    # bootstrap intervals in ``frame['lower_cl']`` / ``frame['upper_cl']``
    # — but the bootstrap *draws* are not stored on the result, so
    # there is no way to re-percentile at a new level / new
    # multiplicity-widening / new one-sided side / new scale. previously
    # ``summary(em_b)`` silently overwrote the percentile CIs with Wald
    # ``t * SE`` intervals — same shape as the posterior path that
    # closed, but for bootstrap. Refuse any state-change kwarg
    # that would force a recompute (scale / level / adjust / side /
    # bias_adjust / sigma); the CI-preservation branch below ensures
    # the existing percentile CIs round-trip through plain
    # ``summary(em_b)`` unchanged. The user-facing workflow is "re-run
    # ``bootstrap_ci(em, level=<new>)`` to bootstrap at a new level".
    is_bootstrap = getattr(obj, "df_method", "default") == "bootstrap"
    if is_bootstrap and show_tests:
        # the bootstrap-recompute lockdown
        # only blocked CI-recompute kwargs. ``show_tests`` itself was
        # not treated as a recompute, so ``summary(ct_b)`` (default
        # contrast infer = (False, True)) silently emitted analytic
        # Wald t-tests derived from ``linfct @ beta`` / ``L V L.T``
        # while the object's stored inference is percentile bootstrap.
        # Same silent inference-mixing class as closed for
        # ``test()`` — but ``summary(em_b, infer=(_, True))`` was the
        # backdoor.
        #
        # Resolution: if the caller did NOT explicitly ask for tests
        # (the per-object default for contrasts is tests-only),
        # silently suppress tests and show stored CIs instead — the
        # user typed ``summary(ct_b)`` to see the bootstrap output,
        # not to ask for Wald analytics. If the caller DID explicitly
        # request tests (``infer=(True, True)`` / ``infer=True`` /
        # ``infer=(False, True)``), refuse with a workflow hint —
        # there's no honest answer.
        if _infer_was_default:
            show_ci, show_tests = True, False
            infer = (True, False)
        else:
            raise ValueError(
                "summary(bootstrap_result, infer=(..., True)) is not "
                "defined — analytic Wald p-values would silently mix "
                "two inference paradigms (percentile bootstrap point "
                "uncertainty vs analytic-Wald test statistics). For "
                "the stored bootstrap CIs, use infer=(True, False); "
                "for true bootstrap inference on contrasts, call "
                "``permutation_test(...)``."
            )
    if is_bootstrap:
        if requested_type != obj_type:
            raise ValueError(
                "summary(bootstrap_emm, type=...) cannot regrid a "
                "bootstrap-derived result because the inverse transform "
                "must apply to each draw, not to the percentile bounds. "
                "Re-run ``bootstrap_ci(em, ...)`` on an EMM that was "
                "already at the desired scale."
            )
        _level_change_b = level is not None and not np.isclose(
            lvl, getattr(obj, "level", lvl)
        )
        _adjust_requested_b = (
            adjust is not None and str(adjust).lower() != "none"
        )
        _side_change_b = side != "two-sided"
        _bias_override_b = bias_adjust is not None or sigma is not None
        if (
            _level_change_b or _adjust_requested_b or _side_change_b
            or _bias_override_b
        ):
            raise ValueError(
                "summary(bootstrap_emm, level=/adjust=/side=/"
                "bias_adjust=/sigma=) cannot recompute percentile "
                "bootstrap intervals at new settings because the "
                "individual bootstrap draws are not stored on the "
                "result — only the precomputed quantiles. To get "
                "intervals at a new level / adjustment / side, "
                "re-run ``bootstrap_ci(em, level=<new>)`` from the "
                "raw (un-bootstrapped) EMM. To inspect the stored "
                "percentile CIs unchanged, call summary() with no "
                "recompute kwargs."
            )

    # If asking for a scale change OR a level change AND we'd need to
    # rebuild the CI endpoints from link scale, recompute via the
    # response-transform helper. Otherwise keep the existing
    # (potentially asymmetric) CIs from `obj.frame` and only adjust
    # them if the user wants a different `level=` on link scale OR
    # one-sided / TOST tests.
    scale_change = requested_type != obj_type
    level_change = level is not None and not np.isclose(
        lvl, getattr(obj, "level", lvl)
    )
    if is_posterior and level_change:
        raise ValueError(
            "summary(posterior_emm, level=...) cannot recompute "
            "percentile credible intervals at a new level from the "
            "EMMResult alone — the underlying draws live on the "
            "PosteriorInfo, not on the result. Re-run "
            "`posterior_emmeans(pinfo, ..., level=<new>)`."
        )

    # Resolve adjust early (also done below for the test branch but
    # the response-scale recompute below needs it).
    if adjust is not None:
        _adj_pre = adjust
    else:
        _adj_pre = getattr(obj, "adjust", None) or _opt("adjust", "none")

    # preserve the original ``bias_adjust`` flag so the
    # response-scale recompute can re-apply the Taylor correction.
    # The caller may override via the ``bias_adjust=`` and ``sigma=``
    # kwargs, mirroring R
    # ``summary(em, type="response", bias.adjust=TRUE, sigma=...)``.
    _ba_pre = bool(
        getattr(obj, "bias_adjust", False) if bias_adjust is None
        else bias_adjust
    )
    # if the caller did not pass ``sigma=``,
    # fall back to the stored ``bias_sigma`` from the source object
    # (set by ``regrid_response(..., sigma=...)``). Without this, a
    # later ``summary(em, level=...)`` silently reverted to
    # ``info.scale`` and produced response means consistent with the
    # default sigma — even though the EMM was built with an
    # explicit override. Mirrors R, which carries ``sigma`` on the
    # emmGrid misc-attributes.
    _sigma_eff = sigma if sigma is not None else getattr(obj, "bias_sigma", None)
    # R's ``summary(pairs(em), type="response",
    # bias.adjust=TRUE)`` on a non-log LHS transform (e.g.
    # ``sqrt(y)``) DOES NOT error — it returns the link-scale
    # contrast frame with a note that "contrasts are still on the
    # sqrt scale". pymmeans previously raised because
    # ``regrid_response(contrast, bias_adjust=True)`` is only defined
    # for log-family transforms. Catch the refusal here and fall back
    # to a link-scale frame with a warning, matching R semantics.
    def _safe_recompute(_obj, **_kw):
        # catch the structured
        # ``NonLogContrastBiasAdjustError`` sentinel instead of the
        # pre-string-match on ``"log-family" in str(exc)``.
        # The subclass IS a ValueError, so generic callers still see
        # the same error; this catch is specifically targeted.
        from pymmeans.transforms import NonLogContrastBiasAdjustError
        try:
            return _recompute_response_from_link(_obj, **_kw), True
        except NonLogContrastBiasAdjustError:
            if _is_contrast_result(_obj):
                import warnings as _w
                _w.warn(
                    "summary(contrast, type='response', "
                    "bias_adjust=True) on a non-log transform is not "
                    "defined; falling back to the link-scale contrast "
                    "frame. (R `emmeans::summary` emits an equivalent "
                    "'contrasts are still on the <link> scale' note.)",
                    UserWarning,
                    stacklevel=3,
                )
                return _obj, False
            raise
    if scale_change and obj_type == "link" and requested_type == "response":
        # Take the link-scale obj, recompute response at the new level.
        obj, _did_regrid = _safe_recompute(
            obj, level=lvl, adjust=_adj_pre, side=side,
            bias_adjust=_ba_pre, sigma=_sigma_eff,
        )
        obj_type = "response" if _did_regrid else obj_type
    elif scale_change and obj_type == "response" and requested_type == "link":
        _recompute_link_from_response(obj) # raises
    elif level_change and obj_type == "response":
        # rebuild a link-scale view from `linfct @ beta` /
        # `sqrt(diag(L V L.T))`, recompute response at the requested
        # level. thread adjust/side so the new CIs are
        # multiplicity-adjusted. also preserve bias_adjust.
        link_view = _response_to_link_result(obj)
        obj, _did_regrid = _safe_recompute(
            link_view, level=lvl, adjust=_adj_pre, side=side,
            bias_adjust=_ba_pre, sigma=_sigma_eff,
        )
        obj_type = "response" if _did_regrid else obj_type

    # Reuse the input's value column and SE; we don't recompute the
    # point estimate or SE here. This is the R `summary` convention:
    # the EMM matrix is fixed, but CI / test endpoints can be
    # re-derived (on link scale, where Wald CIs are symmetric and well-
    # defined).
    value_col = _value_col(obj)
    point = obj.frame[value_col].to_numpy(dtype=float)
    se = obj.frame["SE"].to_numpy(dtype=float)
    df_arr = obj.frame["df"].to_numpy(dtype=float)

    # R always runs the hypothesis test on the LINK scale, even when
    # display is on the
    # response scale. The previous pymmeans implementation used
    # `point / se_response` as the test statistic — wrong by a factor
    # of `inverse_deriv(eta)` (e.g. ~exp(eta) for a log link).
    #
    # When `obj_type == 'response'`, derive link-scale point + SE from
    # the contrast / EMM matrix L: eta = L @ beta and
    # SE_eta = sqrt(diag(L V L.T)). We keep `point` and `se` as the
    # display-scale (response) values; `point_test` / `se_test` are
    # the link-scale companions used for the t-ratio / p-value.
    if obj_type == "response":
        try:
            linfct = obj.linfct
            beta = obj.model_info.beta
            vcov = obj.model_info.vcov
            point_test = linfct @ beta
            cov_test = linfct @ vcov @ linfct.T
            with np.errstate(invalid="ignore"):
                se_test = np.sqrt(np.clip(np.diag(cov_test), 0.0, None))
        except Exception:
            # Fall back gracefully if linfct / vcov aren't available
            # (hand-built result, etc.) — tests will use display scale.
            point_test, se_test = point, se
    else:
        point_test, se_test = point, se

    # resolve `adj` early so
    # the CI branch can also see it. R `summary.emmGrid` widens
    # EMM-row CIs by the family-wise critical value of the requested
    # adjustment.
    if adjust is not None:
        adj = adjust
    else:
        adj = getattr(obj, "adjust", None) or _opt("adjust", "none")

    out = obj.frame.copy()
    # CI endpoints
    if show_ci:
        # for response-scale results, the
        # existing lower_cl / upper_cl in obj.frame are already
        # asymmetric (from the inverse-link transform of link-scale
        # endpoints). DO NOT overwrite them with symmetric Wald.
        # Symmetric Wald is correct on link scale only.
        if is_posterior:
            # posterior credible intervals live on
            # `obj.frame` as PERCENTILES of the draws. Wald
            # recomputation would silently overwrite the
            # asymmetric / skew-aware quantile bounds with symmetric
            # t-intervals (negative lower bounds on lognormal-style
            # posteriors). Preserve in place.
            pass
        elif is_bootstrap:
            # bootstrap CIs are PERCENTILE intervals
            # — same preservation rule as posterior. Any
            # recompute-forcing kwarg has already been refused above
            # (level / adjust / side / scale / bias_adjust / sigma).
            # Default ``summary(em_b)`` passes through the stored
            # percentile bounds unchanged.
            pass
        elif obj_type == "response":
            # if the response-
            # scale frame doesn't have `lower_cl` / `upper_cl`
            # columns (which is the case for `regrid_response(pairs(em))`
            # — contrast frames don't carry CIs by default), rebuild
            # them by routing through the link-scale CI computation
            # plus inverse transform. silently returned a
            # frame with NO CI columns; users running
            # `confint(rct)` got back nothing.
            #
            # We also rebuild when an explicit multiplicity adjustment
            # OR a one-sided ``side``
            # is requested, otherwise the response-scale CI silently
            # ignores the adjustment / side flag. R `summary.emmGrid`
            # widens the response CIs by the family-wise critical
            # value (sidak / bonferroni / tukey / scheffe). Before the
            # fix, ``summary(rb, adjust="sidak")`` returned the same
            # unadjusted endpoints regardless of the kwarg.
            missing_ci = (
                "lower_cl" not in out.columns
                or "upper_cl" not in out.columns
            )
            needs_adjusted_ci = bool(adj and adj.lower() != "none")
            # also force recompute when caller passed
            # ``bias_adjust=`` or ``sigma=`` explicitly.
            user_bias_override = (
                bias_adjust is not None or sigma is not None
            )
            if (
                missing_ci or needs_adjusted_ci or side != "two-sided"
                or user_bias_override
            ):
                link_view = _response_to_link_result(obj)
                obj = _recompute_response_from_link(
                    link_view, level=lvl, adjust=adj, side=side,
                    bias_adjust=_ba_pre, sigma=_sigma_eff,
                )
                out = obj.frame.copy()
            # Otherwise keep existing asymmetric CIs (computed at
            # obj.level by `regrid_response` or `emmeans(type='response')`).
            # level_change for response was rerouted above so any
            # surviving state is correct.
        else:
            # + EMM and contrast CIs both
            # use the family-wise critical value when an adjustment is
            # requested. R `confint(pairs(em))` widens contrast CIs by
            # the Tukey-q under the underlying EMM family size
            # mirror that via `_contrast_ci_critical_value`. The
            # release wired this for EMMs only — contrasts
            # silently used the unadjusted t-critical and returned CIs
            # ~25% too narrow vs R.
            if (
                _is_emm_result(obj)
                and adj
                and adj.lower() != "none"
                and len(point) > 1
            ):
                # only count
                # ESTIMABLE rows in the family-wise multiplicity
                # calculation. R `summary.emmGrid` drops non-estimable
                # rows from the `n` reported in the conf-level
                # adjustment note; pymmeans was inflating the family
                # by including the NaN cells too, producing CIs ~5%
                # too narrow vs R on rank-deficient designs.
                valid = np.isfinite(point) & np.isfinite(se)
                n_valid = int(valid.sum())
                if n_valid <= 1:
                    crit = stats.t.ppf(
                        1.0 - (1.0 - lvl) / 2.0 if side == "two-sided" else lvl,
                        df_arr,
                    )
                else:
                    crit = np.full(len(point), np.nan)
                    crit_valid = _emm_ci_critical_value(
                        adj, lvl, df_arr[valid], n_valid, side
                    )
                    crit[valid] = crit_valid
            elif (
                _is_contrast_result(obj)
                and adj
                and adj.lower() != "none"
                and len(point) > 1
            ):
                crit = _contrast_ci_critical_value(obj, adj, lvl, df_arr, side)
            elif side == "two-sided":
                alpha = 1.0 - lvl
                crit = stats.t.ppf(1.0 - alpha / 2.0, df_arr)
            else:
                crit = stats.t.ppf(lvl, df_arr)

            if side == "two-sided":
                out["lower_cl"] = point - crit * se
                out["upper_cl"] = point + crit * se
            elif side == "<":
                out["lower_cl"] = -np.inf
                out["upper_cl"] = point + crit * se
            else: # ">"
                out["lower_cl"] = point - crit * se
                out["upper_cl"] = np.inf
    else:
        out = out.drop(columns=["lower_cl", "upper_cl"], errors="ignore")

    # Hypothesis test columns
    if show_tests:
        # tests run on link scale when display is
        # response (R convention). `point_test` / `se_test` were
        # derived from `linfct` above and equal `point` / `se` on
        # link-scale objects.
        # `delta > 0` is NOT always TOST. R interprets
        # side='two-sided' (or 'equivalence') + delta>0 -> TOST
        # side='>' (noninferiority) + delta>0 -> one-sided upper
        # side='<' (nonsuperiority) + delta>0 -> one-sided lower
        with np.errstate(divide="ignore", invalid="ignore"):
            if delta > 0 and side == "two-sided":
                # TOST equivalence (Schuirmann 1987): reject the
                # composite null `mu <= null - delta OR mu >= null +
                # delta` (i.e. "not equivalent") via TWO one-sided
                # tests, declaring equivalence iff both reject.
                #
                # H0_lower: mu ≤ null - delta vs H1: mu > null - delta
                # t_lo = (point_test - (null - delta)) / SE_test;
                # p_lo = P(T ≥ t_lo)
                #
                # H0_upper: mu ≥ null + delta vs H1: mu < null + delta
                # t_hi = (point_test - (null + delta)) / SE_test;
                # p_hi = P(T ≤ t_hi)
                #
                # TOST p-value = max(p_lo, p_hi); reject when
                # max < alpha. Small p ⇒ strong equivalence evidence.
                t_lo = (point_test - (null - delta)) / se_test
                t_hi = (point_test - (null + delta)) / se_test
                p_lo = stats.t.sf(t_lo, df_arr)
                p_hi = stats.t.cdf(t_hi, df_arr)
                p = np.maximum(p_lo, p_hi)
                # R prints the
                # SIGNED DISTANCE TO THE EQUIVALENCE BOUNDARY divided
                # by SE:
                # t_ratio = (|point - null| - delta) / SE
                # Negative inside the equivalence region (the
                # observation is closer to the null than `delta`,
                # which is the GOAL for TOST — `point` is within the
                # equivalence margin so we want to reject the
                # composite null of non-equivalence). Positive when
                # the point is OUTSIDE the equivalence band.
                # used `sign(point - null) * min(|t_lo|,
                # |t_hi|)` which gives the wrong sign — it reports
                # "direction of deviation from null", not "distance
                # to boundary".
                with np.errstate(invalid="ignore"):
                    t_ratio = (np.abs(point_test - null) - delta) / se_test
            elif delta > 0 and side == ">":
                # Noninferiority: H0: mu - null <= -delta vs
                # H1: mu - null > -delta. t = (point - (null - delta))/SE.
                t_ratio = (point_test - (null - delta)) / se_test
                p = stats.t.sf(t_ratio, df_arr)
            elif delta > 0 and side == "<":
                # Nonsuperiority: H0: mu - null >= delta vs
                # H1: mu - null < delta. t = (point - (null + delta))/SE.
                t_ratio = (point_test - (null + delta)) / se_test
                p = stats.t.cdf(t_ratio, df_arr)
            else:
                t_ratio = np.where(
                    se_test > 0, (point_test - null) / se_test, np.nan
                )
                if side == "two-sided":
                    p = 2.0 * stats.t.sf(np.abs(t_ratio), df_arr)
                elif side == "<":
                    p = stats.t.cdf(t_ratio, df_arr)
                else: # ">"
                    p = stats.t.sf(t_ratio, df_arr)
        out["t_ratio"] = t_ratio
        # `adj` resolved above (hoisted it out of the
        # test branch so the CI branch can use it too).
        from pymmeans.adjustments import adjust_pvalues

        # `delta > 0` (TOST equivalence) still honours the multiplicity
        # adjustment for sidak-style methods; tukey-on-equivalence emits
        # a warning and falls back to sidak (matches R `emmeans`).
        # the warning + sidak fallback must include `dunnettx`
        # (default for `trt.vs.ctrl`, so the silently-wrong case fires
        # on the default contrast path) and `dunnett` / `mvt`. R
        # actually implements one-sided MVT for `dunnett` / `mvt` +
        # equivalence; v0.1 falls back to sidak with a warning and
        # documents the partial (candidate). Sidak is the
        # safer, more conservative answer in the meantime.
        adj_lower_chk = adj.lower() if isinstance(adj, str) else ""
        is_one_sided_or_tost = (delta > 0) or (side in ("<", ">"))

        # a: R `emmeans`'s `.adjust.fun` mvt branch uses
        # ONE-SIDED multivariate-t integration for `mvt` / `dunnett` +
        # delta>0 / one-sided. Rounds 31-34 fell back to Sidak with a
        # warning because the exact algorithm wasn't pinned.
        # pinned R's behaviour:
        #
        # equivalence (TOST): per-family tail integrals at the
        # observed `t_lo` and `t_hi`, take the max as the
        # intersection-union FWER-adjusted p-value.
        # noninferiority (`side='>'`): family-wise tail integral
        # at `t_lo` only (upper tail).
        # nonsuperiority (`side='<'`): family-wise tail integral
        # at `t_hi` only (lower tail).
        #
        # `tukey` and `dunnettx` still demote to Sidak — R emits the
        # same "method was changed to sidak" Note for those (the
        # studentized-range CDF and R's `.pdunnx` approximation both
        # assume a two-sided statistic).
        _ONE_SIDED_SIDAK_FALLBACK = ("tukey", "dunnettx")
        if (
            is_one_sided_or_tost
            and adj_lower_chk in _ONE_SIDED_SIDAK_FALLBACK
            and len(p) > 1
            and _is_contrast_result(obj)
        ):
            import warnings as _warn
            _warn.warn(
                f"adjust={adj_lower_chk!r} is not appropriate for "
                "one-sided / equivalence tests (studentized-range / "
                "dunnettx mixture assume two-sided statistics); "
                "falling back to adjust='sidak' to match R `emmeans`'s "
                "documented Note.",
                UserWarning,
                stacklevel=2,
            )
            adj = "sidak"
            adj_lower_chk = "sidak"

        # when `by=None` (flatten) or `by="col"`
        # (regroup) is requested, the family-meta needs to be
        # rebuilt BEFORE the one-sided MVT branch runs — otherwise
        # the MVT integral is computed per ORIGINAL by-group and the
        # by-override is silently dropped. Detect and apply early.
        _by_state = {"flatten": False, "regroup_cols": None}
        if (
            by is not _BY_UNSET
            and _is_contrast_result(obj)
            and len(p) > 1
        ):
            if by is None or (hasattr(by, "__len__") and len(by) == 0):
                _by_state["flatten"] = True
            else:
                _req = [by] if isinstance(by, str) else list(by)
                _missing = [c for c in _req if c not in obj.frame.columns]
                if _missing:
                    raise ValueError(
                        "'by' variables are not all in the grid: "
                        f"{_missing} not in {list(obj.frame.columns)}."
                    )
                _by_state["regroup_cols"] = _req

        def _apply_by_override_inplace():
            """Mutate (and return) the local p, t_ratio, point_test,
            se_test, df_arr, out frame, and obj._adjust_meta state to
            reflect the requested by-override. Returns (p, t_ratio,
            point_test, se_test, df_arr, out, new_meta, new_obj) so
            the caller can rebind these locals. Also issues the
            tukey→sidak warning when relevant."""
            nonlocal adj, adj_lower_chk
            adj_lower_now = (adj or "none").lower()
            if _by_state["flatten"]:
                if adj_lower_now == "tukey":
                    import warnings as _warn
                    _warn.warn(
                        'adjust = "tukey" was changed to "sidak" '
                        "because \"tukey\" is only appropriate for one "
                        "set of pairwise comparisons ("
                        "#4: R Note matched).",
                        UserWarning,
                        stacklevel=3,
                    )
                    adj = "sidak"
                    adj_lower_chk = "sidak"
                    adj_lower_now = "sidak"
                new_meta = {
                    "families": [
                        {
                            "start": 0,
                            "stop": len(p),
                            "n_rows": len(p),
                            "n_rows_valid": int(np.isfinite(p).sum()),
                            "n_means": len(p),
                            "n_means_valid": len(p),
                            "df": (
                                float(np.nanmin(obj.frame["df"]))
                                if "df" in obj.frame
                                else np.inf
                            ),
                            "correlation": None,
                            "by_key": (),
                        }
                    ]
                }
                return p, t_ratio, point_test, se_test, df_arr, out, new_meta, obj
            # Regroup
            cols = _by_state["regroup_cols"]
            grp = obj.frame[cols].astype(object)
            key_tuples = list(map(tuple, grp.to_numpy()))
            unique_keys = sorted(set(key_tuples))
            if adj_lower_now == "tukey":
                import warnings as _warn
                _warn.warn(
                    'adjust = "tukey" was changed to "sidak" because '
                    '"tukey" is only appropriate for one set of '
                    "pairwise comparisons (R `emmeans` documents the same fallback).",
                    UserWarning,
                    stacklevel=3,
                )
                adj = "sidak"
                adj_lower_chk = "sidak"
            row_order = []
            fam_list = []
            cursor = 0
            for key in unique_keys:
                idx = [i for i, k in enumerate(key_tuples) if k == key]
                if not idx:
                    continue
                row_order.extend(idx)
                fam_n = len(idx)
                fam_list.append({
                    "start": cursor,
                    "stop": cursor + fam_n,
                    "n_rows": fam_n,
                    "n_rows_valid": int(np.isfinite(p[idx]).sum()),
                    "n_means": fam_n,
                    "n_means_valid": fam_n,
                    "df": (
                        float(np.nanmin(obj.frame["df"].iloc[idx]))
                        if "df" in obj.frame
                        else np.inf
                    ),
                    "correlation": None,
                    "by_key": key,
                })
                cursor += fam_n
            order_arr = np.asarray(row_order, dtype=int)
            new_p = p[order_arr]
            new_t = t_ratio[order_arr]
            new_pt = point_test[order_arr]
            new_st = se_test[order_arr]
            new_df = df_arr[order_arr]
            new_out = out.iloc[order_arr].reset_index(drop=True)
            # Also reorder the obj.linfct so MVT correlation
            # rebuilds use the regrouped row order.
            from dataclasses import replace as _dc_replace
            new_linfct = obj.linfct[order_arr]
            new_obj = _dc_replace(obj, linfct=new_linfct)
            return new_p, new_t, new_pt, new_st, new_df, new_out, {"families": fam_list}, new_obj

        # Pre-compute the override before either adjustment branch.
        _override_meta = None
        if _by_state["flatten"] or _by_state["regroup_cols"]:
            (p, t_ratio, point_test, se_test, df_arr, out,
             _override_meta, obj) = _apply_by_override_inplace()

        # a: one-sided MVT branch for `mvt` / `dunnett` +
        # `delta>0` / one-sided. Operates per family using
        # `_adjust_meta` correlations (built fresh from `obj.linfct`
        # if the original contrast wasn't built with a MVT-family
        # default). Sets `p_adj` directly and short-circuits the
        # generic dispatch below.
        mvt_one_sided_handled = False
        if (
            is_one_sided_or_tost
            and adj_lower_chk in ("mvt", "dunnett")
            and len(p) > 1
            and _is_contrast_result(obj)
        ):
            from pymmeans.adjustments import adjust_mvt_tail
            # if a by-override was applied above,
            # use it instead of obj._adjust_meta so the MVT integral
            # is computed over the *requested* family structure.
            meta = _override_meta or getattr(obj, "_adjust_meta", None)
            families = (meta or {}).get("families") if meta else None
            p_adj_full = np.clip(p.copy(), 0.0, 1.0)
            if families:
                for fam in families:
                    sl = slice(fam["start"], fam["stop"])
                    valid_fam = np.isfinite(p[sl]) & np.isfinite(t_ratio[sl])
                    n_valid = int(valid_fam.sum())
                    if n_valid == 0:
                        continue
                    corr = fam.get("correlation")
                    if corr is None:
                        # Build correlation from the contrast linfct
                        # (filtered to estimable rows).
                        try:
                            L_fam = obj.linfct[sl][valid_fam]
                            V = obj.model_info.vcov
                            cov_c = L_fam @ V @ L_fam.T
                            se_v = np.sqrt(np.clip(np.diag(cov_c), 0.0, None))
                            outer = np.outer(se_v, se_v)
                            with np.errstate(divide="ignore", invalid="ignore"):
                                corr = np.where(outer > 0, cov_c / outer, 0.0)
                            np.fill_diagonal(corr, 1.0)
                        except Exception:
                            corr = None
                    else:
                        # Filter the precomputed correlation to the
                        # estimable subset.
                        try:
                            corr = np.asarray(corr)
                            idx = np.where(valid_fam)[0]
                            corr = corr[np.ix_(idx, idx)]
                        except Exception:
                            corr = None
                    if corr is None or corr.shape[0] < 1:
                        continue
                    df_fam = float(fam["df"])
                    # Per-family one-sided t-stats. For TOST, t_lo is
                    # the upper-tail test (noninferiority), t_hi is
                    # the lower-tail test (nonsuperiority).
                    pt = point_test[sl][valid_fam]
                    se_v_t = se_test[sl][valid_fam]
                    t_lo_fam = (pt - (null - delta)) / se_v_t
                    t_hi_fam = (pt - (null + delta)) / se_v_t
                    if delta > 0 and side == "two-sided":
                        # TOST equivalence: max(p_lo_adj, p_hi_adj)
                        p_lo_adj = adjust_mvt_tail(
                            t_lo_fam, corr, df_fam, tail=+1
                        )
                        p_hi_adj = adjust_mvt_tail(
                            t_hi_fam, corr, df_fam, tail=-1
                        )
                        p_fam_adj = np.maximum(p_lo_adj, p_hi_adj)
                    elif side == ">":
                        # Noninferiority (with or without delta).
                        # Without delta: t_lo collapses to the
                        # standard upper-tail t = (point - null)/se.
                        p_fam_adj = adjust_mvt_tail(
                            t_lo_fam, corr, df_fam, tail=+1
                        )
                    else: # side == "<": nonsuperiority
                        p_fam_adj = adjust_mvt_tail(
                            t_hi_fam, corr, df_fam, tail=-1
                        )
                    # Scatter back into the full-length array.
                    out_slice = p_adj_full[sl]
                    idx_valid_local = np.where(valid_fam)[0]
                    out_slice[idx_valid_local] = p_fam_adj
                    p_adj_full[sl] = out_slice
                p_adj = p_adj_full
                mvt_one_sided_handled = True
            # If no `_adjust_meta` (rare; hand-built contrast), fall
            # through to the generic dispatch below.

        # EMM-row adjustment branch — treat all EMMs in
        # the frame as a single family of size `len(p)`. Use the
        # family-aware adjust_pvalues, falling back to bonferroni-
        # equivalent for correlation-required methods (we don't have
        # the same `_adjust_meta` infrastructure for EMMs).
        if mvt_one_sided_handled:
            pass
        elif (
            adj
            and adj != "none"
            and len(p) > 1
            and _is_emm_result(obj)
        ):
            # skip NaN rows so the family count `m`
            # excludes non-estimable EMMs. R's `summary.emmGrid` does
            # the same; counting NaN rows over-inflated the
            # adjustment and silently produced p_values ~2× too
            # conservative.
            valid_p = np.isfinite(p) & np.isfinite(t_ratio)
            n_valid_p = int(valid_p.sum())
            if n_valid_p <= 1:
                p_adj = np.clip(p, 0.0, 1.0)
            else:
                try:
                    p_adj_valid = adjust_pvalues(
                        p[valid_p],
                        adj,
                        n_means=n_valid_p,
                        df=float(np.nanmean(df_arr[valid_p])),
                        t_ratios=t_ratio[valid_p],
                    )
                except (ValueError, TypeError):
                    # Correlation-required methods would raise without
                    # correlation matrix; bonferroni-fallback.
                    p_adj_valid = np.clip(
                        p[valid_p] * n_valid_p, 0.0, 1.0
                    )
                p_adj = np.full(len(p), np.nan)
                p_adj[valid_p] = p_adj_valid
        elif adj and adj != "none" and len(p) > 1 and _is_contrast_result(obj):
            # use `_adjust_meta` (populated
            # per-family by `_contrast_one_family`) so we apply the
            # correction within each family with the correct n_means /
            # df / correlation. The previous implementation used
            # `len(model_info.param_names)` as n_means, which is the
            # outer parameter count (e.g. 5 for a full interaction
            # model), NOT the family size — Tukey-HSD with n_means=5
            # over k=3 contrasts gave the wrong studentized-range CDF
            # parameter and silently inflated adjusted p-values.
            # by-override (if any) was applied
            # above. Use the pre-computed override meta to keep both
            # the MVT one-sided branch and this contrast branch
            # consistent. Falls through to obj._adjust_meta when no
            # by-override was requested.
            meta = _override_meta or getattr(obj, "_adjust_meta", None)
            p_adj_full = np.clip(p.copy(), 0.0, 1.0)
            adj_lower = adj.lower()
            # these two flags were preserved as
            # informational-only locals after the review
            # moved the override application above. Ruff F841 flagged
            # them; the comment-only purpose can be served by the
            # surrounding prose without the dead bindings.
            # (by-flatten = _by_state["flatten"] and contrast-result;
            # by-regroup-cols = _by_state["regroup_cols"].)
            # by-override (flatten / regroup) is
            # already applied above via `_apply_by_override_inplace`
            # — no duplicate work needed here. ``adj`` may have been
            # auto-promoted to ``"sidak"`` (tukey→sidak under the
            # override); refresh `adj_lower` so the downstream
            # correlation-dispatch picks the right method.
            adj_lower = (adj or "none").lower()
            need_corr = adj_lower in ("dunnett", "mvt")
            if meta and meta.get("families"):
                for fam in meta["families"]:
                    sl = slice(fam["start"], fam["stop"])
                    p_fam = p[sl]
                    if len(p_fam) == 0:
                        continue
                    # filter to estimable rows so the
                    # family count `n` matches R's "method for N
                    # tests" note. Non-estimable rows pass through
                    # as NaN.
                    t_fam = t_ratio[sl]
                    valid_fam = np.isfinite(p_fam) & np.isfinite(t_fam)
                    n_valid_fam = int(valid_fam.sum())
                    if n_valid_fam == 0:
                        # All NaN — leave as NaN.
                        p_adj_full[sl] = p_fam
                        continue
                    corr = fam.get("correlation")
                    # If user requested dunnett/mvt but correlation wasn't
                    # precomputed at contrast-build time, rebuild it from
                    # the contrast linfct against model_info.vcov,
                    # filtered to estimable rows.
                    if need_corr and corr is None and n_valid_fam > 1:
                        try:
                            L_fam = obj.linfct[sl]
                            L_valid = L_fam[valid_fam]
                            V = obj.model_info.vcov
                            cov_c = L_valid @ V @ L_valid.T
                            se_v = np.sqrt(np.clip(np.diag(cov_c), 0.0, None))
                            outer = np.outer(se_v, se_v)
                            with np.errstate(divide="ignore", invalid="ignore"):
                                corr = np.where(outer > 0, cov_c / outer, 0.0)
                            np.fill_diagonal(corr, 1.0)
                        except Exception:
                            corr = None
                    # Prefer the -added valid counts; fall
                    # through to raw counts on older pickled metadata.
                    n_means_use = int(
                        fam.get("n_means_valid", fam["n_means"]) or fam["n_means"]
                    )
                    try:
                        p_adj_valid = adjust_pvalues(
                            p_fam[valid_fam],
                            adj,
                            n_means=n_means_use,
                            df=fam["df"],
                            t_ratios=t_fam[valid_fam],
                            correlation=corr,
                        )
                        p_adj_full[sl] = np.where(
                            valid_fam, np.nan, np.clip(p_fam, 0.0, 1.0)
                        )
                        idx_valid = np.where(valid_fam)[0]
                        # p_adj_full[sl] is a view of p_adj_full;
                        # need to assign positionally.
                        out_slice = p_adj_full[sl]
                        out_slice[idx_valid] = p_adj_valid
                        p_adj_full[sl] = out_slice
                    except (ValueError, TypeError):
                        # Fall back to raw within this family
                        p_adj_full[sl] = np.clip(p_fam, 0.0, 1.0)
                p_adj = p_adj_full
            else:
                # No family metadata — treat as a single family. We
                # don't have correlation, so dunnett-style methods will
                # raise; tukey / bonferroni / fdr still work.
                n_means = len(p) + 1 # heuristic: k means produce k(k-1)/2 pairs
                try:
                    p_adj = adjust_pvalues(
                        p, adj,
                        n_means=n_means,
                        df=float(np.nanmean(df_arr)),
                        t_ratios=t_ratio,
                    )
                except (ValueError, TypeError):
                    p_adj = np.clip(p, 0.0, 1.0)
        else:
            p_adj = np.clip(p, 0.0, 1.0)

        # ``cross_adjust=`` second-stage adjustment. Mirrors R
        # ``summary(em, adjust=..., cross.adjust=...)``: after the
        # per-family ``adjust`` correction (above), pool every row's
        # adjusted p-value into a single family and apply
        # ``cross_adjust`` across that pool. The closure principle
        # (Marcus, Peritz & Gabriel 1976) makes the composition a
        # valid familywise-error-rate procedure.
        #
        # Family-internal methods (tukey / dunnett / scheffe / mvt)
        # depend on a correlation structure across THE specific
        # contrasts inside one family; they don't generalise to a
        # pool of already-adjusted p-values across heterogeneous
        # families. Refuse those for cross_adjust.
        # Only APPLY cross_adjust when pooling >= 2 families, but
        # always VALIDATE the method name when cross_adjust is given.
        # R treats ``cross.adjust`` on a single family as a clean
        # no-op — the family-internal ``adjust`` has already
        # controlled the familywise error, and a second pass would
        # double-penalise. Gating on the contrast count rather than
        # the family count would wrongly double-adjust a single
        # family of pairwise contrasts. ``cross_adjust`` is only
        # meaningful when pooling >= 2 families (the closure principle
        # composes per-family corrections into a cross-family one).
        # We still reject family-internal / correlation-aware methods
        # up front regardless of family count so a typo or misuse
        # surfaces immediately instead of silently no-opping.
        if cross_adjust is not None:
            cross_lower = cross_adjust.lower()
            if cross_lower in (
                "tukey", "dunnett", "scheffe", "mvt", "dunnettx",
            ):
                raise ValueError(
                    f"cross_adjust={cross_adjust!r} is a family-"
                    "internal correlation-aware adjustment and can't "
                    "be used as the cross-family correction. Pick a "
                    "step-down / step-up method: 'bonferroni' / "
                    "'sidak' / 'holm' / 'bh' / 'by' / 'hochberg' / "
                    "'hommel'."
                )
            # Read family meta directly from the object — the local
            # ``meta`` is only bound on the family-adjust path, but
            # cross_adjust can run after ``adjust='none'`` too.
            _cross_meta = getattr(obj, "_adjust_meta", None)
            n_families = (
                len(_cross_meta["families"])
                if (_cross_meta and _cross_meta.get("families"))
                else 1
            )
            if cross_lower in ("none", "no"):
                # explicit no-op (R parity)
                pass
            elif n_families <= 1:
                # Single family: ``cross_adjust`` is a clean no-op.
                # The family-internal ``adjust`` already controls the
                # familywise error for this single family.
                pass
            else:
                # auditor V13-A1 P0: R ``emmeans::summary(cross.adjust=)``
                # arranges per-family adjusted p-values into a matrix
                # with ``nrow = contrasts_per_family`` and
                # ``ncol = n_families``, then applies the cross-adjust
                # method ROW-WISE across the family columns
                # (R source: ``R/summary.R`` lines ~705-720 — the
                # ``mat = matrix(p, nrow = len)`` + ``apply(mat, 1,
                # p.adjust)`` block). The Bonferroni multiplier is
                # therefore ``n_families``, NOT the total pool size.
                # The earlier pymmeans implementation flattened the
                # matrix and multiplied by the pool size, silently
                # diverging from R by a factor of
                # ``contrasts_per_family``. Per R's documented
                # condition ("they are all the same size"), we only
                # apply cross-adjust when every family has the same
                # row count; otherwise it is a silent no-op (R parity).
                _fam_meta = _cross_meta["families"]
                _fam_sizes = [int(f["n_rows"]) for f in _fam_meta]
                _all_same_size = (
                    len(set(_fam_sizes)) == 1 and _fam_sizes[0] > 0
                )
                if not _all_same_size:
                    # Mismatched family sizes — R silently no-ops
                    # cross-adjust in this regime.
                    pass
                else:
                    _contrasts_per_family = _fam_sizes[0]
                    # Reshape into (contrasts_per_family, n_families) by
                    # stacking the per-family slices as columns. The
                    # families are stored in row-order so column k
                    # corresponds to family k.
                    try:
                        _cols = [
                            p_adj[int(f["start"]):int(f["stop"])]
                            for f in _fam_meta
                        ]
                        _mat = np.column_stack(_cols)
                    except Exception as exc:
                        raise ValueError(
                            f"cross_adjust={cross_adjust!r}: failed to "
                            "reshape p-values into the per-family "
                            f"matrix ({type(exc).__name__}: {exc}). "
                            "This is an internal error — please report."
                        ) from exc

                    # Apply the cross-adjust method ROW-WISE across the
                    # n_families columns. R's row-wise ``p.adjust`` with
                    # ``method="bonferroni"`` multiplies each row's
                    # p-values by ``ncol(mat) = n_families``.
                    _adjusted_rows = np.empty_like(_mat)
                    for _ri in range(_contrasts_per_family):
                        _row = _mat[_ri, :]
                        _row_valid = np.isfinite(_row)
                        _n_row_valid = int(_row_valid.sum())
                        if _n_row_valid > 1:
                            try:
                                _adj_row = adjust_pvalues(
                                    _row[_row_valid],
                                    cross_adjust,
                                    n_means=_n_row_valid,
                                    df=float(np.inf),
                                    t_ratios=np.zeros(_n_row_valid),
                                )
                                _out_row = np.full(len(_row), np.nan)
                                _out_row[_row_valid] = _adj_row
                                _adjusted_rows[_ri, :] = _out_row
                            except (ValueError, TypeError) as exc:
                                raise ValueError(
                                    f"cross_adjust={cross_adjust!r} "
                                    f"failed to apply: "
                                    f"{type(exc).__name__}: {exc}. "
                                    "Try 'bonferroni' as a conservative "
                                    "fallback."
                                ) from exc
                        else:
                            _adjusted_rows[_ri, :] = _row

                    # Unstack the adjusted matrix back into ``p_adj``
                    # by writing each column to its family's slice.
                    _out_cross = p_adj.copy()
                    for _ci, _f in enumerate(_fam_meta):
                        _out_cross[
                            int(_f["start"]):int(_f["stop"])
                        ] = _adjusted_rows[:, _ci]
                    p_adj = _out_cross
        out["p_value"] = p_adj
    else:
        out = out.drop(columns=["t_ratio", "p_value"], errors="ignore")

    # `df` column always stays
    out = out.reset_index(drop=True)
    return out


def confint(
    obj: Any,
    level: float | None = None,
    side: str = "two-sided",
    adjust: str | None = None,
    type: str | None = None,
    bias_adjust: bool | None = None,
    sigma: float | None = None,
) -> pd.DataFrame:
    """R `confint.emmGrid` analogue — show CIs only.

    Equivalent to ``summary(obj, infer=(True, False), level=, side=,
    adjust=, type=, bias_adjust=, sigma=)``.

    ``level`` defaults to ``None`` so it falls through
    the same precedence chain as :func:`summary` (explicit kwarg >
    ``emm_options('level')`` > ``obj.level`` > 0.95). Previously
    hard-coded to 0.95, which made ``emm_options(level=...)``
    silently ineffective on ``confint``.

    R's ``confint.emmGrid`` has a hard ``level = 0.95`` default that
    overrides any baked-in level on the EMM object — so
    ``confint(update(em, level=.99))`` returns a **95%** CI in R.

    pymmeans matches R *strictly* — ``level=None`` resolves to
    **0.95**, independent of ``emm_options(level=...)``. R does not
    consult its option here (the ``confint.emmGrid`` formal is a bare
    ``level = 0.95``). To globally raise the CI level via pymmeans,
    pass ``level=`` explicitly or call
    ``summary(obj, infer=(True, False), level=...)``, which *does*
    honour ``emm_options(level=...)``.

    The ``bias_adjust=`` and ``sigma=`` kwargs are forwarded to
    :func:`summary` and ultimately :func:`regrid_response`.
    """
    # hard default 0.95 — no `emm_options(level=)`
    # fall-through. Strict R parity for `confint.emmGrid`.
    #
    # exception — bootstrap-derived results store
    # percentile CIs at ``obj.level`` (whatever the user passed to
    # ``bootstrap_ci(level=...)``). Forcing ``level=0.95`` here turns
    # a no-argument display call (``confint(em_b)``) into a recompute
    # request for any bootstrap object NOT built at 0.95, which the
    # summary guard then refuses. That's user-hostile —
    # ``confint`` is the natural CI-only accessor; it should "just
    # work" on a result whose CIs are already stored. Use the
    # bootstrap's own level when none is explicitly supplied.
    if level is None:
        if getattr(obj, "df_method", "default") == "bootstrap":
            level = getattr(obj, "level", 0.95)
        else:
            level = 0.95
    return summary(
        obj, infer=(True, False), level=level, side=side,
        adjust=adjust, type=type,
        bias_adjust=bias_adjust, sigma=sigma,
    )


def test(
    obj: Any,
    null: float = 0.0,
    side: str = "two-sided",
    delta: float = 0.0,
    adjust: str | None = None,
    type: str | None = None,
) -> pd.DataFrame:
    """R `test.emmGrid` analogue — show hypothesis tests only.

    Equivalent to ``summary(obj, infer=(False, True), null=, side=,
    delta=, adjust=, type=)``.
    """
    # refuse bootstrap-derived results — sibling of
    # the CI-recompute refusal. previously ``test(em_b)``
    # silently emitted analytic Wald t-tests derived from
    # ``linfct @ beta`` / ``L V L.T``, but the object's stored
    # inference is percentile bootstrap — mixing two inference
    # paradigms. The summary-layer bootstrap guard only blocks
    # *CI*-recompute kwargs (``level/adjust/side/type/...``)
    # "show tests" path didn't trip it because the kwargs were
    # left at defaults. Users wanting bootstrap p-values should run
    # ``permutation_test(em_b)`` or
    # ``bootstrap_ci(pairs(raw_em), n_samples=...)`` and read the
    # bootstrap percentile CIs directly.
    if getattr(obj, "df_method", "default") == "bootstrap":
        raise ValueError(
            "test() is not defined for a bootstrap-derived result "
            "(df_method='bootstrap'). The stored inference is "
            "percentile bootstrap; the analytic t-test would "
            "silently mix two inference paradigms. For bootstrap "
            "uncertainty quantification, either inspect the stored "
            "percentile CIs (``summary(em_b)``) or call "
            "``permutation_test(pairs(raw_em), ...)`` for a true "
            "non-parametric p-value."
        )
    return summary(
        obj, infer=(False, True), null=null, side=side, delta=delta,
        adjust=adjust, type=type,
    )


# R-name column mapping
_R_COLUMN_MAP = {
    "lower_cl": "lower.CL",
    "upper_cl": "upper.CL",
    "t_ratio": "t.ratio",
    "p_value": "p.value",
    "z_ratio": "z.ratio",
}


def _glm_response_value_name(model_info: Any) -> str | None:
    """polish map a GLM family to
    R's response-scale value column name.

    R uses ``prob`` for Binomial, ``rate`` for Poisson, and
    ``response`` for everything else (Gamma, NegBin, ...). pymmeans
    keeps ``emmean`` internally; we apply the rename only when the
    user explicitly asks for an R-style frame via ``as_r_frame``.

    Returns the target column name, or ``None`` if the input isn't a
    GLM (in which case `emmean` stays as-is). For an LHS-transform
    OLS fit (e.g. ``np.log(y) ~ ...``) on the response scale, R uses
    ``response``; we mirror that.
    """
    family = getattr(model_info, "family", None)
    if family is None:
        # OLS / GLS / MixedLM with an LHS transform: response display
        # column in R is `response`.
        if getattr(model_info, "response_name", None):
            return "response"
        return None
    fname = type(family).__name__.lower()
    if "binomial" in fname:
        return "prob"
    if "poisson" in fname:
        return "rate"
    return "response"


def as_r_frame(
    frame_or_result: Any, include_set: bool = False
) -> pd.DataFrame:
    """Return a DataFrame with R-style dot-separated column names.

    pymmeans uses Pythonic underscore
    column names (``lower_cl``, ``t_ratio``, ``p_value``); R uses
    dot-separated (``lower.CL``, ``t.ratio``, ``p.value``). For users
    porting R `emmeans` code that grep-matches columns, this helper
    renames in place.

    polish: for response-scale GLM frames, R additionally
    renames the value column to a family-specific name (``prob`` for
    Binomial, ``rate`` for Poisson, ``response`` for Gamma /
    Inverse-Gaussian / LHS-transformed OLS). The CI columns become
    ``asymp.LCL`` / ``asymp.UCL`` when ``df=inf`` (Wald asymptotic
    intervals). Both renames only fire when ``as_r_frame`` is given
    a result object (so the family can be inspected); raw-DataFrame
    inputs are renamed with the generic map only.

    Accepts either an ``EMMResult`` / ``ContrastResult`` (uses
    ``.frame``) or a raw ``pandas.DataFrame``.

    ``EmmList`` is also accepted. Mirrors R
    ``as.data.frame.emm_list``: combines each member's summary into
    one DataFrame, padding non-shared columns with ``"."``, and emits
    R's UserWarning about combined results affecting adjusted
    P-values.

    The ``.set`` provenance column is opt-in via
    ``include_set=True`` (default ``False``). R's
    ``as.data.frame.emm_list`` does NOT emit a ``.set`` column —
    members are just stacked. Strict R-parity is the default; pass
    ``include_set=True`` if you want the pymmeans provenance column.
    """
    # EmmList branch. Combine members like
    # R `as.data.frame.emm_list`. Members are first summarised so
    # the frames carry CIs / tests; then concatenated with sparse
    # union of columns, padding missing cells with ".".
    from pymmeans.contrasts import EmmList
    if isinstance(frame_or_result, EmmList):
        import warnings as _w
        _w.warn(
            "Note: 'as_r_frame' has combined your "
            f"{len(frame_or_result)} sets of results into one object, "
            "and this affects things like adjusted P values. Refer "
            "to the annotations.",
            UserWarning,
            stacklevel=2,
        )
        pieces: list[pd.DataFrame] = []
        for name, member in zip(
            frame_or_result.names, frame_or_result, strict=True
        ):
            # Recurse into nested EmmLists by passing include_set down.
            piece = as_r_frame(member, include_set=include_set).copy()
            if include_set:
                piece.insert(0, ".set", name)
            pieces.append(piece)
        out = pd.concat(pieces, ignore_index=True, sort=False)
        # R pads missing cells with "." (string); pandas default is
        # NaN. For non-numeric columns (factors, contrast labels),
        # convert to object dtype first so the "." sentinel doesn't
        # collide with categorical-allowed values. Numeric columns
        # keep NaN so downstream code can still arithmetic on them.
        for col in out.columns:
            if not out[col].isna().any():
                continue
            if pd.api.types.is_numeric_dtype(out[col]):
                continue
            out[col] = out[col].astype(object).where(out[col].notna(), ".")
        return out

    if isinstance(frame_or_result, pd.DataFrame):
        return frame_or_result.copy().rename(columns=_R_COLUMN_MAP)

    frame = frame_or_result.frame.copy()

    # drop analytic-Wald test columns
    # (``t_ratio`` / ``p_value`` / ``z_ratio``) on a bootstrap-
    # derived result. These columns were computed by ``pairs()`` /
    # ``contrast()`` BEFORE ``bootstrap_ci`` was called and remain in
    # the stored frame. Without this strip, ``as_r_frame(ct_b)``
    # emits ``t.ratio`` / ``p.value`` columns side-by-side with the
    # percentile CI columns — silently mixing two inference paradigms
    # (same composition class as , but through the
    # column-rename door instead of the ``summary`` recompute door).
    # The ``summary(ct_b, infer=(_, True))`` refusal
    # makes the user-facing test access path explicit; ``as_r_frame``
    # should be similarly faithful to the stored inference.
    if getattr(frame_or_result, "df_method", "default") == "bootstrap":
        _bootstrap_test_cols = {"t_ratio", "p_value", "z_ratio"}
        _drop_cols = [c for c in frame.columns if c in _bootstrap_test_cols]
        if _drop_cols:
            frame = frame.drop(columns=_drop_cols)

    rename = dict(_R_COLUMN_MAP)

    # Family-specific value-column rename for response-scale frames.
    if (
        getattr(frame_or_result, "type", "link") == "response"
        and "emmean" in frame.columns
    ):
        name = _glm_response_value_name(frame_or_result.model_info)
        if name is not None and name != "emmean":
            rename["emmean"] = name

    # Asymptotic CIs use `asymp.LCL` / `asymp.UCL` when df is infinite
    # on every row. R distinguishes finite-df t-intervals
    # (`lower.CL` / `upper.CL`) from asymptotic-Wald (`asymp.LCL` /
    # `asymp.UCL`). Match that distinction.
    if "df" in frame.columns:
        df_arr = frame["df"].to_numpy()
        if df_arr.size > 0 and np.all(~np.isfinite(df_arr)):
            rename["lower_cl"] = "asymp.LCL"
            rename["upper_cl"] = "asymp.UCL"
            # `t.ratio` becomes `z.ratio` under asymptotic inference.
            if "t_ratio" in frame.columns:
                rename["t_ratio"] = "z.ratio"

    return frame.rename(columns=rename)


def update(obj: Any, **kwargs: Any) -> Any:
    """R `update.emmGrid` analogue — return a copy with fields replaced.

 R `update()` lets users tweak
    EMM-object fields (level, adjust, type, bias_adjust, ...) without
    rebuilding from scratch. pymmeans's frozen-dataclass result
    objects make this a thin wrapper over ``dataclasses.replace`` that
    also re-derives the summary columns when level/adjust/type change.

    Examples
    --------
    >>> em2 = update(em, level=0.99) # doctest: +SKIP
    >>> ct2 = update(ct, adjust='bonferroni') # doctest: +SKIP
    """
    from dataclasses import replace as _dc_replace

    # refuse any field that controls *reconstruction*.
    # added ``method`` / ``method_args`` / ``at`` / ``weights``
    # to ``ContrastResult`` / ``EMMResult``; ``update()`` was wired to
    # accept *every* dataclass field via ``dataclasses.replace``, but
    # these are NOT passive display fields — they drive case-bootstrap
    # / permutation-test rebuilds. Stamping ``at={"x":[10]}`` on a
    # result whose ``frame`` / ``linfct`` were built at ``x=mean(x)``
    # produces a "split-brain" object whose point estimate disagrees
    # with whatever any rebuild-aware op does next. Force users to
    # re-call ``emmeans()`` / ``pairs()`` / ``contrast()`` instead.
    _REBUILD_ONLY = {
        # ``frame`` is the *displayed* table but it
        # is paired with ``linfct`` / ``model_info`` for the rebuild
        # path. Stamping ``frame=corrupted_frame`` via ``update`` was
        # accepted the input and produced a split-brain object whose
        # displayed ``emmean`` disagreed with what ``pairs`` /
        # ``contrast`` recomputed from ``linfct @ beta``.
        # refuses; users who need to mutate display columns should
        # work on ``obj.frame.copy()`` outside the dataclass.
        "frame",
        "at", "weights", "method", "method_args",
        "target", "by", "linfct", "model_info",
        "_adjust_meta", "_pair_indices",
        # ``bias_sigma`` is the user-supplied
        # ``sigma=`` override stored by ``regrid_response(...,
        # bias_adjust=True, sigma=...)``. Stamping it
        # via ``update`` without rebuilding the frame's ``emmean``
        # column produces a split-brain object whose displayed
        # values were computed with the OLD sigma while any later
        # ``summary(em, level=...)`` recomputes use the stamped NEW
        # sigma. Force the user to re-call ``regrid_response`` at the
        # desired sigma instead.
        "bias_sigma",
        # ``bias_adjust`` is NOT a passive label.
        # Stamping it via ``update()`` on a response-scale EMM
        # produces a split-brain object whose ``frame['emmean']``
        # still carries the un-bias-corrected inverse-transform
        # values while the dataclass field claims they were Taylor-
        # corrected. The correct workflow is
        # ``regrid_response(em, bias_adjust=True)``, which rebuilds
        # the frame via ``tran.bias_mean``.
        "bias_adjust",
        # ``inference_kind`` / ``df_method`` /
        # ``_satt_cache`` are inference-control fields. Stamping
        # ``inference_kind="posterior"`` on a Wald EMM doesn't
        # convert the CIs to percentile credible intervals;
        # stamping ``df_method="kenward_roger"`` doesn't inflate
        # ``vcov`` to V_KR. The frame columns continue to report
        # Wald inference while downstream consumers
        # (``apply_satterthwaite`` / ``EMMResult._scale`` posterior
        # refusal / contrast df propagation) read the stamped
        # field, silently mixing two inference paradigms.
        # Force users to call ``posterior_emmeans`` /
        # ``apply_satterthwaite`` / ``apply_kenward_roger``.
        "inference_kind", "df_method", "_satt_cache",
    }
    rebuild_attempted = _REBUILD_ONLY & set(kwargs)
    if rebuild_attempted:
        raise TypeError(
            "update() cannot mutate reconstruction-control metadata "
            f"{sorted(rebuild_attempted)} — these fields are used by "
            "case-bootstrap / permutation_test / pairs(simple=) to "
            "rebuild the result faithfully, and stamping them without "
            "rebuilding the frame produces a split-brain object whose "
            "point estimate disagrees with its claimed grid / method. "
            "Recompute via emmeans() / contrast() / pairs() instead "
            "(or regrid_response(..., bias_adjust=True) for "
            "bias_adjust; apply_satterthwaite / apply_kenward_roger "
            "for df_method; posterior_emmeans for inference_kind)."
        )

    # Fields that can be set via replace
    settable = set(obj.__dataclass_fields__.keys())
    direct_args = {k: v for k, v in kwargs.items() if k in settable}
    invalid = set(kwargs) - settable
    if invalid:
        raise TypeError(
            f"update(): unknown field(s) {sorted(invalid)}. "
            f"Available: {sorted(settable)}."
        )

    # drop ``level=None`` / ``type=None`` from
    # direct_args — these are "no change" semantically. previously
    # ``update(em, level=None)`` stamped ``level=None`` on the new
    # dataclass, then ``summary(new, level=new.level)`` crashed deep
    # in ``_validate_level`` with ``TypeError: '<' not supported
    # between instances of 'float' and 'NoneType'``. Treat None as
    # "leave as-is" for the strictly-validated numeric / string
    # fields. (``adjust=None`` is genuinely meaningful — it falls
    # back to the option / default — so keep that as-is.
    # ``bias_sigma`` is in ``_REBUILD_ONLY`` and refused earlier.)
    for _none_safe in ("level", "type"):
        if _none_safe in direct_args and direct_args[_none_safe] is None:
            direct_args.pop(_none_safe)

    # bootstrap-derived results cannot be re-level'd /
    # re-typed / re-adjusted because the per-draw values aren't
    # stored. Refuse early with a workflow hint pointing at
    # ``bootstrap_ci(raw_em, level=<new>)``. Without this guard,
    # ``update(em_b, level=0.99)`` flowed into ``summary()`` and
    # silently overwrote the bootstrap percentile CIs with t-Wald
    # CIs at the new level. Symmetric with in
    # ``summary()``.
    if getattr(obj, "df_method", "default") == "bootstrap":
        # ``bias_sigma`` is also a recompute key —
        # changing it would force the response back-transform to
        # rebuild without bootstrap draws.
        _recompute_keys = {"level", "type", "adjust", "bias_adjust", "bias_sigma"}
        # a kwarg present in ``direct_args`` but
        # whose value already equals the object's stored value is a
        # no-op and should pass through silently. previously
        # ``update(em_b, type='link')`` (where em_b is already
        # link-scale) raised — user-hostile. Only flag actual
        # mutations as recompute attempts.
        _recompute_attempted: set[str] = set()
        for _key in _recompute_keys & set(direct_args):
            _old = getattr(obj, _key, None)
            _new_val = direct_args[_key]
            try:
                if _key == "level":
                    if _new_val is None or (
                        _old is not None
                        and np.isclose(float(_new_val), float(_old))
                    ):
                        continue
                else:
                    # Normalise "none" / "" / None as equivalent for
                    # the ``adjust`` field; otherwise compare equal.
                    if _key == "adjust":
                        _old_norm = (str(_old).lower()
                                     if _old is not None else "none")
                        _new_norm = (str(_new_val).lower()
                                     if _new_val is not None else "none")
                        if _old_norm in ("none", "") and _new_norm in ("none", ""):
                            continue
                        if _old_norm == _new_norm:
                            continue
                    elif _new_val == _old:
                        continue
            except (TypeError, ValueError):
                # If the equality check itself raises (e.g. weird
                # numpy comparisons), fall through to flag as a
                # mutation.
                pass
            _recompute_attempted.add(_key)
        if _recompute_attempted:
            raise ValueError(
                "update() cannot mutate "
                f"{sorted(_recompute_attempted)} on a bootstrap-derived "
                "result because the per-draw values are not stored — "
                "only the precomputed percentile bounds. Re-run "
                "``bootstrap_ci(raw_em, ...)`` from the un-bootstrapped "
                "EMM at the desired settings."
            )

    # if `type=` changes, recompute via
    # the response-transform helper INSTEAD of just stamping the new
    # type on the frame (which would silently mislabel link-scale
    # values as response-scale).
    if "type" in direct_args:
        new_type = direct_args["type"]
        if new_type not in ("link", "response"):
            raise ValueError(
                f"type must be 'link' or 'response', got {new_type!r}."
            )
        if new_type != getattr(obj, "type", "link"):
            if new_type == "response":
                # sibling: also thread ``bias_sigma``
                # through the type-change recompute. Same logic as
                # the level-change branch below.
                obj = _recompute_response_from_link(
                    obj,
                    level=direct_args.get("level", getattr(obj, "level", 0.95)),
                    bias_adjust=bool(
                        direct_args.get(
                            "bias_adjust", getattr(obj, "bias_adjust", False),
                        )
                    ),
                    sigma=getattr(obj, "bias_sigma", None),
                )
            else:
                _recompute_link_from_response(obj) # raises
        # Drop type from direct_args; the helper set it.
        direct_args.pop("type", None)
        direct_args.pop("level", None) # level handled by helper

    # ``dataclasses.replace`` is a shallow copy; if
    # ``target`` / ``by`` are lists or ``_adjust_meta`` is a nested
    # dict, the new and old objects share the same mutable refs. A
    # later ``new.target.append("BAD")`` would silently mutate the
    # original. Defensively deep-copy these fields on every
    # ``update()`` so frozen-dataclass invariants hold for downstream
    # users (matches the contract that
    # ``dataclasses.replace`` is documented unsafe; this is the
    # ``update()``-path equivalent).
    import copy as _copy
    _shielded = {}
    for _attr in ("target", "by"):
        if _attr not in direct_args and hasattr(obj, _attr):
            _val = getattr(obj, _attr)
            if isinstance(_val, list):
                _shielded[_attr] = list(_val)
    if "_adjust_meta" not in direct_args and hasattr(obj, "_adjust_meta"):
        _am = obj._adjust_meta
        if _am is not None:
            _shielded["_adjust_meta"] = _copy.deepcopy(_am)
    if hasattr(obj, "method_args") and "method_args" not in direct_args:
        _ma = obj.method_args
        if isinstance(_ma, dict):
            _shielded["method_args"] = _copy.deepcopy(_ma)

    new = _dc_replace(obj, **{**direct_args, **_shielded})
    # If level changed (and type didn't trigger a recompute), update CI
    # endpoints. response-scale targets are now supported
    # by rebuilding the link-scale view (`_response_to_link_result`)
    # and routing through `_recompute_response_from_link` at the new
    # level — same path `summary` uses. refused this on the
    # incorrect assumption that the link state was lost.
    if "level" in direct_args:
        if getattr(new, "type", "link") == "response":
            link_view = _response_to_link_result(new)
            # forward stored ``bias_sigma`` so the
            # level-recompute uses the user's sigma override instead
            # of silently reverting to ``info.scale``. previously the
            # response-scale ``update(em, level=...)`` path called
            # ``_recompute_response_from_link`` without
            # ``sigma=``, so ``regrid_response(em, bias_adjust=True,
            # sigma=2.0)`` then ``update(level=0.9)`` produced
            # response means computed with the model default sigma
            # — absolute error ~3 on the the bug reproduction.
            # ``summary(em, level=...)`` already threaded
            # ``_sigma_eff`` per ; this brings ``update``
            # in line.
            new = _recompute_response_from_link(
                link_view, level=new.level,
                bias_adjust=bool(getattr(new, "bias_adjust", False)),
                sigma=getattr(new, "bias_sigma", None),
            )
        else:
            new_frame = new.frame.copy()
            new_cols = summary(new, infer=(True, False), level=new.level)
            if "lower_cl" in new_cols.columns:
                new_frame["lower_cl"] = new_cols["lower_cl"]
                new_frame["upper_cl"] = new_cols["upper_cl"]
            new = _dc_replace(new, frame=new_frame)
    # if `adjust=` changed on a contrast,
    # recompute p_value / t_ratio using the new method.
    if "adjust" in direct_args and _is_contrast_result(new):
        new_cols = summary(new, infer=(False, True), adjust=new.adjust)
        new_frame = new.frame.copy()
        new_frame["p_value"] = new_cols["p_value"]
        if "t_ratio" in new_cols.columns:
            new_frame["t_ratio"] = new_cols["t_ratio"]
        new = _dc_replace(new, frame=new_frame)
    # (, partial): when `adjust=` changes
    # on an EMMResult, R also widens the CIs by the family-wise
    # critical value. The summary-layer side fix lets `summary(em,
    # adjust=...)` return the right CIs, but `update(em, adjust=...)`
    # also needs to write them back into `new.frame` so downstream
    # consumers reading the frame directly (without going through
    # summary) see the adjusted endpoints.
    if "adjust" in direct_args and _is_emm_result(new):
        new_cols = summary(new, infer=(True, False), adjust=new.adjust)
        if "lower_cl" in new_cols.columns:
            new_frame = new.frame.copy()
            new_frame["lower_cl"] = new_cols["lower_cl"]
            new_frame["upper_cl"] = new_cols["upper_cl"]
            new = _dc_replace(new, frame=new_frame)
    return new
