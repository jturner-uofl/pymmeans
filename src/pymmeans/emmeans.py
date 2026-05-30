"""Estimated marginal means.

EMMs are model-predicted averages over a reference grid, optionally conditioned
on by-factors. Math:

    mu = L_marg @ beta
    var(mu) = diag(L_marg @ V @ L_marg.T)

where ``L_marg`` averages the rows of the full grid's L matrix that share a
target level (and a by-level, if conditioned).

For very large grids (e.g. an interaction of 16 factors with 50-level
target — the GitHub issue #282 scenario) the full L matrix can exceed
memory. In that case the implementation streams the cartesian product in
chunks and accumulates per-key sums + counts, never materializing the
full L matrix.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from patsy import build_design_matrices
from scipy import stats

from pymmeans.ref_grid import build_grid_spec, grid_size
from pymmeans.utils import ModelInfo, from_fitted

# ``RefGrid`` is referenced only in the
# ``emmeans()`` return-type annotation (forward string with
# ``from __future__ import annotations``); pull it into the
# TYPE_CHECKING block so the symbol resolves for type checkers
# and IDEs without producing an import cycle at runtime.
if TYPE_CHECKING:
    from pymmeans.ref_grid import RefGrid


def _validate_level(level: float) -> float:
    """Confirm ``level`` is a valid confidence level in ``(0, 1)``.

    Centralised because every public function that builds a confidence
    interval (``emmeans``, ``bootstrap_ci``, ``emtrends``, plotting
    helpers) needs the same guard. ``level=95`` (a
    common typo for ``level=0.95``) silently producing NaN CIs and
    ``level=-0.1`` returning inverted CIs.
    """
    try:
        lvl = float(level)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"level must be a number; got {level!r}.") from exc
    if not (0.0 < lvl < 1.0):
        raise ValueError(
            f"level must lie strictly in (0, 1); got {lvl}. "
            "Use 0.95 for a 95% confidence interval, not 95."
        )
    return lvl


def _validate_chunk_size(chunk_size: int | None) -> int | None:
    """Confirm ``chunk_size`` is a positive integer, or ``None``.

    ``chunk_size=0`` would make ``itertools.islice`` produce an empty
    iterator, the marginalisation loop would never run, and every EMM
    would come back zero — so we refuse ``0`` explicitly.
    """
    if chunk_size is None:
        return None
    try:
        cs = int(chunk_size)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"chunk_size must be an integer; got {chunk_size!r}."
        ) from exc
    if cs < 1:
        raise ValueError(
            f"chunk_size must be >= 1; got {cs}. Use None for the "
            "auto-streaming default."
        )
    return cs

# Auto-stream when the full L matrix would exceed this many bytes.
_STREAM_MEMORY_THRESHOLD_BYTES = 200 * 1024 * 1024
_STREAM_CHUNK_ROWS_DEFAULT = 100_000


@dataclass(frozen=True)
class EMMResult:
    """Estimated marginal means with optional by-group conditioning.

    The ``linfct`` matrix is always on the link scale so that downstream
    contrasts (pairs, contrast) can do their math regardless of the scale
    selected for display in ``frame``.

    .. warning::
       Do **not** mutate this dataclass directly via
       ``dataclasses.replace`` (or any other field-level write).
       ``frame`` / ``linfct`` / ``at`` / ``weights`` / ``bias_adjust``
       / ``inference_kind`` / ``df_method`` are coupled: changing one
       without rebuilding the others produces a split-brain object
       whose displayed values silently disagree with its metadata.
       Use :func:`pymmeans.update` for *display* fields
       (``level`` / ``adjust`` / ``type``); recompute via
       :func:`pymmeans.emmeans` / :func:`pymmeans.regrid_response` /
       :func:`pymmeans.apply_satterthwaite` /
       :func:`pymmeans.apply_kenward_roger` /
       :func:`pymmeans.posterior.posterior_emmeans` for *reconstruction*
       fields. ``update()`` enforces this distinction by refusing to
       stamp reconstruction-control kwargs.
    """

    frame: pd.DataFrame
    linfct: np.ndarray
    model_info: ModelInfo
    target: list[str]
    by: list[str]
    level: float
    type: str = "link"
    bias_adjust: bool = False
    """True iff the response-scale values were produced via
    ``regrid_response(..., bias_adjust=True)``. ``bootstrap_ci`` and
    other downstream tooling need this to apply the same R-style Taylor
    bias correction to bootstrap draws — without it the bootstrap CIs
    silently revert to the unadjusted inverse transform (#4)."""
    bias_sigma: Any = None
    """the ``sigma=`` override (if any) passed to
    ``regrid_response(..., bias_adjust=True, sigma=...)``. Carried so
    later ``summary(em, level=...)`` / case-bootstrap recomputes keep
    using the user's sigma instead of silently reverting to
    ``model_info.scale``. ``None`` means "use ``info.scale``" (the
    default path). Mirrors R's ``sigma`` argument on
    ``summary.emmGrid(..., bias.adjust=TRUE, sigma=...)``."""
    inference_kind: str = "wald"
    """``"wald"`` (default) means CIs are computed via the Wald
    approximation around the point estimate; ``"posterior"`` means
    they are posterior credible intervals from
    :func:`pymmeans.posterior.posterior_emmeans`. Set so that
    ``apply_satterthwaite`` / ``apply_kenward_roger`` can refuse
    posterior results (their t-based CI overwrite would be
    meaningless on a posterior; caught the silent
    overwrite)."""
    df_method: str = "default"
    """Which df / vcov correction has been applied. ``"default"``
    means the model's native df (``df_resid`` for OLS, ``inf`` for
    GLM / mixed); ``"satterthwaite"`` means
    :func:`~pymmeans.apply_satterthwaite` has updated df via Satterthwaite
    and SE via the original V_beta; ``"kenward_roger"`` means
    :func:`~pymmeans.apply_kenward_roger` has inflated V_beta to V_KR
    and updated df at V_KR. uses this so ``pairs`` /
    ``contrast`` can propagate the same correction to the contrast-level
    L matrix instead of silently demoting back to z-inference."""
    adjust: str | None = None
    """ default multiplicity
    adjustment for ``summary(em, infer=(...))``. ``None`` is the v0.1
    default and is treated as ``'none'`` by :func:`summary`; setting
    it via ``update(em, adjust='sidak')`` lets users carry a default
    adjustment on the EMM object so subsequent ``summary`` /
    ``confint`` calls widen CIs without an explicit kwarg. Mirrors
    R `update(em, adjust='sidak')` semantics. ContrastResult has had
    this field since v0.1 dev; adding it to EMMResult unifies the
    update API."""
    _satt_cache: Any = None
    """pickle-safe snapshot of the REML state used by
    :func:`~pymmeans.apply_satterthwaite`. Populated automatically the
    first time the correction runs on a MixedLM EMM. Without this,
    ``pickle.dumps`` + ``pickle.loads`` + ``pairs(...)`` failed
    because the propagation path needed ``raw_result`` (which
    ``ModelInfo.__getstate__`` correctly drops on pickle). With the
    cache, the post-pickle path skips ``raw_result`` entirely and
    flows through. Type: ``Optional[_SattCache]`` (NamedTuple)."""
    at: dict[str, Any] | None = None
    """the ``at=`` overrides used to build this EMM (or
    ``None`` if none were supplied). Carried so case-bootstrap and
    ``pairs(simple=)`` rebuild paths can pass the SAME grid restrictions
    to the rebuilt EMMs. Without this, ``pairs(em, simple="a")`` on
    an EMM built with ``at={"x": [10]}`` silently rebuilt at
    ``x = mean(x)`` and gave wrong simple-effect estimates."""
    weights: str = "equal"
    """the ``weights=`` mode used to build this EMM
    (``"equal"`` default, or ``"proportional"`` / ``"outer"`` /
    ``"cells"``). Carried so rebuild paths preserve the marginalisation
    scheme."""

    def __repr__(self) -> str:
        pct = round(self.level * 100)
        scale = "response" if self.type == "response" else "link"
        target = ", ".join(self.target) or "(none)"
        header = f"EMMs of {target} on {scale} scale, {pct}% CI"
        if self.by:
            header += f" (by {', '.join(self.by)})"
        return f"{header}\n{self.frame!r}"

    @property
    def n_rows(self) -> int:
        """Number of EMM rows in the summary frame."""
        return len(self.frame)

    def __getitem__(self, key: Any) -> EMMResult:
        """Row-level subsetting that keeps ``frame`` and ``linfct`` aligned.

        Mirrors R ``emmGrid``'s ``[`` operator: select a subset of EMM
        rows (by integer index, slice, integer iterable, or boolean
        mask) and get back a new :class:`EMMResult` with both the
        summary frame and the underlying ``linfct`` matrix subsetted
        in lockstep so downstream :func:`pairs` /
        :func:`pymmeans.contrast` math still works.

        Examples::

            em = emmeans(fit, "g")
            em[0]                    # first EMM row
            em[1:3]                  # rows 1 and 2
            em[[0, 2]]               # fancy integer index
            em[em.frame["emmean"] > 0]   # boolean mask

        String keys are deliberately NOT supported here — they would
        be ambiguous between "select a column" and "select a row by
        level label". Use ``em.frame[col]`` for columns; for label-
        based row selection, build a boolean mask
        (``em[em.frame["g"] == "a"]``).
        """
        from dataclasses import replace as _dc_replace

        n = self.n_rows
        if n == 0:
            raise IndexError("cannot index an empty EMMResult")

        if isinstance(key, str):
            raise TypeError(
                "EMMResult does NOT support string keys. Use "
                "`em.frame[col]` for column access, or build a "
                "boolean mask like `em[em.frame['g'] == 'a']` for "
                "label-based row selection."
            )

        if isinstance(key, int | np.integer):
            i = int(key)
            if i < -n or i >= n:
                raise IndexError(
                    f"EMMResult index {i} out of range for {n} rows."
                )
            if i < 0:
                i += n
            idx = np.array([i])
        elif isinstance(key, slice):
            idx = np.arange(*key.indices(n))
        else:
            arr = np.asarray(key)
            if arr.dtype == bool:
                if arr.shape != (n,):
                    raise IndexError(
                        f"EMMResult boolean mask shape {arr.shape} "
                        f"does not match {n} rows."
                    )
                idx = np.flatnonzero(arr)
            elif np.issubdtype(arr.dtype, np.integer):
                idx = arr.astype(int)
                if idx.ndim != 1:
                    raise IndexError(
                        "EMMResult fancy index must be 1-D; got shape "
                        f"{idx.shape}."
                    )
                bad = (idx < -n) | (idx >= n)
                if bad.any():
                    raise IndexError(
                        "EMMResult fancy index out of range: "
                        f"{idx[bad].tolist()} (n_rows={n})."
                    )
                idx = np.where(idx < 0, idx + n, idx)
            else:
                raise TypeError(
                    "EMMResult index must be int, slice, integer "
                    f"iterable, or boolean mask; got {type(key).__name__} "
                    f"(dtype={arr.dtype})."
                )

        new_frame = (
            self.frame.iloc[idx].reset_index(drop=True).copy()
        )
        new_linfct = self.linfct[idx, :].copy()
        return _dc_replace(self, frame=new_frame, linfct=new_linfct)

    # emmGrid arithmetic, matching R `emmGrid`'s `+ / - / * / /`
    # operators. Two EMMs from the SAME model with the SAME shape can be
    # combined linearly: the result is a new EMMResult whose linfct is
    # the linear combination of the inputs' linfcts, with the point
    # estimate, SE, and CI re-derived from `L @ beta` and the original
    # vcov. Use cases: difference-in-differences-style EMMs (em1 - em2),
    # averaging two reference grids ((em1 + em2) / 2), or rescaling
    # (em * 0.5).
    #
    # Result columns: factor / by columns are taken from `self`'s frame
    # (the caller is responsible for ensuring the row order matches
    # `other`'s — we don't auto-align). For `- other`, the contrast
    # column is renamed; for `+ other`, kept as-is.
    #
    # We deliberately return EMMResult (not ContrastResult) for `-`
    # because R does the same: emmGrid arithmetic produces another
    # emmGrid, not a hypothesis-testing contrast. Call `pairs(...)` or
    # `contrast(...)` on the result if you want adjusted p-values.

    def __add__(self, other: Any) -> EMMResult:
        return self._linear_combine(other, sign=+1.0, op_name="+")

    def __sub__(self, other: Any) -> EMMResult:
        return self._linear_combine(other, sign=-1.0, op_name="-")

    def __mul__(self, scalar: float) -> EMMResult:
        # refuse non-finite scalars
        # so accidental NaN/Inf inputs surface as clean errors rather
        # than producing all-NaN EMM tables.
        s = float(scalar)
        if not np.isfinite(s):
            raise ValueError(
                f"EMMResult scalar multiplier must be finite; got "
                f"{scalar!r}. NaN/Inf would propagate to every cell of "
                "the resulting EMM table."
            )
        return self._scale(s, op_name="*")

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> EMMResult:
        s = float(scalar)
        if not np.isfinite(s):
            raise ValueError(
                f"EMMResult scalar divisor must be finite; got "
                f"{scalar!r}."
            )
        if s == 0:
            raise ZeroDivisionError(
                "EMMResult division by zero; use `* 0` if you really "
                "want a zeroed result, but the SE would be zero too."
            )
        return self._scale(1.0 / s, op_name="/")

    def _scale(self, k: float, op_name: str) -> EMMResult:
        from dataclasses import replace as _dc_replace

        # EMMResult arithmetic recomputes
        # `emmean` from `L @ beta` on the LINK scale, which silently
        # mis-stamps the result as `type="response"` if the source was
        # already on response scale. The displayed `emmean` would then
        # be link-scale values labelled as response — exactly the
        # silent-wrong category. Refuse cleanly with a workflow hint.
        if getattr(self, "type", "link") == "response":
            raise ValueError(
                f"EMMResult {op_name} is only defined on link-scale "
                "results because the recomputed `emmean = L @ beta` is "
                "on the link scale. Compute the linear combination on "
                "link-scale EMMs first, then `regrid_response` the "
                "result if you need response-scale display values."
            )
        # posterior EMMs carry percentile
        # credible intervals; the arithmetic path re-derives CIs from
        # `t.ppf(1 - alpha/2, df_arr)` on the corrected SE, which is
        # WALD inference. Mixing posterior point estimates with Wald
        # intervals is a category error — refuse.
        if getattr(self, "inference_kind", "wald") == "posterior":
            raise ValueError(
                f"EMMResult {op_name} on posterior summaries would "
                "replace percentile credible intervals with Wald "
                "intervals (the arithmetic path uses t.ppf on the "
                "corrected SE). Combine the underlying beta_samples "
                "directly and rerun `posterior_emm_summary` instead."
            )
        new_linfct = self.linfct * k
        new_frame = self.frame.copy()
        # Re-derive estimate and SE from the rescaled L.
        info = self.model_info
        beta = info.beta
        vcov = info.vcov
        new_emm = new_linfct @ beta
        new_var = np.einsum("ij,jk,ik->i", new_linfct, vcov, new_linfct)
        new_se = np.sqrt(np.clip(new_var, 0.0, None))
        new_frame["emmean"] = new_emm
        new_frame["SE"] = new_se
        # CI from t-quantile at the existing df
        from scipy import stats as _stats

        df_arr = new_frame["df"].to_numpy(dtype=float) if "df" in new_frame.columns \
            else np.full(len(new_frame), np.inf)
        with np.errstate(invalid="ignore"):
            crit = _stats.t.ppf(1.0 - (1.0 - self.level) / 2.0, df_arr)
        new_frame["lower_cl"] = new_emm - crit * new_se
        new_frame["upper_cl"] = new_emm + crit * new_se
        return _dc_replace(self, frame=new_frame, linfct=new_linfct)

    def _linear_combine(
        self, other: Any, sign: float, op_name: str
    ) -> EMMResult:
        from dataclasses import replace as _dc_replace

        if not isinstance(other, EMMResult):
            return NotImplemented # type: ignore[return-value]
        # Compatibility checks
        # ( / #5): refuse arithmetic on
        # response-scale or posterior operands BEFORE the model-info
        # identity check, so users get the most-actionable error first.
        if getattr(self, "type", "link") == "response" or \
                getattr(other, "type", "link") == "response":
            raise ValueError(
                f"EMMResult {op_name} is only defined on link-scale "
                "results; the recomputed `emmean = L @ beta` is on "
                "the link scale, so combining response-scale operands "
                "would silently mix scales. Compute on link-scale "
                "EMMs first, then `regrid_response` the result."
            )
        if getattr(self, "inference_kind", "wald") == "posterior" or \
                getattr(other, "inference_kind", "wald") == "posterior":
            raise ValueError(
                f"EMMResult {op_name} on posterior summaries would "
                "replace percentile credible intervals with Wald "
                "intervals. Combine the underlying beta_samples "
                "directly via `posterior_emm_summary(samp, L_new)` "
                "where L_new = L1 + sign * L2."
            )
        # identity-based check (`is`)
        # fails after `pickle.loads` even when the two EMMs came from
        # the same fit, because unpickling creates new ModelInfo
        # objects with the same content. Fall back to structural
        # equality on the inference-critical fields (param_names,
        # beta, vcov) when the identity check fails.
        if self.model_info is not other.model_info:
            # the fallback used
            # `np.allclose(..., atol=0)` which still uses the default
            # `rtol=1e-05`, accepting near-equal-but-different models.
            # Pickle round-trip preserves floats bitwise, so strict
            # `array_equal` is the right test; also extend the field
            # list to include scale / df_resid / family-type /
            # response_name (any of which differing means the inference
            # would differ).
            mi_a, mi_b = self.model_info, other.model_info
            same_struct = (
                mi_a.param_names == mi_b.param_names
                and mi_a.beta.shape == mi_b.beta.shape
                and np.array_equal(mi_a.beta, mi_b.beta)
                and mi_a.vcov.shape == mi_b.vcov.shape
                and np.array_equal(mi_a.vcov, mi_b.vcov)
                and mi_a.df_resid == mi_b.df_resid
                and mi_a.scale == mi_b.scale
                and mi_a.response_name == mi_b.response_name
                and mi_a.is_mixed == mi_b.is_mixed
                and type(mi_a.family) is type(mi_b.family)
                and mi_a.offset_mean == mi_b.offset_mean
            )
            if not same_struct:
                raise ValueError(
                    f"EMMResult {op_name} requires both operands from "
                    "the same model. Identity differs (typical after "
                    "pickle round-trip) AND structural fields (param_"
                    "names / beta / vcov) don't match either. Combine "
                    "grids from different fits manually via linfct."
                )
        if self.linfct.shape != other.linfct.shape:
            raise ValueError(
                f"EMMResult {op_name} requires matching shapes; got "
                f"{self.linfct.shape} vs {other.linfct.shape}. Use "
                "`at=` or `by=` to align the two grids before combining."
            )
        # the result frame's factor /
        # by columns are copied from `self`. If the two operands have
        # different factor labels per row (e.g. `em_at_b_X - em_at_b_Y`
        # subtracts row-by-row but the result frame still says
        # `b=X`), downstream readers can't tell what the contrast is.
        # Refuse cleanly when the target+by labels don't line up.
        label_cols = list(dict.fromkeys((self.target or []) + (self.by or [])))
        if label_cols:
            # `DataFrame.equals` is
            # dtype-sensitive — categorical vs object with the same
            # string values returns False. Compare element-wise after
            # casting to object so the round-trip (pickle, replace)
            # doesn't falsely refuse identical labels.
            self_labels = self.frame[label_cols].reset_index(drop=True)
            other_labels = other.frame[label_cols].reset_index(drop=True)
            self_obj = self_labels.astype(object).values.tolist()
            other_obj = other_labels.astype(object).values.tolist()
            if self_obj != other_obj:
                raise ValueError(
                    f"EMMResult {op_name} requires identical target/by "
                    f"labels in the same row order. Self labels:\n"
                    f"{self_labels}\nOther labels:\n{other_labels}\n"
                    "Build the combination manually via linfct if the "
                    "asymmetry is intentional."
                )
        new_linfct = self.linfct + sign * other.linfct
        info = self.model_info
        beta = info.beta
        vcov = info.vcov
        new_emm = new_linfct @ beta
        new_var = np.einsum("ij,jk,ik->i", new_linfct, vcov, new_linfct)
        new_se = np.sqrt(np.clip(new_var, 0.0, None))
        # Frame: take factor columns from self, replace emmean/SE/CI
        new_frame = self.frame.copy()
        new_frame["emmean"] = new_emm
        new_frame["SE"] = new_se
        from scipy import stats as _stats

        df_arr = new_frame["df"].to_numpy(dtype=float) if "df" in new_frame.columns \
            else np.full(len(new_frame), np.inf)
        with np.errstate(invalid="ignore"):
            crit = _stats.t.ppf(1.0 - (1.0 - self.level) / 2.0, df_arr)
        new_frame["lower_cl"] = new_emm - crit * new_se
        new_frame["upper_cl"] = new_emm + crit * new_se
        return _dc_replace(self, frame=new_frame, linfct=new_linfct)


def _as_list(value: Any) -> list[str]:
    # the cells weighting path calls
    # this on at-values from a user dict. Previously a scalar (e.g.
    # `at={"x": 1.5}`) hit `list(value)` and raised `TypeError:
    # 'float' object is not iterable`. Accept scalars as single-element
    # lists, matching the behaviour of `ref_grid._as_list` (the
    # canonical helper); the duplicate definition here predates the
    # ref_grid one and should ideally be removed (followup).
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, np.ndarray, pd.Series)):
        return list(value)
    return [value]


def _type_submodel_rhs(
    info: ModelInfo, primary: set[str], kind: str
) -> str:
    """Build the reduced-formula RHS for ``submodel="minimal"/"type2"/"type3"``.

    ``kind="minimal"`` (and ``"type3"``, which R emmeans treats
    identically) keeps ONLY the terms whose factors are all primary
    (target / by) — every non-primary main effect, covariate, and
    mixed interaction is dropped. For an EMM of ``A`` from
    ``y ~ A*B + x`` this is the ``~ A`` submodel, matching R
    ``emmeans(fit, ~A, submodel="minimal")``. (The conventional
    equally-weighted Type-III LS-means is pymmeans's *default*
    ``emmeans()`` output; R's ``submodel="type3"`` string is the
    projection-onto-primary-terms convention, which we mirror here.)

    ``kind="type2"`` applies the Type-II marginality principle: a
    primary-factor effect is estimated adjusting for every term that
    does NOT contain it. Operationally we DROP every model term whose
    factor set contains BOTH a primary (target / by) factor AND a
    non-primary factor — i.e. the interactions that mix a primary
    factor with a covariate or non-primary factor. Main effects,
    pure-primary interactions (needed to span the EMM grid), and
    pure-non-primary terms are kept. For an additive model (no such
    mixed interactions) this is the full model, so type-II EMMs equal
    the default — they only differ when the target interacts with
    another predictor.

    The RHS is reconstructed from ``design_info.term_names`` (patsy's
    own term strings), so it round-trips exactly through patsy when
    re-evaluated by :func:`_apply_submodel`.
    """
    di = info.design_info
    k = kind.lower()

    def _canon(name: str) -> str:
        return info.aliases.get(name, name)

    primary_canon = {_canon(p) for p in primary}
    kept: list[str] = []
    has_intercept = False
    for term, tname in zip(di.terms, di.term_names, strict=True):
        fac_codes = {_canon(f.name()) for f in term.factors}
        if not fac_codes:  # intercept term
            has_intercept = True
            continue
        has_primary = bool(fac_codes & primary_canon)
        has_nonprimary = bool(fac_codes - primary_canon)
        if k in ("minimal", "type3"):
            # Keep ONLY terms whose factors are ALL primary (drop every
            # non-primary main effect, covariate, and mixed interaction).
            # Matches R emmeans's submodel="minimal" / "type3" — the
            # projection onto the primary-factor-only submodel (e.g.
            # ``~ A`` for an EMM of ``A`` from ``y ~ A*B + x``).
            if has_primary and not has_nonprimary:
                kept.append(tname)
        else:  # type2
            # Drop terms mixing a primary and a non-primary factor; keep
            # pure-primary, pure-non-primary, and main effects.
            if not (has_primary and has_nonprimary):
                kept.append(tname)
    parts = (["1"] if has_intercept else ["0"]) + kept
    return " + ".join(parts)


def _apply_submodel(
    info: ModelInfo, submodel: str, primary: set[str] | None = None
) -> ModelInfo:
    """Project ``info`` onto a nested submodel via closed-form projection.

    For a nested submodel ``X_sub ⊂ span(X_full)``:

        beta_sub = pinv(X_sub' X_sub) · X_sub' · X_full · beta_full
        vcov_sub = J · vcov_full · J.T,   J := pinv(X_sub' X_sub) · X_sub' · X_full

    Exact for OLS; for GLMs the projection treats the full fit's
    coefficient covariance as the "true" variance and propagates it
    through the linear projection, which matches R
    ``emmeans(fit, submodel = ~ ...)`` numerically.

    ``submodel`` accepts an explicit formula RHS, ``"minimal"``
    (intercept-only), or ``"type2"`` / ``"type3"`` (resolved from the
    model's term structure via :func:`_type_submodel_rhs`, which needs
    the ``primary`` = target+by factor names).
    """
    from dataclasses import replace as _dc_replace

    from patsy import dmatrix

    # Reject post-pickle or otherwise dataless fits early — we need
    # the training frame to rebuild both X_full (via design_info) and
    # X_sub (via the submodel formula). Without it we can't compute
    # the projection.
    data = info.data
    if data is None or len(data) == 0 or info.design_info is None:
        raise ValueError(
            "submodel= requires the original training data and patsy "
            "design_info on the ModelInfo, but ``info.data`` is empty "
            "or ``info.design_info`` is None (typically because the "
            "fit was pickled — pickling drops both). Fixes:\n"
            "  - Call emmeans(..., submodel=...) on a fresh fit in "
            "the current process.\n"
            "  - Drop submodel= and use ``contrast(..., method=...)`` "
            "to encode the reduced-model comparison manually."
        )

    s = submodel.strip()
    if s.lower() in ("minimal", "type2", "type3"):
        if primary is None:
            raise ValueError(
                f"submodel={s!r} needs the primary (target / by) "
                "factors to derive the reduced model; this is an "
                "internal-call error (emmeans passes them "
                "automatically)."
            )
        sub_rhs = _type_submodel_rhs(info, primary, s.lower())
    else:
        # accept both "~ rhs" and bare "rhs" forms
        sub_rhs = s.lstrip("~").strip() or "1"

    # Build X_full and X_sub from the training frame.
    X_full_df = dmatrix(info.design_info, data, return_type="dataframe")
    X_full = np.asarray(X_full_df, dtype=float)
    try:
        X_sub_df = dmatrix(sub_rhs, data, return_type="dataframe")
    except Exception as exc:
        raise ValueError(
            f"submodel={submodel!r}: patsy could not parse the "
            f"reduced formula RHS {sub_rhs!r} against the training "
            "frame. Make sure the submodel only references columns "
            f"that are in the original fit (factors: "
            f"{sorted(info.factors)}; numerics: "
            f"{sorted(info.numeric_means)})."
        ) from exc
    X_sub = np.asarray(X_sub_df, dtype=float)

    if X_sub.shape[0] != X_full.shape[0]:
        raise ValueError(
            f"submodel={submodel!r}: reduced design has "
            f"{X_sub.shape[0]} rows but the full design has "
            f"{X_full.shape[0]}; this usually means the submodel "
            "formula uses a column with different NA handling than "
            "the original fit. Pre-clean the training data so both "
            "formulas see the same rows."
        )

    # closed-form projection:
    #   J = pinv(X_sub' X_sub) · X_sub' · X_full   shape (q, p)
    # so beta_sub = J · beta_full and vcov_sub = J · vcov_full · J.T.
    XtX = X_sub.T @ X_sub
    # Use solve when well-conditioned; fall back to pinv otherwise so a
    # rank-deficient submodel design (e.g. ``submodel='g'`` when ``g``
    # is exactly co-linear with another factor) still produces a
    # well-defined projection.
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
    J = XtX_inv @ X_sub.T @ X_full
    beta_sub = J @ info.beta
    vcov_sub = J @ info.vcov @ J.T
    # Symmetrize tiny floating-point noise before downstream
    # consumers (Satterthwaite / KR / Cholesky) see it.
    vcov_sub = 0.5 * (vcov_sub + vcov_sub.T)

    sub_design_info = X_sub_df.design_info
    sub_param_names = list(sub_design_info.column_names)

    # multi_col_factors only makes sense in the
    # original design's column structure; the submodel's column
    # structure is different, so drop it. Anything not present in the
    # submodel design simply can't be addressed via ``at={col: ...}``
    # anymore — that's the user-visible contract of submodel=.
    return _dc_replace(
        info,
        beta=beta_sub,
        vcov=vcov_sub,
        param_names=sub_param_names,
        design_info=sub_design_info,
        multi_col_factors={},
    )


def _marginalize_streamed(
    info: ModelInfo,
    spec: dict[str, list],
    group_cols: list[str],
    chunk_size: int,
    weights_mode: str = "equal",
    factor_weights: dict[str, np.ndarray] | None = None,
    proportional_joint: dict[tuple, float] | None = None,
) -> tuple[np.ndarray, list[tuple]]:
    """Compute the marginalized L matrix without materializing the full grid.

    Iterates the cartesian product in chunks, builds the chunk's L matrix via
    patsy, and accumulates per-key weighted sums. Returns (L_marg, keys).

    Weight semantics match the analytic path:

    - ``equal``: uniform 1/k_levels (each row contributes weight 1 to its
      key, normalised by count).
    - ``outer``: each row's weight is the product of per-factor marginal
      frequencies over the non-target factors at that row's values.
    - ``proportional``: each row's weight is the joint frequency of its
      non-target factor combination (from training data), normalised
      within the spec-allowed levels.

    Non-target numeric covariates with multi-value ``at=`` contribute a
    uniform 1/k_values per row in all three modes (matching analytic
    behaviour). the previous version silently
    falling back to ``equal`` for the streamed path; this version
    propagates the weights through.
    """
    cols = list(spec.keys())
    val_lists = [spec[c] for c in cols]

    unique_keys: list[tuple] = list(itertools.product(*[spec[c] for c in group_cols]))
    key_to_row = {key: i for i, key in enumerate(unique_keys)}

    n_keys = len(unique_keys)
    n_params = info.n_params
    L_sum = np.zeros((n_keys, n_params))
    weight_sum = np.zeros(n_keys, dtype=float)

    non_target_factors = [
        c for c in cols if c not in group_cols and c in info.factors
    ]

    def _row_weight(row_vals: tuple) -> float:
        """Per-row weight given its full value-tuple in `cols` order."""
        if weights_mode == "equal":
            return 1.0
        # Extract the non-target factor combo for this row
        combo = tuple(
            row_vals[cols.index(fname)] for fname in non_target_factors
        )
        if weights_mode == "outer":
            w = 1.0
            for fname, lvl in zip(non_target_factors, combo, strict=True):
                fw = factor_weights.get(fname) if factor_weights else None
                if fw is None:
                    continue
                idx = info.factors[fname].index(lvl)
                # Normalise the factor's weight vector to sum 1 within
                # the spec-allowed levels.
                allowed_idx = [info.factors[fname].index(v) for v in spec[fname]]
                allowed_total = float(fw[allowed_idx].sum())
                if allowed_total > 0:
                    w *= float(fw[idx]) / allowed_total
                else:
                    w *= 1.0 / len(allowed_idx)
            return w
        if weights_mode == "proportional":
            joint = proportional_joint or {}
            return float(joint.get(combo, 0.0))
        raise AssertionError(f"unknown weights_mode {weights_mode!r}")

    product_iter = itertools.product(*val_lists)
    while True:
        chunk_tuples = list(itertools.islice(product_iter, chunk_size))
        if not chunk_tuples:
            break
        chunk_df = pd.DataFrame(chunk_tuples, columns=cols)
        for name, levels in info.factors.items():
            if name in chunk_df.columns:
                chunk_df[name] = pd.Categorical(chunk_df[name], categories=levels)
        [L_chunk] = build_design_matrices(
            [info.design_info], chunk_df, return_type="matrix"
        )
        L_chunk = np.asarray(L_chunk, dtype=float)

        row_weights = np.fromiter(
            (_row_weight(rt) for rt in chunk_tuples), dtype=float
        )

        groups = chunk_df.groupby(group_cols, observed=True, sort=False).indices
        for key, indices in groups.items():
            key_tuple = key if isinstance(key, tuple) else (key,)
            row = key_to_row[key_tuple]
            idx = np.asarray(indices)
            w = row_weights[idx]
            L_sum[row] += (w[:, None] * L_chunk[idx]).sum(axis=0)
            weight_sum[row] += w.sum()

    nonempty = weight_sum > 0
    L_marg = np.zeros_like(L_sum)
    L_marg[nonempty] = L_sum[nonempty] / weight_sum[nonempty, None]
    return L_marg, unique_keys


def _marginalize_cells(
    info: ModelInfo,
    spec: dict[str, list],
    group_cols: list[str],
    at: dict[str, Any] | None,
    fit_weights: np.ndarray | None,
    nuisance_cols: list[str] = (),
) -> tuple[np.ndarray, list[tuple]]:
    """Build ``L_marg`` using observed cell frequencies (R's
    ``weights='cells'``).

    Unlike the analytic / streamed paths which iterate the cartesian
    product of ``spec``, this path walks the actual training rows in
    ``info.data``, computes the design row for each, and averages
    within each unique (target, by) combination — weighted by
    ``fit_weights`` if present.

    Differences from ``weights='proportional'``:

    - **proportional** uses the MARGINAL joint frequency of non-target
      factors (one weighting table shared across all target levels).
    - **cells** uses the CONDITIONAL joint frequency of non-target
      factors WITHIN each target level (a separate weighting per
      target level). The two are identical on balanced designs but
      can differ substantially under unbalance.

    ``at=`` semantics:
    - Categorical override: keeps only rows whose categorical value is
      in the override list (`isin` filter).
    - Numeric override (single value): overrides the column to that
      value for every retained row, so L is computed at the at-
      specified value rather than the row's actual value.
    - Numeric override (multi-value): refused — ``cells`` assumes a
      single covariate value per row; multi-value at= would
      double-count rows. Use ``weights='proportional'`` or
      ``weights='equal'`` if you need a covariate sweep.

    Cell with zero matching rows: row in the returned key list, with
    all-NaN L row (estimability check will mark it non-estimable).
    """
    import itertools as _it

    from patsy import dmatrix as _dmatrix

    canonical_to_raw = {v: k for k, v in info.aliases.items()}

    # Build the data subset with at= overrides applied
    data = info.data.copy()
    fw = fit_weights.copy() if fit_weights is not None else None
    keep_mask = np.ones(len(data), dtype=bool)

    at = at or {}
    for k, v in at.items():
        v_list = _as_list(v)
        canonical = info.aliases.get(k, k)
        raw = canonical_to_raw.get(canonical, canonical)
        # Categorical: restrict to rows whose value is in the override list
        if canonical in info.factors:
            if raw in data.columns:
                keep_mask &= data[raw].isin(v_list).to_numpy()
            continue
        # Numeric covariate: single value -> override column; multi
        # value -> refuse (would double-count rows).
        if canonical in info.numeric_means:
            if len(v_list) > 1:
                raise NotImplementedError(
                    f"weights='cells' with multi-value at={{'{k}': "
                    f"{v_list}}} is not supported because the cells "
                    "path assumes a single covariate value per row. "
                    "Use weights='proportional' or weights='equal' for "
                    "a covariate sweep."
                )
            if raw in data.columns:
                val = float(v_list[0])
                # when the user fits
                # `y ~ np.log(x)` and passes `at={"np.log(x)": log(2)}`,
                # the canonical name is `np.log(x)` and the raw column
                # is `x`. We can't just write `data.x = log(2)` because
                # patsy will then evaluate `np.log(log(2))`. Invert the
                # canonical's transform on the override value to get
                # the raw-column value that gives the requested
                # canonical value when patsy re-evaluates.
                if canonical != raw:
                    from pymmeans.transforms import detect_transform
                    tr = detect_transform(canonical)
                    if tr is None:
                        raise NotImplementedError(
                            f"weights='cells' with at={{'{k}': ...}} on "
                            f"a transformed numeric ({canonical!r}) "
                            "requires the transform's inverse to set "
                            "the raw column value before patsy re-"
                            "evaluation. The transform isn't auto-"
                            "detected; pass `at={raw_col_name: ...}` "
                            "directly (with the pre-transform value), "
                            "or use weights='proportional'."
                        )
                    val = float(tr.inverse(np.asarray(val)))
                data[raw] = val

    data = data.loc[keep_mask].reset_index(drop=True)
    if fw is not None:
        fw = fw[keep_mask]
    if len(data) == 0:
        raise ValueError(
            "weights='cells': no training-data rows match the spec / at="
            " filter; nothing to aggregate."
        )

    # Build design matrix for the retained rows
    X = np.asarray(_dmatrix(info.design_info, data), dtype=float)

    # Group by (target, by) combo
    target_cols_raw = [canonical_to_raw.get(c, c) for c in group_cols]
    missing = [c for c, raw in zip(group_cols, target_cols_raw, strict=True)
               if raw not in data.columns]
    if missing:
        raise ValueError(
            f"weights='cells' needs the raw factor columns "
            f"{missing} in info.data; one or more is missing. (This "
            "usually means the model was fit on a transformed or "
            "pre-aggregated frame.)"
        )

    key_list: list[tuple] = list(_it.product(*[spec[c] for c in group_cols]))
    L_marg = np.zeros((len(key_list), info.n_params))

    # Build a per-row combo key from the raw columns
    keys_arr = list(zip(*[data[r].tolist() for r in target_cols_raw], strict=True))
    key_index = {k: i for i, k in enumerate(key_list)}

    if not nuisance_cols:
        # Existing plain cells path: sum design rows in each target cell,
        # divide by total weight.
        weight_sum = np.zeros(len(key_list))
        for row_idx, key in enumerate(keys_arr):
            ki = key_index.get(key)
            if ki is None:
                continue
            w = 1.0 if fw is None else float(fw[row_idx])
            L_marg[ki] += w * X[row_idx]
            weight_sum[ki] += w
        empty = weight_sum == 0
        with np.errstate(invalid="ignore", divide="ignore"):
            L_marg[~empty] = L_marg[~empty] / weight_sum[~empty, None]
        L_marg[empty] = np.nan
        return L_marg, key_list

    # `weights='cells'` × ``nuisance=`` path. R's `wt.nuis='equal'`
    # collapses nuisance levels uniformly — including empty
    # (target, nuisance, non-nuisance non-target) sub-cells. We can't
    # reproduce that via observed-cell averaging (rows that don't exist
    # contribute nothing), so we switch to ANALYTICAL EXTRAPOLATION at
    # each grid combo via patsy: build the design row for every
    # (target × nuisance × non-nuisance non-target × covariates-at-mean)
    # combination, average uniformly across nuisance levels, then weight
    # the (target, non-nuisance non-target) cells by their observed
    # frequency in the training data. Matches R `emmeans(..., weights=
    # "cells", nuisance=, wt.nuis="equal")` to floating-point precision.
    from patsy import build_design_matrices as _build_dm  # local re-import for clarity
    non_nuisance_non_target = [
        c for c in info.factors
        if c not in group_cols and c not in nuisance_cols
    ]
    nn_raw = [canonical_to_raw.get(c, c) for c in non_nuisance_non_target]

    target_levels_lists = [spec[c] for c in group_cols]
    nuisance_levels_lists = [list(info.factors[n]) for n in nuisance_cols]
    nn_levels_lists = [list(info.factors[c]) for c in non_nuisance_non_target]

    target_cells = list(_it.product(*target_levels_lists))
    nuisance_combos = list(_it.product(*nuisance_levels_lists))
    nn_combos = list(_it.product(*nn_levels_lists))
    if not nn_combos:
        nn_combos = [()]  # all non-target factors are nuisance

    # Build single grid DataFrame for all (target × nuisance × nn) combos
    # so patsy is called once.
    grid_rows: list[dict[str, Any]] = []
    for t_combo in target_cells:
        for n_combo in nuisance_combos:
            for nn_combo in nn_combos:
                row: dict[str, Any] = {}
                for cname, val in zip(group_cols, t_combo, strict=True):
                    raw = canonical_to_raw.get(cname, cname)
                    row[raw] = val
                for cname, val in zip(nuisance_cols, n_combo, strict=True):
                    raw = canonical_to_raw.get(cname, cname)
                    row[raw] = val
                for cname, val in zip(non_nuisance_non_target, nn_combo, strict=True):
                    raw = canonical_to_raw.get(cname, cname)
                    row[raw] = val
                for ncov, nmean in info.numeric_means.items():
                    raw = canonical_to_raw.get(ncov, ncov)
                    if raw not in row:
                        row[raw] = nmean
                grid_rows.append(row)
    grid_df = pd.DataFrame(grid_rows)
    # Categorical encoding so patsy treatments line up with the fit.
    for cname, levels in info.factors.items():
        raw = canonical_to_raw.get(cname, cname)
        if raw in grid_df.columns:
            grid_df[raw] = pd.Categorical(grid_df[raw], categories=levels)
    L_grid = np.asarray(_build_dm([info.design_info], grid_df)[0], dtype=float)
    n_t_cells = len(target_cells)
    n_nuis = len(nuisance_combos)
    n_nn_cells = len(nn_combos)
    L_grid = L_grid.reshape(n_t_cells, n_nuis, n_nn_cells, info.n_params)
    # Average over nuisance — R wt.nuis="equal", over ALL combos
    # (including those whose observed cell is empty; the analytical L is
    # well-defined at the model's design space).
    L_t_nn = L_grid.mean(axis=1)  # (n_t, n_nn, p)

    # cells-frequency weights on non-nuisance non-target, computed from
    # observed data (raw marginal freq(nn | target)).
    if nn_raw:
        tn_keys = list(zip(
            *[data[c].tolist() for c in target_cols_raw + nn_raw],
            strict=True,
        ))
    else:
        tn_keys = [(*tk,) for tk in zip(
            *[data[c].tolist() for c in target_cols_raw],
            strict=True,
        )]
    n_t_raw = len(target_cols_raw)
    target_count: dict[tuple, float] = {}
    target_nn_count: dict[tuple, float] = {}
    for row_idx, fk in enumerate(tn_keys):
        t_key = fk[:n_t_raw]
        nn_key = fk[n_t_raw:]
        if t_key not in key_index:
            continue
        w = 1.0 if fw is None else float(fw[row_idx])
        target_count[t_key] = target_count.get(t_key, 0.0) + w
        target_nn_count[(t_key, nn_key)] = target_nn_count.get(
            (t_key, nn_key), 0.0
        ) + w

    nn_combo_map = {nn: idx for idx, nn in enumerate(nn_combos)}
    weight_sum = np.zeros(len(key_list))
    for ti, t_combo in enumerate(target_cells):
        ki = key_index.get(t_combo)
        if ki is None:
            continue
        t_total = target_count.get(t_combo, 0.0)
        if t_total <= 0:
            L_marg[ki] = np.nan
            continue
        accumulated = np.zeros(info.n_params)
        for nn_combo in nn_combos:
            count = target_nn_count.get((t_combo, nn_combo), 0.0)
            if count <= 0:
                continue
            nn_idx = nn_combo_map[nn_combo]
            accumulated += (count / t_total) * L_t_nn[ti, nn_idx]
            weight_sum[ki] += count / t_total
        L_marg[ki] = accumulated
    empty = weight_sum == 0
    L_marg[empty] = np.nan
    return L_marg, key_list


_SENTINEL = object()


def emmeans(
    model: Any,
    specs: str | list[str],
    by: str | list[str] | None = None,
    at: dict[str, Any] | None = None,
    level: float | None = None,
    type: str | None = None,
    weights: str | None = None,
    chunk_size: int | None = None,
    *,
    cov_reduce: dict[str, Callable[..., float] | float] | None = None,
    cov_keep: list[str] | None = None,
    nuisance: str | list[str] | None = None,
    counterfactuals: dict[str, Any] | None = None,
    nesting: dict[str, str | list[str]] | None = None,
    vcov: np.ndarray | None = None,
    submodel: str | None = None,
) -> EMMResult | RefGrid:
    """Compute estimated marginal means.

    Parameters
    ----------
    model
        Fitted statsmodels OLS/GLM result or a ``ModelInfo``.
    specs
        Target factor name(s) for which to compute marginal means. If a list,
        the cross product is used.

        Also accepts R-style formula specs as a string — e.g.
        ``"pairwise ~ g"``, ``"trt.vs.ctrl ~
        g"``, ``"consec ~ g | block"``. The LHS names any contrast
        method (pairwise, revpairwise, tukey, trt.vs.ctrl /
        trt.vs.ctrl1 / trt.vs.ctrlk, poly, consec, eff, del.eff,
        mean_chg, identity, helmert). The RHS lists target factors;
        ``|`` separates target from by-factors (matches R's
        ``emmeans(fit, pairwise ~ g | block)`` syntax). Return value
        is an :class:`EmmList` with ``.emmeans`` and ``.contrasts``
        members.
    by
        Optional factor name(s) to condition on; one EMM per by-level.
    at
        Optional dict of values for grid construction (see ``ref_grid``).
    level
        Confidence level for intervals; default 0.95.
    type
        Scale for the displayed EMMs: ``"link"`` (default) for the linear
        predictor scale, or ``"response"`` to back-transform via the GLM's
        inverse link. SEs are delta-method on the response scale. The
        ``linfct`` matrix attached to the result is always link-scale so
        downstream contrasts work uniformly.
    weights
        Weighting scheme for averaging over non-target factors. Matches R
        emmeans modes:

        - ``"equal"`` (default): uniform weights over levels.
        - ``"proportional"``: joint cell frequencies of NON-target factors
          (one shared weighting table across all target levels).
        - ``"outer"``: product of per-factor marginal frequencies (same as
          ``"proportional"`` under the analytic Kronecker path on
          additive models).
        - ``"cells"``: observed CONDITIONAL cell frequencies INCLUDING the
          target factor. Walks the training data row-by-row, so it
          respects unbalanced designs where the non-target distribution
          varies by target level. Multi-value numeric ``at=`` is refused
          on this path (would double-count rows).
    chunk_size
        Optional. If set, force-stream the cartesian product in chunks of
        this many rows. By default, streaming auto-engages when the full
        L matrix would exceed ~200 MB.
    cov_reduce
        Optional dict ``{numeric_col: callable | scalar}`` overriding
        the default training-data-mean reduction for the named
        numeric covariates only. See :func:`pymmeans.ref_grid` for
        the full surface; ``emmeans`` threads ``cov_reduce`` through
        :func:`pymmeans.ref_grid.build_grid_spec` unchanged. Useful
        for sensitivity analyses ("predict at the 75th percentile
        of x instead of the mean") without rebuilding the
        reference grid by hand.
    cov_keep
        Optional list of numeric-covariate names to **keep** at their
        distinct observed values (treated as grid factors) rather than
        reducing to a single summary. Mirrors R
        ``emmeans(..., cov.keep = ...)``: the EMM table then carries
        one row per ``(target × by × kept-covariate-value)``
        combination — useful when a numeric predictor takes only a
        few values (dose levels, year, an ordinal score) and you want
        an EMM at each rather than at the mean. Forces the eager
        grid-materialised path. Requires live training data (the
        unique values come from ``info.data``; refused post-pickle —
        pin the values via ``at={col: [...]}`` instead). Refuses
        overlap with ``cov_reduce`` (keep-vs-reduce is contradictory)
        and with ``specs`` / ``by`` (a covariate can't be both target
        and kept dimension).
    nuisance
        Optional factor name or list of names to **explicitly mark
        as nuisance** — averaged over with equal weights regardless
        of the displayed grid. The validation catches typos and
        intent mismatches:

        - Every name must be a categorical factor in the model.
        - No name may appear in ``specs`` or ``by`` (would
          contradict the role declaration).

        Under the default ``weights='equal'`` (and the alias
        ``'flat'``), nuisance factors are already averaged
        uniformly, so the kwarg is **declarative only** — it
        doesn't change numbers, it just documents intent and
        catches typos before they propagate.

        Under ``weights='outer'`` and ``weights='proportional'``,
        ``nuisance=`` overrides the per-factor weight construction
        for the named factor(s): they are averaged with **equal
        marginals** (matching R ``ref_grid(..., nuisance=,
        wt.nuis='equal')`` — the R default), while the remaining
        non-target factors keep their training-data weighting.
        Validated against R `emmeans` machine-precision (see
        ``jss_audit/§VIII``).

        ``nuisance=`` with ``weights='cells'`` is still refused: the
        observed-cell aggregation path needs nuisance-specific
        pre-collapse logic distinct from the marginal / joint
        reweighting used by 'outer' / 'proportional'. Workarounds:
        use ``weights='proportional'`` (identical to 'cells' under
        balance over the nuisance factor), or pin the nuisance
        factors with ``at={...}``.
    nesting
        Optional ``dict[nested -> enclosing | list-of-enclosing]``
        declaring that a categorical factor's levels are only
        meaningful within another's (e.g. school within district).
        Mirrors R ``emmeans(..., nesting = list(school =
        "district"))``: the reference grid is filtered to only
        the ``(nested, *enclosing)`` tuples that actually appear
        in the training data, so phantom (district, school-not-in-
        that-district) cells are dropped from the EMM result.
        Under the default ``weights='equal'`` the averaging then
        operates on only the valid grid rows, giving the R-correct
        nested-EMM result.

        Examples::

            emmeans(fit, "district", nesting={"school": "district"})
            emmeans(fit, "school", nesting={"school": ["district", "region"]})

        Refused under non-equal weight modes
        (``weights='proportional'``, ``'outer'``, ``'cells'``):
        those weight modes have additional R-parity semantics
        around per-cell training-data frequencies that v0.1's
        filter-only implementation doesn't yet handle correctly.
        Use ``weights='equal'`` with ``nesting=`` for now; full
        R-parity nested-weights is a 0.2.0 candidate.

        Forces the eager (grid-materialised) marginalization path
        because the Kronecker analytic path assumes factor
        independence, which nesting breaks. Refuses post-pickle
        (the filter needs ``info.data`` to discover valid tuples).
    submodel
        Optional nested-submodel specification. Computes EMMs *as if*
        the model had been fit with the reduced formula, without
        actually refitting — uses the closed-form projection
        ``beta_sub = pinv(X_sub) @ X_full @ beta_full`` (exact for
        OLS / weighted-OLS when the submodel is nested in the full
        design; approximate for GLMs). Mirrors R
        ``emmeans(fit, ..., submodel = ~ a + b)``.

        Accepted forms:

        - ``"minimal"`` → project onto the **primary-factor-only**
          submodel: keep only terms whose factors are all
          target / by factors, dropping every non-primary main
          effect, covariate, and mixed interaction. For an EMM of
          ``g`` from ``y ~ g*h + x`` this is the ``~ g`` submodel.
          Matches R ``emmeans(fit, ~g, submodel="minimal")``.
        - String formula RHS, e.g. ``"g + x"`` or ``"~ g + x"`` →
          patsy-parsed reduced design. The formula must reference
          only columns from the original fit's design.
        - ``"type2"`` → Type-II marginality: drops every model term
          whose factors mix a primary (target / by) factor with a
          non-primary one (the interactions of the target with other
          predictors), keeping main effects, pure-primary
          interactions, and pure-non-primary terms. For an additive
          model this equals the default EMM — type-II only differs
          when the target interacts with another predictor. The
          contrasts match R ``emmeans(submodel="type2")`` exactly.
        - ``"type3"`` → alias for ``"minimal"`` (matches R emmeans's
          ``submodel="type3"``). NOTE: the conventional
          equally-weighted **Type-III LS-means** is pymmeans's
          *default* ``emmeans()`` output (no ``submodel=`` needed);
          this string follows R's projection-onto-primary-terms
          convention.

        Not yet supported:

        - User-supplied ``(n, q)`` matrix.

        Refused when ``info.data`` is empty (post-pickle), since the
        reduced design must be rebuilt from the training frame.
    vcov
        Optional ``(p, p)`` ndarray to override the model's default
        coefficient covariance. Use this to plug in robust /
        sandwich / cluster-robust / HAC SEs computed outside the fit
        without rebuilding the model. Mirrors R
        ``emmeans(fit, "g", vcov. = sandwich::vcovHC(fit))``.

        Validation: must be a square 2-D array of shape
        ``(n_params, n_params)``, symmetric to ``1e-8``, and
        positive-semidefinite (min eigenvalue ``> -1e-8``).
        ``info.vcov`` is replaced via ``dataclasses.replace`` so the
        rest of the EMM/contrast pipeline picks it up unchanged.

        If you already have a :class:`ModelInfo` (e.g. from
        :func:`from_survey` or :func:`qdrg`) carrying the SE you want,
        pass it directly as ``model=`` instead — this kwarg exists for
        the common case of having a vanilla statsmodels fit plus a
        custom vcov matrix.
    counterfactuals
        Optional dict ``{col: scalar_or_list}`` declaring
        counterfactual values for grid construction. Semantically
        equivalent to passing the same keys via ``at=``, but the
        named API documents the analyst's intent (this is a
        counterfactual-prediction workflow) and adds stricter
        validation:

        - Every key must be a known factor level or numeric
          covariate in the model.
        - No key may overlap with ``at=`` (would double-specify
          the value).
        - No key may overlap with ``specs`` / ``target`` (a target
          factor is what the EMM varies over, not what it pins
          to a counterfactual).

        Mirrors R ``ref_grid(..., counterfactuals=...)`` for the
        common case where the user wants to predict at specific
        intervention levels rather than at training-data defaults.
        Internally merges into the effective ``at=`` map after
        validation.

    Returns
    -------
    EMMResult
        Container with a summary ``frame``, the marginalized ``linfct`` matrix,
        the source ``model_info``, and the ``target``/``by``/``level`` used.

    Examples
    --------
    >>> import pandas as pd
    >>> import statsmodels.formula.api as smf
    >>> df = pd.DataFrame({
    ... "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ... "g": pd.Categorical(["a", "a", "b", "b", "c", "c"]),
    ... })
    >>> emm = emmeans(smf.ols("y ~ g", data=df).fit(), "g")
    >>> emm.frame[["g", "emmean"]]  # doctest: +NORMALIZE_WHITESPACE
       g  emmean
    0  a     1.5
    1  b     3.5
    2  c     5.5
    """
    # consult emm_options for any
    # kwarg the caller didn't supply explicitly. Explicit kwargs ALWAYS
    # win — this matches R semantics. The sentinel-vs-None pattern
    # lets users explicitly pass `None` (rare but defensible) and have
    # it differ from "not supplied".
    from pymmeans.options import get_emm_option as _opt
    if level is None:
        level = _opt("level", 0.95)
    if type is None:
        type = _opt("type", "link")
    if weights is None:
        weights = _opt("weights", "equal")
    level = _validate_level(level)
    chunk_size = _validate_chunk_size(chunk_size)
    # accept a `RefGrid` as
    # input so the R `library(lsmeans); lsmeans(ref.grid(fit, at=…),
    # "factor")` workflow ports directly. R's `lsmeans` first builds
    # a reference grid via `ref.grid` (which applies `at=`), then
    # averages over non-target factors. We translate by deriving an
    # `at=` from the RefGrid's grid and routing through the normal
    # emmeans path with the same model_info.
    from pymmeans.ref_grid import RefGrid as _RefGrid
    if isinstance(model, _RefGrid):
        if at is not None:
            raise ValueError(
                "emmeans(ref_grid, ..., at=...) is ambiguous: the "
                "RefGrid already encodes its own `at` values. Pass "
                "`at=` to `ref_grid(fit, at=...)` instead of `emmeans`."
            )
        target_for_at = _as_list(specs) + _as_list(by)
        # Derive an `at=` dict from non-target columns of the
        # reference grid. Treat numeric columns by their unique
        # values; categorical columns are skipped (handled by the
        # weighted-average path) unless they're the only way to
        # parameterise the grid.
        rg_at: dict[str, Any] = {}
        rg_info = model.model_info
        rg_grid = model.grid
        for col in rg_grid.columns:
            if col in target_for_at:
                continue
            values = list(pd.unique(rg_grid[col]))
            # Only set `at=` on numeric covariates (matches R: factor
            # levels are uniformly averaged regardless of the grid).
            if col in (rg_info.numeric_means or {}):
                rg_at[col] = values
        info = rg_info
        if rg_at:
            at = rg_at
    else:
        info = model if isinstance(model, ModelInfo) else from_fitted(model)

    # vcov= override.
    # Validates shape, symmetry, and PSD-ness BEFORE swapping, so
    # downstream Satterthwaite / KR / contrast code only sees a clean
    # covariance matrix and we surface bad input at the call site
    # instead of as a cryptic eig/cholesky failure deep in the stack.
    if vcov is not None:
        import dataclasses as _dc

        v = np.asarray(vcov, dtype=float)
        p = info.n_params
        if v.ndim != 2 or v.shape != (p, p):
            raise ValueError(
                f"vcov= must be a square ({p}, {p}) ndarray to match "
                f"the model's coefficient vector; got shape {v.shape}."
            )
        if not np.allclose(v, v.T, atol=1e-8, rtol=0.0):
            raise ValueError(
                "vcov= must be symmetric to atol=1e-8. Did you pass a "
                "non-symmetric robust SE matrix? Symmetrise via "
                "`0.5 * (V + V.T)` first."
            )
        # symmetrise the tiny numerical asymmetry that
        # made it past the tolerance check, so eigvalsh is well-defined.
        v_sym = 0.5 * (v + v.T)
        lam_min = float(np.linalg.eigvalsh(v_sym)[0])
        if lam_min < -1e-8:
            raise ValueError(
                "vcov= must be positive-semidefinite (min eigenvalue "
                f"> -1e-8); got {lam_min:.3e}. A negative eigenvalue "
                "usually means the matrix isn't a valid covariance — "
                "double-check the upstream sandwich/cluster "
                "estimator."
            )
        info = _dc.replace(info, vcov=v_sym)

    # submodel= projection.
    # Builds a new ModelInfo whose design is the reduced (nested)
    # submodel via closed-form projection of the full-fit beta /
    # vcov onto the submodel column space. Exact for OLS; for GLMs
    # it computes the "marginal-prediction" submodel as R does (uses
    # the full fit's coefficient covariance, propagated by the
    # projection — matches `emmeans(... , submodel = ~ ...)`
    # numerically for nested submodels).
    # type2/type3 need the target+by factors to derive
    # the reduced model, which aren't parsed yet — defer those two to
    # just after target/by canonicalisation. Formula / "minimal"
    # submodels don't need the target, so apply them now.
    _deferred_type_submodel: str | None = None
    if submodel is not None:
        if isinstance(submodel, str) and submodel.strip().lower() in (
            "minimal", "type2", "type3",
        ):
            # All three derive the reduced model from the primary
            # (target / by) factors, which aren't parsed yet — defer.
            _deferred_type_submodel = submodel
        else:
            info = _apply_submodel(info, submodel)

    # Cox PH advisory: the partial likelihood does not identify the
    # baseline hazard, so the link-scale ``emmean`` column is a
    # relative log-hazard against an arbitrary reference (the row for
    # the patsy-baseline level always shows ``emmean=0, SE=0``). The
    # identifiable views are ``pairs(emm)`` (log-hazard ratios) and
    # ``emmeans(fit, ..., type="response")`` / ``regrid_response(emm)``
    # (hazard ratios). R `emmeans.coxph` does not display the raw
    # link-scale column for the same reason.
    # Match the results class directly OR (for MI-pooled / proxy-wrapped
    # results) the underlying model class accessible via ``raw_result.model``.
    # Hardcoded string match on a single class name silently misses
    # ``MIResults`` / custom result wrappers and the warning never fires.
    def _is_cox_ph(raw_result: Any) -> bool:
        # Module-qualified isinstance check: name-only matching
        # ({"PHRegResults", ...}) collides with any user/library class
        # that happens to share the same short name. Anchoring to
        # ``statsmodels.duration.hazard_regression.{PHReg, PHRegResults,
        # PHRegResultsWrapper}`` eliminates that false-positive while
        # still firing on MI-pooled / proxy-wrapped results whose
        # ``.model`` is the real Cox fit.
        #
        # ``PHRegResultsWrapper`` was added later than ``PHReg`` /
        # ``PHRegResults``; on older statsmodels releases the symbol
        # is absent and a single
        # ``from ... import PHReg, PHRegResults, PHRegResultsWrapper``
        # raises ``ImportError`` for the whole tuple, silently
        # disabling the Cox PH advisory warning. Import each name in
        # isolation so a missing wrapper degrades only the wrapper
        # detection while keeping the bare-class detection intact.
        cox_type_list: list[type] = []
        try:
            from statsmodels.duration.hazard_regression import PHReg
        except ImportError:
            pass
        else:
            cox_type_list.append(PHReg)
        try:
            from statsmodels.duration.hazard_regression import PHRegResults
        except ImportError:
            pass
        else:
            cox_type_list.append(PHRegResults)
        try:
            from statsmodels.duration.hazard_regression import (
                PHRegResultsWrapper,
            )
        except ImportError:
            pass
        else:
            cox_type_list.append(PHRegResultsWrapper)
        if not cox_type_list:
            return False
        cox_types: tuple[type, ...] = tuple(cox_type_list)
        if isinstance(raw_result, cox_types):
            return True
        mdl = getattr(raw_result, "model", None)
        return mdl is not None and isinstance(mdl, cox_types)

    if info.raw_result is not None and type == "link" and _is_cox_ph(info.raw_result):
        import warnings as _w
        _w.warn(
            "Cox PH `emmeans(..., type='link')` returns a relative "
            "log-hazard column: the partial likelihood does not "
            "identify the baseline hazard, so the row for the patsy "
            "reference level shows `emmean=0` by construction. Use "
            "`pairs(emm)` for log-hazard-ratio contrasts (identifiable) "
            "or pass `type='response'` / wrap with `regrid_response(emm)` "
            "for hazard ratios. R `emmeans.coxph` omits this column for "
            "the same reason.",
            UserWarning,
            stacklevel=2,
        )

    # #8: pickling a ModelInfo drops design_info + data, so
    # we cannot build a fresh reference grid from it. Detect early so
    # the user sees a clear error instead of a confusing
    # `AttributeError: 'NoneType' object has no attribute 'terms'`
    # deep inside the grid-construction path.
    if info.design_info is None:
        # distinguish ``emmobj()`` origin
        # (no design_info ever existed) from the pickle-origin
        # case (design_info was dropped during pickling).
        # ``emmobj()`` constructs a ModelInfo with a sentinel
        # ``_emmobj_origin`` column on the data DataFrame; pickle
        # origin preserves the original column structure.
        data_attr = getattr(info, "data", None)
        is_emmobj_origin = (
            data_attr is not None
            and hasattr(data_attr, "columns")
            and "_emmobj_origin" in data_attr.columns
        )
        if is_emmobj_origin:
            raise ValueError(
                "emmeans() cannot build a reference grid from a "
                "ModelInfo constructed via ``emmobj(bhat, V, levels)`` "
                "— ``emmobj`` is a low-level constructor that does "
                "not carry a patsy ``design_info``, which the grid "
                "logic needs. Use ``qdrg(formula, data, coef, "
                "vcov, df)`` instead: it parses a formula via patsy "
                "and produces a fully-functional ModelInfo."
            )
        raise ValueError(
            "This ModelInfo has been pickled and lost its patsy "
            "design_info — emmeans() cannot build a new reference "
            "grid from it. Pickled EMMResult / ContrastResult objects "
            "remain inference-safe (they store the already-built "
            "linfct matrix); pickling a ModelInfo and re-running "
            "emmeans() on it is not supported. Either keep the "
            "original fit alive, or compute and pickle the "
            "EMMResult/ContrastResult directly."
        )
    # R-style formula specs like ``pairwise ~ g``,
    # ``trt.vs.ctrl ~ g``, ``revpairwise ~ g | block``,
    # etc. Compute the EMM, run the named contrast on it, and return
    # an ``EmmList(emmeans=em, contrasts=ct)``. Mirrors R's
    # ``emmeans(fit, pairwise ~ g)`` return shape exactly.
    if isinstance(specs, str) and "~" in specs:
        lhs, rhs = (s.strip() for s in specs.split("~", 1))
        valid_methods = {
            "pairwise", "revpairwise", "tukey",
            "trt.vs.ctrl", "trt.vs.ctrl1", "trt.vs.ctrlk",
            "poly", "consec", "eff", "del.eff",
            "mean_chg", "identity", "helmert",
        }
        if lhs in valid_methods:
            # Parse the RHS: "g" or "g | block" or "g * h | block".
            if "|" in rhs:
                rhs_spec, rhs_by = (s.strip() for s in rhs.split("|", 1))
            else:
                rhs_spec, rhs_by = rhs, None
            # The RHS may contain "*" or "+" — treat both as
            # listing multiple target factors (R semantics).
            rhs_targets = [
                t.strip() for t in rhs_spec.replace("*", "+").split("+")
                if t.strip()
            ]
            inner_by = (
                [t.strip() for t in rhs_by.replace("*", "+").split("+")
                 if t.strip()]
                if rhs_by else None
            )
            em = emmeans(
                model,
                rhs_targets if len(rhs_targets) > 1 else rhs_targets[0],
                by=inner_by if inner_by else by,
                at=at, level=level, type=type, weights=weights,
                chunk_size=chunk_size,
            )
            from pymmeans.contrasts import EmmList
            from pymmeans.contrasts import contrast as _contrast
            ct = _contrast(em, method=lhs)
            return EmmList(emmeans=em, contrasts=ct)

    target = _as_list(specs)
    by_list = _as_list(by)

    type = type.lower()
    if type not in ("link", "response"):
        raise ValueError(
            f"'type' must be 'link' or 'response', got {type!r}."
        )

    if not target:
        raise ValueError("'specs' must name at least one factor.")

    # Allow users to refer to factors by their underlying column name even
    # when the formula wraps them in an expression (e.g. "C(percent)").
    # for multi-column basis terms (``bs(x, df=3)``, etc.),
    # the underlying column name (``x``) IS the canonical user-facing
    # name — applying the alias here would map ``x`` to the patsy
    # expression and then fail the numeric_means lookup.
    multi_col = getattr(info, "multi_col_factors", {}) or {}
    def _maybe_alias(name: str) -> str:
        canonical = info.aliases.get(name, name)
        if canonical in multi_col:
            return name # keep the underlying column name
        return canonical

    target = [_maybe_alias(t) for t in target]
    by_list = [_maybe_alias(b) for b in by_list]

    # Apply the deferred type2/type3 submodel now that the primary
    # (target + by) factors are known. The Type-II reduced model drops
    # interactions mixing a primary factor with a non-primary one;
    # Type-III is the full model (Type-III LS-means = default). See
    # :func:`_type_submodel_rhs`.
    if _deferred_type_submodel is not None:
        info = _apply_submodel(
            info, _deferred_type_submodel,
            primary=set(target) | set(by_list),
        )

    # ``nuisance`` validation. Declarative-only under
    # ``weights='equal'`` (the default); refused under non-equal
    # weight modes because nuisance-overrides-weights semantics
    # need a more invasive implementation than v0.1 ships.
    if nuisance is not None:
        nuisance_list = _as_list(nuisance)
        # Canonicalise via the same alias map used for target / by.
        nuisance_list = [_maybe_alias(n) for n in nuisance_list]
        if not nuisance_list:
            raise ValueError(
                "nuisance= must be a non-empty string or list of "
                "factor names (or None to opt out); got an empty list."
            )
        # Each name must be a categorical factor of the fitted model.
        unknown_nuisance = [
            n for n in nuisance_list if n not in info.factors
        ]
        if unknown_nuisance:
            raise ValueError(
                "nuisance references unknown categorical factors: "
                f"{sorted(unknown_nuisance)}. Known factors: "
                f"{sorted(info.factors)}. Numeric covariates are not "
                "valid nuisance entries — pin them via ``at=`` or "
                "override their reduction via ``cov_reduce=``."
            )
        # A nuisance factor by definition shouldn't also be a target
        # or a by-conditioner; both would contradict the
        # "average-this-away" intent.
        target_overlap = [n for n in nuisance_list if n in target]
        if target_overlap:
            raise ValueError(
                "nuisance overlaps with specs (target): "
                f"{sorted(target_overlap)}. A factor cannot be both "
                "a target of the EMM and a nuisance averaged over; "
                "drop it from one of the two."
            )
        by_overlap = [n for n in nuisance_list if n in by_list]
        if by_overlap:
            raise ValueError(
                "nuisance overlaps with by: "
                f"{sorted(by_overlap)}. A by-conditioner cannot also "
                "be marked nuisance; drop it from one of the two."
            )
        # nuisance × weights interaction. The kernel now overrides the
        # per-factor weight construction for `weights='outer'` and
        # `weights='proportional'` so that nuisance factors are forced
        # to equal-marginal averaging while the remaining non-target
        # factors retain the requested weighting scheme — matching R's
        # `wt.nuis='equal'` (the default in `ref_grid(..., nuisance=)`).
        # `weights='cells'` remains refused: its observed-cell aggregation
        # path needs nuisance-specific pre-collapse logic that is
        # genuinely more invasive than the marginal/joint reweighting
        # used by outer/proportional. Workaround: weights='proportional'
        # gives the same answer when the design is balanced over the
        # nuisance factor(s).
        _nuisance_weights = (weights or "equal").lower()
        if _nuisance_weights == "flat":
            _nuisance_weights = "equal"
        # weights='cells' × nuisance= is now supported in
        # _marginalize_cells via analytical-extrapolation extrapolation
        # at each (target, nuisance, non-nuisance non-target) grid combo
        # — matches R `wt.nuis='equal'` including empty sub-cells.
        # Under weights='equal' / 'flat' the underlying math is unchanged
        # — non-target factors are already averaged uniformly so nuisance
        # is declarative only (catches typos, documents intent). Under
        # 'outer' / 'proportional' the construction below applies the
        # override.

    # ``counterfactuals`` validation + merge into ``at``. Mirrors R
    # ``ref_grid(..., counterfactuals=...)`` for the
    # counterfactual-prediction workflow: every key is a column the
    # caller wants pinned at specific intervention values for the
    # grid, semantically equivalent to ``at=`` but with stricter
    # validation that the named columns aren't targets / by-
    # conditioners (those vary across grid rows, not pin to a CF).
    if counterfactuals is not None:
        if not isinstance(counterfactuals, dict):
            # Note: ``type`` is shadowed by the ``type`` parameter of
            # ``emmeans()`` (the link/response scale kwarg), so we
            # spell the class lookup via ``__class__`` to avoid
            # calling the parameter as a function.
            raise TypeError(
                "counterfactuals= must be a dict[column_name -> "
                f"scalar | list]; got {counterfactuals.__class__.__name__}."
            )
        if not counterfactuals:
            raise ValueError(
                "counterfactuals= must contain at least one entry "
                "(or be None to opt out); got an empty dict."
            )
        # Canonicalise keys via the same alias map used for ``at``.
        cf_canonical: dict[str, Any] = {}
        for raw_key, raw_val in counterfactuals.items():
            if not isinstance(raw_key, str):
                # Same ``type`` parameter shadow as the dict check
                # above — spell the class lookup via ``__class__``.
                raise TypeError(
                    "counterfactuals= keys must be strings; got "
                    f"non-string key of type {raw_key.__class__.__name__}."
                )
            canon = info.aliases.get(raw_key, raw_key)
            cf_canonical[canon] = raw_val
        # Every key must reference a known model variable (factor or
        # numeric covariate). Reuses the same membership rules
        # ``build_grid_spec`` applies to ``at``, but surfaced earlier
        # so the error message names the kwarg the user actually
        # passed.
        known = set(info.factors) | set(info.numeric_means)
        unknown_cf = [k for k in cf_canonical if k not in known]
        if unknown_cf:
            raise ValueError(
                "counterfactuals= references unknown columns: "
                f"{sorted(unknown_cf)}. Known factors / covariates: "
                f"{sorted(known)}."
            )
        # No key may be a target — a counterfactual *pins* a column,
        # whereas a target *varies* across grid rows. Both at once is
        # contradictory.
        target_cf_overlap = [k for k in cf_canonical if k in target]
        if target_cf_overlap:
            raise ValueError(
                "counterfactuals= overlaps with specs (target): "
                f"{sorted(target_cf_overlap)}. A target factor varies "
                "across the EMM result rows; a counterfactual pins a "
                "value. Drop the column from one of the two — pass "
                "the counterfactual values via ``at=`` if you wanted "
                "them as the target's grid points."
            )
        # No key may be a by-conditioner — same logic as target.
        by_cf_overlap = [k for k in cf_canonical if k in by_list]
        if by_cf_overlap:
            raise ValueError(
                "counterfactuals= overlaps with by: "
                f"{sorted(by_cf_overlap)}. A by-conditioner varies "
                "across grid panels; a counterfactual pins a value. "
                "Drop the column from one of the two."
            )
        # No key may collide with ``at=`` — that would double-specify
        # the column. Refuse rather than silently picking one source.
        if at is not None:
            at_keys_for_cf = {info.aliases.get(k, k) for k in at}
            at_cf_overlap = [k for k in cf_canonical if k in at_keys_for_cf]
            if at_cf_overlap:
                raise ValueError(
                    "counterfactuals= overlaps with at=: "
                    f"{sorted(at_cf_overlap)}. Specify each column "
                    "via exactly one of the two kwargs — "
                    "counterfactuals= for the documented-intent "
                    "counterfactual workflow, or at= for general "
                    "grid pinning."
                )
        # Merge into ``at``. From here on the grid build sees the
        # combined map; counterfactual columns flow through the same
        # ``build_grid_spec`` validation (finite-numerics check,
        # factor-levels check) as everything else.
        at = {**(at or {}), **cf_canonical}

    # ``nesting`` validation. The full structural validation lives
    # in :func:`pymmeans.ref_grid._validate_nesting`; here we add
    # the emmeans-specific guards (weight-mode interaction,
    # post-pickle data check) before delegating the rest. When
    # nesting is supplied we ALSO force the eager (grid-
    # materialised) path because the Kronecker analytic path
    # assumes factor independence — nesting breaks that and would
    # average over phantom (nested, enclosing) tuples that don't
    # appear in the data.
    # Resolve nesting="auto" → detected dict (or None) before the
    # validation / weight-mode guards below see it.
    from pymmeans.ref_grid import _resolve_auto_nesting as _ran
    nesting = _ran(info, nesting)
    if nesting is not None:
        from pymmeans.ref_grid import _validate_nesting as _vn
        nesting_norm_validated = _vn(info, nesting)
        if info.data is None or len(info.data) == 0:
            raise ValueError(
                "nesting= requires the training data on "
                "``info.data`` to discover the valid (nested, "
                "enclosing) tuples, but ``info.data`` is None or "
                "empty (typically because the ModelInfo / EMM was "
                "pickled — pickling drops the training data). "
                "Fixes:\n"
                "  - Call emmeans(fit, ..., nesting=...) on a fresh "
                "fit in the current process.\n"
                "  - Drop nesting= and pin the valid combinations "
                "explicitly via ``at={...}`` (one entry per "
                "factor)."
            )
        _nesting_weights = (weights or "equal").lower()
        if _nesting_weights == "flat":
            _nesting_weights = "equal"
        if _nesting_weights != "equal":
            raise ValueError(
                f"nesting= is not yet supported with "
                f"weights={weights!r}. Under non-equal weight modes "
                "the v0.1 filter-only implementation can't produce "
                "R-correct per-cell frequencies on the nested "
                "factors. Two workarounds:\n"
                "  - Drop nesting= and accept the non-equal "
                "weights (each (nested, enclosing) tuple gets its "
                "raw training-data weight; phantom cells will be "
                "marked NaN by the estimability check).\n"
                "  - Drop weights=, use the default equal weights, "
                "and keep nesting= for the R-style grid filter.\n"
                "Full nesting+non-equal-weights is a 0.2.0 "
                "candidate."
            )
        # Surface a flag the path-choice block below reads to force
        # the eager grid-materialised marginalization.
        _nesting_force_eager = True
        # Stash the normalized nesting for the eager fallback's
        # ``ref_grid`` call below to pick up (the eager fallback
        # calls ``_ref_grid(info, at=at)`` and we want the same
        # filtered grid).
        _nesting_norm = nesting_norm_validated
    else:
        _nesting_force_eager = False
        _nesting_norm = None

    # cov_keep covariates are retained as grouping
    # dimensions (one EMM row per kept value), not averaged. Force the
    # eager grid-materialised path: the analytic Kronecker path assumes
    # factor independence, which is fragile for a numeric covariate
    # kept at multiple values under an interaction term (e.g.
    # ``g * dose``). The eager path materialises the grid and groups
    # correctly. Normalise to a list for the group_cols append below.
    _cov_keep_names: list[str] = []
    if cov_keep is not None:
        _cov_keep_names = (
            [cov_keep] if isinstance(cov_keep, str) else list(cov_keep)
        )
        if _cov_keep_names:
            _nesting_force_eager = True

    # GLMGam smoother models must use the eager
    # path: the analytic Kronecker marginalization assumes a pure
    # patsy factorial design and has no way to append the spline
    # basis. The eager fallback routes through ref_grid(), which
    # re-evaluates the smoother at each grid point.
    if getattr(info, "smoother_info", None) is not None:
        _nesting_force_eager = True

    # target / by may be a categorical factor (default) or a
    # numeric covariate when `at={...}` provides explicit values for it
    # (matches R `emmeans(model, ~x, at=list(x=c(-1,0,1)))`). Without
    # at=, a numeric target collapses to a single row at the mean which
    # is identical to `emmeans()` with no target — refuse to avoid
    # silently-confusing one-row output.
    at_keys = set((at or {}).keys())
    # Alias-aware: a user-supplied raw column name maps to the canonical.
    at_keys_canonical = {info.aliases.get(k, k) for k in at_keys}
    at_keys_all = at_keys | at_keys_canonical
    for name in target + by_list:
        if name in info.factors:
            continue
        if name in info.numeric_means:
            if name not in at_keys_all:
                raise ValueError(
                    f"'{name}' is a numeric covariate; using it as a "
                    "target / by requires `at={...}` with explicit "
                    f"values (e.g. `at={{'{name}': [-1, 0, 1]}}`). "
                    f"Without at=, a numeric target collapses to a "
                    f"single row at the training-data mean. For the "
                    "slope of the response w.r.t. a covariate, use "
                    f"`emtrends(info, ..., var='{name}')` instead."
                )
            continue
        # when the user passes a canonical
        # multi-col-basis name as target (e.g. ``"bs(x, df=3)"``),
        # point them at the underlying column they should use.
        if name in (multi_col or {}):
            underlying = (multi_col or {})[name]
            suggestion = (
                underlying[0] if len(underlying) == 1
                else f"one of {underlying!r}"
            )
            raise ValueError(
                f"'{name}' is a multi-column basis expression, not a "
                "target you can specify directly. Pass the underlying "
                f"covariate as the target instead: ``target={suggestion!r}``"
                f" (with ``at={{'{suggestion}': [values]}}`` to enumerate)."
            )
        raise ValueError(
            f"'{name}' is not a factor or numeric covariate in the model. "
            f"Known factors: {sorted(info.factors)}; known numerics: "
            f"{sorted(info.numeric_means)}."
        )

    overlap = set(target) & set(by_list)
    if overlap:
        raise ValueError(
            f"Variables {sorted(overlap)} appear in both specs and by."
        )

    group_cols = target + by_list
    # cov_keep covariates become retained grouping dimensions (one row
    # per kept value), appended after target + by. They're validated
    # numeric covariates with explicit grid values in `spec`, so they
    # bypass the "numeric target needs at=" guard above (cov_keep IS
    # the explicit opt-in to grid a covariate). Overlap with target /
    # by is refused — a covariate can't be both the EMM target and a
    # kept grouping dimension.
    if _cov_keep_names:
        keep_overlap = [c for c in _cov_keep_names if c in group_cols]
        if keep_overlap:
            raise ValueError(
                f"cov_keep names {sorted(keep_overlap)} also appear in "
                "specs / by. A covariate can't be both an EMM target / "
                "by-conditioner and a kept grouping dimension; drop it "
                "from one."
            )
        group_cols = group_cols + [
            c for c in _cov_keep_names if c not in group_cols
        ]
    # ``nesting`` breaks the Kronecker factor-independence assumption
    # of the analytic path (one factor's valid levels depend on
    # another's), so force the eager grid-materialised
    # marginalization. Set BEFORE ``use_analytic`` is read by
    # downstream logic.
    use_analytic = (chunk_size is None) and not _nesting_force_eager
    # `weights='cells'` walks the actual
    # training rows via `_marginalize_cells`; it doesn't enumerate the
    # spec's cartesian product. The "require plain identifiers" guard
    # is a streaming-path restriction that doesn't apply to cells.
    # Suppress it so the cells transformed-`at=` workflow keeps working
    # when the user also passes `chunk_size=`.
    weights_lc_lookahead = (weights or "equal").lower()
    require_plain = (not use_analytic) and weights_lc_lookahead != "cells"
    spec = build_grid_spec(
        info, at,
        require_plain_identifiers=require_plain,
        cov_reduce=cov_reduce,
        cov_keep=cov_keep,
    )

    # Build per-factor weights from training data when requested.
    # Semantics match R emmeans:
    # - 'equal': uniform 1/k_levels (Kronecker; default)
    # - 'outer': product of per-factor marginal frequencies (Kronecker)
    # - 'proportional': joint cell frequencies of non-target factors
    # (does NOT decompose; requires per-combination enumeration)
    # - 'cells': joint with target included; walks the training data
    weights_lc = weights.lower()
    # R `lsmeans` / `emmeans` accept two extra string options that pymmeans
    # didn't recognise before (80%-parity push):
    #
    # * `weights='flat'` is functionally an alias for `weights='equal'`
    # in the common case of full-rank designs. R semantics: equal
    # weight on every CELL with positive observed frequency; pymmeans's
    # `equal` Kronecker path already gives each existing level equal
    # weight, and the estimability check correctly marks
    # non-estimable (zero-observation) cells as NaN. For balanced /
    # full-rank designs they're mathematically identical; for severely
    # rank-deficient designs, R's `flat` would give weight 1 to
    # present cells and 0 to empty ones — pymmeans's `equal` + NaN
    # filter accomplishes the same statistical conclusion. Map to
    # `equal` and proceed.
    #
    # * `weights='show.levels'` does NOT compute marginal means; it
    # short-circuits the marginalization and returns a summary of the
    # reference grid. R returns the un-marginalized grid; pymmeans
    # hands back a `RefGrid` object the same way `ref_grid()` does.
    # Callers wanting the grid as a frame can read `result.grid`.
    if weights_lc == "flat":
        weights_lc = "equal"
    # Import the public ``ref_grid`` once; both the ``show.levels``
    # short-circuit AND the eager fallback below need it. The
    # previous code imported lazily inside the ``show.levels`` branch
    # only, so the eager fallback at the bottom of this function
    # raised ``UnboundLocalError`` when neither the analytic nor
    # the streaming path applied (e.g., when ``nesting=`` forces
    # the eager fallback at any data size).
    from pymmeans.ref_grid import ref_grid as _ref_grid

    if weights_lc == "show.levels":
        # Special case: return the un-marginalized reference grid as a
        # RefGrid (matches R `emmeans(..., weights='show.levels')`,
        # which prints the grid and returns it unchanged rather than
        # producing an emmGrid of marginal means).
        return _ref_grid(info, at=at, nesting=nesting)
    if weights_lc not in ("equal", "proportional", "outer", "cells"):
        raise ValueError(
            f"weights must be 'equal' / 'proportional' / 'outer' / 'cells' "
            f"(plus R aliases 'flat' and 'show.levels'); got {weights!r}."
        )
    # `weights='cells'` is handled via a dedicated path that
    # aggregates the design matrix per observed (target, by) cell, with
    # at= overrides applied row-by-row. Implementation lives in
    # _marginalize_cells; we jump there after the spec is built.
    # (Sentinel handled inline below.)

    canonical_to_raw = {v: k for k, v in info.aliases.items()}

    # Use fit weights when present so 'outer'/'proportional' match R's
    # weighted ``emmeans``. #2: unweighted value_counts ignored
    # WLS / freq_weights and gave wildly wrong EMMs on weighted fits.
    fw = info.fit_weights
    if fw is not None and len(fw) != len(info.data):
        fw = None # length mismatch (post-pickle or filtered frame); skip

    factor_weights: dict[str, np.ndarray] | None = None
    if weights_lc == "outer":
        # Per-factor marginal frequency table (matches R 'outer' under the
        # analytic Kronecker structure).
        factor_weights = {}
        for fname, levels in info.factors.items():
            source_col = canonical_to_raw.get(fname, fname)
            if source_col in info.data.columns:
                if fw is not None:
                    # Weighted marginal: sum of fit weights per level.
                    col = info.data[source_col].to_numpy()
                    w = np.array(
                        [float(fw[col == lv].sum()) for lv in levels],
                        dtype=float,
                    )
                else:
                    counts = info.data[source_col].value_counts()
                    w = np.array(
                        [float(counts.get(lv, 0)) for lv in levels],
                        dtype=float,
                    )
                if w.sum() > 0:
                    factor_weights[fname] = w
        # nuisance override: force equal marginals on the named factor(s)
        # so the per-row product in `_per_row_weight` normalises against a
        # uniform vector regardless of the training-data marginals. Mirrors
        # R `wt.nuis='equal'` under weights='outer'.
        if nuisance is not None:
            for nf in nuisance_list:
                if nf in info.factors:
                    factor_weights[nf] = np.ones(len(info.factors[nf]))

    proportional_joint = None
    if weights_lc == "proportional":
        non_target_factor_names = [
            c for c in info.factors if c not in target and c not in by_list
        ]
        non_target_src = [
            canonical_to_raw.get(c, c) for c in non_target_factor_names
        ]
        valid = [
            (n, s)
            for n, s in zip(non_target_factor_names, non_target_src, strict=True)
            if s in info.data.columns
        ]
        if valid:
            _, cols = zip(*valid, strict=True)
            if fw is not None:
                # Weighted joint: group by the non-target factor combo and
                # sum fit weights within each cell, then normalise.
                sub = info.data[list(cols)].assign(_w=fw)
                grouped = sub.groupby(list(cols), observed=True, sort=False)["_w"].sum()
                total = float(grouped.sum())
                if total > 0:
                    proportional_joint = {
                        (k if isinstance(k, tuple) else (k,)): float(v) / total
                        for k, v in grouped.items()
                    }
                else:
                    proportional_joint = {}
            else:
                counts = info.data[list(cols)].value_counts(normalize=True)
                # pandas already returns tuple keys for DataFrame.value_counts,
                # so we just round-trip through tuple() defensively.
                proportional_joint = {
                    tuple(k): float(v) for k, v in counts.items()
                }
        else:
            # No non-target factors to weight over
            proportional_joint = {(): 1.0}
        # nuisance override (proportional path): marginalise the joint over
        # the nuisance dimensions and re-expand uniformly over them. This
        # forces equal-marginal averaging on the nuisance factor(s) while
        # the remaining non-target factors keep their data-marginal weights
        # — matching R `wt.nuis='equal'` with non-equal `weights=`.
        if nuisance is not None and proportional_joint:
            _nset = set(nuisance_list)
            _nt_pos_nuis = [
                i for i, c in enumerate(non_target_factor_names) if c in _nset
            ]
            if _nt_pos_nuis:
                _nt_pos_pure = [
                    i for i, c in enumerate(non_target_factor_names)
                    if c not in _nset
                ]
                _nuis_names_nt = [
                    non_target_factor_names[i] for i in _nt_pos_nuis
                ]
                _nuis_levels_nt = [
                    list(info.factors[nf]) for nf in _nuis_names_nt
                ]
                _nuis_count = 1
                for _lvls in _nuis_levels_nt:
                    _nuis_count *= max(len(_lvls), 1)
                _pos_pure = set(_nt_pos_pure)
                _pure_joint: dict[tuple, float] = {}
                for _k, _v in proportional_joint.items():
                    _pure_key = tuple(_k[i] for i in _nt_pos_pure)
                    _pure_joint[_pure_key] = _pure_joint.get(_pure_key, 0.0) + _v
                _expanded: dict[tuple, float] = {}
                for _pure_k, _pure_w in _pure_joint.items():
                    for _nuis_combo in itertools.product(*_nuis_levels_nt):
                        _full = [None] * len(non_target_factor_names)
                        _pi = iter(_pure_k)
                        _ni = iter(_nuis_combo)
                        for _i in range(len(non_target_factor_names)):
                            if _i in _pos_pure:
                                _full[_i] = next(_pi)
                            else:
                                _full[_i] = next(_ni)
                        _expanded[tuple(_full)] = _pure_w / _nuis_count
                proportional_joint = _expanded
    total_rows = grid_size(spec)
    full_l_bytes = total_rows * info.n_params * 8
    should_stream = (
        not use_analytic and full_l_bytes > _STREAM_MEMORY_THRESHOLD_BYTES
    )

    unique_keys: pd.MultiIndex | list[tuple]
    if weights_lc == "cells":
        # 'cells' uses observed cell frequencies including the target —
        # walks the actual data rows rather than the cartesian product.
        _nuis_for_cells = (
            list(nuisance_list) if nuisance is not None else []
        )
        L_marg, key_list = _marginalize_cells(
            info, spec, group_cols, at, fw, nuisance_cols=_nuis_for_cells,
        )
        unique_keys = pd.MultiIndex.from_tuples(key_list, names=group_cols)
    elif use_analytic:
        if weights_lc == "proportional":
            from pymmeans.analytic import analytic_marginalize_proportional
            L_marg, key_list = analytic_marginalize_proportional(
                info, spec, group_cols, proportional_joint
            )
        else:
            from pymmeans.analytic import analytic_marginalize
            L_marg, key_list = analytic_marginalize(
                info, spec, group_cols, factor_weights=factor_weights
            )
        unique_keys = pd.MultiIndex.from_tuples(key_list, names=group_cols)
    elif (should_stream or chunk_size is not None) and not _nesting_force_eager:
        # The streaming path materialises the cartesian product in
        # chunks and never sees ``nesting=`` — it would silently
        # average over phantom (nested, enclosing) tuples that
        # don't appear in the training data. When nesting is on,
        # skip streaming and fall through to the eager fallback,
        # which applies the filter via :func:`_apply_nesting_filter`
        # before averaging.
        L_marg, key_list = _marginalize_streamed(
            info,
            spec,
            group_cols,
            chunk_size if chunk_size is not None else _STREAM_CHUNK_ROWS_DEFAULT,
            weights_mode=weights_lc,
            factor_weights=factor_weights,
            proportional_joint=proportional_joint,
        )
        unique_keys = pd.MultiIndex.from_tuples(key_list, names=group_cols)
    else:
        # Pass ``nesting`` through so the eager fallback filters the
        # cartesian grid to valid (nested, enclosing) tuples before
        # averaging — under ``weights='equal'`` this is exactly the
        # R-style nesting EMM.
        rg = _ref_grid(info, at=at, nesting=nesting, cov_keep=cov_keep)
        keys = pd.MultiIndex.from_frame(rg.grid[group_cols].astype(object))
        unique_keys = keys.unique()
        L_marg = np.empty((len(unique_keys), info.n_params))
        for i, key in enumerate(unique_keys):
            mask = keys == key
            L_marg[i] = rg.linfct[mask].mean(axis=0)

    # Estimability check for rank-deficient designs. For full-rank X the
    # mask is all-True and this is fast; we always run it so non-estimable
    # rows surface as NaN with a clear warning instead of silently nonsense.
    # Priority: precomputed basis (always available post-adapter
    # call, signals known rank) > live X SVD > assume estimable.
    #
    # When the adapter built ModelInfo it ran an SVD on X to decide
    # whether to store a null-space basis (#5). If that SVD
    # found full column rank, `estimability_basis is None` and every
    # contrast row is estimable -- no need to redo the SVD here. This
    # is the dominant cost (~12ms on a 20k x 50 design after 's
    # full_matrices=False; ~3.2s before).
    from pymmeans.estimability import estimable_mask, estimable_mask_from_basis

    X_design = None
    if info.raw_result is not None and hasattr(info.raw_result, "model"):
        X_design = np.asarray(getattr(info.raw_result.model, "exog", None))
    has_basis_field = hasattr(info, "estimability_basis")
    full_rank_known = has_basis_field and info.estimability_basis is None
    X_usable = (
        X_design is not None
        and X_design.ndim == 2
        and X_design.shape[1] == info.n_params
    )
    if full_rank_known and X_usable:
        # Adapter already verified full rank; skip the second SVD.
        estimable = np.ones(L_marg.shape[0], dtype=bool)
    elif X_usable:
        estimable = estimable_mask(L_marg, X_design)
    elif info.estimability_basis is not None:
        estimable = estimable_mask_from_basis(L_marg, info.estimability_basis)
    else:
        estimable = np.ones(L_marg.shape[0], dtype=bool)
    if not estimable.all():
        import warnings

        warnings.warn(
            f"{int((~estimable).sum())} of {len(estimable)} EMM rows "
            "are not estimable under the model's design (rank deficiency); "
            "marking them NaN.",
            UserWarning,
            stacklevel=2,
        )

    mu = L_marg @ info.beta
    # Offset (GLM `offset=` / Poisson `exposure=`): R `emmeans` honours
    # `offset()` by adding the *mean* of the offset vector to eta before
    # the inverse link. We materialised that mean at adapter time
    # (`info.offset_mean`); 0.0 for non-GLM or unoffset fits is a no-op.
    if info.offset_mean:
        mu = mu + info.offset_mean
    var = np.einsum("ij,jk,ik->i", L_marg, info.vcov, L_marg)
    se = np.sqrt(np.clip(var, 0.0, None))
    if not estimable.all():
        mu = np.where(estimable, mu, np.nan)
        se = np.where(estimable, se, np.nan)

    df_value: float = (
        np.inf if (info.family is not None or info.is_mixed) else info.df_resid
    )
    df_arr = np.full(len(unique_keys), df_value)

    alpha = 1.0 - level
    crit = float(stats.t.ppf(1.0 - alpha / 2.0, df_value))
    lower = mu - crit * se
    upper = mu + crit * se

    if type == "response":
        # Two response-scale paths:
        # 1. GLM family present → use link.inverse / link.inverse_deriv
        # 2. No family but detect_transform sees a recognised LHS
        # expression in response_name → use Transform.inverse /
        # inverse_deriv.
        #
        # Before , only path 1 fired. Path 2 was silently a
        # no-op for OLS fits with LHS transforms like `np.log(y) ~ ...`,
        # so `emmeans(..., type='response')` returned LINK-scale values
        # stamped with `type='response'` — a real silent-wrong bug.
        # Now we route through whichever path applies; users who want
        # the bias-adjusted Taylor-corrected response-scale mean keep
        # using `regrid_response(em, bias_adjust=True)` as before.
        applied_response = False
        if info.family is not None:
            link = info.family.link
            mu_resp = link.inverse(mu)
            se_resp = np.abs(link.inverse_deriv(mu)) * se
            lo_resp = link.inverse(lower)
            up_resp = link.inverse(upper)
            mu = mu_resp
            se = se_resp
            lower = np.minimum(lo_resp, up_resp)
            upper = np.maximum(lo_resp, up_resp)
            applied_response = True
        else:
            from pymmeans.transforms import (
                _interval_inverse as _iv_inv,
            )
            from pymmeans.transforms import (
                detect_transform as _detect,
            )

            tran = _detect(info.response_name or "")
            if tran is not None:
                mu_resp = tran.inverse(mu)
                se_resp = np.abs(tran.inverse_deriv(mu)) * se
                # use _interval_inverse so
                # non-monotone inverses (sqrt, sqrt_const) handle a
                # CI-crossing-zero correctly. For monotone inverses
                # this is just min/max of the two endpoints.
                lower, upper = _iv_inv(tran, lower, upper)
                mu = mu_resp
                se = se_resp
                applied_response = True
        if not applied_response:
            # the user wrote
            # `np.log(y + 1)` (or similar composite LHS), detect_transform
            # refuses composite inner expressions, and the
            # earlier warning kept returning link-scale values
            # stamped as `type='response'`. That's still silent-wrong:
            # downstream consumers (`pairs`, `cld`, `pwpp`,
            # `bootstrap_ci`) read `.type` and trust it. Refuse cleanly
            # for any response_name that LOOKS like a function call
            # (contains `(` but didn't match `detect_transform`); plain
            # identity OLS (`y ~ a`) silently returns link-scale values
            # because in that case link == response by definition.
            response_name = info.response_name or ""
            if "(" in response_name:
                raise ValueError(
                    f"emmeans(..., type='response') with response_name="
                    f"{response_name!r} cannot be auto-back-transformed "
                    "because the inner expression is composite. Pass an "
                    "explicit transform via `regrid_response(em, "
                    "tran=make_tran('genlog', base=1))` for "
                    "`log(y + 1)`-style fits, or `make_tran(name, "
                    "inverse=, inverse_deriv=)` for arbitrary LHS "
                    "expressions."
                )
            # Plain identity LHS (e.g. `y ~ a`): link == response, so
            # silently returning link-scale values stamped as response
            # is mathematically correct. Emit a soft note for the
            # warning-as-error users; warnings module dedups by source
            # location so the same call site won't spam.
            import warnings as _w
            _w.warn(
                "emmeans(..., type='response') on a plain-identity LHS "
                f"({response_name!r}) is a no-op: link == response. "
                "Pass type='link' (the default) to suppress this note.",
                UserWarning, stacklevel=2,
            )

    frame = pd.DataFrame(
        {col: [k[i] for k in unique_keys] for i, col in enumerate(group_cols)}
    )
    for name in group_cols:
        # only convert categorical-factor columns to ordered
        # pd.Categorical; numeric covariate targets (used with at=)
        # stay as their native dtype so downstream tooling sees floats.
        if name not in info.factors:
            continue
        levels = info.factors[name]
        present = [lv for lv in levels if lv in set(frame[name])]
        frame[name] = pd.Categorical(frame[name], categories=present)
    frame["emmean"] = mu
    frame["SE"] = se
    frame["df"] = df_arr
    frame["lower_cl"] = lower
    frame["upper_cl"] = upper

    sort_cols = by_list + target
    if sort_cols:
        frame = frame.sort_values(sort_cols, kind="stable").reset_index(drop=True)
        key_index = unique_keys.tolist()
        order = [
            key_index.index(tuple(row))
            for row in frame[group_cols].itertuples(index=False, name=None)
        ]
        L_marg = L_marg[order]

    return EMMResult(
        frame=frame,
        linfct=L_marg,
        model_info=info,
        target=target,
        by=by_list,
        level=level,
        type=type,
        at=dict(at) if at is not None else None,
        weights=weights if weights is not None else "equal",
    )
