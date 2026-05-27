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

    weight_sum = np.zeros(len(key_list))
    for row_idx, key in enumerate(keys_arr):
        ki = key_index.get(key)
        if ki is None:
            continue
        w = 1.0 if fw is None else float(fw[row_idx])
        L_marg[ki] += w * X[row_idx]
        weight_sum[ki] += w
    # Average within each cell. Empty cells -> NaN row, surfaces via
    # estimability check downstream.
    empty = weight_sum == 0
    with np.errstate(invalid="ignore", divide="ignore"):
        L_marg[~empty] = L_marg[~empty] / weight_sum[~empty, None]
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
        try:
            from statsmodels.duration.hazard_regression import (
                PHReg,
                PHRegResults,
                PHRegResultsWrapper,
            )
        except ImportError:
            return False
        cox_types: tuple[type, ...] = (PHReg, PHRegResults, PHRegResultsWrapper)
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
    use_analytic = chunk_size is None
    # `weights='cells'` walks the actual
    # training rows via `_marginalize_cells`; it doesn't enumerate the
    # spec's cartesian product. The "require plain identifiers" guard
    # is a streaming-path restriction that doesn't apply to cells.
    # Suppress it so the cells transformed-`at=` workflow keeps working
    # when the user also passes `chunk_size=`.
    weights_lc_lookahead = (weights or "equal").lower()
    require_plain = (not use_analytic) and weights_lc_lookahead != "cells"
    spec = build_grid_spec(
        info, at, require_plain_identifiers=require_plain
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
    if weights_lc == "show.levels":
        # Special case: return the un-marginalized reference grid as a
        # RefGrid (matches R `emmeans(..., weights='show.levels')`,
        # which prints the grid and returns it unchanged rather than
        # producing an emmGrid of marginal means).
        from pymmeans.ref_grid import ref_grid as _ref_grid
        return _ref_grid(info, at=at)
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
    total_rows = grid_size(spec)
    full_l_bytes = total_rows * info.n_params * 8
    should_stream = (
        not use_analytic and full_l_bytes > _STREAM_MEMORY_THRESHOLD_BYTES
    )

    unique_keys: pd.MultiIndex | list[tuple]
    if weights_lc == "cells":
        # 'cells' uses observed cell frequencies including the target —
        # walks the actual data rows rather than the cartesian product.
        L_marg, key_list = _marginalize_cells(info, spec, group_cols, at, fw)
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
    elif should_stream or chunk_size is not None:
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
        rg = _ref_grid(info, at=at)
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
