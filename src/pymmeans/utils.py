"""Model extraction helpers.

Pull beta, vcov, design info, and factor metadata out of fitted statsmodels
results into a single ``ModelInfo`` container so the rest of pymmeans can stay
focused on the EMM math.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


def detect_value_column(frame: pd.DataFrame) -> tuple[str, str] | None:
    """Return ``(kind, column_name)`` for a pymmeans result frame.

    The kind tag identifies the structural shape of the result so
    downstream code can dispatch without inspecting column names directly
    (we found repeatedly fragile —
    emtrends uses ``<var>.trend``, regridded log-family contrasts use
    ``ratio``, ordinary EMMs use ``emmean``, etc.). Returns ``None`` if
    the frame doesn't look like a pymmeans result.

    Kinds:

    - ``"emm"`` — EMMResult on link or response scale; column ``emmean``.
    - ``"contrast"`` — ContrastResult on link scale; column ``estimate``.
    - ``"ratio"`` — ContrastResult after ``regrid_response`` on a
      log-family transform; column ``ratio``.
    - ``"trend"`` — emtrends result; column ``<var>.trend``.
    """
    cols = list(frame.columns)
    if "emmean" in cols:
        return ("emm", "emmean")
    if "ratio" in cols:
        return ("ratio", "ratio")
    if "estimate" in cols:
        return ("contrast", "estimate")
    for c in cols:
        if isinstance(c, str) and c.endswith(".trend"):
            return ("trend", c)
    return None


def _underlying_columns(code: str, data_cols: set[str]) -> list[str]:
    """Return the data columns that appear inside a patsy factor expression.

    For ``code='percent'`` returns ``['percent']``.
    For ``code='C(percent)'`` returns ``['percent']``.
    For ``code='np.log(x + 1)'`` returns ``['x']``.
    For ``code='x + y'`` returns ``['x', 'y']`` (ambiguous; caller decides).

    when the user has a data column whose
    name collides with a patsy basis function name (e.g. a column
    literally named ``'bs'`` and a formula ``bs(x, df=3)``), the
    bare ``ast.Name`` walker would pick up ``bs`` as an underlying
    column. We now exclude the *head identifier* of each
    ``ast.Call`` node (e.g. the ``bs`` in ``bs(x, df=3)``) so
    only the function arguments contribute to the underlying-column
    set. Verified: ``_underlying_columns("bs(x, df=3)", {"bs", "x"})``
    returns ``["x"]`` (previously ``["bs", "x"]``).
    """
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError:
        return [code] if code in data_cols else []
    # Collect the Name nodes that are *call heads* — exclude them
    # from the underlying-column candidate set.
    call_head_ids: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            head = node.func
            # Unwrap attribute chains (``np.log`` → exclude ``np``).
            while isinstance(head, ast.Attribute):
                head = head.value
            if isinstance(head, ast.Name):
                call_head_ids.add(head.id)
    names = {
        n.id for n in ast.walk(tree)
        if isinstance(n, ast.Name) and n.id not in call_head_ids
    }
    return sorted(names & data_cols)


def _extract_offset(result: Any, model: Any) -> float:
    """Return the offset contribution to eta at the reference grid.

    statsmodels keeps ``offset=`` and ``exposure=`` on **separate**
    attributes of the model object:

    - ``model.offset`` stores the user-supplied offset vector verbatim
      (typically already on the link scale, e.g. ``offset=log(t)``).
      #1 fixed the previous assumption that exposure was
      folded into offset.
    - ``model.exposure`` stores ``log(exposure)`` (statsmodels logs it
      on entry). The user supplied raw exposure values.

    R ``emmeans`` averages the **underlying** column of an ``offset()``
    expression: for ``offset(log_t)`` it uses ``mean(log_t)``, for
    ``offset(log(t))`` it uses ``log(mean(t))``. That mirrors
    statsmodels' split: ``offset=`` corresponds to the pre-transformed
    case (use ``mean(model.offset)``), and ``exposure=`` corresponds to
    the in-formula log case (use ``log(mean(exp(model.exposure)))``,
    which by Jensen's inequality differs from ``mean(model.exposure)``).
    Both contribute additively to eta.

    Returns 0.0 when neither offset nor exposure was supplied.
    """
    if model is None:
        return 0.0
    total = 0.0
    offset = getattr(model, "offset", None)
    if offset is not None:
        arr = np.asarray(offset, dtype=float)
        if arr.size:
            total += float(arr.mean())
    exposure = getattr(model, "exposure", None)
    if exposure is not None:
        arr = np.asarray(exposure, dtype=float)
        if arr.size:
            # `model.exposure` is log(raw_exposure); recover raw before
            # averaging to match R's `log(mean(exposure))` convention.
            total += float(np.log(np.mean(np.exp(arr))))
    return total


def _extract_fit_weights(result: Any, model: Any) -> np.ndarray | None:
    """Return per-observation fit weights, or None for unweighted fits.

    Priority order:
    1. ``model.freq_weights`` (GLM frequency weights — these are the
       analog of R's ``weights`` argument for `glm`).
    2. ``model.var_weights`` (GLM variance weights).
    3. ``model.weights`` (WLS / statsmodels weighted OLS).

    Returns a copy as a 1-D ndarray. We materialise these once at adapter
    time because the underlying model can be discarded (e.g. after pickle)
    and the weighting logic in :func:`emmeans` would otherwise silently
    fall back to unweighted frequencies.
    """
    if model is None:
        return None
    for attr in ("freq_weights", "var_weights", "weights"):
        w = getattr(model, attr, None)
        if w is None:
            continue
        arr = np.asarray(w, dtype=float).ravel()
        if arr.size == 0:
            continue
        # statsmodels OLS sets `model.weights = 1.0` (a Python scalar) for
        # unweighted fits; skip those.
        if arr.size == 1 and float(arr[0]) == 1.0:
            continue
        # WLS expands a scalar to a 1-vector; if all weights are equal we
        # don't need to carry them through.
        if arr.size > 1 and np.allclose(arr, arr[0]):
            continue
        return arr
    return None


def _build_estimability_basis(
    X: np.ndarray, tol: float | None = None
) -> np.ndarray | None:
    """Return an orthonormal row-space basis of X (for post-pickle estimability).

    A contrast row ``c`` is estimable iff it lies in the row space of X.
    We store ``V_r`` (the right-singular vectors of X with non-trivial
    singular values) so the contrast estimability check can verify
    ``||c - V_r V_r^T c|| <= tol`` even after the original X has been
    dropped from the unpickled ModelInfo. Returns ``None`` when X is empty
    or full-rank up to numerical tolerance (no rank deficiency, so every
    contrast on the column dimension is estimable and the basis isn't
    needed).
    """
    if X is None or X.size == 0 or X.ndim != 2:
        return None
    try:
        _, s, vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    if s.size == 0:
        return None
    cutoff = (tol if tol is not None else max(X.shape) * np.finfo(float).eps) * s.max()
    rank = int((s > cutoff).sum())
    if rank == X.shape[1]:
        # Full column rank: every contrast in R^p is estimable, no basis
        # needed (the estimability check short-circuits to all-True).
        return None
    return vt[:rank].astype(float, copy=True)


@dataclass(frozen=True)
class ModelInfo:
    """Everything pymmeans needs from a fitted model."""

    beta: np.ndarray
    vcov: np.ndarray
    param_names: list[str]
    factors: dict[str, list[str]]
    numeric_means: dict[str, float]
    df_resid: float
    design_info: Any
    data: pd.DataFrame
    response_name: str
    family: Any | None = None
    scale: float = 1.0 # sigma^2 for OLS, dispersion for GLM
    is_mixed: bool = False # True for MixedLM (z-tests by default)
    aliases: dict[str, str] = field(default_factory=dict)
    """Underlying-column -> patsy-canonical name mapping. Lets a user write
    ``emmeans(model, "percent")`` even when the formula was
    ``y ~ C(percent)`` (canonical name is ``"C(percent)"``)."""
    raw_result: Any | None = None
    """The underlying fitted result, kept for operations that need to
    re-evaluate at perturbed parameters (e.g. Satterthwaite df)."""
    offset_mean: float = 0.0
    """Mean of the ``offset=`` (or GLM ``exposure=``) vector at fit time.
    Added to the linear predictor before applying the inverse link in
    ``emmeans(type="response")``. Zero when no offset was used. R's
    ``emmeans`` honours ``offset()`` terms automatically; #1
    fixed our previous silent zero-offset behaviour."""
    fit_weights: np.ndarray | None = None
    """Per-observation weights from the fit (WLS ``weights=`` or GLM
    ``var_weights`` / ``freq_weights``). Used by ``weights="proportional"``
    and ``weights="outer"`` to build frequency tables that match R's
    weighted ``emmeans``. ``None`` for unweighted fits."""
    estimability_basis: np.ndarray | None = None
    """Right-singular vectors of the training design matrix corresponding
    to non-trivial singular values; an orthonormal basis for the row space
    of X. Used by the contrast estimability check after a pickle round
    trip, when ``raw_result`` (and hence the original design matrix) has
    been dropped. ``None`` if no design matrix was available at adapter
    construction time."""
    multi_col_factors: dict[str, list[str]] = field(default_factory=dict)
    """``{patsy_factor_name → [underlying data column(s)]}``
    for numerical factors whose patsy expansion spans more than one
    design column (e.g. ``"bs(x, df=3)"`` → ``["x"]``, ``"poly(x, 3)"``
    → ``["x"]``). Pre-these were refused at adapter time
    with a ``NotImplementedError``; makes the analytic grid
    re-evaluate the basis at user-supplied / mean covariate values
    via ``patsy.build_design_matrices``, closing the GAM-smooth-terms
    parity gap with R ``emmeans``."""

    @property
    def n_params(self) -> int:
        """Number of fixed-effect coefficients in ``beta``."""
        return len(self.beta)

    def __getstate__(self) -> dict[str, Any]:
        """Pickle-safe state.

        Drops the patsy ``design_info``, the raw statsmodels/linearmodels
        result, and the training DataFrame — none of which round-trip
        through ``pickle`` reliably (patsy raises ``NotImplementedError``
        on its DesignInfo, and statsmodels Results aren't designed for
        serialisation). The ``estimability_basis`` array is preserved so
        the contrast estimability check still works post-unpickle
        (without this, unpickled rank-deficient results would silently
        treat all contrasts as estimable). Downstream use of the
        resulting EMMResult / ContrastResult retains everything needed
        for inference, plotting, and Satterthwaite/KR fall-through.
        """
        state = self.__dict__.copy()
        state["design_info"] = None
        state["raw_result"] = None
        state["data"] = pd.DataFrame() # empty placeholder
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Pickle reconstruction (frozen dataclass workaround)."""
        for k, v in state.items():
            object.__setattr__(self, k, v)


def from_linearmodels(result: Any, data: pd.DataFrame | None = None) -> ModelInfo:
    """Build a ``ModelInfo`` from a fitted linearmodels result.

    Supports ``linearmodels.PanelOLS`` and formula-based IV models
    (``IV2SLS``, ``IVGMM``, ``IVLIML``) when ``data=`` is supplied.
    For IV fits the adapter strips the ``[endog ~ instruments]``
    section out of the formula, reparses the structural formula with
    patsy, and reuses the IV-corrected ``result.params`` /
    ``result.cov``. The bracket-section parsing is regex-
    based; complex IV formulas with multiple brackets / interactions
    with the endogenous regressors should be smoke-tested before
    relied on.

    Parameters
    ----------
    result
        Fitted linearmodels result.
    data
        Original DataFrame passed to ``from_formula``. linearmodels does
        NOT preserve raw factor columns post-fit (only the coded design
        matrix), so reconstructing patsy's factor metadata requires the
        original frame. If omitted, an attempt is made to recover from
        ``result.model.dependent.dataframe`` + ``.exog.dataframe`` — this
        works only when the formula has no categorical variables.

    Notes
    -----
    The formula must include an explicit ``1`` intercept (e.g.
    ``y ~ 1 + x + g``) so that linearmodels uses patsy-compatible
    treatment coding instead of full categorical encoding. Entity / time /
    cluster effects are NOT modeled in the EMMeans design; inference
    assumes the reported beta and vcov already absorb whatever fixed
    effects linearmodels applied.
    """
    from patsy import dmatrices

    model = getattr(result, "model", None)
    if model is None or not hasattr(model, "formula"):
        raise TypeError(
            "Expected a linearmodels result with .model.formula; got "
            f"{type(result).__name__}."
        )
    formula = model.formula
    # linearmodels IV formulas embed `[endog ~ instruments]`
    # for the first-stage regression. Patsy can't parse that syntax, but
    # for EMM purposes we only need the STRUCTURAL design (the user's
    # right-hand side minus the IV bracket section). The IV fit returns
    # consistent beta and a robust-to-endogeneity vcov for the
    # structural columns; the bracket-stage instruments don't appear in
    # `result.params`. So strip the `[...]` and reparse with patsy on
    # the cleaned formula.
    #
    # Three caveats:
    # - The user MUST pass `data=` (the original DataFrame). IV models
    # don't expose a unified `.dataframe`, and the bracket-stage
    # data lives separately.
    # - Endogenous regressors stay in the structural formula (they
    # appear in `result.params`); only the bracket section is dropped.
    # - Multiple endogenous variables (`[e1 e2 ~ z1 z2]`) work the
    # same way — strip the whole bracket, the endog names are still
    # in the LHS of `[...]` and must be added to the cleaned formula
    # (patsy needs them as explicit terms).
    _iv_bracket = re.search(r"\[([^]]+)\]", formula)
    if _iv_bracket is not None:
        if data is None:
            raise ValueError(
                "linearmodels IV models (IV2SLS / IVGMM / IVLIML) require "
                "the original DataFrame via `from_linearmodels(result, "
                "data=df)` so pymmeans can rebuild the structural design "
                f"matrix. Got formula {formula!r} with no data= override."
            )
        bracket_content = _iv_bracket.group(1)
        # LHS of "endog ~ instr" inside the bracket is the endogenous
        # variable list. Each name must remain in the cleaned formula
        # as a structural term so patsy includes it.
        endog_section = bracket_content.split("~", 1)[0].strip()
        endog_names = [
            tok.strip() for tok in endog_section.split("+") if tok.strip()
        ]
        # Strip the full `[...]` token (possibly preceded / followed by `+`)
        cleaned = re.sub(r"\s*\+?\s*\[[^]]+\]\s*\+?\s*", " ", formula)
        # Re-add endogenous names as plain structural terms
        rhs = cleaned.split("~", 1)[1] if "~" in cleaned else cleaned
        rhs_clean = rhs.strip().strip("+").strip()
        endog_segment = " + ".join(endog_names)
        if rhs_clean:
            new_rhs = f"{rhs_clean} + {endog_segment}"
        else:
            new_rhs = endog_segment
        formula = f"{cleaned.split('~', 1)[0].strip()} ~ {new_rhs}"

    iv_classes = ("IV2SLS", "IVGMM", "IVLIML", "IV3SLS")
    # Note: we now ACCEPT IV classes if their formula was stripped
    # successfully above. Only refuse if formula stripping somehow
    # failed AND we still have an IV-typed model. (Defensive belt +
    # suspenders.)
    if type(model).__name__ in iv_classes and _iv_bracket is None:
        raise NotImplementedError(
            f"linearmodels {type(model).__name__} fit without an IV "
            "bracket section is unusual; please report this as a bug "
            "with the formula used."
        )
    # Strip linearmodels' absorbed-effect tokens (EntityEffects, TimeEffects,
    # FixedEffects) — patsy can't parse them, and they don't contribute
    # additional columns to the FE design (linearmodels absorbs them out).
    _absorbed_tokens = ("EntityEffects", "TimeEffects", "FixedEffects")
    clean_formula = formula
    for tok in _absorbed_tokens:
        # Remove "+ tok" or "tok +" wherever they appear in the RHS
        for pat in (f"+ {tok}", f"{tok} +", f"+{tok}", f"{tok}+", tok):
            clean_formula = clean_formula.replace(pat, "").strip()
    formula = clean_formula
    if "1 +" not in formula and not formula.strip().split("~")[1].lstrip().startswith("1"):
        raise ValueError(
            "linearmodels formulas must include an explicit '1' intercept "
            f"for pymmeans (got {formula!r}). Add '1 + ' to the RHS."
        )

    def _lm_response_name(dep_container: Any) -> str:
        # Panel data exposes `.dataframe` (DataFrame); IV data exposes
        # `.cols` (list of names) and `.pandas` (DataFrame). Try both.
        if hasattr(dep_container, "dataframe"):
            return str(dep_container.dataframe.columns[0])
        if hasattr(dep_container, "cols"):
            return str(dep_container.cols[0])
        if hasattr(dep_container, "pandas"):
            return str(dep_container.pandas.columns[0])
        raise TypeError(
            f"Cannot extract response name from {type(dep_container).__name__}; "
            "expected `.dataframe`, `.cols`, or `.pandas`."
        )

    if data is not None:
        frame = data.reset_index() if isinstance(data.index, pd.MultiIndex) else data.copy()
        response_name = _lm_response_name(result.model.dependent)
    else:
        dep_frame = result.model.dependent.dataframe
        exog_frame = result.model.exog.dataframe
        frame = pd.concat([dep_frame, exog_frame], axis=1)
        response_name = dep_frame.columns[0]

    _, design_matrix = dmatrices(formula, frame, return_type="dataframe")
    design_info = design_matrix.design_info
    patsy_names = list(design_info.column_names)
    lm_names = list(result.params.index)
    if set(lm_names) != set(patsy_names):
        raise ValueError(
            "linearmodels params don't match the reconstructed patsy "
            f"design columns. params={lm_names}, patsy={patsy_names}."
        )

    # Reorder linearmodels' beta/vcov to patsy's column order
    order = [lm_names.index(n) for n in patsy_names]
    beta = np.asarray(result.params, dtype=float)[order]
    vcov_full = np.asarray(result.cov, dtype=float)
    vcov = vcov_full[np.ix_(order, order)]

    factors: dict[str, list[str]] = {}
    numeric_means: dict[str, float] = {}
    multi_col_factors_dict: dict[str, list[str]] = {}
    exog = np.asarray(design_matrix)
    frame_cols = set(frame.columns)
    aliases: dict[str, str] = {}
    for factor, fi in design_info.factor_infos.items():
        name = factor.name()
        if fi.type == "categorical":
            # NOTE: keep patsy's full category list (including
            # unused pandas-Categorical levels) so that
            # ``weights="cells"`` and similar paths can still emit
            # the documented "NaN row for the unobserved cell"
            # contract. The Tukey/Šidák family-size correction for
            # unused levels lives in
            # ``contrasts._contrast_one_family``.
            factors[name] = list(fi.categories)
        elif name in frame.columns:
            numeric_means[name] = float(frame[name].mean())
        else:
            # also support multi-col basis
            # expressions (``bs(x, df=3)``, ``cr(x, df=4)``, etc.)
            # in the linearmodels adapter — matches the
            # ``from_statsmodels`` handling.
            standalone_width = getattr(fi, "num_columns", None) or 0
            handled_via_main = False
            for term in design_info.terms:
                if len(term.factors) == 1 and term.factors[0] is factor:
                    tslice = design_info.term_slices[term]
                    width = tslice.stop - tslice.start
                    if width == 1:
                        numeric_means[name] = float(exog[:, tslice].mean())
                    elif width > 1:
                        underlying_cols = _underlying_columns(
                            factor.code, frame_cols
                        )
                        if underlying_cols:
                            multi_col_factors_dict[name] = underlying_cols
                            for col in underlying_cols:
                                if col in frame.columns:
                                    numeric_means.setdefault(
                                        col, float(frame[col].mean())
                                    )
                    handled_via_main = True
                    break
            if not handled_via_main and standalone_width > 1:
                # Interaction-only multi-col factor.
                underlying_cols = _underlying_columns(
                    factor.code, frame_cols
                )
                if underlying_cols:
                    multi_col_factors_dict[name] = underlying_cols
                    for col in underlying_cols:
                        if col in frame.columns:
                            numeric_means.setdefault(
                                col, float(frame[col].mean())
                            )
        if name not in frame_cols:
            underlying = _underlying_columns(factor.code, frame_cols)
            if len(underlying) == 1 and underlying[0] != name:
                aliases.setdefault(underlying[0], name)

    est_basis = _build_estimability_basis(exog) if exog is not None else None

    return ModelInfo(
        beta=beta,
        vcov=vcov,
        param_names=patsy_names,
        factors=factors,
        numeric_means=numeric_means,
        df_resid=float(result.df_resid),
        design_info=design_info,
        data=frame,
        response_name=str(response_name),
        family=None,
        scale=1.0,
        multi_col_factors=multi_col_factors_dict,
        is_mixed=False,
        aliases=aliases,
        raw_result=result,
        offset_mean=0.0,
        fit_weights=None,
        estimability_basis=est_basis,
    )


def from_fitted(result: Any, **kwargs: Any) -> ModelInfo:
    """Dispatch helper: pick a registered adapter and build a ModelInfo.

    The first registered adapter whose ``detects(result)`` returns True
    wins. Built-in adapters: ``StatsmodelsAdapter`` and
    ``LinearmodelsAdapter``. Register more via
    ``pymmeans.adapters.register_adapter``.
    """
    # Imported lazily because adapters.py imports back from utils.py.
    from pymmeans.adapters import dispatch

    return dispatch(result, **kwargs)


def _response_family_for_model(model: Any) -> Any:
    """Best-effort `family` extraction for response-scale EMMs.

    adapter previously stored only
    ``getattr(model, "family", None)``, which is None for several
    statsmodels classes that nevertheless have a meaningful response-
    scale link:

    - BetaModel: exposes ``model.link`` (logit by default).
    - statsmodels.discrete Probit / Logit / NegativeBinomial:
      have a fixed link by definition.
    - PHReg (Cox PH): handled separately via the
      ``_cox_response_name_override`` synthetic ``response_name``.

    When ``family`` is None but a link CAN be inferred, return a
    minimal family-like object (``SimpleNamespace`` with a ``link``
    attribute) that ``emmeans(type='response')`` consumes via the
    ``family.link.inverse`` / ``family.link.inverse_deriv`` calls.
    """
    from types import SimpleNamespace as _SimpleNamespace

    fam = getattr(model, "family", None)
    if fam is not None:
        return fam
    link = getattr(model, "link", None)
    if link is not None:
        return _SimpleNamespace(link=link)
    # Discrete-model fallback: family-less classes with fixed links
    cls = type(model).__name__
    if cls in ("NegativeBinomial", "NegativeBinomialP"):
        import statsmodels.api as _sm
        return _SimpleNamespace(link=_sm.families.links.Log())
    if cls == "Probit":
        import statsmodels.api as _sm
        return _SimpleNamespace(link=_sm.families.links.Probit())
    if cls == "Logit":
        import statsmodels.api as _sm
        return _SimpleNamespace(link=_sm.families.links.Logit())
    return None


def _cox_response_name_override(cls_name: str, endog_name: str) -> str:
    """Cox PH (`PHReg`) fits the linear predictor on the
    LOG-HAZARD scale but exposes no family / link to advertise that
    to ``emmeans(..., type='response')``. R `coxph` users expect
    ``type='response'`` to return hazard RATIOS (exponentiated linear
    predictor), matching the equivalent log-link Poisson GLM.

    We synthesize ``np.log(<endog>)`` as the ``response_name`` so the
    existing ``detect_transform`` machinery picks up the ``log`` family
    and ``type='response'`` applies ``exp`` automatically. Contrast
    back-transforms also flip ``estimate`` → ``ratio`` (the standard
    hazard-ratio interpretation) thanks to the log-family
    ``is_log=True`` flag.

    For non-Cox classes this is a no-op; we return the raw endog name.
    """
    if cls_name in ("PHRegResults", "PHReg"):
        return f"np.log({endog_name})"
    return endog_name


def from_statsmodels(result: Any) -> ModelInfo:
    """Build a ``ModelInfo`` from a fitted statsmodels OLS or GLM result.

    Parameters
    ----------
    result
        A fitted result from ``smf.ols(...).fit()`` or ``smf.glm(...).fit()``.

    Raises
    ------
    TypeError
        If ``result`` does not look like a statsmodels Results object.
    ValueError
        If the model was not fit via the formula API (no patsy design info).

    Examples
    --------
    >>> import pandas as pd
    >>> import statsmodels.formula.api as smf
    >>> df = pd.DataFrame({
    ... "y": [1.0, 2.0, 3.0, 4.0],
    ... "g": pd.Categorical(["a", "b", "a", "b"]),
    ... })
    >>> info = from_statsmodels(smf.ols("y ~ g", data=df).fit())
    >>> info.factors
    {'g': ['a', 'b']}
    """
    model = getattr(result, "model", None)
    if model is None:
        raise TypeError(
            "Expected a fitted statsmodels Results object (with .model); "
            f"got {type(result).__name__}."
        )

    # refuse model classes whose parameter vector is shaped
    # in a way pymmeans's scalar-y EMM math cannot use without a
    # dedicated adapter. MNLogit has a 2-D params matrix (one column
    # per non-reference outcome category); the user can fit a binary
    # model per category as a workaround.
    #
    # OrderedModel (cumulative-link ordinal regression) is
    # now supported via the existing by-name design-column slice
    # (which discards the trailing transformed-threshold parameters)
    # and a dedicated :mod:`pymmeans.ordinal` module for the
    # response-scale modes (per-category prob, mean.class, etc.).
    # The ``latent`` linear-predictor EMM works through the standard
    # ``emmeans()`` entry point.
    _cls_name = type(model).__name__
    if _cls_name == "MNLogitResults" or _cls_name == "MNLogit":
        raise NotImplementedError(
            "MNLogit (multinomial logit) has a 2-D parameter matrix "
            "(one column per non-reference outcome category) that the "
            "generic ``emmeans`` path cannot reduce to a single "
            "contrast. added the dedicated entry point:\n\n"
            " from pymmeans import multinom_emmeans\n"
            " res = multinom_emmeans(fit, 'g', mode='prob')\n\n"
            "Supported modes: 'prob' (per-category probabilities) and "
            "'latent' (centered log-odds, matching R "
            "``emmeans(... , mode='latent')``). As a fallback for "
            "ad-hoc category contrasts, fit "
            "``smf.logit('I(y == k) ~ ...')`` per category and call "
            "emmeans on each."
        )
    # Also detect by params shape — MNLogit fits return a Results
    # wrapper whose .params is a 2-D DataFrame.
    params = getattr(result, "params", None)
    if params is not None and getattr(params, "ndim", 1) > 1:
        raise NotImplementedError(
            f"Model class {_cls_name} returns a 2-D parameter matrix "
            f"(shape {params.shape}); pymmeans v0.1 only supports "
            "models with a 1-D structural beta vector. Likely a "
            "multinomial / multivariate fit — see MNLogit workaround "
            "above."
        )

    # (P0, silent-numerical-wrongness): a Binomial GLM
    # fit with a comparison-style LHS (e.g. ``(y > c) ~ x``) expands
    # the response to a 2-column endog matrix ``[<expr>[False],
    # <expr>[True]]``, and statsmodels then fits P(False) (column 0)
    # as success — NOT P(True), which is what the user almost
    # certainly intended. Pymmeans faithfully reproduces statsmodels'
    # choice (verified against ``fit.predict()`` to zero ulp), but
    # the resulting response-scale probabilities are the COMPLEMENT
    # of what the user wanted, and the displayed probability column
    # gives no hint of the class flip. refuses this fit
    # shape with an actionable error pointing to the binary 0/1
    # workaround. had previously made the 2-col path
    # "not crash" in regrid_response; the defensive
    # check in transforms.py is preserved as defense-in-depth, but
    # the canonical refusal now lives at adapter-construction time
    # so neither type='response' nor regrid_response can silently
    # produce wrong probabilities.
    endog_names = getattr(model, "endog_names", None)
    family = getattr(model, "family", None)
    if (
        isinstance(endog_names, list)
        and len(endog_names) == 2
        and family is not None
        and type(family).__name__ == "Binomial"
    ):
        raise NotImplementedError(
            f"2-column-LHS Binomial GLMs (endog_names={endog_names!r}) "
            "are refused by pymmeans because statsmodels models the "
            "FIRST column as success — for a comparison-LHS like "
            "``(y > c) ~ x`` that means it fits P(y <= c), the "
            "OPPOSITE class from what users almost always intend. "
            "The displayed response-scale probabilities would silently "
            "be the complement of P(y > c). Refit with an explicit "
            "binary outcome instead:\n\n"
            " df['outcome'] = (y > c).astype(int)\n"
            " fit = smf.glm('outcome ~ x', df, "
            "family=sm.families.Binomial()).fit()\n\n"
            "and pymmeans will return P(outcome=1) on the response "
            "scale, matching the user's intent."
        )

    data_attr = getattr(model, "data", None)
    design_info = getattr(data_attr, "design_info", None) if data_attr else None
    if design_info is None:
        raise ValueError(
            "Model has no design_info. pymmeans requires a model fit via the "
            "statsmodels formula API (smf.ols / smf.glm) so that factor "
            "structure is available."
        )

    frame = getattr(data_attr, "frame", None)
    if frame is None:
        raise ValueError(
            "Original DataFrame not attached to model. Fit via smf.ols / "
            "smf.glm with data=<pandas DataFrame>."
        )

    # MixedLM exposes fe_params (fixed-effects-only) separate from .params
    # (which also includes variance components). Use fe_params when present
    # and slice cov_params to the fixed-effects sub-block.
    is_mixed = hasattr(result, "fe_params")
    if is_mixed:
        beta_series = result.fe_params
        beta = np.asarray(beta_series, dtype=float)
        param_names = list(beta_series.index)
        full_vcov = result.cov_params()
        if hasattr(full_vcov, "loc"):
            vcov = np.asarray(
                full_vcov.loc[param_names, param_names], dtype=float
            )
        else:
            n_fe = len(beta)
            vcov = np.asarray(full_vcov, dtype=float)[:n_fe, :n_fe]
    else:
        beta = np.asarray(result.params, dtype=float)
        vcov = np.asarray(result.cov_params(), dtype=float)
        param_names = (
            list(result.params.index)
            if hasattr(result.params, "index")
            else list(model.exog_names)
        )
        # some statsmodels model classes (BetaModel, Tobit
        # truncated regression, ZIP) append auxiliary nuisance
        # parameters to `result.params` alongside the structural fixed-
        # effect columns. `model.k_extra` advertises an extra count
        # but does NOT specify position — BetaModel APPENDS precision,
        # ZeroInflatedPoisson PREPENDS the inflation parameter. The
        # naive "drop last k_extra entries" approach () was
        # therefore silently wrong for ZIP: it kept the inflation
        # parameter and dropped a real structural coefficient.
        #
        # replace positional slicing with
        # by-NAME selection against `design_info.column_names`. When
        # the design columns are a subset of `param_names`, reorder
        # beta + vcov to the design's column order. Fall back to the
        # positional slice (with a warning) when names don't align.
        # Belt-and-suspenders refusal for zero-inflated models in
        # particular — `inflate_*` parameters in the result indicate
        # the structural slice is component-specific math we don't
        # yet model.
        design_cols = list(design_info.column_names)
        param_names_str = [str(n) for n in param_names]
        if set(design_cols).issubset(set(param_names_str)):
            order = [param_names_str.index(c) for c in design_cols]
            beta = beta[order]
            vcov = vcov[np.ix_(order, order)]
            param_names = design_cols
        else:
            k_extra = int(getattr(model, "k_extra", 0) or 0)
            if k_extra:
                # Positional fallback: only safe when extras are appended
                # AND none of them collide with design column names.
                # We can't tell from k_extra alone, so emit a warning
                # so users notice if their results look off.
                import warnings as _w
                _w.warn(
                    f"from_statsmodels: positional k_extra slice "
                    f"({k_extra} extras off the end of "
                    f"{len(param_names_str)} params). Design column "
                    f"names {design_cols} are not a subset of "
                    f"param_names {param_names_str}, so by-name "
                    "selection isn't available. If your model class "
                    "prepends extras (e.g. ZeroInflatedPoisson), the "
                    "wrong coefficients will be kept — verify the "
                    "result or build a dedicated adapter.",
                    UserWarning, stacklevel=3,
                )
                n_design = len(design_cols)
                beta = beta[:n_design]
                vcov = vcov[:n_design, :n_design]
                param_names = param_names[:n_design]
        # defensive refusal: even if
        # by-name selection succeeded, zero-inflated models carry
        # `inflate_*` parameters in the FULL param vector. The cleanly-
        # selected structural sub-block is correct, but the user might
        # have intended inference on the inflation component, which
        # we don't support. Tell them explicitly so they don't get a
        # silently-truncated answer.
        if any(
            str(n).startswith("inflate_") or str(n).startswith("zero_inf")
            for n in param_names_str
        ):
            raise NotImplementedError(
                "Zero-inflated models (ZeroInflatedPoisson / "
                "ZeroInflatedNegativeBinomialP) need a dedicated adapter "
                "because the inflation and structural components are "
                "separate equations. EMMs on the structural component "
                "alone are mathematically defined (and the by-name "
                "selection in extracts them correctly), but "
                "we refuse to ship that as the default since users "
                "frequently want the response-scale (probability-of-"
                "zero-adjusted) mean. Tracked for v0.2."
            )

    factors: dict[str, list[str]] = {}
    numeric_means: dict[str, float] = {}
    multi_col_factors_dict: dict[str, list[str]] = {}
    exog = np.asarray(model.exog) if hasattr(model, "exog") else None
    frame_cols = set(frame.columns)
    aliases: dict[str, str] = {}

    for factor, fi in design_info.factor_infos.items():
        name = factor.name()
        if fi.type == "categorical":
            # NOTE: see ``from_linearmodels`` — 's fix
            # for unused-categorical family-size inflation is on the
            # contrast side, not here.
            factors[name] = list(fi.categories)
        elif name in frame.columns:
            numeric_means[name] = float(frame[name].mean())
        elif exog is not None:
            # Numerical expression like np.log(x): use the mean of the
            # evaluated column from the design matrix.
            # also handle factors that appear ONLY in
            # interaction terms (e.g. ``bs(x, df=3):g``). Patsy's
            # ``factor_info.num_columns`` gives the factor's
            # standalone width regardless of which term(s) it
            # appears in.
            standalone_width = getattr(fi, "num_columns", None)
            if standalone_width is None:
                standalone_width = 0
                for term in design_info.terms:
                    if (
                        len(term.factors) == 1 and term.factors[0] is factor
                    ):
                        tsl = design_info.term_slices[term]
                        standalone_width = tsl.stop - tsl.start
                        break
            for term in design_info.terms:
                if len(term.factors) == 1 and term.factors[0] is factor:
                    tslice = design_info.term_slices[term]
                    width = tslice.stop - tslice.start
                    if width == 1:
                        numeric_means[name] = float(exog[:, tslice].mean())
                    elif width > 1:
                        # multi-column numerical expressions
                        # like ``bs(x, df=3)``, ``ns(x, df=4)``,
                        # ``poly(x, 3)``. Pre-these were
                        # refused at adapter time; makes the
                        # analytic grid re-evaluate the basis at user
                        # / mean covariate values via
                        # ``patsy.build_design_matrices``.
                        underlying_cols = _underlying_columns(
                            factor.code, set(frame.columns)
                        )
                        if not underlying_cols:
                            # Couldn't identify the underlying column —
                            # fall back to the refusal with the same
                            # workaround hint.
                            raise NotImplementedError(
                                f"Multi-column numerical expression '{name}' "
                                f"(spans {width} design columns) references no "
                                "identifiable column in the DataFrame; "
                                "pymmeans cannot enumerate the reference grid. "
                                "Pre-compute the basis columns (e.g. via "
                                "``patsy.bs(x, df=3)`` outside the formula) "
                                "and pass them as plain numeric columns."
                            )
                        multi_col_factors_dict[name] = underlying_cols
                        # Store the mean of each underlying column for
                        # marginalisation when the user doesn't supply
                        # ``at={col: ...}``.
                        for col in underlying_cols:
                            if col in frame.columns:
                                numeric_means.setdefault(
                                    col, float(frame[col].mean())
                                )
                    break
            else:
                # No main-effect term — factor appears only in
                # interaction(s). Use ``standalone_width`` from
                # patsy's factor_info to detect multi-col status.
                if standalone_width > 1:
                    underlying_cols = _underlying_columns(
                        factor.code, set(frame.columns)
                    )
                    if underlying_cols:
                        multi_col_factors_dict[name] = underlying_cols
                        for col in underlying_cols:
                            if col in frame.columns:
                                numeric_means.setdefault(
                                    col, float(frame[col].mean())
                                )

        # Build alias: underlying single-column -> patsy canonical name.
        if name not in frame_cols:
            underlying = _underlying_columns(factor.code, frame_cols)
            if len(underlying) == 1 and underlying[0] != name:
                aliases.setdefault(underlying[0], name)

    # Extract offset terms (statsmodels GLM `offset=` parameter and
    # PoissonRegression `exposure=`). The fit predicts on eta + offset,
    # so the reference-grid eta needs the offset's mean added back in.
    offset = _extract_offset(result, model)

    # Extract per-observation fit weights (WLS `weights=`, GLM `var_weights`
    # or `freq_weights`). we used unweighted `value_counts()`
    # for the `outer`/`proportional` weighting schemes, which silently
    # ignored frequency weights and gave wrong EMMs on weighted fits.
    fit_weights = _extract_fit_weights(result, model)

    # Build a picklable estimability basis: the right-singular vectors of
    # X corresponding to non-trivial singular values. After unpickling,
    # `raw_result` is None, so the contrast estimability check cannot
    # rebuild the row space — without this basis, unpickled results
    # silently treat all contrasts as estimable (#5).
    est_basis = _build_estimability_basis(exog) if exog is not None else None

    # Cox PH (PHReg) inference is asymptotic; R
    # reports `df = Inf` and uses z-quantiles for CIs / p-values. The
    # statsmodels `df_resid` is `n - p` (finite), which made pymmeans
    # use t(226) instead of z on the lung dataset — p=0.0025 vs R's
    # p=0.0022 (12% relative drift). Override to inf so `summary` /
    # `confint` / `pairs` use the asymptotic distribution that R uses.
    if _cls_name in ("PHRegResults", "PHReg"):
        df_resid_use = float("inf")
    else:
        df_resid_use = float(result.df_resid)

    return ModelInfo(
        beta=beta,
        vcov=vcov,
        param_names=param_names,
        factors=factors,
        numeric_means=numeric_means,
        df_resid=df_resid_use,
        design_info=design_info,
        data=frame,
        response_name=_cox_response_name_override(_cls_name, model.endog_names),
        family=_response_family_for_model(model),
        scale=float(getattr(result, "scale", 1.0) or 1.0),
        is_mixed=is_mixed,
        aliases=aliases,
        raw_result=result,
        offset_mean=offset,
        fit_weights=fit_weights,
        estimability_basis=est_basis,
        multi_col_factors=multi_col_factors_dict,
    )
