"""Multinomial-logit EMMs (``statsmodels.MNLogit``).

bringing pymmeans up to R `emmeans`'s multinomial-model
coverage. Adds response-scale modes for ``statsmodels.MNLogit``
fits:

- ``mode="latent"`` — per-non-reference-category log-odds
  ``η_j = X β_j``. Multi-row output (one row per
  target-level × non-reference category).
- ``mode="prob"`` (default) — per-category probabilities
  ``P(Y = k | x) = softmax(0, η_1, ..., η_{J-1})_k``.
  Multi-row output (one row per target-level × all J categories,
  including the reference).

For the response-scale mode we apply the delta method through
the softmax, accounting for the block covariance between the
non-reference-category coefficient vectors.

R `emmeans` does exactly the same thing under the hood; pymmeans
matches at floating-point precision on a synthetic reference
case (validated against ``nnet::multinom`` + ``emmeans``).

References
----------
- Greene, W. H. (2018). *Econometric Analysis* (8th ed.).
  Pearson. Chapter on multinomial logit and softmax response.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pymmeans.emmeans import EMMResult


def _extract_multinom_state(result: Any) -> dict[str, Any]:
    """Pull β matrix, full flat vcov, and category labels from
    a fitted ``MNLogit``.

    statsmodels' ``MNLogit`` returns ``params`` as a DataFrame of
    shape ``(k_vars, J - 1)`` where columns are non-reference
    outcome categories (the first category is the reference). The
    flat ``cov_params()`` matrix is ordered **column-major**:
    equation-1's coefficients first (in design-column order), then
    equation-2's, and so on. Verified by inspecting the
    MultiIndex on ``cov_params().index``.

    Returns
    -------
    dict
        - ``beta_mat`` : ``np.ndarray`` of shape ``(k_vars, J-1)``
        - ``vcov_flat`` : ``np.ndarray`` of shape
          ``(k_vars*(J-1), k_vars*(J-1))``, column-major
        - ``categories`` : J category labels (with the reference
          category at index 0)
        - ``column_names`` : list of k_vars structural design
          column names
    """
    params_df = result.params
    if not hasattr(params_df, "values") or params_df.values.ndim != 2:
        raise TypeError(
            "multinom_emmeans: `fit.params` must be a 2-D DataFrame "
            "(shape (k_vars, J-1)). Got "
            f"{type(params_df).__name__} with ndim "
            f"{getattr(getattr(params_df, 'values', None), 'ndim', 'n/a')}."
        )
    beta_mat = np.asarray(params_df.values, dtype=float)
    k_vars, J_minus_1 = beta_mat.shape
    cov_full = np.asarray(result.cov_params(), dtype=float)
    expected = k_vars * J_minus_1
    if cov_full.shape != (expected, expected):
        raise ValueError(
            "multinom_emmeans: cov_params() shape "
            f"{cov_full.shape} does not match expected "
            f"({expected}, {expected})."
        )
    # Categories: pull from the model's endog / wendog mapping.
    # MNLogit stores them on ``model.wendog`` (the dummy-coded
    # response) and ``model._ynames_map`` when present.
    model = result.model
    cat_labels: list[Any]
    if hasattr(model, "_ynames_map") and model._ynames_map:
        # _ynames_map is {original_label: integer_code}; invert
        # and sort by code.
        m = model._ynames_map
        cat_labels = [k for k, _ in sorted(m.items(), key=lambda kv: kv[1])]
    else:
        # Fall back to integer codes.
        cat_labels = list(range(J_minus_1 + 1))

    column_names = list(getattr(params_df, "index", []))
    if not column_names:
        # Synthesize from model.exog_names.
        column_names = list(getattr(model, "exog_names", [])) or [
            f"x{i}" for i in range(k_vars)
        ]
    return dict(
        beta_mat=beta_mat,
        vcov_flat=cov_full,
        categories=cat_labels,
        column_names=column_names,
    )


def _build_multinom_modelinfo_stub(fit: Any) -> Any:
    """Construct a stub :class:`ModelInfo` for multinom grid building.

    MNLogit's ``from_statsmodels`` path refuses the 2-D params
    matrix; this helper sidesteps that by manually building a
    ModelInfo whose ``beta`` is the first non-reference equation's
    coefficient vector (any column works — only ``design_info`` /
    ``factors`` / ``numeric_means`` are used by the grid logic).
    The SE returned by ``emmeans()`` on this stub is discarded; we
    only care about ``linfct`` (the L matrix at the reference grid)
    and the frame labels.
    """
    from pymmeans.utils import ModelInfo

    model = fit.model
    data_attr = getattr(model, "data", None)
    design_info = getattr(data_attr, "design_info", None) if data_attr else None
    frame = getattr(data_attr, "frame", None)
    if design_info is None or frame is None:
        raise TypeError(
            "multinom_emmeans: MNLogit fit must come from the "
            "formula API (``smf.mnlogit(...)``) so that "
            "``design_info`` and the source DataFrame are "
            "available for grid construction."
        )

    beta_mat = np.asarray(fit.params.values, dtype=float)
    k_vars = beta_mat.shape[0]
    cov_full = np.asarray(fit.cov_params(), dtype=float)
    # Stub β + vcov: first equation only. The grid logic doesn't
    # use β values — only sizes — so this is a safe choice.
    stub_beta = beta_mat[:, 0].copy()
    stub_vcov = cov_full[:k_vars, :k_vars].copy()

    # Factor / numeric structure (same logic as from_statsmodels).
    factors: dict[str, list[str]] = {}
    numeric_means: dict[str, float] = {}
    exog = np.asarray(model.exog) if hasattr(model, "exog") else None
    for factor, fi in design_info.factor_infos.items():
        name = factor.name()
        if fi.type == "categorical":
            factors[name] = list(fi.categories)
        elif name in frame.columns:
            numeric_means[name] = float(frame[name].mean())
        elif exog is not None:
            for term in design_info.terms:
                if len(term.factors) == 1 and term.factors[0] is factor:
                    tslice = design_info.term_slices[term]
                    width = tslice.stop - tslice.start
                    if width == 1:
                        numeric_means[name] = float(exog[:, tslice].mean())
                    break

    response_name = str(getattr(model, "endog_names", "y"))
    if isinstance(response_name, list):
        response_name = response_name[0]
    column_names = list(getattr(fit.params, "index", []))
    if not column_names:
        column_names = list(getattr(model, "exog_names", [])) or [
            f"x{i}" for i in range(k_vars)
        ]

    return ModelInfo(
        beta=stub_beta,
        vcov=stub_vcov,
        param_names=column_names,
        factors=factors,
        numeric_means=numeric_means,
        df_resid=float(getattr(fit, "df_resid", np.inf)),
        design_info=design_info,
        data=frame,
        response_name=response_name,
        raw_result=None,
    )


def _build_multinom_reference_grid(
    fit: Any,
    specs: str | list[str],
    by: Any = None,
    at: Any = None,
) -> Any:
    """Return the latent EMMResult from a stub ModelInfo.

    The stub β is the first non-reference equation's coefficient
    vector — only ``linfct`` (the L matrix) and the frame labels
    matter; the SE on the stub frame is discarded.
    """
    from pymmeans.emmeans import emmeans as _emm

    info = _build_multinom_modelinfo_stub(fit)
    return _emm(info, specs, by=by, at=at)


def multinom_emmeans(
    fit: Any,
    specs: Any,
    mode: str = "prob",
    by: Any = None,
    at: Any = None,
    level: float = 0.95,
) -> EMMResult:
    """Multinomial-logit EMMs at the reference grid.

    Parameters
    ----------
    fit
        A fitted ``statsmodels.discrete.discrete_model.MNLogitResults``.
    specs
        Factors / numeric predictors to marginalise over (same
        semantics as :func:`pymmeans.emmeans`).
    mode
        Response-scale mode. One of:

        - ``"latent"`` — per-non-reference-category log-odds
          ``η_j = X β_j``. Returns J-1 rows per target cell.
        - ``"prob"`` (default) — per-category probabilities
          ``P(Y = k | x) = softmax(0, η_1, ..., η_{J-1})_k``.
          Returns J rows per target cell, including the
          reference category.

    by, at, level
        Forwarded to the reference-grid builder (same semantics
        as :func:`emmeans`).

    Returns
    -------
    EMMResult
        With a tidy ``frame`` containing a ``category`` column,
        the point estimate, delta-method SE, df (treated as Inf
        because MNLogit's MLE is asymptotic), and Wald CIs.

    Validation
    ----------
    Matches R ``nnet::multinom`` + ``emmeans(fit, ~ ... | y,
    mode="prob")`` at floating-point precision on a synthetic
    3-category reference case. See
    ``tests/r_reference/multinom_reference.R``.
    """
    if mode not in ("latent", "prob"):
        raise ValueError(
            f"multinom_emmeans: unknown mode {mode!r}. Valid: "
            "'latent', 'prob'."
        )

    state = _extract_multinom_state(fit)
    beta_mat = state["beta_mat"]
    vcov_flat = state["vcov_flat"]
    categories = state["categories"]

    em_stub = _build_multinom_reference_grid(fit, specs, by=by, at=at)
    L = em_stub.linfct
    frame = em_stub.frame.copy()
    n_cells, k_vars = L.shape
    J_minus_1 = beta_mat.shape[1]
    J = J_minus_1 + 1

    # Latent: eta[c, j] = L_c @ beta_mat[:, j] (n_cells, J-1)
    eta = L @ beta_mat

    # Build per-category point estimates and gradients-w.r.t.-flat-β.
    # Flat-β is column-major: [β_eq0_pred0, β_eq0_pred1, ..., β_eq0_pred_{k-1},
    # β_eq1_pred0, ..., β_eq_{J-2}_pred_{k-1}].
    # Output point shape: (n_cells, n_out_cats); gradient: per (cell,
    # output) a (k_vars * (J-1))-vector.
    if mode == "latent":
        # R `emmeans` convention: report J centered log-odds (one
        # per category including the reference) where each value
        # is ``η_k - mean(η_·)`` over all J categories
        # (with the reference's η_0 = 0). This puts the reference
        # category on the same footing as the others and centers
        # the latent values around 0 for direct comparability.
        eta_full = np.column_stack([np.zeros(n_cells), eta]) # (n_cells, J)
        mean_eta = eta_full.mean(axis=1, keepdims=True)
        point = eta_full - mean_eta # (n_cells, J)
        n_out = J
        out_cat_labels = categories # all J categories
    else: # prob
        # Softmax with reference at η_0 = 0.
        # exp_full[c, k] for k in [0, J-1]:
        # k = 0: 1
        # k ≥ 1: exp(eta[c, k - 1])
        exp_full = np.column_stack([
            np.ones(n_cells),
            np.exp(eta),
        ]) # (n_cells, J)
        denom = exp_full.sum(axis=1, keepdims=True)
        probs = exp_full / denom # (n_cells, J)
        n_out = J
        point = probs
        out_cat_labels = categories # all J categories

    # Build the gradient and variance per (cell, output category).
    # Vectorised across cells; loop over output categories.
    se = np.zeros((n_cells, n_out))
    if mode == "latent":
        # centered log-odds: ζ_k = η_k - mean(η_·) for k ∈ [0, J-1]
        # with η_0 ≡ 0. Mean(η_·) = sum(η_1..η_{J-1}) / J.
        # ∂ζ_k/∂β_jl:
        # if k = 0 (reference): ∂ζ_0/∂β_jl = -L_l / J
        # (only the mean contribution from non-reference β)
        # if k ≥ 1: ∂ζ_k/∂β_jl = (δ_{k, j+1} - 1/J) · L_l
        # (β_{j} indexes the (j+1)-th non-reference equation)
        for c in range(n_cells):
            for k in range(J):
                grad = np.zeros(k_vars * J_minus_1)
                for j in range(J_minus_1):
                    softmax_j = j + 1
                    if k == 0:
                        coef = -1.0 / J
                    else:
                        coef = (1.0 if k == softmax_j else 0.0) - 1.0 / J
                    grad[j * k_vars : (j + 1) * k_vars] = coef * L[c]
                se[c, k] = float(np.sqrt(max(grad @ vcov_flat @ grad, 0.0)))
    else: # prob
        # ∂p_k/∂η_j = p_k · (δ_kj - p_j) for k ∈ [0, J-1], j ∈ [1, J-1]
        # (η_0 = 0 is fixed; only η_1..η_{J-1} are free, indexed in
        # `eta` as columns 0..J-2)
        # Chain rule: ∂p_k/∂β_jl = ∂p_k/∂η_{j+1} · L_l (j+1 because
        # column-0 of beta_mat is the first non-reference equation)
        for c in range(n_cells):
            p_c = probs[c] # (J,)
            for k in range(J): # output category k ∈ [0, J-1]
                grad = np.zeros(k_vars * J_minus_1)
                for j in range(J_minus_1):
                    # j indexes the non-reference equation (so the
                    # underlying softmax index is j + 1)
                    softmax_j = j + 1
                    dpk_deta = p_c[k] * (
                        (1.0 if k == softmax_j else 0.0) - p_c[softmax_j]
                    )
                    grad[j * k_vars : (j + 1) * k_vars] = dpk_deta * L[c]
                se[c, k] = float(np.sqrt(max(grad @ vcov_flat @ grad, 0.0)))

    # Assemble output frame.
    from scipy import stats
    # R `emmeans` for multinomial fits uses
    # ``df = n_free_params = k_vars * (J - 1)`` for t-quantile CIs
    # (a conservative finite-sample choice that's slightly wider
    # than the asymptotic z-CI). Match R's convention so the
    # response-scale CIs aren't ~18 % narrower than R's.
    n_free_params = k_vars * J_minus_1
    df_out = np.full(n_cells * n_out, float(n_free_params))
    base_cols = [c for c in frame.columns if c not in
                 ("emmean", "SE", "df", "lower_cl", "upper_cl")]
    rep_base = frame[base_cols].loc[
        frame.index.repeat(n_out)
    ].reset_index(drop=True)
    cat_col = np.tile(out_cat_labels, n_cells)
    point_flat = point.reshape(-1)
    se_flat = se.reshape(-1)
    crit = stats.t.ppf(1.0 - (1.0 - level) / 2.0, df_out)
    out_frame = pd.DataFrame({
        **{c: rep_base[c].to_numpy() for c in rep_base.columns},
        "category": cat_col,
        "emmean": point_flat,
        "SE": se_flat,
        "df": df_out,
        "lower_cl": point_flat - crit * se_flat,
        "upper_cl": point_flat + crit * se_flat,
    })

    # Wrap as EMMResult by replacing the stub's frame. The stub
    # carries the correct target / by / level / linfct from the
    # grid-construction call; we only need to swap in the new
    # multi-row output frame and stamp ``type=mode``.
    from dataclasses import replace
    return replace(em_stub, frame=out_frame, type=mode, level=level)
