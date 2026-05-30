"""Accelerated Failure Time (AFT) survival models — marginal means.

R's `survreg(Surv(T, E) ~ x, dist="weibull")` (and the equivalent log-normal /
log-logistic) is the canonical parametric survival alternative to Cox; R
`emmeans` consumes it directly and returns EMMs on the *log-time linear
predictor* scale (with `df = n - n_params`). pymmeans bridges the same
surface for the **lifelines** AFT family (``WeibullAFTFitter``,
``LogNormalAFTFitter``, ``LogLogisticAFTFitter``, ...).

Parameterisation
----------------

Both R `survreg(dist="weibull")` and lifelines ``WeibullAFTFitter`` use the
same AFT linear-predictor parametrisation:

.. math::

   \\log T = X\\beta + \\sigma\\,\\varepsilon,

where ``β`` lives in lifelines' ``params_['lambda_']`` block (Intercept +
covariate effects) and ``σ = 1/exp(rho_['Intercept'])``. On the seeded
n=300 probe the two engines agree to ~1e-6 on β and σ — so an EMM built
from the lifelines ``lambda_`` block + the corresponding
``variance_matrix_`` submatrix reproduces R `emmeans(survreg, ...)` to
floating-point precision.

API
---

:func:`from_aft` wraps a fitted AFT fitter as a :class:`ModelInfo` so the
standard ``emmeans(...)`` / ``contrast(...)`` / ``pairs(...)`` /
``summary(...)`` pipeline works on it directly — no separate
``aft_emmeans`` function needed.

References
----------
- Kalbfleisch & Prentice (2002). *The Statistical Analysis of Failure Time
  Data*, §3.2 (the AFT family).
- Davidson-Pilon, C. (2019). lifelines: survival analysis in Python.
  *Journal of Open Source Software*, 4(40), 1317.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from patsy import dmatrices

from pymmeans.utils import ModelInfo


def from_aft(fitter: Any, data: pd.DataFrame,
             formula: str | None = None) -> ModelInfo:
    """Wrap a fitted lifelines AFT fitter as a :class:`ModelInfo`.

    Parameters
    ----------
    fitter
        A fitted lifelines ``WeibullAFTFitter`` / ``LogNormalAFTFitter`` /
        ``LogLogisticAFTFitter``. The fitter must expose ``params_``
        (a MultiIndex Series with a ``'lambda_'`` block) and
        ``variance_matrix_`` (the joint covariance, with the same
        MultiIndex). pymmeans pulls the location-scale linear-predictor
        block (``lambda_``) and ignores the ancillary shape parameter
        (``rho_`` / similar) — only the linear predictor enters the EMM.
    data
        The DataFrame used to fit ``fitter``. Required so non-specs
        numeric covariates can be averaged to their sample mean.
    formula
        Override for the patsy RHS formula. If ``None``, attempts to
        recover from ``fitter.formula`` or ``fitter._formula``. Pass
        explicitly when lifelines doesn't expose it.

    Returns
    -------
    ModelInfo
        A standard pymmeans model info. Use with
        ``pymmeans.emmeans(info, specs)`` and the full contrast /
        summary / multiplicity surface.
    """
    if not hasattr(fitter, "params_"):
        raise TypeError(
            "from_aft expected a fitted lifelines AFT fitter "
            "(WeibullAFTFitter / LogNormalAFTFitter / LogLogisticAFTFitter / "
            "GeneralizedGammaRegressionFitter); "
            f"got {type(fitter).__name__}."
        )
    params = fitter.params_
    if not hasattr(params, "index") or not hasattr(
        params.index, "get_level_values"
    ):
        raise TypeError(
            "from_aft: fitter.params_ does not have a 2-level MultiIndex "
            "(location-block, covariate); pymmeans relies on the AFT "
            "linear-predictor parameterisation."
        )

    if formula is None:
        formula = getattr(fitter, "formula", None) or getattr(
            fitter, "_formula", None
        )
    if not formula:
        raise ValueError(
            "from_aft: could not recover the patsy formula from the "
            "fitter; pass formula='...' explicitly."
        )

    # Build patsy design_info with a dummy LHS so we can call dmatrices.
    data_d = data.assign(__pmm_y__=np.zeros(len(data)))
    _, X = dmatrices(
        f"__pmm_y__ ~ {formula}", data_d, return_type="dataframe"
    )
    design_info = X.design_info
    design_cols = list(X.columns)

    # lifelines uses different block names per AFT distribution:
    # Weibull -> 'lambda_', LogNormal -> 'mu_', LogLogistic -> 'alpha_',
    # GeneralizedGamma -> 'mu_'. Pick the block whose covariate index
    # exactly covers the patsy design columns (the location-scale linear
    # predictor block) rather than hard-coding a name.
    design_set = set(design_cols)
    blocks = list(params.index.get_level_values(0).unique())
    location_block: str | None = None
    for blk in blocks:
        blk_cov = [str(c) for c in params.loc[blk].index]
        if set(blk_cov) == design_set:
            location_block = blk
            break
    if location_block is None:
        raise ValueError(
            "from_aft: no params_ block matches the patsy design columns. "
            f"Blocks = {blocks}; design = {design_cols}. lifelines may have "
            "an unfamiliar parameterisation; pass an explicit formula= or "
            "open an issue."
        )

    lam = params.loc[location_block]
    beta = np.asarray(lam.to_numpy(), dtype=float)
    beta_names = [str(c) for c in lam.index]

    V = fitter.variance_matrix_
    if hasattr(V, "loc"):
        vcov = np.asarray(
            V.loc[location_block, location_block].to_numpy(), dtype=float
        )
    else:
        # Fallback: top-left k×k block of a plain array.
        vcov = np.asarray(V, dtype=float)[:len(beta), :len(beta)]

    # Align beta / vcov to patsy's column order (lifelines may emit them in
    # a different order; reorder defensively).
    if design_cols != beta_names:
        idx = [beta_names.index(c) for c in design_cols]
        beta = beta[idx]
        vcov = vcov[np.ix_(idx, idx)]
        beta_names = design_cols

    factor_infos = getattr(design_info, "factor_infos", {}) or {}
    factors: dict[str, list[str]] = {}
    numeric_means: dict[str, float] = {}
    for f, fi in factor_infos.items():
        name = f.name()
        if getattr(fi, "type", None) == "categorical":
            factors[name] = list(fi.categories)
        elif name in data.columns and pd.api.types.is_numeric_dtype(data[name]):
            numeric_means[name] = float(data[name].mean())

    n_obs = len(data)
    # Match R survreg's reporting: df = n − (location + scale params). For
    # lifelines AFT that's len(lambda_) + len(rho_/sigma_/...) — total of
    # params_, which includes lambda_ and the ancillary block.
    n_params = len(params)
    df_resid = float(max(n_obs - n_params, 0))

    return ModelInfo(
        beta=beta,
        vcov=vcov,
        param_names=beta_names,
        factors=factors,
        numeric_means=numeric_means,
        df_resid=df_resid,
        design_info=design_info,
        data=data.copy(),
        response_name=f"log_{getattr(fitter, 'duration_col', 'T')}",
        family=None,
        scale=1.0,
        raw_result=fitter,
    )
