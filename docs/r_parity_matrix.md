# pymmeans vs R `emmeans` — feature parity matrix

Function-by-function map between `pymmeans` (Python) and R's `emmeans` /
`lsmeans` / `pbkrtest` packages.

## Beyond-R-parity features (pymmeans-only)

Features `pymmeans` ships that R `emmeans` does not have:

- **`from_predict(predict_fn, data, factors=, numerics=)` +
  `ml_emmeans` + `ml_pairs`** — marginal means for any ML model with a
  `.predict()` method (sklearn / xgboost / lightgbm / pytorch /
  custom) via prediction-surface averaging (g-computation). R
  `emmeans` is tied to R's S4 model-class system and does not handle
  generic prediction-only models. `bootstrap_ci(kind="case",
  refit_fn=...)` gives proper CIs by resampling, refitting, and
  recomputing the EMM.
- **`bootstrap_ci(obj, kind="case")`** — true non-parametric case-
  resampling bootstrap on `EMMResult` and `ContrastResult`. R
  requires manual `boot::boot` integration.
- **`permutation_test(contrast, n_permutations=)`** — label-shuffle
  p-values with Phipson-Smyth correction. No R `emmeans` equivalent.
- **`eta_squared(model)`** — per-term partial η² / Hays' ω² /
  Cohen's f with 90 % noncentral-F confidence intervals, in a single
  call alongside `joint_tests`. R `emmeans` requires a separate
  `effectsize` package call; `pymmeans` bakes it in. Bit-exact match
  with R `effectsize::eta_squared` on canonical references.
- **`effect_size(c, measure="odds_ratio" | "risk_ratio" |
  "hazard_ratio")`** — exponentiated effect-size measures for
  binomial-logit, log-link Poisson, and Cox PH contrasts, with
  delta-method SE and Wald-then-exp CIs.
- **`pairs(emm, max_contrasts=50)`** — guard against the silent
  multi-factor-pairwise explosion footgun. R's `pairs()` on a
  (5, 4, 3)-level model silently returns 1 770 contrasts;
  `pymmeans` raises with explicit escape-hatch options.
- **`pairs(emm, simple="each" | "factor" | list)`** — per-factor
  decomposition that mirrors R's `pairs(simple=)` semantic but
  also returns a proper `EmmList` keyed by factor name.
- **`bootstrap_ci(emm, method="streaming")`** — Jain & Chlamtac P²
  online percentile estimator for memory-bounded parametric
  bootstrap CIs (any `n_samples` in constant memory).
- **Pandas-first API** — `.frame` is a tidy DataFrame at every
  layer. No S4 → `data.frame` coercion step.
- **Thread-isolated `emm_options`** — ContextVar-based; each thread
  (including `concurrent.futures.ThreadPoolExecutor` workers) sees
  its own option state, and does not inherit the submitter's
  context.

## Legend

- ✅ **Full parity** — same behaviour, same defaults, same output shape
- 🟡 **Partial** — exists but differs in defaults, semantics, or output column names
- ❌ **Missing** — not in `pymmeans`
- N/A — irrelevant to the Python ecosystem

## Core API

| R feature | pymmeans status | Notes |
|---|---|---|
| `emmeans(object, specs, by, at)` | ✅ Full | R defaults match; options system available |
| `pairs()` | ✅ Full | Tukey default; per-by family adjustment |
| `contrast(emm, method=..., interaction=, simple=, combine=)` | ✅ Full | Default `"eff"` (R parity); 13 named methods plus callable `method=`, `interaction=` (Kronecker contrasts), `simple=` / `combine=` per R `contrast.emmGrid(..., simple=, combine=)` |
| `summary(emm, infer=, level=, side=, null=, delta=, type=, adjust=, bias_adjust=, sigma=, by=)` | ✅ Full | All keyword arguments supported; response-scale stamping and `emm_options` propagation match R |
| `confint(emm, level=, side=, adjust=, bias_adjust=, sigma=)` | ✅ Full | Default `level=0.95`. Strict R parity: `level=None` is 0.95 unconditionally and does not consult `emm_options(level=...)` |
| `test(emm, null=, side=, delta=)` | ✅ Full | `side` aliases include `noninferiority` and `nonsuperiority` |
| `update(emm, **kwargs)` | ✅ Full | `type=` reroutes through the inverse-link helper; `adjust=` recomputes p-values via the family-aware re-adjustment path |
| `joint_tests(model, by=...)` | ✅ Full | `by=` supported; by-cell ordering matches R `expand.grid`. `show0df`, `cov.reduce` are display-side knobs deferred to v0.2 |
| `emtrends(model, var=, delta_var=)` | ✅ Full | `delta_var=` alias supported; `max_degree=` (higher-order polynomial trends, k≤4) supported with R's Taylor-coefficient convention (jss_audit §XXIX); estimability check returns NaN (R: `nonEst`) for constant `var` |
| `cld(emm)` | ✅ Full | Piepho (2004) algorithm; by-groups, `alpha=`, `reverse=` |
| `pwpp(emm, method=, sort=, values=)` | ✅ Full | Any contrast method (not just pairwise); `sort=` for ascending tick order; `values=` annotates ticks with numeric estimates |
| `pwpm(emm, type=)` | ✅ Full | Matrix display; `type='response'` produces a ratio matrix for log-family models |
| `emmip(model, formula, PIs=, dodge=, CIs=, plotit=)` | ✅ Full | `PIs=True`, `dodge=`, `plotit=False` returning R-style DataFrame, `CIs=` as R alias for `show_ci=`. `abbr.len` deferred to v0.2 |
| `plot(emm)` | 🟡 Partial | Basic forest plot; `comparisons=`, `sep=` missing |
| `ref_grid(model)` | ✅ Full | `nesting=` / `nuisance=` supported; `cov_reduce=` accepts a bare callable (applied to all numeric covariates, R-style `cov.reduce = median`) or a per-column dict, on both `ref_grid()` and `emmeans()`. Validated in jss_audit §XXX. |
| `regrid(emm, transform=, N.sim=)` | ✅ Full | R-style wrapper; aliases `"response"` / `"mu"` / `"unlink"` route to `regrid_response`; `"pass"` / `"none"` / `None` are no-ops. `n_sim=` gives R's `N.sim=` simulation-based regrid (draw from `MVN(β̂, V)`, no MCMC / no refit), with optional `hpd=` and `random_state=`; converges to Wald on the identity scale and gives the correct asymmetric interval on nonlinear back-transforms (jss_audit §XXXIII) |
| `eff_size(emm, sigma=, edf=, method=)` | ✅ Full | Cohen's d + Hedges' g; emits R-style `effect_size` / `effect_size_SE` / `effect_size_lower_cl` / `effect_size_upper_cl` columns |
| `make.tran(type, ...)` | ✅ Full | R aliases: `asin.sqrt`, `log+1`, `sqrt+.5`, `identity` |
| **`lsmeans` R package coverage** (v2.30-2) | ✅ Full for v0.1 surface | See dedicated section below |
| `as.data.frame(emm)` | ✅ Full | `result.frame` (DataFrame already) |
| `as_r_frame(emm)` | ✅ Full | Dot-name rename plus family-specific value-column renames (`prob` Binomial, `rate` Poisson, `response` Gamma / Inverse-Gaussian / log-LHS OLS) and `asymp.LCL` / `asymp.UCL` / `z.ratio` for `df=inf` Wald frames |
| `emm_options(...)` | ✅ Full | ContextVar-based, thread-isolated |
| `qdrg(formula, data, coef, vcov, df)` | ✅ Full | Builds a `ModelInfo` from raw inputs via patsy; mirrors R's `qdrg` signature |
| `emmobj()` / `as.emmGrid()` | ✅ Full | `emmobj(bhat, V, levels)` — formula-less low-level constructor matching R `emmobj` |
| `rbind(c1, c2, ..., adjust=)` | ✅ Full | Concatenates `ContrastResult`s and applies a joint multiplicity adjustment over the combined family |
| `emm_list` | ✅ Full | `EmmList` named / positional container; `summary` / `confint` / `test` recurse; `as_r_frame(EmmList)` combines members into one DataFrame |
| `as.glht()` / `as.mcmc()` | ❌ Missing | R-specific adapters |
| `hpd.summary()` (HPD credible intervals) | ✅ Full | `posterior_emmeans(..., hpd=True)` / `posterior_emm_summary(..., hpd=True)` report highest-posterior-density intervals via the Chen–Shao algorithm; matches `arviz.hdi` to the bit (jss_audit §XXXII) |
| `mvcontrast()` | ✅ Full | `pymmeans.mvcontrast` on a `MultivariateEMM` from `multivariate_emmeans(_MultivariateOLSResults, …)`; Hotelling T² / F per between-contrast, Sidak default; matches R machine-precision (see jss_audit §VII.5). `mvregrid()` still missing. |
| `add_grouping`, `comb_facs`, `split_fac`, `permute_levels` | ✅ Full | Grid manipulation utilities. All four are exported from ``pymmeans`` and covered by ``tests/test_public_api_smoke.py``. |
| `nesting=`, `nuisance=` kwargs on ``emmeans()`` | ✅ Full | Auto-nesting detection plus user-supplied dict; nuisance-over-weights override on all four weight schemes (see `src/pymmeans/emmeans.py`). |
| `counterfactuals=` | 🟡 Partial | Per-cell counterfactual frequency override is implemented for the eager path; the streaming-eager auto-switch covers it. R's full counterfactual scaffolding (e.g. `weights="cells" × by="..."` interactions with non-trivial nesting) is not exhaustively tested. |

## Contrast methods (R emmc family)

| R method | pymmeans status |
|---|---|
| `pairwise` | ✅ Full |
| `revpairwise` | ✅ Full |
| `tukey` | ✅ Full (alias for pairwise + Tukey adjust) |
| `dunnett` / `trt.vs.ctrl` | ✅ Full (default adjust = `dunnettx`) |
| `trt.vs.ctrl1` | ✅ Full |
| `trt.vs.ctrlk` | ✅ Full |
| `poly` | 🟡 Partial — integer-scaled via `Fractions`, exact through k=20. For k > 20 the LCM of denominators blows up to ~1e6 and `pymmeans` falls back to a stable orthonormal output. Semantically valid contrasts but not the exact integer coefficients R emits |
| `consec` | ✅ Full (default adjust = `mvt`) |
| `mean_chg` | ✅ Full (default adjust = `mvt`) |
| `eff` | ✅ Full (default adjust = `fdr`) |
| `del.eff` | ✅ Full (default adjust = `fdr`) |
| `identity` | ✅ Full |
| `helmert` | ✅ Full |
| `opoly` | ✅ Full | Call ``pymmeans.opoly(k, kind="orthonormal")`` for the R ``emmeans::opoly`` (unit-row-norm) form. Default ``kind="integer"`` returns the R ``poly.emmc`` integer-scaled form for backward compatibility. |
| `nrmlz` | ❌ Missing (normalization wrapper; v0.2) |
| `wtcon` | ❌ Missing (weighted contrasts; v0.2) |
| Custom `method=function(levs, ...)` | ✅ Full — callable accepted; returns dict, DataFrame, or ndarray. Default `adjust="none"` matches R `.emmc_*` semantics |
| `interaction=` (Kronecker contrasts) | ✅ Full |
| `simple=` / `combine=` | ✅ Full on `contrast()` (and on `pairs()`) |

## Multiplicity adjustments

| R adjust | pymmeans status |
|---|---|
| `none` | ✅ Full |
| `bonferroni` | ✅ Full |
| `holm` | ✅ Full |
| `sidak` | ✅ Full |
| `tukey` | ✅ Full (faster than SciPy at large k) |
| `dunnett` | ✅ Full (exact Genz QMC; deterministic across calls) |
| `mvt` | ✅ Full — two-sided MVT with singular-correlation ridge regularisation; one-sided / TOST uses R's exact `.my.pmvt` algorithm. Matches R `mvtnorm::pmvt` to 3–4 decimals on the reference fits |
| `dunnettx` | ✅ Full (R `.pdunnx` mixture, ported — no longer an exact-Dunnett alias) |
| `BH` / `fdr` | ✅ Full |
| `BY` | ✅ Full |
| `hochberg` | ✅ Full |
| `hommel` | ✅ Full |
| `scheffe` | ✅ Full (`df=inf` → χ² limit) |
| `cross.adjust` | ✅ Full | `summary(..., cross_adjust=)` applies the method to each contrast's by-group family of size G (bonferroni ×G, Šidák 1−(1−p)^G); matches R to ~1e-11 (jss_audit §XXX). |
| `side=` / `null=` / `delta=` | ✅ Full |

## Transforms

| R name | pymmeans | Notes |
|---|---|---|
| `log`, `log10`, `log2`, `log1p` | ✅ Full | Plus auto-detect from patsy LHS |
| `sqrt` | ✅ Full | |
| `genlog`, `boxcox` | ✅ Full | Picklable via `functools.partial` |
| `logit`, `probit`, `cloglog` | ✅ Full | |
| `asin.sqrt` / `asin_sqrt` | ✅ Full | Both spellings accepted |
| `scale` | ✅ Full | Linear contrast back-transform too |
| `log+1`, `log+.5`, `sqrt+.5`, `+.5` | ✅ Full | |
| `identity` | ✅ Full | Explicit no-op |
| `power`, `sympower`, `atanh`, `asin_sqrt`, `bcnPower`, `yj.power` | ✅ Full | Parametric power transforms via `make_tran(...)`. `power` matches R `make.tran('power')` exactly; `bcnPower` matches `car::bcnPowerInverse` (Hawkins–Weisberg); `yj.power` round-trips through `car::yjPower`; `atanh`/`asin_sqrt` are in the auto-detect registry. `sympower` back-transformed estimates match R exactly, but its derivative is the mathematically-correct `(1/λ)|z|^{1/λ−1}` — R `emmeans`' `sympower` `mu.eta` drops the `1/λ` factor (an emmeans bug), which pymmeans intentionally does not replicate. Validated in jss\_audit §XXVIII. |

## Model classes

| R model class | pymmeans status | Notes |
|---|---|---|
| `lm` | ✅ Full | OLS / `smf.ols` |
| `glm` (canonical and non-canonical links) | ✅ Full | |
| `lmer` / `glmer` | ✅ Full | `statsmodels.MixedLM` |
| Satterthwaite df | ✅ Full | `atol < 1e-3` vs `lmerTest` (finite-difference Hessian noise dominates) |
| Kenward-Roger | ✅ Full | Six headline `pbkrtest` user-facing functions ported (`vcovAdj`, `getKR`, `Lb_ddf`, `KRmodcomp`, `SATmodcomp`, `PBmodcomp`). Matches `pbkrtest::vcovAdj` at `atol < 1e-4` on the random-slopes sleepstudy fit; random-intercept fits have a documented ~2.6% residual on the intercept SE (see `docs/vs-r.md`) |
| `aov` (single stratum) | ✅ Full | Through OLS adapter |
| `aovlist` (multi-stratum) | ❌ Missing | Use MixedLM equivalent |
| `geeglm` / GEE | ✅ Full | `statsmodels.GEE` |
| `gls` / `wls` | ✅ Full | |
| `coxph` / `coxme` | ✅ Full | `statsmodels.PHReg` |
| `survreg` (parametric AFT) | ❌ Missing | v0.2 candidate; `statsmodels`' parametric AFT path is incomplete |
| `polr` / `clm` / `clmm` (ordinal) | ✅ Full | `statsmodels.OrderedModel` adapter + `ordinal_emmeans(fit, mode="prob"/"cum.prob"/"exc.prob"/"mean.class")`. Matches R `ordinal::clm + emmeans` at `atol < 1e-5` on probabilities and SEs |
| `multinom` (multinomial logit) | ✅ Full | `statsmodels.MNLogit` adapter + `multinom_emmeans(fit, mode="prob"/"latent")`. Matches R `nnet::multinom + emmeans` at `atol < 1e-4` |
| `betareg` | ✅ Full | `statsmodels.othermod.BetaModel` |
| `gam` / `mgcv` (smooth terms in formula) | 🟡 Partial — `bs(x, ...)` | Multi-column basis expressions (`bs(x, df=3)`, `cr(x, df=4)`, spline × factor interactions) match R at `atol < 1e-4` via patsy's `FactorInfo.eval`-based re-evaluation. Patsy does not expose `ns(x, ...)` or R's `poly(x, ...)` natively — use `cr(x, df=4)` or pre-compute orthogonal polynomial columns. Full `mgcv` smooth-term machinery is a v0.2+ piece (`gam` model class itself is not in `statsmodels`) |
| Multivariate (`mvmodel`) | ❌ Missing | v0.2+; no Python ecosystem analogue |
| `brms` / `rstanarm` | N/A (R-only) | PyMC posterior path exists |
| `linearmodels` `PanelOLS` / `IV2SLS` | ✅ Full | Via `from_linearmodels(result, data=df)` |
| `statsmodels.othermod.BetaModel` | ✅ Full | |
| Discrete `Probit` / `Logit` / `NegBin` | ✅ Full | |

## EMMResult / ContrastResult arithmetic

| Feature | pymmeans status |
|---|---|
| `+`, `-`, `*`, `/` on results | ✅ Full |
| Identity check + structural-equality fallback | ✅ Full |
| Label-alignment refusal | ✅ Full |
| Refuse on response / posterior | ✅ Full |
| Pickle round-trip via Satterthwaite cache | ✅ Full |
| `rbind` (stack different grids) | ✅ Full — `rbind(c1, c2, ..., adjust="bonferroni")` |
| `as.list()` / `emm_list` | ✅ Full — `EmmList` named / positional container |

## `lsmeans` R package coverage

The `lsmeans` R package (CRAN v2.30-2) is officially a transitional
front-end to `emmeans` — its CRAN help topic `transition` literally
states "Users may use **emmeans** in almost exactly the same way as
**lsmeans**." For users porting code that still references the older
`lsmeans` namespace (a substantial body of published agronomy /
pharma / social-science analyses), `pymmeans` provides every name
`library(lsmeans)` exposes, modulo the Python PEP 8 dot-to-underscore
translation.

| R `lsmeans` name | pymmeans | Mechanism |
|---|---|---|
| `lsmeans(model, "g")` | `lsmeans(model, "g")` | Identity alias for `emmeans` |
| `lsm(model, "g")` | `lsm(model, "g")` | Short-form alias for `emmeans` |
| `lstrends(model, "g", var="x")` | `lstrends(model, "g", var="x")` | Identity alias for `emtrends` |
| `lsmip(model, ~ a \| b)` | `lsmip(model, x="a", by="b")` | Identity alias for `emmip` (formula → kwargs) |
| `contrast(emm_or_lsm, ...)` | `contrast(emm, ...)` | Same function in `pymmeans` |
| `lsm.options(level=0.99)` | `lsm_options(level=0.99)` | Identity alias for `emm_options`; same backing ContextVar |
| `get.lsm.option("adjust")` | `get_lsm_option("adjust")` | Identity alias for `get_emm_option` |
| `ref.grid(model)` | `ref_grid(model)` | Same function, PEP 8 underscore name |
| `recover.data(model)` | (via adapter protocol) | Not user-facing; the equivalent is `from_fitted` / the `register_adapter` API |
| `lsm.basis(model)` | (via adapter protocol) | Same; not part of the v0.1 public surface (an internal extension hook) |
| `lsmobj(bhat, V, levels)` | ✅ via `emmobj` | `pymmeans.emmobj(bhat, V, levels)` is the formula-less low-level constructor. R's `lsmobj` is a transitional alias for `emmobj` per the `lsmeans` `transition` help topic |

`pymmeans` covers the **lsmeans analysis-facing API** — the script-
level functions a user writes in their analysis (the seven names in
the upper block above). `lsmeans(ref_grid(fit, at=...), "factor")`
workflows run end-to-end (matching the CRAN `fiber` example from
Lenth's documentation). The remaining lower-block items
(`lsmobj`, `lsm.basis`, `recover.data`) are low-level constructor /
extension hooks that v0.1 routes through the adapter protocol.

## Validation

- 313 public-surface unit tests pass, 1 skipped, 0 failed
- Cross-validated against R `emmeans` via 18 reference comparisons
  across 12 distinct fits in `tests/test_r_benchmark.py` (committed
  CSVs; no R required in CI)
- Poly contrasts: matched R `emmeans::poly.emmc` integer-scaled
  values exactly for k = 3..20 (k > 20 uses orthonormal fallback)
- Satterthwaite SE: matches `lmerTest::lmer + emmeans` at `atol < 1e-4`
- `mvt` / `dunnett` p-values are deterministic across identical |t|
- Response-scale `summary` / `test` run on the link scale and match
  a direct `emmeans(type='response')` round-trip to `rtol < 1e-10`
  for `emmean`, `SE`, `lower_cl`, `upper_cl`

## Numeric parity score

| Tier | Count |
|---|---|
| ✅ Full | 87 |
| 🟡 Partial | 3 |
| ❌ Missing | 10 |

**87 / 100 strict parity on the public surface (90 / 100 if Partial
counts).** The remaining 10 ❌ items are distributed as:

- **No-Python-equivalent model classes** (4): `aovlist`, `survreg`,
  `mvmodel`, `glmmTMB` / `MCMCglmm`
- **R-specific glue** (3): `as.glht`, `as.mcmc`, `brms` adapter
- **Niche power-user features** (5): `cross.adjust`, grid-manipulation
  utilities (`add_grouping`, etc.), `nesting=` / `nuisance=` /
  `counterfactuals=`

For the typical applied-stats workflow — OLS, GLM, MixedLM
(Satterthwaite / Kenward-Roger / parametric-bootstrap F-test
inference), GEE, Cox, Beta, ordinal, multinomial, B-spline formulas,
`linearmodels` panel, survey-weighted designs, PyMC posteriors —
`pymmeans` matches R `emmeans` + `pbkrtest` at `atol = 1e-4` (and
tighter in many subsystems; see the per-row tolerances above).
Documented residuals: the Kenward-Roger random-intercept intercept SE
differs from `pbkrtest` by ~2.6% on the canonical reference fit
(see `docs/vs-r.md`); Satterthwaite df match within 6% across the
broad test surface.
