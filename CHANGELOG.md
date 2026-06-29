# Changelog

All notable changes to `pymmeans` will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.21.0] — 2026-06-28

### Added — explicit estimand / "which average?" decision aid (wish-list #4)

- **`estimands(model, specs, by=)`** — computes the EMM of `specs` under each
  marginalisation scheme (`equal` / `proportional` / `cells`) side by side, as
  a tidy DataFrame with one `emmean[scheme]` column per scheme (and
  `SE[scheme]` with `se=True`). `emmeans` (like R) defaults to *equal*
  weighting — the balanced / experimental estimand — which for observational
  data is often the wrong target; these schemes diverge substantially on an
  imbalanced design (Heiss 2022; the `ggeffects` `marginalmeans`-vs-`empirical`
  distinction) and coincide on a balanced one. This makes the estimand choice
  explicit and its consequences visible — the deepest critique of EMMs,
  addressed (`docs/wishlist.md` #4).
- **`EMMResult.describe()`** now interprets the marginalisation estimand
  (e.g. "weights: equal — balanced / experimental estimand") and, for the
  equal-weight default, points to `estimands()` to compare the alternatives.

## [0.20.0] — 2026-06-28

### Added — self-describing results (wish-list sweetener cluster)

- **`EMMResult.describe()`** — a one-call, plain-English account of what an
  EMM actually is, surfaced on demand without changing the default frame
  output. It reports:
  - the **active scale** (link vs response; a logit/GLM link scale is
    explicitly labelled "NOT the response scale");
  - which factors were **averaged over** and with what **weights**;
  - which **covariates are held fixed** and at what value;
  - a **flag when an averaged-over factor interacts with the target**, with a
    `by=` suggestion (the EMM can otherwise be misleading);
  - **named non-estimable cells** ("g=C, blk=y") instead of a bare `NonEst`;
  - **comparison guidance** (use `pairs()`/`contrast()`; don't read
    significance from overlapping CIs).

  These target the largest cluster of R `emmeans` user confusions catalogued
  in `docs/wishlist.md` (FAQ entries #5/#7/#12/#14/#16/#18/#19). Neither R
  `emmeans` nor `marginaleffects` ships an equivalent. `docs/wishlist.md`
  updated to mark sweeteners #1/#3/#5/#6 shipped.

## [0.19.0] — 2026-06-28

### Changed — mistake-catching guidance (wish-list sweetener)

- The error raised when a **numeric focal covariate** is used as `specs`/`by`
  without `at=` now **names the most likely fix**. When the covariate has few
  distinct values — the signature of a 3+-level factor coded as a number,
  which is R `emmeans`' single most-repeated user confusion (it silently
  returns one estimate; the author declined to warn, rvlenth/emmeans #523) —
  the message leads with `C(x)` / `pd.Categorical`. A genuinely continuous
  covariate keeps the `at=` / `emtrends` guidance, with no spurious factor
  hint. First of the `docs/wishlist.md` "self-describing result" sweeteners
  (a sourced audit of emmeans community pain points, added this release).

## [0.18.0] — 2026-06-28

### Added / R-emmeans parity

- **`vcov=` now accepts an R-style callable** (`vcov.=fn`). In addition to a
  covariance matrix, `emmeans(..., vcov=fn)` evaluates `fn` on the fitted
  model to obtain the robust / sandwich / cluster-robust covariance — e.g.
  `vcov=lambda m: m.get_robustcov_results("HC3").cov_params()`. Matches the
  explicit-matrix form exactly.

### Documentation / validation

- Documented and validated the **robust / cluster-robust EMM** path
  end-to-end: a statsmodels fit with `cov_type="cluster"`/`"HC3"`
  auto-propagates its robust covariance to the marginal means *and* their
  contrasts, with the marginal SE equal to the sandwich identity
  `sqrt(diag(L V Lᵀ))` to machine precision. New parity-matrix row, new
  tests (`tests/test_robust_vcov.py`), and validation-notebook Section
  XXXIV. The executed notebook records 340 contracts (176 cross-validation
  + 119 structural + 45 Monte-Carlo), zero failures.

## [0.17.0] — 2026-06-28

### Added / R-emmeans parity

- **Simulation-based regrid** (R `regrid(object, N.sim=)`).
  `regrid(em, n_sim=k)` draws `k` samples from the asymptotic
  `MVN(β̂, V)` of the fitted coefficients and reports sample-based
  intervals — no MCMC, no refit. On the identity scale
  (`transform="pass"`) it converges to the analytic Wald interval; on a
  nonlinear back-transform (`transform="response"`) it gives the correct
  asymmetric interval (e.g. logit response intervals lie in `(0, 1)`,
  with the simulated mean equal to `E[plogis(η)]`) where the delta method
  cannot. Optional `hpd=True` (HPD intervals on the simulated sample) and
  `random_state=` (reproducible draws). Reuses the posterior-summary
  machinery; supported for a link-scale `EMMResult`.

### Fixed

- Validation-notebook scorecard: three Monte-Carlo checks (the
  bootstrap-coverage contract in §XXVII plus the two new §XXXIII
  simulation contracts) were tagged `"Monte-Carlo"` but the categoriser
  matched `"Monte Carlo"`, so they were silently counted as
  cross-validations. Normalised the label; the counts now read 176
  cross-validation + 116 structural + 45 Monte-Carlo.

### Documentation

- R-parity matrix: `regrid` row documents the new `N.sim=` capability.
  New validation-notebook Section XXXIII; the executed notebook records
  337 contracts, zero failures.

## [0.16.0] — 2026-06-28

### Added / R-emmeans parity

- **HPD credible intervals** (R `emmeans`'s `hpd.summary`).
  `posterior_emmeans(..., hpd=True)` and
  `posterior_emm_summary(..., hpd=True)` now report highest-posterior-
  density intervals — the narrowest interval carrying the requested
  probability mass — instead of the default equal-tailed percentile
  credible interval. Implemented with the Chen–Shao order-statistic
  algorithm; the endpoints match `arviz.hdi` (the reference HDI) to
  machine precision on skewed (log-normal, gamma) posteriors. For a
  skewed posterior the HPD interval is strictly narrower than the
  equal-tailed one; for a symmetric posterior they coincide. Default
  behaviour (`hpd=False`) is unchanged.

### Documentation

- R-parity matrix: `hpd.summary` split out of the R-adapter "missing" row
  and marked full. New validation-notebook Section XXXII; the executed
  notebook records 334 contracts (177 cross-validation + 115 structural +
  42 Monte-Carlo), zero failures.

## [0.15.1] — 2026-06-28

### Validation

- Locked in `effect_size`'s SD-uncertainty standard error against R
  `emmeans::eff_size`. The `effect_size_SE` column propagates uncertainty in
  the standardising SD via `edf` —
  `SE(d) = sqrt(SE_inf^2 + d^2 / (2 * edf))` — matching R to `~5e-11` on
  shared data across several `edf` values. This is the one capability the
  marginaleffects JSS paper concedes to `emmeans`; pymmeans matches it (and
  marginaleffects has no standardised-effect-size SE at all). New tests
  (`tests/test_effect_size_edf.py`) and validation-notebook Section XXXI; the
  executed notebook records 331 contracts (176 cross-validation + 113
  structural + 42 Monte-Carlo), zero failures. No behaviour change.

## [0.15.0] — 2026-06-28

### Added / R-emmeans parity

- **`cov_reduce=` now accepts a bare callable** on `emmeans()` and
  `ref_grid()`, applied to every numeric covariate (R-style
  `ref_grid(..., cov.reduce = median)`) — previously only a per-column
  `{name: callable}` dict was accepted. The EMM shifts by exactly
  `coef × (reduce(x) − mean(x))`. (Moves `ref_grid` to full parity.)
- **`cross_adjust=`** (second-stage multiplicity adjustment across `by`
  groups, in `summary()`) is validated against R `emmeans`'
  `cross.adjust`: `bonferroni` multiplies each within-group-adjusted
  p-value by the number of by-groups `G`, and `sidak` gives `1 − (1 − p)^G`
  — matching R to `~1e-11`. (Already implemented; the R-parity matrix had
  it wrongly listed as missing.)

### Documentation

- R-parity matrix corrected: `cross.adjust` and `ref_grid` `cov.reduce`
  moved to full parity (both had stale "missing"/"partial" entries). Strict
  parity is now **87/100 (90/100 with partial)**, with the README and paper
  reconciled. New validation-notebook Section XXX; the executed notebook
  records 329 contracts (175 cross-validation + 112 structural + 42
  Monte-Carlo), zero failures.

## [0.14.0] — 2026-06-28

### Added

- `emtrends(..., max_degree=k)` — higher-order polynomial trends (R
  `emtrends(max.degree=)`). Returns a trend for each degree 1..k (k ≤ 4)
  with a `degree` column (`linear`, `quadratic`, ...). The degree-d trend is
  the d-th derivative of the linear predictor divided by `d!` (R's
  Taylor/polynomial-coefficient convention, so a raw `y = b2 x²` fit reports
  `quadratic = b2`, not `2 b2`). Validated against R: the quadratic trend
  equals `b2` exactly, the cubic trend of a quadratic model is zero, and the
  degree-1 row is identical to the single-degree `emtrends` (the default
  path is unchanged). `response_derivative=True` remains single-degree only.

### Documentation

- Removed the "`max.degree > 1` deferred" note from the R-parity matrix's
  `emtrends` row. New validation-notebook Section XXIX; the executed
  notebook records 326 contracts (173 cross-validation + 111 structural + 42
  Monte-Carlo), zero failures.

## [0.13.0] — 2026-06-28

### Fixed

- **`make_tran("bcnPower", ...)`** computed the wrong transform. It applied
  a plain Box-Cox to a shifted response `y + gamma`, but the Box-Cox-with-
  Negatives transform (Hawkins & Weisberg 2017; R `car::bcnPower`) applies
  the Box-Cox to the *smoothed* response `s = 0.5(y + sqrt(y² + gamma²))`,
  with inverse `y = s − gamma²/(4s)` and a derivative that carries the
  smoothing term. The inverse and its derivative are corrected and now
  match `car::bcnPowerInverse` to `~1e-11`. Any response-scale
  back-transform that used `bcnPower` previously produced incorrect values.

### Validated / documented (R-emmeans parity)

- The parametric power transforms `power`, `sympower`, `yj.power`
  (Yeo–Johnson), and `bcnPower` are now cross-validated against R
  `emmeans::make.tran` / `car`: `power` matches `make.tran('power')`
  exactly; `yj.power` round-trips through `car::yjPower`; `bcnPower` matches
  `car::bcnPowerInverse`. `sympower`'s back-transformed estimates match R
  exactly, but pymmeans uses the mathematically-correct derivative
  `(1/λ)|z|^{1/λ−1}` — R `emmeans`' `sympower` `mu.eta` drops the `1/λ`
  factor (an emmeans bug), which pymmeans intentionally does not replicate.
- **Parity re-audit.** The R-parity matrix had undercounted: the parametric
  transforms (and `atanh`/`asin_sqrt`) were already implemented but listed
  as missing. Corrected to **85/100 strict (89/100 with partial)**, and the
  README/paper reconciled to the matrix (the audited source of truth).

### Documentation

- New validation-notebook Section XXVIII with the transform contracts; the
  executed notebook records 321 contracts (172 cross-validation + 107
  structural + 42 Monte-Carlo), zero failures.
- New `docs/vs-marginaleffects.md` competitive comparison; the user-facing
  showcase notebook refreshed with the v0.8–v0.12 surface (and a stale
  broken cell fixed).

## [0.12.0] — 2026-06-28

### Added

- `ml_avg_slopes()` / `ml_avg_comparisons()` — marginal effects for
  black-box predictive models (scikit-learn, gradient boosting, neural
  nets; anything with a `.predict()`). A black-box model exposes no
  coefficient covariance, so the delta method does not apply: the point
  estimate is numerical g-computation on `predict_fn`, and the standard
  error / percentile confidence interval come from a **pairs bootstrap**
  that resamples the data and refits via `refit_fn`. Without a `refit_fn`
  the point estimate is still returned, with a `NaN` standard error. The
  bootstrap is validated three ways: the point estimate equals the OLS
  coefficient for a linear learner (exactly); the bootstrap standard error
  recovers the analytic OLS standard error (within Monte-Carlo tolerance);
  and the 95% percentile-bootstrap interval achieves nominal coverage in a
  Monte-Carlo calibration study. This closes the last roadmap item — the
  black-box "reach" — that `marginaleffects` itself does not provide an
  inference path for.

  `ml_avg_slopes` is a numerical derivative and is meaningful only for a
  *smooth* `predict_fn`; for piecewise-constant tree ensembles use
  `ml_avg_comparisons` (a discrete change). For binary/categorical
  treatment effects, the efficient influence-function path
  (`cross_fit_ml_emmeans`, `aipw_ate`) remains available and is preferred
  when a propensity/outcome learner pair is on hand.

### Documentation

- New validation-notebook Section XXVII with the OLS-identity, analytic-SE-
  recovery, and Monte-Carlo coverage contracts for the bootstrap marginal
  effects; the executed notebook records 315 contracts (167
  cross-validation + 106 structural + 42 Monte-Carlo), zero failures.

## [0.11.0] — 2026-06-27

### Added

- `hypothesis=` argument on `avg_predictions` / `avg_slopes` /
  `avg_comparisons` — test linear combinations of the result's rows (e.g.
  whether a marginal effect differs across groups) with an *exact*
  delta-method standard error. Each row's Jacobian with respect to the
  coefficients is retained, so a contrast matrix `L` is applied as
  `sqrt(diag(L J V Jᵀ Lᵀ))`. Accepts the emmeans-style shortcuts
  `"pairwise"`, `"revpairwise"`, `"reference"` / `"trt.vs.ctrl"`,
  `"sequential"` / `"consec"`, or an explicit numeric contrast matrix. The
  contrast SEs match an independently reconstructed analytic delta method
  to machine precision.
- `transform=` argument on the same functions — back-transform the point
  estimate and confidence limits (e.g. exponentiating a log-odds-ratio
  contrast), with the standard error set to NaN on the transformed scale,
  matching `marginaleffects` and `emmeans`.

### Changed

- `by=` groups in `avg_predictions` / `avg_slopes` / `avg_comparisons` are
  now returned in factor-level order (categories for a Categorical, sorted
  otherwise), matching `emmeans`, rather than data-appearance order. Row
  *values* are unchanged.

### Documentation

- New validation-notebook Section XXVI with `hypothesis=` (exact contrast
  SE vs independent analytic delta) and `transform=` (vs `marginaleffects`)
  contracts; the executed notebook records 311 contracts (165
  cross-validation + 104 structural + 42 Monte-Carlo), zero failures.

## [0.10.0] — 2026-06-27

### Added

- `datagrid()` — build a counterfactual reference grid to pass as
  `newdata=`. Named variables are crossed (Cartesian product); every other
  predictor is held at a typical value (mean for numeric, mode for
  categorical), matching `marginaleffects.datagrid`.
- `newdata=` argument on `avg_predictions` / `predictions` / `avg_slopes` /
  `slopes` / `avg_comparisons` / `comparisons` — evaluate the estimand at a
  supplied grid (e.g. from `datagrid`) instead of averaging over the
  observed sample.
- Richer `avg_comparisons` / `comparisons` change grammar. `variables` now
  accepts a dict `{name: spec}` for per-variable change specifications; a
  numeric `spec` may be a number (centred step), `"sd"` / `"2sd"` (centred
  SD step), `"iqr"` (Q1→Q3), `"minmax"` (min→max), or a `(lo, hi)` pair; a
  categorical `spec` is the explicit list of levels to contrast. The
  `comparison` argument additionally accepts any callable `(hi, lo) ->
  value`. All specs reproduce `marginaleffects` (estimate and standard
  error) to machine precision, verified in-process.

### Documentation

- New validation-notebook Section XXV with `datagrid` / `newdata` and
  change-spec contracts; the executed notebook records 302 contracts (163
  cross-validation + 97 structural + 42 Monte-Carlo), zero failures.
- New API page for the `datagrid` module.

## [0.9.0] — 2026-06-27

### Added

- `avg_predictions()` / `predictions()` — average adjusted predictions, the
  third leg of the marginaleffects triad. The model's fitted value on the
  response or link scale, averaged over the observed sample or within `by`
  groups, with a delta-method standard error. On the link scale the result
  is exact (`X̄β`, SE `√(X̄VX̄ᵀ)`); for a logistic MLE fit the response-scale
  average prediction equals the observed outcome mean (score-equation
  calibration). Estimate and SE reproduce `marginaleffects.avg_predictions`
  to machine precision (committed `importorskip` cross-validation).
- `plot_predictions()` / `plot_slopes()` / `plot_comparisons()` — a
  plot-ready visual layer over the prediction/slope/comparison frames.
  `plot_predictions` draws an average-adjusted-prediction curve (numeric
  condition: line + confidence band; categorical: points + error bars) by
  g-computation over a grid of the focal variable; `plot_slopes` and
  `plot_comparisons` are forest plots of the averaged estimates with
  confidence intervals and a reference line. Matplotlib is imported lazily
  (the `[plot]` extra); the plotted point estimates equal the
  `avg_slopes` / `avg_comparisons` values they visualise.

### Documentation

- New validation-notebook Section XXIV with `avg_predictions` closed-form,
  calibration, and cross-validation contracts plus plot-layer structural
  checks; the executed notebook records 285 contracts (153
  cross-validation + 90 structural + 42 Monte-Carlo), zero failures.
- New API page for the `predictions` module.

## [0.8.0] — 2026-06-26

### Added

- `avg_comparisons()` / `comparisons()` — counterfactual comparisons
  (g-computation), the discrete-change companion to `avg_slopes`. For a
  numeric predictor the default is a centred unit change
  `mean[h(X(x+1/2)β) − h(X(x−1/2)β)]`; for a categorical predictor, each
  non-reference level versus the reference. The `comparison=` argument
  selects the function applied to the two averaged predictions:
  `difference` (default), `ratio`, `lnratio`, `lnor` (log odds ratio), or
  `lift`. Standard errors are the delta method (shared `_beta_jacobian`
  machinery). On a linear model the difference over a step `s` equals `s`
  times the OLS coefficient exactly; on a GLM all five comparison
  functions reproduce `marginaleffects.avg_comparisons` (estimate and
  standard error) to machine precision, verified in-process via a
  committed `importorskip` cross-validation test. Closes the last
  substantive capability gap against `marginaleffects`.

### Documentation

- New validation-notebook Section XXIII with closed-form and
  marginaleffects cross-validation contracts for `avg_comparisons`; the
  executed notebook records 277 contracts (151 cross-validation + 84
  structural + 42 Monte-Carlo), zero failures.
- New API page for the `comparisons` module; `marginaleffects` added as an
  optional test dependency.

## [0.7.0] — 2026-06-26

### Added

- `avg_slopes()` / `slopes()` — average marginal effects over the observed
  sample, on the link or response scale, with optional `by=` grouping.
  Closes the headline gap against `marginaleffects`: on a linear model the
  average slope and its standard error equal the OLS coefficient exactly
  (verified to `1e-9` / `1e-8`); on a logistic GLM the response-scale
  average marginal effect matches the closed-form `mean(p(1-p)beta)` to
  `6e-15`. Against R `marginaleffects::avg_slopes` the point estimate
  agrees to `~1e-9` and the standard error to within marginaleffects' own
  finite-difference tolerance (`~1e-7`); pymmeans reproduces the *exact*
  analytic delta-method standard error (to `~1e-12`), which the
  double-finite-differenced reference does not.
- `slopes()` returns per-observation marginal effects with full
  delta-method standard errors (the per-row Jacobian carries the inverse
  link's curvature term, not a point-evaluated approximation).
- `hypotheses()` — nonlinear `g(beta)` tests by the delta method with a
  finite-difference Jacobian. Coefficient-ratio standard errors match the
  closed-form ratio delta method and R `car::deltaMethod`.
- `from_pyfixest()` and `PyFixestAdapter` — coefficient-level support for
  `pyfixest` high-dimensional fixed-effects models. Within-fixed-effect
  coefficients, covariances, residual degrees of freedom (including the
  absorbed-fixed-effect dimensions, and asymptotic `z` inference for
  `fepois`), and their delta-method tests reproduce pyfixest's own output
  and a dummy-encoded `statsmodels` fit. Reference-grid operations
  (`emmeans`, `avg_slopes`, `ref_grid`) remain patsy-only and raise a
  clear, steering error on `pyfixest` fits.

### Documentation

- New validation-notebook Section XXII closing the `marginaleffects` gaps,
  with nine closed-form / structural contracts (tolerances `1e-6` to
  `1e-10`); the executed notebook records 260 contracts, zero failures.
- New API pages for the `slopes` and `hypotheses` modules.

## [0.6.0] — 2026-06-08

### Added

- Top-level `Makefile` providing canonical reproduction entry points
  (`make reproduce`, `make test`, `make notebook`, `make html`,
  `make benchmarks`, `make clean`) — closes the JSS replication-materials
  submission checklist item.

### Documentation

- Major README polish with badges, headline numbers, validation evidence
  pointers, and citation block.
- New `CONTRIBUTING.md` + `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1).
- New `.github/ISSUE_TEMPLATE/` (bug report, R-parity discrepancy, feature
  request) and `.github/PULL_REQUEST_TEMPLATE.md`.
- `CITATION.cff` refreshed with v0.6.0 metadata and full keyword list.

### Confirmed shipped (no code change)

- Hochberg and Hommel multiplicity adjustments (delegated to
  `statsmodels.multipletests`, verified against R `p.adjust` to
  floating-point precision).

## [0.5.0] — 2026-06-01

### Added — TABLE-1.8 #1 + #2 (double machine learning)

- `pymmeans.cross_fit_ml_emmeans()`: K-fold cross-fitted g-computation
  through `ml_emmeans`. Every prediction used in the marginal-mean
  average is out-of-sample with respect to the model that produced it
  (the structural property double machine learning depends on;
  Chernozhukov et al. 2018).
- `pymmeans.aipw_ate()`: doubly-robust augmented inverse-probability-
  weighted ATE estimator (Robins, Rotnitzky & Zhao 1994) with
  influence-function-based standard errors. Consistent if EITHER the
  outcome model OR the propensity model is correctly specified.
- `pymmeans.AIPWResult` public dataclass.
- `src/pymmeans/double_ml.py` module + `tests/test_double_ml.py`
  (15 unit tests).
- §XXI of the validation notebook: AIPW double-robust property
  verified by 200-replication Monte-Carlo; cross-fit structural
  correctness verified at machine precision. 8 new contracts.

## [0.4.0] — 2026-05-31

### Added — TABLE-1.8 #3 + #5 (conformal prediction)

- `pymmeans.split_conformal_pi()`: split-conformal prediction intervals
  on `ml_emmeans` cell predictions (Vovk et al. 2005, Lei et al. 2018
  JASA). Distribution-free finite-sample marginal-coverage guarantee
  under exchangeability.
- `pymmeans.conformal_counterfactual_pi()`: weighted split-conformal
  counterfactual prediction intervals (Lei & Candès 2021 JRSSB
  Algorithm 1). Valid coverage of unobserved `Y(t*) | X` even at units
  where `t*` was not observed.
- `pymmeans.ConformalPIResult`, `pymmeans.ConformalCounterfactualResult`
  public dataclasses.
- `src/pymmeans/conformal.py` module + `tests/test_conformal.py`
  (22 unit tests).
- §XX of the validation notebook: split-conformal coverage at within
  ±0.006 of nominal across Gaussian, Student-t with 3 df, and
  contaminated errors; Lei-Candès counterfactual coverage of unobserved
  Y(1) at control-arm test units within ±0.005 of nominal.
  15 new contracts.

## [0.3.0] — 2026-05-30

### Added — TABLE-1.8 #7 + #8 (sensitivity + MI pooling)

- `pymmeans.e_value()`: VanderWeele-Ding (2017) E-value sensitivity
  statistic. Reproduces the VanderWeele-Ding smoking worked example
  (RR=10.73 → E≈20.95) to within 0.01. Conversions for odds ratios,
  hazard ratios, and standardised mean differences via Mathur et al.
  (2018).
- `pymmeans.pool_imputed()`: Rubin's (1987) rules for pooling EMM /
  contrast results across multiply-imputed datasets, with the
  Barnard-Rubin (1999) small-sample degrees-of-freedom correction.
  Total-variance identity verified at machine precision.
- `pymmeans.EValueResult`, `pymmeans.PooledImputationResult` public
  dataclasses.
- `src/pymmeans/sensitivity.py` + `src/pymmeans/imputation.py` modules
  + 27 unit tests.
- §XIX of the validation notebook: E-value closed-form match to
  published values; Rubin identity to machine precision; FMI in [0, 1].
  7 new contracts.

## [0.2.0] — 2026-05-28

### Added — applied case studies, performance benchmarks, ML-adapter showcase

- §XIV LaLonde NSW (Dehejia-Wahba 1999 experimental subset) applied
  case study: regression-adjusted ATE plus GBM g-computation via the
  ML adapter. Published DiM = \$1,794.34 recovered.
- §XV RHC (Connors et al. 1996 JAMA, n=5735) case study: logistic-GLM-
  adjusted risk difference plus GBM g-computation.
- §XVI IHDP (Hill 2011 JCGS semi-synthetic, n=672 × 10 reps) case study
  with KNOWN simulated true ATE. GBM g-comp RMSE = 0.17 vs OLS-adjusted
  RMSE = 1.59 — **9.5× tighter** through `from_predict`.
- §XVII SUPPORT2 (Knaus et al. 1995, n=9037) case study: Cox PH
  pairwise log-HR with Tukey adjustment. Cancer-highest-hazard
  ordering reproduced after covariate adjustment.
- §XVIII performance benchmarks vs R `emmeans` on six representative
  workloads. `pymmeans` 1.6-4.1× faster on four; the two slower paths
  (multivariate-t QMC adjustment and Cox PH) are diagnosed and
  explained as deliberate precision trade-off + Python ecosystem gap.
- Cross-`adjust` matrix-row Bonferroni rule (R parity fix); degenerate-
  posterior warning on `posterior_emmeans` when empirical SE is below
  the floating-point floor.

## [0.1.8] — 2026-05-27

### Fixed (memory)

- **`pairs(emm, max_contrasts=None)` no longer allocates an O(m²)
  contrast-covariance matrix** when the adjustment method does not
  need the off-diagonal correlation. Previously, every adjust path
  built the full ``L_c @ V @ L_c.T`` so it could fall through to
  the Dunnett correlation construction; for k=250 group levels via
  pairwise that is ~7.2 GB on float64, which OOM'd typical
  hardware. The contrast covariance is now built lazily: only the
  ``"dunnett"`` / ``"mvt"`` paths construct the full matrix, and
  every other adjustment computes the diagonal SEs via
  ``np.einsum("ij,jk,ik->i", L_c, V, L_c)`` — O(m·p) memory.
  Empirically at k=250, peak RSS dropped from ~7.7 GB to ~270 MB
  with identical numerical output. Tukey HSD at k=250 is now
  feasible on a laptop.

### Fixed (defaults)

- **``set_emm_options(dunnett_max_k=...)`` default tightened from
  100 to 50.** Empirical exact-Dunnett wall time on the reference
  machine: k=30 finishes in ~5 s, k=50 in ~4 min, k=100 in
  effectively unbounded time for an interactive call. The
  previous 100 cap suggested feasibility the QMC integrator can't
  deliver. Callers who genuinely need exact Dunnett at higher k
  (sensitivity analyses, batch jobs) can bump the cap explicitly;
  the steering error now also names empirical k-vs-wall-time
  numbers so the override decision is informed.

### Fixed (refusal completeness)

- **``joint_tests(EmmList(...))`` now scans every member's
  ``inference_kind`` instead of only the first.** The 0.1.7 fix
  unwrapped the first ``model_info``-carrying member; a mixed
  ``EmmList(freq_em, post_em)`` (frequentist first, posterior
  second) silently dispatched on the frequentist member and
  computed a Wald F / chi-squared on the bundle, leaving the
  posterior content unguarded. ``joint_tests`` now raises if ANY
  member has ``inference_kind='posterior'`` and names the offending
  positions in the message.

### Fixed (errors)

- **``bootstrap_ci(em, kind='case', refit_fn=...)`` now accepts an
  explicit ``data=`` kwarg** so the post-pickle path is actually
  usable. The 0.1.7 error message recommended supplying
  ``refit_fn=`` after a pickle round-trip, but the resampling step
  still required a non-empty DataFrame on ``info.data`` and the
  ``refit_fn`` was never reached. ``bootstrap_ci`` now resolves
  the DataFrame to resample in this order: live ``info.data``
  (live-fit path), otherwise the caller's ``data=`` kwarg
  (post-pickle path), otherwise a clear error naming all three
  fixes. Passing ``data=`` AND ``info.data`` populated emits a
  UserWarning and prefers ``info.data`` (the source the fit was
  actually conditioned on); silently overriding it would be a
  wrong-numerics footgun.

### Fixed (documentation)

- ``src/pymmeans/options.py`` module docstring now documents the
  process-boundary propagation gap explicitly. ``ContextVar`` is
  thread-safe but does NOT cross to ``joblib.Parallel`` workers,
  ``ProcessPoolExecutor`` workers, or any
  ``multiprocessing`` spawn-start child. The supported pattern
  (re-call ``set_emm_options(...)`` inside the worker, or use
  ``joblib.Parallel(initializer=...)``) is named. A matching
  warning admonition is added to ``set_emm_options``'s docstring
  so users hitting the new ``dunnett_max_k`` option in a parallel
  sensitivity-analysis pipeline see the limitation at the call
  site rather than at the resulting ``ValueError`` in the worker.

- ``CITATION.cff``, ``docs/index.md``,
  ``docs/PERFORMANCE_REPORT.md`` version stamps synced to 0.1.8.

## [0.1.7] — 2026-05-27

### Fixed

- **`joint_tests()` now refuses posterior-derived `EmmList` inputs.**
  The 0.1.4 posterior refusal landed on bare ``EMMResult`` /
  ``ContrastResult`` but missed the ``EmmList`` wrapper produced
  by ``emmeans(fit, pairwise ~ x)``-style formulas; the previous
  behaviour raised ``TypeError: No pymmeans adapter recognises
  EmmList``. ``joint_tests`` now unwraps the first ``model_info``-
  carrying member of an ``EmmList`` and routes it through the
  same ``inference_kind`` check, so posterior containers get the
  steering ``ValueError`` and frequentist containers proceed
  normally.

- **Cox PH advisory warning now fires on older statsmodels
  releases.** The 0.1.6 Cox detection imported ``PHReg``,
  ``PHRegResults``, and ``PHRegResultsWrapper`` in a single
  ``from ... import`` statement. ``PHRegResultsWrapper`` is absent
  on older statsmodels releases (e.g. 0.14.6); the missing symbol
  raised ``ImportError`` for the whole tuple and silently disabled
  the relative-log-hazard warning. Each name is now imported in
  isolation so a missing wrapper degrades only the wrapper
  detection — bare ``PHRegResults`` instances and
  ``.model``-wrapped results both warn as intended.

- **Cox class-name collisions in the adapter metadata.**
  ``src/pymmeans/utils.py`` still had two short-name string
  checks (``cls_name in ("PHRegResults", "PHReg")``) that routed
  any unrelated proxy class with one of those names into the
  Cox-PH code paths (synthetic ``np.log(<endog>)``
  ``response_name``, ``df_resid=inf`` override). Both call sites
  now go through a new ``_is_phreg(model)`` helper that does a
  module-qualified ``isinstance`` against
  ``statsmodels.duration.hazard_regression``, matching the 0.1.6
  fix to ``emmeans.py``.

- **``apply_kenward_roger`` on non-MixedLM fits now raises a
  ``ValueError`` instead of an ``AttributeError``.** Mirrors the
  ``apply_satterthwaite`` "MixedLM-only" refusal already in
  place. Pre-fix:
  ``AttributeError: 'OLSResults' object has no attribute
  'cov_re'`` deep inside the KR derivative chain.

- **``joint_tests()`` on a pickled EMM/contrast now raises a
  clear ``ValueError``.** Pickling drops the unpicklable patsy
  ``design_info``; ``joint_tests`` needs it to walk the term
  structure. The pre-fix error was
  ``AttributeError: 'NoneType' object has no attribute 'terms'``
  with no hint that pickling was the cause. The new error names
  the cause and lists three fixes (run before pickling, re-fit
  in the current process, or accept that hand-built EMMs are not
  supported).

- **``bootstrap_ci(em, kind='case', ...)`` on a pickled EMM now
  raises a clear ``ValueError``.** ``ModelInfo.__getstate__``
  replaces ``data`` with an empty DataFrame placeholder (not
  ``None``), so the pre-fix code path hit
  ``ValueError: Cannot bootstrap a zero-row dataset.`` — true
  but unhelpful. The new error names the pickle round-trip as
  the likely cause and lists three fixes (run before pickling,
  re-fit in the current process, supply ``refit_fn=`` with the
  original data closed over).

- **Exact Dunnett at large k is now bounded by a configurable
  safety cap.** ``adjust='dunnett'`` at k=200 (199 comparisons,
  199-dimensional MVT integral) ran past 70 s in benchmarks and
  was killed. The QMC integration cost in
  ``scipy.stats.multivariate_t.cdf`` scales roughly as
  ``O(maxpts * k)`` with ``maxpts`` itself ``50_000 * k``, so the
  curve becomes effectively unbounded for k ≥ ~150. ``_dunnett``
  now raises ``ValueError`` at k > 100 (overridable via
  ``set_emm_options(dunnett_max_k=...)``) and the message steers
  the user to ``adjust='dunnettx'`` (R's closed-form ``.pdunnx``
  approximation, finite-time at any k). A proper redesign that
  scales to arbitrary k is targeted for 0.2.0.

### Fixed (documentation)

- README's ``apply_satterthwaite`` / ``apply_kenward_roger`` rows
  previously claimed "any RE structure" coverage. The current
  implementation handles ``cov_re`` / ``re_formula=`` MixedLMs
  but raises ``NotImplementedError`` on crossed ``vc_formula=``
  random effects (statsmodels variance-component syntax). The
  README rows now state the actual coverage; the ``vc_formula``
  extension remains a 0.2.0 candidate.

- ``CITATION.cff``, ``docs/index.md``,
  ``docs/PERFORMANCE_REPORT.md`` version stamps synced to 0.1.7.

## [0.1.6] — 2026-05-26

### Fixed (statistical correctness)

- **`pairs(apply_kenward_roger(em))`** now matches
  `apply_kenward_roger(pairs(em))` to floating-point precision.
  The 0.1.4 KR-idempotency guard was tripping on freshly-built
  contrast results that *inherit* `df_method="kenward_roger"` from
  the source EMM but whose ``SE`` was still computed from the
  uncorrected ``V_beta`` — the guard short-circuited and the
  KR-corrected SE was never written. The contrast builder now
  initialises the new result with ``df_method="default"`` and lets
  ``_apply_correction`` stamp the final marker, so the propagation
  recomputes ``L_c @ V_KR @ L_c.T`` as intended. Same recipe for
  ``apply_satterthwaite``. Bug surfaced as ~1e-4 SE drift on
  ``pairs(em_kr)`` vs ``apply_kr(pairs(em))``.
- **`apply_satterthwaite` / `apply_kenward_roger` now refuse cross-
  correction inputs.** Previously
  ``apply_satterthwaite(apply_kenward_roger(em))`` silently rebuilt
  ``SE`` from the *uncorrected* ``V_beta`` (the Satt path doesn't
  carry the KR vcov inflation forward), discarding the KR step
  while keeping the Satt df — a hybrid that is neither correct nor
  documented. Both functions now raise ``ValueError`` with a
  steer-the-user message when fed a result already carrying the
  other small-sample correction; the same-method idempotency
  short-circuits are unchanged.
- **`joint_tests` refuses posterior-derived inputs.** The 0.1.4
  EMMResult dispatch path accepted any input that exposed
  ``model_info``, including posterior ``EMMResult``\\ s. The Wald
  F / chi-squared statistic assumes a frequentist sampling
  distribution of ``beta_hat`` and is meaningless on a posterior
  covariance (the test silently returned NaN or, in the
  ``df_denom=inf`` branch, a value that looked frequentist).
  Posterior inputs now raise ``ValueError`` and steer the user to
  ``posterior_emm_summary``.
- **Cox PH detection no longer false-positives on class-name
  collision.** 0.1.5 widened the Cox PH check to walk
  ``raw_result.__class__`` and ``raw_result.model.__class__`` but
  matched on bare class names (``{"PHReg", "PHRegResults",
  "PHRegResultsWrapper"}``), so any unrelated class with those
  short names triggered the relative-log-hazard warning. The
  check is now a module-qualified ``isinstance`` against
  ``statsmodels.duration.hazard_regression``, eliminating the
  collision while preserving the MI-pooled / proxy-wrapped result
  detection from 0.1.5.

### Fixed (API)

### Fixed (tooling)

- **`benchmarks/bench_performance.py`** now passes
  ``max_contrasts=None`` to ``pairs(emmeans(model, "group"))`` so
  the documented ``pairwise_k20`` … ``pairwise_k200`` rows in
  ``docs/PERFORMANCE_REPORT.md`` are reachable; the
  ``pairs()`` safety guard added in an earlier release was
  refusing every k ≥ 20 case and the report rows were stale.
  The runner also adds a failure-collection gate: if any scenario
  raises, the script now refuses to write a partial report rather
  than silently dropping rows.
- **Static analysis cleanup.** ``ruff check src/ tests/`` is clean
  (the residual ``RUF046`` ``int(len(...))`` cast in
  ``contrasts.py`` and the over-long ``removeprefix("~")`` ternary
  in ``qdrg.py`` are gone; the ``arviz`` availability probe in
  ``posterior.py`` is back to the noqa-suppressed
  ``import arviz`` form because the
  ``importlib.util.find_spec`` variant blew up on the
  ``monkeypatch.setitem(sys.modules, "arviz", SimpleNamespace())``
  pattern used in posterior tests).
- **Strict-warnings test fix.** The narrow
  ``filterwarnings`` decorator on
  ``test_pbmodcomp_bootstrap_more_conservative_in_small_sample``
  is now anchored on the ``"only N/M bootstrap iterations"``
  convergence message instead of catching every ``UserWarning``,
  so an unrelated warning regression in the same test would no
  longer be silently swallowed.

### Fixed (documentation)

- README banner version: ``v0.1.5`` → ``v0.1.6``;
  ``docs/index.md`` and ``docs/PERFORMANCE_REPORT.md`` synced.
  ``CITATION.cff`` version and date stamped to 0.1.6.
- README performance highlights table: ``~14× faster`` and
  ``~8× faster`` were rounded values from a different run;
  corrected to ``~11.5× faster`` and ``~7.7× faster`` so they
  match ``docs/PERFORMANCE_REPORT.md`` exactly.
- **Doctest collection is now clean.** ``ml.py::from_predict``
  had an unindented function body in its example (``... X_new =
  ...`` directly under ``>>> def predict(d):``) which crashed
  ``pytest --doctest-modules src/pymmeans`` at run time. The
  example is now ``+SKIP``-marked (the runtime path needs a real
  sklearn fit + function definition the doctest collector can't
  reasonably construct). ``emmeans.py``'s doctest gained
  ``+NORMALIZE_WHITESPACE`` so it tolerates pandas's column-
  alignment spacing. ``cld.py``'s example now seeds its RNG
  (``np.random.default_rng(0)``) so the printed ``.group``
  column is reproducible.
- **Empty parenthetical scrub artifacts removed from ~40
  docstrings and comments** spanning ``contrasts.py``,
  ``transforms.py``, ``plotting.py``, ``ml.py``,
  ``summary_layer.py``, ``summary.py``, ``emmeans.py``,
  ``posterior.py``, ``trends.py``, ``joint.py``,
  ``satterthwaite.py``, ``pwpm.py``, ``utils.py``. (Round /
  PR-number markers like ``()`` and ``(): foo bar`` left behind
  when audit-trail context was scrubbed for the public release.)

## [0.1.5] — 2026-05-26

### Fixed

- **`regrid_response(em)` on a Gaussian-identity GLM** is now a
  no-op (was: raised `ValueError: No transform recognised`). The
  identity-link path now resolves to `make_tran("identity")`
  instead of routing through `detect_transform("y")`, which only
  recognises function-call shapes. Matches R `emmeans::regrid`
  on `glm(y ~ x, family=gaussian())`.
- **Cox PH baseline-unidentified warning** now also detects the
  Cox model when it appears as `info.raw_result.model.__class__`,
  not only as `info.raw_result.__class__`. Wrapped result objects
  (e.g. MI-pooled, custom proxies) now trigger the warning as
  intended. Plain `PHRegResults` behaviour is unchanged.
- **Three tests now pass under `pytest -W error`** (strict CI):
  `test_response_for_ols_is_noop`, `test_response_scale_binomial_in_unit_interval`,
  and `test_pbmodcomp_bootstrap_more_conservative_in_small_sample`
  each carry a `@pytest.mark.filterwarnings(...)` decorator for
  the warning the test legitimately expects.
- **`qdrg("~ x", ...)`** (R-style RHS-only formula with leading
  tilde) is now accepted alongside the existing `"x"` and
  `"y ~ x"` forms.

### Fixed (documentation)

- README banner version: `(v0.1.3)` → `(v0.1.5)` (banner was
  unchanged across 0.1.3 → 0.1.4 → 0.1.5).
- `docs/index.md` and `docs/PERFORMANCE_REPORT.md` version stamps
  synced to 0.1.5.
- README and `docs/r_parity_matrix.md` test count: `312` → `313`
  (measured with the `[parallel,plot,tutorial]` extras installed,
  which include the optional `linearmodels` / `joblib` tests).
- `docs/PERFORMANCE_REPORT.md` speedup arithmetic corrected on
  four rows: `13.7x → 11.5x` (n=1000), `8.1x → 7.5x` (n=10000 and
  n=100000), `7.6x → 7.7x` (n=500000). Underlying timings
  unchanged; the displayed ratios now match `R_time / pymmeans_time`
  exactly.

## [0.1.4] — 2026-05-26

### Fixed (statistical correctness)

- **`regrid_response(em, bias_adjust=True)`** on a GLM with a
  non-identity link now raises `ValueError` unless an explicit
  `sigma=` is supplied. The previous default (`info.scale`) used
  the residual dispersion on the *response* scale as the Jensen
  correction's σ² — which silently inflated the response-scale
  EMM by a family-dependent factor (1.5× on canonical Poisson(log),
  ~3.25× on Gaussian(log)). R `emmeans::summary(...,
  bias.adjust=TRUE)` enforces the same constraint. OLS with an
  LHS transform (`lm(log(y) ~ ...)`) is unchanged — `info.scale`
  there *is* the correct sigma².
- **Cox PH `emmeans(fit, ..., type='link')`** now emits a
  `UserWarning` advising that the `emmean` column is on the
  *relative log-hazard* scale (the partial likelihood does not
  identify the baseline hazard, so the reference-level row shows
  `emmean=0` by construction). `pairs(emm)` and `regrid_response(emm)`
  remain identifiable and unchanged. R `emmeans.coxph` omits the
  link-scale column for the same reason.
- **`apply_kenward_roger`** is now idempotent: a second call on an
  already-KR-corrected EMM/contrast returns the input unchanged.
  Previously a second application recomputed K-R from the
  already-inflated vcov and drifted the df by ~0.05 due to finite-
  difference noise on the doubly-corrected Hessian.

### Fixed (API)

- **`joint_tests(emm_result)`** now accepts `EMMResult` /
  `ContrastResult` / `RefGrid` input by dispatching to the underlying
  `model_info`, matching R `emmeans::joint_tests` behaviour.
  Previously raised `TypeError: No adapter recognises EMMResult`.
- **`pairs(emm)`** now emits a `UserWarning` when the input EMM
  contains any non-estimable rows (`emmean` is NaN, typically from
  a rank-deficient design). Contrasts touching those rows still
  propagate NaN to the result — the warning surfaces the issue so
  users are not surprised by silent NaN downstream. R `emmeans`
  marks these as `nonEst`; `pymmeans` flags them at the contrast
  step.

### Fixed (documentation)

- **`apply_satterthwaite`** docstring now documents the per-row
  property of Satterthwaite df: df at an EMM cell can differ from
  df on a pair contrast by an order of magnitude on the same model
  (different linear-combination matrix `L`), and both are correct.
  Includes the canonical sleepstudy numbers (EMM at `Days=9` →
  df ≈ 23.4, `Days=9 − Days=0` contrast → df ≈ 161).
- **`survey.py`** module docstring now matches the implementation
  for the simple-random-sample variance (the code applies the
  `n/(n−1)` finite-sample correction with score centring; the
  docstring previously cited the EHW uncentred form, which agrees
  at the MLE FOC but diverges if the GLM has not fully converged).
- **`_regularize_corr_for_mvt`** docstring no longer claims R's
  `mvtnorm::pmvt` does "the same thing internally" — R uses
  Cholesky pivoting and dimension reduction; `pymmeans` uses a
  ridge that introduces an O(1e-9) bias invisible at the 1e-4
  validation tolerance.

### Deferred to 0.2.0

- Kenward-Roger algorithmic refactor (analytic derivatives or
  chunked finite-difference) for memory-bounded `n_theta ≥ 20`
  fits. Current implementation works for typical mixed-model
  sizes (`n_theta ≤ ~10`).
- Hessian-step Richardson extrapolation for poorly-scaled fits
  (heritability-like parameters ≈ 1e-3). Current step formula is
  adequate for the canonical reference suite but accumulates
  10-50 % roundoff at extreme parameter scales.

## [0.1.3] — 2026-05-25

### Fixed

- Stripped residual scrub artifacts: empty parenthetical fragments
  (`()`) left in 10+ docstrings and module-level comments
  (`contrasts.py`, `summary_layer.py`, `emmeans.py`, `summary.py`,
  `utils.py`, `pbmodcomp.py`, `satterthwaite.py`); audit-trail
  markers `P0 #5` / `P1-3 (audits)` removed from the public source.
- Recounted the `docs/r_parity_matrix.md` summary table: the
  previous totals (85 / 3 / 10 = 98) did not match the matrix rows.
  Actual is 84 / 4 / 12 = 100; README banner updated accordingly.
- Softened "floating-point parity with no qualifying footnotes"
  language in `r_parity_matrix.md` and `vs-r.md` to match the
  measured tolerances (`atol < 1e-4` typical, with the documented
  ~2.6% residual on the K-R random-intercept SE).
- `docs/v0_2_roadmap.md` removed entries for `bs(x, ...)` spline
  support (already shipped in v0.1) and custom callable contrast
  methods (already accepted by `contrast()`).
- `docs/index.md` version banner synced to v0.1.3.
- `summary_layer.py`: dropped unused `nonlocal adj_lower` declaration.
- `tests/test_pbmodcomp.py`: replaced `import joblib  # noqa: F401`
  with `pytest.importorskip("joblib")`.
- `pyproject.toml`: added `Programming Language :: Python :: 3.13`
  classifier (PyPI 0.1.1 was already known to install cleanly on
  Python 3.13).
- README quickstart output corrected to match what `pandas` actually
  prints (the displayed `p_value` is `0.0` from underflow, not
  `< 1e-9`); added pointer to `summary(pairs(...))` / `as_r_frame`
  for "<.0001"-style formatting.

## [0.1.2] — 2026-05-25

### Fixed

- `patsy` is now declared as a direct dependency (`pyproject.toml`).
  It was always required at import time (`ref_grid.py`, `emmeans.py`,
  `trends.py`) but was previously pulled in transitively via
  `statsmodels`; declaring it directly future-proofs against
  `statsmodels` switching to `formulaic`.
- Two docstring paragraphs in `summary_layer.py` (`adjust` and `null`
  parameters) had stale half-finished edits visible in
  `help(summary)`; rewritten in clean voice.
- README banner now reports the correct version, public-surface test
  count, and line-coverage percentage. The R-parity claim now matches
  the `r_parity_matrix.md` figure exactly.
- `docs/v0_2_roadmap.md` removed entries for features that already
  shipped (OrderedModel, MNLogit, interaction contrasts,
  `rbind` / `emm_list`).
- `docs/PERFORMANCE_REPORT.md`, `docs/getting-started.md` version
  references synced to 0.1.2.

## [0.1.1] — 2026-05-25

### Fixed

- README relative links and the interaction-plot image reference were
  converted to absolute GitHub URLs so they resolve correctly on the
  PyPI project page (PyPI's README renderer does not follow GitHub
  relative paths).

## [0.1.0] — 2026-05-25

Initial release.

### Added

- Reference-grid construction (`ref_grid`) and estimated marginal means
  extraction (`emmeans`, `lsmeans`) for fitted `statsmodels` models,
  `linearmodels` panel / IV results, and any user-supplied
  `predict_fn(data) -> ndarray` callable.
- Pairwise contrasts (`pairs`), generic linear contrasts (`contrast`),
  compact letter displays (`cld`), and pairwise p-value matrices
  (`pwpm`).
- Multiplicity adjustments: Tukey HSD (exact studentised-range
  integral), Dunnett (exact via the multivariate-_t_ CDF), Šidák,
  Bonferroni, Holm, Benjamini–Hochberg FDR, and the generic `mvt`
  integral.
- Small-sample mixed-model inference: `apply_satterthwaite`,
  `apply_kenward_roger`, plus the six headline `pbkrtest`
  equivalents (`kenward_roger_vcov`, `get_kr`, `ddf_lb`, `krmodcomp`,
  `satmodcomp`, `pbmodcomp`).
- Response-scale back-transformation (`type="response"`) for
  GLM-style links, including bias-adjusted estimates for log links.
- Bootstrap (`bootstrap_ci`, both parametric and case-resampling)
  and permutation tests (`permutation_test`).
- Bayesian-posterior EMMs via `from_pymc` / `posterior_emmeans` /
  `posterior_emm_summary` (optional `arviz` / `PyMC` integration).
- Survey-weighted EMMs via `from_survey` (Lumley-style designs).
- ML adapter (`from_predict`, `ml_emmeans`, `ml_pairs`,
  `ml_contrast`) for tree ensembles, gradient-boosted models, neural
  networks, and any other `.predict()`-capable estimator.
- `emtrends` for derivatives of the regression surface at focal points.
- `OrderedModel` (cumulative-link ordinal) and `MNLogit` (multinomial
  logit) support.

### Tested

- 312 public-surface unit tests against `statsmodels`,
  `linearmodels`, `scikit-learn`, `xgboost`, `lightgbm`, `PyTorch`,
  `survey`, and R-side reference values from `emmeans`,
  `lme4` + `lmerTest`, `pbkrtest`, `marginaleffects`, and `survey`.
- R parity tolerances range from `atol < 1e-7` (survey-weighted
  Gaussian) to `atol < 1e-3` (finite-difference Satterthwaite df);
  the six `pbkrtest` equivalents match the R reference at
  `atol < 1e-5` on identical $\hat\theta$ inputs.

