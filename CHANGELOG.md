# Changelog

All notable changes to `pymmeans` will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

