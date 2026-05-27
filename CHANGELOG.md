# Changelog

All notable changes to `pymmeans` will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

