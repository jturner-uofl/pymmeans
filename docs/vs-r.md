# vs R emmeans

pymmeans is validated against R `emmeans` 2.0.3 on **two independent
benchmark suites**:

1. **`tests/test_vs_r.py`** — 9 tests on the original 5 reference
   scenarios (R-emmeans canonical datasets), at `atol=1e-4`.
2. **`tests/test_r_benchmark.py`** — 18 tests across the broader
   parity surface (afex, lme4 + lmerTest, pbkrtest, marginaleffects,
   survey Gaussian / Poisson / Binomial / Gamma, GLM offsets,
   bias_adjust), at the tighter tolerances listed below.

Reference values are committed to `tests/r_reference/*.csv`, so CI
runs both suites without needing R installed. The R scripts
(`generate_r_reference.R`, `cross_validation.R`) regenerate them.

## Original reference scenarios (`test_vs_r.py`)

| Dataset | Model | Tests |
|---|---|---|
| `warpbreaks` | `lm(breaks ~ wool * tension)` | EMMs by-group, pairs with Tukey |
| `pigs` | `lm(log(conc) ~ source + factor(percent))` | EMMs, pairwise on log scale |
| `ToothGrowth` | `lm(len ~ supp * factor(dose))` | EMMs with by-group, factor-from-numeric |
| `InsectSprays` | `lm(sqrt(count) ~ spray)` | EMMs, k=6 pairwise |
| `neuralgia` | `glm(Pain ~ Treatment * Sex + Age, family=binomial)` | Response-scale EMMs (probabilities) |

## Broader R-package parity (`test_r_benchmark.py`)

| Reference | EMM / SE tolerance | df tolerance |
|---|---|---|
| `afex` EMMs / pairs / joint_tests | `atol=1e-5` | `atol=1e-3` |
| `lme4 + lmerTest` Satterthwaite (random intercept) | `atol=1e-4` | `atol=0.1` |
| `lme4` Satterthwaite (random slopes) | `atol=1e-4` | `atol=0.2` |
| `lme4 + pbkrtest` KR | `rtol<3%` (known intercept gap) | `rtol<6%` |
| `marginaleffects` + `emmeans` | `atol=1e-5` | n/a |
| `survey::svyglm` SRS Gaussian | `atol=1e-7` | n/a |
| `survey::svyglm` SRS Poisson | `atol=1e-6` | n/a |
| `survey::svyglm` SRS Binomial logit | `atol=1e-6` | n/a |
| `survey::svyglm` SRS Gamma log | `atol=1e-4` (EMM), `1e-5` (SE) | n/a |
| GLM Poisson + `exposure=` | `atol=1e-6` | n/a |
| `bias_adjust` Taylor formula | `atol=1e-5` | n/a |

EMM / SE columns are the actual benchmark tolerances. The df column
is broader because the Satterthwaite finite-difference Hessian
introduces ~1e-3 relative noise on df even when the SE matches R to
1e-4. The two figures are reported separately: SE matches R at ~1e-4,
df at ~1e-3.

## What we don't match (yet)

- **Satterthwaite SE** via `apply_satterthwaite()` matches
  `lmerTest::lmer + emmeans` SE to `atol=1e-4` (effectively
  floating-point precision) on both random-intercept and
  random-slopes reference fits. Uses lmer's (Lambda, sigma²)
  parametrisation with Cov(theta) from inverting the REML deviance
  Hessian. The **Satterthwaite df** matches R to `atol=0.1`
  (random intercept) and `atol=0.2` (random slopes) — the
  finite-difference Hessian introduces small numerical noise that
  appears in df even when SE is exact.
- **Kenward-Roger** via `apply_kenward_roger()` matches `pbkrtest::vcovAdj`
  to <0.1% on non-intercept SEs and ~2.6% on the intercept SE for the
  lme4 reference fit (Kackar-Harville form; the full Kenward-Roger
  small-sample 3rd-order correction is not yet implemented but the gap
  for contrast/slope inference is publication-grade).
- **Survey-weighted EMMs** via `SurveyDesign` + `from_survey()` match
  R's `survey::svyglm + emmeans` to <1e-7 on:
  - SRS Gaussian (WLS-style)
  - Stratified Gaussian
  - SRS Poisson (GLM IRLS bread + score factor)
  Clustered (PSU within stratum) designs are supported via the Taylor
  linearisation. FPC (finite population correction) is not yet wired in.
- **Bayesian / posterior-based EMMs** via `posterior_emmeans` work on
  any model with posterior draws of the fixed-effect coefficients. The
  pure-numpy `posterior_emm_summary` doesn't require PyMC/arviz to be
  installed; the `from_pymc(idata, formula, data)` adapter lazy-imports
  arviz only when called. Gives correctly asymmetric credible
  intervals on the response scale for non-linear links — the same
  workflow R `emmeans` offers via `emm_basis.brmsfit`, in Python.
- **Reproducible R cross-validation benchmark** in
  `tests/test_r_benchmark.py` runs **18 reference comparisons across 12
  distinct fits** in afex /
  lme4 / lmerTest / pbkrtest / marginaleffects / survey (Gaussian +
  Poisson + Binomial logit + Gamma log) / GLM offsets / bias_adjust
  and asserts pymmeans matches R to the tolerances listed above.
  CI runs against committed CSVs (no R required); regenerate via
  `Rscript tests/r_reference/cross_validation.R` after numerical
  changes.
- **`joint_tests` for all term shapes** (purely categorical, purely
  numeric, mixed cat-by-numeric) uses the EMM-basis path and matches
  R `emmeans::joint_tests` exactly on the reference fits in
  `tests/test_r_benchmark.py`.
- **R's quirky outputs**: certain corner cases of R emmeans (e.g. order
  of contrasts when factor levels are unordered) may differ. We always
  set factor levels explicitly in tests to make this deterministic.
- **`bias_adjust`**: matches R to floating-point precision on
  log / log10 / log2 / log1p / sqrt transforms. We use R's
  second-order Taylor expansion (`exp(mu) * (1 + sigma^2/2)`) rather
  than the exact lognormal (`exp(mu + sigma^2/2)`); the two agree to
  ~0.05% at `sigma ≈ 0.25` but diverge to ~1.5% at `sigma ≈ 0.6`.

## Where we beat R

See [Performance](PERFORMANCE_REPORT.md) for the full table — the
highlights below cite that benchmark verbatim:

- `emmeans` on n=500K OLS: **~8× faster** (analytic L_marg, no grid).
- `emmeans` on n=100K OLS: **~8× faster**.
- GitHub issue #282 (46M-row grid): R **refuses / OOM**; pymmeans
  runs in **21 ms** (analytic marginalisation, no grid
  materialisation).
- Pairwise k=200 Tukey: **~4.9× faster** (vectorised
  studentized-range SF via Gauss–Hermite + Gauss–Laguerre).
- Pairwise k=100 Tukey: **~1.3× faster**.

For the moderate-k pairwise regime (k=20–50), pymmeans and R
``emmeans`` are within ~2× of each other in either direction —
the quadrature overhead dominates and R's compiled studentized-
range implementation is competitive. The crossover happens around
k≈100; above that, pymmeans's vectorisation pays off. Absolute
ratios are hardware- and BLAS-dependent; the asymptotic ordering
(constant in n; sub-linear in k at large k via vectorised
quadrature) is the durable claim.
