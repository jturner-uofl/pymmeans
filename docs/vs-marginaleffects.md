# pymmeans vs. marginaleffects

[`marginaleffects`](https://marginaleffects.com/) (Arel-Bundock, Greifer &
Heiss, *JSS* 111(9), 2024) is an excellent, mature package for predictions,
comparisons, slopes, and hypothesis tests across 100+ model types. pymmeans
and marginaleffects come from different traditions — marginaleffects from
the *observed-sample* (g-computation) view, pymmeans from the R `emmeans` /
`pbkrtest` *balanced-design* view — and they are best understood as
**complementary**.

As of pymmeans v0.12, pymmeans implements the full marginaleffects core
surface (`avg_predictions` / `avg_comparisons` / `avg_slopes` /
`hypotheses` / `hypothesis=` / `transform=` / `datagrid` / `newdata=` /
`plot_*`), plus bootstrap marginal effects for black-box models — so the
two packages now overlap substantially. This page records, honestly and
reproducibly, where pymmeans is measurably ahead, and where marginaleffects
is genuinely better. Every number below is reproduced by a script in the
repository and by the validation notebook.

## Where pymmeans is measurably ahead

These five claims are backed by numbers anyone can reproduce. The Python
edition of marginaleffects (0.5.1) was used throughout.

| # | Claim | The measurement |
|---|-------|-----------------|
| 1 | **Analytic-exact standard errors.** pymmeans differentiates an exact design and finite-differences only over the coefficients; marginaleffects double-finite-differences by default (analytic gradients require an optional JAX install). | Response-scale average-marginal-effect SE error vs the hand-built analytic delta-method truth (logistic GLM, n=2000): pymmeans **2.4e-13**, marginaleffects **7.4e-8**. |
| 2 | **Step-size robustness.** | SE drift across the finite-difference step: pymmeans **6e-10**; marginaleffects **6.3e-4**, and it collapses (SE 0.00865 vs 0.00928) at `eps=1e-8`. |
| 3 | **Speed on the common paths.** | Best-of-7, n=4000 logistic GLM: `avg_predictions` **3.1×**, `avg_slopes` **5.8×**, `avg_comparisons` **6.6×** faster. |
| 4 | **Uncertainty for black-box ML models.** marginaleffects returns a point estimate with no standard error for a scikit-learn model; pymmeans `ml_avg_*` returns a pairs-bootstrap SE and percentile interval (with Monte-Carlo-validated coverage). | sklearn `GradientBoostingRegressor`: marginaleffects frame = `estimate` only; pymmeans = estimate **±SE, CI** (requires a `refit_fn`). |
| 5 | **Native Kenward–Roger / Satterthwaite degrees of freedom.** A source search of marginaleffects for `kenward`/`satterthwaite` returns zero matches; it reports infinite-df *z* on a mixed model. | `MixedLM`, 8 groups: pymmeans yields a finite **KR df ≈ 38.9** with a variance-component-corrected SE; matches R `pbkrtest::vcovAdj` to `atol ≈ 5e-7`. |
| 6 | **Standardised effect sizes with SD-uncertainty SEs.** marginaleffects has no standardised-effect-size helper; pymmeans `effect_size` returns Cohen's *d* / Hedges' *g* with a standard error that propagates uncertainty in the standardising SD via `edf`, exactly as R `emmeans::eff_size`. | `SE(d) = sqrt(SE_∞² + d²/(2·edf))`; pymmeans matches R `eff_size` to **~5e-11** across `edf` on shared data (jss_audit §XXXI). |

One correctness note in pymmeans' favour: pymmeans' `hypothesis=` argument
refers to result rows by **factor-level labels** (`"pairwise"`,
`"reference"`, or an explicit contrast matrix), so there is no positional
`b0`/`b1` coefficient indexing — and therefore none of the R-vs-Python
off-by-one inconsistency that reached a *JSS* erratum for marginaleffects'
string-based `hypothesis` syntax (its `b`-indexing starts at 0 in Python
but 1 in R,
[errata #1293](https://github.com/vincentarelbundock/marginaleffects/issues/1293)).

And one strategic, non-numerical fact: a user requested an emmeans-style
Python function and the marginaleffects maintainers closed it as **"not
planned"**
([#1474](https://github.com/vincentarelbundock/marginaleffects/issues/1474));
the JSS paper itself *recommends* emmeans for that use case. pymmeans is a
native-Python implementation of exactly that.

## What pymmeans adds that marginaleffects does not cover

The balanced-grid **estimated marginal means** estimand as the default;
**Kenward–Roger / Satterthwaite** df; exact **Tukey / Dunnett / `mvt`**
multiplicity and **compact letter displays**; and the
sensitivity/uncertainty surface — **E-value** sensitivity analysis,
**AIPW**, **cross-fit double-machine-learning**, and the
prediction-surface ML adapter. The default estimand difference is concrete:
on an unbalanced design, the equal-weight EMM grand mean is `2.035` while
marginaleffects' `avg_predictions` default is the proportional average
`1.182` (it reaches the EMM only via an explicit balanced `datagrid`).

## Where marginaleffects is genuinely better (read this part)

Intellectual honesty requires stating the other side plainly. These are
real, and a few of them were "obvious" pymmeans advantages that **did not
survive measurement**:

- **Cold-start import time.** marginaleffects imports in ~0.015 s and
  lazy-loads its heavy dependencies; pymmeans front-loads scipy/pandas and
  imports in ~0.67 s (~44× slower). *(A lazy-import refactor is on the
  pymmeans backlog.)*
- **Small-sample `t` vs `z`.** marginaleffects-py 0.5.1 **does** use the
  model's residual df with a *t* distribution for statsmodels-backed
  models. The "marginaleffects defaults to z / infinite df" claim is false
  for OLS/GLM — pymmeans' df advantage is specific to **mixed models**
  (Kenward–Roger / Satterthwaite), not basic models.
- **Conformal prediction, multiple-imputation (Rubin's-rules) pooling, and
  observation weights** are all **supported by marginaleffects** (R, and
  partially in Python). pymmeans does not "own" these.
- **`multcomp="single-step"`** in marginaleffects *is* a correlation-aware
  (multivariate-*t*) adjustment — not merely an independent-test
  correction.
- **Model-backend breadth and ergonomics.** `me.fit_sklearn("y ~ x", data,
  engine=...)` is one line across many backends; pymmeans' black-box path
  asks for a `predict_fn` and a `refit_fn`.
- **Maturity, adoption, documentation, and a peer-reviewed JSS paper.**
  marginaleffects is field-tested by thousands of users. pymmeans is newer.

## Summary

For analysis on a parametric (statsmodels / GLM / mixed) model, pymmeans is
faster, numerically more precise, and adds the EMM / Kenward–Roger /
multiplicity / causal-sensitivity world that marginaleffects scopes out —
while now also covering the marginaleffects core surface. For maximum
model-type reach, smoother black-box ergonomics, and a mature ecosystem,
marginaleffects remains the more battle-tested choice. The honest one-line
summary: **pymmeans wins the engineering comparison on the merits;
marginaleffects wins on mileage and reach-of-hand.**
