# pymmeans wish-list — sweeteners beyond R `emmeans` parity

This is **not** a parity list (see [`r_parity_matrix.md`](r_parity_matrix.md) for
that). It is a sourced roadmap of features that R `emmeans` *users* repeatedly
ask for, are confused by, or were told "no" on — i.e. opportunities to make
`pymmeans` not just a port but a friendlier tool.

Built from a 3-vein audit (2026-06-28) of: the `rvlenth/emmeans` GitHub issue
tracker (API-verified), question-and-answer friction (anchored on emmeans' own
23-entry FAQ vignette as a recurrence proxy, since SO/CrossValidated were not
directly fetchable), and author-acknowledged limitations + external critiques.

## Headline: pymmeans already ships much of the wish-list

Many "a Python port should do X" opportunities are already done — useful as a
"why pymmeans, not rpy2-wrapped emmeans" story for the paper.

| emmeans community wish / critique | Source | pymmeans status |
|---|---|---|
| First-class tidy dataframe output (the #1 Q&A friction — `emmGrid` is S4, CIs "buried deeply") | [Posit](https://forum.posit.co/t/emmeans-and-dplyr/84087) | ✅ `.frame` everywhere |
| Native finite-sample df (emmeans gives `df=Inf`/z without pbkrtest) | [models vignette](https://cran.r-project.org/web/packages/emmeans/vignettes/models.html), [#306](https://github.com/rvlenth/emmeans/issues/306) | ✅ KR + Satterthwaite built in |
| Exact/simulation back-transform vs 2nd-order bias approx | [transformations vignette](https://rvlenth.github.io/emmeans/articles/transformations.html) | ✅ `regrid(n_sim=)` (v0.17) |
| Quantile credible intervals, not just HPD (asked by brms's author) | [#538](https://github.com/rvlenth/emmeans/issues/538) | ✅ percentile default + `hpd=` (v0.16) |
| Higher-degree `emtrends` | [#133](https://github.com/rvlenth/emmeans/issues/133) (27c) | ✅ `max_degree=` (v0.14) |
| Multivariate contrasts | [#281](https://github.com/rvlenth/emmeans/issues/281) (37c, most-commented) | ✅ `mvcontrast` (mvregrid still missing) |
| Multiple-imputation pooling | [#80](https://github.com/rvlenth/emmeans/issues/80) | ✅ Rubin's rules |
| Robust / cluster SEs; observational-data weighting | Heiss; marginaleffects JSS | ✅ robust vcov (v0.18) + weight schemes + g-computation |

## The sweeteners — strategic lane: self-describing, mistake-catching results

The highest-recurrence *unmet* pains are all things emmeans **deliberately
refuses** to do: Lenth's stance is *"emmeans summarizes the model, not the
data; bad model in → bad results out"* ([FAQ](https://rvlenth.github.io/emmeans/articles/FAQs.html),
[#523 wontfix](https://github.com/rvlenth/emmeans/issues/523)). That refusal is
the opening — a durable differentiator, not catch-up.

| # | Sweetener | Evidence | Effort | Moat | Status |
|---|---|---|---|---|---|
| 1 | **Annotate the result**: the active *scale*, *held-at-mean covariates* ("age=43.2"), and what was averaged over (with weights) | FAQ #7/#14/#18/#19/#20 — largest pain cluster | Low | emmeans won't clutter output | ✅ `EMMResult.describe()` (v0.20) |
| 2 | **Footgun warnings**: warn when a 3+level factor is coded numeric (FAQ #3/#10/#11); warn when `log(x)`/`poly(x)` makes the grid use `mean(x)` not `mean(log x)` | [#523 wontfix](https://github.com/rvlenth/emmeans/issues/523), FAQ | Low–Med | **Explicitly declined by emmeans** | ✅ numeric-target error names the `C(x)` fix (v0.19); the `log(x)` grid case still open |
| 3 | **Name which cell is non-estimable and why** ("no data for A=2, B=3") instead of bare `NonEst`/NaN | FAQ #12, [#71](https://github.com/rvlenth/emmeans/issues/71) | Low | Pure diagnostics | ✅ `describe()` names them (v0.20) |
| 4 | **Estimand made explicit** + a "which average?" helper (equal / proportional / counterfactual), labeled in output | Heiss "Marginalia"; [marginaleffects JSS](https://arelbundock.com/research/arel-bundock_greifer_heiss_2024_how_to_interpret_statistical_models_using_marginaleffects_in_r_and_python.pdf); ggeffects | Med | The deepest critique (experimental-vs-observational) | open (design conversation) |
| 5 | **Inline "did you mean `by=`?"** when averaging over an interacting factor | FAQ #5/#16 | Low | Turns the scariest warning into a guided fix | ✅ `describe()` flags it (v0.20) |
| 6 | **Comparison-appropriate uncertainty by default** (steer from overlapping-CI fallacy toward `pwpp`/arrows) | [comparisons vignette](https://rvlenth.github.io/emmeans/articles/comparisons.html) | Low | Default-steering emmeans only nags about | ✅ `describe()` gives the guidance (v0.20); `pwpp`/`cld` already shipped |

### Lower-priority / higher-effort

- **`mvregrid`** — regrid for multivariate EMMs (genuine parity gap).
- **Risk-ratio `type=`** for binomial summaries — emmeans resisted ([#48](https://github.com/rvlenth/emmeans/issues/48)); pymmeans has it in `effect_size`, just not as a summary `type`.
- **Reusable grid object** powering means + contrasts + trends — emmeans recomputes per covariate ([#233](https://github.com/rvlenth/emmeans/issues/233): 7 covariates ≈ 1 hr).
- **No-hard-cap large-grid scaling** — emmeans blows up memory ([#282](https://github.com/rvlenth/emmeans/issues/282), [#218](https://github.com/rvlenth/emmeans/issues/218): 20 GB); emmeans mitigates only with a 10k-row `rg.limit`.
- **Richer contrast constructors** — `nrmlz`, `opoly`, `helmert` (emmeans NEWS v1.11.1); pymmeans lists `nrmlz` missing.

## Recommendation / build order

Build **#1–#3 as a "self-describing result" sequence** — low-effort, hits the
top Q&A pains, lives in the lane emmeans structurally won't enter, and each
annotation/warning is a clean measure-three contract. #4 (explicit estimand) is
the strategic one but is a design conversation, not a quick cut.

### Sourcing caveats

GitHub demand signals were API-verified (the repo keeps almost no standing
backlog — most requests are *closed-as-implemented*, so demand shows as long
comment threads + shipped NEWS entries, not 👍 pile-ons). SO/CrossValidated
pages were not directly fetchable; the Q&A vein is anchored on emmeans' own FAQ
vignette (each numbered entry = a question the author answers repeatedly) plus
Posit Community + GitHub. "Recurrence" is a qualitative ranking, not a measured
view count.
