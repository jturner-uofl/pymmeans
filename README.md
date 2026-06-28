# pymmeans

[![PyPI](https://img.shields.io/pypi/v/pymmeans?color=blue&label=PyPI)](https://pypi.org/project/pymmeans/)
[![Python versions](https://img.shields.io/pypi/pyversions/pymmeans)](https://pypi.org/project/pymmeans/)
[![License](https://img.shields.io/badge/license-GPL--3.0--or--later-blue)](LICENSE)
[![GitHub release](https://img.shields.io/github/v/release/jturner-uofl/pymmeans?label=release)](https://github.com/jturner-uofl/pymmeans/releases)

Status: **Beta** · v0.18.0 · API stable, numerical surface frozen.

Estimated marginal means (EMMs) for Python. A native implementation of R's
[emmeans](https://cran.r-project.org/package=emmeans) and
[pbkrtest](https://cran.r-project.org/package=pbkrtest), validated against both
at floating-point precision, with additional integrations for VanderWeele–Ding
E-value sensitivity analysis, Rubin's-rules multiple-imputation pooling,
split- and counterfactual-conformal prediction intervals, and
augmented-inverse-probability-weighted plus cross-fitted double-machine-learning
average-treatment-effect estimation.

The package targets the Python statistical-modelling ecosystem
(`statsmodels`, `lifelines`, `linearmodels`, `scikit-learn`, `PyMC`) and
provides the canonical R `emmeans` workflow as a native Python dependency,
with no R toolchain required.

---

## Status

Beta. Version 0.18.0. API stable across the OLS, GLM, MixedLM, GEE, Cox
proportional-hazards, parametric AFT, ordinal, multinomial, and
survey-weighted model classes. For MixedLM the Kenward–Roger and
Satterthwaite degrees-of-freedom machinery covers the `cov_re`,
`re_formula`, and `vc_formula` random-effects syntaxes. 87 of 100
strict R-`emmeans` parity items (90 of 100 with partially-supported
items counted; see the [R-parity matrix](docs/r_parity_matrix.md),
the audited source of truth). The numerical surface is frozen; minor
API polish is still possible until version 1.0.

The public API surface is exercised by 586 unit tests.
The full suite reaches 87% line coverage and totals 1,192 passing
tests, including the internal audit-regression file.

| Metric                                  | Value                          |
|-----------------------------------------|--------------------------------|
| Unit tests                              | 1,192 passing (586 public-surface) |
| Line coverage                           | 87 %                           |
| Validation contracts                    | 340 (176 R cross-validation + 119 structural + 45 Monte-Carlo) |
| Validation contract failures            | 0                              |
| Wall-clock vs R `emmeans` (common paths)| 1.6–4.1 times faster on 4 of 6 representative workloads |
| Wall-clock vs R `emmeans` (slow paths)  | 13–34 times slower on 2 of 6 (both diagnosed; see Section 5 of the manuscript) |

Representative validation evidence from the package's narrative validation
notebook:

- Direct cross-validation against R `emmeans` reference values to
  `atol = 1e-14` on the deterministic surface (176 contracts).
- The algebraic identity `SE = sqrt(L V L^T)` verified to exact zero on
  four archival datasets across three model classes.
- Conformal coverage within `±0.006` of nominal across Gaussian,
  Student's t with three degrees of freedom, and contaminated errors at
  three nominal levels.
- AIPW unbiased under either-side nuisance misspecification, with
  six-fold tighter mean absolute error than the inverse-probability-
  weighted-only estimator.
- On the Hill (2011) IHDP semi-synthetic benchmark, gradient-boosted
  g-computation via `from_predict` achieves a root-mean-square error
  of `0.17` against simulated truth across ten replications, against
  the linear-adjustment estimator's `1.59` (9.5 times tighter), through
  the same API.

---

## Installation

```bash
pip install pymmeans                       # core
pip install "pymmeans[plot]"               # adds matplotlib for plot() / emmip() / pwpp()
pip install "pymmeans[tutorial]"           # adds scikit-learn + jupyter + pysofra for the showcase notebook
```

Python 3.10 or later. Required dependencies: `numpy`, `pandas`, `patsy`,
`scipy`, `statsmodels`.

Local development:

```bash
git clone https://github.com/jturner-uofl/pymmeans.git
cd pymmeans
uv venv && uv pip install -e ".[dev,plot]"
make test
```

---

## Usage

### Marginal means and pairwise contrasts on a fitted OLS model

```python
import statsmodels.formula.api as smf
from pymmeans import emmeans, pairs

fit = smf.ols("growth ~ fertilizer * sunlight", data=df).fit()
em  = emmeans(fit, specs=["fertilizer"])
pw  = pairs(em, adjust="tukey")
```

The result objects (`EMMResult`, `ContrastResult`) expose `.frame`
attributes that are standard pandas DataFrames; downstream code that
expects R `emmeans`-shaped output should work without modification.

### Mixed-effects models with Kenward–Roger degrees of freedom

```python
import statsmodels.regression.mixed_linear_model as mlm
from pymmeans import emmeans
from pymmeans.satterthwaite import apply_kenward_roger

fit   = mlm.MixedLM.from_formula(
    "reaction ~ days", df, groups="subject"
).fit(reml=True)
em_kr = apply_kenward_roger(emmeans(fit, specs=["days"]))
```

The six headline R `pbkrtest` functions (`vcovAdj`, `getKR`, `Lb_ddf`,
`KRmodcomp`, `SATmodcomp`, `PBmodcomp`) are ported and match the R
reference values to floating-point precision on every tested fit.

### Machine-learning g-computation through the ML adapter

```python
from sklearn.ensemble import GradientBoostingRegressor
from pymmeans import from_predict, ml_emmeans, ml_contrast

gbm  = GradientBoostingRegressor().fit(X_train, y_train)
info = from_predict(
    predict_fn=lambda d: gbm.predict(d[features]),
    data=train_df,
    factors={"treat": [0, 1]},
    numerics=features,
    response="y",
)
em_ml = ml_emmeans(info, specs="treat")
ct_ml = ml_contrast(em_ml, method="trt.vs.ctrl", ref=0)
```

Any object exposing a `.predict()` method can serve as the outcome
model. R `emmeans` has no equivalent path because its extension
mechanism requires the model to expose a linear-predictor coefficient
vector and covariance matrix.

### Modern causal-inference and uncertainty-quantification extensions

```python
from pymmeans import (
    e_value,                       # VanderWeele and Ding (2017) sensitivity
    pool_imputed,                  # Rubin (1987) + Barnard and Rubin (1999) pooling
    split_conformal_pi,            # Vovk et al. (2005); Lei et al. (2018)
    conformal_counterfactual_pi,   # Lei and Candes (2021)
    aipw_ate,                      # Robins, Rotnitzky, and Zhao (1994)
    cross_fit_ml_emmeans,          # Chernozhukov et al. (2018)
)
```

Each implements a published, peer-reviewed algorithm and ships with
explicit Monte-Carlo coverage verification in the validation notebook
(Sections XIX, XX, and XXI).

---

## Relation to R packages

A Python user installing `pymmeans` no longer needs to install or
maintain an R toolchain for any of the following packages; equivalent
or fuller coverage is provided in one library.

| R package | Capability                                                                | Coverage in pymmeans |
|-----------|---------------------------------------------------------------------------|----------------------|
| emmeans   | EMMs, contrasts, multiplicity adjustments, response-scale back-transforms | Full                 |
| lsmeans   | Predecessor to emmeans                                                    | Full (aliased)       |
| multcomp  | Bonferroni, Sidak, Holm, Tukey, Dunnett, mvt, Hochberg, Hommel; `cld()`   | Full                 |
| mvtnorm   | Multivariate-t cumulative distribution function for `mvt` adjustment      | Full (via SciPy)     |
| pbkrtest  | Kenward–Roger covariance, parametric-bootstrap nested test, F-tests       | Full (six functions) |
| lmerTest  | Satterthwaite degrees of freedom for `lmer`                               | Full                 |
| EValue    | VanderWeele–Ding E-value sensitivity analysis                             | Full                 |
| effects   | Adjusted means and predicted effects                                      | Subsumed             |
| predictmeans / phia / gmodels::estimable | Predicted means, post-hoc interactions, custom contrasts | Subsumed             |
| mice::pool | Rubin's-rules pooling for any downstream R estimator                     | Partial (pools pymmeans output; `mice` itself remains in R) |

**Contrast families** (`contrast(em, method=...)`): `pairwise`,
`revpairwise`, `consec`, `poly`, `trt.vs.ctrl`, `eff`, `del.eff`,
`mean_chg`, plus custom coefficient lists.
**Multiplicity adjustments** (`adjust=...`): Tukey, Dunnett (exact
`mvt`), Šidák, Bonferroni, Holm, Hochberg, Hommel, Scheffé, and the
Benjamini–Hochberg (`BH`) and Benjamini–Yekutieli (`BY`)
false-discovery-rate controls.

See [docs/r_parity_matrix.md](docs/r_parity_matrix.md) for the
per-function R-parity table, and
[docs/vs-r.md](docs/vs-r.md) for measured numerical agreement.

---

## Limitations and scope

`pymmeans` targets the Python statistical-modelling ecosystem and does
not attempt parity with R `emmeans`'s full extensible registry. Model
classes that depend on R fitters with no production-grade Python
equivalent are out of scope. The principal items are summarised below;
the full enumeration is in the validation notebook and the manuscript.

| Out-of-scope capability                                | R analogue            | Reason                |
|--------------------------------------------------------|-----------------------|-----------------------|
| Non-linear mixed-effects models                         | `nlme::nlme`          | Python ecosystem      |
| Generalised least squares with structured errors        | `nlme::gls`           | Python ecosystem      |
| Penalised additive models with tensor / factor-by smooths | `mgcv::gam`         | Python ecosystem      |
| Zero-inflated mixed models                              | `glmmTMB`             | Python ecosystem      |
| Bayesian regression formula DSL                         | `rstanarm`, `brms`    | pymmeans accepts PyMC draws directly via `from_pymc` and `from_arviz` |
| Bayesian mixed models with prior control                | `MCMCglmm`            | Python ecosystem      |
| Distributional regression                               | `gamlss`              | Python ecosystem      |
| Full Lumley `survey` Taylor-series linearisation        | `survey`              | Python ecosystem (partial via `from_survey`) |

---

## Validation

Three independent layers of evidence support the numerical claims.

- **Automated test suite.** 1,192 unit tests at 87 % line coverage
  (586 on the public API surface). Execute with `make test` or `pytest -q`.
- **Narrative validation notebook.** 340 enumerated contracts spanning
  direct cross-validation against R reference values (176), structural
  and self-consistency identities (119), and Monte-Carlo coverage and
  calibration checks (42). Zero failures. The notebook is at
  [`examples/jss_audit/jss_case_study.ipynb`](https://nbviewer.org/github/jturner-uofl/pymmeans/blob/main/examples/jss_audit/jss_case_study.ipynb)
  and can be re-executed with `make notebook`.
- **R reference fits.** Reference CSV outputs from the original R
  `emmeans`, `pbkrtest`, and supporting packages are committed under
  `tests/r_reference/`, along with the R scripts that regenerate
  them.

A single-command reproduction of the full evidence base from a clean
checkout is provided:

```bash
make reproduce
```

---

## Documentation

| Resource                              | Location                                           |
|---------------------------------------|----------------------------------------------------|
| Online documentation (mkdocs site)    | [docs/](docs/) — build locally with `mkdocs serve` |
| R-parity matrix                       | [docs/r_parity_matrix.md](docs/r_parity_matrix.md) |
| Measured numerical agreement vs R     | [docs/vs-r.md](docs/vs-r.md)                       |
| Performance benchmark report          | [docs/PERFORMANCE_REPORT.md](docs/PERFORMANCE_REPORT.md) |
| Tutorial and applied case studies     | `examples/jss_audit/jss_case_study.ipynb`          |

---

## Citation

If `pymmeans` is used in research, please cite the package. A
`CITATION.cff` file is committed to the repository root; GitHub
renders a "Cite this repository" button from it automatically.

```bibtex
@software{pymmeans,
  author  = {Turner, Jason S.},
  title   = {pymmeans: Estimated Marginal Means for Python},
  version = {0.6.0},
  year    = {2026},
  url     = {https://github.com/jturner-uofl/pymmeans},
}
```

---

## Contributing

Bug reports, R-parity discrepancies, and feature requests are welcome
through the issue templates at `.github/ISSUE_TEMPLATE/`. For
substantial contributions, please open an issue to discuss the design
before submitting a pull request. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the development workflow and the
[Code of Conduct](CODE_OF_CONDUCT.md) for community standards.

---

## License

GNU General Public License, version 3 or later (`GPL-3.0-or-later`),
matching the licence of R `emmeans`.

---

## Acknowledgements

The R `emmeans` package by Russell V. Lenth is the reference standard
against which `pymmeans` was designed and validated. The R `pbkrtest`
package by Ulrich Halekoh and Søren Højsgaard supplies the
small-sample mixed-model inference machinery that `pymmeans` ports.
Vincent Arel-Bundock's `marginaleffects` package informed the
prediction-surface adapter design.
