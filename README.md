<h1 align="center">pymmeans</h1>

<p align="center"><b>Estimated marginal means for Python.</b><br/>A native implementation of R's <a href="https://cran.r-project.org/package=emmeans">emmeans</a> + <a href="https://cran.r-project.org/package=pbkrtest">pbkrtest</a>, validated to floating-point precision, integrated with modern causal-inference and uncertainty-quantification methods.</p>

<p align="center">
  <a href="https://pypi.org/project/pymmeans/"><img src="https://img.shields.io/pypi/v/pymmeans?color=blue&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/pymmeans/"><img src="https://img.shields.io/pypi/pyversions/pymmeans" alt="Python versions"></a>
  <a href="https://github.com/jturner-uofl/pymmeans/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-GPL--3.0--or--later-blue" alt="License"></a>
  <a href="https://github.com/jturner-uofl/pymmeans/releases"><img src="https://img.shields.io/github/v/release/jturner-uofl/pymmeans?label=release" alt="GitHub release"></a>
  <img src="https://img.shields.io/badge/tests-1%2C037%20passing-brightgreen" alt="tests">
  <img src="https://img.shields.io/badge/coverage-87%25-brightgreen" alt="coverage">
  <img src="https://img.shields.io/badge/validation%20contracts-251%20%2F%200%20fail-brightgreen" alt="validation">
</p>

> **Status: Beta (v0.6.0).** API stable across the OLS / GLM / MixedLM / GEE / Cox / AFT / ordinal / multinomial / survey surface. **90/100 strict R-`emmeans` parity (94/100 with partially-supported items).** **1,037 unit tests passing at 87% line coverage.** **1.6–4.1× faster than R `emmeans` on the four most-used code paths.** The numerical surface is frozen; minor API polish still possible.

---

## Why pymmeans

| If you want… | Reach for… |
|---|---|
| The full R `emmeans` workflow in Python with no R toolchain | **`pymmeans`** |
| The Kenward–Roger / Satterthwaite / parametric-bootstrap stack from `pbkrtest`, in Python | **`pymmeans`** (the six headline `pbkrtest` functions are ported and FP-precision-matched) |
| To plug a sklearn / XGBoost / LightGBM / PyTorch model in as the outcome and get population-average contrasts | **`pymmeans.from_predict`** (no R analog) |
| Modern causal-inference & UQ extensions integrated with the EMM grammar — E-value sensitivity, MI pooling (Rubin), split + counterfactual conformal PIs, AIPW, cross-fitted DML | **`pymmeans`** (no single R-`emmeans`-adjacent package bundles all of these) |

### Headline numbers

| Claim | Evidence |
|---|---|
| **Matches R `emmeans` to floating-point precision** on its deterministic surface | 134 direct cross-validation contracts, mostly at `atol=1e-14` |
| **1.6–4.1× faster than R `emmeans`** on four of six representative workloads | §XVIII of the [validation notebook](https://nbviewer.org/github/jturner-uofl/pymmeans/blob/main/examples/jss_audit/jss_case_study.ipynb) |
| **9.5× tighter RMSE than OLS-adjusted ATE** via the ML adapter on the Hill 2011 IHDP benchmark | §XVI of the validation notebook |
| **Distribution-free conformal coverage** within ±0.006 of nominal across Gaussian / t₃ / contaminated errors | §XX |
| **AIPW unbiased under either-side nuisance misspecification**, 6× tighter MAE than IPW-only | §XXI |
| **251 documented validation contracts, 0 failures** in a single executable notebook | 109-cell `examples/jss_audit/jss_case_study.ipynb` |

---

## Install

```bash
pip install pymmeans                       # core
pip install "pymmeans[plot]"               # + matplotlib for plot() / emmip() / pwpp()
pip install "pymmeans[tutorial]"           # + sklearn + jupyter + pysofra for the showcase notebook
```

Local development:

```bash
git clone https://github.com/jturner-uofl/pymmeans.git
cd pymmeans
uv venv && uv pip install -e ".[dev,plot]"
```

Requires Python ≥ 3.10. Dependencies: `numpy`, `pandas`, `patsy`, `scipy`, `statsmodels` (all standard PyData).

---

## 60-second tour

### Parametric EMM + pairwise contrasts

```python
import statsmodels.formula.api as smf
from pymmeans import emmeans, pairs

fit = smf.ols("growth ~ fertilizer * sunlight", data=df).fit()
em  = emmeans(fit, specs=["fertilizer"])
pw  = pairs(em, adjust="tukey")
```

### Mixed-effects with Kenward–Roger df

```python
import statsmodels.regression.mixed_linear_model as mlm
from pymmeans import emmeans
from pymmeans.satterthwaite import apply_kenward_roger

fit = mlm.MixedLM.from_formula(
    "reaction ~ days", df, groups="subject"
).fit(reml=True)
em_kr = apply_kenward_roger(emmeans(fit, specs=["days"]))
```

### Machine-learning g-computation via the ML adapter

```python
from sklearn.ensemble import GradientBoostingRegressor
from pymmeans import from_predict, ml_emmeans, ml_contrast

gbm = GradientBoostingRegressor().fit(X_train, y_train)
info = from_predict(
    predict_fn=lambda d: gbm.predict(d[features]),
    data=train_df, factors={"treat": [0, 1]},
    numerics=features, response="y",
)
em_ml = ml_emmeans(info, specs="treat")
ct_ml = ml_contrast(em_ml, method="trt.vs.ctrl", ref=0)
```

### Modern UQ + causal-inference extensions

```python
from pymmeans import (
    e_value,                       # VanderWeele-Ding 2017 sensitivity
    pool_imputed,                  # Rubin 1987 + Barnard-Rubin 1999 MI pooling
    split_conformal_pi,            # Vovk / Lei et al. 2018 split conformal
    conformal_counterfactual_pi,   # Lei-Candes 2021 weighted-conformal CFs
    aipw_ate,                      # Robins-Rotnitzky-Zhao 1994 doubly-robust
    cross_fit_ml_emmeans,          # Chernozhukov et al. 2018 cross-fit DML
)
```

Each implements a published peer-reviewed algorithm and ships with explicit Monte-Carlo coverage verification.

---

## What pymmeans replaces

A Python user installing `pymmeans` no longer needs to install (or maintain an R toolchain for) any of these R packages — they get equivalent or fuller coverage in one library:

| R package | What it does | pymmeans coverage |
|---|---|:---:|
| `emmeans` | Estimated marginal means, contrasts, multiplicity | ✅ full (the core) |
| `lsmeans` | Predecessor to `emmeans` | ✅ aliased (`lsmeans`, `lstrends`, `lsmip`, `lsm_options`) |
| `multcomp` | Bonferroni / Šidák / Holm / Tukey / Dunnett / mvt + `cld()` | ✅ full (incl. Hochberg, Hommel) |
| `mvtnorm` | Multivariate-t CDF for `mvt` adjustment | ✅ full (via scipy) |
| `pbkrtest` | Kenward–Roger + parametric-bootstrap nested test + KR/Satterthwaite F-tests | ✅ full (six headline functions ported) |
| `lmerTest` | Satterthwaite df for `lmer` | ✅ full |
| `EValue` | VanderWeele–Ding E-value | ✅ full |
| `effects` | Adjusted means / predicted effects | ✅ via `emmeans()` + `summary()` |
| `predictmeans`, `phia`, `gmodels::estimable` | Predicted means, post-hoc interactions, custom contrasts | ✅ subsumed |
| `mice::pool` | Rubin's-rules pooling | ⚠️ partial (pools `pymmeans` output; `mice` itself stays in R) |

See [docs/r_parity_matrix.md](docs/r_parity_matrix.md) for the per-function parity table and [docs/vs-r.md](docs/vs-r.md) for measured numerical agreement.

---

## Validation evidence

Three independent layers of evidence:

| Layer | What it covers | Where |
|---|---|---|
| **`pytest` automated suite** | 1,037 unit tests, 87% line coverage | `make test` |
| **Narrative validation notebook** | 251 documented contracts: 134 direct R cross-validations + 67 structural identities + 50 Monte-Carlo coverage checks. **0 failures.** | [`examples/jss_audit/jss_case_study.ipynb`](https://nbviewer.org/github/jturner-uofl/pymmeans/blob/main/examples/jss_audit/jss_case_study.ipynb) |
| **R reference CSVs** | 18+ reference fits regenerable from the committed R scripts | `tests/r_reference/` |

Reproduce the full evidence base in one command:

```bash
make reproduce
```

---

## Documentation

| Resource | Link |
|---|---|
| API reference (mkdocs site) | [docs/](docs/) — `make html` or `mkdocs serve` |
| R-parity matrix (90/100 strict) | [docs/r_parity_matrix.md](docs/r_parity_matrix.md) |
| Measured numerical agreement vs R | [docs/vs-r.md](docs/vs-r.md) |
| Performance benchmark report | [docs/PERFORMANCE_REPORT.md](docs/PERFORMANCE_REPORT.md) |
| Showcase notebook (tutorial + applied case studies) | `examples/jss_audit/jss_case_study.ipynb` |

---

## Limitations and scope

`pymmeans` targets the Python statistical-modelling ecosystem. R-only model classes that depend on R fitters with no production Python equivalent are out of scope — see the [limitations table in the manuscript](paper/article.pdf) for the full enumeration. Notable out-of-scope items:

- Non-linear mixed-effects (`nlme::nlme`)
- Generalized least squares with structured errors (`nlme::gls`)
- Penalized GAMs with tensor / factor-by smooths (`mgcv` beyond `statsmodels.GLMGam`)
- Zero-inflated mixed models (`glmmTMB`)
- The full `rstanarm` / `brms` formula DSL for Bayesian regression (pymmeans accepts PyMC draws directly via `from_pymc` / `from_arviz`)
- `MCMCglmm`, `gamlss`, `VGAM`, full Lumley-`survey` Taylor-series linearization

---

## Cite this work

If `pymmeans` enables your research, please cite the package:

```bibtex
@software{pymmeans,
  author       = {Turner, Jason S.},
  title        = {pymmeans: Estimated Marginal Means for Python},
  version      = {0.6.0},
  year         = {2026},
  url          = {https://github.com/jturner-uofl/pymmeans},
}
```

A `CITATION.cff` is committed to the repository root — GitHub renders a "Cite this repository" button from it automatically.

---

## Contributing

Bug reports and feature requests are welcome through GitHub Issues. For substantial contributions, please open an issue first to discuss the design before submitting a pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) for details. All contributors are expected to adhere to the [code of conduct](CODE_OF_CONDUCT.md).

---

## License

GNU General Public License v3.0 or later (GPL-3.0-or-later), matching R `emmeans`. See [LICENSE](LICENSE).

---

## Acknowledgements

The R `emmeans` package by Russell V. Lenth is the reference standard against which `pymmeans` was designed and validated. The R `pbkrtest` package by Ulrich Halekoh and Søren Højsgaard supplies the small-sample mixed-model inference machinery that `pymmeans` ports. Vincent Arel-Bundock's `marginaleffects` ecosystem informed the prediction-surface adapter design.
