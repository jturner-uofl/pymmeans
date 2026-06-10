---
title: pymmeans — Estimated Marginal Means for Python
hide:
  - navigation
  - toc
---

<div class="hero" markdown>

# pymmeans { .hero-title }

**Estimated marginal means for Python.** A native implementation of R's
`emmeans` and `pbkrtest`, validated to floating-point precision,
integrated with modern causal-inference and uncertainty-quantification
methods.

[Get started](getting-started.md){ .md-button .md-button--primary }
[GitHub](https://github.com/jturner-uofl/pymmeans){ .md-button }
[Validation notebook](https://nbviewer.org/github/jturner-uofl/pymmeans/blob/main/examples/jss_audit/jss_case_study.ipynb){ .md-button }

</div>

---

## At a glance

<div class="grid cards" markdown>

-   :material-check-decagram: __Floating-point R parity__

    ---

    134 direct cross-validation contracts against R `emmeans` reference
    values, mostly at `atol = 1e-14`.

    [R-parity matrix →](r_parity_matrix.md)

-   :material-speedometer: __Faster than R on common paths__

    ---

    1.6–4.1 times faster than R `emmeans` across four of six
    representative workloads; the remaining two are diagnosed.

    [Performance report →](PERFORMANCE_REPORT.md)

-   :material-robot-outline: __Machine-learning g-computation__

    ---

    Plug any sklearn-style `predict()` callable into the EMM grammar
    via `from_predict`. No R `emmeans` analogue.

    [Tour →](getting-started.md)

-   :material-shield-check-outline: __Modern causal-inference & UQ__

    ---

    E-value sensitivity, Rubin's-rules pooling, split + counterfactual
    conformal, AIPW, cross-fitted DML. All bundled.

    [API reference →](api/emmeans.md)

-   :material-test-tube: __1,037 tests, 87 % coverage__

    ---

    251 enumerated validation contracts in a single executable notebook.
    Zero failures.

    [Validation →](vs-r.md)

-   :material-language-python: __Pure Python__

    ---

    Depends only on `numpy`, `pandas`, `patsy`, `scipy`, `statsmodels`.
    No R toolchain required.

    [Install ↓](#installation)

</div>

---

## Installation

```bash
pip install pymmeans
```

Optional extras for plotting and the showcase notebook:

```bash
pip install "pymmeans[plot]"
pip install "pymmeans[tutorial]"
```

Python 3.10 or later required.

---

## Usage

=== "Parametric EMMs"

    ```python
    import statsmodels.formula.api as smf
    from pymmeans import emmeans, pairs

    fit = smf.ols("growth ~ fertilizer * sunlight", data=df).fit()
    em  = emmeans(fit, specs=["fertilizer"])
    pw  = pairs(em, adjust="tukey")
    ```

=== "Mixed models + Kenward–Roger"

    ```python
    import statsmodels.regression.mixed_linear_model as mlm
    from pymmeans import emmeans
    from pymmeans.satterthwaite import apply_kenward_roger

    fit   = mlm.MixedLM.from_formula(
        "reaction ~ days", df, groups="subject"
    ).fit(reml=True)
    em_kr = apply_kenward_roger(emmeans(fit, specs=["days"]))
    ```

=== "ML adapter"

    ```python
    from sklearn.ensemble import GradientBoostingRegressor
    from pymmeans import from_predict, ml_emmeans, ml_contrast

    gbm  = GradientBoostingRegressor().fit(X_train, y_train)
    info = from_predict(
        predict_fn=lambda d: gbm.predict(d[features]),
        data=train_df, factors={"treat": [0, 1]},
        numerics=features, response="y",
    )
    em_ml = ml_emmeans(info, specs="treat")
    ct_ml = ml_contrast(em_ml, method="trt.vs.ctrl", ref=0)
    ```

=== "Causal-inference extensions"

    ```python
    from pymmeans import (
        e_value,                       # VanderWeele & Ding (2017)
        pool_imputed,                  # Rubin (1987); Barnard & Rubin (1999)
        split_conformal_pi,            # Vovk et al. (2005); Lei et al. (2018)
        conformal_counterfactual_pi,   # Lei & Candès (2021)
        aipw_ate,                      # Robins, Rotnitzky & Zhao (1994)
        cross_fit_ml_emmeans,          # Chernozhukov et al. (2018)
    )
    ```

---

## What pymmeans replaces

A Python user installing `pymmeans` no longer needs to install or
maintain an R toolchain for any of the following packages:

| R package | Capability                                                              | Coverage   |
|-----------|-------------------------------------------------------------------------|------------|
| `emmeans`   | EMMs, contrasts, multiplicity, response-scale back-transforms         | Full       |
| `lsmeans`   | Predecessor to `emmeans`                                              | Full       |
| `multcomp`  | Bonferroni / Šidák / Holm / Tukey / Dunnett / mvt / Hochberg / Hommel | Full       |
| `mvtnorm`   | Multivariate-t CDF for `mvt` adjustment                              | Full (SciPy backend) |
| `pbkrtest`  | Kenward–Roger covariance + parametric-bootstrap nested test + F-tests | Full (6 functions) |
| `lmerTest`  | Satterthwaite df for `lmer`                                          | Full       |
| `EValue`    | VanderWeele–Ding E-value                                             | Full       |
| `effects`, `predictmeans`, `phia`, `gmodels::estimable` | Adjusted means, post-hoc, custom contrasts | Subsumed |
| `mice::pool` | Rubin's-rules pooling                                                | Partial    |

See the [R-parity matrix](r_parity_matrix.md) for the full per-function
table.

---

## Cite this work

```bibtex
@software{pymmeans,
  author  = {Turner, Jason S.},
  title   = {pymmeans: Estimated Marginal Means for Python},
  version = {0.6.0},
  year    = {2026},
  url     = {https://github.com/jturner-uofl/pymmeans},
}
```

A `CITATION.cff` file is committed to the repository root; GitHub
renders a "Cite this repository" button from it automatically.

---

<small>
`pymmeans` is released under GPL-3.0-or-later, matching the licence of
R `emmeans`. See the [GitHub repository](https://github.com/jturner-uofl/pymmeans)
for source, releases, issue tracker, and contribution guidelines.
</small>
