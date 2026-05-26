# pymmeans

Estimated marginal means (EMMs) for Python — a native implementation of R's
[emmeans](https://cran.r-project.org/package=emmeans) package, with no R
dependency.

```python
import statsmodels.formula.api as smf
from pymmeans import emmeans, pairs

model = smf.ols("growth ~ fertilizer * sunlight", data=df).fit()
print(emmeans(model, "fertilizer"))
print(pairs(emmeans(model, "fertilizer")))
```

## What's in the box

- **`emmeans`** — marginal means with optional `by` conditioning, `at`
  overrides, response-scale back-transformation, and analytic
  marginalization that handles 4M+ row grids in milliseconds.
- **`pairs`, `contrast`** — pairwise, revpairwise, trt.vs.ctrl, polynomial,
  consecutive, and user-supplied contrasts with Tukey/Bonferroni/Holm/Šidák
  multiplicity correction.
- **`emtrends`** — slopes of continuous predictors at each EMM grid point.
- **`bootstrap_ci`** — parametric percentile CIs from N(β̂, V) for
  asymmetric / response-scale intervals.
- **`effect_size`** — Cohen's d and Hedge's g on contrast output.
- **`plot`, `emmip`** — forest and interaction plots via matplotlib.
- **`joint_tests`** — Type III joint Wald tests for every term in the model.
- **MixedLM support** — `statsmodels.formula.api.mixedlm` results work out
  of the box.

## Status

Beta (v0.1.3). Feature-complete for the v0.1 surface. Validated
against the canonical R `emmeans` reference at `atol=1e-4` on 18
reference comparisons spanning OLS, GLM, MixedLM, GEE, Cox PH,
beta regression, response-scale, by-grouped, and factor-from-numeric
edge cases.

See [Getting started](getting-started.md) for installation and the
[API reference](api/emmeans.md) for full function signatures.
