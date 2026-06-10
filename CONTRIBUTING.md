# Contributing to pymmeans

Thank you for considering a contribution. This document covers the most common cases: filing bug reports, proposing features, and submitting pull requests.

## Filing issues

Three issue templates are configured under `.github/ISSUE_TEMPLATE/`:

| Template | When to use |
|---|---|
| **Bug report** | Crashes, unexpected behaviour, or numerical mismatch against an internal pymmeans claim (test failure, docs disagreement, etc.) |
| **R-parity discrepancy** | Pymmeans output diverges from R `emmeans` / `pbkrtest` / `multcomp` on a case the docs claim should match |
| **Feature request** | New capability, model class, or extension you'd like to see |

For bug reports, the most helpful thing you can do is **include a minimal reproducing example** with all data inline (or as a small synthetic dataset that triggers the bug). For R-parity discrepancies, please include the **R reference output and the `sessionInfo()`** so we can rerun on our side.

## Development workflow

```bash
git clone https://github.com/jturner-uofl/pymmeans.git
cd pymmeans
uv venv && uv pip install -e ".[dev,plot,tutorial]"

# Run the test suite (1,037 tests)
make test

# Or, more directly
.venv/bin/pytest -q

# Lint
.venv/bin/ruff check .

# Type-check (non-strict)
.venv/bin/mypy src/pymmeans
```

Before submitting a PR, please ensure:

- `make test` passes locally
- `.venv/bin/ruff check .` is clean
- Public-facing changes include docstrings
- If the change touches the EMM algebra, an R-side reference CSV is added under `tests/r_reference/` along with the R script that produced it

## R-parity policy

`pymmeans` is validated against R `emmeans` to floating-point precision on its deterministic surface. Documented tolerance bounds are:

| Tolerance class | Example sources |
|---|---|
| Machine precision (`\|Δ\| <= 1e-14`) | Pure linear algebra: L-matrix, vcov propagation |
| Solver bound (`~1e-6`) | GLM IRLS, REML optimiser |
| QMC bound (`~1e-4`) | Multivariate-t adjustments |
| KR finite-difference bound (`~3-4%`) | Kenward-Roger df on `vc_formula` corner cases |
| Posterior MC bound (`~1e-3`) | Bayesian / PyMC paths |

Any PR claiming "matches R" must specify which class the agreement falls in. The validation notebook (`examples/jss_audit/_build_case_study.py`) and the per-test `assert ... atol=` annotations are the source of truth.

## Code style

- Black + isort default settings (handled by `ruff format`)
- Type annotations on public functions
- Docstrings in numpydoc style for public API; brief one-liners for internals
- No new dependencies without discussion in an issue first

## Scope

`pymmeans` targets the Python statistical-modelling ecosystem (`statsmodels`, `lifelines`, `linearmodels`, `scikit-learn`, `PyMC`). R-only model classes (`nlme::nlme`, `mgcv` with tensor smooths, `glmmTMB`, `rstanarm` / `brms` formula DSL, `MCMCglmm`, `gamlss`, `VGAM`, full `survey`-package Taylor-series linearisation) are out of scope and will not be added — see the manuscript's limitations table for the full list.

## Releases

The numerical surface is frozen; minor API polish is still possible until v1.0. Semver applies: breaking public-API changes require a major version bump.

## License

By contributing, you agree that your contributions will be licensed under GPL-3.0-or-later, matching the package as a whole.
