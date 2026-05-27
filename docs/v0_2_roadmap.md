# pymmeans v0.2 Roadmap

Items deferred from v0.1. Priority order reflects expected user-base
demand × implementation effort.

## Tier 1 — high user value, ~1 week each

### `summary(..., cross.adjust=)` (~8 hours)

Second-stage multiplicity correction across by-groups. Currently
each by-group is its own family.

### Full `mgcv`-style penalised smooths (~24-40 hours)

`bs(x, df=k)` and `cr(x, df=k)` basis expansions are already supported
in v0.1 via patsy's `FactorInfo.eval` re-evaluation; `ns(x, ...)` and
R's `poly(x, ...)` are not in patsy's built-in namespace and require
either user-side substitution (`cr(x, df=k)` for `ns`, pre-computed
orthonormal columns for `poly`) or a v0.2 patsy-extension shim.
True `mgcv`-style penalised smooths (`s(x, k=10, bs="tp")`) are an
R-only model class with no `statsmodels` equivalent.

## Tier 2 — niche but documented

### Kenward-Roger algorithmic refactor (~24-40 hours)

The current `_compute_pbkrtest_aux_core` materialises a
``(n_theta, n_theta, p, p)`` derivative tensor that scales O(n_theta²
· n_groups · n_g³) and risks OOM on multi-level RE structures with
`n_theta >= 20`. Two refactor paths:

- Chunk the `(r, s)` derivative loop so only one `(p, p)` slice is
  alive at a time.
- Use the analytic derivatives Kenward & Roger 1997 give in closed
  form, eliminating the finite-difference step entirely.

The current finite-difference Hessian also uses `h = max(|θ|, 1e-6) *
1e-3 ≈ 1e-9` absolute on poorly-scaled fits (e.g. heritability θ ≈
1e-3), where second-derivative roundoff hits 10-50 % relative error.
The fix is `h ≈ eps^(1/4) · max(|θ|, 1) ≈ 1.2e-4` plus Richardson
extrapolation. Both changes need a full re-validation pass against
`pbkrtest` to confirm the current `atol < 1e-4` parity claim still
holds.

### Remaining transform aliases (~16 hours)

- `power` / `sympower`
- `atanh` (Fisher's transformation)
- `asinh.sqrt`
- `bcnPower` (Box-Cox negative)
- `yj.power` (Yeo-Johnson)

## Tier 3 — defer indefinitely

### Multivariate (`mvmodel`) (~40 hours)

Multiple response variables fit jointly. No Python equivalent has
gained traction. Skip unless a specific user surfaces.

### Advanced ref-grid features (~24-40 hours)

`nesting`, `nuisance`, `counterfactuals`, `cov.keep`, `mult.names` —
these are powerful R features but the audience is small. Add on
demand.

### R-specific adapters

`as.glht`, `as.mcmc`, BRMS / rstanarm — these are R-side glue; the
Python equivalent is direct access to `posterior.beta_samples`
(already supported).

## Excluded permanently

- Multi-stratum `aovlist` — Python ecosystem uses MixedLM
- `brms`-specific path — R-only; `pymmeans` has PyMC equivalent
- `glmmTMB` / `MCMCglmm` — no Python equivalent

## Suggested v0.2 release scope

For a meaningful v0.2:

1. **GAM smooth terms** (Tier 1) — closes the spline gap on OLS/GLM
2. **`cross.adjust=`** (Tier 1) — completes the multiplicity surface
3. **Custom callable contrast methods** (Tier 2) — last common R
   parity item

Total: ~40-50 hours of focused work, deliverable as v0.2.0 within a
2-month window post-v0.1.

Multivariate / advanced ref-grid features → v0.3+.
