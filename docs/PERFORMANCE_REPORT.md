# pymmeans vs R emmeans — Performance Comparison

Timings on the developer's machine. Numbers are wall-clock seconds for the EMM/pairs operation only (model fit excluded). pymmeans 0.2.0; R 4.6, emmeans 2.0.3.

| Scenario | R emmeans | pymmeans | Speedup |
|---|---|---|---|
| scaling_n1000 | 0.023 s | 0.002 s | 11.5x |
| scaling_n10000 | 0.015 s | 0.002 s | 7.5x |
| scaling_n100000 | 0.045 s | 0.006 s | 7.5x |
| scaling_n500000 | 0.178 s | 0.023 s | 7.7x |
| pairwise_k20 | 0.016 s | 0.018 s | 1.13x slower |
| pairwise_k50 | 0.060 s | 0.101 s | 1.68x slower |
| pairwise_k100 | 0.561 s | 0.416 s | 1.3x |
| pairwise_k200 | 9.704 s | 1.961 s | 4.9x |
| issue_282 | refuses / OOM | 0.021 s | ∞ (only pymmeans completes) |

## Notes

- `scaling_n*` runs `emmeans(model, 'f1')` after fitting `y ~ f1 * f2 + x` on n rows. pymmeans is dramatically faster because the EMM computation depends only on `beta` and `vcov` (already extracted from the fit), not on n. The analytic marginalization path also skips materializing the reference grid entirely.
- `pairwise_k*` runs `pairs(emmeans(model, 'group'))` with k group levels and Tukey adjustment. pymmeans implements the studentized-range SF directly via Gauss-Hermite quadrature (inner integral) and Gauss-Laguerre quadrature over chi-squared (outer integral for finite df), vectorized across all comparisons in one shot.
- `issue_282` reproduces the GitHub `emmeans` issue #282: GLM Gamma log-link with 16 factors and an interaction, then `emmeans(model, 'p')` over a 46M-row reference grid. R refuses at default `rg.limit=10000`. pymmeans uses analytic marginalization — no grid materialization, no patsy round-trip — so it finishes in milliseconds.
