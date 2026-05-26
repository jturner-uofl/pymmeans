# R reference outputs

Ground-truth CSVs that `tests/test_vs_r.py` asserts against.

## Generating

You need R (≥ 4.0) with the `emmeans` package installed:

```bash
# macOS via Homebrew
brew install r
R -e 'install.packages(c("emmeans"), repos="https://cloud.r-project.org")'

# Then regenerate:
Rscript tests/r_reference/generate_r_reference.R
```

This writes one CSV per scenario into this directory. Commit the CSVs so CI
can run the comparison without R.

## Scenarios

| CSV | Model | What's tested |
|---|---|---|
| `warp_emm_tension_by_wool.csv` | `lm(breaks ~ wool * tension)` | EMMs, by-grouping, two-way interaction |
| `warp_pairs_tension_by_wool.csv` | same | Pairwise within by-group, Tukey adjustment |
| `pigs_emm_source.csv` | `lm(log(conc) ~ source + factor(percent))` | LHS transformation, marginalization over a numeric-as-factor |
| `pigs_pairs_source.csv` | same | Pairwise on link scale |
| `tooth_emm_supp_by_dose.csv` | `lm(len ~ supp * factor(dose))` | Two-way interaction, by-grouping |
| `spray_emm.csv` | `lm(sqrt(count) ~ spray)` | LHS transformation, one-way |
| `spray_pairs.csv` | same | Pairwise, k=6 levels |
| `neuralgia_emm_treatment_response.csv` | `glm(Pain ~ Treatment * Sex + Age, family=binomial)` | GLM, response-scale (v0.2+) |

## Why not bundle the data?

R datasets are loaded from the [Rdatasets mirror](https://vincentarelbundock.github.io/Rdatasets/)
via `statsmodels.datasets.get_rdataset()` at test time, matching whatever R
used. This keeps the repo small and avoids licensing issues.
