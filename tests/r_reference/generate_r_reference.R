# Generate R emmeans reference outputs for validation tests.
# Run from the repo root:
#   Rscript tests/r_reference/generate_r_reference.R
#
# Writes one CSV per scenario into tests/r_reference/.
# These CSVs are the ground truth that test_vs_r.py asserts against.

library(emmeans)

out_dir <- "tests/r_reference"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

write_emm <- function(emm, name) {
  df <- as.data.frame(summary(emm))
  write.csv(df, file.path(out_dir, paste0(name, ".csv")), row.names = FALSE)
}

# --- pigs: one-way EMMs + pairwise ---
# pigs is in the emmeans package and not on the Rdatasets mirror, so we
# also export the raw data for the Python-side test to load.
write.csv(pigs, file.path(out_dir, "pigs_data.csv"), row.names = FALSE)
pigs.lm <- lm(log(conc) ~ source + factor(percent), data = pigs)
write_emm(emmeans(pigs.lm, "source"), "pigs_emm_source")
write_emm(pairs(emmeans(pigs.lm, "source")), "pigs_pairs_source")

# --- warpbreaks: interaction + by groups ---
warp.lm <- lm(breaks ~ wool * tension, data = warpbreaks)
write_emm(emmeans(warp.lm, ~ tension | wool), "warp_emm_tension_by_wool")
write_emm(pairs(emmeans(warp.lm, ~ tension | wool)), "warp_pairs_tension_by_wool")

# --- neuralgia: logistic GLM ---
write.csv(neuralgia, file.path(out_dir, "neuralgia_data.csv"), row.names = FALSE)
neuralgia.glm <- glm(Pain ~ Treatment * Sex + Age,
                     family = binomial, data = neuralgia)
write_emm(emmeans(neuralgia.glm, "Treatment", type = "response"),
          "neuralgia_emm_treatment_response")

# --- ToothGrowth: factorial OLS ---
tooth.lm <- lm(len ~ supp * factor(dose), data = ToothGrowth)
write_emm(emmeans(tooth.lm, ~ supp | dose), "tooth_emm_supp_by_dose")

# --- InsectSprays: one-way ---
spray.lm <- lm(sqrt(count) ~ spray, data = InsectSprays)
write_emm(emmeans(spray.lm, "spray"), "spray_emm")
write_emm(pairs(emmeans(spray.lm, "spray")), "spray_pairs")

cat("Reference CSVs written to", out_dir, "\n")
