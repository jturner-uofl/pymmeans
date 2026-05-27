#!/usr/bin/env Rscript
# Cross-validation reference data generator.
#
# Runs every R-package model we cross-validate against and writes the
# expected values to CSV files in this directory. `tests/test_r_benchmark.py`
# loads these CSVs and asserts pymmeans matches.
#
# Required R packages: afex, emmeans, lme4, lmerTest, pbkrtest, survey,
# marginaleffects.
#
# Re-run when:
# - You change a numerical algorithm in pymmeans that touches one of the
# covered model classes (OLS/GLM/MixedLM/survey).
# - You bump a referenced R package and want to re-baseline.
#
# Each block:
# 1. Generates the dataset in Python first (so RNG is shared across
# pymmeans + R; we read the CSV the Python harness writes).
# 2. Fits the R reference model.
# 3. Writes EMMs / contrasts / df / SE to a named CSV.

suppressMessages({
 library(afex)
 library(emmeans)
 library(lme4)
 library(lmerTest)
 library(pbkrtest)
 library(survey)
 library(marginaleffects)
})

# Resolve script directory — works under Rscript (commandArgs has --file=)
# and under source() (sys.frame trick) fallbacks. CI calls via Rscript.
args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
if (length(script_arg) > 0) {
 OUT_DIR <- normalizePath(dirname(sub("^--file=", "", script_arg)))
} else {
 OUT_DIR <- normalizePath(".")
}

write_emm <- function(em, path, type_col = "emmean") {
 df <- as.data.frame(summary(em))
 # Normalize column name to a consistent set
 names(df)[names(df) == "emmean"] <- "emmean"
 names(df)[names(df) == "response"] <- "emmean"
 names(df)[names(df) == "rate"] <- "emmean"
 names(df)[names(df) == "prob"] <- "emmean"
 write.csv(df, path, row.names = FALSE)
}

write_pairs <- function(p, path) {
 df <- as.data.frame(summary(p))
 write.csv(df, path, row.names = FALSE)
}

# === Block 1: afex factorial ANOVA ===
{
 df <- read.csv(file.path(OUT_DIR, "afex_data.csv"))
 df$A <- factor(df$A); df$B <- factor(df$B)
 fit <- lm(y ~ A * B, data = df)
 write_emm(emmeans(fit, ~ A | B), file.path(OUT_DIR, "afex_emm_A_by_B.csv"))
 write_pairs(pairs(emmeans(fit, ~ A | B)), file.path(OUT_DIR, "afex_pairs_A_by_B.csv"))
 write.csv(as.data.frame(joint_tests(fit)),
 file.path(OUT_DIR, "afex_joint_tests.csv"), row.names = FALSE)
}

# === Block 2: lme4 + lmerTest Satterthwaite (random intercept) ===
{
 df <- read.csv(file.path(OUT_DIR, "lme4_ri_data.csv"))
 df$treatment <- factor(df$treatment)
 fit <- lmer(y ~ treatment + x + (1 | subject), data = df, REML = TRUE)
 em <- emmeans(fit, ~ treatment, lmer.df = "satterthwaite")
 write_emm(em, file.path(OUT_DIR, "lme4_ri_emm_satt.csv"))
 write_pairs(pairs(em), file.path(OUT_DIR, "lme4_ri_pairs_satt.csv"))
}

# === Block 3: lme4 Satterthwaite (random slopes) ===
{
 df <- read.csv(file.path(OUT_DIR, "lme4_rs_data.csv"))
 df$treatment <- factor(df$treatment)
 fit <- lmer(y ~ treatment + x + (1 + x | subject), data = df, REML = TRUE)
 em <- emmeans(fit, ~ treatment, lmer.df = "satterthwaite")
 write_emm(em, file.path(OUT_DIR, "lme4_rs_emm_satt.csv"))
}

# === Block 4: lme4 + pbkrtest Kenward-Roger ===
{
 df <- read.csv(file.path(OUT_DIR, "lme4_ri_data.csv"))
 df$treatment <- factor(df$treatment)
 fit <- lmer(y ~ treatment + x + (1 | subject), data = df, REML = TRUE)
 em <- emmeans(fit, ~ treatment, lmer.df = "kenward-roger")
 write_emm(em, file.path(OUT_DIR, "lme4_ri_emm_kr.csv"))
 # Also write the V_KR diagonal directly for direct vcov check
 V_KR <- vcovAdj(fit)
 out <- data.frame(
 param = colnames(V_KR),
 var_kr = diag(V_KR),
 se_kr = sqrt(diag(V_KR))
 )
 write.csv(out, file.path(OUT_DIR, "lme4_ri_vcov_kr.csv"), row.names = FALSE)
}

# === Block 5: marginaleffects + emmeans ===
{
 df <- read.csv(file.path(OUT_DIR, "marginal_data.csv"))
 df$a <- factor(df$a); df$b <- factor(df$b)
 fit <- lm(y ~ a * b + z, data = df)
 write_emm(emmeans(fit, ~ a), file.path(OUT_DIR, "marginal_emm_a.csv"))
 write_pairs(pairs(emmeans(fit, ~ a)),
 file.path(OUT_DIR, "marginal_pairs_a.csv"))
}

# === Block 6: survey::svyglm SRS (Gaussian) ===
{
 df <- read.csv(file.path(OUT_DIR, "survey_srs_data.csv"))
 df$a <- factor(df$a)
 design <- svydesign(ids = ~1, weights = ~w, data = df)
 fit <- svyglm(y ~ a + x, design = design)
 write_emm(emmeans(fit, ~ a), file.path(OUT_DIR, "survey_srs_emm.csv"))
 coefs <- as.data.frame(summary(fit)$coefficients)
 coefs$param <- rownames(coefs)
 write.csv(coefs, file.path(OUT_DIR, "survey_srs_coef.csv"), row.names = FALSE)
}

# === Block 6b: survey::svyglm Poisson (regression) ===
{
 df <- read.csv(file.path(OUT_DIR, "survey_poisson_data.csv"))
 df$a <- factor(df$a)
 design <- svydesign(ids = ~1, weights = ~w, data = df)
 fit <- svyglm(y ~ a + x, design = design, family = quasipoisson())
 write_emm(emmeans(fit, ~ a, type = "response"),
 file.path(OUT_DIR, "survey_poisson_emm.csv"))
 coefs <- as.data.frame(summary(fit)$coefficients)
 coefs$param <- rownames(coefs)
 write.csv(coefs, file.path(OUT_DIR, "survey_poisson_coef.csv"),
 row.names = FALSE)
}

# === Block 6c: survey::svyglm Binomial logit (coverage) ===
{
 df <- read.csv(file.path(OUT_DIR, "survey_binomial_data.csv"))
 df$a <- factor(df$a)
 design <- svydesign(ids = ~1, weights = ~w, data = df)
 fit <- svyglm(y ~ a + x, design = design, family = quasibinomial())
 write_emm(emmeans(fit, ~ a, type = "response"),
 file.path(OUT_DIR, "survey_binomial_emm.csv"))
 coefs <- as.data.frame(summary(fit)$coefficients)
 coefs$param <- rownames(coefs)
 write.csv(coefs, file.path(OUT_DIR, "survey_binomial_coef.csv"),
 row.names = FALSE)
}

# === Block 6d: survey::svyglm Gamma log (non-canonical link) ===
{
 df <- read.csv(file.path(OUT_DIR, "survey_gamma_data.csv"))
 df$a <- factor(df$a)
 design <- svydesign(ids = ~1, weights = ~w, data = df)
 fit <- svyglm(y ~ a + x, design = design, family = Gamma(link = "log"))
 write_emm(emmeans(fit, ~ a, type = "response"),
 file.path(OUT_DIR, "survey_gamma_emm.csv"))
 coefs <- as.data.frame(summary(fit)$coefficients)
 coefs$param <- rownames(coefs)
 write.csv(coefs, file.path(OUT_DIR, "survey_gamma_coef.csv"),
 row.names = FALSE)
}

# === Block 7: GLM exposure offset (Poisson with log(exposure)) ===
{
 df <- read.csv(file.path(OUT_DIR, "exposure_data.csv"))
 df$a <- factor(df$a)
 fit <- glm(y ~ a + offset(log(exposure)), data = df, family = poisson())
 write_emm(emmeans(fit, ~ a, type = "response"),
 file.path(OUT_DIR, "exposure_emm_response.csv"))
}

# === Block 8: bias_adjust on log-LHS OLS (R's Taylor formula) ===
{
 df <- read.csv(file.path(OUT_DIR, "bias_adjust_data.csv"))
 df$a <- factor(df$a)
 fit <- lm(log(y) ~ a, data = df)
 out <- summary(emmeans(fit, ~ a, type = "response",
 bias.adjust = TRUE, sigma = sigma(fit)))
 write.csv(as.data.frame(out),
 file.path(OUT_DIR, "bias_adjust_emm.csv"), row.names = FALSE)
}

cat("Cross-validation reference CSVs written to", OUT_DIR, "\n")
