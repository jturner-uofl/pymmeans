# Reference for the multivariate-OLS marginal-means surface
# (`multivariate_emmeans` + `mvcontrast`). pymmeans's MVP slice supports
# statsmodels' `_MultivariateOLS` — R's analogue is `lm(cbind(y1,y2,y3)
# ~ x)`, an `mlm` fit that `emmeans` extends with a `rep.meas`
# pseudo-factor and `mvcontrast` reduces to Hotelling-T² / F-tests.
#
# This script:
#   1. Generates the 3-response × 3-group seeded fixture (n=60).
#   2. Writes the data CSV so the Python test fits an identical model.
#   3. Writes the per-cell × per-response EMM table (R reference).
#   4. Writes the `mvcontrast` pairwise Hotelling output (R reference).

suppressMessages({
 library(emmeans)
})

set.seed(1010)
n <- 60
g <- factor(sample(c("a", "b", "c"), n, replace = TRUE))
x <- rnorm(n)
beta_mv <- c(a = 0, b = 0.5, c = 1)
y1 <- beta_mv[as.character(g)] + 0.3 * x + rnorm(n, sd = 0.5)
y2 <- 0.7 * beta_mv[as.character(g)]      + rnorm(n, sd = 0.5)
y3 <- 0.4 * beta_mv[as.character(g)] + 0.2 * x + rnorm(n, sd = 0.5)
dat <- data.frame(g = g, x = x, y1 = y1, y2 = y2, y3 = y3)
write.csv(dat, "tests/r_reference/multivariate_data.csv", row.names = FALSE)

mv_fit <- lm(cbind(y1, y2, y3) ~ g + x, data = dat)

em <- as.data.frame(emmeans(mv_fit, ~ g | rep.meas))
write.csv(data.frame(
 g        = em$g,
 rep_meas = em$rep.meas,
 emmean   = em$emmean,
 SE       = em$SE,
 df       = em$df,
 lower_cl = em$lower.CL,
 upper_cl = em$upper.CL
), "tests/r_reference/multivariate_emm.csv", row.names = FALSE)

mvc <- as.data.frame(mvcontrast(
 emmeans(mv_fit, ~ g, by = "rep.meas"),
 "pairwise", mult.name = "rep.meas"
))
write.csv(data.frame(
 contrast = mvc$contrast,
 T_square = mvc$T.square,
 df1      = mvc$df1,
 df2      = mvc$df2,
 F_ratio  = mvc$F.ratio,
 p_value  = mvc$p.value
), "tests/r_reference/multivariate_mvc.csv", row.names = FALSE)

cat("wrote multivariate_data.csv, multivariate_emm.csv, multivariate_mvc.csv\n")
