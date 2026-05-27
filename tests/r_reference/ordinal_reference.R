# Reference for ordinal-regression EMMs (round 79).
#
# Fit a cumulative-link ordinal regression via ``ordinal::clm`` and
# dump the canonical R emmeans output for:
#  - mode="prob"       (per-category probabilities)
#  - mode="mean.class" (expected category)
#  - mode="cum.prob"   (cumulative probabilities)
#  - mode="linear.predictor" (≡ mode="latent")
#
# pymmeans' ``ordinal_emmeans`` is validated against these
# reference CSVs at ``atol=1e-3`` on point estimates and
# ``atol=1e-2`` on SEs.

suppressMessages({
  library(ordinal)
  library(emmeans)
})

set.seed(42)
n <- 500
x1 <- rnorm(n)
g  <- sample(c("a", "b", "c"), n, replace = TRUE)
g_b <- as.numeric(g == "b")
g_c <- as.numeric(g == "c")
eta <- 0.8 * x1 + 0.5 * g_b - 0.3 * g_c
y_lat <- eta + rlogis(n)
y_int <- ifelse(y_lat < -0.5, 0,
        ifelse(y_lat <  0.8, 1, 2))

dat <- data.frame(
  y  = factor(y_int, levels = c(0, 1, 2), ordered = TRUE),
  x1 = x1,
  g  = factor(g, levels = c("a", "b", "c"))
)
write.csv(dat, "tests/r_reference/ordinal_data.csv", row.names = FALSE)

m <- clm(y ~ x1 + g, data = dat, link = "logit")
cat("clm coef:\n"); print(coef(m))

dump_emm <- function(em, file) {
  s <- as.data.frame(summary(em))
  write.csv(s, file, row.names = FALSE)
}

# Mode = "linear.predictor"  (latent η)
em_lat <- emmeans(m, ~ g, mode = "linear.predictor")
dump_emm(em_lat, "tests/r_reference/ordinal_emm_latent.csv")

# Mode = "prob"  — need the response factor in the rhs of the
# formula to keep per-category rows.  ``emmeans(m, ~ y | g, mode="prob")``
# returns J rows per ``g`` level.
em_prob <- emmeans(m, ~ y | g, mode = "prob")
dump_emm(em_prob, "tests/r_reference/ordinal_emm_prob.csv")

# Mode = "cum.prob"  — same shape (J-1 rows per g level).
em_cum <- emmeans(m, ~ cut | g, mode = "cum.prob")
dump_emm(em_cum, "tests/r_reference/ordinal_emm_cumprob.csv")

# Mode = "mean.class"
em_mc <- emmeans(m, ~ g, mode = "mean.class")
dump_emm(em_mc, "tests/r_reference/ordinal_emm_meanclass.csv")

cat("wrote ordinal_emm_{latent,prob,cumprob,meanclass}.csv\n")
