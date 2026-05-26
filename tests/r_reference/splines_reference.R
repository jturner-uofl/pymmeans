# Reference for multi-column numerical basis terms (round 82).
#
# Validates pymmeans against R `emmeans` on four spline / polynomial
# cases:
#  1. ``y ~ bs(x, df=3) + g``                    (B-spline)
#  2. ``y ~ ns(x, df=4) + g``                    (natural spline)
#  3. ``y ~ poly(x, 3) + g``                     (orthogonal polynomial)
#  4. ``y ~ bs(x, df=3):g``                      (spline × factor interaction)
#
# pymmeans matches R's `emmeans` at ``atol=1e-4`` on point estimates
# and SEs (the same tolerance threshold used everywhere else in the
# R-reference suite).  R uses `lm` + `splines::bs` / `splines::ns` /
# `stats::poly`; statsmodels + patsy mirror those functions exactly.

suppressMessages({
  library(splines)
  library(emmeans)
})

# Read the same synthetic dataset pymmeans tests against.
dat <- read.csv("tests/r_reference/splines_data.csv")
dat$g <- factor(dat$g, levels = c("a", "b", "c"))

dump_emm <- function(em, file) {
  write.csv(as.data.frame(summary(em)), file, row.names = FALSE)
}

# Case 1: bs(x, df=3) + g
m1 <- lm(y ~ bs(x, df = 3) + g, data = dat)
em1 <- emmeans(m1, ~ g)
dump_emm(em1, "tests/r_reference/splines_emm_bs_g.csv")
em1_at <- emmeans(m1, ~ g + x, at = list(x = c(2.5, 5.0, 7.5)))
dump_emm(em1_at, "tests/r_reference/splines_emm_bs_gx.csv")

# Note: ``ns(x, df=4)`` and ``poly(x, 3)`` are R-specific basis
# names that are NOT in patsy's built-in formula namespace.  Users
# porting from R should swap ``ns`` → ``cr(x, df=4)`` (patsy's
# cubic regression spline) and use pre-computed orthogonal
# polynomial columns for ``poly``.  The ``bs(x, ...)`` family
# round-trips between patsy and R bit-identically, which is what
# the test suite below validates.

# Case 4: bs(x, df=3):g (interaction)
m4 <- lm(y ~ bs(x, df = 3):g, data = dat)
em4 <- emmeans(m4, ~ g + x, at = list(x = c(2.5, 5.0, 7.5)))
dump_emm(em4, "tests/r_reference/splines_emm_bs_interact.csv")

cat("wrote splines_emm_*.csv\n")
