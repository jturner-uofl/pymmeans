# PBmodcomp reference: run pbkrtest::PBmodcomp on a sleepstudy
# nested-model test so the pymmeans port can be validated against
# the canonical R reference.
#
# Round 77 (parametric-bootstrap model comparison): the pbkrtest
# function tests H0 (small model is sufficient) vs H1 (large model
# is needed) by drawing nsim parametric bootstrap replicates from
# the fitted small model.  We dump the observed LRT, the chi²
# p-value, and a 5,000-iteration bootstrap p-value for sleepstudy
# `Reaction ~ Days` vs `Reaction ~ 1`.
#
# The bootstrap p-value depends on the RNG seed, so we also write
# the simulated LRT distribution to a separate CSV so the pymmeans
# guard can compare to the empirical distribution at a tolerance
# rather than to a single fragile p-value.

suppressMessages({
  library(lme4)
  library(pbkrtest)
})

data(sleepstudy, package = "lme4")
write.csv(
  sleepstudy[, c("Reaction", "Days", "Subject")],
  "tests/r_reference/pbmodcomp_data.csv",
  row.names = FALSE
)

# Use ML (REML log-likelihoods are not comparable across nestings).
large <- lmer(Reaction ~ Days + (1 | Subject),
              data = sleepstudy, REML = FALSE)
small <- lmer(Reaction ~ 1 + (1 | Subject),
              data = sleepstudy, REML = FALSE)

set.seed(20260522)
res <- PBmodcomp(large, small, nsim = 5000, cl = NULL)

lrt_obs   <- as.numeric(res$test$stat[1])
chi2_p    <- as.numeric(res$test$p.value[1])
boot_p    <- as.numeric(res$test$p.value[2])

# Save the simulated LRT distribution (the actual bootstrap draws),
# not just the summary, so pymmeans can compare its own bootstrap
# percentiles to R's at distribution level rather than at scalar
# p-value level.
ref_dist <- as.numeric(res$ref)

write.csv(
  data.frame(
    metric = c("lrt_obs", "chi2_p", "boot_p", "n_sim", "df"),
    value  = c(lrt_obs, chi2_p, boot_p, 5000, 1)
  ),
  "tests/r_reference/pbmodcomp_summary.csv",
  row.names = FALSE
)
write.csv(
  data.frame(lrt = ref_dist),
  "tests/r_reference/pbmodcomp_lrt_dist.csv",
  row.names = FALSE
)
cat("wrote pbmodcomp_summary.csv and pbmodcomp_lrt_dist.csv\n")
