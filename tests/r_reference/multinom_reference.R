# Reference for multinomial-logit EMMs (round 80).
#
# Fit a multinomial logit via ``nnet::multinom`` and dump R
# ``emmeans`` output for ``mode="prob"`` and ``mode="latent"``.
# pymmeans' ``multinom_emmeans`` is validated against these CSVs
# at ``atol=1e-3`` on point estimates and SEs.

suppressMessages({
  library(nnet)
  library(emmeans)
})

dat <- read.csv("tests/r_reference/multinom_data.csv")
dat$g <- factor(dat$g, levels = c("a", "b", "c"))
dat$y <- factor(dat$y, levels = c(0, 1, 2))

m <- multinom(y ~ x1 + g, data = dat, trace = FALSE)
cat("multinom coef:\n")
print(coef(m))

dump_emm <- function(em, file) {
  s <- as.data.frame(summary(em))
  write.csv(s, file, row.names = FALSE)
}

# prob mode: ~ y | g gives J rows per g level
em_prob <- emmeans(m, ~ y | g, mode = "prob")
dump_emm(em_prob, "tests/r_reference/multinom_emm_prob.csv")

# latent mode: ~ y | g excluding reference category
em_lat <- emmeans(m, ~ y | g, mode = "latent")
dump_emm(em_lat, "tests/r_reference/multinom_emm_latent.csv")

cat("wrote multinom_emm_{prob,latent}.csv\n")
