# Kenward-Roger reference: fit lmer models, dump pbkrtest's KR-adjusted
# vcov + df for two designs (a random-intercept fit and a random-slopes
# fit) so the pymmeans implementation can be validated against the
# canonical R reference for both shapes.
#
# #3: previously the script generated only a random-intercept
# reference, so pymmeans had no independent confirmation that its KR
# implementation handled k_re > 1. This script now also writes
# kr_reference_rs.csv + kr_reference_rs_data.csv for a random-slopes
# fit (sleepstudy ~ Days, (Days|Subject)).

suppressMessages({
 library(lme4)
 library(lmerTest)
 library(pbkrtest)
})

# ---------------------------------------------------------------------
# (1) Random-intercept reference — unchanged from the script
# ---------------------------------------------------------------------

set.seed(42)
n_groups <- 20
n_per <- 8
n <- n_groups * n_per
subj <- rep(1:n_groups, each = n_per)
g <- sample(c("a", "b", "c"), n, replace = TRUE)
u <- rnorm(n_groups, sd = 0.6)[subj]
y <- 1.0 + 0.4 * (g == "b") + 0.9 * (g == "c") + u + rnorm(n, sd = 0.5)

dat <- data.frame(y = y, g = factor(g), subj = factor(subj))
write.csv(dat, "tests/r_reference/kr_reference_data.csv", row.names = FALSE)

m <- lmer(y ~ g + (1 | subj), data = dat, REML = TRUE)
V_naive <- as.matrix(vcov(m))
V_KR <- as.matrix(pbkrtest::vcovAdj(m))

satt_df <- summary(m, ddf = "Satterthwaite")$coefficients[, "df"]
kr_df <- summary(m, ddf = "Kenward-Roger")$coefficients[, "df"]

write.csv(
 data.frame(
 name = rownames(V_KR),
 se_naive = sqrt(diag(V_naive)),
 se_kr = sqrt(diag(V_KR)),
 df_satt = unname(satt_df),
 df_kr = unname(kr_df)
 ),
 "tests/r_reference/kr_reference.csv",
 row.names = FALSE
)
cat("wrote kr_reference.csv\n")

# ---------------------------------------------------------------------
# (2) Random-slopes reference — #3 addition.
#
# Use the classic sleepstudy fit: Reaction ~ Days + (Days | Subject).
# This exercises k_re = 2 with a correlated random-effects matrix and
# is the canonical lme4 demo dataset, so anyone can sanity-check the
# reference against published pbkrtest output.
# ---------------------------------------------------------------------

data(sleepstudy, package = "lme4")
write.csv(
 sleepstudy[, c("Reaction", "Days", "Subject")],
 "tests/r_reference/kr_reference_rs_data.csv",
 row.names = FALSE
)

m_rs <- lmer(
 Reaction ~ Days + (Days | Subject),
 data = sleepstudy, REML = TRUE
)
V_naive_rs <- as.matrix(vcov(m_rs))
V_KR_rs <- as.matrix(pbkrtest::vcovAdj(m_rs))
satt_df_rs <- summary(m_rs, ddf = "Satterthwaite")$coefficients[, "df"]
kr_df_rs <- summary(m_rs, ddf = "Kenward-Roger")$coefficients[, "df"]

write.csv(
 data.frame(
 name = rownames(V_KR_rs),
 se_naive = sqrt(diag(V_naive_rs)),
 se_kr = sqrt(diag(V_KR_rs)),
 df_satt = unname(satt_df_rs),
 df_kr = unname(kr_df_rs)
 ),
 "tests/r_reference/kr_reference_rs.csv",
 row.names = FALSE
)
cat("wrote kr_reference_rs.csv\n")
