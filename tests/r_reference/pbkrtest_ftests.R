# Reference for pbkrtest's F-test functions KRmodcomp + SATmodcomp.
#
# pymmeans ports the user-facing pbkrtest API. After
# we had `vcovAdj` (-76) and `PBmodcomp` (round
# 77). adds the asymptotic F-test counterparts to
# `PBmodcomp` — `KRmodcomp` (Kenward-Roger F) and `SATmodcomp`
# (Satterthwaite F). This script dumps the canonical pbkrtest
# output for two nested-mixed-model cases so the pymmeans port can
# be validated against floating-point.

suppressMessages({
 library(lme4)
 library(pbkrtest)
})

# -----------------------------------------------------------------
# Case 1 — sleepstudy fixed-effect addition test:
# large: Reaction ~ Days + (1 | Subject)
# small: Reaction ~ 1 + (1 | Subject)
# Same as the PBmodcomp reference; gives a textbook
# rank-1 F-test (df_num = 1).
# -----------------------------------------------------------------

data(sleepstudy, package = "lme4")

large_1 <- lmer(Reaction ~ Days + (1 | Subject), data = sleepstudy, REML = TRUE)
small_1 <- lmer(Reaction ~ 1 + (1 | Subject), data = sleepstudy, REML = TRUE)

kr_1 <- KRmodcomp(large_1, small_1)
sat_1 <- SATmodcomp(large_1, small_1)

kr_1_t <- kr_1$test
sat_1_t <- sat_1$test

cat("Case 1 KR:\n"); print(kr_1_t)
cat("Case 1 SAT:\n"); print(sat_1_t)

# -----------------------------------------------------------------
# Case 2 — multi-DF F-test (rank-2):
# large: Reaction ~ Days + I(Days^2) + (Days | Subject)
# small: Reaction ~ 1 + (Days | Subject)
# Tests both Days *and* Days^2 simultaneously; df_num = 2.
# Exercises the multi-DF branch of K-R 1997.
# -----------------------------------------------------------------

sleepstudy$Days2 <- sleepstudy$Days^2

large_2 <- lmer(
 Reaction ~ Days + Days2 + (Days | Subject),
 data = sleepstudy, REML = TRUE
)
small_2 <- lmer(
 Reaction ~ 1 + (Days | Subject),
 data = sleepstudy, REML = TRUE
)

kr_2 <- KRmodcomp(large_2, small_2)
sat_2 <- SATmodcomp(large_2, small_2)

kr_2_t <- kr_2$test
sat_2_t <- sat_2$test

cat("Case 2 KR:\n"); print(kr_2_t)
cat("Case 2 SAT:\n"); print(sat_2_t)

# -----------------------------------------------------------------
# Case 3 — vc_formula= variance-component model (the new pymmeans
# support). Canonical split-plot `oats` with TWO independent
# scalar variance components — Block and Block:Variety — plus the
# residual. We test the sub-plot `nitro` fixed effect (4 levels =>
# df_num = 3). This exercises the variance-component blocks in the
# Kenward-Roger V_beta / W / P_list kernel that the cov_re-only
# path ignores. The 63-row subset matches the Sprint-7.A fixture
# (statsmodels MixedLM converges to lme4's solution here; an
# arbitrary crossed design can collapse a component to the
# boundary in statsmodels).
# -----------------------------------------------------------------

data(oats, package = "MASS")
oa <- data.frame(
  Block = oats$B, Variety = oats$V,
  nitro = as.numeric(sub("cwt", "", as.character(oats$N))), yld = oats$Y
)
drop_idx <- c(1, 2, 3, 5, 8, 13, 21, 34, 55)   # 1-based; Sprint-7.A subset
oa <- oa[-drop_idx, ]
oa$BlockVariety <- factor(paste(oa$Block, oa$Variety, sep = ":"))
oa$Block <- factor(oa$Block)
oa$Variety <- factor(oa$Variety)
oa$nitro <- factor(oa$nitro)

large_3 <- lmer(yld ~ Variety + nitro + (1 | Block) + (1 | BlockVariety),
                data = oa, REML = TRUE)
small_3 <- lmer(yld ~ Variety + (1 | Block) + (1 | BlockVariety),
                data = oa, REML = TRUE)

kr_3 <- KRmodcomp(large_3, small_3)
sat_3 <- SATmodcomp(large_3, small_3)

kr_3_t <- kr_3$test
sat_3_t <- sat_3$test

cat("Case 3 KR (vc_formula):\n"); print(kr_3_t)
cat("Case 3 SAT (vc_formula):\n"); print(sat_3_t)

write.csv(
  oa[, c("Block", "Variety", "nitro", "yld", "BlockVariety")],
  "tests/r_reference/pbkrtest_ftests_vc_data.csv", row.names = FALSE
)

# -----------------------------------------------------------------
# Save sleepstudy data + the F-test summaries as CSVs.
# pbkrtest's `$test` data-frame has columns:
# stat, ndf, ddf, F.scaling (KR only), p.value
# -----------------------------------------------------------------

write.csv(
 sleepstudy[, c("Reaction", "Days", "Days2", "Subject")],
 "tests/r_reference/pbkrtest_ftests_data.csv",
 row.names = FALSE
)

# Flatten the four results into one tidy data frame. KRmodcomp's
# `$test` is a 2-row data.frame ("Ftest" with scaling, "FtestU"
# unscaled); SATmodcomp's `$test` is a 1-row data.frame with
# column names `statistic`/`ndf`/`ddf`/`p.value` (no F.scaling).
make_row_kr <- function(case, t) {
 data.frame(
 case = case,
 method = "KR",
 stat = as.numeric(t["Ftest", "stat"]),
 ndf = as.numeric(t["Ftest", "ndf"]),
 ddf = as.numeric(t["Ftest", "ddf"]),
 F.scaling = as.numeric(t["Ftest", "F.scaling"]),
 p.value = as.numeric(t["Ftest", "p.value"])
 )
}
make_row_sat <- function(case, t) {
 data.frame(
 case = case,
 method = "SAT",
 stat = as.numeric(t[1, "statistic"]),
 ndf = as.numeric(t[1, "ndf"]),
 ddf = as.numeric(t[1, "ddf"]),
 F.scaling = NA_real_,
 p.value = as.numeric(t[1, "p.value"])
 )
}

ftests <- rbind(
 make_row_kr ("case1", kr_1_t),
 make_row_sat("case1", sat_1_t),
 make_row_kr ("case2", kr_2_t),
 make_row_sat("case2", sat_2_t),
 make_row_kr ("case3", kr_3_t),
 make_row_sat("case3", sat_3_t)
)
write.csv(ftests, "tests/r_reference/pbkrtest_ftests.csv", row.names = FALSE)
cat("wrote pbkrtest_ftests.csv and pbkrtest_ftests_data.csv\n")
