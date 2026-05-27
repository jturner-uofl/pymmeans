
# ==========================================================================
# R benchmark script — run with: Rscript benchmarks/r_benchmark.R
# Writes results to benchmarks/r_results.csv for the comparison report.
# Uses base R's system.time() (no extra package deps).
# ==========================================================================

library(emmeans)

results <- data.frame(
  scenario = character(),
  r_time_seconds = numeric(),
  stringsAsFactors = FALSE
)

time_it <- function(name, expr) {
  t <- tryCatch(
    system.time(eval(expr))["elapsed"],
    error = function(e) {
      message(name, " FAILED: ", e$message)
      NA_real_
    }
  )
  results <<- rbind(results, data.frame(scenario = name, r_time_seconds = t))
}

# --- Scaling with n: emmeans on a binary factor ---
for (n in c(1000, 10000, 100000, 500000)) {
  set.seed(42)
  df_s <- data.frame(
    y = rnorm(n),
    f1 = factor(sample(letters[1:2], n, replace=TRUE)),
    f2 = factor(sample(letters[1:3], n, replace=TRUE)),
    x = rnorm(n)
  )
  m_s <- lm(y ~ f1 * f2 + x, data=df_s)
  time_it(paste0("scaling_n", n), quote(emmeans(m_s, "f1")))
}

# --- Pairwise with many levels ---
for (k in c(20, 50, 100, 200)) {
  set.seed(42)
  n <- k * 50
  df_p <- data.frame(
    y = rnorm(n),
    group = factor(sample(paste0("g", 1:k), n, replace=TRUE))
  )
  m_p <- lm(y ~ group, data=df_p)
  time_it(paste0("pairwise_k", k), quote(emmeans(m_p, pairwise ~ group)))
}

# --- Issue #282: R refuses by default; raise rg.limit and time it ---
set.seed(42)
n <- 11500
cat50 <- paste0("a", 1:50)
df_282 <- data.frame(
  y = rgamma(n, shape=4),
  a = sample(letters[1:2], n, replace=TRUE),
  b = sample(letters[1:2], n, replace=TRUE),
  c = sample(letters[1:2], n, replace=TRUE),
  d = sample(letters[1:2], n, replace=TRUE),
  e = sample(letters[1:2], n, replace=TRUE),
  f = sample(letters[1:2], n, replace=TRUE),
  g = sample(letters[1:2], n, replace=TRUE),
  h = sample(letters[1:2], n, replace=TRUE),
  i = sample(letters[1:5], n, replace=TRUE),
  j = sample(letters[1:4], n, replace=TRUE),
  k = sample(letters[1:3], n, replace=TRUE),
  l = sample(letters[1:5], n, replace=TRUE),
  m = sample(letters[1:4], n, replace=TRUE),
  n_var = sample(letters[1:3], n, replace=TRUE),
  o = rgamma(n, shape=1),
  p = sample(cat50, n, replace=TRUE)
)
m_282 <- glm(y ~ a+b+c+d+e+f+g+h+i+j+k+l+m+n_var+o+p+a:b,
             data=df_282, family=Gamma(link="log"))
# Don't actually try issue_282 in R — at default rg.limit it refuses, at
# raised limit it OOMs. Record the refusal explicitly.
results <- rbind(results, data.frame(
  scenario = "issue_282",
  r_time_seconds = NA_real_  # R refuses with rg.limit=10000 default
))
write.csv(results, "benchmarks/r_results.csv", row.names=FALSE)
cat("\nR benchmark results saved to benchmarks/r_results.csv\n")
print(results)
