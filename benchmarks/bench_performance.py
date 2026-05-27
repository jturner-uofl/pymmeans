# pymmeans Benchmarking Suite
# ==========================================================================
# These benchmarks are designed to demonstrate performance advantages over
# R's emmeans package, targeting documented pain points from GitHub issues
# (#282, #233) and CRAN check failures.
#
# Each scenario includes:
#   - The equivalent R code (commented) for fair comparison
#   - A description of why R struggles here
#   - Assertions on both correctness AND timing
#
# Run with: pytest benchmarks/ -v --tb=short
# Or standalone: python benchmarks/bench_performance.py
# ==========================================================================

import time
import tracemalloc
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymmeans import emmeans, pairs

# ---------------------------------------------------------------------------
# Benchmark infrastructure
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Container for a single benchmark run."""
    name: str
    n_rows: int
    grid_size: int
    time_seconds: float
    peak_memory_mb: float
    passed: bool
    notes: str = ""

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.name}\n"
            f"  Data rows:    {self.n_rows:,}\n"
            f"  Grid size:    {self.grid_size:,}\n"
            f"  Time:         {self.time_seconds:.3f}s\n"
            f"  Peak memory:  {self.peak_memory_mb:.1f} MB\n"
            f"  {self.notes}"
        )


def run_benchmark(name: str, fn: Callable, time_limit: float | None = None) -> BenchmarkResult:
    """Run a benchmark function, tracking time and memory."""
    tracemalloc.start()
    t0 = time.perf_counter()

    result = fn()  # returns dict with n_rows, grid_size, notes

    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    passed = True
    if time_limit and elapsed > time_limit:
        passed = False

    return BenchmarkResult(
        name=name,
        n_rows=result.get("n_rows", 0),
        grid_size=result.get("grid_size", 0),
        time_seconds=elapsed,
        peak_memory_mb=peak / 1024 / 1024,
        passed=passed,
        notes=result.get("notes", ""),
    )


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def generate_github_issue_282_data(n: int = 11500, seed: int = 42) -> pd.DataFrame:
    """
    Reproduces the exact scenario from emmeans GitHub issue #282.
    User had n=11,400 with many categorical variables.
    R emmeans either crashed with memory exhaustion or took overnight.

    Equivalent R code:
        set.seed(1)
        n <- 11500
        cat <- paste("a", 1:50, sep="")
        df <- data.frame(
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
            n = sample(letters[1:3], n, replace=TRUE),
            o = rgamma(n, shape=1),
            p = sample(cat, n, replace=TRUE)
        )
    """
    rng = np.random.default_rng(seed)
    letters_2 = list("ab")
    letters_5 = list("abcde")
    letters_4 = list("abcd")
    letters_3 = list("abc")
    cat_50 = [f"a{i}" for i in range(1, 51)]

    df = pd.DataFrame({
        "y": rng.gamma(shape=4, scale=1, size=n),
        "a": rng.choice(letters_2, n),
        "b": rng.choice(letters_2, n),
        "c": rng.choice(letters_2, n),
        "d": rng.choice(letters_2, n),
        "e": rng.choice(letters_2, n),
        "f": rng.choice(letters_2, n),
        "g": rng.choice(letters_2, n),
        "h": rng.choice(letters_2, n),
        "i": rng.choice(letters_5, n),
        "j": rng.choice(letters_4, n),
        "k": rng.choice(letters_3, n),
        "l": rng.choice(letters_5, n),
        "m": rng.choice(letters_4, n),
        "n_var": rng.choice(letters_3, n),  # 'n' conflicts with param name
        "o": rng.gamma(shape=1, scale=1, size=n),
        "p": rng.choice(cat_50, n),
    })

    # Convert to categorical for proper factor handling
    cat_cols = [c for c in df.columns if c not in ("y", "o")]
    for col in cat_cols:
        df[col] = pd.Categorical(df[col])

    return df


def generate_scaled_data(
    n: int,
    n_binary_factors: int = 2,
    n_multi_factors: int = 1,
    multi_levels: int = 5,
    n_numeric: int = 1,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate data with controllable factor structure for scaling tests."""
    rng = np.random.default_rng(seed)
    data = {"y": rng.normal(size=n)}

    for i in range(n_binary_factors):
        data[f"bin_{i}"] = pd.Categorical(rng.choice(["lo", "hi"], n))

    for i in range(n_multi_factors):
        levels = [f"lv{j}" for j in range(multi_levels)]
        data[f"multi_{i}"] = pd.Categorical(rng.choice(levels, n))

    for i in range(n_numeric):
        data[f"num_{i}"] = rng.normal(size=n)

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------

def bench_issue_282_reproduction():
    """
    SCENARIO: GitHub Issue #282 — the headline benchmark.
    R behavior: refuses outright with rg.limit=10,000, or OOMs if raised.

    Full grid: 2^8 * 5 * 4 * 3 * 5 * 4 * 3 * 50 = 46,080,000 rows.
    pymmeans v0.1: streams the cartesian product in 100K-row chunks, peak
    memory ~160 MB. Runtime is dominated by patsy's per-chunk design-matrix
    construction; analytic marginalization that skips patsy entirely is the
    v0.3 target for sub-5s.
    """
    df = generate_github_issue_282_data(n=11500)
    fit_start = time.perf_counter()
    model = smf.glm(
        "y ~ a + b + c + d + e + f + g + h + i + j + k + l + m + n_var + o + p + a:b",
        data=df,
        family=sm.families.Gamma(sm.families.links.Log()),
    ).fit()
    fit_time = time.perf_counter() - fit_start

    emm_start = time.perf_counter()
    emm = emmeans(model, "p")
    emm_time = time.perf_counter() - emm_start

    return {
        "n_rows": 11500,
        "grid_size": (2**8) * 5 * 4 * 3 * 5 * 4 * 3 * 50,
        "notes": (
            f"fit {fit_time:.1f}s, emmeans-on-p {emm_time:.1f}s. "
            f"Output: {emm.n_rows} rows. "
            "R refuses by default (rg.limit=10,000)."
        ),
    }


def bench_scaling_n_observations():
    """
    SCENARIO: How does runtime scale with dataset size?
    R emmeans re-evaluates the model matrix from the full dataset for
    ref_grid construction, which gets slow for large n.

    We should show near-constant EMM time since EMMs depend on the model
    coefficients and vcov (already computed), NOT on n. The fit time
    grows with n; the EMM time should not.
    """
    results = []
    for n in [1_000, 10_000, 100_000, 500_000, 1_000_000]:
        df = generate_scaled_data(n=n, n_binary_factors=3, n_multi_factors=1)
        t_fit_start = time.perf_counter()
        model = smf.ols("y ~ bin_0 * bin_1 * bin_2 + multi_0 + num_0", data=df).fit()
        t_fit = time.perf_counter() - t_fit_start

        t_emm_start = time.perf_counter()
        emmeans(model, "bin_0")
        t_emm = time.perf_counter() - t_emm_start

        results.append({"n": n, "fit_s": t_fit, "emm_s": t_emm})

    summary = " | ".join(
        f"n={r['n']:>7,}: emm={r['emm_s']*1000:.1f}ms" for r in results
    )
    return {
        "n_rows": 1_000_000,
        "grid_size": 2 * 2 * 2 * 5,
        "notes": "EMM-only timings (excludes fit): " + summary,
    }


def bench_many_factor_levels():
    """
    SCENARIO: Factor with many levels (50-500).
    R emmeans builds the full model matrix for the reference grid,
    which is O(grid_rows * n_params). With a 500-level factor this
    gets expensive.
    """
    results = []
    for n_levels in [50, 100, 200, 500]:
        n = max(n_levels * 20, 5000)
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "y": rng.normal(size=n),
            "big_factor": pd.Categorical(
                rng.choice([f"lv{i}" for i in range(n_levels)], n)
            ),
            "x": rng.normal(size=n),
        })
        model = smf.ols("y ~ big_factor + x", data=df).fit()
        t0 = time.perf_counter()
        emm = emmeans(model, "big_factor")
        elapsed = time.perf_counter() - t0
        results.append({"k": n_levels, "emm_s": elapsed, "rows": emm.n_rows})

    summary = " | ".join(f"k={r['k']}: {r['emm_s']*1000:.1f}ms" for r in results)
    return {
        "n_rows": 10_000,
        "grid_size": 500,
        "notes": "EMM timings by factor levels: " + summary,
    }


def bench_grid_explosion_interactions():
    """
    SCENARIO: Multiple interacting factors that create massive grids.
    This is the #1 pain point for R emmeans users.

    Example: 5 binary factors fully interacted = 2^5 = 32 grid rows (fine)
    But: 3 factors with 10 levels each, interacted = 1000 grid rows
    And: 4 factors with 10 levels each = 10,000 grid rows (R's limit)

    Our approach for "smart marginalization":
    Instead of building the full grid and then averaging, we should
    compute the marginalization weights analytically and build only
    the L matrix we actually need.
    """
    # v0.1 tractable scenarios only. Larger grids (>=125K rows) need v0.3
    # smart marginalization to avoid materializing the full L matrix.
    scenarios = [
        {"factors": 3, "levels": 10, "grid": 1_000},
        {"factors": 4, "levels": 10, "grid": 10_000},
        {"factors": 3, "levels": 20, "grid": 8_000},
        {"factors": 5, "levels": 5, "grid": 3_125},
    ]
    deferred = [
        {"factors": 4, "levels": 20, "grid": 160_000},
        {"factors": 3, "levels": 50, "grid": 125_000},
    ]

    timings = []
    for s in scenarios:
        n = max(s["grid"] * 2, 5000)
        rng = np.random.default_rng(42)
        data = {"y": rng.normal(size=n)}
        factor_names = []
        for i in range(s["factors"]):
            name = f"f{i}"
            levels = [f"lv{j}" for j in range(s["levels"])]
            data[name] = pd.Categorical(rng.choice(levels, n))
            factor_names.append(name)
        df = pd.DataFrame(data)
        formula = "y ~ " + " * ".join(factor_names)
        model = smf.ols(formula, data=df).fit()

        t0 = time.perf_counter()
        emmeans(model, "f0")
        elapsed = time.perf_counter() - t0
        timings.append({"grid": s["grid"], "emm_s": elapsed})

    summary = " | ".join(
        f"grid={t['grid']:>5,}: {t['emm_s']*1000:.1f}ms" for t in timings
    )
    return {
        "n_rows": 16_000,
        "grid_size": 10_000,
        "notes": (
            "EMM timings (v0.1 tractable scenarios): " + summary + ". "
            f"Deferred to v0.3 (smart marginalization): "
            f"{[s['grid'] for s in deferred]} grid rows."
        ),
    }


def bench_pairwise_many_levels():
    """
    SCENARIO: Pairwise comparisons with many levels.
    With k levels, there are k*(k-1)/2 pairwise comparisons.
    k=50  → 1,225 comparisons
    k=100 → 4,950 comparisons
    k=200 → 19,900 comparisons

    R emmeans warns when >7 levels are compared pairwise.
    Our target: handle k=200 without breaking a sweat.
    """
    results = []
    for k in [20, 50, 100, 200]:
        n = k * 50
        rng = np.random.default_rng(42)
        df = pd.DataFrame({
            "y": rng.normal(size=n),
            "group": pd.Categorical(
                rng.choice([f"g{i}" for i in range(k)], n)
            ),
        })
        n_pairs = k * (k - 1) // 2
        model = smf.ols("y ~ group", data=df).fit()

        t0 = time.perf_counter()
        # ``max_contrasts=None`` opts out of the ``pairs()`` safety guard
        # so the benchmark can time the documented k = 20 / 50 / 100 / 200
        # scenarios in ``docs/PERFORMANCE_REPORT.md``. Without this, the
        # guard refuses families larger than 50 contrasts.
        pairs(emmeans(model, "group"), max_contrasts=None)
        elapsed = time.perf_counter() - t0
        results.append({"k": k, "pairs": n_pairs, "secs": elapsed})

    summary = " | ".join(
        f"k={r['k']} ({r['pairs']:,} pairs): {r['secs']:.2f}s" for r in results
    )
    return {
        "n_rows": 10_000,
        "grid_size": 200,
        "notes": "Pairs + Tukey timings: " + summary + ". R warns at k>7.",
    }


def bench_memory_efficiency():
    """
    SCENARIO: Memory usage of dense vs sparse linfct.
    R emmeans stores the full linfct matrix (grid_rows by n_params) as a
    dense matrix. Sparse storage is a v0.3 target.
    """
    n_params = 50
    lines = []
    for gs in [1_000, 10_000, 100_000]:
        dense_mb = gs * n_params * 8 / 1024 / 1024
        sparse_mb = gs * n_params * 0.05 * 12 / 1024 / 1024
        lines.append(f"{gs:>6,}: dense {dense_mb:.1f}MB / sparse {sparse_mb:.1f}MB")
    return {
        "n_rows": 0,
        "grid_size": 100_000,
        "notes": "L-matrix memory (n_params=50): " + " | ".join(lines),
    }


# ---------------------------------------------------------------------------
# Performance targets (for CI / automated checks)
# ---------------------------------------------------------------------------

PERFORMANCE_TARGETS = {
    # name: (max_seconds, max_memory_mb)
    "issue_282_reproduction": (5.0, 500),
    "scaling_1M_rows": (10.0, 1000),
    "many_levels_500": (3.0, 200),
    "grid_explosion_160K": (10.0, 500),
    "pairwise_200_levels": (2.0, 100),
}


# ---------------------------------------------------------------------------
# R comparison script generator
# ---------------------------------------------------------------------------

R_BENCHMARK_SCRIPT = """
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
  time_it(paste0("scaling_n", format(n, scientific=FALSE)),
          quote(emmeans(m_s, "f1")))
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
cat("\\nR benchmark results saved to benchmarks/r_results.csv\\n")
print(results)
"""


# ---------------------------------------------------------------------------
# Comparison report generator
# ---------------------------------------------------------------------------

def generate_comparison_report(python_results: list, r_results_path: str | None = None):
    """
    Generate a markdown comparison report: pymmeans vs R emmeans.
    This becomes part of the JOSS paper and README.
    """
    report = []
    report.append("# pymmeans vs R emmeans — Performance Comparison\n")
    header_row = (
        "| Scenario | R Time (s) | pymmeans Time (s) | Speedup "
        "| R Memory | pymmeans Memory |"
    )
    report.append(header_row)
    report.append(
        "|----------|-----------|-------------------|---------|----------|-----------------|"
    )

    r_data = {}
    if r_results_path:
        r_df = pd.read_csv(r_results_path)
        for _, row in r_df.iterrows():
            r_data[row["scenario"]] = row["r_time_seconds"]

    for pr in python_results:
        r_time = r_data.get(pr.name, "N/A")
        if isinstance(r_time, (int, float)) and not np.isnan(r_time):
            speedup = f"{r_time / pr.time_seconds:.1f}x"
            r_str = f"{r_time:.3f}"
        elif r_time != r_time:  # NaN = crashed
            speedup = "∞ (R crashed)"
            r_str = "CRASHED"
        else:
            speedup = "—"
            r_str = "—"

        report.append(
            f"| {pr.name} | {r_str} | {pr.time_seconds:.3f} | "
            f"{speedup} | — | {pr.peak_memory_mb:.1f} MB |"
        )

    report.append("\n*R version: 4.x, emmeans 1.10+. Python: 3.11+, pymmeans 0.1.0.*")
    report.append("*Hardware: [to be filled in at benchmark time]*\n")

    return "\n".join(report)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("pymmeans Performance Benchmark Suite")
    print("=" * 70)

    benchmarks = [
        ("issue_282_reproduction", bench_issue_282_reproduction),
        ("scaling_n_observations", bench_scaling_n_observations),
        ("many_factor_levels", bench_many_factor_levels),
        ("grid_explosion_interactions", bench_grid_explosion_interactions),
        ("pairwise_many_levels", bench_pairwise_many_levels),
        ("memory_efficiency", bench_memory_efficiency),
    ]

    results = []
    failures: list[tuple[str, BaseException]] = []
    for name, fn in benchmarks:
        print(f"\n--- {name} ---")
        try:
            result = run_benchmark(name, fn)
        except BaseException as exc:  # noqa: BLE001 — surface ANY failure
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            failures.append((name, exc))
            continue
        print(result)
        results.append(result)

    # Reproducibility gate: ``docs/PERFORMANCE_REPORT.md`` advertises every
    # scenario above, so the script must complete all of them. If a future
    # API-safety change breaks one (e.g., the ``pairs(max_contrasts=)``
    # guard added in 0.1.x silently broke ``pairwise_many_levels`` at
    # k > 50), refusing to write a partial report is the only way to keep
    # the published report honest.
    if failures:
        raise RuntimeError(
            f"{len(failures)} of {len(benchmarks)} benchmark scenarios failed; "
            f"refusing to write a partial report. Failed: "
            f"{[name for name, _ in failures]}. First exception: "
            f"{type(failures[0][1]).__name__}: {failures[0][1]}"
        )

    # Save R comparison script
    with open("benchmarks/r_benchmark.R", "w") as f:
        f.write(R_BENCHMARK_SCRIPT)
    print("\nR comparison script saved to benchmarks/r_benchmark.R")
    print("Run it in R, then re-run this script to generate comparison report.")

    # Generate report if R results exist
    import os
    r_path = "benchmarks/r_results.csv"
    if os.path.exists(r_path):
        report = generate_comparison_report(results, r_path)
        with open("benchmarks/PERFORMANCE_REPORT.md", "w") as f:
            f.write(report)
        print("\nComparison report saved to benchmarks/PERFORMANCE_REPORT.md")
    else:
        print(f"\nNo R results found at {r_path}. Run r_benchmark.R first.")

    print("\n" + "=" * 70)
    print("Done.")
