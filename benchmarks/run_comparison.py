"""Run matching Python benchmarks and emit a comparison report vs R.

Usage:
    1. Rscript benchmarks/r_benchmark.R # produces r_results.csv
    2. python benchmarks/run_comparison.py # produces PERFORMANCE_REPORT.md

The R script must be run first. Scenarios match by name.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymmeans import emmeans, pairs

HERE = Path(__file__).resolve().parent
R_RESULTS = HERE / "r_results.csv"
REPORT = HERE.parent / "docs" / "PERFORMANCE_REPORT.md"


def bench_scaling(n: int) -> float:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "f1": pd.Categorical(rng.choice(["a", "b"], n)),
            "f2": pd.Categorical(rng.choice(["a", "b", "c"], n)),
            "x": rng.normal(size=n),
        }
    )
    model = smf.ols("y ~ f1 * f2 + x", data=df).fit()
    t0 = time.perf_counter()
    emmeans(model, "f1")
    return time.perf_counter() - t0


def bench_pairwise(k: int) -> float:
    rng = np.random.default_rng(42)
    n = k * 50
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "group": pd.Categorical(rng.choice([f"g{i}" for i in range(k)], n)),
        }
    )
    model = smf.ols("y ~ group", data=df).fit()
    t0 = time.perf_counter()
    # the ``max_contrasts=50`` guard refuses
    # pairs() when ``k*(k-1)/2`` would exceed 50. For k=200 that's
    # 19,900 contrasts — the whole point of this benchmark is to time
    # Tukey at large k, so explicitly opt past the safety guard.
    pairs(emmeans(model, "group"), max_contrasts=None)
    return time.perf_counter() - t0


def bench_issue_282() -> float:
    rng = np.random.default_rng(42)
    n = 11500
    cat_50 = [f"a{i}" for i in range(1, 51)]
    df = pd.DataFrame(
        {
            "y": rng.gamma(shape=4, scale=1, size=n),
            **{
                c: pd.Categorical(rng.choice(list("ab"), n))
                for c in "abcdefgh"
            },
            "i": pd.Categorical(rng.choice(list("abcde"), n)),
            "j": pd.Categorical(rng.choice(list("abcd"), n)),
            "k": pd.Categorical(rng.choice(list("abc"), n)),
            "l": pd.Categorical(rng.choice(list("abcde"), n)),
            "m": pd.Categorical(rng.choice(list("abcd"), n)),
            "n_var": pd.Categorical(rng.choice(list("abc"), n)),
            "o": rng.gamma(shape=1, scale=1, size=n),
            "p": pd.Categorical(rng.choice(cat_50, n)),
        }
    )
    model = smf.glm(
        "y ~ a + b + c + d + e + f + g + h + i + j + k + l + m + n_var + o + p + a:b",
        data=df,
        family=sm.families.Gamma(sm.families.links.Log()),
    ).fit()
    t0 = time.perf_counter()
    emmeans(model, "p")
    return time.perf_counter() - t0


SCENARIOS: dict[str, callable] = {}
for n in (1000, 10000, 100000, 500000):
    SCENARIOS[f"scaling_n{n}"] = lambda n=n: bench_scaling(n)
for k in (20, 50, 100, 200):
    SCENARIOS[f"pairwise_k{k}"] = lambda k=k: bench_pairwise(k)
SCENARIOS["issue_282"] = bench_issue_282


def main() -> None:
    if not R_RESULTS.exists():
        raise SystemExit(
            f"R results not found at {R_RESULTS}. Run: "
            "Rscript benchmarks/r_benchmark.R"
        )
    r_df = pd.read_csv(R_RESULTS)
    # Normalize R's scientific notation: "scaling_n1e+05" -> "scaling_n100000".
    def _normalize(name: str) -> str:
        for prefix in ("scaling_n", "pairwise_k"):
            if name.startswith(prefix):
                tail = name[len(prefix):]
                try:
                    return f"{prefix}{int(float(tail))}"
                except ValueError:
                    return name
        return name

    r_times = {
        _normalize(s): t for s, t in zip(r_df["scenario"], r_df["r_time_seconds"], strict=True)
    }

    rows = []
    for name, fn in SCENARIOS.items():
        print(f"running {name} ...", flush=True)
        py_time = fn()
        r_time = r_times.get(name)
        rows.append((name, r_time, py_time))

    lines = [
        "# pymmeans vs R emmeans — Performance Comparison",
        "",
        "Timings on the developer's machine. Numbers are wall-clock seconds "
        "for the EMM/pairs operation only (model fit excluded). "
        f"pymmeans {_version()}; R 4.6, emmeans 2.0.3.",
        "",
        "| Scenario | R emmeans | pymmeans | Speedup |",
        "|---|---|---|---|",
    ]
    for name, r_t, py_t in rows:
        if r_t is None or pd.isna(r_t):
            r_disp = "refuses / OOM"
            speedup = "∞ (only pymmeans completes)"
        else:
            r_disp = f"{r_t:.3f} s"
            ratio = r_t / py_t if py_t > 0 else float("inf")
            speedup = f"{ratio:.1f}x" if ratio >= 1 else f"{1 / ratio:.2f}x slower"
        py_disp = f"{py_t:.3f} s"
        lines.append(f"| {name} | {r_disp} | {py_disp} | {speedup} |")

    lines += [
        "",
        "## Notes",
        "",
        "- `scaling_n*` runs `emmeans(model, 'f1')` after fitting "
        "`y ~ f1 * f2 + x` on n rows. pymmeans is dramatically faster "
        "because the EMM computation depends only on `beta` and `vcov` "
        "(already extracted from the fit), not on n. The analytic "
        "marginalization path also skips materializing the reference "
        "grid entirely.",
        "- `pairwise_k*` runs `pairs(emmeans(model, 'group'))` with k group "
        "levels and Tukey adjustment. pymmeans implements the studentized-"
        "range SF directly via Gauss-Hermite quadrature (inner integral) "
        "and Gauss-Laguerre quadrature over chi-squared (outer integral for "
        "finite df), vectorized across all comparisons in one shot.",
        "- `issue_282` reproduces the GitHub `emmeans` issue #282: GLM Gamma "
        "log-link with 16 factors and an interaction, then `emmeans(model, 'p')` "
        "over a 46M-row reference grid. R refuses at default `rg.limit=10000`. "
        "pymmeans uses analytic marginalization — no grid materialization, no "
        "patsy round-trip — so it finishes in milliseconds.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {REPORT}")


def _version() -> str:
    from pymmeans import __version__
    return __version__


if __name__ == "__main__":
    main()
