"""Generate cross-validation reference datasets.

Writes the exact same CSV files that ``cross_validation.R`` reads, so
the R cross-validation and Python tests work on identical data.

Run from the repo root:
    python tests/r_reference/generate_cv_data.py

Then run R to produce the reference outputs:
    Rscript tests/r_reference/cross_validation.R

Then run the Python harness:
    pytest tests/test_r_benchmark.py -v
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent


def _afex() -> pd.DataFrame:
    """2x3 factorial between-subjects ANOVA."""
    rng = np.random.default_rng(1234)
    n_per_cell = 20
    rows = []
    for a in ("a1", "a2"):
        for b in ("b1", "b2", "b3"):
            for _ in range(n_per_cell):
                interaction = 0.5 if (a == "a2" and b == "b2") else 0.0
                y = (
                    2.0
                    + 0.4 * (a == "a2")
                    + 0.6 * (b == "b2")
                    + 0.3 * (b == "b3")
                    + interaction
                    + rng.normal(scale=0.4)
                )
                rows.append({"A": a, "B": b, "y": y})
    return pd.DataFrame(rows)


def _lme4_ri() -> pd.DataFrame:
    """Random-intercept dataset: y ~ treatment + x + (1|subject)."""
    rng = np.random.default_rng(42)
    n_subj, n_per = 20, 6
    subj = np.repeat(np.arange(n_subj), n_per)
    treatment = np.tile(np.repeat(["ctrl", "drug"], n_per // 2), n_subj)
    x = rng.normal(size=n_subj * n_per)
    u = rng.normal(scale=0.5, size=n_subj)
    y = (
        1.0
        + 0.4 * (treatment == "drug")
        + 0.3 * x
        + u[subj]
        + rng.normal(scale=0.4, size=n_subj * n_per)
    )
    return pd.DataFrame({"subject": subj, "treatment": treatment, "x": x, "y": y})


def _lme4_rs() -> pd.DataFrame:
    """Random intercept + slope on x."""
    rng = np.random.default_rng(2025)
    n_subj, n_per = 30, 8
    rows = []
    u_int = rng.normal(scale=0.5, size=n_subj)
    u_slope = rng.normal(scale=0.2, size=n_subj)
    for s in range(n_subj):
        for t in range(n_per):
            treatment = ("ctrl", "drug")[t % 2]
            x = rng.normal()
            y = (
                1.0
                + 0.4 * (treatment == "drug")
                + 0.3 * x
                + u_int[s]
                + u_slope[s] * x
                + rng.normal(scale=0.3)
            )
            rows.append({"subject": s, "trial": t, "treatment": treatment, "x": x, "y": y})
    return pd.DataFrame(rows)


def _marginal() -> pd.DataFrame:
    """Marginaleffects reference: y ~ a*b + z."""
    rng = np.random.default_rng(7)
    n = 300
    df = pd.DataFrame(
        {
            "a": rng.choice(["A", "B", "C"], n),
            "b": rng.choice(["x", "y"], n),
            "z": rng.normal(size=n),
        }
    )
    df["y"] = (
        1
        + 0.5 * (df["a"] == "B")
        - 0.3 * (df["a"] == "C")
        + 0.7 * (df["b"] == "y")
        + 0.4 * df["z"]
        + 0.2 * (df["a"] == "B") * (df["b"] == "y")
        + rng.normal(scale=0.4, size=n)
    )
    return df


def _survey_srs() -> pd.DataFrame:
    """Survey SRS reference: y ~ a + x with weights."""
    rng = np.random.default_rng(99)
    n = 200
    df = pd.DataFrame(
        {
            "a": rng.choice(["A", "B", "C"], n),
            "x": rng.normal(size=n),
            "w": rng.uniform(1.0, 5.0, n),
        }
    )
    df["y"] = (
        1
        + 0.5 * (df["a"] == "B")
        - 0.3 * (df["a"] == "C")
        + 0.7 * df["x"]
        + rng.normal(scale=0.4, size=n)
    )
    return df


def _survey_poisson() -> pd.DataFrame:
    """Survey Poisson GLM reference (#1 regression). The
    previous code was using an OLS-style sandwich + working residuals
    for GLMs; this dataset exercises the GLM-specific code path."""
    rng = np.random.default_rng(123)
    n = 300
    df = pd.DataFrame(
        {
            "a": rng.choice(["A", "B", "C"], n),
            "x": rng.normal(size=n),
            "w": rng.uniform(0.5, 4.0, n),
        }
    )
    eta = 0.2 + 0.5 * (df["a"] == "B") - 0.3 * (df["a"] == "C") + 0.4 * df["x"]
    df["y"] = rng.poisson(np.exp(eta))
    return df


def _survey_binomial() -> pd.DataFrame:
    """Survey Binomial logit GLM reference (coverage gap):
    verify the IRLS bread + score factor are correct for a logit link,
    not just log link."""
    rng = np.random.default_rng(456)
    n = 250
    df = pd.DataFrame(
        {
            "a": rng.choice(["A", "B"], n),
            "x": rng.normal(size=n),
            "w": rng.uniform(0.5, 3.0, n),
        }
    )
    eta = -0.3 + 0.8 * (df["a"] == "B") + 0.5 * df["x"]
    p = 1.0 / (1.0 + np.exp(-eta))
    df["y"] = (rng.uniform(size=n) < p).astype(int)
    return df


def _survey_gamma() -> pd.DataFrame:
    """Survey Gamma log GLM reference. Non-canonical link (score factor
    is 1/mu, not 1), so this exercises the score_factor branch
    distinctly from Poisson / Binomial."""
    rng = np.random.default_rng(789)
    n = 250
    df = pd.DataFrame(
        {
            "a": rng.choice(["A", "B", "C"], n),
            "x": rng.normal(size=n),
            "w": rng.uniform(0.5, 3.0, n),
        }
    )
    eta = 0.5 + 0.4 * (df["a"] == "B") - 0.3 * (df["a"] == "C") + 0.2 * df["x"]
    mu = np.exp(eta)
    df["y"] = rng.gamma(shape=2.0, scale=mu / 2.0)
    return df


def _exposure() -> pd.DataFrame:
    """Poisson GLM with exposure offset."""
    rng = np.random.default_rng(101)
    df = pd.DataFrame(
        {
            "a": np.repeat(["A", "B"], 10),
            "exposure": np.linspace(0.2, 5.0, 20),
        }
    )
    df["y"] = rng.poisson(
        np.exp(0.3 + 0.5 * (df["a"] == "B") + np.log(df["exposure"]))
    )
    return df


def _bias_adjust() -> pd.DataFrame:
    """log(y) ~ a OLS with moderately-large sigma."""
    rng = np.random.default_rng(44)
    a = np.repeat(["A", "B"], 60)
    y = np.exp(np.where(a == "A", 1.0, 1.5) + rng.normal(scale=0.6, size=120))
    return pd.DataFrame({"a": a, "y": y})


def main() -> None:
    datasets = {
        "afex_data.csv": _afex(),
        "lme4_ri_data.csv": _lme4_ri(),
        "lme4_rs_data.csv": _lme4_rs(),
        "marginal_data.csv": _marginal(),
        "survey_srs_data.csv": _survey_srs(),
        "survey_poisson_data.csv": _survey_poisson(),
        "survey_binomial_data.csv": _survey_binomial(),
        "survey_gamma_data.csv": _survey_gamma(),
        "exposure_data.csv": _exposure(),
        "bias_adjust_data.csv": _bias_adjust(),
    }
    for name, df in datasets.items():
        df.to_csv(OUT / name, index=False)
        print(f"wrote {OUT / name} ({len(df)} rows)")


if __name__ == "__main__":
    main()
