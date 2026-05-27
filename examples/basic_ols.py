"""Basic OLS example: EMMs and pairwise comparisons on a synthetic dataset.

Run with:
    python examples/basic_ols.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from pymmeans import contrast, emmeans, pairs


def main() -> None:
    rng = np.random.default_rng(42)
    n_per_cell = 30
    cells = [
        ("hi", "shade", 1.0),
        ("hi", "sun", 2.5),
        ("med", "shade", 0.5),
        ("med", "sun", 2.0),
        ("lo", "shade", 0.0),
        ("lo", "sun", 1.0),
    ]
    rows = []
    for fert, sun, mean in cells:
        for _ in range(n_per_cell):
            rows.append({"fertilizer": fert, "sunlight": sun, "growth": rng.normal(mean, 0.5)})
    df = pd.DataFrame(rows)
    df["fertilizer"] = pd.Categorical(df["fertilizer"], categories=["lo", "med", "hi"])
    df["sunlight"] = pd.Categorical(df["sunlight"], categories=["shade", "sun"])

    model = smf.ols("growth ~ fertilizer * sunlight", data=df).fit()
    print("Model R^2:", round(model.rsquared, 4), "\n")

    print("=== Marginal means for fertilizer (averaging over sunlight) ===")
    print(emmeans(model, "fertilizer"), "\n")

    print("=== Marginal means for fertilizer, conditional on sunlight ===")
    print(emmeans(model, "fertilizer", by="sunlight"), "\n")

    print("=== Pairwise comparisons of fertilizer levels (Tukey-adjusted) ===")
    print(pairs(emmeans(model, "fertilizer")), "\n")

    print("=== Treatment vs control (lo as reference) ===")
    print(contrast(emmeans(model, "fertilizer"), method="trt.vs.ctrl", ref="lo"))


if __name__ == "__main__":
    main()
