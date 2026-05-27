"""Generate a sample emmip()-style interaction plot for the README.

Run with:
    python examples/make_plot_artifact.py
Writes docs/example_interaction_plot.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from pymmeans import emmip

OUT = Path(__file__).resolve().parent.parent / "docs" / "example_interaction_plot.png"


def main() -> None:
    rng = np.random.default_rng(42)
    n_per_cell = 50
    rows = []
    for fert, sun, mean in [
        ("lo", "shade", 0.0),
        ("lo", "sun", 1.0),
        ("med", "shade", 0.5),
        ("med", "sun", 2.0),
        ("hi", "shade", 1.0),
        ("hi", "sun", 2.5),
    ]:
        for _ in range(n_per_cell):
            rows.append({"fertilizer": fert, "sunlight": sun, "growth": rng.normal(mean, 0.4)})
    df = pd.DataFrame(rows)
    df["fertilizer"] = pd.Categorical(df["fertilizer"], categories=["lo", "med", "hi"])
    df["sunlight"] = pd.Categorical(df["sunlight"], categories=["shade", "sun"])

    model = smf.ols("growth ~ fertilizer * sunlight", data=df).fit()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    emmip(model, x="fertilizer", by="sunlight", ax=ax)
    ax.set_title("emmip(model, x='fertilizer', by='sunlight')")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
