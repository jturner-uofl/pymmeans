"""Logistic GLM example: marginal probabilities on both link and response scales.

Run with:
    python examples/glm_logistic.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from pymmeans import emmeans, pairs


def main() -> None:
    rng = np.random.default_rng(7)
    n = 600
    sex = rng.choice(["F", "M"], n)
    treatment = rng.choice(["placebo", "drug_a", "drug_b"], n)
    age = rng.normal(50, 10, n)
    eta = (
        -1.0
        + 0.6 * (treatment == "drug_a")
        + 1.1 * (treatment == "drug_b")
        + 0.2 * (sex == "M")
        + 0.02 * (age - 50)
    )
    p = 1.0 / (1.0 + np.exp(-eta))
    pain_relief = rng.binomial(1, p, size=n)

    df = pd.DataFrame(
        {
            "pain_relief": pain_relief,
            "treatment": pd.Categorical(treatment, categories=["placebo", "drug_a", "drug_b"]),
            "sex": pd.Categorical(sex),
            "age": age,
        }
    )

    model = smf.glm(
        "pain_relief ~ treatment * sex + age",
        data=df,
        family=sm.families.Binomial(),
    ).fit()
    print(model.summary().tables[1], "\n")

    print("=== EMMs by treatment (logit scale) ===")
    print(emmeans(model, "treatment"), "\n")

    print("=== EMMs by treatment (response scale = probabilities) ===")
    print(emmeans(model, "treatment", type="response"), "\n")

    print("=== EMMs by treatment, conditional on sex (response scale) ===")
    print(emmeans(model, "treatment", by="sex", type="response"), "\n")

    print("=== Pairwise treatment comparisons (Tukey on logit scale) ===")
    print(pairs(emmeans(model, "treatment")))


if __name__ == "__main__":
    main()
