"""Test for the AnovaRM long-format RM equivalence claim.

R's `aov(y ~ cond + Error(subj/cond))` and `lmer(y ~ cond + (1|subj))`
produce IDENTICAL EMMs (point estimate AND standard error). The natural
Python fit for this design is statsmodels MixedLM with subject as the
random intercept — pymmeans's EMM machinery on that fit reproduces
R lmer / aov+Error to optimiser precision.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import emmeans


def test_mixedlm_emm_matches_R_lmer_and_aov_error():
    """The §XI notebook claim: statsmodels MixedLM EMMs reproduce R's
    `lmer(y ~ cond + (1|subj))` (== R's `aov + Error(subj/cond)`) to
    optimiser precision on the seeded 1-way RM fixture. R reference
    values come from ``cs_ref/rm_lmer_emm.csv``."""
    data_path = Path("examples/jss_audit/cs_ref/rm_data.csv")
    ref_path = Path("examples/jss_audit/cs_ref/rm_lmer_emm.csv")
    if not (data_path.exists() and ref_path.exists()):
        pytest.skip(
            "Run examples/jss_audit/generate_case_study_reference.R "
            "to seed the AnovaRM equivalence references."
        )
    d = pd.read_csv(data_path)
    d["subj"] = pd.Categorical(d["subj"])
    d["cond"] = pd.Categorical(d["cond"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm("y ~ cond", d, groups="subj").fit(reml=True)
    em = (
        emmeans(fit, "cond").frame
        .sort_values("cond").reset_index(drop=True)
    )
    rr = pd.read_csv(ref_path).sort_values("cond").reset_index(drop=True)
    # Point estimate matches R lmer / aov+Error to machine precision
    # (any difference is REML-optimiser-level noise between statsmodels
    # and lme4 on the variance components).
    np.testing.assert_allclose(em["emmean"], rr["emmean"], atol=1e-6)
    np.testing.assert_allclose(em["SE"], rr["SE"], atol=1e-5)
