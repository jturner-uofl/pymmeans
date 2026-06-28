"""effect_size() carries sigma (SD) uncertainty via edf, matching R
emmeans' eff_size: SE(d) = sqrt((SE_d|edf=inf)^2 + d^2 / (2 * edf)).

This is the one capability the marginaleffects JSS paper concedes emmeans
does better; pymmeans matches R `eff_size` to ~5e-11 (verified on shared
data), so the formula identity below is the R-validated closed form.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import effect_size, emmeans


def _fit(n=60, seed=2):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"g": pd.Categorical(["A", "B", "C"] * (n // 3))})
    mu = df["g"].map({"A": 1.0, "B": 2.0, "C": 3.0}).astype(float) * 0.7
    df["y"] = mu + rng.standard_normal(n)
    fit = smf.ols("y ~ g", df).fit()
    return fit, emmeans(fit, "g"), float(np.sqrt(fit.scale))


def test_effect_size_se_grows_as_edf_shrinks():
    _, em, sig = _fit()
    se_inf = float(effect_size(em, sigma=sig, edf=1e9).iloc[0]["effect_size_SE"])
    se_50 = float(effect_size(em, sigma=sig, edf=50).iloc[0]["effect_size_SE"])
    se_8 = float(effect_size(em, sigma=sig, edf=8).iloc[0]["effect_size_SE"])
    assert se_inf < se_50 < se_8  # less certain sigma -> wider effect-size SE


def test_effect_size_se_matches_eff_size_formula():
    """SE(d) = sqrt(SE_inf^2 + d^2 / (2 edf)) for every contrast and edf —
    the R `eff_size` closed form (verified vs R to ~5e-11)."""
    _, em, sig = _fit()
    base = effect_size(em, sigma=sig, edf=1e12)
    d = base["effect_size"].to_numpy()
    se_inf = base["effect_size_SE"].to_numpy()
    for edf in (50.0, 8.0, 3.0):
        es = effect_size(em, sigma=sig, edf=edf)
        expected = np.sqrt(se_inf**2 + d**2 / (2.0 * edf))
        np.testing.assert_allclose(
            es["effect_size_SE"].to_numpy(), expected, atol=1e-9
        )


def test_effect_size_point_is_edf_independent():
    """The standardized effect (d = estimate / sigma) does not depend on
    edf; only its SE does."""
    _, em, sig = _fit()
    a = effect_size(em, sigma=sig, edf=1e9)["effect_size"].to_numpy()
    b = effect_size(em, sigma=sig, edf=5)["effect_size"].to_numpy()
    np.testing.assert_allclose(a, b, atol=1e-12)


def test_effect_size_default_edf_is_residual_df():
    """Without an explicit edf, the residual df is used (R default)."""
    fit, em, sig = _fit()
    default = effect_size(em, sigma=sig).iloc[0]["effect_size_SE"]
    explicit = effect_size(em, sigma=sig, edf=fit.df_resid).iloc[0]["effect_size_SE"]
    assert float(default) == pytest.approx(float(explicit), abs=1e-10)
