"""Tests for response-scale back-transformations (regrid)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import (
    Transform,
    detect_transform,
    emmeans,
    pairs,
    regrid_response,
)


def test_detect_log():
    assert detect_transform("np.log(conc)").name == "log"
    assert detect_transform("log(conc)").name == "log"
    assert detect_transform("np.log10(y)").name == "log10"
    assert detect_transform("np.log1p(y)").name == "log1p"
    assert detect_transform("np.sqrt(count)").name == "sqrt"
    assert detect_transform("y") is None
    assert detect_transform("") is None


def test_regrid_response_log():
    rng = np.random.default_rng(0)
    n = 60
    df = pd.DataFrame(
        {
            "y": np.exp(rng.normal(loc=1.0, scale=0.3, size=n)),
            "g": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        }
    )
    fit = smf.ols("np.log(y) ~ g", data=df).fit()
    emm_link = emmeans(fit, "g")
    emm_resp = regrid_response(emm_link)
    # All response-scale means should be positive (exp of finite values)
    assert (emm_resp.frame["emmean"] > 0).all()
    # Back-transformed means = exp(link means)
    np.testing.assert_array_almost_equal(
        emm_resp.frame["emmean"].to_numpy(),
        np.exp(emm_link.frame["emmean"].to_numpy()),
    )


def test_regrid_response_log_bias_adjusted():
    rng = np.random.default_rng(1)
    n = 100
    y = np.exp(rng.normal(loc=2.0, scale=0.4, size=n))
    df = pd.DataFrame({"y": y, "g": pd.Categorical(["a"] * n)})
    fit = smf.ols("np.log(y) ~ g", data=df).fit()
    emm_link = emmeans(fit, "g")
    emm_resp = regrid_response(emm_link, bias_adjust=False)
    emm_resp_bias = regrid_response(emm_link, bias_adjust=True)
    # Bias-adjusted exp(mu + sigma^2/2) > exp(mu)
    assert emm_resp_bias.frame["emmean"].iloc[0] > emm_resp.frame["emmean"].iloc[0]


def test_regrid_response_sqrt():
    rng = np.random.default_rng(2)
    n = 80
    counts = rng.poisson(lam=4, size=n)
    df = pd.DataFrame(
        {"count": counts, "g": pd.Categorical(rng.choice(["a", "b"], n))}
    )
    fit = smf.ols("np.sqrt(count) ~ g", data=df).fit()
    emm_link = emmeans(fit, "g")
    emm_resp = regrid_response(emm_link)
    # Squared means should be in count-scale (non-negative)
    np.testing.assert_array_almost_equal(
        emm_resp.frame["emmean"].to_numpy(),
        emm_link.frame["emmean"].to_numpy() ** 2,
    )


def test_regrid_response_custom_transform():
    rng = np.random.default_rng(3)
    n = 50
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "g": pd.Categorical(rng.choice(["a", "b"], n)),
        }
    )
    fit = smf.ols("y ~ g", data=df).fit()
    emm_link = emmeans(fit, "g")
    # Custom "identity" transform should be a no-op
    identity = Transform("identity", lambda x: x, lambda x: np.ones_like(x), False)
    emm_resp = regrid_response(emm_link, tran=identity)
    np.testing.assert_array_almost_equal(
        emm_resp.frame["emmean"].to_numpy(),
        emm_link.frame["emmean"].to_numpy(),
    )


def test_regrid_response_lower_le_upper_after_inverse():
    """For log, exp is monotone increasing so order is preserved; for
    transforms that aren't monotone increasing (e.g. ``np.sqrt`` operates
    on positive predictions, fine), we swap to keep lower <= upper."""
    rng = np.random.default_rng(4)
    n = 60
    df = pd.DataFrame(
        {
            "y": np.exp(rng.normal(loc=2, scale=0.5, size=n)),
            "g": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        }
    )
    fit = smf.ols("np.log(y) ~ g", data=df).fit()
    emm_resp = regrid_response(emmeans(fit, "g"))
    assert (emm_resp.frame["lower_cl"] <= emm_resp.frame["upper_cl"]).all()


def test_regrid_response_on_pairs():
    """Log-family contrasts back-transform to ratios; labels rename
    'a - b' -> 'a / b' and the column is renamed estimate -> ratio."""
    rng = np.random.default_rng(5)
    n = 80
    df = pd.DataFrame(
        {
            "y": np.exp(rng.normal(size=n)),
            "g": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        }
    )
    fit = smf.ols("np.log(y) ~ g", data=df).fit()
    pw = pairs(emmeans(fit, "g"))
    pw_resp = regrid_response(pw)
    expected_ratios = np.exp(pw.frame["estimate"].to_numpy())
    np.testing.assert_array_almost_equal(
        pw_resp.frame["ratio"].to_numpy(), expected_ratios
    )
    # Contrast labels should be renamed
    assert all(" / " in c for c in pw_resp.frame["contrast"])
    assert "estimate" not in pw_resp.frame.columns


def test_regrid_response_bias_adjust_log10_taylor_matches_r():
    """log10 bias adjust uses R's second-order Taylor formula:
    ``10^mu * (1 + (ln 10)^2 * sigma^2 / 2)`` — not the exact lognormal
    ``10^(mu + sigma^2/2)``. #5 switched to R parity."""
    rng = np.random.default_rng(6)
    mu_true, sigma_true = 1.0, 0.5
    n = 1000
    y = 10 ** (rng.normal(loc=mu_true, scale=sigma_true, size=n))
    df = pd.DataFrame({"y": y, "g": pd.Categorical(["a"] * n)})
    fit = smf.ols("np.log10(y) ~ g", data=df).fit()
    emm_link = emmeans(fit, "g")
    emm_resp = regrid_response(emm_link, bias_adjust=True)
    mu_hat = float(emm_link.frame["emmean"].iloc[0])
    sigma2_hat = fit.scale
    expected = (10**mu_hat) * (1.0 + (np.log(10) ** 2) * sigma2_hat / 2.0)
    np.testing.assert_array_almost_equal(
        emm_resp.frame["emmean"].to_numpy(), [expected], decimal=6
    )


def test_regrid_response_bias_adjust_log1p_taylor_matches_r():
    """log1p bias adjust uses R's Taylor formula:
    ``expm1(mu) + exp(mu) * sigma^2 / 2``."""
    rng = np.random.default_rng(7)
    y = np.expm1(rng.normal(loc=2.0, scale=0.4, size=200))
    df = pd.DataFrame({"y": y, "g": pd.Categorical(["a"] * 200)})
    fit = smf.ols("np.log1p(y) ~ g", data=df).fit()
    emm_resp = regrid_response(emmeans(fit, "g"), bias_adjust=True)
    mu_hat = float(emmeans(fit, "g").frame["emmean"].iloc[0])
    sigma2_hat = fit.scale
    expected = np.expm1(mu_hat) + np.exp(mu_hat) * sigma2_hat / 2.0
    np.testing.assert_allclose(
        emm_resp.frame["emmean"].to_numpy(), [expected], atol=1e-6
    )


def test_regrid_response_bias_adjust_undefined_raises():
    """The exp() inverse transform doesn't have a closed-form bias mean."""
    rng = np.random.default_rng(8)
    n = 60
    df = pd.DataFrame(
        {
            "y": np.log(rng.uniform(0.5, 5.0, n)),
            "g": pd.Categorical(rng.choice(["a", "b"], n)),
        }
    )
    fit = smf.ols("np.exp(y) ~ g", data=df).fit()
    with pytest.raises(ValueError, match="bias_adjust"):
        regrid_response(emmeans(fit, "g"), bias_adjust=True)
