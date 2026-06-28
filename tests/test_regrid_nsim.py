"""Simulation-based regrid — regrid(..., n_sim=) / R regrid(object, N.sim=).

Draws from the asymptotic MVN(beta_hat, V) of the coefficients and reports
sample-based intervals: converges to the Wald interval on the identity
scale, and gives the correct asymmetric interval on a nonlinear
back-transform.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy.special import expit

from pymmeans import emmeans, pairs, regrid, summary


def _ols():
    rng = np.random.default_rng(0)
    n = 200
    d = pd.DataFrame({"g": pd.Categorical(rng.choice(["A", "B", "C"], n))})
    d["y"] = d["g"].map({"A": 1.0, "B": 2.0, "C": 3.0}).astype(float) + rng.standard_normal(n)
    return emmeans(smf.ols("y ~ g", d).fit(), "g")


def _logit():
    rng = np.random.default_rng(0)
    n = 200
    d = pd.DataFrame({"g": pd.Categorical(rng.choice(["A", "B"], n))})
    p = d["g"].map({"A": 0.2, "B": 0.8}).astype(float)
    d["yb"] = (rng.random(n) < p).astype(int)
    return emmeans(smf.glm("yb ~ g", d, family=sm.families.Binomial()).fit(), "g")


def test_pass_sim_converges_to_wald_on_identity_scale():
    em = _ols()
    wald = summary(em)
    sim = regrid(em, transform="pass", n_sim=300000, random_state=1).frame
    np.testing.assert_allclose(sim["emmean"].to_numpy(), wald["emmean"].to_numpy(), atol=5e-3)
    np.testing.assert_allclose(sim["SE"].to_numpy(), wald["SE"].to_numpy(), atol=5e-3)
    np.testing.assert_allclose(sim["lower_cl"].to_numpy(), wald["lower_cl"].to_numpy(), atol=1e-2)
    np.testing.assert_allclose(sim["upper_cl"].to_numpy(), wald["upper_cl"].to_numpy(), atol=1e-2)


def test_sim_is_reproducible_with_random_state():
    em = _ols()
    a = regrid(em, transform="pass", n_sim=10000, random_state=7).frame
    b = regrid(em, transform="pass", n_sim=10000, random_state=7).frame
    np.testing.assert_allclose(a["emmean"].to_numpy(), b["emmean"].to_numpy(), atol=0)
    np.testing.assert_allclose(a["lower_cl"].to_numpy(), b["lower_cl"].to_numpy(), atol=0)


def test_response_sim_recovers_expected_probability():
    """On the response scale the sim mean is E[plogis(eta)], close to
    plogis(eta_hat) up to the (small) Jensen gap; CIs lie in (0, 1)."""
    em = _logit()
    eta = em.frame["emmean"].to_numpy()
    sim = regrid(em, transform="response", n_sim=300000, random_state=2).frame
    assert np.max(np.abs(sim["emmean"].to_numpy() - expit(eta))) < 1e-2
    assert (sim["lower_cl"] > 0).all() and (sim["upper_cl"] < 1).all()


def test_response_sim_intervals_are_asymmetric():
    em = _logit()
    sim = regrid(em, transform="response", n_sim=200000, random_state=2).frame
    up = sim["upper_cl"].to_numpy() - sim["emmean"].to_numpy()
    lo = sim["emmean"].to_numpy() - sim["lower_cl"].to_numpy()
    assert np.any(np.abs(up - lo) > 1e-3)


def test_hpd_sim_is_no_wider_than_equal_tailed():
    em = _logit()
    eti = regrid(em, transform="response", n_sim=200000, random_state=2).frame
    hpd = regrid(em, transform="response", n_sim=200000, random_state=2, hpd=True).frame
    w_eti = (eti["upper_cl"] - eti["lower_cl"]).to_numpy()
    w_hpd = (hpd["upper_cl"] - hpd["lower_cl"]).to_numpy()
    assert (w_hpd <= w_eti + 1e-9).all()


# ---------------------------------------------------------------------- refusals


def test_nsim_refuses_contrast_result():
    em = _ols()
    with pytest.raises(NotImplementedError, match="EMMResult"):
        regrid(pairs(em), n_sim=100)


def test_nsim_refuses_already_response_grid():
    responded = regrid(_logit(), transform="response")  # GLM: genuinely response-scale
    assert responded.type == "response"
    with pytest.raises(NotImplementedError, match="response scale"):
        regrid(responded, n_sim=100)


def test_nsim_refuses_named_forward_scale():
    em = _ols()
    with pytest.raises(NotImplementedError, match="transform="):
        regrid(em, transform="log", n_sim=100)


def test_nsim_rejects_too_few_samples():
    em = _ols()
    with pytest.raises(ValueError, match="n_sim"):
        regrid(em, transform="pass", n_sim=1)
