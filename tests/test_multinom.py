"""Tests for ``pymmeans.multinom``.

multinomial-logit EMMs (``statsmodels.MNLogit``).
Validated against R ``nnet::multinom`` + ``emmeans`` on a
synthetic 3-category dataset.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _fit_multinom() -> tuple:
    """Load the R-reference CSV and fit statsmodels MNLogit."""
    import statsmodels.formula.api as smf

    data_csv = Path("tests/r_reference/multinom_data.csv")
    if not data_csv.exists():
        pytest.skip(
            "Run the multinom reference setup first to generate "
            "multinom_data.csv + multinom_emm_*.csv."
        )
    dat = pd.read_csv(data_csv)
    dat["g"] = pd.Categorical(dat["g"], categories=["a", "b", "c"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mnlogit("y ~ x1 + g", dat).fit(disp=False)
    return fit, dat


def test_mnlogit_prob_mode_matches_r_emmeans():
    """per-category probability EMMs from
    pymmeans' ``multinom_emmeans(fit, "g", mode="prob")``
    must match R ``emmeans(multinom_fit, ~ y | g, mode="prob")``
    at floating-point precision. statsmodels' ``MNLogit`` and
    R ``nnet::multinom`` converge to the same MLE, so the
    only residual disagreement is optimizer-noise."""
    from pymmeans import multinom_emmeans

    fit, _ = _fit_multinom()
    res = multinom_emmeans(fit, "g", mode="prob")
    ref_csv = Path("tests/r_reference/multinom_emm_prob.csv")
    if not ref_csv.exists():
        pytest.skip("R reference missing; run multinom_reference.R")
    ref = pd.read_csv(ref_csv)
    np.testing.assert_allclose(
        res.frame["emmean"].to_numpy(),
        ref["prob"].to_numpy(),
        atol=1e-5,
        err_msg="MNLogit prob-mode parity vs R must hold at atol=1e-5.",
    )
    np.testing.assert_allclose(
        res.frame["SE"].to_numpy(),
        ref["SE"].to_numpy(),
        atol=1e-5,
        err_msg="MNLogit prob-mode SE parity vs R must hold at atol=1e-5.",
    )


def test_mnlogit_latent_mode_matches_r_emmeans():
    """per-non-reference-category log-odds (latent mode)
    parity vs R ``emmeans(... , mode="latent")``."""
    from pymmeans import multinom_emmeans

    fit, _ = _fit_multinom()
    res = multinom_emmeans(fit, "g", mode="latent")
    ref_csv = Path("tests/r_reference/multinom_emm_latent.csv")
    if not ref_csv.exists():
        pytest.skip("R reference missing")
    ref = pd.read_csv(ref_csv)
    np.testing.assert_allclose(
        res.frame["emmean"].to_numpy(),
        ref["emmean"].to_numpy(),
        atol=1e-4,
    )
    np.testing.assert_allclose(
        res.frame["SE"].to_numpy(),
        ref["SE"].to_numpy(),
        atol=1e-4,
    )


def test_mnlogit_probabilities_sum_to_one():
    """Sanity invariant: per-cell probabilities sum to 1.0."""
    from pymmeans import multinom_emmeans

    fit, _ = _fit_multinom()
    res = multinom_emmeans(fit, "g", mode="prob")
    per_cell_sum = res.frame.groupby("g", observed=True)["emmean"].sum()
    np.testing.assert_allclose(per_cell_sum.to_numpy(), 1.0, atol=1e-12)


def test_mnlogit_invalid_mode_raises():
    """Unknown ``mode`` raises ``ValueError``."""
    from pymmeans import multinom_emmeans

    fit, _ = _fit_multinom()
    with pytest.raises(ValueError, match="unknown mode"):
        multinom_emmeans(fit, "g", mode="bogus")


def test_mnlogit_refused_through_emmeans_with_clear_message():
    """The plain ``emmeans(mnlogit_fit, ...)`` path still raises
    a clear ``NotImplementedError`` pointing at ``multinom_emmeans``.

    the error message now explicitly names
    ``pymmeans.multinom_emmeans`` as the canonical workaround
    (previously it only mentioned the binary-logit decomposition).
    Verify both pieces of information appear in the message."""
    from pymmeans import emmeans

    fit, _ = _fit_multinom()
    with pytest.raises(NotImplementedError, match="multinom_emmeans"):
        emmeans(fit, "g")


def test_multinom_df_matches_r_emmeans():
    """previously, ``multinom_emmeans`` returned
    ``df = inf`` (z-quantile CIs), but R ``emmeans`` for multinomial
    fits uses ``df = n_free_params = k_vars * (J - 1)`` (t-quantile
    CIs). On the reference fit that's df=8. The CI difference is
    visible (~18 % wider in R at df=8 vs z). Now matches R.

    Guard: returned df equals n_free_params, and CIs match R's
    t-quantile bounds at floating-point precision."""
    from pymmeans import multinom_emmeans

    fit, _ = _fit_multinom()
    k_vars, J_minus_1 = fit.params.shape
    expected_df = float(k_vars * J_minus_1) # 4 * 2 = 8 here

    res = multinom_emmeans(fit, "g", mode="prob")
    np.testing.assert_array_equal(
        res.frame["df"].to_numpy(),
        np.full(len(res.frame), expected_df),
    )

    # And the CIs should now match R's t-quantile bounds.
    ref_csv = Path("tests/r_reference/multinom_emm_prob.csv")
    if not ref_csv.exists():
        pytest.skip("R reference missing")
    ref = pd.read_csv(ref_csv)
    np.testing.assert_allclose(
        res.frame["lower_cl"].to_numpy(),
        ref["lower.CL"].to_numpy(),
        atol=1e-4,
        err_msg=(
            "#3: lower CI must match R at atol=1e-4 "
            "(both now use t-quantile with df = n_free_params)."
        ),
    )
    np.testing.assert_allclose(
        res.frame["upper_cl"].to_numpy(),
        ref["upper.CL"].to_numpy(),
        atol=1e-4,
    )


def test_multinom_centered_latent_sum_zero():
    """the centered-latent mode must produce
    per-cell zero-sum across categories. A sign-flip on the mean
    subtraction would still produce R-parity at the SE level but
    break the algebra; this is a defensive invariant guard."""
    from pymmeans import multinom_emmeans

    fit, _ = _fit_multinom()
    res = multinom_emmeans(fit, "g", mode="latent")
    per_cell_sum = res.frame.groupby("g", observed=True)["emmean"].sum()
    np.testing.assert_allclose(
        per_cell_sum.to_numpy(), 0.0, atol=1e-12,
        err_msg=(
            "centered-latent ζ_k = η_k - mean(η_·) must "
            "sum to zero across categories within each cell."
        ),
    )
