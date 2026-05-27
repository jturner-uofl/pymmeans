"""Tests for the health_check diagnostic bundle (leapfrog feature)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import (
    Check,
    HealthReport,
    contrast,
    emmeans,
    health_check,
)


def _ols_fit(rng: np.random.Generator, n: int = 200):
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "g": pd.Categorical(rng.choice(["a", "b", "c"], n)),
            "x": rng.normal(size=n),
        }
    )
    return smf.ols("y ~ g + x", data=df).fit()


def test_health_check_clean_ols_reports_all_ok():
    rng = np.random.default_rng(0)
    fit = _ols_fit(rng)
    emm = emmeans(fit, "g")
    rep = health_check(emm)
    assert rep.ok
    assert not rep.critical
    names = [c.name for c in rep]
    for required in [
        "estimability",
        "conditioning",
        "rank",
        "df_sanity",
        "cell_counts",
        "influence",
    ]:
        assert required in names, f"missing check: {required}"


def test_health_check_near_collinear_flags_conditioning():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "g": pd.Categorical(rng.choice(["a", "b"], n)),
            "x1": x,
            "x2": x + 1e-8 * rng.normal(size=n),
        }
    )
    fit = smf.ols("y ~ g + x1 + x2", data=df).fit()
    rep = health_check(emmeans(fit, "g"))
    cond_check = next(c for c in rep if c.name == "conditioning")
    assert cond_check.severity == "critical"
    assert "condition number" in cond_check.message.lower()


def test_health_check_zero_count_cell_flags_three_problems():
    """A factor level present in the dtype but absent from observed data
    triggers estimability, rank, and cell_counts critical findings."""
    rng = np.random.default_rng(2)
    n = 200
    g_obs = rng.choice(["a", "b"], n)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "g": pd.Categorical(g_obs, categories=["a", "b", "c"]),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit = smf.ols("y ~ g", data=df).fit()
        rep = health_check(emmeans(fit, "g"))
    critical_names = {c.name for c in rep.critical}
    assert {"estimability", "rank", "cell_counts"}.issubset(critical_names)


def test_health_check_works_on_contrast_result():
    rng = np.random.default_rng(3)
    fit = _ols_fit(rng)
    cr = contrast(emmeans(fit, "g"), method="pairwise")
    rep = health_check(cr)
    # cell_counts and influence may not apply to contrasts but estimability /
    # conditioning / rank should still fire.
    assert any(c.name == "estimability" for c in rep)
    assert any(c.name == "conditioning" for c in rep)


def test_health_report_repr_contains_all_severities():
    rep = HealthReport()
    rep.add("a", "ok", "fine")
    rep.add("b", "warning", "borderline")
    rep.add("c", "critical", "bad")
    s = repr(rep)
    assert "1 ok" in s and "1 warning" in s and "1 critical" in s
    assert "[ok]" in s and "[warn]" in s and "[CRIT]" in s


def test_health_report_to_frame_round_trips():
    rep = HealthReport()
    rep.add("x", "ok", "msg", value=42)
    df = rep.to_frame()
    assert list(df.columns) == ["name", "severity", "message", "value"]
    assert df.iloc[0]["value"] == 42


def test_check_dataclass_is_iterable_from_report():
    rep = HealthReport()
    rep.add("x", "ok", "msg")
    for c in rep:
        assert isinstance(c, Check)


def test_health_check_handles_pickled_emm_result():
    """After pickle, ModelInfo.raw_result is None — the rank check must
    fall back to the estimability_basis path and not crash."""
    import pickle as _pickle

    rng = np.random.default_rng(4)
    fit = _ols_fit(rng)
    emm = emmeans(fit, "g")
    emm2 = _pickle.loads(_pickle.dumps(emm))
    # No exception.
    rep = health_check(emm2)
    # estimability still works (uses NaN in frame).
    assert any(c.name == "estimability" for c in rep)


def test_health_check_mixedlm_boundary_fit():
    """When MixedLM hits a sigma_u^2 ~ 0 boundary, boundary_fit is critical."""
    rng = np.random.default_rng(23)
    n_groups = 10
    n_per = 6
    n = n_groups * n_per
    subj = np.repeat(np.arange(n_groups), n_per)
    y = rng.normal(size=n)  # no true RE effect
    df = pd.DataFrame(
        {
            "y": y,
            "subj": subj,
            "g": pd.Categorical(rng.choice(["a", "b", "c"], n)),
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm("y ~ g", data=df, groups="subj").fit()
    if float(fit.cov_re.iloc[0, 0]) > 1e-6:
        pytest.skip("This seed didn't hit the boundary on this scipy version")
    rep = health_check(emmeans(fit, "g"))
    boundary = next(c for c in rep if c.name == "boundary_fit")
    assert boundary.severity == "critical"
