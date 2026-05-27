"""Validation against R emmeans reference outputs.

Loads CSVs from ``tests/r_reference/`` (produced by
``generate_r_reference.R``), refits the equivalent model in Python with
``pymmeans``, and asserts the outputs match within ``atol=1e-4``.

If the CSVs are missing, individual tests skip with instructions for
regenerating them. If a required R dataset can't be fetched (offline),
the same skip applies.

To regenerate CSVs (requires R + emmeans):
    Rscript tests/r_reference/generate_r_reference.R
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import emmeans, pairs

REF_DIR = Path(__file__).parent / "r_reference"
ATOL = 1e-4


def _require_csv(name: str) -> pd.DataFrame:
    path = REF_DIR / f"{name}.csv"
    if not path.exists():
        pytest.skip(
            f"R reference {path.name} missing. Run: "
            f"Rscript tests/r_reference/generate_r_reference.R"
        )
    return pd.read_csv(path)


def _r_dataset(dataset: str, package: str) -> pd.DataFrame:
    # Prefer a local CSV (exported by generate_r_reference.R) when present,
    # so packages that aren't on the Rdatasets mirror (emmeans::pigs) work.
    local = REF_DIR / f"{dataset}_data.csv"
    if local.exists():
        return pd.read_csv(local)
    try:
        from statsmodels.datasets import get_rdataset
        return get_rdataset(dataset, package, cache=True).data
    except Exception as e:  # network / mirror unavailable
        pytest.skip(f"Could not fetch {package}::{dataset}: {e}")


def _normalize_r_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(
        columns={
            "lower.CL": "lower_cl",
            "upper.CL": "upper_cl",
            "asymp.LCL": "lower_cl",
            "asymp.UCL": "upper_cl",
            "t.ratio": "t_ratio",
            "z.ratio": "t_ratio",
            "p.value": "p_value",
            # Response-scale column names that R uses per family.
            "prob": "emmean",
            "rate": "emmean",
            "response": "emmean",
        }
    )


def _compare_emm(expected_raw: pd.DataFrame, actual_frame: pd.DataFrame, keys: list[str]) -> None:
    expected = _normalize_r_columns(expected_raw).copy()
    actual = actual_frame.copy()
    for k in keys:
        expected[k] = expected[k].astype(str)
        actual[k] = actual[k].astype(str)
    expected = expected.set_index(keys).sort_index()
    actual = actual.set_index(keys).sort_index()
    for col in ("emmean", "SE", "lower_cl", "upper_cl"):
        if col in expected.columns:
            np.testing.assert_allclose(
                actual[col].to_numpy(),
                expected[col].to_numpy(),
                atol=ATOL,
                err_msg=f"{col} mismatch",
            )


def _compare_pairs(expected_raw: pd.DataFrame, actual_frame: pd.DataFrame, keys: list[str]) -> None:
    expected = _normalize_r_columns(expected_raw).copy()
    actual = actual_frame.copy()
    for k in keys:
        expected[k] = expected[k].astype(str)
        actual[k] = actual[k].astype(str)
    expected = expected.set_index(keys).sort_index()
    actual = actual.set_index(keys).sort_index()
    for col in ("estimate", "SE", "t_ratio"):
        np.testing.assert_allclose(
            actual[col].to_numpy(),
            expected[col].to_numpy(),
            atol=ATOL,
            err_msg=f"{col} mismatch",
        )
    if "p_value" in expected.columns:
        np.testing.assert_allclose(
            actual["p_value"].to_numpy(),
            expected["p_value"].to_numpy(),
            atol=1e-3,
            err_msg="p_value mismatch",
        )


def test_warpbreaks_emm_matches_r():
    expected = _require_csv("warp_emm_tension_by_wool")
    data = _r_dataset("warpbreaks", "datasets").copy()
    # Match R's factor level order so contrast ordering aligns
    data["tension"] = pd.Categorical(data["tension"], categories=["L", "M", "H"])
    data["wool"] = pd.Categorical(data["wool"], categories=["A", "B"])
    fit = smf.ols("breaks ~ wool * tension", data=data).fit()
    emm = emmeans(fit, "tension", by="wool")
    _compare_emm(expected, emm.frame, keys=["tension", "wool"])


def test_warpbreaks_pairs_matches_r():
    expected = _require_csv("warp_pairs_tension_by_wool")
    data = _r_dataset("warpbreaks", "datasets").copy()
    data["tension"] = pd.Categorical(data["tension"], categories=["L", "M", "H"])
    data["wool"] = pd.Categorical(data["wool"], categories=["A", "B"])
    fit = smf.ols("breaks ~ wool * tension", data=data).fit()
    pw = pairs(emmeans(fit, "tension", by="wool"))
    _compare_pairs(expected, pw.frame, keys=["contrast", "wool"])


def _pigs_data() -> pd.DataFrame:
    data = _r_dataset("pigs", "emmeans").copy()
    # R's emmeans::pigs defines `source` with levels (fish, soy, skim) in
    # definition order. Preserve that so pair ordering matches.
    data["source"] = pd.Categorical(
        data["source"], categories=["fish", "soy", "skim"]
    )
    data["percent_cat"] = pd.Categorical(data["percent"])
    return data


def test_pigs_emm_matches_r():
    expected = _require_csv("pigs_emm_source")
    data = _pigs_data()
    fit = smf.ols("np.log(conc) ~ source + percent_cat", data=data).fit()
    emm = emmeans(fit, "source")
    _compare_emm(expected, emm.frame, keys=["source"])


def test_pigs_emm_C_expression_matches_r():
    """The R-canonical 'factor(percent)' form matches via C(percent)."""
    expected = _require_csv("pigs_emm_source")
    data = _pigs_data()
    fit = smf.ols("np.log(conc) ~ source + C(percent)", data=data).fit()
    emm = emmeans(fit, "source")
    _compare_emm(expected, emm.frame, keys=["source"])


def test_pigs_pairs_matches_r():
    expected = _require_csv("pigs_pairs_source")
    data = _pigs_data()
    fit = smf.ols("np.log(conc) ~ source + percent_cat", data=data).fit()
    pw = pairs(emmeans(fit, "source"))
    _compare_pairs(expected, pw.frame, keys=["contrast"])


def test_toothgrowth_emm_matches_r():
    expected = _require_csv("tooth_emm_supp_by_dose")
    data = _r_dataset("ToothGrowth", "datasets").copy()
    data["dose_cat"] = pd.Categorical(data["dose"])
    fit = smf.ols("len ~ supp * dose_cat", data=data).fit()
    emm = emmeans(fit, "supp", by="dose_cat")
    actual = emm.frame.rename(columns={"dose_cat": "dose"})
    _compare_emm(expected, actual, keys=["supp", "dose"])


def test_insectsprays_emm_matches_r():
    expected = _require_csv("spray_emm")
    data = _r_dataset("InsectSprays", "datasets")
    fit = smf.ols("np.sqrt(count) ~ spray", data=data).fit()
    emm = emmeans(fit, "spray")
    _compare_emm(expected, emm.frame, keys=["spray"])


def test_insectsprays_pairs_matches_r():
    expected = _require_csv("spray_pairs")
    data = _r_dataset("InsectSprays", "datasets")
    fit = smf.ols("np.sqrt(count) ~ spray", data=data).fit()
    pw = pairs(emmeans(fit, "spray"))
    _compare_pairs(expected, pw.frame, keys=["contrast"])


def test_neuralgia_response_scale_matches_r():
    import statsmodels.api as sm
    expected = _require_csv("neuralgia_emm_treatment_response")
    data = _r_dataset("neuralgia", "emmeans").copy()
    # Match R's factor level order; Pain is the binary response coded Yes/No.
    data["Treatment"] = pd.Categorical(
        data["Treatment"], categories=["A", "B", "P"]
    )
    data["Sex"] = pd.Categorical(data["Sex"])
    data["Pain_num"] = (data["Pain"] == "Yes").astype(int)
    fit = smf.glm(
        "Pain_num ~ Treatment * Sex + Age",
        data=data,
        family=sm.families.Binomial(),
    ).fit()
    emm = emmeans(fit, "Treatment", type="response")
    _compare_emm(expected, emm.frame, keys=["Treatment"])
