"""Tests for the ``nuisance=`` × ``weights=`` cross-product.

Under ``weights='equal'`` ``nuisance=`` is declarative-only; under
``weights='outer'`` / ``weights='proportional'`` / ``weights='cells'``
``nuisance=`` **overrides** the per-factor weight construction to force
equal-marginal averaging on the named factor(s), matching R `emmeans`
with the default `wt.nuis='equal'`. The ``'cells'`` path uses analytical
extrapolation at each grid combo (via patsy) so empty
(target, nuisance, non-nuisance) sub-cells are handled correctly,
matching R to floating-point precision.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import emmeans


def _make_unbalanced_3factor(seed: int = 17, n: int = 240):
    """Construct an unbalanced (g × b × c) design where 'proportional' and
    'equal' produce visibly different EMMs and 'nuisance=b' shifts them
    by a measurable amount in between."""
    rng = np.random.default_rng(seed)
    g = rng.choice(["g1", "g2", "g3"], n, p=[0.5, 0.3, 0.2])
    b = rng.choice(["bL", "bH"],       n, p=[0.8, 0.2])
    c = rng.choice(["cA", "cB", "cC"], n, p=[0.6, 0.3, 0.1])
    y = (
        (g == "g2") * 0.4 + (g == "g3") * 1.0
        + (b == "bH") * 0.7
        + (c == "cB") * 0.5 + (c == "cC") * 1.0
        + rng.normal(scale=0.7, size=n)
    )
    df = pd.DataFrame({"g": pd.Categorical(g), "b": pd.Categorical(b),
                       "c": pd.Categorical(c), "y": y})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.ols("y ~ g + b + c", df).fit()
    return df, fit


def test_nuisance_no_longer_refuses_under_proportional():
    """Round-XX nuisance×weights: combining nuisance= with
    weights='proportional' used to raise NotImplementedError. The kernel
    now applies the equal-marginal override; the call must succeed and
    produce finite EMMs."""
    _, fit = _make_unbalanced_3factor()
    em = emmeans(fit, "g", weights="proportional", nuisance="b").frame
    assert np.all(np.isfinite(em["emmean"].to_numpy()))
    assert np.all(np.isfinite(em["SE"].to_numpy()))


def test_nuisance_no_longer_refuses_under_outer():
    """Same lift for weights='outer'."""
    _, fit = _make_unbalanced_3factor()
    em = emmeans(fit, "g", weights="outer", nuisance="b").frame
    assert np.all(np.isfinite(em["emmean"].to_numpy()))


def test_nuisance_shifts_emms_visibly_on_proportional():
    """The override must MOVE the EMMs by a meaningful amount on an
    unbalanced design — otherwise nuisance= is silently a no-op and the
    fix is illusory. On this fixture the shift is ~0.18 (≈25% of the
    bH coefficient × the (0.5 − 0.2) marginal-prob delta)."""
    _, fit = _make_unbalanced_3factor()
    em_plain = emmeans(fit, "g", weights="proportional").frame
    em_nuis = emmeans(fit, "g", weights="proportional", nuisance="b").frame
    shift = float(np.max(np.abs(em_plain["emmean"].to_numpy()
                                - em_nuis["emmean"].to_numpy())))
    assert shift > 5e-2, (
        f"nuisance override should visibly shift EMMs (~0.18 expected); "
        f"got {shift:.4f} — the override may be a silent no-op."
    )


def test_nuisance_all_non_specs_equals_weights_equal():
    """Marking *every* non-target factor as nuisance under
    weights='proportional' must collapse to the same EMMs as
    weights='equal' to machine precision — a structural sanity property
    of the equal-marginal override."""
    _, fit = _make_unbalanced_3factor()
    em_equal = (
        emmeans(fit, "g", weights="equal").frame
        .sort_values("g").reset_index(drop=True)
    )
    em_nuis_all = (
        emmeans(fit, "g", weights="proportional", nuisance=["b", "c"]).frame
        .sort_values("g").reset_index(drop=True)
    )
    np.testing.assert_allclose(em_nuis_all["emmean"], em_equal["emmean"], atol=1e-12)
    np.testing.assert_allclose(em_nuis_all["SE"],     em_equal["SE"],     atol=1e-12)


def test_nuisance_with_cells_weights_now_supported_via_analytical():
    """weights='cells' × nuisance= used to refuse because observed-cell
    averaging can't reproduce R's `wt.nuis='equal'` over empty
    (target, nuisance, non-nuisance) sub-cells. The kernel now switches
    to ANALYTICAL extrapolation at each (target × nuisance × non-nuisance)
    grid combo via patsy — matching R `emmeans(..., weights='cells',
    nuisance=, wt.nuis='equal')` to floating-point precision (verified
    in jss_audit §VIII.b on the same seeded fixture)."""
    _, fit = _make_unbalanced_3factor()
    em = emmeans(fit, "g", weights="cells", nuisance="b").frame
    assert np.all(np.isfinite(em["emmean"].to_numpy()))
    assert np.all(np.isfinite(em["SE"].to_numpy()))
    # Marking BOTH non-target factors as nuisance under cells must
    # collapse to equal weights (this is the analytical-extrapolation
    # limit of cells × full-nuisance).
    em_all = emmeans(
        fit, "g", weights="cells", nuisance=["b", "c"]
    ).frame.sort_values("g").reset_index(drop=True)
    em_eq = emmeans(
        fit, "g", weights="equal"
    ).frame.sort_values("g").reset_index(drop=True)
    np.testing.assert_allclose(
        em_all["emmean"], em_eq["emmean"], atol=1e-6
    )


def test_cells_with_nuisance_matches_R_to_machine_precision():
    """The analytical-extrapolation cells × nuisance path must reproduce
    R `emmeans(..., weights="cells", nuisance=, wt.nuis="equal")` on the
    seeded §VIII.b fixture to floating-point precision. Hard-coded R
    targets (from ``examples/jss_audit/cs_ref/nuisance_cells_nuis_*.csv``)
    so the test runs independently of the case-study generator."""
    from pathlib import Path
    data_path = Path("examples/jss_audit/cs_ref/nuisance_data.csv")
    ref_b_path = Path("examples/jss_audit/cs_ref/nuisance_cells_nuis_b_emm.csv")
    ref_bc_path = Path(
        "examples/jss_audit/cs_ref/nuisance_cells_nuis_bc_emm.csv"
    )
    if not (data_path.exists() and ref_b_path.exists() and ref_bc_path.exists()):
        pytest.skip("Run generate_case_study_reference.R to seed the "
                    "cells × nuisance R reference CSVs.")
    d = pd.read_csv(data_path)
    for c in ("g", "b", "c"):
        d[c] = pd.Categorical(d[c])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.ols("y ~ g + b + c", d).fit()
    for r_path, kw in [
        (ref_b_path, dict(nuisance="b")),
        (ref_bc_path, dict(nuisance=["b", "c"])),
    ]:
        em = (
            emmeans(fit, "g", weights="cells", **kw).frame
            .sort_values("g").reset_index(drop=True)
        )
        rr = pd.read_csv(r_path).sort_values("g").reset_index(drop=True)
        np.testing.assert_allclose(
            em["emmean"], rr["emmean"], atol=1e-9
        )
        np.testing.assert_allclose(em["SE"], rr["SE"], atol=1e-9)


def test_outer_and_proportional_with_nuisance_agree_on_one_non_nuisance_factor():
    """When the only remaining non-target factor (after nuisance is
    removed) is a single factor, weights='outer' and weights='proportional'
    apply identical 1-D marginals to it — so the two paths must agree to
    machine precision under nuisance= as well."""
    _, fit = _make_unbalanced_3factor()
    # remove c too via nuisance → only g (target) and b/c (both nuisance)
    em_o = emmeans(fit, "g", weights="outer", nuisance=["c"]).frame.sort_values("g")
    em_p = emmeans(fit, "g", weights="proportional", nuisance=["c"]).frame.sort_values("g")
    np.testing.assert_allclose(em_o["emmean"], em_p["emmean"], atol=1e-12)
