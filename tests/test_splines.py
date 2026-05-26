"""Tests for multi-column numerical basis terms.

Validates pymmeans against R `emmeans` on B-spline, polynomial, and
spline-by-factor-interaction formulas at floating-point precision.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _fit_and_emm(formula: str, target, at=None) -> pd.DataFrame:
    """Common test scaffold: load the shared synthetic dataset, fit
    an OLS via ``smf.ols``, and return the sorted EMM frame."""
    import statsmodels.formula.api as smf

    from pymmeans import emmeans

    data_csv = Path("tests/r_reference/splines_data.csv")
    if not data_csv.exists():
        pytest.skip(
            "Run tests/r_reference/splines_reference.R to generate "
            "the spline reference fixtures."
        )
    dat = pd.read_csv(data_csv)
    dat["g"] = pd.Categorical(dat["g"], categories=["a", "b", "c"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.ols(formula, dat).fit()
    res = emmeans(fit, target, at=at).frame
    sort_cols = [c for c in ("g", "x") if c in res.columns]
    if sort_cols:
        res = res.sort_values(sort_cols).reset_index(drop=True)
    return res


def _load_ref(name: str) -> pd.DataFrame:
    """Load R reference CSV, sorted to match pymmeans's row order."""
    path = Path(f"tests/r_reference/{name}")
    if not path.exists():
        pytest.skip(
            "Run tests/r_reference/splines_reference.R to generate "
            f"{name}."
        )
    ref = pd.read_csv(path)
    sort_cols = [c for c in ("g", "x") if c in ref.columns]
    if sort_cols:
        ref = ref.sort_values(sort_cols).reset_index(drop=True)
    return ref


def test_bs_main_effect_at_mean_x():
    """``bs(x, df=3) + g`` with ``emmeans(fit, "g")``
    averages over x at ``mean(x)`` (R's ``cov.reduce=mean`` default).
    Per-cell EMMs must match R at floating-point precision."""
    res = _fit_and_emm("y ~ bs(x, df=3) + g", "g")
    ref = _load_ref("splines_emm_bs_g.csv")
    np.testing.assert_allclose(
        res["emmean"].to_numpy(), ref["emmean"].to_numpy(), atol=1e-12,
        err_msg=(
            "bs(x, df=3) + g EMMs at mean(x) must match R "
            "splines::bs + emmeans at FP precision."
        ),
    )
    np.testing.assert_allclose(
        res["SE"].to_numpy(), ref["SE"].to_numpy(), atol=1e-12,
    )


def test_bs_main_effect_at_x_grid():
    """``emmeans(fit, ["g", "x"], at={"x": [2.5, 5, 7.5]})`` returns
    the 3x3 EMM grid at the spline's basis evaluated at each x."""
    res = _fit_and_emm(
        "y ~ bs(x, df=3) + g", ["g", "x"], at={"x": [2.5, 5.0, 7.5]},
    )
    ref = _load_ref("splines_emm_bs_gx.csv")
    assert len(res) == 9
    np.testing.assert_allclose(
        res["emmean"].to_numpy(), ref["emmean"].to_numpy(), atol=1e-12,
    )
    np.testing.assert_allclose(
        res["SE"].to_numpy(), ref["SE"].to_numpy(), atol=1e-12,
    )


def test_bs_factor_interaction_at_x_grid():
    """Spline-by-factor interaction (``bs(x, df=3):g``) — the bs(x)
    basis appears only in interaction terms, no main-effect term.
    Pymmeans uses patsy's ``factor.eval(state, ...)`` to recover
    the standalone basis at user x values regardless of which
    terms it appears in."""
    res = _fit_and_emm(
        "y ~ bs(x, df=3):g", ["g", "x"], at={"x": [2.5, 5.0, 7.5]},
    )
    ref = _load_ref("splines_emm_bs_interact.csv")
    assert len(res) == 9
    np.testing.assert_allclose(
        res["emmean"].to_numpy(), ref["emmean"].to_numpy(), atol=1e-12,
    )
    np.testing.assert_allclose(
        res["SE"].to_numpy(), ref["SE"].to_numpy(), atol=1e-12,
    )


def test_multi_col_factor_no_longer_refused():
    """multi-column numeric expressions
    at adapter time with a ``NotImplementedError``.
    lifts that refusal via patsy's ``factor.eval(state, ...)``
    re-evaluation. Verify the refusal is gone and ``from_fitted``
    accepts a ``bs(x, df=3)`` formula cleanly."""
    import statsmodels.formula.api as smf

    from pymmeans import emmeans
    from pymmeans.utils import from_fitted

    data_csv = Path("tests/r_reference/splines_data.csv")
    if not data_csv.exists():
        pytest.skip("splines_data.csv missing")
    dat = pd.read_csv(data_csv)
    dat["g"] = pd.Categorical(dat["g"], categories=["a", "b", "c"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.ols("y ~ bs(x, df=3) + g", dat).fit()
    info = from_fitted(fit)
    # The adapter now populates multi_col_factors with the
    # underlying column for the basis expression.
    assert info.multi_col_factors == {"bs(x, df=3)": ["x"]}
    assert "x" in info.numeric_means
    # And emmeans no longer raises NotImplementedError.
    em = emmeans(info, "g")
    assert len(em.frame) == 3


def test_ref_grid_supports_multi_col_basis():
    """previously, ``ref_grid(fit_with_bs)`` raised
    ``NotImplementedError`` with a misleading "v0.1 supports only
    identifier factors" message even though the analytic path
    already handled it. Now the eager ``ref_grid`` builder
    accepts registered multi-col factors and produces a usable
    grid + L matrix."""
    import statsmodels.formula.api as smf

    from pymmeans import ref_grid

    dat = pd.read_csv("tests/r_reference/splines_data.csv")
    dat["g"] = pd.Categorical(dat["g"], categories=["a", "b", "c"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.ols("y ~ bs(x, df=3) + g", dat).fit()
    rg = ref_grid(fit)
    # 3 g × 1 x_mean → 3 rows; design has 6 columns (Intercept + 2 g
    # dummies + 3 bs basis).
    assert rg.grid.shape == (3, 2)
    assert rg.linfct.shape == (3, 6)


def test_chunk_size_streaming_supports_multi_col_basis():
    """previously, the streaming
    ``emmeans(..., chunk_size=N)`` path raised ``NotImplementedError``
    on multi-col basis terms because ``_validate_design`` rejected
    them. Now the streaming path produces identical results
    to the analytic path."""
    import statsmodels.formula.api as smf

    from pymmeans import emmeans

    dat = pd.read_csv("tests/r_reference/splines_data.csv")
    dat["g"] = pd.Categorical(dat["g"], categories=["a", "b", "c"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.ols("y ~ bs(x, df=3) + g", dat).fit()
    em_analytic = emmeans(fit, "g")
    em_stream = emmeans(fit, "g", chunk_size=8)
    np.testing.assert_allclose(
        em_stream.frame["emmean"].to_numpy(),
        em_analytic.frame["emmean"].to_numpy(),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        em_stream.frame["SE"].to_numpy(),
        em_analytic.frame["SE"].to_numpy(),
        atol=1e-12,
    )


def test_pickle_round_trip_preserves_multi_col_factors():
    """pickle round-trip of a ``ModelInfo``
    with multi-col factors must preserve ``multi_col_factors``
    (otherwise post-pickle ``apply_satterthwaite`` / contrast
    re-computation paths would silently lose the basis routing)."""
    import pickle

    import statsmodels.formula.api as smf

    from pymmeans.utils import from_fitted

    dat = pd.read_csv("tests/r_reference/splines_data.csv")
    dat["g"] = pd.Categorical(dat["g"], categories=["a", "b", "c"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.ols("y ~ bs(x, df=3) + g", dat).fit()
    info = from_fitted(fit)
    info2 = pickle.loads(pickle.dumps(info))
    assert info2.multi_col_factors == info.multi_col_factors
    assert info2.multi_col_factors == {"bs(x, df=3)": ["x"]}


def test_canonical_basis_name_as_target_helpful_error():
    """previously, passing the canonical basis
    expression as target (``emmeans(fit, "bs(x, df=3)")``) raised
    a generic "not a factor or numeric covariate" error that didn't
    suggest the underlying column name. Now the error points
    at ``x`` directly."""
    import statsmodels.formula.api as smf

    from pymmeans import emmeans

    dat = pd.read_csv("tests/r_reference/splines_data.csv")
    dat["g"] = pd.Categorical(dat["g"], categories=["a", "b", "c"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.ols("y ~ bs(x, df=3) + g", dat).fit()
    with pytest.raises(ValueError, match="underlying covariate.*'x'"):
        emmeans(fit, "bs(x, df=3)")


def test_underlying_columns_excludes_basis_function_name():
    """previously, the AST walker in
    ``_underlying_columns`` picked up function-call head identifiers
    (the ``bs`` in ``bs(x, df=3)``) when they coincided with data
    column names. Now only argument names contribute to the
    underlying-column set."""
    from pymmeans.utils import _underlying_columns

    # No collision: returns just the argument
    assert _underlying_columns("bs(x, df=3)", {"x"}) == ["x"]
    # Collision: the ``bs`` data column should NOT be picked up
    assert _underlying_columns("bs(x, df=3)", {"bs", "x"}) == ["x"]
    # Attribute call: ``np.log(x)`` should not pick up ``np``
    assert _underlying_columns("np.log(x)", {"np", "x"}) == ["x"]
    # Plain identifier (not a call): still works
    assert _underlying_columns("percent", {"percent"}) == ["percent"]
    # Multi-arg basis: ``te(x, y)`` returns both
    assert _underlying_columns("te(x, y)", {"te", "x", "y"}) == ["x", "y"]


def test_at_x_uses_underlying_column_name():
    """The user passes ``at={"x": ...}`` where ``x`` is the
    underlying column of a multi-col basis (``bs(x, df=3)``).
    Pre-the ``ref_grid`` ambiguity check raised
    ``NotImplementedError`` because ``x → bs(x, df=3)`` looked
    like a transform alias; routes multi-col aliases to
    a no-op consistency check (``x`` is the only valid name)."""
    import statsmodels.formula.api as smf

    from pymmeans import emmeans

    data_csv = Path("tests/r_reference/splines_data.csv")
    if not data_csv.exists():
        pytest.skip("splines_data.csv missing")
    dat = pd.read_csv(data_csv)
    dat["g"] = pd.Categorical(dat["g"], categories=["a", "b", "c"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.ols("y ~ bs(x, df=3) + g", dat).fit()
    # Should NOT raise NotImplementedError on the ambiguity check.
    em = emmeans(fit, ["g", "x"], at={"x": [2.5, 7.5]})
    assert len(em.frame) == 6
