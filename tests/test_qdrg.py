"""Tests for ``pymmeans.qdrg``.

``qdrg`` / ``emmobj`` low-level constructors for
building reference grids from raw ``β̂`` + ``V̂`` + formula
inputs (no fitted model required).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_qdrg_round_trip_matches_hand_calculation():
    """``qdrg`` builds a ModelInfo from raw β + V +
    formula; calling ``emmeans`` on it must produce the
    hand-calculated EMM for each factor level."""
    from pymmeans import emmeans, qdrg

    df = pd.DataFrame({"y": [0.0] * 9, "g": pd.Categorical(["a", "b", "c"] * 3)})
    info = qdrg(
        "g", df, coef=np.array([2.0, 0.5, -0.3]),
        vcov=np.diag([0.04, 0.04, 0.04]), df=50,
    )
    em = emmeans(info, "g")
    # Patsy "g" formula gives [Intercept, g[T.b], g[T.c]] →
    # EMM at g=a is just the Intercept (2.0), at g=b is
    # Intercept+g[T.b] = 2.5, at g=c is Intercept+g[T.c] = 1.7.
    expected = {"a": 2.0, "b": 2.5, "c": 1.7}
    actual = dict(zip(em.frame["g"], em.frame["emmean"], strict=True))
    for k, v in expected.items():
        np.testing.assert_allclose(actual[k], v, atol=1e-10)


def test_qdrg_accepts_full_formula_and_rhs_only():
    """Both ``"y ~ g"`` and ``"g"`` (RHS-only) must work."""
    from pymmeans import emmeans, qdrg

    df = pd.DataFrame({"y": [0.0] * 6, "g": pd.Categorical(["a", "b"] * 3)})
    for formula in ("y ~ g", "g"):
        info = qdrg(
            formula, df, coef=np.array([1.0, 0.3]),
            vcov=np.diag([0.01, 0.01]), df=10,
        )
        em = emmeans(info, "g")
        assert len(em.frame) == 2


def test_qdrg_validates_shape_mismatch():
    """Raise ``ValueError`` if coef vs vcov shapes disagree, or
    if coef length doesn't match the formula's design columns."""
    from pymmeans import qdrg

    df = pd.DataFrame({"y": [0.0] * 4, "g": pd.Categorical(["a", "b"] * 2)})
    # Wrong vcov shape
    with pytest.raises(ValueError, match="square matrix"):
        qdrg("g", df, coef=np.array([1.0, 0.3]),
             vcov=np.array([[1.0, 0.0]]))
    # coef vs vcov size mismatch
    with pytest.raises(ValueError, match="must match"):
        qdrg("g", df, coef=np.array([1.0, 0.3]),
             vcov=np.eye(3))
    # coef vs design columns mismatch
    with pytest.raises(ValueError, match="design columns"):
        qdrg("g", df, coef=np.array([1.0]),
             vcov=np.eye(1))


def test_emmobj_builds_modelinfo_with_factor_levels():
    """``emmobj`` is the low-level constructor without a formula.
    It should produce a ModelInfo that carries the supplied β,
    V, and factor levels — even though many emmeans paths require
    ``design_info`` (which emmobj leaves as ``None``)."""
    from pymmeans import emmobj
    from pymmeans.utils import ModelInfo

    info = emmobj(
        bhat=np.array([1.0, 0.5, -0.3]),
        V=np.diag([0.04, 0.04, 0.04]),
        levels={"g": ["a", "b", "c"]},
        df=20,
    )
    assert isinstance(info, ModelInfo)
    assert info.factors == {"g": ["a", "b", "c"]}
    assert info.design_info is None
    np.testing.assert_array_equal(info.beta, [1.0, 0.5, -0.3])
    np.testing.assert_array_equal(info.vcov, np.diag([0.04, 0.04, 0.04]))
    assert info.df_resid == 20.0


def test_qdrg_rejects_non_symmetric_vcov():
    """previously, ``qdrg`` would accept a
    non-symmetric vcov and silently produce wrong SEs. now,
    a non-symmetric matrix is refused with a clear message that
    points at the likely cause (lower-/upper-triangular slice with
    the other half implicit). Symmetric matrices are still
    accepted, including those with tiny FP-noise asymmetry."""
    from pymmeans import qdrg

    df = pd.DataFrame({"y": [0.0] * 6, "g": pd.Categorical(["a", "b"] * 3)})

    # Wildly asymmetric (off-diagonals have opposite signs).
    bad = np.array([[1.0, 0.5], [-0.5, 1.0]])
    with pytest.raises(ValueError, match="symmetric"):
        qdrg("g", df, coef=np.array([1.0, 0.3]), vcov=bad)

    # FP-noise asymmetry is tolerated and symmetrised silently.
    almost_sym = np.array(
        [[1.0, 0.5 + 1e-13], [0.5, 1.0]]
    )
    info = qdrg("g", df, coef=np.array([1.0, 0.3]), vcov=almost_sym)
    np.testing.assert_allclose(info.vcov, info.vcov.T, atol=1e-15)


def test_emmobj_design_info_error_is_helpful():
    """previously, calling ``emmeans(emmobj(...), "g")``
    raised a generic 'pickled and lost design_info' error — but the
    user did not pickle anything; they used ``emmobj``. now,
    the error message specifically mentions ``emmobj`` and
    suggests ``qdrg`` as the workaround."""
    from pymmeans import emmeans, emmobj

    info = emmobj(
        bhat=np.array([1.0, 0.5, -0.3]),
        V=np.diag([0.04, 0.04, 0.04]),
        levels={"g": ["a", "b", "c"]},
    )
    with pytest.raises(ValueError, match="emmobj.*qdrg"):
        emmeans(info, "g")


def test_qdrg_then_pairs_works_end_to_end():
    """End-to-end: ``qdrg → emmeans → pairs`` produces a sensible
    contrast table."""
    from pymmeans import emmeans, pairs, qdrg

    df = pd.DataFrame({"y": [0.0] * 9, "g": pd.Categorical(["a", "b", "c"] * 3)})
    info = qdrg(
        "g", df, coef=np.array([2.0, 0.5, -0.3]),
        vcov=np.diag([0.04, 0.04, 0.04]), df=50,
    )
    em = emmeans(info, "g")
    p = pairs(em)
    # 3 levels → 3 pairwise contrasts
    assert len(p.frame) == 3
    # a-b = -0.5, a-c = +0.3, b-c = +0.8
    contrasts = dict(zip(p.frame["contrast"], p.frame["estimate"], strict=True))
    np.testing.assert_allclose(contrasts["a - b"], -0.5, atol=1e-10)
    np.testing.assert_allclose(contrasts["a - c"], +0.3, atol=1e-10)
    np.testing.assert_allclose(contrasts["b - c"], +0.8, atol=1e-10)
