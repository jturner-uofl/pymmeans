"""Tests for the model-adapter protocol and registry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

from pymmeans import (
    LinearmodelsAdapter,
    StatsmodelsAdapter,
    emmeans,
    register_adapter,
)
from pymmeans.adapters import _ADAPTERS, adapters, dispatch
from pymmeans.utils import ModelInfo


def test_statsmodels_adapter_detects_ols():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.normal(size=20), "g": pd.Categorical(rng.choice(["a", "b"], 20))})
    fit = smf.ols("y ~ g", data=df).fit()
    assert StatsmodelsAdapter.detects(fit)
    assert not LinearmodelsAdapter.detects(fit)


def test_dispatch_returns_modelinfo():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"y": rng.normal(size=20), "g": pd.Categorical(rng.choice(["a", "b"], 20))})
    fit = smf.ols("y ~ g", data=df).fit()
    info = dispatch(fit)
    assert isinstance(info, ModelInfo)


def test_dispatch_raises_for_unknown_object():
    with pytest.raises(TypeError, match="No pymmeans adapter"):
        dispatch("not a model")


def test_custom_adapter_can_be_registered():
    """User registers a stub adapter that handles a custom marker class."""
    class _CustomResult:
        pass

    class _CustomAdapter:
        name = "custom"

        @staticmethod
        def detects(result):
            return isinstance(result, _CustomResult)

        @staticmethod
        def build(result, **kwargs):
            raise NotImplementedError("stub")

    assert isinstance(_CustomAdapter, type)
    register_adapter(_CustomAdapter)
    try:
        # The adapter should now match for _CustomResult instances
        assert _CustomAdapter in adapters()
        with pytest.raises(NotImplementedError, match="stub"):
            dispatch(_CustomResult())
    finally:
        # Undo registration so test isolation holds
        _ADAPTERS.remove(_CustomAdapter)


def test_register_adapter_rejects_non_classes():
    with pytest.raises(TypeError, match="must be a class"):
        register_adapter("not a class") # type: ignore[arg-type]


def test_register_adapter_rejects_non_callable_members():
    """an adapter with detects=True / build=True (not
    callable) used to slip through the registration check."""
    class BadDetects:
        name = "bad"
        detects = True # not callable

        @staticmethod
        def build(result):
            ...

    with pytest.raises(TypeError, match=r"detects.*callable"):
        register_adapter(BadDetects)

    class BadBuild:
        name = "bad"

        @staticmethod
        def detects(result):
            return False

        build = "string instead of callable"

    with pytest.raises(TypeError, match=r"build.*callable"):
        register_adapter(BadBuild)


def test_register_adapter_rejects_missing_name():
    class NoName:
        @staticmethod
        def detects(result):
            return False

        @staticmethod
        def build(result):
            ...

    with pytest.raises(TypeError, match="`name`"):
        register_adapter(NoName)


def test_register_adapter_prepend_allows_override():
    """built-in adapters were always tried first, so a
    user adapter couldn't override statsmodels detection. `prepend=True`
    fixes this."""
    class StatsmodelsOverride:
        name = "override"
        called = False

        @staticmethod
        def detects(result):
            return StatsmodelsAdapter.detects(result)

        @staticmethod
        def build(result, **kwargs):
            StatsmodelsOverride.called = True
            return StatsmodelsAdapter.build(result, **kwargs)

    register_adapter(StatsmodelsOverride, prepend=True)
    try:
        rng = np.random.default_rng(0)
        df = pd.DataFrame(
            {"y": rng.normal(size=10), "g": pd.Categorical(rng.choice(["a", "b"], 10))}
        )
        fit = smf.ols("y ~ g", data=df).fit()
        dispatch(fit)
        assert StatsmodelsOverride.called
    finally:
        _ADAPTERS.remove(StatsmodelsOverride)


def test_emmeans_uses_registry_by_default():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(
        {
            "y": rng.normal(size=30),
            "g": pd.Categorical(rng.choice(["a", "b", "c"], 30)),
        }
    )
    fit = smf.ols("y ~ g", data=df).fit()
    emm = emmeans(fit, "g")
    assert emm.n_rows == 3
