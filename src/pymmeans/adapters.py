"""Pluggable model-adapter protocol.

Each fitted-model framework gets one ``ModelAdapter`` subclass that
implements detection (does this result come from my framework?) and
construction (build a ``ModelInfo`` from it). The registry below holds
the built-in adapters for statsmodels and linearmodels; third-party
packages can extend it with ``register_adapter``.

Adapter-protocol design (instead of bespoke ``from_pymc()`` /
``from_tsa()`` functions in ``utils.py``): every new framework just
provides a small adapter class that the dispatcher picks via
``detects()``.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from pymmeans.utils import ModelInfo, from_linearmodels, from_statsmodels


@runtime_checkable
class ModelAdapter(Protocol):
    """Adapter for plugging a fitted-model framework into pymmeans.

    Implementations are stateless; both methods are class/staticmethods
    and the only instance state is the class-level ``name`` attribute
    (used for debugging and error messages).

    A correct adapter must satisfy two contracts:

    1. ``detects(result)`` returns ``True`` only for fitted-result
       objects this adapter can fully construct a :class:`ModelInfo`
       from. It MUST NOT raise on alien inputs — return ``False``.
    2. ``build(result, **kwargs)`` returns a populated
       :class:`pymmeans.utils.ModelInfo`. Adapters that need extra
       state (e.g. ``LinearmodelsAdapter`` needing the original
       DataFrame) accept it via keyword arguments propagated through
       :func:`pymmeans.utils.from_fitted`.
    """

    name: str

    @staticmethod
    def detects(result: Any) -> bool:
        """Cheap duck-type check: can this adapter handle ``result``?

        Must never raise — return ``False`` for inputs that don't match.
        """
        ...

    @staticmethod
    def build(result: Any, **kwargs: Any) -> ModelInfo:
        """Construct a :class:`ModelInfo` from the fitted result.

        Free to raise informative errors if ``result`` looks right at
        the detection level but turns out to be unsupported (e.g. a
        random-effects structure we don't yet handle).
        """
        ...


class StatsmodelsAdapter:
    """Built-in adapter for ``statsmodels`` results (OLS / GLM / MixedLM).

    Detection relies on patsy's ``design_info`` being attached to
    ``result.model.data``, which is true for any model fit through the
    formula API (``smf.ols``, ``smf.glm``, ``smf.mixedlm``, ...).
    """

    name = "statsmodels"

    @staticmethod
    def detects(result: Any) -> bool:
        """True iff ``result`` carries a patsy ``design_info`` OR is a
        statsmodels result class we know about even when ``design_info``
        is unattached (so the ``from_statsmodels`` path can surface a
        targeted refusal with workaround steering instead of the
        generic ``No adapter recognises X`` dispatch error)."""
        model = getattr(result, "model", None)
        if model is None:
            return False
        data = getattr(model, "data", None)
        if data is not None and hasattr(data, "design_info"):
            return True
        # GLMGam appends spline-basis columns AFTER the patsy parse,
        # leaving design_info unset. Recognise it here so
        # ``from_statsmodels`` can raise the qdrg-workaround refusal at
        # the right layer.
        return type(model).__name__ == "GLMGam"

    @staticmethod
    def build(result: Any, **_kwargs: Any) -> ModelInfo:
        """Delegate to :func:`pymmeans.utils.from_statsmodels`."""
        return from_statsmodels(result)


class LinearmodelsAdapter:
    """Built-in adapter for ``linearmodels`` panel / IV results.

    linearmodels doesn't expose a patsy ``design_info`` post-fit, so we
    rebuild patsy's design from the formula + the user-supplied raw
    ``data=`` frame. See :func:`pymmeans.utils.from_linearmodels` for
    the round-trip caveats (explicit ``1 + ...`` intercept, absorbed
    effects stripped).
    """

    name = "linearmodels"

    @staticmethod
    def detects(result: Any) -> bool:
        """True iff ``result`` has a ``model.formula`` but no patsy
        ``design_info`` — the linearmodels post-fit shape."""
        model = getattr(result, "model", None)
        if model is None:
            return False
        data = getattr(model, "data", None)
        return hasattr(model, "formula") and not (
            data is not None and hasattr(data, "design_info")
        )

    @staticmethod
    def build(result: Any, *, data=None, **_kwargs: Any) -> ModelInfo:
        """Delegate to :func:`pymmeans.utils.from_linearmodels`.

        Pass the original raw DataFrame as ``data=``; linearmodels
        destroys the raw factor columns after the fit, so reconstructing
        patsy's factor metadata needs the original frame.
        """
        return from_linearmodels(result, data=data)


class PyFixestAdapter:
    """Built-in adapter for ``pyfixest`` ``Feols`` / ``Fepois`` results.

    ``pyfixest`` parses formulas with ``formulaic`` and *absorbs* fixed
    effects, so it exposes no patsy ``design_info`` and no
    reference-grid-able design for the absorbed terms. Reference-grid
    operations (:func:`pymmeans.emmeans`, :func:`pymmeans.avg_slopes`)
    therefore require patsy and are not yet supported on ``pyfixest``
    fits. What *is* supported is the coefficient-level surface that
    needs only the within-fixed-effect coefficient vector and its
    covariance: :func:`pymmeans.hypotheses` (linear and nonlinear
    delta-method tests of the estimated coefficients). The extracted
    coefficients and covariance match a dummy-encoded ``statsmodels``
    OLS to floating-point precision.
    """

    name = "pyfixest"

    @staticmethod
    def detects(result: Any) -> bool:
        """True iff ``result`` is a fitted pyfixest estimator.

        Duck-typed (so importing pyfixest is never required): a fitted
        ``Feols`` exposes ``coef()``, ``vcov()`` and ``_coefnames``.
        """
        return (
            hasattr(result, "coef")
            and hasattr(result, "vcov")
            and hasattr(result, "_coefnames")
            and type(result).__module__.split(".")[0] == "pyfixest"
        )

    @staticmethod
    def build(result: Any, **_kwargs: Any) -> ModelInfo:
        """Delegate to :func:`pymmeans.utils.from_pyfixest`."""
        from pymmeans.utils import from_pyfixest

        return from_pyfixest(result)


_ADAPTERS: list[type[ModelAdapter]] = [
    StatsmodelsAdapter,
    LinearmodelsAdapter,
    PyFixestAdapter,
]


def register_adapter(
    adapter: type[ModelAdapter], *, prepend: bool = False
) -> None:
    """Register a new model adapter.

    Adapters are tried in resolution order; the first whose ``detects``
    returns True wins.

    The adapter class must expose:
    - ``name`` (string)
    - ``detects(result) -> bool`` (callable)
    - ``build(result, **kwargs) -> ModelInfo`` (callable)

    Parameters
    ----------
    adapter
        Adapter class implementing the :class:`ModelAdapter` protocol.
    prepend
        If ``True``, insert ``adapter`` at the FRONT of the registry so
        it is tried before the built-in statsmodels / linearmodels
        adapters. Useful when a third-party framework wraps a
        statsmodels result and you want your adapter to take
        precedence. Default ``False`` (appended at the end).
    """
    if not isinstance(adapter, type):
        raise TypeError(
            "adapter must be a class implementing the ModelAdapter protocol."
        )
    if not isinstance(getattr(adapter, "name", None), str):
        raise TypeError(
            f"Adapter {adapter.__name__} must expose a string `name` attribute."
        )
    for attr in ("detects", "build"):
        fn = getattr(adapter, attr, None)
        if fn is None or not callable(fn):
            raise TypeError(
                f"Adapter {adapter.__name__}.{attr} must be a callable; "
                f"got {type(fn).__name__}."
            )
    if prepend:
        _ADAPTERS.insert(0, adapter)
    else:
        _ADAPTERS.append(adapter)


def adapters() -> list[type[ModelAdapter]]:
    """Return the current list of registered adapters (in resolution order)."""
    return list(_ADAPTERS)


def dispatch(result: Any, **kwargs: Any) -> ModelInfo:
    """Find the first matching adapter and build a ``ModelInfo``.

    Raises ``TypeError`` with a helpful message if no adapter recognises
    the input.
    """
    # special-case known-unsupported summary-only result
    # objects so users get a focused workaround pointer instead of
    # the generic "no adapter" message.
    cls_name = type(result).__name__
    if cls_name == "AnovaResults":
        # statsmodels' AnovaRM (`from statsmodels.stats.anova import
        # AnovaRM`) returns a summary-only ``AnovaResults`` object —
        # just an ANOVA table, no fitted model / beta / vcov. The
        # Python equivalent of R `aov(... + Error(subj))` is fitting a
        # MixedLM with the same random structure (which pymmeans
        # fully supports, including Satterthwaite / Kenward-Roger).
        raise NotImplementedError(
            "statsmodels' AnovaRM returns a summary-only ANOVA table "
            "(no fitted model / coefficients / vcov), so pymmeans "
            "cannot compute EMMs on it. The Python equivalent of "
            "R's `aov(y ~ a + Error(subj))` is a MixedLM with the "
            "same random structure: "
            "`smf.mixedlm('y ~ a', df, groups='subj').fit()`. "
            "pymmeans supports MixedLM with Satterthwaite "
            "(`apply_satterthwaite`) and Kenward-Roger "
            "(`apply_kenward_roger`) df corrections for "
            "lmerTest-grade inference."
        )
    for ad in _ADAPTERS:
        if ad.detects(result):
            return ad.build(result, **kwargs)
    names = ", ".join(ad.name for ad in _ADAPTERS)
    raise TypeError(
        f"No pymmeans adapter recognises {type(result).__name__}. "
        f"Registered adapters: {names}. Use register_adapter() to add one."
    )
