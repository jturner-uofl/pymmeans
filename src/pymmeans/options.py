"""Process / context-local default options for pymmeans (R `emm_options`).

 R `emmeans` uses `emm_options(...)`
to set global defaults that downstream functions read. pymmeans
previously required explicit kwargs on every call.

This module exposes:

- `emm_options(**kwargs)` — context manager that temporarily sets
  options. Use as ``with emm_options(level=0.99, adjust='bonferroni'):``.
- `get_emm_option(name, default=None)` — read the current value.
- `set_emm_options(**kwargs)` / `reset_emm_options()` — programmatic
  setters for non-context usage.

Backed by ``contextvars.ContextVar``, so options are thread-safe.
The contract (verified by regression test):

- Within a ``with emm_options(level=0.99)`` block, the calling
  thread / async-task sees ``0.99``.
- A separately-started thread (raw ``threading.Thread`` or
  ``concurrent.futures.ThreadPoolExecutor.submit``) does NOT
  inherit the submitter's options — the worker thread sees the
  default value, NOT ``0.99``. This is **thread isolation**, not
  thread propagation; if you want a worker to use a non-default
  option you must call ``emm_options`` inside the worker.
- Joblib parallel workers (separate processes) likewise do NOT
  inherit options.

This matches the ``_SattCache`` pickle pattern and avoids
surprise leakage across threads / processes.

**Explicit function kwargs always win.** Options are a *default*; if
you pass ``level=0.95`` to ``emmeans()``, it overrides whatever the
option says. This mirrors R's behavior and is essential for
reproducibility.

Supported option names (any extras are accepted but ignored by the
core functions that don't consult them; this is intentional so users
can plug new keys without modifying pymmeans):

- ``level``: default CI level for ``emmeans``, ``summary``, ``confint``
- ``adjust``: default adjustment for ``pairs``, ``contrast``
- ``type``: default scale for ``emmeans`` (``'link'`` or ``'response'``)
- ``weights``: default weighting for ``emmeans`` (``'equal'``,
  ``'proportional'``, ``'outer'``, ``'cells'``)
- ``infer``: default ``(show_ci, show_tests)`` for ``summary``
- ``dunnett_max_k``: safety cap on the exact-Dunnett MVT integral
  dimension (default 50; raise via ``set_emm_options(dunnett_max_k=...)``
  for batch jobs that can spend minutes per family)

Parallel-worker propagation
---------------------------

``ContextVar`` is **process-local**. Options set in the main
process do NOT propagate to:

- ``joblib.Parallel(..., backend="loky")`` workers
- ``concurrent.futures.ProcessPoolExecutor`` workers
- ``multiprocessing`` workers with ``start_method="spawn"`` or
  ``"forkserver"``

A worker sees the *defaults* hardcoded in each option's consumer
(e.g. ``dunnett_max_k=50``, ``level=0.95``), not whatever the parent
process set via ``emm_options(...)`` / ``set_emm_options(...)``.

If you need a non-default option inside a worker, call
``set_emm_options(...)`` **inside the worker function** (or wrap the
worker body in ``with emm_options(...):``). For ``joblib.Parallel``
this is the standard ``initializer=`` pattern. Document this in
sensitivity-analysis pipelines that rely on a raised
``dunnett_max_k`` — the workers will revert to 50 and raise
``ValueError`` unless you re-set inside.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

_OPTIONS: ContextVar[dict[str, Any] | None] = ContextVar(
    "pymmeans_options", default=None
)


def _current_options() -> dict[str, Any]:
    """hygiene (ruff B039): `ContextVar(default={})`
    shares a single mutable dict instance across all readers in the
    default state. We never mutate it (set/reset both go through
    ``copy()``-then-set), but the lint warning is correct that the
    pattern is dangerous. Switch to `default=None` and synthesise a
    fresh empty dict on read.
    """
    current = _OPTIONS.get()
    return current if current is not None else {}


@contextmanager
def emm_options(**kwargs: Any):
    """Temporarily set pymmeans default options for a code block.

    Examples
    --------
    >>> from pymmeans import emm_options, emmeans # doctest: +SKIP
    >>> with emm_options(level=0.99, adjust='bonferroni'): # doctest: +SKIP
    ... em = emmeans(model, 'a')
    ... pr = pairs(em)

    Outside the ``with`` block, defaults revert.

    Notes
    -----
    - Explicit kwargs always override the option. E.g.
      ``emmeans(model, 'a', level=0.90)`` uses 0.90 even if
      ``emm_options(level=0.99)`` is active.
    - Nested ``emm_options`` calls compose: inner overrides outer
      within its block, outer remains in effect after inner exits.
    - Each thread / async-task has its own option state (ContextVar
      semantics). Joblib parallel workers don't inherit options.
    """
    current = _current_options()
    new = {**current, **kwargs}
    token = _OPTIONS.set(new)
    try:
        yield
    finally:
        _OPTIONS.reset(token)


def get_emm_option(name: str, default: Any = None) -> Any:
    """Return the current value of an option, or `default` if unset.

    Useful for library code that wants to read an option as a fallback
    when the caller didn't pass an explicit kwarg::

        level = level if level is not None else get_emm_option('level', 0.95)
    """
    return _current_options().get(name, default)


def set_emm_options(**kwargs: Any) -> None:
    """Set options globally for the current context, NOT scoped.

    Use sparingly — prefer the ``emm_options()`` context manager. This
    is here for notebook / REPL workflows where users want persistent
    defaults across cells without nesting `with` blocks.

    To clear, call :func:`reset_emm_options`.

    .. warning::
       Options are backed by :class:`contextvars.ContextVar`, which is
       **process-local**. ``set_emm_options(dunnett_max_k=500)`` in the
       main process does NOT propagate to ``joblib.Parallel`` workers,
       ``ProcessPoolExecutor`` workers, or any other separate-process
       worker. Re-call ``set_emm_options(...)`` inside the worker
       function (or use the ``initializer=`` hook in your parallel
       runner). See the module docstring for the full propagation
       table.
    """
    current = _current_options().copy()
    current.update(kwargs)
    _OPTIONS.set(current)


def reset_emm_options() -> None:
    """Clear all options. Useful for tests and notebook cleanup."""
    _OPTIONS.set(None)
