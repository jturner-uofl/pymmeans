"""Shared pytest fixtures.

Tests that need canonical R-emmeans datasets (``pigs``, ``warpbreaks``,
``neuralgia``, etc.) fetch them directly from
``statsmodels.datasets.get_rdataset``.
"""

from __future__ import annotations
