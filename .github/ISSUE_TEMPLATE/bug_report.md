---
name: Bug report
about: Report a numerical mismatch, crash, or unexpected behaviour.
title: "[bug] "
labels: bug
assignees: ''
---

## Summary

<!-- One sentence: what's wrong? -->

## Minimal reproducing example

```python
# Smallest self-contained Python snippet that demonstrates the bug.
# Please include the dataset (or a minimal synthetic version that
# triggers it) and any seed required for reproducibility.
import pymmeans
...
```

## Expected vs. actual

- **Expected** (e.g., what R `emmeans` produces, or what the docs claim):
- **Actual** (what `pymmeans` produces):

## Environment

| | |
|---|---|
| `pymmeans` version | `import pymmeans; pymmeans.__version__` |
| Python | `python --version` |
| OS | macOS / Linux / Windows + version |
| Key dependency versions | `numpy`, `pandas`, `scipy`, `statsmodels` |

## Additional context

<!-- Reference to a CSV / dataset, related issues, or anything else
that helps reproduce the problem. -->
