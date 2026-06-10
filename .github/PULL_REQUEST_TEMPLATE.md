## Summary

<!-- 1-2 sentences: what does this PR change, and why? -->

## Type of change

- [ ] Bug fix (numerical correctness, crash, or regression)
- [ ] New feature (new public API surface or extension to an existing one)
- [ ] R-parity update (closing a gap relative to R `emmeans` / `pbkrtest`)
- [ ] Documentation
- [ ] Internal refactor / cleanup (no public API change)
- [ ] Test coverage

## Validation evidence

- [ ] Added unit tests covering the change
- [ ] All existing tests still pass (`make test` or `.venv/bin/pytest -q`)
- [ ] Where applicable, added a contract in `examples/jss_audit/_build_case_study.py`
- [ ] For R-parity changes: added an R-side reference CSV under `tests/r_reference/`
- [ ] For numerical changes: documented the tolerance class in the test

## Checklist

- [ ] I have read and agree to the [Code of Conduct](../CODE_OF_CONDUCT.md)
- [ ] The change preserves backwards compatibility on the documented public API, OR breaking changes are listed in the PR description
- [ ] Public-facing additions include docstrings
- [ ] `ruff check` is clean (`.venv/bin/ruff check .`)

## References

<!-- Link any related issues, papers, or R documentation. -->
