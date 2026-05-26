"""Analytic marginalization — the heart of pymmeans's performance story.

Computes the marginalized L matrix (``L_marg``) directly from patsy's
term / factor / contrast metadata, without materializing the reference
grid. For each Term in the design, the column at a particular row of
the FULL grid is the row-wise Kronecker product of its factors' coded
values (patsy's standard expansion; see Bates & Maechler's writeups on
model-matrix construction). Under uniform sampling over the cartesian
product of factor levels, factors are independent — so the expectation
of the product factorises into the Kronecker product of per-factor
expectations:

    E[ kron(coded_F1, coded_F2, ...) ] = kron(E[coded_F1], E[coded_F2], ...)

That identity is what lets us skip the grid. For each factor F in a
term:

- Categorical and **fixed** at a target/by level: row of the
  contrast matrix corresponding to that level
- Categorical and **averaged** over its available spec levels: mean of
  contrast-matrix rows (weighted by per-factor freqs for ``weights=
  "outer"`` — see :func:`analytic_marginalize`'s ``factor_weights``
  argument)
- Numerical and fixed: the value itself
- Numerical and averaged: the mean of its spec values

Patsy interleaves interaction columns so that EARLIER factors in
``subterm.factors`` vary fastest (inner index). ``np.kron(A, B)`` puts
B as the inner index, so we fold from the LEFT: ``result = f0;
result = kron(f1, result); ...``. The Kronecker-ordering invariant is
verified against the eager streamed path in
``tests/test_analytic.py::test_analytic_matches_streamed``.

Complexity is O(n_terms · n_target_keys) — no patsy calls, no grid
materialization. For the GitHub ``emmeans`` issue #282 scenario
(46-million-row reference grid) this collapses runtime from about 88
seconds (streamed chunked path) to ~13 ms (analytic).

For R-style ``weights="proportional"`` (joint cell counts), the
Kronecker factorisation no longer holds — we enumerate the cartesian
product of non-target factor levels per target key (see
:func:`analytic_marginalize_proportional`). Slower, but matches R when
non-target factors are correlated in the data.

References
----------
- Searle, Speed & Milliken (1980), the population-marginal-means
  framework that defines what we're computing.
- patsy documentation, "Coding categorical data", for the term/subterm
  expansion convention.
"""

from __future__ import annotations

import itertools

import numpy as np

from pymmeans.utils import ModelInfo


def _subterm_columns(
    subterm,
    design_info,
    key_dict: dict[str, object],
    spec: dict[str, list],
    factor_weights: dict[str, np.ndarray] | None = None,
    info: ModelInfo | None = None,
) -> np.ndarray:
    if not subterm.factors:
        return np.array([1.0])

    factor_values: list[np.ndarray] = []
    for factor in subterm.factors:
        factor_info = design_info.factor_infos[factor]
        fname = factor.name()

        if factor_info.type == "categorical":
            contrast = subterm.contrast_matrices[factor].matrix # (k, k')
            levels = list(factor_info.categories)
            if fname in key_dict:
                idx = levels.index(key_dict[fname])
                factor_values.append(contrast[idx, :].astype(float))
            else:
                avail = spec.get(fname, levels)
                indices = [levels.index(lv) for lv in avail]
                rows = contrast[indices, :]
                # Apply per-level weights if provided (proportional/outer modes)
                if factor_weights is not None and fname in factor_weights:
                    full_w = factor_weights[fname]
                    sub_w = full_w[indices]
                    s = sub_w.sum()
                    w = sub_w / s if s > 0 else np.full_like(sub_w, 1.0 / len(sub_w))
                    factor_values.append((w[:, None] * rows).sum(axis=0))
                else:
                    factor_values.append(rows.mean(axis=0))
        else: # numerical
            # handle multi-column numeric factors (splines,
            # polynomials, basis expansions like ``bs(x, df=3)``,
            # ``ns(x, df=4)``, ``poly(x, 3)``). Detect via the
            # info.multi_col_factors registry populated at adapter time.
            is_multi_col = (
                info is not None
                and fname in info.multi_col_factors
            )
            if is_multi_col:
                factor_values.append(
                    _evaluate_multi_col_factor(
                        factor=factor,
                        factor_info=factor_info,
                        info=info,
                        key_dict=key_dict,
                        spec=spec,
                    )
                )
            elif fname in key_dict:
                factor_values.append(np.array([float(key_dict[fname])]))
            else:
                avail = spec.get(fname, [0.0])
                factor_values.append(np.array([float(np.mean(avail))]))

    result = factor_values[0]
    for fv in factor_values[1:]:
        result = np.kron(fv, result)
    return result


def _evaluate_multi_col_factor(
    factor,
    factor_info,
    info: ModelInfo,
    key_dict: dict[str, object],
    spec: dict[str, list],
) -> np.ndarray:
    """Evaluate a multi-column numeric basis (``bs``/``ns``/``poly``/etc.) at
    the requested covariate value(s) and return the factor's basis row.

    Strategy: build a 1-row DataFrame containing the underlying
    column(s) at their target/grid/mean values, then call
    ``patsy.build_design_matrices`` against the fit's stored
    ``design_info`` to re-evaluate the basis using the same knots /
    coefficients patsy stored at fit time. Slice out the term's
    columns and return them.

    pre-this-function pymmeans refused multi-column numeric
    expressions at adapter time. Reusing ``build_design_matrices``
    sidesteps the need to re-implement spline / polynomial / arbitrary
    basis math in pymmeans — patsy already has it.
    """
    import patsy

    design_info = info.design_info
    fname = factor.name()
    underlying = info.multi_col_factors[fname]

    # Build a 1-row template DataFrame with all underlying columns
    # set to either the user-supplied value (key_dict) / grid value
    # (spec) or the data mean (numeric_means).
    template: dict[str, object] = {}
    for col in underlying:
        if col in key_dict:
            template[col] = float(key_dict[col])
        elif col in spec:
            # Marginalising over this covariate: R `emmeans`'s
            # default `cov.reduce=mean` plugs in the mean of the
            # underlying covariate (not the mean of the basis
            # values), which we match here.
            template[col] = float(np.mean(spec[col]))
        else:
            template[col] = float(info.numeric_means.get(col, 0.0))
    # Evaluate just THIS factor's basis at the requested values.
    # ``factor.eval(state, data_dict)`` uses the factor's stored
    # stateful-transform metadata (knots for splines, orthogonal
    # polynomial coefficients, etc.) to re-evaluate the basis
    # without going through the full design matrix. This works
    # for both main-effect terms and interaction-only terms (where
    # the factor never appears as a standalone term).
    # narrowed the previous bare ``except``
    # to specific known patsy error types so unexpected exceptions
    # (e.g. user-data type errors, OOM) propagate rather than
    # silently dropping into the slow fallback path.
    data_dict = {col: np.array([float(val)]) for col, val in template.items()}
    factor_state = getattr(factor_info, "state", None)
    try:
        if factor_state is None:
            raise AttributeError(
                "factor_info.state is None; cannot use factor.eval()"
            )
        basis_row = np.asarray(
            factor.eval(factor_state, data_dict), dtype=float
        )
    except (AttributeError, KeyError, TypeError):
        # Fall back to the build_design_matrices path when the
        # factor doesn't expose a state (or the state isn't in the
        # form ``factor.eval`` expects). Defense-in-depth for
        # unusual factor classes; ordinary patsy splines /
        # polynomials always have a usable state.
        new_row = info.data.iloc[:1].copy()
        for col, val in template.items():
            new_row[col] = val
        new_X = np.asarray(
            patsy.build_design_matrices([design_info], new_row)[0]
        )
        for term in design_info.terms:
            if len(term.factors) == 1 and term.factors[0] is factor:
                tslice = design_info.term_slices[term]
                return np.asarray(new_X[0, tslice], dtype=float)
        cols = [
            i for i, n in enumerate(design_info.column_names)
            if n.startswith(fname + "[")
        ]
        return np.asarray(new_X[0, cols], dtype=float)
    # ``factor.eval`` returns a (1, k) array — flatten to (k,).
    return basis_row.reshape(-1)


def _term_marginal_columns(
    term,
    design_info,
    key_dict: dict[str, object],
    spec: dict[str, list],
    factor_weights: dict[str, np.ndarray] | None = None,
    info: ModelInfo | None = None,
) -> np.ndarray:
    if not term.factors:
        return np.array([1.0])

    subterms = design_info.term_codings[term]
    return np.concatenate(
        [
            _subterm_columns(
                st, design_info, key_dict, spec, factor_weights, info=info,
            )
            for st in subterms
        ]
    )


def analytic_marginalize_proportional(
    info: ModelInfo,
    spec: dict[str, list],
    group_cols: list[str],
    joint_counts,
) -> tuple[np.ndarray, list[tuple]]:
    """Marginalize using **joint** non-target cell frequencies (R 'proportional').

    Semantics match R ``emmeans(weights='proportional')``:

    - Non-target *factor* combinations are weighted by their joint
      frequency in the training data, **filtered to the levels the spec
      actually allows** (``at={"h": ["x"]}`` restricts the joint table to
      rows where ``h == "x"`` and **renormalises** the remaining mass to
      sum to 1).
    - Non-target *numeric* covariates with multi-value ``at=`` (e.g.
      ``at={"x": [0, 10]}``) are iterated uniformly over the supplied
      values, mirroring the equal/outer paths.

    Slower than the Kronecker path (which only works under independence,
    i.e. R's 'outer'), but matches R when non-target factors are
    correlated in the data OR when ``at=`` restricts the grid.
    """
    design_info = info.design_info
    target_keys: list[tuple] = list(
        itertools.product(*[spec[c] for c in group_cols])
    )

    non_target_factors = [
        c for c in spec if c not in group_cols and c in info.factors
    ]
    non_target_numerics = [
        c for c in spec if c not in group_cols and c in info.numeric_means
    ]

    # Filter the joint factor table to spec-allowed levels and renormalise.
    spec_levels = {c: set(spec[c]) for c in non_target_factors}
    filtered_joint = {
        combo: w
        for combo, w in joint_counts.items()
        if len(combo) == len(non_target_factors)
        and all(
            lvl in spec_levels[fname]
            for fname, lvl in zip(non_target_factors, combo, strict=False)
        )
    }
    total = sum(filtered_joint.values())
    if total > 0:
        filtered_joint = {k: v / total for k, v in filtered_joint.items()}
    elif non_target_factors:
        # Spec excluded every observed combination; fall back to uniform
        # over the spec's cartesian product so the result is at least
        # interpretable rather than zero.
        all_combos = list(itertools.product(*[spec[c] for c in non_target_factors]))
        filtered_joint = {combo: 1.0 / len(all_combos) for combo in all_combos}
    elif not non_target_factors:
        filtered_joint = {(): 1.0}

    # Numeric non-targets contribute a uniform 1/N weight over their
    # spec values.
    n_numeric_combos = 1
    for c in non_target_numerics:
        n_numeric_combos *= len(spec[c])
    numeric_weight = 1.0 / n_numeric_combos if n_numeric_combos else 1.0

    L_marg = np.zeros((len(target_keys), info.n_params))

    for ki, target_key in enumerate(target_keys):
        target_dict = dict(zip(group_cols, target_key, strict=True))
        for factor_combo in itertools.product(
            *[spec[c] for c in non_target_factors]
        ) if non_target_factors else [()]:
            factor_w = float(filtered_joint.get(tuple(factor_combo), 0.0))
            if factor_w == 0.0:
                continue
            for numeric_combo in itertools.product(
                *[spec[c] for c in non_target_numerics]
            ) if non_target_numerics else [()]:
                kd = dict(target_dict)
                for fname, lvl in zip(non_target_factors, factor_combo, strict=True):
                    kd[fname] = lvl
                for cname, val in zip(non_target_numerics, numeric_combo, strict=True):
                    kd[cname] = val
                w = factor_w * numeric_weight
                for term in design_info.terms:
                    tslice = design_info.term_slices[term]
                    cols = _term_marginal_columns(
                        term, design_info, kd, spec, info=info,
                    )
                    L_marg[ki, tslice] += w * cols

    return L_marg, target_keys


def analytic_marginalize(
    info: ModelInfo,
    spec: dict[str, list],
    group_cols: list[str],
    factor_weights: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, list[tuple]]:
    """Compute (L_marg, unique_keys) analytically.

    Parameters
    ----------
    info
        Model metadata.
    spec
        Output of ``build_grid_spec(info, at)`` — value lists per variable.
    group_cols
        Variables that vary across L_marg rows (target + by).

    Returns
    -------
    L_marg : ndarray of shape (n_keys, n_params)
    keys : list of tuples, one per L_marg row
    """
    design_info = info.design_info
    keys: list[tuple] = list(itertools.product(*[spec[c] for c in group_cols]))
    L_marg = np.empty((len(keys), info.n_params))

    # Cache term metadata once per term — saves a tiny amount per row.
    term_meta = [
        (term, design_info.term_slices[term]) for term in design_info.terms
    ]

    for ki, key in enumerate(keys):
        key_dict = dict(zip(group_cols, key, strict=True))
        for term, tslice in term_meta:
            cols = _term_marginal_columns(
                term, design_info, key_dict, spec, factor_weights, info=info,
            )
            expected = tslice.stop - tslice.start
            if len(cols) != expected:
                raise RuntimeError(
                    f"Term {term} produced {len(cols)} columns, expected "
                    f"{expected}. (Analytic marginalization mismatch — please "
                    "report this as a bug.)"
                )
            L_marg[ki, tslice] = cols

    return L_marg, keys
