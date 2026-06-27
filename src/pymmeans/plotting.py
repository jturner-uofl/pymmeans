"""Plotting helpers: forest plot of EMMs and interaction plots.

Two visual conventions matching R ``emmeans``:

- :func:`plot` is the **forest plot** — one row per EMM with a
  horizontal CI bar. Mirrors ``emmeans::plot.emmGrid(..., horiz=TRUE)``.
- :func:`emmip` is the **interaction plot** — one line per by-level
  across the x-factor's levels, with optional CI ribbon. Mirrors
  ``emmeans::emmip(model, ~ x | by)``.

Both lazily import matplotlib so the package stays importable without
the ``[plot]`` extra; calling either without matplotlib installed
raises a clear ``ImportError`` pointing at the install command.

References
----------
- Lewis, J. R. & Lenth, R. (2024). ``emmip`` documentation in the
  ``emmeans`` R package.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pymmeans.emmeans import EMMResult, emmeans

if TYPE_CHECKING:
    import pandas as pd
    from matplotlib.axes import Axes


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "Plotting requires matplotlib. Install it with: pip install pymmeans[plot]"
        ) from e
    return plt


def _prediction_interval_bounds(
    emm: EMMResult, level: float
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Per-row prediction-interval bounds for a forest / interaction plot.

    Returns ``(pi_lower, pi_upper)`` or ``(None, None)``. Mirrors R
    ``plot.emmGrid(PIs=TRUE)`` / ``emmip(PIs=TRUE)``:

    * OLS (no GLM family): ``PI = emmean ± qt(level, df) · sqrt(SE² +
      σ²)`` where ``σ²`` is the residual variance (``info.scale``).
      The per-row SE is backed out of the stored CI half-width so the
      same df / multiplier convention is reused.
    * GLM family: residual variance on the link scale isn't well-
      defined, so R refuses PIs and warns. We mirror that — warn and
      return ``(None, None)``.
    """
    if emm.model_info.family is not None:
        import warnings as _w
        _w.warn(
            "Prediction intervals are not available for this object",
            UserWarning,
            stacklevel=3,
        )
        return None, None
    from scipy import stats as _stats

    sigma2 = float(getattr(emm.model_info, "scale", 0.0) or 0.0)
    frame = emm.frame
    half = (
        frame["upper_cl"].to_numpy() - frame["lower_cl"].to_numpy()
    ) / 2.0
    df_arr = frame["df"].to_numpy(dtype=float)
    crit = _stats.t.ppf(0.5 + level / 2.0, df_arr)
    with np.errstate(divide="ignore", invalid="ignore"):
        SE = np.where(crit > 0, half / crit, half)
    pi_half = crit * np.sqrt(SE ** 2 + sigma2)
    emm_vals = frame["emmean"].to_numpy()
    return emm_vals - pi_half, emm_vals + pi_half


def plot(
    emm: EMMResult,
    ax: Axes | None = None,
    ref_line: float | None = None,
    *,
    sep: str | None = None,
    comparisons: bool = False,
    comparison_level: float = 0.95,
    comparison_adjust: str = "tukey",
    PIs: bool = False,
    level: float = 0.95,
) -> Axes | np.ndarray:
    """Forest plot of EMMs with confidence intervals.

    Parameters
    ----------
    emm
        Result from ``emmeans(...)``.
    ax
        Optional existing matplotlib Axes; a new figure is created
        otherwise. Ignored when ``sep=`` is set (the function
        creates a fresh grid of facet Axes).
    ref_line
        Optional x-value at which to draw a vertical reference line
        (e.g. 0 to flag "no effect" or 0.5 for response-scale
        binomial).
    sep
        Optional factor name. When supplied, the forest plot is
        split into one Axes per unique level of ``sep``; the function
        returns a 1-D ``np.ndarray`` of Axes objects (one per panel)
        instead of a single Axes. The named factor must be a column
        in ``emm.frame`` AND must currently appear in
        ``emm.target + emm.by``. The factor is removed from the row
        labels of each panel (the panel header carries the value).
    comparisons
        When ``True``, overlay pairwise-comparison "arrows" on each
        EMM in addition to the standard CI bars. Each arrow's
        half-width is sized so that two arrows overlap iff the
        corresponding pair is **not** significantly different at
        ``comparison_level`` under ``comparison_adjust``. Mirrors R
        ``plot.emmGrid(..., comparisons=TRUE)``. Drawn in a
        contrasting style above (or, when arrows would overlap the
        CI, below) the CI bar.

        Implementation note: the half-width is computed as
        ``c_i = q · SE_i / sqrt(2)`` where ``q`` is the studentised
        critical value at the requested level + adjustment. This
        produces the symmetric Gabriel-style comparison interval —
        easy to verify, slightly conservative versus R's exact
        asymmetric solve, but the overlap-iff-nonsignificant
        interpretation holds for the balanced case. R's exact
        asymmetric arrows are a 0.2.0 candidate.
    comparison_level
        Confidence level for the overlap test. Default 0.95.
    comparison_adjust
        Multiplicity adjustment driving the critical value for the
        overlap test. ``"tukey"`` (default) matches R's
        plot.emmGrid behaviour; ``"none"`` produces unadjusted
        per-pair intervals; ``"bonferroni"`` / ``"sidak"`` are
        accepted as conservative alternatives.
    PIs
        When ``True``, overlay a wider **prediction interval** band
        (``emmean ± qt(level, df) · sqrt(SE² + σ²)``, ``σ²`` =
        residual variance) behind the CI bars. Mirrors R
        ``plot.emmGrid(PIs = TRUE)``. Refused (with a warning, like R)
        on GLM fits where a link-scale residual variance isn't
        well-defined.
    level
        Confidence level for the prediction-interval band. Default
        0.95. (The CI bars use the EMM's own stored ``level``.)

    Returns
    -------
    Axes | ndarray of Axes
        A single Axes when ``sep`` is ``None``; otherwise a 1-D
        ndarray of Axes (one per facet).
    """
    plt = _require_matplotlib()

    # ``sep`` short-circuit: facet split. Each facet calls back
    # into ``plot`` recursively with the row-subset EMMResult,
    # ``sep=None``, and the same ``comparisons`` / ``ref_line``
    # settings — so the comparison-arrow code below only runs in
    # the non-faceted branch (and per-facet recursively).
    if sep is not None:
        return _plot_faceted(
            emm,
            sep=sep,
            ref_line=ref_line,
            comparisons=comparisons,
            comparison_level=comparison_level,
            comparison_adjust=comparison_adjust,
            PIs=PIs,
            level=level,
        )

    if ax is None:
        _, ax = plt.subplots(figsize=(7, max(3.0, len(emm.frame) * 0.35)))

    label_cols = emm.target + emm.by
    labels = [
        ", ".join(f"{c}={v}" for c, v in zip(label_cols, row, strict=True))
        for row in emm.frame[label_cols].itertuples(index=False, name=None)
    ]
    y_pos = np.arange(len(emm.frame))
    means = emm.frame["emmean"].to_numpy()
    lo = emm.frame["lower_cl"].to_numpy()
    hi = emm.frame["upper_cl"].to_numpy()

    # Prediction-interval band (drawn FIRST so the CI bar sits on top).
    if PIs:
        pi_lo, pi_hi = _prediction_interval_bounds(emm, level)
        if pi_lo is not None:
            ax.errorbar(
                means,
                y_pos,
                xerr=[means - pi_lo, pi_hi - means],
                fmt="none",
                capsize=7,
                elinewidth=1.0,
                color="C0",
                alpha=0.35,
                label="PI",
            )

    ax.errorbar(
        means,
        y_pos,
        xerr=[means - lo, hi - means],
        fmt="o",
        capsize=4,
        color="C0",
        label="CI",
    )
    if PIs:
        ax.legend(loc="best", fontsize="small", frameon=False)

    # Pairwise-comparison arrows overlay. Half-width formula:
    # c_i = q · SE_i / sqrt(2). Two arrows overlap iff
    # |mu_i - mu_j| < (c_i + c_j) = q · (SE_i + SE_j) / sqrt(2),
    # which is the Gabriel-style approximation to "pair (i, j) is
    # not significantly different at level alpha under the chosen
    # adjustment". For balanced designs and Tukey adjustment this
    # is essentially R's plot.emmGrid output; for unbalanced designs
    # the symmetric formula is slightly conservative versus R's
    # exact asymmetric solve (the visual interpretation —
    # "overlapping = not significantly different" — holds in either
    # case).
    if comparisons:
        _overlay_comparison_arrows(
            ax,
            emm,
            means=means,
            y_pos=y_pos,
            level=comparison_level,
            adjust=comparison_adjust,
        )
        # Add a tiny legend marker for the arrow series so a viewer
        # can distinguish CI bars (the wider segment) from
        # comparison arrows (the narrower one). Use the default
        # matplotlib legend; only render when there's something to
        # show beyond the single ``errorbar`` call.
        ax.legend(loc="best", fontsize="small", frameon=False)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    scale = "response" if emm.type == "response" else "link"
    pct = round(emm.level * 100)
    interval = (
        "CrI"
        if getattr(emm, "inference_kind", "wald") == "posterior"
        else "CI"
    )
    ax.set_xlabel(f"EMM ({scale} scale, {pct}% {interval})")

    if ref_line is not None:
        ax.axvline(ref_line, color="red", linestyle="--", alpha=0.5)

    return ax


def _plot_faceted(
    emm: EMMResult,
    *,
    sep: str,
    ref_line: float | None,
    comparisons: bool,
    comparison_level: float,
    comparison_adjust: str,
    PIs: bool = False,
    level: float = 0.95,
) -> np.ndarray:
    """Split the forest plot into facets by the named factor."""
    plt = _require_matplotlib()
    from dataclasses import replace as _dc_replace

    import pandas as pd

    if not isinstance(sep, str) or not sep:
        raise ValueError(
            "plot(sep=...) must be a non-empty string naming a "
            "factor column in emm.frame."
        )
    if sep not in emm.frame.columns:
        raise ValueError(
            f"plot(sep={sep!r}): column not present in emm.frame. "
            f"Available columns: {list(emm.frame.columns)}."
        )
    # ``sep`` should genuinely be a target / by factor — otherwise
    # we'd be slicing on something that doesn't vary across grid
    # rows in a meaningful way (each row would be its own facet).
    sep_dims = emm.target + emm.by
    if sep not in sep_dims:
        raise ValueError(
            f"plot(sep={sep!r}): the facet column must appear in "
            f"emm.target or emm.by (current dimensions: {sep_dims}). "
            f"Got a non-dimension column; that would produce one row "
            "per facet which isn't useful."
        )
    # Discover the levels in the column's natural order (Categorical
    # cat.categories if present, else first-occurrence).
    col = emm.frame[sep]
    if isinstance(col.dtype, pd.CategoricalDtype):
        levels = [lv for lv in col.cat.categories if lv in set(col)]
    else:
        levels = list(pd.unique(col))
    if not levels:
        raise ValueError(
            f"plot(sep={sep!r}): no levels present in the column."
        )

    n_facets = len(levels)
    # Per-facet row count for sizing the panel heights.
    rows_per_facet = [
        int((col == lv).sum()) for lv in levels
    ]
    max_rows = max(rows_per_facet) if rows_per_facet else 1

    fig, axes = plt.subplots(
        n_facets,
        1,
        figsize=(7, max(3.0, max_rows * 0.35) * n_facets * 0.9),
        sharex=True,
    )
    if n_facets == 1:
        axes = np.array([axes])
    else:
        axes = np.asarray(axes).reshape(-1)

    # Slice the EMMResult per facet. Drop ``sep`` from target / by
    # of each sub-EMM so the row labels in each facet don't repeat
    # the panel-header information.
    sub_target = [t for t in emm.target if t != sep]
    sub_by = [b for b in emm.by if b != sep]
    for ax_facet, lv in zip(axes, levels, strict=True):
        mask = (col == lv).to_numpy()
        if not mask.any():
            ax_facet.set_visible(False)
            continue
        sub_frame = emm.frame.loc[mask].reset_index(drop=True)
        sub_linfct = emm.linfct[mask]
        sub_emm = _dc_replace(
            emm,
            frame=sub_frame,
            linfct=sub_linfct,
            target=sub_target,
            by=sub_by,
        )
        # Recurse with sep=None so the per-facet plot uses the
        # full single-Axes machinery (including comparisons
        # overlay if requested).
        plot(
            sub_emm,
            ax=ax_facet,
            ref_line=ref_line,
            sep=None,
            comparisons=comparisons,
            comparison_level=comparison_level,
            comparison_adjust=comparison_adjust,
            PIs=PIs,
            level=level,
        )
        ax_facet.set_title(f"{sep} = {lv}", fontsize=10, loc="left")
    fig.tight_layout()
    return axes


def _overlay_comparison_arrows(
    ax: Axes,
    emm: EMMResult,
    *,
    means: np.ndarray,
    y_pos: np.ndarray,
    level: float,
    adjust: str,
) -> None:
    """Draw Gabriel-style comparison intervals on a forest Axes.

    Each EMM gets a horizontal segment of half-width
    ``c_i = q · SE_i / sqrt(2)``, where ``q`` is the critical
    value at the chosen level + adjustment. Two segments overlap
    iff the corresponding pair is non-significantly different
    under that adjustment (Gabriel, 1978).
    """
    from scipy import stats

    n = len(means)
    if n < 2:
        # No pairs to draw — degenerate to a no-op rather than
        # erroring (a user might splat ``comparisons=True``
        # uniformly across multiple EMM objects of varying size).
        return
    se = emm.frame["SE"].to_numpy()
    df_arr = emm.frame["df"].to_numpy()
    # Use a representative df for the critical value. Per-row df is
    # the right thing in principle but the overlap-as-significance
    # interpretation needs a SHARED critical value across all
    # arrows. Use the harmonic mean to handle ragged df without
    # silently letting one large-df row dominate.
    df_finite = df_arr[np.isfinite(df_arr)]
    if len(df_finite) == 0:
        df_for_q = np.inf
    else:
        with np.errstate(divide="ignore", invalid="ignore"):
            df_for_q = len(df_finite) / np.sum(1.0 / df_finite)
        if not np.isfinite(df_for_q):
            df_for_q = float(np.median(df_finite))
    alpha = 1.0 - level
    adj_lower = adjust.lower()
    if adj_lower in ("none", "no", "wald"):
        q = stats.t.ppf(1.0 - alpha / 2.0, df_for_q)
    elif adj_lower == "bonferroni":
        # n_pairs = n*(n-1)/2 comparisons in the family.
        n_pairs = n * (n - 1) // 2
        q = stats.t.ppf(1.0 - alpha / (2.0 * n_pairs), df_for_q)
    elif adj_lower == "sidak":
        n_pairs = n * (n - 1) // 2
        q_alpha = 1.0 - (1.0 - alpha) ** (1.0 / n_pairs)
        q = stats.t.ppf(1.0 - q_alpha / 2.0, df_for_q)
    else:
        # Tukey: studentised range divided by sqrt(2) gives the
        # equivalent t-style critical value for pairwise
        # comparisons. R uses qtukey; scipy exposes
        # ``stats.studentized_range``.
        q_tukey = stats.studentized_range.ppf(level, n, df_for_q)
        q = q_tukey / np.sqrt(2.0)
    half = q * se / np.sqrt(2.0)
    # Stagger the arrows slightly below the CI errorbar so both
    # are readable. ``y_pos`` is integer-spaced; offset by 0.18
    # which is well inside the 1.0 step.
    arrow_y = y_pos + 0.18
    for i in range(n):
        if not (np.isfinite(half[i]) and np.isfinite(means[i])):
            continue
        ax.plot(
            [means[i] - half[i], means[i] + half[i]],
            [arrow_y[i], arrow_y[i]],
            color="C3",
            lw=2.0,
            solid_capstyle="butt",
            label="comparison" if i == 0 else None,
        )


def _emmip_frame(
    emm: EMMResult,
    x: str,
    by: str | None,
    show_ci: bool,
    PIs: bool,
    level: float,
) -> pd.DataFrame:
    """Build R-style ``emmip(plotit=FALSE)`` data frame.

    Mirrors R `emmip.emmGrid`'s ``plotit = FALSE`` return. R's column
    order is:

        <by>, <x>, yvar, SE, df, [LCL, UCL,] [LPL, UPL,] tvar, xvar

    where the original factor columns (``<by>``, ``<x>``) are
    retained alongside the synthetic ``tvar`` / ``xvar`` plotting
    helpers. Row order is x-major (e.g. tension L, M, H — model's
    factor order), then by-major (e.g. wool A, B), matching R's
    `expand.grid` layout.
    """
    import numpy as np
    import pandas as pd

    frame = emm.frame.copy()
    # R orders rows: x varies slowest, by varies fastest within each x.
    # In pymmeans's EMM frame, by-cells are usually contiguous (rows
    # are by-major). Reorder to match R.
    if by is not None:
        # Use Categorical to preserve factor-level order of x and by.
        frame_sorted = frame.sort_values(
            by=[x, by],
            key=lambda s: pd.Categorical(
                s,
                categories=(
                    list(emm.model_info.factors.get(s.name, s.unique()))
                ),
                ordered=True,
            ).codes,
        ).reset_index(drop=True)
    else:
        frame_sorted = frame.sort_values(
            by=[x],
            key=lambda s: pd.Categorical(
                s,
                categories=list(emm.model_info.factors.get(s.name, s.unique())),
                ordered=True,
            ).codes,
        ).reset_index(drop=True)

    cols: dict[str, np.ndarray] = {}
    # Original factor columns first (R parity).
    if by is not None:
        cols[by] = frame_sorted[by].to_numpy()
    cols[x] = frame_sorted[x].to_numpy()
    cols["yvar"] = frame_sorted["emmean"].to_numpy()
    cols["SE"] = frame_sorted["SE"].to_numpy()
    cols["df"] = frame_sorted["df"].to_numpy()
    if show_ci:
        cols["LCL"] = frame_sorted["lower_cl"].to_numpy()
        cols["UCL"] = frame_sorted["upper_cl"].to_numpy()
    if PIs:
        if emm.model_info.family is not None:
            import warnings as _w
            _w.warn(
                "Prediction intervals are not available for this object",
                UserWarning,
                stacklevel=3,
            )
        else:
            from scipy import stats as _stats
            sigma2 = float(getattr(emm.model_info, "scale", 0.0) or 0.0)
            df_arr = frame_sorted["df"].to_numpy(dtype=float)
            crit = _stats.t.ppf(0.5 + level / 2.0, df_arr)
            half_ci = (
                frame_sorted["upper_cl"].to_numpy()
                - frame_sorted["lower_cl"].to_numpy()
            ) / 2.0
            with np.errstate(divide="ignore", invalid="ignore"):
                SE = np.where(crit > 0, half_ci / crit, half_ci)
            pi_half = crit * np.sqrt(SE ** 2 + sigma2)
            emm_vals = frame_sorted["emmean"].to_numpy()
            cols["LPL"] = emm_vals - pi_half
            cols["UPL"] = emm_vals + pi_half
    # R-style synthetic columns at the end: tvar (=by) and xvar (=x).
    if by is not None:
        cols["tvar"] = frame_sorted[by].astype(str).to_numpy()
    else:
        cols["tvar"] = "1"
    cols["xvar"] = frame_sorted[x].astype(str).to_numpy()
    return pd.DataFrame(cols)


def emmip(
    model: Any,
    x: str,
    by: str | None = None,
    at: dict[str, Any] | None = None,
    level: float = 0.95,
    type: str = "link",
    ax: Axes | None = None,
    show_ci: bool = True,
    PIs: bool = False,
    dodge: float = 0.0,
    CIs: bool | None = None,
    plotit: bool = True,
) -> Axes | pd.DataFrame:
    """Interaction plot of EMMs.

    Plots EMM of the target factor ``x`` on the y-axis, with one connected
    line per level of ``by`` (when provided). Mirrors R emmeans's
    ``emmip(~ x | by)`` for the no-faceting case.

    Parameters
    ----------
    model
        Fitted statsmodels model.
    x
        Factor whose levels form the x-axis.
    by
        Optional factor whose levels become separate lines.
    at, level, type
        Forwarded to ``emmeans()``.
    ax
        Optional existing matplotlib Axes.
    show_ci
        If True, draw a shaded band around each line at ``level`` CI bounds.
        Pymmeans default ``True`` differs from R `emmip(CIs = FALSE)`;
        pass ``CIs=False`` (R-style alias, takes precedence) for parity.
    CIs
        R-style alias for ``show_ci``. When supplied, overrides
        ``show_ci``. R's default is ``FALSE``; pymmeans keeps
        ``show_ci=True`` for backwards compatibility unless ``CIs`` is
        given.
    PIs
        80%-parity push: if True, also draw a wider band at
        the **prediction interval** bounds (mirrors R
        ``emmip(PIs = TRUE)``). The prediction SE for each EMM row is
        ``sqrt(SE^2 + sigma^2)`` where ``sigma^2`` is the model's
        residual variance (``model_info.scale``). Drawn at half the
        alpha of the CI band so both can be read together.
    dodge
        horizontal shift applied to each by-level's line
        (and band) so overlapping points / error bars stay readable.
        Lines are dodged at ``[-dodge*(n-1)/2, ..., +dodge*(n-1)/2]``,
        centered on zero. Default ``0.0`` (no dodge). Has no effect
        when ``by`` is None.
    plotit
        If ``False``, return the underlying ``pandas.DataFrame`` (with
        R-style columns ``xvar`` / ``yvar`` / ``SE`` / ``df`` /
        ``tvar`` / optional ``LCL`` / ``UCL`` / ``LPL`` / ``UPL``)
        instead of drawing the plot. Mirrors R
        ``emmip(..., plotit = FALSE)``. When ``plotit=False`` the R
        default of ``CIs=FALSE`` applies unless the caller explicitly
        passes ``CIs=True`` (this keeps the data-frame return
        columnar-stable for grep-style column matching from ported R
        code).
    """

    # Resolve CI flag: explicit `CIs=` overrides `show_ci=`. In the
    # plotit=False branch, R defaults to CIs=FALSE — match that when
    # the user didn't say otherwise.
    if CIs is not None:
        show_ci = bool(CIs)
    elif not plotit:
        show_ci = False

    emm = emmeans(model, x, by=by, at=at, level=level, type=type)
    frame = emm.frame

    # plotit=False short-circuit: build R-style data frame and return.
    if not plotit:
        return _emmip_frame(emm, x, by, show_ci=show_ci, PIs=PIs, level=level)

    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    # Build a prediction-SE for the optional PI band, matching R
    # `emmip.emmGrid(PIs=TRUE)` exactly:
    #
    # * OLS (no GLM family): use a **t**-quantile at the row's df,
    # not z. R does ``qt(.975, df)`` for the multiplier. Using z
    # produced bounds ~1.8% too narrow at df=66 (matched the bug
    # earlier reports flagged on InsectSprays: pymmeans got
    # [6.488, 22.512] vs R's [6.350, 22.650]).
    #
    # * GLM family (Poisson, Binomial, Gamma, ...): R refuses PIs
    # and warns "Prediction intervals are not available for this
    # object". The notion of a residual variance σ² on the LINK
    # scale isn't well-defined for a non-Gaussian GLM (the
    # variance is a function of the mean), so the PI band would
    # be meaningless. We mirror R: warn and disable PIs.
    if PIs:
        if emm.model_info.family is not None:
            import warnings as _w
            _w.warn(
                "Prediction intervals are not available for this object",
                UserWarning,
                stacklevel=2,
            )
            PIs = False
            pi_lower = pi_upper = None
        else:
            from scipy import stats as _stats
            sigma2 = float(getattr(emm.model_info, "scale", 0.0) or 0.0)
            half = (
                frame["upper_cl"].to_numpy() - frame["lower_cl"].to_numpy()
            ) / 2.0
            df_arr = frame["df"].to_numpy(dtype=float)
            # Per-row t critical value at this level. ``qt(p, Inf)`` →
            # z, so this remains z-equivalent for ``df=inf`` (mixed /
            # GEE results that still want a PI band).
            crit = _stats.t.ppf(0.5 + level / 2.0, df_arr)
            # Back out per-row SE then forward to PI half-width.
            with np.errstate(divide="ignore", invalid="ignore"):
                SE = np.where(crit > 0, half / crit, half)
            pi_half = crit * np.sqrt(SE ** 2 + sigma2)
            emm_vals = frame["emmean"].to_numpy()
            pi_lower = emm_vals - pi_half
            pi_upper = emm_vals + pi_half
    else:
        pi_lower = pi_upper = None

    def _dodge_xs(xs: np.ndarray, shift: float) -> np.ndarray:
        """Apply horizontal dodge to *numeric* x positions. Strings
        are first replaced by their positional index 0..n-1."""
        try:
            arr = np.asarray(xs, dtype=float)
        except (TypeError, ValueError):
            arr = np.arange(len(xs), dtype=float)
        return arr + shift

    if by is None:
        x_vals = frame[x].astype(str).to_numpy()
        means = frame["emmean"].to_numpy()
        ax.plot(x_vals, means, "o-", color="C0", label=x)
        if show_ci:
            ax.fill_between(
                x_vals,
                frame["lower_cl"].to_numpy(),
                frame["upper_cl"].to_numpy(),
                alpha=0.2,
                color="C0",
            )
        if PIs and pi_lower is not None:
            ax.fill_between(
                x_vals, pi_lower, pi_upper,
                alpha=0.10, color="C0",
                label="PI" if not show_ci else None,
            )
    else:
        by_levels = list(frame[by].cat.categories)
        n_by = len(by_levels)
        # Compute centred dodge offsets:
        # [-dodge*(n-1)/2, ..., +dodge*(n-1)/2]
        if dodge and n_by > 1:
            shifts = (np.arange(n_by) - (n_by - 1) / 2.0) * dodge
        else:
            shifts = np.zeros(n_by)

        for color_idx, by_level in enumerate(by_levels):
            sub = frame[frame[by] == by_level]
            sub_idx = sub.index.to_numpy()
            x_vals_raw = sub[x].astype(str).to_numpy()
            color = f"C{color_idx}"
            shift = shifts[color_idx]
            if shift != 0.0:
                x_pos = _dodge_xs(np.arange(len(x_vals_raw)), shift)
                # Hide the implicit x-tick text under categorical mode
                # by drawing string ticks once after the loop.
                ax.set_xticks(np.arange(len(x_vals_raw)))
                ax.set_xticklabels(x_vals_raw)
                x_use = x_pos
            else:
                x_use = x_vals_raw
            means = sub["emmean"].to_numpy()
            ax.plot(x_use, means, "o-", color=color, label=f"{by}={by_level}")
            if show_ci:
                ax.fill_between(
                    x_use,
                    sub["lower_cl"].to_numpy(),
                    sub["upper_cl"].to_numpy(),
                    alpha=0.2,
                    color=color,
                )
            if PIs and pi_lower is not None:
                ax.fill_between(
                    x_use,
                    pi_lower[sub_idx], pi_upper[sub_idx],
                    alpha=0.10, color=color,
                )
        ax.legend()

    ax.set_xlabel(x)
    scale = "response" if type == "response" else "link"
    ax.set_ylabel(f"EMM ({scale} scale)")
    return ax


def pwpp(
    emm: EMMResult,
    ax: Axes | None = None,
    adjust: str = "tukey",
    alpha: float = 0.05,
    method: str = "pairwise",
    sort: bool = True,
    values: bool = True,
) -> Axes:
    """Pairwise-p-value plot (Lenth, R ``emmeans::pwpp``).

    For each pair of EMMs ``(i, j)`` we plot a horizontal segment
    connecting the two EMM point estimates at the vertical position
    given by the adjusted p-value for ``emm_i - emm_j``. P-values are
    drawn on a base-10 log scale so small ones sit at the top.

    Pairs that are NOT significantly different at ``alpha`` are drawn
    in grey; significant pairs in red. A horizontal reference line is
    drawn at ``p = alpha``. This makes it easy to read off both which
    pairs are different and *how close* the non-significant ones are
    to the threshold — information you can't get from a forest plot of
    EMMs alone.

    Parameters
    ----------
    emm
        Result from ``emmeans(...)`` on the link scale (response-scale
        is refused via the same guard used by ``pairs``).
    ax
        Optional existing matplotlib Axes.
    adjust
        Multiplicity correction passed to ``pairs``/``contrast``.
        Default ``tukey``.
    alpha
        Threshold for the colour split and the reference line. Must
        be in ``(0, 1)``. Default 0.05.
    method
        Contrast method, mirroring R ``pwpp(method=...)``. Default
        ``"pairwise"``. Any string accepted by :func:`contrast`
        (``"pairwise"``, ``"revpairwise"``, ``"tukey"``,
        ``"trt.vs.ctrl"``, ``"trt.vs.ctrl1"``, ``"trt.vs.ctrlk"``,
        ...). Non-pairwise methods still plot one segment per
        contrast row, connecting the two levels whose labels show
        up as ``"a - b"`` in the contrast frame.
    sort
        If True (default), x-axis tick order follows ascending EMM
        estimate (matches R ``pwpp(sort = TRUE)``). If False, the
        x-axis is just the numeric EMM value with no reordering
        cosmetic.
    values
        If True (default), annotate each EMM tick with its numeric
        estimate just above the x-axis (matches R
        ``pwpp(values = TRUE)``).

    Returns
    -------
    matplotlib.axes.Axes

    Notes
    -----
    By-grouped EMMs: pwpp shows ALL pairs across groups in a single
    panel for v0.1 — call separately per by-group if you want the
    R-style faceted output. (Implementation note: R's pwpp also draws
    EMM tick marks on the x-axis at each estimate value; we do the
    same.)
    """
    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots(figsize=(8.0, 5.0))
    if getattr(emm, "type", "link") == "response":
        raise ValueError(
            "pwpp() requires a link-scale EMMResult; on response-scale "
            "the pairwise differences would be on the wrong scale. "
            "Pass an EMM with type='link' (the default)."
        )
    # refuse a bootstrap-derived EMM with a clear,
    # pwpp-specific error. Previously the internal ``pairs(emm)`` /
    # ``contrast(emm)`` call surfaced the refusal,
    # confusing users who hadn't called those wrappers themselves.
    # pwpp's job is to plot ADJUSTED PAIRWISE P-VALUES — bootstrap
    # results don't carry the per-pair analytic p-values needed for
    # the plot.
    if getattr(emm, "df_method", "default") == "bootstrap":
        raise ValueError(
            "pwpp() is not defined for a bootstrap-derived EMMResult "
            "(df_method='bootstrap'). The pairwise-p-value plot "
            "requires analytic-Wald per-pair p-values, which would "
            "silently mix with the stored percentile bootstrap "
            "uncertainty. Compute the plot on the raw EMM:\n"
            " pwpp(emmeans(model, ...)) # raw EMM\n"
        )
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}.")

    # 80%-parity push: route through `contrast()` so the
    # full method= surface (pairwise, revpairwise, trt.vs.ctrl,
    # tukey, ...) is reachable. Default keeps the old behaviour
    # (pairwise + Tukey) so existing callers are unaffected.
    from pymmeans.contrasts import contrast as _contrast
    from pymmeans.contrasts import pairs as _pairs

    if method == "pairwise":
        pr = _pairs(emm, adjust=adjust)
    else:
        pr = _contrast(emm, method=method, adjust=adjust)
    # The EMM estimate values give the x-coordinate; the contrast name
    # ("A - B") tells us which two EMMs to draw between.
    if "emmean" in emm.frame.columns:
        emm_value_col = "emmean"
    else:
        # Fall back to estimate-style columns
        from pymmeans.utils import detect_value_column

        kind_info = detect_value_column(emm.frame)
        if kind_info is None or kind_info[0] not in ("emm",):
            raise ValueError(
                "pwpp() needs an EMMResult with an 'emmean' column."
            )
        emm_value_col = kind_info[1]

    # use the structural row indices
    # via the `_pair_indices` dataclass field (used frame
    # columns; moved them off the public surface). The
    # label-parsing fallback is preserved for ContrastResults from
    # outside `pairs()`.
    from pymmeans.contrasts import _iter_by_groups, _row_labels

    sub = emm.frame.reset_index(drop=True)
    pair_indices = getattr(pr, "_pair_indices", None)
    has_indices = pair_indices is not None and len(pair_indices) == len(pr.frame)
    if has_indices:
        # Per-by-group x-position map: (by_key, within_group_idx) -> x
        group_x: dict[tuple, dict[int, float]] = {}
        for by_key, indices in _iter_by_groups(emm):
            group_x[by_key] = {
                local_i: float(sub[emm_value_col].iloc[orig_i])
                for local_i, orig_i in enumerate(indices)
            }
    else:
        emm_labels = _row_labels(sub, emm.target)
        label_to_x = {lab: float(sub[emm_value_col].iloc[i])
                      for i, lab in enumerate(emm_labels)}

    log10_alpha = np.log10(alpha)
    pmin = 1e-10 # floor for the log axis
    sig_color = "tab:red"
    ns_color = "0.55"

    for row_idx, (_, row) in enumerate(pr.frame.iterrows()):
        if has_indices:
            by_key = tuple(row[c] for c in emm.by) if emm.by else ()
            i, j = pair_indices[row_idx]
            xmap = group_x.get(by_key, {})
            if i not in xmap or j not in xmap:
                continue
            x1, x2 = xmap[i], xmap[j]
        else:
            label = row["contrast"]
            if " - " not in label:
                continue
            a, b = label.split(" - ", 1)
            if a not in label_to_x or b not in label_to_x:
                continue
            x1, x2 = label_to_x[a], label_to_x[b]
        p = float(row["p_value"])
        p = max(p, pmin)
        y = np.log10(p)
        is_sig = p < alpha
        color = sig_color if is_sig else ns_color
        ax.plot(
            [x1, x2], [y, y], "-",
            color=color, alpha=0.75 if is_sig else 0.6, linewidth=1.5,
        )

    # EMM tick marks at each estimate
    if has_indices:
        tick_xs = [x for xmap in group_x.values() for x in xmap.values()]
    else:
        tick_xs = list(label_to_x.values())
    for x in tick_xs:
        ax.axvline(x, color="black", alpha=0.15, linewidth=0.8)
    # Alpha reference line
    ax.axhline(log10_alpha, color="tab:red", linestyle="--", linewidth=0.8,
               alpha=0.7, label=f"alpha = {alpha}")

    # ``values=True`` (R parity) — annotate each EMM tick
    # with its numeric estimate, near the top of the plot just below
    # the p=1 line so they don't collide with segment endpoints.
    if values:
        # ``invert_yaxis()`` is called below, but at this point top
        # of axis is at y = -log10(pmin) and bottom is at 0. We want
        # the annotations near the bottom of the *flipped* axis,
        # which means small log10(p) — i.e. near p=1. After invert,
        # y=0.05 will be at the bottom edge; we annotate at y=0.03
        # so labels sit right above the x-axis baseline.
        ann_y = 0.03
        for x in tick_xs:
            ax.text(
                x, ann_y, f"{x:.3g}",
                ha="center", va="bottom",
                fontsize=8, color="black", alpha=0.8,
            )

    # ``sort=True`` (R parity) — re-order x-axis ticks to
    # ascending EMM estimate. This is purely cosmetic; the segment
    # endpoints are already at the EMM values themselves.
    if sort:
        sorted_xs = sorted(set(tick_xs))
        ax.set_xticks(sorted_xs)
        # Label each tick with its numeric value (helps when ``values``
        # is False but sort is True — gives the reader a way to read
        # off which EMM each tick corresponds to).
        ax.set_xticklabels([f"{x:.3g}" for x in sorted_xs])

    # Y-axis: show p-value (10^y) labels rather than raw log10
    yticks = [-3, -2, np.log10(0.05), -1, 0]
    ylabels = ["0.001", "0.01", "0.05", "0.1", "1"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylim(np.log10(pmin), 0.1)
    ax.invert_yaxis() # small p (top) -> big p (bottom)

    target_label = " x ".join(emm.target) if emm.target else "EMM"
    ax.set_xlabel(f"{target_label} estimate")
    ax.set_ylabel(f"Pairwise p-value (adjust={adjust})")
    ax.legend(loc="best", frameon=False)
    return ax


# ---------------------------------------------------------------------------
# marginaleffects-style plots: predictions / slopes / comparisons.
#
# These visualise the result frames produced by ``avg_predictions``,
# ``avg_slopes``, and ``avg_comparisons``. They mirror the
# ``marginaleffects`` ``plot_predictions`` / ``plot_slopes`` /
# ``plot_comparisons`` family: a fitted curve (or points) with a
# confidence band for predictions across a focal variable, and forest-style
# point-and-interval plots for the averaged slopes / comparisons.
# ---------------------------------------------------------------------------


def _condition_grid(col: pd.Series, grid_n: int):
    """Grid of values for a focal ``condition`` variable.

    Returns ``(values, is_numeric)``: a numeric linspace over the observed
    range, or the ordered categorical levels.
    """
    import numpy as _np
    import pandas as _pd

    if _pd.api.types.is_numeric_dtype(col):
        lo, hi = float(_np.nanmin(col)), float(_np.nanmax(col))
        return _np.linspace(lo, hi, grid_n), True
    levels = list(col.cat.categories) if hasattr(col, "cat") else sorted(_pd.unique(col))
    return levels, False


def plot_predictions(
    obj: Any,
    condition: str,
    *,
    type: str = "response",
    grid_n: int = 50,
    level: float = 0.95,
    ax: Axes | None = None,
):
    """Plot adjusted predictions across the values of a focal variable.

    For each value ``v`` of ``condition`` the variable is set to ``v`` for
    every observed row (g-computation), the model predicts, and the
    predictions are averaged -- an *average adjusted prediction* curve.
    Numeric conditions render as a line with a confidence band; categorical
    conditions render as points with error bars.

    Parameters
    ----------
    obj
        A fitted model or pymmeans result carrying ``model_info``.
    condition
        Name of the focal predictor on the x-axis.
    type
        ``"response"`` (default) or ``"link"``.
    grid_n
        Number of grid points for a numeric condition.
    level
        Confidence level for the band.
    ax
        Optional existing Axes.

    Returns
    -------
    matplotlib Axes.
    """
    import numpy as _np
    import scipy.stats as _st

    from pymmeans.comparisons import _cf_design, _pred_fn
    from pymmeans.slopes import _beta_jacobian, _df_value, _get_info, _require_reference_data

    plt = _require_matplotlib()
    info = _get_info(obj)
    data = _require_reference_data(info, "plot_predictions")
    if condition not in data.columns:
        raise ValueError(f"{condition!r} is not a column of the model data.")
    pred = _pred_fn(info, type)
    beta = _np.asarray(info.beta, dtype=float)
    vcov = _np.asarray(info.vcov, dtype=float)
    df_value = _df_value(info)
    crit = (
        _st.t.isf((1.0 - level) / 2.0, df_value)
        if _np.isfinite(df_value)
        else _st.norm.isf((1.0 - level) / 2.0)
    )

    values, is_numeric = _condition_grid(data[condition], grid_n)
    ests, ses = [], []
    n = len(data)
    for v in values:
        newval = _np.full(n, v, dtype=float) if is_numeric else v
        design = _cf_design(info, data, condition, newval)

        def theta(b, _d=design):
            return [pred(_d @ b).mean()]

        est, jac = _beta_jacobian(theta, beta)
        ests.append(float(est[0]))
        ses.append(float(_np.sqrt(max((jac @ vcov @ jac.T)[0, 0], 0.0))))
    ests = _np.asarray(ests)
    ses = _np.asarray(ses)
    lo, hi = ests - crit * ses, ests + crit * ses

    if ax is None:
        _, ax = plt.subplots()
    if is_numeric:
        ax.plot(values, ests, color="C0", lw=2)
        ax.fill_between(values, lo, hi, color="C0", alpha=0.2)
    else:
        xpos = _np.arange(len(values))
        ax.errorbar(xpos, ests, yerr=[ests - lo, hi - ests], fmt="o", color="C0", capsize=4)
        ax.set_xticks(xpos)
        ax.set_xticklabels([str(v) for v in values])
    ax.set_xlabel(condition)
    ax.set_ylabel(f"Predicted ({type})")
    ax.set_title(f"Adjusted predictions over {condition}")
    return ax


def _forest(frame, value_col, label_col, *, ref, xlabel, title, ax):
    """Shared forest plotter: one point + CI per row of a result frame."""
    import numpy as _np

    plt = _require_matplotlib()
    if ax is None:
        _, ax = plt.subplots()
    labels = [str(v) for v in frame[label_col]]
    ypos = _np.arange(len(frame))[::-1]
    est = frame[value_col].to_numpy(dtype=float)
    lo = frame["lower_cl"].to_numpy(dtype=float)
    hi = frame["upper_cl"].to_numpy(dtype=float)
    ax.errorbar(
        est, ypos, xerr=[est - lo, hi - est], fmt="o", color="C0", capsize=4
    )
    if ref is not None:
        ax.axvline(ref, color="0.5", ls="--", lw=1)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return ax


def plot_slopes(
    obj: Any,
    var: str,
    *,
    by: str | list[str] | None = None,
    type: str = "response",
    level: float = 0.95,
    ax: Axes | None = None,
):
    """Forest plot of the average marginal effect of ``var``.

    One point-and-interval per ``by`` group (or a single point for the
    overall average marginal effect). A dashed reference line marks zero.

    Returns the matplotlib Axes.
    """
    from pymmeans.slopes import avg_slopes

    res = avg_slopes(obj, var, by=by, type=type, level=level)
    frame = res.frame.copy()
    by_list = [by] if isinstance(by, str) else (list(by) if by else [])
    if by_list:
        frame["_label"] = frame[by_list].astype(str).agg(", ".join, axis=1)
        label_col = "_label"
    else:
        frame["_label"] = var
        label_col = "_label"
    return _forest(
        frame, "slope", label_col, ref=0.0,
        xlabel=f"d E[y]/d {var} ({type})",
        title=f"Average marginal effect of {var}",
        ax=ax,
    )


def plot_comparisons(
    obj: Any,
    var: str,
    *,
    by: str | list[str] | None = None,
    comparison: str = "difference",
    type: str = "response",
    level: float = 0.95,
    ax: Axes | None = None,
):
    """Forest plot of the average counterfactual comparison(s) of ``var``.

    One point-and-interval per contrast (and per ``by`` group). The
    reference line is 0 for ``difference``/``lnratio``/``lnor``/``lift`` and
    1 for ``ratio``.

    Returns the matplotlib Axes.
    """
    from pymmeans.comparisons import avg_comparisons

    res = avg_comparisons(obj, var, by=by, comparison=comparison, type=type, level=level)
    frame = res.frame.copy()
    by_list = [by] if isinstance(by, str) else (list(by) if by else [])
    parts = ["contrast"] + by_list if by_list else ["contrast"]
    frame["_label"] = frame[parts].astype(str).agg(", ".join, axis=1)
    ref = 1.0 if comparison == "ratio" else 0.0
    return _forest(
        frame, "estimate", "_label", ref=ref,
        xlabel=f"{comparison} ({type})",
        title=f"Average comparison of {var}",
        ax=ax,
    )
