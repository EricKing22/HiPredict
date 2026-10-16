"""Per-metadata-field NMAE distributions for the fields SHAP ranks highest.

Companion figure to the SHAP summary plots: SHAP ranks *which* metadata fields
drive the loss, this shows *how the loss is distributed* within each of them. For
the top-N fields (ordered by SHAP importance) each panel splits the field into
quantile bins and overlays one NMAE histogram per bin, over a grey silhouette of
the overall validation distribution.

Histograms rather than per-bin means: NMAE is strongly right-skewed, so a bin's
mean is set by its tail. Seeing the whole shape shows whether a bin is uniformly
worse or merely carries more outliers.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# EDR metadata uses -9998.0 as its "value not available" sentinel
MISSING_SENTINEL = -9998.0

DEFAULT_METRIC = "CNN NMAE"


def observation_metadata_table(EDR_df, meta_cols):
    """Collapse the per-CCD EDR table to one row per observation.

    EDR_df carries one row per CCD channel, so merging it straight onto
    per-image results would duplicate every result row ~13x. Numeric fields are
    averaged across the channels of an observation, with the missing sentinel
    masked out first so it cannot drag the mean.

    Called from corr_evals, which already holds EDR_df in memory - the CSV is
    ~1.7 GB, so it must not be re-loaded to build this table.
    """
    cols = [c for c in meta_cols if c in EDR_df.columns]
    if not cols:
        return None

    tbl = EDR_df[["OBSERVATION_ID"] + cols].copy()
    tbl[cols] = tbl[cols].replace(MISSING_SENTINEL, np.nan)
    return tbl.groupby("OBSERVATION_ID", as_index=False)[cols].mean()


def _fmt_edge(value):
    return f"{value:.3g}"


def _unique_labels(labels):
    """Bin labels must be unique to be used as categories; disambiguate collisions
    (adjacent edges that round to the same short string) with a bin index."""
    if len(set(labels)) == len(labels):
        return labels
    return [f"{i + 1}: {lab}" for i, lab in enumerate(labels)]


def _bin_field(series, mode, n_bins, max_discrete):
    """Bin one metadata field, returning an ordered categorical Series or None.

    Fields with few unique values are grouped by exact value instead of binned,
    matching the categorical/continuous split used in eval_correlations.py. The
    result is always a Series on the input index so callers can mask with it.
    """
    valid = series.dropna()
    n_unique = valid.nunique()
    if n_unique < 2:
        return None

    if n_unique <= max_discrete:
        values = np.sort(valid.unique())
        ordered = _unique_labels([_fmt_edge(v) for v in values])
        labels = dict(zip(values, ordered))
        binned = pd.Categorical(series.map(labels), categories=ordered, ordered=True)
        return pd.Series(binned, index=series.index)

    try:
        if mode == "quantile":
            intervals = pd.qcut(series, q=n_bins, duplicates="drop")
        else:
            intervals = pd.cut(series, bins=n_bins)
    except (ValueError, IndexError):
        return None

    cats = intervals.cat.categories
    if len(cats) < 2:
        return None
    labels = _unique_labels([f"{_fmt_edge(c.left)}–{_fmt_edge(c.right)}" for c in cats])
    return intervals.cat.rename_categories(labels)


def _hist_edges(values, n_hist_bins):
    """Shared log-spaced NMAE bin edges so every panel and series is comparable.

    Cut at the 0.5/99.5th percentiles of the whole validation set; callers clip
    into that range rather than dropping, so the tails stay visible in the end
    bins instead of silently leaving the plot.
    """
    finite = values[np.isfinite(values) & (values > 0)]
    if finite.size < 2:
        return None
    lo = float(np.percentile(finite, 0.5))
    hi = float(np.percentile(finite, 99.5))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return np.logspace(np.log10(lo), np.log10(hi), n_hist_bins + 1)


def _series_pct(values, edges):
    """Percentage of `values` falling in each NMAE bin, tails clipped inward."""
    clipped = np.clip(values, edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    total = counts.sum()
    if total == 0:
        return None
    return 100.0 * counts / total


def _draw_hist_panel(ax, field, rank, sub, binned, metric_col, edges, ref_pct, min_bin_count):
    """One panel: overlaid per-bin NMAE histograms over the global silhouette.

    Bins are ordered, so they get a single-hue light-to-dark ramp rather than
    categorical colours - the ramp direction carries the field's ordering.
    Returns the per-bin stats rows for the stdout table, or None if unplottable.
    """
    cats = list(binned.cat.categories)
    cmap = plt.get_cmap("Blues")
    # Floor at 0.45 so the lowest bin still separates from the grey silhouette
    colours = cmap(np.linspace(0.45, 1.0, len(cats)))

    if ref_pct is not None:
        ax.stairs(
            ref_pct,
            edges,
            fill=True,
            color="0.90",
            edgecolor="0.75",
            linewidth=0.8,
            zorder=1,
            label="all validation images",
        )

    rows = []
    for cat, colour in zip(cats, colours):
        values = sub.loc[binned == cat, metric_col].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size < min_bin_count:
            continue
        pct = _series_pct(values, edges)
        if pct is None:
            continue

        median = float(np.median(values))
        ax.stairs(pct, edges, color=colour, linewidth=1.8, zorder=3)
        # Median as a dotted rule in the series colour - the histograms overlap
        # heavily at the mode, so the central tendency needs its own mark
        ax.axvline(median, color=colour, linestyle=":", linewidth=1.2, zorder=2)
        ax.plot([], [], color=colour, linewidth=1.8,
                label=f"{cat}  (n={values.size}, med={median:.2f})")

        rows.append({
            "field": field,
            "bin": str(cat),
            "n": values.size,
            "mean": float(np.mean(values)),
            "median": median,
            "p90": float(np.percentile(values, 90)),
        })

    if not rows:
        return None

    ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    # NMAE spans well under a decade here, so decade-only ticks leave the axis
    # near-unlabelled - tick at 1/2/5 per decade and print plain numbers
    ax.xaxis.set_major_locator(mticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_title(f"#{rank}  {field}", fontsize=10)
    ax.set_xlabel("DN NMAE (log scale, tails clipped into end bins)", fontsize=8)
    ax.set_ylabel("% of images in bin", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
    return rows


def _print_bin_tables(metric_col, stats_rows):
    """Per-bin distribution stats, plus the worst/best median ratio per field."""
    if not stats_rows:
        return
    table = pd.DataFrame(stats_rows)
    print(f"\n{metric_col} distribution by metadata bin (top fields by SHAP importance):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    spread = table.groupby("field", sort=False)["median"].agg(["min", "max"])
    spread["worst/best"] = spread["max"] / (spread["min"] + 1e-12)
    spread = spread.sort_values("worst/best", ascending=False)
    print(f"\n{metric_col} median spread across bins, worst-bin/best-bin ratio:")
    print(spread.to_string(float_format=lambda v: f"{v:.4g}"))


def plot_nmae_hist_by_metadata(
    global_val_df,
    corr_vals,
    cnn_name,
    SAVE_DIR,
    metric_col=DEFAULT_METRIC,
    top_n=6,
    mode="quantile",
    n_bins=5,
    max_discrete=7,
    min_bin_count=20,
    n_hist_bins=36,
    n_cols=3,
):
    """Save one figure: per-bin NMAE histograms for the top-N SHAP metadata fields.

    Fields are taken in SHAP-importance order (from ``corr_vals``); degenerate
    ones (constant over the validation set, or too few populated bins) are
    skipped and the next-ranked field is used instead, so the figure carries a
    full ``top_n`` panels where the data allows.
    """
    if global_val_df is None or global_val_df.empty:
        print("Skipping metadata NMAE histograms: no global validation rows found.")
        return

    missing = [c for c in ["Observation", metric_col] if c not in global_val_df.columns]
    if missing:
        print(f"Skipping metadata NMAE histograms: missing columns {missing}")
        return

    corr_vals = corr_vals or {}
    ranking = list(corr_vals.get("feature_ranking") or [])
    if not ranking:
        print("Skipping metadata NMAE histograms: no SHAP feature ranking available.")
        return

    meta_tbl = corr_vals.get("meta_obs_tbl")
    if meta_tbl is None or meta_tbl.empty:
        print("Skipping metadata NMAE histograms: no per-observation metadata table in corr_vals.")
        return
    meta_cols = [c for c in meta_tbl.columns if c != "OBSERVATION_ID"]

    merged = global_val_df.merge(
        meta_tbl,
        how="left",
        left_on="Observation",
        right_on="OBSERVATION_ID",
    ).drop(columns=["OBSERVATION_ID"], errors="ignore")
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=[metric_col])
    if merged.empty:
        print("Skipping metadata NMAE histograms: no finite metric rows after merge.")
        return

    edges = _hist_edges(merged[metric_col].to_numpy(dtype=np.float64), n_hist_bins)
    if edges is None:
        print("Skipping metadata NMAE histograms: metric has no usable positive range.")
        return
    ref_pct = _series_pct(merged[metric_col].to_numpy(dtype=np.float64), edges)

    panels = []
    for field in [f for f in ranking if f in meta_cols]:
        if len(panels) >= top_n:
            break
        sub = merged.dropna(subset=[field])
        if sub.empty:
            continue
        binned = _bin_field(sub[field], mode, n_bins, max_discrete)
        if binned is None:
            continue
        panels.append((field, ranking.index(field) + 1, sub, binned))

    if not panels:
        print("Skipping metadata NMAE histograms: no plottable fields.")
        return

    n_rows = int(np.ceil(len(panels) / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.0 * n_cols, 4.4 * n_rows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    stats_rows = []
    drawn = 0
    for field, rank, sub, binned in panels:
        rows = _draw_hist_panel(
            axes[drawn], field, rank, sub, binned, metric_col, edges, ref_pct, min_bin_count
        )
        if rows is None:
            axes[drawn].clear()
            continue
        stats_rows.extend(rows)
        drawn += 1

    if drawn == 0:
        plt.close(fig)
        print("Skipping metadata NMAE histograms: no field had enough populated bins.")
        return

    for ax in axes[drawn:]:
        ax.set_visible(False)

    fig.suptitle(
        f"{metric_col} distribution by metadata bin – top {drawn} fields by SHAP importance "
        f"({mode} bins, dotted rule = bin median, grey = all validation images)",
        fontsize=13,
    )

    out_path = os.path.join(SAVE_DIR, f"EVALS_{cnn_name}_nmae_hist_by_metadata.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved metadata NMAE histograms to {out_path}")

    _print_bin_tables(metric_col, stats_rows)
