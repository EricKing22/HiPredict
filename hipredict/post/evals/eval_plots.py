import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LogNorm
import os
import shap

from hipredict.data.testing.np_file_trends import plot_hexbin


def ecdf(x):
    x = np.sort(x)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def eval_db_plot(eval_cats, evals, cnn_name, SAVE_DIR=None):
    cats_theme = [k for k, v in eval_cats.items() if k in evals and str(v.get('eval_category', '')).casefold() == 'theme']

    nmae_cnn = np.array([evals[c]['NMAE_CNN'] for c in cats_theme], dtype=float)
    nmae_base = np.array([evals[c]['NMAE_BASE'] for c in cats_theme], dtype=float)
    gap = nmae_base - nmae_cnn
    order = np.argsort(gap)[::-1]  # biggest improvement first
    cats_ord = [cats_theme[i] for i in order]
    cnn_vals = nmae_cnn[order]
    base_vals = nmae_base[order]
    x = np.arange(len(cats_ord))
    ymin = np.minimum(cnn_vals, base_vals)
    ymax = np.maximum(cnn_vals, base_vals)

    fig, ax = plt.subplots(figsize=(max(10, 0.8 * len(cats_ord)), 4), constrained_layout=True)
    ax.vlines(x, ymin=ymin, ymax=ymax, linewidth=1, alpha=0.7, color='k', linestyles='--')
    ax.scatter(x, cnn_vals, s=30, label="HiPredict")
    ax.scatter(x, base_vals, s=30, label="SYN4")

    for i, g in enumerate((base_vals - cnn_vals)):
        ax.text(i, ymax[i] + 0.02 * (np.max(ymax) - np.min(ymin) + 1e-6),
                f"-{(100.0 * g / max(base_vals[i], 1e-8)):.0f}%",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(cats_ord, rotation=45, ha='right')
    ax.set_ylabel("NMAE (% of mean |true|, DN)")
    ax.set_ylim(bottom=0)
    ax.set_title("DN NMAE of HiPredict vs SYN4 by Scientific Theme")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()
    if SAVE_DIR:
        dumbbell_path = os.path.join(SAVE_DIR, f"EVALS_{cnn_name}_by_theme_NMAE.png")
        plt.savefig(dumbbell_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved NMAE dumbbell figure to {dumbbell_path}")
    else:
        return fig, ax  # for testing


def plot_syn4_vs_hipredict_hexbin(global_val_df, cnn_name, SAVE_DIR):
    if global_val_df is None or global_val_df.empty:
        print("Skipping SYN4 vs HiPredict hexbin: no global validation rows found.")
        return

    required = ["SYN4 NMAE", "CNN NMAE"]
    missing = [c for c in required if c not in global_val_df.columns]
    if missing:
        print(f"Skipping SYN4 vs HiPredict hexbin: missing columns {missing}")
        return

    df = global_val_df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    if df.empty:
        print("Skipping SYN4 vs HiPredict hexbin: no finite NMAE rows found.")
        return

    x = df["SYN4 NMAE"].to_numpy(dtype=np.float64)
    y = df["CNN NMAE"].to_numpy(dtype=np.float64)
    positive = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    x = x[positive]
    y = y[positive]

    if x.size == 0:
        print("Skipping SYN4 vs HiPredict hexbin: no positive finite NMAE rows found.")
        return

    out_path = os.path.join(SAVE_DIR, f"EVALS_{cnn_name}_syn4_vs_hipredict_dn_nmae_hexbin.png")
    plot_hexbin(
        x,
        y,
        a_name="SYN4 DN NMAE (% of mean |true|)",
        b_name="HiPredict DN NMAE (% of mean |true|)",
        fname=out_path,
        gridsize=60,
        sample=None,
        xyline=True,
        log_counts=True,
        scale="Image",
        positive_only=True,
        xscale="log",
        yscale="log",
        match_axes=True,
        title="HiPredict vs SYN4 DN NMAE Across Validation Images",
    )


def plot_syn4_binned_improvement(global_val_df, cnn_name, SAVE_DIR, n_bins=14, min_bin_count=10):
    if global_val_df is None or global_val_df.empty:
        print("Skipping binned SYN4 improvement plot: no global validation rows found.")
        return

    required = ["SYN4 NMAE", "CNN NMAE"]
    missing = [c for c in required if c not in global_val_df.columns]
    if missing:
        print(f"Skipping binned SYN4 improvement plot: missing columns {missing}")
        return

    df = global_val_df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    df = df[(df["SYN4 NMAE"] > 0) & (df["CNN NMAE"] > 0)].copy()
    if df.empty:
        print("Skipping binned SYN4 improvement plot: no positive finite NMAE rows found.")
        return

    x = df["SYN4 NMAE"].to_numpy(dtype=np.float64)
    ratio = (df["CNN NMAE"] / df["SYN4 NMAE"]).to_numpy(dtype=np.float64)

    df["_hipredict_syn4_ratio"] = ratio
    bin_edges = np.logspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
    bin_ids = np.digitize(x, bin_edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)
    df["_syn4_nmae_bin"] = bin_ids

    centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    medians = np.full(n_bins, np.nan, dtype=np.float64)
    q25 = np.full(n_bins, np.nan, dtype=np.float64)
    q75 = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=int)

    bin_summary_rows = []
    for i in range(n_bins):
        vals = ratio[bin_ids == i]
        if vals.size == 0:
            continue
        medians[i] = np.nanmedian(vals)
        q25[i] = np.nanpercentile(vals, 25)
        q75[i] = np.nanpercentile(vals, 75)
        counts[i] = vals.size
        bin_df = df[df["_syn4_nmae_bin"] == i]
        max_ratio_idx = bin_df["_hipredict_syn4_ratio"].idxmax()
        max_ratio_row = bin_df.loc[max_ratio_idx]
        bin_summary_rows.append({
            "bin": i,
            "syn4_left": bin_edges[i],
            "syn4_center": centers[i],
            "syn4_right": bin_edges[i + 1],
            "count": vals.size,
            "ratio_q25": q25[i],
            "ratio_median": medians[i],
            "ratio_q75": q75[i],
            "ratio_p90": np.nanpercentile(vals, 90),
            "frac_ratio_gt_0p75": np.nanmean(vals > 0.75),
            "frac_ratio_gt_1": np.nanmean(vals > 1.0),
            "cnn_nmae_median": np.nanmedian(bin_df["CNN NMAE"].to_numpy(dtype=np.float64)),
            "syn4_nmae_median": np.nanmedian(bin_df["SYN4 NMAE"].to_numpy(dtype=np.float64)),
            "max_ratio": max_ratio_row["_hipredict_syn4_ratio"],
            "max_ratio_observation": max_ratio_row.get("Observation", ""),
            "max_ratio_set": max_ratio_row.get("Set name", ""),
            "max_ratio_date": max_ratio_row.get("Date", ""),
        })

    if bin_summary_rows:
        bin_summary = pd.DataFrame(bin_summary_rows)
        print("\nBinned HiPredict/SYN4 relative NMAE summary:")
        print(
            bin_summary.to_string(
                index=False,
                float_format=lambda value: f"{value:.4g}",
            )
        )

    valid = counts > 0
    if not np.any(valid):
        print("Skipping binned SYN4 improvement plot: no populated bins.")
        return

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    valid_idx = np.flatnonzero(valid)
    keep = valid_idx[counts[valid_idx] >= min_bin_count]
    if keep.size == 0:
        print(f"Skipping binned SYN4 improvement plot: no bins with at least {min_bin_count} samples.")
        return

    last_keep = keep[-1] + 1
    plot_mask = valid.copy()
    plot_mask[last_keep:] = False

    ax.fill_between(centers[plot_mask], q25[plot_mask], q75[plot_mask], color="tab:red", alpha=0.2, label="25th-75th percentile")
    ax.plot(centers[plot_mask], medians[plot_mask], color="tab:red", marker="o", linewidth=2, label="Median HiPredict / SYN4")
    ax.axhline(1.0, color="k", linestyle="--", linewidth=1, alpha=0.8, label="Parity")
    ax.set_xscale("log")
    ax.set_xlabel("SYN4 DN NMAE (% of mean |true|)")
    ax.set_ylabel("HiPredict NMAE / SYN4 NMAE")
    ax.set_title("HiPredict Relative Error vs SYN4 Difficulty")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper right")

    out_path = os.path.join(SAVE_DIR, f"EVALS_{cnn_name}_syn4_binned_relative_nmae.png")
    ax.set_ylim(bottom=0.0)
    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved binned SYN4 improvement plot to {out_path}")


def plot_syn4_vs_hipredict_binned(global_val_df, cnn_name, SAVE_DIR, n_bins=14, min_bin_count=10):
    """Reader-friendly restatement of the relative-error plot.

    Same SYN4-difficulty binning, but the y-axis shows the actual HiPredict DN NMAE
    (median + 25th-75th band per bin) against a y=x parity line instead of a ratio.
    Median/band below the diagonal => HiPredict beats SYN4.
    """
    if global_val_df is None or global_val_df.empty:
        print("Skipping binned SYN4 vs HiPredict plot: no global validation rows found.")
        return

    required = ["SYN4 NMAE", "CNN NMAE"]
    missing = [c for c in required if c not in global_val_df.columns]
    if missing:
        print(f"Skipping binned SYN4 vs HiPredict plot: missing columns {missing}")
        return

    df = global_val_df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    df = df[(df["SYN4 NMAE"] > 0) & (df["CNN NMAE"] > 0)].copy()
    if df.empty:
        print("Skipping binned SYN4 vs HiPredict plot: no positive finite NMAE rows found.")
        return

    x = df["SYN4 NMAE"].to_numpy(dtype=np.float64)
    y = df["CNN NMAE"].to_numpy(dtype=np.float64)

    bin_edges = np.logspace(np.log10(x.min()), np.log10(x.max()), n_bins + 1)
    bin_ids = np.digitize(x, bin_edges, right=False) - 1
    bin_ids = np.clip(bin_ids, 0, n_bins - 1)

    centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    medians = np.full(n_bins, np.nan, dtype=np.float64)
    q25 = np.full(n_bins, np.nan, dtype=np.float64)
    q75 = np.full(n_bins, np.nan, dtype=np.float64)
    counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        vals = y[bin_ids == i]
        if vals.size == 0:
            continue
        medians[i] = np.nanmedian(vals)
        q25[i] = np.nanpercentile(vals, 25)
        q75[i] = np.nanpercentile(vals, 75)
        counts[i] = vals.size

    valid = counts > 0
    if not np.any(valid):
        print("Skipping binned SYN4 vs HiPredict plot: no populated bins.")
        return

    valid_idx = np.flatnonzero(valid)
    keep = valid_idx[counts[valid_idx] >= min_bin_count]
    if keep.size == 0:
        print(f"Skipping binned SYN4 vs HiPredict plot: no bins with at least {min_bin_count} samples.")
        return

    last_keep = keep[-1] + 1
    plot_mask = valid.copy()
    plot_mask[last_keep:] = False

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)

    # y=x parity line spanning the plotted SYN4 range
    kept_bins = np.flatnonzero(plot_mask)
    line_lo = bin_edges[kept_bins[0]]
    line_hi = bin_edges[kept_bins[-1] + 1]
    ax.plot([line_lo, line_hi], [line_lo, line_hi], color="k", linestyle="--", linewidth=1, alpha=0.8, label="Parity (y = x)")

    ax.fill_between(centers[plot_mask], q25[plot_mask], q75[plot_mask], color="tab:red", alpha=0.2, label="25th-75th percentile")
    ax.plot(centers[plot_mask], medians[plot_mask], color="tab:red", marker="o", linewidth=2, label="Median HiPredict DN NMAE")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SYN4 DN NMAE (% of mean |true|)", fontsize=14)
    ax.set_ylabel("HiPredict DN NMAE (% of mean |true|)", fontsize=14)
    ax.tick_params(axis="both", which="both", labelsize=12)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc="upper left", fontsize=12)

    # Region cues, pinned to the parity line at the top-right so "better" is unambiguous:
    # anything above the dashed line is a win for SYN4, anything below is a win for HiPredict.
    log_lo, log_hi = np.log10(line_lo), np.log10(line_hi)
    span = log_hi - log_lo
    anchor = 10 ** (log_lo + 0.99 * span)
    gap = 0.015 * span  # vertical clearance from the line, in decades
    ax.text(
        anchor, 10 ** (np.log10(anchor) + gap), "SYN4 better",
        ha="right", va="bottom", rotation=45, rotation_mode="anchor", transform_rotates_text=True,
        fontsize=12, color="dimgray", style="italic",
    )
    ax.text(
        anchor, 10 ** (np.log10(anchor) - gap), "HiPredict better",
        ha="right", va="top", rotation=45, rotation_mode="anchor", transform_rotates_text=True,
        fontsize=12, color="dimgray", style="italic",
    )

    # Spell out the proxy the x-axis encodes: high SYN4 error => colourful, complex scenes.
    ax.annotate(
        "", xy=(1.0, -0.135), xytext=(0.0, -0.135),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="dimgray", linewidth=1.2, shrinkA=0, shrinkB=0),
        annotation_clip=False,
    )
    ax.text(
        0.5, -0.165, "increasing image complexity", transform=ax.transAxes,
        ha="center", va="top", fontsize=12, color="dimgray", style="italic",
    )

    ax.xaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.xaxis.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    out_path = os.path.join(SAVE_DIR, f"EVALS_{cnn_name}_syn4_vs_hipredict_binned_nmae.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved binned SYN4 vs HiPredict plot to {out_path}")


def plot_syn4_vs_hipredict_hist2d(global_val_df, cnn_name, SAVE_DIR, n_bins=45):
    """2D-histogram restatement of the SYN4 vs HiPredict DN NMAE comparison.

    Same axes as the hexbin / binned plots (x = SYN4 DN NMAE, y = HiPredict DN NMAE),
    rendered as a log-log density heatmap with a y=x parity line. Mass below the
    diagonal => HiPredict beats SYN4.
    """
    if global_val_df is None or global_val_df.empty:
        print("Skipping SYN4 vs HiPredict 2D histogram: no global validation rows found.")
        return

    required = ["SYN4 NMAE", "CNN NMAE"]
    missing = [c for c in required if c not in global_val_df.columns]
    if missing:
        print(f"Skipping SYN4 vs HiPredict 2D histogram: missing columns {missing}")
        return

    df = global_val_df.replace([np.inf, -np.inf], np.nan).dropna(subset=required).copy()
    df = df[(df["SYN4 NMAE"] > 0) & (df["CNN NMAE"] > 0)].copy()
    if df.empty:
        print("Skipping SYN4 vs HiPredict 2D histogram: no positive finite NMAE rows found.")
        return

    x = df["SYN4 NMAE"].to_numpy(dtype=np.float64)
    y = df["CNN NMAE"].to_numpy(dtype=np.float64)

    # Shared log-spaced edges so both axes match and the y=x diagonal is meaningful
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    edges = np.logspace(np.log10(lo), np.log10(hi), n_bins + 1)

    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    _, _, _, mesh = ax.hist2d(x, y, bins=[edges, edges], norm=LogNorm(), cmap="viridis")
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Validation image count")

    ax.plot([lo, hi], [lo, hi], color="k", linestyle="--", linewidth=1, alpha=0.8, label="Parity (y = x)")

    # Region cues for readers
    ax.text(0.97, 0.03, "HiPredict better", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="white", style="italic")
    ax.text(0.03, 0.97, "SYN4 better", transform=ax.transAxes, ha="left", va="top", fontsize=9, color="white", style="italic")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("SYN4 DN NMAE (% of mean |true|)")
    ax.set_ylabel("HiPredict DN NMAE (% of mean |true|)")
    ax.set_title("HiPredict vs SYN4 DN NMAE Density Across Validation Images")
    ax.legend(loc="upper left")

    out_path = os.path.join(SAVE_DIR, f"EVALS_{cnn_name}_syn4_vs_hipredict_hist2d.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved SYN4 vs HiPredict 2D histogram to {out_path}")


def plot_evals(
    all_abs_err_norm_cnn,
    all_abs_err_norm_syn4,
    evals,
    eval_cats,
    sampled_base_pixels,
    sampled_pred_pixels,
    sampled_true_pixels,
    pred_mean_perimg,
    true_mean_perimg,
    cnn_name,
    SAVE_DIR,
    corr_vals,
):

    print("\n" + "=" * 80)
    print("PLOTS")
    print("=" * 80)

    # === Figure 1: CDF of absolute error (normalized by image DR) ===
    rng = np.random.default_rng(42)

    def sample_concat(chunks, cap=1_000_000, dtype=np.float32):
        if isinstance(chunks, np.ndarray):
            xs = chunks.astype(dtype, copy=False).ravel()
        else:
            arrs = [np.asarray(c, dtype=dtype).ravel() for c in chunks if np.asarray(c).size > 0]
            xs = np.concatenate(arrs) if arrs else np.array([], dtype=dtype)

        if xs.size > cap:
            idx = rng.choice(xs.size, size=cap, replace=False)
            xs = xs[idx]
        return xs

    assert all_abs_err_norm_cnn is not None and len(all_abs_err_norm_cnn) > 0, "No CNN error data available for CDF plot"
    err_cnn = sample_concat(all_abs_err_norm_cnn)  # in fraction of DR
    err_base = sample_concat(all_abs_err_norm_syn4)
    assert err_cnn.size > 0, "No CNN ECDF values after concatenation"
    assert err_base.size > 0, "No SYN4 ECDF values after concatenation"

    x_cnn, y_cnn = ecdf(100.0 * err_cnn)   # convert to %DR on x-axis
    x_bas, y_bas = ecdf(100.0 * err_base)

    # x val at percentile
    percentile = 99.
    xmax = max(np.percentile(x_cnn, percentile), np.percentile(x_bas, percentile), 1.0)
    print(f"\nCDF plot x-axis `limit` set to {xmax:.2f}% of dynamic range ({percentile}th percentile of absolute error)", flush=True)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    ax.plot(x_cnn, y_cnn, label="HiPredict")
    ax.plot(x_bas, y_bas, label="SYN4")
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(1))   # Minor axis locators
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax.set_xlim(left=0, right=xmax)
    ax.set_ylim(bottom=0, top=1.02)
    ax.set_xlabel("Absolute error (% of dynamic range)")
    ax.set_ylabel("Fraction of pixels (ECDF)")
    ax.set_title(f"CDF of normalised absolute error (showing top {percentile}%)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(loc='lower right')
    cdf_fig_path = os.path.join(SAVE_DIR, f"EVALS_{cnn_name}_cdf_abs_error.png")
    plt.savefig(cdf_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved CDF figure to {cdf_fig_path}")

    # === Figure 2: NMAE dumbbell by category ===
    # Build arrays and sort by improvement (baseline - cnn)
    eval_db_plot(eval_cats, evals, cnn_name, SAVE_DIR)

    # === Figure 3: Hexbin of mean pixel value per image (CNN vs True) ===
    # Clip perimg arrs to 1st and 99th percentile to avoid outliers dominating the plot
    pred_mean_perimg = np.clip(pred_mean_perimg, np.percentile(pred_mean_perimg, 1), np.percentile(pred_mean_perimg, 99))
    true_mean_perimg = np.clip(true_mean_perimg, np.percentile(true_mean_perimg, 1), np.percentile(true_mean_perimg, 99))
    plot_hexbin(
        np.asarray(pred_mean_perimg),
        np.asarray(true_mean_perimg),
        a_name="Mean predicted image value (normalized)",
        b_name="Mean true image value (normalized)",
        fname=os.path.join(SAVE_DIR, "hexbin_perimg.png"),
        gridsize=175,
        sample=None,
        xyline=True,
        log_counts=True,
        scale='Array',
        title="Array-level hexbin in normalized space: mean predicted vs true",
    )

    # === Figure 3b: Hexbin of sampled pixels (CNN vs True) ===
    sampled_pred_pixels = np.asarray(sampled_pred_pixels, dtype=np.float32)
    sampled_true_pixels = np.asarray(sampled_true_pixels, dtype=np.float32)
    if sampled_pred_pixels.size > 0 and sampled_true_pixels.size > 0:
        plot_hexbin(
            sampled_pred_pixels,
            sampled_true_pixels,
            a_name="Predicted pixel value (DN)",
            b_name="True pixel value (DN)",
            fname=os.path.join(SAVE_DIR, "hexbin_sampled_pixels_cnn.png"),
            gridsize=175,
            sample=None,
            xyline=True,
            log_counts=True,
            scale='Pixel',
            title="Pixel-level hexbin in DN space: sampled predicted vs true",
        )

    # === Figure 3c: Hexbin of sampled pixels (SYN4 vs True) ===
    sampled_base_pixels = np.asarray(sampled_base_pixels, dtype=np.float32)
    sampled_base_pixels = np.clip(sampled_base_pixels, np.percentile(sampled_base_pixels, 0), np.max(sampled_pred_pixels))  # clip to max of pred to avoid extreme outliers dominating the plot
    if sampled_base_pixels.size > 0 and sampled_true_pixels.size > 0:
        plot_hexbin(
            sampled_base_pixels,
            sampled_true_pixels,
            a_name="SYN4 pixel value (DN)",
            b_name="True pixel value (DN)",
            fname=os.path.join(SAVE_DIR, "hexbin_sampled_pixels_syn4.png"),
            gridsize=175,
            sample=None,
            xyline=True,
            log_counts=True,
            scale='Pixel',
            title="Pixel-level hexbin in DN space: sampled SYN4 vs true",
        )

    # === Figure 4: SHAP summary plot for metadata correlations ===
    # This plot shows feature importance AND the direction of the effect
    plt.figure(figsize=(12, 8))
    shap_values = np.asarray(corr_vals['shap_values'])
    shap_clip = np.nanpercentile(np.abs(shap_values), 99.5)
    shap_values_plot = np.clip(shap_values, -shap_clip, shap_clip)
    print(f"Clipping SHAP plot values to +/-{shap_clip:.6f} (99.5th percentile absolute SHAP value).")
    shap.summary_plot(shap_values_plot, corr_vals['shap_x'], show=False)
    plt.title("Impact of Metadata on CNN Loss (SHAP Values)", fontsize=14)
    plt.tight_layout()
    shap_path = os.path.join(SAVE_DIR, f"EVALS_{cnn_name}_shap.png")
    plt.savefig(shap_path, dpi=300, bbox_inches="tight")
    plt.savefig(shap_path, dpi=300)
    plt.close()
    print(f"SHAP Analysis complete. Plot saved to: {shap_path}")

    # === Figure 4b: same plot restricted to the top 6 features ===
    # The full plot compresses every feature into one axis; this one gives the
    # fields that actually matter enough vertical room to be read, and pairs
    # with the per-field NMAE histograms (eval_meta_bins.py) covering the same 6
    shap.summary_plot(shap_values_plot, corr_vals['shap_x'], max_display=6, show=False)
    plt.title("Impact of Metadata on CNN Loss (SHAP Values) - top 6 features", fontsize=14)
    plt.tight_layout()
    shap_top_path = os.path.join(SAVE_DIR, f"EVALS_{cnn_name}_shap_top6.png")
    plt.savefig(shap_top_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Top-6 SHAP plot saved to: {shap_top_path}")
    """READING SUMMARY GRAPH:
    - Vertical ranking:     Features at the top are most influential on loss
    - Colour:               Value of metadata (red = high, blue = low)
    - Horizontal axis:      Positive: meta value increased the loss, Negative: meta value decreased the loss
    """         # noqa
