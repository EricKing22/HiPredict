"""Hexbin plotting shared by the retained evaluation figures."""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

EPS = 5e-4


def plot_hexbin(a, b, a_name, b_name, fname, gridsize=150, sample=None, xyline=True, log_counts=True, scale='Pixel', positive_only=True, title=None, xscale='linear', yscale='linear', match_axes=False, bestfit=False):
    rng = np.random.default_rng(42)

    if positive_only:
        mask = (a > EPS) & (b > EPS) & np.isfinite(a) & np.isfinite(b)
    else:
        mask = np.isfinite(a) & np.isfinite(b)

    if not np.any(mask):
        raise ValueError("No valid pixel pairs found after applying EPS filter.")

    x = a[mask].ravel()
    y = b[mask].ravel()

    if sample is not None and x.size > sample:
        idx = rng.choice(x.size, size=sample, replace=False)
        x = x[idx]
        y = y[idx]

    x_lo = float(x.min())
    x_hi = float(x.max())
    y_lo = float(y.min())
    y_hi = float(y.max())

    x_pad = 0.05 * max(x_hi - x_lo, 1e-9)
    y_pad = 0.05 * max(y_hi - y_lo, 1e-9)

    if positive_only:
        x_lims = (max(0.0, x_lo - x_pad), x_hi + x_pad)
        y_lims = (max(0.0, y_lo - y_pad), y_hi + y_pad)
    else:
        x_lims = (x_lo - x_pad, x_hi + x_pad)
        y_lims = (y_lo - y_pad, y_hi + y_pad)

    if match_axes:
        y_lims = x_lims

    extent = (x_lims[0], x_lims[1], y_lims[0], y_lims[1])
    n = x.size

    fig, ax = plt.subplots(figsize=(8, 6))
    hexbin_kwargs = dict(
        x=x,
        y=y,
        gridsize=gridsize,
        mincnt=1,
        cmap="Reds",
        bins="log" if log_counts else None,
        xscale=xscale,
        yscale=yscale,
    )
    if xscale == 'linear' and yscale == 'linear':
        hexbin_kwargs["extent"] = extent
    ax.hexbin(**hexbin_kwargs)

    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    if xyline:
        lo = max(x_lims[0], y_lims[0])
        hi = min(x_lims[1], y_lims[1])
        ax.plot((lo, hi), (lo, hi), "k--", linewidth=1, alpha=0.8)

    legend_handles = [Line2D([], [], linestyle='none', label=f"N={n}")]

    if bestfit:
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.array([x_lims[0], x_lims[1]])
        ax.plot(x_line, slope * x_line + intercept, "k--", linewidth=1.5, alpha=0.9)
        sign = "+" if intercept >= 0 else "-"
        eq_label = f"y = {slope:.3f}x {sign} {abs(intercept):.4f}"
        legend_handles.append(Line2D([], [], color="k", linestyle="--", label=eq_label))

    ax.legend(
        handles=legend_handles,
        handlelength=1.5 if bestfit else 0,
        frameon=True, loc="best"
    )

    ax.set_xlim(x_lims)
    ax.set_ylim(y_lims)
    if positive_only:
        ax.set_ylim(bottom=0.0)
    ax.set_xlabel(a_name)
    ax.set_ylabel(b_name)
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"{scale}-level hexbin: {a_name} vs {b_name}")
    ax.grid(True, alpha=0.3)
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {a_name} vs {b_name} hexbin plot to {fname}")
