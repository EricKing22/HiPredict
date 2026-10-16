from __future__ import annotations

from hipredict.data.normalization import MAD_TO_NORMAL_STD


import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import subprocess
import tempfile
from pathlib import Path

from hipredict.paths import PACKAGE_ROOT


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def create_icy_plot(true, pred, syn4, ir, bg, savedir, obs_ids=None, use_center_pixel=False, ref_offset=100, plot_reference=True):
    """
    true, pred, syn4, ir, bg: arrays of shape (B, 256, 256)

    - Different colours for different models
    - Different shapes for different images
    - BG/RED = BG / RED
    - WATER-ICE = mean(BG, RED) / IR
    """
    true = _to_numpy(true)
    pred = _to_numpy(pred)
    syn4 = _to_numpy(syn4)
    ir = _to_numpy(ir)
    bg = _to_numpy(bg)

    if true.shape != pred.shape or true.shape != syn4.shape or true.shape != ir.shape or true.shape != bg.shape:
        raise ValueError(
            "All inputs must have the same shape (B, 256, 256). Got: "
            f"true={true.shape}, pred={pred.shape}, syn4={syn4.shape}, ir={ir.shape}, bg={bg.shape}"
        )

    os.makedirs(savedir, exist_ok=True)

    eps = 1e-8
    def _clamp_center(y, x, h, w):
        y = int(np.clip(y, 0, h - 1))
        x = int(np.clip(x, 0, w - 1))
        return y, x

    cycle = plt.rcParams.get("axes.prop_cycle", None)
    colors = None
    if cycle is not None:
        colors = cycle.by_key().get("color", None)
    if not colors:
        colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]

    model_names = ["RED4", "HiPredict", "SYN4"]
    model_arrays = {"RED4": true, "HiPredict": pred, "SYN4": syn4}
    model_colors = {name: colors[i % len(colors)] for i, name in enumerate(model_names)}

    marker_list = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "*", "h", "8"]

    fig, ax = plt.subplots(figsize=(10,5))
    ax.set_xlabel("WATER-ICE")
    ax.set_ylabel("BG/RED")

    ax.set_xlim(0.4, 1.8)
    ax.set_ylim(0.2, 1.2)

    ax.axvspan(0.8, 1.1, color="blue", alpha=0.08, zorder=0)
    ax.axhspan(0.5, 0.8, color="red", alpha=0.08, zorder=0)

    ax.text(0.02, 0.02, "Ice-poor", transform=ax.transAxes, ha="left", va="bottom", weight="bold")
    ax.text(0.98, 0.98, "Ice-rich", transform=ax.transAxes, ha="right", va="top", weight="bold")


    B = true.shape[0]
    if obs_ids is not None and len(obs_ids) != B:
        raise ValueError(f"obs_ids must have length {B}, got {len(obs_ids)}")

    used = []

    for b in range(B):
        marker_b = marker_list[b % len(marker_list)]
        obs_label = obs_ids[b] if obs_ids is not None else f"img {b}"

        ir_b = ir[b]
        bg_b = bg[b]
        r_true = true[b]

        h, w = r_true.shape
        if use_center_pixel:
            y, x = h // 2, w // 2
        else:
            wi_true = ((bg_b + r_true) / 2.0) / (ir_b + eps)
            med = np.nanmedian(wi_true)
            mad = np.nanmedian(np.abs(wi_true - med))
            z = np.abs((wi_true - med) / (MAD_TO_NORMAL_STD * mad + eps))

            flat_idx = int(np.nanargmax(z))
            y, x = np.unravel_index(flat_idx, wi_true.shape)

        y_ref, x_ref = _clamp_center(y + ref_offset, x + ref_offset, h, w)

        wi_points = {}
        wi_points_ref = {}

        for name in model_names:
            r = model_arrays[name][b]
            wi_points[name] = ((bg_b[y, x] + r[y, x]) / 2.0) / (ir_b[y, x] + eps)

            if plot_reference:
                wi_points_ref[name] = ((bg_b[y_ref, x_ref] + r[y_ref, x_ref]) / 2.0) / (ir_b[y_ref, x_ref] + eps)

        if np.any(np.array(list(wi_points.values())) < -50.0):
            continue
        if plot_reference and np.any(np.array(list(wi_points_ref.values())) < -50.0):
            continue

        used.append((b, marker_b, obs_label))

        for name in model_names:
            r = model_arrays[name][b]

            bg_red = bg_b[y, x] / (r[y, x] + eps)
            ax.scatter(wi_points[name], bg_red, c=model_colors[name], marker=marker_b, zorder=2, alpha=0.9)

            if plot_reference:
                bg_red_ref = bg_b[y_ref, x_ref] / (r[y_ref, x_ref] + eps)
                ax.scatter(
                    wi_points_ref[name],
                    bg_red_ref,
                    facecolors="none",
                    edgecolors=model_colors[name],
                    marker=marker_b,
                    zorder=2,
                    alpha=0.9,
                    linewidths=1.0
                )

    model_handles = [
        Line2D([0], [0], marker="o", linestyle="None", color=model_colors[name], label=name)
        for name in model_names
    ]
    legend1 = ax.legend(
        handles=model_handles,
        title="Model",
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=9,
        title_fontsize=10
    )
    ax.add_artist(legend1)

    if len(used) <= 12:
        image_handles = []
        for _, m, lab in used:
            # main (icy) point: filled marker
            image_handles.append(
                Line2D([0], [0], marker=m, linestyle="None", color="black", label=f"{lab} (icy)")
            )
            # reference point: hollow marker
            image_handles.append(
                Line2D(
                    [0], [0],
                    marker=m,
                    linestyle="None",
                    markerfacecolor="none",
                    markeredgecolor="black",
                    color="black",
                    label=f"{lab} (ref)"
                )
            )

        ax.legend(
            handles=image_handles,
            title="Observation",
            loc="lower left",
            bbox_to_anchor=(1.02, 0.0),
            borderaxespad=0.0,
            fontsize=9,
            title_fontsize=10
        )

    outpath = os.path.join(savedir, "icy_scatter.png")
    fig.tight_layout(rect=[0.0, 0.0, 0.95, 1.0])              # type: ignore
    fig.savefig(outpath, dpi=200)
    plt.close(fig)
    print(f"Saved icy scatter plot to: {outpath}")

    return outpath


def save_domain_imgs(true, pred, syn4, ir, bg, savedir, labels=None, file_ids=None, annotate_center=False, box_half_size=3, save_num_in=6, ref_offset=100):
    """
    Saves quicklook mosaics for the first n samples.

    Draws a small red box at the image center.
    """
    from matplotlib.patches import Rectangle

    true = _to_numpy(true)
    pred = _to_numpy(pred)
    syn4 = _to_numpy(syn4)
    ir = _to_numpy(ir)
    bg = _to_numpy(bg)

    if true.shape != pred.shape or true.shape != syn4.shape or true.shape != ir.shape or true.shape != bg.shape:
        raise ValueError(
            "All inputs must have the same shape (B, 256, 256). Got: "
            f"true={true.shape}, pred={pred.shape}, syn4={syn4.shape}, ir={ir.shape}, bg={bg.shape}"
        )

    B = true.shape[0]
    if labels is not None and len(labels) != B:
        raise ValueError(f"labels must have length {B}, got {len(labels)}")
    if file_ids is not None and len(file_ids) != B:
        raise ValueError(f"file_ids must have length {B}, got {len(file_ids)}")

    os.makedirs(savedir, exist_ok=True)
    eps = 1e-8
    n_save = min(save_num_in, B)

    def _norm01(img):
        lo, hi = np.nanpercentile(img, [2, 98])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = np.nanmin(img)
            hi = np.nanmax(img)
        return np.clip((img - lo) / (hi - lo + eps), 0.0, 1.0)

    def _clamp_center(y, x, h, w, half):
        y = int(np.clip(y, half, h - 1 - half))
        x = int(np.clip(x, half, w - 1 - half))
        return y, x

    for b in range(n_save):
        ir_b = ir[b]
        bg_b = bg[b]
        r_true = true[b]

        irb_true = np.stack([_norm01(ir_b), _norm01(r_true), _norm01(bg_b)], axis=-1)
        irb_pred = np.stack([_norm01(ir_b), _norm01(pred[b]), _norm01(bg_b)], axis=-1)
        irb_syn4 = np.stack([_norm01(ir_b), _norm01(syn4[b]), _norm01(bg_b)], axis=-1)

        panels = [
            ("RED4", r_true),   ("RED4 (IRB)", irb_true),
            ("HiPredict ", pred[b]),   ("HiPredict (IRB)", irb_pred),
            ("Syn4", syn4[b]),     ("Syn4 (IRB)", irb_syn4)
        ]

        fig, axes = plt.subplots(3,2, figsize=(8, 10))
        axes_flat = axes.flatten()      # type: ignore

        for ax, (title, img) in zip(axes_flat, panels):
            if img.ndim == 3:
                ax.imshow(img)
            else:
                lo, hi = np.nanpercentile(img, [1, 99])
                if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
                    lo = np.nanmin(img)
                    hi = np.nanmax(img)
                ax.imshow(img, cmap="gray", vmin=lo, vmax=hi)

            ax.set_title(title, fontsize=10)
            ax.set_axis_off()

            if annotate_center:
                h, w = img.shape[:2]
                curr_y, curr_x = h // 2, w // 2
                curr_y, curr_x = _clamp_center(curr_y, curr_x, h, w, box_half_size)

                rect = Rectangle(
                    (curr_x - box_half_size - 0.5, curr_y - box_half_size - 0.5),
                    2 * box_half_size + 1, 2 * box_half_size + 1,
                    linewidth=1.5, edgecolor="red", facecolor="none", alpha=1.0
                )
                ax.add_patch(rect)

                ref_y, ref_x = _clamp_center(curr_y + ref_offset, curr_x + ref_offset, h, w, box_half_size)
                ref_rect = Rectangle(
                    (ref_x - box_half_size - 0.5, ref_y - box_half_size - 0.5),
                    2 * box_half_size + 1, 2 * box_half_size + 1,
                    linewidth=1.5, edgecolor="black", facecolor="none", alpha=1.0
                )
                ax.add_patch(ref_rect)

        label = labels[b] if labels is not None else f"img {b}"

        fig.suptitle(label, fontsize=11)

        fid = file_ids[b] if file_ids is not None else f"img_{b}"
        outpath = os.path.join(savedir, f"{fid}_quicklook.png")
        fig.tight_layout()
        fig.savefig(outpath, dpi=200)
        plt.close(fig)


def save_domain_diffs(true, pred, syn4, ir, bg, savedir, labels=None, file_ids=None, annotate_center=False, box_half_size=3, save_num_in=6, ref_offset=100):
    """
    Saves quicklook mosaics for the first n samples.

    Layout: 3 rows x 2 cols
      - Left col:  true vs pred (HiPredict)
      - Right col: true vs syn4
    Rows show residuals smoothed over increasing window sizes:
      1) 1x1 (per-pixel)
      2) 3x3
      3) 5x5

    Draws a small red box at the image center and a black ref box offset by ref_offset (optional).
    """
    true = _to_numpy(true)
    pred = _to_numpy(pred)
    syn4 = _to_numpy(syn4)
    ir = _to_numpy(ir)
    bg = _to_numpy(bg)

    if true.shape != pred.shape or true.shape != syn4.shape or true.shape != ir.shape or true.shape != bg.shape:
        raise ValueError(
            "All inputs must have the same shape (B, 256, 256). Got: "
            f"true={true.shape}, pred={pred.shape}, syn4={syn4.shape}, ir={ir.shape}, bg={bg.shape}"
        )

    B = true.shape[0]
    if labels is not None and len(labels) != B:
        raise ValueError(f"labels must have length {B}, got {len(labels)}")
    if file_ids is not None and len(file_ids) != B:
        raise ValueError(f"file_ids must have length {B}, got {len(file_ids)}")

    os.makedirs(savedir, exist_ok=True)
    n_save = min(save_num_in, B)

    def _clamp_center(y, x, h, w, half):
        y = int(np.clip(y, half, h - 1 - half))
        x = int(np.clip(x, half, w - 1 - half))
        return y, x

    def _mean_filter_2d(x, k):
        if k == 1:
            return x

        if k % 2 == 0:
            raise ValueError(f"k must be odd, got {k}")

        pad = k // 2
        x_pad = np.pad(x, ((pad, pad), (pad, pad)), mode="reflect")

        c = np.cumsum(np.cumsum(x_pad, axis=0), axis=1)
        c = np.pad(c, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)

        s = c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]
        return s / float(k * k)

    def _sym_vlim(img, pct=99.0):
        v = np.nanpercentile(np.abs(img), pct)
        if not np.isfinite(v) or v <= 0:
            v = np.nanmax(np.abs(img))
        if not np.isfinite(v) or v <= 0:
            v = 1.0
        return float(v)

    windows = [1, 3, 5]

    for b in range(n_save):
        r_true = true[b]
        r_pred = pred[b]
        r_syn4 = syn4[b]

        fig, axes = plt.subplots(3, 2, figsize=(10, 10))

        # Pre-calculate global vlim for consistency across all 6 subplots
        all_res = [(_mean_filter_2d(r_pred - r_true, k), _mean_filter_2d(r_syn4 - r_true, k)) for k in windows]
        v_max = _sym_vlim(np.array(all_res), pct=99.0)

        for i, k in enumerate(windows):
            res_pred, res_syn4 = all_res[i]


            ax_l = axes[i, 0]      # type: ignore
            ax_r = axes[i, 1]      # type: ignore

            ax_l.imshow(res_pred, cmap="magma", vmin=-v_max, vmax=v_max)
            im_r = ax_r.imshow(res_syn4, cmap="magma", vmin=-v_max, vmax=v_max)

            fig.colorbar(im_r, ax=ax_r, fraction=0.046, pad=0.04)
            ax_l.set_title(f"HiPredict - True ({k}x{k})", fontsize=10)
            ax_r.set_title(f"Syn4 - True ({k}x{k})", fontsize=10)

            ax_l.set_axis_off()
            ax_r.set_axis_off()

            if annotate_center:
                h, w = r_true.shape
                curr_y, curr_x = h // 2, w // 2
                curr_y, curr_x = _clamp_center(curr_y, curr_x, h, w, box_half_size)

                for ax in (ax_l, ax_r):
                    rect = Rectangle(
                        (curr_x - box_half_size - 0.5, curr_y - box_half_size - 0.5),
                        2 * box_half_size + 1,
                        2 * box_half_size + 1,
                        linewidth=1.5,
                        edgecolor="red",
                        facecolor="none",
                        alpha=1.0
                    )
                    ax.add_patch(rect)

                    ref_y, ref_x = _clamp_center(curr_y + ref_offset, curr_x + ref_offset, h, w, box_half_size)
                    ref_rect = Rectangle(
                        (ref_x - box_half_size - 0.5, ref_y - box_half_size - 0.5),
                        2 * box_half_size + 1,
                        2 * box_half_size + 1,
                        linewidth=1.5,
                        edgecolor="black",
                        facecolor="none",
                        alpha=1.0
                    )
                    ax.add_patch(ref_rect)

        label = labels[b] if labels is not None else f"img {b}"
        fig.suptitle(label, fontsize=11)

        fid = file_ids[b] if file_ids is not None else f"img_{b}"
        outpath = os.path.join(savedir, f"{fid}_diffs.png")

        fig.tight_layout()
        fig.savefig(outpath, dpi=200)
        plt.close(fig)


def save_domain_features(true, pred, syn4, ir, bg, savedir, labels=None, file_ids=None, annotate_center=False, box_half_size=3, save_num_in=6, ref_offset=100):
    """
    Saves a single feature-focused mosaic per sample (first n samples).

    Layout: 3 rows x 4 cols
      Rows: residual smoothing window size (1x1, 3x3, 5x5)
      Cols:
        1) Grad mask + HiPredict
        2) Grad mask + Syn4
        3) Anom mask + HiPredict
        4) Anom mask + Syn4

    Visualization:
      - Background is true RED4 in grayscale for context
      - Residual overlay is shown only where the feature mask is true
      - A thin mask contour is drawn so "good performance" doesn't look like "no mask"
      - Optional center/ref boxes (annotate_center)

    Also prints masked MAE and bias in each panel title.
    """
    true = _to_numpy(true)
    pred = _to_numpy(pred)
    syn4 = _to_numpy(syn4)
    ir = _to_numpy(ir)
    bg = _to_numpy(bg)

    if true.shape != pred.shape or true.shape != syn4.shape or true.shape != ir.shape or true.shape != bg.shape:
        raise ValueError(
            "All inputs must have the same shape (B, 256, 256). Got: "
            f"true={true.shape}, pred={pred.shape}, syn4={syn4.shape}, ir={ir.shape}, bg={bg.shape}"
        )

    B = true.shape[0]
    if labels is not None and len(labels) != B:
        raise ValueError(f"labels must have length {B}, got {len(labels)}")
    if file_ids is not None and len(file_ids) != B:
        raise ValueError(f"file_ids must have length {B}, got {len(file_ids)}")

    os.makedirs(savedir, exist_ok=True)
    n_save = min(save_num_in, B)

    windows = [1, 3, 5]

    morph_k = 3

    # For edge
    edge_smooth_k = 3          # light denoise before gradients
    edge_roi_half = 64         # thresholds computed in central ROI
    edge_high_pct = 96.0       # high threshold = percentile in ROI
    edge_low_frac = 0.4        # low threshold = frac * high
    edge_band_k = 3            # dilate edges to a band region


    def _clamp_center(y, x, h, w, half):
        y = int(np.clip(y, half, h - 1 - half))
        x = int(np.clip(x, half, w - 1 - half))
        return y, x

    def _norm_gray_limits(img):
        lo, hi = np.nanpercentile(img, [1, 99])
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = np.nanmin(img)
            hi = np.nanmax(img)
        if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
            lo = 0.0
            hi = 1.0
        return float(lo), float(hi)

    def _mean_filter_2d(x, k):
        if k == 1:
            return x

        if k % 2 == 0:
            raise ValueError(f"k must be odd, got {k}")

        pad = k // 2
        xp = np.pad(x, ((pad, pad), (pad, pad)), mode="reflect")

        I = np.zeros((xp.shape[0] + 1, xp.shape[1] + 1), dtype=xp.dtype)
        I[1:, 1:] = xp.cumsum(axis=0).cumsum(axis=1)

        s = I[k:, k:] - I[:-k, k:] - I[k:, :-k] + I[:-k, :-k]
        return s / float(k * k)


    def _sobel_grad_mag(x):
        x0 = np.nan_to_num(x, nan=np.nanmedian(x))
        xp = np.pad(x0, ((1, 1), (1, 1)), mode="reflect")

        tl = xp[:-2, :-2]
        tc = xp[:-2, 1:-1]
        tr = xp[:-2, 2:]

        ml = xp[1:-1, :-2]
        mr = xp[1:-1, 2:]

        bl = xp[2:, :-2]
        bc = xp[2:, 1:-1]
        br = xp[2:, 2:]

        gx = (tl + 2.0 * ml + bl) - (tr + 2.0 * mr + br)
        gy = (tl + 2.0 * tc + tr) - (bl + 2.0 * bc + br)
        g = np.hypot(gx, gy)
        return g

    def _binary_dilate(mask, k):
        pad = k // 2
        h, w = mask.shape
        mp = np.pad(mask, ((pad, pad), (pad, pad)), mode="constant", constant_values=False)
        out = np.zeros((h, w), dtype=bool)

        for dy in range(k):
            for dx in range(k):
                out |= mp[dy:dy + h, dx:dx + w]

        return out

    def _binary_erode(mask, k):
        pad = k // 2
        h, w = mask.shape
        mp = np.pad(mask, ((pad, pad), (pad, pad)), mode="constant", constant_values=True)
        out = np.ones((h, w), dtype=bool)

        for dy in range(k):
            for dx in range(k):
                out &= mp[dy:dy + h, dx:dx + w]

        return out

    def _binary_close(mask, k):
        return _binary_erode(_binary_dilate(mask, k), k)

    def _binary_fill_holes(mask):
        h, w = mask.shape
        bg = ~mask
        seen = np.zeros((h, w), dtype=bool)
        stack = []

        for x in range(w):
            if bg[0, x] and not seen[0, x]:
                seen[0, x] = True
                stack.append((0, x))
            if bg[h - 1, x] and not seen[h - 1, x]:
                seen[h - 1, x] = True
                stack.append((h - 1, x))

        for y in range(h):
            if bg[y, 0] and not seen[y, 0]:
                seen[y, 0] = True
                stack.append((y, 0))
            if bg[y, w - 1] and not seen[y, w - 1]:
                seen[y, w - 1] = True
                stack.append((y, w - 1))

        while stack:
            y, x = stack.pop()
            for ny, nx in (
                (y - 1, x), (y + 1, x),
                (y, x - 1), (y, x + 1),
            ):
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if not bg[ny, nx]:
                    continue
                if seen[ny, nx]:
                    continue

                seen[ny, nx] = True
                stack.append((ny, nx))

        holes = bg & ~seen
        return mask | holes


    def _hysteresis_edges(g, high_thr, low_thr):
        strong = g >= high_thr
        weak = g >= low_thr

        h, w = g.shape
        out = np.zeros((h, w), dtype=bool)

        stack = list(map(tuple, np.argwhere(strong)))
        for y, x in stack:
            out[y, x] = True

        while stack:
            y, x = stack.pop()
            for ny, nx in (
                (y - 1, x - 1), (y - 1, x), (y - 1, x + 1),
                (y, x - 1), (y, x + 1),
                (y + 1, x - 1), (y + 1, x), (y + 1, x + 1),
            ):
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if out[ny, nx]:
                    continue
                if not weak[ny, nx]:
                    continue

                out[ny, nx] = True
                stack.append((ny, nx))

        return out

    def _mask_edge(true_img):
        x = np.nan_to_num(true_img, nan=np.nanmedian(true_img))

        if edge_smooth_k > 1:
            x = _mean_filter_2d(x, edge_smooth_k)

        g = _sobel_grad_mag(x)

        h, w = g.shape
        cy, cx = h // 2, w // 2

        y0 = max(0, cy - edge_roi_half)
        y1 = min(h, cy + edge_roi_half + 1)
        x0 = max(0, cx - edge_roi_half)
        x1 = min(w, cx + edge_roi_half + 1)
        groi = g[y0:y1, x0:x1]

        high = np.nanpercentile(groi, edge_high_pct)
        if not np.isfinite(high):
            high = np.nanmax(groi)
        if not np.isfinite(high) or high <= 0:
            return np.zeros((h, w), dtype=bool)

        low = float(edge_low_frac) * float(high)

        edges = _hysteresis_edges(g, high, low)

        edges = _binary_close(edges, morph_k)

        region = _binary_dilate(edges, edge_band_k)
        region = _keep_center_component(region)
        region = _binary_fill_holes(region)

        return region

    def _masked_stats(res, mask):
        n = int(mask.sum())
        if n == 0:
            return np.nan, np.nan, 0.0

        vals = res[mask]
        mae = float(np.nanmean(np.abs(vals)))
        bias = float(np.nanmean(vals))
        cov = float(n) / float(mask.size)
        return mae, bias, cov

    def _row_vmax(masked_residuals):
        vals = []
        for r in masked_residuals:
            if r is None:
                continue
            a = np.abs(r)
            a = a[np.isfinite(a)]
            if a.size > 0:
                vals.append(a)

        if len(vals) == 0:
            return 1.0

        allv = np.concatenate(vals, axis=0)
        vmax = np.nanpercentile(allv, 99.0)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = np.nanmax(allv)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = 1.0
        return float(vmax)
    
    def _nearest_true_pixel(mask, cy, cx):
        coords = np.argwhere(mask)
        if coords.size == 0:
            return None

        dy = coords[:, 0] - cy
        dx = coords[:, 1] - cx
        d2 = dy * dy + dx * dx
        idx = int(np.argmin(d2))
        return int(coords[idx, 0]), int(coords[idx, 1])

    def _flood_fill_component(mask, sy, sx):
        h, w = mask.shape
        out = np.zeros((h, w), dtype=bool)

        stack = [(sy, sx)]
        out[sy, sx] = True

        while stack:
            y, x = stack.pop()

            for ny, nx in (
                (y - 1, x - 1), (y - 1, x), (y - 1, x + 1),
                (y, x - 1), (y, x + 1),
                (y + 1, x - 1), (y + 1, x), (y + 1, x + 1),
            ):
                if ny < 0 or ny >= h or nx < 0 or nx >= w:
                    continue
                if not mask[ny, nx]:
                    continue
                if out[ny, nx]:
                    continue

                out[ny, nx] = True
                stack.append((ny, nx))

        return out

    def _keep_center_component(mask):
        h, w = mask.shape
        cy, cx = h // 2, w // 2

        if not mask.any():
            return mask

        if mask[cy, cx]:
            sy, sx = cy, cx
        else:
            seed = _nearest_true_pixel(mask, cy, cx)
            if seed is None:
                return mask
            sy, sx = seed

        return _flood_fill_component(mask, sy, sx)


    for b in range(n_save):
        r_true = true[b]
        r_pred = pred[b]
        r_syn4 = syn4[b]

        m_edge = _mask_edge(r_true)

        base_lo, base_hi = _norm_gray_limits(r_true)

        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(
            nrows=3,
            ncols=4,
            width_ratios=[1, 1, 1, 0.06],
            wspace=0.03,
            hspace=0.22,
        )

        axes = np.empty((3, 3), dtype=object)
        caxes = np.empty((3,), dtype=object)
        for ii in range(3):
            for jj in range(3):
                axes[ii, jj] = fig.add_subplot(gs[ii, jj])
            caxes[ii] = fig.add_subplot(gs[ii, 3])


        for i, k in enumerate(windows):
            res_pred = _mean_filter_2d(r_pred - r_true, k)
            res_syn4 = _mean_filter_2d(r_syn4 - r_true, k)

            err_pred = np.abs(res_pred)
            err_syn4 = np.abs(res_syn4)

            r_e_pred = np.where(m_edge, err_pred, np.nan)
            r_e_syn4 = np.where(m_edge, err_syn4, np.nan)

            vmax = _row_vmax([r_e_pred, r_e_syn4])

            row_mappable = None

            panels = [
                ("Edge", "HiPredict", m_edge, r_e_pred, res_pred, True),
                ("Edge", "Syn4", m_edge, r_e_syn4, res_syn4, True),
                ("True", "Edge outline", m_edge, None, None, False),
            ]

            for j, (mask_name, model_name, mask, r_masked, res_full, show_residual) in enumerate(panels):
                ax = axes[i, j]

                ax.imshow(r_true, cmap="gray", vmin=base_lo, vmax=base_hi)

                if show_residual:
                    im = ax.imshow(r_masked, cmap="magma", vmin=0., vmax=vmax, alpha=0.85)
                    if row_mappable is None:
                        row_mappable = im

                    mae, bias, cov = _masked_stats(res_full, mask)
                    cov_pct = 100.0 * cov

                    ax.set_title(
                        f"{mask_name} | {model_name} | {k}x{k}\n"
                        f"cov={cov_pct:.1f}%  MAE={mae:.3g}  B={bias:.3g}",
                        fontsize=9,
                        pad=6
                    )
                else:
                    cov = float(mask.sum()) / float(mask.size)
                    cov_pct = 100.0 * cov
                    ax.set_title(
                        f"{mask_name} | {model_name} | {k}x{k}\n"
                        f"cov={cov_pct:.1f}%",
                        fontsize=9,
                        pad=6
                    )


                if mask.any():
                    ax.contour(mask.astype(np.float32), levels=[0.5], colors="white", linewidths=0.6)
                ax.set_axis_off()

                if annotate_center:
                    h, w = r_true.shape
                    curr_y, curr_x = h // 2, w // 2
                    curr_y, curr_x = _clamp_center(curr_y, curr_x, h, w, box_half_size)

                    rect = Rectangle(
                        (curr_x - box_half_size - 0.5, curr_y - box_half_size - 0.5),
                        2 * box_half_size + 1,
                        2 * box_half_size + 1,
                        linewidth=1.5,
                        edgecolor="red",
                        facecolor="none",
                        alpha=1.0
                    )
                    ax.add_patch(rect)

                    ref_y, ref_x = _clamp_center(curr_y + ref_offset, curr_x + ref_offset, h, w, box_half_size)
                    ref_rect = Rectangle(
                        (ref_x - box_half_size - 0.5, ref_y - box_half_size - 0.5),
                        2 * box_half_size + 1,
                        2 * box_half_size + 1,
                        linewidth=1.5,
                        edgecolor="black",
                        facecolor="none",
                        alpha=1.0
                    )
                    ax.add_patch(ref_rect)

            if row_mappable is not None:
                cb = fig.colorbar(row_mappable, cax=caxes[i])
                cb.ax.tick_params(labelsize=8)
                cb.set_label("Residual (model - true)", fontsize=9)
            else:
                caxes[i].set_axis_off()


        label = labels[b] if labels is not None else f"img {b}"
        fig.suptitle(label, fontsize=11)


        fid = file_ids[b] if file_ids is not None else f"img_{b}"
        outpath = os.path.join(savedir, f"{fid}_feature.png")

        fig.subplots_adjust(top=0.93, bottom=0.02, left=0.02, right=0.96)
        fig.savefig(outpath, dpi=200)
        plt.close(fig)


def _require_gdal():
    try:
        from osgeo import gdal
    except ImportError as e:
        raise ImportError(
            "GDAL is required for ISIS cube export. Install system GDAL plus the optional "
            "Python binding, then re-run with the cube export dependencies enabled."
        ) from e
    return gdal


def _valid_mask_from_arr(arr, invalid_values=(-9998,), border=0):
    x = np.asarray(arr)

    if x.ndim == 2:
        valid = np.isfinite(x)
        for v in invalid_values:
            valid &= x != v
    elif x.ndim == 3:
        valid = np.all(np.isfinite(x), axis=-1)
        for v in invalid_values:
            valid &= ~np.any(x == v, axis=-1)
    else:
        raise ValueError(f"Unsupported array shape: {x.shape}")

    if border > 0:
        border = min(border, valid.shape[0] // 2, valid.shape[1] // 2)
        if border > 0:
            valid[:border, :] = False
            valid[-border:, :] = False
            valid[:, :border] = False
            valid[:, -border:] = False
    return valid


def _sample_band_for_stretch(band, valid_mask=None, max_samples=1_000_000):
    h, w = band.shape
    n_px = h * w
    if n_px <= max_samples:
        return band if valid_mask is None else band[valid_mask]

    stride = int(np.ceil(np.sqrt(n_px / max_samples)))
    sampled = band[::stride, ::stride]
    if valid_mask is None:
        return sampled
    return sampled[valid_mask[::stride, ::stride]]


def _percentile_bounds(sample, lower_pct, upper_pct, clip_high_pct=None):
    if sample.size == 0:
        return None, None

    if clip_high_pct is not None:
        clip_hi = float(np.percentile(sample, clip_high_pct))
        sample = sample[sample <= clip_hi]
        if sample.size == 0:
            return None, None

    lo = float(np.percentile(sample, lower_pct))
    hi = float(np.percentile(sample, upper_pct))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(sample))
        hi = float(np.nanmax(sample))

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None, None

    return lo, hi


def _stretch_for_display(arr, dtype, lower_pct=0.1, upper_pct=99.99, invalid_values=(-9998,), border=200):
    if not (
        np.issubdtype(dtype, np.integer)
        or np.issubdtype(dtype, np.floating)
    ):
        raise TypeError(f"Unsupported dtype for stretched TIFF export: {dtype}")

    squeeze = arr.ndim == 2
    x = arr[..., None] if squeeze else np.asarray(arr)
    out = np.zeros(x.shape, dtype=np.float32)

    valid = _valid_mask_from_arr(x, invalid_values=invalid_values, border=0)
    stat_valid = _valid_mask_from_arr(x, invalid_values=invalid_values, border=border)

    if not np.any(stat_valid):
        stat_valid = valid

    if not np.any(valid):
        return out[..., 0] if squeeze else out.astype(dtype)

    for band_idx in range(x.shape[-1]):
        band = x[..., band_idx].astype(np.float32, copy=False)
        finite_sample = _sample_band_for_stretch(band, stat_valid)

        if finite_sample.size == 0:
            continue

        # For RGB browse products, stop a tiny number of very bright icy pixels
        # from dominating the per-band white point, while keeping channels
        # independently stretched as in HiRISE-style enhanced products.
        clip_high_pct = 99.9 if x.shape[-1] == 3 else None
        lo, hi = _percentile_bounds(
            finite_sample,
            lower_pct,
            upper_pct,
            clip_high_pct=clip_high_pct,
        )
        if lo is None or hi is None:
            continue

        scaled = np.zeros_like(band, dtype=np.float32)
        scaled[valid] = np.clip((band[valid] - lo) / (hi - lo), 0.0, 1.0)
        out[..., band_idx] = scaled

    if np.issubdtype(dtype, np.integer):
        out = np.rint(out * np.iinfo(dtype).max).astype(dtype)
    else:
        out = out.astype(dtype)

    return out[..., 0] if squeeze else out


def np_to_tiff(arr, out_path, dtype=np.float32, stretch=False):
    from tifffile import imwrite

    target_dtype = np.dtype(dtype)
    x = np.asarray(arr)
    if x.ndim not in (2, 3):
        raise ValueError(f"Unsupported output_img shape: {x.shape}")

    if stretch:
        x = _stretch_for_display(x, target_dtype)
    else:
        x = x.astype(target_dtype)

    kwargs = {
        "compression": "deflate",
        "metadata": None,
    }
    if x.ndim == 3 and x.shape[-1] == 3:
        kwargs["photometric"] = "rgb"

    imwrite(out_path, x, **kwargs)
    return


def _write_isis3_cube_from_array(arr: np.ndarray, out_cub: Path, gdal_type: int) -> None:
    gdal = _require_gdal()
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array (H, W). Got {arr.shape}")

    h, w = arr.shape
    drv = gdal.GetDriverByName("ISIS3")
    if drv is None:
        raise RuntimeError("GDAL ISIS3 driver not available.")

    ds = drv.Create(str(out_cub), w, h, 1, gdal_type, options=["TILED=YES"])
    if ds is None:
        raise RuntimeError(f"Failed to create ISIS cube: {out_cub}")

    ds.GetRasterBand(1).WriteArray(arr)
    ds.FlushCache()
    ds = None


def np_to_cube(
    pred: np.ndarray,
    reference_cube: str | Path,
    output_cube: str | Path,
    ccd_id: str = "RED4",
    inject_script: str | Path = PACKAGE_ROOT / "inference" / "generate_cube.sh",
) -> Path:
    reference_cube = Path(reference_cube)
    output_cube = Path(output_cube)
    inject_script = Path(inject_script)
    print("np to cube triggered")

    if not reference_cube.is_file():
        raise FileNotFoundError(f"reference_cube not found: {reference_cube}")
    if not inject_script.is_file():
        raise FileNotFoundError(f"inject_script not found: {inject_script}")

    pred = np.asarray(pred)
    print(f"Arr shape: {pred.shape}, dtype: {pred.dtype}")
    if pred.ndim != 2:
        raise ValueError(
            "pred must be 2D (Lines, Samples) for single-band cube export. "
            f"Got {pred.shape}. Multi-band strip export is only supported for NumPy and TIFF outputs."
        )

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        tmp_cub = td / "pred_tmp.cub"

        gdal = _require_gdal()
        arr_to_write = pred.astype(np.float32, copy=False)
        gdal_type = gdal.GDT_Float32

        _write_isis3_cube_from_array(arr_to_write, tmp_cub, gdal_type)

        cmd = [
            "bash",
            str(inject_script),
            str(tmp_cub),
            str(reference_cube),
            str(output_cube),
            str(ccd_id),
        ]

        proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                "inject script failed\n"
                f"cmd: {cmd}\n"
                f"returncode: {proc.returncode}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )

    if not output_cube.exists():
        raise FileNotFoundError(f"Output cube not created: {output_cube}")
    else:
        print(f"Successfully created output cube: {output_cube}")

    return output_cube
