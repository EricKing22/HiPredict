import os
import matplotlib.pyplot as plt
import pandas as pd
try:
    import wandb
except ImportError:
    wandb = None

import numpy as np
import torch
import yaml
import re
import torch.nn as nn
import torch.nn.functional as F
import random

from hipredict.predict.models import SYN4
from hipredict.paths import resolve_path, files_dir, models_dir


def _wandb_ready():
    return wandb is not None and getattr(wandb, "run", None) is not None


def load_cfg(config_file, overrides=None):
    with open(config_file) as handle:
        cfg = yaml.safe_load(handle)
    if overrides is not None:
        for kv in overrides:
            k, v = kv.split("=", 1)
            sub = cfg
            *parents, last = k.split(".")
            for key in parents:
                if key not in sub:
                    raise KeyError(f"Key {key} not found in config.")
                sub = sub[key]
            if last not in sub:
                raise KeyError(f"Key {last} not found in config.")

            sub[last] = yaml.safe_load(v)  # converts numbers/bools correctly

    data = cfg.setdefault("data", {})
    data["root"] = str(resolve_path(data.get("root") or "."))
    save = cfg.setdefault("save", {})
    defaults = {
        "model_dir": models_dir(), "img_dir": files_dir() / "pred_images",
        "evals_dir": files_dir() / "evals", "maps_dir": files_dir() / "maps",
    }
    for key, default in defaults.items():
        save[key] = str(resolve_path(save[key], data["root"])) if save.get(key) else str(default)
    return cfg


def parse_early_stopping(early_stopping):
    if not early_stopping:
        return None

    if isinstance(early_stopping, dict):
        if not early_stopping.get("enabled", True):
            return None

        cfg = {
            "patience": int(early_stopping.get("patience", 10)),
            "min_delta": float(early_stopping.get("min_delta", 0.0)),
            "min_epochs": int(early_stopping.get("min_epochs", 0)),
            "restore_best": bool(early_stopping.get("restore_best", True)),
        }
    else:
        cfg = {
            "patience": 10,
            "min_delta": 0.0,
            "min_epochs": 0,
            "restore_best": True,
        }

    assert cfg["patience"] >= 1, "early_stopping.patience must be >= 1"
    assert cfg["min_delta"] >= 0, "early_stopping.min_delta must be >= 0"
    assert cfg["min_epochs"] >= 0, "early_stopping.min_epochs must be >= 0"
    return cfg


def worker_init_fn(seed_main):
    def _init(worker_id):
        seed = seed_main + worker_id
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
    return _init


def split_by_date(data_record, train_size, test_size, val_size, random_state=42, using_wandb=True):
    """
    Split the data into training and testing sets based on the date.
    """
    data_record["Date"] = pd.to_datetime(data_record["Date"])
    unique_sets = (
        data_record.groupby("Set")["Date"].first().sort_values()
    )  # Assume each set has one associated date. Here we take the first date per set.
    total = len(unique_sets)

    train_end = int(total * train_size)
    remaining = total - train_end
    val_count = int(remaining * (val_size / (test_size + val_size)))

    train_sets = unique_sets.iloc[:train_end].index.tolist()
    val_sets = unique_sets.iloc[train_end: train_end + val_count].index.tolist()
    test_sets = unique_sets.iloc[train_end + val_count:].index.tolist()

    train_val_date = unique_sets.iloc[train_end]
    val_test_date = unique_sets.iloc[train_end + val_count]

    if using_wandb and _wandb_ready():
        wandb.run.summary["train_val_date"] = train_val_date    # type: ignore
        wandb.run.summary["val_test_date"] = val_test_date      # type: ignore

    return train_sets, test_sets, val_sets, train_val_date, val_test_date


def limits_robust(x, lower=1.0, upper=99.0):
    lo, hi = np.percentile(x, [lower, upper])
    if hi <= lo:
        hi = lo + 1e-6
    return float(lo), float(hi)


def make_rgb(ir, red, bg, ir_lims, red_lims, bg_lims, gamma=None):
    (ir_lo, ir_hi) = ir_lims
    (red_lo, red_hi) = red_lims
    (bg_lo, bg_hi) = bg_lims
    r = np.clip((ir - ir_lo) / max(ir_hi - ir_lo, 1e-6), 0, 1)
    g = np.clip((red - red_lo) / max(red_hi - red_lo, 1e-6), 0, 1)
    b = np.clip((bg - bg_lo) / max(bg_hi - bg_lo, 1e-6), 0, 1)
    rgb = np.dstack([r, g, b])
    if gamma is not None:
        rgb = np.clip(rgb, 0, 1) ** (1.0 / float(gamma))
    return rgb


def create_prediction_tile(
    pred_img, ir_img, bg_img, red3_img, red5_img,
    true_img=None, base_img=None,
    obs_id="N/A", set_name="N/A", date="N/A",
    save_path=None, idx=0, show=False
):
    """
    Constructs the 5x4 visualization tile from NumPy arrays.
    If true_img or base_img are None, placeholders are used.
    """
    H, W = pred_img.shape

    # Handle missing Ground Truth or Base (for pure inference)
    if true_img is None:
        true_img = np.zeros_like(pred_img)  # Placeholder
    if base_img is None:
        base_img = np.zeros_like(pred_img)  # Placeholder

    # Ensure 2D
    def _sq(x):
        return x.squeeze() if x.ndim > 2 else x
    pred_img, true_img, base_img = _sq(pred_img), _sq(true_img), _sq(base_img)
    ir_img, bg_img = _sq(ir_img), _sq(bg_img)
    red3_img, red5_img = _sq(red3_img), _sq(red5_img)

    # ===== Build 5×4 figure =====
    fig, axes = plt.subplots(
        5, 4, figsize=(15, 18), constrained_layout=True,
        gridspec_kw={'width_ratios': [1, 1, 1, 0.05]}
    )

    AX = axes  # AX[row, col]   # type: ignore
    AX: np.ndarray = AX
    IMG_COLS = [0, 1, 2]
    CB_COL = 3
    box_aspect = H / float(W)

    for r in range(5):
        for c in IMG_COLS:
            AX[r, c].set_box_aspect(box_aspect)

    # ---------- Row 1: Context ----------
    AX[0, 0].imshow(ir_img, cmap="gray")
    AX[0, 0].set_title("IR10")
    AX[0, 1].imshow(bg_img, cmap="gray")
    AX[0, 1].set_title("BG12")

    # RED3/RED5 joint context
    red_context_min = min(np.min(true_img), np.min(pred_img), np.min(base_img),
                          np.min(red3_img), np.min(red5_img))
    red_context_max = max(np.max(true_img), np.max(pred_img), np.max(base_img),
                          np.max(red3_img), np.max(red5_img))

    separator_value = 100
    separator_width = 64
    separator_arr = np.full((H, separator_width), separator_value)

    # Slice to H just in case
    red_context_img = np.concatenate([
        separator_arr, red3_img[:H, :],
        separator_arr, red5_img[:H, :],
        separator_arr
    ], axis=1)

    im_ctx = AX[0, 2].imshow(red_context_img, cmap="gray", vmin=red_context_min, vmax=red_context_max)
    AX[0, 2].set_title("RED3/RED5 (Context)")

    # ---------- Row 2: Grayscale RED (joint) ----------
    red_joint_min = min(np.min(true_img), np.min(pred_img), np.min(base_img))
    red_joint_max = max(np.max(true_img), np.max(pred_img), np.max(base_img))

    im_r2_true = AX[1, 0].imshow(true_img, cmap="gray", vmin=red_joint_min, vmax=red_joint_max)
    AX[1, 0].set_title("RED4 True")
    AX[1, 1].imshow(pred_img, cmap="gray", vmin=red_joint_min, vmax=red_joint_max)
    AX[1, 1].set_title("RED4 Pred")
    AX[1, 2].imshow(base_img, cmap="gray", vmin=red_joint_min, vmax=red_joint_max)
    AX[1, 2].set_title("RED4 Base")

    # ---------- Row 3: RGB anchored to True ----------
    ir_lo_A, ir_hi_A = limits_robust(ir_img)
    bg_lo_A, bg_hi_A = limits_robust(bg_img)
    red_lo_Anc, red_hi_Anc = limits_robust(true_img if np.any(true_img) else pred_img)

    rgb_true_anch = make_rgb(ir_img, true_img, bg_img, (ir_lo_A, ir_hi_A), (red_lo_Anc, red_hi_Anc), (bg_lo_A, bg_hi_A))
    rgb_pred_anch = make_rgb(ir_img, pred_img, bg_img, (ir_lo_A, ir_hi_A), (red_lo_Anc, red_hi_Anc), (bg_lo_A, bg_hi_A))
    rgb_base_anch = make_rgb(ir_img, base_img, bg_img, (ir_lo_A, ir_hi_A), (red_lo_Anc, red_hi_Anc), (bg_lo_A, bg_hi_A))

    AX[2, 0].imshow(rgb_true_anch)
    AX[2, 0].set_title("RGB True (anchored)")
    AX[2, 1].imshow(rgb_pred_anch)
    AX[2, 1].set_title("RGB Pred (anchored)")
    AX[2, 2].imshow(rgb_base_anch)
    AX[2, 2].set_title("RGB Base (anchored)")

    # ---------- Row 4: RGB independent ----------
    ir_lo_I, ir_hi_I = limits_robust(ir_img)
    bg_lo_I, bg_hi_I = limits_robust(bg_img)
    red_lo_true_I, red_hi_true_I = limits_robust(true_img)
    red_lo_pred_I, red_hi_pred_I = limits_robust(pred_img)
    red_lo_base_I, red_hi_base_I = limits_robust(base_img)

    rgb_true_ind = make_rgb(ir_img, true_img, bg_img, (ir_lo_I, ir_hi_I), (red_lo_true_I, red_hi_true_I), (bg_lo_I, bg_hi_I))
    rgb_pred_ind = make_rgb(ir_img, pred_img, bg_img, (ir_lo_I, ir_hi_I), (red_lo_pred_I, red_hi_pred_I), (bg_lo_I, bg_hi_I))
    rgb_base_ind = make_rgb(ir_img, base_img, bg_img, (ir_lo_I, ir_hi_I), (red_lo_base_I, red_hi_base_I), (bg_lo_I, bg_hi_I))

    AX[3, 0].imshow(rgb_true_ind)
    AX[3, 0].set_title("RGB True (indep)")
    AX[3, 1].imshow(rgb_pred_ind)
    AX[3, 1].set_title("RGB Pred (indep)")
    AX[3, 2].imshow(rgb_base_ind)
    AX[3, 2].set_title("RGB Base (indep)")

    # ---------- Row 5: Error maps ----------
    diff_tp = np.abs(true_img - pred_img)
    diff_tb = np.abs(true_img - base_img)
    diff_pb = np.abs(pred_img - base_img)
    vmax_err = np.percentile(np.concatenate([diff_tp.ravel(), diff_tb.ravel(), diff_pb.ravel()]), 99.0)

    im_err_tp = AX[4, 0].imshow(diff_tp, cmap="magma", vmin=0, vmax=vmax_err)
    AX[4, 0].set_title("|True - Pred|")
    AX[4, 1].imshow(diff_tb, cmap="magma", vmin=0, vmax=vmax_err)
    AX[4, 1].set_title("|True - Base|")
    AX[4, 2].imshow(diff_pb, cmap="magma", vmin=0, vmax=vmax_err)
    AX[4, 2].set_title("|Pred - Base|")

    # ---------- Cosmetics ----------
    for r in range(5):
        for c in IMG_COLS:
            AX[r, c].axis("off")

    # Colorbars
    fig.colorbar(im_ctx, cax=AX[0, CB_COL])
    fig.colorbar(im_r2_true, cax=AX[1, CB_COL])
    fig.colorbar(im_err_tp, cax=AX[4, CB_COL])
    AX[2, CB_COL].set_visible(False)
    AX[3, CB_COL].set_visible(False)

    fig.suptitle(f"Obs: {obs_id}\nSet: {set_name} | Date: {date}", fontsize=14, fontweight="bold", y=1.02)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.2)

    if not show:
        plt.close(fig)

    return fig


def save_individual_arrays(
    pred_img, ir_img, bg_img, red3_img, red5_img,
    true_img=None, base_img=None,
    obs_id="N/A", save_path=None,
    save_neigh_bands=False, neigh_width=32
):
    """
    Saves each image panel from the 5x4 prediction tile as an individual PNG
    without axes, titles, or padding.

    When save_neigh_bands is True, also saves the six long, thin neighbour
    bands that are fed to the model as the six channels of x_neigh.
    """
    if save_path is None:
        return

    # Create the 'extras' directory relative to the tile save path
    base_dir = os.path.dirname(save_path)
    extras_dir = os.path.join(base_dir, "extras")
    os.makedirs(extras_dir, exist_ok=True)

    if true_img is None:
        true_img = np.zeros_like(pred_img)
    if base_img is None:
        base_img = np.zeros_like(pred_img)

    def _sq(x):
        return x.squeeze() if x.ndim > 2 else x

    pred_img = _sq(pred_img)
    true_img = _sq(true_img)
    base_img = _sq(base_img)
    ir_img = _sq(ir_img)
    bg_img = _sq(bg_img)
    red3_img = _sq(red3_img)
    red5_img = _sq(red5_img)

    H, W = pred_img.shape

    # -----------------------------
    # Shared grayscale limits
    # -----------------------------
    red_joint_min = min(np.min(true_img), np.min(pred_img), np.min(base_img))
    red_joint_max = max(np.max(true_img), np.max(pred_img), np.max(base_img))

    red_context_min = min(
        np.min(true_img), np.min(pred_img), np.min(base_img),
        np.min(red3_img), np.min(red5_img)
    )
    red_context_max = max(
        np.max(true_img), np.max(pred_img), np.max(base_img),
        np.max(red3_img), np.max(red5_img)
    )

    # -----------------------------
    # RED3/RED5 context strip
    # Same construction as tile
    # -----------------------------
    separator_value = 100
    separator_width = 64
    separator_arr = np.full((H, separator_width), separator_value)

    red_context_img = np.concatenate([
        separator_arr,
        red3_img[:H, :],
        separator_arr,
        red5_img[:H, :],
        separator_arr
    ], axis=1)

    # -----------------------------
    # RGB anchored
    # -----------------------------
    ir_lo_A, ir_hi_A = limits_robust(ir_img)
    bg_lo_A, bg_hi_A = limits_robust(bg_img)
    red_lo_Anc, red_hi_Anc = limits_robust(true_img if np.any(true_img) else pred_img)

    rgb_true_anch = make_rgb(
        ir_img, true_img, bg_img,
        (ir_lo_A, ir_hi_A),
        (red_lo_Anc, red_hi_Anc),
        (bg_lo_A, bg_hi_A)
    )
    rgb_pred_anch = make_rgb(
        ir_img, pred_img, bg_img,
        (ir_lo_A, ir_hi_A),
        (red_lo_Anc, red_hi_Anc),
        (bg_lo_A, bg_hi_A)
    )
    rgb_base_anch = make_rgb(
        ir_img, base_img, bg_img,
        (ir_lo_A, ir_hi_A),
        (red_lo_Anc, red_hi_Anc),
        (bg_lo_A, bg_hi_A)
    )

    # -----------------------------
    # RGB independent
    # -----------------------------
    ir_lo_I, ir_hi_I = limits_robust(ir_img)
    bg_lo_I, bg_hi_I = limits_robust(bg_img)
    red_lo_true_I, red_hi_true_I = limits_robust(true_img)
    red_lo_pred_I, red_hi_pred_I = limits_robust(pred_img)
    red_lo_base_I, red_hi_base_I = limits_robust(base_img)

    rgb_true_ind = make_rgb(
        ir_img, true_img, bg_img,
        (ir_lo_I, ir_hi_I),
        (red_lo_true_I, red_hi_true_I),
        (bg_lo_I, bg_hi_I)
    )
    rgb_pred_ind = make_rgb(
        ir_img, pred_img, bg_img,
        (ir_lo_I, ir_hi_I),
        (red_lo_pred_I, red_hi_pred_I),
        (bg_lo_I, bg_hi_I)
    )
    rgb_base_ind = make_rgb(
        ir_img, base_img, bg_img,
        (ir_lo_I, ir_hi_I),
        (red_lo_base_I, red_hi_base_I),
        (bg_lo_I, bg_hi_I)
    )

    # -----------------------------
    # Error maps
    # -----------------------------
    diff_tp = np.abs(true_img - pred_img)
    diff_tb = np.abs(true_img - base_img)
    diff_pb = np.abs(pred_img - base_img)

    vmax_err = np.percentile(
        np.concatenate([diff_tp.ravel(), diff_tb.ravel(), diff_pb.ravel()]),
        99.0
    )

    # -----------------------------
    # Save all 15 panels
    # -----------------------------
    images_to_save = {
        # Row 1
        "ir10": (ir_img, "gray", None, None),
        "bg12": (bg_img, "gray", None, None),
        "red3_red5_context": (red_context_img, "gray", red_context_min, red_context_max),

        # Row 2
        "red4_true": (true_img, "gray", red_joint_min, red_joint_max),
        "red4_pred": (pred_img, "gray", red_joint_min, red_joint_max),
        "red4_base": (base_img, "gray", red_joint_min, red_joint_max),

        # Row 3
        "rgb_true_anch": (rgb_true_anch, None, None, None),
        "rgb_pred_anch": (rgb_pred_anch, None, None, None),
        "rgb_base_anch": (rgb_base_anch, None, None, None),

        # Row 4
        "rgb_true_ind": (rgb_true_ind, None, None, None),
        "rgb_pred_ind": (rgb_pred_ind, None, None, None),
        "rgb_base_ind": (rgb_base_ind, None, None, None),

        # Row 5
        "error_true_pred": (diff_tp, "magma", 0, vmax_err),
        "error_true_base": (diff_tb, "magma", 0, vmax_err),
        "error_pred_base": (diff_pb, "magma", 0, vmax_err),
    }

    for name, (data, cmap, vmin, vmax) in images_to_save.items():
        fname = f"{obs_id}_{name}.png"
        fpath = os.path.join(extras_dir, fname)
        plt.imsave(fpath, data, cmap=cmap, vmin=vmin, vmax=vmax)

    # -----------------------------
    # Six long, thin neighbour bands (the channels of x_neigh)
    # -----------------------------
    if save_neigh_bands:
        w = int(neigh_width)

        # Channel order matches torch.stack([...]) in HiPredict._build_inputs:
        #   [right_bg, right_ir, left_r5, left_bg, left_ir, right_r3]
        neigh_bands = {
            "neigh0_bg12_right": bg_img[:, -w:],
            "neigh1_ir10_right": ir_img[:, -w:],
            "neigh2_red5_left":  red5_img[:, :w],
            "neigh3_bg12_left":  bg_img[:, :w],
            "neigh4_ir10_left":  ir_img[:, :w],
            "neigh5_red3_right": red3_img[:, -w:],
        }

        for name, data in neigh_bands.items():
            fname = f"{obs_id}_{name}.png"
            fpath = os.path.join(extras_dir, fname)
            plt.imsave(fpath, data, cmap="gray")


def save_batch_imgs(batch, model, save_dir, device, criterion, arch, img_type="pred", save_diag=False, save_indiv=True, stats_path=None):
    """
    Save predicted images and ground truth images side by side.
    !! Only call from within evaluate function !!
    """

    arch_dir = os.path.join(save_dir, arch)
    extra_dir = os.path.join(arch_dir, "extra")
    os.makedirs(arch_dir, exist_ok=True)
    os.makedirs(extra_dir, exist_ok=True)

    x_band = batch["x_band"].to(device).float()
    x_neigh = batch["x_neigh"].to(device).float()
    x_meta = batch["x_meta"].to(device).float()
    y_test = batch["y"].to(device).float()
    obs_list = batch["obs_id"]
    set_list = batch["set_name"]
    date_list = batch["date"]
    patch_x = batch["patch_x"].to(device).float()
    obs_width = batch["obs_width"].to(device).float()

    y_pred = model(x_band, x_neigh, x_meta, patch_x=patch_x, obs_width=obs_width)
    y_base_pred = SYN4.from_global_stats(device, stats_path=stats_path).eval()(x_band, x_neigh, x_meta)
    y_test = y_test.type_as(y_pred)

    def squeeze_ch(t):
        return t.squeeze(1) if (t.ndim == 4 and t.size(1) == 1) else t

    y_pred = squeeze_ch(y_pred)
    y_test = squeeze_ch(y_test)
    y_base_pred = squeeze_ch(y_base_pred)

    # Shape of true imgs
    B, H, W = y_test.shape
    max_iter = min(16, x_band.shape[0])  # Save at most 16 images

    wandb_imgs = []
    per_img_losses = []

    pix_ranges = compute_batch_ranges(model, batch, device, stats_path=stats_path)

    def _gaussian_kernel_t(k=21, sigma=6.0, device=None):
        x = torch.arange(k, device=device).float() - (k - 1) / 2
        g = torch.exp(-0.5 * (x / sigma) ** 2)
        g = g / (g.sum() + 1e-8)
        kg = g.view(1, 1, 1, k) * g.view(1, 1, k, 1)
        return kg

    def plot_lf_nd_diagnostics(ir_t, bg_t, true_t, pred_t, save_dir, arch, img_type, i, title):
        # ir_t, bg_t: (H,W) torch; true_t, pred_t: (H,W) torch
        device = true_t.device
        H, W = true_t.shape
        eps = 1e-8

        # to 4D for conv
        def _to4(z):
            return z.view(1, 1, H, W)

        K = _gaussian_kernel_t(k=21, sigma=6.0, device=device)
        pad = (K.size(-1) // 2, K.size(-1) // 2, K.size(-2) // 2, K.size(-2) // 2)

        def _blur(z2d):
            z4 = _to4(z2d)
            z4 = F.pad(z4, pad, mode='reflect')
            return F.conv2d(z4, K).view(H, W)

        # blurred fields
        r_t = _blur(true_t)
        r_p = _blur(pred_t)
        b_lf = _blur(bg_t)

        # residuals
        delta = pred_t - true_t
        d_lf = _blur(delta)
        d_hf = delta - d_lf

        # ND maps on LF fields (RED vs BG)
        nd_t = (r_t - b_lf) / (r_t + b_lf + eps)
        nd_p = (r_p - b_lf) / (r_p + b_lf + eps)
        d_nd = nd_p - nd_t

        # numpy views
        r_t_np = r_t.detach().cpu().numpy()
        r_p_np = r_p.detach().cpu().numpy()
        d_lf_np = d_lf.detach().cpu().numpy()
        nd_t_np = nd_t.detach().cpu().numpy()
        nd_p_np = nd_p.detach().cpu().numpy()
        d_nd_np = d_nd.detach().cpu().numpy()
        delta_np = delta.detach().cpu().numpy()
        dhf_abs_np = d_hf.abs().detach().cpu().numpy()
        b_lf_np = b_lf.detach().cpu().numpy()

        # robust limits
        def _sym_lims(a):
            p1, p99 = np.percentile(a, [1, 99])
            v = max(abs(p1), abs(p99), 1e-6)
            return -v, v

        rtmin, rtmax = np.percentile(np.concatenate([r_t_np.ravel(), r_p_np.ravel()]), [1, 99])
        dlf_vmin, dlf_vmax = _sym_lims(d_lf_np)
        nd_vmin, nd_vmax = _sym_lims(np.concatenate([nd_t_np.ravel(), nd_p_np.ravel()]))
        d_raw_vmin, d_raw_vmax = _sym_lims(delta_np)
        dhf_max = np.percentile(dhf_abs_np, 99)
        bmin, bmax = np.percentile(b_lf_np, [1, 99])

        # simple scalars to annotate
        E_LF = float((d_lf_np ** 2).sum() / ((delta_np ** 2).sum() + eps))
        rho_dlf_nd = np.corrcoef(d_lf_np.ravel() - d_lf_np.mean(), nd_t_np.ravel() - nd_t_np.mean())[0, 1]
        mu_pred = float(pred_t.mean().item())
        mu_true = float(true_t.mean().item())

        # 3x3 figure
        fig, ax = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)     # type: ignore
        ax: np.ndarray = ax
        for r in range(3):
            for c in range(3):
                ax[r, c].set_axis_off()

        im00 = ax[0, 0].imshow(r_t_np, cmap='gray', vmin=rtmin, vmax=rtmax)
        ax[0, 0].set_title('RED4 True (LF)', fontsize=10)
        _ = ax[0, 1].imshow(r_p_np, cmap='gray', vmin=rtmin, vmax=rtmax)
        ax[0, 1].set_title('RED4 Pred (LF)', fontsize=10)
        im02 = ax[0, 2].imshow(d_lf_np, cmap='coolwarm', vmin=dlf_vmin, vmax=dlf_vmax)
        ax[0, 2].set_title('δ_LF = Pred_LF − True_LF', fontsize=10)

        im10 = ax[1, 0].imshow(nd_t_np, cmap='coolwarm', vmin=nd_vmin, vmax=nd_vmax)
        ax[1, 0].set_title('ND_true (LF)', fontsize=10)
        _ = ax[1, 1].imshow(nd_p_np, cmap='coolwarm', vmin=nd_vmin, vmax=nd_vmax)
        ax[1, 1].set_title('ND_pred (LF)', fontsize=10)
        im12 = ax[1, 2].imshow(d_nd_np, cmap='coolwarm', vmin=_sym_lims(d_nd_np)[0], vmax=_sym_lims(d_nd_np)[1])
        ax[1, 2].set_title('ΔND = ND_pred − ND_true', fontsize=10)

        im20 = ax[2, 0].imshow(delta_np, cmap='coolwarm', vmin=d_raw_vmin, vmax=d_raw_vmax)
        ax[2, 0].set_title('δ = Pred − True (raw)', fontsize=10)
        im21 = ax[2, 1].imshow(dhf_abs_np, cmap='magma', vmin=0.0, vmax=dhf_max)
        ax[2, 1].set_title('|δ_HF| (abs high-freq residual)', fontsize=10)
        im22 = ax[2, 2].imshow(b_lf_np, cmap='gray', vmin=bmin, vmax=bmax)
        ax[2, 2].set_title('BG_LF (blurred)', fontsize=10)

        # colorbars on the right of each row
        fig.colorbar(im00, ax=[ax[0, 0], ax[0, 1]], fraction=0.046, pad=0.02)
        fig.colorbar(im02, ax=[ax[0, 2]], fraction=0.046, pad=0.02)
        fig.colorbar(im10, ax=[ax[1, 0], ax[1, 1]], fraction=0.046, pad=0.02)
        fig.colorbar(im12, ax=[ax[1, 2]], fraction=0.046, pad=0.02)
        fig.colorbar(im20, ax=[ax[2, 0]], fraction=0.046, pad=0.02)
        fig.colorbar(im21, ax=[ax[2, 1]], fraction=0.046, pad=0.02)
        fig.colorbar(im22, ax=[ax[2, 2]], fraction=0.046, pad=0.02)

        fig.suptitle(f"{title}\nμ_pred−μ_true={mu_pred - mu_true:+.3e}   E_LF={E_LF:.3f}   corr(δ_LF, ND_true)={rho_dlf_nd:+.3f}",
                     fontsize=12, y=1.02)

        out_path = os.path.join(save_dir, f"{arch}_{img_type}_{i}_diag.png")
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(out_path, dpi=600, bbox_inches='tight', pad_inches=0.15)
        plt.close(fig)

    # --------------------------------------------------------------------------------

    print(f"\nRed min: {pix_ranges['RED']['min']:.4f}, max: {pix_ranges['RED']['max']:.4f}")

    model_name = model.__class__.__name__

    for i in range(max_iter):
        # 1. KEEP TENSORS FOR LOSS CALCULATION & DIAGNOSTICS
        pred_t = y_pred[i]       # (H, W)
        true_t = y_test[i]       # (H, W)
        base_t = y_base_pred[i]  # (H, W)
        ir_t = x_band[i, 0]
        bg_t = x_band[i, 1]

        # 2. CALCULATE LOSS (Required for the scatter plot at the end)
        pred_loss = criterion(pred_t, true_t)[0]
        base_loss = criterion(base_t, true_t)[0]

        # Safe item retrieval
        p_loss_val = pred_loss.detach().item() if hasattr(pred_loss, "detach") else float(pred_loss)
        b_loss_val = base_loss.detach().item() if hasattr(base_loss, "detach") else float(base_loss)

        per_img_losses.append({
            "idx": i,
            "obs_id": obs_list[i],
            "set": set_list[i],
            "pred_loss": p_loss_val,
            "base_loss": b_loss_val
        })

        if save_indiv:
            ir_img = ir_t.detach().cpu().numpy()
            bg_img = bg_t.detach().cpu().numpy()
            pred_img = pred_t.detach().cpu().numpy()
            true_img = true_t.detach().cpu().numpy()
            base_img = base_t.detach().cpu().numpy()
            red5_img = x_neigh[i, 2].detach().cpu().numpy()
            red3_img = x_neigh[i, 5].detach().cpu().numpy()

            red_joint_min = min(pred_img.min(), true_img.min(), base_img.min())
            red_joint_max = max(pred_img.max(), true_img.max(), base_img.max())
            red_context_min = min(
                true_img.min(), pred_img.min(), base_img.min(),
                red3_img.min(), red5_img.min()
            )
            red_context_max = max(
                true_img.max(), pred_img.max(), base_img.max(),
                red3_img.max(), red5_img.max()
            )

            diff_tp = np.abs(true_img - pred_img)
            diff_tb = np.abs(true_img - base_img)
            diff_pb = np.abs(pred_img - base_img)
            vmax_err = np.percentile(
                np.concatenate([diff_tp.ravel(), diff_tb.ravel(), diff_pb.ravel()]),
                99.0
            )

            def _save_raw(arr, fname, cmap='gray', vmin=None, vmax=None):
                path = os.path.join(extra_dir, fname)
                plt.imsave(path, arr, cmap=cmap, vmin=vmin, vmax=vmax)

            _save_raw(ir_img, f"{arch}_{img_type}_{i}_IR10.png")
            _save_raw(bg_img, f"{arch}_{img_type}_{i}_BG12.png")

            H, _ = pred_img.shape
            separator_value = 100
            separator_width = 64
            separator_arr = np.full((H, separator_width), separator_value)
            red_context_img = np.concatenate([
                separator_arr,
                red3_img[:H, :],
                separator_arr,
                red5_img[:H, :],
                separator_arr
            ], axis=1)
            _save_raw(red_context_img, f"{arch}_{img_type}_{i}_RED3_RED5.png", vmin=red_context_min, vmax=red_context_max)

            # RED4 Joint Normalized
            _save_raw(true_img, f"{arch}_{img_type}_{i}_RED4_true_joint.png", vmin=red_joint_min, vmax=red_joint_max)
            _save_raw(pred_img, f"{arch}_{img_type}_{i}_RED4_pred_joint.png", vmin=red_joint_min, vmax=red_joint_max)
            _save_raw(base_img, f"{arch}_{img_type}_{i}_RED4_base_joint.png", vmin=red_joint_min, vmax=red_joint_max)

            #  Error Maps (Absolute Differences)
            _save_raw(diff_tp, f"{arch}_{img_type}_{i}_ERR_true_minus_pred.png", cmap="magma", vmin=0, vmax=vmax_err)
            _save_raw(diff_tb, f"{arch}_{img_type}_{i}_ERR_true_minus_base.png", cmap="magma", vmin=0, vmax=vmax_err)
            _save_raw(diff_pb, f"{arch}_{img_type}_{i}_ERR_pred_minus_base.png", cmap="magma", vmin=0, vmax=vmax_err)

            # RGB Compositions (Anchored vs Independent)
            ir_lo_A, ir_hi_A = limits_robust(ir_img)
            bg_lo_A, bg_hi_A = limits_robust(bg_img)
            red_lo_Anc, red_hi_Anc = limits_robust(true_img if np.any(true_img) else pred_img)

            rgb_true_anch = make_rgb(
                ir_img, true_img, bg_img,
                (ir_lo_A, ir_hi_A),
                (red_lo_Anc, red_hi_Anc),
                (bg_lo_A, bg_hi_A)
            )
            rgb_pred_anch = make_rgb(
                ir_img, pred_img, bg_img,
                (ir_lo_A, ir_hi_A),
                (red_lo_Anc, red_hi_Anc),
                (bg_lo_A, bg_hi_A)
            )
            rgb_base_anch = make_rgb(
                ir_img, base_img, bg_img,
                (ir_lo_A, ir_hi_A),
                (red_lo_Anc, red_hi_Anc),
                (bg_lo_A, bg_hi_A)
            )

            ir_lo_I, ir_hi_I = limits_robust(ir_img)
            bg_lo_I, bg_hi_I = limits_robust(bg_img)
            red_lo_true_I, red_hi_true_I = limits_robust(true_img)
            red_lo_pred_I, red_hi_pred_I = limits_robust(pred_img)
            red_lo_base_I, red_hi_base_I = limits_robust(base_img)

            rgb_true_ind = make_rgb(
                ir_img, true_img, bg_img,
                (ir_lo_I, ir_hi_I),
                (red_lo_true_I, red_hi_true_I),
                (bg_lo_I, bg_hi_I)
            )
            rgb_pred_ind = make_rgb(
                ir_img, pred_img, bg_img,
                (ir_lo_I, ir_hi_I),
                (red_lo_pred_I, red_hi_pred_I),
                (bg_lo_I, bg_hi_I)
            )
            rgb_base_ind = make_rgb(
                ir_img, base_img, bg_img,
                (ir_lo_I, ir_hi_I),
                (red_lo_base_I, red_hi_base_I),
                (bg_lo_I, bg_hi_I)
            )

            _save_raw(rgb_true_anch, f"{arch}_{img_type}_{i}_RGB_true_anchored.png", cmap=None)
            _save_raw(rgb_pred_anch, f"{arch}_{img_type}_{i}_RGB_pred_anchored.png", cmap=None)
            _save_raw(rgb_base_anch, f"{arch}_{img_type}_{i}_RGB_base_anchored.png", cmap=None)
            _save_raw(rgb_true_ind, f"{arch}_{img_type}_{i}_RGB_true_indep.png", cmap=None)
            _save_raw(rgb_pred_ind, f"{arch}_{img_type}_{i}_RGB_pred_indep.png", cmap=None)
            _save_raw(rgb_base_ind, f"{arch}_{img_type}_{i}_RGB_base_indep.png", cmap=None)

        # 3. DIAGNOSTICS (Requires Tensors)
        if save_diag:
            plot_lf_nd_diagnostics(
                ir_t=ir_t, bg_t=bg_t, true_t=true_t, pred_t=pred_t,
                save_dir=extra_dir, arch=arch, img_type=img_type, i=i,
                title=f"{obs_list[i]} — {set_list[i]} — {date_list[i]}"
            )

        # 4. CONVERT TO NUMPY FOR TILE GENERATION
        pred_img_np = pred_t.detach().cpu().numpy()
        true_img_np = true_t.detach().cpu().numpy()
        base_img_np = base_t.detach().cpu().numpy()

        ir_img_np = ir_t.detach().cpu().numpy()
        bg_img_np = bg_t.detach().cpu().numpy()
        red5_img_np = x_neigh[i, 2].detach().cpu().numpy()
        red3_img_np = x_neigh[i, 5].detach().cpu().numpy()

        # 5. GENERATE TILE USING NEW HELPER
        image_path = os.path.join(arch_dir, f"{arch}_{img_type}_{i}.png")

        create_prediction_tile(
            pred_img=pred_img_np,
            ir_img=ir_img_np,
            bg_img=bg_img_np,
            red3_img=red3_img_np,
            red5_img=red5_img_np,
            true_img=true_img_np,
            base_img=base_img_np,
            obs_id=str(obs_list[i]),
            set_name=str(set_list[i]),
            date=str(date_list[i]),
            save_path=image_path
        )

        if _wandb_ready():
            wandb_imgs.append(wandb.Image(image_path, caption=f"Prediction {i}"))

        # Console logging
        print(f"============ Image {i} ============")
        print(f"Pred loss: {p_loss_val:.2e}, Base loss: {b_loss_val:.2e}")

    if _wandb_ready() and wandb_imgs:
        wandb.log({"Predictions": wandb_imgs[:6]}, step=None)

    # ---- Final scatter of per-image losses (pred vs base) ----
    idxs = [d["idx"] for d in per_img_losses]
    pred_losses = [d["pred_loss"] for d in per_img_losses]
    base_losses = [d["base_loss"] for d in per_img_losses]

    x_pred = np.array(idxs, dtype=float)
    x_base = np.array(idxs, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.scatter(x_pred, pred_losses, s=22, label=f"{model_name} pred")
    ax.scatter(x_base, base_losses, s=22, label="SYN4 base")

    for xp, xb, yp, yb in zip(x_pred, x_base, pred_losses, base_losses):
        ax.plot([xp, xb], [yp, yb], linestyle="--", color="k", linewidth=1, alpha=0.7)

    ax.set_xticks(idxs)
    ax.set_xlabel("Image index")
    ax.set_ylabel("Loss")
    ax.set_ylim(bottom=0)
    ax.set_title("Per-image loss comparison (pred vs base)")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend()

    loss_fig_path = os.path.join(arch_dir, f"{arch}_{img_type}_loss_scatter.png")
    plt.savefig(loss_fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if _wandb_ready():
        wandb.log({"Loss Scatter": wandb.Image(loss_fig_path)}, step=None)


def compute_batch_ranges(model, batch, device, stats_path=None):
    ranges = {
        "IR": {"min": +float("inf"), "max": -float("inf"), "std": 0},
        "BG": {"min": +float("inf"), "max": -float("inf"), "std": 0},
        "RED": {"min": +float("inf"), "max": -float("inf"), "std": 0},
    }
    model.eval()
    base_model = SYN4.from_global_stats(device, stats_path=stats_path).eval()

    with torch.no_grad():
        x_band = batch["x_band"].to(device).float()
        x_neigh = batch["x_neigh"].to(device).float()
        x_meta = batch["x_meta"].to(device).float()
        y_true = batch["y"].float().to(device)  # (B,1,H,W)
        patch_x = batch["patch_x"].to(device).float()
        obs_width = batch["obs_width"].to(device).float()

        y_pred = model(x_band, x_neigh, x_meta, patch_x=patch_x, obs_width=obs_width)  # (B,1,H,W)
        y_base_pred = base_model(x_band, x_neigh, x_meta)  # (B,1,H,W)

        # collapse batch & channel dims to a flat array per component
        ir = x_band[:, 0].detach().cpu().numpy().ravel()
        bg = x_band[:, 1].detach().cpu().numpy().ravel()

        pr = y_pred.cpu().numpy().ravel()
        tr = y_true.cpu().numpy().ravel()
        bs = y_base_pred.cpu().numpy().ravel()
        red_all = np.concatenate([tr, pr, bs])

        ranges["IR"]["min"] = min(ranges["IR"]["min"], ir.min())
        ranges["IR"]["max"] = max(ranges["IR"]["max"], ir.max())
        ranges["IR"]["std"] = np.std(ir)

        ranges["BG"]["min"] = min(ranges["BG"]["min"], bg.min())
        ranges["BG"]["max"] = max(ranges["BG"]["max"], bg.max())
        ranges["BG"]["std"] = np.std(bg)

        ranges["RED"]["min"] = min(ranges["RED"]["min"], red_all.min())
        ranges["RED"]["max"] = max(ranges["RED"]["max"], red_all.max())
        ranges["RED"]["std"] = np.std(red_all)

    lo, hi = np.percentile(red_all, [1, 99])
    ranges["RED"]["min"], ranges["RED"]["max"] = lo, hi

    return ranges


def plot_losses(losses_dict, save_path, rolling_window=None):
    """
    Plot training and validation losses over epochs.
    """
    plt.figure(figsize=(14, 10))
    for key, values in losses_dict.items():
        if rolling_window is not None and rolling_window > 1:
            line = plt.plot(values, alpha=0.3, linewidth=1)
            colour = line[0].get_color()
            series = pd.Series(values)
            rolling_mean = series.rolling(window=rolling_window).mean()
            plt.plot(rolling_mean, label=f"{key} Loss", linewidth=2, color=colour)
        else:
            plt.plot(values, label=f"{key} Loss", linewidth=1, alpha=0.7)
    plt.xlabel("Step")
    plt.ylabel("Loss (log scale)")
    plt.yscale("log")
    plt.title("Training Losses across Steps")
    plt.legend()
    plt.grid(True, alpha=0.7, linestyle="--")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved loss plot to {save_path}")

    if _wandb_ready():
        wandb.log({"Loss Metrics": wandb.Image(save_path)}, step=None)


def get_model_details(name, model_save_dir="models/"):
    files = [f for f in os.listdir(model_save_dir) if f.startswith(name) and f.endswith(".pt")]
    if not files:
        print("Found no files in model_save_dir, returning None")
        return None

    if len(files) == 1:
        print(f"Found one file in {model_save_dir}: {files[0]}")
        return files[0]

    # Matches ..._HHMM-DDMM.pt at the end
    rx = re.compile(r'_(\d{2})(\d{2})-(\d{2})(\d{2})\.pt$')

    print("rx match found")

    parsed = []
    for f in files:
        m = rx.search(f)
        if m is None:
            continue
        hh = int(m.group(1))
        mi = int(m.group(2))
        dd = int(m.group(3))
        mo = int(m.group(4))
        parsed.append(((mo, dd, hh, mi), f))

    if not parsed:
        return None

    # Primary sort: month, then day, then hour, then minute
    best_key = max(parsed, key=lambda t: t[0])[0]
    candidates = [f for k, f in parsed if k == best_key]

    if len(candidates) == 1:
        return candidates[0]

    # Tie-break with modification time
    return max(candidates, key=lambda fn: os.path.getmtime(os.path.join(model_save_dir, fn)))


def load_latest_into(model, name, latest_path, strict=False):
    ckpt = torch.load(latest_path, map_location='cpu', weights_only=True)

    if isinstance(ckpt, nn.Module):
        loaded = ckpt
    else:
        if isinstance(ckpt, dict):
            sd = ckpt.get('state_dict', ckpt.get('model_state_dict', ckpt))
        else:
            raise TypeError(f"Unexpected checkpoint type for {name}: {type(ckpt)}")
        if any(k.startswith('module.') for k in sd.keys()):
            sd = {k.replace('module.', '', 1): v for k, v in sd.items()}
        missing, unexpected = model.load_state_dict(sd, strict=strict)
        if missing or unexpected:
            print(f"load_state_dict ({name}): missing={missing}, unexpected={unexpected}")
        loaded = model

    return loaded
