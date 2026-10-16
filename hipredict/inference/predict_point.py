"""
Single-point inference for an already-processed observation.

Reads the existing *_processed.cub files directly (no re-calibration),
crops a window around the target pixel, runs inference, and saves the
same tile + extras + spectral-profile outputs as the domain eval scripts.

Edit the CONFIG block, then run with:
    python -m hipredict.inference.predict_point
"""
import os
import time
from pathlib import Path

import numpy as np
import torch

from hipredict.predict.get_model import get_model
from hipredict.paths import files_dir
from hipredict.predict.utils import create_prediction_tile, save_individual_arrays
from hipredict.inference.hipredict import HiPredict
from hipredict.inference.predict_domain import save_center_spectral_plot, calc_metrics
from hipredict.data.isis2np import isis2np

OBS_ID = "ESP_012940_1655"
CENTER_X = 750
CENTER_Y = 4200
IMG_BASE_DIR = str(files_dir() / "isis_imgs" / "observations")
CROP_SIZE = 512    # side length of the extracted crop in pixels
OUTPUT_DIR = str(files_dir() / "figs" / "point_evals")
CNN_NAME = "hipredict"


if __name__ == "__main__":
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cnn_model, cfg = get_model(CNN_NAME, device=device)
    pred = HiPredict(cnn_model, cfg, device=device)
    print(f"Model loaded in {time.time() - start:.2f}s")

    obs_dir = Path(IMG_BASE_DIR) / OBS_ID
    cubefiles = sorted(obs_dir.glob("*_processed.cub"))
    if len(cubefiles) != 5:
        raise FileNotFoundError(
            f"Expected 5 *_processed.cub files in {obs_dir}, found {len(cubefiles)}"
        )
    # Alphabetical sort gives: BG12, IR10, RED3, RED4, RED5
    bg_full, ir_full, r3_full, r4_full, r5_full = (
        isis2np(str(f)) for f in cubefiles
    )
    print(f"Cubefiles read in {time.time() - start:.2f}s  (shape: {bg_full.shape})")

    half = CROP_SIZE // 2
    img_h, img_w = bg_full.shape
    if not (half <= CENTER_X <= img_w - half):
        raise ValueError(
            f"CENTER_X={CENTER_X} is too close to an edge for a {CROP_SIZE}px crop. "
            f"The center must be at least CROP_SIZE/2 = {half}px from every edge, so "
            f"for this {img_w}px-wide image CENTER_X must be in [{half}, {img_w - half}]."
        )
    if not (half <= CENTER_Y <= img_h - half):
        raise ValueError(
            f"CENTER_Y={CENTER_Y} is too close to an edge for a {CROP_SIZE}px crop. "
            f"The center must be at least CROP_SIZE/2 = {half}px from every edge, so "
            f"for this {img_h}px-tall image CENTER_Y must be in [{half}, {img_h - half}]."
        )

    y0, y1 = CENTER_Y - half, CENTER_Y + half
    x0, x1 = CENTER_X - half, CENTER_X + half

    sample = {
        "bg": bg_full[y0:y1, x0:x1],
        "ir": ir_full[y0:y1, x0:x1],
        "r3": r3_full[y0:y1, x0:x1],
        "r5": r5_full[y0:y1, x0:x1],
        "y": r4_full[y0:y1, x0:x1],
        "obs_id": OBS_ID,
    }

    print(f"Running inference on {CROP_SIZE}x{CROP_SIZE} crop...")
    output_img = pred.predict([sample])[0]
    syn4_img = pred.predict([sample], syn4=True)[0]
    print(f"Inference complete in {time.time() - start:.2f}s")

    # ── Metrics vs. true RED4 ──────────────────────────────────────────────────
    true_arr = np.squeeze(sample["y"]).astype(np.float64)
    pred_arr = np.squeeze(output_img).astype(np.float64)
    base_arr = np.squeeze(syn4_img).astype(np.float64)

    def _report(name, model_arr):
        mse = float(np.mean((model_arr - true_arr) ** 2))
        _, pwt, nmae, ssim_val, bias = calc_metrics(model_arr, true_arr, mse)
        print(
            f"  {name:<10s} NMAE={nmae:.4f}  MSE={mse:.3e}  SSIM={ssim_val:.4f}  "
            f"PWT={pwt:5.1f}%  bias={bias:+.3e}"
        )

    print(f"Metrics vs true RED4 (center {CROP_SIZE}x{CROP_SIZE} crop):")
    _report("HiPredict", pred_arr)
    _report("SYN4", base_arr)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    tile_save_path = os.path.join(OUTPUT_DIR, f"tiles_{OBS_ID}.png")

    create_prediction_tile(
        ir_img=sample["ir"],
        bg_img=sample["bg"],
        red3_img=sample["r3"],
        red5_img=sample["r5"],
        pred_img=output_img,
        true_img=sample["y"],
        base_img=syn4_img,
        obs_id=OBS_ID,
        set_name="Inference",
        save_path=tile_save_path,
    )

    save_individual_arrays(
        ir_img=sample["ir"],
        bg_img=sample["bg"],
        red3_img=sample["r3"],
        red5_img=sample["r5"],
        pred_img=output_img,
        true_img=sample["y"],
        base_img=syn4_img,
        obs_id=OBS_ID,
        save_path=tile_save_path,
        save_neigh_bands=True,
    )

    save_center_spectral_plot(
        bg_img=sample["bg"],
        true_img=sample["y"],
        syn4_img=syn4_img,
        pred_img=output_img,
        ir_img=sample["ir"],
        obs_id=OBS_ID,
        save_path=tile_save_path,
        window_size=3,
    )

    print(f"Outputs saved to {OUTPUT_DIR}")
    print(f"  tile:     tiles_{OBS_ID}.png")
    print("  extras:   extras/")
    print("  spectral: spectral_profiles/")
    print(f"Done in {time.time() - start:.2f}s")
