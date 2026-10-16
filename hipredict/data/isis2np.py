"""
File to convert ISIS cube files to NumPy arrays and save them as images and patches.
"""

import shlex
import pvl
import numpy as np
import os
import subprocess
import time
from concurrent.futures import as_completed, ProcessPoolExecutor
from pathlib import Path
import csv
import traceback
import pandas as pd
import hashlib
import random
import glob
from typing import Any

from hipredict.data.normalization import _robust_center_scale_from_predictors_np, _valid_pixel_mask
from hipredict.utils.utils import save_img
from hipredict.data.load_metadata import gen_edr_df
from hipredict.paths import PACKAGE_ROOT, files_dir

np.random.seed(42)
random.seed(42)

PATCH_CCDS = ("BG12", "IR10", "RED3", "RED4", "RED5")
MISSING_VALUE = -3.663e-4
PATCH_X_BORDER = 150
DATA_RECORD_COLUMNS = [
    "Observation",
    "Set",
    "Patch",
    "CCD",
    "Path",
    "Height",
    "Width",
    "Channels",
    "Date",
    "Pix_min",
    "Pix_max",
    "Binning",
    "TDI",
    "CPMM_temp",
    "Emission_angle",
    "FPA_negY_temp",
    "FPA_posY_temp",
    "FPE_temp",
    "IEA_PWS_temp",
    "IEA_temp",
    "Image_center_lat",
    "Image_center_lon",
    "Incidence_angle",
    "Local_time",
    "North_azimuth",
    "Orbit_number",
    "Phase_angle",
    "Slant_distance",
    "Solar_distance",
    "Solar_longitude",
    "Spacecraft_altitude",
    "Sub_solar_azimuth",
    "Sub_solar_latitude",
    "Sub_solar_longitude",
    "Sub_spacecraft_latitude",
    "Sub_spacecraft_longitude",
    "Target_center_distance",
    "Patch_x",
    "Patch_y",
    "Obs_center",
    "Obs_scale",
]
PATCH_X_COLUMN = "Patch_x"
PATCH_Y_COLUMN = "Patch_y"
OBS_CENTER_COLUMN = "Obs_center"
OBS_SCALE_COLUMN = "Obs_scale"
SET_COLUMN_INDEX = DATA_RECORD_COLUMNS.index("Set")

_OBS_META = None
_HIPROC_ENV = None


def init_worker(obs_meta, hiproc_env):
    global _OBS_META, _HIPROC_ENV
    _OBS_META = obs_meta
    _HIPROC_ENV = hiproc_env


def _coerce_patch_bands(img_arr):
    if isinstance(img_arr, np.ndarray):
        assert img_arr.ndim == 3, f"Input image must be a 3D array (H, W, C), got shape {img_arr.shape}."
        assert img_arr.shape[2] == 5, f"Expected 5 arrays in the 3rd dimension, got {img_arr.shape[2]}. Shape: {img_arr.shape}"
        bands = tuple(np.asarray(img_arr[..., idx]) for idx in range(img_arr.shape[2]))
    else:
        bands = tuple(np.asarray(arr) for arr in img_arr)
        assert len(bands) == 5, f"Expected 5 aligned 2D arrays, got {len(bands)}."
        assert all(arr.ndim == 2 for arr in bands), "All band arrays must be 2D."
        shapes = {arr.shape for arr in bands}
        assert len(shapes) == 1, f"All band arrays must share the same shape, got {sorted(shapes)}."

    H, W = bands[0].shape
    return bands, H, W


def _sample_unique_patch_positions(rng, num_patches, min_x, max_x, min_y, max_y, existing_coords=None):
    if min_x > max_x or min_y > max_y:
        raise ValueError(f"Invalid patch sampling bounds: x=[{min_x}, {max_x}], y=[{min_y}, {max_y}]")

    existing = {
        (int(coord[0]), int(coord[1]))
        for coord in (existing_coords or [])
        if coord is not None and min_x <= int(coord[0]) <= max_x and min_y <= int(coord[1]) <= max_y
    }
    selected = []
    max_unique = (max_x - min_x + 1) * (max_y - min_y + 1)
    if len(existing) + num_patches > max_unique:
        raise ValueError(
            f"Requested {num_patches} new patches but only {max_unique - len(existing)} unique positions remain."
        )

    attempts = 0
    max_attempts = max(1_000, num_patches * 200)
    while len(selected) < num_patches and attempts < max_attempts:
        attempts += 1
        coord = (
            int(rng.integers(min_x, max_x + 1)),
            int(rng.integers(min_y, max_y + 1)),
        )
        if coord in existing:
            continue
        existing.add(coord)
        selected.append(coord)

    if len(selected) == num_patches:
        return selected

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            coord = (x, y)
            if coord in existing:
                continue
            existing.add(coord)
            selected.append(coord)
            if len(selected) == num_patches:
                return selected

    raise ValueError("Unable to find enough unique patch positions without duplicates.")


def generate_patches_3d(
    img_arr,
    num_patches,
    obs,
    target_patch_size=(256, 256),
    context_patch_size=(1024, 32),
    x=None,
    y=None,
    x_border=PATCH_X_BORDER,
    existing_coords=None,
):
    """
    Input: Either a stacked 3D array (H, W, C) or five aligned 2D arrays
    ordered as [BG, IR, RED3, RED4, RED5].

    Returns:
    Three lists:

    - target_patches:  A list of length 'num_patches'; each patch  is (256, 256, 3) with channels [BG12, IR10, RED4].
                       Top-left corner decided by sampled (x, y) positions.
    - context_patches: A list of length 'num_patches'; 2 3D patches, each patch is (512, 32, 3) with channels [RED3, RED5] and overlapping colour arrs
                       Top row same as for target_patches but extends lower.
    - patch_positions: Top-left (x, y) position of each target patch in the cropped observation frame.
    Length of list is number of patches for this specific observation
    With sizes of 256x256 for target and 512x32 for context, the number of context pixels is half (32,768) of the target pixels (65,536).

    If x and y are provided, returns a single patch centered at that coordinate instead of random patches.
    """
    assert num_patches > 0, f"Number of patches must be greater than 0, got {num_patches}."

    bands, H, W = _coerce_patch_bands(img_arr)
    target_h, target_w = target_patch_size  # Default patch size for RED4, IR10, BG12
    context_h, context_w = context_patch_size  # Context size for RED3, RED5 overlaps
    bg_arr, ir_arr, r3_arr, r4_arr, r5_arr = bands

    w = int(context_w)

    # Determine patch starting coords
    assert W >= target_w and H >= target_h, f"Input image is too small to extract patches. Require size {target_h}x{target_w} and got {H}x{W}."

    if (x is None) != (y is None):
        raise ValueError("x and y must be provided together, or both be None")

    # Handle case where x and y are provided
    if x is not None and y is not None:
        assert num_patches == 1, "When providing x and y coordinates, num_patches must be 1."
        x = int(x)
        y = int(y)
        # Center the target patch around the given x, y coordinate
        target_x = x - target_w // 2
        target_y = y - target_h // 2

        # Enforce that the shared top-left y also allows the CONTEXT patch to fit.
        min_center_x = target_w // 2
        max_center_x = (W - target_w) + (target_w // 2)

        min_center_y = target_h // 2
        max_center_y = (H - context_h) + (target_h // 2)

        if not (min_center_x <= x <= max_center_x and min_center_y <= y <= max_center_y):
            raise ValueError(
                f"(x, y)=({x}, {y}) cannot be the center of a target patch while also fitting context_h={context_h}. "
                f"Valid center ranges: x in [{min_center_x}, {max_center_x}], y in [{min_center_y}, {max_center_y}]."
            )

        # Clamp to valid ranges
        target_x = max(0, min(target_x, W - target_w))
        target_y = max(0, min(target_y, H - context_h))
        selected_positions = [(target_x, target_y)]

    else:   # Randomly sample patches
        seed = int(hashlib.sha256(obs.encode("utf-8")).hexdigest()[:8], 16)  # Create a seed from the observation ID
        rng = np.random.default_rng(seed)  # Create a random number generator with the seed

        need_h = max(target_h, context_h)
        assert H >= need_h, f"Image height {H} is too small to fit target/context patch height {need_h}."
        min_target_x = int(x_border)
        max_target_x = W - target_w - int(x_border)
        max_target_y = H - context_h
        if max_target_x < min_target_x:
            raise ValueError(
                f"Image width {W} is too small to sample {target_w}px patches "
                f"while keeping {x_border}px clearance from each horizontal border."
            )

        selected_positions = _sample_unique_patch_positions(
            rng,
            num_patches,
            min_target_x,
            max_target_x,
            0,
            max_target_y,
            existing_coords=existing_coords,
        )

    r3_right = slice(W - w, W)
    bgir_left = slice(0, w)
    r5_left = slice(0, w)
    bgir_right = slice(W - w, W)

    target_patches = []
    context_patches = []
    patch_positions = []

    for target_x, target_y in selected_positions:
        y = target_y
        patch_positions.append((int(target_x), int(target_y)))

        target_patch = np.stack([
            bg_arr[y:y + target_h, target_x:target_x + target_w],
            ir_arr[y:y + target_h, target_x:target_x + target_w],
            r4_arr[y:y + target_h, target_x:target_x + target_w],
        ], axis=-1)
        target_patches.append(np.ascontiguousarray(target_patch))

        # R3 side overlap
        c3_bg = bg_arr[y:y + context_h, bgir_left]
        c3_ir = ir_arr[y:y + context_h, bgir_left]
        c3_r3 = r3_arr[y:y + context_h, r3_right]
        c3_r4 = r4_arr[y:y + context_h, bgir_left]
        context3_patch = np.stack([c3_bg, c3_ir, c3_r3, c3_r4], axis=-1)

        # R5 side overlap
        c5_bg = bg_arr[y:y + context_h, bgir_right]
        c5_ir = ir_arr[y:y + context_h, bgir_right]
        c5_r5 = r5_arr[y:y + context_h, r5_left]
        c5_r4 = r4_arr[y:y + context_h, bgir_right]
        context5_patch = np.stack([c5_bg, c5_ir, c5_r5, c5_r4], axis=-1)

        context3_patch = np.ascontiguousarray(context3_patch)
        context5_patch = np.ascontiguousarray(context5_patch)
        context_patches.append((context3_patch, context5_patch))

    assert all(p.shape == (target_h, target_w, 3) for p in target_patches)
    assert all(c3.shape == (context_h, context_w, 4) and c5.shape == (context_h, context_w, 4) for c3, c5 in context_patches)

    return target_patches, context_patches, patch_positions


def isis2np(cub_loc, img_save_loc=None):
    """Convert an ISIS cube file to a NumPy array."""
    assert cub_loc.endswith(".cub"), f"Input file must be a .cub file, not {cub_loc}"
    assert os.path.exists(cub_loc), f"File not found: {cub_loc}"
    metadata: Any = pvl.load(cub_loc)
    assert metadata is not None

    fmt = str(metadata["IsisCube"]["Core"].get("Format", "Tile"))
    if fmt != "Tile":
        raise ValueError(f"Unsupported Core.Format={fmt} (this reader assumes Tile)")

    img_dtype = str(metadata["IsisCube"]["Core"]["Pixels"]["Type"])
    byteorder = str(metadata["IsisCube"]["Core"]["Pixels"]["ByteOrder"])

    img_lines = metadata["IsisCube"]["Core"]["Dimensions"]["Lines"]
    img_samples = metadata["IsisCube"]["Core"]["Dimensions"]["Samples"]
    tile_samples = metadata["IsisCube"]["Core"]["TileSamples"]
    tile_lines = metadata["IsisCube"]["Core"]["TileLines"]
    img_start_byte = (metadata["IsisCube"]["Core"]["StartByte"] - 1)  # Adjust to 0-based index

    if img_dtype == "Real":
        img_bits = 32
    elif img_dtype == "SignedWord":
        img_bits = 16
    else:
        raise ValueError(f"Unhandled Pixel Type: {img_dtype}")
    sample_bytes = img_bits // 8

    if (byteorder == "Lsb" and img_dtype == "Real"):
        dtype = "<f4"  # Little-endian 32-bit float
    elif (byteorder == "Lsb" and img_dtype == "SignedWord"):
        dtype = "<i2"   # Little-endian 16-bit signed integer
    else:
        raise ValueError(
            f"Unsupported Type: {img_dtype} or ByteOrder: {byteorder}"
        )

    # ISIS pads tiled cubes out to whole tiles, so round the tile counts up.
    n_tiles_x = -(-img_samples // tile_samples)
    n_tiles_y = -(-img_lines // tile_lines)
    padded_samples = n_tiles_x * tile_samples
    padded_lines = n_tiles_y * tile_lines

    with open(cub_loc, "rb") as f:
        f.seek(img_start_byte)
        total_bytes = padded_lines * padded_samples * sample_bytes
        raw_data = f.read(total_bytes)
        if len(raw_data) != total_bytes:
            raise ValueError(
                f"{os.path.basename(cub_loc)}: expected {total_bytes} bytes of tile data "
                f"from byte {img_start_byte} ({n_tiles_y}x{n_tiles_x} tiles of "
                f"{tile_lines}x{tile_samples} for a {img_lines}x{img_samples} image), "
                f"read {len(raw_data)}"
            )
        raw_array = np.frombuffer(raw_data, dtype=dtype)
        tiles = raw_array.reshape(n_tiles_y, n_tiles_x, tile_lines, tile_samples)
        img_arr = tiles.transpose(0, 2, 1, 3).reshape(padded_lines, padded_samples)
        if (padded_lines, padded_samples) != (img_lines, img_samples):
            img_arr = img_arr[:img_lines, :img_samples]

    if img_dtype == "SignedWord":
        base = metadata["IsisCube"]["Core"]["Pixels"]["Base"]
        multiplier = metadata["IsisCube"]["Core"]["Pixels"]["Multiplier"]
        img_arr = img_arr.astype(np.float32) * multiplier + base

    if img_arr.min() < -1e38:
        print(
            f"Null pixel detected in {os.path.basename(cub_loc)}: {img_arr.min():.1e}. Setting to min."
        )
        img_arr[img_arr < -1e38] = min(img_arr[img_arr > -1e38])

    if img_arr.min() < -1:
        num_negs = np.sum(img_arr < -1)
        print(
            f"{num_negs} pixel values less than -1 detected in {os.path.basename(cub_loc)}. Min: {img_arr.min():.4f}. Setting all to -1."
        )
        img_arr[img_arr < -1] = -1

    if img_arr.max() > 1:
        print(
            f"Pixel value over 1 detected in {os.path.basename(cub_loc)}. Max: {img_arr.max():.4f}. Setting to 1."
        )
        img_arr[img_arr > 1] = 1

    if img_save_loc is not None:
        name = os.path.basename(cub_loc)[16:20]
        save_img(img_arr, img_save_loc, name, f"Calibrated Image: {name}", dpi=600)

    return img_arr


def crop_offset(
    stacked_arrs,
    thresh=0.01,
    eps=1e-3,
    use_missing=True,
    missing_value=-3.663e-4,
    min_keep=(16, 16),
    max_iters=10,
    min_edge_run=3,
    invalid_frac_thresh=0.85,
    crop_band_indices=(0, 1),
):
    """Find a single crop window by thresholding invalid fractions on rows/cols.

    Rows/columns are kept if their invalid fraction is below ``invalid_frac_thresh``
    for all crop-driving bands. By default the crop is driven only by BG12/IR10.
    """
    if isinstance(stacked_arrs, np.ndarray):
        if stacked_arrs.ndim != 3:
            raise ValueError(f"Expected a 3D stacked array, got shape {stacked_arrs.shape}.")
        H, W, _ = stacked_arrs.shape
        band_views = tuple(stacked_arrs[..., idx] for idx in range(stacked_arrs.shape[2]))
        return_stacked = True
    else:
        band_views = tuple(np.asarray(arr) for arr in stacked_arrs)
        if not band_views:
            raise ValueError("At least one band array is required for cropping.")
        if any(arr.ndim != 2 for arr in band_views):
            raise ValueError("All band arrays must be 2D.")
        shapes = {arr.shape for arr in band_views}
        if len(shapes) != 1:
            raise ValueError(f"All band arrays must share the same shape, got {sorted(shapes)}.")
        H, W = band_views[0].shape
        return_stacked = False

    crop_bands = tuple(band_views[idx] for idx in crop_band_indices)
    invalid_masks = [
        ~_valid_pixel_mask(
            edge,
            use_missing=use_missing,
            missing_value=missing_value,
            eps=eps,
        )
        for edge in crop_bands
    ]

    row_invalid_frac = np.maximum.reduce([mask.mean(axis=1) for mask in invalid_masks])
    col_invalid_frac = np.maximum.reduce([mask.mean(axis=0) for mask in invalid_masks])

    keep_rows = np.flatnonzero(row_invalid_frac < invalid_frac_thresh)
    keep_cols = np.flatnonzero(col_invalid_frac < invalid_frac_thresh)

    if keep_rows.size == 0 or keep_cols.size == 0:
        raise ValueError(
            "No crop window remained after thresholding invalid row/column fractions: "
            f"rows_kept={keep_rows.size}, cols_kept={keep_cols.size}, "
            f"invalid_frac_thresh={invalid_frac_thresh}"
        )

    top = int(keep_rows[0])
    bottom = int(keep_rows[-1]) + 1
    left = int(keep_cols[0])
    right = int(keep_cols[-1]) + 1

    if (bottom - top) < min_keep[0] or (right - left) < min_keep[1]:
        raise ValueError(
            f"Crop window became too small after thresholding invalid borders: "
            f"height={bottom - top}, width={right - left}, "
            f"invalid_frac_thresh={invalid_frac_thresh}"
        )

    if return_stacked:
        cropped = stacked_arrs[top:bottom, left:right, :]
    else:
        cropped = tuple(arr[top:bottom, left:right] for arr in band_views)
    return cropped, (top, bottom, left, right)


def compute_meta_dict(EDR_df, obs_ids):
    obs_meta = {}
    # Confirm all observations are present in EDR_df
    missing_obs = set(obs_ids) - set(EDR_df["OBSERVATION_ID"])
    if missing_obs:
        print(f"Error creating meta dict: Missing observations in EDR_df:\n{missing_obs}\n", flush=True)
    for obs, df in EDR_df[EDR_df["OBSERVATION_ID"].isin(obs_ids)].groupby("OBSERVATION_ID"):
        obs_meta[obs] = {
            "OBSERVATION_START_TIME": df["OBSERVATION_START_TIME"].iloc[0],
            "per_ccd": {row["CCD_NAME"]: (int(row["BINNING"]), int(row["TDI"])) for _, row in df.iterrows()},
            "CPMM_temp": df["CPMM_PWS_BOARD_TEMPERATURE"].iloc[0],
            "Emission_angle": df["EMISSION_ANGLE"].iloc[0],
            "FPA_negY_temp": df["FPA_NEGATIVE_Y_TEMPERATURE"].iloc[0],
            "FPA_posY_temp": df["FPA_POSITIVE_Y_TEMPERATURE"].iloc[0],
            "FPE_temp": df["FPE_TEMPERATURE"].iloc[0],
            "IEA_PWS_temp": df["IEA_PWS_BOARD_TEMPERATURE"].iloc[0],
            "IEA_temp": df["IEA_TEMPERATURE"].iloc[0],
            "Image_center_lat": df["IMAGE_CENTER_LATITUDE"].iloc[0],
            "Image_center_lon": df["IMAGE_CENTER_LONGITUDE"].iloc[0],
            "Incidence_angle": df["INCIDENCE_ANGLE"].iloc[0],
            "Local_time": df["LOCAL_TIME"].iloc[0],
            "North_azimuth": df["NORTH_AZIMUTH"].iloc[0],
            "Orbit_number": df["ORBIT_NUMBER"].iloc[0],
            "Phase_angle": df["PHASE_ANGLE"].iloc[0],
            "Slant_distance": df["SLANT_DISTANCE"].iloc[0],
            "Solar_distance": df["SOLAR_DISTANCE"].iloc[0],
            "Solar_longitude": df["SOLAR_LONGITUDE"].iloc[0],
            "Spacecraft_altitude": df["SPACECRAFT_ALTITUDE"].iloc[0],
            "Sub_solar_azimuth": df["SUB_SOLAR_AZIMUTH"].iloc[0],
            "Sub_solar_latitude": df["SUB_SOLAR_LATITUDE"].iloc[0],
            "Sub_solar_longitude": df["SUB_SOLAR_LONGITUDE"].iloc[0],
            "Sub_spacecraft_latitude": df["SUB_SPACECRAFT_LATITUDE"].iloc[0],
            "Sub_spacecraft_longitude": df["SUB_SPACECRAFT_LONGITUDE"].iloc[0],
            "Target_center_distance": df["TARGET_CENTER_DISTANCE"].iloc[0],
        }

    # Assert all obs in obs list inmeta dict
    missing_meta = set(obs_ids) - set(obs_meta.keys())
    if missing_meta:
        print(f"Error creating meta dict: Missing observations in meta dict:\n{missing_meta}\n", flush=True)
    else:
        print(f"Successfully created meta dict for all {len(obs_meta)} observations.", flush=True)
    return obs_meta


def _load_reusable_patch_records(data_record, obs_ids):
    if data_record is None or not os.path.exists(data_record):
        return {}, {}

    existing_df = pd.read_csv(data_record)
    if existing_df.empty:
        return {}, {}

    required_cols = {"Observation", "Patch", "CCD", "Path"}
    missing_required = [col for col in required_cols if col not in existing_df.columns]
    if missing_required:
        print(
            f"WARNING: Existing data_record is missing required columns {missing_required}. "
            "Existing manifest rows and patch coordinates will not be reused.",
            flush=True,
        )
        return {}, {}

    existing_df = existing_df[existing_df["Observation"].isin(obs_ids)].copy()
    if existing_df.empty:
        return {}, {}

    existing_df["Patch"] = existing_df["Patch"].astype(int)
    expected_ccds = set(PATCH_CCDS)
    reusable_records = {}
    patch_coords = {}
    missing_cols = [col for col in DATA_RECORD_COLUMNS if col not in existing_df.columns]
    can_reuse_records = not missing_cols

    if missing_cols:
        print(
            f"WARNING: Existing data_record is missing columns {missing_cols}. "
            "Existing manifest rows will not be reused, but existing patch coordinates will still be used when available.",
            flush=True,
        )

    for (obs, patch_idx), patch_df in existing_df.groupby(["Observation", "Patch"], sort=False):
        assert isinstance(patch_idx, (int, np.integer)), (
            f"Expected integer patch index, got {type(patch_idx).__name__}"
        )
        patch_idx = int(patch_idx)

        if {PATCH_X_COLUMN, PATCH_Y_COLUMN}.issubset(patch_df.columns):
            xy_df = patch_df[[PATCH_X_COLUMN, PATCH_Y_COLUMN]].drop_duplicates()
            if len(xy_df) == 1:
                patch_x = xy_df[PATCH_X_COLUMN].iloc[0]
                patch_y = xy_df[PATCH_Y_COLUMN].iloc[0]
                if pd.notna(patch_x) and pd.notna(patch_y):
                    patch_coords.setdefault(obs, {})[patch_idx] = (int(patch_x), int(patch_y))

        if not can_reuse_records:
            continue
        if len(patch_df) != len(PATCH_CCDS):
            continue
        if set(patch_df["CCD"]) != expected_ccds:
            continue

        reusable_records.setdefault(obs, {})[patch_idx] = list(
            patch_df.loc[:, DATA_RECORD_COLUMNS].itertuples(index=False, name=None)
        )

    return reusable_records, patch_coords


def _reset_record_set_values(obs, records_by_patch, n_patches, base_idx):
    rows_out = []
    for patch_idx in range(n_patches):
        patch_records = records_by_patch.get(patch_idx)
        if patch_records is None:
            raise ValueError(f"Missing reusable manifest rows for observation {obs}, patch {patch_idx}.")
        set_value = base_idx + patch_idx
        for record in patch_records:
            row = list(record)
            row[SET_COLUMN_INDEX] = set_value
            rows_out.append(row)
    return rows_out


def processed_cubes(img_base_addr, obs):
    """
    Return the sorted ``*_processed.cub`` files already on disk for an observation.

    A complete set has one cube per entry in PATCH_CCDS; anything else means the
    HiPROC pipeline still has to run. Callers can use this to work out up front
    whether the hiproc+isis backend is needed at all.
    """
    file_loc = os.fspath(Path(img_base_addr) / obs)
    return sorted(glob.glob(f"{file_loc}/*_processed.cub"))


def processed_cubes_complete(img_base_addr, obs):
    """True when every PATCH_CCDS cube for this observation is already processed."""
    return len(processed_cubes(img_base_addr, obs)) == len(PATCH_CCDS)


def snapshot_hiproc_env():
    """
    Activate the shared hiproc+isis backend ONCE in a subshell, then return env_dict suitable for env in subprocess.run().
    """
    import json

    py_probe = r"""
import os, json
print(json.dumps(dict(os.environ)))
"""

    activate_script = PACKAGE_ROOT / "data" / "activate_hiproc_backend.sh"    # Change environment
    if not activate_script.is_file():
        raise FileNotFoundError(f"Shared backend activation script not found: {activate_script}")

    bash = rf"""
set -eo pipefail
source {shlex.quote(str(activate_script))}
python - <<'PY'
{py_probe}
PY
"""
    try:
        p = subprocess.run(["bash", "-lc", bash], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Snapshot hiproc+isis backend failed")
        print(f"\tstdout: {e.stdout}")
        print(f"\tstderr: {e.stderr}")
        raise

    try:
        env_json = p.stdout.strip()
        env_dict = json.loads(env_json.splitlines()[-1])
    except Exception as e:
        print("Snapshot failed at JSON parsing")
        print(f"Error: {e}")
        print(f"\tstdout: {p.stdout}")
        print(f"\tstderr: {p.stderr}")
        raise

    env_dict.setdefault("HIPREDICT_LOG_DIR", str(files_dir() / "hiproc_logs"))

    # Avoid oversubscription when running many workers
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env_dict.pop(var, None)  # Remove these variables to avoid oversubscription

    return env_dict


def compute_observation(np_save, img_base_addr, obs, calib_path, npy_exists, n_patches, base_idx, existing_patch_count=0, existing_patch_coords=None, patch_x=None, patch_y=None, force_cub_gen=False, gen_np_patches=True, reuse_existing_records=False, verbose=False):
    """
    Returns error codes:
    0: success
    1: fail - add to fail list
    2: fail - dont add to fail list
    """
    global _OBS_META, _HIPROC_ENV
    ccd_order = list(PATCH_CCDS)
    assert _OBS_META is not None
    meta = _OBS_META.get(obs)
    if meta is None:
        print(f"ERROR: Missing metadata for observation {obs}. Skipping.")
        return {obs: 2}, None

    # Check if HIPROC files exist - run hiproc_pipeline if not
    file_loc = os.fspath(Path(img_base_addr) / obs)
    cubefiles = processed_cubes(img_base_addr, obs)
    cubefiles_exist = len(cubefiles) == len(PATCH_CCDS)

    if force_cub_gen:
        cubefiles_exist = False  # FORCE CUBEFILE REGENERATION

    if not cubefiles_exist:
        # env=None would silently inherit the caller's environment and fail deep
        # inside HiPROC. If the backend was never snapshotted, this observation
        # simply cannot be processed - say so and skip it.
        if _HIPROC_ENV is None:
            print(
                f"ERROR: Observation {obs} needs the hiproc+isis backend "
                f"({len(cubefiles)}/{len(PATCH_CCDS)} processed cubes in {file_loc}), "
                f"but no backend environment is available. Skipping."
            )
            return {obs: 2}, None

        try:
            subprocess.run(
                [calib_path, obs, file_loc],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                env=_HIPROC_ENV,
            )
        except subprocess.CalledProcessError as e:
            err_code = e.returncode
            stderr = e.stderr or ""
            print(f"[HIPROC ] Error processing obs {obs} - RC: {err_code} - stderr: {stderr}")

            if err_code > 127:
                return {obs: 2}, None
            if err_code in (10, 13):
                return {obs: 2}, None
            if err_code in (11, 12):
                return {obs: 1}, None
            return {obs: 2}, None

        cubefiles = processed_cubes(img_base_addr, obs)

    if len(cubefiles) != len(PATCH_CCDS):
        print(f"ERROR: Expected {len(PATCH_CCDS)} processed cubefiles for {obs}, found {len(cubefiles)}.")
        return {obs: 2}, None

    if not gen_np_patches:
        return {obs: 0}, None

    if np_save is None:
        raise ValueError("np_save must be provided when patch generation is enabled")

    if reuse_existing_records and npy_exists:
        return {obs: 0}, None
    existing_patch_count = int(existing_patch_count)
    existing_patch_coords = dict(existing_patch_coords or {})

    bg_img_hiproc = isis2np(cubefiles[0], img_save_loc=None)
    ir_img_hiproc = isis2np(cubefiles[1], img_save_loc=None)
    r3_img_hiproc = isis2np(cubefiles[2], img_save_loc=None)
    r4_img_hiproc = isis2np(cubefiles[3], img_save_loc=None)
    r5_img_hiproc = isis2np(cubefiles[4], img_save_loc=None)

    for img in [bg_img_hiproc, ir_img_hiproc, r3_img_hiproc, r4_img_hiproc, r5_img_hiproc]:
        assert img.ndim == 2, f"Found converted np array not in 2D. Ndims: {img.ndim}"

    try:
        cropped_bands, crop_dims = crop_offset(
            (bg_img_hiproc, ir_img_hiproc, r3_img_hiproc, r4_img_hiproc, r5_img_hiproc),
            invalid_frac_thresh=0.85,
            crop_band_indices=(0, 1),
        )
    except ValueError as e:
        print(f"ERROR: Failed crop for obs {obs}: {e}. Skipping.")
        return {obs: 2}, None

    try:
        obs_center, obs_scale = _robust_center_scale_from_predictors_np(
            cropped_bands[0],
            cropped_bands[1],
        )
    except ValueError as e:
        print(f"ERROR: Failed predictor normalization for obs {obs}: {e}. Skipping.")
        return {obs: 2}, None

    if (patch_x is None) != (patch_y is None):
        raise ValueError("patch_x and patch_y must be provided together, or both be None")

    patch_x_c = None
    patch_y_c = None
    if patch_x is not None and patch_y is not None:
        crop_top, crop_bottom, crop_left, crop_right = crop_dims
        patch_x_i = int(patch_x)
        patch_y_i = int(patch_y)

        if not (crop_left <= patch_x_i < crop_right and crop_top <= patch_y_i < crop_bottom):
            print(
                f"ERROR: Requested (patch_x, patch_y)=({patch_x_i}, {patch_y_i}) is outside "
                f"the kept crop window x:[{crop_left},{crop_right}) y:[{crop_top},{crop_bottom}) for {obs}."
            )
            return {obs: 2}, None

        patch_x_c = patch_x_i - crop_left
        patch_y_c = patch_y_i - crop_top

    new_H, new_W = cropped_bands[0].shape
    if new_H < 1024 or new_W < 256:
        print(f"ERROR: Cropped image for obs {obs} is too small: {new_H}x{new_W}. Skipping.")
        return {obs: 2}, None

    patch_list = [[] for _ in range(5)]
    patch_positions = []
    for i in range(existing_patch_count):
        for ccd_idx, ccd in enumerate(ccd_order):
            path = os.path.join(np_save, f"{obs}_{i}_{ccd}.npy")
            if not os.path.exists(path):
                print(f"ERROR: {path} not found for observation {obs}. Skipping.")
                return {obs: 2}, None
            patch = np.load(path)
            patch_list[ccd_idx].append(patch)
        patch_positions.append(existing_patch_coords.get(i))

    new_patch_count = n_patches - existing_patch_count
    if new_patch_count > 0:
        target_patches, context_patches, new_positions = generate_patches_3d(
            cropped_bands,
            new_patch_count,
            obs,
            x=patch_x_c,
            y=patch_y_c,
            existing_coords=[existing_patch_coords.get(i) for i in range(existing_patch_count)],
        )
        assert len(target_patches) == len(context_patches) == len(new_positions) == new_patch_count, (
            f"Expected {new_patch_count} patches but got {len(target_patches)} target, "
            f"{len(context_patches)} context, {len(new_positions)} position entries."
        )

        for t, (c3, c5), (patch_x_i, patch_y_i) in zip(target_patches, context_patches, new_positions):
            patch_list[0].append(t[..., 0])
            patch_list[1].append(t[..., 1])
            patch_list[2].append(c3)
            patch_list[3].append(t[..., 2])
            patch_list[4].append(c5)
            patch_positions.append((patch_x_i, patch_y_i))

    assert all(len(lst) == n_patches for lst in patch_list), f"Expected {n_patches} patches per CCD, but found {[len(lst) for lst in patch_list]}."
    assert len(patch_list) == 5, f"Expected 5 CCDs, but found {len(patch_list)}."
    assert len(patch_positions) == n_patches, f"Expected {n_patches} patch positions, found {len(patch_positions)}."

    ccd_names = {
        0: "BG12",
        1: "IR10",
        2: "RED3",
        3: "RED4",
        4: "RED5",
    }

    if verbose:
        missing_val = -3.663e-4
        decimals = 7
        eps = 5e-4
        mv_round = np.around(missing_val, decimals)
        context_chan_names = {
            "RED3": ["BG12", "IR10", "RED3", "RED4"],
            "RED5": ["BG12", "IR10", "RED5", "RED4"],
        }

        for ccd_idx, patches in enumerate(patch_list):
            ccd = ccd_names[ccd_idx]
            for i, patch in enumerate(patches):
                arr = np.asarray(patch)

                if arr.ndim == 2:
                    mv_mask = (np.around(arr, decimals) == mv_round)
                    eps_mask = (np.abs(arr) <= eps)
                    n_mv = int(mv_mask.sum())
                    n_eps = int(eps_mask.sum())
                    if n_mv or n_eps:
                        print(f"[{obs}] {ccd}: patch {i}: ==MV(round7)={n_mv}, <=EPS(5e-4)={n_eps}")

                elif arr.ndim == 3 and arr.shape[-1] == 4 and ccd in ("RED3", "RED5"):
                    for ch, ch_name in enumerate(context_chan_names[ccd]):
                        plane = arr[..., ch]
                        mv_mask = (np.around(plane, decimals) == mv_round)
                        eps_mask = (np.abs(plane) <= eps)
                        n_mv = int(mv_mask.sum())
                        n_eps = int(eps_mask.sum())
                        if n_mv or n_eps:
                            print(f"[{obs}] {ccd}_{ch_name}: patch {i}: ==MV(round7)={n_mv}, <=EPS(5e-4)={n_eps}")
                else:
                    raise ValueError(f"Unexpected array shape for {obs} patch {i} CCD {ccd}: {arr.shape}")

    # Now we have variable 'patch_list' which is a list of lists:
    # patch_list[0] = [patch_BG12_0, patch_BG12_1, ..., patch_BG12_n]
    # patch_list[1] = [patch_IR10_0, patch_IR10_1, ..., patch_IR10_n]
    # etc...

    date = meta["OBSERVATION_START_TIME"]
    records_out = []

    # Go through each list of patches, update data record and save if needed
    for ccd_num, ccd_patch_list in enumerate(patch_list):
        # Get metadata for this CCD
        ccd = ccd_names[ccd_num]
        binning, tdi = meta["per_ccd"][ccd]

        pix_min = min(p.min() for p in ccd_patch_list)
        pix_max = max(p.max() for p in ccd_patch_list)
        h = ccd_patch_list[0].shape[0]
        w = ccd_patch_list[0].shape[1]
        c = ccd_patch_list[0].shape[2] if ccd_patch_list[0].ndim == 3 else 1

        # Save patches and update data record
        for j, patch in enumerate(ccd_patch_list):
            path = os.path.join(np_save, f"{obs}_{j}_{ccd}.npy")
            if j >= existing_patch_count:
                np.save(path, patch)  # noqa

            set_value = base_idx + j
            patch_x_val, patch_y_val = patch_positions[j] if patch_positions[j] is not None else (np.nan, np.nan)
            record = [obs,
                      set_value,
                      j,
                      ccd,
                      path,
                      h,
                      w,
                      c,
                      date,
                      pix_min,
                      pix_max,
                      binning,
                      tdi,
                      meta["CPMM_temp"],
                      meta["Emission_angle"],
                      meta["FPA_negY_temp"],
                      meta["FPA_posY_temp"],
                      meta["FPE_temp"],
                      meta["IEA_PWS_temp"],
                      meta["IEA_temp"],
                      meta["Image_center_lat"],
                      meta["Image_center_lon"],
                      meta["Incidence_angle"],
                      meta["Local_time"],
                      meta["North_azimuth"],
                      meta["Orbit_number"],
                      meta["Phase_angle"],
                      meta["Slant_distance"],
                      meta["Solar_distance"],
                      meta["Solar_longitude"],
                      meta["Spacecraft_altitude"],
                      meta["Sub_solar_azimuth"],
                      meta["Sub_solar_latitude"],
                      meta["Sub_solar_longitude"],
                      meta["Sub_spacecraft_latitude"],
                      meta["Sub_spacecraft_longitude"],
                      meta["Target_center_distance"],
                      patch_x_val,
                      patch_y_val,
                      obs_center,
                      obs_scale,
                      ]
            records_out.append(record)

    return {obs: 0}, records_out


def _wrap_compute(*args, **kwargs):
    obs = kwargs.get("obs", args[2] if len(args) > 2 else "unknown")
    try:
        result, record = compute_observation(*args, **kwargs)
    except Exception as e:
        np_save = kwargs.get("np_save", args[0] if len(args) > 0 else None)
        img_base_addr = kwargs.get("img_base_addr", args[1] if len(args) > 1 else None)
        n_patches = kwargs.get("n_patches", args[5] if len(args) > 5 else None)
        base_idx = kwargs.get("base_idx", args[6] if len(args) > 6 else None)
        print(
            "ERROR in compute_observation "
            f"for obs {obs} "
            f"(img_base_addr={img_base_addr}, np_save={np_save}, "
            f"n_patches={n_patches}, base_idx={base_idx}): "
            f"{type(e).__name__}: {e}",
            flush=True,
        )
        raise
    return result, record


def _is_relative_to(path, base):
    try:
        path.relative_to(base)
    except ValueError:
        return False
    return True


def _format_sample_paths(paths, max_count=5):
    paths = list(paths)
    suffix = "" if len(paths) <= max_count else f" ... and {len(paths) - max_count:,} more"
    return ", ".join(str(path) for path in paths[:max_count]) + suffix


def _prune_orphan_npy_files(np_save_base, data_record):
    np_root = Path(np_save_base).resolve()
    if not np_root.is_dir():
        raise FileNotFoundError(f"np_save_base does not exist or is not a directory: {np_root}")

    manifest_df = pd.read_csv(data_record)
    if "Path" not in manifest_df.columns:
        raise ValueError(f"Cannot prune orphan .npy files because {data_record} has no Path column.")

    keep_paths = {
        _resolve_record_path(data_record, record_path).resolve()
        for record_path in manifest_df["Path"].dropna().unique()
    }

    outside_paths = sorted(path for path in keep_paths if not _is_relative_to(path, np_root))
    if outside_paths:
        raise ValueError(
            f"Refusing to prune because {len(outside_paths):,} manifest path(s) are outside "
            f"np_save_base={np_root}: {_format_sample_paths(outside_paths)}"
        )

    missing_paths = sorted(path for path in keep_paths if not path.is_file())
    if missing_paths:
        raise FileNotFoundError(
            f"Refusing to prune because {len(missing_paths):,} manifest path(s) are missing: "
            f"{_format_sample_paths(missing_paths)}"
        )

    orphan_paths = []
    for path in np_root.rglob("*.npy"):
        if not path.is_file():
            continue
        if path.resolve() not in keep_paths:
            orphan_paths.append(path)

    deleted_files = 0
    deleted_bytes = 0
    for path in orphan_paths:
        try:
            deleted_bytes += path.stat().st_size
            path.unlink()
            deleted_files += 1
        except FileNotFoundError:
            continue

    removed_dirs = 0
    dirs = sorted(
        (path for path in np_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    for directory in dirs:
        try:
            if not any(directory.iterdir()):
                directory.rmdir()
                removed_dirs += 1
        except OSError:
            continue

    return deleted_files, deleted_bytes, removed_dirs


def run_processing(
    data_record,
    n_workers,
    img_base_addr,
    obs_ids,
    np_save_base,
    n_ccds,
    hiproc_fail_txt=None,
    n_patches=1,
    remaining_patches=0,
    force_cub_gen=False,
    force_npy_gen=False,
    prune_orphan_npy=False,
    gen_np_patches=True,
    calib_path=str(PACKAGE_ROOT / "data" / "hiproc_pipe.sh"),
    flush_every=10_000,
    verbose=False,
    patch_targets=None,
    existing_patch_sets=None,
):
    """Convert all observations in a list to patched NumPy arrays and save each image from each array."""

    # global DATA_RECORD
    start_time = time.time()
    img_base_addr = str(Path(img_base_addr).resolve())
    assert n_ccds == 5, "Only 5 CCDs are currently supported."
    ccd_list = ["RED3", "RED4", "RED5", "IR10", "BG12"]
    if verbose:
        print(f"\nProcessing for {n_ccds} ccds: {ccd_list}")

    assert isinstance(obs_ids, list), "Observation IDs must be provided as a list."
    patch_targets = dict(patch_targets or {})
    existing_patch_sets = dict(existing_patch_sets or {})

    def _count_complete_patch_sets(np_save, obs):
        patch_sets = 0
        while all(
            os.path.isfile(os.path.join(np_save, f"{obs}_{patch_sets}_{ccd}.npy"))
            for ccd in ccd_list
        ):
            patch_sets += 1
        return patch_sets

    if gen_np_patches:
        if np_save_base is None:
            raise ValueError("np_save_base must be provided when gen_np_patches=True")
        if data_record is None:
            raise ValueError("data_record must be provided when gen_np_patches=True")
        np_save_base = str(Path(np_save_base).resolve())
        data_record = str(Path(data_record).resolve())
        Path(data_record).parent.mkdir(parents=True, exist_ok=True)
        os.makedirs(np_save_base, exist_ok=True)
    else:
        if force_npy_gen:
            print("WARNING: force_npy_gen=True ignored because patch generation is disabled.")
        if prune_orphan_npy:
            print("WARNING: prune_orphan_npy=True ignored because patch generation is disabled.")
        if np_save_base is not None:
            print(f"Patch generation disabled; np_save_base={np_save_base!r} will not be used.")
        if data_record is not None:
            print(f"Patch generation disabled; data_record={data_record!r} will not be written.")

    EDR_df = gen_edr_df()

    calib_path = os.fspath(Path(calib_path).resolve())
    print("\n")

    reusable_records_by_obs = {}
    existing_patch_coords_by_obs = {}
    if gen_np_patches:
        reusable_records_by_obs, existing_patch_coords_by_obs = _load_reusable_patch_records(data_record, set(obs_ids))
        # Create data record (wipe first if already exists)
        if os.path.exists(data_record):
            os.remove(data_record)
        with open(data_record, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(DATA_RECORD_COLUMNS)
    running_base = 0
    obs_plans = []

    for i, obs in enumerate(obs_ids):
        np_save = None
        npy_exists = False
        temp_patches = 0
        complete_patch_sets = 0
        reuse_existing_records = False

        if gen_np_patches:
            np_save = os.path.join(np_save_base, obs)
            os.makedirs(np_save, exist_ok=True)

            if patch_targets:
                temp_patches = int(patch_targets[obs])
            else:
                temp_patches = n_patches + 1 if i < remaining_patches else n_patches
            complete_patch_sets = existing_patch_sets.get(obs)
            if complete_patch_sets is None:
                complete_patch_sets = _count_complete_patch_sets(np_save, obs)
            if force_npy_gen:
                complete_patch_sets = 0
            npy_exists = complete_patch_sets >= temp_patches

            if npy_exists:
                obs_reusable_records = reusable_records_by_obs.get(obs, {})
                reuse_existing_records = all(patch_idx in obs_reusable_records for patch_idx in range(temp_patches))
                if reuse_existing_records:
                    print(f"Observation {obs} already processed. Reusing existing np files and data record rows.")
                else:
                    print(f"Observation {obs} already processed. Using existing np files.")

        obs_plans.append(
            {
                "obs": obs,
                "np_save": np_save,
                "npy_exists": npy_exists,
                "temp_patches": temp_patches,
                "base_idx": running_base,
                "existing_patch_count": min(complete_patch_sets, temp_patches),
                "existing_patch_coords": existing_patch_coords_by_obs.get(obs, {}),
                "reuse_existing_records": reuse_existing_records,
            }
        )
        if gen_np_patches:
            running_base += temp_patches

    # Build tiny metadata only for the obs we care about
    obs_set = set(obs_ids)
    obs_meta = compute_meta_dict(EDR_df, obs_set)

    needs_backend = force_cub_gen or any(not processed_cubes_complete(img_base_addr, obs) for obs in obs_ids)
    HIPROC_ENV = snapshot_hiproc_env() if needs_backend else None

    with ProcessPoolExecutor(max_workers=n_workers,
                             initializer=init_worker,
                             initargs=(obs_meta, HIPROC_ENV)) as executor:
        futures = {}

        for plan in obs_plans:
            future = executor.submit(
                    _wrap_compute,
                    plan["np_save"],
                    img_base_addr,
                    plan["obs"],
                    calib_path,
                    plan["npy_exists"],
                    plan["temp_patches"],
                    base_idx=plan["base_idx"],
                    existing_patch_count=plan["existing_patch_count"],
                    existing_patch_coords=plan["existing_patch_coords"],
                    force_cub_gen=force_cub_gen,
                    gen_np_patches=gen_np_patches,
                    reuse_existing_records=plan["reuse_existing_records"],
                    verbose=verbose,
                )
            futures[future] = plan

        results = {}
        print(f"All futures submitted after {time.time() - start_time:.1f}s.")

        if gen_np_patches:
            with open(data_record, "a", newline="") as f:
                writer = csv.writer(f)
                for i, future in enumerate(as_completed(futures)):
                    plan = futures[future]
                    try:
                        result, records = future.result()
                        results.update(result)
                        if records:
                            for record in records:
                                record[4] = os.path.relpath(record[4], Path(data_record).parent)
                            writer.writerows(records)
                        elif plan["reuse_existing_records"] and result.get(plan["obs"]) == 0:
                            writer.writerows(
                                _reset_record_set_values(
                                    plan["obs"],
                                    reusable_records_by_obs.get(plan["obs"], {}),
                                    plan["temp_patches"],
                                    plan["base_idx"],
                                )
                            )

                        if (i + 1) % flush_every == 0:
                            f.flush()
                    except Exception as e:
                        print(f"Error: {e}")
                        traceback.print_exc()
                        raise
        else:
            for future in as_completed(futures):
                try:
                    result, _ = future.result()
                    results.update(result)
                except Exception as e:
                    print(f"Error: {e}")
                    traceback.print_exc()
                    raise

    print(f"Futures computed after {time.time() - start_time:.1f}s.")
    succeeded_obs  = [obs for obs, code in results.items() if code == 0]
    failed_obs     = [obs for obs, code in results.items() if code == 1]    # Only obs with exit code 1 are added to fail list
    nonfatal_fails = [obs for obs, code in results.items() if code == 2]
    stage = "patch generation" if gen_np_patches else "cubefile preparation"
    print(f"\n--- Results ({stage}) ---")
    print(f"  {len(succeeded_obs):>3} succeeded : {succeeded_obs}")
    print(f"  {len(failed_obs):>3} failed    (fatal, will be added to fail list): {failed_obs}")
    print(f"  {len(nonfatal_fails):>3} failed    (nonfatal, not added to fail list) : {nonfatal_fails}")
    print("---")

    if hiproc_fail_txt is not None:
        hiproc_fail_txt = Path(hiproc_fail_txt)
        existing = set()
        if hiproc_fail_txt.exists():
            with open(hiproc_fail_txt, "r", encoding="utf-8") as f:
                existing = {line.rstrip("\n") for line in f}

        new_fails = [obs for obs in failed_obs if obs not in existing]

        print(f"Computing fails after {time.time() - start_time:.1f}s.")
        if new_fails:
            hiproc_fail_txt.parent.mkdir(parents=True, exist_ok=True)
            with open(hiproc_fail_txt, "a", encoding="utf-8") as f:
                for obs in new_fails:
                    f.write(f"{obs}\n")

        print(f"Logged {len(new_fails)} new failed obs to {hiproc_fail_txt} after {time.time() - start_time:.1f}s.")
    else:
        print(
            f"hiproc_fail_txt not provided; skipping fail-list logging after {time.time() - start_time:.1f}s."
        )

    if gen_np_patches:
        if patch_targets:
            exp_sets = sum(int(patch_targets[obs]) for obs in obs_ids)
        else:
            exp_sets = (len(obs_ids) * n_patches) + remaining_patches
        true_sets = pd.read_csv(data_record)["Set"].nunique()
        print(f"\nWritten {true_sets:,} sets of patches out of {exp_sets:,} expected.")
        if patch_targets:
            patch_counts = [int(patch_targets[obs]) for obs in obs_ids]
            print(
                f"Requested patch targets ranged from {min(patch_counts):,} to {max(patch_counts):,} "
                f"across {len(patch_counts):,} observations."
            )
        elif remaining_patches > 0:
            print(
                f"Created {n_patches:,} patch(es) per obs for {len(obs_ids):,} plus an additional patch for {remaining_patches:,} observations."
            )
        else:
            print(f"Created {n_patches:,} patch(es) per obs for {len(obs_ids):,} observations with no additional patches.")
        print(f"Data record written to: {data_record}")

        if prune_orphan_npy:
            if failed_obs:
                print(
                    "Pruning orphan .npy files despite "
                    f"{len(failed_obs):,} failed observation(s). "
                    "The data record is the source of truth for retained patch files."
                )
            if nonfatal_fails:
                print(
                    "Pruning orphan .npy files despite "
                    f"{len(nonfatal_fails):,} nonfatal failed observation(s). "
                    "The data record is the source of truth for retained patch files."
                )
            if true_sets != exp_sets:
                print(
                    f"Pruning orphan .npy files even though the data record has {true_sets:,} set(s), "
                    f"but {exp_sets:,} were expected. The data record is the source of truth."
                )
            try:
                deleted_files, deleted_bytes, removed_dirs = _prune_orphan_npy_files(np_save_base, data_record)
                deleted_gib = deleted_bytes / (1024 ** 3)
                print(
                    f"Pruned {deleted_files:,} orphan .npy file(s) from {np_save_base} "
                    f"({deleted_gib:.2f} GiB) and removed {removed_dirs:,} empty observation directories."
                )
            except (FileNotFoundError, ValueError) as e:
                print(f"WARNING: Skipping orphan .npy pruning: {e}")

        print(f"Patch numpy files saved under: {np_save_base}")
    else:
        print("\nPatch generation disabled; no patch sets or data_record were written.")

    print(f"Processed {len(obs_ids)} observations in {(time.time() - start_time) / 60:.2f} mins.")
    return


def _resolve_record_path(data_record, record_path):
    record_path = Path(record_path)
    if record_path.exists():
        return record_path.resolve()
    return (Path(data_record).resolve().parent / record_path).resolve()


