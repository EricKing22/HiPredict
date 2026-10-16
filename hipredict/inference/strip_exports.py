"""Disk exports for the strip prediction workflow (.npy / .tif / .cub).

Takes the predicted RED4 arrays produced by the main workflow and writes them
out in whichever formats each observation's config entry requests. Also owns the
canonical model identities (main reconstruction model + SYN4 baseline) and the
file-name tags they map to, since naming saved outputs is an export concern.
"""

import os

import numpy as np

from hipredict.inference.utils import np_to_tiff, np_to_cube
from hipredict.paths import files_dir


# Main reconstruction model and SYN4 baseline. The `npy`/`tif`/`cub` config
# fields select which of these to export per format by model name.
MAIN_MODEL = "hipredict"
SYN4_MODEL = "SYN4"
MODEL_FILE_TAGS = {MAIN_MODEL: "HiPredict", "unet_conv_b22": "HiPredict", SYN4_MODEL: "SYN4"}

# Directory of neighbour RED5 cubes used as the geometry reference when writing
# .cub exports. This data lives outside the repo, so it is machine-specific:
# override with the HIPREDICT_NEIGH_CUBE_DIR environment variable.
NEIGH_CUBE_DIR = os.environ.get(
    "HIPREDICT_NEIGH_CUBE_DIR",
    str(files_dir() / "isis_imgs" / "observations"),
)


def resolve_model_tag(model_name):
    return MODEL_FILE_TAGS.get(model_name, model_name)


def normalise_model_list(entry, key):
    """Read entry[key] (npy/tif/cub) into a de-duplicated list of model names."""
    cfg = entry.get(key, None)

    if cfg is None or cfg is False:
        return []
    if cfg is True:
        # Back-compat: a bare True used to mean "save the main model".
        return [MAIN_MODEL]
    if isinstance(cfg, str):
        models = [cfg]
    elif isinstance(cfg, (list, tuple)):
        models = [str(model) for model in cfg]
    else:
        raise TypeError(
            f"{entry['id']}: {key} must be a model name, a list of model names, or false. "
            f"Got {cfg!r}"
        )

    models = [MAIN_MODEL if name == "unet_conv_b22" else name for name in models]
    return list(dict.fromkeys(models))


def normalise_band_counts(entry, key, default=1):
    """Read entry[key] (npy_bands/tif_bands) into a de-duplicated list of 1/3."""
    band_cfg = entry.get(key, default)

    if isinstance(band_cfg, int):
        band_counts = [band_cfg]
    elif isinstance(band_cfg, (list, tuple)):
        band_counts = list(band_cfg)
    else:
        raise TypeError(
            f"{entry['id']}: {key} must be 1, 3, or a list containing 1/3. "
            f"Got {band_cfg!r}"
        )

    if not band_counts:
        raise ValueError(f"{entry['id']}: {key} must not be empty.")

    try:
        band_counts = [int(band) for band in band_counts]
    except (TypeError, ValueError) as e:
        raise TypeError(
            f"{entry['id']}: {key} must contain only integers 1 or 3. "
            f"Got {band_cfg!r}"
        ) from e

    invalid = [band for band in band_counts if band not in (1, 3)]
    if invalid:
        raise ValueError(
            f"{entry['id']}: unsupported {key}={band_cfg!r}. Only 1 and 3 are supported."
        )

    return list(dict.fromkeys(band_counts))


def build_export_array(output_img, ir_img, bg_img, band_count):
    if band_count == 1:
        return output_img
    if band_count == 3:
        # Save 3-band strips in IR10 / predicted RED4 / BG12 order.
        return np.dstack([ir_img, output_img, bg_img])
    raise ValueError(f"Unsupported band_count: {band_count}")


def restore_crop(arr, original_shape, crop_dims, fill_value=np.nan):
    top, bottom, left, right = [int(x) for x in crop_dims]
    x = np.asarray(arr)
    if x.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D array to restore, got shape {x.shape}")

    crop_shape = (bottom - top, right - left)
    if x.shape[:2] != crop_shape:
        raise ValueError(
            f"Cannot restore crop with shape {x.shape[:2]} into crop window {crop_shape}"
        )

    full_shape = tuple(int(v) for v in original_shape)
    out_shape = full_shape if x.ndim == 2 else full_shape + (x.shape[-1],)
    out = np.full(out_shape, fill_value, dtype=x.dtype)
    out[top:bottom, left:right, ...] = x
    return out


def build_full_export_array(output_img, ir_img, bg_img, sample, band_count):
    export_arr = build_export_array(output_img, ir_img, bg_img, band_count)
    return restore_crop(
        export_arr, sample["original_shape"], sample["crop_dims"]
    )


def band_suffix(band_count):
    return "" if band_count == 1 else f"_{band_count}band"


def save_observation_exports(
    entry,
    sample,
    pred_arrays,
    r4_true,
    npy_band_counts,
    tif_band_counts,
    save_root,
    verbose,
    img_base_dir=None,
):
    """Save the configured per-format exports for a single observation.

    ``pred_arrays`` maps model name -> predicted RED4 array. The entry's
    ``npy``/``tif``/``cub`` fields each list the model names to export for that
    format (see normalise_model_list).
    """
    obs = entry["id"]
    npy_models = normalise_model_list(entry, "npy")
    tif_models = normalise_model_list(entry, "tif")
    cub_models = normalise_model_list(entry, "cub")

    def _pred_for(model_name, fmt):
        if model_name not in pred_arrays:
            print(
                f"WARNING: {obs}: requested {fmt} model {model_name!r} is not available; "
                f"skipping. Available: {sorted(pred_arrays)}"
            )
            return None
        return pred_arrays[model_name]

    if npy_models:
        strips_dir = os.path.join(save_root, "strips")
        os.makedirs(strips_dir, exist_ok=True)
        for model_name in npy_models:
            pred_img = _pred_for(model_name, "npy")
            if pred_img is None:
                continue
            tag = resolve_model_tag(model_name)
            for band_count in npy_band_counts:
                export_arr = build_full_export_array(
                    pred_img,
                    sample["ir"],
                    sample["bg"],
                    sample,
                    band_count,
                )
                out_path = os.path.join(strips_dir, f"{obs}_{tag}{band_suffix(band_count)}.npy")
                np.save(out_path, export_arr)
                if verbose:
                    print(f"Saved {obs} {tag} as {band_count}-band NumPy array at {out_path}")

        # Always-saved single-band reference arrays (predicted RED4, truth, SYN4).
        pred_out_path = os.path.join(strips_dir, f"{obs}_HiPredict_red.npy")
        true_out_path = os.path.join(strips_dir, f"{obs}_RED4_true.npy")
        syn4_out_path = os.path.join(strips_dir, f"{obs}_SYN4_red.npy")

        pred_full = restore_crop(
            pred_arrays[MAIN_MODEL],
            sample["original_shape"],
            sample["crop_dims"],
        )
        true_full = restore_crop(
            r4_true,
            sample["original_shape"],
            sample["crop_dims"],
        )
        syn4_full = restore_crop(
            pred_arrays[SYN4_MODEL],
            sample["original_shape"],
            sample["crop_dims"],
        )

        np.save(pred_out_path, pred_full.astype(np.float32, copy=False))
        np.save(true_out_path, true_full.astype(np.float32, copy=False))
        np.save(syn4_out_path, syn4_full.astype(np.float32, copy=False))

        if verbose:
            print(f"Saved {obs} HiPred RED4 array at {pred_out_path}")
            print(f"Saved {obs} true RED4 array at {true_out_path}")
            print(f"Saved {obs} SYN4 RED4 array at {syn4_out_path}")

    if tif_models:
        tifs_dir = os.path.join(save_root, "tifs")
        os.makedirs(tifs_dir, exist_ok=True)
        for model_name in tif_models:
            pred_img = _pred_for(model_name, "tif")
            if pred_img is None:
                continue
            tag = resolve_model_tag(model_name)
            for band_count in tif_band_counts:
                export_arr = build_full_export_array(
                    pred_img,
                    sample["ir"],
                    sample["bg"],
                    sample,
                    band_count,
                )
                out_path = os.path.join(tifs_dir, f"{obs}_{tag}{band_suffix(band_count)}.tif")
                np_to_tiff(
                    export_arr,
                    out_path,
                    dtype=np.float32,
                    stretch=False,
                )
                if verbose:
                    print(f"Saved {obs} {tag} as {band_count}-band GeoTIFF at {out_path} (float32, no stretch)")

    if cub_models:
        cubes_dir = os.path.join(save_root, "cubes")
        os.makedirs(cubes_dir, exist_ok=True)
        neigh_cube = os.path.join(img_base_dir or NEIGH_CUBE_DIR, obs, f"{obs}_RED5_processed.cub")
        if not os.path.exists(neigh_cube):
            print(f"Neighbor cube {neigh_cube} not found, skipping cube saving for {obs}")
        else:
            for model_name in cub_models:
                pred_img = _pred_for(model_name, "cub")
                if pred_img is None:
                    continue
                tag = resolve_model_tag(model_name)
                out_path = os.path.join(cubes_dir, f"{obs}_{tag}.cub")
                output_full = restore_crop(
                    pred_img,
                    sample["original_shape"],
                    sample["crop_dims"],
                )
                np_to_cube(output_full, neigh_cube, out_path)
                print(f"Saved {obs} {tag} as cube at {out_path}")
