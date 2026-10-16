import json
from pathlib import Path

import numpy as np
from hipredict.paths import CONFIGS_DIR, files_dir, resolve_path

MAD_TO_NORMAL_STD = 1.4826  # Converts MAD to Gaussian-equivalent std: 1 / Phi^-1(0.75).
DEFAULT_GLOBAL_STATS_PATH = "files/global_stats.json"
DEFAULT_NORMALIZATION_MODE = "global_robust"
_REPO_ROOT = Path(__file__).resolve().parents[2]


def _resolve_global_stats_path(cfg, stats_path=None):
    data_cfg = (cfg or {}).get("data", {})
    path_like = (
        stats_path
        or data_cfg.get("global_stats")
        or data_cfg.get("global_stats_path")
        or DEFAULT_GLOBAL_STATS_PATH
    )
    path = Path(path_like)
    if path.is_absolute():
        return path

    candidates = []
    if path_like == DEFAULT_GLOBAL_STATS_PATH:
        candidates.append(files_dir() / "global_stats.json")
    root = data_cfg.get("root")
    if root is not None:
        candidates.append(resolve_path(path, root))
    candidates.extend([_REPO_ROOT / path, Path.cwd() / path])
    if path_like == DEFAULT_GLOBAL_STATS_PATH:
        candidates.append(CONFIGS_DIR / "global_stats.json")

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def load_global_normalization_stats(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Global normalization stats not found: {path}")

    with path.open("r") as f:
        stats = json.load(f)

    norm = stats.get("normalization", {})
    if norm.get("mode") != "global_robust":
        raise ValueError(f"Expected global_robust stats in {path}, found {norm.get('mode')!r}")

    center = float(norm["center"])
    scale = float(norm["scale"])
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"Invalid global normalization center/scale in {path}: {center=}, {scale=}")
    return center, scale


def normalization_kwargs_from_config(cfg, norm_mode=None):
    data_cfg = (cfg or {}).get("data", {})
    mode = norm_mode if norm_mode is not None else data_cfg.get("normalisation_mode", DEFAULT_NORMALIZATION_MODE)
    kwargs = {"norm_mode": mode}
    if mode == "global_robust":
        center, scale = load_global_normalization_stats(_resolve_global_stats_path(cfg))
        kwargs.update(global_center=center, global_scale=scale)
    return kwargs


def _valid_pixel_mask(arr, use_missing=True, missing_value=-3.663e-4, atol=1e-7, eps=1e-3):
    arr = np.asarray(arr)
    finite = np.isfinite(arr)
    if use_missing:
        return finite & (~np.isclose(arr, missing_value, atol=atol))
    return finite & (np.abs(arr) > eps)


def _robust_center_scale_from_predictors_np(bg_arr, ir_arr):
    """Canonical observation-level IR/BG normalization used by data prep and strip inference."""
    bg_arr = np.asarray(bg_arr, dtype=np.float32)
    ir_arr = np.asarray(ir_arr, dtype=np.float32)
    valid = _valid_pixel_mask(bg_arr) & _valid_pixel_mask(ir_arr)
    if not np.any(valid):
        raise ValueError("Cannot compute observation-level normalization stats without valid IR/BG pixels.")

    flat = np.concatenate([ir_arr[valid], bg_arr[valid]], axis=0).astype(np.float32, copy=False)
    a = float(np.median(flat))
    mad = float(np.median(np.abs(flat - a)))
    s = max(MAD_TO_NORMAL_STD * mad, 1e-3)
    return a, s
