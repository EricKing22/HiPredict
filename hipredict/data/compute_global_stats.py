"""Compute global normalization statistics from processed HiRISE cubefiles.

This mirrors the data-generation path used in ``isis2np.compute_observation``:
load BG12/IR10/RED3/RED4/RED5 processed cubefiles, crop with ``crop_offset``
using BG12/IR10 as the crop-driving bands, then compute robust normalization
statistics from valid BG12/IR10 pixels.
"""

import argparse
import csv
import json
import math
from pprint import pprint
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from hipredict.data.isis2np import MISSING_VALUE, PATCH_CCDS, crop_offset, isis2np
from hipredict.data.normalization import MAD_TO_NORMAL_STD, _valid_pixel_mask
from hipredict.utils.utils import get_max_workers
from hipredict.paths import CONFIGS_DIR, files_dir


PREDICTOR_CCDS = ("BG12", "IR10")
IMG_BASE_DIR = str(files_dir() / "isis_imgs" / "observations")
MAX_OBS = None
WORKERS = 8
HIST_BINS = 2**17
HIST_MIN = -1.0
HIST_MAX = 1.0
PROGRESS_EVERY = 1000
MAX_FAILURES = 20


def _read_observations_from_data_record(data_record):
    obs_ids = []
    seen = set()
    with open(data_record, newline="") as f:
        reader = csv.DictReader(f)
        if "Observation" not in (reader.fieldnames or []):
            raise ValueError(f"{data_record} has no Observation column.")
        for row in reader:
            obs = row["Observation"]
            if obs and obs not in seen:
                seen.add(obs)
                obs_ids.append(obs)
    return obs_ids


def _discover_observations(img_base_dir):
    obs_ids = []
    for path in sorted(Path(img_base_dir).iterdir()):
        if path.is_dir() and any(path.glob("*_processed.cub")):
            obs_ids.append(path.name)
    return obs_ids


def _find_processed_cubes(img_base_dir, obs):
    obs_dir = Path(img_base_dir) / obs
    if not obs_dir.is_dir():
        raise FileNotFoundError(f"Observation directory not found: {obs_dir}")

    cube_paths = {ccd: [] for ccd in PATCH_CCDS}
    for path in sorted(obs_dir.glob("*_processed.cub")):
        for ccd in PATCH_CCDS:
            if ccd in path.name:
                cube_paths[ccd].append(path)

    selected = []
    for ccd in PATCH_CCDS:
        matches = cube_paths[ccd]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one processed cube for {obs} {ccd}, found {len(matches)}."
            )
        selected.append(str(matches[0]))
    return selected


def _load_cropped_predictors(img_base_dir, obs):
    cube_paths = _find_processed_cubes(img_base_dir, obs)
    bands = tuple(isis2np(path, img_save_loc=None) for path in cube_paths)
    cropped_bands, crop_dims = crop_offset(
        bands,
        invalid_frac_thresh=0.85,
        crop_band_indices=(0, 1),
    )
    bg_arr = np.asarray(cropped_bands[0], dtype=np.float32)
    ir_arr = np.asarray(cropped_bands[1], dtype=np.float32)
    return bg_arr, ir_arr, crop_dims


def _empty_band_summary():
    return {
        "finite_count": 0,
        "nonfinite_count": 0,
        "missing_count": 0,
        "near_zero_count": 0,
        "finite_min": math.nan,
        "finite_max": math.nan,
        "finite_sum": 0.0,
        "finite_sumsq": 0.0,
    }


def _band_summary(arr):
    arr = np.asarray(arr, dtype=np.float32)
    finite = np.isfinite(arr)
    out = _empty_band_summary()
    out["finite_count"] = int(finite.sum())
    out["nonfinite_count"] = int(arr.size - out["finite_count"])
    out["missing_count"] = int(np.isclose(arr, MISSING_VALUE, atol=1e-7).sum())
    out["near_zero_count"] = int((np.abs(arr) <= 1e-3).sum())
    if out["finite_count"] > 0:
        vals = arr[finite].astype(np.float64, copy=False)
        out["finite_min"] = float(vals.min())
        out["finite_max"] = float(vals.max())
        out["finite_sum"] = float(vals.sum())
        out["finite_sumsq"] = float(np.square(vals).sum())
    return out


def _merge_band_summary(total, part):
    total["finite_count"] += part["finite_count"]
    total["nonfinite_count"] += part["nonfinite_count"]
    total["missing_count"] += part["missing_count"]
    total["near_zero_count"] += part["near_zero_count"]
    total["finite_sum"] += part["finite_sum"]
    total["finite_sumsq"] += part["finite_sumsq"]
    if math.isnan(total["finite_min"]):
        total["finite_min"] = part["finite_min"]
    elif not math.isnan(part["finite_min"]):
        total["finite_min"] = min(total["finite_min"], part["finite_min"])
    if math.isnan(total["finite_max"]):
        total["finite_max"] = part["finite_max"]
    elif not math.isnan(part["finite_max"]):
        total["finite_max"] = max(total["finite_max"], part["finite_max"])


def _finish_moment_stats(count, total_sum, total_sumsq):
    if count <= 0:
        return {"mean": math.nan, "std": math.nan}
    mean = total_sum / count
    var = max((total_sumsq / count) - (mean * mean), 0.0)
    return {"mean": float(mean), "std": float(math.sqrt(var))}


def _hist_quantile(hist, edges, q):
    total = int(hist.sum())
    if total <= 0:
        return math.nan
    if q <= 0:
        idx = int(np.flatnonzero(hist)[0])
        return float(edges[idx])
    if q >= 1:
        idx = int(np.flatnonzero(hist)[-1])
        return float(edges[idx + 1])

    rank = q * (total - 1)
    cdf = np.cumsum(hist, dtype=np.int64)
    idx = int(np.searchsorted(cdf, rank + 1, side="left"))
    prev = int(cdf[idx - 1]) if idx > 0 else 0
    bin_count = int(hist[idx])
    if bin_count <= 0:
        return float((edges[idx] + edges[idx + 1]) / 2.0)
    frac = min(max((rank - prev) / bin_count, 0.0), 1.0)
    return float(edges[idx] + frac * (edges[idx + 1] - edges[idx]))


def _quantile_dict(hist, edges, prefix):
    quantiles = {
        "p001": 0.001,
        "p01": 0.01,
        "p05": 0.05,
        "p10": 0.10,
        "p25": 0.25,
        "p50": 0.50,
        "p75": 0.75,
        "p90": 0.90,
        "p95": 0.95,
        "p99": 0.99,
        "p999": 0.999,
    }
    return {f"{prefix}_{name}": _hist_quantile(hist, edges, q) for name, q in quantiles.items()}


def _process_observation_values(task):
    img_base_dir, obs, hist_edges = task
    try:
        bg_arr, ir_arr, crop_dims = _load_cropped_predictors(img_base_dir, obs)
        valid_bg = _valid_pixel_mask(bg_arr)
        valid_ir = _valid_pixel_mask(ir_arr)
        valid_pair = valid_bg & valid_ir

        values = np.concatenate(
            [bg_arr[valid_pair], ir_arr[valid_pair]],
            axis=0,
        ).astype(np.float32, copy=False)

        hist, _ = np.histogram(values, bins=hist_edges)
        below_range = int((values < hist_edges[0]).sum())
        above_range = int((values > hist_edges[-1]).sum())

        if values.size:
            masked_min = float(values.min())
            masked_max = float(values.max())
            masked_sum = float(values.astype(np.float64).sum())
            masked_sumsq = float(np.square(values.astype(np.float64)).sum())
            obs_center = float(np.median(values))
            obs_mad = float(np.median(np.abs(values - obs_center)))
            obs_scale = float(max(MAD_TO_NORMAL_STD * obs_mad, 1e-3))
        else:
            masked_min = masked_max = masked_sum = masked_sumsq = math.nan
            obs_center = obs_mad = obs_scale = math.nan

        h, w = bg_arr.shape
        return {
            "ok": True,
            "obs": obs,
            "hist": hist.astype(np.int64, copy=False),
            "crop_dims": tuple(int(x) for x in crop_dims),
            "cropped_height": int(h),
            "cropped_width": int(w),
            "cropped_pixels": int(h * w),
            "valid_pair_pixels": int(valid_pair.sum()),
            "valid_bg_pixels": int(valid_bg.sum()),
            "valid_ir_pixels": int(valid_ir.sum()),
            "masked_count": int(values.size),
            "masked_min": masked_min,
            "masked_max": masked_max,
            "masked_sum": masked_sum,
            "masked_sumsq": masked_sumsq,
            "below_hist_range": below_range,
            "above_hist_range": above_range,
            "bg_summary": _band_summary(bg_arr),
            "ir_summary": _band_summary(ir_arr),
            "obs_center": obs_center,
            "obs_mad": obs_mad,
            "obs_scale": obs_scale,
        }
    except Exception as exc:
        return {"ok": False, "obs": obs, "error": repr(exc)}


def _run_pool(tasks, worker_fn, workers, label, progress_every):
    results = []
    done = 0
    total = len(tasks)
    if workers == 1:
        for task in tasks:
            results.append(worker_fn(task))
            done += 1
            if done % progress_every == 0 or done == total:
                print(f"{label}: {done:,}/{total:,} observations", flush=True)
        return results

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker_fn, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())
            done += 1
            if done % progress_every == 0 or done == total:
                print(f"{label}: {done:,}/{total:,} observations", flush=True)
    return results


def _obs_distribution(values, prefix):
    vals = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if vals.size == 0:
        return {
            f"{prefix}_min": math.nan,
            f"{prefix}_p25": math.nan,
            f"{prefix}_p50": math.nan,
            f"{prefix}_p75": math.nan,
            f"{prefix}_max": math.nan,
        }
    return {
        f"{prefix}_min": float(np.min(vals)),
        f"{prefix}_p25": float(np.quantile(vals, 0.25)),
        f"{prefix}_p50": float(np.quantile(vals, 0.50)),
        f"{prefix}_p75": float(np.quantile(vals, 0.75)),
        f"{prefix}_max": float(np.max(vals)),
    }


def _deviation_histogram_from_value_histogram(hist, value_edges, center, dev_edges):
    value_centers = (value_edges[:-1] + value_edges[1:]) / 2.0
    deviations = np.abs(value_centers - center)
    dev_hist, _ = np.histogram(deviations, bins=dev_edges, weights=hist)
    return np.rint(dev_hist).astype(np.int64, copy=False)


def compute_global_stats(
    img_base_dir,
    data_record=None,
    max_obs=20,
    workers=None,
    hist_bins=HIST_BINS,
    hist_min=-1.0,
    hist_max=1.0,
    progress_every=PROGRESS_EVERY,
    max_failures=20,
):
    if img_base_dir is None:
        raise ValueError("img_base_dir must be set before computing global stats.")

    if data_record:
        obs_ids = _read_observations_from_data_record(data_record)
    else:
        obs_ids = _discover_observations(img_base_dir)

    if max_obs is not None:
        obs_ids = obs_ids[: max_obs]
    if not obs_ids:
        raise ValueError("No observations selected.")

    workers = workers if workers is not None else get_max_workers()
    workers = max(1, int(workers))
    hist_bins = int(hist_bins)
    hist_edges = np.linspace(hist_min, hist_max, hist_bins + 1, dtype=np.float64)

    print(f"Selected {len(obs_ids):,} observation(s).", flush=True)
    print(f"Using {workers} worker(s), {hist_bins:,} histogram bins.", flush=True)

    value_tasks = [(img_base_dir, obs, hist_edges) for obs in obs_ids]
    value_results = _run_pool(
        value_tasks,
        _process_observation_values,
        workers,
        "value pass",
        progress_every,
    )

    successes = [result for result in value_results if result["ok"]]
    failures = [result for result in value_results if not result["ok"]]
    if not successes:
        raise ValueError(f"No observations succeeded. First failures: {failures[:5]}")

    hist = np.zeros(hist_bins, dtype=np.int64)
    band_summaries = {ccd: _empty_band_summary() for ccd in PREDICTOR_CCDS}
    totals = {
        "cropped_pixels": 0,
        "valid_pair_pixels": 0,
        "valid_bg_pixels": 0,
        "valid_ir_pixels": 0,
        "masked_count": 0,
        "masked_sum": 0.0,
        "masked_sumsq": 0.0,
        "below_hist_range": 0,
        "above_hist_range": 0,
    }
    masked_min = math.nan
    masked_max = math.nan

    for result in successes:
        hist += result["hist"]
        for key in totals:
            totals[key] += result[key]
        _merge_band_summary(band_summaries["BG12"], result["bg_summary"])
        _merge_band_summary(band_summaries["IR10"], result["ir_summary"])
        if math.isnan(masked_min):
            masked_min = result["masked_min"]
        else:
            masked_min = min(masked_min, result["masked_min"])
        if math.isnan(masked_max):
            masked_max = result["masked_max"]
        else:
            masked_max = max(masked_max, result["masked_max"])

    global_center = _hist_quantile(hist, hist_edges, 0.5)
    dev_max = max(abs(masked_min - global_center), abs(masked_max - global_center), 1e-6)
    dev_edges = np.linspace(0.0, dev_max, hist_bins + 1, dtype=np.float64)
    dev_hist = _deviation_histogram_from_value_histogram(hist, hist_edges, global_center, dev_edges)
    dev_failures = []

    global_mad = _hist_quantile(dev_hist, dev_edges, 0.5)
    global_scale = float(max(MAD_TO_NORMAL_STD * global_mad, 1e-3))
    masked_moments = _finish_moment_stats(
        totals["masked_count"],
        totals["masked_sum"],
        totals["masked_sumsq"],
    )

    result = {
        "normalization": {
            "mode": "global_robust",
            "center": float(global_center),
            "scale": global_scale,
            "mad": float(global_mad),
            "method": "histogram_approximation_over_cropped_valid_BG12_IR10_pixels",
            "scale_formula": f"max({MAD_TO_NORMAL_STD} * MAD, 1e-3)",
        },
        "masked_predictor_values": {
            "count": int(totals["masked_count"]),
            "min": float(masked_min),
            "max": float(masked_max),
            **masked_moments,
            **_quantile_dict(hist, hist_edges, "value"),
        },
        "validity": {
            "observations_requested": len(obs_ids),
            "observations_succeeded": len(successes),
            "observations_failed_value_pass": len(failures),
            "observations_failed_deviation_pass": 0,
            "cropped_pixels_per_predictor": int(totals["cropped_pixels"]),
            "valid_pair_pixels": int(totals["valid_pair_pixels"]),
            "valid_pair_fraction": float(totals["valid_pair_pixels"] / totals["cropped_pixels"]),
            "valid_bg_pixels": int(totals["valid_bg_pixels"]),
            "valid_ir_pixels": int(totals["valid_ir_pixels"]),
            "below_hist_range": int(totals["below_hist_range"]),
            "above_hist_range": int(totals["above_hist_range"]),
        },
        "unmasked_finite_values": {
            "BG12": {**band_summaries["BG12"], **_finish_moment_stats(
                band_summaries["BG12"]["finite_count"],
                band_summaries["BG12"]["finite_sum"],
                band_summaries["BG12"]["finite_sumsq"],
            )},
            "IR10": {**band_summaries["IR10"], **_finish_moment_stats(
                band_summaries["IR10"]["finite_count"],
                band_summaries["IR10"]["finite_sum"],
                band_summaries["IR10"]["finite_sumsq"],
            )},
        },
        "absolute_deviation_from_center": {
            **_quantile_dict(dev_hist, dev_edges, "absdev"),
            "max": float(dev_max),
        },
        "observation_level_distribution": {
            **_obs_distribution([r["obs_center"] for r in successes], "obs_center"),
            **_obs_distribution([r["obs_scale"] for r in successes], "obs_scale"),
            **_obs_distribution([r["obs_mad"] for r in successes], "obs_mad"),
        },
        "histogram": {
            "bins": int(hist_bins),
            "value_min": float(hist_min),
            "value_max": float(hist_max),
            "deviation_min": 0.0,
            "deviation_method": "derived_from_value_histogram_bin_centers",
            "deviation_max": float(dev_max),
        },
        "failures": {
            "value_pass": failures[: max_failures],
            "deviation_pass": dev_failures[: max_failures],
            "truncated_to": int(max_failures),
        },
    }
    return result


def write_global_stats(result, output_path=None):
    output_path = Path(output_path or (files_dir() / "generated" / "statistics" / "global_stats.json"))
    references = {files_dir() / "global_stats.json", CONFIGS_DIR / "global_stats.json"}
    if output_path.resolve() in {path.resolve() for path in references}:
        raise ValueError("Choose a generated-output path; released normalization statistics are reference files.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--img-dir", default=IMG_BASE_DIR)
    parser.add_argument("--record", help="Optional observation-selection CSV; otherwise use available processed observations.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-obs", type=int, default=MAX_OBS)
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args(argv)
    result = compute_global_stats(
        img_base_dir=args.img_dir,
        data_record=args.record,
        max_obs=args.max_obs,
        workers=args.workers,
        hist_bins=HIST_BINS,
        hist_min=HIST_MIN,
        hist_max=HIST_MAX,
        progress_every=PROGRESS_EVERY,
        max_failures=MAX_FAILURES,
    )
    pprint(result, sort_dicts=True)
    output = write_global_stats(result, args.output)
    print(f"Computed statistics saved to {output.resolve()}")


if __name__ == "__main__":
    main()
