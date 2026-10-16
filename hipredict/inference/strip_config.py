"""Config-driven assembly of the observation set for the strip workflow.

Turns the postprocess YAML into the concrete list of observations to process:
resolves each entry's group label, optionally auto-selects a random set of
standard-eval strips, and looks up the patch subsets used for the patch-vs-strip
comparison. This is the "what to run" layer; the main workflow just consumes it.
"""

import os

import numpy as np
import pandas as pd

from hipredict.paths import files_dir, resolve_path


DEFAULT_EVAL_RECORD = str(files_dir() / "data_record_evals.csv")
REQUIRED_PATCH_CCDS = {"BG12", "IR10", "RED3", "RED4", "RED5"}


def entry_group(entry):
    """Group label for an entry, from its `group` field or its description text."""
    group = entry.get("group")
    if group:
        return str(group)

    desc = str(entry.get("description", "")).strip().lower()
    if "icy" in desc:
        return "icy_impact"
    if "standard" in desc or "eval" in desc:
        return "standard_eval"
    return desc.replace(" ", "_") or "ungrouped"


def available_observation_ids(img_base_addr):
    """Observation ids under img_base_addr that have local IMG/cube inputs."""
    base = os.fspath(img_base_addr)
    if not os.path.isdir(base):
        return set()

    available = set()
    for name in os.listdir(base):
        obs_dir = os.path.join(base, name)
        if not os.path.isdir(obs_dir):
            continue
        has_inputs = any(fn.lower().endswith(".img") or fn.endswith("_processed.cub") for fn in os.listdir(obs_dir))
        if has_inputs:
            available.add(name)
    return available


def standard_eval_entries(postproc_cfg, existing_obs, img_base_addr=None, default_model="hipredict"):
    """Randomly pick a set of standard-eval observations from the data record."""
    cfg = postproc_cfg.get("standard_eval_observations", {}) or {}
    if not cfg.get("enabled", False):
        return []

    count = int(cfg.get("count", 0))
    if count <= 0:
        return []

    data_record = resolve_path(cfg.get("data_record", DEFAULT_EVAL_RECORD))
    eval_df = pd.read_csv(data_record, usecols=["Observation"])
    candidates = sorted(
        str(obs)
        for obs in eval_df["Observation"].dropna().unique()
        if str(obs) not in existing_obs
    )
    if not candidates:
        raise ValueError(f"No standard eval observations found in {data_record}")

    if bool(cfg.get("require_available_inputs", True)):
        available_obs = available_observation_ids(img_base_addr) if img_base_addr is not None else set()
        available_candidates = [obs for obs in candidates if obs in available_obs]
        skipped = len(candidates) - len(available_candidates)
        if skipped:
            print(
                f"Standard eval selection skipped {skipped} observation(s) without local IMG/cube inputs "
                f"under {img_base_addr}."
            )
        candidates = available_candidates
        if not candidates:
            print(
                "WARNING: No locally available standard eval strip inputs were found. "
                "Standard eval strip comparison will be skipped."
            )
            return []

    rng = np.random.default_rng(int(cfg.get("seed", 0)))
    chosen = sorted(rng.choice(candidates, size=min(count, len(candidates)), replace=False).tolist())
    if len(chosen) < count:
        print(f"WARNING: Requested {count} standard eval observations, but only {len(chosen)} are locally available.")

    return [
        {
            "id": obs,
            "group": cfg.get("group", "standard_eval"),
            "description": cfg.get("description", "standard eval"),
            "npy_bands": cfg.get("npy_bands", 3),
            "npy": cfg.get("npy", [default_model]),
            "tif_bands": cfg.get("tif_bands", 3),
            "tif": cfg.get("tif", []),
            "cub": cfg.get("cub", []),
        }
        for obs in chosen
    ]


def load_patch_comparison_record(postproc_cfg):
    """Load the patch-comparison data record if the feature is enabled."""
    cfg = postproc_cfg.get("patch_comparison", {}) or {}
    if not cfg.get("enabled", False):
        return None
    return pd.read_csv(resolve_path(cfg.get("data_record", DEFAULT_EVAL_RECORD)))


def first_patch_subset_for_obs(patch_record_df, obs):
    """First patch Set for obs that covers all REQUIRED_PATCH_CCDS, with its RED4 patch index."""
    if patch_record_df is None:
        return None, None, None

    obs_df = patch_record_df[patch_record_df["Observation"] == obs]
    if obs_df.empty:
        return None, None, None

    for set_id in sorted(obs_df["Set"].dropna().unique()):
        set_df = obs_df[obs_df["Set"] == set_id].copy()
        if REQUIRED_PATCH_CCDS.issubset(set(set_df["CCD"])):
            patch_idx = None
            patch_vals = set_df.loc[set_df["CCD"] == "RED4", "Patch"]
            if not patch_vals.empty:
                patch_idx = int(patch_vals.iloc[0])
            return set_df, int(set_id), patch_idx

    return None, None, None
