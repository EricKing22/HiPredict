import random
import os
import warnings
from pathlib import Path
import pandas as pd

from hipredict.data.isis2np import run_processing
from hipredict.data.load_metadata import gen_edr_df
from hipredict.utils.utils import check_obs_list, get_max_workers
from hipredict.paths import PACKAGE_ROOT


PATCH_CCDS = ("BG12", "IR10", "RED3", "RED4", "RED5")
VALID_BIN_TDI_PAIRS = ((1, 64), (1, 128), (2, 64), (2, 128))


def _plan_patch_targets_basic(obs_ids, existing_patch_sets, n_patch_sets, min_patches_per_obs, preferred_obs=None):
    if n_patch_sets <= 0:
        raise ValueError("n_patch_sets must be a positive integer")
    if min_patches_per_obs <= 0:
        raise ValueError("min_patches_per_obs must be a positive integer")
    if not obs_ids:
        return [], {}

    preferred_obs = set(preferred_obs or ())
    existing_obs = [obs for obs in obs_ids if existing_patch_sets.get(obs, 0) > 0]
    preferred_new_obs = [obs for obs in obs_ids if existing_patch_sets.get(obs, 0) == 0 and obs in preferred_obs]
    other_new_obs = [obs for obs in obs_ids if existing_patch_sets.get(obs, 0) == 0 and obs not in preferred_obs]
    existing_obs.sort(key=lambda obs: (-existing_patch_sets.get(obs, 0), obs))
    random.shuffle(preferred_new_obs)
    random.shuffle(other_new_obs)
    ordered_obs = existing_obs + preferred_new_obs + other_new_obs

    if n_patch_sets < min_patches_per_obs:
        # Preserve the existing/basic ordering preference when there is budget for only one partial selection.
        obs = ordered_obs[0]
        return [obs], {obs: max(existing_patch_sets.get(obs, 0), n_patch_sets)}

    target_obs_count = min(len(ordered_obs), max(1, n_patch_sets // min_patches_per_obs))
    selected_obs = ordered_obs[:target_obs_count]
    patch_targets = {
        obs: max(existing_patch_sets.get(obs, 0), min_patches_per_obs)
        for obs in selected_obs
    }

    allocated = sum(patch_targets.values())
    while allocated > n_patch_sets and len(selected_obs) > 1:
        dropped_obs = selected_obs.pop()
        allocated -= patch_targets.pop(dropped_obs)

    if allocated > n_patch_sets and len(selected_obs) == 1:
        only_obs = selected_obs[0]
        existing_count = existing_patch_sets.get(only_obs, 0)
        if existing_count > n_patch_sets:
            print(
                f"WARNING: Observation {only_obs} already has {existing_count} complete patch set(s), "
                f"which exceeds requested n_patch_sets={n_patch_sets}. Keeping the existing patches."
            )
            patch_targets[only_obs] = existing_count
            return selected_obs, patch_targets

        patch_targets[only_obs] = n_patch_sets
        return selected_obs, patch_targets

    remaining = n_patch_sets - allocated
    if remaining > 0 and selected_obs:
        extra_each, extra_remainder = divmod(remaining, len(selected_obs))
        for idx, obs in enumerate(selected_obs):
            patch_targets[obs] += extra_each + (1 if idx < extra_remainder else 0)

    return selected_obs, patch_targets


def _qcut_labels(series, q, prefix):
    labels = pd.Series(index=series.index, dtype="object")
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return labels

    q = max(1, min(int(q), int(numeric.nunique())))
    if q == 1:
        labels.loc[numeric.index] = f"{prefix}_0"
        return labels

    binned = pd.qcut(numeric, q=q, labels=False, duplicates="drop")
    labels.loc[binned.index] = [f"{prefix}_{int(code)}" for code in binned.astype(int)]
    return labels


def _fixed_cut_labels(series, edges, prefix, right=True):
    labels = pd.Series(index=series.index, dtype="object")
    numeric = pd.to_numeric(series, errors="coerce")
    binned = pd.cut(numeric, bins=edges, labels=False, include_lowest=True, right=right)
    valid = binned.dropna().astype(int)
    labels.loc[valid.index] = [f"{prefix}_{code}" for code in valid.tolist()]
    return labels


def _build_observation_sampling_frame(
    edr_df,
    obs_ids,
    patch_ccds=PATCH_CCDS,
    spatial_lat_bins=6,
    spatial_lon_bins=12,
    orbit_bins=8,
    solar_longitude_bins=12,
    pair_min_obs_fraction=0.01,
):
    cols = [
        "OBSERVATION_ID",
        "CCD_NAME",
        "BINNING",
        "TDI",
        "IMAGE_CENTER_LATITUDE",
        "IMAGE_CENTER_LONGITUDE",
        "ORBIT_NUMBER",
        "SOLAR_LONGITUDE",
    ]
    subset = edr_df[
        edr_df["OBSERVATION_ID"].isin(obs_ids)
        & edr_df["CCD_NAME"].isin(patch_ccds)
    ][cols].copy()

    if subset.empty:
        return pd.DataFrame(index=pd.Index([], name="OBSERVATION_ID")), tuple()

    subset["BINNING"] = pd.to_numeric(subset["BINNING"], errors="coerce")
    subset["TDI"] = pd.to_numeric(subset["TDI"], errors="coerce")
    subset["IMAGE_CENTER_LATITUDE"] = pd.to_numeric(subset["IMAGE_CENTER_LATITUDE"], errors="coerce")
    subset["IMAGE_CENTER_LONGITUDE"] = pd.to_numeric(subset["IMAGE_CENTER_LONGITUDE"], errors="coerce")
    subset["ORBIT_NUMBER"] = pd.to_numeric(subset["ORBIT_NUMBER"], errors="coerce")
    subset["SOLAR_LONGITUDE"] = pd.to_numeric(subset["SOLAR_LONGITUDE"], errors="coerce")

    obs_frame = (
        subset.sort_values(["OBSERVATION_ID", "CCD_NAME"])
        .groupby("OBSERVATION_ID", sort=False)[
            ["IMAGE_CENTER_LATITUDE", "IMAGE_CENTER_LONGITUDE", "ORBIT_NUMBER", "SOLAR_LONGITUDE"]
        ]
        .first()
        .reindex(obs_ids)
    )

    lat_edges = [-90.0 + (180.0 * idx / spatial_lat_bins) for idx in range(spatial_lat_bins + 1)]
    lon_edges = [360.0 * idx / spatial_lon_bins for idx in range(spatial_lon_bins + 1)]
    solar_edges = [360.0 * idx / solar_longitude_bins for idx in range(solar_longitude_bins + 1)]

    lat_labels = _fixed_cut_labels(obs_frame["IMAGE_CENTER_LATITUDE"], lat_edges, "lat")
    lon_labels = _fixed_cut_labels(obs_frame["IMAGE_CENTER_LONGITUDE"].mod(360.0), lon_edges, "lon", right=False)
    spatial_labels = pd.Series(index=obs_frame.index, dtype="object")
    spatial_mask = lat_labels.notna() & lon_labels.notna()
    spatial_labels.loc[spatial_mask] = [
        f"{lat_labels.loc[idx]}__{lon_labels.loc[idx]}"
        for idx in obs_frame.index[spatial_mask]
    ]

    obs_frame["spatial_bin"] = spatial_labels
    obs_frame["orbit_bin"] = _qcut_labels(obs_frame["ORBIT_NUMBER"], orbit_bins, "orbit")
    obs_frame["solar_bin"] = _fixed_cut_labels(
        obs_frame["SOLAR_LONGITUDE"].mod(360.0),
        solar_edges,
        "solar",
        right=False,
    )

    pair_df = subset[
        subset["BINNING"].isin([1, 2])
        & subset["TDI"].isin([64, 128])
    ].copy()

    if pair_df.empty:
        return obs_frame, tuple()

    pair_df["pair_key"] = list(zip(pair_df["BINNING"].astype(int), pair_df["TDI"].astype(int)))
    pair_obs_presence = (
        pair_df.groupby(["OBSERVATION_ID", "pair_key"]).size().unstack(fill_value=0).reindex(obs_frame.index, fill_value=0)
    )

    obs_total = max(1, len(obs_frame.index))
    kept_pairs = tuple(
        pair
        for pair in VALID_BIN_TDI_PAIRS
        if float((pair_obs_presence.get(pair, pd.Series(0, index=obs_frame.index)) > 0).sum()) / obs_total >= pair_min_obs_fraction
    )

    if not kept_pairs:
        return obs_frame, kept_pairs

    kept_pair_counts = pair_obs_presence.reindex(columns=list(kept_pairs), fill_value=0)
    kept_pair_totals = kept_pair_counts.sum(axis=1).replace(0, pd.NA)
    kept_pair_shares = kept_pair_counts.div(kept_pair_totals, axis=0).fillna(0.0)

    for pair in kept_pairs:
        obs_frame[f"pair_{pair[0]}_{pair[1]}"] = kept_pair_shares[pair].astype(float)

    return obs_frame, kept_pairs


def _build_sampling_categories(obs_frame, kept_pairs, n_patch_sets):
    categories = {}

    for name, col, even_target in (
        ("spatial", "spatial_bin", True),
        ("orbit", "orbit_bin", True),
        ("solar", "solar_bin", False),
    ):
        values = obs_frame[col].dropna()
        if values.empty:
            continue

        if even_target:
            unique_values = sorted(values.unique().tolist())
            weights = {value: 1.0 / len(unique_values) for value in unique_values}
        else:
            norm = values.value_counts(normalize=True)
            weights = {value: float(weight) for value, weight in norm.items()}

        contribs = {}
        for obs, value in obs_frame[col].items():
            if pd.isna(value):
                continue
            contribs[obs] = {value: 1.0}

        categories[name] = {
            "target_counts": {value: n_patch_sets * weight for value, weight in weights.items()},
            "contribs": contribs,
        }

    if kept_pairs:
        pair_cols = {
            pair: f"pair_{pair[0]}_{pair[1]}"
            for pair in kept_pairs
            if f"pair_{pair[0]}_{pair[1]}" in obs_frame.columns
        }
        pair_weights_raw = {
            pair: float(obs_frame[col].sum())
            for pair, col in pair_cols.items()
        }
        total_pair_weight = sum(pair_weights_raw.values())

        if total_pair_weight > 0:
            target_counts = {
                pair: n_patch_sets * (weight / total_pair_weight)
                for pair, weight in pair_weights_raw.items()
                if weight > 0
            }
            contribs = {}
            for obs in obs_frame.index:
                obs_weights = {}
                for pair, col in pair_cols.items():
                    value = obs_frame.at[obs, col]
                    value = float(value) if pd.notna(value) else 0.0
                    if value > 0:
                        obs_weights[pair] = value
                if obs_weights:
                    contribs[obs] = obs_weights

            if target_counts and contribs:
                categories["pair"] = {
                    "target_counts": target_counts,
                    "contribs": contribs,
                }

    return categories


def _initialise_category_counts(categories):
    return {
        name: {value: 0.0 for value in category["target_counts"]}
        for name, category in categories.items()
    }


def _add_observation_contribution(current_counts, categories, obs, patch_count):
    for name, category in categories.items():
        obs_contrib = category["contribs"].get(obs, {})
        for value, weight in obs_contrib.items():
            current_counts[name][value] = current_counts[name].get(value, 0.0) + (patch_count * weight)


def _score_candidate_bundle(current_counts, categories, obs, bundle_size):
    score = 0.0
    for name, category in categories.items():
        obs_contrib = category["contribs"].get(obs, {})
        if not obs_contrib:
            continue
        category_counts = current_counts[name]
        for value, weight in obs_contrib.items():
            target = category["target_counts"].get(value, 0.0)
            current = category_counts.get(value, 0.0)
            updated = current + (bundle_size * weight)
            score += abs(target - current) - abs(target - updated)
    return score


def _choose_best_candidate(candidates, patch_targets, current_counts, categories, preferred_obs=None):
    if not candidates:
        return None

    preferred_obs = set(preferred_obs or ())
    scored = []
    for obs, bundle_size in candidates:
        score = _score_candidate_bundle(current_counts, categories, obs, bundle_size)
        scored.append((score, obs in preferred_obs, patch_targets.get(obs, 0), obs, bundle_size))

    scored.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    _, _, _, obs, bundle_size = scored[0]
    return obs, bundle_size


def _plan_patch_targets(
    obs_ids,
    existing_patch_sets,
    n_patch_sets,
    min_patches_per_obs,
    obs_frame=None,
    kept_pairs=(),
    preferred_obs=None,
):
    if n_patch_sets <= 0:
        raise ValueError("n_patch_sets must be a positive integer")
    if min_patches_per_obs <= 0:
        raise ValueError("min_patches_per_obs must be a positive integer")
    if not obs_ids:
        return [], {}
    if obs_frame is None or obs_frame.empty:
        return _plan_patch_targets_basic(obs_ids, existing_patch_sets, n_patch_sets, min_patches_per_obs, preferred_obs=preferred_obs)

    obs_frame = obs_frame.reindex(obs_ids)
    categories = _build_sampling_categories(obs_frame, kept_pairs, n_patch_sets)
    if not categories:
        print("WARNING: Representative sampling metadata unavailable. Falling back to baseline patch planning.")
        return _plan_patch_targets_basic(obs_ids, existing_patch_sets, n_patch_sets, min_patches_per_obs, preferred_obs=preferred_obs)

    existing_total = sum(int(existing_patch_sets.get(obs, 0)) for obs in obs_ids if existing_patch_sets.get(obs, 0) > 0)
    if existing_total > n_patch_sets:
        print(
            "WARNING: Existing complete patch sets take priority over cube-only preferences. "
            f"WARNING: Existing complete patch sets total {existing_total:,}, which exceeds "
            f"requested n_patch_sets={n_patch_sets:,}. Falling back to baseline patch planning."
        )
        return _plan_patch_targets_basic(obs_ids, existing_patch_sets, n_patch_sets, min_patches_per_obs)

    if kept_pairs:
        kept_pair_msg = ", ".join(f"({pair[0]}, {pair[1]})" for pair in kept_pairs)
        print(f"Representative sampling enabled with retained BINNING/TDI pairs: {kept_pair_msg}")
    else:
        print("Representative sampling enabled without a retained BINNING/TDI pair target.")

    preferred_obs = {
        obs
        for obs in (preferred_obs or ())
        if obs in obs_ids and int(existing_patch_sets.get(obs, 0)) == 0
    }
    if preferred_obs:
        print(
            f"Prioritising {len(preferred_obs):,} observation(s) with existing processed cubefiles for initial patch allocation."
        )
    patch_targets = {}
    current_counts = _initialise_category_counts(categories)

    for obs in obs_ids:
        existing_count = int(existing_patch_sets.get(obs, 0))
        if existing_count <= 0:
            continue
        patch_targets[obs] = existing_count
        _add_observation_contribution(current_counts, categories, obs, existing_count)

    remaining = n_patch_sets - sum(patch_targets.values())

    while remaining >= min_patches_per_obs:
        # While any preferred observations still need their initial min-patch bundle,
        # allocate only among that preferred subset using the same representative score.
        bundle_candidates = []
        for obs in obs_ids:
            current = patch_targets.get(obs, 0)
            if current == 0:
                bundle_candidates.append((obs, min_patches_per_obs))
            elif current < min_patches_per_obs:
                bundle_candidates.append((obs, min_patches_per_obs - current))

        if not bundle_candidates:
            break

        preferred_bundle_candidates = [
            candidate
            for candidate in bundle_candidates
            if candidate[0] in preferred_obs
        ]
        candidate_pool = preferred_bundle_candidates or bundle_candidates

        chosen = _choose_best_candidate(
            candidate_pool,
            patch_targets,
            current_counts,
            categories,
            preferred_obs=preferred_obs,
        )
        if chosen is None:
            break

        obs, bundle_size = chosen
        patch_targets[obs] = patch_targets.get(obs, 0) + bundle_size
        _add_observation_contribution(current_counts, categories, obs, bundle_size)
        remaining -= bundle_size

    if remaining > 0:
        selected_obs = [obs for obs in obs_ids if patch_targets.get(obs, 0) > 0]

        if not selected_obs:
            chosen = _choose_best_candidate(
                [(obs, remaining) for obs in obs_ids],
                patch_targets,
                current_counts,
                categories,
                preferred_obs=preferred_obs,
            )
            if chosen is not None:
                obs, bundle_size = chosen
                patch_targets[obs] = patch_targets.get(obs, 0) + bundle_size
                _add_observation_contribution(current_counts, categories, obs, bundle_size)
                remaining -= bundle_size
                selected_obs = [obs]

        while remaining > 0 and selected_obs:
            chosen = _choose_best_candidate(
                [(obs, 1) for obs in selected_obs],
                patch_targets,
                current_counts,
                categories,
                preferred_obs=preferred_obs,
            )
            if chosen is None:
                break

            obs, bundle_size = chosen
            patch_targets[obs] = patch_targets.get(obs, 0) + bundle_size
            _add_observation_contribution(current_counts, categories, obs, bundle_size)
            remaining -= bundle_size

    selected_obs = [obs for obs in obs_ids if patch_targets.get(obs, 0) > 0]
    return selected_obs, patch_targets


def PrepObservations(
        n_patch_sets=None,
        min_patches_per_obs=1,
        data_record=None,
        obs_list=None,
        np_base_dir=None,
        img_base_dir=None,
        n_ccds=5,
        workers=None,
        hiproc_fail_txt=None,
        force_cub_gen=False,
        force_npy_gen=False,
        prune_orphan_npy=False,
        use_existing_cubes=False,
        gen_np_patches=False,
        prep_mode=None,
        calib_path=str(PACKAGE_ROOT / "data" / "hiproc_pipe.sh"),
):
    """
    Convert downloaded HiRISE .IMG files into calibrated ISIS cubefiles and,
    optionally, numpy patch arrays ready for model inference.

    Filters the observation list to those that are downloaded, not on the
    hiproc fail list, and have valid BINNING/TDI pairs before processing.

    Two modes:
      - 'strips': runs the hiproc calibration pipeline to produce processed
        cubefiles only. Use this when full-strip inference is the goal.
      - 'patches': additionally extracts fixed-size numpy patch arrays from
        the cubefiles for patch-level model inference or training. Patch
        allocation across observations is planned using representative
        sampling over spatial location, orbit number, solar longitude, and
        BINNING/TDI pair to ensure the patch set reflects the diversity of
        the full observation catalogue.

    Args:
        n_patch_sets: Total patch sets to generate across all selected
            observations. Defaults to tot_obs * min_patches_per_obs.
            Ignored in 'strips' mode.
        min_patches_per_obs: Minimum patch sets assigned to any single
            selected observation. Ignored in 'strips' mode.
        data_record: Path to the CSV tracking processed observations.
            Required in 'patches' mode.
        obs_list: Observation IDs as a list of strings or a path to a .txt
            file with one ID per line.
        np_base_dir: Root directory for numpy patch output files. Required
            in 'patches' mode.
        img_base_dir: Root directory containing per-observation subdirectories
            of downloaded .IMG files.
        n_ccds: Number of CCD channels to process per observation.
        workers: Number of parallel processing workers. Defaults to all
            available CPUs minus one.
        hiproc_fail_txt: Writable failure log; IDs already in it are skipped
            and new failures are appended. Use a new file rather than the
            shipped files/hiproc_fails.txt, which the CLI reads via --skip-list.
        force_cub_gen: If True, regenerate cubefiles via hiproc even if they
            already exist on disk.
        force_npy_gen: If True, regenerate numpy patches even if complete sets
            already exist. No effect in 'strips' mode.
        prune_orphan_npy: If True, remove numpy files that belong to
            incomplete patch sets.
        use_existing_cubes: If True, prioritise observations that already have
            complete processed cubefiles when planning patch allocation.
            No effect when force_cub_gen=True.
        gen_np_patches: Deprecated flag; sets prep_mode to 'patches' when
            True. Prefer setting prep_mode directly.
        prep_mode: Processing mode — 'strips' produces only cubefiles;
            'patches' also generates numpy patch arrays. Inferred from
            gen_np_patches if not set.
        calib_path: Path to the hiproc calibration shell script.
    """
    print("PrepObservations called with:")
    for arg_name, arg_value in {
        "n_patch_sets": n_patch_sets,
        "min_patches_per_obs": min_patches_per_obs,
        "data_record": data_record,
        "obs_list": obs_list,
        "np_base_dir": np_base_dir,
        "img_base_dir": img_base_dir,
        "force_cub_gen": force_cub_gen,
        "force_npy_gen": force_npy_gen,
        "prune_orphan_npy": prune_orphan_npy,
        "use_existing_cubes": use_existing_cubes,
        "gen_np_patches": gen_np_patches,
        "calib_path": calib_path,
    }.items():
        print(f"  {arg_name}: {arg_value!r}")
    print()

    if obs_list is None:
        raise ValueError("obs_list must be provided")
    if img_base_dir is None:
        raise ValueError("img_base_dir must be provided")

    assert os.path.exists(img_base_dir), f"Image base dir does not exist: {img_base_dir}"
    assert Path(img_base_dir).is_dir(), f"Image base dir is not a directory: {img_base_dir}"
    img_base_dir = Path(img_base_dir).resolve()

    valid_prep_modes = {"strips", "patches"}
    if prep_mode is None:
        prep_mode = "patches" if gen_np_patches else "strips"
    if prep_mode not in valid_prep_modes:
        raise ValueError(f"Unsupported prep_mode: {prep_mode}. Expected one of {sorted(valid_prep_modes)}.")
    if gen_np_patches and prep_mode == "strips":
        raise ValueError("gen_np_patches=True conflicts with prep_mode='strips'. Use prep_mode='patches' or set gen_np_patches=False.")
    patch_enabled = prep_mode == "patches"

    if force_cub_gen:
        print("\nWARNING: force_cub_gen is set to True: All observations will regenerate cubefiles via hiproc_pipeline.\n")
    if force_npy_gen and patch_enabled:
        print("\nWARNING: force_npy_gen is set to True: All observations will regenerate numpy files even if they exist.\n")
    elif force_npy_gen and not patch_enabled:
        warnings.warn("force_npy_gen=True has no effect when prep_mode='strips'.", stacklevel=2)
    if use_existing_cubes and force_cub_gen:
        warnings.warn("use_existing_cubes=True has no effect when force_cub_gen=True.", stacklevel=2)

    # Load and check observation list
    # ------------------------
    all_ids = check_obs_list(obs_list)
    obs_ids = list(all_ids)
    print(f"Found {len(all_ids)} observations to process in list/file")

    skipped_fail = []
    if isinstance(hiproc_fail_txt, (str, Path)) and os.path.exists(hiproc_fail_txt):
        with open(hiproc_fail_txt, "r", encoding="utf-8") as f:     # type: ignore
            fail_set = {line.rstrip("\n") for line in f}
        skipped_fail = [obs for obs in all_ids if obs in fail_set]
        if skipped_fail:
            print(f"Skipping {len(skipped_fail)} of {len(all_ids)} requested observation(s) already in fail list ({hiproc_fail_txt}): {skipped_fail}")
        else:
            print(f"None of the {len(all_ids)} requested observation(s) are in the fail list ({hiproc_fail_txt}).")
        obs_ids = [obs for obs in all_ids if obs not in fail_set]
        print(f"Proceeding with {len(obs_ids)} of {len(all_ids)} observation(s) after fail-list filter.")

    # Check requested observations are downloaded
    # ------------------------
    downloaded_obs = set(
        subdir.name for subdir in img_base_dir.iterdir()
        if subdir.is_dir() and (
            next(subdir.glob('*.[iI][mM][gG]'), None) or # noqa
            next(subdir.glob('*_processed.cub'), None)
        )
    )
    skipped_no_download = [obs for obs in obs_ids if obs not in downloaded_obs]
    if skipped_no_download:
        print(f"Skipping {len(skipped_no_download)} of {len(obs_ids)} observation(s) with no downloaded images: {skipped_no_download}")
    obs_ids = [obs for obs in obs_ids if obs in downloaded_obs]
    print(f"Found {len(obs_ids)} of {len(all_ids)} requested observation(s) with downloaded images.")

    edr_df = gen_edr_df()
    ccd_config_subset = edr_df[
        edr_df["OBSERVATION_ID"].isin(obs_ids)
        & edr_df["CCD_NAME"].isin(["RED3", "RED4", "RED5", "IR10", "BG12"])
    ][["OBSERVATION_ID", "CCD_NAME", "BINNING", "TDI"]].copy()
    ccd_config_subset["BINNING"] = pd.to_numeric(ccd_config_subset["BINNING"], errors="coerce")
    ccd_config_subset["TDI"] = pd.to_numeric(ccd_config_subset["TDI"], errors="coerce")

    def _valid_bin_tdi_pair(row):
        binning = row["BINNING"]
        tdi = row["TDI"]
        if pd.isna(binning) or pd.isna(tdi):
            return False
        if not float(binning).is_integer() or not float(tdi).is_integer():
            return False
        return (int(binning), int(tdi)) in VALID_BIN_TDI_PAIRS

    invalid_pair_obs = set()
    invalid_binning_obs = set()
    if not ccd_config_subset.empty:
        invalid_pair_obs = set(
            ccd_config_subset.loc[
                ~ccd_config_subset.apply(_valid_bin_tdi_pair, axis=1),
                "OBSERVATION_ID",
            ]
        )

        binning_subset = ccd_config_subset[["OBSERVATION_ID", "CCD_NAME", "BINNING"]]
        obs_bins = (
            binning_subset.groupby(["OBSERVATION_ID", "CCD_NAME"])["BINNING"]
            .agg(lambda s: tuple(sorted({int(v) for v in s.dropna()})))
            .unstack()
        )

        def _bins_match(row, ccds):
            bins = []
            for ccd in ccds:
                value = row.get(ccd)
                if not isinstance(value, tuple):
                    return True
                if len(value) != 1:
                    return False
                bins.append(value[0])
            return len(set(bins)) == 1

        invalid_binning_obs = set(obs_bins.index[~obs_bins.apply(lambda row: _bins_match(row, ["RED3", "RED4", "RED5"]) and _bins_match(row, ["IR10", "BG12"]), axis=1)])

    invalid_obs = invalid_pair_obs | invalid_binning_obs
    skipped_invalid = [obs for obs in obs_ids if obs in invalid_obs]
    obs_ids = [obs for obs in obs_ids if obs not in invalid_obs]
    if invalid_pair_obs:
        print(f"Removed {len(invalid_pair_obs)} observation(s) from BINNING/TDI validity filter: {sorted(invalid_pair_obs)}")
    else:
        print("Removed 0 observations from BINNING/TDI validity filter.")
    extra_binning = invalid_binning_obs - invalid_pair_obs
    if extra_binning:
        print(f"Removed {len(extra_binning)} additional observation(s) from binning-consistency filter: {sorted(extra_binning)}")
    else:
        print("Removed 0 additional observations from binning-consistency filter.")

    tot_obs = len(obs_ids)
    assert len(obs_ids) > 0, "No requested observations have downloaded images. Please check your obs_list and img_base_dir paths."

    print(
        f"Prep mode: {prep_mode} "
        f"(processed cubefiles enabled; patch outputs {'enabled' if patch_enabled else 'disabled'})."
    )
    existing_patch_sets = None
    patch_targets = None
    preferred_cube_obs = set()

    if patch_enabled:
        if np_base_dir is None:
            raise ValueError("np_base_dir must be provided when prep_mode='patches'.")
        if data_record is None:
            raise ValueError("data_record must be provided when prep_mode='patches'.")
        patch_ccds = PATCH_CCDS

        def _complete_patch_sets(obs: str) -> int:
            obs_dir = Path(np_base_dir) / obs
            if not obs_dir.is_dir():
                return 0

            patch_sets = 0
            while all(
                (obs_dir / f"{obs}_{patch_sets}_{ccd}.npy").is_file()
                for ccd in patch_ccds
            ):
                patch_sets += 1
            return patch_sets

        def _has_processed_cubefiles(obs: str) -> bool:
            obs_dir = img_base_dir / obs
            if not obs_dir.is_dir():
                return False
            return sum(
                1 for _ in obs_dir.glob("*_processed.cub")
            ) == len(patch_ccds)

        if n_patch_sets is None:
            n_patch_sets = tot_obs * int(min_patches_per_obs)
        n_patch_sets = int(n_patch_sets)
        min_patches_per_obs = int(min_patches_per_obs)
        assert n_patch_sets > 0, "n_patch_sets must be a positive integer"
        assert min_patches_per_obs > 0, "min_patches_per_obs must be a positive integer"

        obs_sampling_frame, kept_pairs = _build_observation_sampling_frame(
            edr_df,
            obs_ids,
            patch_ccds=patch_ccds,
        )
        existing_patch_sets = {obs: _complete_patch_sets(obs) for obs in obs_ids}
        if force_npy_gen:
            print(
                "Ignoring existing patch-set counts during planning because "
                "force_npy_gen=True."
            )
            existing_patch_sets = {obs: 0 for obs in obs_ids}
        if use_existing_cubes and not force_cub_gen:
            preferred_cube_obs = {obs for obs in obs_ids if _has_processed_cubefiles(obs)}
            print(
                f"Found {len(preferred_cube_obs):,} observation(s) with complete processed cubefiles. "
                "These will be prioritised for initial patch allocation."
            )
        elif use_existing_cubes and force_cub_gen:
            preferred_cube_obs = set()

        obs_ids, patch_targets = _plan_patch_targets(
            obs_ids,
            existing_patch_sets,
            n_patch_sets,
            min_patches_per_obs,
            obs_frame=obs_sampling_frame,
            kept_pairs=kept_pairs,
            preferred_obs=preferred_cube_obs,
        )
        n_patch_sets = sum(patch_targets.values())
        selected_existing = [obs for obs in obs_ids if existing_patch_sets.get(obs, 0) > 0]
        patch_counts = [patch_targets[obs] for obs in obs_ids]

        print(f"Found {len(selected_existing):,} selected observation(s) with at least one complete existing patch set.")
        print(
            f"Generating {n_patch_sets:,} patch set(s) with min_patches_per_obs={min_patches_per_obs:,} "
            f"across {len(obs_ids):,} selected observations."
        )
        if use_existing_cubes and not force_cub_gen:
            selected_existing_cubes = sum(1 for obs in obs_ids if obs in preferred_cube_obs)
            print(
                f"Selected {selected_existing_cubes:,} observation(s) with complete processed cubefiles."
            )
        print(
            f"Selected {len(selected_existing):,} existing observation(s) and "
            f"{len(obs_ids) - len(selected_existing):,} new observation(s)."
        )
        print(
            f"Per-observation patch targets range from {min(patch_counts):,} to {max(patch_counts):,}."
        )
        n_patches = 0
        remaining_patches = 0
    else:
        if n_patch_sets is not None:
            print(f"Ignoring n_patch_sets={int(n_patch_sets):,} because prep_mode='strips'.")
        if np_base_dir is not None:
            print(f"Ignoring np_base_dir={np_base_dir!r} because prep_mode='strips'.")
        if data_record is not None:
            print(f"Ignoring data_record={data_record!r} because prep_mode='strips'.")
        n_patches = 0
        remaining_patches = 0
        print(f"Patch generation disabled. Preparing processed cubefiles for {tot_obs:,} observation(s).")

    print(f"\n--- Observation funnel ({len(all_ids)} requested) ---")
    if skipped_fail:
        print(f"  {len(skipped_fail):>3} skipped  (in fail list)      : {skipped_fail}")
    if skipped_no_download:
        print(f"  {len(skipped_no_download):>3} skipped  (no downloaded images): {skipped_no_download}")
    if skipped_invalid:
        print(f"  {len(skipped_invalid):>3} skipped  (invalid BINNING/TDI)  : {skipped_invalid}")
    print(f"  {len(obs_ids):>3} proceeding to preparation : {obs_ids}")
    print("---\n")

    requested_workers = int(workers) if workers is not None else (os.cpu_count() or 1)
    resolved_workers = get_max_workers(max_cap=requested_workers, leave_free=1, verbose=True)

    run_processing(
        data_record=data_record,
        n_workers=resolved_workers,
        img_base_addr=img_base_dir,
        obs_ids=obs_ids,
        np_save_base=np_base_dir,
        n_ccds=n_ccds,
        hiproc_fail_txt=hiproc_fail_txt,
        n_patches=n_patches,
        remaining_patches=remaining_patches,
        force_cub_gen=force_cub_gen,
        force_npy_gen=force_npy_gen,
        prune_orphan_npy=prune_orphan_npy,
        gen_np_patches=patch_enabled,
        calib_path=calib_path,
        patch_targets=patch_targets,
        existing_patch_sets=existing_patch_sets,
    )


def main(argv=None):
    import argparse
    from hipredict.paths import files_dir
    parser = argparse.ArgumentParser(description="Preprocess HiRISE EDRs using the patched HiPROC + ISIS backend.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--obs", nargs="+")
    source.add_argument("--obs-list")
    parser.add_argument("--img-dir", default=str(files_dir() / "isis_imgs" / "observations"))
    parser.add_argument("--mode", choices=["strips", "patches"], default="strips")
    parser.add_argument("--np-dir", default=str(files_dir() / "npy_files"))
    parser.add_argument("--record", default=str(files_dir() / "data_record.csv"))
    parser.add_argument("--patch-sets", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--use-existing-cubes", action="store_true")
    parser.add_argument("--skip-list", type=Path, help="Read-only observation IDs to skip, e.g. files/hiproc_fails.txt.")
    parser.add_argument("--fail-log", type=Path, help="Writable log for new processing failures; use a generated-output path.")
    args = parser.parse_args(argv)
    observations = check_obs_list(args.obs or args.obs_list)
    if args.skip_list is not None:
        skip_ids = set(check_obs_list(str(args.skip_list)))
        skipped = [obs for obs in observations if obs in skip_ids]
        observations = [obs for obs in observations if obs not in skip_ids]
        print(f"Skipping {len(skipped)} observation(s) listed in {args.skip_list}.")
    if args.fail_log is not None:
        reference = files_dir() / "hiproc_fails.txt"
        if args.fail_log.resolve() == reference.resolve() or (args.skip_list and args.fail_log.resolve() == args.skip_list.resolve()):
            parser.error("--fail-log must be separate from the read-only skip/reference list.")
    if not observations:
        print("No observations remain to process.")
        return
    PrepObservations(obs_list=observations, img_base_dir=args.img_dir,
                     workers=args.workers, prep_mode=args.mode,
                     np_base_dir=args.np_dir if args.mode == "patches" else None,
                     data_record=args.record if args.mode == "patches" else None,
                     n_patch_sets=args.patch_sets, use_existing_cubes=args.use_existing_cubes,
                     hiproc_fail_txt=args.fail_log)


if __name__ == "__main__":
    main()
