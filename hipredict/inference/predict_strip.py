"""Run the HiRISE reconstruction model over full observation strips.

This is the reference example for the end-to-end inference workflow: point it at
one or more observations, apply the trained model, and save the reconstructed
output.

    for each observation:
        1. INPUT        load the observation's input bands from disk
        2. APPLY MODEL  run the model to reconstruct its RED4 band
        3. SAVE         write the prediction to disk (.npy / .tif / .cub)

Sibling modules:
    strip_config   -> which observations to run + how each is configured
    strip_eval     -> score the model against the SYN4 baseline; print reports
    strip_exports  -> write predictions to .npy / .tif / .cub

What you normally edit:
    configs/postprocess.yaml  -> which observations to process and what to export
    MAIN_MODEL                -> which trained model to run (defined in strip_exports)
"""

import time

import torch
import yaml

from hipredict import get_model
from hipredict.predict.loss import get_criterion
from hipredict.inference.hipredict import HiPredict
from hipredict.paths import CONFIGS_DIR, files_dir

from hipredict.inference import strip_config, strip_eval, strip_exports
from hipredict.inference.strip_exports import MAIN_MODEL, SYN4_MODEL


POSTPROC_CONFIG = str(CONFIGS_DIR / 'postprocess.yaml')     # Config listing the observations to run and the formats to export for each.
SAVE_ROOT = str(files_dir() / "figs")   # Root under which predictions are written (strips/, tifs/, cubes/ subfolders).


if __name__ == "__main__":
    start = time.time()
    # Strip inference is heavy, so use the GPU whenever one is available.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    cnn_name = MAIN_MODEL              # trained model to run (weights loaded below)
    smooth_norm_kernel = 257          # Only used when normalisation_mode == "scene_smooth"
    img_base_addr = str(files_dir() / "isis_imgs" / "observations")  # input root: one folder per observation

    with open(POSTPROC_CONFIG, "r") as f:
        postproc_cfg = yaml.safe_load(f)

    # --- Assemble the observation set ("what to run") --------------------------
    # `observations` in the YAML is the explicit list; `standard_eval_entries`
    # can additionally draw a random, seeded sample of eval strips. Each "entry"
    # is a dict describing one observation and how it should be exported.
    verbose = bool(postproc_cfg.get("verbose", False))
    base_entries = list(postproc_cfg['observations'])
    standard_entries = strip_config.standard_eval_entries(
        postproc_cfg,
        existing_obs={entry["id"] for entry in base_entries},
        img_base_addr=img_base_addr,
        default_model=MAIN_MODEL,
    )
    entries = base_entries + standard_entries
    # Optional table of labelled patches, for the patch-vs-strip comparison below.
    patch_record_df = strip_config.load_patch_comparison_record(postproc_cfg)

    # --- Build the model and predictor ----------------------------------------
    # get_model returns the trained network plus its config; criterion is the
    # loss used to score predictions. HiPredict wraps the model and takes care of
    # normalisation and sliding-window inference over full-size strips.
    cnn_model, model_cfg = get_model(cnn_name)
    cnn_model = cnn_model.to(device).eval()
    criterion = get_criterion(model_cfg).to(device)

    pred = HiPredict(
        cnn_model,
        model_cfg,
        smooth_norm_kernel=smooth_norm_kernel,
        device=device,
        verbose=verbose,
    )
    print(f"Setup complete after {time.time() - start:.2f}s (norm mode: {pred.norm_mode}, smooth kernel: {smooth_norm_kernel})\n\n")
    if standard_entries:
        print(
            "Selected standard eval observations: "
            + ", ".join(entry["id"] for entry in standard_entries)
        )

    # Sliding-window settings: the model runs over `window_size`-tall tiles with
    # `pad_x`/`pad_y` overlap so tile boundaries don't leave visible seams.
    window_size = 2 ** 13
    pad_x = 256    # Vertical padding
    pad_y = 64     # Horizontal padding
    print(f"pad_x: {pad_x}, pad_y: {pad_y}, window_size: {window_size}")

    results = []       # per-observation metric rows -> end-of-run summary
    failures = []      # observations whose inputs could not be loaded

    for entry in entries:
        obs = entry['id']
        group = strip_config.entry_group(entry)      # label used to bucket summary rows
        # Which band layouts to export (1 = predicted RED4 only, 3 = IR10/RED4/BG12).
        npy_band_counts = strip_exports.normalise_band_counts(entry, "npy_bands", default=1)
        tif_band_counts = strip_exports.normalise_band_counts(entry, "tif_bands", default=1)

        print(f"\n[{group}] {obs} started after {time.time()-start:.2f}s")

        # === 1. INPUT: load this observation's bands ==========================
        # `pred.load` returns one sample per observation; each sample is a dict
        # holding the input bands plus the ground-truth RED4 under key "y".
        try:
            imgs = pred.load(img_base_addr, mode="strips", obs_ids=[obs])
        except (FileNotFoundError, ValueError) as e:
            # A missing/unreadable observation shouldn't abort the whole run:
            # record it and carry on with the next one.
            failures.append((obs, group, str(e)))
            print(f"[{group}] {obs}: failed to load strip inputs: {e}")
            continue

        sample = imgs[0]
        r4_true = sample["y"]        # ground-truth RED4 (for scoring + saved as a reference)

        # HiPredict records per-tile normalisation stats as it runs; clear them
        # so each observation starts fresh and they don't accumulate across the loop.
        if hasattr(pred, "batch_stats"):
            try:
                pred.batch_stats.clear()
            except Exception:
                pred.batch_stats = []

        # === 2. APPLY MODEL: reconstruct RED4 across the whole strip ==========
        # The essential call is `pred.predict(imgs, ...)`, which slides the model
        # over the strip in tiles. `evaluate_pair` wraps it: it runs the main
        # model AND the SYN4 baseline and scores both against `r4_true`.
        strip_start = time.time()
        strip_result = strip_eval.evaluate_pair(
            pred,
            imgs,
            r4_true,
            criterion,
            device,
            predict_kwargs={"window_size": window_size, "pad_x": pad_x, "pad_y": pad_y},
        )
        output_img = strip_result["hip_output"]      # the reconstructed RED4 strip
        syn4_img = strip_result["syn4_output"]       # baseline output, for comparison/export

        # --- Score and report (not required to produce the output) ------------
        # Flatten metrics into a summary row and build the live per-strip line.
        # If patch comparison is enabled, repeat the evaluation on a single
        # labelled patch (a small crop) and append its numbers to the line.
        result = strip_eval.strip_record(obs, group, strip_result)

        line = strip_eval.format_result_line(
            group, obs, strip_result, strip_result["hip_stats"], time.time() - strip_start
        )
        patch_subset, patch_set, patch_idx = strip_config.first_patch_subset_for_obs(patch_record_df, obs)
        if patch_subset is not None:
            try:
                patch_imgs = pred.load(patch_subset, mode="patches")
                patch_result = strip_eval.evaluate_pair(
                    pred, [patch_imgs[0]], patch_imgs[0]["y"], criterion, device,
                )
                strip_eval.add_patch_record(result, patch_set, patch_idx, patch_result)
                line += "\n" + strip_eval.format_patch_line(group, obs, patch_set, patch_idx, patch_result)
            except (FileNotFoundError, ValueError, KeyError) as e:
                line += f"\n[{group}] {obs} patch comparison skipped: {e}"

        results.append(result)
        print(line)

        # === 3. SAVE: write the prediction(s) to disk =========================
        # `pred_arrays` maps model name -> reconstructed array. The YAML entry
        # decides which of these are written, in which formats and band layouts.
        pred_arrays = {cnn_name: output_img, SYN4_MODEL: syn4_img}
        strip_exports.save_observation_exports(
            entry,
            sample,
            pred_arrays,
            r4_true,
            npy_band_counts,
            tif_band_counts,
            SAVE_ROOT,
            verbose,
            img_base_dir=img_base_addr,
        )

        # Free the large strip arrays before moving to the next observation.
        del sample, imgs
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # Aggregate table across all observations (HP vs SYN4, per group).
    strip_eval.print_summary(results, failures)
    print(f"Finished processing images after {time.time() - start:.2f} seconds")
