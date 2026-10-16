"""Predict selected HiRISE observations with HiPredict and/or SYN4."""

import argparse
from pathlib import Path

import numpy as np
import torch

from hipredict import Download, PrepObservations, HiPredict, get_model
from hipredict.paths import files_dir
from hipredict.inference.strip_exports import restore_crop
from hipredict.inference.utils import np_to_tiff, np_to_cube
from hipredict.utils.utils import check_obs_list


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--obs", nargs="+")
    source.add_argument("--obs-list")
    parser.add_argument("--img-dir", type=Path, default=files_dir() / "isis_imgs" / "observations")
    parser.add_argument("--output-dir", type=Path, default=files_dir() / "predictions")
    parser.add_argument("--model", choices=["hipredict", "SYN4", "both"], default="hipredict")
    parser.add_argument("--format", choices=["npy", "tif", "cub"], nargs="+", default=["npy", "tif"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--preprocess", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--hf-repo")
    parser.add_argument("--revision")
    args = parser.parse_args()
    obs_ids = check_obs_list(args.obs or args.obs_list)
    if args.download:
        Download(obs_ids, args.img_dir, workers=args.workers)
    if args.preprocess:
        PrepObservations(obs_list=obs_ids, img_base_dir=args.img_dir, workers=args.workers)
    model, cfg = get_model("SYN4" if args.model == "SYN4" else "hipredict",
                           device=args.device, hf_repo_id=args.hf_repo, revision=args.revision)
    predictor = HiPredict(model, cfg, device=args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for obs_id in obs_ids:
        samples = predictor.load(args.img_dir, mode="strips", obs_ids=[obs_id])
        outputs = {}
        settings = dict(window_size=8192, pad_x=256, pad_y=64)
        if args.model != "SYN4":
            outputs["HiPredict"] = predictor.predict(samples, **settings)[0]
        if args.model in {"SYN4", "both"}:
            outputs["SYN4"] = predictor.predict(samples, syn4=True, **settings)[0]
        for name, output in outputs.items():
            full = restore_crop(output, samples[0]["original_shape"], samples[0]["crop_dims"])
            full = full.astype(np.float32, copy=False)
            stem = args.output_dir / f"{obs_id}_{name}"
            if "npy" in args.format: np.save(str(stem) + ".npy", full)
            if "tif" in args.format: np_to_tiff(full, str(stem) + ".tif", dtype=np.float32, stretch=False)
            if "cub" in args.format:
                reference = args.img_dir / obs_id / f"{obs_id}_RED5_processed.cub"
                np_to_cube(full, str(reference), str(stem) + ".cub")
        print(f"Saved {obs_id}: {', '.join(outputs)} to {args.output_dir}")


if __name__ == "__main__":
    main()
