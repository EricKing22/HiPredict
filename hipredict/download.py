from pathlib import Path

from hipredict.data.fetch_img import fetch_imgs_parallel
from hipredict.data.load_metadata import gen_edr_df
from hipredict.utils.utils import check_obs_list


def Download(obs_list, img_base_dir, max_obs=None, required_ccds=["RED3", "RED4", "RED5", "IR10", "BG12"], workers=None, print_stats=False):
    """
    Download HiRISE EDR image files for a list of observations.

    Fetches the required CCD channel files (.IMG) for each observation in
    parallel, skipping any already present on disk.

    Args:
        obs_list: Observation IDs as a list of strings or a path to a .txt
            file with one ID per line.
        img_base_dir: Root directory under which downloaded images are saved.
        max_obs: Cap on the number of observations to process. If None, all
            observations are processed.
        required_ccds: CCD channels to download per observation. Each channel
            produces two .IMG files (even/odd detector lines), so the expected
            file count is len(obs_ids) * len(required_ccds) * 2.
        workers: Number of parallel download workers. Defaults to the
            fetch_imgs_parallel default when None.
        print_stats: If True, print per-file download statistics.
    """
    obs_ids = check_obs_list(obs_list)
    total_obs = len(obs_ids)

    EDR_df = gen_edr_df()
    img_base_dir = Path(img_base_dir)

    if max_obs is not None:
        obs_ids = obs_ids[:max_obs]

    limit_applied = (max_obs is not None) and (max_obs < total_obs)
    if isinstance(obs_list, str):
        print(f"Obs txt: {obs_list}")
    print(f"Observations provided: {total_obs:,}")
    if max_obs is not None:
        print(f"max_obs: {max_obs:,} (limit applied: {limit_applied})")
    print(f"Number of observations to process: {len(obs_ids):,}")

    fetch_imgs_parallel(
        EDR_df,
        obs_ids,
        img_base_dir,
        workers,
        download_stats=print_stats,
        required_ccds=required_ccds
    )

    n_obs = len(obs_ids)
    expected_imgs = n_obs * len(required_ccds) * 2
    tot_imgs = 0
    tot_bytesize = 0
    for obs_id in obs_ids:
        obs_dir = img_base_dir / obs_id
        if obs_dir.is_dir():
            for f in obs_dir.iterdir():
                if f.is_file() and f.suffix.upper() == ".IMG":
                    tot_imgs += 1
                    tot_bytesize += f.stat().st_size
    tot_gbs = tot_bytesize / 1024 / 1024 / 1024

    print(f"Downloaded/found {tot_imgs:,} .IMG files across {n_obs:,} requested observations (expected {expected_imgs:,} files). Total size: {tot_gbs:.2f} GB")


def main():
    import argparse
    from hipredict.paths import files_dir
    parser = argparse.ArgumentParser(description="Download the five HiRISE detectors used by HiPredict.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--obs", nargs="+", help="Observation IDs")
    source.add_argument("--obs-list", help="Text file with one observation ID per line")
    parser.add_argument("--img-dir", default=str(files_dir() / "isis_imgs" / "observations"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-obs", type=int)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    Download(args.obs or args.obs_list, args.img_dir, max_obs=args.max_obs,
             workers=args.workers, print_stats=args.verbose)


if __name__ == "__main__":
    main()
