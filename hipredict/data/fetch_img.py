import os
import concurrent.futures
import asyncio
import aiohttp
from urllib.parse import urljoin
import random

from hipredict.utils.utils import download_img_async

BASE_URL = "https://hirise-pds.lpl.arizona.edu/PDS/"


def _build_expected_specs_map(EDR_df, required_ccds):
    filtered = EDR_df.loc[
        EDR_df["CCD_NAME"].isin(required_ccds),
        ["OBSERVATION_ID", "FILE_NAME_SPECIFICATION"],
    ]
    return (
        filtered.groupby("OBSERVATION_ID")["FILE_NAME_SPECIFICATION"]
        .agg(list)
        .to_dict()
    )


def _list_existing_files(obs_dir):
    if not os.path.isdir(obs_dir):
        return set()

    with os.scandir(obs_dir) as entries:
        return {entry.name for entry in entries if entry.is_file()}


def _missing_file_specs(obs_dir, expected_specs):
    existing_files = _list_existing_files(obs_dir)
    return [
        file_spec
        for file_spec in expected_specs
        if os.path.basename(file_spec) not in existing_files
    ]


def _process_observation(
    obs_id,
    expected_specs,
    download_items,
    img_save_dir,
    workers,
    download_stats,
    expected_count,
):
    obs_dir = os.path.join(img_save_dir, obs_id)
    os.makedirs(obs_dir, exist_ok=True)

    assert len(expected_specs) == expected_count, (
        f"Expected {expected_count} images for observation {obs_id}, "
        f"found {len(expected_specs)}"
    )

    if not download_items:
        print(f"All images already downloaded for observation ID {obs_id}")
        return

    async def download_task_async(file_spec, session, sem, max_attempts=20, debug=False):
        """
        Async download task:
        - construct full url and local path
        - Uses sem to limit concurrency
        - Measures download duration and speed
        """
        path = urljoin(BASE_URL, file_spec)
        local_file = os.path.join(obs_dir, os.path.basename(file_spec))
        delay = 1
        e_msg = None

        for attempt in range(1, max_attempts + 1):
            jitter = random.uniform(0.5, 1.5)
            try:
                async with sem:
                    result = await download_img_async(session, path, local_file)
                return result

            except aiohttp.ClientResponseError as e:
                retry_after = e.headers.get("Retry-After") if e.headers else None
                if retry_after:
                    if debug:
                        print(f"Retry after: {retry_after}")
                if e.status == 503:
                    e_msg = "503 Service Unavailable"
                    await asyncio.sleep(delay + jitter)
                else:
                    e_msg = f"{e.status} {e.message}"
                    if debug:
                        print(f"ERROR: {e} - {e.__class__.__name__} - {repr(e)}. Attempt {attempt}/{max_attempts}. Retrying in {delay + jitter:.2f} seconds...")
                    await asyncio.sleep(delay + jitter)
            except Exception as e:
                e_msg = str(e)
                if debug:
                    print(f"ERROR: {e} - {e.__class__.__name__} - {repr(e)}. Attempt {attempt}/{max_attempts}. Retrying in {delay + jitter:.2f} seconds...")
                await asyncio.sleep(delay + jitter)

            delay += 12

        print(f"ERROR: Failed to download {path}: {e_msg} after {max_attempts} attempts.")
        return None

    async def download_all():
        sem = asyncio.Semaphore(workers)
        connector = aiohttp.TCPConnector(limit=workers, use_dns_cache=True)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = [
                download_task_async(file_spec, session, sem)
                for file_spec in download_items
            ]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    print(f"Error: {result}")

    asyncio.run(download_all())


def fetch_imgs_parallel(
    EDR_df, obs_ids, img_save_dir, workers, download_stats, required_ccds
):
    expected_count = len(required_ccds) * 2
    expected_specs_by_obs = _build_expected_specs_map(EDR_df, required_ccds)

    obs_ids = list(dict.fromkeys(obs_ids))

    edr_obs_ids = set(EDR_df["OBSERVATION_ID"].unique())
    missing_obs = [obs_id for obs_id in obs_ids if obs_id not in edr_obs_ids]

    if missing_obs:
        raise Exception(f"Observation IDs {missing_obs} not found in the EDR dataframe")

    obs_jobs = []          # (obs_id, expected_specs, download_items, eff_count)
    complete_obs = []
    edr_mismatch_obs = set()

    for obs_id in obs_ids:
        expected_specs = expected_specs_by_obs.get(obs_id, [])
        if len(expected_specs) != expected_count:
            print(
                f"WARNING: Expected {expected_count} images for observation {obs_id}, "
                f"but found {len(expected_specs)} in EDR dataframe. This may indicate "
                "missing CCDs or an issue with the EDR dataframe."
            )
            edr_mismatch_obs.add(obs_id)
            # Use len(expected_specs) as effective count so the assert in _process_observation passes
            obs_jobs.append((obs_id, expected_specs, expected_specs, len(expected_specs)))
            continue

        download_items = _missing_file_specs(
            os.path.join(img_save_dir, obs_id),
            expected_specs,
        )
        if download_items:
            obs_jobs.append((obs_id, expected_specs, download_items, expected_count))
        else:
            complete_obs.append(obs_id)

    # Pre-download classification summary
    normal_download = [o for o, *_ in obs_jobs if o not in edr_mismatch_obs]
    mismatch_download = [o for o, *_ in obs_jobs if o in edr_mismatch_obs]
    print(f"\n--- Pre-download classification ({len(obs_ids)} observations) ---")
    if complete_obs:
        print(f"  {len(complete_obs):>3} already complete (all files on disk)        : {complete_obs}")
    if normal_download:
        print(f"  {len(normal_download):>3} to download (files missing)                 : {normal_download}")
    if mismatch_download:
        print(f"  {len(mismatch_download):>3} to download (EDR count mismatch, WARNING)   : {mismatch_download}")
    print("---\n")

    if not obs_jobs:
        print("All observations are already complete.")
        return

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_obs = {
            executor.submit(
                _process_observation,
                obs_id,
                expected_specs,
                download_items,
                img_save_dir,
                workers,
                download_stats,
                eff_count,
            ): obs_id
            for obs_id, expected_specs, download_items, eff_count in obs_jobs
        }
        concurrent.futures.wait(future_to_obs)

    # Check for exceptions raised during download
    download_exceptions = {
        obs_id: future.exception()
        for future, obs_id in future_to_obs.items()
        if future.exception() is not None
    }
    if download_exceptions:
        for obs_id, exc in download_exceptions.items():
            print(f"ERROR: {obs_id} raised {type(exc).__name__}: {exc}")

    # Post-download per-obs verification
    newly_complete = []
    newly_incomplete = []
    for obs_id, expected_specs, _, _ in obs_jobs:
        if obs_id in download_exceptions:
            continue
        missing = _missing_file_specs(os.path.join(img_save_dir, obs_id), expected_specs)
        if not missing:
            newly_complete.append(obs_id)
        else:
            newly_incomplete.append((obs_id, len(expected_specs) - len(missing), len(expected_specs)))

    failed_obs = list(download_exceptions.keys())
    print(f"--- Download results ({len(obs_ids)} observations) ---")
    if complete_obs:
        print(f"  {len(complete_obs):>3} already complete, no download needed         : {complete_obs}")
    if newly_complete:
        print(f"  {len(newly_complete):>3} successfully downloaded (all files present)  : {newly_complete}")
    if newly_incomplete:
        detail = [f"{o} ({got}/{exp} files)" for o, got, exp in newly_incomplete]
        print(f"  {len(newly_incomplete):>3} incomplete after download                    : {detail}")
    if failed_obs:
        detail = [f"{o} ({type(download_exceptions[o]).__name__})" for o in failed_obs]
        print(f"  {len(failed_obs):>3} failed (exception during download)            : {detail}")
    print("---\n")


if __name__ == "__main__":
    from hipredict.download import main
    main()
