import pandas as pd
import torch
import os
import numpy as np
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import defaultdict
import errno
import fcntl
from contextlib import contextmanager
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

from hipredict.predict.get_model import get_model
from hipredict.predict.utils import create_prediction_tile, save_individual_arrays
from hipredict.predict.loss import get_criterion
import hipredict.data.isis2np as i2np
from hipredict.inference.hipredict import HiPredict
from hipredict.data.load_metadata import gen_edr_df
from hipredict.inference.utils import create_icy_plot, save_domain_features, save_domain_imgs, save_domain_diffs
from hipredict.utils.utils import get_max_workers
from hipredict.paths import PACKAGE_ROOT, CONFIGS_DIR, files_dir

DOMAIN_CFG_LIST = [str(CONFIGS_DIR / 'icy_crater_coords.csv'), str(CONFIGS_DIR / 'ridge_spectra_coords.csv'), str(CONFIGS_DIR / 'chloride_coords.csv')]
img_base_addr = str(files_dir() / "isis_imgs" / "observations")
np_save_base = str(files_dir() / "npy_files_eval")
fig_savedir_base = str(files_dir() / "figs" / "domain_evals")
calib_path = str(PACKAGE_ROOT / "data" / "hiproc_pipe.sh")

MAX_ICY_PLOT_OBS = 6
REF_OFFSET = 100

OMMITED_OBS = ['ESP_036658_2550', 'ESP_036085_2455', 'ESP_036836_2360', 'ESP_059453_1335']

METRIC_KEYS = ("loss", "pwt", "nmae", "ssim", "bias")


def calc_metrics(pred, true, loss_val):
    data_range = true.max() - true.min()

    ssim_val = ssim(true, pred, data_range=data_range)
    nmae = 100.0 * np.mean(np.abs(pred - true)) / np.mean(np.abs(true))
    bias = np.mean(pred - true)
    tol = 0.01 * data_range
    pwt = np.mean(np.abs(pred - true) < tol) * 100

    return loss_val, pwt, nmae, ssim_val, bias


def save_center_spectral_plot(
    bg_img,
    true_img,
    syn4_img,
    pred_img,
    ir_img,
    obs_id,
    save_path,
    window_size=3,
):
    """
    Save a small spectral-profile plot using the mean DN from the center window.

    The plot uses three discrete band positions:
      BG12 -> RED4 -> IR10

    RED4 is shown for three sources:
      - True
      - Syn4
      - HiPredict
    """
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")

    def _sq(x):
        return x.squeeze() if x.ndim > 2 else x

    bg_img = _sq(bg_img)
    true_img = _sq(true_img)
    syn4_img = _sq(syn4_img)
    pred_img = _sq(pred_img)
    ir_img = _sq(ir_img)

    h, w = true_img.shape
    half = window_size // 2
    cy = int(np.clip(h // 2, half, h - 1 - half))
    cx = int(np.clip(w // 2, half, w - 1 - half))

    y_slice = slice(cy - half, cy + half + 1)
    x_slice = slice(cx - half, cx + half + 1)

    def _window_mean(img):
        return float(np.nanmean(img[y_slice, x_slice]))

    bg_val = _window_mean(bg_img)
    ir_val = _window_mean(ir_img)

    spectra = {
        "True": [bg_val, _window_mean(true_img), ir_val],
        "Syn4": [bg_val, _window_mean(syn4_img), ir_val],
        "HiPredict": [bg_val, _window_mean(pred_img), ir_val],
    }
    styles = {
        "True": {"color": "forestgreen", "marker": "o"},
        "Syn4": {"color": "tab:orange", "marker": "s"},
        "HiPredict": {"color": "tab:blue", "marker": "^"},
    }

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(3)
    red4_x = x[1]

    for name, values in spectra.items():
        ax.plot(
            x,
            values,
            linewidth=1.2,
            alpha=0.75,
            color=styles[name]["color"],
            label=name,
        )
        ax.scatter(
            [red4_x],
            [values[1]],
            s=40,
            color=styles[name]["color"],
            marker=styles[name]["marker"],
            zorder=3,
        )

    ax.scatter([x[0], x[2]], [bg_val, ir_val], s=22, color="black", marker="o", zorder=4)

    ax.set_xticks(x, ["BG12", "RED4", "IR10"])
    ax.set_xlabel("Spectral Band")
    ax.set_ylabel("DN")
    ax.set_title(f"{obs_id} center {window_size}x{window_size} spectral profile")
    ax.grid(axis="y", alpha=0.3)
    ax.set_xlim(-0.15, 2.15)
    ax.legend(fontsize=9)
    fig.tight_layout()

    base_dir = os.path.dirname(save_path)
    spectral_dir = os.path.join(base_dir, "spectral_profiles")
    os.makedirs(spectral_dir, exist_ok=True)
    outpath = os.path.join(spectral_dir, f"{obs_id}_center_spectrum.png")
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)

    return outpath


@contextmanager
def _obs_file_lock(img_base_addr, real_obs, timeout_s=1800, poll_s=0.2):
    """
    Cross-process lock per observation using a lock file in the shared obs dir.
    Prevents concurrent HiPROC / cube generation collisions for the same observation.
    """
    obs_dir = os.path.join(img_base_addr, real_obs)
    os.makedirs(obs_dir, exist_ok=True)

    lock_path = os.path.join(obs_dir, ".compute_observation.lock")
    fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o664)

    start = time.time()
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (errno.EACCES, errno.EAGAIN):
                    raise
                if (time.time() - start) > timeout_s:
                    raise TimeoutError(f"Timeout waiting for obs lock: {real_obs}")
                time.sleep(poll_s)

        yield

    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _compute_observation_task(task):
    """
    Worker wrapper for process pool. Keeps payload small and returns enough
    metadata for downstream processing in the main process.
    """
    try:
        with _obs_file_lock(task["img_base_addr"], task["real_obs"]):
            result, record = i2np.compute_observation(
                task["np_save"],
                task["img_base_addr"],
                task["real_obs"],
                task["calib_path"],
                task["np_exists"],
                1,
                task["base_idx"],
                patch_x=task["x_coord"],
                patch_y=task["y_coord"],
            )

        return {
            "ok": True,
            "unique_obs_id": task["unique_obs_id"],
            "real_obs": task["real_obs"],
            "desired_channel": task["desired_channel"],
            "x_coord": task["x_coord"],
            "y_coord": task["y_coord"],
            "np_save": task["np_save"],
            "result": result,
            "record": record,
        }

    except Exception as e:
        return {
            "ok": False,
            "unique_obs_id": task["unique_obs_id"],
            "real_obs": task["real_obs"],
            "error": repr(e),
        }


def _make_unique_obs_id(real_obs, obs_counter):
    count = obs_counter[real_obs]
    obs_counter[real_obs] += 1
    return real_obs if count == 0 else f"{real_obs}_{count}"


def _build_domain_tasks(df_filtered):
    tasks = []
    obs_counter = defaultdict(int)

    for base_idx, (_, row) in enumerate(df_filtered.iterrows(), start=1):
        real_obs = row["Observation"]
        unique_obs_id = _make_unique_obs_id(real_obs, obs_counter)
        desired_channel = row["channel"]

        x_coord = int(row["x"])
        y_coord = int(row["y"])
        np_save = f"{np_save_base}/{unique_obs_id}"
        os.makedirs(np_save, exist_ok=True)

        tasks.append({
            "np_save": np_save,
            "img_base_addr": img_base_addr,
            "real_obs": real_obs,
            "calib_path": calib_path,
            "np_exists": False,
            "base_idx": base_idx,
            "x_coord": x_coord,
            "y_coord": y_coord,
            "unique_obs_id": unique_obs_id,
            "desired_channel": desired_channel,
        })

    return sorted(tasks, key=lambda task: task["real_obs"])


def _load_domain_samples(tasks, domain_meta, hiproc_env, mp_ctx, max_workers, predictor, start_time):
    records = []
    sample_meta_by_set = {}

    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=mp_ctx,
        initializer=i2np.init_worker,
        initargs=(domain_meta, hiproc_env),
    ) as ex:
        futures = [ex.submit(_compute_observation_task, task) for task in tasks]

        for fut in as_completed(futures):
            try:
                out = fut.result()
            except Exception as e:
                print(f"ERROR: Future failed before returning payload: {e}")
                continue

            if not out["ok"]:
                print(f"ERROR: Unexpected error for observation {out['unique_obs_id']}: {out['error']}")
                continue

            unique_obs_id = out["unique_obs_id"]
            real_obs = out["real_obs"]
            desired_channel = out["desired_channel"]
            x_coord = out["x_coord"]
            y_coord = out["y_coord"]
            result = out["result"]
            record = out["record"]

            if record is None:
                print(f"ERROR: Observation {unique_obs_id}: no record returned, continuing...")
                continue

            if result.get(real_obs, 2) != 0:
                print(f"Failed to compute observation {unique_obs_id}, continuing...")
                continue

            set_ids = {row[1] for row in record}
            if len(set_ids) != 1:
                print(f"ERROR: Observation {unique_obs_id} produced multiple set ids: {sorted(set_ids)}")
                continue

            set_id = int(next(iter(set_ids)))
            sample_meta_by_set[set_id] = {
                "unique_obs_id": unique_obs_id,
                "real_obs": real_obs,
                "desired_channel": desired_channel,
                "x_coord": x_coord,
                "y_coord": y_coord,
            }
            records.extend(record)

            print(
                f"\tProcessed observation {real_obs} ({desired_channel}) with patch centered on "
                f"x_coord={x_coord}, y_coord={y_coord} - Time: {time.time() - start_time:.2f}s"
            )

    if not records:
        return [], []

    data_record = (
        pd.DataFrame(records, columns=i2np.DATA_RECORD_COLUMNS)
        .sort_values("Set")
        .reset_index(drop=True)
    )
    samples = predictor.load(data_record, mode="patches")
    sample_meta = [sample_meta_by_set[int(sample["set_name"])] for sample in samples]
    return samples, sample_meta


def _new_metrics_store():
    return {name: {key: [] for key in METRIC_KEYS} for name in ("HiPred", "Syn4")}


def _append_metrics(metrics_store, model_name, metric_values):
    for key, value in zip(METRIC_KEYS, metric_values):
        metrics_store[model_name][key].append(value)


def _print_metrics_summary(metrics_store, header):
    print("\n" + "=" * len(header))
    print(f"{'SUMMARY':^15} | {'Type':<8} | {'Mean ± Std':<40}")
    print("=" * len(header))
    for model_name in ("HiPred", "Syn4"):
        avg_l = np.mean(metrics_store[model_name]["loss"])
        avg_p = np.mean(metrics_store[model_name]["pwt"])
        avg_n = np.mean(metrics_store[model_name]["nmae"])
        avg_s = np.mean(metrics_store[model_name]["ssim"])
        std_l = np.std(metrics_store[model_name]["loss"])
        std_p = np.std(metrics_store[model_name]["pwt"])
        std_n = np.std(metrics_store[model_name]["nmae"])
        std_s = np.std(metrics_store[model_name]["ssim"])

        print(f"{'AVERAGE':<15} | {model_name:<8} | Loss: {avg_l:.2e} (±{std_l:.1e})  | SSIM: {avg_s:.4f} (±{std_s:.2f})")
        print(f"{'':<15} | {'':<8} | PWT:  {avg_p:05.2f}% (±{std_p:.2f})       | NMAE: {avg_n:.4f} (±{std_n:.4f})")
        print("-" * len(header))


def _save_sample_artifacts(sample, output_img, syn4_img, unique_obs_id, dir_name):
    x_band = sample["x_band"]
    x_neigh = sample["x_neigh"]
    true_img = sample["y"]
    tile_save_path = (
        f"{fig_savedir_base}"
        f"{dir_name}/tiles_{unique_obs_id}.png"
    )

    create_prediction_tile(
        ir_img=x_band[0],
        bg_img=x_band[1],
        red3_img=x_neigh[5, :, :],
        red5_img=x_neigh[2, :, :],
        pred_img=output_img,
        true_img=true_img,
        base_img=syn4_img,
        obs_id=unique_obs_id,
        set_name="Inference",
        save_path=tile_save_path
    )

    save_individual_arrays(
        ir_img=x_band[0],
        bg_img=x_band[1],
        red3_img=x_neigh[5, :, :],
        red5_img=x_neigh[2, :, :],
        pred_img=output_img,
        true_img=true_img,
        base_img=syn4_img,
        obs_id=unique_obs_id,
        save_path=tile_save_path
    )

    save_center_spectral_plot(
        bg_img=x_band[1],
        true_img=true_img,
        syn4_img=syn4_img,
        pred_img=output_img,
        ir_img=x_band[0],
        obs_id=unique_obs_id,
        save_path=tile_save_path,
        window_size=3,
    )


def _pending_hiproc_obs(domain_cfg_list):
    """
    Observations across every domain config whose processed cubes are not all
    on disk yet. These are the only ones that need the hiproc+isis backend.
    """
    pending = set()
    for domain_cfg in domain_cfg_list:
        domain_df = pd.read_csv(domain_cfg)
        df_filtered = domain_df[~domain_df['Observation'].isin(OMMITED_OBS)]
        for real_obs in df_filtered['Observation'].unique():
            if not i2np.processed_cubes_complete(img_base_addr, real_obs):
                pending.add(real_obs)
    return sorted(pending)


def _get_hiproc_env(domain_cfg_list):
    """
    Snapshot the hiproc+isis backend, but only if some observation actually
    needs it. Returns None when every cube is already processed, or when the
    backend is unavailable - observations needing it are then skipped rather
    than taking the whole run down with them.
    """
    pending = _pending_hiproc_obs(domain_cfg_list)
    if not pending:
        print("All processed cubes present - skipping hiproc+isis backend snapshot.")
        return None

    print(f"{len(pending)} observation(s) need HiPROC processing: {', '.join(pending)}")
    try:
        return i2np.snapshot_hiproc_env()
    except Exception as e:
        print(
            f"WARNING: hiproc+isis backend unavailable ({e!r}).\n"
            f"         Continuing without it; the {len(pending)} observation(s) above will be skipped."
        )
        return None


def _stack_domain_arrays(samples, batch_outputs, batch_syn4):
    true_b = np.stack([sample["y"] for sample in samples], axis=0)
    pred_b = np.stack(batch_outputs, axis=0) if not isinstance(batch_outputs, np.ndarray) else batch_outputs
    syn4_b = np.stack(batch_syn4, axis=0) if not isinstance(batch_syn4, np.ndarray) else batch_syn4
    ir_b = np.stack([sample["x_band"][0] for sample in samples], axis=0)
    bg_b = np.stack([sample["x_band"][1] for sample in samples], axis=0)
    return true_b, pred_b, syn4_b, ir_b, bg_b


if __name__ == "__main__":
    start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    EDR_df = gen_edr_df()
    hiproc_env = _get_hiproc_env(DOMAIN_CFG_LIST)

    mp_ctx = mp.get_context("spawn")
    max_workers = get_max_workers(max_cap=8, leave_free=1, verbose=True)

    cnn_name = "hipredict"
    cnn_model, cfg = get_model(cnn_name, device=device)
    criterion = get_criterion(cfg).to(device)
    pred = HiPredict(cnn_model, cfg, device=device)

    print(f"Setup complete after {time.time() - start:.2f}s\n\n")

    icy_all_true = []
    icy_all_pred = []
    icy_all_syn4 = []
    icy_all_ir = []
    icy_all_bg = []
    icy_all_obs_ids = []
    icy_all_is_icy = []

    for domain_cfg in DOMAIN_CFG_LIST:
        print(f"\n=== {domain_cfg} ===")
        print(f"\tStarted after {time.time() - start:.2f}s", flush=True)

        domain_df = pd.read_csv(domain_cfg)
        df_filtered = domain_df[~domain_df['Observation'].isin(OMMITED_OBS)]
        unique_obs_list = sorted(df_filtered['Observation'].unique())
        domain_meta = i2np.compute_meta_dict(EDR_df, unique_obs_list)
        tasks = _build_domain_tasks(df_filtered)

        if not tasks:
            print(f"No valid observations found for domain config {domain_cfg}. Skipping.")
            continue

        samples, sample_meta = _load_domain_samples(
            tasks,
            domain_meta,
            hiproc_env,
            mp_ctx,
            max_workers,
            pred,
            start,
        )
        if not samples:
            print(f"No valid observations found for domain config {domain_cfg}. Skipping.")
            continue

        print(f"\tCompleted domain config {domain_cfg} - Batch size: {len(samples)} - Time: {time.time() - start:.2f}s", flush=True)

        batch_outputs = pred.predict(samples, window_size=512, pad_x=50, pad_y=32)
        batch_syn4 = pred.predict(samples, syn4=True)

        print(f"Inference complete. Processing results individually... Time: {time.time() - start:.2f}s")

        assert len(batch_outputs) == len(samples)
        assert len(batch_syn4) == len(samples)

        metrics_store = _new_metrics_store()

        header = f"{'Obs ID':<17} | {'Type':<8} | {'Loss':<8}   | {'PWT 1%':<8} | {'NMAE':<8} | {'SSIM':<8} | {'Bias':<8}"
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        dir_name = os.path.splitext(os.path.basename(domain_cfg))[0].lower()

        print(f"Processing {len(samples)} observations for domain '{dir_name} - Time: {time.time() - start:.2f}s", flush=True)
        for sample, meta, output_img, syn4_img in zip(samples, sample_meta, batch_outputs, batch_syn4):
            unique_obs = meta["unique_obs_id"]
            real_obs = meta["real_obs"]
            r4_true = sample["y"]

            output_t = torch.from_numpy(output_img).float().to(device)
            r4_true_t = torch.from_numpy(r4_true).float().to(device)
            syn4_t = torch.from_numpy(syn4_img).float().to(device)

            pred_loss, _ = criterion(output_t, r4_true_t)
            syn4_loss, _ = criterion(syn4_t, r4_true_t)

            hp_metrics = calc_metrics(output_img, r4_true, pred_loss.item())
            sy_metrics = calc_metrics(syn4_img, r4_true, syn4_loss.item())
            _append_metrics(metrics_store, "HiPred", hp_metrics)
            _append_metrics(metrics_store, "Syn4", sy_metrics)

            hp_l, hp_p, hp_n, hp_s, hp_b = hp_metrics
            sy_l, sy_p, sy_n, sy_s, sy_b = sy_metrics

            print(f"{unique_obs:<17} | {'HiPred':<8} | {hp_l:.2e}   | {hp_p:05.2f}%   | {hp_n:.4f}   | {hp_s:.4f}   | {hp_b:.2e}")
            print(f"{'':<17} | {'Syn4':<8} | {sy_l:.2e}   | {sy_p:05.2f}%   | {sy_n:.4f}   | {sy_s:.4f}   | {sy_b:.2e}")
            print("-" * len(header))

            _save_sample_artifacts(sample, output_img, syn4_img, unique_obs, dir_name)

        _print_metrics_summary(metrics_store, header)

        print(f"Finished processing domain {domain_cfg} - Time: {time.time() - start:.2f}s", flush=True)
        is_icy = dir_name.startswith("icy_")

        quicklook_savedir = f"{fig_savedir_base}{dir_name}/quicklooks/"
        true_b, pred_b, syn4_b, ir_b, bg_b = _stack_domain_arrays(samples, batch_outputs, batch_syn4)
        labels = [f"{meta['unique_obs_id']} ({meta['real_obs']})" for meta in sample_meta]
        file_ids = [meta["unique_obs_id"] for meta in sample_meta]

        save_domain_imgs(
            true=true_b,
            pred=pred_b,
            syn4=syn4_b,
            ir=ir_b,
            bg=bg_b,
            savedir=quicklook_savedir,
            labels=labels,
            file_ids=file_ids,
            annotate_center=True,
            box_half_size=3,
            save_num_in=20,
            ref_offset=REF_OFFSET
        )

        save_domain_diffs(
            true=true_b,
            pred=pred_b,
            syn4=syn4_b,
            ir=ir_b,
            bg=bg_b,
            savedir=quicklook_savedir,
            labels=labels,
            file_ids=file_ids,
            annotate_center=True,
            box_half_size=3,
            save_num_in=20,
            ref_offset=REF_OFFSET
        )

        save_domain_features(
            true=true_b,
            pred=pred_b,
            syn4=syn4_b,
            ir=ir_b,
            bg=bg_b,
            savedir=quicklook_savedir,
            labels=labels,
            file_ids=file_ids,
            annotate_center=False,
            save_num_in=20,
        )
        print(f"Domain images and diffs saved to {quicklook_savedir} - Time: {time.time() - start:.2f}\n\n")

        icy_all_true.append(true_b)
        icy_all_pred.append(pred_b)
        icy_all_syn4.append(syn4_b)
        icy_all_ir.append(ir_b)
        icy_all_bg.append(bg_b)
        icy_all_obs_ids.extend(file_ids)
        icy_all_is_icy.extend([is_icy] * len(file_ids))

    if len(icy_all_true) > 0:
        icy_true = np.concatenate(icy_all_true, axis=0)
        icy_pred = np.concatenate(icy_all_pred, axis=0)
        icy_syn4 = np.concatenate(icy_all_syn4, axis=0)
        icy_ir = np.concatenate(icy_all_ir, axis=0)
        icy_bg = np.concatenate(icy_all_bg, axis=0)
        icy_obs_ids = list(icy_all_obs_ids)
        icy_mask = np.array(icy_all_is_icy, dtype=bool)

        idx_icy = np.where(icy_mask)[0]
        if idx_icy.size == 0:
            print("No icy observations found for icy plot. Skipping.")
        else:
            if MAX_ICY_PLOT_OBS is not None and idx_icy.size > int(MAX_ICY_PLOT_OBS):
                idx_icy = idx_icy[:int(MAX_ICY_PLOT_OBS)]

            icy_true = icy_true[idx_icy]
            icy_pred = icy_pred[idx_icy]
            icy_syn4 = icy_syn4[idx_icy]
            icy_ir = icy_ir[idx_icy]
            icy_bg = icy_bg[idx_icy]
            icy_obs_ids = [icy_obs_ids[i] for i in idx_icy.tolist()]

            create_icy_plot(
                icy_true,
                icy_pred,
                icy_syn4,
                icy_ir,
                icy_bg,
                f"{fig_savedir_base}/icy_plots",
                obs_ids=icy_obs_ids,
                use_center_pixel=True,
                ref_offset=REF_OFFSET,
                plot_reference=True
            )
