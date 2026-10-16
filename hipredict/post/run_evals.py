import pandas as pd
import yaml
import torch
import os
import numpy as np
from skimage.metrics import structural_similarity as sk_ssim
import joblib
import torch.nn.functional as F
import gc

from hipredict.predict.get_model import get_model
from hipredict.predict.utils import split_by_date
from hipredict.predict.dataset import FilteredHiRISEDataset, get_loader, collate_fn
from hipredict.data.normalization import normalization_kwargs_from_config
from hipredict.predict.loss import get_criterion

from hipredict.post.evals.eval_plots import plot_evals, plot_syn4_vs_hipredict_hexbin, plot_syn4_binned_improvement, plot_syn4_vs_hipredict_binned, plot_syn4_vs_hipredict_hist2d
from hipredict.post.evals.eval_correlations import corr_evals
from hipredict.post.evals.eval_meta_bins import plot_nmae_hist_by_metadata
from hipredict.post.evals.eval_passes import run_train_val_eval_pass, RESIDUAL_BLUR_KERNELS
from hipredict.post.evals.eval_summary import build_eval_summary
from hipredict.post.evals.eval_reporting import print_global_eval_report, print_worst_syn4_subset_report
from hipredict.post.evals.eval_categories import run_category_evals
from hipredict.post.evals.eval_cat_reporting import save_category_results_and_print, compare_global_vs_category_stats
from hipredict.post.evals.eval_train_vs_eval import compare_train_eval_sets
from hipredict.predict.utils import worker_init_fn
from hipredict.paths import CONFIGS_DIR, files_dir


DATA_RECORD = str(files_dir() / "data_record_alls.csv")
EVAL_DR = str(files_dir() / "data_record_evals.csv")
EVALS_CONFIG = str(CONFIGS_DIR / 'evals.yaml')
MODEL_NAME = "hipredict"
SAVE_DIR = str(files_dir() / "evals")

PWT_THRESH = 0.01  # Threshold for perc. within tol. metric
PIX_SAMPLES_PER_IMAGE = 20  # Number of aligned true/pred pixels to sample per image for downstream plots
WORST_SYN4_TOP_N = 100
PWT_PERC = int(100 * PWT_THRESH)  # in percentage points


def _to_hw(img):
    """Ensure (H,W) numpy float64"""
    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim == 4:
        arr = arr[0, 0]
    elif arr.ndim == 3:
        if arr.shape[0] in (1, 3):
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
    return arr


def _per_img_dynamic_range(y):
    ymin = np.min(y)
    ymax = np.max(y)
    dy_range = float(max(ymax - ymin, 1e-8))
    return ymin, ymax, dy_range


def _ssim_safe(y_true, y_pred):
    """Input as 2d arrs"""
    tmin, tmax = np.min(y_true), np.max(y_true)
    rng = max(tmax - tmin, 1e-8)
    yt = (y_true - tmin) / rng
    yp = (y_pred - tmin) / rng
    return float(sk_ssim(yt, yp, data_range=1.0))      # type: ignore[arg-type]


def compute_per_img_metrics(y_true, y_pred, pwt_threshold=0.01, use_magnitude=True):
    """Input as batch tensors (B,C,H,W)
    Normalise by magnitude by default"""
    y_true_b = y_true.detach().cpu().numpy()
    y_pred_b = y_pred.detach().cpu().numpy()
    if y_true_b.ndim == 4 and y_true_b.shape[1] == 1:
        y_true_b = y_true_b[:, 0]
        y_pred_b = y_pred_b[:, 0]

    pwt_list, nmae_list, bias_list, ssim_list = [], [], [], []
    for i in range(y_true_b.shape[0]):
        yt = _to_hw(y_true_b[i])
        yp = _to_hw(y_pred_b[i])

        _, _, dy_range = _per_img_dynamic_range(yt)
        dy_mag = np.mean(np.abs(yt)) if use_magnitude else dy_range

        err = yp - yt
        abs_err = np.abs(err)

        pwt = 100.0 * np.mean(abs_err <= (pwt_threshold * dy_range))
        nmae = 100.0 * np.mean(abs_err) / dy_mag        # Uses chosen denom
        bias = 100.0 * np.mean(err) / dy_mag            # Used chosen denom
        ssim = _ssim_safe(yt, yp)

        pwt_list.append(pwt)
        nmae_list.append(nmae)
        bias_list.append(bias)
        ssim_list.append(ssim)

    return {
        'pwt': np.array(pwt_list, dtype=np.float32),
        'nmae': np.array(nmae_list, dtype=np.float32),
        'bias': np.array(bias_list, dtype=np.float32),
        'ssim': np.array(ssim_list, dtype=np.float32),
    }


def summarize_metric(arr, fn="mean"):
    arr = np.asarray(arr, dtype=float)
    if fn == "median":
        return float(np.nanmedian(arr))
    elif fn == "mean":
        return float(np.nanmean(arr))
    else:
        raise ValueError(f"Unknown summary fn: {fn}")


def _blur7(x):
    k = 7
    x = F.pad(x, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
    return F.avg_pool2d(x, kernel_size=k, stride=1)


def print_all_models_eval_table(val_dataloader, device, compute_per_img_metrics, summarize_metric, pwt_thresh, cfg=None):
    """Evaluate every model on the val set and print one bias/ssim/pwt summary table.

    Self-contained: loads each model fresh via get_model, runs it over the val set in
    DN space, computes per-image metrics, then frees it before loading the next so we
    never hold all models on the GPU at once. Models that fail to load yield a 'nan' row
    rather than aborting the table.
    """
    # (display name, registry key) in the requested row order
    model_specs = [("SYN4", "SYN4"), ("HiPredict", "hipredict")]

    rows = []
    for disp_name, key in model_specs:
        try:
            if key == "SYN4":
                model, _ = get_model(key, device, cfg=cfg)
            else:
                model, _ = get_model(key, device)
            model = model.eval()
            nmae_all, bias_all, ssim_all, pwt_all = [], [], [], []
            with torch.no_grad():
                for batch in val_dataloader:
                    x_band = batch["x_band"].to(device).float()
                    x_neigh = batch["x_neigh"].to(device).float()
                    x_meta = batch["x_meta"].to(device).float()
                    y_true = batch["y"].to(device)
                    patch_x = batch["patch_x"].to(device).float()
                    obs_width = batch["obs_width"].to(device).float()
                    stats = batch["stats"].to(device).float()
                    a = stats[:, 0].view(-1, 1, 1, 1)
                    s = stats[:, 1].view(-1, 1, 1, 1)

                    y_pred = model(x_band, x_neigh, x_meta, patch_x=patch_x, obs_width=obs_width)
                    y_true_dn = y_true.float() * s + a
                    y_pred_dn = y_pred.float() * s + a

                    m = compute_per_img_metrics(y_true_dn, y_pred_dn, pwt_threshold=pwt_thresh, use_magnitude=True)
                    nmae_all.extend(m["nmae"].tolist())
                    bias_all.extend(m["bias"].tolist())
                    ssim_all.extend(m["ssim"].tolist())
                    pwt_all.extend(m["pwt"].tolist())

            rows.append((
                disp_name,
                {
                    name: (
                        summarize_metric(np.asarray(vals), "mean"),
                        summarize_metric(np.asarray(vals), "median"),
                        float(np.nanstd(np.asarray(vals, dtype=float))),
                    )
                    for name, vals in (("NMAE", nmae_all), ("Bias", bias_all), ("SSIM", ssim_all), ("PWT", pwt_all))
                },
            ))
        except Exception as e:
            print(f"WARNING: failed to evaluate '{disp_name}' ({key}): {type(e).__name__}: {e}")
            rows.append((disp_name, None))
        finally:
            try:
                del model
            except NameError:
                pass
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    # Each metric shows mean and median side by side under a grouped header.
    metrics = ["NMAE", "Bias", "SSIM", "PWT"]
    name_w = 28
    fw = 10               # width of each value field
    sep = "  "            # gap between metric groups
    group_w = fw * 2      # a metric group spans its mean + median fields

    group_header = f"{'':<{name_w}}" + sep.join(m.center(group_w) for m in metrics)
    sub_header = f"{'Name':<{name_w}}" + sep.join(f"{'mean':>{fw}}{'median':>{fw}}" for _ in metrics)
    rule = "-" * len(sub_header)

    print("\n=== All-models evaluation table (test set, DN space) ===")
    print(group_header)
    print(sub_header)
    print(rule)
    for name, stats in rows:
        if stats is None:
            cells = sep.join(f"{'nan':>{fw}}{'nan':>{fw}}" for _ in metrics)
        else:
            cells = sep.join(f"{stats[m][0]:>{fw}.4f}{stats[m][1]:>{fw}.4f}" for m in metrics)
        print(f"{name:<{name_w}}{cells}")
    print()

    # Per-image standard deviations for the two headline models, one value per metric.
    std_specs = [("UNet conv b22 (HiPredict)", "HiPredict"), ("SYN4", "SYN4")]
    std_rows = {name: stats for name, stats in rows}

    std_name_w = name_w
    std_header = f"{'Model':<{std_name_w}}" + sep.join(f"{m:>{fw}}" for m in metrics)
    std_rule = "-" * len(std_header)

    print("=== Per-image standard deviations (test set, DN space) ===")
    print(std_header)
    print(std_rule)
    for disp_name, label in std_specs:
        stats = std_rows.get(disp_name)
        if stats is None:
            cells = sep.join(f"{'nan':>{fw}}" for _ in metrics)
        else:
            cells = sep.join(f"{stats[m][2]:>{fw}.4f}" for m in metrics)
        print(f"{label:<{std_name_w}}{cells}")
    print()


def main():
    global DATA_RECORD, EVAL_DR, EVALS_CONFIG, SAVE_DIR

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models first so the active config defines all data/output locations.
    cnn_model, cfg = get_model(MODEL_NAME, device)
    syn4, _ = get_model('SYN4', device, cfg=cfg)

    data_root = cfg['data'].get('root', None)
    DATA_RECORD = os.path.join(data_root, cfg['data']['data_record']) if data_root else DATA_RECORD
    EVAL_DR = os.path.join(data_root, cfg['evals']['eval_dr']) if data_root else EVAL_DR
    EVALS_CONFIG = os.path.join(data_root, cfg['evals']['eval_cfg']) if data_root else EVALS_CONFIG
    SAVE_DIR = cfg.get('save', {}).get('evals_dir', SAVE_DIR)
    os.makedirs(SAVE_DIR, exist_ok=True)

    dr = pd.read_csv(DATA_RECORD)
    dr.attrs["record_dir"] = os.path.dirname(os.path.abspath(DATA_RECORD))
    data_root = cfg["data"].get("patch_root")
    dr_eval = pd.read_csv(EVAL_DR)
    dr_eval.attrs["record_dir"] = os.path.dirname(os.path.abspath(EVAL_DR))
    with open(EVALS_CONFIG, 'r') as file:
        eval_cats = yaml.safe_load(file)

    # Remove categories with empty observation lists
    eval_cats = {k: v for k, v in eval_cats.items() if v and v.get('oids')}


    linear_model = None

    # SYN4 denormalizes with the global (center, scale); this is only valid when the
    # dataset is also normalized in global_robust mode (constant per-sample affine).
    assert cfg['data'].get('normalisation_mode') == 'global_robust', (
        "SYN4 baseline requires global_robust normalization; eval cfg uses "
        f"{cfg['data'].get('normalisation_mode')!r}.")


    # Load the fitted scaler for normalising metadata features
    try:
        shared_meta_scaler = joblib.load(cfg['shared_meta_scalar'])
    except FileNotFoundError:
        print("WARNING: meta scalar file not found")
        shared_meta_scaler = None

    criterion = get_criterion(cfg)
    meta_cols = cfg['data'].get('meta_cols', None)

    linear_avail = False
    linear_model = None
    try:
        linear_model, _ = get_model("linear_with_meta", device)
        expected_n_meta = getattr(linear_model, "n_meta", None)
        actual_n_meta = len(meta_cols or [])
        if expected_n_meta is not None and expected_n_meta != actual_n_meta:
            print("\n*** WARNING: linear_model metadata width mismatch")
            print(f"*** linear_model expects n_meta={expected_n_meta}, eval config provides n_meta={actual_n_meta}")
            print("*** Continuing without linear model. Related metrics will be 'nan'.\n")
            linear_model = None
        else:
            linear_model = linear_model.eval()
            linear_avail = True
    except Exception as e:
        print("\n*** WARNING: Failed to load linear_model")
        print(f"*** {type(e).__name__}: {e}")
        print("*** Continuing without linear model. Related metrics will be 'nan'.\n")

    cnn_model = cnn_model.eval()
    syn4 = syn4.eval()

    # Custom split by date
    train_sets, test_sets, val_sets, train_val_date, val_test_date = split_by_date(
        dr, train_size=cfg['data']['train_size'], test_size=cfg['data']['test_size'], val_size=cfg['data']['val_size'], using_wandb=False
    )

    # Obtain meta cols
    meta_cols = cfg['data'].get('meta_cols', None)
    if meta_cols is not None:
        missing_cols = set(meta_cols) - set(dr.columns)
        assert not missing_cols, f"Missing meta columns in data record: {missing_cols}"

    normalization_kwargs = normalization_kwargs_from_config(cfg)
    train_dataset = FilteredHiRISEDataset(
        allowed_sets=train_sets, data_record=dr, sweep=False, transform=None,
        meta_cols=meta_cols, **normalization_kwargs, data_root=data_root
    )

    val_dataset = FilteredHiRISEDataset(
        allowed_sets=val_sets, data_record=dr, sweep=False, transform=None,
        meta_cols=meta_cols, meta_scaler=shared_meta_scaler,
        **normalization_kwargs, data_root=data_root
    )
    test_dataset = FilteredHiRISEDataset(
        allowed_sets=test_sets, data_record=dr, sweep=False, transform=None,
        meta_cols=meta_cols, meta_scaler=shared_meta_scaler,
        **normalization_kwargs, data_root=data_root
    )

    train_dataloader = get_loader(
        train_dataset,
        cfg['optim']['batch_size'],
        num_workers=1,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn(cfg['seed']),
        seed=cfg['seed'],
    )
    val_dataloader = get_loader(
        val_dataset,
        cfg['optim']['batch_size'],
        num_workers=1,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn(cfg['seed']),
        seed=cfg['seed'],
    )
    test_dataloader = get_loader(
        test_dataset,
        cfg['optim']['batch_size'],
        num_workers=1,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn(cfg['seed']),
        seed=cfg['seed'],
    )

    def create_buffers():
        train_losses, val_losses = [], []
        train_base, val_base = [], []
        val_pwt_cnn_list, val_pwt_base_list = [], []
        val_nmae_cnn_list, val_nmae_base_list = [], []
        val_ssim_cnn_list, val_ssim_base_list = [], []
        lf_mae_cnn_all = []
        hf_ceil_mae_all = []
        hf_model_mae_all = []
        dcfix_mae_all = []
        res_e_tot_all = []
        res_e_dc_all = []
        res_e_lf_all = {k: [] for k in RESIDUAL_BLUR_KERNELS}
        res_e_hf_all = {k: [] for k in RESIDUAL_BLUR_KERNELS}
        dc_err_direct_all = []
        out_centered_dc_all = []
        lf_bias_dc_all = []
        r3_only_mae_all = []
        r5_only_mae_all = []
        mu_true_all_rows = []
        mixopt_top_mae_all = []
        alpha_opt_rows = []
        true_mean_perimg = []
        pred_mean_perimg = []
        dn_bias_cnn_all = []
        sampled_true_pixels = []
        sampled_pred_pixels = []
        sampled_base_pixels = []
        dn_bias_base_all = []
        dn_true_mag_all = []
        dn_mean_err_all = []
        out_cent_dn_all = []
        global_bias_dn_all = []
        m_hat_dn_all = []
        lf_bias_dn_all = []
        pred_mean_dn_all = []
        true_mean_dn_all = []
        bias_cnn_all = []
        bias_base_all = []
        dn_nmae_cnn_all = []
        dn_nmae_base_all = []
        dn_pwt_cnn_all = []
        dn_pwt_base_all = []
        dn_ssim_cnn_all = []
        dn_ssim_base_all = []
        a_dn_all = []
        s_all = []
        ecdf_abs_err_dn_cnn_all = []
        ecdf_abs_err_dn_base_all = []
        dn_true_dr_all = []
        dn_mae_cnn_all = []
        dn_mae_base_all = []
        dn_bias_raw_cnn_all = []
        dn_bias_raw_base_all = []
        global_val_rows = []
        return locals()

    buffers = create_buffers()

    # Run train/val eval pass - fill buffers
    buffers = run_train_val_eval_pass(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        cnn_model=cnn_model,
        syn4=syn4,
        criterion=criterion,
        compute_per_img_metrics=compute_per_img_metrics,
        summarize_metric=summarize_metric,
        _blur7=_blur7,
        PWT_THRESH=PWT_THRESH,
        PIX_SAMPLES_PER_IMAGE=PIX_SAMPLES_PER_IMAGE,
        buffers=buffers,
    )

    # Aggregate per-image buffer lists into global summary statistics
    summary = build_eval_summary(buffers)
    # Print CNN vs SYN4 global metric comparison to stdout
    print_global_eval_report(summary)

    # Collect per-observation val-set metrics into a DataFrame for plotting
    global_val_df = pd.DataFrame(buffers["global_val_rows"])
    # Print the observations where SYN4 performs worst
    print_worst_syn4_subset_report(global_val_df, top_n=WORST_SYN4_TOP_N)

    # Print a single bias/ssim/pwt table comparing all models on the test set
    print_all_models_eval_table(test_dataloader, device, compute_per_img_metrics, summarize_metric, PWT_THRESH, cfg=cfg)

    # Per-category evals
    category_out = run_category_evals(
        eval_cats=eval_cats,
        dr=dr_eval,
        FilteredHiRISEDataset=FilteredHiRISEDataset,
        meta_cols=meta_cols,
        shared_meta_scaler=shared_meta_scaler,
        get_loader=get_loader,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        cfg=cfg,
        device=device,
        cnn_model=cnn_model,
        syn4=syn4,
        linear_avail=linear_avail,
        linear_model=linear_model,
        criterion=criterion,
        compute_per_img_metrics=compute_per_img_metrics,
        summarize_metric=summarize_metric,
        PWT_THRESH=PWT_THRESH,
        _to_hw=_to_hw,
    )

    # Save category CSV + print category tables
    save_category_results_and_print(
        category_out=category_out,
        SAVE_DIR=SAVE_DIR,
        cnn_name=MODEL_NAME,
        PWT_PERC=PWT_PERC,
    )
    # Sanity-check that per-category metrics are consistent with global val-set metrics
    compare_global_vs_category_stats(
        buffers=buffers,
        category_out=category_out,
    )
    results_df = pd.DataFrame(category_out["rows"])

    # Compute correlations between per-image errors and metadata features
    corr_vals = corr_evals(results_df, val_dataset)
    gc.collect()

    evals = category_out["evals"]
    pred_mean_perimg = buffers["pred_mean_perimg"]
    true_mean_perimg = buffers["true_mean_perimg"]
    sampled_pred_pixels = buffers["sampled_pred_pixels"]
    sampled_true_pixels = buffers["sampled_true_pixels"]
    sampled_base_pixels = buffers["sampled_base_pixels"]

    # Generate and save all standard evaluation plots (CDFs, hexbins, dumbbell, SHAP)
    plot_evals(
        summary["ecdf_arr_cnn"],
        summary["ecdf_arr_base"],
        evals,
        eval_cats,
        sampled_base_pixels,
        sampled_pred_pixels,
        sampled_true_pixels,
        pred_mean_perimg,
        true_mean_perimg,
        MODEL_NAME,
        SAVE_DIR,
        corr_vals=corr_vals,
    )

    # Plot per-observation SYN4 vs CNN DN-space NMAE as a hexbin scatter
    plot_syn4_vs_hipredict_hexbin(
        global_val_df=global_val_df,
        cnn_name=MODEL_NAME,
        SAVE_DIR=SAVE_DIR,
    )
    # Plot CNN improvement over SYN4, binned by SYN4 NMAE
    plot_syn4_binned_improvement(
        global_val_df=global_val_df,
        cnn_name=MODEL_NAME,
        SAVE_DIR=SAVE_DIR,
    )
    # Same comparison restated: HiPredict DN NMAE vs SYN4 difficulty with a y=x parity line
    plot_syn4_vs_hipredict_binned(
        global_val_df=global_val_df,
        cnn_name=MODEL_NAME,
        SAVE_DIR=SAVE_DIR,
    )
    # Same comparison as a log-log 2D-histogram density heatmap with a y=x parity line
    plot_syn4_vs_hipredict_hist2d(
        global_val_df=global_val_df,
        cnn_name=MODEL_NAME,
        SAVE_DIR=SAVE_DIR,
    )
    # Per-bin DN NMAE histograms for the six fields SHAP ranks as most influential
    plot_nmae_hist_by_metadata(
        global_val_df=global_val_df,
        corr_vals=corr_vals,
        cnn_name=MODEL_NAME,
        SAVE_DIR=SAVE_DIR,
    )

    # Plot temporal and spatial overlap between training and evaluation observation sets
    compare_train_eval_sets(SAVE_DIR, TRAIN_DR=DATA_RECORD, EVAL_DR=EVAL_DR)


if __name__ == "__main__":
    main()
