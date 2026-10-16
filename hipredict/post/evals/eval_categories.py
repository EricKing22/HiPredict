# eval_categories.py
import numpy as np
import torch

from hipredict.data.normalization import normalization_kwargs_from_config


def run_category_evals(
    *,
    eval_cats,
    dr,
    FilteredHiRISEDataset,
    meta_cols,
    shared_meta_scaler,
    get_loader,
    collate_fn,
    worker_init_fn,
    cfg,
    device,
    cnn_model,
    syn4,
    linear_avail,
    linear_model,
    criterion,
    compute_per_img_metrics,
    summarize_metric,
    PWT_THRESH,
    _to_hw,
):
    evals = {}
    rows = []
    cnn_avg_pix = []
    syn4_avg_pix = []
    true_avg_pix = []
    ir_avg_pix = []
    bg_avg_pix = []
    r5_avg_pix = []
    r3_avg_pix = []
    normalization_kwargs = normalization_kwargs_from_config(cfg)

    for cat, cat_data in eval_cats.items():
        obs_list = cat_data['oids']
        allowed_sets = dr[dr['Observation'].isin(obs_list)].drop_duplicates('Observation')['Set'].tolist()      # Only take one set for each observation
        if not allowed_sets:
            print(f"Skipping category {cat}: no observations in the supplied record.")
            continue
        try:
            dataset = FilteredHiRISEDataset(
                data_record=dr,
                allowed_sets=allowed_sets,
                sweep=True,
                meta_cols=meta_cols,
                meta_scaler=shared_meta_scaler,
                **normalization_kwargs,
                data_root=cfg['data'].get('patch_root')
            )
            dataloader = get_loader(
                dataset,
                batch_size=64,
                collate_fn=collate_fn,
                worker_init_fn=worker_init_fn(cfg['seed']),
                seed=cfg['seed']
            )

            # Stream metrics per batch to avoid holding all predictions in RAM
            nmae_cnn_all_cat = []
            nmae_syn4_all_cat = []
            nmae_linear_all_cat = []

            pwt_cnn_all_cat = []
            pwt_syn4_all_cat = []
            pwt_linear_all_cat = []

            bias_cnn_all_cat = []
            bias_syn4_all_cat = []
            bias_linear_all_cat = []

            ssim_cnn_all_cat = []
            ssim_syn4_all_cat = []
            ssim_linear_all_cat = []

            cnn_loss_all = []
            base_loss_all = []
            linear_loss_all = []

            mae_cnn_dn_all_cat = []
            mae_syn4_dn_all_cat = []
            mae_linear_dn_all_cat = []

            bias_cnn_dn_raw_all_cat = []
            bias_syn4_dn_raw_all_cat = []
            bias_linear_dn_raw_all_cat = []

            true_mag_dn_all_cat = []
            true_dr_dn_all_cat = []

            obs_ids_all = []
            set_names_all = []
            dates_all = []
            cnn_sum = 0.0
            syn4_sum = 0.0
            true_sum = 0.0
            n_pix_total = 0

            with torch.no_grad():
                for batch in dataloader:
                    x_band = batch["x_band"].to(device).float()
                    x_neigh = batch["x_neigh"].to(device).float()
                    x_meta = batch["x_meta"].to(device).float()
                    y_true = batch['y'].to(device)
                    patch_x = batch["patch_x"].to(device).float()
                    obs_width = batch["obs_width"].to(device).float()

                    y_pred = cnn_model(x_band, x_neigh, x_meta, patch_x=patch_x, obs_width=obs_width)
                    y_base_pred = syn4(x_band, x_neigh, x_meta)

                    stats = batch["stats"].to(device).float()
                    a = stats[:, 0].view(-1, 1, 1, 1)
                    s = stats[:, 1].view(-1, 1, 1, 1)

                    y_true_dn = y_true.float() * s + a
                    y_pred_dn = y_pred.float() * s + a
                    y_base_dn = y_base_pred.float() * s + a
                    B = y_true.size(0)

                    # DN-space channel means (assumes x_band/x_neigh use the same per-sample a,s normalisation)
                    x_band_dn = x_band.float() * s + a
                    x_neigh_dn = x_neigh.float() * s + a

                    for i in range(B):
                        ir_avg_pix.append(float(x_band_dn[i, 0].mean().detach().cpu().item()))
                        bg_avg_pix.append(float(x_band_dn[i, 1].mean().detach().cpu().item()))
                        r5_avg_pix.append(float(x_neigh_dn[i, 2].mean().detach().cpu().item()))
                        r3_avg_pix.append(float(x_neigh_dn[i, 5].mean().detach().cpu().item()))

                    cnn_sum += y_pred_dn.detach().sum().cpu().item()
                    syn4_sum += y_base_dn.detach().sum().cpu().item()
                    true_sum += y_true_dn.detach().sum().cpu().item()
                    n_pix_total += y_true_dn.numel()

                    cnn_loss_i = [criterion(y_pred[i: i + 1], y_true[i: i + 1])[0].detach().cpu().item() for i in range(B)]
                    base_loss_i = [criterion(y_base_pred[i: i + 1], y_true[i: i + 1])[0].detach().cpu().item() for i in range(B)]

                    if linear_avail:
                        y_linear = linear_model(x_band, x_neigh, x_meta, patch_x=patch_x, obs_width=obs_width)
                        y_linear_dn = y_linear.float() * s + a
                        assert y_pred.ndim == 4 and y_base_pred.ndim == 4 and y_linear.ndim == 4
                        linear_loss_i = [criterion(y_linear[i: i + 1], y_true[i: i + 1])[0].detach().cpu().item() for i in range(B)]
                    else:
                        y_linear = None
                        assert y_pred.ndim == 4 and y_base_pred.ndim == 4
                        linear_loss_i = [np.nan] * B

                    # Per-image DN diagnostics for category-vs-global comparison
                    mae_cnn_dn_b = (y_pred_dn - y_true_dn).abs().mean(dim=(1, 2, 3)).detach().cpu().numpy()
                    mae_syn4_dn_b = (y_base_dn - y_true_dn).abs().mean(dim=(1, 2, 3)).detach().cpu().numpy()

                    bias_cnn_dn_raw_b = (y_pred_dn - y_true_dn).mean(dim=(1, 2, 3)).detach().cpu().numpy()
                    bias_syn4_dn_raw_b = (y_base_dn - y_true_dn).mean(dim=(1, 2, 3)).detach().cpu().numpy()

                    y_true_dn_flat = y_true_dn.view(y_true_dn.shape[0], -1)

                    true_mag_dn_b = y_true_dn.abs().mean(dim=(1, 2, 3)).detach().cpu().numpy()
                    true_dr_dn_b = (
                        y_true_dn_flat.max(dim=1).values - y_true_dn_flat.min(dim=1).values
                    ).detach().cpu().numpy()

                    if linear_avail:
                        mae_linear_dn_b = (y_linear_dn - y_true_dn).abs().mean(dim=(1, 2, 3)).detach().cpu().numpy()
                        bias_linear_dn_raw_b = (y_linear_dn - y_true_dn).mean(dim=(1, 2, 3)).detach().cpu().numpy()
                    else:
                        mae_linear_dn_b = np.full(B, np.nan, dtype=np.float64)
                        bias_linear_dn_raw_b = np.full(B, np.nan, dtype=np.float64)

                    mae_cnn_dn_all_cat.extend(mae_cnn_dn_b.tolist())
                    mae_syn4_dn_all_cat.extend(mae_syn4_dn_b.tolist())
                    mae_linear_dn_all_cat.extend(mae_linear_dn_b.tolist())

                    bias_cnn_dn_raw_all_cat.extend(bias_cnn_dn_raw_b.tolist())
                    bias_syn4_dn_raw_all_cat.extend(bias_syn4_dn_raw_b.tolist())
                    bias_linear_dn_raw_all_cat.extend(bias_linear_dn_raw_b.tolist())

                    true_mag_dn_all_cat.extend(true_mag_dn_b.tolist())
                    true_dr_dn_all_cat.extend(true_dr_dn_b.tolist())

                    cnn_loss_all.extend(cnn_loss_i)
                    base_loss_all.extend(base_loss_i)
                    linear_loss_all.extend(linear_loss_i)

                    obs_ids_all.extend(list(batch['obs_id']))
                    set_names_all.extend(list(batch['set_name']))
                    dates_all.extend([str(d) for d in batch['date']])

                    m_cnn_b = compute_per_img_metrics(y_true_dn, y_pred_dn, pwt_threshold=PWT_THRESH, use_magnitude=True)
                    m_syn4_b = compute_per_img_metrics(y_true_dn, y_base_dn, pwt_threshold=PWT_THRESH, use_magnitude=True)

                    if linear_avail:
                        m_linear_b = compute_per_img_metrics(y_true_dn, y_linear_dn, pwt_threshold=PWT_THRESH, use_magnitude=True)
                    else:
                        nan_array_b = np.full(B, np.nan)
                        m_linear_b = {"pwt": nan_array_b, "nmae": nan_array_b, "bias": nan_array_b, "ssim": nan_array_b}

                    nmae_cnn_all_cat.extend(m_cnn_b["nmae"].tolist())
                    nmae_syn4_all_cat.extend(m_syn4_b["nmae"].tolist())
                    nmae_linear_all_cat.extend(m_linear_b["nmae"].tolist())

                    pwt_cnn_all_cat.extend(m_cnn_b["pwt"].tolist())
                    pwt_syn4_all_cat.extend(m_syn4_b["pwt"].tolist())
                    pwt_linear_all_cat.extend(m_linear_b["pwt"].tolist())

                    bias_cnn_all_cat.extend(m_cnn_b["bias"].tolist())
                    bias_syn4_all_cat.extend(m_syn4_b["bias"].tolist())
                    bias_linear_all_cat.extend(m_linear_b["bias"].tolist())

                    ssim_cnn_all_cat.extend(m_cnn_b["ssim"].tolist())
                    ssim_syn4_all_cat.extend(m_syn4_b["ssim"].tolist())
                    ssim_linear_all_cat.extend(m_linear_b["ssim"].tolist())

            cnn_avg_pix.append(cnn_sum / max(n_pix_total, 1))
            syn4_avg_pix.append(syn4_sum / max(n_pix_total, 1))
            true_avg_pix.append(true_sum / max(n_pix_total, 1))

            cnn_loss_i = cnn_loss_all
            base_loss_i = base_loss_all
            linear_loss_i = linear_loss_all

            B = len(obs_ids_all)

            assert len(set_names_all) == B
            assert len(dates_all) == B
            assert len(cnn_loss_i) == B
            assert len(base_loss_i) == B
            assert len(linear_loss_i) == B

            m_cnn = {
                "nmae": np.asarray(nmae_cnn_all_cat, dtype=np.float64),
                "pwt": np.asarray(pwt_cnn_all_cat, dtype=np.float64),
                "bias": np.asarray(bias_cnn_all_cat, dtype=np.float64),
                "ssim": np.asarray(ssim_cnn_all_cat, dtype=np.float64),
            }
            m_syn4 = {
                "nmae": np.asarray(nmae_syn4_all_cat, dtype=np.float64),
                "pwt": np.asarray(pwt_syn4_all_cat, dtype=np.float64),
                "bias": np.asarray(bias_syn4_all_cat, dtype=np.float64),
                "ssim": np.asarray(ssim_syn4_all_cat, dtype=np.float64),
            }
            m_linear = {
                "nmae": np.asarray(nmae_linear_all_cat, dtype=np.float64),
                "pwt": np.asarray(pwt_linear_all_cat, dtype=np.float64),
                "bias": np.asarray(bias_linear_all_cat, dtype=np.float64),
                "ssim": np.asarray(ssim_linear_all_cat, dtype=np.float64),
            }

            wins = np.sum(m_cnn["nmae"] < m_syn4["nmae"])
            n_imgs = len(m_cnn["nmae"])
            if linear_avail:
                wins_l = np.sum(m_cnn["nmae"] < m_linear["nmae"])
                win_rate_l = 100 * wins_l / max(n_imgs, 1)
            else:
                win_rate_l = np.nan
            win_rate = 100 * wins / max(n_imgs, 1)

            pwt_c = summarize_metric(m_cnn["pwt"])
            pwt_b = summarize_metric(m_syn4["pwt"])
            pwt_l = summarize_metric(m_linear["pwt"]) if linear_avail else np.nan

            nmae_c = summarize_metric(m_cnn["nmae"])
            nmae_b = summarize_metric(m_syn4["nmae"])
            nmae_l = summarize_metric(m_linear["nmae"]) if linear_avail else np.nan

            bias_c = summarize_metric(m_cnn["bias"])
            bias_b = summarize_metric(m_syn4["bias"])
            bias_l = summarize_metric(m_linear["bias"]) if linear_avail else np.nan

            ssim_c = summarize_metric(m_cnn["ssim"])
            ssim_b = summarize_metric(m_syn4["ssim"])
            ssim_l = summarize_metric(m_linear["ssim"]) if linear_avail else np.nan

            pwt_delta_pp = pwt_c - pwt_b
            nmae_rel_red = (nmae_b - nmae_c) / max(nmae_b, 1e-8)
            ssim_delta = ssim_c - ssim_b
            evals[cat] = dict(
                PWT_CNN=pwt_c, PWT_BASE=pwt_b, PWT_DELTA=pwt_delta_pp, PWT_LINEAR=pwt_l,
                NMAE_CNN=nmae_c, NMAE_BASE=nmae_b, NMAE_REL_RED=nmae_rel_red, NMAE_LINEAR=nmae_l,
                BIAS_CNN=bias_c, BIAS_BASE=bias_b, BIAS_LINEAR=bias_l,
                SSIM_CNN=ssim_c, SSIM_BASE=ssim_b, SSIM_DELTA=ssim_delta, SSIM_LINEAR=ssim_l,
                WIN_RATE=win_rate, N=n_imgs, WIN_RATE_LINEAR=win_rate_l,
            )

            obs_ids = obs_ids_all
            set_names = set_names_all
            dates = dates_all

            for i in range(B):
                rows.append({
                    'Category': cat,
                    'Observation': obs_ids[i],
                    'Set name': set_names[i],
                    'Date': dates[i],
                    'CNN Loss': cnn_loss_i[i],
                    'SYN4 Loss': base_loss_i[i],
                    'Linear Loss': linear_loss_i[i],
                    'CNN PWT': float(m_cnn['pwt'][i]),
                    'SYN4 PWT': float(m_syn4['pwt'][i]),
                    'Linear PWT': float(m_linear['pwt'][i]),
                    'CNN NMAE': float(m_cnn['nmae'][i]),
                    'SYN4 NMAE': float(m_syn4['nmae'][i]),
                    'Linear NMAE': float(m_linear['nmae'][i]),
                    'CNN Bias': float(m_cnn['bias'][i]),
                    'SYN4 Bias': float(m_syn4['bias'][i]),
                    'Linear Bias': float(m_linear['bias'][i]),
                    'CNN SSIM': float(m_cnn['ssim'][i]),
                    'SYN4 SSIM': float(m_syn4['ssim'][i]),
                    'Linear SSIM': float(m_linear['ssim'][i]),
                    'CNN MAE_DN': float(mae_cnn_dn_all_cat[i]),
                    'SYN4 MAE_DN': float(mae_syn4_dn_all_cat[i]),
                    'Linear MAE_DN': float(mae_linear_dn_all_cat[i]),
                    'CNN Bias_DN_raw': float(bias_cnn_dn_raw_all_cat[i]),
                    'SYN4 Bias_DN_raw': float(bias_syn4_dn_raw_all_cat[i]),
                    'Linear Bias_DN_raw': float(bias_linear_dn_raw_all_cat[i]),
                    'TrueMag_DN': float(true_mag_dn_all_cat[i]),
                    'TrueDR_DN': float(true_dr_dn_all_cat[i]),
                })

        except ValueError as e:
            import traceback
            print(f"WARNING ValueError in cat={cat} | n_obs={len(obs_list)} | n_sets={len(allowed_sets)} | {e}")
            print(traceback.format_exc())
            continue

    print(f"run_category_evals - Num categories: {len(evals)} | Total images: {len(rows)}")

    return {
        "evals": evals,
        "rows": rows,
        "cnn_avg_pix": cnn_avg_pix,
        "syn4_avg_pix": syn4_avg_pix,
        "true_avg_pix": true_avg_pix,
        "ir_avg_pix": ir_avg_pix,
        "bg_avg_pix": bg_avg_pix,
        "r5_avg_pix": r5_avg_pix,
        "r3_avg_pix": r3_avg_pix,
    }
