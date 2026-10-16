import torch
import torch.nn.functional as F
import numpy as np


# Box-blur kernel sizes (px) used to split the residual into LF/HF bands.
# The split point is otherwise arbitrary, so we sweep it as a sensitivity check.
RESIDUAL_BLUR_KERNELS = (3, 7, 15)
# Kernel used for the headline DC/LF/HF numbers (must be in RESIDUAL_BLUR_KERNELS).
PRIMARY_BLUR_KERNEL = 7


def _avg_blur(x, k):
    """Reflect-padded k x k box blur (stride 1). Matches _blur7 when k == 7."""
    pad = k // 2
    x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    return F.avg_pool2d(x, kernel_size=k, stride=1)


def run_train_val_eval_pass(
    *,
    train_dataloader,
    val_dataloader,
    device,
    cnn_model,
    syn4,
    criterion,
    compute_per_img_metrics,
    summarize_metric,
    _blur7,
    PWT_THRESH,
    PIX_SAMPLES_PER_IMAGE,
    buffers,
):

    # Explicit aliases
    train_losses = buffers["train_losses"]
    val_losses = buffers["val_losses"]
    train_base = buffers["train_base"]
    val_base = buffers["val_base"]
    val_pwt_cnn_list = buffers["val_pwt_cnn_list"]
    val_pwt_base_list = buffers["val_pwt_base_list"]
    val_nmae_cnn_list = buffers["val_nmae_cnn_list"]
    val_nmae_base_list = buffers["val_nmae_base_list"]
    val_ssim_cnn_list = buffers["val_ssim_cnn_list"]
    val_ssim_base_list = buffers["val_ssim_base_list"]
    lf_mae_cnn_all = buffers["lf_mae_cnn_all"]
    hf_ceil_mae_all = buffers["hf_ceil_mae_all"]
    hf_model_mae_all = buffers["hf_model_mae_all"]
    dcfix_mae_all = buffers["dcfix_mae_all"]
    res_e_tot_all = buffers["res_e_tot_all"]
    res_e_dc_all = buffers["res_e_dc_all"]
    res_e_lf_all = buffers["res_e_lf_all"]
    res_e_hf_all = buffers["res_e_hf_all"]
    dc_err_direct_all = buffers["dc_err_direct_all"]
    out_centered_dc_all = buffers["out_centered_dc_all"]
    lf_bias_dc_all = buffers["lf_bias_dc_all"]
    r3_only_mae_all = buffers["r3_only_mae_all"]
    r5_only_mae_all = buffers["r5_only_mae_all"]
    mu_true_all_rows = buffers["mu_true_all_rows"]
    mixopt_top_mae_all = buffers["mixopt_top_mae_all"]
    alpha_opt_rows = buffers["alpha_opt_rows"]
    true_mean_perimg = buffers["true_mean_perimg"]
    pred_mean_perimg = buffers["pred_mean_perimg"]
    dn_bias_cnn_all = buffers["dn_bias_cnn_all"]
    sampled_true_pixels = buffers["sampled_true_pixels"]
    sampled_pred_pixels = buffers["sampled_pred_pixels"]
    sampled_base_pixels = buffers["sampled_base_pixels"]
    dn_bias_base_all = buffers["dn_bias_base_all"]
    dn_true_mag_all = buffers["dn_true_mag_all"]
    dn_mean_err_all = buffers["dn_mean_err_all"]
    out_cent_dn_all = buffers["out_cent_dn_all"]
    global_bias_dn_all = buffers["global_bias_dn_all"]
    m_hat_dn_all = buffers["m_hat_dn_all"]
    lf_bias_dn_all = buffers["lf_bias_dn_all"]
    pred_mean_dn_all = buffers["pred_mean_dn_all"]
    true_mean_dn_all = buffers["true_mean_dn_all"]
    bias_cnn_all = buffers["bias_cnn_all"]
    bias_base_all = buffers["bias_base_all"]
    dn_nmae_cnn_all = buffers["dn_nmae_cnn_all"]
    dn_nmae_base_all = buffers["dn_nmae_base_all"]
    dn_pwt_cnn_all = buffers["dn_pwt_cnn_all"]
    dn_pwt_base_all = buffers["dn_pwt_base_all"]
    dn_ssim_cnn_all = buffers["dn_ssim_cnn_all"]
    dn_ssim_base_all = buffers["dn_ssim_base_all"]
    a_dn_all = buffers["a_dn_all"]
    s_all = buffers["s_all"]
    ecdf_abs_err_dn_cnn_all = buffers["ecdf_abs_err_dn_cnn_all"]
    ecdf_abs_err_dn_base_all = buffers["ecdf_abs_err_dn_base_all"]
    dn_true_dr_all = buffers["dn_true_dr_all"]
    dn_mae_cnn_all = buffers["dn_mae_cnn_all"]
    dn_mae_base_all = buffers["dn_mae_base_all"]
    dn_bias_raw_cnn_all = buffers["dn_bias_raw_cnn_all"]
    dn_bias_raw_base_all = buffers["dn_bias_raw_base_all"]
    global_val_rows = buffers["global_val_rows"]
    rng = np.random.default_rng(42)

    with torch.no_grad():
        for batch in train_dataloader:
            x_band = batch["x_band"].to(device).float()
            x_neigh = batch["x_neigh"].to(device).float()
            x_meta = batch["x_meta"].to(device).float()
            y = batch['y'].to(device).float()
            patch_x = batch["patch_x"].to(device).float()
            obs_width = batch["obs_width"].to(device).float()

            y_pred = cnn_model(x_band, x_neigh, x_meta, patch_x=patch_x, obs_width=obs_width)
            y_base_pred = syn4(x_band, x_neigh, x_meta)

            train_losses.append(criterion(y_pred, y)[0].detach().cpu().mean())
            train_base.append(criterion(y_base_pred, y)[0].detach().cpu().mean())

        for batch in val_dataloader:
            x_band = batch["x_band"].to(device).float()
            x_neigh = batch["x_neigh"].to(device).float()
            x_meta = batch["x_meta"].to(device).float()
            y_true = batch['y'].to(device).float()
            patch_x = batch["patch_x"].to(device).float()
            obs_width = batch["obs_width"].to(device).float()

            y_pred = cnn_model(x_band, x_neigh, x_meta, patch_x=patch_x, obs_width=obs_width)
            y_base_pred = syn4(x_band, x_neigh, x_meta)

            stats = batch["stats"].to(device).float()
            a = stats[:, 0].view(-1, 1, 1, 1)
            s = stats[:, 1].view(-1, 1, 1, 1)

            y_true_dn = y_true * s + a
            y_pred_dn = y_pred * s + a
            y_base_dn = y_base_pred * s + a

            # Per-image DN diagnostics (for comparing global val vs category subsets)
            mae_cnn_dn = (y_pred_dn - y_true_dn).abs().mean(dim=(1, 2, 3)).detach().cpu().numpy()
            mae_base_dn = (y_base_dn - y_true_dn).abs().mean(dim=(1, 2, 3)).detach().cpu().numpy()

            bias_raw_cnn_dn = (y_pred_dn - y_true_dn).mean(dim=(1, 2, 3)).detach().cpu().numpy()
            bias_raw_base_dn = (y_base_dn - y_true_dn).mean(dim=(1, 2, 3)).detach().cpu().numpy()


            y_true_dn_flat = y_true_dn.view(y_true_dn.shape[0], -1)
            true_dr_dn = (
                y_true_dn_flat.max(dim=1).values - y_true_dn_flat.min(dim=1).values
            ).detach().cpu().numpy()

            dn_mae_cnn_all.extend(mae_cnn_dn.tolist())
            dn_mae_base_all.extend(mae_base_dn.tolist())
            dn_bias_raw_cnn_all.extend(bias_raw_cnn_dn.tolist())
            dn_bias_raw_base_all.extend(bias_raw_base_dn.tolist())
            dn_true_dr_all.extend(true_dr_dn.tolist())

            m = cnn_model.module if hasattr(cnn_model, "module") else cnn_model

            # Pixelwise ECDF source (DN space, mag normalised by per image mean |true|)
            y_true_dn_np = y_true_dn.detach().cpu().numpy()
            y_pred_dn_np = y_pred_dn.detach().cpu().numpy()
            y_base_dn_np = y_base_dn.detach().cpu().numpy()

            if y_true_dn_np.ndim == 4 and y_true_dn_np.shape[1] == 1:
                y_true_dn_np = y_true_dn_np[:, 0]
                y_pred_dn_np = y_pred_dn_np[:, 0]
                y_base_dn_np = y_base_dn_np[:, 0]

            for j in range(y_true_dn_np.shape[0]):
                yt = y_true_dn_np[j]
                yc = y_pred_dn_np[j]
                yb = y_base_dn_np[j]

                denom = float(max(np.mean(np.abs(yt)), 1e-8))

                cnn_err = np.abs(yc - yt).ravel() / denom
                base_err = np.abs(yb - yt).ravel() / denom

                ecdf_abs_err_dn_cnn_all.append(cnn_err.astype(np.float32, copy=False))
                ecdf_abs_err_dn_base_all.append(base_err.astype(np.float32, copy=False))

            outc_mean_dn = (m.out_centered.mean(dim=(2, 3)).view(-1) * s.view(-1)).detach().cpu().numpy()       # type:ignore
            mhat_mean_dn = (m.m_hat.mean(dim=(2, 3)).view(-1) * s.view(-1)).detach().cpu().numpy()              # type:ignore
            lfb_mean_dn = (m.lf_bias.mean(dim=(2, 3)).view(-1) * s.view(-1)).detach().cpu().numpy()             # type:ignore
            gb_mean_dn = (m.global_bias.view(1) * s.view(-1)).detach().cpu().numpy()                            # type:ignore

            out_cent_dn_all.extend(outc_mean_dn.tolist())
            m_hat_dn_all.extend(mhat_mean_dn.tolist())
            lf_bias_dn_all.extend(lfb_mean_dn.tolist())
            global_bias_dn_all.extend(gb_mean_dn.tolist())

            pred_mean_dn = y_pred_dn.mean(dim=(1, 2, 3)).detach().cpu().numpy()
            true_mean_dn = y_true_dn.mean(dim=(1, 2, 3)).detach().cpu().numpy()
            pred_mean_dn_all.extend(pred_mean_dn.tolist())
            true_mean_dn_all.extend(true_mean_dn.tolist())

            a_dn_all.extend(a.view(-1).detach().cpu().numpy().tolist())
            s_all.extend(s.view(-1).detach().cpu().numpy().tolist())

            # Compute metrics on denormalised data
            # Note: compute_per_img_metrics calculates range internally for PWT
            metrics_cnn_dn = compute_per_img_metrics(y_true_dn, y_pred_dn, pwt_threshold=PWT_THRESH, use_magnitude=True)
            metrics_syn4_dn = compute_per_img_metrics(y_true_dn, y_base_dn, pwt_threshold=PWT_THRESH, use_magnitude=True)

            dn_nmae_cnn_all.extend(metrics_cnn_dn["nmae"].tolist())
            dn_nmae_base_all.extend(metrics_syn4_dn["nmae"].tolist())
            dn_pwt_cnn_all.extend(metrics_cnn_dn["pwt"].tolist())
            dn_pwt_base_all.extend(metrics_syn4_dn["pwt"].tolist())
            dn_ssim_cnn_all.extend(metrics_cnn_dn["ssim"].tolist())
            dn_ssim_base_all.extend(metrics_syn4_dn["ssim"].tolist())
            dn_bias_cnn_all.extend(metrics_cnn_dn["bias"].tolist())
            dn_bias_base_all.extend(metrics_syn4_dn["bias"].tolist())

            obs_ids = list(batch["obs_id"])
            set_names = list(batch["set_name"])
            dates = [str(d) for d in batch["date"]]

            for j in range(y_true.shape[0]):
                global_val_rows.append({
                    "Observation": obs_ids[j],
                    "Set name": set_names[j],
                    "Date": dates[j],
                    "CNN NMAE": float(metrics_cnn_dn["nmae"][j]),
                    "SYN4 NMAE": float(metrics_syn4_dn["nmae"][j]),
                    "CNN PWT": float(metrics_cnn_dn["pwt"][j]),
                    "SYN4 PWT": float(metrics_syn4_dn["pwt"][j]),
                    "CNN Bias": float(metrics_cnn_dn["bias"][j]),
                    "SYN4 Bias": float(metrics_syn4_dn["bias"][j]),
                    "CNN SSIM": float(metrics_cnn_dn["ssim"][j]),
                    "SYN4 SSIM": float(metrics_syn4_dn["ssim"][j]),
                    "CNN MAE_DN": float(mae_cnn_dn[j]),
                    "SYN4 MAE_DN": float(mae_base_dn[j]),
                    "CNN Bias_DN_raw": float(bias_raw_cnn_dn[j]),
                    "SYN4 Bias_DN_raw": float(bias_raw_base_dn[j]),
                })

            # Flatten all dimensions except the batch (N) and get mean per image
            y_true_means_per_img = y_true.view(y_true.shape[0], -1).mean(dim=1)
            y_pred_means_per_img = y_pred.view(y_pred.shape[0], -1).mean(dim=1)

            # Extend the main lists with the means from this batch
            true_mean_perimg.extend(y_true_means_per_img.detach().cpu().tolist())
            pred_mean_perimg.extend(y_pred_means_per_img.detach().cpu().tolist())

            if PIX_SAMPLES_PER_IMAGE > 0:
                y_true_dn_flat = y_true_dn.view(y_true_dn.shape[0], -1).detach().cpu().numpy()
                y_pred_dn_flat = y_pred_dn.view(y_pred_dn.shape[0], -1).detach().cpu().numpy()
                y_base_dn_flat = y_base_dn.view(y_base_dn.shape[0], -1).detach().cpu().numpy()

                for j in range(y_true_dn_flat.shape[0]):
                    yt_flat = y_true_dn_flat[j]
                    yp_flat = y_pred_dn_flat[j]
                    yb_flat = y_base_dn_flat[j]

                    valid_idx = np.flatnonzero(np.isfinite(yt_flat) & np.isfinite(yp_flat) & np.isfinite(yb_flat))
                    if valid_idx.size == 0:
                        continue

                    n_take = min(PIX_SAMPLES_PER_IMAGE, valid_idx.size)
                    chosen_idx = rng.choice(valid_idx, size=n_take, replace=False)

                    sampled_true_pixels.extend(
                        yt_flat[chosen_idx].astype(np.float32, copy=False).tolist()
                    )
                    sampled_pred_pixels.extend(
                        yp_flat[chosen_idx].astype(np.float32, copy=False).tolist()
                    )
                    sampled_base_pixels.extend(
                        yb_flat[chosen_idx].astype(np.float32, copy=False).tolist()
                    )

            val_losses.append(criterion(y_pred, y_true)[0].detach().cpu().mean())
            val_base.append(criterion(y_base_pred, y_true)[0].detach().cpu().mean())

            metrics_cnn = compute_per_img_metrics(y_true, y_pred, pwt_threshold=PWT_THRESH)
            metrics_syn4 = compute_per_img_metrics(y_true, y_base_pred, pwt_threshold=PWT_THRESH)

            bias_cnn_all.extend(metrics_cnn["bias"].tolist())
            bias_base_all.extend(metrics_syn4["bias"].tolist())

            val_pwt_cnn_list.append(summarize_metric(metrics_cnn["pwt"]))
            val_pwt_base_list.append(summarize_metric(metrics_syn4["pwt"]))

            val_nmae_cnn_list.append(summarize_metric(metrics_cnn["nmae"]))
            val_nmae_base_list.append(summarize_metric(metrics_syn4["nmae"]))

            val_ssim_cnn_list.append(summarize_metric(metrics_cnn["ssim"]))
            val_ssim_base_list.append(summarize_metric(metrics_syn4["ssim"]))

            B = y_true.size(0)

            # LF fields
            y_t_lf = _blur7(y_true)
            y_p_lf = _blur7(y_pred)

            # Model LF MAE
            lf_cnn_mae = float(F.l1_loss(y_p_lf, y_t_lf).item())

            # HF magnitude “noise ceiling” (how much of true is HF)
            hf_ceil_mae = float(F.l1_loss(y_true, y_t_lf).item())

            # Model HF MAE (how well model captures hf)
            hf_true = y_true - y_t_lf
            hf_pred = y_pred - y_p_lf
            hf_model_mae = float(F.l1_loss(hf_pred, hf_true).item())

            lf_mae_cnn_all.append((lf_cnn_mae, B))
            hf_ceil_mae_all.append((hf_ceil_mae, B))
            hf_model_mae_all.append((hf_model_mae, B))

            # DP/DDP-safe access to exposed parts
            m = cnn_model.module if hasattr(cnn_model, 'module') else cnn_model
            out_centered = m.out_centered         # [B,1,H,W]
            lf_bias = m.lf_bias                   # [B,1,H,W]

            # (a) DC-median fix: how much MAE drops if we only adjust image-wise DC
            delta = (y_true - y_pred).median(dim=3, keepdim=True)[0].median(dim=2, keepdim=True)[0]  # [B,1,1,1]
            y_dcfix = y_pred + delta
            dcfix_mae = float(F.l1_loss(y_dcfix, y_true).item())
            dcfix_mae_all.append((dcfix_mae, B))

            # (d) Residual band split (L2 energy): DC/LF/HF shares of the final error.
            # Energy (mean-square) is additive up to a small LF/HF cross-term (the box
            # blur is not an exact projection), which we record so the partition can be
            # disclosed honestly. DC is exactly orthogonal to the zero-mean residual.
            # Per-image values are kept so the summary can report median + IQR, and the
            # LF/HF split is swept over kernel sizes as a sensitivity check.
            r = y_true - y_pred                                       # [B,1,H,W]
            r_dc = r.mean(dim=(2, 3), keepdim=True)                   # [B,1,1,1] per-image DC offset
            z = r - r_dc                                             # zero-mean residual
            e_tot = (r * r).mean(dim=(1, 2, 3))                       # [B] per-image mean-square
            e_dc = (r_dc * r_dc).mean(dim=(1, 2, 3))                  # [B]
            res_e_tot_all.extend(e_tot.detach().cpu().numpy().tolist())
            res_e_dc_all.extend(e_dc.detach().cpu().numpy().tolist())
            for k in RESIDUAL_BLUR_KERNELS:
                lf_k = _avg_blur(z, k)
                hf_k = z - lf_k
                e_lf = (lf_k * lf_k).mean(dim=(1, 2, 3))              # [B]
                e_hf = (hf_k * hf_k).mean(dim=(1, 2, 3))              # [B]
                res_e_lf_all[k].extend(e_lf.detach().cpu().numpy().tolist())
                res_e_hf_all[k].extend(e_hf.detach().cpu().numpy().tolist())

            y_pred_dc = y_pred.mean(dim=(2, 3), keepdim=True)
            y_true_dc = y_true.mean(dim=(2, 3), keepdim=True)
            dc_err_direct = (y_pred_dc - y_true_dc).abs().mean().item()
            outc_mean = out_centered.mean(dim=(2, 3), keepdim=True)
            lfb_mean = lf_bias.mean(dim=(2, 3), keepdim=True)
            outc_dc = outc_mean.abs().mean().item()
            lfb_dc = lfb_mean.abs().mean().item()
            dc_err_direct_all.append((float(dc_err_direct), B))
            out_centered_dc_all.append((float(outc_dc), B))
            lf_bias_dc_all.append((float(lfb_dc), B))

            # ensure tensor scalars for R3/R5 means
            mu_r3_t = x_neigh[:, 5:6].mean(dim=(2, 3), keepdim=True)  # [B,1,1,1]
            mu_r5_t = x_neigh[:, 2:3].mean(dim=(2, 3), keepdim=True)  # [B,1,1,1]
            y_r3only = m.out_centered + m.lf_bias + mu_r3_t
            mae_r3only = float(F.l1_loss(y_r3only, y_true).item())
            r3_only_mae_all.append((mae_r3only, B))

            y_r5only = m.out_centered + m.lf_bias + mu_r5_t
            mae_r5only = float(F.l1_loss(y_r5only, y_true).item())
            r5_only_mae_all.append((mae_r5only, B))

            # collect for calibrator fits
            mu_true_all_rows.append(y_true.mean(dim=(2, 3)).view(B, 1).detach().cpu().numpy())

            # --- per-image optimal R3/R5 mix upper bound ---
            eps = 1e-8
            mu_r3_t = x_neigh[:, 5:6].mean(dim=(2, 3), keepdim=True)   # [B,1,1,1]
            mu_r5_t = x_neigh[:, 2:3].mean(dim=(2, 3), keepdim=True)   # [B,1,1,1]
            mu_true_t = y_true.mean(dim=(2, 3), keepdim=True)    # [B,1,1,1]

            den = (mu_r3_t - mu_r5_t)
            alpha_opt = torch.clamp((mu_true_t - mu_r5_t) / (den + eps), 0.0, 1.0)  # [B,1,1,1]
            alpha_opt_rows.append(alpha_opt.view(B, 1).detach().cpu().numpy())

            m_hat_mixopt = alpha_opt * mu_r3_t + (1 - alpha_opt) * mu_r5_t

            # MixOpt only
            y_mixopt_top = m.out_centered + m.lf_bias + m_hat_mixopt
            mixopt_top_mae = float(F.l1_loss(y_mixopt_top, y_true).item())
            mixopt_top_mae_all.append((mixopt_top_mae, B))

            err_dn = (y_pred_dn - y_true_dn).mean(dim=(1, 2, 3)).detach().cpu().numpy()
            true_mag = y_true_dn.abs().mean(dim=(1, 2, 3)).detach().cpu().numpy()
            dn_mean_err_all.extend(err_dn.tolist())
            dn_true_mag_all.extend(true_mag.tolist())

    return buffers
