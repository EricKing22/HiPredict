import numpy as np

from hipredict.post.evals.eval_passes import RESIDUAL_BLUR_KERNELS, PRIMARY_BLUR_KERNEL


def _wmean_pairs(pairs):
    num = sum(v * n for v, n in pairs if not np.isnan(v))
    den = sum(n for v, n in pairs if not np.isnan(v))
    return float(num / max(den, 1))


def build_eval_summary(buffers):
    # pull from buffers
    dn_nmae_cnn_all = buffers["dn_nmae_cnn_all"]
    dn_nmae_base_all = buffers["dn_nmae_base_all"]
    dn_pwt_cnn_all = buffers["dn_pwt_cnn_all"]
    dn_pwt_base_all = buffers["dn_pwt_base_all"]
    dn_ssim_cnn_all = buffers["dn_ssim_cnn_all"]
    dn_ssim_base_all = buffers["dn_ssim_base_all"]
    dn_bias_cnn_all = buffers["dn_bias_cnn_all"]
    dn_bias_base_all = buffers["dn_bias_base_all"]

    val_pwt_cnn_list = buffers["val_pwt_cnn_list"]
    val_pwt_base_list = buffers["val_pwt_base_list"]
    val_nmae_cnn_list = buffers["val_nmae_cnn_list"]
    val_nmae_base_list = buffers["val_nmae_base_list"]
    val_ssim_cnn_list = buffers["val_ssim_cnn_list"]
    val_ssim_base_list = buffers["val_ssim_base_list"]
    bias_cnn_all = buffers["bias_cnn_all"]
    bias_base_all = buffers["bias_base_all"]

    train_losses = buffers["train_losses"]
    val_losses = buffers["val_losses"]
    train_base = buffers["train_base"]
    val_base = buffers["val_base"]

    dn_mean_err_all = buffers["dn_mean_err_all"]
    dn_true_mag_all = buffers["dn_true_mag_all"]
    true_mean_dn_all = buffers["true_mean_dn_all"]
    pred_mean_dn_all = buffers["pred_mean_dn_all"]

    ecdf_abs_err_dn_cnn_all = buffers["ecdf_abs_err_dn_cnn_all"]
    ecdf_abs_err_dn_base_all = buffers["ecdf_abs_err_dn_base_all"]

    dn_nmae_cnn_arr = np.asarray(dn_nmae_cnn_all, dtype=np.float64)
    dn_nmae_base_arr = np.asarray(dn_nmae_base_all, dtype=np.float64)
    syn4_over_10_mask = (
        np.isfinite(dn_nmae_cnn_arr)
        & np.isfinite(dn_nmae_base_arr)
        & (dn_nmae_base_arr > 10.0)
    )
    dn_nmae_syn4_over_10_n = int(np.sum(syn4_over_10_mask))
    if dn_nmae_syn4_over_10_n:
        syn4_over_10_mean = float(np.nanmean(dn_nmae_base_arr[syn4_over_10_mask]))
        hipredict_over_10_mean = float(np.nanmean(dn_nmae_cnn_arr[syn4_over_10_mask]))
        dn_nmae_syn4_over_10_improvement = (
            syn4_over_10_mean / max(hipredict_over_10_mean, 1e-8)
        )
    else:
        syn4_over_10_mean = np.nan
        hipredict_over_10_mean = np.nan
        dn_nmae_syn4_over_10_improvement = np.nan

    if dn_nmae_cnn_all:
        dn_nmae_cnn = float(np.nanmean(dn_nmae_cnn_all))
        dn_nmae_cnn_std = float(np.nanstd(dn_nmae_cnn_all))
        dn_nmae_base = float(np.nanmean(dn_nmae_base_all))
        dn_nmae_base_std = float(np.nanstd(dn_nmae_base_all))

        dn_pwt_cnn = float(np.nanmean(dn_pwt_cnn_all))
        dn_pwt_cnn_std = float(np.nanstd(dn_pwt_cnn_all))
        dn_pwt_base = float(np.nanmean(dn_pwt_base_all))
        dn_pwt_base_std = float(np.nanstd(dn_pwt_base_all))

        dn_ssim_cnn = float(np.nanmean(dn_ssim_cnn_all))
        dn_ssim_cnn_std = float(np.nanstd(dn_ssim_cnn_all))
        dn_ssim_base = float(np.nanmean(dn_ssim_base_all))
        dn_ssim_base_std = float(np.nanstd(dn_ssim_base_all))

        dn_bias_cnn = float(np.nanmean(dn_bias_cnn_all))
        dn_bias_cnn_std = float(np.nanstd(dn_bias_cnn_all))
        dn_bias_base = float(np.nanmean(dn_bias_base_all))
        dn_bias_base_std = float(np.nanstd(dn_bias_base_all))

        dn_nmae_rel = (dn_nmae_base - dn_nmae_cnn) / max(dn_nmae_base, 1e-8)
        dn_pwt_delta = dn_pwt_cnn - dn_pwt_base
        dn_ssim_delta = dn_ssim_cnn - dn_ssim_base
    else:
        dn_nmae_cnn = dn_nmae_base = dn_pwt_cnn = dn_pwt_base = dn_ssim_cnn = dn_ssim_base = dn_bias_cnn = dn_bias_base = 0.0
        dn_nmae_cnn_std = dn_nmae_base_std = dn_pwt_cnn_std = dn_pwt_base_std = dn_ssim_cnn_std = dn_ssim_base_std = dn_bias_cnn_std = dn_bias_base_std = 0.0

        dn_nmae_rel = (dn_nmae_base - dn_nmae_cnn) / max(dn_nmae_base, 1e-8)
        dn_pwt_delta = dn_pwt_cnn - dn_pwt_base
        dn_ssim_delta = dn_ssim_cnn - dn_ssim_base

    val_pwt_cnn, val_pwt_cnn_std = np.mean(val_pwt_cnn_list), np.std(val_pwt_cnn_list)
    val_pwt_base, val_pwt_base_std = np.mean(val_pwt_base_list), np.std(val_pwt_base_list)
    val_nmae_cnn, val_nmae_cnn_std = np.mean(val_nmae_cnn_list), np.std(val_nmae_cnn_list)
    val_nmae_base, val_nmae_base_std = np.mean(val_nmae_base_list), np.std(val_nmae_base_list)
    val_ssim_cnn, val_ssim_cnn_std = np.mean(val_ssim_cnn_list), np.std(val_ssim_cnn_list)
    val_ssim_base, val_ssim_base_std = np.mean(val_ssim_base_list), np.std(val_ssim_base_list)
    val_bias_cnn = float(np.nanmean(bias_cnn_all))
    val_bias_cnn_std = float(np.nanstd(bias_cnn_all))
    val_bias_base = float(np.nanmean(bias_base_all))
    val_bias_base_std = float(np.nanstd(bias_base_all))

    train_mean, train_std = np.mean(train_losses), np.std(train_losses)
    val_mean, val_std = np.mean(val_losses), np.std(val_losses)
    train_base_mean, train_base_std = np.mean(train_base), np.std(train_base)
    val_base_mean, val_base_std = np.mean(val_base), np.std(val_base)

    pwt_delta_pp = val_pwt_cnn - val_pwt_base
    nmae_rel_red = (val_nmae_base - val_nmae_cnn) / max(val_nmae_base, 1e-8)
    ssim_delta = val_ssim_cnn - val_ssim_base

    bias_dn_metric_like = float(np.mean(100.0 * np.array(dn_mean_err_all) / np.maximum(np.array(dn_true_mag_all), 1e-8)))
    bias_dn_global = float(100.0 * np.mean(dn_mean_err_all) / max(np.mean(dn_true_mag_all), 1e-8))

    t = np.asarray(true_mean_dn_all, dtype=np.float64)
    p = np.asarray(pred_mean_dn_all, dtype=np.float64)
    m = np.isfinite(t) & np.isfinite(p)
    t = t[m]
    p = p[m]

    # Extra diagnostics previously computed in print_global_eval_report
    out_cent_dn_all = buffers["out_cent_dn_all"]
    m_hat_dn_all = buffers["m_hat_dn_all"]
    lf_bias_dn_all = buffers["lf_bias_dn_all"]
    global_bias_dn_all = buffers["global_bias_dn_all"]
    a_dn_all = buffers["a_dn_all"]

    pred_mean_dn = float(np.nanmean(pred_mean_dn_all))
    true_mean_dn = float(np.nanmean(true_mean_dn_all))
    dc_err_mean_dn = float(np.nanmean(np.array(pred_mean_dn_all) - np.array(true_mean_dn_all)))
    dc_err_abs_dn = float(np.nanmean(np.abs(np.array(pred_mean_dn_all) - np.array(true_mean_dn_all))))

    outc_mean_dn = float(np.nanmean(out_cent_dn_all))
    mhat_mean_dn = float(np.nanmean(m_hat_dn_all))
    lfb_mean_dn = float(np.nanmean(lf_bias_dn_all))
    global_bias_mean_dn = float(np.nanmean(global_bias_dn_all))

    a_mean_dn = float(np.nanmean(a_dn_all))
    closure = a_mean_dn + outc_mean_dn + mhat_mean_dn + lfb_mean_dn + global_bias_mean_dn
    pred_minus_closure = pred_mean_dn - closure

    if t.size >= 2 and np.std(t) > 1e-12:
        t0 = t - t.mean()
        p0 = p - p.mean()
        mean_calib_slope = float(np.dot(t0, p0) / max(np.dot(t0, t0), 1e-12))
        mean_calib_intercept = float(p.mean() - mean_calib_slope * t.mean())
        mean_calib_corr = float(np.corrcoef(t, p)[0, 1]) if (np.std(p) > 1e-12) else 0.0
        mean_calib_ok = True
    else:
        mean_calib_slope = 0.0
        mean_calib_intercept = 0.0
        mean_calib_corr = 0.0
        mean_calib_ok = False

    e = np.asarray(dn_mean_err_all, dtype=np.float64)
    m_dn = np.asarray(dn_true_mag_all, dtype=np.float64)
    r = 100.0 * e / np.maximum(m_dn, 1e-8)

    bias_ratio_mean = float(np.nanmean(r))
    bias_ratio_median = float(np.nanmedian(r))
    bias_ratio_p90 = float(np.nanpercentile(r, 90))
    bias_ratio_p99 = float(np.nanpercentile(r, 99))

    true_mag_mean = float(np.nanmean(m_dn))
    true_mag_p10 = float(np.nanpercentile(m_dn, 10))
    true_mag_p1 = float(np.nanpercentile(m_dn, 1))

    posthoc_global_bias_dn = dc_err_mean_dn

    # Residual band decomposition (L2 energy). Per-image mean-square energies were
    # collected per band; shares are taken against the true total energy so the
    # LF/HF cross-term (non-orthogonality of the box blur) is disclosed rather than
    # hidden. DC is exactly orthogonal to the zero-mean residual.
    e_tot = np.asarray(buffers["res_e_tot_all"], dtype=np.float64)
    e_dc = np.asarray(buffers["res_e_dc_all"], dtype=np.float64)
    n_res_imgs = int(e_tot.size)
    E_tot = float(np.nanmean(e_tot)) if n_res_imgs else float("nan")
    E_tot_safe = E_tot if (np.isfinite(E_tot) and E_tot > 0) else float("nan")

    def _energy_share(e_band):
        e_band = np.asarray(e_band, dtype=np.float64)
        return float(np.nanmean(e_band) / E_tot_safe) if e_band.size else float("nan")

    k0 = PRIMARY_BLUR_KERNEL
    e_lf0 = np.asarray(buffers["res_e_lf_all"][k0], dtype=np.float64)
    e_hf0 = np.asarray(buffers["res_e_hf_all"][k0], dtype=np.float64)

    residual_dc_share = _energy_share(e_dc)
    residual_lf_share = _energy_share(e_lf0)
    residual_hf_share = _energy_share(e_hf0)
    residual_cross_share = 1.0 - (residual_dc_share + residual_lf_share + residual_hf_share)
    residual_primary_kernel = k0

    # Per-image share distribution (median + IQR) for the primary kernel.
    def _share_quantiles(e_band):
        e_band = np.asarray(e_band, dtype=np.float64)
        if e_band.size == 0 or n_res_imgs == 0:
            return (float("nan"), float("nan"), float("nan"))
        with np.errstate(divide="ignore", invalid="ignore"):
            frac = e_band / e_tot
        frac = frac[np.isfinite(frac)]
        if frac.size == 0:
            return (float("nan"), float("nan"), float("nan"))
        return (
            float(np.nanmedian(frac)),
            float(np.nanpercentile(frac, 25)),
            float(np.nanpercentile(frac, 75)),
        )

    residual_share_dc_med, residual_share_dc_q25, residual_share_dc_q75 = _share_quantiles(e_dc)
    residual_share_lf_med, residual_share_lf_q25, residual_share_lf_q75 = _share_quantiles(e_lf0)
    residual_share_hf_med, residual_share_hf_q25, residual_share_hf_q75 = _share_quantiles(e_hf0)

    # Kernel-size sensitivity check (DC share is kernel-independent).
    residual_share_sensitivity = {}
    for k in RESIDUAL_BLUR_KERNELS:
        lf_s = _energy_share(buffers["res_e_lf_all"][k])
        hf_s = _energy_share(buffers["res_e_hf_all"][k])
        residual_share_sensitivity[k] = {
            "lf": lf_s,
            "hf": hf_s,
            "cross": 1.0 - (residual_dc_share + lf_s + hf_s),
        }

    dc_err_direct_mean = _wmean_pairs(buffers["dc_err_direct_all"])
    out_centered_dc_mean = _wmean_pairs(buffers["out_centered_dc_all"])
    lf_bias_dc_mean = _wmean_pairs(buffers["lf_bias_dc_all"])

    ecdf_arr_cnn = (
        np.concatenate(ecdf_abs_err_dn_cnn_all)
        if ecdf_abs_err_dn_cnn_all else np.array([], dtype=np.float32)
    )
    ecdf_arr_base = (
        np.concatenate(ecdf_abs_err_dn_base_all)
        if ecdf_abs_err_dn_base_all else np.array([], dtype=np.float32)
    )

    return {
        "dn_nmae_cnn": dn_nmae_cnn, "dn_nmae_cnn_std": dn_nmae_cnn_std,
        "dn_nmae_base": dn_nmae_base, "dn_nmae_base_std": dn_nmae_base_std,
        "dn_pwt_cnn": dn_pwt_cnn, "dn_pwt_cnn_std": dn_pwt_cnn_std,
        "dn_pwt_base": dn_pwt_base, "dn_pwt_base_std": dn_pwt_base_std,
        "dn_ssim_cnn": dn_ssim_cnn, "dn_ssim_cnn_std": dn_ssim_cnn_std,
        "dn_ssim_base": dn_ssim_base, "dn_ssim_base_std": dn_ssim_base_std,
        "dn_bias_cnn": dn_bias_cnn, "dn_bias_cnn_std": dn_bias_cnn_std,
        "dn_bias_base": dn_bias_base, "dn_bias_base_std": dn_bias_base_std,
        "dn_nmae_rel": dn_nmae_rel, "dn_pwt_delta": dn_pwt_delta, "dn_ssim_delta": dn_ssim_delta,
        "dn_nmae_syn4_over_10_improvement": dn_nmae_syn4_over_10_improvement,
        "dn_nmae_syn4_over_10_n": dn_nmae_syn4_over_10_n,
        "dn_nmae_syn4_over_10_mean": syn4_over_10_mean,
        "dn_nmae_hipredict_over_10_mean": hipredict_over_10_mean,

        "val_pwt_cnn": val_pwt_cnn, "val_pwt_cnn_std": val_pwt_cnn_std,
        "val_pwt_base": val_pwt_base, "val_pwt_base_std": val_pwt_base_std,
        "val_nmae_cnn": val_nmae_cnn, "val_nmae_cnn_std": val_nmae_cnn_std,
        "val_nmae_base": val_nmae_base, "val_nmae_base_std": val_nmae_base_std,
        "val_ssim_cnn": val_ssim_cnn, "val_ssim_cnn_std": val_ssim_cnn_std,
        "val_ssim_base": val_ssim_base, "val_ssim_base_std": val_ssim_base_std,
        "val_bias_cnn": val_bias_cnn, "val_bias_cnn_std": val_bias_cnn_std,
        "val_bias_base": val_bias_base, "val_bias_base_std": val_bias_base_std,

        "train_mean": train_mean, "train_std": train_std,
        "val_mean": val_mean, "val_std": val_std,
        "train_base_mean": train_base_mean, "train_base_std": train_base_std,
        "val_base_mean": val_base_mean, "val_base_std": val_base_std,

        "pwt_delta_pp": pwt_delta_pp,
        "nmae_rel_red": nmae_rel_red,
        "ssim_delta": ssim_delta,

        "bias_dn_metric_like": bias_dn_metric_like,
        "bias_dn_global": bias_dn_global,
        "t": t, "p": p,

        "pred_mean_dn": pred_mean_dn,
        "true_mean_dn": true_mean_dn,
        "dc_err_mean_dn": dc_err_mean_dn,
        "dc_err_abs_dn": dc_err_abs_dn,
        "outc_mean_dn": outc_mean_dn,
        "mhat_mean_dn": mhat_mean_dn,
        "lfb_mean_dn": lfb_mean_dn,
        "global_bias_mean_dn": global_bias_mean_dn,
        "a_mean_dn": a_mean_dn,
        "closure": closure,
        "pred_minus_closure": pred_minus_closure,

        "mean_calib_ok": mean_calib_ok,
        "mean_calib_intercept": mean_calib_intercept,
        "mean_calib_slope": mean_calib_slope,
        "mean_calib_corr": mean_calib_corr,

        "bias_ratio_mean": bias_ratio_mean,
        "bias_ratio_median": bias_ratio_median,
        "bias_ratio_p90": bias_ratio_p90,
        "bias_ratio_p99": bias_ratio_p99,

        "true_mag_mean": true_mag_mean,
        "true_mag_p10": true_mag_p10,
        "true_mag_p1": true_mag_p1,

        "posthoc_global_bias_dn": posthoc_global_bias_dn,

        "residual_dc_share": residual_dc_share,
        "residual_lf_share": residual_lf_share,
        "residual_hf_share": residual_hf_share,
        "residual_cross_share": residual_cross_share,
        "residual_primary_kernel": residual_primary_kernel,
        "residual_n_images": n_res_imgs,
        "residual_share_dc_med": residual_share_dc_med,
        "residual_share_dc_q25": residual_share_dc_q25,
        "residual_share_dc_q75": residual_share_dc_q75,
        "residual_share_lf_med": residual_share_lf_med,
        "residual_share_lf_q25": residual_share_lf_q25,
        "residual_share_lf_q75": residual_share_lf_q75,
        "residual_share_hf_med": residual_share_hf_med,
        "residual_share_hf_q25": residual_share_hf_q25,
        "residual_share_hf_q75": residual_share_hf_q75,
        "residual_share_sensitivity": residual_share_sensitivity,

        "dc_err_direct_mean": dc_err_direct_mean,
        "out_centered_dc_mean": out_centered_dc_mean,
        "lf_bias_dc_mean": lf_bias_dc_mean,

        "ecdf_arr_cnn": ecdf_arr_cnn,
        "ecdf_arr_base": ecdf_arr_base,
    }
