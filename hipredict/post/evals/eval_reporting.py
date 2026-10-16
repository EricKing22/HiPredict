

import numpy as np


def print_worst_syn4_subset_report(global_val_df, top_n=100):
    if global_val_df is None or global_val_df.empty:
        print("No global validation per-image rows found for worst-SYN4 subset report.")
        return

    cols = ["SYN4 NMAE", "CNN NMAE", "SYN4 PWT", "CNN PWT", "SYN4 SSIM", "CNN SSIM"]
    missing = [c for c in cols if c not in global_val_df.columns]
    if missing:
        print(f"Missing columns for worst-SYN4 subset report: {missing}")
        return

    df = global_val_df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["SYN4 NMAE", "CNN NMAE"])
    if df.empty:
        print("No finite rows available for worst-SYN4 subset report.")
        return

    n_sel = min(int(top_n), len(df))
    worst_df = df.sort_values("SYN4 NMAE", ascending=False).head(n_sel).copy()

    syn4_mean = float(worst_df["SYN4 NMAE"].mean())
    cnn_mean = float(worst_df["CNN NMAE"].mean())
    syn4_median = float(worst_df["SYN4 NMAE"].median())
    cnn_median = float(worst_df["CNN NMAE"].median())
    mean_gap = syn4_mean - cnn_mean
    rel_red = mean_gap / max(syn4_mean, 1e-8)
    win_rate = 100.0 * float((worst_df["CNN NMAE"] < worst_df["SYN4 NMAE"]).mean())

    syn4_pwt = float(worst_df["SYN4 PWT"].mean())
    cnn_pwt = float(worst_df["CNN PWT"].mean())
    syn4_ssim = float(worst_df["SYN4 SSIM"].mean())
    cnn_ssim = float(worst_df["CNN SSIM"].mean())

    print("\n" + "=" * 70)
    print(f"WORST SYN4 SUBSET REPORT (Top {n_sel} by DN NMAE)")
    print("=" * 70)
    print("Subset definition: select validation images with the highest SYN4 DN NMAE.")
    print(f"SYN4 DN NMAE:      mean={syn4_mean:.4f}  median={syn4_median:.4f}")
    print(f"HiPredict DN NMAE: mean={cnn_mean:.4f}  median={cnn_median:.4f}")
    print(f"NMAE gap:          mean(SYN4 - HiPredict) = {mean_gap:.4f}")
    print(f"NMAE rel. reduct.: {100.0 * rel_red:.2f}%")
    print(f"Win rate:          {win_rate:.1f}%  (HiPredict DN NMAE < SYN4 DN NMAE)")
    print(f"PWT mean:          HiPredict={cnn_pwt:.2f}%  SYN4={syn4_pwt:.2f}%  delta={cnn_pwt - syn4_pwt:+.2f} pp")
    print(f"SSIM mean:         HiPredict={cnn_ssim:.4f}  SYN4={syn4_ssim:.4f}  delta={cnn_ssim - syn4_ssim:+.4f}")

    preview_cols = [c for c in ["Observation", "Set name", "Date", "SYN4 NMAE", "CNN NMAE"] if c in worst_df.columns]
    if preview_cols:
        print("\nTop 10 worst SYN4 cases in this subset:")
        print(worst_df.loc[:, preview_cols].head(10).to_string(index=False))

    print("=" * 70)


def print_global_eval_report(summary):

    # local aliases for readability
    train_mean = summary["train_mean"]; train_std = summary["train_std"]                        # noqa
    val_mean = summary["val_mean"]; val_std = summary["val_std"]                                # noqa
    train_base_mean = summary["train_base_mean"]; train_base_std = summary["train_base_std"]    # noqa
    val_base_mean = summary["val_base_mean"]; val_base_std = summary["val_base_std"]            # noqa

    val_pwt_cnn = summary["val_pwt_cnn"]; val_pwt_cnn_std = summary["val_pwt_cnn_std"]          # noqa
    val_pwt_base = summary["val_pwt_base"]; val_pwt_base_std = summary["val_pwt_base_std"]      # noqa
    val_nmae_cnn = summary["val_nmae_cnn"]; val_nmae_cnn_std = summary["val_nmae_cnn_std"]      # noqa
    val_nmae_base = summary["val_nmae_base"]; val_nmae_base_std = summary["val_nmae_base_std"]  # noqa
    val_bias_cnn = summary["val_bias_cnn"]; val_bias_cnn_std = summary["val_bias_cnn_std"]      # noqa
    val_bias_base = summary["val_bias_base"]; val_bias_base_std = summary["val_bias_base_std"]  # noqa
    val_ssim_cnn = summary["val_ssim_cnn"]; val_ssim_cnn_std = summary["val_ssim_cnn_std"]      # noqa
    val_ssim_base = summary["val_ssim_base"]; val_ssim_base_std = summary["val_ssim_base_std"]  # noqa
    pwt_delta_pp = summary["pwt_delta_pp"]
    nmae_rel_red = summary["nmae_rel_red"]
    ssim_delta = summary["ssim_delta"]

    dn_pwt_cnn = summary["dn_pwt_cnn"]; dn_pwt_cnn_std = summary["dn_pwt_cnn_std"]             # noqa
    dn_pwt_base = summary["dn_pwt_base"]; dn_pwt_base_std = summary["dn_pwt_base_std"]         # noqa
    dn_nmae_cnn = summary["dn_nmae_cnn"]; dn_nmae_cnn_std = summary["dn_nmae_cnn_std"]         # noqa
    dn_nmae_base = summary["dn_nmae_base"]; dn_nmae_base_std = summary["dn_nmae_base_std"]     # noqa
    dn_bias_cnn = summary["dn_bias_cnn"]; dn_bias_cnn_std = summary["dn_bias_cnn_std"]         # noqa
    dn_bias_base = summary["dn_bias_base"]; dn_bias_base_std = summary["dn_bias_base_std"]     # noqa
    dn_ssim_cnn = summary["dn_ssim_cnn"]; dn_ssim_cnn_std = summary["dn_ssim_cnn_std"]         # noqa
    dn_ssim_base = summary["dn_ssim_base"]; dn_ssim_base_std = summary["dn_ssim_base_std"]     # noqa
    dn_pwt_delta = summary["dn_pwt_delta"]
    dn_nmae_rel = summary["dn_nmae_rel"]
    dn_ssim_delta = summary["dn_ssim_delta"]
    dn_nmae_syn4_over_10_improvement = summary["dn_nmae_syn4_over_10_improvement"]
    dn_nmae_syn4_over_10_n = summary["dn_nmae_syn4_over_10_n"]

    print('=' * 70)
    print("GLOBAL EVALUATION SUMMARY")
    print('=' * 70)

    print('-' * 20 + ' Training  ' + '-' * 20)
    print(f"Average train loss: {train_mean:.4f} (std: {train_std:.4f})")
    print(f"Average val loss: {val_mean:.4f} (std: {val_std:.4f})")
    print(f"Average train base loss: {train_base_mean:.4f} (std: {train_base_std:.4f})")
    print(f"Average val base loss: {val_base_mean:.4f} (std: {val_base_std:.4f})")

    print('\n' + '-' * 20 + ' NORMALISED SPACE (Magnitude-based) ' + '-' * 20)
    print(" Metrics computed on normalised tensors. Denom = Mean Signal Magnitude.")
    print('-' * 70)
    print(f"PWT:     cnn: {val_pwt_cnn:.4f} ({val_pwt_cnn_std:.4f})   base: {val_pwt_base:.4f} ({val_pwt_base_std:.4f})   delta:{pwt_delta_pp:.4f}")
    print(f"NMAE:    cnn: {val_nmae_cnn:.4f} ({val_nmae_cnn_std:.4f})   base: {val_nmae_base:.4f} ({val_nmae_base_std:.4f})    rel_red:{nmae_rel_red:.2%}")
    print(f"BIAS:    cnn: {val_bias_cnn:.4f} ({val_bias_cnn_std:.4f})   base: {val_bias_base:.4f} ({val_bias_base_std:.4f})")
    print(f"SSIM:    cnn: {val_ssim_cnn:.4f} ({val_ssim_cnn_std:.4f})   base: {val_ssim_base:.4f} ({val_ssim_base_std:.4f})    delta:{ssim_delta:.4f}")

    print('\n' + '-' * 20 + ' PHYSICAL UNITS (DN) (Signal-based) ' + '-' * 17)
    print(" Metrics computed on original DNs. Denom = Mean Signal Magnitude.")
    print('-' * 70)
    print(f"PWT:     cnn: {dn_pwt_cnn:.4f} ({dn_pwt_cnn_std:.4f})  base: {dn_pwt_base:.4f} ({dn_pwt_base_std:.4f})    delta:{dn_pwt_delta:.4f}")
    print(f"NMAE:    cnn: {dn_nmae_cnn:.4f} ({dn_nmae_cnn_std:.4f})   base: {dn_nmae_base:.4f} ({dn_nmae_base_std:.4f})    rel_red:{dn_nmae_rel:.2%}")
    if dn_nmae_syn4_over_10_n:
        print(
            f"{dn_nmae_syn4_over_10_improvement:.2f} times improvement between HiPredict and SYN4 "
            "for images where SYN4 DN NMAE is over 10%"
        )
    else:
        print("No validation images where SYN4 DN NMAE is over 10%")
    print(f"BIAS:    cnn: {dn_bias_cnn:.4f} ({dn_bias_cnn_std:.4f})   base: {dn_bias_base:.4f} ({dn_bias_base_std:.4f})")
    print(f"SSIM:    cnn: {dn_ssim_cnn:.4f} ({dn_ssim_cnn_std:.4f})   base: {dn_ssim_base:.4f} ({dn_ssim_base_std:.4f})    delta:{dn_ssim_delta:.4f}", flush=True)

    print("\n" + "=" * 70)
    print("Model Error Decomposition Analysis")
    print("=" * 70)

    print('-' * 20 + ' DC diagnostics (signed means) ' + '-' * 20)
    print(f"pred mean (DN):         {summary['pred_mean_dn']: .6f}")
    print(f"true mean (DN):         {summary['true_mean_dn']: .6f}")
    print(f"pred-true mean (DN):    {summary['dc_err_mean_dn']: .6f}")
    print(f"|pred-true| mean (DN):  {summary['dc_err_abs_dn']: .6f}")
    print(f"out_centered mean (DN): {summary['outc_mean_dn']: .6f}   (should be ~0)")
    print(f"m_hat mean (DN):        {summary['mhat_mean_dn']: .6f}   (should supply brightness)")
    print(f"lf_bias mean (DN):      {summary['lfb_mean_dn']: .6f}   (should be ~0)")
    print(f"BIAS DN cross-check (per-image): {summary['bias_dn_metric_like']:.4f}")
    print(f"BIAS DN cross-check (global):    {summary['bias_dn_global']:.4f}")
    print()
    print(f"a mean (DN):              {summary['a_mean_dn']: .6f}")
    print(f"closure (DN):             {summary['closure']: .6f}   (should match pred mean)")
    print(f"pred - closure (DN):      {summary['pred_minus_closure']: .6f}")

    if summary["mean_calib_ok"]:
        print(
            f"Mean calib fit (DN): pred ≈ {summary['mean_calib_intercept']:+.6e} + "
            f"{summary['mean_calib_slope']:.6f} * true   (corr={summary['mean_calib_corr']:.4f})"
        )
    else:
        print("Mean calib fit (DN): insufficient variance/samples")

    print(
        f"Bias ratio stats (%): mean={summary['bias_ratio_mean']:.4f}  "
        f"median={summary['bias_ratio_median']:.4f}  "
        f"p90={summary['bias_ratio_p90']:.4f}  "
        f"p99={summary['bias_ratio_p99']:.4f}"
    )
    print(
        f"True mag stats (DN): mean={summary['true_mag_mean']:.6e}  "
        f"p10={summary['true_mag_p10']:.6e}  "
        f"p1={summary['true_mag_p1']:.6e}"
    )

    print(f"Post-hoc global bias correction (DN): subtract {summary['posthoc_global_bias_dn']:+.6e}")

    k0 = summary['residual_primary_kernel']
    print('\n' + '-' * 20 + ' Residual energy shares ' + '-' * 20)
    print(f"(L2 energy of error residual, partitioned into DC/LF/HF bands; "
          f"LF/HF split at {k0}px box blur; n={summary['residual_n_images']} images)")

    print(f"DC: {summary['residual_dc_share']:.2%}   "
          f"(per-image median {summary['residual_share_dc_med']:.2%}, "
          f"IQR {summary['residual_share_dc_q25']:.2%}-{summary['residual_share_dc_q75']:.2%})")
    print(f"LF: {summary['residual_lf_share']:.2%}   "
          f"(per-image median {summary['residual_share_lf_med']:.2%}, "
          f"IQR {summary['residual_share_lf_q25']:.2%}-{summary['residual_share_lf_q75']:.2%})")
    print(f"HF: {summary['residual_hf_share']:.2%}   "
          f"(per-image median {summary['residual_share_hf_med']:.2%}, "
          f"IQR {summary['residual_share_hf_q25']:.2%}-{summary['residual_share_hf_q75']:.2%})")
    print(f"LF/HF cross-term (non-additivity): {summary['residual_cross_share']:.2%}   "
          f"(small => bands are near-orthogonal, partition is meaningful)")

    print("\nKernel-size sensitivity (DC share is kernel-independent):")
    print(f"  {'kernel':>8} {'LF':>8} {'HF':>8} {'cross':>8}")
    for k, sh in summary['residual_share_sensitivity'].items():
        flag = "  <- primary" if k == k0 else ""
        print(f"  {f'{k}px':>8} {sh['lf']:>8.2%} {sh['hf']:>8.2%} {sh['cross']:>8.2%}{flag}")

    print('\n' + '-' * 20 + ' Identity checks (normalised tensor units) ' + '-' * 20)

    print(f"  |mean(y_pred)-mean(y_true)|: {summary['dc_err_direct_mean']:.6f}")
    print(f"  mean(out_centered):          {summary['out_centered_dc_mean']:.6f}")
    print(f"  mean(lf_bias):               {summary['lf_bias_dc_mean']:.6f}")

    print("=" * 70 + "\n")
