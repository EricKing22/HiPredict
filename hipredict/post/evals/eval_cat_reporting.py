# eval_category_reporting.py
import os
import numpy as np
import pandas as pd


def save_category_results_and_print(
    *,
    category_out,
    SAVE_DIR,
    cnn_name,
    PWT_PERC,
):
    evals = category_out["evals"]
    rows = category_out["rows"]
    cnn_avg_pix = category_out["cnn_avg_pix"]
    syn4_avg_pix = category_out["syn4_avg_pix"]
    true_avg_pix = category_out["true_avg_pix"]
    ir_avg_pix = category_out["ir_avg_pix"]
    bg_avg_pix = category_out["bg_avg_pix"]
    r5_avg_pix = category_out["r5_avg_pix"]
    r3_avg_pix = category_out["r3_avg_pix"]

    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(SAVE_DIR, f"{cnn_name}_evals.csv")
    os.makedirs(SAVE_DIR, exist_ok=True)
    results_df.to_csv(csv_path, index=False)
    print(f"Saved per-sample metrics to {csv_path}")

    print(f"Mean pixel value for cnn: {np.mean(cnn_avg_pix):.4f}, std: {np.std(cnn_avg_pix):.4f}")
    print(f"Median pixel value for cnn: {np.median(cnn_avg_pix):.4f}, range: {max(cnn_avg_pix):.4f} - {min(cnn_avg_pix)}")
    print(f"Mean pixel value for syn4: {np.mean(syn4_avg_pix):.4f}, std: {np.std(syn4_avg_pix):.4f}")
    print(f"Mean pixel value for ytrue: {np.mean(true_avg_pix):.4f}, std: {np.std(true_avg_pix):.4f}")
    print(f"Mean pixel value for ir: {np.mean(ir_avg_pix):.4f}, std: {np.std(ir_avg_pix):.4f}")
    print(f"Mean pixel value for bg: {np.mean(bg_avg_pix):.4f}, std: {np.std(bg_avg_pix):.4f}")
    print(f"Mean pixel value for r3: {np.mean(r3_avg_pix):.4f}, std: {np.std(r3_avg_pix):.4f}")
    print(f"Mean pixel value for r5: {np.mean(r5_avg_pix):.4f}, std: {np.std(r5_avg_pix):.4f}")

    def print_evals(cats, eval_dict):
        print(f"\n========== PWT @ {PWT_PERC}% across categories ==========")
        print(f"\t- PWT @ {PWT_PERC}%: Percentage of pixels with absolute error within a magnitude-based tolerance.")
        print(f"\t- Pixel is within tol if |pred - true| ≤ {PWT_PERC}% of mean |true signal| (per image, DN space).")
        print("\t- higher percentage is better")
        print("\t- Δ shown in percentage points (pp) vs baseline.\n")

        print(f"| {'Category':<20} | {'CNN':<10} | {'Baseline':<10} | {'Δ (pp)':<6} | {'Linear':<10} | {'N':<6} |")
        print(f"|{'=' * 22}|{'=' * 12}|{'=' * 12}|{'=' * 8}|{'=' * 12}|{'=' * 8}|")
        for cat in cats:
            e = eval_dict[cat]
            print(f"| {cat:<20.20} | {e['PWT_CNN']:>9.1f}% | {e['PWT_BASE']:>9.1f}% | {e['PWT_DELTA']:+6.1f} | {e['PWT_LINEAR']:>9.1f}% | {e['N']!s:>6} |")

        avg_pwt_evals = np.mean([v['PWT_CNN'] for v in eval_dict.values()])
        avg_pwt_evals_base = np.mean([v['PWT_BASE'] for v in eval_dict.values()])
        print(f"\nAverage PWT% across categories - CNN: {avg_pwt_evals:.2f}% - SYN4: {avg_pwt_evals_base:.2f}%")

        print("\n========== NMAE% across categories ==========")
        print("\t- NMAE%: Mean absolute error as a percentage of mean |true signal| (per image, DN space).")
        print("\t- Lower is better.")
        print("\t- ↓Rel is relative reduction vs baseline.\n")

        print(f"| {'Category':<20} | {'CNN':<10} | {'Baseline':<10} | {'↓Rel':<6} | {'Linear':<10} | {'N':>6} |")
        print(f"|{'=' * 22}|{'=' * 12}|{'=' * 12}|{'=' * 8}|{'=' * 12}|{'=' * 8}|")
        for cat in cats:
            e = eval_dict[cat]
            rel = 100.0 * e['NMAE_REL_RED'] if np.isfinite(e['NMAE_REL_RED']) else np.nan
            print(f"| {cat:<20.20} | {e['NMAE_CNN']:>10.2f} | {e['NMAE_BASE']:>10.2f} | {rel:5.0f}% | {e['NMAE_LINEAR']:>10.2f} | {e['N']!s:>6} |")

        avg_nmae_evals = np.mean([v['NMAE_CNN'] for v in eval_dict.values()])
        avg_nmae_evals_base = np.mean([v['NMAE_BASE'] for v in eval_dict.values()])
        print(f"\nAverage NMAE% across categories - CNN: {avg_nmae_evals:.2f} - SYN4: {avg_nmae_evals_base:.2f}")

        print("\n========== Bias %DR across categories ==========")
        print("\t- Bias %Mag: Signed mean error as a percentage of mean |true signal| (negative = under-prediction).")
        print("\t- Closer to zero is better.")
        print("\t- Signed mean error as % of mean |true|; Shift = CNN - Baseline (percentage points).\n")
        print(f"| {'Category':<20} | {'CNN':<10} | {'Baseline':<10} | {'Shift (pp)':<13} | {'Linear':<10} | {'N':>6} |")
        print(f"|{'=' * 22}|{'=' * 12}|{'=' * 12}|{'=' * 15}|{'=' * 12}|{'=' * 8}|")
        for cat in cats:
            e = eval_dict[cat]
            shift = e['BIAS_CNN'] - e['BIAS_BASE']
            print(f"| {cat:<20.20} | {e['BIAS_CNN']:+10.2f} | {e['BIAS_BASE']:+10.2f} | {shift:+13.2f} | {e['BIAS_LINEAR']:+10.2f} | {e['N']!s:>6} |")

        avg_bias_evals = np.mean([v['BIAS_CNN'] for v in eval_dict.values()])
        avg_bias_evals_base = np.mean([v['BIAS_BASE'] for v in eval_dict.values()])
        print(f"\nAverage Bias% across categories - CNN: {avg_bias_evals:.2f}% - SYN4: {avg_bias_evals_base:.2f}%")

        print("\n========== SSIM across categories ==========")
        print("\t- SSIM: Structural Similarity index (0-1), measuring perceptual/structural fidelity.")
        print("\t- Higher is better.")
        print("\t- Δ is the absolute SSIM gain (CNN - Baseline).\n")

        print(f"| {'Category':<20} | {'CNN':<6} | {'Baseline':<8} | {'Δ':<6} | {'Linear':<10} | {'N':<6} |")
        print(f"|{'=' * 22}|{'=' * 8}|{'=' * 10}|{'=' * 8}|{'=' * 12}|{'=' * 8}|")
        for cat in cats:
            e = eval_dict[cat]
            print(f"| {cat:<20.20} | {e['SSIM_CNN']:6.3f} | {e['SSIM_BASE']:8.3f} | {e['SSIM_DELTA']:+6.3f} | {e['SSIM_LINEAR']:>10.3f} | {e['N']!s:>6} |")

        avg_ssim_evals = np.mean([v['SSIM_CNN'] for v in eval_dict.values()])
        avg_ssim_evals_base = np.mean([v['SSIM_BASE'] for v in eval_dict.values()])
        print(f"\nAverage SSIM across categories - CNN: {avg_ssim_evals:.3f} - SYN4: {avg_ssim_evals_base:.3f}")

        print("\n========== Win rate across categories ==========")
        print("\t- Win rate: % of images where the CNN's NMAE% is lower than the baseline within the category.")
        print("\t- Higher is better.")
        print("\t- % of images where CNN NMAE% < Baseline NMAE%.\n")

        print(f"| {'Category':<20} | {'Win rate':<10} | {'Linear':<10} | {'N':<6} |")
        print(f"|{'=' * 22}|{'=' * 12}|{'=' * 12}|{'=' * 8}|")
        for cat in cats:
            e = eval_dict[cat]
            print(f"| {cat:<20.20} | {e['WIN_RATE']:>9.1f}% | {e['WIN_RATE_LINEAR']:>10.1f}% | {e['N']!s:>6} |")
        print("\n\n")

    print_evals(list(evals.keys()), evals)


def compare_global_vs_category_stats(*, buffers, category_out):
    rows = category_out["rows"]
    if not rows:
        print("No category rows found for diagnostics.")
        return

    df = pd.DataFrame(rows)
    print(f"compare_global_vs_category_stats - Num lines in evals: {len(rows)}")

    # Global val arrays from buffers
    g_nmae_cnn = np.asarray(buffers["dn_nmae_cnn_all"], dtype=np.float64)
    g_nmae_base = np.asarray(buffers["dn_nmae_base_all"], dtype=np.float64)
    g_mae_cnn = np.asarray(buffers["dn_mae_cnn_all"], dtype=np.float64)
    g_mae_base = np.asarray(buffers["dn_mae_base_all"], dtype=np.float64)
    g_mag = np.asarray(buffers["dn_true_mag_all"], dtype=np.float64)
    g_dr = np.asarray(buffers["dn_true_dr_all"], dtype=np.float64)

    print("\n" + "=" * 80)
    print("GLOBAL vs CATEGORY DATASET DIAGNOSTICS (DN space)")
    print("=" * 80)

    print("\n--- Global val (all images) ---")
    print(f"N images:      {g_nmae_cnn.size}")
    print(f"CNN NMAE%:     mean={np.nanmean(g_nmae_cnn):.3f}  median={np.nanmedian(g_nmae_cnn):.3f}")
    print(f"SYN4 NMAE%:    mean={np.nanmean(g_nmae_base):.3f}  median={np.nanmedian(g_nmae_base):.3f}")
    print(f"CNN MAE_DN:    mean={np.nanmean(g_mae_cnn):.6f}  median={np.nanmedian(g_mae_cnn):.6f}")
    print(f"SYN4 MAE_DN:   mean={np.nanmean(g_mae_base):.6f}  median={np.nanmedian(g_mae_base):.6f}")
    print(f"TrueMag_DN:    mean={np.nanmean(g_mag):.6f}  p10={np.nanpercentile(g_mag, 10):.6f}  p1={np.nanpercentile(g_mag, 1):.6f}")
    print(f"TrueDR_DN:     mean={np.nanmean(g_dr):.6f}  p10={np.nanpercentile(g_dr, 10):.6f}  p1={np.nanpercentile(g_dr, 1):.6f}")

    # Category summary
    grp = df.groupby("Category", dropna=False)

    cat_stats = grp.agg(
        N=("CNN NMAE", "size"),
        CNN_NMAE_mean=("CNN NMAE", "mean"),
        SYN4_NMAE_mean=("SYN4 NMAE", "mean"),
        CNN_NMAE_median=("CNN NMAE", "median"),
        SYN4_NMAE_median=("SYN4 NMAE", "median"),
        CNN_MAE_DN_mean=("CNN MAE_DN", "mean"),
        SYN4_MAE_DN_mean=("SYN4 MAE_DN", "mean"),
        TrueMag_DN_mean=("TrueMag_DN", "mean"),
        TrueMag_DN_median=("TrueMag_DN", "median"),
        TrueDR_DN_mean=("TrueDR_DN", "mean"),
        CNN_Bias_DN_mean=("CNN Bias_DN_raw", "mean"),
        SYN4_Bias_DN_mean=("SYN4 Bias_DN_raw", "mean"),
    ).reset_index()

    # Relative-to-global ratios (super helpful)
    cat_stats["CNN_NMAE_vs_global"] = cat_stats["CNN_NMAE_mean"] / max(np.nanmean(g_nmae_cnn), 1e-8)
    cat_stats["CNN_MAE_DN_vs_global"] = cat_stats["CNN_MAE_DN_mean"] / max(np.nanmean(g_mae_cnn), 1e-8)
    cat_stats["TrueMag_vs_global"] = cat_stats["TrueMag_DN_mean"] / max(np.nanmean(g_mag), 1e-8)

    print("\n--- Category means (sorted by CNN NMAE%) ---")
    cat_stats = cat_stats.sort_values("CNN_NMAE_mean", ascending=False)

    for _, r in cat_stats.iterrows():
        print(
            f"{r['Category']:<20.20} | N={int(r['N']):>4} | "
            f"NMAE cnn/base={r['CNN_NMAE_mean']:>7.2f}/{r['SYN4_NMAE_mean']:>7.2f} | "
            f"MAE_DN cnn/base={r['CNN_MAE_DN_mean']:.5f}/{r['SYN4_MAE_DN_mean']:.5f} | "
            f"TrueMag={r['TrueMag_DN_mean']:.5f} | TrueDR={r['TrueDR_DN_mean']:.5f} | "
            f"(NMAE x{r['CNN_NMAE_vs_global']:.1f}, MAE x{r['CNN_MAE_DN_vs_global']:.1f}, Mag x{r['TrueMag_vs_global']:.2f})"
        )

    # Quick overlap warning (categories are not disjoint)
    # Same observation can appear in multiple categories; this is expected but worth surfacing.
    if "Observation" in df.columns:
        obs_per_cat = grp["Observation"].nunique()
        total_unique_obs = df["Observation"].nunique()
        total_rows = len(df)
        print("\n--- Overlap diagnostics ---")
        print(f"Total category rows: {total_rows}")
        print(f"Unique observations across all categories: {total_unique_obs}")
        print(f"Mean unique observations per category: {obs_per_cat.mean():.1f}")
