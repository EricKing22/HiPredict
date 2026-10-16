"""Evaluation, metrics and reporting for the strip prediction workflow.

This module owns the "score it" half of ``predict_strip``: it runs the
HiPredict model and the SYN4 baseline over the same inputs, scores both against
the truth, packages the per-observation record, and renders the live per-strip
lines plus the final aggregate summary tables. The prediction/apply-model step
itself lives in the main workflow; everything here is reporting.
"""

import numpy as np
import torch


def prediction_metrics(pred_arr, true_arr):
    """MAE/NMAE, bias and error-tail stats for one prediction vs truth."""
    pred = np.asarray(pred_arr, dtype=np.float32)
    true = np.asarray(true_arr, dtype=np.float32)
    valid = np.isfinite(pred) & np.isfinite(true)
    if not np.any(valid):
        return {"mae": np.nan, "nmae": np.nan, "bias": np.nan}

    err = pred[valid] - true[valid]
    abs_err = np.abs(err)
    mae = float(np.mean(abs_err))
    true_mag = float(np.mean(np.abs(true[valid])))
    denom = max(true_mag, 1e-8)
    p95, p99 = (float(v) for v in np.percentile(abs_err, [95, 99]))
    return {
        "mae": mae,
        "nmae": 100.0 * mae / denom,
        "bias": float(err.mean()),
        "abs_err_std": float(np.std(abs_err)),
        "abs_err_p95": p95,
        "abs_err_p95_nmae": 100.0 * p95 / denom,
        "abs_err_p99": p99,
        "abs_err_p99_nmae": 100.0 * p99 / denom,
        "true_mag": true_mag,
    }


def evaluate_pair(pred, samples, true_arr, criterion, device, predict_kwargs=None):
    """Run the HiPredict model and the SYN4 baseline and score both vs truth."""
    predict_kwargs = dict(predict_kwargs or {})

    hip_output = pred.predict(samples, **predict_kwargs)[0]
    hip_stats = pred.batch_stats[0] if getattr(pred, "batch_stats", None) else (np.nan, np.nan)

    syn4_output = pred.predict(samples, syn4=True)[0]
    syn4_stats = pred.batch_stats[0] if getattr(pred, "batch_stats", None) else (np.nan, np.nan)

    true_t = torch.from_numpy(true_arr).float().to(device)
    hip_t = torch.from_numpy(hip_output).float().to(device)
    syn4_t = torch.from_numpy(syn4_output).float().to(device)

    hip_loss, _ = criterion(hip_t, true_t)
    syn4_loss, _ = criterion(syn4_t, true_t)

    return {
        "hip_output": hip_output,
        "syn4_output": syn4_output,
        "hip_stats": hip_stats,
        "syn4_stats": syn4_stats,
        "hip_loss": float(hip_loss.item()),
        "syn4_loss": float(syn4_loss.item()),
        "hip_metrics": prediction_metrics(hip_output, true_arr),
        "syn4_metrics": prediction_metrics(syn4_output, true_arr),
    }


# Metric keys copied from each side's metrics dict into the flat summary record.
_RECORD_METRIC_KEYS = (
    "mae", "nmae", "bias", "abs_err_std", "abs_err_p95", "abs_err_p99", "abs_err_p99_nmae",
)


def strip_record(obs, group, strip_result):
    """Flatten an ``evaluate_pair`` result into the row dict ``print_summary`` reads."""
    record = {
        "obs": obs,
        "group": group,
        "hip_loss": strip_result["hip_loss"],
        "syn4_loss": strip_result["syn4_loss"],
        "patch_set": None,
        "patch_idx": None,
        "patch_hip_loss": np.nan,
        "patch_syn4_loss": np.nan,
    }
    for side, metrics_key in (("hip", "hip_metrics"), ("syn4", "syn4_metrics")):
        metrics = strip_result[metrics_key]
        for key in _RECORD_METRIC_KEYS:
            record[f"{side}_{key}"] = metrics.get(key, np.nan)
    return record


def add_patch_record(record, patch_set, patch_idx, patch_result):
    """Merge a patch ``evaluate_pair`` result into an existing strip record."""
    hip = patch_result["hip_metrics"]
    syn = patch_result["syn4_metrics"]
    record.update({
        "patch_set": patch_set,
        "patch_idx": patch_idx,
        "patch_hip_loss": patch_result["hip_loss"],
        "patch_syn4_loss": patch_result["syn4_loss"],
        "patch_hip_mae": hip["mae"],
        "patch_syn4_mae": syn["mae"],
        "patch_hip_nmae": hip["nmae"],
        "patch_syn4_nmae": syn["nmae"],
    })
    return record


def format_result_line(group, obs, strip_result, pred_stats, elapsed):
    """The live per-strip line printed as each observation finishes (consumer A)."""
    hip = strip_result["hip_metrics"]
    syn = strip_result["syn4_metrics"]
    return (
        f"[{group}] {obs}: "
        f"strip loss HP/S4={strip_result['hip_loss']:.4f}/{strip_result['syn4_loss']:.4f}, "
        f"MAE HP/S4={hip['mae']:.6f}/{syn['mae']:.6f}, "
        f"DN NMAE% HP/S4={hip['nmae']:.2f}/{syn['nmae']:.2f}, "
        f"bias HP/S4={hip['bias']:+.6f}/{syn['bias']:+.6f}, "
        f"norm a/s={pred_stats[0]:.6f}/{pred_stats[1]:.6f}, "
        f"time={elapsed:.2f}s"
    )


def format_patch_line(group, obs, patch_set, patch_idx, patch_result):
    """The optional patch-comparison line appended to a strip's result line."""
    hip = patch_result["hip_metrics"]
    syn = patch_result["syn4_metrics"]
    return (
        f"[{group}] {obs} patch set={patch_set}, patch={patch_idx}: "
        f"loss HP/S4={patch_result['hip_loss']:.4f}/{patch_result['syn4_loss']:.4f}, "
        f"MAE HP/S4={hip['mae']:.6f}/{syn['mae']:.6f}, "
        f"NMAE% HP/S4={hip['nmae']:.2f}/{syn['nmae']:.2f}"
    )


def print_summary(results, failures):
    """Render the final aggregate summary tables over all observations (consumer B)."""
    if not results:
        print("\nNo successful observations to summarise.")
        return

    print("\nSummary")
    summary_sep = "-" * 142
    print(summary_sep)
    print(
        "group           n     loss HP/S4     MAE mean HP/S4   MAE med HP/S4  "
        "DN NMAE mean  DN NMAE med   MAE red.      bias HP/S4       HP wins"
    )
    print(summary_sep)
    groups = []
    for result in results:
        if result["group"] not in groups:
            groups.append(result["group"])
    groups.append("all")

    for group in groups:
        rows = results if group == "all" else [r for r in results if r["group"] == group]
        if not rows:
            continue

        hip_loss = np.nanmean([r["hip_loss"] for r in rows])
        syn_loss = np.nanmean([r["syn4_loss"] for r in rows])
        hip_mae = np.nanmean([r["hip_mae"] for r in rows])
        syn_mae = np.nanmean([r["syn4_mae"] for r in rows])
        hip_nmae = np.nanmean([r["hip_nmae"] for r in rows])
        syn_nmae = np.nanmean([r["syn4_nmae"] for r in rows])
        hip_mae_med = np.nanmedian([r["hip_mae"] for r in rows])
        syn_mae_med = np.nanmedian([r["syn4_mae"] for r in rows])
        hip_nmae_med = np.nanmedian([r["hip_nmae"] for r in rows])
        syn_nmae_med = np.nanmedian([r["syn4_nmae"] for r in rows])
        hip_bias = np.nanmean([r["hip_bias"] for r in rows])
        syn_bias = np.nanmean([r["syn4_bias"] for r in rows])
        wins = sum(r["hip_loss"] < r["syn4_loss"] for r in rows)
        mae_rel_red = 100.0 * (syn_mae - hip_mae) / max(syn_mae, 1e-8)

        loss_pair = f"{hip_loss:.4f}/{syn_loss:.4f}"
        mae_mean_pair = f"{hip_mae:.6f}/{syn_mae:.6f}"
        mae_med_pair = f"{hip_mae_med:.6f}/{syn_mae_med:.6f}"
        nmae_mean_pair = f"{hip_nmae:.2f}/{syn_nmae:.2f}"
        nmae_med_pair = f"{hip_nmae_med:.2f}/{syn_nmae_med:.2f}"
        bias_pair = f"{hip_bias:+.6f}/{syn_bias:+.6f}"
        wins_pair = f"{wins}/{len(rows)}"

        print(
            f"{group:<14} {len(rows):>3}  "
            f"{loss_pair:>16}  "
            f"{mae_mean_pair:>19}  "
            f"{mae_med_pair:>17}  "
            f"{nmae_mean_pair:>12}  "
            f"{nmae_med_pair:>11}  "
            f"{mae_rel_red:5.1f}%  "
            f"{bias_pair:>20}  "
            f"{wins_pair:>7}"
        )
    print(summary_sep)

    print("\nStrip error dispersion (DN absolute error)")
    disp_sep = "-" * 84
    print(disp_sep)
    print("group           n    err std HP/S4        p95 HP/S4        p99 HP/S4")
    print(disp_sep)
    for group in groups:
        rows = results if group == "all" else [r for r in results if r["group"] == group]
        if not rows:
            continue

        hip_std = np.nanmean([r["hip_abs_err_std"] for r in rows])
        syn_std = np.nanmean([r["syn4_abs_err_std"] for r in rows])
        hip_p95 = np.nanmean([r["hip_abs_err_p95"] for r in rows])
        syn_p95 = np.nanmean([r["syn4_abs_err_p95"] for r in rows])
        hip_p99 = np.nanmean([r["hip_abs_err_p99"] for r in rows])
        syn_p99 = np.nanmean([r["syn4_abs_err_p99"] for r in rows])

        std_pair = f"{hip_std:.6f}/{syn_std:.6f}"
        p95_pair = f"{hip_p95:.6f}/{syn_p95:.6f}"
        p99_pair = f"{hip_p99:.6f}/{syn_p99:.6f}"

        print(
            f"{group:<14} {len(rows):>3}  "
            f"{std_pair:>18}  "
            f"{p95_pair:>16}  "
            f"{p99_pair:>16}"
        )
    print(disp_sep)

    print("\nHighlights (DN NMAE%)")
    print("- Strongest SYN4 tails by strip p99 error:")
    tail_rows = sorted(
        results,
        key=lambda r: r.get("syn4_abs_err_p99_nmae", np.nan),
        reverse=True,
    )[:2]
    for r in tail_rows:
        print(
            f"  {r['obs']} ({r['group']}): "
            f"p99 HP/S4={r['hip_abs_err_p99_nmae']:.2f}/{r['syn4_abs_err_p99_nmae']:.2f}, "
            f"mean NMAE HP/S4={r['hip_nmae']:.2f}/{r['syn4_nmae']:.2f}"
        )

    syn4_wins = sorted(
        [r for r in results if r["hip_nmae"] > r["syn4_nmae"]],
        key=lambda r: r["hip_nmae"] - r["syn4_nmae"],
        reverse=True,
    )[:2]
    print("- Biggest strip cases where SYN4 beats HP:")
    if syn4_wins:
        for r in syn4_wins:
            print(
                f"  {r['obs']} ({r['group']}): "
                f"mean NMAE HP/S4={r['hip_nmae']:.2f}/{r['syn4_nmae']:.2f}, "
                f"delta={r['hip_nmae'] - r['syn4_nmae']:+.2f}"
            )
    else:
        print("  none")

    patch_rows_all = [r for r in results if np.isfinite(r.get("patch_hip_loss", np.nan))]
    if patch_rows_all:
        print("\nAverage NMAE by prediction category (DN NMAE%)")
        nmae_sep = "-" * 92
        print(nmae_sep)
        print(
            "group           n  hipred patch  hipred strip  syn4 patch  syn4 strip"
        )
        print(nmae_sep)
        for group in groups:
            rows = patch_rows_all if group == "all" else [
                r for r in patch_rows_all if r["group"] == group
            ]
            if not rows:
                continue

            hipred_patch_nmae = np.nanmean([r["patch_hip_nmae"] for r in rows])
            hipred_strip_nmae = np.nanmean([r["hip_nmae"] for r in rows])
            syn4_patch_nmae = np.nanmean([r["patch_syn4_nmae"] for r in rows])
            syn4_strip_nmae = np.nanmean([r["syn4_nmae"] for r in rows])

            print(
                f"{group:<14} {len(rows):>3}  "
                f"{hipred_patch_nmae:>12.2f}  "
                f"{hipred_strip_nmae:>12.2f}  "
                f"{syn4_patch_nmae:>10.2f}  "
                f"{syn4_strip_nmae:>10.2f}"
            )
        print(nmae_sep)

        print("\nPatch vs strip averages")
        patch_sep = "-" * 88
        print(patch_sep)
        print(
            "group           n  strip loss HP/S4  patch loss HP/S4  "
            "strip HP wins  patch HP wins"
        )
        print(patch_sep)
        for group in groups:
            rows = patch_rows_all if group == "all" else [
                r for r in patch_rows_all if r["group"] == group
            ]
            if not rows:
                continue

            strip_hip_loss = np.nanmean([r["hip_loss"] for r in rows])
            strip_syn_loss = np.nanmean([r["syn4_loss"] for r in rows])
            patch_hip_loss = np.nanmean([r["patch_hip_loss"] for r in rows])
            patch_syn_loss = np.nanmean([r["patch_syn4_loss"] for r in rows])
            strip_wins = sum(r["hip_loss"] < r["syn4_loss"] for r in rows)
            patch_wins = sum(r["patch_hip_loss"] < r["patch_syn4_loss"] for r in rows)

            strip_loss_pair = f"{strip_hip_loss:.4f}/{strip_syn_loss:.4f}"
            patch_loss_pair = f"{patch_hip_loss:.4f}/{patch_syn_loss:.4f}"
            strip_wins_pair = f"{strip_wins}/{len(rows)}"
            patch_wins_pair = f"{patch_wins}/{len(rows)}"

            print(
                f"{group:<14} {len(rows):>3}  "
                f"{strip_loss_pair:>16}  "
                f"{patch_loss_pair:>16}  "
                f"{strip_wins_pair:>13}  "
                f"{patch_wins_pair:>13}"
            )
        print(patch_sep)

        patch_syn4_wins = sorted(
            [r for r in patch_rows_all if r["patch_hip_nmae"] > r["patch_syn4_nmae"]],
            key=lambda r: r["patch_hip_nmae"] - r["patch_syn4_nmae"],
            reverse=True,
        )[:2]
        print("\nPatch highlights (DN NMAE%)")
        print("- Largest patch cases where SYN4 beats HP:")
        if patch_syn4_wins:
            for r in patch_syn4_wins:
                print(
                    f"  {r['obs']} ({r['group']} set={r['patch_set']}, patch={r['patch_idx']}): "
                    f"patch NMAE HP/S4={r['patch_hip_nmae']:.2f}/{r['patch_syn4_nmae']:.2f}, "
                    f"delta={r['patch_hip_nmae'] - r['patch_syn4_nmae']:+.2f}"
                )
        else:
            print("  none")

    if failures:
        failed_obs = ", ".join(f"{obs}({group})" for obs, group, _ in failures)
        print(f"Failed observations: {failed_obs}")
