
from hipredict.data.normalization import _resolve_global_stats_path
import torch
import numpy as np
import torch.nn.functional as F
import pandas as pd
import wandb

from hipredict.predict.models import SYN4
from hipredict.post.create_maps import plot_aggregate
from hipredict.data.load_metadata import gen_edr_df
from hipredict.post.evals.eval_categories import run_category_evals
from hipredict.predict.dataset import FilteredHiRISEDataset, get_loader, collate_fn
from hipredict.post.run_evals import _to_hw, compute_per_img_metrics, summarize_metric
from hipredict.post.evals.eval_plots import eval_db_plot
from hipredict.paths import files_dir


def validate(model, criterion, val_dataloader, device, L1):
    model.to(device).eval()
    val_loss = 0.0
    val_l1 = 0.0
    with torch.no_grad():
        for batch in val_dataloader:
            x_band = batch["x_band"].to(device).float()
            x_neigh = batch["x_neigh"].to(device).float()
            x_meta = batch["x_meta"].to(device).float()
            y_val = batch["y"].to(device).float()
            patch_x = batch["patch_x"].to(device).float()
            obs_width = batch["obs_width"].to(device).float()

            y_pred = model(x_band, x_neigh, x_meta, patch_x=patch_x, obs_width=obs_width)
            loss, loss_dict = criterion(y_pred, y_val)
            val_loss += loss.item()

            loss = L1(y_pred, y_val)
            val_l1 += loss.item()

    avg_val_loss = val_loss / len(val_dataloader)
    val_l1 = val_l1 / len(val_dataloader)

    return avg_val_loss, val_l1


def evaluate(model, dataloader, criterion, device, shared_meta_scaler, eval_dr, eval_cats,
             cfg, meta_cols, worker_init_fn, print_adv_metrics=False):
    eps = 1e-8
    _gk_cache = {}

    def _ensure_tensor_output(y):
        return y[0] if isinstance(y, (tuple, list)) else y

    def _gaussian_kernel(k=21, sigma=6.0, device='cpu'):
        key = (int(k), float(sigma), str(device))
        if key in _gk_cache:
            return _gk_cache[key]
        x = torch.arange(k, device=device).float() - (k - 1) / 2
        g = torch.exp(-0.5 * (x / sigma) ** 2)
        g = g / (g.sum() + eps)
        g2 = (g.view(1, 1, k, 1) * g.view(1, 1, 1, k)).contiguous()
        _gk_cache[key] = g2
        return g2

    def _blur(x, k=21, sigma=6.0):
        g2 = _gaussian_kernel(k, sigma, x.device).expand(x.size(1), 1, k, k)
        x = F.pad(x, (k // 2, k // 2, k // 2, k // 2), mode='reflect')
        return F.conv2d(x, g2, groups=x.size(1))

    # gradient-share probe to compare per-term ∂L/∂y magnitudes on one small batch
    def _grad_share_probe(model, _ensure_tensor_output, x_band, x_neigh, x_meta, y_true, loss_weights, patch_x=None, obs_width=None, k=31, sigma=9.0):
        was_training = model.training
        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            model.train()
            y = _ensure_tensor_output(model(x_band, x_neigh, x_meta, patch_x=patch_x, obs_width=obs_width)).clone().detach().requires_grad_(True)
            B, C, H, W = y.shape

            # losses to probe (mirror your training terms)
            mu_p = y.mean(dim=(2, 3), keepdim=True)
            mu_t = y_true.mean(dim=(2, 3), keepdim=True)
            L_mean_mae = F.l1_loss(mu_p, mu_t)
            L_mae = F.l1_loss(y, y_true)
            L_mse = F.mse_loss(y, y_true)

            # LF path MIRRORING TRAINING: downsample by 2 then Gaussian blur (k=15, sigma=5)
            def _lf_fields(z):
                z_ds = F.avg_pool2d(z, kernel_size=2, stride=2)
                return _blur(z_ds, k=k, sigma=sigma)

            yb = _lf_fields(y)
            tb = _lf_fields(y_true)
            L_lf_mse = F.mse_loss(yb, tb)
            L_lf_l1 = F.l1_loss(yb, tb)

            grads = {}

            def _grab_grad(L, tag):
                model.zero_grad(set_to_none=True)
                if y.grad is not None:
                    y.grad.zero_()
                L.backward(retain_graph=True)
                g = y.grad.detach()
                grads[tag] = dict(mean=g.abs().mean().item(), sum=g.abs().sum().item())

            for tag, L in [
                ("mean_mae", L_mean_mae),
                ("mae", L_mae),
                ("mse", L_mse),
                ("lf_mse", L_lf_mse),
                ("lf_l1", L_lf_l1),
            ]:
                _grab_grad(L, tag)

            if not was_training:
                model.eval()

        w = {
            "mae": float(loss_weights.get("mae", 0.0)),
            "mse": float(loss_weights.get("mse", 0.0)),
            "mean_mae": float(loss_weights.get("mean_mae", loss_weights.get("mean_l1", 0.0))),
            "lf_mse": float(loss_weights.get("lf_mse", 0.0)),
            "lf_l1": float(loss_weights.get("lf_l1", 0.0)),
        }
        gw = {k: w.get(k, 0.0) * grads.get(k, {"sum": 0.0})["sum"] for k in grads}
        total = sum(gw.values()) if gw else 0.0
        shares = {k: (gw[k] / total if total > 0 else 0.0) for k in grads}

        print(f"\n[GRAD] probe B={B} H={H} W={W}", flush=True)
        for k in ["mean_mae", "mae", "mse", "lf_mse", "lf_l1"]:
            if k in grads:
                print(f"[GRAD] {k:8s} mean|dL/dy|={grads[k]['mean']:.3e}  sum|dL/dy|={grads[k]['sum']:.3e}  weighted_share={shares.get(k, 0.0):.2%}", flush=True)

        if "mae" in grads and grads["mae"]["sum"] > 0:
            lf_sum = 0.0
            if w.get("lf_l1", 0.0) > 0 and "lf_l1" in grads:
                lf_sum += grads["lf_l1"]["sum"]
            if w.get("lf_mse", 0.0) > 0 and "lf_mse" in grads:
                lf_sum += grads["lf_mse"]["sum"]
            if lf_sum > 0:
                print(f"[GRAD] LF_vs_full ratio ((lf)/(mae)) = {lf_sum / grads['mae']['sum']:.3e}", flush=True)

    model.to(device).eval()
    stats_path = _resolve_global_stats_path(cfg)
    base_model = SYN4.from_global_stats(device, stats_path=stats_path).eval()

    losses = []
    base_losses = []

    # per-image metrics
    E_LF_list = []
    dlf_mae_list = []
    E_HF_list = []
    dc_share_list = []    # DC-ness of LF residual (1.0 ~ uniform offset, 0 ~ structured LF error)

    # mean-head and strain diagnostics
    mu_pred_list = []
    mu_true_list = []

    ran_grad_probe = False

    print(f"[LOSS] config: {getattr(criterion, 'weights', {})}", flush=True)

    with torch.no_grad():
        for batch in dataloader:
            x_band = batch["x_band"].to(device).float()
            x_neigh = batch["x_neigh"].to(device).float()
            x_meta = batch["x_meta"].to(device).float()
            y_true = batch["y"].to(device).float()
            patch_x = batch["patch_x"].to(device).float()
            obs_width = batch["obs_width"].to(device).float()

            y_pred = _ensure_tensor_output(model(x_band, x_neigh, x_meta, patch_x=patch_x, obs_width=obs_width))
            loss, _ = criterion(y_pred, y_true)
            losses.append(loss)

            y_base = _ensure_tensor_output(base_model(x_band, x_neigh, x_meta))
            base_loss, _ = criterion(y_base, y_true)
            base_losses.append(base_loss)

            if not print_adv_metrics:
                continue

            if not ran_grad_probe:
                Bslice = min(y_true.size(0), 2)
                _grad_share_probe(
                    model,
                    _ensure_tensor_output,
                    x_band[:Bslice],
                    x_neigh[:Bslice],
                    x_meta[:Bslice],
                    y_true[:Bslice],
                    getattr(criterion, 'weights', {}),
                    patch_x=patch_x[:Bslice],
                    obs_width=obs_width[:Bslice],
                )
                ran_grad_probe = True

            # means
            mu_t = y_true.mean(dim=(2, 3)).view(-1).detach().cpu().numpy()
            mu_p = y_pred.mean(dim=(2, 3)).view(-1).detach().cpu().numpy()
            mu_true_list.extend(mu_t.tolist())
            mu_pred_list.extend(mu_p.tolist())

            # LF energy and MAE (per image)
            delta = y_pred - y_true
            d_lf = _blur(delta)
            num = (d_lf**2).flatten(1).sum(dim=1)
            den = (delta**2).flatten(1).sum(dim=1).clamp_min(eps)
            E_LF_list.extend((num / den).detach().cpu().numpy().tolist())
            dlf_mae_list.extend(d_lf.abs().flatten(1).mean(dim=1).detach().cpu().numpy().tolist())

            # HF share
            E_HF_list.extend((1.0 - (num / den)).detach().cpu().numpy().tolist())

            # DC share of LF
            dc = d_lf.mean(dim=(2, 3)).abs()
            dca = d_lf.abs().mean(dim=(2, 3)).clamp_min(eps)
            dc_share_list.extend((dc / dca).detach().cpu().numpy().tolist())

    losses = torch.stack(losses).detach().cpu().numpy()
    base_losses = torch.stack(base_losses).detach().cpu().numpy()

    # Per-category evals
    category_out = run_category_evals(
        eval_cats=eval_cats,
        dr=eval_dr,
        FilteredHiRISEDataset=FilteredHiRISEDataset,
        meta_cols=meta_cols,
        shared_meta_scaler=shared_meta_scaler,
        get_loader=get_loader,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        cfg=cfg,
        device=device,
        cnn_model=model,
        syn4=base_model,
        linear_avail=False,
        linear_model=None,
        criterion=criterion,
        compute_per_img_metrics=compute_per_img_metrics,
        summarize_metric=summarize_metric,
        PWT_THRESH=0.01,
        _to_hw=_to_hw,
    )

    if category_out["evals"]:
        avg_eval_nmae = np.mean([v['NMAE_CNN'] for v in category_out['evals'].values()])
        avg_eval_nmae_base = np.mean([v['NMAE_BASE'] for v in category_out['evals'].values()])
        avg_eval_pwt = np.mean([v['PWT_CNN'] for v in category_out['evals'].values()])
        avg_eval_pwt_base = np.mean([v['PWT_BASE'] for v in category_out['evals'].values()])
        avg_eval_bias = np.mean([v['BIAS_CNN'] for v in category_out['evals'].values()])
        avg_eval_bias_base = np.mean([v['BIAS_BASE'] for v in category_out['evals'].values()])
        avg_eval_ssim = np.mean([v['SSIM_CNN'] for v in category_out['evals'].values()])
        avg_eval_ssim_base = np.mean([v['SSIM_BASE'] for v in category_out['evals'].values()])

        wandb.run.summary['avg_eval_nmae'] = avg_eval_nmae
        wandb.run.summary['avg_eval_nmae_base'] = avg_eval_nmae_base
        wandb.run.summary['avg_eval_pwt'] = avg_eval_pwt
        wandb.run.summary['avg_eval_pwt_base'] = avg_eval_pwt_base
        wandb.run.summary['avg_eval_bias'] = avg_eval_bias
        wandb.run.summary['avg_eval_bias_base'] = avg_eval_bias_base
        wandb.run.summary['avg_eval_ssim'] = avg_eval_ssim
        wandb.run.summary['avg_eval_ssim_base'] = avg_eval_ssim_base

        if any(k in category_out["evals"] and str(v.get("eval_category", "")).casefold() == "theme" for k, v in eval_cats.items()):
            fig, _ = eval_db_plot(eval_cats, category_out["evals"], cnn_name=cfg["architecture"]["name"])
            wandb.log({"EVALS_db_plot": wandb.Image(fig)})

    def _m(v):
        return float(np.mean(v)) if len(v) else float("nan")

    def _s(v):
        return float(np.std(v)) if len(v) else float("nan")

    if print_adv_metrics:
        # mean-head and drift summary
        if hasattr(model, "mean_head"):
            try:
                W = model.mean_head.weight.detach().cpu().numpy().reshape(-1)
                b0 = float(model.mean_head.bias.detach().cpu().numpy().reshape(()))
                print("\n===== Scene Mean Head =====", flush=True)
                print(f"weights [mu_ir, mu_bg, mu_r3, mu_r5] = [{W[0]:+.4f}, {W[1]:+.4f}, {W[2]:+.4f}, {W[3]:+.4f}], bias={b0:+.4f}", flush=True)
            except Exception:
                pass

        print("\n===== Core Diagnostics =====", flush=True)
        print(f"[LF]    E_LF  (low-freq error share: cast/bias)           mean={_m(E_LF_list):.3f} std={_s(E_LF_list):.3f}")
        print(f"[LF]    MAE(|δ_LF|) (avg magnitude of LF residuals)       mean={_m(dlf_mae_list):.6f} std={_s(dlf_mae_list):.6f}")
        print(f"[HF]    E_HF  (high-freq share: texture/edges)            mean={_m(E_HF_list):.3f} std={_s(E_HF_list):.3f}", flush=True)

        if len(mu_pred_list) == len(mu_true_list) and len(mu_true_list) > 3:
            dmu = np.array(mu_pred_list) - np.array(mu_true_list)
            print(f"[MEAN] residual drift |μ_pred-μ_true| (scene bias)     MAE={float(np.mean(np.abs(dmu))):.3e}", flush=True)

        # DC-ness of LF residual
        if len(dc_share_list) > 3:
            dcs = np.array(dc_share_list, dtype=np.float64)
            print(f"[LF]   DC share of δ_LF (flat cast vs structured LF)   mean={float(dcs.mean()):.3f} std={float(dcs.std()):.3f}", flush=True)

        if category_out["evals"]:
            print(f"[EVAL] Average eval NMAE: CNN={avg_eval_nmae:.3f}, SYN4={avg_eval_nmae_base:.3f}")
            print(f"[EVAL] Average eval PWT:  CNN={avg_eval_pwt:.3f}, SYN4={avg_eval_pwt_base:.3f}")
            print(f"[EVAL] Average eval Bias: CNN={avg_eval_bias:.6f}, SYN4={avg_eval_bias_base:.6f}")
            print(f"[EVAL] Average eval SSIM: CNN={avg_eval_ssim:.4f}, SYN4={avg_eval_ssim_base:.4f}")

    return losses, base_losses


def create_loss_map(model, criterion, loader, device, datatype='val', output_dir=None, cfg=None):
    output_dir = output_dir or str(files_dir() / 'maps')
    cfg = cfg or {}
    edr_csv = cfg.get("evals", {}).get("edr_csv")
    edr_df = gen_edr_df(EDR_csv_path=edr_csv) if edr_csv else gen_edr_df()
    model.to(device).eval()
    stats_path = _resolve_global_stats_path(cfg or {})
    base = SYN4.from_global_stats(device, stats_path=stats_path).eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            x_band = batch["x_band"].to(device).float()
            x_neigh = batch["x_neigh"].to(device).float()
            x_meta = batch["x_meta"].to(device).float()
            y_true = batch["y"].to(device).float()
            B = y_true.size(0)
            patch_x = batch["patch_x"].to(device).float()
            obs_width = batch["obs_width"].to(device).float()

            y_pred = model(x_band, x_neigh, x_meta, patch_x=patch_x, obs_width=obs_width)
            y_pred_base = base(x_band, x_neigh, x_meta)
            losses = [criterion(y_pred[i: i + 1], y_true[i: i + 1])[0].detach().cpu().item() for i in range(B)]
            losses_base = [criterion(y_pred_base[i: i + 1], y_true[i: i + 1])[0].detach().cpu().item() for i in range(B)]

            obs_ids = list(batch['obs_id'])
            for i in range(B):
                oid = obs_ids[i]
                edr_row = edr_df[edr_df['OBSERVATION_ID'] == oid].iloc[0]
                lat = edr_row['IMAGE_CENTER_LATITUDE']
                lon = edr_row['IMAGE_CENTER_LONGITUDE']
                rows.append({
                    'Lat': lat,
                    'Lon': lon,
                    'Loss': losses[i],
                    'Loss_base': losses_base[i],
                })

    df = pd.DataFrame(rows)
    print(f"Plotting loss map with {len(df)} points")

    upper_bound = df['Loss'].quantile(0.95)
    df['Loss_clamp'] = df['Loss'].clip(upper=upper_bound)

    plot_aggregate(
        df=df,
        value_col='Loss',
        lon_col='Lon',
        lat_col='Lat',
        reducer='mean',
        gridsize=40,
        title='Mean val Loss (clamped at 95th percentile)',
        cb_label_input='Mean val loss per hex',
        out_path=f'{output_dir}/mars_{datatype}_loss_mean.png',
        vmin=0,
        cmap='magma'
    )

    # Calculate relative improvement: (Baseline - New) / Baseline
    # Positive % = New model is better. Negative % = New model is worse.
    df['Rel_Imp'] = (df['Loss_base'] - df['Loss']) / df['Loss_base']

    # Clamp improvement outliers (e.g., between 5th and 95th percentiles) to center the map
    # This prevents division-by-small-number artifacts from ruining the scale
    q_low = df['Rel_Imp'].quantile(0.05)
    q_high = df['Rel_Imp'].quantile(0.95)
    df['Rel_Imp'] = df['Rel_Imp'].clip(lower=q_low, upper=q_high)
    df['RED_Imp'] = df['Rel_Imp'] * 100
    print("Plotting improvement map...")
    plot_aggregate(
        df=df,
        value_col='Rel_Imp',
        lon_col='Lon',
        lat_col='Lat',
        reducer='mean',
        gridsize=40,
        title="Mean Relative Improvement over SYN4",
        cb_label_input='Mean Relative Improvement (%) per hex',
        out_path=f'{output_dir}/mars_{datatype}_improvement.png',
        cmap='RdBu',    # Divergent colormap: Red (worse) to Blue (better)
        vmin=-0.5,      # Fix scales to e.g., -50% to +50% for consistency, or remove to auto-scale
        vmax=0.5
    )

    return df
