import torch
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import os
import time
import wandb
from datetime import datetime
import pandas as pd
import argparse
import sys
from pprint import pprint
import random
import joblib
import yaml
from pathlib import Path
from hipredict.paths import files_dir, resolve_path

from hipredict.predict.get_model import get_model
from hipredict.predict.dataset import FilteredHiRISEDataset, collate_fn, get_loader
from hipredict.data.normalization import normalization_kwargs_from_config, _resolve_global_stats_path
from hipredict.predict.evaluate import evaluate, create_loss_map
from hipredict.predict.loss import get_criterion
from hipredict.predict.utils import split_by_date, save_batch_imgs, load_cfg, plot_losses, worker_init_fn
from hipredict.predict.train import train


def main(n_workers, cfg, sweep, data_record_path, run_name, notrain, eval_cfg_path,
         test=False, eval_mode=True, save_loss_map=False, eval_dr_path=None, ):
    main_start = time.time()
    split_sum = np.round(cfg['data']['train_size'] + cfg['data']['test_size'] + cfg['data']['val_size'], decimals=1)
    assert split_sum == 1.0, (
        f"Train, test, val sizes must sum to 1 not {split_sum:.1f}."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = cfg['architecture']['name']
    model_savedir = f"{cfg['save']['model_dir']}/{model_name}"
    normalization_kwargs = normalization_kwargs_from_config(cfg)
    if (eval_mode or test) and normalization_kwargs["norm_mode"] != "global_robust":
        raise ValueError("SYN4 evaluation requires global_robust normalization.")
    wandb.config.update(
        {
            "device": str(device),
            "cuda available": torch.cuda.is_available(),
            "n_workers": n_workers,
        }
    )

    data_record = pd.read_csv(data_record_path)
    data_record = data_record[data_record["Set"] < cfg['data']['num_sets']]

    data_record.attrs["record_dir"] = str(Path(data_record_path).resolve().parent)
    eval_dr, eval_cats = None, {}
    if eval_mode or test:
        eval_dr = pd.read_csv(eval_dr_path)
        eval_dr.attrs["record_dir"] = str(Path(eval_dr_path).resolve().parent)
        with open(eval_cfg_path, 'r') as file:
            eval_cats = yaml.safe_load(file) or {}
        eval_cats = {k: v for k, v in eval_cats.items() if v and v.get('oids')}

    true_sets = data_record["Set"].unique()
    wandb.config.update({"num_sets": cfg['data']['num_sets'], "true_sets": len(true_sets)})

    if not sweep:
        print("Config:")
        pprint(dict(wandb.config))
        print()

    # Custom split by date
    train_sets, test_sets, val_sets, train_val_date, val_test_date = split_by_date(
        data_record, train_size=cfg['data']['train_size'], test_size=cfg['data']['test_size'], val_size=cfg['data']['val_size']
    )


    # Obtain meta cols
    meta_cols = cfg['data'].get('meta_cols', None)
    if meta_cols is not None:
        missing_cols = set(meta_cols) - set(data_record.columns)
        assert not missing_cols, f"Missing meta columns in data record: {missing_cols}"

    # Set data root for dataset
    data_root = cfg['data'].get('patch_root')
    if notrain:
        # Create datasets with shared scalar
        try:
            shared_meta_scaler = joblib.load(os.path.join(model_savedir, 'latest/meta_scalar.joblib'))
            print(f"Meta scalar successfully loaded from {model_savedir}/latest")
        except FileNotFoundError:
            if meta_cols:
                raise FileNotFoundError(f"Required metadata scaler missing in {model_savedir}/latest")
            shared_meta_scaler = None

        train_dataset = FilteredHiRISEDataset(
            allowed_sets=train_sets, data_record=data_record, sweep=sweep, transform=None, meta_cols=meta_cols, meta_scaler=shared_meta_scaler, **normalization_kwargs, data_root=data_root
        )
        shared_meta_scaler = train_dataset.meta_scaler if meta_cols is not None else None
        val_dataset = FilteredHiRISEDataset(
            allowed_sets=val_sets, data_record=data_record, sweep=sweep, transform=None, meta_cols=meta_cols, meta_scaler=shared_meta_scaler, **normalization_kwargs, data_root=data_root
        )

    else:
        # Create meta scalar from train dataset
        train_dataset = FilteredHiRISEDataset(
            allowed_sets=train_sets, data_record=data_record, sweep=sweep, transform=None, meta_cols=meta_cols, **normalization_kwargs, data_root=data_root
        )
        shared_meta_scaler = train_dataset.meta_scaler if meta_cols is not None else None
        val_dataset = FilteredHiRISEDataset(
            allowed_sets=val_sets, data_record=data_record, sweep=sweep, transform=None, meta_cols=meta_cols, meta_scaler=shared_meta_scaler, **normalization_kwargs, data_root=data_root
        )

    train_dataloader = get_loader(
        train_dataset,
        cfg['optim']['batch_size'],
        num_workers=n_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn(cfg['seed']),
        seed=cfg['seed'],
    )
    val_dataloader = get_loader(
        val_dataset,
        cfg['optim']['batch_size'],
        num_workers=n_workers,
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn(cfg['seed']),
        seed=cfg['seed'],
    )
    if not sweep:
        print(f"Data loaders created after {time.time() - main_start:.2f} seconds")


    criterion = get_criterion(cfg)

    if notrain:
        # Get most recently saved model
        trained_model, _ = get_model(
            model_name, device=device, trained=True,
            model_save_dir=os.path.join(model_savedir, "latest"), cfg=cfg
        )
        trained_model = trained_model.to(device)
        print(f"Loaded saved model from {os.path.join(model_savedir, 'latest')} after {time.time() - main_start:.2f} seconds")

    else:
        sched_cfg = cfg['optim'].get('scheduler', None)
        if sched_cfg:
            if sched_cfg.get('name') == 'one_cycle':
                start_lr = sched_cfg['max_lr'] / sched_cfg.get('div_factor', 10)
            else:
                raise ValueError(f"Unknown schedule name given: {sched_cfg.get('name')} ")
        else:
            start_lr = cfg['optim']['lr']

        model, _ = get_model(model_name, device=device, trained=False, cfg=cfg)
        if hasattr(model, 'mean_head'):
            head_params = list(model.mean_head.parameters())    # type: ignore
            body_params = [p for n, p in model.named_parameters() if not n.startswith('mean_head')]
            optimiser = AdamW(
                [
                    {'params': head_params, 'lr': start_lr * 2.0, 'weight_decay': cfg['optim']['weight_decay']},
                    {'params': body_params, 'lr': start_lr, 'weight_decay': cfg['optim']['weight_decay']}
                ]
            )
        else:
            optimiser = AdamW(model.parameters(), lr=start_lr, weight_decay=cfg['optim']['weight_decay'])

        if sched_cfg and sched_cfg.get('name') == 'one_cycle':
            scheduler = OneCycleLR(
                optimiser,
                max_lr=sched_cfg['max_lr'],
                total_steps=(len(train_dataloader) * cfg['optim']['epochs']),
                pct_start=sched_cfg.get('pct_start', 0.15),
                div_factor=sched_cfg.get('div_factor', 10),
                final_div_factor=sched_cfg.get('final_div_factor', 200),
                anneal_strategy='cos'
            )
        else:
            scheduler = None

        wandb.run.summary["n_params"] = sum(p.numel() for p in model.parameters())
        wandb.config.update(
            {
                "model": model.__class__.__name__,
                "criterion": criterion.__class__.__name__,
                "optimiser": optimiser.__class__.__name__,
                "scheduler": cfg['optim'].get('scheduler', {}).get('name', 'none'),
            }, allow_val_change=True
        )
        sys.stdout.flush()

        train_start = time.time()
        if not sweep:
            print(f"\nStarting training after {time.time() - main_start:.2f} seconds...")

        trained_model, losses_dict = train(
            model,
            criterion,
            optimiser,
            train_dataloader,
            val_dataloader,
            device,
            epochs=cfg['optim']['epochs'],
            scheduler=scheduler,
            SEED=cfg['seed'],
            early_stopping=cfg['optim'].get('early_stopping', False),
        )

        training_time = time.time() - train_start
        wandb.run.summary["training_time"] = training_time
        if not sweep:
            print(f"Training complete after {training_time:.2f} seconds.")

    if not sweep:
        print(f"Total elapsed time: {time.time() - main_start:.2f} seconds\n")

    save_imgs = cfg.get('save', {}).get('imgs', False)
    save_model = cfg.get('save', {}).get('model', False)

    if save_imgs:
        with torch.no_grad():
            trained_model.eval()
            first_batch = next(iter(val_dataloader))
            imgs_indiv = cfg.get('save', {}).get('imgs_indiv', False)
            pred_images_dir = cfg.get('save', {}).get('img_dir') or str(files_dir() / 'pred_images')
            stats_path = _resolve_global_stats_path(cfg)
            save_batch_imgs(
                first_batch, trained_model, pred_images_dir, device,
                img_type='pred', criterion=criterion, arch=model_name,
                save_indiv=imgs_indiv, stats_path=stats_path
            )
            if imgs_indiv:
                print(f"Saved first batch of pred images (plus indiv) after {time.time() - main_start:.2f} seconds")
            else:
                print(f"Saved first batch of pred images after {time.time() - main_start:.2f} seconds")
        if not notrain:
            evals_dir = cfg.get('save', {}).get('evals_dir') or str(files_dir() / 'evals')
            os.makedirs(evals_dir, exist_ok=True)
            plot_losses(losses_dict, save_path=f'{evals_dir}/loss_plots.png', rolling_window=25)

    if eval_mode:
        train_losses, base_train_losses = evaluate(
            trained_model, train_dataloader, criterion, device, shared_meta_scaler,
            eval_dr, eval_cats, cfg, meta_cols, worker_init_fn=worker_init_fn, print_adv_metrics=False
        )
        val_losses, base_val_losses = evaluate(
            trained_model, val_dataloader, criterion, device, shared_meta_scaler,
            eval_dr, eval_cats, cfg, meta_cols, worker_init_fn=worker_init_fn, print_adv_metrics=True
        )
        wandb.run.summary["mean_val_loss"] = val_losses.mean()
        wandb.run.summary["base_val_loss"] = base_val_losses.mean()
        if not sweep:
            print(f"\nMean validation loss: {val_losses.mean():.2e}   ----- max: {val_losses.max():.2e}, min: {val_losses.min():.2e}, std: {val_losses.std():.2e}")
            print(f"Base loss (val imgs): {base_val_losses.mean():.2e}   ----- max: {base_val_losses.max():.2e}, min: {base_val_losses.min():.2e}, std: {base_val_losses.std():.2e}")
            print(f"Mean train loss:      {train_losses.mean():.2e}   ----- max: {train_losses.max():.2e}, min: {train_losses.min():.2e}, std: {train_losses.std():.2e}")
            print(f"Evaluation complete after {time.time() - main_start:.2f} seconds")

    if test:
        test_dataset = FilteredHiRISEDataset(
            allowed_sets=test_sets, data_record=data_record, sweep=sweep, meta_cols=meta_cols,
            meta_scaler=shared_meta_scaler, **normalization_kwargs, data_root=data_root
        )
        test_dataloader = get_loader(
            test_dataset,
            batch_size=cfg['optim']['batch_size'],
            num_workers=n_workers,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn(cfg['seed']),
            seed=cfg['seed'],
        )

        test_losses, base_test_losses = evaluate(
            trained_model, test_dataloader, criterion, device, shared_meta_scaler,
            eval_dr, eval_cats, cfg, meta_cols, worker_init_fn=worker_init_fn, print_adv_metrics=True
        )

        wandb.run.summary["mean_test_mse"] = test_losses.mean()
        wandb.run.summary["base_test_mse"] = base_test_losses.mean()
        if not sweep:
            print(f"\nMean test MSE: {test_losses.mean():.2e}  ----- max: {test_losses.max():.2e}, min: {test_losses.min():.2e}, std: {test_losses.std():.2e}")
            print(f"Base test MSE: {base_test_losses.mean():.2e}   ----- max: {base_test_losses.max():.2e}, min: {base_test_losses.min():.2e}, std: {base_test_losses.std():.2e}")
            print(f"Evaluation complete after {time.time() - main_start:.2f} seconds")

    if save_model:
        print("Save model flag triggered", flush=True)
        if os.path.isdir(model_savedir) is False:
            os.makedirs(model_savedir)
            print(f"Model save dir not found. Created {model_savedir}")

        version = str(cfg['architecture']['version']).replace('.', '_')
        run_name = wandb.run.name
        run_date = "-".join(run_name.split("-")[:2])
        model_save_name = f"{model_name}_v{version}_{run_date}.pt"
        save_name = f"{model_savedir}/{model_save_name}"
        torch.save(trained_model.state_dict(), save_name)       # Save model weights
        print(f"Model saved to {save_name}")

        latest_dir = os.path.join(model_savedir, "latest")
        os.makedirs(latest_dir, exist_ok=True)
        for f in os.listdir(latest_dir):
            os.remove(os.path.join(latest_dir, f))
        torch.save(trained_model.state_dict(), os.path.join(latest_dir, f"{model_save_name}"))
        print(f"Model replaced latest in {latest_dir}")

        if shared_meta_scaler is not None:
            joblib.dump(shared_meta_scaler, os.path.join(latest_dir, "meta_scalar.joblib"))
            print(f"Meta scalar saved to {latest_dir}")

    if eval_mode and save_loss_map:
        maps_dir = cfg.get("save", {}).get("maps_dir") or str(files_dir() / "maps")
        os.makedirs(maps_dir, exist_ok=True)
        _ = create_loss_map(trained_model, criterion, val_dataloader, device, output_dir=maps_dir, cfg=cfg)
    sys.stdout.flush()  # Flush the output buffer to ensure all logs are written before wandb output


def set_global_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, default=1, help="Number of workers for data loader")
    parser.add_argument("--config", required=True, help="Path to the config file")
    parser.add_argument("overrides", nargs="*")      # optional k=v pairs
    parser.add_argument("--sweep", action="store_true", help="Run in sweep mode")
    return parser.parse_args()


def flatten_keys(d: dict, parent_key: str = "", sep: str = ".") -> list[str]:
    """
    Recursively flatten a nested dict structure into a list of dot-separated keys.
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_keys(v, new_key, sep=sep))
        else:
            items.append(new_key)
    return items


if __name__ == "__main__":
    args = parse_args()
    base_cfg = load_cfg(args.config, args.overrides)

    notrain = base_cfg.get('flags', {}).get('notrain', False)

    wandb_mode = base_cfg.get("wandb", {}).get("mode", "disabled") if not notrain else "disabled"
    if wandb_mode == "online":
        wandb.login()

    if notrain:
        base_cfg['save']['model'] = False    # don't save model if not training

    with wandb.init(
        project=base_cfg["wandb"]["project"],
        group=base_cfg["wandb"].get("group", None),
        config=base_cfg,
        mode=wandb_mode,
        resume='allow' if args.sweep else 'never',              # allow resuming runs in sweep mode, will raise if same id is used
    ) as run:

        cfg = wandb.config.as_dict()
        dr_path = str(resolve_path(cfg['data']['data_record'], cfg['data']['root']))
        eval_dr_path = str(resolve_path(cfg['evals']['eval_dr'], cfg['data']['root']))
        eval_cfg_path = str(resolve_path(cfg['evals']['eval_cfg'], cfg['data']['root']))

        set_global_seed(cfg["seed"])

        now = datetime.now()
        run_name = f"{cfg['architecture']['name']}{now.strftime('%H%M-%d%m')}"
        run.name = run_name
        save_loss_map = base_cfg.get('save', {}).get('loss_map', False)

        main(args.n_workers, cfg, args.sweep, dr_path, run_name,
             notrain=notrain, save_loss_map=save_loss_map,
             eval_mode=cfg.get('flags', {}).get('evaluate', False),
             test=cfg.get('flags', {}).get('test', False),
             eval_dr_path=eval_dr_path, eval_cfg_path=eval_cfg_path)
