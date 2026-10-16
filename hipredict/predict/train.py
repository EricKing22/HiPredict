from copy import deepcopy
import torch
import time
import wandb
import warnings
import numpy as np
import torch.nn as nn

from hipredict.predict.evaluate import validate
from hipredict.predict.utils import parse_early_stopping

warnings.filterwarnings("ignore", message="Can't initialize NVML")


def train(
    model,
    criterion,
    optimiser,
    train_dataloader,
    val_dataloader,
    device,
    epochs,
    scheduler=None,
    SEED=42,
    early_stopping=False,
):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    L1 = nn.L1Loss(reduction="mean")
    MSE = nn.MSELoss(reduction="mean")

    model.to(device)
    if wandb.run is not None and not getattr(wandb.run, "disabled", False):
        wandb.watch(model, criterion, log="all")

    L1_losses, MSE_losses, criterion_losses = [], [], []
    epoch_step_losses = []
    bias_num = 0.0
    bias_den = 0
    m = model.module if hasattr(model, "module") else model

    epoch_times = []
    step_count = 0
    outputs = torch.tensor([])
    early_stopping_cfg = parse_early_stopping(early_stopping)
    best_val_loss = float("inf")
    best_epoch = 0
    best_model_state = None
    epochs_without_improvement = 0
    y = torch.tensor([])
    print("Started training loop")
    print(f"Num epochs: {epochs}")
    print(f"Num train batches: {len(train_dataloader)}", flush=True)
    print(f"Device: {device}", flush=True)
    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_start = time.time()
        data_timer = time.time()
        model.train()

        has_brightness_params = hasattr(m, "mean_head") or hasattr(m, "global_bias")
        if has_brightness_params:
            if epoch == 0:
                # freeze everything
                for p in m.parameters():
                    p.requires_grad = False

                # unfreeze the brightness mechanism and global bias
                if hasattr(m, "mean_head"):
                    for p in m.mean_head.parameters():
                        p.requires_grad = True
                if hasattr(m, "global_bias"):
                    m.global_bias.requires_grad = True

            if epoch == 2:
                for p in m.parameters():
                    p.requires_grad = True

        for batch in train_dataloader:
            load_time = time.time() - data_timer
            comp_start = time.time()

            optimiser.zero_grad()
            x_band = batch["x_band"].to(device).float()
            x_neigh = batch["x_neigh"].to(device).float()
            x_meta = batch["x_meta"].to(device).float()
            y = batch["y"].to(device).float()
            patch_x = batch["patch_x"].to(device).float()
            obs_width = batch["obs_width"].to(device).float()
            if step_count == 0 and epoch == 0:
                print("Shapes:")
                print(f"\tx_band: {x_band.shape}")
                print(f"\tx_neigh: {x_neigh.shape}")
                print(f"\tx_meta: {x_meta.shape}")
                print(f"\ty: {y.shape}")
            outputs = model(x_band, x_neigh, x_meta, patch_x=patch_x, obs_width=obs_width)
            loss, shift = criterion(outputs, y)


            L1_losses.append(L1(outputs, y).item())
            MSE_losses.append(MSE(outputs, y).item())
            criterion_losses.append(loss.item())

            epoch_step_losses.append(loss.item())
            diff = (outputs.detach() - y.detach())
            bias_num += diff.double().sum().item()
            bias_den += diff.numel()

            loss.backward()  # Calculate gradients

            optimiser.step()  # Update weights
            epoch_loss += loss.item()
            step_count += 1

            comp_time = time.time() - comp_start
            wandb.log(
                {
                    "step_count": step_count,
                    "data_load_time": load_time,
                    "comp_time": comp_time,
                    "loss": loss.item(),
                }
            )
            data_timer = time.time()

            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = optimiser.param_groups[0]["lr"]

        grad_norm = (
            sum(
                p.grad.norm(2).item() ** 2
                for p in model.parameters()
                if p.grad is not None
            ) ** 0.5
        )  # Gradient norm to monitor exploding/vanishing gradients

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        val_loss, val_L1 = validate(model, criterion, val_dataloader, device, L1)

        epoch_L1 = L1(outputs, y).item()

        mean_dff = torch.mean(outputs - y)

        train_loss_p50 = float(np.median(epoch_step_losses))
        train_loss_p90 = float(np.percentile(epoch_step_losses, 90))
        epoch_mean_bias = bias_num / max(1, bias_den)

        wandb.log(
            {
                "epoch": epoch + 1,
                "epoch loss": avg_epoch_loss,
                "epoch_time": epoch_time,
                "val_loss": val_loss,
                "grad_norm": grad_norm,
                "lr": current_lr,
                "epoch_L1": epoch_L1,
                "val_L1": val_L1,
                "mean_dff": mean_dff.item(),
                "train_loss_p50": train_loss_p50,          # median step loss (bulk learning)
                "train_loss_p90": train_loss_p90,          # tail loss (hard cases)
                "epoch_mean_bias": epoch_mean_bias,        # mean offset y_pred - y
            },
        )
        if epoch % 5 == 0 or epoch == (epochs - 1):
            print(
                f"| Epoch {epoch + 1:03d} | Train loss: {loss.item():.6f} | Val loss: {val_loss:.6f} | Epoch time: {epoch_time:.2f} seconds |",
                flush=True,
            )

        if early_stopping_cfg is not None:
            min_delta = early_stopping_cfg["min_delta"]
            improved = val_loss < best_val_loss - min_delta

            if improved:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                epochs_without_improvement = 0
                if early_stopping_cfg["restore_best"]:
                    best_model_state = deepcopy(model.state_dict())
            else:
                epochs_without_improvement += 1

            should_stop = (
                epoch + 1 >= early_stopping_cfg["min_epochs"]
                and epochs_without_improvement >= early_stopping_cfg["patience"]
            )
            if should_stop:
                print(
                    f"EARLY STOPPING: Epoch {epoch + 1}, "
                    f"best epoch: {best_epoch}, "
                    f"best val loss: {best_val_loss:.6f}, "
                    f"current val loss: {val_loss:.6f}",
                    flush=True,
                )
                break

    if early_stopping_cfg is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
        wandb.run.summary["early_stopping_best_epoch"] = best_epoch
        wandb.run.summary["early_stopping_best_val_loss"] = best_val_loss

    avg_epoch_time = np.mean(epoch_times)
    wandb.run.summary["avg_epoch_time"] = avg_epoch_time

    losses_dict = {
        "L1": L1_losses,
        "MSE": MSE_losses,
        "Criterion": criterion_losses,
    }

    return model, losses_dict
