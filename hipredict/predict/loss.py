import torch.nn as nn
import torch
import torch.nn.functional as F


def gaussian_blur(x, k=21, sigma=6.0):
    coords = torch.arange(k, device=x.device).float() - (k - 1) / 2
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    g = g / g.sum()
    w_w = g.view(1, 1, 1, k).expand(x.size(1), 1, 1, k)
    w_h = g.view(1, 1, k, 1).expand(x.size(1), 1, k, 1)
    x = F.pad(x, (k // 2, k // 2, 0, 0), mode='reflect')
    x = F.conv2d(x, w_w, groups=x.size(1))
    x = F.pad(x, (0, 0, k // 2, k // 2), mode='reflect')
    x = F.conv2d(x, w_h, groups=x.size(1))
    return x


class HiRISELoss(nn.Module):
    def __init__(self, loss_config):
        """
        Dynamic loss
        Args:
        loss_config (dict): Dictionary containing the loss configuration where keys are loss names and values are weights
        eg:
        {
            "mse": 0.5,
            "mae": 0.5,
            "rmse": 0.0
        }
        """
        super().__init__()

        allowed = {"mse", "mae", "rmse", "ssim", "gdl", "mean_mae", "lf_mse", "local_mae", "local_mse", "std_mae", "lf_l1"}
        unknown = set(loss_config) - allowed
        if unknown:
            raise ValueError(f"Only {allowed} losses are implemented, got {unknown}")

        self.weights = loss_config
        self.fns = nn.ModuleDict()

        if loss_config.get("mse", 0):
            self.fns["mse"] = nn.MSELoss()
        if loss_config.get("mae", 0):
            self.fns["mae"] = nn.L1Loss()
        if loss_config.get("ssim", 0):
            import piq
            self.fns["ssim"] = piq.SSIMLoss(data_range=1., reduction="mean")    # Assuming data range is [0, 1], wraps (1-SSIM) such that lower is better here
        if loss_config.get("gdl", 0):
            self.fns["gdl"] = GradientDifferenceLoss(reduction="mean")

        # Assert sum of weights is 1
        total_weight = sum(self.weights.values())
        assert abs(total_weight - 1.0) < 1e-6, f"Sum of weights must be 1, got {total_weight}."

    @staticmethod
    def _per_image_mean(x):
        return x.mean(dim=(2, 3), keepdim=True)

    @staticmethod
    def _per_image_std(x, eps=1e-8):
        m = x.mean(dim=(2, 3), keepdim=True)
        v = ((x - m) ** 2).mean(dim=(2, 3), keepdim=True)
        return (v + eps).sqrt()

    def forward(self, y_pred, y_true):
        total = torch.tensor(0.0, device=y_pred.device)
        details = {}

        if y_pred.dim() == 2:
            y_pred = y_pred.unsqueeze(0).unsqueeze(0)
        elif y_pred.dim() == 3:
            y_pred = y_pred.unsqueeze(0)
        if y_true.dim() == 2:
            y_true = y_true.unsqueeze(0).unsqueeze(0)
        elif y_true.dim() == 3:
            y_true = y_true.unsqueeze(0)

        mu_p = self._per_image_mean(y_pred)
        mu_t = self._per_image_mean(y_true)
        std_p = self._per_image_std(y_pred)
        std_t = self._per_image_std(y_true)

        y_pred_local = y_pred - mu_p
        y_true_local = y_true - mu_t

        # prepare low-frequency tensors if any lf_* loss is active
        use_lf = (self.weights.get("lf_l1", 0) > 0) or (self.weights.get("lf_mse", 0) > 0)
        if use_lf:
            s = 2
            low_pred = gaussian_blur(F.avg_pool2d(y_pred_local, kernel_size=s, stride=s), k=15, sigma=5.0)
            with torch.no_grad():
                low_true = gaussian_blur(F.avg_pool2d(y_true_local, kernel_size=s, stride=s), k=15, sigma=5.0)
            H, W = y_pred.shape[-2:]
            h, w = low_pred.shape[-2:]

        for name, w in self.weights.items():
            if w == 0: continue   # noqa

            if name == "rmse":
                val = torch.sqrt(F.mse_loss(y_pred, y_true) + 1e-8)
            elif name == "ssim":
                val = self.fns[name](y_pred.clamp(0, 1), y_true)    # SSIM expects inputs in [0, 1]
            elif name == "mse":
                val = self.fns[name](y_pred, y_true)
            elif name == "mae":
                val = self.fns[name](y_pred, y_true)
            elif name == "local_mse":
                val = F.mse_loss(y_pred_local, y_true_local)
            elif name == "local_mae":
                val = F.l1_loss(y_pred_local, y_true_local)
            elif name == "mean_mae":
                val = F.l1_loss(mu_p, mu_t)
            elif name == "std_mae":
                val = F.l1_loss(std_p, std_t)
            elif name == "gdl":
                val = self.fns[name](y_pred, y_true)
            elif name == "lf_mse":
                val = F.mse_loss(low_pred, low_true)
            elif name == "lf_l1":
                val = F.l1_loss(low_pred, low_true)
            else:
                raise ValueError(f"Unknown loss key: {name}")

            total += w * val
            details[name] = val.detach().item()

        details["total"] = total.detach().item()

        return total, details


class GradientDifferenceLoss(nn.Module):
    """
    Gradient-Difference Loss (GDL)

    L = L1(∂x) + L1(∂y)
    """
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Horizontal gradients  (width direction)
        grad_pred_x = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])
        grad_true_x = torch.abs(y_true[:, :, :, 1:] - y_true[:, :, :, :-1])

        # Vertical gradients    (height direction)
        grad_pred_y = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
        grad_true_y = torch.abs(y_true[:, :, 1:, :] - y_true[:, :, :-1, :])

        diff_x = torch.abs(grad_true_x - grad_pred_x)
        diff_y = torch.abs(grad_true_y - grad_pred_y)

        if self.reduction == "mean":
            return diff_x.mean() + diff_y.mean()
        elif self.reduction == "sum":
            return diff_x.sum() + diff_y.sum()
        else:  # "none"  → return per-sample tensor shape (B,)
            # average each diff over (C,H,W') respectively, keep batch dim
            loss_x = diff_x.view(diff_x.size(0), -1).mean(dim=1)
            loss_y = diff_y.view(diff_y.size(0), -1).mean(dim=1)
            return loss_x + loss_y


class ScalarLoss(nn.Module):
    """Give standard PyTorch losses the same result interface as HiRISELoss."""
    def __init__(self, loss, name):
        super().__init__()
        self.loss, self.name = loss, name

    def forward(self, prediction, target):
        value = self.loss(prediction, target)
        return value, {self.name: value.detach()}


def get_criterion(cfg):
    if cfg['loss']['name'] == 'HiRISELoss':
        return HiRISELoss(loss_config=cfg['loss']['config'])
    elif cfg['loss']['name'] == 'MSELoss':
        return ScalarLoss(torch.nn.MSELoss(), "mse")
    elif cfg['loss']['name'] == 'L1Loss':
        return ScalarLoss(torch.nn.L1Loss(), "mae")
    else:
        raise ValueError(f"Unknown loss function: {cfg['loss']['name']}")
