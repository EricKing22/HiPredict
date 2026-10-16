"""Public model loading for HiPredict and SYN4."""

from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import torch

from hipredict.data.normalization import load_global_normalization_stats, _resolve_global_stats_path
from hipredict.paths import CONFIGS_DIR, models_dir
from hipredict.predict.hub import download_model, validate_release
from hipredict.predict.models import SYN4, Linear, LinearWithMeta, BasicUNet, UNetCNN_local, UNetCNN_conv
from hipredict.predict.utils import load_cfg, load_latest_into, get_model_details


def get_model(
    model_name: str = "hipredict",
    device: Union[str, torch.device] = "cpu",
    model_save_dir: Optional[str] = None,
    trained: bool = True,
    cfg: Optional[dict] = None,
    hf_repo_id: Optional[str] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
):
    """Load HiPredict from Hugging Face or a local release directory.

    ``unet_conv_b22`` remains an alias for HiPredict. SYN4 requires no weights
    or metadata. Other registered architectures can be trained from scratch,
    or loaded locally with an explicit configuration. Baseline configurations
    and pretrained baseline weights are not distributed.
    """
    name = "unet_conv_b22" if model_name.lower() in {"hipredict", "unet_conv_b22"} else model_name
    if name.lower() == "syn4":
        name = "SYN4"
    registry = {
        "linear": Linear, "linear_with_meta": LinearWithMeta,
        "basic_unet": BasicUNet, "unet_local": UNetCNN_local,
        "unet_conv": UNetCNN_conv, "unet_conv_b22": UNetCNN_conv,
    }
    if name not in registry and name != "SYN4":
        raise ValueError(f"Unknown model {model_name!r}. Public models: hipredict, SYN4.")
    if name not in {"unet_conv_b22", "SYN4"} and cfg is None:
        raise ValueError(f"{name} has no released configuration. Supply an explicit cfg for local training/loading.")
    device = torch.device(device)
    directory = None
    if trained and name != "SYN4":
        if model_save_dir is not None:
            directory = Path(model_save_dir).expanduser().resolve()
        elif name == "unet_conv_b22":
            directory = download_model(hf_repo_id, revision, local_files_only)
        else:
            directory = models_dir() / name / "latest"
        if not directory.is_dir():
            raise FileNotFoundError(f"Model directory not found: {directory}. Only HiPredict pretrained weights are released.")

    if cfg is None:
        config_path = directory / "config.yaml" if directory else None
        cfg = load_cfg(config_path if config_path and config_path.is_file() else CONFIGS_DIR / f"{name if name != 'SYN4' else 'unet_conv_b22'}.yaml")
    else:
        cfg = deepcopy(cfg)

    if directory and (directory / "model.pt").is_file():
        validate_release(directory)
        cfg["data"]["global_stats"] = str(directory / "global_stats.json")

    if name == "SYN4":
        center, scale = load_global_normalization_stats(_resolve_global_stats_path(cfg))
        cfg.setdefault("data", {}).update(normalisation_mode="global_robust", meta_cols=[])
        cfg["architecture"] = {"name": "SYN4", "args": {}}
        cfg.pop("shared_meta_scalar", None)
        return SYN4(center=center, scale=scale).to(device).eval(), cfg

    if cfg["architecture"]["name"] != name:
        raise ValueError(f"Config architecture {cfg['architecture']['name']!r} does not match model {name!r}")
    args = dict(cfg["architecture"].get("args") or {})
    args.setdefault("n_meta", len(cfg["data"].get("meta_cols", [])))
    args.setdefault("in_ch", 8 + args["n_meta"])
    model = registry[name](**args)
    if trained:
        checkpoint = directory / "model.pt"
        if not checkpoint.is_file():
            filename = get_model_details(name, str(directory))
            if filename is None:
                raise FileNotFoundError(f"No {name} checkpoint found in {directory}")
            checkpoint = directory / filename
        model = load_latest_into(model, name, str(checkpoint), strict=True)
        scaler = directory / "meta_scalar.joblib"
        if cfg["data"].get("meta_cols"):
            if not scaler.is_file():
                raise FileNotFoundError(f"Metadata scaler required by {name}: {scaler}")
            cfg["shared_meta_scalar"] = str(scaler)
    print(f"{model_name} loaded with {sum(p.numel() for p in model.parameters()):,} parameters.")
    return model.to(device), cfg
