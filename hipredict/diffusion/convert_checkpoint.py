"""Convert irp-ed425 checkpoints to the hipredict.diffusion checkpoint format.

irp-ed425 checkpoints pickle their configs as classes from a top-level
``config`` module, so they are read here with a remapping unpickler. Only
convert files from a trusted source: reading them executes pickled code.

    python -m hipredict.diffusion.convert_checkpoint OLD.pt NEW.pt [--weights_only]
"""

import argparse
import logging
import pickle
import types
from pathlib import Path

import torch

from hipredict.diffusion.config import FMConfig, DDPMConfig, TrainingConfig
from hipredict.diffusion.utils import checkpoint_dict

logger = logging.getLogger(__name__)

# irp-ed425 settings that no longer exist. Model fields must hold the listed
# value; non-zero retired loss weights change what training optimises.
RETIRED_MODEL_FIELDS = {"normalize_method": "global"}
RETIRED_LOSS_WEIGHTS = (
    "lambda_mean",
    "lambda_lf",
    "lambda_local",
    "lambda_block",
    "lambda_edge_x1",
    "lambda_ms_edge_x1",
)


class _LegacyUnpickler(pickle.Unpickler):
    """Map irp-ed425's top-level ``config`` module onto hipredict.diffusion.config."""

    def find_class(self, module, name):
        if module == "config":
            module = "hipredict.diffusion.config"
        return super().find_class(module, name)


# torch.load takes a pickle-like module and subclasses its Unpickler.
_legacy_pickle = types.SimpleNamespace(
    load=pickle.load, Unpickler=_LegacyUnpickler, __name__="pickle"
)


def _stored_fields(cfg) -> dict:
    """The field values a pickled config object was saved with."""
    return dict(cfg.__dict__) if hasattr(cfg, "__dict__") else dict(cfg)


def _convert_model_config(stored: dict):
    for field, required in RETIRED_MODEL_FIELDS.items():
        value = stored.pop(field, required)
        if value != required:
            raise ValueError(f"Unsupported checkpoint: {field}={value!r}")
    model_type = stored.get("type")
    if model_type not in (
        "UnidirectionalDDPM",
        "BidirectionalDDPM",
        "UnidirectionalFM",
        "BidirectionalFM",
    ):
        raise ValueError(f"Unsupported model type: {model_type!r}")
    cls = DDPMConfig if "DDPM" in model_type else FMConfig
    return cls(**{k: v for k, v in stored.items() if k in cls.model_fields})


def _convert_training_config(stored: dict, weights_only: bool):
    retired = {k: stored.pop(k) for k in RETIRED_LOSS_WEIGHTS if k in stored}
    active = {k: v for k, v in retired.items() if v}
    if active and not weights_only:
        raise ValueError(
            f"The checkpoint was trained with retired loss terms {active}, so its "
            "training state cannot be resumed; convert it with --weights_only"
        )
    unknown = sorted(k for k in stored if k not in TrainingConfig.model_fields)
    if unknown:
        logger.warning("Dropping unknown training settings: %s", ", ".join(unknown))
    return TrainingConfig(
        **{k: v for k, v in stored.items() if k in TrainingConfig.model_fields}
    )


def convert_checkpoint(source, destination, weights_only: bool = False) -> None:
    """Rewrite an irp-ed425 checkpoint in the hipredict.diffusion format.

    With ``weights_only``, only the inference weights (EMA when available) and
    configs are kept, so the result cannot resume training.
    """
    legacy = torch.load(
        source, map_location="cpu", weights_only=False, pickle_module=_legacy_pickle
    )
    if legacy.get("format") is not None:
        raise ValueError(f"{source} is already in the {legacy['format']} format")

    cfg_model = _convert_model_config(_stored_fields(legacy["cfg_model"]))
    cfg_train = _convert_training_config(
        _stored_fields(legacy["cfg_train"]), weights_only
    )

    if weights_only:
        state = legacy.get("ema_model") or legacy["model"]
        converted = checkpoint_dict(legacy["step"], state, cfg_model, cfg_train)
    else:
        converted = checkpoint_dict(
            legacy["step"],
            legacy["model"],
            cfg_model,
            cfg_train,
            ema_state=legacy.get("ema_model"),
            optimizer_state=legacy["optimizer"],
            scheduler_state=legacy["scheduler"],
        )

    Path(destination).parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, destination)
    logger.info("Wrote %s (%s, step %d)", destination, cfg_model.type, legacy["step"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("source", help="irp-ed425 checkpoint (.pt)")
    parser.add_argument("destination", help="Output checkpoint path")
    parser.add_argument(
        "--weights_only",
        action="store_true",
        help="Keep only the inference weights and configs (cannot resume training)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if Path(args.source).resolve() == Path(args.destination).resolve():
        parser.error("destination must differ from source")
    convert_checkpoint(args.source, args.destination, weights_only=args.weights_only)


if __name__ == "__main__":
    main()
