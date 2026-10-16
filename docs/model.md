# Model weights

HiPredict's trained weights are not stored in this repository. They are published on Hugging Face at [`ardan-S/HiPredict`](https://huggingface.co/ardan-S/HiPredict), and `get_model("hipredict")` downloads them on first use. SYN4 needs no weights.

## Release contents

| File | Contents |
| --- | --- |
| `model.pt` | PyTorch state dict of the HiPredict U-Net |
| `meta_scalar.joblib` | The fitted scaler for the nine metadata inputs |
| `config.yaml` | Architecture, metadata column order and normalisation settings |
| `global_stats.json` | The fixed normalisation statistics the model was trained with |
| `provenance.json` | Source identity and SHA-256 checksums of the files above |

The four model files are used together. `global_stats.json` is identical to `files/global_stats.json` in this repository.

## Choosing the release

By default, `get_model("hipredict")` downloads the release pinned in `hipredict/predict/hub.py`: the published repository at the commit this version of the code was validated against. To load a different release, set the repository and revision with:

1. the `hf_repo_id` and `revision` arguments, or
2. the `HIPREDICT_HF_REPO` and `HIPREDICT_HF_REVISION` environment variables.

Arguments take precedence over environment variables. When you name another repository without a revision, its latest version is used.

```python
from hipredict import get_model

model, config = get_model("hipredict", hf_repo_id="ardan-S/HiPredict", revision="main", device="cuda")
```

Files are cached by Hugging Face (set `HF_HOME` to move the cache). HiPredict never modifies cached files. `unet_conv_b22`, the model's research name, is accepted as an alias for `hipredict`.

## Checks on load

Before returning the model, `get_model` checks that all four model files are present. If `provenance.json` is present, it also verifies every checksum, so a mixed or corrupted set of files fails with an error. It then loads the weights strictly, so any mismatch between the config and the checkpoint is an error, and it fails if the metadata scaler is missing.

## Offline use and local copies

- `get_model("hipredict", local_files_only=True)` uses a release already in the Hugging Face cache, without network access.
- `get_model("hipredict", model_save_dir="/path/to/release")` loads a downloaded or copied release directory.

## Models you train yourself

Training writes checkpoints to `models/<architecture>/` (or `HIPREDICT_MODELS_DIR`), with the newest copy and its metadata scaler in `latest/`. Load one with:

```python
model, config = get_model("unet_conv_b22", model_save_dir="models/unet_conv_b22/latest")
```

If you changed the architecture or metadata columns, also pass the matching config as `cfg=`. The other architectures in `hipredict/predict/models.py` (`linear`, `linear_with_meta`, `basic_unet`, `unet_local`, `unet_conv`) can be trained and loaded with an explicit `cfg`; no configs or pretrained weights are provided for them.
