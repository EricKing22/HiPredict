# Workflows

## Where files go

Commands read and write under `files/` in the repository by default. Set `HIPREDICT_FILES_DIR` to keep data elsewhere, and `HIPREDICT_MODELS_DIR` for models you train yourself. Hugging Face downloads go to its own cache (see `HF_HOME`).

| Location | Contents |
| --- | --- |
| `files/isis_imgs/observations/<observation>/` | Raw `.IMG` files and processed cubes |
| `files/metadata/EDR_df.csv` | Cached HiRISE EDR index, built on first use |
| `files/predictions/` | Output of `hipredict.inference.predict` |
| `files/hiproc_logs/` | Preprocessing logs |
| `files/data_record_*.csv`, `files/npy_files*/` | The HiPredict dataset, once downloaded |
| `files/generated/` | Recomputed statistics and observation lists |
| `models/` | Checkpoints from your own training runs |

The EDR index is downloaded from the HiRISE PDS the first time you download, preprocess or run the model with metadata. Later runs reuse the cached copy.

## The HiPredict dataset

The patches and records HiPredict was trained and evaluated on are published on Hugging Face at [`ardan-S/HiPredict`](https://huggingface.co/datasets/ardan-S/HiPredict), as about 220 GB of uncompressed tar shards plus two records:

| File | Contents |
| --- | --- |
| `data_record_alls.csv` | The patch sets used for training, validation and testing. The training config splits them by date. |
| `data_record_evals.csv` | The evaluation patch sets for the per-category metrics in `hipredict/configs/evals.yaml` |
| `npy_files/<observation>/` | Arrays for `data_record_alls.csv` |
| `npy_files_evals/<observation>/` | Arrays for `data_record_evals.csv` |
| `dataset_manifest.json` | Shard checksums and record provenance |

Download it into `files/` and unpack the shards there:

```bash
uv run hf download ardan-S/HiPredict --repo-type dataset --local-dir files \
  --exclude README.md --exclude .gitattributes
for shard in files/shards/*/*.tar; do tar -xf "$shard" -C files; done
```

Excluding `README.md` keeps the shipped `files/README.md`; the dataset's own description is on its Hugging Face page. Unpacking needs as much space again as the download. Once it has finished you can delete `files/shards/`, but running the download again will fetch them again.

Paths in the records are relative to the CSV files, so `files/data_record_alls.csv` finds its arrays in `files/npy_files/`. Training reads `files/data_record_alls.csv` by default, and `run_evals` reads both records from `files/`. To keep the dataset on another disk, download and unpack it into that directory instead and set `HIPREDICT_FILES_DIR` to it. `run_evals` then finds the records there. For training, also pass `data.data_record=` and `evals.eval_dr=` with the full paths to the two CSVs, because config paths are resolved against `data.root` (the repository), not `HIPREDICT_FILES_DIR`.

## Generating patches

The default preprocessing mode produces processed cubes only. To also cut training/evaluation patches and write a patch record (a CSV manifest):

```bash
uv run -m hipredict.process \
  --obs ESP_028693_1535 ESP_012940_1655 \
  --mode patches --patch-sets 16 \
  --np-dir files/npy_files_demo --record files/data_record_demo.csv --workers 2
```

The Python equivalent is `PrepObservations(..., prep_mode="patches", n_patch_sets=16, np_base_dir=..., data_record=...)`.

The requested number of patch sets is spread across the observations by sampling over location, orbit, solar longitude and binning/TDI. Each set holds BG12, IR10 and RED4 patches plus RED3 and RED5 context arrays. The record stores, for every array, its path, dimensions, observation metadata, patch position, and observation-level centre and scale. Existing cubes and patches are reused unless you pass `force_cub_gen=True` or `force_npy_gen=True`. The record is rewritten on each run, so use a new record file for each experiment.

### Record paths

The `Path` column in a record is relative to the CSV file's folder (absolute paths also work), so a record can be moved together with its arrays. To resolve relative paths against a different folder, set `data.patch_root` in the training config. A DataFrame passed directly to the API resolves relative paths against the working directory, unless `df.attrs["record_dir"]` is set.

### Patch inference

```python
from hipredict import get_model, HiPredict

model, config = get_model("hipredict")
predictor = HiPredict(model, config)
samples = predictor.load("files/data_record_demo.csv", mode="patches")
outputs = predictor.predict(samples)
syn4_outputs = predictor.predict(samples, syn4=True)
```

### Normalisation

HiPredict uses fixed `global_robust` normalisation: every input is scaled with the dataset-wide centre and scale in `files/global_stats.json`, and predictions are scaled back the same way. This is also the default for training and for the dataset classes. The observation-level centre and scale in patch records are used only if you explicitly choose `scene_robust` normalisation for a model of your own. SYN4 requires `global_robust` normalisation, because it forms its brightness ratio in physical units.

## Training

Install the `train` and `postprocess` extras. The released configuration is `hipredict/configs/unet_conv_b22.yaml`: the HiPredict architecture, loss weights and optimiser, with patch sets split chronologically into 70% training, 10% validation and 20% test. Paths in the config are relative to `data.root`, which defaults to the repository root. Any config value can be overridden on the command line as `key=value`.

A one-epoch check with the demo patches from above:

```bash
uv run --extra train --extra postprocess -m hipredict.predict.train_model \
  --config hipredict/configs/unet_conv_b22.yaml --n_workers 0 \
  data.data_record=files/data_record_demo.csv \
  optim.epochs=1 optim.batch_size=2 \
  save.model_dir=models/demo flags.evaluate=false
```

A small run like this only checks that training works. To train a useful model, use [the HiPredict dataset](#the-hipredict-dataset) or a large set of your own patches, and keep the shipped `files/global_stats.json` unless you deliberately recompute it. The metadata scaler is fitted on the training split and reused for validation and test. Checkpoints and the scaler are written to `<save.model_dir>/<architecture>/`, with the newest copy in `latest/`. Weights & Biases logging is off by default; set `wandb.mode=online` to enable it.

To evaluate after training against the evaluation record and the categories in `hipredict/configs/evals.yaml`:

```bash
uv run --extra train --extra postprocess -m hipredict.predict.train_model \
  --config hipredict/configs/unet_conv_b22.yaml \
  data.data_record=files/data_record_demo.csv \
  evals.eval_dr=files/data_record_evals.csv \
  flags.evaluate=true flags.test=true save.model_dir=models/demo
```

## Evaluation

These scripts compare HiPredict with SYN4. They need the `train` and `postprocess` extras and the model weights.

- **`python -m hipredict.post.run_evals`**: the full patch-level evaluation. It reports per-image and per-category metrics (NMAE, bias, SSIM, the fraction of pixels within tolerance), metadata correlations and SHAP attributions. It reads `files/data_record_alls.csv`, `files/data_record_evals.csv` and `hipredict/configs/evals.yaml`, and writes to `files/evals/`. The two records are part of [the HiPredict dataset](#the-hipredict-dataset).
- **`python -m hipredict.inference.predict_strip`**: batch full-strip prediction and export for the observations listed in `hipredict/configs/postprocess.yaml`, which also sets the formats to save. Output goes to `files/figs/`.
- **`python -m hipredict.inference.predict_domain`**: predictions on patches centred on the locations in `hipredict/configs/*_coords.csv` (icy craters, ridges and chloride deposits). It saves image, difference, feature and spectral-profile plots plus metrics to `files/figs/domain_evals/`. Observations that are not yet processed need the preprocessing backend.
- **`python -m hipredict.inference.predict_point`**: one 512×512 crop of one observation, set by the constants at the top of the file.
- **`python -m hipredict.post.create_maps`**: global maps of observation density, incidence angle, spacecraft altitude, TDI and HiPredict error, from the patch record and the `run_evals` output. Output goes to `files/maps/`.

## Statistics and observation lists

`files/global_stats.json` and the observation lists in `files/` describe the released model and dataset. Leave them unchanged. The helpers that regenerate them write to `files/generated/` instead, and refuse to overwrite the shipped copies:

```bash
uv run -m hipredict.data.compute_global_stats \
  --img-dir files/isis_imgs/observations --record files/data_record_demo.csv \
  --output files/generated/statistics/global_stats.json --workers 2
uv run -m hipredict.data.img_selection --output-dir files/generated/selection
```

Without `--record`, statistics are computed over every processed observation in `--img-dir`. Recomputed statistics only apply to models you train with them. `img_selection` applies the HiPredict selection filters to the current EDR index. That index grows as new observations are released, so its output can differ from the shipped lists.

To skip the observations that failed during the original processing, and record any new failures separately:

```bash
uv run -m hipredict.process --obs ESP_028693_1535 ESP_012940_1655 \
  --skip-list files/hiproc_fails.txt \
  --fail-log files/generated/preprocessing/failures.txt
```

## Exports

- **NumPy and TIFF** files are float32 arrays with the same shape as the processed cube, with NaN outside the valid crop. TIFFs are not georeferenced.
- **ISIS cubes** (`--format cub`) take their geometry from the observation's RED5 cube, which only approximates RED4's geometry. They are not a map-projected or geometrically validated product.

## Storage

Raw download size depends on the observation's length and binning. For each observation, allow for ten raw channel files, five processed cubes, temporary cubes during processing, and any patches. A float32 array takes `4 × height × width` bytes. A patch set (three 256×256 patches and two 1024×32×4 context arrays) takes about 1.75 MiB. Process one observation first to gauge the storage you need.
