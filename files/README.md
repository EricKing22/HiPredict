# files/

Default location for downloaded observations, processed cubes, patches, predictions, logs and other runtime data. Generated content is ignored by Git. Set `HIPREDICT_FILES_DIR` to keep it somewhere else.

The tracked files describe the released model and dataset:

| File | Contents |
| --- | --- |
| `global_stats.json` | The fixed normalisation statistics HiPredict was trained with. A copy is also packaged in `hipredict/configs/` for installed use. |
| `all_obs.txt` | Observations that passed the HiPredict selection filters and have all five detectors (BG12, IR10, RED3, RED4, RED5). |
| `selected_obs_bin1.txt`, `selected_obs_bin2.txt`, `selected_obs_bin12.txt` | The observations from `all_obs.txt` grouped by binning: all five detectors at binning 1; all at binning 2; or RED detectors at binning 1 with IR10 and BG12 at binning 2. |
| `eval_observations.txt` | Observations used for the evaluation categories in `hipredict/configs/evals.yaml`. |
| `hiproc_fails.txt` | Observations that failed preprocessing when the dataset was built. Pass it with `--skip-list` to skip them. |

Leave these files unchanged when using the released model. Use `hipredict.data.compute_global_stats` and `hipredict.data.img_selection` to generate new versions; they write to `files/generated/` (see [Workflows](../docs/workflows.md)).
