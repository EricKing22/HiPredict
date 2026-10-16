# Preprocessing

`hipredict.process` (or `PrepObservations`) turns the raw HiRISE EDR files into radiometrically calibrated, stitched and registered ISIS cubes, one per detector. It runs [HiPROC](https://github.com/rbeyer/hiproc) and [USGS ISIS](https://isis.astrogeology.usgs.gov/) in their own conda environments, separate from the uv environment that runs HiPredict.

You only need this backend to process raw `.IMG` files or to export predictions as ISIS cubes. Inference on observations that already have processed cubes does not use it.

## Setting up the backend

1. Install [Miniforge](https://github.com/conda-forge/miniforge) or another conda distribution.
2. Create the ISIS environment and point it at an ISIS data area that contains the MRO mission data. HiPredict's training data were processed with ISIS 9.0.0. See the [ISIS installation guide](https://astrogeology.usgs.gov/docs/how-to-guides/environment-setup-and-maintenance/installing-isis-via-anaconda/).

   ```bash
   conda create -n isis -c usgs-astrogeology -c conda-forge isis=9.0.0 -y
   conda activate isis
   conda env config vars set ISISROOT=$CONDA_PREFIX ISISDATA=/path/to/isis_data
   conda deactivate
   ```

3. Create a separate HiPROC environment, which is stacked on top of ISIS at run time. HiPredict's training data were processed with the unmodified HiPROC 0.11.0, so install that version to get matching cubes:

   ```bash
   conda create -n hiproc -c conda-forge python=3.11 hiproc=0.11.0 kalasiris pvl numpy "setuptools<81" -y
   ```

4. If your conda installation or environment names differ from the defaults, tell HiPredict where they are:

   ```bash
   export HIPREDICT_CONDA_SH="$HOME/miniforge3/etc/profile.d/conda.sh"
   export HIPREDICT_ISIS_ENV="isis"
   export HIPREDICT_HIPROC_ENV="hiproc"
   ```

   By default HiPredict finds `conda.sh` from `CONDA_EXE`, `~/miniforge3` or `~/miniconda3`.

5. Check the setup:

   ```bash
   ./scripts/verify_hiproc_backend.sh
   ```

   This activates both environments, checks that every ISIS and HiPROC command the pipeline uses is available, and checks that `ISISROOT` and `ISISDATA` are set. It warns if the installed versions are not ISIS 9.0.0 and HiPROC 0.11.0.

## What the pipeline does

For each observation, [`hipredict/data/hiproc_pipe.sh`](../hipredict/data/hiproc_pipe.sh) runs:

1. `EDR_Stats` and `HiCal` on each of the ten channel files (two per detector).
2. `HiStitch` to join the two channels of each detector.
3. `HiccdStitch` across RED3, RED4 and RED5. IR10 and BG12 are single detectors in their colour groups and skip this step.
4. `HiColorInit`, `HiJitReg` and `HiSlither` to register the colour bands to the RED detectors.

The results are written as `<observation>_<detector>_processed.cub` in the observation's folder. Intermediate cubes for that observation are then deleted. Raw `.IMG` files, final cubes and any other files are kept.

Only observations whose five detectors use one of the supported binning/TDI combinations (binning 1 or 2, TDI 64 or 128), with matching binning across the RED detectors and across the colour detectors, are processed. Other observations are skipped and reported before processing starts.

## Logs and failures

Each step logs to `files/hiproc_logs/<observation>/`, or to `HIPREDICT_LOG_DIR` if set. Set `HIPROC_VERBOSE=1` for verbose HiPROC output.

When an observation fails in calibration or a later HiPROC step, it is reported and processing continues with the others. Pass `--fail-log path.txt` to record these failures; observations already in that file are skipped on later runs. Missing input files and full disks are reported but not recorded, so those observations are retried. `files/hiproc_fails.txt` lists observations that failed during the original processing; pass it with `--skip-list` to skip them. It is read-only and must not be used as the `--fail-log`.

## Reusing processed cubes

Observations that already have all five processed cubes are not reprocessed, and if every requested observation has them, HiPROC is not activated at all. Such observations are accepted even if their raw `.IMG` files have been deleted. Pass `force_cub_gen=True` to `PrepObservations` to regenerate the cubes.
