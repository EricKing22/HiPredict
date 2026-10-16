import numpy as np
import torch
import pandas as pd
import torch.nn.functional as F
import joblib
from pathlib import Path
from typing import Optional, Union

from hipredict.data.sample_build import (
    assemble_strip_inputs,
    normalizer_from_mode,
)
from hipredict.data.normalization import (
    DEFAULT_NORMALIZATION_MODE,
    normalization_kwargs_from_config,
)
from hipredict.predict.dataset import (
    PatchInferenceDataset,
    get_loader,
    identity_collate,
)
from hipredict.data.isis2np import isis2np, crop_offset
from hipredict.data.load_metadata import gen_edr_df
from hipredict.utils.utils import check_obs_list


class HiPredict():
    def __init__(self, model, config, norm_mode: Optional[str] = None, global_center=None, global_scale=None,
                 device: Union[str, torch.device] = 'cpu', max_img_size=2**14, max_px_per_batch=6_000_000,
                 smooth_norm_kernel=257, verbose=True, normalizer=None):
        """
        Initialize the HiPredict inference engine.

        Normalization defaults to the configured mode, or global_robust with
        recorded statistics when the mode is omitted. Scene modes are explicit
        options for separately trained models.

        Args:
            model (nn.Module): The trained PyTorch model.
            config (dict): Configuration dictionary containing model and data parameters.
            norm_mode (str, optional): Normalization strategy, globally robust by default.
                - 'scene_robust': Normalizes based on current image stats.
                - 'scene_smooth': Normalizes with a smooth row-wise field derived from IR/BG strip statistics.
                - 'global_robust': Uses fixed global dataset stats (requires global_center/scale).
            global_center (float, optional): Dataset median for global_robust mode.
            global_scale (float, optional): Dataset scale (MAD) for global_robust mode.
            device (torch.device, optional): Device to run inference on (cuda/cpu).
            max_img_size (int, optional): Max dimension before switching to sliding window inference.
        """
        self.model = model.eval()
        self.device = device
        self.model.to(self.device)
        self.max_img_size = max_img_size
        self.cfg = config

        self.cfg = config
        self.meta_cols = list(self.cfg.get("data", {}).get("meta_cols", []) or [])
        self.meta_ccd = self.cfg.get("data", {}).get("meta_ccd", "RED3")
        assert self.meta_ccd in {"BG12", "IR10", "RED3", "RED4", "RED5"}, \
            f"Invalid meta_ccd: {self.meta_ccd}"

        self.norm_mode = norm_mode if norm_mode is not None else self.cfg.get("data", {}).get("normalisation_mode", DEFAULT_NORMALIZATION_MODE)
        if self.norm_mode == "global_robust" and (global_center is None or global_scale is None):
            normalization_kwargs = normalization_kwargs_from_config(self.cfg, norm_mode=self.norm_mode)
            if global_center is None:
                global_center = normalization_kwargs.get("global_center")
            if global_scale is None:
                global_scale = normalization_kwargs.get("global_scale")
        self.global_center = global_center
        self.global_scale = global_scale
        self.verbose = bool(verbose)

        arch_args = self.cfg.get("architecture", {}).get("args", {})
        depth = int(arch_args.get("depth", 4))
        self.unet_stride = 2 ** depth
        self.max_px_per_batch = max_px_per_batch
        self.smooth_norm_kernel = int(smooth_norm_kernel)
        if self.smooth_norm_kernel < 1:
            raise ValueError(f"smooth_norm_kernel must be >= 1, got {self.smooth_norm_kernel}")
        self.batch_denorm = []

        self.meta_scaler = None
        meta_scaler_path = self.cfg.get("shared_meta_scalar", None)
        if meta_scaler_path is not None:
            try:
                self.meta_scaler = joblib.load(meta_scaler_path)
            except FileNotFoundError:
                raise FileNotFoundError(f"Metadata scaler not found: {meta_scaler_path}")

        if self.meta_cols and self.meta_scaler is None:
            raise ValueError("A fitted metadata scaler is required when meta_cols is enabled.")

        # Only needed when metadata is enabled
        self._edr_df = gen_edr_df() if self.meta_cols else None
        self._obs_meta_cache = {}
        # ----------------------------------------


        assert self.norm_mode in ['scene_robust', 'scene_smooth', 'global_robust', None], f"Invalid norm_mode given: {self.norm_mode}"
        if self.norm_mode == 'global_robust':
            assert self.global_center is not None and self.global_scale is not None, \
                "global_center/global_scale required for global_robust"
            assert self.global_scale > 0, "global_scale must be > 0"

        # Pluggable normalization strategy. Defaults to the built-in matching norm_mode;
        # callers may inject a custom Normalizer to override it. An explicitly injected
        # normalizer takes precedence over BOTH norm_mode and any per-sample/record
        # obs_center/obs_scale (see _build_inputs); the default path is unchanged.
        self._normalizer_injected = normalizer is not None
        self._normalizer = normalizer if normalizer is not None else normalizer_from_mode(
            self.norm_mode,
            global_center=self.global_center,
            global_scale=self.global_scale,
            smooth_kernel=self.smooth_norm_kernel,
            include_red_means=False,
        )

        self.meta_key_map = {
            "EMISSION_ANGLE": "Emission_angle",
            "INCIDENCE_ANGLE": "Incidence_angle",
            "PHASE_ANGLE": "Phase_angle",
            "NORTH_AZIMUTH": "North_azimuth",
            "LOCAL_TIME": "Local_time",
            "IMAGE_LINES": "Height",
            "LINE_SAMPLES": "Width",
            "CCD_NAME": "CCD",
            "ORBIT_NUMBER": "Orbit_number",
            "SLANT_DISTANCE": "Slant_distance",
            "SPACECRAFT_ALTITUDE": "Spacecraft_altitude",
            "TARGET_CENTER_DISTANCE": "Target_center_distance",
            "SOLAR_DISTANCE": "Solar_distance",
            "SOLAR_LONGITUDE": "Solar_longitude",
            "SUB_SOLAR_AZIMUTH": "Sub_solar_azimuth",
            "SUB_SOLAR_LATITUDE": "Sub_solar_latitude",
            "SUB_SOLAR_LONGITUDE": "Sub_solar_longitude",
            "SUB_SPACECRAFT_LATITUDE": "Sub_spacecraft_latitude",
            "SUB_SPACECRAFT_LONGITUDE": "Sub_spacecraft_longitude",
            "IMAGE_CENTER_LATITUDE": "Image_center_lat",
            "IMAGE_CENTER_LONGITUDE": "Image_center_lon",
            "BINNING": "Binning",
            "TDI": "TDI",
            "OBSERVATION_START_TIME": "Date",

            "CPMM_PWS_BOARD_TEMPERATURE": "CPMM_temp",
            "FPA_NEGATIVE_Y_TEMPERATURE": "FPA_negY_temp",
            "FPA_POSITIVE_Y_TEMPERATURE": "FPA_posY_temp",
            "FPE_TEMPERATURE": "FPE_temp",
            "IEA_PWS_BOARD_TEMPERATURE": "IEA_PWS_temp",
            "IEA_TEMPERATURE": "IEA_temp",
        }   # Note CHANNEL_NUMBER != Channels - First is [1, 2] second is 3rd dim of input.

    def _normalise(self, bands, neighbours, norm_stats=None):
        """
        Apply normalization to band and neighbour tensors.

        Statistics come from an explicit ``norm_stats`` when provided (e.g. record- or
        observation-level stats precomputed before tiling), otherwise from the
        configured pluggable normalizer (see hipredict.data.sample_build).
        """
        if norm_stats is not None:
            a, s = norm_stats
        else:
            a, s = self._normalizer.stats(bands, neighbours)

        bands = (bands - a) / s
        neighbours = (neighbours - a) / s
        # Make attributes to scale true images
        self.a = a
        self.s = s
        return bands, neighbours, a, s

    @staticmethod
    def _summarise_stat(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().float().cpu()
            if x.numel() == 1:
                return float(x.item())
            return float(x.median().item())
        return float(x)

    @staticmethod
    def _to_denorm_param(x):
        if isinstance(x, torch.Tensor):
            x = x.detach().float().cpu().numpy()
            return float(x.reshape(())) if x.size == 1 else x
        return float(x)

    def _translate_meta_cols(self, meta_in):
        """
        Convert a raw metadata dictionary into a normalized PyTorch tensor.
        HAndles list of dicts when batching

        - Translate any PDS-style keys into internal ones
        - Accept keys that are already in internal form (if theyre in self.meta_cols)
        - Ignore everything that isnt in meta_cols
        - Reorder
        - Assert none missing
        - Return as torch.Tensor of shape (1, n_meta)

        Args:
            meta_in (dict): Dictionary of metadata keys (e.g., 'EMISSION_ANGLE') and values.
                            Can use PDS labels or internal column names.

        Returns:
            torch.Tensor: Normalized metadata tensor of shape (1, n_meta).

        Raises:
            KeyError: If required metadata columns defined in config are missing.
        """

        def process_single_dict(d):
            translated = {}
            for k, v in d.items():
                if k in self.meta_cols:
                    translated[k] = v
                else:
                    internal_key = self.meta_key_map.get(k)
                    if internal_key is not None and internal_key in self.meta_cols:
                        translated.setdefault(internal_key, v)

            values = []
            missing = []

            for col in self.meta_cols:
                if col not in translated:
                    missing.append(col)
                else:
                    values.append(translated[col])

            if missing:
                raise KeyError(
                    f"Missing required meta columns: {missing}."
                    f"Provided keys: {list(d.keys())}"
                )
            return values

        # Check if input is list (batch) or dict (single)
        if isinstance(meta_in, list):
            batch_values = [process_single_dict(m) for m in meta_in]    # [[v1, v2..], [v1, v2..]]
            vals_np = np.array(batch_values)                            # (B, n_meta)
        else:
            vals_np = np.array([process_single_dict(meta_in)])          # (1, n_meta)

        if self.meta_scaler is not None:
            df = pd.DataFrame(vals_np, columns=self.meta_cols)

            scaler_cols = getattr(self.meta_scaler, "feature_names_in_", None)
            if scaler_cols is not None:
                scaler_cols = list(scaler_cols)

                missing = [c for c in scaler_cols if c not in df.columns]
                extra = [c for c in df.columns if c not in scaler_cols]
                if missing:
                    raise KeyError(f"meta_scalar expects columns {missing} which are missing from meta_cols={self.meta_cols}")
                if extra:
                    # config has columns the scaler didn't see at fit-time
                    df = df.drop(columns=extra)

                df = df.reindex(columns=scaler_cols)

            vals_np = self.meta_scaler.transform(df)
        return torch.from_numpy(vals_np).float().to(self.device)  # Return tensor with shape (n_meta,)

    def _pad_to_stride(self, x_band, x_neigh, stride):
        # x_band: (B, 2, H, W), x_neigh: (B, 6, H, 32)
        B, Cb, H, W = x_band.shape

        pad_h = (stride - (H % stride)) % stride
        pad_w = (stride - (W % stride)) % stride

        if pad_h == 0 and pad_w == 0:
            return x_band, x_neigh, (0, 0, 0, 0), (H, W)

        pt = pad_h // 2
        pb = pad_h - pt
        pl = pad_w // 2
        pr = pad_w - pl

        # Band: reflect is consistent with conv padding_mode="reflect"
        x_band = F.pad(x_band, (pl, pr, pt, pb), mode="reflect")

        # Neigh: only pad height (width is fixed 32); replicate is a safe choice
        if x_neigh is not None:
            x_neigh = F.pad(x_neigh, (0, 0, pt, pb), mode="replicate")

        return x_band, x_neigh, (pt, pb, pl, pr), (H, W)

    def _load_data_record_df(self, source):
        if isinstance(source, pd.DataFrame):
            return source.copy()

        if isinstance(source, (str, Path)):
            source = Path(source)
            if not source.is_file():
                raise FileNotFoundError(f"data_record file not found: {source}")
            if source.suffix.lower() != ".csv":
                raise ValueError(f"Expected a .csv data_record path, got: {source}")
            df = pd.read_csv(source)
            df.attrs["record_dir"] = str(source.resolve().parent)
            return df

        raise TypeError(
            "For mode='patches', source must be a pandas DataFrame or a path to a .csv data_record."
        )

    def _coerce_obs_ids(self, obs_ids):
        if obs_ids is None:
            return None
        return check_obs_list(obs_ids) if isinstance(obs_ids, (str, list)) else list(obs_ids)

    def _build_patch_dataset(self, source, obs_ids=None, sets=None):
        """Build a PatchInferenceDataset from a data_record source.

        Shared by _load_patch_samples and load_patch_loader so both apply the
        same validation and filtering.
        """
        df = self._load_data_record_df(source)

        required_cols = {"Set", "Observation", "Patch", "CCD", "Path"}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            raise KeyError(f"data_record is missing required columns: {sorted(missing_cols)}")

        obs_ids = self._coerce_obs_ids(obs_ids)
        return PatchInferenceDataset(df, obs_ids=obs_ids, sets=sets,
                                     data_root=self.cfg.get("data", {}).get("patch_root"))

    def _load_patch_samples(self, source, obs_ids=None, sets=None):
        """
        Load explicit patch samples from a PrepObservations data_record.

        Returns a list of dicts accepted directly by predict():
            {"x_band": ..., "x_neigh": ..., "obs_id": ..., "y": ...}
        """
        dataset = self._build_patch_dataset(source, obs_ids=obs_ids, sets=sets)
        samples = [dataset[i] for i in range(len(dataset))]
        if not samples:
            raise ValueError("No valid patch samples could be loaded from data_record.")
        return samples

    def load_patch_loader(self, source, obs_ids=None, sets=None, batch_size=1,
                          num_workers=0, shuffle=False):
        """Return a DataLoader over patch samples for the same source as load().

        Each yielded batch is a plain list of sample dicts (via identity_collate),
        so it can be passed straight to predict(): e.g.

            for batch in pred.load_patch_loader(data_record):
                out = pred.predict(batch, ...)

        At batch_size=1 this reproduces the per-sample list path exactly.
        """
        dataset = self._build_patch_dataset(source, obs_ids=obs_ids, sets=sets)
        return get_loader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=identity_collate,
        )

    def _load_strip_samples(self, source, obs_ids=None):
        """
        Load full-scene reconstruct-mode samples from processed cubefiles.

        - If source points to a single observation directory containing `*_processed.cub`,
          load that observation directly.
        - Otherwise source is treated as a base directory and obs_ids must identify
          observation subdirectories beneath it.
        """
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"strip source path not found: {source}")

        direct_cubes = sorted(source.glob("*_processed.cub"))
        if direct_cubes:
            obs_dirs = [(source.name, source)]
        else:
            obs_ids = self._coerce_obs_ids(obs_ids)
            if obs_ids is None:
                raise ValueError(
                    "For mode='strips', obs_ids must be provided when source is a base directory."
                )
            obs_dirs = [(obs_id, source / obs_id) for obs_id in obs_ids]

        samples = []
        for obs_id, obs_dir in obs_dirs:
            if not obs_dir.is_dir():
                raise FileNotFoundError(f"Observation directory not found for {obs_id}: {obs_dir}")

            cubefiles = sorted(obs_dir.glob("*_processed.cub"))
            if len(cubefiles) != 5:
                raise ValueError(
                    f"Expected 5 processed cubefiles for {obs_id}, found {len(cubefiles)} in {obs_dir}"
                )

            bg_img_hiproc = isis2np(str(cubefiles[0]), img_save_loc=None)
            ir_img_hiproc = isis2np(str(cubefiles[1]), img_save_loc=None)
            r3_img_hiproc = isis2np(str(cubefiles[2]), img_save_loc=None)
            r4_img_hiproc = isis2np(str(cubefiles[3]), img_save_loc=None)
            r5_img_hiproc = isis2np(str(cubefiles[4]), img_save_loc=None)

            arr_stack = np.dstack([bg_img_hiproc, ir_img_hiproc, r3_img_hiproc, r4_img_hiproc, r5_img_hiproc])
            original_shape = arr_stack.shape[:2]
            obs_arr_cropped, crop_dims = crop_offset(arr_stack)

            samples.append(
                {
                    "bg": obs_arr_cropped[..., 0],
                    "ir": obs_arr_cropped[..., 1],
                    "r3": obs_arr_cropped[..., 2],
                    "r5": obs_arr_cropped[..., 4],
                    "obs_id": obs_id,
                    "y": obs_arr_cropped[..., 3],
                    "original_shape": original_shape,
                    "crop_dims": crop_dims,
                }
            )

        if not samples:
            raise ValueError("No strip samples could be loaded.")
        return samples

    def load(self, source, mode="patches", obs_ids=None, sets=None):
        """
        Prepare inference-ready samples for predict().

        Args:
            source:
                - mode='patches': data_record CSV path or pandas DataFrame.
                - mode='strips': processed cubefile directory for a single observation,
                  or a base directory containing observation subdirectories.
            mode (str): One of {'patches', 'strips'}.
            obs_ids: Optional observation filter. Accepts a list or `.txt` path.
            sets: Optional set filter for mode='patches'.

        Returns:
            list[dict]: Samples accepted directly by predict().
        """
        if mode == "patches":
            return self._load_patch_samples(source, obs_ids=obs_ids, sets=sets)
        if mode == "strips":
            return self._load_strip_samples(source, obs_ids=obs_ids)
        raise ValueError(f"Unsupported load mode: {mode}. Expected 'patches' or 'strips'.")

    def _get_obs_meta(self, obs_id):
        if not self.meta_cols:
            return None

        if obs_id in self._obs_meta_cache:
            return self._obs_meta_cache[obs_id]

        assert self._edr_df is not None
        df = self._edr_df[self._edr_df["OBSERVATION_ID"] == obs_id]
        if df.empty:
            raise KeyError(f"Observation {obs_id} not found in EDR_df")

        obs_meta = {
            "OBSERVATION_START_TIME": df["OBSERVATION_START_TIME"].iloc[0],
            "per_ccd": {
                row["CCD_NAME"]: (int(row["BINNING"]), int(row["TDI"]))
                for _, row in df.iterrows()
            },
            "CPMM_temp": df["CPMM_PWS_BOARD_TEMPERATURE"].iloc[0],
            "Emission_angle": df["EMISSION_ANGLE"].iloc[0],
            "FPA_negY_temp": df["FPA_NEGATIVE_Y_TEMPERATURE"].iloc[0],
            "FPA_posY_temp": df["FPA_POSITIVE_Y_TEMPERATURE"].iloc[0],
            "FPE_temp": df["FPE_TEMPERATURE"].iloc[0],
            "IEA_PWS_temp": df["IEA_PWS_BOARD_TEMPERATURE"].iloc[0],
            "IEA_temp": df["IEA_TEMPERATURE"].iloc[0],
            "Image_center_lat": df["IMAGE_CENTER_LATITUDE"].iloc[0],
            "Image_center_lon": df["IMAGE_CENTER_LONGITUDE"].iloc[0],
            "Incidence_angle": df["INCIDENCE_ANGLE"].iloc[0],
            "Local_time": df["LOCAL_TIME"].iloc[0],
            "North_azimuth": df["NORTH_AZIMUTH"].iloc[0],
            "Orbit_number": df["ORBIT_NUMBER"].iloc[0],
            "Phase_angle": df["PHASE_ANGLE"].iloc[0],
            "Slant_distance": df["SLANT_DISTANCE"].iloc[0],
            "Solar_distance": df["SOLAR_DISTANCE"].iloc[0],
            "Solar_longitude": df["SOLAR_LONGITUDE"].iloc[0],
            "Spacecraft_altitude": df["SPACECRAFT_ALTITUDE"].iloc[0],
            "Sub_solar_azimuth": df["SUB_SOLAR_AZIMUTH"].iloc[0],
            "Sub_solar_latitude": df["SUB_SOLAR_LATITUDE"].iloc[0],
            "Sub_solar_longitude": df["SUB_SOLAR_LONGITUDE"].iloc[0],
            "Sub_spacecraft_latitude": df["SUB_SPACECRAFT_LATITUDE"].iloc[0],
            "Sub_spacecraft_longitude": df["SUB_SPACECRAFT_LONGITUDE"].iloc[0],
            "Target_center_distance": df["TARGET_CENTER_DISTANCE"].iloc[0],
        }

        self._obs_meta_cache[obs_id] = obs_meta
        return obs_meta

    def _resolve_sample_meta(self, sample, is_explicit):
        if not self.meta_cols:
            return None

        if not isinstance(sample, dict) or "obs_id" not in sample:
            raise ValueError(
                "When meta_cols is non-empty, each sample passed to predict() must be a dict "
                "containing 'obs_id'. For explicit mode use "
                "{'x_band': ..., 'x_neigh': ..., 'obs_id': ...}. "
                "For reconstruct mode use "
                "{'bg': ..., 'ir': ..., 'r3': ..., 'r5': ..., 'obs_id': ...}."
            )

        obs_id = sample["obs_id"]
        obs_meta = self._get_obs_meta(obs_id)
        assert obs_meta is not None

        if self.meta_ccd not in obs_meta["per_ccd"]:
            raise KeyError(f"{self.meta_ccd} not found in per_ccd metadata for {obs_id}")

        binning, tdi = obs_meta["per_ccd"][self.meta_ccd]

        if is_explicit:
            h = int(sample["x_band"].shape[-2])
            w = int(sample["x_band"].shape[-1])
        else:
            h = int(sample["bg"].shape[-2])
            w = int(sample["bg"].shape[-1])

        meta_dict = {
            "Observation": obs_id,
            "CCD": self.meta_ccd,
            "Height": h,
            "Width": w,
            "Channels": 1,
            "Date": obs_meta["OBSERVATION_START_TIME"],
            "Binning": binning,
            "TDI": tdi,
            "CPMM_temp": obs_meta["CPMM_temp"],
            "Emission_angle": obs_meta["Emission_angle"],
            "FPA_negY_temp": obs_meta["FPA_negY_temp"],
            "FPA_posY_temp": obs_meta["FPA_posY_temp"],
            "FPE_temp": obs_meta["FPE_temp"],
            "IEA_PWS_temp": obs_meta["IEA_PWS_temp"],
            "IEA_temp": obs_meta["IEA_temp"],
            "Image_center_lat": obs_meta["Image_center_lat"],
            "Image_center_lon": obs_meta["Image_center_lon"],
            "Incidence_angle": obs_meta["Incidence_angle"],
            "Local_time": obs_meta["Local_time"],
            "North_azimuth": obs_meta["North_azimuth"],
            "Orbit_number": obs_meta["Orbit_number"],
            "Phase_angle": obs_meta["Phase_angle"],
            "Slant_distance": obs_meta["Slant_distance"],
            "Solar_distance": obs_meta["Solar_distance"],
            "Solar_longitude": obs_meta["Solar_longitude"],
            "Spacecraft_altitude": obs_meta["Spacecraft_altitude"],
            "Sub_solar_azimuth": obs_meta["Sub_solar_azimuth"],
            "Sub_solar_latitude": obs_meta["Sub_solar_latitude"],
            "Sub_solar_longitude": obs_meta["Sub_solar_longitude"],
            "Sub_spacecraft_latitude": obs_meta["Sub_spacecraft_latitude"],
            "Sub_spacecraft_longitude": obs_meta["Sub_spacecraft_longitude"],
            "Target_center_distance": obs_meta["Target_center_distance"],
        }
        return meta_dict

    def _build_inputs(self, im, is_explicit):
        """
        Parse and batch raw input samples into model-ready tensors.

        When metadata is enabled, each sample must be a dict carrying obs_id so
        metadata can be resolved internally.
        """
        band_list = []
        neigh_list = []
        meta_dicts = []
        self.batch_denorm = []
        self.batch_stats = []


        for sample in im:
            if is_explicit:
                if isinstance(sample, dict):
                    band = sample["x_band"]
                    neigh = sample["x_neigh"]
                else:
                    band, neigh = sample

                band_t = torch.as_tensor(band)
                neigh_t = torch.as_tensor(neigh)
                norm_stats = None
                # An explicitly injected normalizer overrides per-sample/record stats.
                if not self._normalizer_injected and self.norm_mode == "scene_robust" and isinstance(sample, dict):
                    obs_center = sample.get("obs_center")
                    obs_scale = sample.get("obs_scale")
                    if obs_center is not None and obs_scale is not None:
                        norm_stats = (
                            torch.tensor(float(obs_center), dtype=band_t.dtype),
                            torch.tensor(float(obs_scale), dtype=band_t.dtype),
                        )

                band_t, neigh_t, a, s = self._normalise(band_t, neigh_t, norm_stats=norm_stats)

                band_list.append(band_t)
                neigh_list.append(neigh_t)
                self.batch_stats.append((self._summarise_stat(a), self._summarise_stat(s)))
                self.batch_denorm.append((self._to_denorm_param(a), self._to_denorm_param(s)))

            else:
                if isinstance(sample, dict):
                    bg = sample["bg"]
                    ir = sample["ir"]
                    r3 = sample["r3"]
                    r5 = sample["r5"]
                else:
                    bg, ir, r3, r5 = sample

                band, neigh = assemble_strip_inputs(bg, ir, r3, r5)

                norm_stats = None
                # An explicitly injected normalizer overrides per-sample/record stats.
                if not self._normalizer_injected and self.norm_mode == "scene_robust" and isinstance(sample, dict):
                    obs_center = sample.get("obs_center")
                    obs_scale = sample.get("obs_scale")
                    if obs_center is not None and obs_scale is not None:
                        norm_stats = (
                            torch.tensor(float(obs_center), dtype=band.dtype),
                            torch.tensor(float(obs_scale), dtype=band.dtype),
                        )

                if norm_stats is None:
                    # Observation-level (pre-tiling) stats from the pluggable normalizer.
                    # Built-in SceneRobust returns the canonical IR/BG predictor stats;
                    # other normalizers return None to normalize per-tile in _normalise.
                    predictor_stats_fn = getattr(self._normalizer, "predictor_stats", None)
                    pre = predictor_stats_fn(bg, ir) if predictor_stats_fn is not None else None
                    if pre is not None:
                        obs_center, obs_scale = pre
                        norm_stats = (
                            torch.tensor(obs_center, dtype=band.dtype),
                            torch.tensor(obs_scale, dtype=band.dtype),
                        )

                band, neigh, a, s = self._normalise(band, neigh, norm_stats=norm_stats)

                band_list.append(band)
                neigh_list.append(neigh)
                self.batch_stats.append((self._summarise_stat(a), self._summarise_stat(s)))
                self.batch_denorm.append((self._to_denorm_param(a), self._to_denorm_param(s)))

            if self.meta_cols:
                meta_dicts.append(self._resolve_sample_meta(sample, is_explicit))

        x_band = torch.stack(band_list, dim=0).float().to(self.device)
        x_neigh = torch.stack(neigh_list, dim=0).float().to(self.device)

        if self.meta_cols:
            x_meta = self._translate_meta_cols(meta_dicts)
        else:
            x_meta = None

        if getattr(self, "zero_meta", False) and x_meta is not None:
            x_meta = torch.zeros_like(x_meta)

        return x_band, x_neigh, x_meta

    @torch.no_grad()
    def predict_full(self, x_band, x_neigh, x_meta):
        """Predict on full image (when it fits in memory)."""
        x_band, x_neigh, pad, orig = self._pad_to_stride(x_band, x_neigh, self.unet_stride)   # Pad to ensure no concat issues in model
        pt, pb, pl, pr = pad
        H0, W0 = orig

        y = self.model(x_band, x_neigh, x_meta)  # (1,O,H,W)

        if pt or pb or pl or pr:
            y = y[:, :, pt:pt + H0, pl:pl + W0]   # Crop back to original size (remove padding)

        y = y.cpu().numpy()  # (O,H,W)

        # 1. Move channels to the last axis: (B, H, W, O)
        y = np.moveaxis(y, 1, -1)

        # 2. If single channel output (e.g. grayscale mask), squeeze it: (B, H, W)
        if y.shape[-1] == 1:
            y = y.squeeze(-1)

        return y

    @torch.no_grad()
    def predict_sliding(self, x_band, x_neigh, x_meta, window_size, pad_x, pad_y, bs=32):
        """Predict using sliding window (for large images)."""
        assert x_band.ndim == 4, f"expected x_band 4D, got {x_band.shape}"
        assert x_neigh.ndim == 4, f"expected x_neigh 4D, got {x_neigh.shape}"

        B, Cb, H, W = x_band.shape
        Bn, Cn, Hn, Wn = x_neigh.shape

        assert B == 1, "predict_sliding currently supports B == 1 (one image at a time)"
        assert Bn == 1, "x_neigh batch size must match x_band and equal 1"
        assert Cb == 2, f"expected 2 band channels, got {Cb}"
        assert Cn == 6, f"expected 6 neighbour channels, got {Cn}"
        assert Hn == H, f"neighbour height {Hn} must match band height {H}"
        assert Wn == 32, f"expected neighbour width 32, got {Wn}"

        band_img = x_band[0]    # (2, H, W)
        neigh_img = x_neigh[0]  # (6, H, 32)

        x_band_p, x_neigh_p, pad, orig = self._pad_to_stride(x_band, x_neigh, self.unet_stride)
        pt, pb, pl, pr = pad
        H0, W0 = orig
        band_img = x_band_p[0]
        neigh_img = x_neigh_p[0]

        neigh_kernel_h = getattr(self.model, "neigh_kernel_h", None)
        neigh_buffer = 0
        if neigh_kernel_h is not None:
            neigh_buffer = int(neigh_kernel_h) // 2

        neigh_img_pad = neigh_img
        if neigh_buffer > 0:
            neigh_img_pad = F.pad(
                neigh_img,
                (0, 0, neigh_buffer, neigh_buffer),
                mode="replicate",
            )

        _, H, W = band_img.shape
        W1 = W
        tile_h = window_size + 2 * pad_x

        need = (-tile_h) % self.unet_stride
        pad_x_old = pad_x
        pad_x = pad_x + (need // 2)

        if need % 2 != 0:
            raise ValueError(
                f"Cannot make tile_h divisible by stride with symmetric pad_x: "
                f"tile_h={tile_h}, stride={self.unet_stride}, need={need} (odd)."
            )

        if pad_x != pad_x_old:
            if self.verbose:
                print(f"    -> Adjusting pad_x {pad_x_old} -> {pad_x} to satisfy tile_h % {self.unet_stride} == 0")

        # Global left/right context (no width-tiling)
        if pad_y > 0:
            assert (2 * pad_y) % self.unet_stride == 0, (
                f"pad_y must keep width divisible by {self.unet_stride}. "
                f"Got pad_y={pad_y}."
            )
            band_img = F.pad(band_img, (pad_y, pad_y, 0, 0), mode="reflect")
            _, H, W = band_img.shape

        # Height-only tiling
        if window_size + 2 * pad_x >= H:
            raise ValueError(
                f"window_size + 2*pad_x must be < H for height-only sliding: "
                f"{window_size} + 2*{pad_x} vs H={H}"
            )

        px_per_tile = int(tile_h) * int(W)
        bs_eff = max(1, min(bs, int(self.max_px_per_batch) // max(1, px_per_tile)))

        if bs_eff < bs:
            if self.verbose:
                print(f"    -> Reducing batch size from {bs} to {bs_eff} to fit within max_px_per_batch={self.max_px_per_batch} (px_per_tile={px_per_tile})")
            bs = bs_eff

        apply_row_dc = getattr(self, "apply_row_dc", False)
        k_w = int(getattr(self.model, "neigh_kernel_w", 33))

        def _meta_single(x_meta_in):
            if x_meta_in is None:
                return None
            if isinstance(x_meta_in, np.ndarray):
                meta = torch.from_numpy(x_meta_in).float().to(self.device)
            else:
                meta = x_meta_in.to(self.device)
            if meta.ndim == 1:
                meta = meta.unsqueeze(0)
            if meta.shape[0] != 1:
                raise ValueError(f"x_meta must have batch size 1, got {meta.shape[0]}")
            return meta

        def _compute_m_hat_row_full(band_img_in, neigh_img_pad_in, x_meta_in):
            _, Hm, Wm = band_img_in.shape
            x_meta_1 = _meta_single(x_meta_in)

            # Representative 33-col window around center x, matching your "use center" logic
            x_ref = Wm // 2
            half_w = k_w // 2
            w0 = max(0, x_ref - half_w)
            w1 = min(Wm, w0 + k_w)
            w0 = max(0, w1 - k_w)

            band_win = band_img_in[:, :, w0:w1]
            if band_win.shape[2] < k_w:
                need_w = k_w - band_win.shape[2]
                pl_w = need_w // 2
                pr_w = need_w - pl_w
                band_win = F.pad(band_win, (pl_w, pr_w, 0, 0), mode="reflect")

            # Band context per row via the same big aggregator
            band_for_agg = torch.zeros(
                1,
                6,
                Hm,
                k_w,
                device=band_img_in.device,
                dtype=band_img_in.dtype,
            )
            band_for_agg[:, 0:2, :, :] = band_win.unsqueeze(0)
            band_ctx = self.model.neigh_spatial_avg(band_for_agg)  # (1, 6, Hm, k_w)
            mu_band_row = band_ctx[:, 0:2, :, k_w // 2]            # (1, 2, Hm)
            mu_band_row = mu_band_row[0].permute(1, 0)             # (Hm, 2)

            # Neighbour context per row
            x_neigh_full = neigh_img_pad_in.unsqueeze(0)           # (1, 6, Hm + 2*buf, 32)
            neigh_ctx_full = self.model.neigh_spatial_avg(x_neigh_full)
            neigh_col = neigh_ctx_full.mean(dim=3, keepdim=True)   # (1, 6, Hn, 1)

            start = (neigh_col.shape[2] - Hm) // 2
            neigh_col_crop = neigh_col[:, :, start:start + Hm, :]  # (1, 6, Hm, 1)
            mu_neigh_row = neigh_col_crop[:, :, :, 0]              # (1, 6, Hm)
            mu_neigh_row = mu_neigh_row[0].permute(1, 0)           # (Hm, 6)

            if getattr(self.model, "n_meta", 0) > 0:
                if x_meta_1 is None:
                    raise ValueError("x_meta must be provided when n_meta > 0")
                meta_row = x_meta_1.expand(Hm, -1)                 # (Hm, n_meta)
                mean_in = torch.cat([mu_band_row, mu_neigh_row, meta_row], dim=1)
            else:
                mean_in = torch.cat([mu_band_row, mu_neigh_row], dim=1)

            m_hat_row = self.model.mean_head(mean_in)              # (Hm, 1)
            return m_hat_row.view(1, 1, Hm, 1)                     # (1, 1, Hm, 1)

        m_hat_row_full = None
        if apply_row_dc:
            m_hat_row_full = _compute_m_hat_row_full(band_img, neigh_img_pad, x_meta)

        stride = window_size
        ixs = np.arange(0, H, stride)
        ixs[-1] = H - window_size

        x_chunks = []
        x_offsets = []
        for ix in ixs:
            ix_pad = ix - pad_x
            x0 = ix_pad
            x1 = ix_pad + window_size + 2 * pad_x
            xoffset = 0

            if x0 < 0:
                xoffset = -x0
                x0 = x0 + xoffset
                x1 = x1 + xoffset
            if x1 > H:
                xoffset = H - x1
                x0 = x0 + xoffset
                x1 = x1 + xoffset

            x_chunks.append((ix, x0, x1))
            x_offsets.append(xoffset)

        if self.verbose:
            print(f"  -> Sliding Inference (height-only): {H}x{W} image split into {len(x_chunks)} patches")

        # Process in batches
        batch_band = []
        batch_neigh = []
        batch_x0 = []
        batch_x1 = []
        batch_ix = []
        batch_xoffset = []

        def _make_meta_batch(x_meta_in, B):
            if x_meta_in is None:
                return None

            if isinstance(x_meta_in, np.ndarray):
                meta = torch.from_numpy(x_meta_in).float().to(self.device)
            else:
                meta = x_meta_in.to(self.device)

            if meta.ndim == 1:
                # (n_meta,) -> (1, n_meta)
                meta = meta.unsqueeze(0)

            if meta.shape[0] == 1 and B > 1:
                meta = meta.expand(B, -1)
            elif meta.shape[0] != B:
                raise ValueError(
                    f"x_meta batch size mismatch: got {meta.shape[0]}, expected 1 or {B}"
                )

            return meta

        tile_ix_list = []
        tile_x0_list = []
        tile_x1_list = []
        tile_m_hat_list = []

        out_t = None
        out_dim = None

        for i, (ix, x0, x1) in enumerate(x_chunks):
            band_tile = band_img[:, x0:x1, :]
            neigh_tile = neigh_img_pad[:, x0:x1 + 2 * neigh_buffer, :]

            batch_band.append(band_tile)
            batch_neigh.append(neigh_tile)

            batch_ix.append(int(ix))
            batch_x0.append(int(x0))
            batch_x1.append(int(x1))
            batch_xoffset.append(int(x_offsets[i]))

            if len(batch_band) == bs or i == len(x_chunks) - 1:
                B_tiles = len(batch_band)

                x_band_batch = torch.stack(batch_band, dim=0)
                x_neigh_batch = torch.stack(batch_neigh, dim=0)
                x_meta_batch = _make_meta_batch(x_meta, B_tiles)

                y = self.model(x_band_batch, x_neigh_batch, x_meta_batch)

                m_hat_np = self.model.m_hat.detach().view(-1).float().cpu().numpy()
                tile_m_hat_list.extend(list(m_hat_np))

                if apply_row_dc:
                    tile_dc = y.mean(dim=(2, 3), keepdim=True)
                    assert m_hat_row_full is not None
                    for b in range(B_tiles):
                        row_dc = m_hat_row_full[:, :, batch_x0[b]:batch_x1[b], :]
                        if y.shape[1] == 1:
                            y[b:b + 1] = y[b:b + 1] - tile_dc[b:b + 1] + row_dc
                        else:
                            y[b:b + 1, 0:1] = y[b:b + 1, 0:1] - tile_dc[b:b + 1, 0:1] + row_dc

                if out_t is None:
                    out_dim = int(y.shape[1])
                    if out_dim == 1:
                        out_t = torch.empty((H, W), dtype=torch.float32, device="cpu")
                    else:
                        out_t = torch.empty((H, W, out_dim), dtype=torch.float32, device="cpu")

                for b in range(B_tiles):
                    ix_b = batch_ix[b]
                    xoffset = batch_xoffset[b]

                    core_y0 = pad_x - xoffset
                    core_y1 = core_y0 + window_size

                    if out_dim == 1:
                        core = y[b, 0, core_y0:core_y1, :].detach().float().cpu()
                        out_t[ix_b:ix_b + window_size, :] = core
                    else:
                        core = y[b, :, core_y0:core_y1, :].detach().float().permute(1, 2, 0).cpu()
                        out_t[ix_b:ix_b + window_size, :, :] = core

                del x_band_batch, x_neigh_batch, x_meta_batch, y

                batch_band = []
                batch_neigh = []
                batch_x0 = []
                batch_x1 = []
                batch_ix = []
                batch_xoffset = []
                tile_ix_list.append(int(ix))
                tile_x0_list.append(int(x0))
                tile_x1_list.append(int(x1))

        assert out_t is not None
        out = out_t.numpy()

        # Remove global horizontal pad (added above)
        if pad_y > 0:
            if out.ndim == 2:
                out = out[:, pad_y:pad_y + W1]
            else:
                out = out[:, pad_y:pad_y + W1, :]

        # Remove stride padding back to original
        if pt or pb or pl or pr:
            if out.ndim == 2:
                out = out[pt:pt + H0, pl:pl + W0]
            else:
                out = out[pt:pt + H0, pl:pl + W0, :]

        mh = np.array(tile_m_hat_list, dtype=np.float64)
        if self.verbose:
            print("m_hat per tile:", np.round(mh, 6))
            print("delta m_hat:", np.round(np.diff(mh), 6))
            print("tile ix:", tile_ix_list)

        return out

    @torch.no_grad()
    def predict(self, im, window_size=None, pad_x=None, pad_y=None, syn4=False):
        """
        Run inference on one or more samples.

        Denormalization is applied after the model forward pass using the same
        affine that was used to normalize the inputs:
        - scalar a/s for scene_robust and global_robust
        - row-wise smooth fields for scene_smooth

        This keeps the output in physical units while allowing strip inference
        to stay closer to the patch-local normalization regime used in training.

        Normalization precedence:
        - If a Normalizer was explicitly injected at construction, it alone decides
          the stats (its .stats() for patches, .predictor_stats() for strips);
          norm_mode and any per-sample/record obs_center/obs_scale are ignored.
        - In scene_robust mode, per-sample/record obs_center/obs_scale win when present, else
          the built-in normalizer selected by norm_mode is used.

        This method accepts either raw samples or the list returned by load().

        Supported input formats:
        - Explicit mode, meta disabled:
            (x_band, x_neigh)
        - Explicit mode, meta enabled:
            {"x_band": ..., "x_neigh": ..., "obs_id": ...}
        - Output of self.load(..., mode="patches")
            {"x_band": ..., "x_neigh": ..., "obs_id": ..., "y": ...}

        - Reconstruct mode, meta disabled:
            [bg, ir, r3, r5]
        - Reconstruct mode, meta enabled:
            {"bg": ..., "ir": ..., "r3": ..., "r5": ..., "obs_id": ...}
        - Output of self.load(..., mode="strips")
            {"bg": ..., "ir": ..., "r3": ..., "r5": ..., "obs_id": ..., "y": ...}
        """

        def _is_explicit(sample):
            if isinstance(sample, dict):
                return "x_band" in sample and "x_neigh" in sample
            return len(sample) == 2 and hasattr(sample[0], "shape") and sample[0].ndim == 3

        def _hw(sample, is_explicit):
            if is_explicit:
                if isinstance(sample, dict):
                    H, W = sample["x_band"].shape[-2:]
                else:
                    H, W = sample[0].shape[-2:]
            else:
                if isinstance(sample, dict):
                    H, W = sample["bg"].shape
                else:
                    H, W = sample[0].shape
            return H, W

        from hipredict.predict.models import SYN4
        syn4 = syn4 or isinstance(self.model, SYN4)
        if syn4 and (not hasattr(self, "_syn4_model") or self._syn4_model is None):
            if self.norm_mode != "global_robust" or self._normalizer_injected:
                raise ValueError("SYN4 requires global_robust normalization without a custom normalizer.")
            self._syn4_model = SYN4(center=self.global_center, scale=self.global_scale)
            self._syn4_model.to(self.device).eval()

        outs = []
        stats = []
        denorm_stats = []

        for sample in im:
            is_explicit = _is_explicit(sample)
            H, W = _hw(sample, is_explicit)
            if H <= 0 or W <= 0:
                raise ValueError(
                    f"Sample has empty spatial dimensions (H={H}, W={W}). This usually "
                    f"means the crop window fell outside the source image bounds (e.g. a "
                    f"CENTER_X/CENTER_Y plus crop half-width exceeding the image extent)."
                )
            max_dim = max(H, W)

            x_band, x_neigh, x_meta = self._build_inputs([sample], is_explicit)

            if syn4:
                y = self._syn4_model(x_band, x_neigh, x_meta).detach().cpu().numpy()
                y = np.moveaxis(y, 1, -1)
                if y.shape[-1] == 1:
                    y = y.squeeze(-1)
                out_i = y[0]
            else:
                if is_explicit or max_dim <= self.max_img_size:
                    out_i = self.predict_full(x_band, x_neigh, x_meta)[0]
                else:
                    if window_size is None or pad_x is None or pad_y is None:
                        raise ValueError("window_size and pad must be provided for sliding predict")
                    out_i = self.predict_sliding(
                        x_band,
                        x_neigh,
                        x_meta,
                        window_size,
                        pad_x=pad_x,
                        pad_y=pad_y
                    )

            if len(self.batch_stats) != 1 or len(self.batch_denorm) != 1:
                raise RuntimeError(
                    f"expected batch_stats and batch_denorm length 1, got "
                    f"{len(self.batch_stats)} and {len(self.batch_denorm)}"
                )

            a_log, s_log = self.batch_stats[0]
            a_denorm, s_denorm = self.batch_denorm[0]

            if isinstance(a_denorm, np.ndarray):
                if out_i.ndim == 3:
                    a_denorm = a_denorm[..., None]
                    s_denorm = s_denorm[..., None]

            out_i = out_i * s_denorm + a_denorm

            outs.append(out_i)
            stats.append((a_log, s_log))
            denorm_stats.append((a_denorm, s_denorm))

        self.batch_stats = stats
        self.batch_denorm = denorm_stats

        try:
            return np.stack(outs, axis=0)
        except Exception:
            return outs
