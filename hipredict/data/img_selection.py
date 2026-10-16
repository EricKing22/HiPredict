import argparse
import random
from pathlib import Path

import pandas as pd

from hipredict.data.load_metadata import gen_edr_df
from hipredict.data.selection_filters import zscore_filter
from hipredict.paths import files_dir


def obs_count(df, n_ccds):
    """
    This function counts the number of observations that contain all the required CCDs.
    Return:
    - n_obs: number of observations that contain all the required CCDs
    """
    assert n_ccds == 5 or n_ccds == 13, "required_ccds must be either 5 or 13"

    if n_ccds == 5:
        required_ccds = ["RED3", "RED4", "RED5", "IR10", "BG12"]
    else:
        required_ccds = [
            "RED0",
            "RED1",
            "RED2",
            "RED3",
            "RED4",
            "RED5",
            "RED6",
            "RED7",
            "RED8",
            "IR10",
            "IR11",
            "BG12",
            "BG13",
        ]


    obs_list = (
        df.groupby("OBSERVATION_ID")["CCD_NAME"]
        .apply(lambda x: set(required_ccds).issubset(set(x.unique())))
    )
    n_obs = obs_list.sum()
    return n_obs


def select_imgs(EDR_df):
    drop_cols = [
        "LOOKUP_TABLE_MINIMUM",
        "LOOKUP_TABLE_MAXIMUM",
        "CHANNEL_NUMBER",
        "LOOKUP_TABLE_K_VALUE",
        "LOOKUP_TABLE_MEDIAN",
        "PRODUCT_VERSION_ID",
    ]
    EDR_updated = EDR_df.drop(columns=drop_cols)
    EDR_updated.reset_index(drop=True, inplace=True)

    updated_df = EDR_updated.copy()

    cross_tab = (
        pd.crosstab(updated_df["TDI"], updated_df["BINNING"], normalize=True) * 100
    ).round(2)
    print("\nConfuguraitons before modificaton:")
    print(cross_tab)

    # --- Category 2b: Remove outliers, z-score method ---
    cat2b_mask = (
        zscore_filter(updated_df, "CPMM_PWS_BOARD_TEMPERATURE")
        & zscore_filter(updated_df, "FPA_NEGATIVE_Y_TEMPERATURE")  # noqa
        & zscore_filter(updated_df, "FPA_POSITIVE_Y_TEMPERATURE")  # noqa
        & zscore_filter(updated_df, "FPE_TEMPERATURE")  # noqa
        & zscore_filter(updated_df, "IEA_PWS_BOARD_TEMPERATURE")  # noqa
        & zscore_filter(updated_df, "IEA_TEMPERATURE")  # noqa
        & zscore_filter(updated_df, "TARGET_CENTER_DISTANCE")  # noqa
        & zscore_filter(updated_df, "SCAN_EXPOSURE_DURATION")  # noqa
        & zscore_filter(updated_df, "DELTA_LINE_TIME_COUNT")  # noqa
        & zscore_filter(updated_df, "SLANT_DISTANCE")  # noqa
        & zscore_filter(updated_df, "SPACECRAFT_ALTITUDE")  # noqa
    )

    # --- Category 3: Keep common ADC combinations + 8-bit samples ---
    cond_bits = updated_df["SAMPLE_BITS"] == 8

    adc_combo_pct = (
        updated_df.loc[cond_bits, ["ADC_CONVERSION_SETTINGS_1", "ADC_CONVERSION_SETTINGS_2"]]
        .value_counts(normalize=True)
        .mul(100)
        .rename("pct_images")
        .reset_index()
        .sort_values("pct_images", ascending=False)
    )

    kept_adc_combos = {
        (int(row["ADC_CONVERSION_SETTINGS_1"]), int(row["ADC_CONVERSION_SETTINGS_2"]))
        for _, row in adc_combo_pct.iterrows()
        if row["pct_images"] >= 1.0
    }

    print(f"\nKeeping ADC combinations with >= 1.00% of 8-bit images: {sorted(kept_adc_combos)}")

    adc_combo_mask = list(
        zip(
            updated_df["ADC_CONVERSION_SETTINGS_1"],
            updated_df["ADC_CONVERSION_SETTINGS_2"],
        )
    )
    cond_adc_combo = pd.Series(
        [combo in kept_adc_combos for combo in adc_combo_mask],
        index=updated_df.index,
    )

    cat3_mask = cond_bits & cond_adc_combo

    adc_combo_counts = (
        updated_df.loc[cond_bits, ["ADC_CONVERSION_SETTINGS_1", "ADC_CONVERSION_SETTINGS_2"]]
        .value_counts()
        .rename("n_images")
        .reset_index()
    )

    adc_combo_summary = adc_combo_pct.merge(
        adc_combo_counts,
        on=["ADC_CONVERSION_SETTINGS_1", "ADC_CONVERSION_SETTINGS_2"],
        how="left",
    )

    adc_combo_summary["status"] = adc_combo_summary.apply(
        lambda row: "KEEP"
        if (int(row["ADC_CONVERSION_SETTINGS_1"]), int(row["ADC_CONVERSION_SETTINGS_2"])) in kept_adc_combos
        else "DROP",
        axis=1,
    )

    print("\nCategory 3 ADC combination decision summary:")
    print(
        adc_combo_summary[
            ["ADC_CONVERSION_SETTINGS_1", "ADC_CONVERSION_SETTINGS_2", "n_images", "pct_images", "status"]
        ].to_string(index=False, float_format=lambda x: f"{x:0.2f}")
    )

    kept_pct = adc_combo_summary.loc[adc_combo_summary["status"] == "KEEP", "pct_images"].sum()
    drop_pct = adc_combo_summary.loc[adc_combo_summary["status"] == "DROP", "pct_images"].sum()
    kept_n = adc_combo_summary.loc[adc_combo_summary["status"] == "KEEP", "n_images"].sum()
    drop_n = adc_combo_summary.loc[adc_combo_summary["status"] == "DROP", "n_images"].sum()

    print(
        f"\nCategory 3 summary (8-bit images only): keeping {kept_n:,} images ({kept_pct:0.2f}%), "
        f"dropping {drop_n:,} images ({drop_pct:0.2f}%)."
    )

    def print_stats(df):
        print(
            f"Df contains {len(df):,} images from {df['OBSERVATION_ID'].nunique():,} unique observations. Contains 5 ccds: {obs_count(df, 5):,}, 13 ccds: {obs_count(df, 13):,}"
        )

    print("\nOriginal dataset, before category 2b:")
    print_stats(updated_df)

    updated_df = updated_df[cat2b_mask.reindex(updated_df.index, fill_value=False)]
    print("\n1st remove outliers - category 2b:")
    print_stats(updated_df)

    updated_df = updated_df[cat3_mask.reindex(updated_df.index, fill_value=False)]
    print("\n2nd select single value - category 3:")
    print_stats(updated_df)

    def cond_orbit_number(df):
        return df["ORBIT_NUMBER"] <= 100_000  # Max:85,514

    def cond_emission_angle(df):
        return df["EMISSION_ANGLE"] <= 15

    def cond_local_time(df):
        return df["LOCAL_TIME"].between(13, 18)

    def cond_incidence(df):
        return df["INCIDENCE_ANGLE"].between(0, 75)

    def cond_binning(df):
        return df["BINNING"].isin([1, 2])

    def cond_tdi(df):
        return df["TDI"].isin([64, 128])

    def cond_phase(df):
        return df["MISSION_PHASE_NAME"].isin(
            ["PrimarySciencePhase", "ExtendedSciencePhase"]
        )

    def cond_line_samples(df):
        return df["LINE_SAMPLES"].isin([512, 1024])


    conditions = {
        "Orbit number": cond_orbit_number,
        "Emission angle": cond_emission_angle,
        "Local time": cond_local_time,
        "Incidence angle": cond_incidence,
        "Binning": cond_binning,
        "TDI": cond_tdi,
        "Mission phase": cond_phase,
        "Line samples": cond_line_samples,
    }

    print("\n3rd, custom selection - category 1:")
    for label, cond_func in conditions.items():
        mask = cond_func(updated_df)
        filtered_df = updated_df[mask]

        prev_size = len(updated_df)
        new_size = len(filtered_df)

        pct_reduction = (
            (prev_size - new_size) / prev_size * 100 if prev_size != 0 else 0
        )
        unique_obs = filtered_df["OBSERVATION_ID"].nunique()

        cond_v_orig = cond_func(EDR_updated).sum() / len(EDR_updated) * 100

        print(
            f"| {label:<17s} | Decreased df by {pct_reduction:05.2f}% | New size: {new_size:9,d} images, {unique_obs:5,d} unique observations "
            f"| 5 CCD: {obs_count(filtered_df, 5):5,d}, 13 CCD: {obs_count(filtered_df, 13):5,d} | Condition satisfies {cond_v_orig:4.2f}% of original df |"
        )

        updated_df = filtered_df

    img_ids = updated_df["FILE_NAME_SPECIFICATION"].tolist()

    return img_ids, updated_df


def write_observation_list(observations, filename, output_dir=None):
    output_dir = Path(output_dir or (files_dir() / "generated" / "selection"))
    if output_dir.resolve() == files_dir().resolve():
        raise ValueError("Choose a generated-output directory; shipped observation lists are reference files.")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / filename
    output.write_text("".join(f"{obs}\n" for obs in observations), encoding="utf-8")
    print(f"List of {len(observations)} selected observations saved to {output.resolve()}")
    return output


def main(output_dir=None):
    EDR_df = gen_edr_df()
    print(f"bin values 1: {EDR_df['BINNING'].value_counts().index.tolist()}")

    img_ids, updated_df = select_imgs(EDR_df)

    print(f"bin values 2: {updated_df['BINNING'].value_counts().index.tolist()}")

    required_ccds = ["RED3", "RED4", "RED5", "IR10", "BG12"]
    full_ccds = [
        "RED0",
        "RED1",
        "RED2",
        "RED3",
        "RED4",
        "RED5",
        "RED6",
        "RED7",
        "RED8",
        "IR10",
        "IR11",
        "BG12",
        "BG13",
    ]

    required_imgs = [f"{ccd}_{i}" for ccd in required_ccds for i in range(2)]

    obs_files = updated_df.groupby("OBSERVATION_ID")["FILE_NAME_SPECIFICATION"].apply(list)

    # Function to check if all required images are present for an observation
    def has_all_required_images(file_list, required_imgs):
        return all(any(req_img in file for file in file_list) for req_img in required_imgs)


    complete_obs = {
        obs_id
        for obs_id, files in obs_files.items()
        if has_all_required_images(files, required_imgs)
    }

    before_obs = updated_df["OBSERVATION_ID"].nunique()
    updated_df = updated_df[updated_df["OBSERVATION_ID"].isin(complete_obs)]
    print(f"Removed {before_obs - updated_df['OBSERVATION_ID'].nunique()} observations due to incomplete CCDs")
    img_ids = updated_df["FILE_NAME_SPECIFICATION"].tolist()

    obs_ccds = updated_df.groupby("OBSERVATION_ID")["CCD_NAME"].apply(
        lambda ccd_names: set(ccd_names)
    )

    obs_missing_ccds = obs_ccds.apply(
        lambda found_ccds: set(required_ccds) - found_ccds
    )
    missing_counts = obs_missing_ccds.value_counts()

    n_valid_obs = obs_count(updated_df, 5)
    n_full_obs = obs_count(updated_df, 13)

    print(f"\nSelected {len(img_ids):,} images from {updated_df['OBSERVATION_ID'].nunique():,} unique observations:")
    print(f"Observations with 5 key CCDs:              {n_valid_obs:,}")
    print(f"Observations with all 13 CCDs (ex R9):     {n_full_obs:,}\n")
    print(f"bin values 3: {updated_df['BINNING'].value_counts().index.tolist()}")
    print(missing_counts)
    print(f"\nEarliest date:                             {updated_df['OBSERVATION_START_TIME'].min()}")
    print(f"Latest date:                               {updated_df['OBSERVATION_START_TIME'].max()}")

    cross_tab = (
        pd.crosstab(updated_df["TDI"], updated_df["BINNING"], normalize=True) * 100
    ).round(2)
    print("\nConfuguraitons after modificaton:")
    print(cross_tab)

    main_configs = [(1, 128), (2, 64), (2, 128)]

    size_before = updated_df.shape[0]

    # Filter the dataset to only include these configurations
    filtered_df = updated_df[
        updated_df[["BINNING", "TDI"]].apply(tuple, axis=1).isin(main_configs)
    ]

    size_after = filtered_df.shape[0]
    print(f"Limiting to 3 main configs changed df len from {size_before:,} to {size_after:,}")

    # Compute relative frequency of CCD_NAME vs. (BINNING, TDI) configurations
    cross_tab_ccd = (
        pd.crosstab(
            filtered_df["CCD_NAME"],
            [filtered_df["BINNING"], filtered_df["TDI"]],
            normalize="index",
        )
        * 100   # noqa
    ).round(2)

    print("\nCCD_NAME Distribution Across Key Calibration Settings:")
    print(cross_tab_ccd)

    full_obs_list = (
        updated_df.groupby("OBSERVATION_ID")["CCD_NAME"]
        .agg(lambda x: set(full_ccds).issubset(set(x)))
        .pipe(lambda df: df[df].index.tolist())
    )

    random.seed(42)
    print(
        f"\nRandom selection of images to download: {random.sample(list(full_obs_list), 5)}"
    )

    obs_list = full_obs_list
    write_observation_list(obs_list, "all_obs.txt", output_dir)

    # per-observation binning and pick first 100 that match each bin (preserving all_obs order)
    # All images must satisfy the binning condition
    obs_to_bins = (updated_df
                   .groupby("OBSERVATION_ID")["BINNING"]
                   .apply(lambda s: {int(b) for b in s.dropna().unique()})
                   .to_dict())

    bin1_obs_list = [obs for obs in full_obs_list if obs_to_bins.get(obs, set()) == {1}]
    bin2_obs_list = [obs for obs in full_obs_list if obs_to_bins.get(obs, set()) == {2}]

    bin12_requirements = {
        "RED3": 1,
        "RED4": 1,
        "RED5": 1,
        "IR10": 2,
        "BG12": 2,
    }

    df_full = updated_df[updated_df["OBSERVATION_ID"].isin(full_obs_list)]
    df_req = df_full[df_full["CCD_NAME"].isin(bin12_requirements.keys())].copy()
    df_req["BINNING"] = df_req["BINNING"].astype(int)

    # for each (OBSERVATION_ID, CCD_NAME) collect the set of binning values
    ccd_bins = (
        df_req.groupby(["OBSERVATION_ID", "CCD_NAME"])["BINNING"]
        .agg(lambda s: set(s.dropna().unique()))
        .unstack("CCD_NAME")
    )

    bin12_candidates = []
    for obs_id, row in ccd_bins.iterrows():
        ok = True
        for ccd, req_bin in bin12_requirements.items():
            bins = row.get(ccd)
            if not isinstance(bins, set):
                ok = False
                break
            if bins != {req_bin}:
                ok = False
                break
        if ok:
            bin12_candidates.append(obs_id)

    # preserve full_obs_list order and guarantee disjointness with bin1/bin2
    bin12_candidate_set = set(bin12_candidates)
    bin12_obs_list = [
        obs
        for obs in full_obs_list
        if obs in bin12_candidate_set and obs not in bin1_obs_list and obs not in bin2_obs_list]

    write_observation_list(bin1_obs_list, "selected_obs_bin1.txt", output_dir)
    write_observation_list(bin2_obs_list, "selected_obs_bin2.txt", output_dir)
    write_observation_list(bin12_obs_list, "selected_obs_bin12.txt", output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select observations from the HiRISE EDR index using the HiPredict selection filters.")
    parser.add_argument("--output-dir", type=Path)
    main(parser.parse_args().output_dir)
