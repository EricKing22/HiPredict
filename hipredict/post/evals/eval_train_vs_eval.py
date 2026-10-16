import pandas as pd
import matplotlib.pyplot as plt
import os

from hipredict.data.load_metadata import gen_edr_df


def compare_train_eval_sets(SAVE_DIR, TRAIN_DR, EVAL_DR):
    """
    What do we want to compare?
    - Dates
    - Locations
    - Brightness/TDI
    - Binning

    """
    train_obs = pd.read_csv(TRAIN_DR)['Observation'].unique()
    eval_obs = pd.read_csv(EVAL_DR)['Observation'].unique()
    edr_df = gen_edr_df()

    train_edr_df = edr_df[edr_df['OBSERVATION_ID'].isin(train_obs)]
    eval_edr_df = edr_df[edr_df['OBSERVATION_ID'].isin(eval_obs)]

    train_dates = pd.to_datetime(train_edr_df['START_TIME'])
    eval_dates = pd.to_datetime(eval_edr_df['START_TIME'])
    tot_dates = pd.to_datetime(edr_df['START_TIME'])

    train_lats = train_edr_df['IMAGE_CENTER_LATITUDE']
    eval_lats = eval_edr_df['IMAGE_CENTER_LATITUDE']
    tot_lats = edr_df['IMAGE_CENTER_LATITUDE']

    train_lons = train_edr_df['IMAGE_CENTER_LONGITUDE']
    eval_lons = eval_edr_df['IMAGE_CENTER_LONGITUDE']
    tot_lons = edr_df['IMAGE_CENTER_LONGITUDE']

    train_tdis = train_edr_df['TDI']
    eval_tdis = eval_edr_df['TDI']
    tot_tdis = edr_df['TDI']

    train_bins = train_edr_df['BINNING']
    eval_bins = eval_edr_df['BINNING']
    tot_bins = edr_df['BINNING']

    def get_pcts(series, buckets):
        counts = series.value_counts(normalize=True) * 100
        return {str(b): f"{counts.get(b, 0):.1f}%" for b in buckets}, f"{100 - sum(counts.get(b, 0) for b in buckets):.1f}%"

    # Table 1: Ranges
    print(f"\n{'Metric':<10} | {'Train Max':<12} | {'Train Min':<12} | {'Eval Max':<12} | {'Eval Min':<12} | {'Tot Max':<12} | {'Tot Min':<12}")
    print("-" * 105)
    print(f"{'Date':<10} | {str(train_dates.max().date()):<12} | {str(train_dates.min().date()):<12} | {str(eval_dates.max().date()):<12} | {str(eval_dates.min().date()):<12} | {str(tot_dates.max().date()):<12} | {str(tot_dates.min().date()):<12}")    # noqa
    print(f"{'Lat':<10} | {train_lats.max():<12.2f} | {train_lats.min():<12.2f} | {eval_lats.max():<12.2f} | {eval_lats.min():<12.2f} | {tot_lats.max():<12.2f} | {tot_lats.min():<12.2f}")
    print(f"{'Lon':<10} | {train_lons.max():<12.2f} | {train_lons.min():<12.2f} | {eval_lons.max():<12.2f} | {eval_lons.min():<12.2f} | {tot_lons.max():<12.2f} | {tot_lons.min():<12.2f}")

    # Table 2: TDI Percentages
    t_tdi_p, t_tdi_o = get_pcts(train_tdis, [128, 64, 32])
    e_tdi_p, e_tdi_o = get_pcts(eval_tdis, [128, 64, 32])
    tot_tdi_p, tot_tdi_o = get_pcts(tot_tdis, [128, 64, 32])
    print(f"\n{'TDI %':<10} | {'128':<10} | {'64':<10} | {'32':<10} | {'Other':<10}")
    print("-" * 55)
    print(f"{'Train':<10} | {t_tdi_p['128']:<10} | {t_tdi_p['64']:<10} | {t_tdi_p['32']:<10} | {t_tdi_o:<10}")
    print(f"{'Eval':<10} | {e_tdi_p['128']:<10} | {e_tdi_p['64']:<10} | {e_tdi_p['32']:<10} | {e_tdi_o:<10}")
    print(f"{'Tot':<10} | {tot_tdi_p['128']:<10} | {tot_tdi_p['64']:<10} | {tot_tdi_p['32']:<10} | {tot_tdi_o:<10}")

    # Table 3: Binning Percentages
    t_bin_p, t_bin_o = get_pcts(train_bins, [1, 2])
    e_bin_p, e_bin_o = get_pcts(eval_bins, [1, 2])
    tot_bin_p, tot_bin_o = get_pcts(tot_bins, [1, 2])
    print(f"\n{'Bin %':<10} | {'1':<10} | {'2':<10} | {'Other':<10}")
    print("-" * 45)
    print(f"{'Train':<10} | {t_bin_p['1']:<10} | {t_bin_p['2']:<10} | {t_bin_o:<10}")
    print(f"{'Eval':<10} | {e_bin_p['1']:<10} | {e_bin_p['2']:<10} | {e_bin_o:<10}")
    print(f"{'Tot':<10} | {tot_bin_p['1']:<10} | {tot_bin_p['2']:<10} | {tot_bin_o:<10}\n")

    # FIGURE: Timeline comparison
    plt.hist(train_dates, bins=30, alpha=0.5, label='Train', color='blue')
    plt.hist(eval_dates, bins=30, alpha=0.5, label='Eval', color='orange')
    plt.title('Observation Timeline')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    timeline_fig_path = os.path.join(SAVE_DIR, "EVALS_train_timeline_comparison.png")
    plt.savefig(timeline_fig_path)
    plt.clf()
    print(f"Saved timeline figure to {timeline_fig_path}")

    # FIGURE: Spatial distribution map
    plt.scatter(train_lons, train_lats, s=1, alpha=0.2, label='Train', color='blue')
    plt.scatter(eval_lons, eval_lats, s=1, alpha=0.8, label='Eval', color='red')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Spatial Distribution')
    plt.legend(markerscale=10)
    plt.tight_layout()
    spatial_fig_path = os.path.join(SAVE_DIR, "EVALS_train_spatial_comparison.png")
    plt.savefig(spatial_fig_path)
    plt.clf()
    print(f"Saved spatial distribution figure to {spatial_fig_path}")

    return
