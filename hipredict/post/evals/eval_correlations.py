import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
from sklearn.ensemble import RandomForestRegressor

from hipredict.data.load_metadata import gen_edr_df
from hipredict.post.evals.eval_meta_bins import observation_metadata_table


def corr_evals(results_df, val_dataset):
    # ===========================================================================================
    # CORRELATIONS WITH METADATA
    # ==========================================================================================
    print("\n" + "=" * 80)
    print("CORRELATION INVESTIGATIONS (DN space)")
    print("=" * 80)
    EDR_df = gen_edr_df()

    df_temp = pd.DataFrame()
    df_temp['Variable'] = sorted(EDR_df.select_dtypes(include='number').columns)
    n_cols = len(list(EDR_df.columns))
    print(f"Full list of {n_cols} metadata columns: {list(EDR_df.columns)}")
    df_temp['Min'] = df_temp['Variable'].apply(lambda x: EDR_df[x].min())
    df_temp['Max'] = df_temp['Variable'].apply(lambda x: EDR_df[x].loc[EDR_df[x] != -9998.0].max())
    df_temp['Contain Missing'] = df_temp['Min'].apply(lambda x: 'Yes' if x == -9998.0 else 'No')
    df_temp['n unique'] = df_temp['Variable'].apply(lambda x: EDR_df[x].nunique())

    def _varies_by_ccd(col):
        s = EDR_df.loc[EDR_df[col] != -9998.0].groupby(['OBSERVATION_ID', 'CCD_NAME'])[col].first()
        g = s.groupby('OBSERVATION_ID').nunique()
        return 'Yes' if (g > 1).any() else 'No'

    df_temp['Varies by CCD'] = df_temp['Variable'].apply(_varies_by_ccd)

    df_temp = df_temp[~df_temp['Variable'].isin(['LOOKUP_TABLE_K_VALUE', 'LOOKUP_TABLE_MEDIAN'])]

    for _, row in df_temp.iterrows():
        if row['Min'] == -9998.0:
            valid_values = EDR_df[EDR_df[row['Variable']] != -9998.0][row['Variable']]
            if not valid_values.empty:
                new_min = valid_values.min()
                df_temp.loc[df_temp['Variable'] == row['Variable'], 'Min'] = new_min

    df_temp['Range'] = df_temp['Max'] - df_temp['Min']


    # ----- Print numeric columns with varying values -----
    df_temp['Mean'] = df_temp['Variable'].apply(lambda x: EDR_df[x].loc[EDR_df[x] != -9998.0].mean())
    df_temp['Median'] = df_temp['Variable'].apply(lambda x: EDR_df.loc[EDR_df[x] != -9998.0, x].median())
    df_temp['Std Dev'] = df_temp['Variable'].apply(lambda x: (EDR_df.loc[EDR_df[x] != -9998.0, x]).std())
    df_temp['Count'] = df_temp['Variable'].apply(lambda x: (EDR_df[x] != -9998.0).sum())    # Count of non-missing values

    loss_col = 'CNN Loss'
    meta_cols = list(df_temp['Variable'])
    print(f"Meta cols: {meta_cols}")
    cols_to_merge = meta_cols + ['OBSERVATION_ID']

    merged_df = results_df.merge(
        EDR_df[cols_to_merge],
        how='left',
        left_on='Observation',
        right_on='OBSERVATION_ID'
    )
    merged_df.drop(columns=['OBSERVATION_ID'], inplace=True, errors='ignore')

    cols_to_check = [loss_col] + meta_cols
    valid_rows = merged_df[cols_to_check].dropna()

    unique_counts = valid_rows.nunique()
    cols_to_drop = unique_counts[unique_counts == 1].index.tolist()       # type:ignore
    if cols_to_drop:
        print(f"Dropping columns with constant values for correlation analysis: {cols_to_drop}")
        valid_rows = valid_rows.drop(columns=cols_to_drop)

    correlation_matrix = valid_rows.corr()
    loss_correlations = correlation_matrix[loss_col].drop(index=loss_col)
    abs_correlations = loss_correlations.abs().sort_values(ascending=False)

    print(f"\nCorrelation of metadata variables with {loss_col}:")
    for feature, corr_value in abs_correlations.items():
        print(f"- {feature:<30}: Correlation = {corr_value:.4f}")

    top_5_features = abs_correlations.head(5).index
    for feature in top_5_features:
        print(f"\n{'='*70}")
        print(f"ANALYSIS: {feature} (Abs Corr: {abs_correlations[feature]:.4f})")
        print(f"{'='*70}")

        unique_count = valid_rows[feature].nunique()

        if unique_count <= 7:
            # --- METHOD A: CATEGORICAL GROUPING ---
            stats = valid_rows.groupby(feature, observed=False)[loss_col].agg(['count', 'mean', 'std']).sort_values('count', ascending=False)
            print(f"Type: Categorical/Discrete ({unique_count} unique values)")
            print(stats)
        else:
            # --- METHOD B: CONTINUOUS BINNING (QUINTILES) ---
            # We split the data into 5 groups with an equal number of samples in each
            try:
                # qcut creates bins based on sample quantiles
                binned_data = pd.qcut(valid_rows[feature], q=5, duplicates='drop')
                stats = valid_rows.groupby(binned_data, observed=False)[loss_col].agg(['count', 'mean', 'std'])
                print("Type: Continuous (Binned into Quantiles)")
                print(stats)
            except Exception as e:
                print(f"Could not bin {feature}: {e}")

        if not stats.empty:
            ratio = stats['mean'].max() / (stats['mean'].min() + 1e-9)
            print(f"\nInsight: The 'worst' group has {ratio:.2f}x the error of the 'best' group.")

    # Print some ranges
    val_obs_ids = {val_dataset[i]["obs_id"] for i in range(len(val_dataset))}       # type:ignore
    val_df = EDR_df[EDR_df["OBSERVATION_ID"].isin(val_obs_ids)].copy()

    want_cols = [
        "LOCAL_TIME",
        "SPACECRAFT_ALTITUDE",
        "SCAN_EXPOSURE_DURATION",
        "ORBIT_NUMBER",
        "IMAGE_CENTER_LATITUDE",
        "IMAGE_CENTER_LONGITUDE",
    ]

    print("\nDesired metadata ranges from EDR_df (ignoring -9998.0):")
    for c in want_cols:
        if c not in val_df.columns:
            print(f"- {c}: MISSING from EDR_df")
            continue

        s_valid = val_df.loc[val_df[c] != -9998.0, c]
        if s_valid.empty:
            print(f"- {c}: all values are -9998.0 (missing) within val_dataset obs")
        else:
            print(f"- {c}: min={s_valid.min()} max={s_valid.max()} n_valid={len(s_valid)}")

    print("", flush=True)

    # SHAP
    plt.switch_backend('agg')
    print("\n" + "=" * 70)
    print("STARTING SHAP GLOBAL EXPLANATION")
    print("=" * 70)

    X = valid_rows.drop(columns=[loss_col])
    y = valid_rows[loss_col]

    # Random Forest handles non-linearities better than Pearson
    regressor = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    regressor.fit(X, y)

    # Calculate SHAP values
    # TreeExplainer is optimized for Random Forests
    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X)

    # Extract numerical SHAP importance for the logs
    # SHAP importance is the mean absolute SHAP value for each feature
    importances = np.abs(shap_values).mean(0)
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values(by='importance', ascending=False)

    print("\nSHAP Global Feature Importance (Avg impact on model output magnitude):")
    for index, row in feature_importance_df.iterrows():
        print(f"- {row['feature']:<30}: {row['importance']:.6f}")

    # Per-observation metadata table for the binned-NMAE plots, built here because
    # EDR_df is already loaded (the CSV is ~1.7 GB and must not be re-read)
    feature_ranking = list(feature_importance_df['feature'])
    meta_obs_tbl = observation_metadata_table(EDR_df, feature_ranking)

    return {
        'shap_values': shap_values,
        'shap_x': X,
        'feature_ranking': feature_ranking,
        'meta_obs_tbl': meta_obs_tbl,
    }
