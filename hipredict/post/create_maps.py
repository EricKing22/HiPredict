import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from pyproj import CRS, Transformer
import pandas as pd

from hipredict.data.load_metadata import gen_edr_df
from hipredict.paths import files_dir

DATA_RECORD = str(files_dir() / "data_record_alls.csv")

MARS_A = 3396190.0      # Equatorial radius in meters
MARS_B = 3376200.0
CENTRAL_LON = 180.0


def create_location_df():
    edr_df = gen_edr_df()
    data_record = pd.read_csv(DATA_RECORD)

    df = edr_df[edr_df['OBSERVATION_ID'].isin(data_record['Observation'].values)].copy()
    df = df.groupby('OBSERVATION_ID', as_index=False).first()

    # Keep longitudes in [0, 360]
    df = df[(df['IMAGE_CENTER_LONGITUDE'] >= 0.0) & (df['IMAGE_CENTER_LONGITUDE'] <= 360.0)]
    df['IMAGE_CENTER_LATITUDE'] = df['IMAGE_CENTER_LATITUDE'].clip(-90, 90)

    # Basic cleaning
    cols = ['IMAGE_CENTER_LONGITUDE', 'IMAGE_CENTER_LATITUDE', 'ORBIT_NUMBER']
    df = df.dropna(subset=cols)

    return df, data_record


def make_globe_and_transform(central_lon=CENTRAL_LON, a=MARS_A, b=MARS_B):
    globe = ccrs.Globe(semimajor_axis=a, semiminor_axis=b, ellipse=None)
    geo_crs = CRS.from_proj4(f"+proj=longlat +a={a} +b={b} +no_defs")


    proj = ccrs.Robinson(central_longitude=central_lon, globe=globe)
    robin_crs = CRS.from_proj4(f"+proj=robin +lon_0={central_lon} +a={a} +b={b} +units=m +no_defs")
    to_xy = Transformer.from_crs(geo_crs, robin_crs, always_xy=True)
    return globe, proj, to_xy


def normalize_reducer(reducer):
    if reducer is None or reducer == "count":
        return None
    if callable(reducer):
        return reducer
    lut = {
        "mean": np.mean,
        "median": np.median,
        "sum": np.sum,
        "min": np.min,
        "max": np.max
    }
    if reducer not in lut:
        raise ValueError("Reducer must be one of: count, mean, median, sum, min, max, or a callable")
    return lut[reducer]


def make_axes(proj, figsize=(11, 5.5)):
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=proj)
    ax.set_global()
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.3, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    return fig, ax


def hexbin_map(ax, x, y, gridsize=160, value=None, reducer=None, mincnt=1, vmin=None, vmax=None, cmap=None, col=None):
    reduce_fn = normalize_reducer(reducer)
    if reduce_fn is None:
        hb = ax.hexbin(x, y, gridsize=gridsize, mincnt=mincnt, linewidths=0.0, vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)
        cb_label = "Observations per hex"
    else:
        hb = ax.hexbin(x, y, C=value, reduce_C_function=reduce_fn, gridsize=gridsize, mincnt=mincnt, linewidths=0.0, vmin=vmin, vmax=vmax, cmap=cmap, rasterized=True)
        if col is None:
            cb_label = "Aggregate per hex"
        else:
            cb_label = f"{reducer} {col} per hex"
    return hb, cb_label


def finalize_and_save(fig, ax, hb, cb_label, title, out_path, dpi=300):
    cb = plt.colorbar(hb, ax=ax, shrink=0.75, pad=0.02)
    cb.set_label(cb_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()
    print(f"Saved map to {out_path}")


def plot_density(df, out_path, lon_col='IMAGE_CENTER_LONGITUDE', lat_col='IMAGE_CENTER_LATITUDE', gridsize=160, central_lon=CENTRAL_LON):
    _, proj, to_xy = make_globe_and_transform(central_lon=central_lon)
    lons = df[lon_col].to_numpy()
    lats = df[lat_col].clip(-90, 90).to_numpy()
    x, y = to_xy.transform(lons, lats)

    fig, ax = make_axes(proj)
    hb, cb_label = hexbin_map(ax, x, y, gridsize=gridsize, reducer="count", mincnt=1)
    title = "Mars observation density"
    finalize_and_save(fig, ax, hb, cb_label, title, out_path)


def plot_aggregate(df, out_path, value_col, lon_col='IMAGE_CENTER_LONGITUDE', lat_col='IMAGE_CENTER_LATITUDE',
                   reducer='mean', gridsize=160, central_lon=CENTRAL_LON, vmin=None, vmax=None, cmap='magma', title=None, cb_label_input=None):
    _, proj, to_xy = make_globe_and_transform(central_lon=central_lon)
    lons = df[lon_col].to_numpy()
    lats = df[lat_col].clip(-90, 90).to_numpy()
    vals = df[value_col].to_numpy()
    mask = np.isfinite(lons) & np.isfinite(lats) & np.isfinite(vals)
    x, y = to_xy.transform(lons[mask], lats[mask])

    fig, ax = make_axes(proj)
    hb, cb_label = hexbin_map(ax, x, y, gridsize=gridsize, value=vals[mask], reducer=reducer, mincnt=1, vmin=vmin, vmax=vmax, col=value_col, cmap=cmap)
    title = f"{(reducer if isinstance(reducer, str) else 'aggregate').capitalize()} {value_col}" if title is None else title
    finalize_and_save(fig, ax, hb, (cb_label if cb_label_input is None else cb_label_input), title, out_path)


def main():
    df, _ = create_location_df()
    print(f"Loaded {len(df)} observations from data record")

    # ---------------------------------------------------------------------
    # Observation density
    # ---------------------------------------------------------------------

    plot_density(
        df=df,
        out_path=str(files_dir() / "maps/mars_observation_density.png"),
        gridsize=100,
    )

    # ---------------------------------------------------------------------
    # Metadata
    # ---------------------------------------------------------------------

    plot_aggregate(
        df=df,
        value_col='INCIDENCE_ANGLE',
        out_path=str(files_dir() / "maps/mars_incidence_angle_mean.png"),
        reducer='mean',
        gridsize=100,
    )

    plot_aggregate(
        df=df,
        value_col='SPACECRAFT_ALTITUDE',
        out_path=str(files_dir() / "maps/mars_spacecraft_altitude_mean.png"),
        reducer='mean',
        gridsize=100,
    )

    plot_aggregate(
        df=df,
        value_col='TDI',
        out_path=str(files_dir() / "maps/mars_spacecraft_tdi_mean.png"),
        reducer='mean',
        gridsize=100,
    )

    # ---------------------------------------------------------------------
    # Loss
    # ---------------------------------------------------------------------

    loss_df = pd.read_csv(str(files_dir() / "evals/hipredict_evals.csv"))

    coords = df[['OBSERVATION_ID', 'IMAGE_CENTER_LATITUDE', 'IMAGE_CENTER_LONGITUDE']]
    loss_df = loss_df.merge(
        coords,
        how='left',
        left_on='Observation',      # Column from the left df (loss_df)
        right_on='OBSERVATION_ID'   # Column from the right df (coords)
    )

    loss_df.drop(columns=['OBSERVATION_ID'], inplace=True)

    plot_aggregate(
        df=loss_df,
        value_col='CNN Loss',
        out_path=str(files_dir() / "maps/mars_evals_mean.png"),
        reducer='mean',
        gridsize=100,
    )


if __name__ == "__main__":
    main()
