"""Observation-selection filter used to build the HiPredict dataset."""
import numpy as np


def zscore_filter(df, col, thresh=2):
    """Return a mask that rejects upper outliers of ``col``.

    The comparison is one-sided: rows more than ``thresh`` standard deviations
    above the mean are rejected, while low values are kept. This is the filter
    used to select the HiPredict dataset, so it is kept as is; it is not a
    symmetric z-score filter.
    """
    return np.abs((df[col] - df[col].mean()) <= (thresh * df[col].std()))
