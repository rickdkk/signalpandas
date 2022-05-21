from typing import Optional

import numpy as np
import pandas as pd

from .filter import lowpass_filter, highpass_filter, bandpass_filter, bandstop_filter


@pd.api.extensions.register_dataframe_accessor("sig")
class DataFrameSignalAccessor:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def _map(self, function, subset: Optional[list[str]] = None, **kwargs):
        """Developer only. Maps functions to the DataFrame and subsets if necessary."""
        df = self._df.copy()
        subset = df.columns if subset is None else subset
        df[subset] = function(df[subset], **kwargs)
        return df

    def square(self, subset: Optional[list[str]] = None) -> pd.DataFrame:
        return self._map(np.square, subset)

    def sqrt(self, subset: Optional[list[str]] = None) -> pd.DataFrame:
        return self._map(np.sqrt, subset)

    def lowpass_filter(self, order, cutoff, sfreq, subset: Optional[list[str]] = None) -> pd.DataFrame:
        """
        Applies a low-pass zero-lag Butterworth filter to the data. If the data is multidimensional it will apply the
        same filter to each column.

        Parameters
        ----------
        data : Signal
            The data that will be filtered
        order : int
            Order of the filter
        cutoff : float
            Cutoff frequency of the filter
        sfreq : Optional[float]
            Sample frequency of the data, optional if data.attrs["sfreq"] is available
        subset : Optional[list[str]]
            Subset of the data that the operation should be applied to

        Returns
        -------
        data : Signal
            A new object with the filtered data

        See Also
        --------
        lowpass_filter, sosfiltfilt
        """
        return self._map(lowpass_filter, subset, order=order, cutoff=cutoff, sfreq=sfreq)

    def highpass_filter(self, order, cutoff, sfreq, subset: Optional[list[str]] = None) -> pd.DataFrame:
        return self._map(highpass_filter, subset, order=order, cutoff=cutoff, sfreq=sfreq)

    def bandpass_filter(self, order, low, high, sfreq, subset: Optional[list[str]] = None) -> pd.DataFrame:
        return self._map(bandpass_filter, subset, order=order, low=low, high=high, sfreq=sfreq)

    def bandstop_filter(self, order, low, high, sfreq, subset: Optional[list[str]] = None) -> pd.DataFrame:
        return self._map(bandstop_filter, subset, order=order, low=low, high=high, sfreq=sfreq)

    def savgol_filter(self):
        raise NotImplementedError


@pd.api.extensions.register_series_accessor("sig")
class SeriesSignalAccessor:
    def __init__(self, series: pd.Series):
        self._series = series

    def square(self) -> Optional[pd.Series]:
        return self._series.copy() ** 2

    def sqrt(self) -> pd.Series:
        return np.sqrt(self._series)

    def highpass_filter(self, order, cutoff, sfreq) -> pd.Series:
        return highpass_filter(self._series, order=order, cutoff=cutoff, sfreq=sfreq)

    def bandpass_filter(self, order, low, high, sfreq) -> pd.Series:
        return bandpass_filter(self._series, order=order, low=low, high=high, sfreq=sfreq)

    def bandstop_filter(self, order, low, high, sfreq) -> pd.Series:
        return bandstop_filter(self._series, order=order, low=low, high=high, sfreq=sfreq)
