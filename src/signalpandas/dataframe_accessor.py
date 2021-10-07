from typing import Optional

import numpy as np
import pandas as pd

from .filter import lowpass_filter, highpass_filter, bandpass_filter, bandstop_filter


@pd.api.extensions.register_dataframe_accessor("sig")
class DataFrameSignalAccessor:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def _map(self, function, subset: Optional[list[str]] = None, **kwargs):
        df = self._df.copy()
        subset = df.columns if subset is None else subset

        return function(df[subset], **kwargs)

    def square(self, subset: Optional[list[str]] = None) -> pd.DataFrame:
        return self._map(np.square, subset)

    def sqrt(self, subset: Optional[list[str]] = None) -> pd.DataFrame:
        return self._map(np.sqrt, subset)

    def lowpass_filter(self, order, cutoff, sfreq, subset: Optional[list[str]] = None) -> pd.DataFrame:
        df = self._df.copy()

        subset = df.columns if subset is None else subset
        df[subset] = lowpass_filter(df[subset], order, cutoff, sfreq)
        return df

    def highpass_filter(self, order, cutoff, sfreq, subset: Optional[list[str]] = None) -> pd.DataFrame:
        df = self._df.copy()

        subset = df.columns if subset is None else subset
        df[subset] = highpass_filter(df[subset], order, cutoff, sfreq)
        return df

    def bandpass_filter(self, order, low, high, sfreq, subset: Optional[list[str]] = None) -> pd.DataFrame:
        df = self._df.copy()

        subset = df.columns if subset is None else subset
        df[subset] = bandpass_filter(df[subset], order, low, high, sfreq)
        return df

    def bandstop_filter(self, order, low, high, sfreq, subset: Optional[list[str]] = None) -> pd.DataFrame:
        """Docstring"""
        df = self._df.copy()

        subset = df.columns if subset is None else subset
        df[subset] = bandstop_filter(df[subset], order, low, high, sfreq)
        return df


@pd.api.extensions.register_series_accessor("sig")
class SeriesSignalAccessor:
    def __init__(self, series: pd.Series):
        self._series = series

    def square(self) -> Optional[pd.Series]:
        return self._series.copy() ** 2

    def sqrt(self) -> pd.Series:
        return np.sqrt(self._series)
