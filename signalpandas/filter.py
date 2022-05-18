from typing import Optional, Union, List

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt
from scipy.signal import savgol_filter as savitzky_golay

from .misc import restore_pandas_object
from .custom_types import Signal
from .validation import has_nulls


def _filter_frame_like(
    data: Signal,
    order: int,
    cutoff: Union[float, List[float]],
    btype: str,
    sfreq: Optional[float],
) -> Signal:
    """Applies a filter to zero-phase Butterworth filter to a numpy or pandas object"""

    if sfreq is None:
        if isinstance(data, (pd.Series, pd.DataFrame)):
            sfreq = data.attrs.get("sfreq")
            if sfreq is None:
                raise ValueError("Didn't specify sample-frequency and object .attrs did not contain it either.")
        else:
            raise ValueError("Didn't specify sample-frequency and object .attrs did not contain it either.")

    if has_nulls(data):
        raise RuntimeError("Data contains missings, filtering will replace the column/all data with missings")

    sos = butter(order, cutoff, btype=btype, output="sos", fs=sfreq)
    output = sosfiltfilt(sos, data, axis=0)

    if isinstance(data, (pd.Series, pd.DataFrame)):
        output = restore_pandas_object(data, output)
    return output


def lowpass_filter(data: Signal, order: int, cutoff: float, sfreq: Optional[float] = None) -> Signal:
    """
    Applies a low-pass zero-lag Butterworth filter to the data. If the data is multidimensional it will apply the same
    filter to each column.

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

    Returns
    -------
    data : Signal
        A new object with the filtered data

    See Also
    --------
    sosfiltfilt
    """
    return _filter_frame_like(data, order, cutoff, "lowpass", sfreq)


def highpass_filter(data: Signal, order: int, cutoff: float, sfreq: Optional[float]) -> Signal:
    """
    Applies a high-pass zero-lag Butterworth filter to the data. If the data is multidimensional it will apply the same
    filter to each column.

    Parameters
    ----------
    data: the data that will be filtered
    order: order of the filter
    cutoff: cutoff frequency of the filter
    sfreq: sample frequency of the data, optional if data.attrs["sfreq"] is available

    Returns
    -------
    A new object with the filtered data
    """
    return _filter_frame_like(data, order, cutoff, "highpass", sfreq)


def bandpass_filter(data: Signal, order: int, low: float, high: float, sfreq: Optional[float]) -> Signal:
    """
    Applies a band-pass zero-lag Butterworth filter to the data. If the data is multidimensional it will apply the same
    filter to each column.

    Parameters
    ----------
    data: the data that will be filtered
    order: order of the filter
    low: lower bound
    high: upper bound
    sfreq: sample frequency of the data, optional if data.attrs["sfreq"] is available

    Returns
    -------
    A new object with the filtered data
    """
    return _filter_frame_like(data, order, [low, high], "bandpass", sfreq)


def bandstop_filter(data: Signal, order: int, low: float, high: float, sfreq: Optional[float]) -> Signal:
    """
    Applies a band-stop zero-lag Butterworth filter to the data. If the data is multidimensional it will apply the same
    filter to each column.

    Parameters
    ----------
    data: the data that will be filtered
    order: order of the filter
    low: lower bound
    high: upper bound
    sfreq: sample frequency of the data, optional if data.attrs["sfreq"] is available

    Returns
    -------
    A new object with the filtered data
    """
    return _filter_frame_like(data, order, [low, high], "bandstop", sfreq)


def savgol_filter(
    data: Signal,
    window_length: int,
    polyorder: int,
    deriv: Optional[int],
    delta: Optional[float],
    mode: Optional[str],
    cval: Union[int, float, complex, None],
) -> Signal:
    """Applies a Savitzky-Golay filter to a numpy or pandas object.

    Parameters
    ----------
    data
    window_length
    polyorder
    deriv
    delta
    mode
    cval

    Returns
    -------
    A new object with the filtered data
    """

    output = savitzky_golay(np.asarray(data), window_length, polyorder, deriv, delta, 0, mode, cval)

    if isinstance(data, pd.DataFrame):
        output = pd.DataFrame(output, data.index, data.columns)
    elif isinstance(data, pd.Series):
        output = pd.Series(output, index=data.index, name=data.name)

    return output
