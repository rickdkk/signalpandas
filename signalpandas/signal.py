import pandas as pd
import scipy.signal

from .misc import restore_pandas_object
from .sigtyping import Signal


def detrend(data: Signal, breakpoints: list[int]) -> Signal:
    new_data = scipy.signal.detrend(data, axis=0, bp=breakpoints)
    if isinstance(data, (pd.Series, pd.DataFrame)):
        new_data = restore_pandas_object(data, new_data)
    return new_data


def resample():
    pass


def time_normalize():
    pass


def find_peaks():
    pass


def find_zero_crossings():
    pass


def fft():
    pass
