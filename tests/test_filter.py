import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from signalpandas import initialize
from signalpandas.filter import lowpass_filter

initialize()


def test_lowpass_filter():
    sfreq = 2000
    t = np.arange(0, 10, 1 / sfreq)

    sine_low = np.sin(2 * np.pi * t * 5)  # 5Hz
    sine_high = np.sin(2 * np.pi * t * 200)  # 200Hz

    signal = sine_low + sine_high

    filtered_sine = lowpass_filter(signal, 8, 30, sfreq)

    # Test 1d case
    assert np.max(np.abs(filtered_sine[500:-500] - sine_low[500:-500])) < 1e-4

    # Test multidimensional case
    signal2d = np.column_stack([signal, signal])
    sine_low2d = np.column_stack([sine_low, sine_low])

    assert np.max(np.abs(lowpass_filter(signal2d, 8, 30, sfreq)[500:-500] - sine_low2d[500:-500])) < 1e-4

    # Test dataframe accessor
    signal2d_df = pd.DataFrame(signal2d)
    assert isinstance(lowpass_filter(signal2d_df, 8, 30, sfreq), pd.DataFrame)
    assert (lowpass_filter(signal2d_df, 8, 30, sfreq).iloc[500:-500, :] - sine_low2d[500:-500]).abs().max().max() < 1e-4

    assert_frame_equal(signal2d_df.sig.lowpass_filter(8, 30, sfreq), lowpass_filter(signal2d_df, 8, 30, sfreq))

    # Test subsetting
    subset_df = pd.DataFrame({0: filtered_sine, 1: signal})
    assert_frame_equal(signal2d_df.sig.lowpass_filter(8, 30, sfreq, subset=[0]), subset_df)

    # Test errors
    with pytest.raises(ValueError):
        lowpass_filter(signal, 8, 30)

    with pytest.raises(RuntimeError):  # We want to explicitly throw an error to stop the propagation of NaNs
        signal_with_nan = signal.copy()
        signal_with_nan[10] = np.NaN
        lowpass_filter(signal_with_nan, 8, 30, sfreq)
