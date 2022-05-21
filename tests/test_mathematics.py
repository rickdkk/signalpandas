import numpy as np
import pandas as pd

from signalpandas.mathematics import transform_standard, integrate, transform_min_max


def test_standardize():
    signal = np.random.normal(loc=10, scale=3, size=100)
    signal_stand = transform_standard(signal)

    assert np.allclose(signal_stand.mean(axis=0), 0.0)
    assert np.allclose(signal_stand.std(axis=0), 1.0)
    assert isinstance(signal_stand, np.ndarray)

    df = pd.DataFrame({"a": signal, "b": signal * 2})
    df_stand = transform_standard(df)

    assert np.allclose(df_stand.mean(axis=0), 0.0)
    assert np.allclose(df_stand.std(axis=0), 1.0)
    assert isinstance(df_stand, pd.DataFrame)

    series_stand = transform_standard(df["a"])

    assert np.allclose(series_stand.mean(axis=0), 0.0)
    assert np.allclose(series_stand.std(axis=0), 1.0)
    assert isinstance(series_stand, pd.Series)


def test_integrate():
    time = np.arange(0, 100, 0.01)
    signal = np.cos(time)
    integral = integrate(signal, time)

    assert signal is not integral  # check if the function is pure
    assert np.allclose(integral, np.sin(time))
    assert isinstance(integral, np.ndarray)

    df = pd.DataFrame({"a": signal, "b": signal})
    df.set_index(time)
    # df_int = integrate(df, df.index.values)

    # assert df_int is not df
    # assert np.allclose(df_int, np.sin(time))


def test_min_max_scale():
    time = np.arange(0, 100, 0.01)
    signal = np.cos(time)

    new_signal = transform_min_max(signal)
    assert new_signal.min() == 0
    assert new_signal.max() == 1
