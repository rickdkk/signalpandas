import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal


def test_series_square():
    series = pd.Series([0., 1, 2, 3, 4, 5])
    series2 = pd.Series([0., 1, 4, 9, 16, 25])

    assert_series_equal(series.sig.square(), series2)


def test_dataframe_square():
    df0 = pd.DataFrame({"a": [0., 1, 2, 3, 4, 5], "b": [0., 1, 2, 3, 4, 5]})
    df1 = pd.DataFrame({"a": [0., 1, 4, 9, 16, 25], "b": [0., 1, 4, 9, 16, 25]})
    df2 = pd.DataFrame({"a": [0., 1, 2, 3, 4, 5], "b": [0., 1, 4, 9, 16, 25]})

    assert_frame_equal(df0.sig.square(), df1)
    assert_frame_equal(df0.sig.square(subset=["b"]), df2[["b"]])


def test_series_sqrt():
    series = pd.Series([0., 1, 4, 9, 16, 25])
    series2 = pd.Series([0., 1, 2, 3, 4, 5])

    assert_series_equal(series.sig.sqrt(), series2)


def test_dataframe_sqrt():
    df0 = pd.DataFrame({"a": [4., 9, 16, 25, 36, 49], "b": [4., 9, 16, 25, 36, 49]})
    df1 = pd.DataFrame({"a": [2., 3, 4, 5, 6, 7], "b": [2., 3, 4, 5, 6, 7]})
    df2 = pd.DataFrame({"a": [4., 9, 16, 25, 36, 49], "b": [2., 3, 4, 5, 6, 7]})

    assert_frame_equal(df0.sig.sqrt(), df1)
    assert_frame_equal(df0.sig.sqrt(subset=["b"]), df2[["b"]])
