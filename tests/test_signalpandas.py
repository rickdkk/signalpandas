import pandas as pd

from signalpandas import __version__


def test_version():
    assert __version__ == "0.1.3"


def test_accessor():
    assert pd.DataFrame.sig
    assert pd.Series.sig
