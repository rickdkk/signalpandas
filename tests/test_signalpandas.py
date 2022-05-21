import pandas as pd

from signalpandas import initialize, __version__


def test_version():
    assert __version__ == "0.1.3"


def test_accessor():
    initialize()

    assert pd.DataFrame.sig
    assert pd.Series.sig
