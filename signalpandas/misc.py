import numpy.typing as npt
import pandas as pd

from .custom_types import Pandas


def detect_outliers():
    pass


def remove_outliers():
    pass


def find_closest():
    pass


def slice_by():
    pass


def to_matlab():
    pass


def from_matlab():
    pass


def from_c3d():
    pass


def from_n3d():
    pass


def restore_pandas_object(original: Pandas, data: npt.ArrayLike) -> Pandas:
    """Restore pandas objects, also make sure the function does not edit the original data"""
    if isinstance(original, pd.DataFrame):
        return pd.DataFrame(data, original.index, original.columns)
    else:
        return pd.Series(data, index=original.index, name=original.name)
