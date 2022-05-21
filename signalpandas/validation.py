import numpy as np

from .sigtyping import Signal


def check_time():
    pass


def check_continuity():
    pass


def has_nulls(data: Signal) -> bool:
    """Returns True if there is any sample that is equal to NaN or Inf"""
    return bool(np.any(~np.isfinite(data)))
