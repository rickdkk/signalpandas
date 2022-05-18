from typing import Any

import numpy as np

from .custom_types import Signal


def check_time():
    pass


def check_continuity():
    pass


def has_nulls(data: Signal) -> Any:
    """Returns True if there is any sample that is equal to NaN or Inf"""
    return np.any(~np.isfinite(data))
