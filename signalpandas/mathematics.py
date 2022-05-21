from typing import Optional

import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid

from .misc import restore_pandas_object
from .sigtyping import Signal, Number


def time_normalize():
    pass


def transform_standard(data: Signal) -> Signal:
    """
    Transforms data such that the mean is 0 and the standard deviation is 1. Also known as a z-transformation.

    Parameters
    ----------
    data : Signal
        The data that will be standardized

    Returns
    -------
    Signal
        The standardized data

    """
    return transform_center(data) / data.std(axis=0)


def transform_center(data: Signal) -> Signal:
    """
    Centers the data so the mean is 0.

    Parameters
    ----------
    data : Signal
        The data that will be centered

    Returns
    -------
    Signal
        The centered data
    """
    return data - data.mean(axis=0)


def transform_log():
    pass


def transform_reciprocal():
    pass


def transform_power():
    pass


def transform_min_max(data: Signal) -> Signal:
    """
    Perform a min-max transformation so the data ranges from 0 to 1.

    Parameters
    ----------
    data: Signal
        The data that will be transformed.

    Returns
    -------
    Signal:
        The transformed data ranging from 0 to 1.

    """
    data = data + abs(data.min(axis=0))
    data = data / data.max(axis=0)
    return data


def transform_rms(data: Signal) -> Signal:
    """
    Perform a root-mean-square transformation on the data.

    Parameters
    ----------
    data: Signal
        The data that will be transformed.

    Returns
    -------
    Signal:
        The root-mean-square of the original data.

    """
    data = data**2
    data = data.mean(axis=0)
    data = data**0.5
    return data


def differentiate(data: Signal, time: Optional[Signal]):
    data = np.gradient(data, time)
    return data


def integrate(
    data: Signal, time: Optional[Signal] = None, dt: Optional[Number] = None, initial: Number = 0.0
) -> Signal:
    """
    Cumulatively integrate data(time) using the composite trapezoidal rule.

    Parameters
    ----------
    data : Signal
        Data to integrate.
    time : Signal, optional
        The coordinate to integrate along.
    dt : Number, optional
        The time steps to use for integration.
    initial : Number
        Insert this value at the beginning of the returned result. Typically, this value should be 0.

    Returns
    -------
    Signal:
        The integrated data.

    """
    if time is None and dt is None:
        raise TypeError("Either time or dt should be specified.")

    new = data.copy()

    if isinstance(new, pd.Series):
        new = new.values

    if new.ndim == 1:
        new = new[:, None]  # make column vector

    integral = cumulative_trapezoid(new, time, dt, axis=0, initial=initial)

    if isinstance(data, (pd.Series, pd.DataFrame)):
        integral = restore_pandas_object(data, integral)
    elif data.ndim == 1:
        integral = integral.squeeze()  # make vector 1 dimensional again
    return integral
