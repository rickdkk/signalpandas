from scipy.integrate import cumulative_trapezoid

from .sigtyping import Signal


def time_normalize():
    pass


def standardize(data: Signal) -> Signal:
    """
    Transforms data such that the mean is 0 and the standard deviation is 1

    Parameters
    ----------
    data : Signal
        The data that will be standardized

    Returns
    -------
    data : Signal
        The standardized data

    """
    return center(data) / data.std(axis=0)


def center(data: Signal) -> Signal:
    """Centers the data so the mean is 0"""
    return data - data.mean(axis=0)


def log_transform():
    pass


def reciprocal_transform():
    pass


def power_transform():
    pass


def min_max_scale():
    pass


def root_mean_square():
    pass


def differentiate():
    pass


def integrate(data: Signal, time: Signal) -> Signal:
    if data.ndim > 1:
        data = data.T

    integral = cumulative_trapezoid(data, time, axis=0, initial=0.0)

    return integral.T
