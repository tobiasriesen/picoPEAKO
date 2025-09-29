import datetime
from itertools import product
import warnings

import numpy as np
import xarray as xr


def enumerate_product(*iterables, repeat=1):
    pools = [list(enumerate(it)) for it in iterables] * repeat
    return product(*pools)


def z2lin(array):
    """
    convert dB values to linear space (for np.array or single number)
    :param array: np.array or single number
    :return:
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = 10**(array/10)
        return out


def lin2z(array):
    """
    convert linear values to dB (for np.array or single number)
    :param array: np.array or single number
    :return:
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = 10 * np.log10(array)
        return out


def format_hms(unixtime):
    """format time stamp in seconds since 01.01.1970 00:00 UTC to HH:MM:SS
    :param unixtime: time stamp (seconds since 01.01.1970 00:00 UTC)
    """
    return datetime.datetime.utcfromtimestamp(unixtime).strftime("%H:%M:%S")


def round_to_odd(f):
    """round to odd number
    :param f: float number to be rounded to odd number
    """
    return round(f) if round(f) % 2 == 1 else round(f) + 1


def argnearest(array, value):
    """
    larda function to find the index of the nearest value in a sorted array, for example time
    or range axis

    :param array: sorted array with values, list and dask arrays will be converted to 1D array
    :param value: value for which to find the nearest neighbor
    :return:
        index of the nearest neighbor in array
    """
    if type(array) in [list, xr.DataArray]:
        array = np.array(array)
    i = np.searchsorted(array, value) - 1

    if not i == array.shape[0] - 1:
        if np.abs(array[i] - value) > np.abs(array[i + 1] - value):
            i = i + 1
    return i


def get_vel_resolution(vel_bins):
    return np.nanmedian(np.diff(vel_bins))


def vel_to_ind(velocities, velbins, fill_value):
    """
    Convert velocities of found peaks to indices

    :param velocities: list of Doppler velocities
    :param velbins: Doppler velocity bins
    :param fill_value: value to be ignored in velocities list
    :return: indices of closest match for each element of velocities in velbins
    """
    indices = np.asarray(
        [argnearest(velbins, v) if ~np.isnan(v) else fill_value for v in velocities]
    )

    return indices


def get_closest_time(time, time_array):
    """"
    :param time: datetime.datetime
    :param time_array: xr.DataArray containing time stamp
    """
    time_array = time_array.values
    if (time_array < 1e9).all() and (time_array > 3e8).all():
        time_array += (
            datetime.datetime(2001, 1, 1) - datetime.datetime(1970, 1, 1)
        ).total_seconds()
    ts = (time - datetime.datetime(1970, 1, 1)).total_seconds()
    return argnearest(time_array, ts)
