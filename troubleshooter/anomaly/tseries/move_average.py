#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@date: 2021-07-21 17:35:59
@author: dreamhomes
@description : Exponential weighted moving average.
"""
import numpy as np


def move_average(values: np.array, s: int = 10) -> np.array:
    """Simple moving average.

    Parameters
    ----------
    values : np.array
        input data.
    s : int, optional
        sliding window, by default 21

    Returns
    -------
    np.array
        output data.
    """
    ret = np.cumsum(values, dtype=float)
    ret[s:] = ret[s:] - ret[:-s]
    ret[s:] = ret[s:] / s
    for i in range(1, s):
        ret[i] /= i + 1

    return ret


def ewma(values: np.array, alpha: float = 0.4, s: int = 10) -> np.array:
    """Exponential weighted moving average.

    Parameters
    ----------
    values : np.array
        input values
    alpha : float
        decay factor
    s : int
        sliding window

    Returns
    -------
    np.array
        ewma predict values.
    """

    def _ewma_next(i):
        numerator = 0.0
        denominator = 0.0
        for j in range(s):
            if i - j - 1 >= 0:
                numerator += ((1 - alpha) ** j) * values[i - j - 1]
                denominator += (1 - alpha) ** j
        return numerator / (denominator + 1e-8) if i != 0 else values[0]

    return [_ewma_next(i) for i in range(values.size)]
