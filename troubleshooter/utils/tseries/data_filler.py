#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
"""
@date: 2021-07-21 11:07:50
@author: dreamhomes
@description : time series filling methods.
"""
import pandas as pd


def filler_v1(series: pd.DataFrame, a: int, l: int) -> pd.DataFrame:
    """https://dreamhomes.top/posts/202103121506.html
    Note: not used.

    Parameters
    ----------
    series : pd.DataFrame
        input dataframe with missing value.
    a : int
        sliding window
    l : int
        periods length.

    Returns
    -------
    pd.DataFrame
        filled dataframe.
    """

    i = 0
    while i < a:
        point = i
        # print(df['value'][i])
        while series["label"][i] == 1:
            i += 1
            if 0 < i - point < 5:
                y1 = series["value"][point - 1]
                y2 = series["value"][point]
                for j in range(point, i):
                    series["value"][j] = (j - i) / (point - 1 - i) * y1 + (j - point + 1) / (
                        i - point + 1
                    ) * y2
            if i - point >= 5:
                for j in range(point, i):
                    series["value"][j] = (series["value"][j - l] + series["value"][j + l]) / 2
        i += 1

    return series
