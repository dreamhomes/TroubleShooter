#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File        : spetral_residual.py
@Time        : 2021-07-14 20:12:11
@Author      : dreamhomes
@Description : Time-Series Anomaly Detection Service at Microsoft.
"""

import pandas as pd
import numpy as np

EPS = 1e-8
TIMESTAMP = "time"
IS_ANOMALY = "isAnomaly"
VALUE = "value"
ANOMALY_SCORE = "anomalyScore"
MAP = "map"

class SpectralResidual:
    def __init__(
        self, series: pd.DataFrame, score_window: int, mag_window: int = 3, threshold: float = 3.0
    ):
        """Spectral Residual algorithm parameters.

        Parameters
        ----------
        series : pd.DataFrame
            input dataframe, columns [timestamp, value]
        score_window : int
            anomaly score window.
        mag_window : int, optional
            anomaly score window, by default 3
        threshold : float, optional
            anomaly threshold, by default 3.0
        """
        self._series = series
        self._threshold = threshold
        self._mag_window = mag_window
        self._score_window = score_window
        self._anomaly_frame = None

    def detect(self):
        if self._anomaly_frame is None:
            self._anomaly_frame = self._detect(self._series)

        return self._anomaly_frame

    def _detect(self, series: pd.DataFrame) -> pd.DataFrame:
        values = series[VALUE].values
        extended_series = self.extend_series(values)
        saliency_map = self.spectral_residual_transform(extended_series)
        anomaly_scores = self.generate_spectral_score(saliency_map)
        anomaly_frame = pd.DataFrame(
            {
                TIMESTAMP: series[TIMESTAMP].values,
                VALUE: values,
                MAP: saliency_map[: len(values)],
                ANOMALY_SCORE: anomaly_scores[: len(values)],
            }
        )
        anomaly_frame[IS_ANOMALY] = np.where(
            anomaly_frame[ANOMALY_SCORE] > self._threshold, True, False
        )

        return anomaly_frame

    def generate_spectral_score(self, mags):
        ave_mag = self.average_filter(mags, n=self._score_window)
        safeDivisors = np.clip(ave_mag, EPS, ave_mag.max())

        raw_scores = np.abs(mags - ave_mag) / safeDivisors
        scores = np.clip(raw_scores, 0, 10)

        return scores

    def spectral_residual_transform(self, values):
        """
        This method transform a time series into spectral residual series
        :param values: list.
            a list of float values.
        :return: mag: list.
            a list of float values as the spectral residual values
        """

        trans = np.fft.fft(values)
        mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)
        eps_index = np.where(mag <= EPS)[0]
        mag[eps_index] = EPS

        mag_log = np.log(mag)
        mag_log[eps_index] = 0

        spectral = np.exp(mag_log - self.average_filter(mag_log, n=self._mag_window))

        trans.real = trans.real * spectral / mag
        trans.imag = trans.imag * spectral / mag
        trans.real[eps_index] = 0
        trans.imag[eps_index] = 0

        wave_r = np.fft.ifft(trans)
        mag = np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2)
        return mag

    @staticmethod
    def predict_next(values: np.array) -> float:
        """Predicts the next value by sum up the slope of the last value with previous values.

        Parameters
        ----------
        values : np.array
            previous values.

        Returns
        -------
        float
            predicted value.

        Raises
        ------
        ValueError
            value error.
        """

        if len(values) <= 1:
            raise ValueError(f"data should contain at least 2 numbers")

        v_last = values[-1]
        n = len(values)

        slopes = [(v_last - v) / (n - 1 - i) for i, v in enumerate(values[:-1])]

        return values[1] + sum(slopes)

    def extend_series(self, values: np.array, extend_num=5, look_ahead=5) -> np.array:
        """extend the array data by the predicted next value

        Parameters
        ----------
        values : np.array
            raw input data
        extend_num : int, optional
            number of values added to the tail of data, by default 5
        look_ahead : int, optional
            number of previous values used in prediction., by default 5

        Returns
        -------
        np.array :
            filled values.

        Raises
        ------
            ValueError
        """

        if look_ahead < 1:
            raise ValueError("look_ahead must be at least 1")

        extension = [self.predict_next(values[-look_ahead - 2 : -1])] * extend_num
        return np.concatenate((values, extension), axis=0)

    @staticmethod
    def average_filter(values: np.array, n=3) -> np.array:
        """moving window average
        res[i] = sum_{j=i-t+1}^{i} values[j] / t, where t = min(n, i+1)

        Parameters
        ----------
        values : np.array
            input data
        n : int, optional
            sliding window, by default 3

        Returns
        -------
        np.array
            output data.
        """
        if n >= len(values):
            n = len(values)

        res = np.cumsum(values, dtype=float)
        res[n:] = res[n:] - res[:-n]
        res[n:] = res[n:] / n

        for i in range(1, n):
            res[i] /= i + 1

        return res

    @staticmethod
    def calculate_expected_value(values, anomaly_index):
        """calculate expected value in time series.

        Parameters
        ----------
        values : series
            np.array
        anomaly_index : list
            anomaly index
        """
        pass
