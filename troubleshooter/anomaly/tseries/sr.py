#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File        : spetral_residual.py
@Time        : 2021-07-14 20:12:11
@Author      : dreamhomes
@Description : Time-Series Anomaly Detection Service at Microsoft.
"""
import numpy as np
from scipy.fftpack import fft, ifft

EPS = 1e-8


class SpectraResidual:
    def __init__(self, X, slice_window=3, map_window=3, tresh=1):
        self.slice_window = slice_window
        self.X = X
        self.map_window = map_window
        self.thresh = tresh

    def run(self):
        Smap = self.get_salience_map(self.X)
        result = np.array([1 if i > self.thresh else 0 for i in Smap])
        return result

    def setslicewindow(self, thresh):
        self.slice_window = thresh

    def plot(self):
        raise NotImplementedError

    def getSR(self, X):
        """
        傅里叶变化、残差谱、反傅里叶变换
        """
        X = X

        # spectral_residual_transform
        yy = fft(X)
        A = yy.real
        P = yy.imag
        V = np.sqrt(A ** 2 + P ** 2)
        eps_index = np.where(V <= EPS)[0]
        V[eps_index] = EPS
        L = np.log(V)
        L[eps_index] = 0
        residual = np.exp(L - self.average_filter(L, self.map_window))
        yy.imag = residual * P / V
        yy.real = residual * A / V
        yy.imag[eps_index] = 0
        yy.real[eps_index] = 0
        result = ifft(yy)
        S = np.sqrt(result.real ** 2 + result.imag ** 2)
        # guass filter
        return S

    def get_salience_map(self, X):
        Map = self.getSR(self.extendseries(X))[: len(X)]
        ave_mag = average_filter(Map, n=self.slice_window)
        ave_mag[np.where(ave_mag <= EPS)] = EPS

        return abs(Map - ave_mag) / ave_mag

    def estimate(self, X):
        """
        get k estimated points which is equal to x(n+1)
        x(n+1)=x(n-m+1)+m*g
        g=sum(g(x(n),x(n-i)))/m
        """
        n = len(X)
        gradients = [(X[-1] - v) / (n - 1 - i) for i, v in enumerate(X[:-1])]
        # g=np.sum(gradients)/m
        return X[1] + np.sum(gradients)

    def extend_series(self, X, k=5):
        """
        use k to extend oringe serie;
        """
        X = np.append(X, self.estimate(X[-k - 2 : -1]).repeat(k))
        return X

    @staticmethod
    def average_filter(X, n=3):
        if n >= len(X):
            n = len(X)

        res = np.cumsum(X, dtype=float)
        res[n:] = res[n:] - res[:-n]
        res[n:] = res[n:] / n

        for i in range(1, n):
            res[i] /= (i + 1)

        return res
