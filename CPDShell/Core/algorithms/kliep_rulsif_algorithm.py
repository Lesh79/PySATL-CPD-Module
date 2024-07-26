from abc import ABC, abstractmethod
from collections.abc import Iterable
from scipy.optimize import minimize
from sklearn.neighbors import KernelDensity
import numpy as np


def kde(X, bandwidth):
    kde = KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(X)
    return kde


def calculate_weights(Xt, Xr, bandwidth, lam, objective_function):
    pt = kde(Xt, bandwidth)
    pr = kde(Xr, bandwidth)

    def obj(alpha):
        w = np.exp(pt.score_samples(Xt) - pr.score_samples(Xt) - alpha)
        return objective_function(w, alpha)

    res = minimize(obj, np.zeros(len(Xt)), method="L-BFGS-B")
    alpha = res.x
    w = np.exp(pt.score_samples(Xt) - pr.score_samples(Xt) - alpha)
    return w / np.mean(w)


class KliepAlgorithm(Algorithm):
    def __init__(self, bandwidth, lam):
        self.bandwidth = bandwidth
        self.lam = lam

    def detect(self, window: Iterable[float]) -> list[int]:
        # Compute importance weights using KLIEP
        w = calculate_weights(
            window, window, self.bandwidth, self.lam, lambda w, alpha: -np.mean(w) + self.lam * np.sum(alpha**2)
        )

        # Determine change points based on importance weights
        change_points = []
        threshold = 1.1  # Example threshold for detecting changes
        for i in range(len(window)):
            if w[i] > threshold:
                change_points.append(i + 1)  # Return the right border
        return change_points

    def localize(self, window: Iterable[float]) -> list[int]:
        w = calculate_weights(
            window, window, self.bandwidth, self.lam, lambda w, alpha: -np.mean(w) + self.lam * np.sum(alpha**2)
        )

        change_points = []
        threshold = 1.1
        for i in range(len(window)):
            if w[i] > threshold:
                change_points.append(i)
        return change_points


class RulsifAlgorithm(Algorithm):
    def __init__(self, bandwidth, lam):
        self.bandwidth = bandwidth
        self.lam = lam

    def detect(self, window: Iterable[float]) -> list[int]:
        # Compute importance weights using RULSIF
        w = calculate_weights(
            window,
            window,
            self.bandwidth,
            self.lam,
            lambda w, alpha: np.mean((w - 1) ** 2) + self.lam * np.sum(alpha**2),
        )

        change_points = []
        threshold = 1.1
        for i in range(len(window)):
            if w[i] > threshold:
                change_points.append(i + 1)
        return change_points

    def localize(self, window: Iterable[float]) -> list[int]:
        w = calculate_weights(
            window,
            window,
            self.bandwidth,
            self.lam,
            lambda w, alpha: np.mean((w - 1) ** 2) + self.lam * np.sum(alpha**2),
        )

        change_points = []
        threshold = 1.1
        for i in range(len(window)):
            if w[i] > threshold:
                change_points.append(i)
        return change_points
