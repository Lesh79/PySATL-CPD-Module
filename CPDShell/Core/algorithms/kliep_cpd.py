from collections.abc import Iterable

import numpy as np

from CPDShell.Core.algorithms.DensityBasedCPD.abstracts.density_based_algorithm import DensityBasedAlgorithm


class KliepAlgorithm(DensityBasedAlgorithm):
    def __init__(self, bandwidth, lam):
        self.bandwidth = bandwidth
        self.lam = lam

    def detect(self, window: Iterable[float]) -> list[int]:
        # Compute importance weights using KLIEP
        weights = self._calculate_weights(
            window, window, self.bandwidth, self.lam, lambda w, alpha: -np.mean(w) + self.lam * np.sum(alpha**2)
        )

        # Determine change points based on importance weights
        change_points = []
        threshold = 1.1  # Example threshold for detecting changes
        for i in range(len(window)):
            if weights[i] > threshold:
                change_points.append(i + 1)  # Return the right border
        return change_points

    def localize(self, window: Iterable[float]) -> list[int]:
        weights = self._calculate_weights(
            window, window, self.bandwidth, self.lam, lambda w, alpha: -np.mean(w) + self.lam * np.sum(alpha**2)
        )

        change_points = []
        threshold = 1.1
        for i in range(len(window)):
            if weights[i] > threshold:
                change_points.append(i)
        return change_points
