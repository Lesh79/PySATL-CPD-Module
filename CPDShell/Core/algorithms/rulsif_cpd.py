from collections.abc import Iterable

import numpy as np

from CPDShell.Core.algorithms.DensityBasedCPD.abstracts import DensityBasedAlgorithm


class RulsifAlgorithm(DensityBasedAlgorithm):
    """Relative Unconstrained Least-Squares Importance Fitting (RULSIF)
    algorithm for change point detection.

    RULSIF estimates the density ratio between two distributions and uses
    the importance weights for detecting changes in the data distribution.
    """

    def __init__(self, bandwidth, regularization_coef, threshold):
        """Initialize the RULSIF algorithm.

        Args:
            bandwidth (float): bandwidth parameter for density estimation.
            regularization_coef (float): regularization parameter.
            threshold (float, optional): threshold for detecting change points.
            Defaults to 1.1.
        """
        self.bandwidth = bandwidth
        self.regularization_coef = regularization_coef
        self.threshold = threshold

    def detect(self, window: Iterable[float]) -> int:
        """Detect the number of change points in the given data window
        using RULSIF.

        Args:
            window (Iterable[float]): the data window to detect change points.

        Returns:
            int: the number of detected change points.
        """
        weights = self._calculate_weights(
            window,
            window,
            self.bandwidth,
            self.regularization_coef,
            lambda density_ratio, alpha: np.mean((density_ratio - 1) ** 2)
            + self.regularization_coef * np.sum(alpha**2),
        )

        change_points = 0
        for time, weight in enumerate(weights):
            if weight > self.threshold:
                change_points += 1
        return change_points

    def localize(self, window: Iterable[float]) -> list[int]:
        """
        Localize the change points in the given data window using RULSIF.

        Args:
            window (Iterable[float]): the data window to localize
            change points.

        Returns:
            list[int]: the indices of the detected change points.
        """
        weights = self._calculate_weights(
            window,
            window,
            self.bandwidth,
            self.regularization_coef,
            lambda density_ratio, alpha: np.mean((density_ratio - 1) ** 2)
            + self.regularization_coef * np.sum(alpha**2),
        )

        change_points = []
        for time, weight in enumerate(weights):
            if weight > self.threshold:
                change_points.append(time)
        return change_points
