"""
Module contains KLIEP (Kullback-Leibler Importance Estimation Procedure)
algorithm implementation for change point detection.
"""

__author__ = "Aleksandra Listkova"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from typing import cast

import numpy as np
import numpy.typing as npt
from numpy import dtype, float64, ndarray

from pysatl_cpd.core.algorithms.density.abstracts import IDensityBasedAlgorithm


class KliepAlgorithm(IDensityBasedAlgorithm):
    """Kullback-Leibler Importance Estimation Procedure (KLIEP) algorithm
    for change point detection.

    KLIEP estimates the density ratio between two distributions and uses
    the importance weights for detecting changes in the data distribution.
    """

    def __init__(self, bandwidth: float, regularization_coef: float, threshold: float = 1.1):
        """Initialize the KLIEP algorithm.

        Args:
            bandwidth (float): bandwidth parameter for density estimation.
            regularization_coef (float): regularization parameter.
            threshold (float, optional): threshold for detecting change points.
            Defaults to 1.1.
        """
        self.bandwidth = bandwidth
        self.regularization_coef = regularization_coef
        self.threshold = np.float64(threshold)

    def _loss_function(self, density_ratio: npt.NDArray[np.float64], alpha: npt.NDArray[np.float64]) -> float:
        """Loss function for KLIEP.

        Args:
            density_ratio (np.ndarray): estimated density ratio.
            alpha (np.ndarray): coefficients for the density ratio.

        Returns:
            float: the computed loss value.
        """
        return -np.mean(density_ratio) + self.regularization_coef * np.sum(alpha**2)

    def detect(self, window: npt.NDArray[np.float64]) -> int:
        """Detect the number of change points in the given data window
        using KLIEP.

        Args:
            window (Iterable[float]): the data window to detect change points.

        Returns:
            int: the number of detected change points.
        """

        window_sample = np.array(window)
        weights = self._calculate_weights(
            test_value=window_sample,
            reference_value=window_sample,
            bandwidth=self.bandwidth,
            objective_function=self._loss_function,
        )

        return int(np.count_nonzero(weights > self.threshold))

    def localize(self, window: npt.NDArray[np.float64]) -> list[int]:
        """Localize the change points in the given data window using KLIEP.

        Args:
            window (Iterable[float]): the data window to localize
            change points.

        Returns:
            List[int]: the indices of the detected change points.
        """
        window_sample = np.array(window)
        weights: ndarray[tuple[int, ...], dtype[float64]] = self._calculate_weights(
            test_value=window_sample,
            reference_value=window_sample,
            bandwidth=self.bandwidth,
            objective_function=self._loss_function,
        )

        return cast(list[int], np.where(weights > self.threshold)[0].tolist())
