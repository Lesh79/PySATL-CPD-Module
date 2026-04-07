"""
Module for implementations of SSA CPD algorithm distance detector using a threshold.
"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.ssa.abstracts.idetector import IDetectorSSA


class DistanceThreshold(IDetectorSSA):
    """
    Class of implementations of SSA CPD algorithm detector using
    the normalized Euclidean distance and a threshold.
    """

    def __init__(self, threshold: float) -> None:
        """
        Initializes SSA CPD algorithm distance detector with given threshold.
        :param threshold: threshold for distance calculation.
        """
        if not(0 <= threshold <= 1):
            raise ValueError("Threshold must be in [0.0, 1.0]")
        self._threshold = threshold

    def detect(
        self, subspace: npt.NDArray[np.float64], test_data: list[np.float64]
    ) -> bool:
        """
        Checks whether a changepoint has occurred at the start of the test data using
        the normalized Euclidean distance and comparing it to a threshold.
        :param subspace: vectors of the training set subspace.
        :param test_data: test sample for breakpoint detection.
        :return: boolean indicating whether a changepoint occurred.
        """
        L = subspace.shape[0]
        n_test = len(test_data) - L + 1
        X_test = np.zeros((L, n_test))

        for i in range(n_test):
            X_test[:, i] = test_data[i : i + L]

        proj_sum = np.sum((subspace.T @ X_test) ** 2)
        total_sum = np.sum(X_test**2)

        if total_sum == 0:
            return False

        return bool(1 - (proj_sum / total_sum) > self._threshold)

    def clean(self) -> None:
        """
        Clears the detector's state.
        """
        return
