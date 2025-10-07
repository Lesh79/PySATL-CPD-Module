"""
Module for implementation of the CUSUM statistical test for Predict&Compare CPD algorithm.
"""

__author__ = "Aleksandra Ri"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np

ZERO = np.float64(0.0)


class CuSum:
    """
    Implements the cumulative sum (CUSUM) algorithm to detect shifts in the value of a process.

    This class tracks two cumulative sums: one for detecting an increase (positive CUSUM)
    and one for detecting a decrease (negative CUSUM). An alarm is triggered if either
    sum exceeds a predefined threshold.
    """

    def __init__(self) -> None:
        self.__k_threshold: np.float64 = ZERO
        self.__h_threshold: np.float64 = ZERO

        self.__positive_cusum: np.float64 = ZERO
        self.__negative_cusum: np.float64 = ZERO

        # History tracks when the CUSUM was at 0.
        # Start with True to ensure a valid index is found on the first detection
        self.__positive_reset_history: list[bool] = [True]
        self.__negative_reset_history: list[bool] = [True]

    def update(self, residual: np.float64) -> None:
        """
        Updates the CUSUM statistics with a new residual value.
        :param residual: the difference between the observed and expected value.
        """
        # Update the positive CUSUM, resetting to 0 if it goes negative.
        self.__positive_cusum = np.maximum(ZERO, self.__positive_cusum + residual - self.__k_threshold)

        # Update the negative CUSUM, resetting to 0 if it goes positive.
        self.__negative_cusum = np.minimum(ZERO, self.__negative_cusum + residual + self.__k_threshold)

        self.__positive_reset_history.append(self.__positive_cusum == ZERO)
        self.__negative_reset_history.append(self.__negative_cusum == ZERO)

    def is_change_detected(self) -> np.bool:
        """
        Checks if either CUSUM statistic has exceeded its threshold.
        :return: True if a change is detected, False otherwise.
        """
        return self.__positive_cusum > self.__h_threshold or self.__negative_cusum < -self.__h_threshold

    def get_last_reset_index(self) -> int:
        """
        Finds the number of time steps since the last CUSUM reset.
        After a change is detected (i.e., `is_change_detected()` is True), this
        method calculates how many steps have passed since the cumulative sum value
        was last at zero. This is essential for localizing the starting point of the
        detected anomaly.
        :return: the number of steps back from the current time to the last reset.
        """
        assert self.is_change_detected(), "This method should only be called when a change is detected."

        last_reset_index: int
        if self.__positive_cusum > self.__h_threshold:
            last_reset_index = list(reversed(self.__positive_reset_history)).index(True)
        elif self.__negative_cusum < -self.__h_threshold:
            last_reset_index = list(reversed(self.__negative_reset_history)).index(True)
        else:  # This branch is unreachable if the assertion holds, but NUMPY
            last_reset_index = len(self.__positive_reset_history)

        return last_reset_index

    def update_thresholds(self, k: np.float64, h: np.float64) -> None:
        """
        Updates the 'k' and 'h' thresholds for the CUSUM detector.
        :param k: the new value for the allowable slack.
        :param h: the new value for the decision threshold.
        :return:
        """
        self.__k_threshold = k
        self.__h_threshold = h

    def clear(self) -> None:
        """
        Resets the CUSUM detector to its initial state.
        :return:
        """
        self.__k_threshold = ZERO
        self.__h_threshold = ZERO

        self.__positive_cusum = ZERO
        self.__negative_cusum = ZERO

        self.__positive_reset_history = [True]
        self.__negative_reset_history = [True]
