"""
Module implements the Bubble Entropy algorithm for online change point detection.
"""

__author__ = "Kirill Gribanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections import deque
from typing import Optional

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class BubbleEntropyAlgorithm(OnlineAlgorithm):
    """
    **BubbleEntropyAlgorithm** detects change points in a time series using bubble entropy.

    The algorithm calculates bubble entropy values based on permutation entropy with varying embedding dimensions.
    It then detects significant changes based on predefined thresholds.

    :param window_size: Size of each sliding window.
    :param embedding_dimension: The embedding dimension used for calculating permutation entropy.
    :param time_delay: Time delay between elements in each state vector for calculating permutation entropy.
    :param threshold: Threshold for detecting changes based on entropy differences.

    **Attributes:**
    - `window_size` (int): Size of each sliding window.
    - `embedding_dimension` (int): The embedding dimension used for calculating permutation entropy.
    - `time_delay` (int): Time delay between elements in each state vector.
    - `threshold` (float): Threshold for change detection based on entropy shift.
    - `min_observations_for_detection` (int): Minimum number of observations required to detect a change point.
    - `_buffer` (deque): A buffer for storing the most recent observations.
    - `_entropy_values` (list): A list to store the calculated entropy values.
    - `_position` (int): The current position in the observation sequence.
    - `_last_change_point` (Optional[int]): The position of the last detected change point.
    """

    def __init__(
        self,
        window_size: int = 100,
        embedding_dimension: int = 3,
        time_delay: int = 1,
        threshold: float = 0.2,
        anomaly_threshold: float = 3.0,
    ):
        """
        Initializes the BubbleEntropyAlgorithm with the specified parameters.

        :param window_size: Size of each sliding window.
        :param embedding_dimension: The embedding dimension used for calculating permutation entropy.
        :param time_delay: Time delay between elements in each state vector for calculating permutation entropy.
        :param threshold: Threshold for detecting changes based on entropy differences.
        """
        super().__init__()
        self._window_size = window_size
        self._embedding_dimension = embedding_dimension
        self._time_delay = time_delay
        self._threshold = threshold
        self._anomaly_threshold = anomaly_threshold

        self._buffer: deque[float] = deque(maxlen=window_size)
        self._entropy_values: deque[float] = deque(maxlen=200)
        self._position: int = 0
        self._last_change_point: Optional[int] = None

    def detect(self, observation: np.float64 | npt.NDArray[np.float64]) -> bool:
        """
        Processes the input observation to detect if a change point occurs in the time series.

        :param observation: A single observation or an array of observations.
        :return: `True` if a change point is detected, otherwise `False`.
        """
        if isinstance(observation, np.ndarray):
            for obs in observation.flat:
                self._process_single_observation(float(obs))
        else:
            self._process_single_observation(float(observation))

        return self._last_change_point is not None

    def localize(self, observation: np.float64 | npt.NDArray[np.float64]) -> Optional[int]:
        """
        Localizes the detected change point based on the observation.

        :param observation: A single observation or an array of observations.
        :return: The position of the detected change point, or `None` if no change point is detected.
        """
        if self.detect(observation):
            change_point = self._last_change_point
            self._last_change_point = None
            return change_point
        return None

    def _process_single_observation(self, observation: float) -> None:
        """
        Processes a single observation and updates the internal state. This method checks for significant deviations,
        computes bubble entropy, and detects change points when applicable.

        :param observation: The observation value to be processed.
        """
        if len(self._buffer) >= self._window_size // 2:
            current_mean = np.mean(self._buffer)
            if abs(observation - current_mean) > self._anomaly_threshold:
                self._last_change_point = self._position

        self._buffer.append(observation)
        self._position += 1

        min_required = (self._embedding_dimension + 1) * self._time_delay + 1
        if len(self._buffer) < self._window_size or len(self._buffer) < min_required:
            return

        current_window = np.fromiter(self._buffer, dtype=float)
        current_entropy = 0.0 if np.std(current_window) == 0 else self._calculate_bubble_entropy(current_window)

        self._entropy_values.append(current_entropy)
        min_history = 2
        if len(self._entropy_values) >= min_history:
            entropy_diff = abs(self._entropy_values[-1] - self._entropy_values[-2])
            if entropy_diff > self._threshold:
                self._last_change_point = self._position - self._window_size // 2

    def _calculate_bubble_entropy(self, time_series: npt.NDArray[np.float64]) -> float:
        """
        Calculates the bubble entropy of a time series by computing the difference in permutation entropy
        between two different embedding dimensions.

        :param time_series: The time series to analyze.
        :return: The computed bubble entropy value.
        """
        h_m = self._calculate_permutation_entropy_vectorized(time_series, self._embedding_dimension)
        h_m_plus_1 = self._calculate_permutation_entropy_vectorized(time_series, self._embedding_dimension + 1)

        denom = np.log((self._embedding_dimension + 1) / self._embedding_dimension)
        if denom == 0:
            return 0.0

        bubble_entropy = (h_m_plus_1 - h_m) / denom
        return float(bubble_entropy)

    def _calculate_permutation_entropy_vectorized(self, time_series: npt.NDArray[np.float64], m: int) -> float:
        """
        Calculates the permutation entropy of a time series based on the given embedding dimension.

        :param time_series: The time series data to analyze.
        :param embedding_dimension: The embedding dimension for the state vectors.
        :return: The computed permutation entropy value.
        """
        N = len(time_series)
        n_vectors = N - (m - 1) * self._time_delay

        if n_vectors <= 0:
            return 0.0

        shape = (n_vectors, m)
        itemsize = time_series.strides[0]
        strides = (itemsize, itemsize * self._time_delay)

        vectors = np.lib.stride_tricks.as_strided(time_series, shape=shape, strides=strides)

        sorted_indices = np.argsort(vectors, axis=1)
        _, counts = np.unique(sorted_indices, axis=0, return_counts=True)
        probs = counts / n_vectors
        entropy = -np.sum(probs * np.log2(probs))

        return float(entropy)

    def get_entropy_history(self) -> list[float]:
        """Returns the history of entropy values."""
        return list(self._entropy_values)

    def reset(self) -> None:
        """Resets the internal state."""
        self._buffer.clear()
        self._entropy_values.clear()
        self._position = 0
        self._last_change_point = None
