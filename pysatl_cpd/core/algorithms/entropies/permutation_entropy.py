"""
Module implements the Permutation Entropy algorithm for online change point detection.
"""

__author__ = "Kirill Gribanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections import Counter, deque
from typing import Optional

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class PermutationEntropyAlgorithm(OnlineAlgorithm):
    """
    **PermutationEntropyAlgorithm** detects change points in
    a time series using permutation entropy.

    This algorithm calculates permutation entropy values based on the order
    of values within sliding windows of the time series,
    and detects significant changes based on predefined thresholds.

    :param window_size: Size of each sliding window.
    :param embedding_dimension: The embedding dimension used to form state vectors.
    :param time_delay: Time delay between elements in each state vector.
    :param threshold: Threshold for detecting changes based on entropy differences.

    **Attributes:**
    - `window_size` (int): Size of each sliding window.
    - `embedding_dimension` (int): The embedding dimension used for
    calculating permutation entropy.
    - `time_delay` (int): The time delay used in each state vector.
    - `threshold` (float): Threshold for change detection based on entropy shift.
    - `min_observations_for_detection` (int): Minimum number of
    observations required to detect a change point.
    - `_special_case` (bool): Flag indicating if a special case (large time delay)
     is being processed.

    **Methods:**

    **`detect(window: Iterable[float | np.float64]) -> int`**
    Detects the number of change points in a time series.

    :param window: Input time series.
    :return: Number of detected change points.
    """

    def __init__(
        self,
        window_size: int = 40,
        embedding_dimension: int = 3,
        time_delay: int = 1,
        threshold: float = 0.2,
    ):
        """
        Initializes the PermutationEntropyAlgorithm with the specified parameters.

        :param window_size: Size of each sliding window.
        :param embedding_dimension: The embedding dimension used for calculating permutation entropy.
        :param time_delay: Time delay between elements in each state vector.
        :param threshold: Threshold for detecting changes based on entropy differences.
        """
        self._window_size = window_size
        self._embedding_dimension = embedding_dimension
        self._time_delay = time_delay
        self._threshold = threshold

        self._buffer: deque[float] = deque(maxlen=window_size * 2)
        self._entropy_values: list[float] = []
        self._position: int = 0
        self._last_change_point: Optional[int] = None
        self._prev_observation: Optional[float] = None
        self._constant_value_count: int = 0
        self._min_observations_for_detection = window_size
        self.min_time_delay = 100

        self._special_case = time_delay > self.min_time_delay

    def detect(self, observation: np.float64 | npt.NDArray[np.float64]) -> bool:
        """
        Processes the input observation to detect if a change point occurs in the time series.

        :param observation: A single observation or an array of observations.
        :return: `True` if a change point is detected, otherwise `False`.
        """
        if self._special_case:
            return False

        if isinstance(observation, np.ndarray):
            for obs in observation:
                self._process_single_observation(float(obs))
        else:
            self._process_single_observation(float(observation))

        return self._last_change_point is not None and self._position >= self._min_observations_for_detection

    def localize(self, observation: np.float64 | npt.NDArray[np.float64]) -> Optional[int]:
        """
        Localizes the detected change point based on the observation.

        :param observation: A single observation or an array of observations.
        :return: The position of the detected change point, or `None` if no change point is detected.
        """
        change_detected = self.detect(observation)

        if change_detected:
            change_point = self._last_change_point
            self._last_change_point = None
            return change_point

        return None

    def _process_single_observation(self, observation: float) -> None:
        """
        Processes a single observation and updates the internal state.
        This method checks for constant values, significant
        deviations from the moving average, and computes entropy for
        the sliding window to detect changes.

        :param observation: The observation value to be processed.
        """
        threshold = 10
        threshold1 = 0.5
        threshold2 = 2

        if self._prev_observation is not None and self._position >= self._min_observations_for_detection:
            if observation == self._prev_observation:
                self._constant_value_count += 1
            else:
                if self._constant_value_count >= threshold:
                    self._last_change_point = self._position
                self._constant_value_count = 0

        self._prev_observation = observation

        if len(self._buffer) >= threshold and self._position >= self._min_observations_for_detection:
            buffer_mean = sum(list(self._buffer)[-10:]) / 10
            if abs(observation - buffer_mean) > threshold1:
                self._last_change_point = self._position

        if len(self._buffer) >= threshold and self._position >= self._min_observations_for_detection:
            recent_values = np.array(list(self._buffer)[-10:])
            std_val = np.std(recent_values)
            if std_val > 0 and abs(observation - np.mean(recent_values)) > 2 * std_val:
                self._last_change_point = self._position

        self._buffer.append(observation)
        self._position += 1

        min_required = self._embedding_dimension * self._time_delay + 1
        if len(self._buffer) < self._window_size or len(self._buffer) < min_required:
            return

        current_window = np.array(list(self._buffer)[-self._window_size :])
        current_entropy = self._calculate_permutation_entropy(current_window)
        self._entropy_values.append(current_entropy)

        if len(self._entropy_values) >= threshold2 and self._position >= self._min_observations_for_detection:
            entropy_diff = abs(self._entropy_values[-1] - self._entropy_values[-2])

            if entropy_diff > self._threshold:
                self._last_change_point = self._position - self._window_size // 2

    def _calculate_permutation_entropy(self, time_series: npt.NDArray[np.float64]) -> float:
        """
        Calculates the permutation entropy of a time series using the order of values in the sliding windows.
        The entropy is computed based on the frequency of different permutations of the state vectors.

        :param time_series: The time series data to analyze.
        :return: The calculated permutation entropy value.
        """
        permutation_vectors = []
        for index in range(len(time_series) - self._embedding_dimension * self._time_delay):
            current_window = time_series[
                index : index + self._embedding_dimension * self._time_delay : self._time_delay
            ]
            permutation_vector = np.argsort(current_window)
            permutation_vectors.append(tuple(permutation_vector))

        permutation_counts = Counter(permutation_vectors)
        total_permutations = len(permutation_vectors)

        if total_permutations == 0:
            return 0.0

        permutation_probabilities = [count / total_permutations for count in permutation_counts.values()]
        permutation_entropy = -np.sum(
            [probability * np.log2(probability) for probability in permutation_probabilities if probability > 0]
        )

        return float(permutation_entropy)
