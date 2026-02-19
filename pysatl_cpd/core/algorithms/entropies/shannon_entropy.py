"""
Module implements the Shannon Entropy algorithm for online change point detection.
"""

__author__ = "Kirill Gribanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections import deque
from typing import Optional

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class ShannonEntropyAlgorithm(OnlineAlgorithm):
    """
    **ShannonEntropyAlgorithm** detects change points in time series using Shannon entropy.

    This algorithm estimates the information content of sliding windows using histogram-based
    Shannon entropy. Significant differences in entropy between windows are used to detect
    structural changes in the signal.

    :param window_size: Size of each sliding window.
    :param step: Step size for moving the sliding window.
    :param bins: Number of bins used to create histograms.
    :param threshold: Threshold for detecting changes based on entropy differences.

    **Attributes:**
    - `window_size` (int): Size of each sliding window.
    - `step` (int): Step size for moving the sliding window.
    - `bins` (int): Number of histogram bins used for entropy estimation.
    - `threshold` (float): Threshold for change detection based on entropy shift.

    **Methods:**

    **`detect(window: Iterable[float | np.float64]) -> int`**
    Detects the number of change points in a time series.

    :param window: Input time series.
    :return: Number of detected change points.
    """

    def __init__(
        self,
        window_size: int = 40,
        bins: int = 30,
        threshold: float = 0.3,
        anomaly_threshold: float = 0.5,
        std_threshold: float = 2.0,
    ):
        """
        Initializes the ShannonEntropyAlgorithm with the specified parameters.

        :param window_size: Size of each sliding window.
        :param step: Step size for moving the sliding window.
        :param bins: Number of bins used to create histograms for entropy calculation.
        :param threshold: Threshold for detecting changes based on entropy differences.
        """
        super().__init__()
        self._window_size = window_size
        self._bins = bins
        self._threshold = threshold
        self._anomaly_threshold = anomaly_threshold
        self._std_threshold = std_threshold
        self._const_threshold = 10

        self._buffer: deque[float] = deque(maxlen=window_size)
        self._entropy_history: deque[float] = deque(maxlen=5)

        self._position: int = 0
        self._last_change_point: Optional[int] = None
        self._prev_observation: Optional[float] = None
        self._constant_value_count: int = 0

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
        change_detected = self.detect(observation)

        if change_detected:
            change_point = self._last_change_point
            self._last_change_point = None
            return change_point

        return None

    def _process_single_observation(self, observation: float) -> None:
        """
        Processes a single observation and updates the internal state to detect potential change points.

        This method checks for constant values, significant deviations from the moving average,
        and computes entropy for the sliding window to detect changes.

        :param observation: The observation value to be processed.
        """
        if self._prev_observation is not None:
            if np.isclose(observation, self._prev_observation):
                self._constant_value_count += 1
                if self._constant_value_count >= self._const_threshold:
                    self._last_change_point = self._position
            else:
                self._constant_value_count = 0
        self._prev_observation = observation

        if len(self._buffer) >= self._window_size // 2:
            arr = np.fromiter(self._buffer, dtype=float)

            if abs(observation - np.mean(arr)) > self._anomaly_threshold:
                self._last_change_point = self._position

            recent_window_size = 10
            recent = arr[-recent_window_size:] if len(arr) >= recent_window_size else arr
            std_val = np.std(recent)
            if std_val > 0 and abs(observation - np.mean(recent)) > self._std_threshold * std_val:
                self._last_change_point = self._position

        self._buffer.append(observation)
        self._position += 1

        if len(self._buffer) < self._window_size:
            return

        window_arr = np.fromiter(self._buffer, dtype=float)
        counts, _ = np.histogram(window_arr, bins=self._bins)
        probs = counts / counts.sum()
        current_entropy = self._compute_entropy(probs)

        if len(self._entropy_history) > 0 and abs(current_entropy - self._entropy_history[-1]) > self._threshold:
            self._last_change_point = self._position - self._window_size // 2

        self._entropy_history.append(current_entropy)

    def _compute_entropy(self, probabilities: npt.NDArray[np.float64]) -> float:
        """
        Computes Shannon entropy based on a probability distribution.

        This method uses the Shannon entropy formula to measure the uncertainty or disorder in
        the distribution of the observations in the sliding window.

        :param probabilities: A numpy array representing the probability distribution of observations.
        :return: The computed Shannon entropy value.
        """
        probabilities = probabilities[probabilities > 0]
        if len(probabilities) == 0:
            return 0.0
        return float(-np.sum(probabilities * np.log2(probabilities)))

    def reset(self) -> None:
        """
        Resets the internal state and buffered statistics.
        """
        self._buffer.clear()
        self._entropy_history.clear()
        self._position = 0
        self._last_change_point = None
        self._prev_observation = None
        self._constant_value_count = 0
