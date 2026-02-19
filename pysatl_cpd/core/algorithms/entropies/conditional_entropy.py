"""
Module implements the Conditional Entropy algorithm for online change point detection.
"""

__author__ = "Kirill Gribanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections import deque
from typing import Optional

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class ConditionalEntropyAlgorithm(OnlineAlgorithm):
    """
    **ConditionalEntropyAlgorithm** detects change points in a time series using conditional entropy.

    This algorithm calculates entropy values based on joint and
    conditional probability distributions of paired observations
    and detects significant changes based on predefined thresholds.

    :param window_size: Size of each sliding window.
    :param bins: Number of bins used to create histograms for the joint probability distribution.
    :param threshold: Threshold for detecting changes based on entropy differences.

    **Attributes:**
    - `window_size` (int): Size of each sliding window.
    - `bins` (int): Number of histogram bins used for entropy calculation.
    - `threshold` (float): Threshold for change detection based on entropy shift.
    - `_buffer_x` (deque): A buffer for storing the X observations.
    - `_buffer_y` (deque): A buffer for storing the Y observations.
    - `_entropy_values` (list): A list that stores the calculated entropy values.
    - `_position` (int): The current position (or index) in the observation sequence.
    - `_last_change_point` (Optional[int]): The last position where a change point was detected.
    - `_prev_x` (Optional[float]): The previous X observation value.
    - `_prev_y` (Optional[float]): The previous Y observation value.
    - `_constant_value_count` (int): A counter for consecutive constant values.

    **Methods:**

    **`detect(window: Iterable[float | np.float64]) -> int`**
    Detects the number of change points in a time series.

    :param window: Input time series.
    :return: Number of detected change points.
    """

    def __init__(
        self,
        window_size: int = 40,
        bins: int = 10,
        threshold: float = 0.3,
        anomaly_threshold: float = 3.0,
    ):
        """
        Initializes the ConditionalEntropyAlgorithm with the specified parameters.

        :param window_size: Size of each sliding window.
        :param bins: Number of bins used to create histograms for the joint probability distribution.
        :param threshold: Threshold for detecting changes based on entropy differences.
        """
        super().__init__()
        self._window_size = window_size
        self._bins = bins
        self._threshold = threshold
        self._anomaly_threshold = anomaly_threshold

        self._buffer_x: deque[float] = deque(maxlen=window_size)
        self._buffer_y: deque[float] = deque(maxlen=window_size)
        self._entropy_values: deque[float] = deque(maxlen=200)
        self._position: int = 0
        self._last_change_point: Optional[int] = None
        self._prev_x: Optional[float] = None
        self._prev_y: Optional[float] = None
        self._constant_value_count: int = 0
        self._const_threshold = 10
        self._correlation_threshold = 0.3
        self._epsilon = 1e-10

    def detect(self, observation: np.float64 | npt.NDArray[np.float64]) -> bool:
        """
        Processes a single observation or an array of observations to detect if a change point occurs in the time series

        :param observation: A single observation or an array of observations to be processed.
        :return: `True` if a change point is detected, otherwise `False`.
        """
        min_dimensions = 2
        if isinstance(observation, np.ndarray) and observation.size >= min_dimensions:
            x_obs = float(observation[0])
            y_obs = float(observation[1])
            self._process_observation_pair(x_obs, y_obs)
        elif isinstance(observation, (list, tuple)) and len(observation) >= min_dimensions:
            self._process_observation_pair(float(observation[0]), float(observation[1]))
        else:
            raise ValueError("Observation must contain at least two values (X and Y).")

        return self._last_change_point is not None

    def localize(self, observation: np.float64 | npt.NDArray[np.float64]) -> Optional[int]:
        """
        Localizes the detected change point based on the observation provided.

        :param observation: A single observation or an array of observations.
        :return: The position of the detected change point, or `None` if no change point is detected.
        """
        if self.detect(observation):
            change_point = self._last_change_point
            self._last_change_point = None
            return change_point
        return None

    def _process_observation_pair(self, x_obs: float, y_obs: float) -> None:
        """
        Processes a pair of observations (x and y) and updates the internal state.
        This method checks for constant values
        significant deviations from the moving averages, and computes conditional
        entropy for the sliding window to detect changes

        :param x_obs: The X observation value to be processed
        :param y_obs: The Y observation value to be processed
        """
        self._handle_constant_values(x_obs, y_obs)
        min_len = self._window_size // 4

        if len(self._buffer_x) >= min_len:
            curr_x = np.array(self._buffer_x)
            curr_y = np.array(self._buffer_y)
            self._process_buffer_deviations(x_obs, y_obs, curr_x, curr_y)
            self._check_correlation_change(x_obs, y_obs, curr_x, curr_y)

        self._buffer_x.append(x_obs)
        self._buffer_y.append(y_obs)
        self._position += 1
        if len(self._buffer_x) < self._window_size:
            return
        window_x = np.array(self._buffer_x)
        window_y = np.array(self._buffer_y)

        self._compute_entropy(window_x, window_y)

    def _handle_constant_values(self, x_obs: float, y_obs: float) -> None:
        """
        Checks if the current observations are constant (i.e., the same as the previous ones).
        If so, it increments the counter
        for constant values and detects a change point when the count exceeds the threshold.

        :param x_obs: The current X observation.
        :param y_obs: The current Y observation.
        :param constant_value_threshold: The threshold for detecting constant values.
        """
        if self._prev_x is not None and self._prev_y is not None:
            if np.isclose(x_obs, self._prev_x) and np.isclose(y_obs, self._prev_y):
                self._constant_value_count += 1
                if self._constant_value_count >= self._const_threshold:
                    self._last_change_point = self._position
            else:
                self._constant_value_count = 0

        self._prev_x = x_obs
        self._prev_y = y_obs

    def _process_buffer_deviations(
        self, x_obs: float, y_obs: float, curr_x: npt.NDArray[np.float64], curr_y: npt.NDArray[np.float64]
    ) -> None:
        """
        Checks if the current observations deviate significantly from
        the moving averages of the previous values in the buffer.
        If a significant deviation is detected, a change point is recorded.

        :param x_obs: The current X observation.
        :param y_obs: The current Y observation.
        :param significant_deviation_threshold: The threshold for detecting significant deviations.
        """
        half_win = self._window_size // 2
        if len(curr_x) >= half_win:
            recent_x = curr_x[-half_win:]
            recent_y = curr_y[-half_win:]

            mean_x = np.mean(recent_x)
            mean_y = np.mean(recent_y)

            if abs(x_obs - mean_x) > self._anomaly_threshold or abs(y_obs - mean_y) > self._anomaly_threshold:
                self._last_change_point = self._position

    def _check_correlation_change(
        self, x_obs: float, y_obs: float, curr_x: npt.NDArray[np.float64], curr_y: npt.NDArray[np.float64]
    ) -> None:
        """
        Checks for changes in the correlation between the most recent observations in the buffer.
         If a significant change in
        correlation is detected, a change point is recorded.

        :param x_obs: The current X observation.
        :param y_obs: The current Y observation.
        :param _epsilon_: A small value to avoid division by zero when calculating standard deviations.
        """
        quarter_win = self._window_size // 4
        if len(curr_x) >= quarter_win:
            recent_x = curr_x[-quarter_win:]
            recent_y = curr_y[-quarter_win:]

            std_x = np.std(recent_x)
            std_y = np.std(recent_y)

            if std_x > self._epsilon and std_y > self._epsilon:
                try:
                    corr_before = np.corrcoef(recent_x, recent_y)[0, 1]
                    test_x = np.append(recent_x, x_obs)
                    test_y = np.append(recent_y, y_obs)

                    corr_after = np.corrcoef(test_x, test_y)[0, 1]

                    if (
                        not np.isnan(corr_before)
                        and not np.isnan(corr_after)
                        and abs(corr_after - corr_before) > self._correlation_threshold
                    ):
                        self._last_change_point = self._position
                except Exception:
                    pass

    def _compute_entropy(self, window_x: npt.NDArray[np.float64], window_y: npt.NDArray[np.float64]) -> None:
        """
        Computes the conditional entropy for the most recent observations in the buffer.
        If the number of entropy values exceeds
        a predefined threshold, it checks if the difference between the latest
        two entropy values is significant, indicating a change point.

        :param max_entropy_value: The maximum number of entropy values to consider when detecting changes.
        """
        current_entropy = self._compute_conditional_entropy(window_x, window_y)
        self._entropy_values.append(current_entropy)

        min_history = 2
        if len(self._entropy_values) >= min_history:
            entropy_diff = abs(self._entropy_values[-1] - self._entropy_values[-2])
            if entropy_diff > self._threshold:
                self._last_change_point = self._position - self._window_size // 2

    def _compute_conditional_entropy(
        self, time_series_x: npt.NDArray[np.float64], time_series_y: npt.NDArray[np.float64]
    ) -> float:
        """
        Computes the conditional entropy of the time series using joint and conditional probability distributions.

        :param time_series_x: The X time series to be analyzed.
        :param time_series_y: The Y time series to be analyzed.
        :return: The calculated conditional entropy value.
        """
        hist2d, _, _ = np.histogram2d(time_series_x, time_series_y, bins=[self._bins, self._bins])
        total_samples = np.sum(hist2d)
        if total_samples == 0:
            return 0.0

        p_xy = hist2d / total_samples

        p_y = np.sum(p_xy, axis=0)
        p_x_given_y = np.divide(p_xy, p_y, where=p_y != 0)
        log_term = np.log2(p_x_given_y, where=p_x_given_y > 0)
        log_term[p_x_given_y <= 0] = 0

        entropy = -np.sum(p_xy * log_term)
        return float(entropy)

    def get_entropy_history(self) -> list[float]:
        return list(self._entropy_values)

    def reset(self) -> None:
        self._buffer_x.clear()
        self._buffer_y.clear()
        self._entropy_values.clear()
        self._position = 0
        self._last_change_point = None
        self._prev_x = None
        self._prev_y = None
        self._constant_value_count = 0
