"""
Module implementing the Rényi Entropy algorithm for online change-point detection.

The detector maintains a rolling (sliding) window over a univariate time series,
estimates the empirical probability distribution in that window via fixed-width
binning, computes Rényi entropy for the current window, and raises a change-point
signal when short-term dynamics of entropy indicate a regime shift.
"""

__author__ = "Kirill Gribanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections import Counter, deque
from typing import Optional

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class RenyiEntropyAlgorithm(OnlineAlgorithm):
    """
    Online change-point detector based on Rényi entropy.

    :param window_size: Sliding window length used to compute :math:`H`. Default: ``100``.
    :type window_size: int
    :param alpha: Rényi order. Must be positive and not equal to 1. Smaller ``alpha < 1``
                  emphasizes support size / rare events; larger ``alpha > 1`` emphasizes
                  frequent events. Default: ``0.5``.
    :type alpha: float
    :param bins: Number of histogram bins used to estimate probabilities in the current window.
                 Bin edges are determined from running global min/max. Default: ``10``.
    :type bins: int
    :param threshold: Decision threshold used by both (i) consecutive-entropy-difference test
                      and (ii) short-term variance test (with an internal scaling).
                      Default: ``0.3``.
    :type threshold: float

    . note::
       - The detector processes observations in a streaming fashion.
       - Change localization is returned as an approximate index near the center
         (or quarter) of the current window depending on which criterion was triggered.
    """

    def __init__(
        self,
        window_size: int = 100,
        alpha: float = 0.5,
        bins: int = 10,
        threshold: float = 0.3,
    ):
        if alpha <= 0 or alpha == 1:
            raise ValueError("Alpha must be positive and not equal to 1")

        self._window_size = window_size
        self._alpha = alpha
        self._bins = bins
        self._threshold = threshold

        self._buffer: deque[float] = deque(maxlen=window_size * 2)
        self._entropy_values: list[float] = []
        self._position: int = 0
        self._last_change_point: Optional[int] = None

        # Running range used to define stable bin edges across windows
        self._global_min: Optional[float] = None
        self._global_max: Optional[float] = None

    def detect(self, observation: np.float64 | npt.NDArray[np.float64]) -> bool:
        """
        Ingest a new observation (or a batch) and update the detection state.

        :param observation: A single value or a 1-D array of values to process sequentially.
        :type observation: float or numpy.ndarray
        :return: ``True`` if a change-point was flagged after processing the input,
                 ``False`` otherwise.
        :rtype: bool
        """
        if isinstance(observation, np.ndarray):
            for obs in observation:
                self._process_single_observation(float(obs))
        else:
            self._process_single_observation(float(observation))

        return self._last_change_point is not None

    def localize(self, observation: np.float64 | npt.NDArray[np.float64]) -> Optional[int]:
        """
        Process input and return the index of a detected change-point if present.

        :param observation: A single value or a 1-D array of values to process.
        :type observation: float or numpy.ndarray
        :return: Estimated change-point index (0-based, relative to the processed stream),
                 or ``None`` if no change-point is detected.
        :rtype: int or None
        """
        change_detected = self.detect(observation)

        if change_detected:
            change_point = self._last_change_point
            self._last_change_point = None
            return change_point

        return None

    def _process_single_observation(self, observation: float) -> None:
        """
        Process a single value, update buffers and Rényi entropy, and run decision rules.

        :param observation: New sample from the stream.
        :type observation: float
        """
        v = 2  # minimal history for first-difference test
        self._buffer.append(observation)
        self._position += 1

        # Update running global min/max for stable binning
        if self._global_min is None or observation < self._global_min:
            self._global_min = observation
        if self._global_max is None or observation > self._global_max:
            self._global_max = observation

        # Wait until we have a full window
        if len(self._buffer) < self._window_size:
            return

        current_window = np.array(list(self._buffer)[-self._window_size :])
        current_entropy = self._calculate_renyi_entropy(current_window)
        self._entropy_values.append(current_entropy)

        # Criterion 1: absolute difference in consecutive entropies
        if len(self._entropy_values) >= v:
            entropy_diff = abs(self._entropy_values[-1] - self._entropy_values[-2])
            if entropy_diff > self._threshold:
                self._last_change_point = self._position - self._window_size // 2

        # Criterion 2: short-term variance over last 5 values
        if len(self._entropy_values) >= v + 3:
            recent_entropies = self._entropy_values[-5:]
            entropy_variance = np.var(recent_entropies)
            if entropy_variance > self._threshold * 2:
                self._last_change_point = self._position - self._window_size // 4

    def _calculate_renyi_entropy(self, time_series: npt.NDArray[np.float64]) -> float:
        """
        Compute Rényi entropy for the given window using histogram probabilities.

        :param time_series: The current rolling window.
        :type time_series: numpy.ndarray
        :return: The computed Rényi entropy (natural logarithm base). Returns ``0.0`` when
                 probabilities are degenerate (e.g., all mass in a single bin) or inputs are too short.
        :rtype: float

        . note::
           - Uses :meth:`_compute_probabilities` with global min/max to form fixed edges.
           - Supports the special cases :math:`\\alpha = 0` and :math:`\\alpha \\to \\infty`;
             :math:`\\alpha = 1` (Shannon) is excluded by input validation.
        """
        if len(time_series) == 0:
            return 0.0

        probabilities = self._compute_probabilities(time_series)
        if len(probabilities) == 0 or all(p == 0 for p in probabilities):
            return 0.0

        # Special cases
        if self._alpha == 0:
            non_zero_count = sum(1 for p in probabilities if p > 0)
            return float(np.log(non_zero_count))
        elif np.isinf(self._alpha):
            max_prob = max(probabilities)
            return float(-np.log(max_prob)) if max_prob > 0 else 0.0

        # General case  alpha ≠ 0,1,∞
        power_sum = sum(p**self._alpha for p in probabilities if p > 0)
        if power_sum <= 0:
            return 0.0
        renyi_entropy = (1 / (1 - self._alpha)) * np.log(power_sum)
        return float(renyi_entropy)

    def _compute_probabilities(self, time_series: npt.NDArray[np.float64]) -> list[float]:
        """
        Estimate a discrete probability distribution in the current window via histogramming.

        :param time_series: Values from the current rolling window.
        :type time_series: numpy.ndarray
        :return: Probability of each bin (summing to 1), or an empty list if binning is not possible.
        :rtype: list[float]

        . notes::
           - Uses ``self._global_min`` and ``self._global_max`` to define ``bins + 1`` edges.
           - If global range collapses to a point, returns ``[1.0]``.
        """
        if self._global_min is None or self._global_max is None:
            return []

        if self._global_max == self._global_min:
            return [1.0]

        bin_edges = np.linspace(self._global_min, self._global_max, self._bins + 1)
        digitized = np.digitize(time_series, bin_edges)

        bin_counts = Counter(digitized)
        total_count = len(time_series)

        probabilities: list[float] = []
        # Only bins 1.bins correspond to intervals; bin 0 or bins+1 catch out-of-range
        for i in range(1, len(bin_edges)):
            key = np.int64(i)
            count = bin_counts.get(key, 0)
            prob = count / total_count if total_count > 0 else 0.0
            probabilities.append(prob)

        return probabilities

    def get_entropy_history(self) -> list[float]:
        """
        Get the history of computed Rényi entropy values.

        :return: A copy of the internal entropy sequence evaluated at processed steps.
        :rtype: list[float]
        """
        return self._entropy_values.copy()

    def reset(self) -> None:
        """
        Clear internal state and buffered statistics.

        :return: ``None``.
        :rtype: None
        """
        self._buffer.clear()
        self._entropy_values.clear()
        self._position = 0
        self._last_change_point = None
        self._global_min = None
        self._global_max = None
