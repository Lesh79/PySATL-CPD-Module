"""
Module implementing the Sample Entropy (SampEn) algorithm for online change-point detection.

This detector maintains a rolling window over a univariate time series, computes the
Sample Entropy of the current window, and emits a change-point when short-term
fluctuations in SampEn indicate a regime shift.

Sample Entropy is defined as:

. math::

   \\mathrm{SampEn}(m, r, N) = -\\log\\left(\frac{A}{B}\right)

where:

- ``m`` is the embedding (pattern) length,
- ``r`` is the tolerance radius (often a fraction of the windowed standard deviation),
- ``B`` is the number of matching pairs of m-length patterns under tolerance ``r``,
- ``A`` is the number of matching pairs of (m+1)-length patterns under tolerance ``r``.

This implementation supports a fixed ``r`` or an adaptive one via ``r_factor * std(window)``.
A detection is triggered when either:

1. The absolute change between consecutive SampEn values exceeds ``threshold``; or
2. The short-term variance of recent SampEn values, normalized by their mean, exceeds ``threshold``.
"""

__author__ = "Kirill Gribanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections import deque
from typing import Optional

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class SampleEntropyAlgorithm(OnlineAlgorithm):
    """
    Online change-point detector based on Sample Entropy (SampEn).

    :param window_size: Sliding window length used to compute SampEn. Default: ``100``.
    :type window_size: int
    :param m: Embedding (pattern) length. Default: ``2``.
    :type m: int
    :param r: Fixed tolerance radius. If ``None``, an adaptive radius is used via ``r_factor``. Default: ``None``.
    :type r: float or None
    :param r_factor: Multiplicative factor for adaptive radius,
     i.e. ``r = r_factor * std(window)`` when ``r`` is ``None``. Default: ``0.2``.
    :type r_factor: float
    :param threshold: Decision threshold used in both the first-order
    SampEn difference test and the normalized short-term variance test. Default: ``0.5``.
    :type threshold: float

    .. note::
       - The detector processes observations in a streaming fashion.
       - Change localization is approximate and centered near the middle (or quarter)
         of the current window depending on which criterion was triggered.
    """

    def __init__(
        self,
        window_size: int = 100,
        m: int = 2,
        r: Optional[float] = None,
        r_factor: float = 0.2,
        threshold: float = 0.5,
        anomaly_threshold: float = 3.0,
    ):
        super().__init__()
        self._window_size = window_size
        self._m = m
        self._r = r
        self._r_factor = r_factor
        self._threshold = threshold
        self._anomaly_threshold = anomaly_threshold

        self._buffer: deque[float] = deque(maxlen=window_size)
        self._entropy_values: deque[float] = deque(maxlen=200)
        self._position: int = 0
        self._last_change_point: Optional[int] = None

    def detect(self, observation: np.float64 | npt.NDArray[np.float64]) -> bool:
        """
        Ingest a new observation (or a batch) and update the internal detection state.

        :param observation: A single value or a 1-D array of values to process sequentially.
        :type observation: float or numpy.ndarray
        :return: ``True`` if a change-point was flagged after processing the input, ``False`` otherwise.
        :rtype: bool
        """
        if isinstance(observation, np.ndarray):
            for obs in observation.flat:
                self._process_single_observation(float(obs))
        else:
            self._process_single_observation(float(observation))

        return self._last_change_point is not None

    def localize(self, observation: np.float64 | npt.NDArray[np.float64]) -> Optional[int]:
        """
        Ingest input and return the index of a detected change-point if present.

        :param observation: A single value or a 1-D array of values to process.
        :type observation: float or numpy.ndarray
        :return: Estimated change-point index (0-based, relative to the processed stream),
                 or ``None`` if no change-point is detected.
        :rtype: int or None
        """
        if self.detect(observation):
            cp = self._last_change_point
            self._last_change_point = None
            return cp
        return None

    def _process_single_observation(self, observation: float) -> None:
        """
        Process a single new observation and update the internal SampEn statistics.

        :param observation: New value to be appended to the rolling buffer.
        :type observation: float
        """
        self._position += 1

        if len(self._buffer) >= self._window_size // 2:
            current_mean = np.mean(self._buffer)
            if abs(observation - current_mean) > self._anomaly_threshold:
                self._last_change_point = self._position

        self._buffer.append(observation)

        if len(self._buffer) < self._window_size:
            return

        current_window = np.fromiter(self._buffer, dtype=float)
        current_entropy = self._calculate_sample_entropy_vectorized(current_window)

        if np.isinf(current_entropy) or np.isnan(current_entropy):
            current_entropy = float("inf")

        if len(self._entropy_values) >= 1:
            prev_entropy = self._entropy_values[-1]

            if np.isinf(prev_entropy) and np.isinf(current_entropy):
                entropy_diff = 0.0
            elif np.isinf(prev_entropy) or np.isinf(current_entropy):
                entropy_diff = float("inf")
            else:
                entropy_diff = abs(current_entropy - prev_entropy)

            if entropy_diff > self._threshold:
                self._last_change_point = self._position - self._window_size // 2

        self._entropy_values.append(current_entropy)

    def _calculate_sample_entropy_vectorized(self, time_series: npt.NDArray[np.float64]) -> float:
        """
        Compute Sample Entropy for the given window.

        :param time_series: Current rolling window.
        :type time_series: numpy.ndarray
        :return: The computed SampEn value. Returns ``inf`` when the input is too short or
                 when there are no matches (degenerate case).
        :rtype: float

        . note::
           - If ``r`` is not provided, it is derived as ``r_factor * std(time_series)``.
           - When the window variance is zero, ``inf`` is returned to reflect undefined entropy.
        """
        N = len(time_series)
        r = self.get_current_r()
        if r is None:
            return float("inf")

        def count_matches_fast(m_val: int) -> int:
            n_vec = N - m_val + 1
            if n_vec <= 0:
                return 0

            shape = (n_vec, m_val)
            strides = (time_series.strides[0], time_series.strides[0])
            vectors = np.lib.stride_tricks.as_strided(time_series, shape=shape, strides=strides)

            diffs = np.abs(vectors[:, None, :] - vectors[None, :, :])
            max_diffs = diffs.max(axis=2)

            return int(np.sum(max_diffs[np.triu_indices(n_vec, k=1)] < r))

        B = count_matches_fast(self._m)
        A = count_matches_fast(self._m + 1)

        if B == 0 or A == 0:
            return float("inf")

        return float(-np.log(A / B))

    def _chebyshev_distance(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
        """
        Compute the Chebyshev (L-infinity) distance between two vectors.

        :param x: First m-length vector.
        :param y: Second m-length vector.
        :type x: numpy.ndarray
        :type y: numpy.ndarray
        :return: Maximum absolute distance.
        :rtype: float
        """
        return float(np.max(np.abs(x - y)))

    def _euclidean_distance(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
        """
        Compute the Euclidean (L2) distance between two vectors.

        :param x: First m-length vector.
        :param y: Second m-length vector.
        :type x: numpy.ndarray
        :type y: numpy.ndarray
        :return: Euclidean distance value.
        :rtype: float
        """
        return float(np.sqrt(np.sum((x - y) ** 2)))

    def get_entropy_history(self) -> list[float]:
        """
        Get the history of computed Sample Entropy values.

        :return: A copy of the internal SampEn sequence evaluated at processed steps.
        :rtype: list[float]
        """
        return list(self._entropy_values)

    def get_current_r(self) -> Optional[float]:
        """
        Get the current tolerance radius ``r``.

        :return: The current tolerance radius.
                 - If a fixed ``r`` was provided, it is returned.
                 - If adaptive, returns ``r_factor * std(current_window)`` if available.
                 - Otherwise, ``None``.
        :rtype: float or None
        """
        if self._r is not None:
            return self._r

        if len(self._buffer) > 0:
            std_dev = np.std(list(self._buffer))
            return float(self._r_factor * std_dev) if std_dev > 0 else None

        return None

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
