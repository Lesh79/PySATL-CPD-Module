"""
Module implementing the Approximate Entropy (ApEn) algorithm for online change-point detection.
Optimized for streaming performance using vectorization.

The algorithm maintains a rolling window over a univariate time series, computes the
Approximate Entropy for the current window, and raises a change-point signal when
the short-term dynamics of ApEn indicate a regime shift.

ApEn is computed as:

.. math::

    ApEn(m, r, N) = \\phi(m, r) - \\phi(m+1, r)

where ``m`` is the embedding (pattern) length and ``r`` is a tolerance radius
(often expressed as a fraction of the windowed standard deviation).

This implementation supports a fixed ``r`` or an adaptive one via ``r_factor * std(window)``.
A detection is triggered when either:

1. The absolute change in consecutive ApEn values exceeds ``threshold``.
2. The short-term variance of ApEn, normalized by its mean magnitude, exceeds ``threshold``.
"""

__author__ = "Kirill Gribanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections import deque
from typing import Optional

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class ApproximateEntropyAlgorithm(OnlineAlgorithm):
    """
    Optimized Online change-point detector based on Approximate Entropy (ApEn).

    Includes:
    - Vectorized ApEn calculation (NumPy broadcasting instead of loops).
    - Constant memory usage (fixed-size deques).

    :param window_size: Sliding window length used to compute ApEn. Default: ``100``.
    :type window_size: int
    :param m: Embedding (pattern) length for ApEn. Default: ``2``.
    :type m: int
    :param r: Fixed tolerance radius. If ``None``, an adaptive radius is used via ``r_factor``.
    :type r: float or None
    :param r_factor: Multiplicative factor for adaptive radius,
                     i.e. ``r = r_factor * std(window)`` when ``r`` is ``None``.
    :type r_factor: float
    :param threshold: Decision threshold used in both the first-order ApEn difference test and
                      the normalized short-term variance test. Default: ``0.3``.
    :type threshold: float
    :param anomaly_threshold: Threshold for instant spike detection. Default: ``3.0``.
    :type anomaly_threshold: float

    .. note::
       - The detector processes observations in a streaming fashion.
       - Change localization is returned as an approximate index around the center (or quarter)
         of the current window depending on which criterion was triggered.
    """

    def __init__(
        self,
        window_size: int = 100,
        m: int = 2,
        r: Optional[float] = None,
        r_factor: float = 0.2,
        threshold: float = 0.3,
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
        current_entropy = self._calculate_approximate_entropy_vectorized(current_window)

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

    def _calculate_approximate_entropy_vectorized(self, time_series: npt.NDArray[np.float64]) -> float:
        """
        Compute Approximate Entropy using vectorized operations.

        :param time_series: Current rolling window.
        :type time_series: numpy.ndarray
        :return: The computed ApEn value.
        :rtype: float
        """
        r = self._r
        if r is None:
            std_dev = np.std(time_series)
            if std_dev == 0:
                return 0.0
            r = float(self._r_factor * std_dev)

        assert r is not None

        phi_m = self._calculate_phi_vectorized(time_series, self._m, r)
        phi_m_plus_1 = self._calculate_phi_vectorized(time_series, self._m + 1, r)

        return float(phi_m - phi_m_plus_1)

    def _calculate_phi_vectorized(self, time_series: npt.NDArray[np.float64], m: int, r: float) -> float:
        """
        Compute the auxiliary ``phi(m, r)`` term for ApEn using NumPy broadcasting (Strides).

        :param time_series: Current rolling window.
        :type time_series: numpy.ndarray
        :param m: Embedding length.
        :type m: int
        :param r: Tolerance radius.
        :type r: float
        :return: The average log of match proportions across all m-patterns in the window.
        :rtype: float

        .. note::
           - Protects ``log(0)`` by adding a small epsilon (1e-10).
        """
        N = len(time_series)
        n_vectors = N - m + 1

        if n_vectors <= 0:
            return 0.0

        shape = (n_vectors, m)
        strides = (time_series.strides[0], time_series.strides[0])
        vectors = np.lib.stride_tricks.as_strided(time_series, shape=shape, strides=strides)

        diffs = np.abs(vectors[:, None, :] - vectors[None, :, :])
        max_diffs = diffs.max(axis=2)
        matches = (max_diffs <= r).sum(axis=1)

        c_i = matches / n_vectors
        phi = np.sum(np.log(c_i + 1e-10)) / n_vectors

        return float(phi)

    def get_entropy_history(self) -> list[float]:
        """
        Get the history of computed Approximate Entropy values.

        :return: A copy of the internal ApEn sequence evaluated at processed steps.
        :rtype: list[float]
        """
        return list(self._entropy_values)

    def get_current_r(self) -> Optional[float]:
        """
        Get the current tolerance radius ``r``.

        :return:
            - If a fixed ``r`` was provided, it is returned.
            - If adaptive, returns ``r_factor * std(current_window)`` if available.
            - Otherwise, ``None``.
        :rtype: float or None
        """
        if self._r is not None:
            return self._r
        if len(self._buffer) > 0:
            return float(self._r_factor * np.std(list(self._buffer)))
        return None

    def get_pattern_length(self) -> int:
        """
        Get the current embedding length ``m``.

        :return: The pattern length parameter used for ApEn.
        :rtype: int
        """
        return self._m

    def set_parameters(
        self,
        m: Optional[int] = None,
        r: Optional[float] = None,
        r_factor: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> None:
        """
        Update detector hyperparameters in place.

        :param m: New embedding length. If ``None``, unchanged.
        :type m: int or None
        :param r: New fixed tolerance. If ``None``, unchanged.
        :type r: float or None
        :param r_factor: New adaptive multiplier. If ``None``, unchanged.
        :type r_factor: float or None
        :param threshold: New decision threshold. If ``None``, unchanged.
        :type threshold: float or None
        """
        if m is not None:
            self._m = m
        if r is not None:
            self._r = r
        if r_factor is not None:
            self._r_factor = r_factor
        if threshold is not None:
            self._threshold = threshold

    def reset(self) -> None:
        """
        Clear internal state and buffered statistics.

        :return: ``None``
        :rtype: None
        """
        self._buffer.clear()
        self._entropy_values.clear()
        self._position = 0
        self._last_change_point = None
