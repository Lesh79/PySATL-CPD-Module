"""
Module implementing the Dispersion Entropy (DisEn) algorithm for online change-point detection.

The detector maintains a rolling window over a univariate time series, maps values to
``c`` discrete classes via the Gaussian CDF, forms dispersion patterns of length ``m``
with time delay ``τ``, and computes the Shannon entropy of the pattern distribution:

. math::

   \\mathrm{DisEn}(m, c, \\tau) = - \\sum_{\\text{pattern}} p(\\text{pattern})\\,\\log p(\\text{pattern})

where ``c`` is the number of classes, ``m`` is the embedding dimension, and ``τ`` is the delay.
Optionally, the entropy can be normalized by the maximal value ``log(c^m)``.

A change-point is signaled when any of the following holds:

1. The absolute difference between consecutive entropy values exceeds ``threshold``;
2. The short-term variance of entropy exceeds ``threshold``;
3. The mean shift between two consecutive short windows of entropy exceeds ``1.5 * threshold``.

The implementation supports streaming (online) processing and returns approximate
change-point indices relative to the processed stream.
"""

__author__ = "Kirill Gribanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections import deque
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class DispersionEntropyAlgorithm(OnlineAlgorithm):
    """
    Online change-point detector based on Dispersion Entropy (DisEn).

    :param window_size: Sliding window length used for entropy computation. Default: ``100``.
    :type window_size: int
    :param embedding_dim: Embedding dimension ``m`` of dispersion patterns. Default: ``3``.
    :type embedding_dim: int
    :param num_classes: Number of discrete classes ``c`` used to quantize the window via Gaussian CDF. Default: ``6``.
    :type num_classes: int
    :param time_delay: Delay ``τ`` between consecutive elements of a dispersion pattern. Default: ``1``.
    :type time_delay: int
    :param threshold: Decision threshold used in the detection criteria. Default: ``0.2``.
    :type threshold: float
    :param normalize: If ``True``, normalize entropy by ``log(c^m)``. Default: ``True``.
    :type normalize: bool

    .. note::
       - If ``c^m >= window_size``, the pattern space becomes too large for reliable
         estimation from the window; a ``ValueError`` is raised.
       - Observations are processed online; change localization is approximate and tied
         to the center/quarter of the active window depending on which criterion triggers.
    """

    def __init__(
        self,
        window_size: int = 100,
        embedding_dim: int = 3,
        num_classes: int = 6,
        time_delay: int = 1,
        threshold: float = 0.2,
        normalize: bool = True,
        anomaly_threshold: float = 3.0,
    ):
        super().__init__()
        self._window_size = window_size
        self._embedding_dim = embedding_dim
        self._num_classes = num_classes
        self._time_delay = time_delay
        self._threshold = threshold
        self._normalize = normalize
        self._anomaly_threshold = anomaly_threshold

        if num_classes**embedding_dim >= window_size:
            raise ValueError(
                f"c^w ({num_classes}^{embedding_dim} = {num_classes**embedding_dim}) "
                f"should be less than window_size ({window_size})"
            )

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
            change_point = self._last_change_point
            self._last_change_point = None
            return change_point
        return None

    def _process_single_observation(self, observation: float) -> None:
        """
        Process a single new observation and update the internal DisEn statistics.

        :param observation: New value to be appended to the rolling buffer.
        :type observation: float
        """
        if len(self._buffer) >= self._window_size // 2:
            current_mean = np.mean(self._buffer)
            if abs(observation - current_mean) > self._anomaly_threshold:
                self._last_change_point = self._position

        self._buffer.append(observation)
        self._position += 1

        v = 10
        v1 = 2

        if len(self._buffer) < self._window_size:
            return

        current_window = np.fromiter(self._buffer, dtype=float)

        if np.std(current_window) == 0:
            current_entropy = 0.0
        else:
            current_entropy = self._calculate_dispersion_entropy_vectorized(current_window)

        if np.isinf(current_entropy) or np.isnan(current_entropy):
            current_entropy = 0.0

        self._entropy_values.append(current_entropy)

        history_len = len(self._entropy_values)
        entropy_history = list(self._entropy_values)

        if history_len >= v1:
            entropy_diff = abs(entropy_history[-1] - entropy_history[-2])
            if entropy_diff > self._threshold:
                self._last_change_point = self._position - self._window_size // 2

        if history_len >= v1 + 3:
            recent_entropies = entropy_history[-5:]
            entropy_variance = np.var(recent_entropies)
            if entropy_variance > self._threshold:
                self._last_change_point = self._position - self._window_size // 4

        if history_len >= v:
            window1 = entropy_history[-10:-5]
            window2 = entropy_history[-5:]
            mean1 = np.mean(window1)
            mean2 = np.mean(window2)
            if abs(mean2 - mean1) > self._threshold * 1.5:
                self._last_change_point = self._position - 2

    def _calculate_dispersion_entropy_vectorized(self, time_series: npt.NDArray[np.float64]) -> float:
        """
        Compute Dispersion Entropy for the given window using vectorized approach.

        **Steps**

        1. Standardize to ``N(μ, o)`` and apply Gaussian CDF to obtain ``y in (0, 1)``;
        2. Discretize into ``c`` classes to get integer-coded series ``z`` in ``{1, ..., c}``;
        3. Form dispersion patterns of length ``m`` with delay ``τ``;
        4. Compute the Shannon entropy of pattern probabilities and (optionally) normalize.

        :param time_series: Current rolling window.
        :type time_series: numpy.ndarray
        :return: Dispersion Entropy value for the window. Returns ``0.0`` when input is too short or degenerate.
        :rtype: float
        """
        N = len(time_series)

        mu = np.mean(time_series)
        sigma = np.std(time_series)

        y_series = norm.cdf(time_series, loc=mu, scale=sigma)
        z_series = np.clip(np.round(self._num_classes * y_series + 0.5), 1, self._num_classes).astype(np.int32)
        n_patterns = N - (self._embedding_dim - 1) * self._time_delay

        if n_patterns <= 0:
            return 0.0

        shape = (n_patterns, self._embedding_dim)
        itemsize = z_series.strides[0]
        strides = (itemsize, itemsize * self._time_delay)
        patterns_view = np.lib.stride_tricks.as_strided(z_series, shape=shape, strides=strides)
        _, counts = np.unique(patterns_view, axis=0, return_counts=True)

        probs = counts / n_patterns
        entropy = -np.sum(probs * np.log(probs))

        if self._normalize:
            max_entropy = np.log(self._num_classes**self._embedding_dim)
            if max_entropy > 0:
                entropy /= max_entropy

        return float(entropy)

    def _discretize_to_classes(self, y_series: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
        """
        Map continuous values ``y in (0, 1)`` to integer classes ``{1, ..., c}``.

        :param y_series: Values after applying the Gaussian CDF to the window.
        :type y_series: numpy.ndarray
        :return: Discrete class labels in the range ``[1, c]``.
        :rtype: numpy.ndarray
        """
        rc_values = self._num_classes * y_series + 0.5
        z_series = np.round(rc_values).astype(np.int32)
        z_series = np.clip(z_series, 1, self._num_classes)
        return z_series

    def _create_dispersion_patterns(self, z_series: npt.NDArray[np.int32]) -> list[tuple[int, ...]]:
        """
        Construct dispersion patterns of length ``m`` with delay ``τ`` from class labels.

        :param z_series: Discrete class labels in ``[1, c]``.
        :type z_series: numpy.ndarray
        :return: List of length-``m`` patterns encoded as integer tuples.
        :rtype: list[tuple[int, ...]]
        """
        N = len(z_series)
        patterns = []
        for i in range(N - (self._embedding_dim - 1) * self._time_delay):
            pattern = []
            for j in range(self._embedding_dim):
                idx = i + j * self._time_delay
                if idx < N:
                    pattern.append(z_series[idx])
            if len(pattern) == self._embedding_dim:
                patterns.append(tuple(pattern))
        return patterns

    def get_entropy_history(self) -> list[float]:
        """
        Get the history of computed Dispersion Entropy values.

        :return: A copy of the internal DisEn sequence evaluated at processed steps.
        :rtype: list[float]
        """
        return list(self._entropy_values)

    def get_current_parameters(self) -> dict[str, int | float | bool]:
        """
        Return a snapshot of the current algorithm parameters and derived capacities.

        :return: Dictionary with keys:
                 ``window_size``, ``embedding_dim``, ``num_classes``, ``time_delay``,
                 ``threshold``, ``normalize``, and ``max_patterns`` (i.e., ``c^m``).
        :rtype: dict
        """
        return {
            "window_size": self._window_size,
            "embedding_dim": self._embedding_dim,
            "num_classes": self._num_classes,
            "time_delay": self._time_delay,
            "threshold": self._threshold,
            "normalize": self._normalize,
            "max_patterns": self._num_classes**self._embedding_dim,
        }

    def set_parameters(
        self,
        embedding_dim: Optional[int] = None,
        num_classes: Optional[int] = None,
        time_delay: Optional[int] = None,
        threshold: Optional[float] = None,
        normalize: Optional[bool] = None,
    ) -> None:
        """
        Update detector hyperparameters in place.

        :param embedding_dim: New embedding dimension ``m``.
        :type embedding_dim: int or None
        :param num_classes: New number of classes ``c``.
        :type num_classes: int or None
        :param time_delay: New delay ``τ`` between pattern elements.
        :type time_delay: int or None
        :param threshold: New decision threshold.
        :type threshold: float or None
        :param normalize: Whether to normalize entropy values.
        :type normalize: bool or None
        :raises ValueError: If the updated configuration violates ``c^m < window_size``.
        """
        if embedding_dim is not None:
            self._embedding_dim = embedding_dim
        if num_classes is not None:
            self._num_classes = num_classes
        if time_delay is not None:
            self._time_delay = time_delay
        if threshold is not None:
            self._threshold = threshold
        if normalize is not None:
            self._normalize = normalize

        if self._num_classes**self._embedding_dim >= self._window_size:
            raise ValueError(
                f"c^w ({self._num_classes}^{self._embedding_dim}) should be less than window_size ({self._window_size})"
            )

    def get_pattern_distribution(self) -> dict[tuple[int, ...], int]:
        """
        Return pattern counts for the current window (for diagnostics/inspection).

        :return: Mapping from pattern to its count in the current window.
                 Returns an empty dict if the buffer is not yet filled enough.
        :rtype: dict[tuple[int, ...], int]
        """
        if len(self._buffer) < self._window_size:
            return {}

        current_window = np.fromiter(self._buffer, dtype=float)
        mu = np.mean(current_window)
        sigma = np.std(current_window)
        if sigma == 0:
            return {}

        y_series = norm.cdf(current_window, loc=mu, scale=sigma)
        z_series = self._discretize_to_classes(y_series)

        n_patterns = len(z_series) - (self._embedding_dim - 1) * self._time_delay
        if n_patterns <= 0:
            return {}

        shape = (n_patterns, self._embedding_dim)
        strides = (z_series.strides[0], z_series.strides[0] * self._time_delay)
        patterns_view = np.lib.stride_tricks.as_strided(z_series, shape=shape, strides=strides)

        unique_patterns, counts = np.unique(patterns_view, axis=0, return_counts=True)

        return {tuple(pat): count for pat, count in zip(unique_patterns, counts)}

    def analyze_complexity(self) -> dict[str, Union[float, int]]:
        """
        Compute diagnostic complexity measures for the current window.

        :return: Diagnostic metrics including:
                 - ``dispersion_entropy``: current DisEn value
                 - ``normalized_entropy``: DisEn / log(c^m)
                 - ``unique_patterns``: number of distinct patterns observed
                 - ``max_possible_patterns``: ``c^m``
                 - ``pattern_diversity``: unique / max_possible
                 - ``window_std``: standard deviation of the window
                 - ``window_mean``: mean of the window

                 Returns an empty dict if the buffer is not yet filled enough.
        :rtype: dict
        """
        if len(self._buffer) < self._window_size:
            return {}

        current_window = np.fromiter(self._buffer, dtype=float)
        current_entropy = self._calculate_dispersion_entropy_vectorized(current_window)

        pattern_dist = self.get_pattern_distribution()
        unique_patterns = len(pattern_dist)

        max_possible_patterns = self._num_classes**self._embedding_dim
        pattern_diversity = unique_patterns / max_possible_patterns if max_possible_patterns > 0 else 0

        max_entropy = np.log(max_possible_patterns) if max_possible_patterns > 0 else 1

        normalized = current_entropy if self._normalize else (current_entropy / max_entropy if max_entropy > 0 else 0)

        return {
            "dispersion_entropy": current_entropy,
            "normalized_entropy": normalized if not self._normalize else current_entropy,
            "unique_patterns": unique_patterns,
            "max_possible_patterns": max_possible_patterns,
            "pattern_diversity": pattern_diversity,
            "window_std": float(np.std(current_window)),
            "window_mean": float(np.mean(current_window)),
        }

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
