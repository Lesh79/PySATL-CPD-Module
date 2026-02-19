"""
Module implementing the Slope Entropy algorithm for online change-point detection.

The detector slides a fixed-size window over a univariate time series, encodes
adjacent differences (slopes) into a finite alphabet using two slope thresholds
``delta`` and ``gamma``, estimates the entropy of the resulting slope-pattern
distribution, and triggers a change when short-term fluctuations in this entropy
exceed a decision threshold.
"""

__author__ = "Kirill Gribanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections import deque
from typing import Any, Optional

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class SlopeEntropyAlgorithm(OnlineAlgorithm):
    """
    Online change-point detector based on slope symbolization and entropy.

    :param window_size: Sliding window length used to compute slope entropy. Default: ``100``.
    :type window_size: int
    :param embedding_dim: Length (in samples) of each subsequence used to form slope patterns.
        Each subsequence of length ``embedding_dim`` yields ``embedding_dim - 1`` slope symbols.
        Default: ``3``.
    :type embedding_dim: int
    :param gamma: Upper slope threshold. Slopes greater than ``gamma`` (or less than ``-gamma``)
        are encoded as extreme symbols ``2`` and ``-2`` respectively. Default: ``1.0``.
    :type gamma: float
    :param delta: Inner slope threshold separating flat/tie regions from gentle slopes.
        Must satisfy ``0 <= delta < gamma``. Default: ``1e-3``.
    :type delta: float
    :param threshold: Decision threshold for triggering changes from short-term entropy dynamics.
        Default: ``0.3``.
    :type threshold: float
    :param normalize: If ``True``, normalize entropy by the maximum possible entropy given the
        set of observed patterns in the current window (base-2). Default: ``True``.
    :type normalize: bool

    . note::
       - The algorithm encodes slopes into five possible symbols: {-2, -1, 0, 1, 2}.
       - Change localization is approximate and based on recent short-term entropy fluctuations.
    """

    def __init__(
        self,
        window_size: int = 100,
        embedding_dim: int = 3,
        gamma: float = 1.0,
        delta: float = 1e-3,
        threshold: float = 0.3,
        normalize: bool = True,
    ):
        super().__init__()
        self._window_size = window_size
        self._embedding_dim = embedding_dim
        self._gamma = gamma
        self._delta = delta
        self._threshold = threshold
        self._normalize = normalize

        if delta >= gamma:
            raise ValueError(f"delta ({delta}) must be less than gamma ({gamma})")

        self._buffer: deque[float] = deque(maxlen=window_size)
        self._entropy_values: deque[float] = deque(maxlen=200)
        self._position: int = 0
        self._last_change_point: Optional[int] = None

    def detect(self, observation: np.float64 | npt.NDArray[np.float64]) -> bool:
        """
        Ingest a new observation (or a batch) and update the internal detection state.

        :param observation: A single value or 1-D array of values to process sequentially.
        :type observation: float or numpy.ndarray
        :return: ``True`` if a change-point was flagged after processing the input,
                 ``False`` otherwise.
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
        Process input and return the index of a detected change-point if present.

        :param observation: A single value or 1-D array of values to process.
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
        Process a single new observation and update slope-entropy statistics.

        :param observation: New value to be appended to the rolling buffer.
        :type observation: float

        . note::
           Detection logic combines three short-term tests:

           1. Absolute difference between the last two entropy values.
           2. Difference between means of two consecutive 5-sample entropy blocks.
           3. Difference between variances of two consecutive 4-sample entropy blocks.

           If any test exceeds its (scaled) threshold, a change-point is flagged and localized
           near the center or end of the current window.
        """
        self._buffer.append(observation)
        self._position += 1

        if len(self._buffer) < self._window_size:
            return

        current_window = np.fromiter(self._buffer, dtype=float)
        current_entropy = self._calculate_slope_entropy_vectorized(current_window)

        if np.isinf(current_entropy) or np.isnan(current_entropy):
            current_entropy = 0.0

        if len(self._entropy_values) >= 1 and abs(current_entropy - self._entropy_values[-1]) > self._threshold:
            self._last_change_point = self._position - self._window_size // 2

        self._entropy_values.append(current_entropy)

        hist_list = list(self._entropy_values)
        long_win = 10
        short_win = 8

        if (
            len(hist_list) >= long_win
            and abs(np.mean(hist_list[-5:]) - np.mean(hist_list[-10:-5])) > self._threshold * 0.8
        ):
            self._last_change_point = self._position - 2

        if (
            len(hist_list) >= short_win
            and abs(np.var(hist_list[-4:]) - np.var(hist_list[-8:-4])) > self._threshold * 0.5
        ):
            self._last_change_point = self._position - 1

    def _calculate_slope_entropy_vectorized(self, time_series: npt.NDArray[np.float64]) -> float:
        """
        Compute slope entropy for the given window using vectorized approach.

        :param time_series: Current rolling window.
        :type time_series: numpy.ndarray
        :return: Base-2 entropy of the empirical distribution of slope patterns.
                 Returns ``0.0`` if the window is too short or if no patterns can be formed.
        :rtype: float

        . note::
           - A slope pattern is formed by encoding the ``embedding_dim - 1`` adjacent
             differences inside each length-``embedding_dim`` subsequence.
           - If ``normalize=True``, the entropy is divided by the maximum possible
             entropy (``log2(#observed_patterns)``) to yield a normalized value in ``[0, 1]``.
        """
        diffs = np.diff(time_series)
        symbols = self._symbolize_slopes(diffs)

        m_pattern = self._embedding_dim - 1
        n_patterns = len(symbols) - m_pattern + 1
        if n_patterns <= 0:
            return 0.0

        shape = (n_patterns, m_pattern)
        strides = (symbols.strides[0], symbols.strides[0])
        patterns = np.lib.stride_tricks.as_strided(symbols, shape=shape, strides=strides)

        _, counts = np.unique(patterns, axis=0, return_counts=True)
        probs = counts / n_patterns
        entropy = -np.sum(probs * np.log2(probs))

        if self._normalize:
            max_entropy = np.log2(len(counts)) if len(counts) > 1 else 1.0
            entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        return float(entropy)

    def _symbolize_slopes(self, diffs: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
        symbols = np.zeros_like(diffs, dtype=int)
        symbols[diffs > self._gamma] = 2
        symbols[(diffs > self._delta) & (diffs <= self._gamma)] = 1
        symbols[(diffs < -self._delta) & (diffs >= -self._gamma)] = -1
        symbols[diffs < -self._gamma] = -2
        return symbols

    def _create_slope_pattern(self, subsequence: npt.NDArray[np.float64]) -> list[int]:
        """
        Encode a length-``embedding_dim`` subsequence into a slope pattern.

        :param subsequence: Subsequence from the current window.
        :type subsequence: numpy.ndarray
        :return: List of slope symbols with values in ``{-2, -1, 0, 1, 2}``.
        :rtype: list[int]

        . rubric:: Encoding rules

        For a slope ``d = x[i] - x[i-1]``:

        - ``d >  gamma`` → ``2``  (steep positive)
        - ``delta < d <= gamma`` → ``1``  (gentle positive)
        - ``|d| <= delta`` → ``0``  (flat/ties)
        - ``-gamma <= d < -delta`` → ``-1`` (gentle negative)
        - ``d < -gamma`` → ``-2`` (steep negative)
        """
        diffs = np.diff(subsequence)
        return [int(x) for x in self._symbolize_slopes(diffs)]

    def get_symbol_meanings(self) -> dict[int, str]:
        return {
            2: f"steep positive (d > {self._gamma})",
            1: f"gentle positive ({self._delta} < d <= {self._gamma})",
            0: f"flat/ties (|d| <= {self._delta})",
            -1: f"gentle negative (-{self._gamma} <= d < -{self._delta})",
            -2: f"steep negative (d < -{self._gamma})",
        }

    def get_pattern_distribution(self) -> dict[tuple[int, ...], float]:
        if len(self._buffer) < self._embedding_dim:
            return {}

        current_window = np.fromiter(self._buffer, dtype=float)
        diffs = np.diff(current_window)
        symbols = self._symbolize_slopes(diffs)

        m_pattern = self._embedding_dim - 1
        n_patterns = len(symbols) - m_pattern + 1
        if n_patterns <= 0:
            return {}

        shape = (n_patterns, m_pattern)
        strides = (symbols.strides[0], symbols.strides[0])
        patterns = np.lib.stride_tricks.as_strided(symbols, shape=shape, strides=strides)

        unique_p, counts = np.unique(patterns, axis=0, return_counts=True)
        return {tuple(p): float(c / n_patterns) for p, c in zip(unique_p, counts)}

    def analyze_slope_characteristics(self) -> dict[str, Any]:
        if len(self._buffer) < self._embedding_dim:
            return {}

        current_window = np.fromiter(self._buffer, dtype=float)
        diffs = np.diff(current_window)
        symbols = self._symbolize_slopes(diffs)

        entropy = self._calculate_slope_entropy_vectorized(current_window)

        steep_pos_sym = 2
        gentle_pos_sym = 1
        flat_sym = 0
        gentle_neg_sym = -1
        steep_neg_sym = -2

        return {
            "slope_entropy": entropy,
            "steep_positive_ratio": float(np.mean(symbols == steep_pos_sym)),
            "gentle_positive_ratio": float(np.mean(symbols == gentle_pos_sym)),
            "flat_ratio": float(np.mean(symbols == flat_sym)),
            "gentle_negative_ratio": float(np.mean(symbols == gentle_neg_sym)),
            "steep_negative_ratio": float(np.mean(symbols == steep_neg_sym)),
            "slope_mean": float(np.mean(diffs)),
            "slope_std": float(np.std(diffs)),
            "slope_variance": float(np.var(diffs)),
            "total_patterns": len(symbols) - (self._embedding_dim - 2),
        }

    def demonstrate_encoding(self, data: list[float]) -> dict[str, Any]:
        if len(data) < self._embedding_dim:
            return {"error": "insufficient data", "original_data": data, "encoding_rules": self.get_symbol_meanings()}

        data_arr = np.array(data)
        diffs = np.diff(data_arr)
        symbols = self._symbolize_slopes(diffs)

        m_pattern = self._embedding_dim - 1
        n_patterns = len(symbols) - m_pattern + 1
        patterns = [tuple(symbols[i : i + m_pattern]) for i in range(n_patterns)]

        return {
            "original_data": data,
            "slopes": diffs.tolist(),
            "symbols": symbols.tolist(),
            "patterns": patterns,
            "slope_entropy": self._calculate_slope_entropy_vectorized(data_arr),
            "encoding_rules": self.get_symbol_meanings(),
        }

    def get_entropy_history(self) -> list[float]:
        """
        Get the history of computed slope-entropy values.

        :return: A copy of the internal slope-entropy sequence evaluated at processed steps.
        :rtype: list[float]
        """
        return list(self._entropy_values)

    def get_current_parameters(self) -> dict[str, Any]:
        """
        Get the current configuration of the detector.

        :return: Dictionary with current parameter settings.
        :rtype: dict
        """
        return {
            "window_size": self._window_size,
            "embedding_dim": self._embedding_dim,
            "gamma": self._gamma,
            "delta": self._delta,
            "threshold": self._threshold,
            "normalize": self._normalize,
            "max_symbols": 5,
            "max_patterns": 5 ** (self._embedding_dim - 1),
        }

    def set_parameters(self, **kwargs: Any) -> None:
        """
        Update detector parameters in place.

        :param embedding_dim: New subsequence length. Default: ``None``.
        :type embedding_dim: int, optional
        :param gamma: New upper slope threshold. Default: ``None``.
        :type gamma: float, optional
        :param delta: New inner slope threshold (must remain strictly less than ``gamma``).
        :type delta: float, optional
        :param threshold: New decision threshold for change detection.
        :type threshold: float, optional
        :param normalize: Whether to normalize entropy to ``[0, 1]``.
        :type normalize: bool, optional
        :raises ValueError: If ``delta`` is not strictly less than ``gamma``.
        """
        for key, value in kwargs.items():
            if hasattr(self, f"_{key}"):
                setattr(self, f"_{key}", value)
        if self._delta >= self._gamma:
            raise ValueError(f"delta ({self._delta}) must be less than gamma ({self._gamma})")

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
