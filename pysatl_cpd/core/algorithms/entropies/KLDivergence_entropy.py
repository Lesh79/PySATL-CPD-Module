"""
Module implementing the Kullback Leibler Divergence (KLD) algorithm for online change-point detection.

The detector maintains two sliding windows: a *reference* window and a *current* window.
It computes the divergence between the empirical distributions of these windows using either
histogram binning or kernel density estimation (KDE). A change-point is signaled when
the divergence exceeds a user-defined threshold or when a sustained divergence trend is observed.

Two options are supported:

1. Histogram-based KL divergence with Laplace smoothing.
2. KDE-based KL divergence with Gaussian kernels and continuous evaluation.

Optionally, the divergence can be made symmetric:

.. math::

   KL_{sym}(P, Q) = 0.5 \\times [ KL(P \\| Q) + KL(Q \\| P) ]

This implementation supports streaming (online) processing of time series data.
"""

__author__ = "Kirill Gribanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections import deque
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from scipy import stats

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class KLDivergenceAlgorithm(OnlineAlgorithm):
    """
    Online change-point detector based on Kullback Leibler Divergence.

    :param window_size: Size of the current sliding window. Default: ``100``.
    :type window_size: int
    :param reference_window_size: Size of the reference window. If ``None``, set equal to ``window_size``.
    :type reference_window_size: int or None
    :param threshold: Threshold above which KL divergence indicates a change. Default: ``0.5``.
    :type threshold: float
    :param num_bins: Number of histogram bins used if histogram mode is active. Default: ``20``.
    :type num_bins: int
    :param use_kde: Whether to use KDE instead of histograms for estimating distributions. Default: ``False``.
    :type use_kde: bool
    :param symmetric: If ``True``, compute symmetric KL divergence. Default: ``True``.
    :type symmetric: bool
    :param smoothing_factor: Additive smoothing factor to avoid
    division by zero in probability estimates. Default: ``1e-10``.
    :type smoothing_factor: float

    .. note::
       - Uses histogram or KDE estimates to approximate probability densities.
       - The reference distribution is updated whenever a change is detected.
    """

    def __init__(
        self,
        window_size: int = 100,
        reference_window_size: Optional[int] = None,
        threshold: float = 0.5,
        num_bins: int = 20,
        use_kde: bool = False,
        symmetric: bool = True,
        smoothing_factor: float = 1e-10,
        anomaly_threshold: float = 3.0,
    ):
        super().__init__()
        self._window_size = window_size
        self._reference_window_size = reference_window_size or window_size
        self._threshold = threshold
        self._num_bins = num_bins
        self._use_kde = use_kde
        self._symmetric = symmetric
        self._smoothing_factor = smoothing_factor
        self._anomaly_threshold = anomaly_threshold

        if self._window_size <= 0 or self._reference_window_size <= 0:
            raise ValueError("Window sizes must be positive")
        if self._num_bins <= 1:
            raise ValueError("Number of bins must be greater than 1")
        if self._threshold <= 0:
            raise ValueError("Threshold must be positive")

        self._reference_buffer: deque[float] = deque(maxlen=self._reference_window_size)
        self._current_buffer: deque[float] = deque(maxlen=self._window_size)

        self._reference_array: Optional[npt.NDArray[np.float64]] = None

        self._kl_values: deque[float] = deque(maxlen=200)

        self._position: int = 0
        self._last_change_point: Optional[int] = None
        self._reference_updated: bool = False

    def detect(self, observation: np.float64 | npt.NDArray[np.float64]) -> bool:
        """
        Ingest a new observation (or batch) and update the detection state.

        :param observation: A single value or array of values to process.
        :type observation: float or numpy.ndarray
        :return: ``True`` if a change-point is flagged, otherwise ``False``.
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
        Process input and return index of detected change-point.

        :param observation: Incoming sample(s).
        :type observation: float or numpy.ndarray
        :return: Approximate index of detected change-point, or ``None`` if none found.
        :rtype: int or None
        """
        if self.detect(observation):
            cp = self._last_change_point
            self._last_change_point = None
            return cp
        return None

    def _process_single_observation(self, observation: float) -> None:
        """
        Process a single observation, update buffers, compute KL divergence if ready.

        :param observation: New value from the stream.
        :type observation: float
        """
        self._position += 1
        if len(self._reference_buffer) < self._reference_window_size:
            self._reference_buffer.append(observation)
            self._current_buffer.append(observation)

            if len(self._reference_buffer) == self._reference_window_size:
                self._reference_array = np.array(self._reference_buffer)
            return

        if len(self._current_buffer) >= self._window_size // 2:
            current_mean = np.mean(self._current_buffer)
            if abs(observation - current_mean) > self._anomaly_threshold:
                self._last_change_point = self._position

        self._current_buffer.append(observation)

        if len(self._current_buffer) < self._window_size:
            return

        if self._reference_array is None:
            self._reference_array = np.array(self._reference_buffer)

        current_data = np.fromiter(self._current_buffer, dtype=float)

        kl_divergence = self._calculate_kl_divergence(self._reference_array, current_data)

        if np.isinf(kl_divergence) or np.isnan(kl_divergence):
            kl_divergence = 0.0

        self._kl_values.append(kl_divergence)

        if kl_divergence > self._threshold:
            self._last_change_point = self._position - self._window_size // 2
            self._update_reference_distribution()
            return

        min_history = 5
        if len(self._kl_values) >= min_history:
            recent_kl = list(self._kl_values)[-min_history:]
            kl_trend = np.mean(recent_kl)
            if kl_trend > self._threshold * 0.8:
                self._last_change_point = self._position - self._window_size // 4
                self._update_reference_distribution()

    def _calculate_kl_divergence(self, ref_data: npt.NDArray[np.float64], curr_data: npt.NDArray[np.float64]) -> float:
        """
        Compute KL divergence between reference and current windows.

        :return: KL divergence estimate.
        :rtype: float
        """
        if self._use_kde:
            return self._calculate_kl_divergence_kde(ref_data, curr_data)
        else:
            return self._calculate_kl_divergence_histogram(ref_data, curr_data)

    def _calculate_kl_divergence_histogram(
        self, ref_data: npt.NDArray[np.float64], curr_data: npt.NDArray[np.float64]
    ) -> float:
        """
        Compute KL divergence using histogram binning.

        :param ref_data: Reference window.
        :type ref_data: numpy.ndarray
        :param curr_data: Current window.
        :type curr_data: numpy.ndarray
        :return: KL divergence (symmetric if ``symmetric=True``).
        :rtype: float
        """
        data_min = min(ref_data.min(), curr_data.min())
        data_max = max(ref_data.max(), curr_data.max())

        margin = (data_max - data_min) * 0.01
        if margin == 0:
            return 0.0

        bin_edges = np.linspace(data_min - margin, data_max + margin, self._num_bins + 1)

        ref_hist, _ = np.histogram(ref_data, bins=bin_edges, density=True)
        curr_hist, _ = np.histogram(curr_data, bins=bin_edges, density=True)

        ref_prob = ref_hist + self._smoothing_factor
        curr_prob = curr_hist + self._smoothing_factor

        ref_prob /= np.sum(ref_prob)
        curr_prob /= np.sum(curr_prob)

        kl_pq = np.sum(ref_prob * np.log(ref_prob / curr_prob))

        if self._symmetric:
            kl_qp = np.sum(curr_prob * np.log(curr_prob / ref_prob))
            return float((kl_pq + kl_qp) / 2)

        return float(kl_pq)

    def _calculate_kl_divergence_kde(
        self, ref_data: npt.NDArray[np.float64], curr_data: npt.NDArray[np.float64]
    ) -> float:
        """
        Compute KL divergence using Gaussian KDE estimates.

        :param ref_data: Reference window.
        :type ref_data: numpy.ndarray
        :param curr_data: Current window.
        :type curr_data: numpy.ndarray
        :return: KL divergence (symmetric if ``symmetric=True``).
        :rtype: float
        """
        try:
            ref_kde = stats.gaussian_kde(ref_data)
            curr_kde = stats.gaussian_kde(curr_data)
        except (np.linalg.LinAlgError, ValueError):
            return 0.0

        data_min = min(ref_data.min(), curr_data.min())
        data_max = max(ref_data.max(), curr_data.max())
        margin = (data_max - data_min) * 0.1

        x_eval = np.linspace(data_min - margin, data_max + margin, 100)
        dx = x_eval[1] - x_eval[0] if len(x_eval) > 1 else 1.0

        ref_density = ref_kde(x_eval) + self._smoothing_factor
        curr_density = curr_kde(x_eval) + self._smoothing_factor

        ref_density /= np.sum(ref_density) * dx
        curr_density /= np.sum(curr_density) * dx

        kl_pq = np.sum(ref_density * np.log(ref_density / curr_density)) * dx

        if self._symmetric:
            kl_qp = np.sum(curr_density * np.log(curr_density / ref_density)) * dx
            return float((kl_pq + kl_qp) / 2)

        return float(kl_pq)

    def _update_reference_distribution(self) -> None:
        """Reset the reference buffer using the current buffer (after detection)."""
        self._reference_buffer.clear()
        self._reference_buffer.extend(self._current_buffer)
        self._reference_array = np.array(self._reference_buffer)
        self._reference_updated = True

    def get_kl_history(self) -> list[float]:
        """
        Return history of computed KL divergence values.

        :return: List of past KL divergence values.
        :rtype: list[float]
        """
        return list(self._kl_values)

    def get_current_parameters(self) -> dict[str, float | int | bool]:
        """
        Get current hyperparameters as dictionary.

        :return: Dictionary with keys:
                 ``window_size``, ``reference_window_size``, ``threshold``,
                 ``num_bins``, ``use_kde``, ``symmetric``, ``smoothing_factor``.
        :rtype: dict
        """
        return {
            "window_size": self._window_size,
            "reference_window_size": self._reference_window_size,
            "threshold": self._threshold,
            "num_bins": self._num_bins,
            "use_kde": self._use_kde,
            "symmetric": self._symmetric,
            "smoothing_factor": self._smoothing_factor,
        }

    def set_parameters(
        self,
        threshold: Optional[float] = None,
        num_bins: Optional[int] = None,
        use_kde: Optional[bool] = None,
        symmetric: Optional[bool] = None,
        smoothing_factor: Optional[float] = None,
    ) -> None:
        """
        Update detector hyperparameters.

        :param threshold: New detection threshold (>0).
        :type threshold: float or None
        :param num_bins: New number of histogram bins (>1).
        :type num_bins: int or None
        :param use_kde: Switch to KDE or histogram mode.
        :type use_kde: bool or None
        :param symmetric: Enable or disable symmetric KL computation.
        :type symmetric: bool or None
        :param smoothing_factor: New smoothing factor for probability estimates.
        :type smoothing_factor: float or None
        :raises ValueError: If provided values are invalid.
        """
        if threshold is not None:
            if threshold <= 0:
                raise ValueError("Threshold must be positive")
            self._threshold = threshold

        if num_bins is not None:
            if num_bins <= 1:
                raise ValueError("Number of bins must be greater than 1")
            self._num_bins = num_bins

        if use_kde is not None:
            self._use_kde = use_kde
        if symmetric is not None:
            self._symmetric = symmetric
        if smoothing_factor is not None:
            self._smoothing_factor = smoothing_factor

    def get_distribution_comparison(self) -> dict[str, float]:
        """
        Compare reference and current distributions using multiple statistics.

        :return: Includes:
                 - ``kl_divergence``
                 - ``reference_mean``, ``reference_std``
                 - ``current_mean``, ``current_std``
                 - ``mean_difference``
                 - ``std_ratio``
                 - ``ks_statistic``, ``ks_pvalue``
        :rtype: dict
        """
        if self._reference_array is None or len(self._current_buffer) < self._window_size:
            return {}

        ref_data = self._reference_array
        curr_data = np.fromiter(self._current_buffer, dtype=float)

        ref_mean, ref_std = np.mean(ref_data), np.std(ref_data)
        curr_mean, curr_std = np.mean(curr_data), np.std(curr_data)

        kl_div = self._calculate_kl_divergence(ref_data, curr_data)
        ks_statistic, ks_pvalue = stats.ks_2samp(ref_data, curr_data)

        return {
            "kl_divergence": kl_div,
            "reference_mean": ref_mean,
            "reference_std": ref_std,
            "current_mean": curr_mean,
            "current_std": curr_std,
            "mean_difference": abs(curr_mean - ref_mean),
            "std_ratio": curr_std / ref_std if ref_std > 0 else float("inf"),
            "ks_statistic": ks_statistic,
            "ks_pvalue": ks_pvalue,
        }

    def analyze_distributions(self) -> dict[str, Any]:
        """
        Provide extended diagnostic statistics between reference and current windows.

        :return: Extends :meth:`get_distribution_comparison` with:
                 - ``reference_entropy``, ``current_entropy``
                 - ``entropy_difference``
                 - ``reference_quantiles``, ``current_quantiles``
                 - ``quantile_differences``
        :rtype: dict
        """
        if self._reference_array is None:
            return {}

        comparison = self.get_distribution_comparison()
        if not comparison:
            return {}

        ref_data = self._reference_array
        curr_data = np.fromiter(self._current_buffer, dtype=float)

        ref_h, _ = np.histogram(ref_data, bins=self._num_bins)
        curr_h, _ = np.histogram(curr_data, bins=self._num_bins)

        ref_entropy = stats.entropy(ref_h + self._smoothing_factor)
        curr_entropy = stats.entropy(curr_h + self._smoothing_factor)

        quantiles = [0.25, 0.5, 0.75]
        ref_quantiles = np.quantile(ref_data, quantiles)
        curr_quantiles = np.quantile(curr_data, quantiles)

        return {
            **comparison,
            "reference_entropy": ref_entropy,
            "current_entropy": curr_entropy,
            "entropy_difference": abs(curr_entropy - ref_entropy),
            "reference_quantiles": ref_quantiles.tolist(),
            "current_quantiles": curr_quantiles.tolist(),
            "quantile_differences": (np.abs(curr_quantiles - ref_quantiles)).tolist(),
        }

    def reset(self) -> None:
        """
        Clear all buffers, history, and reset state.

        :return: ``None``.
        :rtype: None
        """
        self._reference_buffer.clear()
        self._current_buffer.clear()
        self._kl_values.clear()
        self._reference_array = None
        self._position = 0
        self._last_change_point = None
        self._reference_updated = False

    def force_reference_update(self) -> None:
        """
        Forcefully refresh the reference buffer with the current buffer.

        .. note::
           This is typically called when no automatic change update is triggered,
           but the user wants to realign reference and current windows.
        """
        if len(self._current_buffer) >= self._window_size:
            self._update_reference_distribution()
