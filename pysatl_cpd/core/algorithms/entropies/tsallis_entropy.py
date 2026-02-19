"""
Module implementing the Tsallis Entropy (S_q) algorithm for online change-point detection.

This detector maintains a sliding window over a univariate time series, estimates the
(probability) distribution in the window (via histogram or KDE), computes Tsallis entropy
for a chosen entropic index ``q``, and raises a change-point signal when short-term dynamics
of S_q indicate a regime shift.

For q ≠ 1, the Tsallis entropy is::

    S_q(P) = k * (1 - Σ_i p_i^q) / (q - 1),

where {p_i} are probabilities over a finite partition (histogram bins) or a continuous
density p(x) via integration. For q → 1, S_q → Shannon entropy (not used directly here).

This implementation supports:
- discrete (histogram-based) and continuous (KDE-based) estimation,
- optional normalization by the maximum Tsallis entropy for the given number of non-zero states,
- multi-q mode that aggregates several q-values into a single score,
- multiple detection criteria (first difference, trend, variance, multi-q voting).
"""

__author__ = "Kirill Gribanov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections import deque
from typing import Any, Optional, Union

import numpy as np
import numpy.typing as npt
from numpy.typing import NDArray
from scipy import stats
from scipy.integrate import quad

from pysatl_cpd.core.algorithms.online_algorithm import OnlineAlgorithm


class TsallisEntropyAlgorithm(OnlineAlgorithm):
    """
    Online change-point detector based on Tsallis entropy S_q.

    :param window_size: Sliding window length used for entropy computation. Default: ``100``.
    :type window_size: int
    :param q_parameter: Entropic index q (q ≠ 1). Smaller q (< 1) emphasizes support/rare events;
                        larger q (> 1) emphasizes frequent events. Default: ``2.0``.
    :type q_parameter: float
    :param k_constant: Constant factor (usually ``k = 1`` in normalized units). Default: ``1.0``.
    :type k_constant: float
    :param threshold: Decision threshold for the detection rules. Default: ``0.1``.
    :type threshold: float
    :param num_bins: Number of histogram bins for discrete S_q estimation. Default: ``20``.
    :type num_bins: int
    :param use_kde: If ``True``, compute continuous Tsallis via Gaussian KDE + numerical integration;
                    if ``False``, use histogram-based discrete S_q. Default: ``False``.
    :type use_kde: bool
    :param normalize: If ``True`` (discrete mode), normalize S_q by its theoretical maximum
                      (uniform distribution over the observed non-zero states). Default: ``True``.
    :type normalize: bool
    :param multi_q: If ``True``, compute S_q for a set of q-values and combine them into one score.
                    Default: ``False``.
    :type multi_q: bool

    . note::
       - The detector processes observations in a streaming fashion.
       - Change localization is returned as an approximate index near the center/quarter of the
         current window depending on which rule was triggered.
    """

    def __init__(
        self,
        window_size: int = 100,
        q_parameter: float = 2.0,
        k_constant: float = 1.0,
        threshold: float = 0.1,
        num_bins: int = 20,
        use_kde: bool = False,
        normalize: bool = True,
        multi_q: bool = False,
    ):
        self._window_size = window_size
        self._q_parameter = q_parameter
        self._k_constant = k_constant
        self._threshold = threshold
        self._num_bins = num_bins
        self._use_kde = use_kde
        self._normalize = normalize
        self._multi_q = multi_q

        if window_size <= 0:
            raise ValueError("Window size must be positive")
        v = 1e-10
        if abs(q_parameter - 1.0) < v:
            raise ValueError("q parameter cannot be 1 (use Shannon entropy instead)")
        if k_constant <= 0:
            raise ValueError("k constant must be positive")
        if num_bins <= 1:
            raise ValueError("Number of bins must be greater than 1")

        if multi_q:
            self._q_values = [0.5, 1.5, 2.0, 2.5, 3.0]
        else:
            self._q_values = [q_parameter]

        self._buffer: deque[float] = deque(maxlen=window_size * 2)
        self._entropy_values: list[float] = []
        self._multi_entropy_values: dict[float, list[float]] = {q: [] for q in self._q_values}
        self._position: int = 0
        self._last_change_point: Optional[int] = None

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
        Process a single sample, update buffers and S_q, and run decision rules.

        :param observation: New sample from the stream.
        :type observation: float
        """
        v1 = 2
        self._buffer.append(observation)
        self._position += 1

        if len(self._buffer) < self._window_size:
            return

        current_window = np.array(list(self._buffer)[-self._window_size :])

        if self._multi_q:
            entropies = {}
            for q in self._q_values:
                entropy = self._calculate_tsallis_entropy(current_window, q)
                entropies[q] = entropy
                self._multi_entropy_values[q].append(entropy)
            current_entropy = self._combine_multi_q_entropies(entropies)
        else:
            current_entropy = self._calculate_tsallis_entropy(current_window, self._q_parameter)

        if np.isinf(current_entropy) or np.isnan(current_entropy):
            current_entropy = 0.0

        self._entropy_values.append(current_entropy)

        if len(self._entropy_values) >= v1:
            entropy_diff = abs(self._entropy_values[-1] - self._entropy_values[-2])
            if entropy_diff > self._threshold:
                self._last_change_point = self._position - self._window_size // 2

        self._detect_entropy_trend()
        self._detect_entropy_variance()

        if self._multi_q:
            self._detect_multi_q_changes()

    def _calculate_tsallis_entropy(self, time_series: npt.NDArray[np.float64], q: float) -> float:
        """
        Compute Tsallis entropy S_q for the given window.

        :param time_series: Current rolling window.
        :type time_series: numpy.ndarray
        :param q: Entropic index (q ≠ 1).
        :type q: float
        :return: Tsallis entropy value (possibly normalized). Returns ``0.0`` on degenerate inputs.
        :rtype: float

        . note::
           - If ``use_kde`` is ``True``, use continuous definition via KDE + integration.
           - Otherwise, compute the discrete version via histogram probabilities.
        """
        if self._use_kde:
            return self._calculate_continuous_tsallis_entropy(time_series, q)
        else:
            return self._calculate_discrete_tsallis_entropy(time_series, q)

    def _calculate_discrete_tsallis_entropy(self, time_series: npt.NDArray[np.float64], q: float) -> float:
        """
        Discrete Tsallis entropy using histogram probabilities.

        :param time_series: Current rolling window.
        :type time_series: numpy.ndarray
        :param q: Entropic index.
        :type q: float
        :return: S_q for the discrete distribution in the window (optionally normalized).
        :rtype: float
        """
        hist, _ = np.histogram(time_series, bins=self._num_bins, density=False)
        probabilities = hist / np.sum(hist)
        probabilities = probabilities[probabilities > 0]

        if len(probabilities) == 0:
            return 0.0

        sum_p_q = np.sum(probabilities**q)

        v = 1e-10
        if abs(q - 1.0) < v:
            tsallis_entropy = -self._k_constant * np.sum(probabilities * np.log(probabilities))
        else:
            tsallis_entropy = self._k_constant * (1.0 / (q - 1.0)) * (1.0 - sum_p_q)

        if self._normalize:
            max_entropy = self._calculate_max_discrete_tsallis_entropy(len(probabilities), q)
            if max_entropy > 0:
                tsallis_entropy = tsallis_entropy / max_entropy

        return float(tsallis_entropy)

    def _calculate_continuous_tsallis_entropy(self, time_series: npt.NDArray[np.float64], q: float) -> float:
        """
        Continuous Tsallis entropy using Gaussian KDE and numerical integration.

        :param time_series: Current rolling window.
        :type time_series: numpy.ndarray
        :param q: Entropic index.
        :type q: float
        :return: Continuous S_q estimated from KDE. Falls back to discrete S_q on numerical errors.
        :rtype: float
        """
        try:
            kde = stats.gaussian_kde(time_series)
            data_min, data_max = np.min(time_series), np.max(time_series)
            margin = (data_max - data_min) * 0.2

            def integrand(x: Union[float, NDArray[np.float64]]) -> float:
                x_array: NDArray[np.float64] = np.atleast_1d(x).astype(np.float64)
                p_x = kde(x_array)[0]
                return float(p_x**q)

            integral_result, _ = quad(
                integrand, data_min - margin, data_max + margin, limit=100, epsabs=1e-8, epsrel=1e-8
            )
            v = 1e-10
            if abs(q - 1.0) < v:

                def shannon_integrand(x: Union[float, NDArray[np.float64]]) -> float:
                    x_array: NDArray[np.float64] = np.atleast_1d(x).astype(np.float64)
                    p_x = kde(x_array)[0]
                    return float(p_x * np.log(p_x + 1e-10))

                shannon_integral, _ = quad(
                    shannon_integrand, data_min - margin, data_max + margin, limit=100, epsabs=1e-8, epsrel=1e-8
                )
                tsallis_entropy = -self._k_constant * shannon_integral
            else:
                tsallis_entropy = (1.0 / (q - 1.0)) * (1.0 - integral_result)

            if np.isinf(tsallis_entropy) or np.isnan(tsallis_entropy):
                return 0.0

            return float(tsallis_entropy)

        except Exception:
            # Fallback to discrete estimate if KDE/integration fails
            return self._calculate_discrete_tsallis_entropy(time_series, q)

    def _calculate_max_discrete_tsallis_entropy(self, n_states: int, q: float) -> float:
        """
        Theoretical maximum of discrete S_q for a uniform distribution over ``n_states``.

        :param n_states: Number of non-zero-probability states in the window.
        :type n_states: int
        :param q: Entropic index.
        :type q: float
        :return: Maximal S_q for the given ``n_states`` (``k * ln(n_states)`` in the Shannon limit).
        :rtype: float
        """
        v = 1e-10
        if n_states <= 1:
            return 0.0

        p_uniform = 1.0 / n_states

        if abs(q - 1.0) < v:
            return float(self._k_constant * np.log(n_states))
        else:
            return float(self._k_constant * (1.0 / (q - 1.0)) * (1.0 - n_states * (p_uniform**q)))

    def _combine_multi_q_entropies(self, entropies: dict[float, float]) -> float:
        """
        Combine multiple S_q values into a single score (simple weighted average).

        :param entropies: Mapping ``{q: S_q(current window)}``.
        :type entropies: dict[float, float]
        :return: Aggregated entropy score across q-values.
        :rtype: float
        """
        weights = {}
        for q in entropies:
            weights[q] = 1.0

        total_weight = sum(weights.values())
        if total_weight == 0:
            return 0.0

        combined_entropy = sum(entropies[q] * weights[q] for q in entropies) / total_weight
        return combined_entropy

    def _detect_entropy_trend(self) -> None:
        """
        Trend test on recent S_q via linear regression slope.

        . note::
           Uses last 10 values; if ``|slope| > 0.5 * threshold``, flags a change and
           localizes near the tail.
        """
        v = 10
        if len(self._entropy_values) >= v:
            recent_entropies = self._entropy_values[-10:]
            x = np.arange(len(recent_entropies))
            slope, _, _, _, _ = stats.linregress(x, recent_entropies)

            if abs(slope) > self._threshold * 0.5:
                self._last_change_point = self._position - 5

    def _detect_entropy_variance(self) -> None:
        """
        Variance-shift test between two recent halves of the S_q history.

        . note::
           Compares ``var(last 10)`` vs ``var(previous 10)``; if ``|Δvar| > 0.3 * threshold``,
           flags a change near the midpoint.
        """
        v = 20
        if len(self._entropy_values) >= v:
            first_half = self._entropy_values[-20:-10]
            second_half = self._entropy_values[-10:]

            var1 = np.var(first_half)
            var2 = np.var(second_half)

            if abs(var2 - var1) > self._threshold * 0.3:
                self._last_change_point = self._position - 10

    def _detect_multi_q_changes(self) -> None:
        """
        Majority-vote rule across q-values based on last-step S_q differences.

        . note::
           For each ``q``, if ``|S_q(t) - S_q(t-1)| > 0.7 * threshold``, that ``q`` votes for a change.
           If votes ≥ half + 1, flag a change near the tail.
        """
        if not self._multi_q:
            return

        change_votes = 0
        v = 2
        v1 = 5
        for q in self._q_values:
            if len(self._multi_entropy_values[q]) >= v1:
                recent_values = self._multi_entropy_values[q][-5:]
                if len(recent_values) >= v:
                    recent_diff = abs(recent_values[-1] - recent_values[-2])
                    if recent_diff > self._threshold * 0.7:
                        change_votes += 1

        if change_votes >= len(self._q_values) // 2 + 1:
            self._last_change_point = self._position - 2

    def get_entropy_history(self) -> list[float]:
        """
        Get the history of aggregated Tsallis entropy values (single- or multi-q).

        :return: A copy of the internal S_q sequence evaluated at processed steps.
        :rtype: list[float]
        """
        return self._entropy_values.copy()

    def get_multi_q_history(self) -> dict[float, list[float]]:
        """
        Get the history of Tsallis entropy values for each ``q`` (only in multi-q mode).

        :return: Mapping ``{q: list of S_q values over time}``.
        :rtype: dict[float, list[float]]
        """
        return {q: values.copy() for q, values in self._multi_entropy_values.items()}

    def get_current_parameters(self) -> dict[str, Any]:
        """
        Return current configuration and important internal parameters.

        :return: Dictionary with keys: ``window_size``, ``q_parameter``, ``k_constant``, ``threshold``,
                 ``num_bins``, ``use_kde``, ``normalize``, ``multi_q``, ``q_values``.
        :rtype: dict
        """
        return {
            "window_size": self._window_size,
            "q_parameter": self._q_parameter,
            "k_constant": self._k_constant,
            "threshold": self._threshold,
            "num_bins": self._num_bins,
            "use_kde": self._use_kde,
            "normalize": self._normalize,
            "multi_q": self._multi_q,
            "q_values": self._q_values.copy(),
        }

    def set_parameters(
        self,
        q_parameter: Optional[float] = None,
        k_constant: Optional[float] = None,
        threshold: Optional[float] = None,
        num_bins: Optional[int] = None,
        use_kde: Optional[bool] = None,
        normalize: Optional[bool] = None,
        multi_q: Optional[bool] = None,
    ) -> None:
        """
        Update detector hyperparameters in place.

        :param q_parameter: New entropic index q (q ≠ 1). Ignored in multi-q mode.
        :type q_parameter: float, optional
        :param k_constant: New k constant (> 0).
        :type k_constant: float, optional
        :param threshold: New decision threshold.
        :type threshold: float, optional
        :param num_bins: New number of histogram bins (> 1).
        :type num_bins: int, optional
        :param use_kde: Toggle continuous KDE-based S_q.
        :type use_kde: bool, optional
        :param normalize: Toggle normalization (discrete mode).
        :type normalize: bool, optional
        :param multi_q: Toggle multi-q mode; resets per-q histories if turned on.
        :type multi_q: bool, optional
        :return: ``None``.
        :rtype: None

        :raises ValueError: If invalid values are provided (e.g., ``q=1`` or non-positive
                            constants / bins).
        """

        def set_q_param(q: float) -> None:
            v = 1e-10
            if abs(q - 1.0) < v:
                raise ValueError("q parameter cannot be 1")
            self._q_parameter = q
            if not self._multi_q:
                self._q_values = [q]

        def set_k_const(k: float) -> None:
            if k <= 0:
                raise ValueError("k constant must be positive")
            self._k_constant = k

        def set_num_bins_func(bins: int) -> None:
            if bins <= 1:
                raise ValueError("Number of bins must be greater than 1")
            self._num_bins = bins

        if q_parameter is not None:
            set_q_param(q_parameter)
        if k_constant is not None:
            set_k_const(k_constant)
        if threshold is not None:
            self._threshold = threshold
        if num_bins is not None:
            set_num_bins_func(num_bins)
        if use_kde is not None:
            self._use_kde = use_kde
        if normalize is not None:
            self._normalize = normalize
        if multi_q is not None:
            self._multi_q = multi_q
            if multi_q:
                self._q_values = [0.5, 1.5, 2.0, 2.5, 3.0]
                self._multi_entropy_values = {q: [] for q in self._q_values}
            else:
                self._q_values = [self._q_parameter]

    def analyze_q_sensitivity(self) -> dict[str, Any]:
        """
        Evaluate S_q sensitivity to a set of q-values on the current window.

        :return: Dictionary with keys:

                 - ``q_entropies``: ``{q: S_q(current window)}``
                 - ``entropy_ratios``: ratios vs base ``q=2.0`` (if defined)
                 - ``most_sensitive_q``: argmax ``|S_q|``
                 - ``entropy_variance_across_q``: variance of ``S_q`` over tested ``q``

        :rtype: dict
        """
        if len(self._buffer) < self._window_size:
            return {}

        current_window = np.array(list(self._buffer)[-self._window_size :])
        test_q_values = [0.1, 0.5, 1.5, 2.0, 2.5, 3.0, 5.0]
        q_entropies: dict[float, float] = {}

        for q in test_q_values:
            try:
                entropy = self._calculate_tsallis_entropy(current_window, q)
                q_entropies[q] = entropy
            except Exception:
                q_entropies[q] = 0.0

        entropy_ratios = {}
        base_q = 2.0
        if base_q in q_entropies:
            base_val = q_entropies[base_q]
            for q, entropy in q_entropies.items():
                if q != base_q and base_val != 0:
                    entropy_ratios[q] = entropy / base_val

        return {
            "q_entropies": q_entropies,
            "entropy_ratios": entropy_ratios,
            "most_sensitive_q": max(q_entropies.keys(), key=lambda x: abs(q_entropies[x])) if q_entropies else None,
            "entropy_variance_across_q": np.var(list(q_entropies.values())) if q_entropies else 0.0,
        }

    def get_complexity_metrics(self) -> dict[str, Any]:
        """
        Compute auxiliary complexity metrics from S_q at several regimes.

        :return: Includes sub-/extensive/super-extensive S_q, their ratio, and basic window stats.
        :rtype: dict
        """
        if len(self._buffer) < self._window_size:
            return {}

        current_window = np.array(list(self._buffer)[-self._window_size :])

        sub_extensive = self._calculate_tsallis_entropy(current_window, 0.5)
        extensive = self._calculate_discrete_tsallis_entropy(current_window, 1.01)
        super_extensive = self._calculate_tsallis_entropy(current_window, 2.5)

        return {
            "sub_extensive_entropy": sub_extensive,
            "extensive_entropy": extensive,
            "super_extensive_entropy": super_extensive,
            "entropy_ratio_sub_super": sub_extensive / super_extensive if super_extensive != 0 else float("inf"),
            "window_std": np.std(current_window),
            "window_mean": np.mean(current_window),
            "data_range": np.max(current_window) - np.min(current_window),
        }

    def reset(self) -> None:
        """
        Clear internal state and buffered statistics.

        :return: ``None``.
        :rtype: None
        """
        self._buffer.clear()
        self._entropy_values.clear()
        self._multi_entropy_values = {q: [] for q in self._q_values}
        self._position = 0
        self._last_change_point = None
