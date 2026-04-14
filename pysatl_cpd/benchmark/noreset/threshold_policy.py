# pysatl_cpd/benchmark/noreset/threshold_policy.py

"""
Threshold policies for signal extraction in NoReset benchmark.

This module provides the ThresholdPolicy protocol and two concrete
implementations: PointBasedPolicy and EventBasedPolicy.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Protocol, cast, runtime_checkable

import numpy as np

from pysatl_cpd.core.typedefs import UnivariateNumericArray


@runtime_checkable
class ThresholdPolicy(Protocol):
    """
    Protocol for signal extraction from a detection function.

    Implementations define how to convert a raw detection function array
    into a list of signal indices given a threshold and known change points.
    """

    def apply(
        self,
        detection_function: UnivariateNumericArray,
        threshold: float,
        change_points: Sequence[int],
    ) -> list[int]:
        """
        Extract signal indices from the detection function.

        Parameters
        ----------
        detection_function : UnivariateNumericArray
            Array of detection statistic values, one per time step.
        threshold : float
            Detection threshold.
        change_points : Sequence[int]
            True change point indices (1-based). Used by some policies
            to define delay windows.

        Returns
        -------
        list[int]
            1-based indices where signals were detected.
        """
        ...


class PointBasedPolicy:
    """
    Signal extraction policy based on point-wise threshold comparison.

    Any position where the detection function satisfies the threshold
    condition is considered a signal. The change_points argument is
    accepted for interface compatibility but is ignored.

    Parameters
    ----------
    strict : bool, default=True
        If True, signal condition is detection_function > threshold.
        If False, signal condition is detection_function >= threshold.
    """

    def __init__(self, strict: bool = True) -> None:
        self.strict = strict

    @staticmethod
    def _exceeds(arr: np.ndarray, threshold: float, strict: bool) -> np.ndarray:
        """
        Check whether array values exceed threshold.

        Parameters
        ----------
        arr : np.ndarray
            Array of values to check.
        threshold : float
            Threshold value.
        strict : bool
            If True, uses strict inequality (>).
            If False, uses non-strict inequality (>=).

        Returns
        -------
        np.ndarray
            Boolean array.
        """
        return arr > threshold if strict else arr >= threshold

    def apply(
        self,
        detection_function: UnivariateNumericArray,
        threshold: float,
        change_points: Sequence[int],
    ) -> list[int]:
        """
        Return 1-based indices where detection function exceeds threshold.

        Parameters
        ----------
        detection_function : UnivariateNumericArray
            Array of detection statistic values.
        threshold : float
            Detection threshold.
        change_points : Sequence[int]
            Ignored. Present for interface compatibility.

        Returns
        -------
        list[int]
            Sorted list of 1-based signal indices.
        """
        if len(detection_function) == 0:
            return []

        res = (np.where(self._exceeds(detection_function, threshold, self.strict))[0] + 1).tolist()
        return cast(list[int], res)


class EventBasedPolicy:
    """
    Signal extraction policy based on rising-edge detection with delay windows.

    In normal (edge) mode, a signal is produced only when the detection
    function crosses the threshold from below (rising edge). Inside delay
    windows [true_cp, true_cp + max_delay] (1-based, inclusive), the policy
    switches to point-based mode to correctly capture detection delay.

    The previous value used for edge detection (prev) is tracked continuously,
    including values inside delay windows (variant A). This means that if the
    detection function is above threshold at the end of a window, the first
    element after the window will not produce an edge signal.

    For the first element, prev is treated as -inf (always below threshold).

    Parameters
    ----------
    max_delay : int
        Maximum allowable detection delay. Defines the right boundary of
        the delay window as true_cp + max_delay (inclusive). Must be >= 0.
    strict_edge : bool, default=True
        If True, rising edge condition requires detection_function > threshold.
        If False, condition is detection_function >= threshold.
        prev is always checked with strict inequality (prev < threshold).
    strict_point : bool, default=True
        If True, point-based condition in delay window is
        detection_function > threshold.
        If False, condition is detection_function >= threshold.

    Raises
    ------
    ValueError
        If max_delay is negative.
    """

    def __init__(
        self,
        max_delay: int,
        strict_edge: bool = True,
        strict_point: bool = True,
    ) -> None:
        if max_delay < 0:
            raise ValueError(f"max_delay must be non-negative, got {max_delay}")
        self.max_delay = max_delay
        self.strict_edge = strict_edge
        self.strict_point = strict_point

    @staticmethod
    def _exceeds(arr: np.ndarray, threshold: float, strict: bool) -> np.ndarray:
        """
        Check whether array values exceed threshold.

        Parameters
        ----------
        arr : np.ndarray
            Array of values to check.
        threshold : float
            Threshold value.
        strict : bool
            If True, uses strict inequality (>).
            If False, uses non-strict inequality (>=).

        Returns
        -------
        np.ndarray
            Boolean array.
        """
        return arr > threshold if strict else arr >= threshold

    def _build_window_mask(
        self,
        length: int,
        change_points: Sequence[int],
    ) -> np.ndarray:
        """
        Build a boolean mask indicating which 0-based indices are in delay windows.

        Uses cumsum trick for fully vectorized computation over change points.

        Parameters
        ----------
        length : int
            Length of the detection function array.
        change_points : Sequence[int]
            True change point indices (1-based).

        Returns
        -------
        np.ndarray
            Boolean array of shape (length,) where True means the position
            is inside a delay window.
        """
        if not change_points:
            return np.zeros(length, dtype=bool)

        lefts = np.clip(np.array(change_points, dtype=int) - 1, 0, length - 1)
        rights = np.clip(lefts + self.max_delay, 0, length - 1)

        marker = np.zeros(length + 1, dtype=int)
        np.add.at(marker, lefts, 1)
        np.add.at(marker, rights + 1, -1)
        return np.cumsum(marker)[:length] > 0

    def apply(
        self,
        detection_function: UnivariateNumericArray,
        threshold: float,
        change_points: Sequence[int],
    ) -> list[int]:
        """
        Extract signal indices using rising-edge detection with delay windows.

        Fully vectorized implementation using numpy masks.

        Parameters
        ----------
        detection_function : UnivariateNumericArray
            Array of detection statistic values.
        threshold : float
            Detection threshold.
        change_points : Sequence[int]
            True change point indices (1-based). Used to define delay windows
            where point-based mode is applied.

        Returns
        -------
        list[int]
            Sorted list of 1-based signal indices.
        """
        n = len(detection_function)
        if n == 0:
            return []

        window_mask = self._build_window_mask(n, change_points)

        # prev[i] = df[i-1], prev[0] = -inf
        prev = np.empty(n, dtype=detection_function.dtype)
        prev[0] = float("-inf")
        prev[1:] = detection_function[:-1]

        # edge signals: rising edge outside windows
        edge = (prev < threshold) & self._exceeds(detection_function, threshold, self.strict_edge) & ~window_mask

        # point signals: threshold exceeded inside windows
        point = self._exceeds(detection_function, threshold, self.strict_point) & window_mask

        res = (np.where(edge | point)[0] + 1).tolist()
        return cast(list[int], res)
