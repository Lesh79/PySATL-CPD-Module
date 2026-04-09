# -*- coding: ascii -*-
"""
Module contains online detection trace container for streaming changepoint detection.

This module provides containers for storing step-by-step results and aggregated
traces from online changepoint detection algorithms.
"""

__author__ = "Vladimir Kutuev, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from pysatl_cpd.core.detection_trace import DetectionTrace
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmState
from pysatl_cpd.core.typedefs import Number, UnivariateNumericArray


def extract_periods(in_periods: Sequence[bool | None]) -> list[tuple[int, int]]:
    """
    Extract continuous periods where condition is True.

    Parameters
    ----------
    in_periods : Sequence[bool | None]
        Sequence of boolean values indicating whether each position
        is in a period. None values are ignored.

    Returns
    -------
    list[tuple[int, int]]
        List of (start, end) indices for each continuous period.
    """
    periods: list[tuple[int, int]] = []
    i = 0
    n = len(in_periods)

    while i < n:
        # Skip None values and False
        if in_periods[i] is not True:
            i += 1
            continue

        # Start of a period
        start = i
        while i < n and in_periods[i] is True:
            i += 1
        # End of period (i-1 is last index)
        periods.append((start, i - 1))

    return periods


@dataclass(kw_only=True)
class OnlineDetectionStepResult[StateT: OnlineAlgorithmState]:
    """
    Result of processing a single observation in online changepoint detection.

    This class captures the complete output for one step of an online detection
    algorithm, including detection flags, computed statistics, and timing
    information.

    Parameters
    ----------
    step_num : int, default=0
        Zero-based index of the processed observation.
    is_forced_change_point : bool, default=False
        Whether a changepoint was forced due to maximum runlength constraint.
    is_signal_change_point : bool, default=False
        Whether a changepoint was detected by the algorithm's detection
        function exceeding the threshold.
    is_in_skip_period : bool, default=False
        Whether this step occurred during a post-detection skip period.
    detection_function : Number, default=nan
        The value of the detection statistic computed for this observation.
    processing_time : Number, default=nan
        Wall-clock time in seconds spent processing this step.
    algorithm_state : StateT or None, default=None
        Snapshot of algorithm internal state after processing this step.
    """

    step_num: int = 0
    is_forced_change_point: bool = False
    is_signal_change_point: bool = False
    is_in_skip_period: bool = False
    detection_function: Number = float("nan")
    processing_time: Number = float("nan")
    algorithm_state: StateT | None = None

    @property
    def is_change_point(self) -> bool:
        """
        Whether a changepoint was detected at this step.

        A changepoint is considered detected if it was either forced
        (due to maximum runlength) or signaled (detection function
        exceeded threshold).

        Returns
        -------
        bool
            True if a forced or signal changepoint occurred, False otherwise.
        """
        return self.is_forced_change_point or self.is_signal_change_point


@dataclass(kw_only=True)
class OnlineDetectionTrace[StateT: OnlineAlgorithmState](DetectionTrace):
    """
    Complete trace of online changepoint detection execution.

    This class aggregates the results of running an online detection algorithm
    over a complete data sequence. It extends DetectionTrace with additional
    metadata specific to online detection, including per-step statistics,
    processing times, and forced detection markers.

    Parameters
    ----------
    algorithm_name : str
        Human-readable name of the algorithm that produced this trace.
    configuration_hash : int
        Hash of the algorithm configuration used during the run.
        Can be used to identify and group runs with identical configurations.
    detected_change_points : list[int]
        Indices where changepoints were detected.
    threshold : Number | None, optional
        Detection threshold used during the run. Default is None.
    processing_time : UnivariateNumericArray
        Processing time for each observation step as a 1-D NumPy array.
    detection_function : UnivariateNumericArray
        Detection function values for each observation as a 1-D NumPy array.
    algorithm_states : list[StateT | None]
        Algorithm state snapshots after processing each observation.
    skip_periods : list[tuple[int, int]], optional
        Indices of beginning and ending of segments where observations were
        skipped during post-detection periods. Default is empty list.
    learning_periods : list[tuple[int, int]], optional
        Indices of beginning and ending of segments where algorithm was
        learning data distribution before change point. Default is empty list.
    forced_change_points : list[int], optional
        Indices where changepoints were forced due to maximum runlength
        constraint. Default is empty list.
    signal_change_points : list[int], optional
        Indices where changepoints were detected due to the algorithm's
        detection function exceeding the threshold. Default is empty list.
    TODO
    """

    algorithm_name: str
    configuration_hash: int
    threshold: Number | None = None
    processing_time: UnivariateNumericArray
    detection_function: UnivariateNumericArray
    forced_change_points: list[int] = field(default_factory=list)
    signal_change_points: list[int] = field(default_factory=list)
    skip_periods: list[tuple[int, int]] = field(default_factory=list)
    learning_periods: list[tuple[int, int]] = field(default_factory=list)
    algorithm_states: list[StateT | None]

    @classmethod
    def from_run(
        cls,
        steps: Sequence[OnlineDetectionStepResult[StateT]],
        algorithm_name: str,
        configuration_hash: int,
        threshold: Number | None = None,
    ) -> "OnlineDetectionTrace[StateT]":
        """
        Construct an OnlineDetectionTrace from a data provider and step results.

        This factory method aggregates per-step results into a complete trace,
        extracting detection indices, processing times, and state snapshots.

        Parameters
        ----------
        steps : Sequence[OnlineDetectionStepResult]
            Sequence of step results from processing each observation.
        algorithm_name : str
            Human-readable name of the algorithm that produced the steps.
        configuration_hash : int
            Hash of the algorithm configuration used during the run.
        threshold : Number | None, optional
            The detection threshold used during execution.

        Returns
        -------
        OnlineDetectionTrace[StateT]
            Aggregated trace containing all detection results and metadata.

        Examples
        --------
        >>> from pysatl_cpd.core.data_providers import NDArrayUnivariateProvider
        >>> import numpy as np
        >>> steps = [
        ...     OnlineDetectionStepResult(step_num=0, detection_function=0.1),
        ...     OnlineDetectionStepResult(step_num=1, detection_function=0.8, is_change_point=True)
        ... ]
        >>> trace = OnlineDetectionTrace.from_run(steps=steps, threshold=0.5)
        >>> trace.detected_changes
        [1]
        """
        # Extract detection function values into a float64 array
        detection_function: UnivariateNumericArray = cast(
            UnivariateNumericArray,
            np.array([step.detection_function for step in steps], dtype=np.float64),
        )

        # Extract processing times into a float64 array
        processing_times: UnivariateNumericArray = cast(
            UnivariateNumericArray,
            np.array([step.processing_time for step in steps], dtype=np.float64),
        )

        # Extract algorithm states preserving None values
        states: list[StateT | None] = [step.algorithm_state for step in steps]
        skip_periods = extract_periods([s.is_in_skip_period for s in steps])
        learning_periods = extract_periods([s.is_in_learning_period if s is not None else None for s in states])

        # Identify indices of different detection types
        detected_indices: list[int] = [step.step_num for step in steps if step.is_change_point]
        forced_indices: list[int] = [step.step_num for step in steps if step.is_forced_change_point]
        signal_indices: list[int] = [step.step_num for step in steps if step.is_signal_change_point]

        return cls(
            algorithm_name=algorithm_name,
            configuration_hash=configuration_hash,
            detected_change_points=detected_indices,
            forced_change_points=forced_indices,
            signal_change_points=signal_indices,
            threshold=threshold,
            detection_function=detection_function,
            processing_time=processing_times,
            algorithm_states=states,
            skip_periods=skip_periods,
            learning_periods=learning_periods,
        )
