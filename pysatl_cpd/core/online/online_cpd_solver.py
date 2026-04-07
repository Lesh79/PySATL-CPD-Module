# -*- coding: ascii -*-
"""
Online change-point detection solver.

This module implements the core execution loop that feeds observations from a
data provider into an online change-point detection algorithm and emits
per-step results.
"""

__author__ = "Alexey Tatyanenko, Vladimir Kutuev, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import time
from collections.abc import Iterator

from pysatl_cpd.core.data_providers import DataProvider
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithm, OnlineAlgorithmConfiguration, OnlineAlgorithmState
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionStepResult
from pysatl_cpd.core.typedefs import Number


class OnlineCpdSolver:
    """
    Sequential executor for online change-point detection.

    Iterates over a data provider, feeds each observation to the algorithm,
    compares the resulting statistic against a threshold, and optionally
    enforces a maximum run-length and a post-detection skip period.

    Parameters
    ----------
    algorithm : OnlineAlgorithm[DataT, ConfugrationT, StateT]
        The online change-point detection algorithm to apply.
    threshold : float, optional
        Detection threshold for the change-point function. A change point is
        declared when the statistic exceeds this value.
        If `nan`, the algorithm will not generate any signals. Default is ``nan``.
    skip_period : int, optional
        Number of steps to skip (suppress detections) after each declared
        change point. Must be non-negative. Default is ``0``.
    max_runlength : int or None, optional
        If not ``None``, forces a change-point declaration once the run
        length exceeds this value. Must be positive if specified.
        Default is ``None``.
    collect_states : bool, optional
        Whether to collect algorithm state snapshots in step results.
        If False, algorithm_state will be None in all step results.
        Default is True.

    Raises
    ------
    ValueError
        If ``skip_period`` is negative.
    ValueError
        If ``max_runlength`` is not None and not positive.
    """

    def __init__(
        self,
        skip_period: int = 0,
        max_runlength: int | None = None,
        collect_states: bool = True,
    ) -> None:
        """
        Initialize the online change-point detection solver.

        Parameters
        ----------
        skip_period : int, optional
            Number of steps to skip after each declared change point.
        max_runlength : int or None, optional
            Maximum run length before forcing a change point.
        collect_states : bool, optional
            Whether to collect algorithm state snapshots.
        """
        # Validate skip_period is non-negative
        if skip_period < 0:
            raise ValueError(f"skip_period must be non-negative, got {skip_period}")

        # Validate max_runlength is positive if specified
        if max_runlength is not None and max_runlength <= 0:
            raise ValueError(f"max_runlength must be positive if specified, got {max_runlength}")

        self.__skip_period = skip_period
        self.__max_runlength = max_runlength
        self.__collect_states = collect_states

        self.__in_skip_period = False

    def run[DataT, ConfugrationT: OnlineAlgorithmConfiguration, StateT: OnlineAlgorithmState](
        self,
        algorithm: OnlineAlgorithm[DataT, ConfugrationT, StateT],
        data_provider: DataProvider[DataT],
        threshold: float = float("nan"),
    ) -> Iterator[OnlineDetectionStepResult[StateT]]:
        """
        Execute the detection loop over all observations.

        Iterates through all observations provided by the data provider,
        processes each through the detection algorithm, and yields per-step
        results. During a skip period following a detected change point,
        observations are processed without change point declarations.

        Parameters
        ----------
        algorithm : OnlineAlgorithm[T]
            The online change-point detection algorithm to apply.
        data_provider : DataProvider[DataT]
            An iterable source of observations.
        threshold : float, optional
            Detection threshold for the change-point function.

        Yields
        ------
        OnlineDetectionStepResult
            Per-step detection result containing the change-point flag,
            statistic value, step index, processing time, and algorithm state.

        Notes
        -----
        The solver maintains three key state variables:
        - run_length: Number of observations since the last change point
        - skip_period_counter: Number of steps remaining in skip period
        - in_skip_period: Flag indicating active skip period
        """
        run_length: int = 0
        skip_period_counter: int = 0
        algorithm.reset()

        for step, observation in enumerate(data_provider):
            # Handle skip period where detections are suppressed
            if self.__in_skip_period:
                if skip_period_counter < self.__skip_period:
                    skip_period_counter += 1
                    yield OnlineDetectionStepResult(
                        step_num=step,
                        is_in_skip_period=True,
                        algorithm_state=None,
                    )
                    continue
                if skip_period_counter == self.__skip_period:
                    self.__in_skip_period = False
                    skip_period_counter = 0

            # Process observation normally
            step_start_time: float = time.perf_counter()
            detection_func: Number = algorithm.process(observation)
            step_finish_time: float = time.perf_counter()

            run_length += 1

            # Determine if change point occurred
            is_forced: bool = self._is_forced_change_point(run_length)
            is_signal: bool = self._is_signal_change_point(detection_func, threshold)
            is_change_point: bool = is_forced | is_signal

            # Get algorithm state if collecting
            algorithm_state = algorithm.state if self.__collect_states else None

            yield OnlineDetectionStepResult(
                step_num=step,
                is_in_skip_period=False,
                is_forced_change_point=is_forced,
                is_signal_change_point=is_signal,
                detection_function=detection_func,
                processing_time=step_finish_time - step_start_time,
                algorithm_state=algorithm_state,
            )

            # Handle change point detection
            if is_change_point:
                algorithm.reset()
                self.__in_skip_period = True
                run_length = 0

    def _is_signal_change_point(self, detection_func: Number, threshold: float) -> bool:
        """
        Determine if detection statistic exceeds threshold.

        Parameters
        ----------
        detection_func : Number
            Current detection statistic value.
        threshold : floatl
            Detection threshold for the change-point function.

        Returns
        -------
        bool
            True if statistic exceeds threshold, False otherwise.
        """
        return bool(detection_func > threshold)

    def _is_forced_change_point(self, run_length: int) -> bool:
        """
        Determine if forced change point is required due to run length.

        Parameters
        ----------
        run_length : int
            Number of observations since last change point.

        Returns
        -------
        bool
            True if run length exceeds max_runlength, False otherwise.
        """
        return bool(self.__max_runlength is not None and run_length > self.__max_runlength)
