# -*- coding: ascii -*-

"""
NoReset detection trace container.

This module provides NoResetDetectionTrace - a lightweight trace produced
by applying a ThresholdPolicy to a pre-computed infinite-threshold trace,
avoiding redundant solver executions.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import cast

import numpy as np

from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmState
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace
from pysatl_cpd.core.typedefs import UnivariateNumericArray


class NoResetDetectionTrace[StateT: OnlineAlgorithmState](OnlineDetectionTrace[StateT]):
    """
    Detection trace produced by the NoReset benchmark strategy.

    Instead of re-running the solver for every threshold, a single
    infinite-threshold trace is computed once and this class wraps it
    with new detected change points obtained by applying a ThresholdPolicy.

    Auxiliary fields (processing_time, algorithm_states, skip_periods,
    learning_periods, forced_change_points, signal_change_points) are
    intentionally left empty - only detection_function and
    detected_change_points carry meaningful data.
    """

    @classmethod
    def from_inf_trace(
        cls,
        source_trace: OnlineDetectionTrace[StateT],
        detected_change_points: list[int],
        threshold: float,
    ) -> "NoResetDetectionTrace[StateT]":
        """
        Construct a NoResetDetectionTrace from an infinite-threshold trace.

        Copies detection_function, algorithm_name, and configuration_hash
        from source_trace. All other fields are set to empty defaults.

        Parameters
        ----------
        source_trace : OnlineDetectionTrace[StateT]
            The pre-computed trace obtained by running the solver with
            threshold=inf. Its detection_function is reused for all
            threshold simulations.
        detected_change_points : list[int]
            Change point indices produced by applying a ThresholdPolicy
            to source_trace.detection_function at a specific threshold.
        threshold : float
            The threshold value used to extract detected_change_points.

        Returns
        -------
        NoResetDetectionTrace[StateT]
            A new trace with the given change points and copied
            detection function.
        """
        empty_processing_time: UnivariateNumericArray = cast(
            UnivariateNumericArray,
            np.array([], dtype=np.float64),
        )

        return cls(
            algorithm_name=source_trace.algorithm_name,
            configuration_hash=source_trace.configuration_hash,
            detected_change_points=detected_change_points,
            threshold=threshold,
            detection_function=source_trace.detection_function.copy(),
            processing_time=empty_processing_time,
            algorithm_states=[],
            skip_periods=[],
            learning_periods=[],
            forced_change_points=[],
            signal_change_points=[],
        )
