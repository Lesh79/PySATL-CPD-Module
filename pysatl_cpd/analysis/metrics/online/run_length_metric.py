# -*- coding: ascii -*-

"""
Module for computing Run Length to False Alarm for online algorithms.

Evaluates the distance (time steps) between algorithm resets and
false positive detections (false alarms).
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.run_metric import RunMetric
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class RunLengthMetric[T: OnlineDetectionTrace[Any], D: LabeledData[Any]](RunMetric[T, D, Sequence[int]]):
    """
    Computes the Run Lengths to False Alarms (ARL) for online detection traces.

    A run length is the distance between consecutive "resets" of the algorithm
    and a False Positive (false alarm). The timer is reset at the start (0),
    after successfully traversing a true change point window, or right after
    a previous false alarm.

    Parameters
    ----------
    max_delay : int
        The maximum allowable delay window that defines valid detections.
        Detections outside these windows are considered False Positives.
    """

    def __init__(self, max_delay: int) -> None:
        self.__max_delay = max_delay

    def evaluate(self, trace: T, data: D) -> Sequence[int]:
        """
        Calculate the run lengths to false alarms.

        Parameters
        ----------
        trace : T
            The online detection trace.
        data : D
            The ground truth data.

        Returns
        -------
        Sequence[int]
            A sequence of distances (run lengths) preceding each false alarm.
        """

        detected_changes = trace.detected_change_points
        true_changes = data.change_points

        false_positives = []
        for detected in detected_changes:
            is_tp = False
            for true_change in true_changes:
                if true_change <= detected <= true_change + self.__max_delay:
                    is_tp = True
                    break

            if not is_tp:
                false_positives.append(detected)

        run_lengths = []
        last_reset_point = 0

        true_idx = 0

        for fp in false_positives:
            while true_idx < len(true_changes) and true_changes[true_idx] + self.__max_delay < fp:
                last_reset_point = true_changes[true_idx] + self.__max_delay
                true_idx += 1

            run_lengths.append(fp - last_reset_point)
            last_reset_point = fp

        return run_lengths
