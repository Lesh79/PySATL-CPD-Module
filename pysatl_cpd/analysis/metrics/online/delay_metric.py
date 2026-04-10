# -*- coding: ascii -*-

"""
Module for computing the detection delay in online change point detection.

Evaluates how quickly an online algorithm reacts to a true change point.
Missed change points are penalized with the maximum allowable delay.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.classification.classification_metric import ClassificationMetric
from pysatl_cpd.analysis.metrics.run_metric import RunMetric
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class DelayMetric[TraceT: OnlineDetectionTrace[Any], ProviderT: LabeledData[Any]](
    RunMetric[TraceT, ProviderT, Sequence[int]]
):
    """
    Computes the detection delay for online change point detection algorithms.

    Delay is defined as the distance between an actual change point and its
    corresponding detection. If a change point is missed (False Negative),
    the algorithm is penalized with the maximum defined delay.

    Parameters
    ----------
    max_delay : int
        The maximum allowable delay window for a valid detection. Also used
        as the penalty value for missed detections (FN).
    """

    def __init__(self, max_delay: int) -> None:
        self._max_delay = max_delay

    def evaluate(self, trace: TraceT, data: ProviderT) -> Sequence[int]:
        """
        Calculate delays for all true change points

        Delay is computed per true change point as the minimum non-negative delay
        among matched detections within [true_change, true_change + max_delay].
        If there is no match, `max_delay` is returned for that change point.

        Parameters
        ----------
        trace : TraceT
            The online detection trace.
        data : ProviderT
            The ground truth data.

        Returns
        -------
        Sequence[int]
            A sequence of delays where each element corresponds to the true
            change point at the same index in ``data.change_points``.
            Length is exactly equal to the number of true change points.
        """

        detected_changes = trace.detected_change_points
        true_changes = data.change_points

        matching = ClassificationMetric.match(detected_changes, true_changes, (0, self._max_delay))

        delays = []
        for cp in true_changes:
            matched_detections = matching[cp]
            delays.append(min(matched_detections) - cp if matched_detections else self._max_delay)

        return delays
