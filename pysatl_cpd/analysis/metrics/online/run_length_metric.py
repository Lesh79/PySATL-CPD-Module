# -*- coding: ascii -*-

"""
Module for computing Run Lengths between consecutive alarms (detections)
for online algorithms.

In this implementation run length is simply the distance (in time steps)
between consecutive detected change points ('positives'). The first run length
is measured from time step 0 to the first detection.

Note
----
Ground truth (`data`) is not used here. Metrics that require TP/FP
separation should be implemented separately (e.g., via classification metrics).
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any, cast

import numpy as np

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.single_run_metric import SingleRunMetric
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class RunLengthMetric[TraceT: OnlineDetectionTrace[Any], ProviderT: LabeledData[Any]](
    SingleRunMetric[TraceT, ProviderT, list[int]]
):
    """
    Computes run lengths between consecutive detections.

    Run length is the distance between two successive detected change points,
    starting from time 0. This is the definition used for Average Run Length
    (ARL) - every detection is treated as a positive, ground-truth is ignored.
    """

    def evaluate(self, trace: TraceT, data: ProviderT) -> list[int]:
        """
        Calculate run lengths between consecutive detections.

        Parameters
        ----------
        trace : TraceT
            The online detection trace.
        data : ProviderT
            Unused. Present for API consistency.

        Returns
        -------
        list[int]
            Distances between consecutive detections, with the first distance measured
            from 0 to the first detection.
        """

        detected_changes = trace.detected_change_points
        if not detected_changes:
            return []

        sorted_changes = np.sort(detected_changes)
        points_with_zero = np.insert(sorted_changes, 0, 0)
        diff_arr = np.diff(points_with_zero)
        return cast(list[int], diff_arr.tolist())
