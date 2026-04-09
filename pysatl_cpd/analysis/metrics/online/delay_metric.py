# -*- coding: ascii -*-

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.run_metric import RunMetric
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class DelayMetric[T: OnlineDetectionTrace[Any], D: LabeledData[Any]](RunMetric[T, D, Sequence[int]]):
    def __init__(self, max_delay: int) -> None:
        self.__max_delay = max_delay

    def evaluate(self, trace: T, data: D) -> Sequence[int]:
        detected_changes = trace.detected_change_points
        true_changes = data.change_points

        delays = []
        used_detections = set()

        for true_change in true_changes:
            covered = False
            for detected_change in detected_changes:
                if detected_change in used_detections:
                    continue

                if true_change <= detected_change <= true_change + self.__max_delay:
                    delays.append(detected_change - true_change)
                    used_detections.add(detected_change)
                    covered = True
                    break

            if not covered:
                delays.append(self.__max_delay)

        return delays
