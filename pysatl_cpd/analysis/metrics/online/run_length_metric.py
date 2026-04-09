# -*- coding: ascii -*-

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.run_metric import RunMetric
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class RunLengthMetric[T: OnlineDetectionTrace[Any], D: LabeledData[Any]](RunMetric[T, D, Sequence[int]]):
    def __init__(self, max_delay: int) -> None:
        self.__max_delay = max_delay

    def evaluate(self, trace: T, data: D) -> Sequence[int]:
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
