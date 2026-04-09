# -*- coding: ascii -*-

"""
Module for computing the False Positive (FP) metric.

A False Positive (false alarm) occurs when a change point is detected
but does not correspond to any true change point within the error margin.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.classification.classification_metric import ClassificationMetric
from pysatl_cpd.core.detection_trace import DetectionTrace


class FalsePositiveMetric[T: DetectionTrace, D: LabeledData[Any]](ClassificationMetric[T, D]):
    """
    Metric for counting False Positives (FP), which are detected change points
    that do not correspond to any actual change point within the error margin.
    """

    @classmethod
    def compute(
        cls, detected_changes: Sequence[int], true_changes: Sequence[int], error_margin: tuple[int, int]
    ) -> int:
        """
        Compute the number of False Positives.

        Parameters
        ----------
        detected_changes : Sequence[int]
            The sequence of predicted change point indices.
        true_changes : Sequence[int]
            The sequence of actual change point indices.
        error_margin : tuple[int, int]
            Tolerance window `(left, right)` for matching.

        Returns
        -------
        int
            The number of false alarm detections.
        """

        return len(detected_changes) - len(cls.match(detected_changes, true_changes, error_margin))

    def evaluate(self, trace: T, data: D) -> float:
        """
        Evaluate the False Positive metric.

        Parameters
        ----------
        trace : T
            The trace containing detected change points.
        data : D
            The ground truth data.

        Returns
        -------
        float
            The number of False Positives.
        """

        return float(
            self.compute(
                trace.detected_change_points,
                data.change_points,
                self._error_margin,
            )
        )
