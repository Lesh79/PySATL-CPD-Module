# -*- coding: ascii -*-

"""
Module for computing the True Positive (TP) metric.

A True Positive occurs when a detected change point successfully matches
a true change point within the specified error margin.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.classification.classification_metric import ClassificationMetric
from pysatl_cpd.core.detection_trace import DetectionTrace


class TruePositiveMetric[T: DetectionTrace, D: LabeledData[Any]](ClassificationMetric[T, D]):
    """
    Metric for counting True Positives (TP), which are correctly identified
    actual change points within the defined error margin.
    """

    @classmethod
    def compute(
        cls, detected_changes: Sequence[int], true_changes: Sequence[int], error_margin: tuple[int, int]
    ) -> int:
        """
        Compute the number of True Positives.

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
            The number of correctly detected change points.
        """

        return len(cls.match(detected_changes, true_changes, error_margin))

    def evaluate(self, trace: T, data: D) -> float:
        """
        Evaluate the True Positive metric.

        Parameters
        ----------
        trace : T
            The trace containing detected change points.
        data : D
            The ground truth data.

        Returns
        -------
        float
            The number of True Positives.
        """

        return float(
            self.compute(
                trace.detected_change_points,
                data.change_points,
                self._error_margin,
            )
        )
