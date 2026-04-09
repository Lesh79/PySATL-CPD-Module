# -*- coding: ascii -*-

"""
Module for computing the False Negative (FN) metric.

A False Negative occurs when an actual change point is missed by the
detection algorithm within the specified tolerance window.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.classification.classification_metric import ClassificationMetric
from pysatl_cpd.core.detection_trace import DetectionTrace


class FalseNegativeMetric[TraceT: DetectionTrace, ProviderT: LabeledData[Any]](ClassificationMetric[TraceT, ProviderT]):
    """
    Metric for counting False Negatives (FN), which are true change points
    that were missed by the detection algorithm.
    """

    @classmethod
    def compute(
        cls, detected_changes: Sequence[int], true_changes: Sequence[int], error_margin: tuple[int, int]
    ) -> int:
        """
        Compute the number of False Negatives.

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
            The number of missed true change points.
        """

        return len(true_changes) - len(cls.match(detected_changes, true_changes, error_margin))

    def evaluate(self, trace: TraceT, data: ProviderT) -> float:
        """
        Evaluate the False Negative metric.

        Parameters
        ----------
        trace : TraceT
            The trace containing detected change points.
        data : ProviderT
            The ground truth data.

        Returns
        -------
        float
            The number of False Negatives.
        """

        return float(
            self.compute(
                trace.detected_change_points,
                data.change_points,
                self._error_margin,
            )
        )
