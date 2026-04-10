# -*- coding: ascii -*-

"""
Module for computing the confusion matrix of change point detections.

Calculates True Positives (TP), False Positives (FP), and False Negatives (FN)
simultaneously to optimize computational overhead.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.classification.classification_metric import ClassificationMetric
from pysatl_cpd.analysis.metrics.run_metric import RunMetric
from pysatl_cpd.core.detection_trace import DetectionTrace


class ConfusionMatrix[TraceT: DetectionTrace, ProviderT: LabeledData[Any]](
    RunMetric[TraceT, ProviderT, dict[str, float]]
):
    """
    Computes True Positives (TP), False Positives (FP), and False Negatives (FN)
    in a single pass to avoid computational overhead.

    Parameters
    ----------
    error_margin : tuple[int, int]
        Tolerance window `(left, right)` around true change points for matching.

    Raises
    ------
    ValueError
        If the left or right margin in the `error_margin` argument is a negative number.
    """

    def __init__(self, error_margin: tuple[int, int]) -> None:
        if error_margin[0] < 0 or error_margin[1] < 0:
            raise ValueError("The left and right margins must be non-negative numbers")

        self._error_margin = error_margin

    def evaluate(self, trace: TraceT, data: ProviderT) -> dict[str, float]:
        """
        Calculate TP, FP, and FN based on detection matches.

        Parameters
        ----------
        trace : TraceT
            The trace containing detected change points.
        data : ProviderT
            The ground truth data containing actual change points.

        Returns
        -------
        dict[str, float]
            tp: number of true change points covered by >=1 detection
            fn: number of true change points with no detections
            fp: number of detections not matched to any true change point
        """

        detected_changes = trace.detected_change_points
        true_changes = data.change_points

        matching = ClassificationMetric.match(detected_changes, true_changes, self._error_margin)
        tp = len([v for v in matching.values() if v])
        fn = len(true_changes) - tp
        fp = len(detected_changes) - sum(len(v) for v in matching.values())

        return {"tp": float(tp), "fp": float(fp), "fn": float(fn)}
