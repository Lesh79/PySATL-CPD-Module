# -*- coding: ascii -*-

"""
Module for computing a comprehensive classification report over a dataset.

Calculates micro-averaged Precision, Recall, and F1 Score simultaneously
to optimize computational overhead across multiple benchmark runs.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.classification.confusion_matrix import ConfusionMatrix
from pysatl_cpd.benchmark.metrics.aggregation_metric import AggregationMetric
from pysatl_cpd.core.detection_trace import DetectionTrace


class ClassificationReport[TraceT: DetectionTrace, ProviderT: LabeledData[Any]](
    AggregationMetric[TraceT, ProviderT, dict[str, float], dict[str, float]]
):
    """
    Computes a comprehensive classification report over the entire dataset.

    Calculates micro-averaged Precision, Recall, and F1 Score by summing
    True Positives (TP), False Positives (FP), and False Negatives (FN)
    across all runs before applying the metric formulas.

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
        self._base_metric = ConfusionMatrix[TraceT, ProviderT](error_margin)

    @property
    def base_metric(self) -> ConfusionMatrix[TraceT, ProviderT]:
        return self._base_metric

    def aggregate(self, values: Sequence[dict[str, float]]) -> dict[str, float]:
        total_tp = sum(v["tp"] for v in values)
        total_fp = sum(v["fp"] for v in values)
        total_fn = sum(v["fn"] for v in values)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"tp": total_tp, "fp": total_fp, "fn": total_fn, "precision": precision, "recall": recall, "f1": f1}
