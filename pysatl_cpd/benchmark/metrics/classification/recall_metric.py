# -*- coding: ascii -*-

"""
Module for computing micro-averaged Recall over a dataset.

Recall measures the proportion of actual change points that were successfully
detected, calculated globally across all benchmark runs.
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


class RecallMetric[TraceT: DetectionTrace, ProviderT: LabeledData[Any]](
    AggregationMetric[TraceT, ProviderT, dict[str, float], float]
):
    """
    Computes the micro-averaged Recall metric over the entire dataset.

    Recall is calculated globally by summing True Positives (TP) and
    False Negatives (FN) across all runs before applying the formula.

    Parameters
    ----------
    error_margin : tuple[int, int]
        Tolerance window `(left, right)` around true change points for matching.
    """

    def __init__(self, error_margin: tuple[int, int]) -> None:
        self._base_metric = ConfusionMatrix[TraceT, ProviderT](error_margin)

    @property
    def base_metric(self) -> ConfusionMatrix[TraceT, ProviderT]:
        return self._base_metric

    def aggregate(self, values: Sequence[dict[str, float]]) -> float:
        total_tp = sum(v["tp"] for v in values)
        total_fn = sum(v["fn"] for v in values)
        return total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
