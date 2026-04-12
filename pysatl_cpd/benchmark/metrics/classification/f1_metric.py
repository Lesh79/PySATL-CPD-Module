# -*- coding: ascii -*-

"""
Module for computing the F1 Score over a dataset.

The F1 Score is the harmonic mean of precision and recall, calculated
by globally aggregating True Positives, False Positives, and False
Negatives across all benchmark runs.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.classification.confusion_matrix import ConfusionMatrix
from pysatl_cpd.benchmark.metrics.aggregation_metric import AggregationMetric
from pysatl_cpd.benchmark.metrics.classification.classification_report import ClassificationReport
from pysatl_cpd.core.detection_trace import DetectionTrace


class F1Metric[TraceT: DetectionTrace, ProviderT: LabeledData[Any]](
    AggregationMetric[TraceT, ProviderT, dict[str, float], float]
):
    """
    Computes the F1 Score over the entire dataset.

    F1 Score is the harmonic mean of precision and recall. Micro-averaging
    is performed by globally aggregating TP, FP, and FN across all runs.

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
        self._error_margin = error_margin

    @property
    def base_metric(self) -> ConfusionMatrix[TraceT, ProviderT]:
        return self._base_metric

    def aggregate(self, values: Sequence[dict[str, float]]) -> float:
        return ClassificationReport(self._error_margin).aggregate(values)["f1"]
