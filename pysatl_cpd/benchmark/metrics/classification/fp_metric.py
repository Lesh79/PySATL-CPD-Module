# -*- coding: ascii -*-

"""
Module for computing aggregated False Positives (FP) over a dataset.

A False Positive occurs when a change point is falsely detected. This metric
sums the total number of false alarms across all benchmark runs.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.classification.fp_metric import FalsePositiveMetric as SingleFP
from pysatl_cpd.benchmark.metrics.aggregation_metric import AggregationMetric
from pysatl_cpd.core.detection_trace import DetectionTrace


class FalsePositiveMetric[TraceT: DetectionTrace, ProviderT: LabeledData[Any]](
    AggregationMetric[TraceT, ProviderT, float, float]
):
    """
    Metric for summing False Positives (FP) across an entire dataset.

    Parameters
    ----------
    error_margin : tuple[int, int]
        Tolerance window `(left, right)` around true change points for matching.
    """

    def __init__(self, error_margin: tuple[int, int]) -> None:
        self._base_metric = SingleFP[TraceT, ProviderT](error_margin)

    @property
    def base_metric(self) -> SingleFP[TraceT, ProviderT]:
        return self._base_metric

    def aggregate(self, values: Sequence[float]) -> float:
        return sum(values)
