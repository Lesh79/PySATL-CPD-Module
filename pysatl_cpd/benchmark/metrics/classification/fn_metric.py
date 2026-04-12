# -*- coding: ascii -*-

"""
Module for computing aggregated False Negatives (FN) over a dataset.

A False Negative occurs when an actual change point is missed. This metric
sums the total number of missed change points across all benchmark runs.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.classification.fn_metric import FalseNegativeMetric as SingleFN
from pysatl_cpd.benchmark.metrics.aggregation_metric import AggregationMetric
from pysatl_cpd.core.detection_trace import DetectionTrace


class FalseNegativeMetric[TraceT: DetectionTrace, ProviderT: LabeledData[Any]](
    AggregationMetric[TraceT, ProviderT, float, float]
):
    """
    Metric for summing False Negatives (FN) across an entire dataset.

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
        self._base_metric = SingleFN[TraceT, ProviderT](error_margin)

    @property
    def base_metric(self) -> SingleFN[TraceT, ProviderT]:
        return self._base_metric

    def aggregate(self, values: Sequence[float]) -> float:
        return sum(values)
