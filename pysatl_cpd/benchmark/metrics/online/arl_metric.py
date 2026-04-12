# -*- coding: ascii -*-

"""
Module for computing the Average Run Length (ARL) over a dataset.

ARL evaluates the mean distance between consecutive detections across
all runs in the benchmark, treating every detection as a signal regardless
of ground truth.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from itertools import chain
from typing import Any

from numpy import mean

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.online.run_length_metric import RunLengthMetric
from pysatl_cpd.benchmark.metrics.aggregation_metric import AggregationMetric
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class ARLMetric[TraceT: OnlineDetectionTrace[Any], ProviderT: LabeledData[Any]](
    AggregationMetric[TraceT, ProviderT, list[int], float]
):
    """
    Computes the Average Run Length (ARL) over the entire dataset.

    ARL is the mean distance between consecutive detections across all evaluated
    time series. Ground truth data is ignored. If no detections occurred in
    any of the runs, the metric returns infinity.
    """

    def __init__(self) -> None:
        self._base_metric = RunLengthMetric[TraceT, ProviderT]()

    @property
    def base_metric(self) -> RunLengthMetric[TraceT, ProviderT]:
        return self._base_metric

    def aggregate(self, results: Sequence[list[int]]) -> float:
        """
        Aggregate run lengths by flattening and computing the mean.

        Parameters
        ----------
        results : Sequence[list[int]]
            A sequence of run length lists collected from each dataset run.

        Returns
        -------
        float
            Mean of all run lengths, or infinity if no detections were made.
        """

        all_run_lengths = list(chain.from_iterable(results))

        if not all_run_lengths:
            return float("inf")

        return float(mean(all_run_lengths))
