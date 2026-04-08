# -*- coding: ascii -*-

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
    def __init__(self) -> None:
        self.__base_metric = RunLengthMetric[TraceT, ProviderT]()

    @property
    def base_metric(self) -> RunLengthMetric[TraceT, ProviderT]:
        return self.__base_metric

    def aggregate(self, results: Sequence[list[int]]) -> float:
        all_run_lengths = list(chain.from_iterable(results))

        if not all_run_lengths:
            return float("inf")

        return float(mean(all_run_lengths))
