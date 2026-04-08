# -*- coding: ascii -*-

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from itertools import chain
from typing import Any

from numpy import mean, median

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.online.delay_metric import DelayMetric
from pysatl_cpd.benchmark.metrics.aggregation_metric import AggregationMetric
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class MeanDelayMetric[TraceT: OnlineDetectionTrace[Any], ProviderT: LabeledData[Any]](
    AggregationMetric[TraceT, ProviderT, list[int], float]
):
    def __init__(self, max_delay: int) -> None:
        self.__base_metric = DelayMetric[TraceT, ProviderT](max_delay)
        self.__max_delay = max_delay

    @property
    def base_metric(self) -> DelayMetric[TraceT, ProviderT]:
        return self.__base_metric

    def aggregate(self, results: Sequence[list[int]]) -> float:
        all_delays = list(chain.from_iterable(results))
        if not all_delays:
            return float(self.__max_delay)

        return float(mean(all_delays))


class MedianDelayMetric[TraceT: OnlineDetectionTrace[Any], ProviderT: LabeledData[Any]](
    AggregationMetric[TraceT, ProviderT, list[int], float]
):
    def __init__(self, max_delay: int) -> None:
        self.__base_metric = DelayMetric[TraceT, ProviderT](max_delay)
        self.__max_delay = max_delay

    @property
    def base_metric(self) -> DelayMetric[TraceT, ProviderT]:
        return self.__base_metric

    def aggregate(self, results: Sequence[list[int]]) -> float:
        all_delays = list(chain.from_iterable(results))
        if not all_delays:
            return float(self.__max_delay)

        return float(median(all_delays))
