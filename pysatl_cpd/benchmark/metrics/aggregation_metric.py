# -*- coding: ascii -*-

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.run_metric import RunMetric as SingleRunMetric
from pysatl_cpd.benchmark.metrics.run_metric import RunMetric
from pysatl_cpd.core.detection_trace import DetectionTrace


class AggregationMetric[TraceT: DetectionTrace, ProviderT: LabeledData[Any], ResultInT, ResultOutT](
    RunMetric[TraceT, ProviderT, ResultOutT]
):
    @property
    @abstractmethod
    def base_metric(self) -> SingleRunMetric[TraceT, ProviderT, ResultInT]:
        raise NotImplementedError

    @abstractmethod
    def aggregate(self, results: Sequence[ResultInT]) -> ResultOutT:
        raise NotImplementedError

    def evaluate(self, runs: Sequence[tuple[TraceT, ProviderT]]) -> ResultOutT:
        results = []
        for trace, data in runs:
            results.append(self.base_metric.evaluate(trace, data))

        return self.aggregate(results)
