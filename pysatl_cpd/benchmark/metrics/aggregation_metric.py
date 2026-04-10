# -*- coding: ascii -*-

"""
Base module for aggregated benchmark metrics.

This module provides the generic `AggregationMetric` base class, which
defines the standard protocol for evaluating a single metric across an
entire dataset (multiple algorithm runs) and aggregating the results.
"""

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
    """
    Abstract base class for metrics that aggregate results over a full benchmark.

    This template class handles the evaluation of multiple detection traces
    against their corresponding datasets. It delegates the per-run evaluation
    to a base metric and then aggregates the collected results.
    """

    @property
    @abstractmethod
    def base_metric(self) -> SingleRunMetric[TraceT, ProviderT, ResultInT]:
        """
        Return the underlying metric used to evaluate a single trace.

        Returns
        -------
        SingleRunMetric[TraceT, ProviderT, ResultInT]
            The metric instance used for individual runs.
        """

    @abstractmethod
    def aggregate(self, results: Sequence[ResultInT]) -> ResultOutT:
        """
        Aggregate a sequence of single-run metric results into a final value.

        Parameters
        ----------
        results : Sequence[ResultInT]
            A sequence of results computed by the base metric for each run.

        Returns
        -------
        ResultOutT
            The aggregated metric result for the entire dataset.
        """

    def evaluate(self, runs: Sequence[tuple[TraceT, ProviderT]]) -> ResultOutT:
        """
        Evaluate the metric over a collection of runs (the entire dataset).

        Parameters
        ----------
        runs : Sequence[tuple[TraceT, ProviderT]]
            A sequence of pairs, each containing a detection trace and its
            corresponding ground truth data provider.

        Returns
        -------
        ResultOutT
            The aggregated metric result.
        """

        results = []
        for trace, data in runs:
            results.append(self.base_metric.evaluate(trace, data))

        return self.aggregate(results)
