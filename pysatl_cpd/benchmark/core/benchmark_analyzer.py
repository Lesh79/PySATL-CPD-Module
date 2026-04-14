# -*- coding: ascii -*-

"""
Benchmark analyzer module.

This module provides a convenient wrapper to apply multiple aggregate metrics
to a single batch of benchmark execution results.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.benchmark.metrics.multiple_run_metric import MultipleRunMetric
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class BenchmarkAnalyzer[TraceT: OnlineDetectionTrace[Any], ProviderT: LabeledData[Any]]:
    """
    Evaluator for applying multiple metrics to a batch of benchmark runs.

    This class encapsulates a dictionary of initialized metrics and provides
    a single entry point to evaluate all of them on the given execution results.

    Parameters
    ----------
    metrics : dict[str, MultipleRunMetric[TraceT, ProviderT, Any]]
        A mapping of metric names to metric instances.
    """

    def __init__(
        self,
        metrics: dict[str, MultipleRunMetric[TraceT, ProviderT, Any]],
    ) -> None:
        self._metrics = metrics

    def analyze(
        self,
        runs: list[tuple[TraceT, ProviderT]],
    ) -> dict[str, Any]:
        """
        Evaluate all registered metrics on the provided batch of runs.

        Parameters
        ----------
        runs : list[tuple[TraceT, ProviderT]]
            A batch of execution results, where each element is a pair of
            (detection_trace, data_provider).

        Returns
        -------
        dict[str, Any]
            A mapping of metric names to their evaluated results.
        """
        return {metric_name: metric.evaluate(runs) for metric_name, metric in self._metrics.items()}
