# -*- coding: ascii -*-

"""
Abstract base class for online benchmark runners.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.benchmark.metrics.multiple_run_metric import MultipleRunMetric
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithm, OnlineAlgorithmConfiguration
from pysatl_cpd.core.online.online_cpd_solver import OnlineCpdSolver
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class OnlineBenchmarkRunner[TraceT: OnlineDetectionTrace[Any], ProviderT: LabeledData[Any]](ABC):
    """
    Abstract base class for online benchmark runners.

    Organises the evaluation loop over algorithms and thresholds,
    delegates data collection to subclasses via _collect_runs(), and
    applies all registered metrics to each batch of runs.

    Parameters
    ----------
    algorithms : Sequence[tuple[OnlineAlgorithm[Any, Any, Any], Sequence[float]]]
        Sequence of (algorithm, thresholds) pairs to evaluate.
    providers : Sequence[ProviderT]
        Sequence of labeled data providers.
    metrics : dict[str, MultipleRunMetric[TraceT, ProviderT, Any]]
        Named metrics to evaluate for each (algorithm, threshold) batch.
    solver : OnlineCpdSolver
        Solver used to run algorithms against providers.
    dump_dir : Path | str | None, optional
        Directory for caching results via BenchmarkExecutor.
        If None, caching is disabled. Default is None.
    """

    def __init__(
        self,
        algorithms: Sequence[tuple[OnlineAlgorithm[Any, Any, Any], Sequence[float]]],
        providers: Sequence[ProviderT],
        metrics: dict[str, MultipleRunMetric[TraceT, ProviderT, Any]],
        solver: OnlineCpdSolver,
        dump_dir: Path | str | None = None,
    ) -> None:
        self._algorithms = algorithms
        self._providers = providers
        self._metrics = metrics
        self._solver = solver
        self._dump_dir = Path(dump_dir) if isinstance(dump_dir, str) else dump_dir

    @abstractmethod
    def _collect_runs(
        self,
        algorithm: OnlineAlgorithm[Any, Any, Any],
        threshold: float,
        providers: Sequence[ProviderT],
    ) -> list[tuple[TraceT, ProviderT]]:
        """
        Collect (trace, provider) pairs for a given algorithm and threshold.

        Parameters
        ----------
        algorithm : OnlineAlgorithm[Any, Any, Any]
            The algorithm to evaluate.
        threshold : float
            The detection threshold.
        providers : Sequence[ProviderT]
            Sequence of data providers to run against.

        Returns
        -------
        list[tuple[TraceT, ProviderT]]
            Batch of (trace, provider) pairs for metric evaluation.
        """

        raise NotImplementedError("Method `_collect_runs` is not implemented yet.")

    def run(
        self,
    ) -> dict[tuple[str, OnlineAlgorithmConfiguration], list[tuple[float, dict[str, Any]]]]:
        """
        Execute the benchmark over all algorithms and thresholds.

        For each (algorithm, threshold) pair, collects runs via
        _collect_runs() and evaluates all registered metrics.

        Returns
        -------
        dict[tuple[str, OnlineAlgorithmConfiguration], list[tuple[float, dict[str, Any]]]]
            Mapping of (algorithm_name, configuration) to a list of
            (threshold, {metric_name: metric_value}) entries, one per threshold.
        """

        results: dict[
            tuple[str, OnlineAlgorithmConfiguration],
            list[tuple[float, dict[str, Any]]],
        ] = {}

        for algorithm, thresholds in self._algorithms:
            key: tuple[str, OnlineAlgorithmConfiguration] = (
                str(algorithm),
                algorithm.configuration,
            )
            results[key] = []

            for threshold in thresholds:
                runs = self._collect_runs(algorithm, threshold, self._providers)

                metric_values: dict[str, Any] = {name: metric.evaluate(runs) for name, metric in self._metrics.items()}

                results[key].append((threshold, metric_values))

        return results
