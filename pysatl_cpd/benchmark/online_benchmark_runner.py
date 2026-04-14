# -*- coding: ascii -*-

"""
Abstract base class for online benchmark runners.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.benchmark.core.benchmark_logger import BenchmarkLogger
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
        verbose: bool = False,
    ) -> None:
        self._algorithms = algorithms
        self._providers = providers
        self._metrics = metrics
        self._solver = solver
        self._dump_dir = Path(dump_dir) if isinstance(dump_dir, str) else dump_dir
        self._verbose = verbose
        self._logger = BenchmarkLogger()

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

        benchmark_start = time.time()

        total_runs = sum(len(thresholds) for _, thresholds in self._algorithms)
        n_algorithms = len(self._algorithms)
        n_providers = len(self._providers)

        if not self._metrics:
            self._logger.warning_no_metrics()

        self._logger.start_benchmark(
            n_algorithms=n_algorithms,
            n_providers=n_providers,
            n_total_runs=total_runs,
        )

        results: dict[
            tuple[str, OnlineAlgorithmConfiguration],
            list[tuple[float, dict[str, Any]]],
        ] = {}

        algo_iterator = tqdm(
            self._algorithms,
            disable=not self._verbose,
            desc="Processing algorithms",
            unit="algo",
        )

        for algorithm, thresholds in algo_iterator:
            algo_name = str(algorithm)

            self._logger.algorithm_start(algo_name, len(thresholds))

            key: tuple[str, OnlineAlgorithmConfiguration] = (
                str(algorithm),
                algorithm.configuration,
            )
            results[key] = []

            threshold_iterator = tqdm(
                thresholds,
                desc=f"  Thresholds ({algo_name})",
                disable=not self._verbose,
                leave=False,
                unit="threshold",
            )

            for threshold in threshold_iterator:
                try:
                    self._logger.debug(
                        "Collecting runs",
                        algo=algo_name,
                        threshold=f"{threshold:.4f}",
                    )

                    runs = self._collect_runs(algorithm, threshold, self._providers)

                    self._logger.metrics_computed(
                        algo_name=algo_name,
                        threshold=threshold,
                        metric_names=list(self._metrics.keys()),
                    )

                    metric_values: dict[str, Any] = {
                        name: metric.evaluate(runs) for name, metric in self._metrics.items()
                    }

                    results[key].append((threshold, metric_values))

                    self._logger.threshold_processed(
                        algo_name=algo_name,
                        threshold=threshold,
                        n_providers=n_providers,
                    )

                except Exception as e:
                    self._logger.error_exception(
                        algo_name=algo_name,
                        threshold=threshold,
                        error=str(e),
                    )
                    raise

        benchmark_end = time.time()
        elapsed = benchmark_end - benchmark_start

        self._logger.benchmark_complete(
            total_runs=total_runs,
            elapsed_sec=elapsed,
        )

        return results
