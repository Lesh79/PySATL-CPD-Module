# -*- coding: ascii -*-

"""
Mock OnlineBenchmarkRunner for testing.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.benchmark.metrics.multiple_run_metric import MultipleRunMetric
from pysatl_cpd.benchmark.online_benchmark_runner import OnlineBenchmarkRunner
from pysatl_cpd.core.algorithm_entry import AlgorithmEntry
from pysatl_cpd.core.online.online_cpd_solver import OnlineCpdSolver
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class MockBenchmarkRunner[TraceT: OnlineDetectionTrace[Any], ProviderT: LabeledData[Any]](
    OnlineBenchmarkRunner[TraceT, ProviderT]
):
    """
    Mock implementation of OnlineBenchmarkRunner for testing.

    Records all _collect_runs calls for assertion in tests.
    Returns a pre-configured list of runs for each call.

    Parameters
    ----------
    entries : Sequence[AlgorithmEntry]
        Sequence of AlgorithmEntry objects containing algorithm, thresholds,
        and an optional data transformer.
    providers : Sequence[ProviderT]
        Sequence of data providers.
    metrics : dict[str, MultipleRunMetric[TraceT, ProviderT, Any]]
        Dictionary of metrics to evaluate.
    solver : OnlineCpdSolver
        Solver instance.
    dump_dir : Path | str | None, optional
        Directory for caching results.
    runs_to_return : list[tuple[TraceT, ProviderT]] | None, optional
        Pre-configured runs returned by _collect_runs.
        If None, returns empty list.
    """

    def __init__(
        self,
        entries: Sequence[AlgorithmEntry[Any, Any, Any]],
        providers: Sequence[ProviderT],
        metrics: dict[str, MultipleRunMetric[TraceT, ProviderT, Any]],
        solver: OnlineCpdSolver,
        dump_dir: Path | str | None = None,
        runs_to_return: list[tuple[TraceT, ProviderT]] | None = None,
    ) -> None:
        super().__init__(
            entries=entries,
            providers=providers,
            metrics=metrics,
            solver=solver,
            dump_dir=dump_dir,
        )
        self._runs_to_return: list[tuple[TraceT, ProviderT]] = runs_to_return or []
        self.collect_runs_calls: list[tuple[AlgorithmEntry[Any, Any, Any], float, Sequence[ProviderT]]] = []

    def _collect_runs(
        self,
        entry: AlgorithmEntry[Any, Any, Any],
        threshold: float,
        providers: Sequence[ProviderT],
    ) -> list[tuple[TraceT, ProviderT]]:
        """
        Record the call and return pre-configured runs.

        Parameters
        ----------
        entry : AlgorithmEntry
            The algorithm configuration entry being evaluated.
        threshold : float
            The detection threshold.
        providers : Sequence[ProviderT]
            Sequence of data providers.

        Returns
        -------
        list[tuple[TraceT, ProviderT]]
            Pre-configured runs set at construction time.
        """
        self.collect_runs_calls.append((entry, threshold, providers))
        return self._runs_to_return
