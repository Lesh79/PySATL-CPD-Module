# -*- coding: ascii -*-

"""
Reset benchmark runner implementation.

This module provides ResetBenchmarkRunner - a benchmark that runs the
solver normally, resetting the algorithm on every detected change point.
Results are cached via BenchmarkExecutor.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.benchmark.core.benchmark_executor import BenchmarkExecutor
from pysatl_cpd.benchmark.metrics.multiple_run_metric import MultipleRunMetric
from pysatl_cpd.benchmark.online_benchmark_runner import OnlineBenchmarkRunner
from pysatl_cpd.core.algorithm_entry import AlgorithmEntry
from pysatl_cpd.core.online.online_cpd_solver import OnlineCpdSolver
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class ResetBenchmarkRunner[TraceT: OnlineDetectionTrace[Any], ProviderT: LabeledData[Any]](
    OnlineBenchmarkRunner[TraceT, ProviderT]
):
    """
    Benchmark runner that uses standard reset behaviour.

    For each (algorithm entry, threshold) pair, runs the solver over all
    providers via BenchmarkExecutor. The algorithm is reset on every
    detected change point (standard solver behaviour). Results are
    cached to disk when dump_dir is provided.

    Parameters
    ----------
    entries : Sequence[AlgorithmEntry]
        Sequence of AlgorithmEntry objects containing algorithm, thresholds,
        and an optional data transformer.
    providers : Sequence[ProviderT]
        Labeled data providers to run against.
    metrics : dict[str, MultipleRunMetric[TraceT, ProviderT, Any]]
        Named metrics to evaluate for each (algorithm, threshold) batch.
    solver : OnlineCpdSolver
        Solver used to run algorithms against providers.
    dump_dir : Path | str | None, optional
        Directory for caching results via BenchmarkExecutor.
        If None, caching is disabled. Default is None.
    verbose : bool, default=False
        If True, displays progress bars during execution.
    """

    def __init__(
        self,
        entries: Sequence[AlgorithmEntry[Any, Any, Any]],
        providers: Sequence[ProviderT],
        metrics: dict[str, MultipleRunMetric[TraceT, ProviderT, Any]],
        solver: OnlineCpdSolver,
        dump_dir: Path | str | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            entries=entries,
            providers=providers,
            metrics=metrics,
            solver=solver,
            dump_dir=dump_dir,
            verbose=verbose,
        )

    def _collect_runs(
        self,
        entry: AlgorithmEntry[Any, Any, Any],
        threshold: float,
        providers: Sequence[ProviderT],
    ) -> list[tuple[TraceT, ProviderT]]:
        """
        Collect runs for a given algorithm entry and threshold via BenchmarkExecutor.

        Creates a BenchmarkExecutor with a single threshold and all providers,
        executes it, and pairs each resulting trace with its provider.

        Parameters
        ----------
        entry : AlgorithmEntry
            The algorithm configuration entry to evaluate.
        threshold : float
            The detection threshold.
        providers : Sequence[ProviderT]
            Data providers to run against.

        Returns
        -------
        list[tuple[TraceT, ProviderT]]
            List of (trace, provider) pairs, one per provider.
        """
        if not providers:
            return []

        # Create a temporary entry with only the current threshold
        single_threshold_entry = dataclasses.replace(entry, thresholds=[threshold])

        executor: BenchmarkExecutor[Any] = BenchmarkExecutor(
            entries=[single_threshold_entry],
            providers=list(providers),
            solver=self._solver,
            dump_dir=self._dump_dir,
        )

        records_and_traces = executor.execute()

        # BenchmarkExecutor returns (BenchmarkRecord, OnlineDetectionTrace) pairs.
        # We need to pair each trace with the correct provider.
        provider_by_name: dict[str, ProviderT] = {provider.name: provider for provider in providers}

        runs: list[tuple[TraceT, ProviderT]] = []
        for record, trace in records_and_traces:
            provider = provider_by_name[record.data]
            runs.append((cast(TraceT, trace), provider))

        return runs
