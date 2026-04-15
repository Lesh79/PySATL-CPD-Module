# -*- coding: ascii -*-

"""
NoReset benchmark runner implementation.

This module provides NoResetBenchmarkRunner - an optimised benchmark for
series with a single change point. The solver is executed only once per
(algorithm, provider) pair with threshold=inf, and all threshold
evaluations are simulated via ThresholdPolicy on the cached trace.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.benchmark.core.benchmark_executor import BenchmarkExecutor
from pysatl_cpd.benchmark.metrics.multiple_run_metric import MultipleRunMetric
from pysatl_cpd.benchmark.noreset.noreset_detection_trace import NoResetDetectionTrace
from pysatl_cpd.benchmark.noreset.threshold_policy import ThresholdPolicy
from pysatl_cpd.benchmark.online_benchmark_runner import OnlineBenchmarkRunner
from pysatl_cpd.core.algorithm_entry import AlgorithmEntry
from pysatl_cpd.core.online.online_cpd_solver import OnlineCpdSolver
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class NoResetBenchmarkRunner[ProviderT: LabeledData[Any]](OnlineBenchmarkRunner[NoResetDetectionTrace[Any], ProviderT]):
    """
    Optimised benchmark runner for series with a single change point.

    For each (algorithm entry, provider) pair the solver is executed exactly
    once with threshold=inf, producing a full detection function trace.
    All threshold evaluations are then simulated by applying a
    ThresholdPolicy to that cached trace, avoiding redundant solver runs.
    Caching is handled entirely by BenchmarkExecutor.

    Parameters
    ----------
    entries : Sequence[AlgorithmEntry]
        Sequence of AlgorithmEntry objects containing algorithm, thresholds,
        and an optional data transformer.
    providers : Sequence[ProviderT]
        Labeled data providers to run against.
    metrics : dict[str, MultipleRunMetric[NoResetDetectionTrace[Any], ProviderT, Any]]
        Named metrics to evaluate for each (algorithm, threshold) batch.
    solver : OnlineCpdSolver
        Solver used to produce inf traces.
    policy : ThresholdPolicy
        Policy used to extract detected change points from the inf trace
        for each threshold.
    dump_dir : Path | str | None, optional
        Directory for caching inf traces via BenchmarkExecutor.
        If None, caching is disabled. Default is None.
    verbose : bool, default=False
        If True, displays progress bars during execution.
    """

    def __init__(
        self,
        entries: Sequence[AlgorithmEntry[Any, Any, Any]],
        providers: Sequence[ProviderT],
        metrics: dict[str, MultipleRunMetric[NoResetDetectionTrace[Any], ProviderT, Any]],
        solver: OnlineCpdSolver,
        policy: ThresholdPolicy,
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
        self._policy = policy

        # Replace all thresholds with inf for initial pre-caching run
        inf_entries = [dataclasses.replace(entry, thresholds=[float("inf")]) for entry in entries]

        executor: BenchmarkExecutor[Any] = BenchmarkExecutor(
            entries=inf_entries,
            providers=list(providers),
            solver=self._solver,
            dump_dir=self._dump_dir,
        )

        self._inf_trace_cache: dict[tuple[str, int, str], OnlineDetectionTrace[Any]] = {}

        for record, trace in executor.execute():
            # record.algorithm maps to entry.full_name, hash maps to entry.full_hash
            key = (record.algorithm, record.configuration_hash, record.data)
            self._inf_trace_cache[key] = trace

    def _collect_runs(
        self,
        entry: AlgorithmEntry[Any, Any, Any],
        threshold: float,
        providers: Sequence[ProviderT],
    ) -> list[tuple[NoResetDetectionTrace[Any], ProviderT]]:
        """
        Collect NoReset runs for a given algorithm entry and threshold.

        For each provider, retrieves the inf trace via BenchmarkExecutor
        and applies the ThresholdPolicy to produce a lightweight
        NoResetDetectionTrace.

        Parameters
        ----------
        entry : AlgorithmEntry
            The algorithm configuration entry to evaluate.
        threshold : float
            The detection threshold to simulate.
        providers : Sequence[ProviderT]
            Data providers to run against.

        Returns
        -------
        list[tuple[NoResetDetectionTrace[Any], ProviderT]]
            List of (noreset_trace, provider) pairs, one per provider.
        """
        if not providers:
            return []

        algo_name = entry.full_name
        config_hash = entry.full_hash
        runs: list[tuple[NoResetDetectionTrace[Any], ProviderT]] = []

        for provider in providers:
            cache_key = (algo_name, config_hash, provider.name)
            inf_trace = self._inf_trace_cache[cache_key]

            detected_change_points: list[int] = self._policy.apply(
                inf_trace.detection_function,
                threshold,
                provider.change_points,
            )

            noreset_trace = NoResetDetectionTrace.from_inf_trace(
                source_trace=inf_trace,
                detected_change_points=detected_change_points,
                threshold=threshold,
            )

            runs.append((noreset_trace, provider))

        return runs
