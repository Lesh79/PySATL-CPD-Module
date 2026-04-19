# -*- coding: ascii -*-

"""
NoReset benchmark runner implementation.
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
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmConfiguration
from pysatl_cpd.core.online.online_cpd_solver import OnlineCpdSolver
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class NoResetBenchmarkRunner[ProviderT: LabeledData[Any]](OnlineBenchmarkRunner[NoResetDetectionTrace[Any], ProviderT]):
    """
    Optimised benchmark runner for series with a single change point.
    """

    def __init__(
        self,
        metrics: dict[str, MultipleRunMetric[NoResetDetectionTrace[Any], ProviderT, Any]],
        solver: OnlineCpdSolver,
        policy: ThresholdPolicy,
        dump_dir: Path | str | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            metrics=metrics,
            solver=solver,
            dump_dir=dump_dir,
            verbose=verbose,
        )
        self._policy = policy
        self._inf_trace_cache: dict[tuple[str, int, str], OnlineDetectionTrace[Any]] = {}

    def run(
        self,
        entries: Sequence[AlgorithmEntry[Any, Any, Any]],
        providers: Sequence[ProviderT],
    ) -> dict[tuple[str, OnlineAlgorithmConfiguration], list[tuple[float, dict[str, Any]]]]:
        """
        Execute the benchmark over all entries and thresholds.

        Pre-calculates detection functions using threshold=inf before executing
        the standard evaluation loop.
        """
        inf_entries = [dataclasses.replace(entry, thresholds=[float("inf")]) for entry in entries]

        executor: BenchmarkExecutor[Any] = BenchmarkExecutor(
            solver=self._solver,
            dump_dir=self._dump_dir,
        )

        self._inf_trace_cache.clear()
        for record, trace in executor.execute(
            entries=inf_entries,
            providers=list(providers),
                ):
            key = (record.algorithm, record.configuration_hash, record.data)
            self._inf_trace_cache[key] = trace

        # Execute standard evaluation loop
        return super().run(entries, providers)

    def _collect_runs(
        self,
        entry: AlgorithmEntry[Any, Any, Any],
        threshold: float,
        providers: Sequence[ProviderT],
    ) -> list[tuple[NoResetDetectionTrace[Any], ProviderT]]:
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
