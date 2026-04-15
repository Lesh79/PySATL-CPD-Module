# -*- coding: ascii -*-

"""
Average Run Length (ARL) benchmark runner.

This module provides the ARLBenchmarkRunner which evaluates the distance
between consecutive false alarms. It automatically applies the ARLMetric
and ensures that the provided datasets do not contain any true change points.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, cast

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.benchmark.metrics.online.arl_metric import ARLMetric
from pysatl_cpd.benchmark.noreset.noreset_benchmark_runner import NoResetBenchmarkRunner
from pysatl_cpd.benchmark.noreset.threshold_policy import PointBasedPolicy
from pysatl_cpd.benchmark.online_benchmark_runner import OnlineBenchmarkRunner
from pysatl_cpd.benchmark.reset_benchmark_runner import ResetBenchmarkRunner
from pysatl_cpd.core.algorithm_entry import AlgorithmEntry
from pysatl_cpd.core.online.online_cpd_solver import OnlineCpdSolver
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class ARLBenchmarkRunner[TraceT: OnlineDetectionTrace[Any], ProviderT: LabeledData[Any]](
    OnlineBenchmarkRunner[TraceT, ProviderT]
):
    """
    Benchmark runner specialized for Average Run Length (ARL) evaluation.

    ARL represents the mean distance between consecutive detections (false alarms)
    when no true change points are present in the data. This runner strictly
    validates that all providers have empty `change_points`.

    It supports two modes:
    - "reset": The algorithm state is reset after every detection (standard behavior).
    - "noreset": The algorithm state is not reset. A single infinite-threshold run
      is cached, and signals are extracted using a strict point-based policy.

    Parameters
    ----------
    entries : Sequence[AlgorithmEntry]
        Sequence of AlgorithmEntry objects containing algorithm, thresholds,
        and an optional data transformer.
    providers : list[ProviderT]
        Labeled data providers to run against. Must have `change_points == []`.
    solver : OnlineCpdSolver
        Solver used to run algorithms against providers.
    mode : Literal["reset", "noreset"]
        Evaluation mode determining whether the algorithm resets after a detection.
    dump_dir : Path | str | None, optional
        Directory for caching results via BenchmarkExecutor.
        If None, caching is disabled. Default is None.
    verbose : bool, default=False
        If True, displays progress bars during execution.

    Raises
    ------
    ValueError
        If any provider contains non-empty `change_points`.
    ValueError
        If `mode` is neither "reset" nor "noreset".
    """

    def __init__(
        self,
        entries: Sequence[AlgorithmEntry[Any, Any, Any]],
        providers: list[ProviderT],
        solver: OnlineCpdSolver,
        mode: Literal["reset", "noreset"],
        dump_dir: Path | str | None = None,
        verbose: bool = False,
    ) -> None:
        for provider in providers:
            if provider.change_points:
                raise ValueError(
                    f"ARL benchmark requires empty change_points, "
                    f"but provider '{provider.name}' has {list(provider.change_points)}."
                )

        metrics = {"arl": ARLMetric[TraceT, ProviderT]()}

        super().__init__(
            entries=entries,
            providers=providers,
            metrics=metrics,  # type: ignore[arg-type]
            solver=solver,
            dump_dir=dump_dir,
            verbose=verbose,
        )

        self._mode = mode
        if mode == "reset":
            # Delegate to standard ResetBenchmarkRunner
            self._inner_runner: OnlineBenchmarkRunner[Any, ProviderT] = ResetBenchmarkRunner(
                entries=entries,
                providers=providers,
                metrics=cast(Any, metrics),
                solver=solver,
                dump_dir=dump_dir,
            )
        elif mode == "noreset":
            # Delegate to optimized NoResetBenchmarkRunner with PointBased policy
            self._inner_runner = NoResetBenchmarkRunner(
                entries=entries,
                providers=providers,
                metrics=cast(Any, metrics),
                solver=solver,
                policy=PointBasedPolicy(strict=True),
                dump_dir=dump_dir,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'reset' or 'noreset'.")

    def _collect_runs(
        self,
        entry: AlgorithmEntry[Any, Any, Any],
        threshold: float,
        providers: Sequence[ProviderT],
    ) -> list[tuple[TraceT, ProviderT]]:
        """
        Collect runs for a given algorithm entry and threshold using the configured mode.

        Delegates the collection to either ResetBenchmarkRunner or
        NoResetBenchmarkRunner depending on the initialized mode.

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
            Batch of (trace, provider) pairs.
        """
        runs = self._inner_runner._collect_runs(entry, threshold, providers)
        return cast(list[tuple[TraceT, ProviderT]], runs)
