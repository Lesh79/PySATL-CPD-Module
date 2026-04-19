# -*- coding: ascii -*-

"""
Average Run Length (ARL) benchmark runner.
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
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmConfiguration
from pysatl_cpd.core.online.online_cpd_solver import OnlineCpdSolver
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class ARLBenchmarkRunner[TraceT: OnlineDetectionTrace[Any], ProviderT: LabeledData[Any]](
    OnlineBenchmarkRunner[TraceT, ProviderT]
):
    """
    Benchmark runner specialized for Average Run Length (ARL) evaluation.
    """

    def __init__(
        self,
        solver: OnlineCpdSolver,
        mode: Literal["reset", "noreset"],
        dump_dir: Path | str | None = None,
        verbose: bool = False,
    ) -> None:
        metrics = {"arl": ARLMetric[TraceT, ProviderT]()}

        super().__init__(
            metrics=metrics,  # type: ignore[arg-type]
            solver=solver,
            dump_dir=dump_dir,
            verbose=verbose,
        )

        self._mode = mode
        if mode == "reset":
            self._inner_runner: OnlineBenchmarkRunner[Any, ProviderT] = ResetBenchmarkRunner(
                metrics=cast(Any, metrics),
                solver=solver,
                dump_dir=dump_dir,
                verbose=verbose,
            )
        elif mode == "noreset":
            self._inner_runner = NoResetBenchmarkRunner(
                metrics=cast(Any, metrics),
                solver=solver,
                policy=PointBasedPolicy(strict=True),
                dump_dir=dump_dir,
                verbose=verbose,
            )
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'reset' or 'noreset'.")

    def run(
        self,
        entries: Sequence[AlgorithmEntry[Any, Any, Any]],
        providers: Sequence[ProviderT],
    ) -> dict[tuple[str, OnlineAlgorithmConfiguration], list[tuple[float, dict[str, Any]]]]:
        """
        Validate datasets and execute the benchmark.
        """
        for provider in providers:
            if provider.change_points:
                raise ValueError(
                    f"ARL benchmark requires empty change_points, "
                    f"but provider '{provider.name}' has {list(provider.change_points)}."
                )

        # Delegate entirely to the inner runner logic
        return self._inner_runner.run(entries, providers)

    def _collect_runs(
        self,
        entry: AlgorithmEntry[Any, Any, Any],
        threshold: float,
        providers: Sequence[ProviderT],
    ) -> list[tuple[TraceT, ProviderT]]:
        runs = self._inner_runner._collect_runs(entry, threshold, providers)
        return cast(list[tuple[TraceT, ProviderT]], runs)
