"""
NoReset benchmark runner implementation.

Provides NoResetBenchmarkRunner - an optimised benchmark for series with
a single change point (bisegments). Returns pandas DataFrames ready for analysis.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import dataclasses
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pysatl_cpd.benchmark.core.benchmark_executor import BenchmarkExecutor
from pysatl_cpd.benchmark.metrics.multiple_run_metric import MultipleRunMetric
from pysatl_cpd.benchmark.noreset.noreset_detection_trace import NoResetDetectionTrace
from pysatl_cpd.benchmark.noreset.threshold_policy import ThresholdPolicy
from pysatl_cpd.core.algorithm_entry import AlgorithmEntry
from pysatl_cpd.core.data_providers.dataset import (
    AnnotationFilter,
    Dataset,
    PandasLabeledDataProvider,
    SegmentFilter,
)
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithm
from pysatl_cpd.core.online.online_cpd_solver import OnlineCpdSolver
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class ThresholdRange(Protocol):
    """Protocol for generating a sequence of thresholds."""

    def get_thresholds(self) -> list[float]: ...


@dataclasses.dataclass
class ManualThresholds(ThresholdRange):
    thresholds: list[float]

    def get_thresholds(self) -> list[float]:
        return self.thresholds


@dataclasses.dataclass
class LinspaceThresholds(ThresholdRange):
    start: float
    stop: float
    num: int

    def get_thresholds(self) -> list[float]:
        return np.linspace(self.start, self.stop, self.num).tolist()


class DataTransformer(Protocol):
    """Protocol for transforming data providers before running the algorithm."""

    def transform(self, provider: PandasLabeledDataProvider) -> PandasLabeledDataProvider: ...


@dataclasses.dataclass
class OnlineBenchmarkEntry:
    """
    Configuration entry for running an online algorithm in the benchmark.
    """

    algorithm: OnlineAlgorithm
    thresholds: ThresholdRange
    data_transformer: DataTransformer | None = None
    entry_name: str | None = None

    def __post_init__(self) -> None:
        if self.entry_name is None:
            name = str(self.algorithm.__class__.__name__)
            if self.data_transformer:
                name += f" + {self.data_transformer.__class__.__name__}"
            self.entry_name = name


class NoResetBenchmark:
    """
    Optimised benchmark runner for bisegments (single change point).

    Evaluates a set of algorithms across thresholds and returns a pandas DataFrame
    for each algorithm containing the computed metrics.
    """

    def __init__(
        self,
        solver: OnlineCpdSolver,
        policy: ThresholdPolicy,
        metrics: dict[str, MultipleRunMetric[NoResetDetectionTrace[Any], PandasLabeledDataProvider, Any]],
        dump_dir: str | Path | None = None,
        verbose: bool = False,
    ) -> None:
        self._solver = solver
        self._policy = policy
        self._metrics = metrics
        self._dump_dir = Path(dump_dir) if dump_dir is not None else None
        self._verbose = verbose

        self._inf_trace_cache: dict[tuple[str, int, str], OnlineDetectionTrace[Any]] = {}

    def run(
        self,
        entries: Sequence[OnlineBenchmarkEntry],
        providers: Sequence[PandasLabeledDataProvider],
    ) -> dict[str, pd.DataFrame]:
        """
        Execute the benchmark over all entries and providers.

        Parameters
        ----------
        entries : Sequence[OnlineBenchmarkEntry]
            Algorithms and their threshold configurations to evaluate.
        providers : Sequence[PandasLabeledDataProvider]
            Prepared data providers (usually bisegments).

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of entry_name to a DataFrame containing thresholds and metrics.
        """
        if not providers:
            return {entry.entry_name: pd.DataFrame() for entry in entries}

        inf_entries: list[AlgorithmEntry[Any, Any, Any]] = []
        for entry in entries:
            inf_entries.append(
                AlgorithmEntry(
                    algorithm=entry.algorithm,
                    thresholds=[float("inf")],
                    transformer=entry.data_transformer,
                )
            )

        executor: BenchmarkExecutor[Any] = BenchmarkExecutor(
            solver=self._solver,
            dump_dir=self._dump_dir,
        )

        self._inf_trace_cache.clear()
        for record, trace in executor.execute(entries=inf_entries, providers=providers):
            key = (record.algorithm, record.configuration_hash, record.data)
            self._inf_trace_cache[key] = trace

        results: dict[str, pd.DataFrame] = {}

        entries_iterator = tqdm(entries, disable=not self._verbose, desc="Evaluating Algorithms")

        for entry in entries_iterator:
            algo_name = entry.entry_name or "UnknownAlgo"
            temp_algo_entry = AlgorithmEntry(algorithm=entry.algorithm, thresholds=[])
            internal_algo_name = temp_algo_entry.full_name
            config_hash = temp_algo_entry.full_hash

            rows = []
            thresholds = entry.thresholds.get_thresholds()

            for thr in tqdm(thresholds, disable=not self._verbose, desc=f"  Thresholds ({algo_name})", leave=False):
                runs: list[tuple[NoResetDetectionTrace[Any], PandasLabeledDataProvider]] = []

                for provider in providers:
                    cache_key = (internal_algo_name, config_hash, provider.name)
                    inf_trace = self._inf_trace_cache[cache_key]

                    detected_change_points = self._policy.apply(
                        inf_trace.detection_function,
                        thr,
                        provider.change_points,
                    )

                    noreset_trace = NoResetDetectionTrace.from_inf_trace(
                        source_trace=inf_trace,
                        detected_change_points=detected_change_points,
                        threshold=thr,
                    )
                    runs.append((noreset_trace, provider))

                metric_results = {"threshold": thr}
                for metric_name, metric in self._metrics.items():
                    metric_results[metric_name] = metric.evaluate(runs)

                rows.append(metric_results)

            results[algo_name] = pd.DataFrame(rows)

        return results

    def evaluate_with_filters(
        self,
        entries: Sequence[OnlineBenchmarkEntry],
        dataset: Dataset,
        annotation_filter: AnnotationFilter | None = None,
        bisegment_filter: SegmentFilter | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Convenience method to filter the dataset and run the benchmark.

        Parameters
        ----------
        entries : Sequence[OnlineBenchmarkEntry]
            Algorithms and threshold configurations to evaluate.
        dataset : Dataset
            The complete dataset containing multiple annotated time series.
        annotation_filter : AnnotationFilter | None
            Filter to apply at the timeseries level (e.g. by scenario).
        bisegment_filter : SegmentFilter | None
            Filter to apply when extracting bisegments.

        Returns
        -------
        dict[str, pd.DataFrame]
            DataFrames with metrics for each algorithm.
        """
        filtered_dataset = dataset
        if annotation_filter is not None:
            filtered_dataset = filtered_dataset.filter_by_annotation(annotation_filter)

        providers = filtered_dataset.select_bisegments_by_filter(bisegment_filter)

        return self.run(entries, providers)
