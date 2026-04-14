# -*- coding: ascii -*-
"""
Benchmark plotter coordinator.
"""

from __future__ import annotations

from typing import Self

import pandas as pd

from pysatl_cpd.analysis.visualization.benchmarking.abstracts import Axes, MetricVisualizer
from pysatl_cpd.analysis.visualization.typedefs import Figure

type MetricVisualizerName = str
type MetricPlotName = str


class BenchmarkPlotter:
    """
    Coordinate benchmark metric visualizers.
    """

    def __init__(self) -> None:
        self._benchmark_table: pd.DataFrame | None = None
        self._metrics: dict[MetricVisualizerName, MetricVisualizer] = {}

    @property
    def requirements(self) -> list[str]:
        all_requirements: list[str] = []
        for metric in self._metrics.values():
            all_requirements.extend(metric.requirements)
        return list(dict.fromkeys(all_requirements))

    def __getitem__(self, metric_name: MetricVisualizerName) -> MetricVisualizer:
        try:
            return self._metrics[metric_name]
        except KeyError as exc:
            raise KeyError(f"Metric visualizer '{metric_name}' is not registered.") from exc

    def set_benchmark_table(self, benchmark_table: pd.DataFrame) -> Self:
        self._benchmark_table = benchmark_table
        for metric in self._metrics.values():
            metric.set_benchmark_table(benchmark_table)
        return self

    def set_metrics(self, metrics: dict[MetricVisualizerName, MetricVisualizer]) -> Self:
        self._metrics = metrics
        if self._benchmark_table is not None:
            for metric in self._metrics.values():
                metric.set_benchmark_table(self._benchmark_table)
        return self

    def draw(self, figure: Figure, axes: dict[MetricVisualizerName, Axes]) -> Figure:
        if self._benchmark_table is None:
            raise ValueError("Benchmark table is not set.")
        if not self._metrics:
            raise ValueError("Metrics are not set.")

        missing_columns = [column for column in self.requirements if column not in self._benchmark_table.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns for BenchmarkPlotter: {missing_columns}")

        missing_axes = [metric_name for metric_name in self._metrics if metric_name not in axes]
        if missing_axes:
            raise ValueError(f"Axes mapping does not contain keys: {missing_axes}")

        for metric_name, metric in self._metrics.items():
            figure = metric.draw(figure=figure, axes=axes[metric_name])
        return figure
