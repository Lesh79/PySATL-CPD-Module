# -*- coding: ascii -*-

"""
Tests for BenchmarkAnalyzer.

Covers metric storage, evaluation routing, and edge cases with empty inputs.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.benchmark.core.benchmark_analyzer import BenchmarkAnalyzer
from pysatl_cpd.benchmark.metrics.multiple_run_metric import MultipleRunMetric
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace
from tests.mocks.analysis.labeled_data import MockLabeledData
from tests.mocks.analysis.metrics.run_metric import MockRunMetric
from tests.mocks.benchmark.metrics.aggregation_metric import MockAggregationMetric
from tests.mocks.core.online.online_detection_trace import MockOnlineDetectionTrace


class TestBenchmarkAnalyzerInit:
    """Tests for BenchmarkAnalyzer.__init__."""

    def test_init_stores_metrics(self) -> None:
        """Analyzer should store the provided metrics dictionary."""
        base: MockRunMetric[OnlineDetectionTrace[Any], LabeledData[Any]] = MockRunMetric(return_values=[1.0])
        metric: MockAggregationMetric[OnlineDetectionTrace[Any], LabeledData[Any]] = MockAggregationMetric(base=base)
        metrics: dict[str, MultipleRunMetric[OnlineDetectionTrace[Any], LabeledData[Any], Any]] = {
            "m1": metric,
        }

        analyzer: BenchmarkAnalyzer[OnlineDetectionTrace[Any], LabeledData[Any]] = BenchmarkAnalyzer(metrics=metrics)
        assert analyzer._metrics is metrics


class TestBenchmarkAnalyzerAnalyze:
    """Tests for BenchmarkAnalyzer.analyze."""

    def test_analyze_evaluates_all_metrics(self) -> None:
        """Analyzer should call evaluate() on every metric and return all results."""
        base1: MockRunMetric[OnlineDetectionTrace[Any], LabeledData[Any]] = MockRunMetric(return_values=[2.0, 3.0])
        base2: MockRunMetric[OnlineDetectionTrace[Any], LabeledData[Any]] = MockRunMetric(return_values=[10.0, 20.0])
        m1: MockAggregationMetric[OnlineDetectionTrace[Any], LabeledData[Any]] = MockAggregationMetric(base=base1)
        m2: MockAggregationMetric[OnlineDetectionTrace[Any], LabeledData[Any]] = MockAggregationMetric(base=base2)
        metrics: dict[str, MultipleRunMetric[OnlineDetectionTrace[Any], LabeledData[Any], Any]] = {
            "sum_small": m1,
            "sum_big": m2,
        }

        analyzer: BenchmarkAnalyzer[OnlineDetectionTrace[Any], LabeledData[Any]] = BenchmarkAnalyzer(metrics=metrics)

        trace1: MockOnlineDetectionTrace = MockOnlineDetectionTrace(detected_change_points=[])
        trace2: MockOnlineDetectionTrace = MockOnlineDetectionTrace(detected_change_points=[])
        data1: MockLabeledData = MockLabeledData(change_points=[], name="d1")
        data2: MockLabeledData = MockLabeledData(change_points=[], name="d2")
        runs: list[tuple[OnlineDetectionTrace[Any], LabeledData[Any]]] = [
            (trace1, data1),
            (trace2, data2),
        ]

        results: dict[str, Any] = analyzer.analyze(runs)

        assert "sum_small" in results
        assert "sum_big" in results
        assert results["sum_small"] == 2.0 + 3.0
        assert results["sum_big"] == 10.0 + 20.0

    def test_analyze_passes_runs_to_base_metric(self) -> None:
        """Base metric inside aggregation should receive the exact runs."""
        base: MockRunMetric[OnlineDetectionTrace[Any], LabeledData[Any]] = MockRunMetric(return_values=[1.0])
        metric: MockAggregationMetric[OnlineDetectionTrace[Any], LabeledData[Any]] = MockAggregationMetric(base=base)
        metrics: dict[str, MultipleRunMetric[OnlineDetectionTrace[Any], LabeledData[Any], Any]] = {
            "m": metric,
        }

        analyzer: BenchmarkAnalyzer[OnlineDetectionTrace[Any], LabeledData[Any]] = BenchmarkAnalyzer(metrics=metrics)

        trace: MockOnlineDetectionTrace = MockOnlineDetectionTrace(detected_change_points=[])
        data: MockLabeledData = MockLabeledData(change_points=[], name="d")
        runs: list[tuple[OnlineDetectionTrace[Any], LabeledData[Any]]] = [(trace, data)]

        analyzer.analyze(runs)

        assert len(base.calls) == 1
        assert base.calls[0][0] is trace
        assert base.calls[0][1] is data

    def test_analyze_with_empty_metrics(self) -> None:
        """Analyzer should return empty dict when no metrics are registered."""
        analyzer: BenchmarkAnalyzer[OnlineDetectionTrace[Any], LabeledData[Any]] = BenchmarkAnalyzer(metrics={})

        trace: MockOnlineDetectionTrace = MockOnlineDetectionTrace(detected_change_points=[])
        data: MockLabeledData = MockLabeledData(change_points=[], name="d")
        runs: list[tuple[OnlineDetectionTrace[Any], LabeledData[Any]]] = [(trace, data)]

        results: dict[str, Any] = analyzer.analyze(runs)
        assert results == {}

    def test_analyze_with_empty_runs(self) -> None:
        """Analyzer should pass empty list to metrics and return their results."""
        base: MockRunMetric[OnlineDetectionTrace[Any], LabeledData[Any]] = MockRunMetric(return_values=[99.0])
        metric: MockAggregationMetric[OnlineDetectionTrace[Any], LabeledData[Any]] = MockAggregationMetric(base=base)
        metrics: dict[str, MultipleRunMetric[OnlineDetectionTrace[Any], LabeledData[Any], Any]] = {
            "m": metric,
        }

        analyzer: BenchmarkAnalyzer[OnlineDetectionTrace[Any], LabeledData[Any]] = BenchmarkAnalyzer(metrics=metrics)

        results: dict[str, Any] = analyzer.analyze([])

        assert results == {"m": 0.0}
        assert len(metric.aggregate_calls) == 1
        assert metric.aggregate_calls[0] == []

    def test_analyze_with_multiple_runs_aggregates_correctly(self) -> None:
        """Aggregation metric should receive all per-run results and sum them."""
        base: MockRunMetric[OnlineDetectionTrace[Any], LabeledData[Any]] = MockRunMetric(return_values=[1.0, 2.0, 3.0])
        metric: MockAggregationMetric[OnlineDetectionTrace[Any], LabeledData[Any]] = MockAggregationMetric(base=base)
        metrics: dict[str, MultipleRunMetric[OnlineDetectionTrace[Any], LabeledData[Any], Any]] = {
            "total": metric,
        }

        analyzer: BenchmarkAnalyzer[OnlineDetectionTrace[Any], LabeledData[Any]] = BenchmarkAnalyzer(metrics=metrics)

        runs: list[tuple[OnlineDetectionTrace[Any], LabeledData[Any]]] = [
            (MockOnlineDetectionTrace([]), MockLabeledData([], name="a")),
            (MockOnlineDetectionTrace([]), MockLabeledData([], name="b")),
            (MockOnlineDetectionTrace([]), MockLabeledData([], name="c")),
        ]

        results: dict[str, Any] = analyzer.analyze(runs)

        assert results == {"total": 6.0}
        assert len(metric.aggregate_calls) == 1
        assert metric.aggregate_calls[0] == [1.0, 2.0, 3.0]
        assert len(base.calls) == 3
