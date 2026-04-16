# -*- coding: ascii -*-

"""
Unit tests for SegmentAggregationMetric.

Verifies that the metric correctly slices traces and providers according to
transition filters, groups them by transition name, and delegates evaluation
to the base aggregation metric.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence

from pysatl_cpd.benchmark.metrics.segment_aggregation_metric import SegmentAggregationMetric
from pysatl_cpd.core.data_providers.dataset import PandasLabeledDataProvider, SegmentFilter, SegmentInfo
from tests.mocks.analysis.metrics.run_metric import MockRunMetric
from tests.mocks.benchmark.metrics.aggregation_metric import MockAggregationMetric
from tests.mocks.core.data_providers.pandas_provider import MockPandasLabeledDataProvider
from tests.mocks.core.online.online_detection_trace import MockOnlineDetectionTrace


def dummy_filter(pair: tuple[SegmentInfo, SegmentInfo]) -> bool:
    """A dummy segment filter for testing."""
    return True


class TestSegmentAggregationMetricInit:
    """Tests for SegmentAggregationMetric initialization."""

    def test_initialization_stores_properties(self) -> None:
        """Metric should store the base metric and transition filters."""
        base_run_metric: MockRunMetric[MockOnlineDetectionTrace, PandasLabeledDataProvider] = MockRunMetric([1.0])
        base_agg_metric: MockAggregationMetric[MockOnlineDetectionTrace, PandasLabeledDataProvider] = (
            MockAggregationMetric(base_run_metric)
        )
        filters: dict[str, SegmentFilter] = {"A->B": dummy_filter}

        metric: SegmentAggregationMetric[MockOnlineDetectionTrace, float, float] = SegmentAggregationMetric(
            base_agg_metric=base_agg_metric,
            transition_filters=filters,
        )

        assert metric.base_agg_metric is base_agg_metric
        assert metric._transition_filters == filters


class TestSegmentAggregationMetricEvaluate:
    """Tests for the evaluate() method of SegmentAggregationMetric."""

    def test_evaluate_empty_runs(self) -> None:
        """Evaluating with an empty runs list should yield an empty result dict."""
        base_run_metric: MockRunMetric[MockOnlineDetectionTrace, PandasLabeledDataProvider] = MockRunMetric([1.0])
        base_agg_metric: MockAggregationMetric[MockOnlineDetectionTrace, PandasLabeledDataProvider] = (
            MockAggregationMetric(base_run_metric)
        )
        filters: dict[str, SegmentFilter] = {"A->B": dummy_filter}

        metric: SegmentAggregationMetric[MockOnlineDetectionTrace, float, float] = SegmentAggregationMetric(
            base_agg_metric=base_agg_metric,
            transition_filters=filters,
        )

        result: dict[str, float] = metric.evaluate([])

        # If no runs provided, no sub_runs are created, so the result should be empty
        assert result == {}
        assert len(base_agg_metric.aggregate_calls) == 0

    def test_evaluate_filters_with_no_matches_are_omitted(self) -> None:
        """Filters that produce no bisegments should not appear in the final output."""
        base_run_metric: MockRunMetric[MockOnlineDetectionTrace, PandasLabeledDataProvider] = MockRunMetric([1.0])
        base_agg_metric: MockAggregationMetric[MockOnlineDetectionTrace, PandasLabeledDataProvider] = (
            MockAggregationMetric(base_run_metric)
        )
        filters: dict[str, SegmentFilter] = {"A->B": dummy_filter, "C->D": dummy_filter}

        metric: SegmentAggregationMetric[MockOnlineDetectionTrace, float, float] = SegmentAggregationMetric(
            base_agg_metric=base_agg_metric,
            transition_filters=filters,
        )

        trace = MockOnlineDetectionTrace(detected_change_points=[])
        provider = MockPandasLabeledDataProvider(name="MainProvider")

        # We configure the provider to return nothing for any query
        provider.mock_bisegments = []
        provider.mock_indexes = []

        runs: Sequence[tuple[MockOnlineDetectionTrace, PandasLabeledDataProvider]] = [(trace, provider)]

        result: dict[str, float] = metric.evaluate(runs)

        assert result == {}
        assert len(base_agg_metric.aggregate_calls) == 0

    def test_evaluate_groups_and_delegates_correctly(self) -> None:
        """
        Metric should slice traces, group by filter name, and call the base
        metric evaluate() with the correctly grouped sub-runs.
        """
        # 1. Setup base metrics. Our mock aggregation metric just sums the results.
        # The base run metric returns 1.0 for every call.
        base_run_metric: MockRunMetric[MockOnlineDetectionTrace, PandasLabeledDataProvider] = MockRunMetric([1.0])
        base_agg_metric: MockAggregationMetric[MockOnlineDetectionTrace, PandasLabeledDataProvider] = (
            MockAggregationMetric(base_run_metric)
        )

        filters: dict[str, SegmentFilter] = {
            "A->B": dummy_filter,
            "C->D": dummy_filter,
        }

        metric: SegmentAggregationMetric[MockOnlineDetectionTrace, float, float] = SegmentAggregationMetric(
            base_agg_metric=base_agg_metric,
            transition_filters=filters,
        )

        # 2. Setup traces and providers
        main_trace = MockOnlineDetectionTrace(detected_change_points=[15, 45])
        main_provider = MockPandasLabeledDataProvider(name="MainProvider")

        # Let's say query_bisegments returns two pieces:
        # First piece: index [10, 15, 20] (covers cp at 15)
        # Second piece: index [40, 45, 50] (covers cp at 45)
        sub_prov1 = MockPandasLabeledDataProvider(name="Sub1")
        sub_prov2 = MockPandasLabeledDataProvider(name="Sub2")

        main_provider.mock_bisegments = [sub_prov1, sub_prov2]
        main_provider.mock_indexes = [(10, 15, 20), (40, 45, 50)]

        runs: Sequence[tuple[MockOnlineDetectionTrace, PandasLabeledDataProvider]] = [(main_trace, main_provider)]

        # 3. Execute
        result: dict[str, float] = metric.evaluate(runs)

        # 4. Verify results
        # The provider is queried TWICE (once for 'A->B', once for 'C->D').
        # Each query returns 2 sub-providers.
        # So 'A->B' group gets 2 sub-runs, 'C->D' group gets 2 sub-runs.
        # Since base_run_metric returns 1.0 for each run, aggregate sum is 2.0 for each group.
        assert "A->B" in result
        assert "C->D" in result
        assert result["A->B"] == 2.0
        assert result["C->D"] == 2.0

        # Verify that slicing happened correctly:
        # The run metric was called 4 times total (2 for 'A->B', 2 for 'C->D').
        assert len(base_run_metric.calls) == 4

        # Let's inspect the first call: it should be sub_prov1 and a sliced trace.
        trace1_sliced, prov1_sliced = base_run_metric.calls[0]
        assert isinstance(trace1_sliced, MockOnlineDetectionTrace)
        assert trace1_sliced.algorithm_name == "MockOnlineAlgorithm"
        # The slice was [10, 20]. The original trace had [15, 45].
        # Sliced trace should have 15 shifted by 10 -> [5].
        assert trace1_sliced.detected_change_points == [5]
        assert prov1_sliced is sub_prov1

        # Let's inspect the second call: it should be sub_prov2.
        trace2_sliced, prov2_sliced = base_run_metric.calls[1]
        assert isinstance(trace2_sliced, MockOnlineDetectionTrace)
        # The slice was [40, 50]. The original trace had [15, 45].
        # Sliced trace should have 45 shifted by 40 -> [5].
        assert trace2_sliced.detected_change_points == [5]
        assert prov2_sliced is sub_prov2
