from collections.abc import Sequence

from tests.mocks.analysis.labeled_data import MockLabeledData
from tests.mocks.analysis.metrics.mock_run_metric import MockRunMetric
from tests.mocks.benchmark.metrics.mock_aggregation_metric import MockAggregationMetric
from tests.mocks.core.detection_trace import MockDetectionTrace


def make_run(detected: Sequence[int], change_points: Sequence[int]) -> tuple[MockDetectionTrace, MockLabeledData]:
    return MockDetectionTrace(detected), MockLabeledData(change_points)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAggregationMetricEvaluate:
    """Tests for AggregationMetric.evaluate."""

    # --- empty input --------------------------------------------------------

    def test_empty_runs_calls_aggregate_with_empty_list(self) -> None:
        """aggregate() must be called exactly once with an empty list."""
        base = MockRunMetric[MockDetectionTrace, MockLabeledData](return_values=[1.0])
        metric = MockAggregationMetric(base)

        metric.evaluate([])

        assert metric.aggregate_calls == [[]]

    def test_empty_runs_returns_zero(self) -> None:
        """sum([]) == 0.0, so evaluate([]) should return 0.0."""

        base = MockRunMetric[MockDetectionTrace, MockLabeledData](return_values=[1.0])
        metric = MockAggregationMetric(base)

        assert metric.evaluate([]) == 0.0

    def test_empty_runs_base_metric_never_called(self) -> None:
        """base_metric.evaluate must not be called when there are no runs."""

        base = MockRunMetric[MockDetectionTrace, MockLabeledData](return_values=[1.0])
        metric = MockAggregationMetric(base)

        metric.evaluate([])

        assert base.calls == []

    # --- single run ---------------------------------------------------------

    def test_single_run_base_metric_called_once(self) -> None:
        """base_metric.evaluate must be called exactly once for one run."""

        trace, data = make_run([5], [5])
        base = MockRunMetric[MockDetectionTrace, MockLabeledData](return_values=[3.0])
        metric = MockAggregationMetric(base)

        metric.evaluate([(trace, data)])

        assert len(base.calls) == 1

    def test_single_run_correct_arguments_forwarded(self) -> None:
        """base_metric.evaluate must receive the exact trace and data objects."""

        trace, data = make_run([5], [5])
        base = MockRunMetric[MockDetectionTrace, MockLabeledData](return_values=[3.0])
        metric = MockAggregationMetric(base)

        metric.evaluate([(trace, data)])

        assert base.calls[0] == (trace, data)

    def test_single_run_aggregate_receives_correct_list(self) -> None:
        """aggregate() must receive a one-element list with base_metric result."""

        trace, data = make_run([5], [5])
        base = MockRunMetric[MockDetectionTrace, MockLabeledData](return_values=[3.0])
        metric = MockAggregationMetric(base)

        metric.evaluate([(trace, data)])

        assert metric.aggregate_calls == [[3.0]]

    def test_single_run_result(self) -> None:
        """evaluate() must return the value produced by aggregate()."""

        trace, data = make_run([5], [5])
        base = MockRunMetric[MockDetectionTrace, MockLabeledData](return_values=[7.0])
        metric = MockAggregationMetric(base)

        assert metric.evaluate([(trace, data)]) == 7.0

    # --- multiple runs ------------------------------------------------------

    def test_multiple_runs_base_metric_called_for_each_run(self) -> None:
        """base_metric.evaluate must be called once per run."""

        runs = [make_run([i], [i]) for i in range(1, 6)]
        base = MockRunMetric[MockDetectionTrace, MockLabeledData](return_values=[1.0])
        metric = MockAggregationMetric(base)

        metric.evaluate(runs)

        assert len(base.calls) == 5

    def test_multiple_runs_arguments_forwarded_in_order(self) -> None:
        """base_metric.evaluate must receive runs in the original order."""

        runs = [make_run([i], [i]) for i in range(1, 4)]
        base = MockRunMetric[MockDetectionTrace, MockLabeledData](return_values=[1.0])
        metric = MockAggregationMetric(base)

        metric.evaluate(runs)

        for call, (expected_trace, expected_data) in zip(base.calls, runs, strict=False):
            assert call == (expected_trace, expected_data)

    def test_multiple_runs_aggregate_receives_all_results(self) -> None:
        """aggregate() must receive one result per run in order."""

        runs = [make_run([i], [i]) for i in range(1, 4)]
        base = MockRunMetric[MockDetectionTrace, MockLabeledData](return_values=[1.0, 2.0, 3.0])
        metric = MockAggregationMetric(base)

        metric.evaluate(runs)

        assert metric.aggregate_calls == [[1.0, 2.0, 3.0]]

    def test_multiple_runs_result_equals_aggregate_output(self) -> None:
        """evaluate() must return whatever aggregate() returns."""

        runs = [make_run([i], [i]) for i in range(1, 4)]
        base = MockRunMetric[MockDetectionTrace, MockLabeledData](return_values=[1.0, 2.0, 3.0])
        metric = MockAggregationMetric(base)

        assert metric.evaluate(runs) == 6.0

    # --- aggregate called exactly once --------------------------------------

    def test_aggregate_called_exactly_once(self) -> None:
        """aggregate() must be called exactly once regardless of run count."""

        runs = [make_run([i], [i]) for i in range(1, 6)]
        base = MockRunMetric[MockDetectionTrace, MockLabeledData](return_values=[1.0])
        metric = MockAggregationMetric(base)

        metric.evaluate(runs)

        assert len(metric.aggregate_calls) == 1
