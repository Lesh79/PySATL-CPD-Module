# -*- coding: ascii -*-

"""
Unit tests for OnlineBenchmarkRunner.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from pathlib import Path

import pytest

from pysatl_cpd.benchmark.online_benchmark_runner import OnlineBenchmarkRunner
from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmConfiguration
from pysatl_cpd.core.online.online_cpd_solver import OnlineCpdSolver
from pysatl_cpd.core.typedefs import Number
from tests.mocks.algorithms.online import MockOnlineAlgorithm
from tests.mocks.analysis.labeled_data import MockLabeledData
from tests.mocks.analysis.metrics.mock_run_metric import MockRunMetric
from tests.mocks.benchmark.metrics.mock_aggregation_metric import MockAggregationMetric
from tests.mocks.benchmark.mock_benchmark_runner import MockBenchmarkRunner
from tests.mocks.core.online.online_detection_trace import MockOnlineDetectionTrace

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def solver() -> OnlineCpdSolver:
    """Default OnlineCpdSolver with no special configuration."""
    return OnlineCpdSolver()


@pytest.fixture
def single_algorithm() -> MockOnlineAlgorithm[Number]:
    """Single mock algorithm with return_sequence=[0.5]."""
    return MockOnlineAlgorithm[Number](name="AlgoA", return_sequence=[0.5])


@pytest.fixture
def two_algorithms() -> list[MockOnlineAlgorithm[Number]]:
    """Two mock algorithms with different configurations."""
    return [
        MockOnlineAlgorithm[Number](name="AlgoA", return_sequence=[0.5]),
        MockOnlineAlgorithm[Number](name="AlgoB", return_sequence=[1.5]),
    ]


@pytest.fixture
def single_provider() -> MockLabeledData:
    """Single labeled data provider with one change point."""
    return MockLabeledData(change_points=[5], name="Provider1")


@pytest.fixture
def two_providers() -> list[MockLabeledData]:
    """Two labeled data providers."""
    return [
        MockLabeledData(change_points=[5], name="Provider1"),
        MockLabeledData(change_points=[10], name="Provider2"),
    ]


@pytest.fixture
def mock_metric() -> MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData]:
    """Single mock aggregation metric."""
    return MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData](base=MockRunMetric(return_values=[1.0]))


@pytest.fixture
def two_metrics() -> dict[str, MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData]]:
    """Two named mock aggregation metrics."""
    return {
        "metric_a": MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData](
            base=MockRunMetric(return_values=[1.0])
        ),
        "metric_b": MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData](
            base=MockRunMetric(return_values=[2.0])
        ),
    }


@pytest.fixture
def single_run() -> list[tuple[MockOnlineDetectionTrace, MockLabeledData]]:
    """Single pre-configured run for MockBenchmarkRunner."""
    return [
        (
            MockOnlineDetectionTrace(detected_change_points=[5]),
            MockLabeledData(change_points=[5], name="Provider1"),
        )
    ]


def make_runner(
    algorithms: Sequence[tuple[MockOnlineAlgorithm[Number], Sequence[float]]],
    providers: Sequence[MockLabeledData],
    metrics: dict[str, MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData]],
    solver: OnlineCpdSolver,
    dump_dir: Path | str | None = None,
    runs_to_return: list[tuple[MockOnlineDetectionTrace, MockLabeledData]] | None = None,
) -> MockBenchmarkRunner[MockOnlineDetectionTrace, MockLabeledData]:
    """Helper to construct MockBenchmarkRunner with given parameters."""
    return MockBenchmarkRunner(
        algorithms=algorithms,
        providers=providers,
        metrics=metrics,  # type: ignore[arg-type]
        solver=solver,
        dump_dir=dump_dir,
        runs_to_return=runs_to_return or [],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOnlineBenchmarkRunnerInit:
    """Tests for OnlineBenchmarkRunner.__init__."""

    def test_stores_algorithms_providers_metrics_solver(
        self,
        single_algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """All constructor parameters are stored as private attributes."""
        algorithms = [(single_algorithm, [1.0])]
        providers = [single_provider]
        metrics = {"m": mock_metric}

        runner = make_runner(algorithms, providers, metrics, solver)

        assert runner._algorithms == algorithms
        assert runner._providers == providers
        assert runner._metrics == metrics
        assert runner._solver is solver

    def test_dump_dir_defaults_to_none(
        self,
        single_algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """dump_dir is None when not provided."""
        runner = make_runner(
            [(single_algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
        )
        assert runner._dump_dir is None

    def test_dump_dir_as_string_is_converted_to_path(
        self,
        single_algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
        tmp_path: Path,
    ) -> None:
        """dump_dir passed as str is stored as Path."""
        runner = make_runner(
            [(single_algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
            dump_dir=str(tmp_path),
        )
        assert isinstance(runner._dump_dir, Path)
        assert runner._dump_dir == tmp_path

    def test_dump_dir_as_path_is_stored_as_path(
        self,
        single_algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
        tmp_path: Path,
    ) -> None:
        """dump_dir passed as Path is stored as Path."""
        runner = make_runner(
            [(single_algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
            dump_dir=tmp_path,
        )
        assert isinstance(runner._dump_dir, Path)
        assert runner._dump_dir == tmp_path


class TestOnlineBenchmarkRunnerAbstract:
    """Tests for OnlineBenchmarkRunner abstract interface."""

    def test_cannot_instantiate_directly(
        self,
        single_algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """OnlineBenchmarkRunner cannot be instantiated directly."""
        with pytest.raises(TypeError):
            OnlineBenchmarkRunner(  # type: ignore[abstract]
                algorithms=[(single_algorithm, [1.0])],
                providers=[single_provider],
                metrics={"m": mock_metric},
                solver=solver,
            )

    def test_subclass_without_collect_runs_cannot_instantiate(
        self,
        single_algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """Subclass without _collect_runs implementation cannot be instantiated."""

        class IncompleteRunner(OnlineBenchmarkRunner):  # type: ignore[type-arg]
            pass

        with pytest.raises(TypeError):
            IncompleteRunner(  # type: ignore[abstract]
                algorithms=[(single_algorithm, [1.0])],
                providers=[single_provider],
                metrics={"m": mock_metric},
                solver=solver,
            )


class TestOnlineBenchmarkRunnerRunStructure:
    """Tests for the structure of run() return value."""

    def test_run_returns_dict(
        self,
        single_algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """run() returns a dict."""
        runner = make_runner(
            [(single_algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
        )
        result = runner.run()
        assert isinstance(result, dict)

    def test_result_key_is_tuple_of_name_and_configuration(
        self,
        single_algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """Keys of result dict are (str, OnlineAlgorithmConfiguration) tuples."""
        runner = make_runner(
            [(single_algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
        )
        result = runner.run()
        for key in result:
            assert isinstance(key, tuple)
            assert len(key) == 2
            assert isinstance(key[0], str)
            assert isinstance(key[1], OnlineAlgorithmConfiguration)

    def test_result_value_is_list_of_threshold_metric_tuples(
        self,
        single_algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """Values of result dict are list[tuple[float, dict[str, Any]]]."""
        runner = make_runner(
            [(single_algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
        )
        result = runner.run()
        for entries in result.values():
            assert isinstance(entries, list)
            for threshold, metrics_dict in entries:
                assert isinstance(threshold, float)
                assert isinstance(metrics_dict, dict)

    def test_one_entry_per_threshold_in_result(
        self,
        single_algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """Each threshold produces exactly one entry in the result list."""
        thresholds = [0.5, 1.0, 1.5]
        runner = make_runner(
            [(single_algorithm, thresholds)],
            [single_provider],
            {"m": mock_metric},
            solver,
        )
        result = runner.run()
        key = (str(single_algorithm), single_algorithm.configuration)
        assert len(result[key]) == len(thresholds)

    def test_metric_names_match_input_dict_keys(
        self,
        single_algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        two_metrics: dict[str, MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData]],
        solver: OnlineCpdSolver,
    ) -> None:
        """Metric names in result match the keys from the metrics dict."""
        runner = make_runner(
            [(single_algorithm, [1.0])],
            [single_provider],
            two_metrics,
            solver,
        )
        result = runner.run()
        for entries in result.values():
            for _, metrics_dict in entries:
                assert set(metrics_dict.keys()) == set(two_metrics.keys())


class TestOnlineBenchmarkRunnerRunLogic:
    """Tests for the logic of run() execution."""

    def test_collect_runs_called_once_per_algorithm_threshold_pair(
        self,
        single_algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """_collect_runs is called exactly once per (algorithm, threshold) pair."""
        thresholds = [0.5, 1.0, 1.5]
        runner = make_runner(
            [(single_algorithm, thresholds)],
            [single_provider],
            {"m": mock_metric},
            solver,
        )
        runner.run()
        assert len(runner.collect_runs_calls) == len(thresholds)

    def test_metric_evaluate_called_once_per_threshold(
        self,
        single_algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """metric.evaluate() is called once per (algorithm, threshold) pair."""
        thresholds = [0.5, 1.0]
        runner = make_runner(
            [(single_algorithm, thresholds)],
            [single_provider],
            {"m": mock_metric},
            solver,
        )
        runner.run()
        assert len(mock_metric.aggregate_calls) == len(thresholds)

    def test_multiple_algorithms_produce_multiple_keys(
        self,
        two_algorithms: list[MockOnlineAlgorithm[Number]],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """Two algorithms produce two distinct keys in result dict."""
        runner = make_runner(
            [(algo, [1.0]) for algo in two_algorithms],
            [single_provider],
            {"m": mock_metric},
            solver,
        )
        result = runner.run()
        assert len(result) == 2

    def test_multiple_thresholds_produce_multiple_entries(
        self,
        single_algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """Two thresholds produce two entries in the result list for one algorithm."""
        thresholds = [0.5, 1.5]
        runner = make_runner(
            [(single_algorithm, thresholds)],
            [single_provider],
            {"m": mock_metric},
            solver,
        )
        result = runner.run()
        key = (str(single_algorithm), single_algorithm.configuration)
        assert len(result[key]) == 2

    def test_multiple_metrics_all_appear_in_result(
        self,
        single_algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        two_metrics: dict[str, MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData]],
        solver: OnlineCpdSolver,
    ) -> None:
        """All metrics from input dict appear in every result entry."""
        runner = make_runner(
            [(single_algorithm, [1.0])],
            [single_provider],
            two_metrics,
            solver,
        )
        result = runner.run()
        for entries in result.values():
            for _, metrics_dict in entries:
                assert "metric_a" in metrics_dict
                assert "metric_b" in metrics_dict

    def test_correct_threshold_passed_to_collect_runs(
        self,
        single_algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """_collect_runs receives exactly the threshold from the input list."""
        thresholds = [0.5, 1.0, 2.0]
        runner = make_runner(
            [(single_algorithm, thresholds)],
            [single_provider],
            {"m": mock_metric},
            solver,
        )
        runner.run()
        called_thresholds = [call[1] for call in runner.collect_runs_calls]
        assert called_thresholds == thresholds

    def test_collect_runs_receives_all_providers(
        self,
        single_algorithm: MockOnlineAlgorithm[Number],
        two_providers: list[MockLabeledData],
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """_collect_runs receives the full list of providers."""
        runner = make_runner(
            [(single_algorithm, [1.0])],
            two_providers,
            {"m": mock_metric},
            solver,
        )
        runner.run()
        assert runner.collect_runs_calls[0][2] == two_providers

    def test_empty_providers_produces_empty_batch(
        self,
        single_algorithm: MockOnlineAlgorithm[Number],
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """Empty providers list results in metric being called with empty runs."""
        runner = make_runner(
            [(single_algorithm, [1.0])],
            [],
            {"m": mock_metric},
            solver,
        )
        runner.run()
        assert mock_metric.aggregate_calls[0] == []

    def test_empty_thresholds_produces_no_entries(
        self,
        single_algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """Empty thresholds list produces empty entries list for the algorithm."""
        runner = make_runner(
            [(single_algorithm, [])],
            [single_provider],
            {"m": mock_metric},
            solver,
        )
        result = runner.run()
        key = (str(single_algorithm), single_algorithm.configuration)
        assert result[key] == []

    def test_result_preserves_threshold_order(
        self,
        single_algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """Thresholds in result appear in the same order as in input list."""
        thresholds = [2.0, 0.5, 1.0]
        runner = make_runner(
            [(single_algorithm, thresholds)],
            [single_provider],
            {"m": mock_metric},
            solver,
        )
        result = runner.run()
        key = (str(single_algorithm), single_algorithm.configuration)
        result_thresholds = [t for t, _ in result[key]]
        assert result_thresholds == thresholds
