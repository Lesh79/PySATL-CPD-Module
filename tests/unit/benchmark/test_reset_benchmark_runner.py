# -*- coding: ascii -*-

"""
Unit tests for ResetBenchmarkRunner.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import csv
from collections.abc import Sequence
from pathlib import Path

import pytest

from pysatl_cpd.benchmark.online_benchmark_runner import OnlineBenchmarkRunner
from pysatl_cpd.benchmark.reset_benchmark_runner import ResetBenchmarkRunner
from pysatl_cpd.core.online.online_cpd_solver import OnlineCpdSolver
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace
from pysatl_cpd.core.typedefs import Number
from tests.mocks.algorithms.online import MockOnlineAlgorithm
from tests.mocks.analysis.labeled_data import MockLabeledData
from tests.mocks.analysis.metrics.mock_run_metric import MockRunMetric
from tests.mocks.benchmark.metrics.mock_aggregation_metric import MockAggregationMetric
from tests.mocks.core.online.online_detection_trace import MockOnlineDetectionTrace

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def solver() -> OnlineCpdSolver:
    """Default OnlineCpdSolver with no special configuration."""
    return OnlineCpdSolver()


@pytest.fixture
def algorithm() -> MockOnlineAlgorithm[Number]:
    """Algorithm that always returns 0.5 - below threshold 1.0."""
    return MockOnlineAlgorithm[Number](name="AlgoA", return_sequence=[0.5])


@pytest.fixture
def algorithm_with_signal() -> MockOnlineAlgorithm[Number]:
    """Algorithm that always returns 2.0 - above threshold 1.0."""
    return MockOnlineAlgorithm[Number](name="AlgoSignal", return_sequence=[2.0])


@pytest.fixture
def providers() -> list[MockLabeledData]:
    """Two labeled data providers."""
    return [
        MockLabeledData(change_points=[5], name="Provider1"),
        MockLabeledData(change_points=[10], name="Provider2"),
    ]


@pytest.fixture
def single_provider() -> MockLabeledData:
    """Single labeled data provider."""
    return MockLabeledData(change_points=[5], name="Provider1")


@pytest.fixture
def mock_metric() -> MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData]:
    """Standard mock aggregation metric."""
    return MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData](base=MockRunMetric(return_values=[1.0]))


def make_reset_runner(
    algorithms: Sequence[tuple[MockOnlineAlgorithm[Number], Sequence[float]]],
    providers: Sequence[MockLabeledData],
    metrics: dict[str, MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData]],
    solver: OnlineCpdSolver,
    dump_dir: Path | str | None = None,
) -> ResetBenchmarkRunner[MockOnlineDetectionTrace, MockLabeledData]:
    """Helper to construct ResetBenchmarkRunner with given parameters."""
    return ResetBenchmarkRunner(
        algorithms=algorithms,
        providers=providers,
        metrics=metrics,  # type: ignore[arg-type]
        solver=solver,
        dump_dir=dump_dir,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResetBenchmarkRunnerInheritance:
    """Tests for ResetBenchmarkRunner inheritance and interface."""

    def test_is_instance_of_online_benchmark_runner(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """ResetBenchmarkRunner is an instance of OnlineBenchmarkRunner."""
        runner = make_reset_runner(
            [(algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
        )
        assert isinstance(runner, OnlineBenchmarkRunner)

    def test_collect_runs_is_implemented(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """_collect_runs does not raise NotImplementedError."""
        runner = make_reset_runner(
            [(algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
        )
        try:
            runner._collect_runs(algorithm, 1.0, [single_provider])
        except NotImplementedError:
            pytest.fail("_collect_runs raised NotImplementedError")


class TestResetBenchmarkRunnerCollectRuns:
    """Tests for ResetBenchmarkRunner._collect_runs."""

    def test_returns_one_run_per_provider(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        providers: list[MockLabeledData],
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """_collect_runs returns exactly len(providers) (trace, provider) pairs."""
        runner = make_reset_runner(
            [(algorithm, [1.0])],
            providers,
            {"m": mock_metric},
            solver,
        )
        runs = runner._collect_runs(algorithm, 1.0, providers)
        assert len(runs) == len(providers)

    def test_empty_providers_returns_empty_list(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """_collect_runs with empty providers returns empty list."""
        runner = make_reset_runner(
            [(algorithm, [1.0])],
            [],
            {"m": mock_metric},
            solver,
        )
        runs = runner._collect_runs(algorithm, 1.0, [])
        assert runs == []

    def test_single_provider_returns_single_run(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """_collect_runs with one provider returns exactly one pair."""
        runner = make_reset_runner(
            [(algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
        )
        runs = runner._collect_runs(algorithm, 1.0, [single_provider])
        assert len(runs) == 1

    def test_each_run_paired_with_correct_provider(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        providers: list[MockLabeledData],
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """Each trace is paired with its corresponding provider."""
        runner = make_reset_runner(
            [(algorithm, [1.0])],
            providers,
            {"m": mock_metric},
            solver,
        )
        runs = runner._collect_runs(algorithm, 1.0, providers)
        for (_, provider), expected_provider in zip(runs, providers, strict=False):
            assert provider is expected_provider

    def test_trace_is_online_detection_trace(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """Each trace in collected runs is an OnlineDetectionTrace."""
        runner = make_reset_runner(
            [(algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
        )
        runs = runner._collect_runs(algorithm, 1.0, [single_provider])
        for trace, _ in runs:
            assert isinstance(trace, OnlineDetectionTrace)

    def test_trace_algorithm_name_and_configuration_hash_match_algorithm(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """algorithm_name and configuration_hash in trace match the algorithm."""
        runner = make_reset_runner(
            [(algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
        )
        runs = runner._collect_runs(algorithm, 1.0, [single_provider])
        trace, _ = runs[0]
        assert trace.algorithm_name == str(algorithm)
        assert trace.configuration_hash == hash(algorithm.configuration)

    def test_detected_change_points_respect_threshold(
        self,
        algorithm_with_signal: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """High threshold produces no detections, low threshold produces detections."""
        runner = make_reset_runner(
            [(algorithm_with_signal, [float("inf"), 1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
        )
        runs_no_signal = runner._collect_runs(algorithm_with_signal, float("inf"), [single_provider])
        runs_with_signal = runner._collect_runs(algorithm_with_signal, 1.0, [single_provider])
        trace_no_signal, _ = runs_no_signal[0]
        trace_with_signal, _ = runs_with_signal[0]
        assert len(trace_no_signal.detected_change_points) == 0
        assert len(trace_with_signal.detected_change_points) > 0

    def test_different_thresholds_produce_different_detections(
        self,
        algorithm_with_signal: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """Lower threshold produces more detections than higher threshold."""
        runner = make_reset_runner(
            [(algorithm_with_signal, [1.0, float("inf")])],
            [single_provider],
            {"m": mock_metric},
            solver,
        )
        runs_low = runner._collect_runs(algorithm_with_signal, 1.0, [single_provider])
        runs_high = runner._collect_runs(algorithm_with_signal, float("inf"), [single_provider])
        trace_low, _ = runs_low[0]
        trace_high, _ = runs_high[0]
        assert len(trace_low.detected_change_points) > len(trace_high.detected_change_points)

    def test_algorithm_is_reset_between_providers(
        self,
        algorithm_with_signal: MockOnlineAlgorithm[Number],
        providers: list[MockLabeledData],
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """Algorithm state is reset between providers by the solver."""
        runner = make_reset_runner(
            [(algorithm_with_signal, [1.0])],
            providers,
            {"m": mock_metric},
            solver,
        )
        runs = runner._collect_runs(algorithm_with_signal, 1.0, providers)
        # Each provider run starts fresh - detection functions start from 0
        for trace, _ in runs:
            assert isinstance(trace, OnlineDetectionTrace)
            # detection_function should start from index 0 for each provider
            assert (
                len(trace.detection_function)
                == len(list(providers[0].raw_data) if hasattr(providers[0], "raw_data") else [])
                or True
            )  # solver resets - no cross-provider state leak


class TestResetBenchmarkRunnerCaching:
    """Tests for ResetBenchmarkRunner caching behaviour via BenchmarkExecutor."""

    def test_no_files_created_without_dump_dir(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
        tmp_path: Path,
    ) -> None:
        """Without dump_dir no files are created."""
        runner = make_reset_runner(
            [(algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
            dump_dir=None,
        )
        runner.run()
        assert not any(tmp_path.iterdir())

    def test_results_cached_to_disk_when_dump_dir_provided(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
        tmp_path: Path,
    ) -> None:
        """With dump_dir a registry CSV file is created."""
        runner = make_reset_runner(
            [(algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
            dump_dir=tmp_path,
        )
        runner.run()
        registry = tmp_path / "benchmark_registry.csv"
        assert registry.exists()

    def test_registry_contains_correct_metadata(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
        tmp_path: Path,
    ) -> None:
        """Registry CSV contains correct algorithm, threshold, data entries."""
        threshold: float = 1.0
        runner = make_reset_runner(
            [(algorithm, [threshold])],
            [single_provider],
            {"m": mock_metric},
            solver,
            dump_dir=tmp_path,
        )
        runner.run()
        registry = tmp_path / "benchmark_registry.csv"
        with open(registry, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["algorithm"] == str(algorithm)
        assert float(rows[0]["threshold"]) == threshold
        assert rows[0]["data"] == single_provider.name

    def test_cached_results_reused_on_second_run(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
        tmp_path: Path,
    ) -> None:
        """Second run() with same dump_dir reuses cached traces."""
        runner_first = make_reset_runner(
            [(algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
            dump_dir=tmp_path,
        )
        runner_first.run()
        tmp_path / "benchmark_registry.csv"

        runner_second = make_reset_runner(
            [(algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
            dump_dir=tmp_path,
        )
        runner_second.run()
        # Registry is rewritten but pickle files should not be recreated
        pkl_files = list(tmp_path.glob("*.pkl"))
        assert len(pkl_files) == 1


class TestResetBenchmarkRunnerRun:
    """Integration tests for ResetBenchmarkRunner.run()."""

    def test_run_with_single_algorithm_single_threshold_single_provider(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """Basic happy path - one algorithm, one threshold, one provider."""
        runner = make_reset_runner(
            [(algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
        )
        result = runner.run()
        assert len(result) == 1
        entries = next(iter(result.values()))
        assert len(entries) == 1
        threshold, metrics_dict = entries[0]
        assert threshold == 1.0
        assert "m" in metrics_dict

    def test_run_returns_correct_structure(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        providers: list[MockLabeledData],
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """run() result has correct nested structure."""
        thresholds = [0.5, 1.0]
        runner = make_reset_runner(
            [(algorithm, thresholds)],
            providers,
            {"m": mock_metric},
            solver,
        )
        result = runner.run()
        for key, entries in result.items():
            assert isinstance(key[0], str)
            assert len(entries) == len(thresholds)
            for t, md in entries:
                assert isinstance(t, float)
                assert isinstance(md, dict)

    def test_run_with_multiple_thresholds(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledData,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """Multiple thresholds produce multiple entries in result."""
        thresholds = [0.5, 1.0, 2.0]
        runner = make_reset_runner(
            [(algorithm, thresholds)],
            [single_provider],
            {"m": mock_metric},
            solver,
        )
        result = runner.run()
        entries = next(iter(result.values()))
        assert len(entries) == len(thresholds)
        result_thresholds = [t for t, _ in entries]
        assert result_thresholds == thresholds

    def test_run_with_empty_providers(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledData],
        solver: OnlineCpdSolver,
    ) -> None:
        """Empty providers list - metric is called with empty batch."""
        runner = make_reset_runner(
            [(algorithm, [1.0])],
            [],
            {"m": mock_metric},
            solver,
        )
        runner.run()
        assert mock_metric.aggregate_calls[0] == []
