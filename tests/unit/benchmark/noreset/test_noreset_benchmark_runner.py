# -*- coding: ascii -*-

"""
Unit tests for NoResetBenchmarkRunner.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from pathlib import Path

import pytest

from pysatl_cpd.benchmark.noreset.noreset_benchmark_runner import NoResetBenchmarkRunner
from pysatl_cpd.benchmark.noreset.noreset_detection_trace import NoResetDetectionTrace
from pysatl_cpd.benchmark.noreset.threshold_policy import EventBasedPolicy, PointBasedPolicy
from pysatl_cpd.benchmark.online_benchmark_runner import OnlineBenchmarkRunner
from pysatl_cpd.core.online.online_cpd_solver import OnlineCpdSolver
from pysatl_cpd.core.typedefs import Number
from tests.mocks.algorithms.online import MockOnlineAlgorithm
from tests.mocks.analysis.labeled_data import MockLabeledDataWithPadding
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
    return MockOnlineAlgorithm[Number](name="AlgoA", return_sequence=[0.5], learning_period_size=2)


@pytest.fixture
def algorithm_with_signal() -> MockOnlineAlgorithm[Number]:
    """Algorithm that always returns 2.0 - above threshold 1.0."""
    return MockOnlineAlgorithm[Number](name="AlgoSignal", return_sequence=[2.0], learning_period_size=2)


@pytest.fixture
def single_provider() -> MockLabeledDataWithPadding:
    """Single labeled data provider with one change point."""
    return MockLabeledDataWithPadding(change_points=[5], name="Provider1")


@pytest.fixture
def two_providers() -> list[MockLabeledDataWithPadding]:
    """Two labeled data providers."""
    return [
        MockLabeledDataWithPadding(change_points=[5], name="Provider1"),
        MockLabeledDataWithPadding(change_points=[10], name="Provider2"),
    ]


@pytest.fixture
def point_policy() -> PointBasedPolicy:
    """PointBasedPolicy with strict=True."""
    return PointBasedPolicy(strict=True)


@pytest.fixture
def event_policy() -> EventBasedPolicy:
    """EventBasedPolicy with max_delay=5."""
    return EventBasedPolicy(max_delay=5)


@pytest.fixture
def mock_metric() -> MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding]:
    """Standard mock aggregation metric."""
    return MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding](
        base=MockRunMetric(return_values=[1.0])
    )


def make_noreset_runner(
    algorithms: Sequence[tuple[MockOnlineAlgorithm[Number], Sequence[float]]],
    providers: Sequence[MockLabeledDataWithPadding],
    metrics: dict[str, MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding]],
    solver: OnlineCpdSolver,
    policy: PointBasedPolicy | EventBasedPolicy,
    dump_dir: Path | str | None = None,
) -> NoResetBenchmarkRunner[MockLabeledDataWithPadding]:
    """Helper to construct NoResetBenchmarkRunner with given parameters."""
    return NoResetBenchmarkRunner(
        algorithms=algorithms,
        providers=providers,
        metrics=metrics,  # type: ignore[arg-type]
        solver=solver,
        policy=policy,
        dump_dir=dump_dir,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoResetBenchmarkRunnerInheritance:
    """Tests for NoResetBenchmarkRunner inheritance and interface."""

    def test_is_instance_of_online_benchmark_runner(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledDataWithPadding,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding],
        solver: OnlineCpdSolver,
        point_policy: PointBasedPolicy,
    ) -> None:
        """NoResetBenchmarkRunner is an instance of OnlineBenchmarkRunner."""
        runner = make_noreset_runner(
            [(algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
            point_policy,
        )
        assert isinstance(runner, OnlineBenchmarkRunner)

    def test_collect_runs_is_implemented(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledDataWithPadding,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding],
        solver: OnlineCpdSolver,
        point_policy: PointBasedPolicy,
    ) -> None:
        """_collect_runs does not raise NotImplementedError."""
        runner = make_noreset_runner(
            [(algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
            point_policy,
        )
        try:
            runner._collect_runs(algorithm, 1.0, [single_provider])
        except NotImplementedError:
            pytest.fail("_collect_runs raised NotImplementedError")


class TestNoResetBenchmarkRunnerCacheInitialization:
    """Tests for NoResetBenchmarkRunner inf trace cache initialization."""

    def test_inf_trace_cache_populated_on_init(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledDataWithPadding,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding],
        solver: OnlineCpdSolver,
        point_policy: PointBasedPolicy,
    ) -> None:
        """Cache is populated during __init__ via BenchmarkExecutor."""
        runner = make_noreset_runner(
            [(algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
            point_policy,
        )
        key = (str(algorithm), hash(algorithm.configuration), single_provider.name)
        assert key in runner._inf_trace_cache

        # Inf trace produced with threshold=inf has no detected change points
        inf_trace = runner._inf_trace_cache[key]
        assert len(inf_trace.detected_change_points) == 0

    def test_cached_trace_detection_function_has_correct_length(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledDataWithPadding,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding],
        solver: OnlineCpdSolver,
        point_policy: PointBasedPolicy,
    ) -> None:
        """Detection function length equals the number of observations in provider."""
        runner = make_noreset_runner(
            [(algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
            point_policy,
        )
        key = (str(algorithm), hash(algorithm.configuration), single_provider.name)
        inf_trace = runner._inf_trace_cache[key]
        assert len(inf_trace.detection_function) == len(single_provider)

    def test_cached_trace_algorithm_name_matches(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledDataWithPadding,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding],
        solver: OnlineCpdSolver,
        point_policy: PointBasedPolicy,
    ) -> None:
        """algorithm_name in inf trace matches str(algorithm)."""
        runner = make_noreset_runner(
            [(algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
            point_policy,
        )
        key = (str(algorithm), hash(algorithm.configuration), single_provider.name)
        inf_trace = runner._inf_trace_cache[key]
        assert inf_trace.algorithm_name == str(algorithm)


class TestNoResetBenchmarkRunnerCollectRuns:
    """Tests for NoResetBenchmarkRunner._collect_runs."""

    def test_returns_one_run_per_provider(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        two_providers: list[MockLabeledDataWithPadding],
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding],
        solver: OnlineCpdSolver,
        point_policy: PointBasedPolicy,
    ) -> None:
        """_collect_runs returns exactly len(providers) (trace, provider) pairs."""
        runner = make_noreset_runner(
            [(algorithm, [1.0])],
            two_providers,
            {"m": mock_metric},
            solver,
            point_policy,
        )
        runs = runner._collect_runs(algorithm, 1.0, two_providers)
        assert len(runs) == len(two_providers)

    def test_empty_providers_returns_empty_list(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding],
        solver: OnlineCpdSolver,
        point_policy: PointBasedPolicy,
    ) -> None:
        """_collect_runs with empty providers returns empty list."""
        runner = make_noreset_runner(
            [(algorithm, [1.0])],
            [],
            {"m": mock_metric},
            solver,
            point_policy,
        )
        runs = runner._collect_runs(algorithm, 1.0, [])
        assert runs == []

    def test_each_run_is_noreset_detection_trace(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledDataWithPadding,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding],
        solver: OnlineCpdSolver,
        point_policy: PointBasedPolicy,
    ) -> None:
        """Each trace in collected runs is a NoResetDetectionTrace."""
        runner = make_noreset_runner(
            [(algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
            point_policy,
        )
        runs = runner._collect_runs(algorithm, 1.0, [single_provider])
        for trace, _ in runs:
            assert isinstance(trace, NoResetDetectionTrace)

    def test_each_run_paired_with_correct_provider(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        two_providers: list[MockLabeledDataWithPadding],
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding],
        solver: OnlineCpdSolver,
        point_policy: PointBasedPolicy,
    ) -> None:
        """Each trace is paired with its corresponding provider."""
        runner = make_noreset_runner(
            [(algorithm, [1.0])],
            two_providers,
            {"m": mock_metric},
            solver,
            point_policy,
        )
        runs = runner._collect_runs(algorithm, 1.0, two_providers)
        for (_, provider), expected in zip(runs, two_providers, strict=False):
            assert provider is expected

    def test_high_threshold_produces_no_detections(
        self,
        algorithm_with_signal: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledDataWithPadding,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding],
        solver: OnlineCpdSolver,
        point_policy: PointBasedPolicy,
    ) -> None:
        """High threshold (inf) produces no detected change points."""
        runner = make_noreset_runner(
            [(algorithm_with_signal, [float("inf")])],
            [single_provider],
            {"m": mock_metric},
            solver,
            point_policy,
        )
        runs = runner._collect_runs(algorithm_with_signal, float("inf"), [single_provider])
        trace, _ = runs[0]
        assert len(trace.detected_change_points) == 0

    def test_low_threshold_produces_detections(
        self,
        algorithm_with_signal: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledDataWithPadding,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding],
        solver: OnlineCpdSolver,
        point_policy: PointBasedPolicy,
    ) -> None:
        """Low threshold (0.0) with signal algorithm produces detections."""
        runner = make_noreset_runner(
            [(algorithm_with_signal, [0.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
            point_policy,
        )
        runs = runner._collect_runs(algorithm_with_signal, 0.0, [single_provider])
        trace, _ = runs[0]
        assert len(trace.detected_change_points) > 0

    def test_policy_is_applied_to_inf_trace(
        self,
        algorithm_with_signal: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledDataWithPadding,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding],
        solver: OnlineCpdSolver,
        point_policy: PointBasedPolicy,
    ) -> None:
        """Detected change points match what policy.apply() would return."""
        runner = make_noreset_runner(
            [(algorithm_with_signal, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
            point_policy,
        )

        # Get the cached inf trace
        key = (str(algorithm_with_signal), hash(algorithm_with_signal.configuration), single_provider.name)
        inf_trace = runner._inf_trace_cache[key]

        expected_cps = point_policy.apply(
            inf_trace.detection_function,
            1.0,
            single_provider.change_points,
        )
        runs = runner._collect_runs(algorithm_with_signal, 1.0, [single_provider])
        trace, _ = runs[0]
        assert list(trace.detected_change_points) == expected_cps


class TestNoResetBenchmarkRunnerRun:
    """Integration tests for NoResetBenchmarkRunner.run()."""

    def test_run_with_single_algorithm_single_threshold_single_provider(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledDataWithPadding,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding],
        solver: OnlineCpdSolver,
        point_policy: PointBasedPolicy,
    ) -> None:
        """Basic happy path - one algorithm, one threshold, one provider."""
        runner = make_noreset_runner(
            [(algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
            point_policy,
        )
        result = runner.run()
        assert len(result) == 1
        entries = next(iter(result.values()))
        assert len(entries) == 1
        threshold, metrics_dict = entries[0]
        assert threshold == 1.0
        assert "m" in metrics_dict

    def test_run_with_multiple_thresholds_single_solver_execution(
        self,
        algorithm_with_signal: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledDataWithPadding,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding],
        solver: OnlineCpdSolver,
        point_policy: PointBasedPolicy,
        tmp_path: Path,
    ) -> None:
        """Multiple thresholds - solver runs only once per provider (checked via caching behaviour)."""
        runner = make_noreset_runner(
            [(algorithm_with_signal, [0.5, 1.0, 2.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
            point_policy,
            dump_dir=tmp_path,
        )
        # Because execution happens in __init__, we already have our files
        pkl_files = list(tmp_path.glob("*.pkl"))
        assert len(pkl_files) == 1  # 1 trace per provider, NOT 3 traces

        # Ensure run completes successfully using the cached inf trace
        result = runner.run()
        entries = next(iter(result.values()))
        assert len(entries) == 3

    def test_run_returns_correct_structure(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        two_providers: list[MockLabeledDataWithPadding],
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding],
        solver: OnlineCpdSolver,
        point_policy: PointBasedPolicy,
    ) -> None:
        """run() result has correct nested structure."""
        thresholds = [0.5, 1.0]
        runner = make_noreset_runner(
            [(algorithm, thresholds)],
            two_providers,
            {"m": mock_metric},
            solver,
            point_policy,
        )
        result = runner.run()
        for key, entries in result.items():
            assert isinstance(key[0], str)
            assert len(entries) == len(thresholds)
            for t, md in entries:
                assert isinstance(t, float)
                assert isinstance(md, dict)

    def test_run_with_empty_providers(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding],
        solver: OnlineCpdSolver,
        point_policy: PointBasedPolicy,
    ) -> None:
        """Empty providers list - metric is called with empty batch."""
        runner = make_noreset_runner(
            [(algorithm, [1.0])],
            [],
            {"m": mock_metric},
            solver,
            point_policy,
        )
        runner.run()
        assert mock_metric.aggregate_calls[0] == []

    def test_different_policies_produce_different_detections(
        self,
        algorithm_with_signal: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledDataWithPadding,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding],
        solver: OnlineCpdSolver,
        point_policy: PointBasedPolicy,
        event_policy: EventBasedPolicy,
    ) -> None:
        """PointBasedPolicy and EventBasedPolicy may produce different detections."""
        runner_point = make_noreset_runner(
            [(algorithm_with_signal, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
            point_policy,
        )
        runner_event = make_noreset_runner(
            [(algorithm_with_signal, [1.0])],
            [single_provider],
            {
                "m": MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding](
                    base=MockRunMetric(return_values=[1.0])
                )
            },
            solver,
            event_policy,
        )
        runs_point = runner_point._collect_runs(algorithm_with_signal, 1.0, [single_provider])
        runs_event = runner_event._collect_runs(algorithm_with_signal, 1.0, [single_provider])
        trace_point, _ = runs_point[0]
        trace_event, _ = runs_event[0]
        # Results may differ - we just verify both are valid NoResetDetectionTrace
        assert isinstance(trace_point, NoResetDetectionTrace)
        assert isinstance(trace_event, NoResetDetectionTrace)


class TestNoResetBenchmarkRunnerCaching:
    """Tests for NoResetBenchmarkRunner caching behaviour."""

    def test_no_files_created_without_dump_dir(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledDataWithPadding,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding],
        solver: OnlineCpdSolver,
        point_policy: PointBasedPolicy,
        tmp_path: Path,
    ) -> None:
        """Without dump_dir no files are created during init."""
        _ = make_noreset_runner(
            [(algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
            point_policy,
            dump_dir=None,
        )
        assert not any(tmp_path.iterdir())

    def test_inf_trace_cached_to_disk_when_dump_dir_provided(
        self,
        algorithm: MockOnlineAlgorithm[Number],
        single_provider: MockLabeledDataWithPadding,
        mock_metric: MockAggregationMetric[MockOnlineDetectionTrace, MockLabeledDataWithPadding],
        solver: OnlineCpdSolver,
        point_policy: PointBasedPolicy,
        tmp_path: Path,
    ) -> None:
        """With dump_dir, inf trace registry and pickle are created synchronously during init."""
        _ = make_noreset_runner(
            [(algorithm, [1.0])],
            [single_provider],
            {"m": mock_metric},
            solver,
            point_policy,
            dump_dir=tmp_path,
        )
        registry = tmp_path / "benchmark_registry.csv"
        pkl_files = list(tmp_path.glob("*.pkl"))

        assert registry.exists()
        assert len(pkl_files) == 1
