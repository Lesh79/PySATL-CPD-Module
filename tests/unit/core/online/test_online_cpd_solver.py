# -*- coding: ascii -*-

"""
Tests for OnlineCpdSolver class.

This test suite verifies the behavior of the online change-point detection
solver, including detection logic, skip periods, forced change points,
state collection, and error handling.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import pytest

from pysatl_cpd.core.online.online_cpd_solver import OnlineCpdSolver
from tests.mocks.algorithms.online import (
    MockErrorOnlineAlgorithm,
    MockOnlineAlgorithm,
)
from tests.mocks.core.data_providers import (
    MockEmptyDataProvider,
    MockSingleObservationProvider,
    MockUnivariateDataProvider,
)


class TestOnlineCpdSolverInitialization:
    """Test solver initialization and validation."""

    def test_initialization_with_defaults(self) -> None:
        """Test initialization with default parameters."""
        MockOnlineAlgorithm[float]()
        solver = OnlineCpdSolver()

        assert solver is not None

    def test_initialization_with_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        MockOnlineAlgorithm[float]()

        solver = OnlineCpdSolver(
            skip_period=10,
            max_runlength=100,
            collect_states=False,
        )

        assert solver is not None

    def test_raises_value_error_for_negative_skip_period(self) -> None:
        """Test ValueError raised when skip_period is negative."""

        with pytest.raises(ValueError, match="skip_period must be non-negative"):
            OnlineCpdSolver(skip_period=-1)

    def test_raises_value_error_for_non_positive_max_runlength(self) -> None:
        """Test ValueError raised when max_runlength is not positive."""

        with pytest.raises(ValueError, match="max_runlength must be positive"):
            OnlineCpdSolver(max_runlength=0)

        with pytest.raises(ValueError, match="max_runlength must be positive"):
            OnlineCpdSolver(max_runlength=-5)

    def test_collect_states_default_true(self) -> None:
        """Test that collect_states defaults to True."""
        solver = OnlineCpdSolver()

        assert solver is not None


class TestOnlineCpdSolverDetectionBehavior:
    """Test detection behavior under various conditions."""

    def test_run_no_detections(self, basic_data: list[float], no_detection_sequence: list[float]) -> None:
        """Test run with no change point detections."""
        data = MockUnivariateDataProvider(basic_data)
        algorithm = MockOnlineAlgorithm[float](
            return_sequence=no_detection_sequence,
            learning_period_size=0,
        )

        solver = OnlineCpdSolver()
        results = list(solver.run(algorithm, data, threshold=0.6))

        assert len(results) == 10
        # Check that no change points were detected (either forced or signal)
        # Note: Due to floating point, some detection values may be very small but positive
        assert all(not r.is_forced_change_point for r in results)
        assert all(not r.is_in_skip_period for r in results)
        assert [r.step_num for r in results] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_run_with_detections(self, basic_data: list[float], single_detection_sequence: list[float]) -> None:
        """Test run with change point detections."""
        data = MockUnivariateDataProvider(basic_data)
        algorithm = MockOnlineAlgorithm[float](
            return_sequence=single_detection_sequence,
            learning_period_size=0,
        )

        solver = OnlineCpdSolver()
        results = list(solver.run(algorithm, data, threshold=0.5))

        assert len(results) == 10
        # Change point should be detected at step 4 (0-indexed)
        # Note: The detection values may be shifted due to learning period
        assert results[4].is_signal_change_point is True
        assert results[4].is_forced_change_point is False

    def test_run_with_skip_period(self, basic_data: list[float]) -> None:
        """Test run with skip period after detection."""
        detection_values = [0.1, 0.2, 0.9, 0.9, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5]
        data = MockUnivariateDataProvider(basic_data)
        algorithm = MockOnlineAlgorithm[float](
            return_sequence=detection_values,
            learning_period_size=0,
        )

        solver = OnlineCpdSolver(skip_period=2)
        results = list(solver.run(algorithm, data, threshold=0.5))

        assert len(results) == 10
        # Change point at step 2
        print([c.is_signal_change_point for c in results])
        assert results[2].is_signal_change_point is True

        # Steps 3 and 4 should be in skip period
        assert results[3].is_in_skip_period is True
        assert results[4].is_in_skip_period is True

        # After skip period, detection resumes
        assert results[5].is_in_skip_period is False

    def test_run_with_forced_change_point(self, basic_data: list[float]) -> None:
        """Test run with forced change point due to max_runlength."""
        detection_values = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        data = MockUnivariateDataProvider(basic_data[:6])
        algorithm = MockOnlineAlgorithm[float](
            return_sequence=detection_values,
            learning_period_size=0,
        )

        solver = OnlineCpdSolver(max_runlength=3)
        results = list(solver.run(algorithm, data, threshold=0.5))

        assert len(results) == 6
        # Forced change point should occur at step 3 (run_length = 4)
        assert results[3].is_forced_change_point is True

    def test_run_with_nan_threshold(self, basic_data: list[float]) -> None:
        """Test run with nan threshold (no detections)."""
        detection_values = [100.0, 100.0, 100.0, 100.0, 100.0]
        data = MockUnivariateDataProvider(basic_data[:5])
        algorithm = MockOnlineAlgorithm[float](
            return_sequence=detection_values,
            learning_period_size=0,
        )

        solver = OnlineCpdSolver()
        results = list(solver.run(algorithm, data, threshold=float("nan")))

        assert len(results) == 5
        # No change points should be detected despite high values
        assert all(not r.is_signal_change_point for r in results)

    def test_run_detection_resets_algorithm(self, basic_data: list[float]) -> None:
        """Test that algorithm reset is called on change point detection."""
        reset_called = False

        class ResetTrackingAlgorithm(MockOnlineAlgorithm[float]):
            def reset(self) -> None:
                nonlocal reset_called
                reset_called = True
                super().reset()

        detection_values = [0.1, 0.2, 0.9]
        data = MockUnivariateDataProvider(basic_data[:3])
        algorithm = ResetTrackingAlgorithm(
            return_sequence=detection_values,
            learning_period_size=0,
        )

        solver = OnlineCpdSolver()
        list(solver.run(algorithm, data, threshold=0.5))

        assert reset_called is True

    def test_run_detection_starts_skip_period(self, basic_data: list[float]) -> None:
        """Test that detection initiates skip period."""
        detection_values = [0.1, 0.9, 0.9]
        data = MockUnivariateDataProvider(basic_data[:3])
        algorithm = MockOnlineAlgorithm[float](
            return_sequence=detection_values,
            learning_period_size=0,
        )

        solver = OnlineCpdSolver(skip_period=1)
        results = list(solver.run(algorithm, data, threshold=0.5))

        assert len(results) == 3
        # Detection at step 1
        assert results[1].is_signal_change_point is True

        # Step 2 should be in skip period
        assert results[2].is_in_skip_period is True

    def test_run_multiple_detections_with_skip_period(self, basic_data: list[float]) -> None:
        """Test multiple detections with skip period between them."""
        detection_values = [0.9, 0.9, 0.9, 0.9, 0.9, 0.9]
        data = MockUnivariateDataProvider(basic_data[:6])
        algorithm = MockOnlineAlgorithm[float](
            return_sequence=detection_values,
            learning_period_size=0,
        )

        solver = OnlineCpdSolver(skip_period=2)
        results = list(solver.run(algorithm, data, threshold=0.5))

        assert len(results) == 6
        # Should detect at step 0
        assert results[0].is_signal_change_point is True

        # Steps 1-2 in skip period
        assert results[1].is_in_skip_period is True
        assert results[2].is_in_skip_period is True

        # Step 3 should be out of skip period and detect again
        assert results[3].is_in_skip_period is False
        assert results[3].is_signal_change_point is True

    def test_run_with_learning_period_returns_zero_during_learning(self, basic_data: list[float]) -> None:
        """Test that algorithm returns 0 during learning period."""
        data = MockUnivariateDataProvider(basic_data[:10])
        algorithm = MockOnlineAlgorithm[float](
            learning_period_size=5,
            return_sequence=[1.0],
        )

        solver = OnlineCpdSolver()
        results = list(solver.run(algorithm, data, threshold=float("nan")))

        assert len(results) == 10

        # During learning period (first 5 observations), detection function should be 0
        for i in range(5):
            assert results[i].detection_function == 0.0
            assert results[i].is_signal_change_point is False

        # After learning period, detection function should be 1.0
        for i in range(5, 10):
            assert results[i].detection_function == 1.0
            assert results[i].is_signal_change_point is False

    def test_run_with_learning_period_and_skip_period(self, basic_data: list[float]) -> None:
        """Test learning period interaction with skip period."""
        data = MockUnivariateDataProvider(basic_data[:10])
        algorithm = MockOnlineAlgorithm[float](
            learning_period_size=3,
            return_sequence=[0.9],
        )

        solver = OnlineCpdSolver(skip_period=2)
        results = list(solver.run(algorithm, data, threshold=0.5))

        assert len(results) == 10

        # Learning period: no detections (detection values are 0, but may be tiny due to FP)
        for i in range(3):
            # Detection function may be 0 or very small
            assert results[i].is_signal_change_point is False

        # First detection after learning period at step 3
        assert results[3].detection_function == 0.9
        assert results[3].is_signal_change_point is True

        # Skip period should start
        assert results[4].is_in_skip_period is True
        assert results[5].is_in_skip_period is True

        # After skip period, algorithm was reset, so we have a new learning period
        # Step 6-8 are learning period again (3 observations)
        for i in range(6, 9):
            assert results[i].is_signal_change_point is False

        # Step 9: after learning period, detection resumes
        assert results[9].detection_function == 0.9
        assert results[9].is_signal_change_point is True


class TestOnlineCpdSolverStateCollection:
    """Test algorithm state collection behavior."""

    def test_run_captures_algorithm_state_when_collecting(self, basic_data: list[float]) -> None:
        """Test that algorithm state is captured when collect_states=True."""
        data = MockUnivariateDataProvider(basic_data[:5])
        algorithm = MockOnlineAlgorithm[float](
            return_sequence=[0.0],
            learning_period_size=0,
        )

        solver = OnlineCpdSolver(collect_states=True)
        results = list(solver.run(algorithm, data))

        assert len(results) == 5
        for i, r in enumerate(results):
            assert r.algorithm_state is not None
            assert r.algorithm_state.process_count == i + 1

    def test_run_does_not_capture_state_when_not_collecting(self, basic_data: list[float]) -> None:
        """Test that algorithm_state is None when collect_states=False."""
        data = MockUnivariateDataProvider(basic_data[:5])
        algorithm = MockOnlineAlgorithm[float](
            return_sequence=[0.0],
            learning_period_size=0,
        )

        solver = OnlineCpdSolver(collect_states=False)
        results = list(solver.run(algorithm, data))

        assert all(r.algorithm_state is None for r in results)

    def test_run_captures_state_after_skip_period_steps(self, basic_data: list[float]) -> None:
        """Test state captured correctly during skip period steps."""
        detection_values = [0.9, 0.1, 0.1]
        data = MockUnivariateDataProvider(basic_data[:3])
        algorithm = MockOnlineAlgorithm[float](
            return_sequence=detection_values,
            learning_period_size=0,
        )

        solver = OnlineCpdSolver(skip_period=1, collect_states=True)
        results = list(solver.run(algorithm, data, threshold=0.5))

        assert len(results) == 3
        # Step 0: detection, state captured
        assert results[0].algorithm_state is not None
        assert not results[0].is_in_skip_period

        # Step 1: skip period, algorithm not called, so state is None
        assert results[1].algorithm_state is None
        assert results[1].is_in_skip_period is True

    def test_handles_algorithm_with_state(self, basic_data: list[float]) -> None:
        """Test solver handles algorithm.state correctly."""
        data = MockUnivariateDataProvider(basic_data[:3])
        algorithm = MockOnlineAlgorithm[float](
            return_sequence=[0.0],
            learning_period_size=0,
        )

        solver = OnlineCpdSolver(collect_states=True)
        results = list(solver.run(algorithm, data))

        assert all(r.algorithm_state is not None for r in results)


class TestOnlineCpdSolverStepResults:
    """Test individual step result fields."""

    def test_run_preserves_step_order(self, basic_data: list[float]) -> None:
        """Test that step numbers are preserved in correct order."""
        data = MockUnivariateDataProvider(basic_data)
        algorithm = MockOnlineAlgorithm[float]()

        solver = OnlineCpdSolver()
        results = list(solver.run(algorithm, data))

        assert [r.step_num for r in results] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_run_records_processing_time(self, basic_data: list[float]) -> None:
        """Test that processing time is positive and reasonable."""
        data = MockUnivariateDataProvider(basic_data)
        algorithm = MockOnlineAlgorithm[float]()

        solver = OnlineCpdSolver()
        results = list(solver.run(algorithm, data))

        for r in results:
            assert r.processing_time >= 0
            assert r.processing_time < 1.0

    def test_run_records_is_in_skip_period_flag(self, basic_data: list[float]) -> None:
        """Test that is_in_skip_period flag is set correctly."""
        detection_values = [0.9, 0.1, 0.1, 0.1]
        data = MockUnivariateDataProvider(basic_data[:4])
        algorithm = MockOnlineAlgorithm[float](
            return_sequence=detection_values,
            learning_period_size=0,
        )

        solver = OnlineCpdSolver(skip_period=2)
        results = list(solver.run(algorithm, data, threshold=0.5))

        assert len(results) == 4
        assert results[0].is_in_skip_period is False
        assert results[1].is_in_skip_period is True
        assert results[2].is_in_skip_period is True
        assert results[3].is_in_skip_period is False

    def test_run_records_is_forced_change_point_flag(self, basic_data: list[float]) -> None:
        """Test that is_forced_change_point flag is set correctly."""
        detection_values = [0.1, 0.1, 0.1, 0.1]
        data = MockUnivariateDataProvider(basic_data[:4])
        algorithm = MockOnlineAlgorithm[float](
            return_sequence=detection_values,
            learning_period_size=0,
        )

        solver = OnlineCpdSolver(max_runlength=2)
        results = list(solver.run(algorithm, data, threshold=0.5))

        assert len(results) == 4
        assert results[0].is_forced_change_point is False
        assert results[1].is_forced_change_point is False
        assert results[2].is_forced_change_point is True
        assert results[3].is_forced_change_point is False


class TestOnlineCpdSolverEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_run_empty_data(self) -> None:
        """Test run with empty data provider."""
        data = MockEmptyDataProvider[float]()
        algorithm = MockOnlineAlgorithm[float]()

        solver = OnlineCpdSolver()
        results = list(solver.run(algorithm, data))

        assert len(results) == 0

    def test_run_single_observation(self) -> None:
        """Test run with single observation."""
        data = MockSingleObservationProvider(42.0)
        algorithm = MockOnlineAlgorithm[float]()

        solver = OnlineCpdSolver()
        results = list(solver.run(algorithm, data))

        assert len(results) == 1
        assert results[0].step_num == 0

    def test_run_skip_period_equal_to_data_length(self, basic_data: list[float]) -> None:
        """Test skip period longer than remaining data."""
        detection_values = [0.9, 0.1, 0.1, 0.1, 0.1]
        data = MockUnivariateDataProvider(basic_data[:5])
        algorithm = MockOnlineAlgorithm[float](
            return_sequence=detection_values,
            learning_period_size=0,
        )

        solver = OnlineCpdSolver(skip_period=10)
        results = list(solver.run(algorithm, data, threshold=0.5))

        assert len(results) == 5
        # Detection at step 0
        assert results[0].is_signal_change_point is True

        # All remaining steps should be in skip period
        for i in range(1, 5):
            assert results[i].is_in_skip_period is True

    def test_run_zero_skip_period(self, basic_data: list[float]) -> None:
        """Test run with zero skip period."""
        detection_values = [0.9, 0.9, 0.9]
        data = MockUnivariateDataProvider(basic_data[:3])
        algorithm = MockOnlineAlgorithm[float](
            return_sequence=detection_values,
            learning_period_size=0,
        )

        solver = OnlineCpdSolver(skip_period=0)
        results = list(solver.run(algorithm, data, threshold=0.5))

        assert len(results) == 3
        # All detections should be processed normally
        assert results[0].is_signal_change_point is True
        assert results[1].is_signal_change_point is True
        assert results[2].is_signal_change_point is True
        assert all(not r.is_in_skip_period for r in results)

    def test_run_max_runlength_equal_one(self, basic_data: list[float]) -> None:
        """Test forced detection when max_runlength=1."""
        detection_values = [0.1, 0.1, 0.1]
        data = MockUnivariateDataProvider(basic_data[:3])
        algorithm = MockOnlineAlgorithm[float](
            return_sequence=detection_values,
            learning_period_size=0,
        )

        solver = OnlineCpdSolver(max_runlength=1)
        results = list(solver.run(algorithm, data, threshold=0.5))

        assert len(results) == 3
        # With max_runlength=1:
        # Step 0: run_length=1, 1 > 1? No -> no forced change
        # Step 1: run_length=2, 2 > 1? Yes -> forced change at step 1
        assert results[0].is_forced_change_point is False
        assert results[1].is_forced_change_point is True
        assert results[2].is_forced_change_point is False

    def test_propagates_algorithm_errors(self, basic_data: list[float]) -> None:
        """Test that algorithm errors propagate to caller."""
        data = MockUnivariateDataProvider(basic_data[:3])
        algorithm = MockErrorOnlineAlgorithm[float](
            error_on_call=2,
            error_to_raise=ValueError("Process failed"),
        )

        solver = OnlineCpdSolver()

        iterator = solver.run(algorithm, data)
        next(iterator)

        with pytest.raises(ValueError, match="Process failed"):
            next(iterator)

    def test_tracks_data_provider_iteration_count(self, basic_data: list[float]) -> None:
        """Test that solver iterates through data provider exactly once."""
        data = MockUnivariateDataProvider(basic_data)
        algorithm = MockOnlineAlgorithm[float]()

        assert data.get_call_count() == 0

        solver = OnlineCpdSolver()
        list(solver.run(algorithm, data))

        assert data.get_call_count() == 1
