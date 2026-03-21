"""
Tests for OnlineCpdSolver class.
"""

from collections.abc import Callable, Iterator, Sequence
from typing import TypeVar

import pytest

from pysatl_cpd._typing import Number
from pysatl_cpd.data_providers import DataProvider
from pysatl_cpd.online.ionline_algorithm import (
    OnlineAlgorithm,
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmState,
)
from pysatl_cpd.online.online_cpd_solver import OnlineCpdSolver

T = TypeVar("T")


class MockDataProvider(DataProvider[T]):
    """Mock data provider for testing."""

    def __init__(self, data: list[T]) -> None:
        self._data = data

    def __iter__(self) -> Iterator[T]:
        return iter(self._data)


class MockOnlineAlgorithm(OnlineAlgorithm[T]):
    """Mock online algorithm for testing."""

    def __init__(
        self,
        process_return_values: Sequence[Number] | None = None,
        reset_callback: Callable[[], None] | None = None,
        state: OnlineAlgorithmState | None = None,
    ) -> None:
        self._process_return_values = process_return_values or [0.0]
        self._call_count = 0
        self._reset_callback = reset_callback
        self._state = state or OnlineAlgorithmState()

    @property
    def name(self) -> str:
        return "MockAlgorithm"

    @property
    def configuration(self) -> OnlineAlgorithmConfiguration:
        return OnlineAlgorithmConfiguration()

    @property
    def state(self) -> OnlineAlgorithmState | None:
        return self._state

    def process(self, observation: T) -> Number:
        value = self._process_return_values[self._call_count % len(self._process_return_values)]
        self._call_count += 1
        return value

    def reset(self) -> None:
        if self._reset_callback:
            self._reset_callback()


class TestOnlineCpdSolver:
    """Test suite for OnlineCpdSolver."""

    def test_initialization_defaults(self) -> None:
        """Test initialization with default parameters."""
        data = MockDataProvider([1, 2, 3])
        algorithm = MockOnlineAlgorithm[float]()

        solver = OnlineCpdSolver(data, algorithm)

        assert solver is not None

    def test_initialization_with_custom_parameters(self) -> None:
        """Test initialization with custom parameters."""
        data = MockDataProvider([1, 2, 3])
        algorithm = MockOnlineAlgorithm[float]()

        solver = OnlineCpdSolver(
            data_provider=data, algorithm=algorithm, threshold=0.5, skip_period=10, max_runlength=100
        )

        assert solver is not None

    def test_raises_value_error_for_negative_skip_period(self) -> None:
        """Test ValueError raised when skip_period is negative."""
        data = MockDataProvider([1, 2, 3])
        algorithm = MockOnlineAlgorithm[float]()

        with pytest.raises(ValueError, match="skip_period must be non-negative"):
            OnlineCpdSolver(data, algorithm, skip_period=-1)

    def test_raises_value_error_for_non_positive_max_runlength(self) -> None:
        """Test ValueError raised when max_runlength is not positive."""
        data = MockDataProvider([1, 2, 3])
        algorithm = MockOnlineAlgorithm[float]()

        with pytest.raises(ValueError, match="max_runlength must be positive"):
            OnlineCpdSolver(data, algorithm, max_runlength=0)

        with pytest.raises(ValueError, match="max_runlength must be positive"):
            OnlineCpdSolver(data, algorithm, max_runlength=-5)

    def test_run_no_detections(self) -> None:
        """Test run with no change point detections."""
        data = MockDataProvider([1, 2, 3, 4, 5])
        algorithm = MockOnlineAlgorithm[float](process_return_values=[0.1, 0.2, 0.3, 0.4, 0.5])

        solver = OnlineCpdSolver(data, algorithm, threshold=0.6)

        results = list(solver.run())

        assert len(results) == 5
        assert all(not r.is_change_point for r in results)
        assert all(not r.is_force_change_point for r in results)
        assert all(not r.is_in_skip_period for r in results)
        assert [r.step_num for r in results] == [0, 1, 2, 3, 4]

    def test_run_with_detections(self) -> None:
        """Test run with change point detections."""
        data = MockDataProvider([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        # Detection function values: low, then high at index 4
        detection_values = [0.1, 0.2, 0.3, 0.4, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5]
        algorithm = MockOnlineAlgorithm[float](process_return_values=detection_values)

        solver = OnlineCpdSolver(data, algorithm, threshold=0.5)

        results = list(solver.run())

        # Change point should be detected at step 4 (0-indexed)
        assert results[4].is_change_point is True
        assert results[4].is_force_change_point is False
        assert results[4].detection_function == 0.9

        # Other steps should not be change points
        for i, r in enumerate(results):
            if i != 4:
                assert r.is_change_point is False

    def test_run_with_skip_period(self) -> None:
        """Test run with skip period after detection."""
        data = MockDataProvider([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        detection_values = [0.1, 0.2, 0.9, 0.9, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5]
        algorithm = MockOnlineAlgorithm[float](process_return_values=detection_values)

        solver = OnlineCpdSolver(data, algorithm, threshold=0.5, skip_period=2)

        results = list(solver.run())

        # Change point at step 2
        assert results[2].is_change_point is True

        # Steps 3 and 4 should be in skip period
        assert results[3].is_in_skip_period is True
        assert results[4].is_in_skip_period is True

        # Detection at step 3 and 4 should be suppressed
        assert results[3].is_change_point is False
        assert results[4].is_change_point is False

        # After skip period, detection resumes
        assert results[5].is_in_skip_period is False

    def test_run_with_forced_change_point(self) -> None:
        """Test run with forced change point due to max_runlength."""
        data = MockDataProvider([1, 2, 3, 4, 5, 6])
        detection_values = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        algorithm = MockOnlineAlgorithm[float](process_return_values=detection_values)

        solver = OnlineCpdSolver(data, algorithm, threshold=0.5, max_runlength=3)

        results = list(solver.run())

        # Forced change point should occur at step 3 (run_length = 4)
        assert results[3].is_force_change_point is True
        assert results[3].is_change_point is True

        # After reset, run length starts over
        assert results[4].is_change_point is False

    def test_run_with_algorithm_reset_on_detection(self) -> None:
        """Test that algorithm reset is called on change point detection."""
        reset_called = False

        def reset_callback() -> None:
            nonlocal reset_called
            reset_called = True

        data = MockDataProvider([1, 2, 3])
        detection_values = [0.1, 0.2, 0.9]
        algorithm = MockOnlineAlgorithm[float](process_return_values=detection_values, reset_callback=reset_callback)

        solver = OnlineCpdSolver(data, algorithm, threshold=0.5)

        list(solver.run())

        assert reset_called is True

    def test_run_with_nan_threshold(self) -> None:
        """Test run with nan threshold (no detections)."""
        data = MockDataProvider([1, 2, 3, 4, 5])
        detection_values = [100.0, 100.0, 100.0, 100.0, 100.0]
        algorithm = MockOnlineAlgorithm[float](process_return_values=detection_values)

        solver = OnlineCpdSolver(data, algorithm, threshold=float("nan"))

        results = list(solver.run())

        # No change points should be detected despite high values
        assert all(not r.is_change_point for r in results)

    def test_run_preserves_step_order(self) -> None:
        """Test that step numbers are preserved in correct order."""
        data = MockDataProvider([1, 2, 3, 4, 5])
        algorithm = MockOnlineAlgorithm[float]()

        solver = OnlineCpdSolver(data, algorithm)

        results = list(solver.run())

        assert [r.step_num for r in results] == [0, 1, 2, 3, 4]

    def test_run_with_algorithm_state_capture(self) -> None:
        """Test that algorithm state is captured in results."""
        state = OnlineAlgorithmState(is_in_learning_period=True)
        data = MockDataProvider([1, 2, 3])
        algorithm = MockOnlineAlgorithm[float](state=state)

        solver = OnlineCpdSolver(data, algorithm)

        results = list(solver.run())

        for r in results:
            assert r.algorithm_state is state

    def test_run_processing_time_positive(self) -> None:
        """Test that processing time is positive and reasonable."""
        data = MockDataProvider([1, 2, 3])
        algorithm = MockOnlineAlgorithm[float]()

        solver = OnlineCpdSolver(data, algorithm)

        results = list(solver.run())

        for r in results:
            assert r.processing_time >= 0

    def test_run_empty_data(self) -> None:
        """Test run with empty data provider."""
        data = MockDataProvider[float]([])
        algorithm = MockOnlineAlgorithm[float]()

        solver = OnlineCpdSolver(data, algorithm)

        results = list(solver.run())

        assert len(results) == 0
