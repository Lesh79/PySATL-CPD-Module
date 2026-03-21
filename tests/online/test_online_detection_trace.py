"""
Tests for online detection trace containers.
"""

from typing import cast

import numpy as np
import pytest

from pysatl_cpd._typing import UnivariateNumericArray
from pysatl_cpd.online.ionline_algorithm import OnlineAlgorithmState
from pysatl_cpd.online.online_detection_trace import (
    OnlineDetectionStepResult,
    OnlineDetectionTrace,
)


class MockAlgorithmState(OnlineAlgorithmState):
    """Mock algorithm state for testing."""


class TestOnlineDetectionStepResult:
    """Test suite for OnlineDetectionStepResult."""

    def test_default_values(self) -> None:
        """Test default values for all fields."""
        result = OnlineDetectionStepResult()

        assert result.step_num == 0
        assert result.is_change_point is False
        assert result.is_force_change_point is False
        assert result.is_in_skip_period is False
        assert np.isnan(result.detection_function)
        assert np.isnan(result.processing_time)
        assert result.algorithm_state is None

    def test_custom_values(self) -> None:
        """Test setting custom values during initialization."""
        state = MockAlgorithmState()
        result = OnlineDetectionStepResult(
            step_num=5,
            is_change_point=True,
            is_force_change_point=False,
            is_in_skip_period=False,
            detection_function=0.85,
            processing_time=0.00123,
            algorithm_state=state,
        )

        assert result.step_num == 5
        assert result.is_change_point is True
        assert result.is_force_change_point is False
        assert result.is_in_skip_period is False
        assert result.detection_function == 0.85
        assert result.processing_time == 0.00123
        assert result.algorithm_state is state


class TestOnlineDetectionTrace:
    """Test suite for OnlineDetectionTrace."""

    @pytest.fixture
    def sample_steps(self) -> list[OnlineDetectionStepResult]:
        """Create sample step results for testing."""
        state1 = MockAlgorithmState()
        state2 = MockAlgorithmState()

        return [
            OnlineDetectionStepResult(
                step_num=0,
                is_change_point=False,
                is_force_change_point=False,
                is_in_skip_period=False,
                detection_function=0.1,
                processing_time=0.001,
                algorithm_state=state1,
            ),
            OnlineDetectionStepResult(
                step_num=1,
                is_change_point=True,
                is_force_change_point=False,
                is_in_skip_period=False,
                detection_function=0.9,
                processing_time=0.002,
                algorithm_state=state2,
            ),
            OnlineDetectionStepResult(
                step_num=2,
                is_change_point=False,
                is_force_change_point=False,
                is_in_skip_period=True,
                detection_function=0.0,
                processing_time=0.0,
                algorithm_state=None,
            ),
            OnlineDetectionStepResult(
                step_num=3,
                is_change_point=False,
                is_force_change_point=True,
                is_in_skip_period=False,
                detection_function=1.2,
                processing_time=0.003,
                algorithm_state=None,
            ),
        ]

    def test_from_online_detection_steps(self, sample_steps: list[OnlineDetectionStepResult]) -> None:
        """Test constructing OnlineDetectionTrace from step results."""
        trace = OnlineDetectionTrace.from_online_detection_steps(threshold=0.5, steps=sample_steps)

        assert trace.threshold == 0.5
        assert isinstance(trace.observation_scores, np.ndarray)
        assert trace.observation_scores.ndim == 1
        assert np.array_equal(trace.observation_scores, np.array([0.1, 0.9, 0.0, 1.2]))
        assert isinstance(trace.processing_time, np.ndarray)
        assert trace.processing_time.ndim == 1
        assert np.array_equal(trace.processing_time, np.array([0.001, 0.002, 0.0, 0.003]))
        assert trace.detected_changes == [1]
        assert trace.skipped_observation == [2]
        assert trace.forced_change_points == [3]
        assert isinstance(trace.algorithm_states, list)
        assert len(trace.algorithm_states) == 4
        assert trace.algorithm_states[0] is not None
        assert trace.algorithm_states[1] is not None
        assert trace.algorithm_states[2] is None
        assert trace.algorithm_states[3] is None

    def test_from_online_detection_steps_with_none_threshold(
        self, sample_steps: list[OnlineDetectionStepResult]
    ) -> None:
        """Test constructing trace with None threshold."""
        trace = OnlineDetectionTrace.from_online_detection_steps(threshold=None, steps=sample_steps)

        assert trace.threshold is None
        assert len(trace.observation_scores) == 4

    def test_from_online_detection_steps_empty(self) -> None:
        """Test constructing trace from empty step sequence."""
        trace = OnlineDetectionTrace.from_online_detection_steps(threshold=0.5, steps=[])

        assert trace.threshold == 0.5
        assert isinstance(trace.observation_scores, np.ndarray)
        assert len(trace.observation_scores) == 0
        assert isinstance(trace.processing_time, np.ndarray)
        assert len(trace.processing_time) == 0
        assert trace.detected_changes == []
        assert trace.skipped_observation == []
        assert trace.forced_change_points == []
        assert trace.algorithm_states == []

    def test_from_online_detection_steps_no_detections(self) -> None:
        """Test constructing trace with no changepoints."""
        steps: list[OnlineDetectionStepResult] = [
            OnlineDetectionStepResult(
                step_num=i,
                is_change_point=False,
                is_force_change_point=False,
                is_in_skip_period=False,
                detection_function=0.1,
                processing_time=0.001,
                algorithm_state=None,
            )
            for i in range(5)
        ]

        trace = OnlineDetectionTrace.from_online_detection_steps(threshold=0.5, steps=steps)

        assert trace.detected_changes == []
        assert trace.forced_change_points == []
        assert trace.skipped_observation == []

    def test_direct_initialization(self) -> None:
        """Test direct initialization of OnlineDetectionTrace."""
        states: list[OnlineAlgorithmState | None] = [MockAlgorithmState() for _ in range(3)]

        observation_scores: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.2, 0.5, 0.8], dtype=np.float64)
        )
        processing_times: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.001, 0.002, 0.003], dtype=np.float64)
        )

        trace = OnlineDetectionTrace(
            threshold=0.75,
            observation_scores=observation_scores,
            processing_time=processing_times,
            algorithm_states=states,
            detected_changes=[2],
            skipped_observation=[1],
            forced_change_points=[],
        )

        assert trace.threshold == 0.75
        assert isinstance(trace.observation_scores, np.ndarray)
        assert len(trace.observation_scores) == 3
        assert isinstance(trace.processing_time, np.ndarray)
        assert len(trace.processing_time) == 3
        assert trace.detected_changes == [2]
        assert trace.skipped_observation == [1]
        assert trace.forced_change_points == []

    def test_inherits_from_detection_trace(self) -> None:
        """Test that OnlineDetectionTrace inherits DetectionTrace functionality."""
        observation_scores: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.1, 0.2, 0.3], dtype=np.float64)
        )
        processing_times: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.001, 0.002, 0.003], dtype=np.float64)
        )

        trace = OnlineDetectionTrace(
            threshold=0.5,
            observation_scores=observation_scores,
            processing_time=processing_times,
            algorithm_states=[],
            detected_changes=[1, 2],
        )

        # Check inherited methods
        assert len(trace) == 2
        assert str(trace) == "DetectionTrace(changes=2, with scores)"

    def test_multiple_detection_types(self) -> None:
        """Test trace with multiple detection types overlapping."""
        steps: list[OnlineDetectionStepResult] = [
            OnlineDetectionStepResult(
                step_num=i,
                is_change_point=(i == 2),
                is_force_change_point=(i == 2),
                is_in_skip_period=(i == 2),
                detection_function=1.0,
                processing_time=0.001,
                algorithm_state=None,
            )
            for i in range(5)
        ]

        trace = OnlineDetectionTrace.from_online_detection_steps(threshold=0.5, steps=steps)

        # Step 2 appears in all three lists
        assert trace.detected_changes == [2]
        assert trace.forced_change_points == [2]
        assert trace.skipped_observation == [2]

    def test_ndarray_dtype_preservation(self) -> None:
        """Test that NumPy arrays preserve float64 dtype."""
        steps: list[OnlineDetectionStepResult] = [
            OnlineDetectionStepResult(
                step_num=i,
                is_change_point=False,
                is_force_change_point=False,
                is_in_skip_period=False,
                detection_function=float(i),
                processing_time=float(i) * 0.001,
                algorithm_state=None,
            )
            for i in range(5)
        ]

        trace = OnlineDetectionTrace.from_online_detection_steps(threshold=0.5, steps=steps)

        assert trace.observation_scores.dtype == np.float64
        assert trace.processing_time.dtype == np.float64

    def test_skipped_observation_default_mutable(self) -> None:
        """Test that skipped_observation default is a new list each instance."""
        observation_scores: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.1, 0.2], dtype=np.float64)
        )
        processing_times: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.001, 0.002], dtype=np.float64)
        )

        trace1 = OnlineDetectionTrace(
            threshold=0.5,
            observation_scores=observation_scores,
            processing_time=processing_times,
            algorithm_states=[],
            detected_changes=[1],
        )

        trace2 = OnlineDetectionTrace(
            threshold=0.5,
            observation_scores=observation_scores,
            processing_time=processing_times,
            algorithm_states=[],
            detected_changes=[1],
        )

        # Modify one instance's skipped_observation
        trace1.skipped_observation.append(5)

        # Other instance should not be affected
        assert trace2.skipped_observation == []

    def test_forced_change_points_default_mutable(self) -> None:
        """Test that forced_change_points default is a new list each instance."""
        observation_scores: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.1, 0.2], dtype=np.float64)
        )
        processing_times: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.001, 0.002], dtype=np.float64)
        )

        trace1 = OnlineDetectionTrace(
            threshold=0.5,
            observation_scores=observation_scores,
            processing_time=processing_times,
            algorithm_states=[],
            detected_changes=[1],
        )

        trace2 = OnlineDetectionTrace(
            threshold=0.5,
            observation_scores=observation_scores,
            processing_time=processing_times,
            algorithm_states=[],
            detected_changes=[1],
        )

        # Modify one instance's forced_change_points
        trace1.forced_change_points.append(3)

        # Other instance should not be affected
        assert trace2.forced_change_points == []
