# -*- coding: ascii -*-

"""
Tests for online detection trace containers.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from typing import cast

import numpy as np
import pytest

from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmState
from pysatl_cpd.core.online.online_detection_trace import (
    OnlineDetectionStepResult,
    OnlineDetectionTrace,
    extract_periods,
)
from pysatl_cpd.core.typedefs import UnivariateNumericArray
from tests.mocks.algorithms.online import MockAlgorithmState


class TestExtractPeriods:
    """Test suite for extract_periods helper function."""

    def test_empty_sequence(self) -> None:
        """Test with empty sequence."""
        result = extract_periods([])
        assert result == []

    def test_no_periods(self) -> None:
        """Test with all False values."""
        result = extract_periods([False, False, False])
        assert result == []

    def test_all_none(self) -> None:
        """Test with all None values."""
        result = extract_periods([None, None, None])
        assert result == []

    def test_single_period(self) -> None:
        """Test with single continuous period."""
        result = extract_periods([True, True, True])
        assert result == [(0, 2)]

    def test_multiple_periods(self) -> None:
        """Test with multiple separate periods."""
        result = extract_periods([True, True, False, True, True])
        assert result == [(0, 1), (3, 4)]

    def test_period_at_beginning(self) -> None:
        """Test with period starting at beginning."""
        result = extract_periods([True, True, False, False])
        assert result == [(0, 1)]

    def test_period_at_end(self) -> None:
        """Test with period ending at end."""
        result = extract_periods([False, False, True, True])
        assert result == [(2, 3)]

    def test_single_true(self) -> None:
        """Test with single True value."""
        result = extract_periods([False, True, False])
        assert result == [(1, 1)]

    def test_with_none_values(self) -> None:
        """Test with None values (should be ignored)."""
        result = extract_periods([None, True, True, None, False, True])
        assert result == [(1, 2), (5, 5)]

    def test_alternating_values(self) -> None:
        """Test with alternating True/False values."""
        result = extract_periods([True, False, True, False, True])
        assert result == [(0, 0), (2, 2), (4, 4)]

    def test_single_false(self) -> None:
        """Test with all False returns empty."""
        result = extract_periods([False, False, False])
        assert result == []

    def test_mixed_with_none_in_period(self) -> None:
        """Test with None inside period (should break period)."""
        result = extract_periods([True, None, True])
        assert result == [(0, 0), (2, 2)]


class TestOnlineDetectionStepResult:
    """Test suite for OnlineDetectionStepResult."""

    def test_default_values(self) -> None:
        """Test default values for all fields."""
        result: OnlineDetectionStepResult[OnlineAlgorithmState] = OnlineDetectionStepResult[OnlineAlgorithmState]()

        assert result.step_num == 0
        assert result.is_forced_change_point is False
        assert result.is_signal_change_point is False
        assert result.is_change_point is False
        assert result.is_in_skip_period is False
        assert np.isnan(result.detection_function)
        assert np.isnan(result.processing_time)
        assert result.algorithm_state is None

    def test_custom_values(self) -> None:
        """Test setting custom values during initialization."""
        state: MockAlgorithmState[float] = MockAlgorithmState[float](
            process_count=1,
            last_observation=5,
        )
        result: OnlineDetectionStepResult[MockAlgorithmState[float]] = OnlineDetectionStepResult(
            step_num=5,
            is_forced_change_point=False,
            is_signal_change_point=True,
            is_in_skip_period=False,
            detection_function=0.85,
            processing_time=0.00123,
            algorithm_state=state,
        )

        assert result.step_num == 5
        assert result.is_forced_change_point is False
        assert result.is_signal_change_point is True
        assert result.is_change_point is True
        assert result.is_in_skip_period is False
        assert result.detection_function == 0.85
        assert result.processing_time == 0.00123
        assert result.algorithm_state is state

    def test_is_change_point_property(self) -> None:
        """Test that is_change_point returns OR of forced and signal."""
        # Both false
        result = OnlineDetectionStepResult[MockAlgorithmState[float]](
            is_forced_change_point=False,
            is_signal_change_point=False,
        )
        assert result.is_change_point is False

        # Only forced
        result = OnlineDetectionStepResult(
            is_forced_change_point=True,
            is_signal_change_point=False,
        )
        assert result.is_change_point is True

        # Only signal
        result = OnlineDetectionStepResult(
            is_forced_change_point=False,
            is_signal_change_point=True,
        )
        assert result.is_change_point is True

        # Both true
        result = OnlineDetectionStepResult(
            is_forced_change_point=True,
            is_signal_change_point=True,
        )
        assert result.is_change_point is True


class TestOnlineDetectionTrace:
    """Test suite for OnlineDetectionTrace."""

    @pytest.fixture
    def sample_steps(self) -> list[OnlineDetectionStepResult[MockAlgorithmState[float]]]:
        """Create sample step results for testing."""
        state1 = MockAlgorithmState[float](process_count=1, last_observation=1.0)
        state2 = MockAlgorithmState[float](process_count=2, last_observation=2.0)

        return [
            OnlineDetectionStepResult(
                step_num=0,
                is_signal_change_point=False,
                is_forced_change_point=False,
                is_in_skip_period=False,
                detection_function=0.1,
                processing_time=0.001,
                algorithm_state=state1,
            ),
            OnlineDetectionStepResult(
                step_num=1,
                is_signal_change_point=True,
                is_forced_change_point=False,
                is_in_skip_period=False,
                detection_function=0.9,
                processing_time=0.002,
                algorithm_state=state2,
            ),
            OnlineDetectionStepResult(
                step_num=2,
                is_signal_change_point=False,
                is_forced_change_point=False,
                is_in_skip_period=True,
                detection_function=0.0,
                processing_time=0.0,
                algorithm_state=None,
            ),
            OnlineDetectionStepResult(
                step_num=3,
                is_signal_change_point=False,
                is_forced_change_point=True,
                is_in_skip_period=False,
                detection_function=1.2,
                processing_time=0.003,
                algorithm_state=None,
            ),
        ]

    def test_from_run(self, sample_steps: list[OnlineDetectionStepResult[MockAlgorithmState[float]]]) -> None:
        """Test constructing OnlineDetectionTrace from run results."""
        trace: OnlineDetectionTrace[MockAlgorithmState[float]] = OnlineDetectionTrace.from_run(
            steps=sample_steps, threshold=0.5
        )

        assert trace.threshold == 0.5
        assert isinstance(trace.detection_function, np.ndarray)
        assert trace.detection_function.ndim == 1
        assert np.array_equal(trace.detection_function, np.array([0.1, 0.9, 0.0, 1.2]))
        assert isinstance(trace.processing_time, np.ndarray)
        assert trace.processing_time.ndim == 1
        assert np.array_equal(trace.processing_time, np.array([0.001, 0.002, 0.0, 0.003]))
        assert trace.detected_change_points == [1, 3]
        assert trace.forced_change_points == [3]
        assert trace.signal_change_points == [1]
        assert trace.skip_periods == [(2, 2)]
        assert isinstance(trace.algorithm_states, list)
        assert len(trace.algorithm_states) == 4
        assert trace.algorithm_states[0] is not None
        assert trace.algorithm_states[1] is not None
        assert trace.algorithm_states[2] is None
        assert trace.algorithm_states[3] is None

    def test_from_run_with_none_threshold(
        self, sample_steps: list[OnlineDetectionStepResult[MockAlgorithmState[float]]]
    ) -> None:
        """Test constructing trace with None threshold."""
        trace: OnlineDetectionTrace[MockAlgorithmState[float]] = OnlineDetectionTrace.from_run(
            steps=sample_steps, threshold=None
        )

        assert trace.threshold is None
        assert len(trace.detection_function) == 4

    def test_from_run_empty(self) -> None:
        """Test constructing trace from empty step sequence."""
        steps: list[OnlineDetectionStepResult[MockAlgorithmState[float]]] = []
        trace: OnlineDetectionTrace[MockAlgorithmState[float]] = OnlineDetectionTrace.from_run(
            steps=steps, threshold=0.5
        )

        assert trace.threshold == 0.5
        assert isinstance(trace.detection_function, np.ndarray)
        assert len(trace.detection_function) == 0
        assert isinstance(trace.processing_time, np.ndarray)
        assert len(trace.processing_time) == 0
        assert trace.detected_change_points == []
        assert trace.forced_change_points == []
        assert trace.signal_change_points == []
        assert trace.skip_periods == []
        assert trace.learning_periods == []
        assert trace.algorithm_states == []

    def test_from_run_no_detections(self) -> None:
        """Test constructing trace with no changepoints."""
        steps: list[OnlineDetectionStepResult[MockAlgorithmState[float]]] = [
            OnlineDetectionStepResult(
                step_num=i,
                is_signal_change_point=False,
                is_forced_change_point=False,
                is_in_skip_period=False,
                detection_function=0.1,
                processing_time=0.001,
                algorithm_state=None,
            )
            for i in range(5)
        ]

        trace: OnlineDetectionTrace[MockAlgorithmState[float]] = OnlineDetectionTrace.from_run(
            steps=steps, threshold=0.5
        )

        assert trace.detected_change_points == []
        assert trace.forced_change_points == []
        assert trace.signal_change_points == []
        assert trace.skip_periods == []

    def test_direct_initialization(self) -> None:
        """Test direct initialization of OnlineDetectionTrace."""
        states: list[MockAlgorithmState[float] | None] = [
            MockAlgorithmState[float](process_count=1, last_observation=1.0),
            MockAlgorithmState[float](process_count=2, last_observation=2.0),
            MockAlgorithmState[float](process_count=3, last_observation=3.0),
        ]

        detection_function: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.2, 0.5, 0.8], dtype=np.float64)
        )
        processing_times: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.001, 0.002, 0.003], dtype=np.float64)
        )

        trace: OnlineDetectionTrace[MockAlgorithmState[float]] = OnlineDetectionTrace(
            detected_change_points=[2],
            threshold=0.75,
            detection_function=detection_function,
            processing_time=processing_times,
            algorithm_states=states,
            forced_change_points=[],
            signal_change_points=[2],
            skip_periods=[(1, 1)],
            learning_periods=[(0, 0)],
        )

        assert trace.threshold == 0.75
        assert isinstance(trace.detection_function, np.ndarray)
        assert len(trace.detection_function) == 3
        assert isinstance(trace.processing_time, np.ndarray)
        assert len(trace.processing_time) == 3
        assert trace.detected_change_points == [2]
        assert trace.forced_change_points == []
        assert trace.signal_change_points == [2]
        assert trace.skip_periods == [(1, 1)]
        assert trace.learning_periods == [(0, 0)]

    def test_inherits_from_detection_trace(self) -> None:
        """Test that OnlineDetectionTrace inherits DetectionTrace functionality."""
        detection_function: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.1, 0.2, 0.3], dtype=np.float64)
        )
        processing_times: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.001, 0.002, 0.003], dtype=np.float64)
        )

        trace: OnlineDetectionTrace[MockAlgorithmState[float]] = OnlineDetectionTrace(
            detected_change_points=[1, 2],
            threshold=0.5,
            detection_function=detection_function,
            processing_time=processing_times,
            algorithm_states=[],
        )

        # Check that detected_change_points are accessible
        assert trace.detected_change_points == [1, 2]

    def test_detected_change_points_property(self) -> None:
        """Test detected_change_points property."""
        detection_function: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.1, 0.2, 0.3], dtype=np.float64)
        )
        processing_times: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.001, 0.002, 0.003], dtype=np.float64)
        )

        trace: OnlineDetectionTrace[MockAlgorithmState[float]] = OnlineDetectionTrace(
            detected_change_points=[1, 2, 3],
            threshold=0.5,
            detection_function=detection_function,
            processing_time=processing_times,
            algorithm_states=[],
        )

        assert trace.detected_change_points == [1, 2, 3]

    def test_multiple_detection_types(
        self, sample_steps: list[OnlineDetectionStepResult[MockAlgorithmState[float]]]
    ) -> None:
        """Test trace with multiple detection types."""
        trace: OnlineDetectionTrace[MockAlgorithmState[float]] = OnlineDetectionTrace.from_run(
            steps=sample_steps, threshold=0.5
        )

        assert trace.detected_change_points == [1, 3]
        assert trace.forced_change_points == [3]
        assert trace.signal_change_points == [1]
        assert trace.skip_periods == [(2, 2)]

    def test_ndarray_dtype_preservation(self) -> None:
        """Test that NumPy arrays preserve float64 dtype."""
        steps: list[OnlineDetectionStepResult[MockAlgorithmState[float]]] = [
            OnlineDetectionStepResult(
                step_num=i,
                is_signal_change_point=False,
                is_forced_change_point=False,
                is_in_skip_period=False,
                detection_function=float(i),
                processing_time=float(i) * 0.001,
                algorithm_state=None,
            )
            for i in range(5)
        ]

        trace: OnlineDetectionTrace[MockAlgorithmState[float]] = OnlineDetectionTrace.from_run(
            steps=steps, threshold=0.5
        )

        assert trace.detection_function.dtype == np.float64
        assert trace.processing_time.dtype == np.float64

    def test_skip_periods_default_mutable(self) -> None:
        """Test that skip_periods default is a new list each instance."""
        detection_function: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.1, 0.2], dtype=np.float64)
        )
        processing_times: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.001, 0.002], dtype=np.float64)
        )

        trace1: OnlineDetectionTrace[MockAlgorithmState[float]] = OnlineDetectionTrace(
            detected_change_points=[1],
            threshold=0.5,
            detection_function=detection_function,
            processing_time=processing_times,
            algorithm_states=[],
        )

        trace2: OnlineDetectionTrace[MockAlgorithmState[float]] = OnlineDetectionTrace(
            detected_change_points=[1],
            threshold=0.5,
            detection_function=detection_function,
            processing_time=processing_times,
            algorithm_states=[],
        )

        trace1.skip_periods.append((2, 3))

        assert trace2.skip_periods == []

    def test_learning_periods_default_mutable(self) -> None:
        """Test that learning_periods default is a new list each instance."""
        detection_function: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.1, 0.2], dtype=np.float64)
        )
        processing_times: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.001, 0.002], dtype=np.float64)
        )

        trace1: OnlineDetectionTrace[MockAlgorithmState[float]] = OnlineDetectionTrace(
            detected_change_points=[1],
            threshold=0.5,
            detection_function=detection_function,
            processing_time=processing_times,
            algorithm_states=[],
        )

        trace2: OnlineDetectionTrace[MockAlgorithmState[float]] = OnlineDetectionTrace(
            detected_change_points=[1],
            threshold=0.5,
            detection_function=detection_function,
            processing_time=processing_times,
            algorithm_states=[],
        )

        trace1.learning_periods.append((0, 1))

        assert trace2.learning_periods == []

    def test_forced_change_points_default_mutable(self) -> None:
        """Test that forced_change_points default is a new list each instance."""
        detection_function: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.1, 0.2], dtype=np.float64)
        )
        processing_times: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.001, 0.002], dtype=np.float64)
        )

        trace1: OnlineDetectionTrace[MockAlgorithmState[float]] = OnlineDetectionTrace(
            detected_change_points=[1],
            threshold=0.5,
            detection_function=detection_function,
            processing_time=processing_times,
            algorithm_states=[],
        )

        trace2: OnlineDetectionTrace[MockAlgorithmState[float]] = OnlineDetectionTrace(
            detected_change_points=[1],
            threshold=0.5,
            detection_function=detection_function,
            processing_time=processing_times,
            algorithm_states=[],
        )

        trace1.forced_change_points.append(3)

        assert trace2.forced_change_points == []

    def test_signal_change_points_default_mutable(self) -> None:
        """Test that signal_change_points default is a new list each instance."""
        detection_function: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.1, 0.2], dtype=np.float64)
        )
        processing_times: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.001, 0.002], dtype=np.float64)
        )

        trace1: OnlineDetectionTrace[MockAlgorithmState[float]] = OnlineDetectionTrace(
            detected_change_points=[1],
            threshold=0.5,
            detection_function=detection_function,
            processing_time=processing_times,
            algorithm_states=[],
        )

        trace2: OnlineDetectionTrace[MockAlgorithmState[float]] = OnlineDetectionTrace(
            detected_change_points=[1],
            threshold=0.5,
            detection_function=detection_function,
            processing_time=processing_times,
            algorithm_states=[],
        )

        trace1.signal_change_points.append(2)

        assert trace2.signal_change_points == []
