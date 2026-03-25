# -*- coding: ascii -*-

"""
Tests for online detection trace containers.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from typing import cast

import numpy as np

from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithmState
from pysatl_cpd.core.online.online_detection_trace import (
    OnlineDetectionStepResult,
    OnlineDetectionTrace,
)
from pysatl_cpd.core.typedefs import UnivariateNumericArray
from tests.mocks.algorithms.online import MockAlgorithmState
from tests.mocks.core.data_providers import MockUnivariateDataProvider


class TestOnlineDetectionStepResult:
    """Test suite for OnlineDetectionStepResult."""

    def test_default_values(self) -> None:
        """Test default values for all fields."""
        result: OnlineDetectionStepResult[OnlineAlgorithmState] = OnlineDetectionStepResult[OnlineAlgorithmState]()

        assert result.step_num == 0
        assert result.is_change_point is False
        assert result.is_force_change_point is False
        assert result.is_in_skip_period is False
        assert np.isnan(result.detection_function)
        assert np.isnan(result.processing_time)
        assert result.algorithm_state is None

    def test_custom_values(self) -> None:
        """Test setting custom values during initialization."""
        state: MockAlgorithmState[int] = MockAlgorithmState[int](
            process_count=1,
            last_observation=5,
        )
        result: OnlineDetectionStepResult[MockAlgorithmState[int]] = OnlineDetectionStepResult(
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

    def test_from_run(
        self,
        sample_data: MockUnivariateDataProvider,
        sample_steps: list[OnlineDetectionStepResult[MockAlgorithmState[int]]],
    ) -> None:
        """Test constructing OnlineDetectionTrace from run results."""
        trace: OnlineDetectionTrace[MockUnivariateDataProvider, MockAlgorithmState[int]] = (
            OnlineDetectionTrace.from_run(data=sample_data, steps=sample_steps, threshold=0.5)
        )

        assert trace.data == sample_data
        assert trace.threshold == 0.5
        assert isinstance(trace.detection_function, np.ndarray)
        assert trace.detection_function.ndim == 1
        assert np.array_equal(trace.detection_function, np.array([0.1, 0.9, 0.0, 1.2]))
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

    def test_from_run_with_none_threshold(
        self,
        sample_data: MockUnivariateDataProvider,
        sample_steps: list[OnlineDetectionStepResult[MockAlgorithmState[int]]],
    ) -> None:
        """Test constructing trace with None threshold."""
        trace: OnlineDetectionTrace[MockUnivariateDataProvider, MockAlgorithmState[int]] = (
            OnlineDetectionTrace.from_run(data=sample_data, steps=sample_steps, threshold=None)
        )

        assert trace.threshold is None
        assert len(trace.detection_function) == 4

    def test_from_run_empty(
        self,
        sample_data: MockUnivariateDataProvider,
    ) -> None:
        """Test constructing trace from empty step sequence."""
        steps: list[OnlineDetectionStepResult[MockAlgorithmState[int]]] = []
        trace: OnlineDetectionTrace[MockUnivariateDataProvider, MockAlgorithmState[int]] = (
            OnlineDetectionTrace.from_run(data=sample_data, steps=steps, threshold=0.5)
        )

        assert trace.threshold == 0.5
        assert isinstance(trace.detection_function, np.ndarray)
        assert len(trace.detection_function) == 0
        assert isinstance(trace.processing_time, np.ndarray)
        assert len(trace.processing_time) == 0
        assert trace.detected_changes == []
        assert trace.skipped_observation == []
        assert trace.forced_change_points == []
        assert trace.algorithm_states == []

    def test_from_run_no_detections(
        self,
        sample_data: MockUnivariateDataProvider,
    ) -> None:
        """Test constructing trace with no changepoints."""
        steps: list[OnlineDetectionStepResult[MockAlgorithmState[int]]] = [
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

        trace: OnlineDetectionTrace[MockUnivariateDataProvider, MockAlgorithmState[int]] = (
            OnlineDetectionTrace.from_run(data=sample_data, steps=steps, threshold=0.5)
        )

        assert trace.detected_changes == []
        assert trace.forced_change_points == []
        assert trace.skipped_observation == []

    def test_direct_initialization(self) -> None:
        """Test direct initialization of OnlineDetectionTrace."""
        states: list[MockAlgorithmState[float] | None] = [
            MockAlgorithmState[float](process_count=1, last_observation=1.0),
            MockAlgorithmState[float](process_count=2, last_observation=2.0),
            MockAlgorithmState[float](process_count=3, last_observation=3.0),
        ]
        data: MockUnivariateDataProvider = MockUnivariateDataProvider([1.0, 2.0, 3.0])

        detection_function: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.2, 0.5, 0.8], dtype=np.float64)
        )
        processing_times: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.001, 0.002, 0.003], dtype=np.float64)
        )

        trace: OnlineDetectionTrace[MockUnivariateDataProvider, MockAlgorithmState[float]] = OnlineDetectionTrace(
            data=data,
            threshold=0.75,
            detection_function=detection_function,
            processing_time=processing_times,
            algorithm_states=states,
            detected_changes=[2],
            skipped_observation=[1],
            forced_change_points=[],
        )

        assert trace.data == data
        assert trace.threshold == 0.75
        assert isinstance(trace.detection_function, np.ndarray)
        assert len(trace.detection_function) == 3
        assert isinstance(trace.processing_time, np.ndarray)
        assert len(trace.processing_time) == 3
        assert trace.detected_changes == [2]
        assert trace.skipped_observation == [1]
        assert trace.forced_change_points == []

    def test_inherits_from_detection_trace(
        self,
        sample_data: MockUnivariateDataProvider,
    ) -> None:
        """Test that OnlineDetectionTrace inherits DetectionTrace functionality."""
        detection_function: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.1, 0.2, 0.3], dtype=np.float64)
        )
        processing_times: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.001, 0.002, 0.003], dtype=np.float64)
        )

        trace: OnlineDetectionTrace[MockUnivariateDataProvider, MockAlgorithmState[int]] = OnlineDetectionTrace(
            data=sample_data,
            threshold=0.5,
            detection_function=detection_function,
            processing_time=processing_times,
            algorithm_states=[],
            detected_changes=[1, 2],
        )

        # Check inherited method
        assert str(trace) == "DetectionTrace(changes=2)"

    def test_str_representation_format(
        self,
        sample_data: MockUnivariateDataProvider,
    ) -> None:
        """Test __str__ method returns correct format."""
        detection_function: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.1, 0.2, 0.3], dtype=np.float64)
        )
        processing_times: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.001, 0.002, 0.003], dtype=np.float64)
        )

        trace: OnlineDetectionTrace[MockUnivariateDataProvider, MockAlgorithmState[int]] = OnlineDetectionTrace(
            data=sample_data,
            threshold=0.5,
            detection_function=detection_function,
            processing_time=processing_times,
            algorithm_states=[],
            detected_changes=[1, 2, 3],
        )

        assert str(trace) == "DetectionTrace(changes=3)"

    def test_multiple_detection_types(
        self,
        sample_data: MockUnivariateDataProvider,
        sample_steps: list[OnlineDetectionStepResult[MockAlgorithmState[int]]],
    ) -> None:
        """Test trace with multiple detection types overlapping."""
        trace: OnlineDetectionTrace[MockUnivariateDataProvider, MockAlgorithmState[int]] = (
            OnlineDetectionTrace.from_run(data=sample_data, steps=sample_steps, threshold=0.5)
        )

        # Step 1 is change point, step 2 is skip, step 3 is forced
        assert trace.detected_changes == [1]
        assert trace.forced_change_points == [3]
        assert trace.skipped_observation == [2]

    def test_ndarray_dtype_preservation(
        self,
        sample_data: MockUnivariateDataProvider,
    ) -> None:
        """Test that NumPy arrays preserve float64 dtype."""
        steps: list[OnlineDetectionStepResult[MockAlgorithmState[int]]] = [
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

        trace: OnlineDetectionTrace[MockUnivariateDataProvider, MockAlgorithmState[int]] = (
            OnlineDetectionTrace.from_run(data=sample_data, steps=steps, threshold=0.5)
        )

        assert trace.detection_function.dtype == np.float64
        assert trace.processing_time.dtype == np.float64

    def test_skipped_observation_default_mutable(self) -> None:
        """Test that skipped_observation default is a new list each instance."""
        detection_function: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.1, 0.2], dtype=np.float64)
        )
        processing_times: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.001, 0.002], dtype=np.float64)
        )
        data: MockUnivariateDataProvider = MockUnivariateDataProvider([1.0, 2.0])

        trace1: OnlineDetectionTrace[MockUnivariateDataProvider, MockAlgorithmState[int]] = OnlineDetectionTrace(
            data=data,
            threshold=0.5,
            detection_function=detection_function,
            processing_time=processing_times,
            algorithm_states=[],
            detected_changes=[1],
        )

        trace2: OnlineDetectionTrace[MockUnivariateDataProvider, MockAlgorithmState[int]] = OnlineDetectionTrace(
            data=data,
            threshold=0.5,
            detection_function=detection_function,
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
        detection_function: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.1, 0.2], dtype=np.float64)
        )
        processing_times: UnivariateNumericArray = cast(
            UnivariateNumericArray, np.array([0.001, 0.002], dtype=np.float64)
        )
        data: MockUnivariateDataProvider = MockUnivariateDataProvider([1.0, 2.0])

        trace1: OnlineDetectionTrace[MockUnivariateDataProvider, MockAlgorithmState[int]] = OnlineDetectionTrace(
            data=data,
            threshold=0.5,
            detection_function=detection_function,
            processing_time=processing_times,
            algorithm_states=[],
            detected_changes=[1],
        )

        trace2: OnlineDetectionTrace[MockUnivariateDataProvider, MockAlgorithmState[int]] = OnlineDetectionTrace(
            data=data,
            threshold=0.5,
            detection_function=detection_function,
            processing_time=processing_times,
            algorithm_states=[],
            detected_changes=[1],
        )

        # Modify one instance's forced_change_points
        trace1.forced_change_points.append(3)

        # Other instance should not be affected
        assert trace2.forced_change_points == []
