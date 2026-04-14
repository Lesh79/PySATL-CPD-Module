# -*- coding: ascii -*-

"""
Unit tests for NoResetDetectionTrace[Any].
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

import pytest

from pysatl_cpd.benchmark.noreset.noreset_detection_trace import NoResetDetectionTrace
from pysatl_cpd.core.detection_trace import DetectionTrace
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace
from tests.mocks.core.online.online_detection_trace import MockOnlineDetectionTrace


@pytest.fixture
def source_trace() -> MockOnlineDetectionTrace:
    """
    Source OnlineDetectionTrace with real detection function values.
    """
    trace = MockOnlineDetectionTrace(detected_change_points=[5, 10])
    return trace


@pytest.fixture
def new_change_points() -> list[int]:
    """New detected change points to assign to NoResetDetectionTrace[Any]."""
    return [3, 7]


class TestNoResetDetectionTraceFromInfTrace:
    """Tests for NoResetDetectionTrace[Any].from_inf_trace factory method."""

    def test_detected_change_points_and_threshold_are_set(
        self,
        source_trace: MockOnlineDetectionTrace,
        new_change_points: list[int],
    ) -> None:
        """New detected_change_points and threshold are stored correctly."""
        threshold: float = 1.0
        trace: NoResetDetectionTrace[Any] = NoResetDetectionTrace.from_inf_trace(
            source_trace=source_trace,
            detected_change_points=new_change_points,
            threshold=threshold,
        )
        assert list(trace.detected_change_points) == new_change_points
        assert trace.threshold == threshold

    def test_algorithm_name_and_configuration_hash_are_copied(
        self,
        source_trace: MockOnlineDetectionTrace,
        new_change_points: list[int],
    ) -> None:
        """algorithm_name and configuration_hash are copied from source_trace."""
        trace: NoResetDetectionTrace[Any] = NoResetDetectionTrace.from_inf_trace(
            source_trace=source_trace,
            detected_change_points=new_change_points,
            threshold=1.0,
        )
        assert trace.algorithm_name == source_trace.algorithm_name
        assert trace.configuration_hash == source_trace.configuration_hash

    def test_auxiliary_fields_are_empty(
        self,
        source_trace: MockOnlineDetectionTrace,
        new_change_points: list[int],
    ) -> None:
        """processing_time, detection_function, algorithm_states, skip_periods,
        learning_periods, forced_change_points, signal_change_points are empty/default."""
        trace: NoResetDetectionTrace[Any] = NoResetDetectionTrace.from_inf_trace(
            source_trace=source_trace,
            detected_change_points=new_change_points,
            threshold=1.0,
        )
        assert len(trace.processing_time) == 0
        assert len(trace.detection_function) == 0
        assert trace.algorithm_states == []
        assert trace.skip_periods == []
        assert trace.learning_periods == []
        assert trace.forced_change_points == []
        assert trace.signal_change_points == []

    def test_source_trace_is_not_mutated(
        self,
        source_trace: MockOnlineDetectionTrace,
        new_change_points: list[int],
    ) -> None:
        """source_trace fields are not modified after from_inf_trace call."""
        original_cps: list[int] = list(source_trace.detected_change_points)
        original_name: str = source_trace.algorithm_name
        original_hash: int = source_trace.configuration_hash

        NoResetDetectionTrace[Any].from_inf_trace(
            source_trace=source_trace,
            detected_change_points=new_change_points,
            threshold=1.0,
        )

        assert list(source_trace.detected_change_points) == original_cps
        assert source_trace.algorithm_name == original_name
        assert source_trace.configuration_hash == original_hash

    def test_with_empty_detected_change_points(
        self,
        source_trace: MockOnlineDetectionTrace,
    ) -> None:
        """from_inf_trace works correctly when detected_change_points is empty."""
        trace: NoResetDetectionTrace[Any] = NoResetDetectionTrace.from_inf_trace(
            source_trace=source_trace,
            detected_change_points=[],
            threshold=1.0,
        )
        assert list(trace.detected_change_points) == []

    def test_with_boundary_threshold_values(
        self,
        source_trace: MockOnlineDetectionTrace,
        new_change_points: list[int],
    ) -> None:
        """from_inf_trace works correctly with threshold=0.0 and threshold=inf."""
        trace_zero: NoResetDetectionTrace[Any] = NoResetDetectionTrace.from_inf_trace(
            source_trace=source_trace,
            detected_change_points=new_change_points,
            threshold=0.0,
        )
        assert trace_zero.threshold == 0.0

        trace_inf: NoResetDetectionTrace[Any] = NoResetDetectionTrace.from_inf_trace(
            source_trace=source_trace,
            detected_change_points=new_change_points,
            threshold=float("inf"),
        )
        assert trace_inf.threshold == float("inf")


class TestNoResetDetectionTraceInheritance:
    """Tests for NoResetDetectionTrace[Any] inheritance chain."""

    def test_is_instance_of_expected_base_classes(
        self,
        source_trace: MockOnlineDetectionTrace,
        new_change_points: list[int],
    ) -> None:
        """NoResetDetectionTrace[Any] is an instance of OnlineDetectionTrace and DetectionTrace."""
        trace: NoResetDetectionTrace[Any] = NoResetDetectionTrace.from_inf_trace(
            source_trace=source_trace,
            detected_change_points=new_change_points,
            threshold=1.0,
        )
        assert isinstance(trace, NoResetDetectionTrace)
        assert isinstance(trace, OnlineDetectionTrace)
        assert isinstance(trace, DetectionTrace)

    def test_detected_change_points_accessible_via_base_property(
        self,
        source_trace: MockOnlineDetectionTrace,
        new_change_points: list[int],
    ) -> None:
        """detected_change_points are accessible through the base class property."""
        trace: NoResetDetectionTrace[Any] = NoResetDetectionTrace.from_inf_trace(
            source_trace=source_trace,
            detected_change_points=new_change_points,
            threshold=1.0,
        )
        base: DetectionTrace = trace
        assert list(base.detected_change_points) == new_change_points
