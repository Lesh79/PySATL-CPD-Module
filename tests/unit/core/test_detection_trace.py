# -*- coding: ascii -*-
"""
Tests for DetectionTrace container class.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.core.detection_trace import DetectionTrace
from pysatl_cpd.core.typedefs import Number
from tests.mocks.algorithms.online import MockOnlineAlgorithm


@pytest.fixture
def mock_algorithm() -> MockOnlineAlgorithm[Number]:
    """Create a mock algorithm instance."""
    return MockOnlineAlgorithm[Number]()


@pytest.fixture
def empty_trace(mock_algorithm: MockOnlineAlgorithm[Number]) -> DetectionTrace:
    """Create empty DetectionTrace instance."""
    return DetectionTrace(
        algorithm_name=mock_algorithm.name,
        configuration_hash=hash(mock_algorithm.configuration),
    )


@pytest.fixture
def trace_with_changes(mock_algorithm: MockOnlineAlgorithm[Number]) -> DetectionTrace:
    """Create DetectionTrace with predefined change points."""
    return DetectionTrace(
        detected_change_points=[2, 5, 8],
        algorithm_name=mock_algorithm.name,
        configuration_hash=hash(mock_algorithm.configuration),
    )


class TestDetectionTrace:
    """Test suite for DetectionTrace class."""

    def test_initialization_with_detected_changes(self, trace_with_changes: DetectionTrace) -> None:
        """Test initialization with detected_changes parameter."""
        assert trace_with_changes.detected_change_points == [2, 5, 8]

    def test_initialization_without_detected_changes(self, empty_trace: DetectionTrace) -> None:
        """Test initialization without change list."""
        assert empty_trace.detected_change_points == []

    def test_initialization_with_empty_detected_changes(self, mock_algorithm: MockOnlineAlgorithm[Number]) -> None:
        """Test initialization with empty change list."""
        trace = DetectionTrace(
            detected_change_points=[],
            algorithm_name=mock_algorithm.name,
            configuration_hash=hash(mock_algorithm.configuration),
        )
        assert trace.detected_change_points == []

    def test_raises_value_error_for_negative_change_index(self, mock_algorithm: MockOnlineAlgorithm[Number]) -> None:
        """Test ValueError raised when detected change index is negative."""
        with pytest.raises(ValueError, match="Detected change indices must be positive"):
            DetectionTrace(
                detected_change_points=[-1, 5, 10],
                algorithm_name=mock_algorithm.name,
                configuration_hash=hash(mock_algorithm.configuration),
            )

        with pytest.raises(ValueError, match="Detected change indices must be positive"):
            DetectionTrace(
                detected_change_points=[5, -2, 10],
                algorithm_name=mock_algorithm.name,
                configuration_hash=hash(mock_algorithm.configuration),
            )

    def test_raises_value_error_for_zero_change_index(self, mock_algorithm: MockOnlineAlgorithm[Number]) -> None:
        """Test ValueError raised when change index is zero (must be positive)."""
        with pytest.raises(ValueError, match="Detected change indices must be positive"):
            DetectionTrace(
                detected_change_points=[0, 5, 10],
                algorithm_name=mock_algorithm.name,
                configuration_hash=hash(mock_algorithm.configuration),
            )

    def test_raises_value_error_for_duplicate_indices(self, mock_algorithm: MockOnlineAlgorithm[Number]) -> None:
        """Test ValueError raised when detected change indices are not unique."""
        with pytest.raises(ValueError, match="Detected change indices must be unique"):
            DetectionTrace(
                detected_change_points=[5, 5, 10],
                algorithm_name=mock_algorithm.name,
                configuration_hash=hash(mock_algorithm.configuration),
            )

        with pytest.raises(ValueError, match="Detected change indices must be unique"):
            DetectionTrace(
                detected_change_points=[2, 5, 5, 8],
                algorithm_name=mock_algorithm.name,
                configuration_hash=hash(mock_algorithm.configuration),
            )

        with pytest.raises(ValueError, match="Detected change indices must be unique"):
            DetectionTrace(
                detected_change_points=[5, 10, 10, 15],
                algorithm_name=mock_algorithm.name,
                configuration_hash=hash(mock_algorithm.configuration),
            )

    def test_multiple_instances_independence(
        self,
        empty_trace: DetectionTrace,
        trace_with_changes: DetectionTrace,
    ) -> None:
        """Test that instances are independent."""
        assert empty_trace.detected_change_points == []
        assert trace_with_changes.detected_change_points == [2, 5, 8]

    def test_warns_for_unsorted_indices(self, mock_algorithm: MockOnlineAlgorithm[Number]) -> None:
        """Test warning issued when detected change indices are not sorted."""
        with pytest.warns(UserWarning, match="Detected change indices are not sorted"):
            DetectionTrace(
                detected_change_points=[5, 2, 4, 1, 3],
                algorithm_name=mock_algorithm.name,
                configuration_hash=hash(mock_algorithm.configuration),
            )

    def test_validation_accepts_single_positive_index(self, mock_algorithm: MockOnlineAlgorithm[Number]) -> None:
        """Test that single positive index is accepted."""
        trace = DetectionTrace(
            detected_change_points=[1],
            algorithm_name=mock_algorithm.name,
            configuration_hash=hash(mock_algorithm.configuration),
        )
        assert trace.detected_change_points == [1]

    def test_validation_accepts_positive_indices(self, mock_algorithm: MockOnlineAlgorithm[Number]) -> None:
        """Test that positive indices are accepted."""
        trace = DetectionTrace(
            detected_change_points=[1, 2, 3, 4, 5],
            algorithm_name=mock_algorithm.name,
            configuration_hash=hash(mock_algorithm.configuration),
        )
        assert trace.detected_change_points == [1, 2, 3, 4, 5]
