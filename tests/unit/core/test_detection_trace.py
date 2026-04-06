# -*- coding: ascii -*-
"""
Tests for DetectionTrace container class.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.core.detection_trace import DetectionTrace


class TestDetectionTrace:
    """Test suite for DetectionTrace class."""

    def test_initialization_with_detected_changes(self) -> None:
        """Test initialization with detected_changes parameter."""
        trace = DetectionTrace(detected_change_points=[2, 5, 8])
        assert trace.detected_change_points == [2, 5, 8]

    def test_initialization_without_detected_changes(self) -> None:
        """Test initialization without change list."""
        trace = DetectionTrace()
        assert trace.detected_change_points == []

    def test_initialization_with_empty_detected_changes(self) -> None:
        """Test initialization with empty change list."""
        trace = DetectionTrace(detected_change_points=[])
        assert trace.detected_change_points == []

    def test_raises_value_error_for_negative_change_index(self) -> None:
        """Test ValueError raised when detected change index is negative."""
        with pytest.raises(ValueError, match="Detected change indices must be positive"):
            DetectionTrace(detected_change_points=[-1, 5, 10])

        with pytest.raises(ValueError, match="Detected change indices must be positive"):
            DetectionTrace(detected_change_points=[5, -2, 10])

    def test_raises_value_error_for_zero_change_index(self) -> None:
        """Test ValueError raised when change index is zero (must be positive)."""
        with pytest.raises(ValueError, match="Detected change indices must be positive"):
            DetectionTrace(detected_change_points=[0, 5, 10])

    def test_raises_value_error_for_duplicate_indices(self) -> None:
        """Test ValueError raised when detected change indices are not unique."""
        with pytest.raises(ValueError, match="Detected change indices must be unique"):
            DetectionTrace(detected_change_points=[5, 5, 10])

        with pytest.raises(ValueError, match="Detected change indices must be unique"):
            DetectionTrace(detected_change_points=[2, 5, 5, 8])

        with pytest.raises(ValueError, match="Detected change indices must be unique"):
            DetectionTrace(detected_change_points=[5, 10, 10, 15])

    def test_multiple_instances_independence(self) -> None:
        """Test that instances are independent."""
        trace1 = DetectionTrace()
        trace2 = DetectionTrace(detected_change_points=[1])

        assert trace1.detected_change_points == []
        assert trace2.detected_change_points == [1]

    def test_warns_for_unsorted_indices(self) -> None:
        """Test warning issued when detected change indices are not sorted."""
        with pytest.warns(UserWarning, match="Detected change indices are not sorted"):
            DetectionTrace(detected_change_points=[5, 2, 4, 1, 3])

    def test_validation_accepts_single_positive_index(self) -> None:
        """Test that single positive index is accepted."""
        trace = DetectionTrace(detected_change_points=[1])
        assert trace.detected_change_points == [1]

    def test_validation_accepts_positive_indices(self) -> None:
        """Test that positive indices are accepted."""
        trace = DetectionTrace(detected_change_points=[1, 2, 3, 4, 5])
        assert trace.detected_change_points == [1, 2, 3, 4, 5]
