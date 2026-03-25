# -*- coding: ascii -*-

# -*- coding: ascii -*-
"""
Tests for DetectionTrace container class.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.core.detection_trace import DetectionTrace
from tests.mocks.core.data_providers import (
    MockEmptyDataProvider,
    MockSingleObservationProvider,
    MockUnivariateDataProvider,
)


class TestDetectionTrace:
    """Test suite for DetectionTrace class."""

    @pytest.fixture
    def mock_data(self) -> MockUnivariateDataProvider:
        """Fixture providing mock data provider."""
        return MockUnivariateDataProvider([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    def test_initialization_with_detected_changes(self, mock_data: MockUnivariateDataProvider) -> None:
        """Test initialization with detected_changes parameter."""
        trace = DetectionTrace(data=mock_data, detected_changes=[2, 5, 8])
        assert trace.detected_changes == [2, 5, 8]
        assert trace.data is mock_data

    def test_initialization_without_detected_changes(self, mock_data: MockUnivariateDataProvider) -> None:
        """Test initialization without change list."""
        trace = DetectionTrace(data=mock_data)
        assert trace.detected_changes == []
        assert trace.data is mock_data

    def test_initialization_with_empty_detected_changes(self, mock_data: MockUnivariateDataProvider) -> None:
        """Test initialization with empty change list."""
        trace = DetectionTrace(data=mock_data, detected_changes=[])
        assert trace.detected_changes == []

    def test_raises_value_error_for_negative_change_index(self, mock_data: MockUnivariateDataProvider) -> None:
        """Test ValueError raised when detected change index is negative."""
        with pytest.raises(ValueError, match="Detected change indices must be positive"):
            DetectionTrace(data=mock_data, detected_changes=[-1, 5, 10])

        with pytest.raises(ValueError, match="Detected change indices must be positive"):
            DetectionTrace(data=mock_data, detected_changes=[5, -2, 10])

    def test_raises_value_error_for_zero_change_index(self, mock_data: MockUnivariateDataProvider) -> None:
        """Test that zero is allowed as a change index."""
        with pytest.raises(ValueError, match="Detected change indices must be positive"):
            DetectionTrace(data=mock_data, detected_changes=[0, 5, 10])

    def test_raises_value_error_for_duplicate_indices(self, mock_data: MockUnivariateDataProvider) -> None:
        """Test ValueError raised when detected change indices are not unique."""
        with pytest.raises(ValueError, match="Detected change indices must be unique"):
            DetectionTrace(data=mock_data, detected_changes=[5, 5, 10])

        with pytest.raises(ValueError, match="Detected change indices must be unique"):
            DetectionTrace(data=mock_data, detected_changes=[2, 5, 5, 8])

        with pytest.raises(ValueError, match="Detected change indices must be unique"):
            DetectionTrace(data=mock_data, detected_changes=[5, 10, 10, 15])

    def test_raises_value_error_for_index_exceeds_data_length(self, mock_data: MockUnivariateDataProvider) -> None:
        """Test ValueError raised when change index exceeds data length."""
        # Data length is 10, indices 0-9 are valid
        with pytest.raises(ValueError, match="exceeds data length"):
            DetectionTrace(data=mock_data, detected_changes=[5, 10])

        with pytest.raises(ValueError, match="exceeds data length"):
            DetectionTrace(data=mock_data, detected_changes=[12])

        with pytest.raises(ValueError, match="exceeds data length"):
            DetectionTrace(data=mock_data, detected_changes=[1, 9, 10])

    def test_str(self, mock_data: MockUnivariateDataProvider) -> None:
        """Test string representation."""
        trace = DetectionTrace(data=mock_data, detected_changes=[1, 2, 3])
        assert str(trace) == "DetectionTrace(changes=3)"

    def test_str_with_empty_changes(self, mock_data: MockUnivariateDataProvider) -> None:
        """Test string representation with no detected changes."""
        trace = DetectionTrace(data=mock_data, detected_changes=[])
        assert str(trace) == "DetectionTrace(changes=0)"

    def test_multiple_instances_independence(self, mock_data: MockUnivariateDataProvider) -> None:
        """Test that instances are independent."""
        trace1 = DetectionTrace(data=mock_data)
        trace2 = DetectionTrace(data=mock_data, detected_changes=[1])

        assert trace1.detected_changes == []
        assert trace2.detected_changes == [1]
        assert trace1.data is mock_data
        assert trace2.data is mock_data

    def test_validation_with_empty_data(self) -> None:
        """Test validation with empty data provider."""
        empty_data = MockEmptyDataProvider[int]()

        # Empty data should allow empty change list
        trace = DetectionTrace(data=empty_data, detected_changes=[])
        assert trace.detected_changes == []

        # Should raise error if changes detected in empty data
        with pytest.raises(ValueError, match="exceeds data length"):
            DetectionTrace(data=empty_data, detected_changes=[1])

    def test_validation_with_single_observation_data(self) -> None:
        """Test validation with single observation data."""
        single_data = MockSingleObservationProvider(42)

        # Index 0 is valid for single observation
        trace = DetectionTrace(data=single_data, detected_changes=[])
        assert trace.detected_changes == []

        # Index 1 is invalid
        with pytest.raises(ValueError, match="exceeds data length"):
            DetectionTrace(data=single_data, detected_changes=[1])

    def test_multiple_warnings_combined(self) -> None:
        """Test that unsorted warning works with multiple indices."""
        data = MockUnivariateDataProvider([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

        with pytest.warns(UserWarning, match="Detected change indices are not sorted"):
            DetectionTrace(data=data, detected_changes=[5, 2, 4, 1, 3])


__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"
