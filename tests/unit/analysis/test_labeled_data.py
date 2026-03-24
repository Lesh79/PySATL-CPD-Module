"""
Tests for LabeledData container class.
"""

import numpy as np
import pytest

from pysatl_cpd.analysis.labeled_data import LabeledData


class TestLabeledData:
    """Test suite for LabeledData class."""

    @pytest.fixture
    def sample_univariate_data(self):
        """Fixture providing sample univariate time series data."""
        return [1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 20.0]

    @pytest.fixture
    def sample_multivariate_data(self):
        """Fixture providing sample multivariate time series data."""
        return np.array([[1, 2], [3, 4], [10, 20], [30, 40]])

    def test_initialization_with_valid_data(self, sample_univariate_data):
        """Test LabeledData initialization with valid parameters."""
        labeled = LabeledData(sample_univariate_data, change_points=[3, 6])

        assert labeled.raw_data == sample_univariate_data
        assert labeled.change_points == [3, 6]
        assert labeled.name is None

    def test_initialization_with_name(self, sample_univariate_data):
        """Test LabeledData initialization with optional name."""
        labeled = LabeledData(sample_univariate_data, change_points=[3], name="Test Dataset")

        assert labeled.name == "Test Dataset"

    def test_initialization_with_empty_change_points(self, sample_univariate_data):
        """Test LabeledData initialization with empty change points list."""
        labeled = LabeledData(sample_univariate_data, change_points=[])

        assert labeled.change_points == []
        assert len(labeled) == len(sample_univariate_data)

    def test_raises_value_error_for_non_positive_change_point(self, sample_univariate_data):
        """Test ValueError raised when change point index is non-positive."""
        with pytest.raises(ValueError, match="Change point indices must be positive"):
            LabeledData(sample_univariate_data, change_points=[0])

        with pytest.raises(ValueError, match="Change point indices must be positive"):
            LabeledData(sample_univariate_data, change_points=[-1, 3])

    def test_raises_value_error_for_exceeding_change_point(self, sample_univariate_data):
        """Test ValueError raised when change point index exceeds data length."""
        max_index = len(sample_univariate_data)

        with pytest.raises(ValueError, match="Change point index exceeds data length"):
            LabeledData(sample_univariate_data, change_points=[max_index + 1])

        with pytest.raises(ValueError, match="Change point index exceeds data length"):
            LabeledData(sample_univariate_data, change_points=[3, max_index + 2])

    def test_iteration_returns_all_observations(self, sample_univariate_data):
        """Test that iteration yields all observations in correct order."""
        labeled = LabeledData(sample_univariate_data, change_points=[3])

        result = list(labeled)
        assert result == sample_univariate_data

    def test_multiple_iterations(self, sample_univariate_data):
        """Test that multiple iterations produce consistent results."""
        labeled = LabeledData(sample_univariate_data, change_points=[3])

        first_iter = list(labeled)
        second_iter = list(labeled)

        assert first_iter == second_iter == sample_univariate_data

    def test_len_returns_correct_length(self, sample_univariate_data):
        """Test that __len__ returns correct number of observations."""
        labeled = LabeledData(sample_univariate_data, change_points=[3])

        assert len(labeled) == len(sample_univariate_data)

    def test_str_without_name(self, sample_univariate_data):
        """Test string representation without name."""
        labeled = LabeledData(sample_univariate_data, change_points=[3])

        assert str(labeled) == f"Labeled Data (len = {len(sample_univariate_data)})"

    def test_str_with_name(self, sample_univariate_data):
        """Test string representation with name."""
        labeled = LabeledData(sample_univariate_data, change_points=[3], name="Test")

        assert str(labeled) == f"Test (len = {len(sample_univariate_data)})"

    def test_with_multivariate_data(self, sample_multivariate_data):
        """Test LabeledData with multivariate numpy array."""
        labeled = LabeledData(sample_multivariate_data, change_points=[2])

        result = list(labeled)
        assert len(result) == len(sample_multivariate_data)
        assert np.array_equal(result[0], sample_multivariate_data[0])
        assert np.array_equal(result[1], sample_multivariate_data[1])

    def test_with_tuple_data(self):
        """Test LabeledData with tuple as raw data."""
        data = (1, 2, 3, 4, 5)
        labeled = LabeledData(data, change_points=[3])

        assert list(labeled) == [1, 2, 3, 4, 5]
        assert len(labeled) == 5

    def test_edge_case_single_change_point_at_end(self, sample_univariate_data):
        """Test change point at the last observation index."""
        last_index = len(sample_univariate_data)
        labeled = LabeledData(sample_univariate_data, change_points=[last_index])

        assert labeled.change_points == [last_index]
        assert len(labeled) == last_index

    def test_edge_case_multiple_change_points(self, sample_univariate_data):
        """Test with multiple change points in sorted order."""
        labeled = LabeledData(sample_univariate_data, change_points=[2, 4, 6])

        assert labeled.change_points == [2, 4, 6]
        assert list(labeled) == sample_univariate_data
