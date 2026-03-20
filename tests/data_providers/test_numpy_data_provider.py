"""
Tests for NumPy-based data provider implementations.
"""

import numpy as np
import pytest

from pysatl_cpd.data_providers.numpy_data_provider import (
    NDArrayMultivariateProvider,
    NDArrayUnivariateProvider,
)


class TestNDArrayUnivariateProvider:
    """Test suite for NDArrayUnivariateProvider."""

    def test_valid_1d_array_initialization(self) -> None:
        """Test initialization with valid 1-dimensional array."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        provider = NDArrayUnivariateProvider(data)

        assert provider is not None

    def test_iteration_returns_correct_values(self) -> None:
        """Test that iteration yields correct univariate values."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        provider = NDArrayUnivariateProvider(data)

        result = list(provider)
        expected = [1.0, 2.0, 3.0, 4.0, 5.0]

        assert result == expected

    def test_multiple_iterations(self) -> None:
        """Test that provider can be iterated multiple times."""
        data = np.array([1.0, 2.0, 3.0])
        provider = NDArrayUnivariateProvider(data)

        assert list(provider) == [1.0, 2.0, 3.0]
        assert list(provider) == [1.0, 2.0, 3.0]
        assert list(provider) == [1.0, 2.0, 3.0]

    def test_empty_array(self) -> None:
        """Test provider with empty array."""
        data = np.array([])
        provider = NDArrayUnivariateProvider(data)

        result = list(provider)
        assert result == []

    def test_integer_array(self) -> None:
        """Test provider with integer array (should work with NumPyNumber)."""
        data = np.array([1, 2, 3, 4, 5])
        provider = NDArrayUnivariateProvider(data)

        result = list(provider)
        assert result == [1, 2, 3, 4, 5]

    def test_raises_value_error_for_2d_array(self) -> None:
        """Test that 2-dimensional array raises ValueError."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])

        with pytest.raises(ValueError, match="Expected 1-dimensional array, got 2 dimensions"):
            NDArrayUnivariateProvider(data)

    def test_raises_value_error_for_3d_array(self) -> None:
        """Test that 3-dimensional array raises ValueError."""
        data = np.array([[[1.0, 2.0], [3.0, 4.0]]])

        with pytest.raises(ValueError, match="Expected 1-dimensional array, got 3 dimensions"):
            NDArrayUnivariateProvider(data)

    def test_preserves_array_order(self) -> None:
        """Test that iteration preserves original array order."""
        data = np.array([5.0, 3.0, 1.0, 4.0, 2.0])
        provider = NDArrayUnivariateProvider(data)

        result = list(provider)
        assert result == [5.0, 3.0, 1.0, 4.0, 2.0]


class TestNDArrayMultivariateProvider:
    """Test suite for NDArrayMultivariateProvider."""

    def test_valid_2d_array_initialization(self) -> None:
        """Test initialization with valid 2-dimensional array."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        provider = NDArrayMultivariateProvider(data)

        assert provider is not None

    def test_iteration_returns_correct_vectors(self) -> None:
        """Test that iteration yields correct multivariate vectors."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        provider = NDArrayMultivariateProvider(data)

        result = list(provider)
        expected = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]

        for res, exp in zip(result, expected, strict=False):
            assert np.array_equal(res, exp)

    def test_multiple_iterations_2d(self) -> None:
        """Test that provider can be iterated multiple times."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        provider = NDArrayMultivariateProvider(data)

        first_iter = list(provider)
        second_iter = list(provider)

        for first, second in zip(first_iter, second_iter, strict=False):
            assert np.array_equal(first, second)

    def test_empty_2d_array(self) -> None:
        """Test provider with empty 2-dimensional array."""
        data = np.array([]).reshape(0, 2)
        provider = NDArrayMultivariateProvider(data)

        result = list(provider)
        assert result == []

    def test_single_observation(self) -> None:
        """Test provider with single observation vector."""
        data = np.array([[1.0, 2.0, 3.0]])
        provider = NDArrayMultivariateProvider(data)

        result = list(provider)
        assert len(result) == 1
        assert np.array_equal(result[0], np.array([1.0, 2.0, 3.0]))

    def test_raises_value_error_for_3d_array(self) -> None:
        """Test that 3-dimensional array is accepted (should work with MultivariateNumericArray)."""
        # 3D array where first dimension indexes observations
        data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

        with pytest.raises(ValueError, match="Expected 2 dimensions, got 3"):
            NDArrayMultivariateProvider(data)

    def test_raises_value_error_for_1d_array(self) -> None:
        """Test that 1-dimensional array raises ValueError."""
        data = np.array([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Expected 2 dimensions, got 1"):
            NDArrayMultivariateProvider(data)

    def test_raises_value_error_for_0d_array(self) -> None:
        """Test that 0-dimensional array (scalar) raises ValueError."""
        data = np.array(5.0)

        with pytest.raises(ValueError, match="Expected 2 dimensions, got 0"):
            NDArrayMultivariateProvider(data)

    def test_integer_multivariate_array(self) -> None:
        """Test provider with integer multivariate array."""
        data = np.array([[1, 2], [3, 4], [5, 6]])
        provider = NDArrayMultivariateProvider(data)

        result = list(provider)
        expected = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]

        for res, exp in zip(result, expected, strict=False):
            assert np.array_equal(res, exp)

    def test_preserves_row_order(self) -> None:
        """Test that iteration preserves original row order."""
        data = np.array([[5, 1], [3, 2], [1, 3], [4, 4]])
        provider = NDArrayMultivariateProvider(data)

        result = list(provider)
        assert np.array_equal(result[0], np.array([5, 1]))
        assert np.array_equal(result[1], np.array([3, 2]))
        assert np.array_equal(result[2], np.array([1, 3]))
        assert np.array_equal(result[3], np.array([4, 4]))
