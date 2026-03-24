"""
Tests for NumPy-based data provider implementations.
"""

import numpy as np
import pytest

from pysatl_cpd._typing import NumericArray
from pysatl_cpd.data_providers.numpy_data_provider import (
    NDArrayMultivariateProvider,
    NDArrayUnivariateProvider,
)


class TestNDArrayUnivariateProvider:
    """Test suite for NDArrayUnivariateProvider."""

    def test_valid_1d_array_initialization(self, univariate_float64_array: NumericArray) -> None:
        """Test initialization with valid 1-dimensional array."""
        provider = NDArrayUnivariateProvider(univariate_float64_array)
        assert provider is not None

    def test_iteration_returns_correct_values(self, univariate_float64_array: NumericArray) -> None:
        """Test that iteration yields correct univariate values."""
        provider = NDArrayUnivariateProvider(univariate_float64_array)
        result = list(provider)
        expected = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert result == expected

    def test_multiple_iterations(self, univariate_float64_array: NumericArray) -> None:
        """Test that provider can be iterated multiple times."""
        provider = NDArrayUnivariateProvider(univariate_float64_array)
        first_iter = list(provider)
        second_iter = list(provider)
        assert first_iter == [1.0, 2.0, 3.0, 4.0, 5.0]
        assert second_iter == [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_empty_array(self, univariate_empty_array: NumericArray) -> None:
        """Test provider with empty array."""
        provider = NDArrayUnivariateProvider(univariate_empty_array)
        result = list(provider)
        assert result == []

    def test_single_element_array(self, univariate_single_element_array: NumericArray) -> None:
        """Test provider with single element array."""
        provider = NDArrayUnivariateProvider(univariate_single_element_array)
        result = list(provider)
        assert result == [42.0]

    def test_integer_array(self, univariate_int64_array: NumericArray) -> None:
        """Test provider with integer array (should work with NumPyNumber)."""
        provider = NDArrayUnivariateProvider(univariate_int64_array)
        result = list(provider)
        assert result == [1, 2, 3, 4, 5]

    def test_preserves_array_order(self, univariate_unsorted_array: NumericArray) -> None:
        """Test that iteration preserves original array order."""
        provider = NDArrayUnivariateProvider(univariate_unsorted_array)
        result = list(provider)
        assert result == [5.0, 3.0, 1.0, 4.0, 2.0]

    def test_array_with_nan_values(self, univariate_array_with_nan: NumericArray) -> None:
        """Test provider with array containing NaN values."""
        provider = NDArrayUnivariateProvider(univariate_array_with_nan)
        result = list(provider)
        assert len(result) == 5
        assert result[0] == 1.0
        assert np.isnan(result[1])
        assert result[2] == 3.0
        assert np.isnan(result[3])
        assert result[4] == 5.0

    def test_array_with_inf_values(self, univariate_array_with_inf: NumericArray) -> None:
        """Test provider with array containing Inf values."""
        provider = NDArrayUnivariateProvider(univariate_array_with_inf)
        result = list(provider)
        assert len(result) == 5
        assert result[0] == 1.0
        assert np.isinf(result[1]) and result[1] > 0
        assert result[2] == 3.0
        assert np.isinf(result[3]) and result[3] < 0
        assert result[4] == 5.0

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

    def test_raises_value_error_for_0d_array(self) -> None:
        """Test that 0-dimensional array (scalar) raises ValueError."""
        data = np.array(5.0)
        with pytest.raises(ValueError, match="Expected 1-dimensional array, got 0 dimensions"):
            NDArrayUnivariateProvider(data)


class TestNDArrayMultivariateProvider:
    """Test suite for NDArrayMultivariateProvider."""

    def test_valid_2d_array_initialization(self, multivariate_float64_array: NumericArray) -> None:
        """Test initialization with valid 2-dimensional array."""
        provider = NDArrayMultivariateProvider(multivariate_float64_array)
        assert provider is not None

    def test_iteration_returns_correct_vectors(self, multivariate_float64_array: NumericArray) -> None:
        """Test that iteration yields correct multivariate vectors."""
        provider = NDArrayMultivariateProvider(multivariate_float64_array)
        result = list(provider)
        expected = [np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0, 6.0])]
        for res, exp in zip(result, expected, strict=True):
            assert np.array_equal(res, exp)

    def test_multiple_iterations(self, multivariate_float64_array: NumericArray) -> None:
        """Test that provider can be iterated multiple times."""
        provider = NDArrayMultivariateProvider(multivariate_float64_array)
        first_iter = list(provider)
        second_iter = list(provider)
        for first, second in zip(first_iter, second_iter, strict=True):
            assert np.array_equal(first, second)

    def test_empty_2d_array(self, multivariate_empty_array: NumericArray) -> None:
        """Test provider with empty 2-dimensional array."""
        provider = NDArrayMultivariateProvider(multivariate_empty_array)
        result = list(provider)
        assert result == []

    def test_single_observation(self, multivariate_single_observation_array: NumericArray) -> None:
        """Test provider with single observation vector."""
        provider = NDArrayMultivariateProvider(multivariate_single_observation_array)
        result = list(provider)
        assert len(result) == 1
        assert np.array_equal(result[0], np.array([1.0, 2.0, 3.0]))

    def test_integer_multivariate_array(self, multivariate_int64_array: NumericArray) -> None:
        """Test provider with integer multivariate array."""
        provider = NDArrayMultivariateProvider(multivariate_int64_array)
        result = list(provider)
        expected = [np.array([1, 2]), np.array([3, 4]), np.array([5, 6])]
        for res, exp in zip(result, expected, strict=True):
            assert np.array_equal(res, exp)

    def test_preserves_row_order(self, multivariate_unsorted_array: NumericArray) -> None:
        """Test that iteration preserves original row order."""
        provider = NDArrayMultivariateProvider(multivariate_unsorted_array)
        result = list(provider)
        assert np.array_equal(result[0], np.array([5.0, 1.0]))
        assert np.array_equal(result[1], np.array([3.0, 2.0]))
        assert np.array_equal(result[2], np.array([1.0, 3.0]))
        assert np.array_equal(result[3], np.array([4.0, 4.0]))

    def test_array_with_nan_values(self, multivariate_array_with_nan: NumericArray) -> None:
        """Test provider with array containing NaN values."""
        provider = NDArrayMultivariateProvider(multivariate_array_with_nan)
        result = list(provider)
        assert len(result) == 3
        # First observation: [1.0, NaN]
        assert result[0][0] == 1.0
        assert np.isnan(result[0][1])
        # Second observation: [3.0, 4.0]
        assert np.array_equal(result[1], np.array([3.0, 4.0]))
        # Third observation: [NaN, 6.0]
        assert np.isnan(result[2][0])
        assert result[2][1] == 6.0

    def test_array_with_inf_values(self, multivariate_array_with_inf: NumericArray) -> None:
        """Test provider with array containing Inf values."""
        provider = NDArrayMultivariateProvider(multivariate_array_with_inf)
        result = list(provider)
        assert len(result) == 3
        # First observation: [1.0, Inf]
        assert result[0][0] == 1.0
        assert np.isinf(result[0][1]) and result[0][1] > 0
        # Second observation: [3.0, 4.0]
        assert np.array_equal(result[1], np.array([3.0, 4.0]))
        # Third observation: [-Inf, 6.0]
        assert np.isinf(result[2][0]) and result[2][0] < 0
        assert result[2][1] == 6.0

    def test_array_with_mixed_special_values(self, multivariate_array_with_mixed_special_values: NumericArray) -> None:
        """Test provider with array containing both NaN and Inf values."""
        provider = NDArrayMultivariateProvider(multivariate_array_with_mixed_special_values)
        result = list(provider)
        assert len(result) == 3
        # First observation: [1.0, NaN]
        assert result[0][0] == 1.0
        assert np.isnan(result[0][1])
        # Second observation: [Inf, 4.0]
        assert np.isinf(result[1][0]) and result[1][0] > 0
        assert result[1][1] == 4.0
        # Third observation: [5.0, -Inf]
        assert result[2][0] == 5.0
        assert np.isinf(result[2][1]) and result[2][1] < 0

    def test_raises_value_error_for_3d_array(self) -> None:
        """Test that 3-dimensional array raises ValueError."""
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

    def test_raises_value_error_for_empty_2d_array_with_invalid_shape(self) -> None:
        """Test that empty array with incorrect shape raises ValueError."""
        data = np.array([]).reshape(0, 0)
        # This should still pass because it's 2D, just empty with zero columns
        provider = NDArrayMultivariateProvider(data)
        result = list(provider)
        assert result == []
