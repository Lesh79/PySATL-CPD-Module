"""
Shared fixtures for data provider tests.
"""

import numpy as np
import pytest

from pysatl_cpd._typing import NumericArray
from tests.mocks.data_providers import (
    MockConstantDataProvider,
    MockEmptyDataProvider,
    MockInfDataProvider,
    MockMultivariateDataProvider,
    MockMultivariateEdgeDataProvider,
    MockNaNDataProvider,
    MockNegativeDataProvider,
    MockSingleObservationProvider,
    MockUnivariateDataProvider,
    MockZeroDataProvider,
)


@pytest.fixture
def univariate_provider() -> MockUnivariateDataProvider:
    """Fixture providing a univariate data provider."""
    return MockUnivariateDataProvider([1, 2, 3])


@pytest.fixture
def multivariate_provider() -> MockMultivariateDataProvider:
    """Fixture providing a multivariate data provider."""
    return MockMultivariateDataProvider([[1, 2], [3, 4]])


@pytest.fixture
def empty_provider() -> MockEmptyDataProvider[int]:
    """Fixture providing an empty data provider."""
    return MockEmptyDataProvider[int]()


@pytest.fixture
def single_observation_provider() -> MockSingleObservationProvider:
    """Fixture providing a provider with one observation."""
    return MockSingleObservationProvider(42)


@pytest.fixture
def constant_provider() -> MockConstantDataProvider:
    """Fixture providing a constant data provider."""
    return MockConstantDataProvider(5, 10)


@pytest.fixture
def nan_provider() -> MockNaNDataProvider:
    """Fixture providing a NaN data provider."""
    return MockNaNDataProvider(5)


@pytest.fixture
def inf_provider() -> MockInfDataProvider:
    """Fixture providing an Inf data provider."""
    return MockInfDataProvider(5)


@pytest.fixture
def zero_provider() -> MockZeroDataProvider:
    """Fixture providing a zero data provider."""
    return MockZeroDataProvider(5)


@pytest.fixture
def negative_provider() -> MockNegativeDataProvider:
    """Fixture providing a negative values provider."""
    return MockNegativeDataProvider([-1, -2, -3])


@pytest.fixture
def multivariate_edge_provider() -> MockMultivariateEdgeDataProvider:
    """Fixture providing a multivariate edge provider."""
    return MockMultivariateEdgeDataProvider(
        [[1, 2], [3, 4], [5, 6]],
        nan_positions=[(0, 1)],
        inf_positions=[(2, 0)],
    )


# ==================== NumPy Data Provider Fixtures ====================


@pytest.fixture
def univariate_float64_array() -> NumericArray:
    """Fixture providing a simple float64 univariate array."""
    return np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)


@pytest.fixture
def univariate_int64_array() -> NumericArray:
    """Fixture providing an int64 univariate array."""
    return np.array([1, 2, 3, 4, 5], dtype=np.int64)


@pytest.fixture
def univariate_empty_array() -> NumericArray:
    """Fixture providing an empty univariate array."""
    return np.array([], dtype=np.float64)


@pytest.fixture
def univariate_single_element_array() -> NumericArray:
    """Fixture providing a univariate array with single element."""
    return np.array([42.0], dtype=np.float64)


@pytest.fixture
def univariate_array_with_nan() -> NumericArray:
    """Fixture providing a univariate array containing NaN."""
    return np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float64)


@pytest.fixture
def univariate_array_with_inf() -> NumericArray:
    """Fixture providing a univariate array containing Inf."""
    return np.array([1.0, np.inf, 3.0, -np.inf, 5.0], dtype=np.float64)


@pytest.fixture
def univariate_unsorted_array() -> NumericArray:
    """Fixture providing a univariate array with unsorted values."""
    return np.array([5.0, 3.0, 1.0, 4.0, 2.0], dtype=np.float64)


@pytest.fixture
def multivariate_float64_array() -> NumericArray:
    """Fixture providing a simple float64 multivariate array."""
    return np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float64)


@pytest.fixture
def multivariate_int64_array() -> NumericArray:
    """Fixture providing an int64 multivariate array."""
    return np.array([[1, 2], [3, 4], [5, 6]], dtype=np.int64)


@pytest.fixture
def multivariate_empty_array() -> NumericArray:
    """Fixture providing an empty multivariate array with 2 columns."""
    return np.array([]).reshape(0, 2)


@pytest.fixture
def multivariate_single_observation_array() -> NumericArray:
    """Fixture providing a multivariate array with single observation."""
    return np.array([[1.0, 2.0, 3.0]], dtype=np.float64)


@pytest.fixture
def multivariate_array_with_nan() -> NumericArray:
    """Fixture providing a multivariate array containing NaN values."""
    return np.array([[1.0, np.nan], [3.0, 4.0], [np.nan, 6.0]], dtype=np.float64)


@pytest.fixture
def multivariate_array_with_inf() -> NumericArray:
    """Fixture providing a multivariate array containing Inf values."""
    return np.array([[1.0, np.inf], [3.0, 4.0], [-np.inf, 6.0]], dtype=np.float64)


@pytest.fixture
def multivariate_array_with_mixed_special_values() -> NumericArray:
    """Fixture providing a multivariate array with both NaN and Inf."""
    return np.array([[1.0, np.nan], [np.inf, 4.0], [5.0, -np.inf]], dtype=np.float64)


@pytest.fixture
def multivariate_unsorted_array() -> NumericArray:
    """Fixture providing a multivariate array with unsorted rows."""
    return np.array([[5.0, 1.0], [3.0, 2.0], [1.0, 3.0], [4.0, 4.0]], dtype=np.float64)
