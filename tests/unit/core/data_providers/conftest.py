# -*- coding: ascii -*-

"""
Shared fixtures for data provider tests.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import numpy as np
import pytest

from pysatl_cpd.core.typedefs import NumericArray
from tests.mocks.core.data_providers import (
    MockEmptyDataProvider,
    MockMultivariateConstantDataProvider,
    MockMultivariateDataProvider,
    MockMultivariateDirtyDataProvider,
    MockMultivariateInfDataProvider,
    MockMultivariateNaNDataProvider,
    MockMultivariateZeroDataProvider,
    MockSingleObservationProvider,
    MockUnivariateConstantDataProvider,
    MockUnivariateDataProvider,
    MockUnivariateDirtyDataProvider,
    MockUnivariateInfDataProvider,
    MockUnivariateNaNDataProvider,
    MockUnivariateZeroDataProvider,
)

# ==================== Basic Provider Fixtures ====================


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
def single_observation_provider() -> MockSingleObservationProvider[float]:
    """Fixture providing a provider with one observation."""
    return MockSingleObservationProvider(42.0)


# ==================== Constant Provider Fixtures ====================


@pytest.fixture
def constant_provider() -> MockUnivariateConstantDataProvider:
    """Fixture providing a constant data provider."""
    return MockUnivariateConstantDataProvider(5, 10)


@pytest.fixture
def zero_provider() -> MockUnivariateZeroDataProvider:
    """Fixture providing a zero data provider."""
    return MockUnivariateZeroDataProvider(5)


@pytest.fixture
def nan_provider() -> MockUnivariateNaNDataProvider:
    """Fixture providing a NaN data provider."""
    return MockUnivariateNaNDataProvider(5)


@pytest.fixture
def inf_provider() -> MockUnivariateInfDataProvider:
    """Fixture providing an Inf data provider."""
    return MockUnivariateInfDataProvider(5)


@pytest.fixture
def multivariate_constant_provider() -> MockMultivariateConstantDataProvider:
    """Fixture providing a constant multivariate data provider."""
    return MockMultivariateConstantDataProvider([1.0, 2.0], 5)


@pytest.fixture
def multivariate_zero_provider() -> MockMultivariateZeroDataProvider:
    """Fixture providing a multivariate zero data provider."""
    return MockMultivariateZeroDataProvider(dimensions=3, length=5)


@pytest.fixture
def multivariate_nan_provider() -> MockMultivariateNaNDataProvider:
    """Fixture providing a multivariate NaN data provider."""
    return MockMultivariateNaNDataProvider(dimensions=3, length=5)


@pytest.fixture
def multivariate_inf_provider() -> MockMultivariateInfDataProvider:
    """Fixture providing a multivariate Inf data provider."""
    return MockMultivariateInfDataProvider(dimensions=3, length=5)


# ==================== Dirty Provider Fixtures ====================


@pytest.fixture
def univariate_dirty_provider() -> MockUnivariateDirtyDataProvider:
    """Fixture providing a univariate dirty data provider."""
    source = MockUnivariateDataProvider([1.0, 2.0, 3.0, 4.0, 5.0])
    return MockUnivariateDirtyDataProvider(
        source=source,
        nan_indices=[1, 3],
        inf_indices=[2],
    )


@pytest.fixture
def multivariate_dirty_provider() -> MockMultivariateDirtyDataProvider:
    """Fixture providing a multivariate dirty data provider."""
    source = MockMultivariateDataProvider([[1, 2], [3, 4], [5, 6]])
    return MockMultivariateDirtyDataProvider(
        source=source,
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
