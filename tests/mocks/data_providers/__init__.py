"""
Mock data providers for testing.

This module provides mock implementations of DataProvider for testing
change-point detection algorithms.
"""

from tests.mocks.data_providers.constant import MockConstantDataProvider
from tests.mocks.data_providers.edge import (
    MockInfDataProvider,
    MockMultivariateEdgeDataProvider,
    MockNaNDataProvider,
    MockNegativeDataProvider,
    MockZeroDataProvider,
)
from tests.mocks.data_providers.empty import MockEmptyDataProvider
from tests.mocks.data_providers.multivariate import MockMultivariateDataProvider
from tests.mocks.data_providers.single import MockSingleObservationProvider
from tests.mocks.data_providers.univariate import MockUnivariateDataProvider

__all__ = [
    "MockUnivariateDataProvider",
    "MockMultivariateDataProvider",
    "MockEmptyDataProvider",
    "MockSingleObservationProvider",
    "MockConstantDataProvider",
    "MockNaNDataProvider",
    "MockInfDataProvider",
    "MockNegativeDataProvider",
    "MockZeroDataProvider",
    "MockMultivariateEdgeDataProvider",
]
