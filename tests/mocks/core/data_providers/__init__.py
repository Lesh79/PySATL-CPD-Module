# -*- coding: ascii -*-

"""
Mock data providers for testing.

This module provides mock implementations of DataProvider for testing
change-point detection algorithms.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from tests.mocks.core.data_providers.dirty import (
    MockMultivariateDirtyDataProvider,
    MockUnivariateDirtyDataProvider,
)
from tests.mocks.core.data_providers.edge import (
    MockEmptyDataProvider,
    MockSingleObservationProvider,
)
from tests.mocks.core.data_providers.multivariate import (
    MockMultivariateConstantDataProvider,
    MockMultivariateDataProvider,
    MockMultivariateInfDataProvider,
    MockMultivariateNaNDataProvider,
    MockMultivariateZeroDataProvider,
)
from tests.mocks.core.data_providers.univariate import (
    MockUnivariateConstantDataProvider,
    MockUnivariateDataProvider,
    MockUnivariateInfDataProvider,
    MockUnivariateNaNDataProvider,
    MockUnivariateZeroDataProvider,
)

__all__ = [
    # Univariate providers
    "MockUnivariateDataProvider",
    "MockUnivariateConstantDataProvider",
    "MockUnivariateZeroDataProvider",
    "MockUnivariateNaNDataProvider",
    "MockUnivariateInfDataProvider",
    "MockUnivariateDirtyDataProvider",
    # Multivariate providers
    "MockMultivariateDataProvider",
    "MockMultivariateConstantDataProvider",
    "MockMultivariateZeroDataProvider",
    "MockMultivariateNaNDataProvider",
    "MockMultivariateInfDataProvider",
    "MockMultivariateDirtyDataProvider",
    # Edge providers
    "MockEmptyDataProvider",
    "MockSingleObservationProvider",
]
