"""
Data Providers Module

This module provides data provider classes that supply sequential observations
to change-point detection algorithms. The module includes an abstract base class
defining the provider interface and concrete implementations for NumPy arrays.
"""

from pysatl_cpd.data_providers.idata_provider import DataProvider
from pysatl_cpd.data_providers.numpy_data_provider import (
    NDArrayMultivariateProvider,
    NDArrayUnivariateProvider,
)

__all__ = [
    "DataProvider",
    "NDArrayMultivariateProvider",
    "NDArrayUnivariateProvider",
]
