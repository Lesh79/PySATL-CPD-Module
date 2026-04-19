# -*- coding: ascii -*-
"""
Data Providers Module

This module provides data provider classes that supply sequential observations
to change-point detection algorithms. The module includes an abstract base class
defining the provider interface and concrete implementations for NumPy arrays.
"""

__author__ = "Vladimir Kutuev, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.core.data_providers.dataset import Annotation, Dataset, PandasLabeledDataProvider, RealDatasetLoader, SegmentInfo
from pysatl_cpd.core.data_providers.idata_provider import DataProvider
from pysatl_cpd.core.data_providers.numpy_data_provider import (
    NDArrayMultivariateProvider,
    NDArrayUnivariateProvider,
)

__all__ = [
    "DataProvider",
    "Annotation",
    "SegmentInfo",
    "PandasLabeledDataProvider",
    "Dataset",
    "RealDatasetLoader",
    "NDArrayMultivariateProvider",
    "NDArrayUnivariateProvider",
]
