# -*- coding: ascii -*-

"""
Dummy data transformer implementation for testing.

This module provides a minimal concrete implementation of IDataTransformer
used strictly to test the default behaviors of the abstract base class.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

from pysatl_cpd.core.data_providers.idata_provider import DataProvider
from pysatl_cpd.core.data_transformers.idata_transformer import IDataTransformer


class DummyTransformer(IDataTransformer[Any, Any]):
    """
    Minimal concrete implementation of IDataTransformer.

    Used for testing the default behaviors of the abstract base class,
    such as the default `name` and `__hash__` properties.
    """

    def transform(self, provider: DataProvider[Any]) -> DataProvider[Any]:
        """
        Dummy implementation that just returns the input provider.

        Parameters
        ----------
        provider : DataProvider[Any]
            The source data provider.

        Returns
        -------
        DataProvider[Any]
            The unmodified input provider.
        """
        return provider
