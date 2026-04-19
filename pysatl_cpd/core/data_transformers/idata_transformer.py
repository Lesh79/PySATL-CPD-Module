# -*- coding: ascii -*-
"""
Interface for data transformers.

This module provides the abstract base class for data transformers, which
are used to adapt data dimensionality or extract specific features before
feeding data into change-point detection algorithms.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

from pysatl_cpd.core.data_providers.idata_provider import DataProvider


class IDataTransformer[DataInT, DataOutT](ABC):
    """
    Abstract base class for data transformers.

    Transformers act as adapters between raw data providers and algorithms,
    allowing, for example, multivariate data to be processed by univariate
    algorithms (e.g., via column selection or norm calculation).
    """

    @abstractmethod
    def transform(self, provider: DataProvider[DataInT]) -> DataProvider[DataOutT]:
        """
        Apply transformation to the given data provider.

        Parameters
        ----------
        provider : DataProvider[DataInT]
            The source data provider.

        Returns
        -------
        DataProvider[DataOut]
            A new data provider yielding transformed observations.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """
        Return the human-readable name of the transformer.

        Returns
        -------
        str
            Transformer identifier used for logging and caching.
        """
        return type(self).__name__

    def __hash__(self) -> int:
        """
        Return the hash of the transformer.

        Used by BenchmarkExecutor to ensure that changes in the transformation
        pipeline correctly invalidate or separate cached traces.

        Returns
        -------
        int
            Hash value based on the transformer's properties.
        """
        return hash(self.name)
