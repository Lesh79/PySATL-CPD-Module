# -*- coding: ascii -*-

"""
Mock data transformer implementations for testing.

This module provides mock implementations of IDataTransformer used for testing
the transformation pipeline in benchmark execution and algorithm evaluation.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.core.data_providers.idata_provider import DataProvider
from pysatl_cpd.core.data_transformers.idata_transformer import IDataTransformer


class MockDataTransformer(IDataTransformer[float, float]):
    """
    Mock data transformer for testing benchmark execution.

    This transformer adds a specified constant value to every observation
    in the dataset and keeps track of how many times the `transform` method
    was applied. It wraps the transformed data back into a `LabeledData` instance.

    Parameters
    ----------
    name : str, default="MockTransform"
        The string identifier for the transformer.
    add_value : float, default=1.0
        The numeric value to add to each observation.
    """

    def __init__(self, name: str = "MockTransform", add_value: float = 1.0) -> None:
        self._name = name
        self.add_value = add_value
        self.call_count = 0

    @property
    def name(self) -> str:
        """
        Return the name of the mock transformer.

        Returns
        -------
        str
            The identifier of this transformer instance.
        """
        return self._name

    def __hash__(self) -> int:
        """
        Return a hash based on the transformer's properties.

        Used to uniquely identify the pipeline configuration in the cache.

        Returns
        -------
        int
            Hash value representing the transformer configuration.
        """
        return hash((self._name, self.add_value))

    def transform(self, provider: DataProvider[float]) -> DataProvider[float]:
        """
        Transform the data by adding a constant value to each element.

        Parameters
        ----------
        provider : DataProvider[float]
            The original data provider.

        Returns
        -------
        DataProvider[float]
            A new `LabeledData` instance containing the transformed values.
        """
        self.call_count += 1

        # Transform data
        new_data: list[float] = [float(x) + self.add_value for x in provider]

        # Preserve change points if the provider has them
        change_points: Any = getattr(provider, "change_points", getattr(provider, "change_point", []))

        return LabeledData(raw_data=new_data, change_points=change_points, name=f"{provider.name}_{self.name}")
