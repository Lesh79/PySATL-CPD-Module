# -*- coding: ascii -*-

"""
Columns Selector Transformer Implementation.

This module provides a transformer that allows selecting specific columns
from multivariate time series data.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np

from pysatl_cpd.core.data_providers.idata_provider import DataProvider
from pysatl_cpd.core.data_providers.numpy_data_provider import (
    NDArrayMultivariateProvider,
    NDArrayUnivariateProvider,
)
from pysatl_cpd.core.data_transformers.idata_transformer import IDataTransformer


class ColumnsSelectorTransformer(IDataTransformer[np.ndarray, np.ndarray | float]):
    """
    Transformer for selecting specific columns from multivariate data.

    If a single integer index is provided, it transforms multivariate data
    into univariate data. If a list of indices is provided, it returns
    multivariate data containing only the specified columns.

    Parameters
    ----------
    columns : list[int] or int
        Indices of columns to select from the input multivariate array.
    """

    def __init__(self, columns: list[int] | int) -> None:
        self.cols = columns

    @property
    def name(self) -> str:
        """
        Return a unique name including selected column indices.

        Returns
        -------
        str
            Formatted name like 'Col_0' or 'Cols_0_2_3'.
        """
        if isinstance(self.cols, int):
            return f"Col_{self.cols}"
        cols_str = "_".join(map(str, self.cols))
        return f"Cols_{cols_str}"

    def transform(self, provider: DataProvider[np.ndarray]) -> DataProvider[np.ndarray | float]:
        """
        Extract selected columns and wrap into a new NumPy data provider.

        Parameters
        ----------
        provider : DataProvider[np.ndarray]
            Multivariate data provider yielding 1-D NumPy arrays.

        Returns
        -------
        DataProvider[Any]
            NDArrayUnivariateProvider if `columns` is int,
            NDArrayMultivariateProvider if `columns` is list[int].

        Raises
        ------
        ValueError
            If the data provided by the source is not 2-dimensional.
        """
        raw_nd_data = np.array(list(provider))

        if raw_nd_data.ndim < 2:
            raise ValueError(
                f"ColumnsSelectorTransformer expects 2D data, "
                f"got {raw_nd_data.ndim}D data from provider '{provider.name}'."
            )

        cols_data = raw_nd_data[:, self.cols]

        new_provider_name = f"{provider.name}_{self.name}"

        if isinstance(self.cols, int):
            return NDArrayUnivariateProvider(data=cols_data, name=new_provider_name)

        return NDArrayMultivariateProvider(data=cols_data, name=new_provider_name)
