# -*- coding: ascii -*-
"""
Data Providers Module - NumPy Array Implementations

This module provides concrete data providers for univariate and multivariate
time series stored as NumPy arrays. These providers wrap array data and
expose it through the DataProvider interface.
"""

__author__ = "Danil Totmyanin, Vladimir Kutuev, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterator
from typing import cast

from pysatl_cpd.core.data_providers.idata_provider import DataProvider
from pysatl_cpd.core.typedefs import MultivariateNumericArray, NumericArray, NumPyNumber, UnivariateNumericArray


class NDArrayUnivariateProvider(DataProvider[NumPyNumber]):
    """
    Data provider for univariate time series stored as a 1-D NumPy array.

    This provider wraps a one-dimensional NumPy array and exposes it as an
    iterable sequence of scalar values. Each iteration yields the next element
    from the array.

    Parameters
    ----------
    data : NumericArray
        A one-dimensional array of univariate observations. The array must contain
        numeric values and have exactly one dimension.

    Raises
    ------
    ValueError
        If ``data.ndim != 1``. The array must be one-dimensional to represent
        univariate time series.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> provider = NDArrayUnivariateProvider(data)
    >>> len(provider)
    5
    >>> list(provider)
    [1.0, 2.0, 3.0, 4.0, 5.0]
    """

    def __init__(self, data: NumericArray) -> None:
        """
        Initialize the univariate provider with a NumPy array.

        Parameters
        ----------
        data : NumericArray
            One-dimensional array containing the time series data.

        Raises
        ------
        ValueError
            If the array is not one-dimensional.
        """
        if data.ndim != 1:
            raise ValueError(f"Expected 1-dimensional array, got {data.ndim} dimensions")
        self.__data = cast(UnivariateNumericArray, data)

    def __iter__(self) -> Iterator[NumPyNumber]:
        """
        Return an iterator over univariate observations.

        Returns
        -------
        Iterator[NumPyNumber]
            An iterator yielding scalar values from the underlying array in order.
        """
        return iter(self.__data)

    def __len__(self) -> int:
        """
        Return the number of observations in the data provider.

        Returns
        -------
        int
            The length of the underlying array (number of observations).
        """
        return self.__data.shape[0]


class NDArrayMultivariateProvider(DataProvider[UnivariateNumericArray]):
    """
    Data provider for multivariate time series stored as a 2-D NumPy array.

    The array is interpreted as a sequence of observation vectors, where each
    row corresponds to a single time point and columns represent different
    variables. Iteration yields complete observation vectors as one-dimensional
    arrays.

    Parameters
    ----------
    data : NumericArray
        An array with ``ndim == 2``. The first axis indexes observations over time,
        while the second axis represents multivariate components.

    Raises
    ------
    ValueError
        If ``data.ndim != 2``. Multivariate data must be exactly two dimensions,
        with the first dimension representing observations.

    Examples
    --------
    >>> import numpy as np
    >>> data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    >>> provider = NDArrayMultivariateProvider(data)
    >>> len(provider)
    3
    >>> list(provider)
    [array([1., 2.]), array([3., 4.]), array([5., 6.])]
    """

    def __init__(self, data: NumericArray) -> None:
        """
        Initialize the multivariate provider with a NumPy array.

        Parameters
        ----------
        data : NumericArray
            Two-dimensional array where each row is an observation vector.

        Raises
        ------
        ValueError
            If the array is not two-dimensional.
        """
        if data.ndim != 2:
            raise ValueError(f"Expected 2 dimensions, got {data.ndim}")
        self.__data = cast(MultivariateNumericArray, data)

    def __iter__(self) -> Iterator[UnivariateNumericArray]:
        """
        Return an iterator over multivariate observations.

        Returns
        -------
        Iterator[UnivariateNumericArray]
            An iterator yielding one-dimensional array slices representing
            individual observation vectors at each time point.
        """
        return iter(self.__data)

    def __len__(self) -> int:
        """
        Return the number of observations in the data provider.

        Returns
        -------
        int
            The number of rows in the underlying array (number of observations).
        """
        return self.__data.shape[0]
