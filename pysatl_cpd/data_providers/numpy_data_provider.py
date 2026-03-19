"""
Data Providers Module - NumPy Array Implementations

This module provides concrete data providers for univariate and multivariate
time series stored as NumPy arrays. These providers wrap array data and
expose it through the DataProvider interface.
"""

from collections.abc import Iterator
from typing import cast

from pysatl_cpd._typing import MultivariateNumericArray, NumericArray, NumPyNumber, UnivariateNumericArray
from pysatl_cpd.data_providers.idata_provider import DataProvider


class NDArrayUnivariateProvider(DataProvider[NumPyNumber]):
    """
    Data provider for univariate time series stored as a 1-D NumPy array.

    Parameters
    ----------
    data : UnivariateNumericArray
        A one-dimensional array of univariate observations. The array must contain
        numeric values and have exactly one dimension.

    Raises
    ------
    ValueError
        If ``data.ndim != 1``. The array must be one-dimensional to represent
        univariate time series.
    """

    def __init__(self, data: NumericArray) -> None:
        if data.ndim != 1:
            raise ValueError(f"Expected 1-dimensional array, got {data.ndim} dimensions")
        self.__data = cast(UnivariateNumericArray, data)

    def __iter__(self) -> Iterator[NumPyNumber]:
        """
        Return an iterator over univariate observations.

        Returns
        -------
        Iterator[NumPyNumber]
            An iterator yielding scalar values from the underlying array.
        """

        return iter(self.__data)


class NDArrayMultivariateProvider(DataProvider[UnivariateNumericArray]):
    """
    Data provider for multivariate time series stored as a 2-D NumPy array.

    The array is interpreted as a sequence of observation vectors, where each
    row corresponds to a single time point and columns represent different
    variables. Iteration yields complete observation vectors.

    Parameters
    ----------
    data : MultivariateNumericArray
        An array with ``ndim == 2``. The first axis indexes observations over time,
        while remaining axes represent multivariate components. For standard
        multivariate time series, a 2-D array with observations as rows is expected.

    Raises
    ------
    ValueError
        If ``data.ndim != 2``. Multivariate data must be exactly two dimensions,
        with the first dimension representing observations.
    """

    def __init__(self, data: NumericArray) -> None:
        if data.ndim != 2:
            raise ValueError(f"Expected at least 2 dimensions, got {data.ndim}")
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
