"""
Data Providers Module - Abstract Interface

This module defines the abstract base class for data providers that supply
sequential observations to change-point detection algorithms. Concrete
implementations of this interface wrap various data sources (NumPy arrays,
datasets, streams) and expose them as iterables.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TypeVar

T = TypeVar("T")


class DataProvider[T](ABC):
    """
    Abstract base class for data providers.

    A data provider is an iterable object that yields observations
    one at a time, suitable for consumption by online change-point
    detection algorithms or for sequential processing in offline
    change-point detection algorithms.

    Parameters
    ----------
    T : TypeVar
        The type of a single observation yielded by the provider.
        For univariate data, T is typically a scalar numeric type.
        For multivariate data, T is typically a one-dimensional array.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[T]:
        """
        Return an iterator over the observations.

        Returns
        -------
        Iterator[T]
            An iterator yielding observations of type T.
        """

        raise NotImplementedError
