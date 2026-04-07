# -*- coding: ascii -*-

"""
Mock univariate data provider for testing.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Iterator, Sequence

from pysatl_cpd.core.data_providers import DataProvider
from pysatl_cpd.core.typedefs import Number


class MockUnivariateDataProvider(DataProvider[Number]):
    """
    Mock data provider for univariate time series testing.

    This provider yields predefined sequences of numeric values, allowing
    controlled testing of detection algorithms with known patterns.

    Parameters
    ----------
    data : Sequence[Number]
        Sequence of observations to yield.
    """

    def __init__(self, data: Sequence[Number], name: str | None = None) -> None:
        super().__init__(name)

        self._data = list(data)
        self._call_count = 0

    def __iter__(self) -> Iterator[Number]:
        """
        Return iterator over the data sequence.

        Returns
        -------
        Iterator[Number]
            Iterator yielding observations in order.
        """
        self._call_count += 1
        return iter(self._data)

    def __len__(self) -> int:
        """Return number of observations."""
        return len(self._data)

    def __getitem__(self, index: int) -> Number:
        """Get observation at specific index."""
        return self._data[index]

    def get_call_count(self) -> int:
        """Return number of times __iter__ was called."""
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the call counter."""
        self._call_count = 0

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MockUnivariateDataProvider(length={len(self)})"


class MockUnivariateConstantDataProvider(MockUnivariateDataProvider):
    """
    Mock data provider that always returns the same value.

    Useful for testing algorithms with constant data streams.

    Parameters
    ----------
    value : Number
        The constant value to return for each observation.
    length : int
        Number of observations to yield.
    """

    def __init__(self, value: Number, length: int, name: str | None = None) -> None:
        self._value = value
        self._length = length
        super().__init__([value] * length, name)


class MockUnivariateZeroDataProvider(MockUnivariateConstantDataProvider):
    """
    Mock data provider that always returns zero.

    Useful for testing algorithms with zero data streams.

    Parameters
    ----------
    length : int
        Number of observations to yield.
    """

    def __init__(self, length: int, name: str | None = None) -> None:
        super().__init__(0.0, length, name)


class MockUnivariateNaNDataProvider(MockUnivariateConstantDataProvider):
    """
    Mock data provider that always returns NaN.

    Useful for testing algorithm robustness with missing data.

    Parameters
    ----------
    length : int
        Number of observations to yield.
    """

    def __init__(self, length: int, name: str | None = None) -> None:
        super().__init__(float("nan"), length, name)


class MockUnivariateInfDataProvider(MockUnivariateConstantDataProvider):
    """
    Mock data provider that always returns Inf.

    Useful for testing algorithm robustness with infinite values.

    Parameters
    ----------
    length : int
        Number of observations to yield.
    """

    def __init__(self, length: int, name: str | None = None) -> None:
        super().__init__(float("inf"), length, name)
