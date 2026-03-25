# -*- coding: ascii -*-

"""
Mock multivariate data provider for testing.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Iterator, Sequence

from pysatl_cpd.core.data_providers import DataProvider
from pysatl_cpd.core.typedefs import Number


class MockMultivariateDataProvider(DataProvider[list[Number]]):
    """
    Mock data provider for multivariate time series testing.

    This provider yields predefined sequences of observation vectors,
    allowing controlled testing of detection algorithms with multivariate data.

    Parameters
    ----------
    data : Sequence[Sequence[Number]]
        Sequence of observations where each observation is a list of numeric values.
        All observations must have the same length.
    """

    def __init__(self, data: Sequence[Sequence[Number]]) -> None:
        if not data:
            self._data: list[list[Number]] = []
            self._dimensions = 0
        else:
            # Convert to list of lists
            self._data = [list(obs) for obs in data]
            # Verify all observations have same length
            dims = {len(obs) for obs in self._data}
            if len(dims) != 1:
                raise ValueError(f"All observations must have same dimensions, got {dims}")
            self._dimensions = dims.pop()
        self._call_count = 0

    def __iter__(self) -> Iterator[list[Number]]:
        """
        Return iterator over the data rows.

        Returns
        -------
        Iterator[list[Number]]
            Iterator yielding observation vectors in order.
        """
        self._call_count += 1
        return iter(self._data)

    def __len__(self) -> int:
        """Return number of observations."""
        return len(self._data)

    def __getitem__(self, index: int) -> list[Number]:
        """Get observation vector at specific index."""
        return self._data[index]

    def get_call_count(self) -> int:
        """Return number of times __iter__ was called."""
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the call counter."""
        self._call_count = 0

    @property
    def dimensions(self) -> int:
        """Return number of dimensions (variables)."""
        return self._dimensions

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MockMultivariateDataProvider(observations={len(self)}, dimensions={self.dimensions})"


class MockMultivariateConstantDataProvider(MockMultivariateDataProvider):
    """
    Mock data provider that always returns the same vector.

    Useful for testing algorithms with constant multivariate data streams.

    Parameters
    ----------
    value : list[Number]
        The constant vector to return for each observation.
    length : int
        Number of observations to yield.
    """

    def __init__(self, value: list[Number], length: int) -> None:
        self._value = value
        self._length = length
        super().__init__([value] * length)


class MockMultivariateZeroDataProvider(MockMultivariateConstantDataProvider):
    """
    Mock data provider that always returns a vector of zeros.

    Useful for testing algorithms with zero data streams.

    Parameters
    ----------
    dimensions : int
        Number of dimensions (variables) in each observation.
    length : int
        Number of observations to yield.
    """

    def __init__(self, dimensions: int, length: int) -> None:
        value = [0.0] * dimensions
        super().__init__(value, length)


class MockMultivariateNaNDataProvider(MockMultivariateConstantDataProvider):
    """
    Mock data provider that always returns a vector of NaN values.

    Useful for testing algorithm robustness with missing data.

    Parameters
    ----------
    dimensions : int
        Number of dimensions (variables) in each observation.
    length : int
        Number of observations to yield.
    """

    def __init__(self, dimensions: int, length: int) -> None:
        value = [float("nan")] * dimensions
        super().__init__(value, length)


class MockMultivariateInfDataProvider(MockMultivariateConstantDataProvider):
    """
    Mock data provider that always returns a vector of Inf values.

    Useful for testing algorithm robustness with infinite values.

    Parameters
    ----------
    dimensions : int
        Number of dimensions (variables) in each observation.
    length : int
        Number of observations to yield.
    """

    def __init__(self, dimensions: int, length: int) -> None:
        value = [float("inf")] * dimensions
        super().__init__(value, length)
