"""
Mock constant data provider for testing.
"""

from collections.abc import Iterator

from pysatl_cpd._typing import Number
from pysatl_cpd.data_providers import DataProvider


class MockConstantDataProvider(DataProvider[Number]):
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

    def __init__(self, value: Number, length: int) -> None:
        self._value = value
        self._length = length
        self._call_count = 0

    def __iter__(self) -> Iterator[Number]:
        """Return iterator yielding constant value `length` times."""
        self._call_count += 1
        return iter([self._value] * self._length)

    def __len__(self) -> int:
        """Return number of observations."""
        return self._length

    def __getitem__(self, index: int) -> Number:
        """Get observation at specific index."""
        if 0 <= index < self._length:
            return self._value
        raise IndexError("Index out of range")

    def get_call_count(self) -> int:
        """Return number of times __iter__ was called."""
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the call counter."""
        self._call_count = 0

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MockConstantDataProvider(value={self._value}, length={self._length})"
