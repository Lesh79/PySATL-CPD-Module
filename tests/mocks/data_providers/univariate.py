"""
Mock univariate data provider for testing.
"""

from collections.abc import Iterator, Sequence

from pysatl_cpd._typing import Number
from pysatl_cpd.data_providers import DataProvider


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

    def __init__(self, data: Sequence[Number]) -> None:
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
