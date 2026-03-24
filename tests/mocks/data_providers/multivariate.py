"""
Mock multivariate data provider for testing.
"""

from collections.abc import Iterator, Sequence

from pysatl_cpd._typing import Number
from pysatl_cpd.data_providers import DataProvider


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
