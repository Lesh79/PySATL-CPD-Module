"""
Mock single observation data provider for testing.
"""

from collections.abc import Iterator

from pysatl_cpd._typing import Number
from pysatl_cpd.data_providers import DataProvider


class MockSingleObservationProvider(DataProvider[Number]):
    """
    Mock data provider with a single observation.

    Useful for testing boundary conditions.
    """

    def __init__(self, observation: Number) -> None:
        self._observation = observation
        self._call_count = 0

    def __iter__(self) -> Iterator[Number]:
        """Return iterator over single observation."""
        self._call_count += 1
        return iter([self._observation])

    def __len__(self) -> int:
        """Return 1."""
        return 1

    def get_call_count(self) -> int:
        """Return number of times __iter__ was called."""
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the call counter."""
        self._call_count = 0

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MockSingleObservationProvider(value={self._observation})"
