"""
Mock empty data provider for testing.
"""

from collections.abc import Iterator

from pysatl_cpd.data_providers import DataProvider


class MockEmptyDataProvider[T](DataProvider[T]):
    """
    Mock data provider that yields no observations.

    Useful for testing edge cases with empty data streams.
    """

    def __init__(self) -> None:
        self._call_count = 0

    def __iter__(self) -> Iterator[T]:
        """Return empty iterator."""
        self._call_count += 1
        return iter([])

    def __len__(self) -> int:
        """Return 0."""
        return 0

    def get_call_count(self) -> int:
        """Return number of times __iter__ was called."""
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the call counter."""
        self._call_count = 0

    def __repr__(self) -> str:
        """Return string representation."""
        return "MockEmptyDataProvider()"
