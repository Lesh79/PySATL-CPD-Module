# -*- coding: ascii -*-

"""
Mock edge case data providers for testing.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Iterator

from pysatl_cpd.core.data_providers import DataProvider


class MockEmptyDataProvider[T](DataProvider[T]):
    """Mock data provider that yields no observations."""

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


class MockSingleObservationProvider[T](DataProvider[T]):
    """Mock data provider with a single observation."""

    def __init__(self, observation: T) -> None:
        self._observation = observation
        self._call_count = 0

    def __iter__(self) -> Iterator[T]:
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
