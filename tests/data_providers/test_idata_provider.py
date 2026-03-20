"""
Tests for the abstract DataProvider interface.
"""

from collections.abc import Iterator
from typing import TypeVar

import pytest

from pysatl_cpd.data_providers.idata_provider import DataProvider

T = TypeVar("T")


class MockDataProvider(DataProvider[T]):
    """
    Mock implementation of DataProvider for testing purposes.

    Parameters
    ----------
    data : list[T]
        List of observations to yield during iteration.
    """

    def __init__(self, data: list[T]) -> None:
        self._data = data

    def __iter__(self) -> Iterator[T]:
        """Return an iterator over the provided data."""
        return iter(self._data)


class TestDataProvider:
    """Test suite for DataProvider abstract base class."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Verify that DataProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataProvider()  # type: ignore

    def test_concrete_implementation_must_implement_iter(self) -> None:
        """Verify that concrete classes must implement __iter__."""

        # Create a class that doesn't implement __iter__
        class IncompleteProvider(DataProvider[T]):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()  # type: ignore

    def test_mock_provider_with_integer_data(self) -> None:
        """Test MockDataProvider with integer observations."""
        data = [1, 2, 3, 4, 5]
        provider = MockDataProvider(data)

        # Verify iteration yields expected values
        result = list(provider)
        assert result == data

        # Verify multiple iterations work
        assert list(provider) == data
        assert list(provider) == data

    def test_mock_provider_with_float_data(self) -> None:
        """Test MockDataProvider with float observations."""
        data = [1.1, 2.2, 3.3, 4.4, 5.5]
        provider = MockDataProvider(data)

        result = list(provider)
        assert result == data

    def test_mock_provider_with_string_data(self) -> None:
        """Test MockDataProvider with non-numeric data (type parameter flexibility)."""
        data = ["a", "b", "c", "d"]
        provider = MockDataProvider(data)

        result = list(provider)
        assert result == data

    def test_mock_provider_with_empty_data(self) -> None:
        """Test MockDataProvider with empty data sequence."""
        provider = MockDataProvider[float]([])

        result = list(provider)
        assert result == []

    def test_provider_is_iterable(self) -> None:
        """Verify that DataProvider instances are properly iterable."""
        provider = MockDataProvider([1, 2, 3])

        # Test iterator protocol
        iterator = iter(provider)
        assert next(iterator) == 1
        assert next(iterator) == 2
        assert next(iterator) == 3

        with pytest.raises(StopIteration):
            next(iterator)
