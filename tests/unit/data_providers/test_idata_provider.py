"""
Tests for the abstract DataProvider interface.

This test suite verifies the contract that all DataProvider implementations
must satisfy, regardless of their concrete implementation details.
"""

import math
from typing import Any

import pytest

from pysatl_cpd.data_providers.idata_provider import DataProvider

# List of all provider fixture names for parametrized tests
ALL_PROVIDER_FIXTURES = [
    "univariate_provider",
    "multivariate_provider",
    "empty_provider",
    "single_observation_provider",
    "constant_provider",
    "nan_provider",
    "inf_provider",
    "zero_provider",
    "negative_provider",
    "multivariate_edge_provider",
]


class TestDataProviderAbstractBase:
    """Test the abstract DataProvider interface constraints."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Verify that DataProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataProvider()  # type: ignore

    def test_concrete_class_must_implement_iter(self) -> None:
        """Verify that concrete classes must implement __iter__."""

        # Create a class that doesn't implement __iter__
        class IncompleteProvider(DataProvider[Any]):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()  # type: ignore


class TestDataProviderContract:
    """Test the common contract that all DataProviders should satisfy."""

    @pytest.mark.parametrize("provider_fixture_name", ALL_PROVIDER_FIXTURES)
    def test_provider_is_iterable(self, request: pytest.FixtureRequest, provider_fixture_name: str) -> None:
        """Verify all providers are iterable (can call iter() on them)."""
        provider = request.getfixturevalue(provider_fixture_name)
        assert iter(provider) is not None

    @pytest.mark.parametrize("provider_fixture_name", ALL_PROVIDER_FIXTURES)
    def test_provider_supports_multiple_iterations(
        self, request: pytest.FixtureRequest, provider_fixture_name: str
    ) -> None:
        """Verify providers can be iterated multiple times."""
        provider = request.getfixturevalue(provider_fixture_name)

        # First iteration
        first_iter = list(provider)
        # Second iteration should work and produce same results
        second_iter = list(provider)

        assert len(first_iter) == len(second_iter)

    @pytest.mark.parametrize("provider_fixture_name", ALL_PROVIDER_FIXTURES)
    def test_provider_raises_stop_iteration_when_exhausted(
        self, request: pytest.FixtureRequest, provider_fixture_name: str
    ) -> None:
        """Verify that iterator raises StopIteration when exhausted."""
        provider = request.getfixturevalue(provider_fixture_name)
        iterator = iter(provider)

        # Exhaust the iterator
        for _ in iterator:
            pass

        # Next call should raise StopIteration
        with pytest.raises(StopIteration):
            next(iterator)

    def test_empty_provider_yields_no_data(self, empty_provider: Any) -> None:
        """Verify empty provider yields no observations."""
        result = list(empty_provider)
        assert result == []

    def test_single_observation_provider_yields_one_value(self, single_observation_provider: Any) -> None:
        """Verify single observation provider yields exactly one value."""
        result = list(single_observation_provider)
        assert len(result) == 1
        assert result[0] == 42

    def test_constant_provider_yields_constant_values(self, constant_provider: Any) -> None:
        """Verify constant provider yields same value for all observations."""
        result = list(constant_provider)
        assert len(result) == 10
        assert all(x == 5 for x in result)

    def test_univariate_provider_yields_numbers(self, univariate_provider: Any) -> None:
        """Verify univariate provider yields numeric values."""
        result = list(univariate_provider)
        assert result == [1, 2, 3]

    def test_multivariate_provider_yields_lists(self, multivariate_provider: Any) -> None:
        """Verify multivariate provider yields lists of numbers."""
        result = list(multivariate_provider)
        assert result == [[1, 2], [3, 4]]

    def test_nan_provider_yields_nan_values(self, nan_provider: Any) -> None:
        """Verify NaN provider yields NaN values."""
        result = list(nan_provider)
        assert len(result) == 5
        assert all(math.isnan(x) for x in result)

    def test_inf_provider_yields_inf_values(self, inf_provider: Any) -> None:
        """Verify Inf provider yields Inf values."""
        result = list(inf_provider)
        assert len(result) == 5
        assert all(math.isinf(x) for x in result)

    def test_zero_provider_yields_zeros(self, zero_provider: Any) -> None:
        """Verify zero provider yields zeros."""
        result = list(zero_provider)
        assert len(result) == 5
        assert all(x == 0 for x in result)

    def test_negative_provider_yields_negative_values(self, negative_provider: Any) -> None:
        """Verify negative provider yields negative values."""
        result = list(negative_provider)
        assert result == [-1, -2, -3]
        assert all(x < 0 for x in result)

    def test_multivariate_edge_provider_injects_nan_and_inf(self, multivariate_edge_provider: Any) -> None:
        """Verify multivariate edge provider injects NaN and Inf at specified positions."""
        result = list(multivariate_edge_provider)
        assert len(result) == 3
        # First observation, second dimension should be NaN
        assert math.isnan(result[0][1])
        # Second observation should be unchanged
        assert result[1] == [3, 4]
        # Third observation, first dimension should be Inf
        assert math.isinf(result[2][0])
        assert result[2][1] == 6
