# -*- coding: ascii -*-

"""
Tests for the abstract DataProvider interface.

This test suite verifies the contract that all DataProvider implementations
must satisfy, regardless of their concrete implementation details.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from typing import Any

import pytest

from pysatl_cpd.core.data_providers.idata_provider import DataProvider

# List of all provider fixture names for parametrized tests
ALL_PROVIDER_FIXTURES = [
    "univariate_provider",
    "multivariate_provider",
    "empty_provider",
    "single_observation_provider",
    "constant_provider",
    "zero_provider",
    "nan_provider",
    "inf_provider",
    "multivariate_constant_provider",
    "multivariate_zero_provider",
    "multivariate_nan_provider",
    "multivariate_inf_provider",
    "univariate_dirty_provider",
    "multivariate_dirty_provider",
]


class TestDataProviderAbstractBase:
    """Test the abstract DataProvider interface constraints."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Verify that DataProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            DataProvider()  # type: ignore

    def test_concrete_class_must_implement_iter(self) -> None:
        """Verify that concrete classes must implement __iter__."""

        class MissingIter(DataProvider[Any]):
            def __len__(self) -> int:
                return 0

        with pytest.raises(TypeError):
            MissingIter()  # type: ignore

    def test_concrete_class_must_implement_len(self) -> None:
        """Verify that concrete classes must implement __len__."""

        class MissingLen(DataProvider[Any]):
            def __iter__(self) -> Any:
                return iter([])

        with pytest.raises(TypeError):
            MissingLen()  # type: ignore


class TestDataProviderContract:
    """Test the common contract that all DataProviders should satisfy."""

    @pytest.mark.parametrize("provider_fixture_name", ALL_PROVIDER_FIXTURES)
    def test_provider_is_iterable(self, request: pytest.FixtureRequest, provider_fixture_name: str) -> None:
        """Verify all providers are iterable (can call iter() on them)."""
        provider = request.getfixturevalue(provider_fixture_name)
        assert iter(provider) is not None

    @pytest.mark.parametrize("provider_fixture_name", ALL_PROVIDER_FIXTURES)
    def test_provider_has_len(self, request: pytest.FixtureRequest, provider_fixture_name: str) -> None:
        """Verify all providers implement __len__ returning non-negative integer."""
        provider = request.getfixturevalue(provider_fixture_name)
        length = len(provider)
        assert isinstance(length, int)
        assert length >= 0

    @pytest.mark.parametrize("provider_fixture_name", ALL_PROVIDER_FIXTURES)
    def test_len_matches_iteration_count(self, request: pytest.FixtureRequest, provider_fixture_name: str) -> None:
        """Verify that __len__ returns the number of elements in iteration."""
        provider = request.getfixturevalue(provider_fixture_name)
        expected_len = len(provider)
        actual_len = len(list(provider))
        assert actual_len == expected_len

    @pytest.mark.parametrize("provider_fixture_name", ALL_PROVIDER_FIXTURES)
    def test_provider_supports_multiple_iterations(
        self, request: pytest.FixtureRequest, provider_fixture_name: str
    ) -> None:
        """Verify providers can be iterated multiple times."""
        provider = request.getfixturevalue(provider_fixture_name)

        first_iter = list(provider)
        second_iter = list(provider)

        assert len(first_iter) == len(second_iter)

    @pytest.mark.parametrize("provider_fixture_name", ALL_PROVIDER_FIXTURES)
    def test_provider_raises_stop_iteration_when_exhausted(
        self, request: pytest.FixtureRequest, provider_fixture_name: str
    ) -> None:
        """Verify that iterator raises StopIteration when exhausted."""
        provider = request.getfixturevalue(provider_fixture_name)
        iterator = iter(provider)

        for _ in iterator:
            pass

        with pytest.raises(StopIteration):
            next(iterator)

    @pytest.mark.parametrize("provider_fixture_name", ALL_PROVIDER_FIXTURES)
    def test_len_consistent_after_multiple_iterations(
        self, request: pytest.FixtureRequest, provider_fixture_name: str
    ) -> None:
        """Verify that __len__ returns same value after multiple iterations."""
        provider = request.getfixturevalue(provider_fixture_name)

        len_before = len(provider)
        list(provider)
        len_after = len(provider)

        assert len_before == len_after
