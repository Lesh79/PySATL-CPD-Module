# -*- coding: ascii -*-

"""
Tests for online algorithm interface and base classes.

This test suite verifies the contract of the abstract OnlineAlgorithm class
and its associated dataclasses, not the behavior of any concrete implementation.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import pytest

from pysatl_cpd.core.online.ionline_algorithm import (
    OnlineAlgorithm,
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmState,
)
from pysatl_cpd.core.typedefs import Number
from tests.mocks.algorithms.online import (
    MockAlgorithmConfiguration,
    MockAlgorithmState,
    MockOnlineAlgorithm,
)


class TestOnlineAlgorithmState:
    """Test the OnlineAlgorithmState dataclass contract."""

    def test_state_has_is_in_learning_period_field(self) -> None:
        """Verify OnlineAlgorithmState has is_in_learning_period field."""
        state = OnlineAlgorithmState()
        assert hasattr(state, "is_in_learning_period")

    def test_state_default_values(self) -> None:
        """Test default values for OnlineAlgorithmState."""
        state = OnlineAlgorithmState()
        assert state.is_in_learning_period is False

    def test_state_custom_values(self) -> None:
        """Test setting custom values in OnlineAlgorithmState."""
        state = OnlineAlgorithmState(is_in_learning_period=True)
        assert state.is_in_learning_period is True

    def test_state_immutability(self) -> None:
        """Test that OnlineAlgorithmState is frozen and immutable."""
        state = OnlineAlgorithmState(is_in_learning_period=False)

        with pytest.raises(AttributeError):
            state.is_in_learning_period = True  # type: ignore


class TestOnlineAlgorithmConfiguration:
    """Test the OnlineAlgorithmConfiguration dataclass contract."""

    def test_config_has_learning_period_size_field(self) -> None:
        """Verify OnlineAlgorithmConfiguration has learning_period_size field."""
        config = OnlineAlgorithmConfiguration()
        assert hasattr(config, "learning_period_size")

    def test_config_default_values(self) -> None:
        """Test default values for OnlineAlgorithmConfiguration."""
        config = OnlineAlgorithmConfiguration()
        assert config.learning_period_size == 0

    def test_config_custom_values(self) -> None:
        """Test setting custom values in OnlineAlgorithmConfiguration."""
        config = OnlineAlgorithmConfiguration(learning_period_size=100)
        assert config.learning_period_size == 100

    def test_config_immutability(self) -> None:
        """Test that OnlineAlgorithmConfiguration is frozen and immutable."""
        config = OnlineAlgorithmConfiguration(learning_period_size=10)

        with pytest.raises(AttributeError):
            config.learning_period_size = 20  # type: ignore


class TestMockAlgorithmState:
    """Test the MockAlgorithmState dataclass contract."""

    def test_mock_state_has_required_fields(self) -> None:
        """Verify MockAlgorithmState has all required fields."""
        state: MockAlgorithmState[int] = MockAlgorithmState[int]()
        assert hasattr(state, "is_in_learning_period")
        assert hasattr(state, "last_observation")
        assert hasattr(state, "process_count")

    def test_mock_state_default_values(self) -> None:
        """Test default values for MockAlgorithmState."""
        state: MockAlgorithmState[int] = MockAlgorithmState[int]()
        assert state.is_in_learning_period is False
        assert state.last_observation is None
        assert state.process_count == 0

    def test_mock_state_custom_values(self) -> None:
        """Test setting custom values in MockAlgorithmState."""
        state: MockAlgorithmState[int] = MockAlgorithmState[int](
            is_in_learning_period=True,
            last_observation=42,
            process_count=5,
        )
        assert state.is_in_learning_period is True
        assert state.last_observation == 42
        assert state.process_count == 5

    def test_mock_state_immutability(self) -> None:
        """Test that MockAlgorithmState is frozen and immutable."""
        state: MockAlgorithmState[int] = MockAlgorithmState[int]()

        with pytest.raises(AttributeError):
            state.process_count = 10  # type: ignore


class TestMockAlgorithmConfiguration:
    """Test the MockAlgorithmConfiguration dataclass contract."""

    def test_mock_config_has_required_fields(self) -> None:
        """Verify MockAlgorithmConfiguration has all required fields."""
        config: MockAlgorithmConfiguration = MockAlgorithmConfiguration(return_sequence=[0.0])
        assert hasattr(config, "learning_period_size")
        assert hasattr(config, "return_sequence")

    def test_mock_config_default_values(self) -> None:
        """Test default values for MockAlgorithmConfiguration."""
        config: MockAlgorithmConfiguration = MockAlgorithmConfiguration(return_sequence=[0.0])
        assert config.learning_period_size == 0
        assert config.return_sequence == [0.0]

    def test_mock_config_custom_values(self) -> None:
        """Test setting custom values in MockAlgorithmConfiguration."""
        config: MockAlgorithmConfiguration = MockAlgorithmConfiguration(
            learning_period_size=50,
            return_sequence=[0.1, 0.2, 0.3],
        )
        assert config.learning_period_size == 50
        assert config.return_sequence == [0.1, 0.2, 0.3]

    def test_mock_config_immutability(self) -> None:
        """Test that MockAlgorithmConfiguration is frozen and immutable."""
        config: MockAlgorithmConfiguration = MockAlgorithmConfiguration(return_sequence=[0.0])

        with pytest.raises(AttributeError):
            config.learning_period_size = 20  # type: ignore


class TestOnlineAlgorithmAbstractBase:
    """Test the abstract OnlineAlgorithm interface."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Verify that OnlineAlgorithm cannot be instantiated directly."""
        with pytest.raises(TypeError):
            OnlineAlgorithm()  # type: ignore

    def test_concrete_class_must_implement_name_property(self) -> None:
        """Verify that concrete classes must implement name property."""

        class MissingName(OnlineAlgorithm[Number, OnlineAlgorithmConfiguration, OnlineAlgorithmState]):
            @property
            def configuration(self) -> OnlineAlgorithmConfiguration:
                return OnlineAlgorithmConfiguration()

            def process(self, observation: Number) -> Number:
                return 0.0

            def reset(self) -> None:
                pass

            @classmethod
            def recreate(
                cls, config: OnlineAlgorithmConfiguration, state: OnlineAlgorithmState | None = None
            ) -> "MissingName":
                return cls()

        with pytest.raises(TypeError):
            MissingName()  # type: ignore

    def test_concrete_class_must_implement_configuration_property(self) -> None:
        """Verify that concrete classes must implement configuration property."""

        class MissingConfig(OnlineAlgorithm[Number, OnlineAlgorithmConfiguration, OnlineAlgorithmState]):
            @property
            def name(self) -> str:
                return "Test"

            def process(self, observation: Number) -> Number:
                return 0.0

            def reset(self) -> None:
                pass

            @classmethod
            def recreate(
                cls, config: OnlineAlgorithmConfiguration, state: OnlineAlgorithmState | None = None
            ) -> "MissingConfig":
                return cls()

        with pytest.raises(TypeError):
            MissingConfig()  # type: ignore

    def test_concrete_class_must_implement_process_method(self) -> None:
        """Verify that concrete classes must implement process method."""

        class MissingProcess(OnlineAlgorithm[Number, OnlineAlgorithmConfiguration, OnlineAlgorithmState]):
            @property
            def name(self) -> str:
                return "Test"

            @property
            def configuration(self) -> OnlineAlgorithmConfiguration:
                return OnlineAlgorithmConfiguration()

            def reset(self) -> None:
                pass

            @classmethod
            def recreate(
                cls, config: OnlineAlgorithmConfiguration, state: OnlineAlgorithmState | None = None
            ) -> "MissingProcess":
                return cls()

        with pytest.raises(TypeError):
            MissingProcess()  # type: ignore

    def test_concrete_class_must_implement_reset_method(self) -> None:
        """Verify that concrete classes must implement reset method."""

        class MissingReset(OnlineAlgorithm[Number, OnlineAlgorithmConfiguration, OnlineAlgorithmState]):
            @property
            def name(self) -> str:
                return "Test"

            @property
            def configuration(self) -> OnlineAlgorithmConfiguration:
                return OnlineAlgorithmConfiguration()

            def process(self, observation: Number) -> Number:
                return 0.0

            @classmethod
            def recreate(
                cls, config: OnlineAlgorithmConfiguration, state: OnlineAlgorithmState | None = None
            ) -> "MissingReset":
                return cls()

        with pytest.raises(TypeError):
            MissingReset()  # type: ignore

    def test_concrete_class_must_implement_recreate_method(self) -> None:
        """Verify that concrete classes must implement recreate method."""

        class MissingRecreate(OnlineAlgorithm[Number, OnlineAlgorithmConfiguration, OnlineAlgorithmState]):
            @property
            def name(self) -> str:
                return "Test"

            @property
            def configuration(self) -> OnlineAlgorithmConfiguration:
                return OnlineAlgorithmConfiguration()

            def process(self, observation: Number) -> Number:
                return 0.0

            def reset(self) -> None:
                pass

        with pytest.raises(TypeError):
            MissingRecreate()  # type: ignore

    def test_state_property_must_return_state(self) -> None:
        """Verify that state property must return a state (not None)."""
        algorithm: MockOnlineAlgorithm[int] = MockOnlineAlgorithm[int](
            name="Test",
            learning_period_size=0,
            return_sequence=[0.0],
        )
        state: MockAlgorithmState[int] = algorithm.state
        assert isinstance(state, MockAlgorithmState)
        assert state is not None

    def test_repr_uses_name_and_configuration(self) -> None:
        """Verify that __repr__ returns string with name and configuration."""
        algorithm: MockOnlineAlgorithm[int] = MockOnlineAlgorithm[int](
            name="TestAlgorithm",
            learning_period_size=10,
            return_sequence=[0.0],
        )
        repr_str: str = repr(algorithm)

        assert "TestAlgorithm" in repr_str
        assert "learning_period_size=10" in repr_str
