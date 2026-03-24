"""
Tests for online algorithm interface and base classes.

This test suite verifies the contract of the abstract OnlineAlgorithm class
and its associated dataclasses, not the behavior of any concrete implementation.
"""

import pytest

from pysatl_cpd._typing import Number
from pysatl_cpd.online.ionline_algorithm import (
    OnlineAlgorithm,
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmState,
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

        with pytest.raises(TypeError):
            MissingReset()  # type: ignore

    def test_state_property_has_default_implementation(self) -> None:
        """Verify that state property returns None by default."""

        class MinimalImplementation(OnlineAlgorithm[Number, OnlineAlgorithmConfiguration, OnlineAlgorithmState]):
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

        algorithm = MinimalImplementation()
        assert algorithm.state is None

    def test_repr_uses_name_and_configuration(self) -> None:
        """Verify that __repr__ returns string with name and configuration."""

        class CustomConfig(OnlineAlgorithmConfiguration):
            custom_value: int = 42

        class ImplementationWithRepr(OnlineAlgorithm[Number, CustomConfig, OnlineAlgorithmState]):
            def __init__(self) -> None:
                self._name = "TestAlgorithm"
                self._config = CustomConfig(learning_period_size=10)

            @property
            def name(self) -> str:
                return self._name

            @property
            def configuration(self) -> CustomConfig:
                return self._config

            def process(self, observation: Number) -> Number:
                return 0.0

            def reset(self) -> None:
                pass

        algorithm = ImplementationWithRepr()
        repr_str = repr(algorithm)

        assert "TestAlgorithm" in repr_str
        assert "learning_period_size=10" in repr_str
