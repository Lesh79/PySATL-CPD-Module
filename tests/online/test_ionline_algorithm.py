"""
Tests for online algorithm interface and base classes.
"""

import pytest

from pysatl_cpd._typing import Number
from pysatl_cpd.online.ionline_algorithm import (
    OnlineAlgorithm,
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmState,
)


class ConcreteAlgorithm(OnlineAlgorithm[float]):
    """Concrete implementation of OnlineAlgorithm for testing."""

    def __init__(
        self,
        name: str = "TestAlgorithm",
        learning_period_size: int = 0,
        process_return_value: Number = 0.0,
        expose_state: bool = True,
    ) -> None:
        self._name = name
        self._config = OnlineAlgorithmConfiguration(learning_period_size=learning_period_size)
        self._process_return_value = process_return_value
        self._expose_state = expose_state
        self._is_reset_called = False
        self._process_count = 0
        self._state = OnlineAlgorithmState(is_in_learning_period=(learning_period_size > 0))

    @property
    def name(self) -> str:
        return self._name

    @property
    def configuration(self) -> OnlineAlgorithmConfiguration:
        return self._config

    @property
    def state(self) -> OnlineAlgorithmState | None:
        if self._expose_state:
            return self._state
        return None

    def process(self, observation: float) -> Number:
        self._process_count += 1
        return self._process_return_value

    def reset(self) -> None:
        self._is_reset_called = True
        self._process_count = 0


class TestOnlineAlgorithmState:
    """Test suite for OnlineAlgorithmState dataclass."""

    def test_default_values(self) -> None:
        """Test default values for OnlineAlgorithmState."""
        state = OnlineAlgorithmState()

        assert state.is_in_learning_period is False

    def test_custom_values(self) -> None:
        """Test setting custom values in OnlineAlgorithmState."""
        state = OnlineAlgorithmState(is_in_learning_period=True)

        assert state.is_in_learning_period is True

    def test_immutability(self) -> None:
        """Test that OnlineAlgorithmState is frozen and immutable."""
        state = OnlineAlgorithmState(is_in_learning_period=False)

        with pytest.raises(AttributeError):
            state.is_in_learning_period = True  # type: ignore


class TestOnlineAlgorithmConfiguration:
    """Test suite for OnlineAlgorithmConfiguration dataclass."""

    def test_default_values(self) -> None:
        """Test default values for OnlineAlgorithmConfiguration."""
        config = OnlineAlgorithmConfiguration()

        assert config.learning_period_size == 0

    def test_custom_values(self) -> None:
        """Test setting custom values in OnlineAlgorithmConfiguration."""
        config = OnlineAlgorithmConfiguration(learning_period_size=100)

        assert config.learning_period_size == 100

    def test_immutability(self) -> None:
        """Test that OnlineAlgorithmConfiguration is frozen and immutable."""
        config = OnlineAlgorithmConfiguration(learning_period_size=10)

        with pytest.raises(AttributeError):
            config.learning_period_size = 20  # type: ignore


class TestOnlineAlgorithm:
    """Test suite for OnlineAlgorithm abstract base class."""

    def test_cannot_instantiate_abstract_class(self) -> None:
        """Test that OnlineAlgorithm cannot be instantiated directly."""
        with pytest.raises(TypeError):
            OnlineAlgorithm()  # type: ignore

    def test_concrete_implementation_works(self) -> None:
        """Test that concrete implementation can be instantiated."""
        algorithm = ConcreteAlgorithm()

        assert algorithm is not None

    def test_name_property(self) -> None:
        """Test name property returns correct value."""
        algorithm = ConcreteAlgorithm(name="CustomAlgorithm")

        assert algorithm.name == "CustomAlgorithm"

    def test_configuration_property(self) -> None:
        """Test configuration property returns correct configuration."""
        algorithm = ConcreteAlgorithm(learning_period_size=50)

        config = algorithm.configuration
        assert isinstance(config, OnlineAlgorithmConfiguration)
        assert config.learning_period_size == 50

    def test_state_property_default_none(self) -> None:
        """Test state property returns None when not exposed."""
        algorithm = ConcreteAlgorithm(expose_state=False)

        assert algorithm.state is None

    def test_state_property_returns_state(self) -> None:
        """Test state property returns state object when exposed."""
        algorithm = ConcreteAlgorithm(expose_state=True)

        state = algorithm.state
        assert isinstance(state, OnlineAlgorithmState)

    def test_process_method(self) -> None:
        """Test process method returns expected value."""
        expected_value: Number = 0.75
        algorithm = ConcreteAlgorithm(process_return_value=expected_value)

        result = algorithm.process(5.0)

        assert result == expected_value

    def test_reset_method(self) -> None:
        """Test reset method is callable."""
        algorithm = ConcreteAlgorithm()

        # Verify reset can be called without errors
        algorithm.reset()

        # Verify reset was called (using internal flag)
        assert algorithm._is_reset_called is True

    def test_process_multiple_observations(self) -> None:
        """Test processing multiple observations."""
        algorithm = ConcreteAlgorithm()

        algorithm.process(1.0)
        algorithm.process(2.0)
        algorithm.process(3.0)

        assert algorithm._process_count == 3

    def test_repr_with_default_configuration(self) -> None:
        """Test string representation with default configuration."""
        algorithm = ConcreteAlgorithm(name="MyAlgo")

        repr_str = repr(algorithm)
        assert repr_str == "MyAlgo(OnlineAlgorithmConfiguration(learning_period_size=0))"

    def test_repr_with_custom_configuration(self) -> None:
        """Test string representation with custom configuration."""
        algorithm = ConcreteAlgorithm(name="MyAlgo", learning_period_size=25)

        repr_str = repr(algorithm)
        assert repr_str == "MyAlgo(OnlineAlgorithmConfiguration(learning_period_size=25))"

    def test_state_snapshot_independence(self) -> None:
        """Test that state snapshots are independent from algorithm state."""
        algorithm = ConcreteAlgorithm()

        state1 = algorithm.state
        # Create a new state (simulating external modification)
        new_state = OnlineAlgorithmState(is_in_learning_period=True)

        # Algorithm's state should remain unchanged
        assert algorithm.state is not state1 or algorithm.state != new_state

    def test_multiple_algorithms_independence(self) -> None:
        """Test that different algorithm instances are independent."""
        algo1 = ConcreteAlgorithm(name="Algo1", learning_period_size=10)
        algo2 = ConcreteAlgorithm(name="Algo2", learning_period_size=20)

        assert algo1.configuration.learning_period_size == 10
        assert algo2.configuration.learning_period_size == 20
        assert algo1.name == "Algo1"
        assert algo2.name == "Algo2"
