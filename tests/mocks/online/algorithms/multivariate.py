"""
Mock multivariate online algorithm implementation.
"""

from dataclasses import dataclass
from typing import Any

from pysatl_cpd._typing import Number
from pysatl_cpd.online.ionline_algorithm import OnlineAlgorithm
from tests.mocks.online.algorithms.base import MockAlgorithmConfiguration, MockAlgorithmState

"""
Mock multivariate online algorithm implementation.
"""


@dataclass(frozen=True, kw_only=True)
class MockMultivariateAlgorithmState(MockAlgorithmState):
    """
    Mock multivariate algorithm state for testing.

    Extends base state with multivariate-specific fields.

    Parameters
    ----------
    last_observation_vector : list[Number] | None, default=None
        Last observation vector processed.
    dimensions : int, default=0
        Number of dimensions in the multivariate data.
    """

    last_observation_vector: list[Number] | None = None
    dimensions: int = 0


@dataclass(frozen=True, kw_only=True)
class MockMultivariateAlgorithmConfiguration(MockAlgorithmConfiguration):
    """
    Mock multivariate algorithm configuration for testing.

    Extends base configuration with multivariate-specific parameters.

    Parameters
    ----------
    expected_dimensions : int, default=0
        Expected number of dimensions. If > 0, validation will check
        that incoming observation vectors match this dimension.
    """

    expected_dimensions: int = 0


class MockMultivariateOnlineAlgorithm(
    OnlineAlgorithm[list[Number], MockMultivariateAlgorithmConfiguration, MockMultivariateAlgorithmState]
):
    """
    Mock implementation of OnlineAlgorithm for multivariate data testing.

    This mock processes observation vectors (lists of numbers) and provides
    configurable behavior for testing multivariate change-point detection.

    Parameters
    ----------
    name : str, default="MultivariateTestAlgorithm"
        Algorithm name for identification.
    learning_period_size : int, default=0
        Number of initial observations for learning period.
    expected_dimensions : int, default=0
        Expected number of dimensions. If > 0, validates observations.
    process_return_value : Number, default=0.0
        Value returned by process() method.
    expose_state : bool, default=True
        Whether to expose state via state property.
    process_return_sequence : list[Number] | None, default=None
        Sequence of return values for process() calls. If provided, cycles
        through values; if None, uses process_return_value for all calls.
    reset_clears_process_count : bool, default=True
        Whether reset() clears the process count in state.
    """

    def __init__(
        self,
        name: str = "MultivariateTestAlgorithm",
        learning_period_size: int = 0,
        expected_dimensions: int = 0,
        process_return_value: Number = 0.0,
        expose_state: bool = True,
        process_return_sequence: list[Number] | None = None,
        reset_clears_process_count: bool = True,
    ) -> None:
        self._name = name
        self._config = MockMultivariateAlgorithmConfiguration(
            learning_period_size=learning_period_size,
            expected_dimensions=expected_dimensions,
            custom_param=0,
            custom_string="default",
        )
        self._process_return_value = process_return_value
        self._process_return_sequence = process_return_sequence
        self._expose_state = expose_state
        self._reset_clears_process_count = reset_clears_process_count
        self._is_reset_called = False
        self._process_count = 0
        self._last_observation_vector: list[Number] | None = None
        self._call_history: list[list[Number]] = []
        self._state = MockMultivariateAlgorithmState(
            is_in_learning_period=(learning_period_size > 0),
            process_count=0,
            last_observation=None,
            last_observation_vector=None,
            dimensions=expected_dimensions,
        )

    @property
    def name(self) -> str:
        """Return the algorithm name."""
        return self._name

    @property
    def configuration(self) -> MockMultivariateAlgorithmConfiguration:
        """Return the algorithm configuration."""
        return self._config

    @property
    def state(self) -> MockMultivariateAlgorithmState | None:
        """Return the current algorithm state snapshot."""
        if self._expose_state:
            return MockMultivariateAlgorithmState(
                is_in_learning_period=self._state.is_in_learning_period,
                process_count=self._process_count,
                last_observation=self._last_observation_vector[-1] if self._last_observation_vector else None,
                last_observation_vector=self._last_observation_vector.copy() if self._last_observation_vector else None,
                dimensions=self._config.expected_dimensions,
                custom_data=self._state.custom_data.copy() if self._state.custom_data else {},
            )
        return None

    def process(self, observation: list[Number]) -> Number:
        """
        Process a single observation vector and return detection statistic.

        Parameters
        ----------
        observation : list[Number]
            New observation vector to process.

        Returns
        -------
        Number
            Detection statistic value.

        Raises
        ------
        ValueError
            If expected_dimensions > 0 and observation length doesn't match.
        """
        # Validate dimensions if configured
        if self._config.expected_dimensions > 0 and len(observation) != self._config.expected_dimensions:
            raise ValueError(f"Expected {self._config.expected_dimensions} dimensions, got {len(observation)}")

        self._process_count += 1
        self._last_observation_vector = observation.copy()
        self._call_history.append(observation.copy())

        # Determine return value
        if self._process_return_sequence is not None:
            idx = (self._process_count - 1) % len(self._process_return_sequence)
            return self._process_return_sequence[idx]

        return self._process_return_value

    def reset(self) -> None:
        """
        Reset the algorithm to its initial state.

        This method clears accumulated state and optionally resets
        the process counter.
        """
        self._is_reset_called = True
        if self._reset_clears_process_count:
            self._process_count = 0
        self._last_observation_vector = None
        # Note: call_history is not cleared to allow inspection

    def get_process_count(self) -> int:
        """Return the number of times process() was called."""
        return self._process_count

    def get_reset_called(self) -> bool:
        """Return whether reset() was called."""
        return self._is_reset_called

    def get_call_history(self) -> list[list[Number]]:
        """
        Return the history of observation vectors passed to process().

        Returns
        -------
        list[list[Number]]
            List of copies of observation vectors.
        """
        return [obs.copy() for obs in self._call_history]

    def get_last_observation_vector(self) -> list[Number] | None:
        """Return the last observation vector processed."""
        return self._last_observation_vector.copy() if self._last_observation_vector is not None else None

    def set_custom_state_data(self, key: str, value: Any) -> None:
        """
        Set custom data in algorithm state for testing.

        Parameters
        ----------
        key : str
            Key for custom data.
        value : Any
            Value to store in state.
        """
        self._state.custom_data[key] = value

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"name={self._name!r}, "
            f"learning_period_size={self._config.learning_period_size}, "
            f"expected_dimensions={self._config.expected_dimensions}, "
            f"process_count={self._process_count})"
        )


class MockMultivariateOnlineAlgorithmWithSequence(MockMultivariateOnlineAlgorithm):
    """
    Mock multivariate algorithm that returns a sequence of values for process() calls.

    This is a convenience subclass for testing with predefined response sequences.

    Parameters
    ----------
    return_sequence : list[Number]
        Sequence of return values for consecutive process() calls.
    name : str, default="MultivariateSequenceAlgorithm"
        Algorithm name.
    learning_period_size : int, default=0
        Learning period size.
    expected_dimensions : int, default=0
        Expected number of dimensions.
    expose_state : bool, default=True
        Whether to expose state.
    """

    def __init__(
        self,
        return_sequence: list[Number],
        name: str = "MultivariateSequenceAlgorithm",
        learning_period_size: int = 0,
        expected_dimensions: int = 0,
        expose_state: bool = True,
    ) -> None:
        super().__init__(
            name=name,
            learning_period_size=learning_period_size,
            expected_dimensions=expected_dimensions,
            process_return_value=0.0,
            expose_state=expose_state,
            process_return_sequence=return_sequence,
        )


class MockMultivariateOnlineAlgorithmWithStateSequence(MockMultivariateOnlineAlgorithm):
    """
    Mock multivariate algorithm that returns custom state snapshots.

    This algorithm cycles through a predefined sequence of states
    each time the state property is accessed.

    Parameters
    ----------
    state_sequence : list[MockMultivariateAlgorithmState]
        Sequence of states to return on consecutive state property accesses.
    name : str, default="MultivariateStateSequenceAlgorithm"
        Algorithm name.
    learning_period_size : int, default=0
        Learning period size.
    expected_dimensions : int, default=0
        Expected number of dimensions.
    """

    def __init__(
        self,
        state_sequence: list[MockMultivariateAlgorithmState],
        name: str = "MultivariateStateSequenceAlgorithm",
        learning_period_size: int = 0,
        expected_dimensions: int = 0,
    ) -> None:
        super().__init__(
            name=name,
            learning_period_size=learning_period_size,
            expected_dimensions=expected_dimensions,
            process_return_value=0.0,
            expose_state=True,
        )
        self._state_sequence = state_sequence
        self._state_index = 0

    @property
    def state(self) -> MockMultivariateAlgorithmState | None:
        """Return the next state from sequence."""
        if self._state_index < len(self._state_sequence):
            state = self._state_sequence[self._state_index]
            self._state_index += 1
            return state
        return None

    def get_state_sequence_index(self) -> int:
        """Return the current index in state sequence."""
        return self._state_index

    def reset_state_sequence(self) -> None:
        """Reset the state sequence index to beginning."""
        self._state_index = 0
