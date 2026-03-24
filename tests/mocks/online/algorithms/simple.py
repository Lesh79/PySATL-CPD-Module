"""
Simple mock online algorithm implementation.
"""

from typing import Any

from pysatl_cpd._typing import Number
from pysatl_cpd.online.ionline_algorithm import OnlineAlgorithm
from tests.mocks.online.algorithms.base import MockAlgorithmConfiguration, MockAlgorithmState


class MockOnlineAlgorithm(OnlineAlgorithm[Number, MockAlgorithmConfiguration, MockAlgorithmState]):
    """
    Simple mock implementation of OnlineAlgorithm for testing.

    This mock allows controlled testing of online detection components
    by providing configurable behavior for process() and reset() methods.

    Parameters
    ----------
    name : str, default="TestAlgorithm"
        Algorithm name for identification.
    learning_period_size : int, default=0
        Number of initial observations for learning period.
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
        name: str = "TestAlgorithm",
        learning_period_size: int = 0,
        process_return_value: Number = 0.0,
        expose_state: bool = True,
        process_return_sequence: list[Number] | None = None,
        reset_clears_process_count: bool = True,
    ) -> None:
        self._name = name
        self._config = MockAlgorithmConfiguration(
            learning_period_size=learning_period_size,
            custom_param=0,
            custom_string="default",
        )
        self._process_return_value = process_return_value
        self._process_return_sequence = process_return_sequence
        self._expose_state = expose_state
        self._reset_clears_process_count = reset_clears_process_count
        self._is_reset_called = False
        self._process_count = 0
        self._last_observation: Number | None = None
        self._call_history: list[Number] = []
        self._state = MockAlgorithmState(
            is_in_learning_period=(learning_period_size > 0),
            process_count=0,
            last_observation=None,
        )

    @property
    def name(self) -> str:
        """Return the algorithm name."""
        return self._name

    @property
    def configuration(self) -> MockAlgorithmConfiguration:
        """Return the algorithm configuration."""
        return self._config

    @property
    def state(self) -> MockAlgorithmState | None:
        """Return the current algorithm state snapshot."""
        if self._expose_state:
            return MockAlgorithmState(
                is_in_learning_period=self._state.is_in_learning_period,
                process_count=self._process_count,
                last_observation=self._last_observation,
                custom_data=self._state.custom_data.copy() if self._state.custom_data else {},
            )
        return None

    def process(self, observation: Number) -> Number:
        """
        Process a single observation and return detection statistic.

        Parameters
        ----------
        observation : Number
            New observation to process.

        Returns
        -------
        Number
            Detection statistic value.
        """
        self._process_count += 1
        self._last_observation = observation
        self._call_history.append(observation)

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
        self._last_observation = None
        # Note: call_history is not cleared to allow inspection

    def get_process_count(self) -> int:
        """Return the number of times process() was called."""
        return self._process_count

    def get_reset_called(self) -> bool:
        """Return whether reset() was called."""
        return self._is_reset_called

    def get_call_history(self) -> list[Number]:
        """Return the history of observations passed to process()."""
        return self._call_history.copy()

    def get_last_observation(self) -> Number | None:
        """Return the last observation processed."""
        return self._last_observation

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
            f"process_count={self._process_count})"
        )
