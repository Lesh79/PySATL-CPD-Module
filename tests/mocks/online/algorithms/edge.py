"""
Edge case mock online algorithm implementations.
"""

from pysatl_cpd._typing import Number
from pysatl_cpd.online.ionline_algorithm import OnlineAlgorithm
from tests.mocks.online.algorithms.base import MockAlgorithmConfiguration, MockAlgorithmState
from tests.mocks.online.algorithms.simple import MockOnlineAlgorithm


class MockOnlineAlgorithmNoState(OnlineAlgorithm[Number, MockAlgorithmConfiguration, MockAlgorithmState]):
    """
    Mock algorithm that never exposes state (state property returns None).

    Useful for testing components that handle algorithms without state exposure.

    Parameters
    ----------
    name : str, default="NoStateAlgorithm"
        Algorithm name.
    learning_period_size : int, default=0
        Learning period size.
    process_return_value : Number, default=0.0
        Value returned by process() method.
    process_return_sequence : list[Number] | None, default=None
        Sequence of return values for process() calls.
    """

    def __init__(
        self,
        name: str = "NoStateAlgorithm",
        learning_period_size: int = 0,
        process_return_value: Number = 0.0,
        process_return_sequence: list[Number] | None = None,
    ) -> None:
        self._name = name
        self._config = MockAlgorithmConfiguration(
            learning_period_size=learning_period_size,
            custom_param=0,
            custom_string="default",
        )
        self._process_return_value = process_return_value
        self._process_return_sequence = process_return_sequence
        self._process_count = 0
        self._call_history: list[Number] = []

    @property
    def name(self) -> str:
        """Return the algorithm name."""
        return self._name

    @property
    def configuration(self) -> MockAlgorithmConfiguration:
        """Return the algorithm configuration."""
        return self._config

    @property
    def state(self) -> None:
        """Always return None (state not exposed)."""
        return None

    def process(self, observation: Number) -> Number:
        """Process observation and return detection statistic."""
        self._process_count += 1
        self._call_history.append(observation)

        # Determine return value
        if self._process_return_sequence is not None:
            idx = (self._process_count - 1) % len(self._process_return_sequence)
            return self._process_return_sequence[idx]

        return self._process_return_value

    def reset(self) -> None:
        """Reset algorithm state."""
        self._process_count = 0
        self._call_history = []

    def get_process_count(self) -> int:
        """Return number of process calls."""
        return self._process_count

    def get_call_history(self) -> list[Number]:
        """Return call history."""
        return self._call_history.copy()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}(name={self._name!r}, learning_period_size={self._config.learning_period_size})"
        )


class MockOnlineAlgorithmErrorInjector(OnlineAlgorithm[Number, MockAlgorithmConfiguration, MockAlgorithmState]):
    """
    Mock algorithm that raises exceptions on process() or reset() calls.

    Useful for testing error handling in solvers and other components.

    Parameters
    ----------
    error_on_process_call : int | None, default=1
        Which process() call number should raise an exception (1-indexed).
        If None, never raises on process().
    error_on_reset : bool, default=False
        Whether reset() should raise an exception.
    error_to_raise : Exception, default=RuntimeError
        Exception to raise when error condition is triggered.
    process_return_value : Number, default=0.0
        Value returned by process() when not raising.
    name : str, default="ErrorAlgorithm"
        Algorithm name.
    learning_period_size : int, default=0
        Learning period size.
    """

    def __init__(
        self,
        error_on_process_call: int | None = 1,
        error_on_reset: bool = False,
        error_to_raise: Exception = RuntimeError("Mock algorithm error"),
        process_return_value: Number = 0.0,
        name: str = "ErrorAlgorithm",
        learning_period_size: int = 0,
    ) -> None:
        self._name = name
        self._config = MockAlgorithmConfiguration(
            learning_period_size=learning_period_size,
            custom_param=0,
            custom_string="default",
        )
        self._process_return_value = process_return_value
        self._error_on_process_call = error_on_process_call
        self._error_on_reset = error_on_reset
        self._error_to_raise = error_to_raise
        self._process_count = 0
        self._reset_called = False
        self._call_history: list[Number] = []

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
        """Return None for simplicity."""
        return None

    def process(self, observation: Number) -> Number:
        """Process observation, optionally raising an exception."""
        self._process_count += 1
        self._call_history.append(observation)

        # Check if we should raise on this call
        if self._error_on_process_call is not None and self._process_count == self._error_on_process_call:
            raise self._error_to_raise

        return self._process_return_value

    def reset(self) -> None:
        """Reset algorithm, optionally raising an exception."""
        self._reset_called = True
        if self._error_on_reset:
            raise self._error_to_raise
        self._process_count = 0
        self._call_history = []

    def get_process_count(self) -> int:
        """Return number of process calls."""
        return self._process_count

    def get_reset_called(self) -> bool:
        """Return whether reset was called."""
        return self._reset_called

    def get_call_history(self) -> list[Number]:
        """Return call history."""
        return self._call_history.copy()


class MockOnlineAlgorithmWithLearningPeriod(MockOnlineAlgorithm):
    """
    Mock algorithm that respects learning period behavior.

    Returns 0 during learning period and configured value after.
    The learning period resets after each reset() call.

    Parameters
    ----------
    learning_period_size : int, default=10
        Number of observations in learning period.
    post_learning_value : Number, default=1.0
        Value returned after learning period.
    name : str, default="LearningPeriodAlgorithm"
        Algorithm name.
    expose_state : bool, default=True
        Whether to expose state.
    process_return_sequence : list[Number] | None, default=None
        Optional sequence of return values to use after learning period.
    """

    def __init__(
        self,
        learning_period_size: int = 10,
        post_learning_value: Number = 1.0,
        name: str = "LearningPeriodAlgorithm",
        expose_state: bool = True,
        process_return_sequence: list[Number] | None = None,
    ) -> None:
        super().__init__(
            name=name,
            learning_period_size=learning_period_size,
            process_return_value=post_learning_value,
            expose_state=expose_state,
            process_return_sequence=process_return_sequence,
        )
        self._post_learning_value = post_learning_value
        self._process_return_sequence = process_return_sequence
        self._learning_period_size = learning_period_size
        self._observations_since_reset = 0

    def process(self, observation: Number) -> Number:
        """
        Process observation, returning 0 during learning period.

        The learning period counts observations since the last reset.

        Returns
        -------
        Number
            0 if observations_since_reset < learning_period_size,
            otherwise post_learning_value or value from sequence.
        """
        # Increment counter since last reset
        self._observations_since_reset += 1

        # Call parent to update counts and history
        super().process(observation)

        # During learning period, always return 0
        if self._observations_since_reset <= self._learning_period_size:
            return 0

        # After learning period, use sequence if provided
        if self._process_return_sequence is not None:
            # Calculate index based on steps after learning period
            idx = (self._observations_since_reset - self._learning_period_size - 1) % len(self._process_return_sequence)
            return self._process_return_sequence[idx]

        return self._post_learning_value

    def reset(self) -> None:
        """Reset algorithm, resetting the observation counter for learning period."""
        super().reset()
        self._observations_since_reset = 0


class MockOnlineAlgorithmWithCustomConfig(OnlineAlgorithm[Number, MockAlgorithmConfiguration, MockAlgorithmState]):
    """
    Mock algorithm with custom configuration parameters.

    Useful for testing components that access algorithm configuration.

    Parameters
    ----------
    name : str, default="CustomConfigAlgorithm"
        Algorithm name.
    learning_period_size : int, default=0
        Learning period size.
    custom_param : int, default=42
        Custom integer parameter.
    custom_string : str, default="test"
        Custom string parameter.
    process_return_value : Number, default=0.0
        Value returned by process().
    """

    def __init__(
        self,
        name: str = "CustomConfigAlgorithm",
        learning_period_size: int = 0,
        custom_param: int = 42,
        custom_string: str = "test",
        process_return_value: Number = 0.0,
    ) -> None:
        self._name = name
        self._config = MockAlgorithmConfiguration(
            learning_period_size=learning_period_size,
            custom_param=custom_param,
            custom_string=custom_string,
        )
        self._process_return_value = process_return_value
        self._process_count = 0
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
        """Return current state."""
        return MockAlgorithmState(
            is_in_learning_period=self._state.is_in_learning_period,
            process_count=self._process_count,
            last_observation=self._call_history[-1] if self._call_history else None,
        )

    def process(self, observation: Number) -> Number:
        """Process observation and return detection statistic."""
        self._process_count += 1
        self._call_history.append(observation)
        return self._process_return_value

    def reset(self) -> None:
        """Reset algorithm state."""
        self._process_count = 0
        self._call_history = []

    def get_process_count(self) -> int:
        """Return number of process calls."""
        return self._process_count

    def get_call_history(self) -> list[Number]:
        """Return call history."""
        return self._call_history.copy()

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"name={self._name!r}, "
            f"custom_param={self._config.custom_param}, "
            f"custom_string={self._config.custom_string!r})"
        )
