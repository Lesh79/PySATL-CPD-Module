# -*- coding: ascii -*-

"""
Error-injecting mock online algorithm implementation.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.core.online.ionline_algorithm import OnlineAlgorithm
from pysatl_cpd.core.typedefs import Number
from tests.mocks.algorithms.online.base import (
    MockAlgorithmConfiguration,
    MockAlgorithmState,
)


class MockErrorOnlineAlgorithm[T](OnlineAlgorithm[T, MockAlgorithmConfiguration, MockAlgorithmState[T]]):
    """
    Mock algorithm that raises an exception on a specific process() call.

    Useful for testing error handling in solvers and other components.

    Parameters
    ----------
    error_on_call : int
        Which process() call number should raise an exception (1-indexed).
    error_to_raise : Exception
        Exception to raise when the call number is reached.
    name : str, default="ErrorAlgorithm"
        Algorithm name for identification.
    learning_period_size : int, default=0
        Number of initial observations for learning period.
    return_sequence : list[Number], default=[0.0]
        Sequence of return values for process() calls. Cycles through values.
    """

    def __init__(
        self,
        error_on_call: int,
        error_to_raise: Exception,
        name: str = "ErrorAlgorithm",
        learning_period_size: int = 0,
        return_sequence: list[Number] | None = None,
    ) -> None:
        self._name = name
        self._config = MockAlgorithmConfiguration(
            learning_period_size=learning_period_size,
            return_sequence=return_sequence or [0.0],
        )
        self._error_on_call = error_on_call
        self._error_to_raise = error_to_raise
        self._process_count = 0
        self._call_history: list[T] = []
        self._last_observation: T | None = None

    @property
    def name(self) -> str:
        """Return the algorithm name."""
        return self._name

    @property
    def configuration(self) -> MockAlgorithmConfiguration:
        """Return the algorithm configuration."""
        return self._config

    @property
    def state(self) -> MockAlgorithmState[T]:
        """Return the current algorithm state snapshot."""
        return MockAlgorithmState(
            is_in_learning_period=self._process_count < self._config.learning_period_size,
            last_observation=self._last_observation,
            process_count=self._process_count,
        )

    def process(self, observation: T) -> Number:
        """
        Process a single observation, raising exception on configured call.

        Parameters
        ----------
        observation : T
            New observation to process.

        Returns
        -------
        Number
            Detection statistic value.

        Raises
        ------
        Exception
            The configured exception when process() call number matches error_on_call.
        """
        self._process_count += 1
        self._last_observation = observation
        self._call_history.append(observation)

        # Check if we should raise on this call
        if self._process_count == self._error_on_call:
            raise self._error_to_raise

        # During learning period, return 0
        if self._process_count <= self._config.learning_period_size:
            return 0

        # After learning period, return from sequence
        idx = (self._process_count - self._config.learning_period_size - 1) % len(self._config.return_sequence)
        return self._config.return_sequence[idx]

    def reset(self) -> None:
        """Reset the algorithm to its initial state."""
        self._process_count = 0
        self._last_observation = None

    @classmethod
    def recreate(
        cls,
        configuration: MockAlgorithmConfiguration,
        state: MockAlgorithmState[T] | None = None,
    ) -> "MockErrorOnlineAlgorithm[T]":
        """
        Recreate the algorithm from configuration and optional state.

        Note: This method cannot recreate the error configuration as it
        was not stored in configuration. Use with caution.

        Parameters
        ----------
        configuration : MockAlgorithmConfiguration
            Configuration to use.
        state : MockAlgorithmState | None, optional
            Optional state to restore.

        Returns
        -------
        MockErrorOnlineAlgorithm[T]
            New algorithm instance (error configuration lost).
        """
        algorithm = cls(
            error_on_call=1,  # Default, will need to be set manually
            error_to_raise=RuntimeError("Recreated error"),
            name="ErrorAlgorithm",
            learning_period_size=configuration.learning_period_size,
            return_sequence=configuration.return_sequence,
        )
        if state is not None:
            algorithm._process_count = state.process_count
            algorithm._last_observation = state.last_observation
        return algorithm

    def get_process_count(self) -> int:
        """Return the number of times process() was called."""
        return self._process_count

    def get_call_history(self) -> list[T]:
        """Return the history of observations passed to process()."""
        return self._call_history.copy()

    def get_last_observation(self) -> T | None:
        """Return the last observation processed."""
        return self._last_observation

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"{self.__class__.__name__}("
            f"name={self._name!r}, "
            f"error_on_call={self._error_on_call}, "
            f"learning_period_size={self._config.learning_period_size}, "
            f"process_count={self._process_count})"
        )
