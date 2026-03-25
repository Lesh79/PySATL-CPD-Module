# -*- coding: ascii -*-

"""
Simple mock online algorithm implementation.
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


class MockOnlineAlgorithm[T](OnlineAlgorithm[T, MockAlgorithmConfiguration, MockAlgorithmState[T]]):
    """
    Flexible mock for testing online algorithms.

    Features:
    - Returns values from configured sequence
    - Learning period returns 0 during configured period
    - Tracks call history and process count
    - State contains last observation and process count

    Parameters
    ----------
    name : str, default="MockAlgorithm"
        Algorithm name for identification.
    learning_period_size : int, default=0
        Number of initial observations for learning period.
    return_sequence : list[Number]
        Sequence of return values for process() calls. Cycles through values.
    """

    def __init__(
        self,
        name: str = "MockAlgorithm",
        learning_period_size: int = 0,
        return_sequence: list[Number] | None = None,
    ) -> None:
        self._name = name
        self._config = MockAlgorithmConfiguration(
            learning_period_size=learning_period_size,
            return_sequence=return_sequence or [0.0],
        )
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
        Process a single observation and return detection statistic.

        During learning period (process_count < learning_period_size),
        returns 0. Otherwise returns next value from configured sequence,
        cycling through if necessary.

        Parameters
        ----------
        observation : T
            New observation to process.

        Returns
        -------
        Number
            Detection statistic value.
        """
        self._process_count += 1
        self._last_observation = observation
        self._call_history.append(observation)

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
    ) -> "MockOnlineAlgorithm[T]":
        """
        Recreate the algorithm from configuration and optional state.

        Parameters
        ----------
        configuration : MockAlgorithmConfiguration
            Configuration to use.
        state : MockAlgorithmState | None, optional
            Optional state to restore.

        Returns
        -------
        MockOnlineAlgorithm[T]
            New algorithm instance.
        """
        algorithm = cls(
            name="MockAlgorithm",
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
            f"learning_period_size={self._config.learning_period_size}, "
            f"process_count={self._process_count})"
        )
