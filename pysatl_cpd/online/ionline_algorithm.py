"""
Interface for online change-point detection algorithms.

This module defines the abstract :class:`OnlineAlgorithm` protocol used by
solvers and concrete detector implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TypeVar

from pysatl_cpd._typing import Number

T = TypeVar("T")


@dataclass(kw_only=True, frozen=True)
class OnlineAlgorithmState:
    """
    Immutable state snapshot of an online change-point detection algorithm.

    This class captures the internal state of an algorithm at a specific point
    in time. Being frozen and immutable ensures state consistency when used
    across different contexts or for debugging purposes.

    Parameters
    ----------
    is_in_learning_period : bool, default=False
        Indicates whether the algorithm is currently in its initial learning
        phase where change-point detection may be disabled or adapted.
    """

    is_in_learning_period: bool = False


@dataclass(kw_only=True, frozen=True)
class OnlineAlgorithmConfiguration:
    """
    Configuration parameters for an online change-point detection algorithm.

    This class holds static configuration settings that define the algorithm's
    behavior. Being frozen ensures configuration immutability after creation.

    Parameters
    ----------
    learning_period_size : int, default=0
        Number of initial observations used for algorithm warm-up or parameter
        estimation before change-point detection begins.
    """

    learning_period_size: int = 0


class OnlineAlgorithm[T](ABC):
    """
    Abstract base class for online change-point detection algorithms.

    Implementations process observations sequentially, updating internal state
    and producing a scalar change-point statistic after each observation.
    Algorithms must support state reset and provide configuration access.

    Parameters
    ----------
    T : type
        Observation type accepted by the algorithm. For univariate data,
        this is typically a numeric scalar. For multivariate data, this is
        typically a one-dimensional array.

    Examples
    --------
    >>> class MyAlgorithm(OnlineAlgorithm[float]):
    ...     def __init__(self) -> None:
    ...         self._state = OnlineAlgorithmState()
    ...         self._config = OnlineAlgorithmConfiguration(learning_period_size=10)
    ...
    ...     @property
    ...     def name(self) -> str:
    ...         return "MyAlgorithm"
    ...
    ...     @property
    ...     def configuration(self) -> OnlineAlgorithmConfiguration:
    ...         return self._config
    ...
    ...     def process(self, observation: float) -> Number:
    ...         return 0.0
    ...
    ...     def reset(self) -> None:
    ...         self._state = OnlineAlgorithmState()
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Human-readable name of the algorithm.

        Returns
        -------
        str
            Algorithm identifier suitable for logging and display.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def configuration(self) -> OnlineAlgorithmConfiguration:
        """
        Configuration parameters of the algorithm.

        Returns
        -------
        OnlineAlgorithmConfiguration
            Immutable configuration object containing algorithm settings.
        """
        raise NotImplementedError

    @property
    def state(self) -> OnlineAlgorithmState | None:
        """
        Current internal state snapshot of the algorithm.

        Returns
        -------
        OnlineAlgorithmState | None
            Immutable state snapshot, or None if state is not exposed.
        """
        return None

    @abstractmethod
    def process(self, observation: T) -> Number:
        """
        Process a single observation and return change-point statistic.

        This method updates the algorithm's internal state with the new
        observation and computes the current change-point detection statistic.

        Parameters
        ----------
        observation : T
            New observation to incorporate into the algorithm's state.

        Returns
        -------
        Number
            Current value of the change-point statistic. Higher values indicate
            higher likelihood of a change-point occurrence.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset the algorithm to its initial state.

        This method clears all accumulated state and returns the algorithm
        to the same condition as after initialization. Concrete implementations
        must provide this capability to enable proper solver behavior after
        change-point detections.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        """
        Return a string representation of the algorithm.

        Returns
        -------
        str
            String combining algorithm name and its configuration.
        """
        return f"{self.name}({self.configuration})"
