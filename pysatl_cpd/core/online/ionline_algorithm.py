# -*- coding: ascii -*-
"""
Interface for online change-point detection algorithms.

This module defines the abstract :class:`OnlineAlgorithm` protocol used by
solvers and concrete detector implementations.

Examples
--------
>>> from pysatl_cpd.core.typedefs import Number
>>> from dataclasses import dataclass
>>>
>>> @dataclass(frozen=True, kw_only=True)
... class MyConfig(OnlineAlgorithmConfiguration):
...     window_size: int = 10
...
>>> @dataclass(frozen=True, kw_only=True)
... class MyState(OnlineAlgorithmState):
...     values: list[float] = field(default_factory=list)
...
>>> class MyAlgorithm(OnlineAlgorithm[float, MyConfig, MyState]):
...     def __init__(self, config: MyConfig) -> None:
...         self._config = config
...         self._state = MyState()
...         self._name = "MyAlgorithm"
...
...     @property
...     def name(self) -> str:
...         return self._name
...
...     @property
...     def configuration(self) -> MyConfig:
...         return self._config
...
...     @property
...     def state(self) -> MyState:
...         return self._state
...
...     def process(self, observation: float) -> Number:
...         new_values = self._state.values + [observation]
...         if len(new_values) > self._config.window_size:
...             new_values = new_values[1:]
...         self._state = MyState(values=new_values)
...         return sum(new_values) / len(new_values) if new_values else 0.0
...
...     def reset(self) -> None:
...         self._state = MyState()
...
...     @classmethod
...     def recreate(cls, configuration: MyConfig, state: MyState | None = None) -> "MyAlgorithm":
...         algorithm = cls(configuration)
...         if state is not None:
...             algorithm._state = state
...         return algorithm
>>>
>>> config = MyConfig(window_size=5)
>>> algorithm = MyAlgorithm(config)
>>> algorithm.name
'MyAlgorithm'
>>> algorithm.process(5.0)
5.0
"""

__author__ = "Alexey Tatyanenko, Vladimir Kutuev, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from abc import ABC, abstractmethod
from dataclasses import dataclass

from pysatl_cpd.core.typedefs import Number


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


class OnlineAlgorithm[DataT, ConfigurationT: OnlineAlgorithmConfiguration, StateT: OnlineAlgorithmState](ABC):
    """
    Abstract base class for online change-point detection algorithms.

    Implementations process observations sequentially, updating internal state
    and producing a scalar change-point statistic after each observation.
    Algorithms must support state reset and provide configuration access.

    Parameters
    ----------
    DataT : type
        Observation type accepted by the algorithm. For univariate data,
        this is typically a numeric scalar. For multivariate data, this is
        typically a one-dimensional array.
    ConfigurationT : OnlineAlgorithmConfiguration
        Configuration type that extends the base configuration.
    StateT : OnlineAlgorithmState
        State type that extends the base state.
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
        raise NotImplementedError  # pragma: no cover

    @property
    @abstractmethod
    def configuration(self) -> ConfigurationT:
        """
        Configuration parameters of the algorithm.

        Returns
        -------
        ConfigurationT
            Immutable configuration object containing algorithm settings.
        """
        raise NotImplementedError  # pragma: no cover

    @property
    @abstractmethod
    def state(self) -> StateT:
        """
        Current internal state snapshot of the algorithm.

        Returns
        -------
        StateT
            Immutable state snapshot of the algorithm.
        """
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def process(self, observation: DataT) -> Number:
        """
        Process a single observation and return detection function value.

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
        raise NotImplementedError  # pragma: no cover

    @abstractmethod
    def reset(self) -> None:
        """
        Reset the algorithm to its initial state.

        This method clears all accumulated state and returns the algorithm
        to the same condition as after initialization. Concrete implementations
        must provide this capability to enable proper solver behavior after
        change-point detections.
        """
        raise NotImplementedError  # pragma: no cover

    @classmethod
    @abstractmethod
    def recreate(
        cls, configuration: ConfigurationT, state: StateT | None = None
    ) -> "OnlineAlgorithm[DataT, ConfigurationT, StateT]":
        """
        Recreate an algorithm instance from configuration and optional state.

        This factory method allows reconstructing an algorithm to a specific
        configuration and state, useful for serialization, checkpointing,
        or resetting to a known state.

        Parameters
        ----------
        configuration : ConfigurationT
            The configuration to use for the recreated algorithm.
        state : StateT | None, optional
            Optional state to restore. If None, the algorithm should be
            initialized to its default initial state.

        Returns
        -------
        OnlineAlgorithm[DataT, ConfigurationT, StateT]
            A new algorithm instance configured with the given parameters
            and optionally restored to the provided state.
        """
        raise NotImplementedError  # pragma: no cover

    def __repr__(self) -> str:
        """
        Return a string representation of the algorithm.

        Returns
        -------
        str
            String combining algorithm name and its configuration.
        """
        return f"{self.name}({self.configuration})"  # pragma: no cover
