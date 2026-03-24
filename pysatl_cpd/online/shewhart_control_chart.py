"""
Shewhart control chart algorithm for online change-point detection.

This module provides :class:`ShewhartControlChart`, an online detector that
tracks running mean and variance, computing a standardized deviation of a
sliding-window mean from the global running mean.
"""

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from pysatl_cpd._typing import Number
from pysatl_cpd.online.ionline_algorithm import (
    OnlineAlgorithm,
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmState,
)


@dataclass(kw_only=True, frozen=True)
class ShewhartControlChartState(OnlineAlgorithmState):
    """
    State snapshot of the Shewhart control chart algorithm.

    This class captures the internal state of the algorithm at a specific point
    in time, allowing for state inspection and debugging.

    Parameters
    ----------
    is_in_learning_period : bool, default=False
        Indicates whether the algorithm is still in the initial learning phase.
    mean : float
        Current running mean estimate.
    variance : float
        Current running variance estimate.
    standard_deviation : float
        Current running standard deviation estimate.
    samples_count : int
        Number of observations processed so far.
    window_mean : float
        Current sliding window mean.
    window_sum : float
        Current sum of values in the sliding window.
    window_size : int
        Size of the sliding window.
    window_contents : list[Number]
        Current contents of the sliding window in order.
    """

    mean: Number = 0.0
    variance: Number = 0.0
    standard_deviation: Number = 0.0
    samples_count: int = 0
    window_mean: Number = 0.0
    window_sum: Number = 0.0
    window_size: int = 0
    window_contents: list[Number] = field(default_factory=list)


@dataclass(kw_only=True, frozen=True)
class ShewhartControlChartConfiguration(OnlineAlgorithmConfiguration):
    """
    Configuration parameters for the Shewhart control chart algorithm.

    Parameters
    ----------
    learning_period_size : int, default=0
        Number of initial observations used for training before non-zero
        statistics are emitted. Inherited from OnlineAlgorithmConfiguration.
    window_size : int
        Size of the sliding window used to compute the local mean statistic.

    Raises
    ------
    ValueError
        If ``learning_period_size`` is not positive.
    ValueError
        If ``window_size`` is not positive.
    ValueError
        If ``window_size`` is greater than ``learning_period_size``.
    """

    window_size: int = 0

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.learning_period_size <= 0:
            raise ValueError(f"learning_period_size must be positive, got {self.learning_period_size}")

        if self.window_size <= 0:
            raise ValueError(f"window_size must be positive, got {self.window_size}")

        if self.window_size > self.learning_period_size:
            raise ValueError(
                f"window_size ({self.window_size}) must be less than or equal to "
                f"learning_period_size ({self.learning_period_size})"
            )

    def __repr__(self) -> str:
        return f"w = {self.window_size}"


class ShewhartControlChart(OnlineAlgorithm[Number, ShewhartControlChartConfiguration, ShewhartControlChartState]):
    """
    Shewhart control chart with sliding-window statistic.

    This algorithm maintains running estimates of mean and variance, and
    computes a standardized deviation between the sliding-window mean and
    the global running mean. The statistic follows the formula:

    .. math::
        S_t = \\frac{|\\bar{x}_w - \\mu| \\sqrt{w}}{\\sigma}

    where:
    - :math:`\\bar{x}_w` is the mean of the last `window_size` observations
    - :math:`\\mu` is the running mean of all observations
    - :math:`w` is the window size
    - :math:`\\sigma` is the running standard deviation

    Parameters
    ----------
    learning_period_size : int
        Number of initial observations used for training before non-zero
        statistics are emitted. Must be positive.
    window_size : int
        Size of the sliding window used to compute the local mean statistic.
        Must be positive and less than or equal to learning_period_size.

    Raises
    ------
    ValueError
        If ``learning_period_size`` or ``window_size`` is not positive.
    ValueError
        If ``window_size`` is greater than ``learning_period_size``.

    Examples
    --------
    >>> chart = ShewhartControlChart(learning_period_size=50, window_size=10)
    >>> for obs in data:
    ...     statistic = chart.process(obs)
    ...     if statistic > threshold:
    ...         print(f"Change detected at {step}")
    ...         chart.reset()
    """

    def __init__(self, learning_period_size: int, window_size: int) -> None:
        """
        Initialize the Shewhart control chart.

        Parameters
        ----------
        learning_period_size : int
            Number of initial observations for training.
        window_size : int
            Size of the sliding window for local mean computation.
        """
        self._configuration = ShewhartControlChartConfiguration(
            learning_period_size=learning_period_size, window_size=window_size
        )

        self._mean: Number = 0.0
        self._previous_mean: Number = 0.0
        self._variance: Number = 0.0
        self._standard_deviation: Number = 0.0
        self._samples_count: int = 0
        self._window: deque[Number] = deque[Number](maxlen=window_size)
        self._window_sum: Number = 0.0
        self._window_mean: Number = 0.0

    @property
    def name(self) -> str:
        """
        Return the short algorithm name.

        Returns
        -------
        str
            Algorithm identifier: ``"ShewhartControlChart"``.
        """
        return "ShewhartControlChart"

    @property
    def configuration(self) -> ShewhartControlChartConfiguration:
        """
        Return the algorithm configuration.

        Returns
        -------
        OnlineAlgorithmConfiguration
            Configuration object containing learning_period_size and window_size.
        """
        return self._configuration

    @property
    def state(self) -> ShewhartControlChartState:
        """
        Return the current algorithm state snapshot.

        Returns
        -------
        ShewhartControlChartState | None
            Immutable state snapshot with current estimates and window contents.
        """
        return ShewhartControlChartState(
            is_in_learning_period=self._samples_count <= self._configuration.learning_period_size,
            mean=self._mean,
            variance=self._variance,
            standard_deviation=self._standard_deviation,
            samples_count=self._samples_count,
            window_mean=self._window_mean,
            window_sum=self._window_sum,
            window_size=self._configuration.window_size,
            window_contents=list(self._window),
        )

    def process(self, observation: Number) -> Number:
        """
        Process a single observation and return chart statistic value.

        Parameters
        ----------
        observation : Number
            New scalar observation (Python ``float`` or NumPy floating).

        Returns
        -------
        Number
            Standardized absolute deviation between the sliding-window mean
            and running mean. Returns ``0.0`` during training period or when
            the running standard deviation is zero.
        """
        self._samples_count += 1

        # Compute detection statistic after learning period
        detection_func: Number = 0.0
        if self._samples_count > self._configuration.learning_period_size and self._standard_deviation > 0:
            detection_func = np.float64(
                np.abs(self._window_mean - self._mean)
                * (self._configuration.window_size**0.5)
                / self._standard_deviation
            )

        # Maintain sliding window
        if len(self._window) == self._configuration.window_size:
            self._window_sum -= self._window.popleft()

        # Update running statistics
        self._mean, self._previous_mean = self._update_mean(observation)
        self._variance = self._update_variance(observation)
        self._standard_deviation = np.sqrt(self._variance)

        # Update sliding window
        self._window.append(observation)
        self._window_sum += observation
        self._window_mean = self._window_sum / self._configuration.window_size

        return detection_func

    def reset(self) -> None:
        """
        Reset the algorithm to its initial state.

        Clears all internal statistics, counters, and the sliding window,
        returning the algorithm to the same state as after initialization.
        """
        self._mean = 0.0
        self._previous_mean = 0.0
        self._variance = 0.0
        self._standard_deviation = 0.0
        self._samples_count = 0
        self._window.clear()
        self._window_sum = 0.0
        self._window_mean = 0.0

    def _update_mean(self, observation: Number) -> tuple[Number, Number]:
        """
        Update running mean using current observation.

        Parameters
        ----------
        observation : Number
            New scalar observation.

        Returns
        -------
        tuple[float, float]
            Pair containing (new_mean, previous_mean).
        """
        new_mean = self._mean + (observation - self._mean) / self._samples_count
        return new_mean, self._mean

    def _update_variance(self, observation: Number) -> Number:
        """
        Update running variance estimate using Welford's algorithm.

        Parameters
        ----------
        observation : Number
            New scalar observation.

        Returns
        -------
        float
            Updated running variance.
        """
        return (
            self._variance
            + ((observation - self._previous_mean) * (observation - self._mean) - self._variance) / self._samples_count
        )
