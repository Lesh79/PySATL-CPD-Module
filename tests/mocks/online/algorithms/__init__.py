"""
Mock online algorithm implementations for testing.

This module provides mock implementations of OnlineAlgorithm for testing
change-point detection components.
"""

from tests.mocks.online.algorithms.base import MockAlgorithmConfiguration, MockAlgorithmState
from tests.mocks.online.algorithms.edge import (
    MockOnlineAlgorithmErrorInjector,
    MockOnlineAlgorithmNoState,
    MockOnlineAlgorithmWithCustomConfig,
    MockOnlineAlgorithmWithLearningPeriod,
)
from tests.mocks.online.algorithms.multivariate import (
    MockMultivariateAlgorithmConfiguration,
    MockMultivariateAlgorithmState,
    MockMultivariateOnlineAlgorithm,
    MockMultivariateOnlineAlgorithmWithSequence,
    MockMultivariateOnlineAlgorithmWithStateSequence,
)
from tests.mocks.online.algorithms.sequence import MockOnlineAlgorithmWithSequence
from tests.mocks.online.algorithms.simple import MockOnlineAlgorithm
from tests.mocks.online.algorithms.state_sequence import MockOnlineAlgorithmWithStateSequence

__all__ = [
    "MockAlgorithmState",
    "MockAlgorithmConfiguration",
    "MockOnlineAlgorithm",
    "MockOnlineAlgorithmWithSequence",
    "MockOnlineAlgorithmWithStateSequence",
    "MockMultivariateAlgorithmState",
    "MockMultivariateAlgorithmConfiguration",
    "MockMultivariateOnlineAlgorithm",
    "MockMultivariateOnlineAlgorithmWithSequence",
    "MockMultivariateOnlineAlgorithmWithStateSequence",
    "MockOnlineAlgorithmNoState",
    "MockOnlineAlgorithmErrorInjector",
    "MockOnlineAlgorithmWithLearningPeriod",
    "MockOnlineAlgorithmWithCustomConfig",
]
