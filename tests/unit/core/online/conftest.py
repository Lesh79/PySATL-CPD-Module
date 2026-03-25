# -*- coding: ascii -*-

"""
Shared fixtures for online algorithm tests.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import pytest

from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionStepResult
from pysatl_cpd.core.typedefs import Number
from tests.mocks.algorithms.online import MockAlgorithmState
from tests.mocks.core.data_providers import MockUnivariateDataProvider


@pytest.fixture
def sample_data() -> MockUnivariateDataProvider:
    """Create sample data provider for testing."""
    return MockUnivariateDataProvider([1.0, 2.0, 3.0, 4.0])


@pytest.fixture
def sample_steps() -> list[OnlineDetectionStepResult[MockAlgorithmState[Number]]]:
    """Create sample step results for testing."""
    state1: MockAlgorithmState[Number] = MockAlgorithmState[Number](
        process_count=1,
        last_observation=1.0,
    )
    state2: MockAlgorithmState[Number] = MockAlgorithmState[Number](
        process_count=2,
        last_observation=2.0,
    )

    return [
        OnlineDetectionStepResult(
            step_num=0,
            is_change_point=False,
            is_force_change_point=False,
            is_in_skip_period=False,
            detection_function=0.1,
            processing_time=0.001,
            algorithm_state=state1,
        ),
        OnlineDetectionStepResult(
            step_num=1,
            is_change_point=True,
            is_force_change_point=False,
            is_in_skip_period=False,
            detection_function=0.9,
            processing_time=0.002,
            algorithm_state=state2,
        ),
        OnlineDetectionStepResult(
            step_num=2,
            is_change_point=False,
            is_force_change_point=False,
            is_in_skip_period=True,
            detection_function=0.0,
            processing_time=0.0,
            algorithm_state=None,
        ),
        OnlineDetectionStepResult(
            step_num=3,
            is_change_point=False,
            is_force_change_point=True,
            is_in_skip_period=False,
            detection_function=1.2,
            processing_time=0.003,
            algorithm_state=None,
        ),
    ]


@pytest.fixture
def basic_data() -> list[Number]:
    """Basic test data sequence."""
    return [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]


@pytest.fixture
def no_detection_sequence() -> list[Number]:
    """Detection values that never exceed threshold."""
    return [0.1, 0.2, 0.3, 0.4, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]


@pytest.fixture
def single_detection_sequence() -> list[Number]:
    """Detection values with one peak exceeding threshold."""
    return [0.1, 0.2, 0.3, 0.4, 0.9, 0.5, 0.4, 0.3, 0.2, 0.1]


@pytest.fixture
def multiple_detection_sequence() -> list[Number]:
    """Detection values with multiple peaks."""
    return [0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9, 0.1, 0.9]


@pytest.fixture
def state_evolution_sequence() -> list[MockAlgorithmState[Number]]:
    """Sequence of states that evolve over time."""
    return [
        MockAlgorithmState[Number](process_count=1, last_observation=1.0),
        MockAlgorithmState[Number](process_count=2, last_observation=2.0),
        MockAlgorithmState[Number](process_count=3, last_observation=3.0),
        MockAlgorithmState[Number](process_count=4, last_observation=4.0),
        MockAlgorithmState[Number](process_count=5, last_observation=5.0),
    ]
