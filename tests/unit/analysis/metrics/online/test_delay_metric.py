# -*- coding: ascii -*-

"""
Tests for DelayMetric class.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import warnings

import hypothesis.strategies as st
import pytest
from hypothesis import given

from pysatl_cpd.analysis.metrics.online.delay_metric import DelayMetric
from tests.mocks.analysis.labeled_data import MockLabeledData
from tests.mocks.core.online.online_detection_trace import MockOnlineDetectionTrace


def test_delay_metric_init() -> None:
    """
    Test successful initialization of DelayMetric.
    """
    max_delay = 50
    metric: DelayMetric[MockOnlineDetectionTrace, MockLabeledData] = DelayMetric(max_delay=max_delay)

    assert metric._max_delay == max_delay


def test_delay_metric_invalid_init() -> None:
    """
    Test invalid initialization of DelayMetric.
    """
    with pytest.raises(ValueError):
        DelayMetric(max_delay=-50)


@pytest.mark.parametrize(
    "detected, true_cps, max_delay, expected_delays",
    [
        # Scenario 1: Empty inputs
        ([], [], 10, []),
        ([5, 10], [], 10, []),  # False alarms, but no true CPs to calculate delay for
        ([], [10, 20], 10, [10, 10]),  # Missed CPs (FN) -> penalized with max_delay
        # Scenario 2: Perfect match (Zero delay)
        ([10, 20], [10, 20], 5, [0, 0]),
        # Scenario 3: Normal delays within max_delay
        # True CP 10 -> detected at 12 (delay 2)
        # True CP 20 -> detected at 25 (delay 5)
        ([12, 25], [10, 20], 10, [2, 5]),
        # Scenario 4: Detection exactly at max_delay boundary
        ([15], [10], 5, [5]),
        # Scenario 5: Early detections (Negative delay) are ignored
        # True CP is 10. Detection is 8.
        # Since margin is (0, max_delay), 8 is not matched to 10.
        # Therefore, 10 is considered missed and penalized.
        ([8], [10], 5, [5]),
        # Scenario 6: Multiple detections after true CP
        # True CP is 10. Detections are 11, 12, 13.
        # The algorithm should pick the FIRST one (minimum delay).
        ([13, 11, 12], [10], 5, [1]),
        # Scenario 7: Mixed complex scenario
        # True CPs: 10, 30, 50. Max delay: 10.
        # - 10 is detected at 12 (delay 2).
        # - 30 is detected at 42 (delay 12 > max_delay) -> unmatched -> penalized (10).
        # - 50 is detected early at 48 (ignored) and then at 55 (delay 5).
        ([12, 42, 48, 55], [10, 30, 50], 10, [2, 10, 5]),
        # Scenario 8: Greedy matching conflict inherited from base class
        # True CPs: 10, 12. Max delay: 5.
        # Detection: 13.
        # 13 binds to 10 (delay 3).
        # 12 has no detections left -> penalized (5).
        ([13], [10, 12], 5, [3, 5]),
        # Scenario 9: Preservation of input order
        # True CPs are unsorted: [30, 10]
        # Detections: [12, 35]
        # max_delay: 10
        # 10 matched to 12 (delay 2), 30 matched to 35 (delay 5)
        # Expected delays should follow input order: [5, 2]
        ([12, 35], [30, 10], 10, [5, 2]),
    ],
)
def test_delay_metric_evaluate(
    detected: list[int],
    true_cps: list[int],
    max_delay: int,
    expected_delays: list[int],
) -> None:
    """
    Test the `evaluate` method for correct delay calculation and penalty assignment.
    """
    warnings.filterwarnings("ignore")

    metric: DelayMetric[MockOnlineDetectionTrace, MockLabeledData] = DelayMetric(max_delay=max_delay)

    trace_mock = MockOnlineDetectionTrace(detected_change_points=detected)
    data_mock = MockLabeledData(change_points=true_cps)

    result = metric.evaluate(trace=trace_mock, data=data_mock)

    # 1. Ensure output is a list of integers
    assert isinstance(result, list)
    assert all(isinstance(d, int) for d in result)

    # 2. Ensure length exactly matches number of true CPs
    assert len(result) == len(true_cps)

    # 3. Ensure values match expected delays
    assert result == expected_delays


@given(
    detected=st.lists(st.integers(min_value=1, max_value=1000), unique=True).map(sorted),
    true_cps=st.lists(st.integers(min_value=1, max_value=1000), unique=True).map(sorted),
    max_delay=st.integers(min_value=0, max_value=100),
)
def test_delay_metric_invariants(detected: list[int], true_cps: list[int], max_delay: int) -> None:
    """
    Hypothesis property-based test to verify mathematical invariants of DelayMetric.
    """
    metric: DelayMetric[MockOnlineDetectionTrace, MockLabeledData] = DelayMetric(max_delay=max_delay)

    trace_mock = MockOnlineDetectionTrace(detected_change_points=detected)
    data_mock = MockLabeledData(change_points=true_cps)

    result = metric.evaluate(trace=trace_mock, data=data_mock)

    # Invariant 1: Result length is always equal to the number of true change points
    assert len(result) == len(true_cps)

    # Invariant 2: Every individual delay is bounded: 0 <= delay <= max_delay
    if result:
        assert all(0 <= delay <= max_delay for delay in result)
