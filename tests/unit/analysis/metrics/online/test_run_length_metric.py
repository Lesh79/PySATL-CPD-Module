# -*- coding: ascii -*-

"""
Tests for RunLengthMetric class.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import warnings

import hypothesis.strategies as st
import pytest
from hypothesis import given

from pysatl_cpd.analysis.metrics.online.run_length_metric import RunLengthMetric
from tests.mocks.analysis.labeled_data import MockLabeledData
from tests.mocks.core.online.online_detection_trace import MockOnlineDetectionTrace


@pytest.mark.parametrize(
    "detected, expected_run_lengths",
    [
        # Scenario 1: No detections
        ([], []),
        # Scenario 2: Single detection at step 5
        # Distance: 5 - 0 = 5
        ([5], [5]),
        # Scenario 3: Multiple detections in sorted order
        # Distances: (5-0), (12-5), (20-12)
        ([5, 12, 20], [5, 7, 8]),
        # Scenario 4: Multiple detections in random order
        # The metric must sort them internally.
        # Sorted: [10, 30, 35] -> Distances: (10-0), (30-10), (35-30)
        ([35, 10, 30], [10, 20, 5]),
        # Scenario 5: Detection at the very first step
        ([1, 2, 3], [1, 1, 1]),
    ],
)
def test_run_length_metric_evaluate(
    detected: list[int],
    expected_run_lengths: list[int],
) -> None:
    """
    Test the `evaluate` method for correct calculation of distances between detections.
    """
    warnings.filterwarnings("ignore")
    metric: RunLengthMetric[MockOnlineDetectionTrace, MockLabeledData] = RunLengthMetric()

    trace_mock = MockOnlineDetectionTrace(detected_change_points=detected)
    # Ground truth data should be ignored by this metric
    data_mock = MockLabeledData(change_points=[999])

    result = metric.evaluate(trace=trace_mock, data=data_mock)

    assert isinstance(result, list)
    assert result == expected_run_lengths


def test_run_length_metric_ignores_ground_truth() -> None:
    """
    Explicitly verify that changing ground truth data does not affect run lengths.
    """
    metric: RunLengthMetric[MockOnlineDetectionTrace, MockLabeledData] = RunLengthMetric()
    trace = MockOnlineDetectionTrace(detected_change_points=[10, 25])

    res1 = metric.evaluate(trace, MockLabeledData(change_points=[10, 25]))
    res2 = metric.evaluate(trace, MockLabeledData(change_points=[5, 50, 100]))
    res3 = metric.evaluate(trace, MockLabeledData(change_points=[]))

    assert res1 == res2 == res3 == [10, 15]


@given(detected=st.lists(st.integers(min_value=1, max_value=10000), unique=True))
def test_run_length_metric_invariants(detected: list[int]) -> None:
    """
    Hypothesis property-based test for RunLengthMetric invariants.
    """
    warnings.filterwarnings("ignore")
    metric: RunLengthMetric[MockOnlineDetectionTrace, MockLabeledData] = RunLengthMetric()
    trace = MockOnlineDetectionTrace(detected_change_points=detected)
    data = MockLabeledData(change_points=[])

    result = metric.evaluate(trace, data)

    # Invariant 1: Number of run lengths equals number of detections
    assert len(result) == len(detected)

    if result:
        # Invariant 2: All run lengths must be positive (since detections are unique and >= 1)
        assert all(rl > 0 for rl in result)

        # Invariant 3: The sum of run lengths must equal the index of the last detection
        assert sum(result) == max(detected)
