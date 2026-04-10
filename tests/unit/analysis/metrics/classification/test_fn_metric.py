# -*- coding: ascii -*-

"""
Tests for FalseNegativeMetric class.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any

import hypothesis.strategies as st
import pytest
from hypothesis import given

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.classification.fn_metric import FalseNegativeMetric
from pysatl_cpd.core.detection_trace import DetectionTrace
from tests.mocks.analysis.labeled_data import MockLabeledData
from tests.mocks.core.detection_trace import MockDetectionTrace


def test_fn_metric_init_valid() -> None:
    """
    Test successful initialization of FalseNegativeMetric with valid error margins.
    """

    margin = (5, 5)
    metric: FalseNegativeMetric[DetectionTrace, LabeledData[Any]] = FalseNegativeMetric(error_margin=margin)

    assert metric._error_margin == margin


@pytest.mark.parametrize(
    "invalid_margin",
    [
        (-1, 5),  # Left margin is negative
        (5, -2),  # Right margin is negative
        (-3, -3),  # Both margins are negative
    ],
    ids=["negative_left", "negative_right", "both_negative"],
)
def test_fn_metric_init_invalid(invalid_margin: tuple[int, int]) -> None:
    """
    Test that FalseNegativeMetric validation raises an error on negative margins.
    """
    with pytest.raises((ValueError, AttributeError), match="must be non-negative"):
        FalseNegativeMetric(error_margin=invalid_margin)


@pytest.mark.parametrize(
    "detected, true_cps, margin, expected_fn",
    [
        # Scenario 1: Empty lists
        ([], [], (2, 2), 0),
        ([10], [], (2, 2), 0),  # Detections exist, but no true CPs -> 0 FN (it's a False Positive)
        ([], [10], (2, 2), 1),  # 1 true CP, no detections -> 1 missed (1 FN)
        # Scenario 2: Perfect match
        ([10, 20], [10, 20], (0, 0), 0),
        # Scenario 3: Order independence (arrays are not sorted)
        # 10 matches 10, 30 matches 30. 20 is missed -> 1 FN.
        ([30, 10], [20, 30, 10], (1, 1), 1),
        # Scenario 4: Multiple detections covering a SINGLE true change point
        # True CP is 10. Detections 9, 10, 11 all fall into the window.
        # TP = 1. Since there is only 1 True CP, FN = 1 - 1 = 0.
        ([9, 10, 11], [10], (2, 2), 0),
        # Scenario 5: One detection overlapping MULTIPLE true change points
        # Margin is (3,3). Detection 12 fits for both 10 [7..13] and 15 [12..18].
        # Due to greedy matching, it binds to 10. 15 remains unmatched -> 1 FN.
        ([12], [10, 15], (3, 3), 1),
        # Scenario 6: False Positives don't inflate or reduce False Negatives
        # 10 is matched. 20 is missed (FN=1). 50 is out of bounds (FP, ignored here).
        ([10, 50], [10, 20], (2, 2), 1),
        # Scenario 7: Mixed complex scenario
        # True CPs: 10, 20, 30, 40.
        # Detected: 8 (matches 10), 40 (matches 40), 55 (FP).
        # 20 and 30 are missed -> 2 FN.
        ([55, 8, 40], [30, 10, 40, 20], (2, 2), 2),
    ],
)
def test_fn_metric_compute(
    detected: Sequence[int], true_cps: Sequence[int], margin: tuple[int, int], expected_fn: int
) -> None:
    """
    Test the `compute` classmethod for correct False Negative calculation.
    """

    result = FalseNegativeMetric.compute(detected_changes=detected, true_changes=true_cps, error_margin=margin)
    assert result == expected_fn


def test_fn_metric_evaluate() -> None:
    """
    Test the `evaluate` instance method using LabeledData and DetectionTrace mocks.
    It should extract the arrays, call `compute`, and return a float.
    """

    margin = (2, 2)
    metric: FalseNegativeMetric[MockDetectionTrace, MockLabeledData] = FalseNegativeMetric(error_margin=margin)

    # 9 matches 10, 21 matches 20.
    # 30 and 40 are missed (FN).
    # 99 is a False Positive.
    trace_mock = MockDetectionTrace(detected_change_points=[9, 21, 99])
    data_mock = MockLabeledData(change_points=[10, 20, 30, 40])

    result = metric.evaluate(trace=trace_mock, data=data_mock)

    # We expect exactly 2.0 FNs (points 30 and 40), and the type must be float.
    assert result == 2.0
    assert isinstance(result, float)


@given(
    detected=st.lists(st.integers(min_value=1, max_value=1000), unique=True),
    true_cps=st.lists(st.integers(min_value=1, max_value=1000), unique=True),
    margin=st.tuples(st.integers(min_value=0, max_value=100), st.integers(min_value=0, max_value=100)),
)
def test_tp_metric_order_independence(detected: list[int], true_cps: list[int], margin: tuple[int, int]) -> None:
    """
    Property-based test: The metric calculation must be completely independent
    of the order of elements in the detected and true change point arrays.
    """

    sorted_detected = sorted(detected)
    sorted_true = sorted(true_cps)

    # Compute on raw (potentially completely unsorted) lists
    result_unsorted = FalseNegativeMetric.compute(detected, true_cps, margin)

    # Compute on strictly sorted lists
    result_sorted = FalseNegativeMetric.compute(sorted_detected, sorted_true, margin)

    # The result must be identical
    assert result_unsorted == result_sorted
