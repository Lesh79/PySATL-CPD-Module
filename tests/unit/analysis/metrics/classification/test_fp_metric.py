# -*- coding: ascii -*-

"""
Tests for FalsePositiveMetric class.
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
from pysatl_cpd.analysis.metrics.classification.fp_metric import FalsePositiveMetric
from pysatl_cpd.core.detection_trace import DetectionTrace
from tests.mocks.analysis.labeled_data import MockLabeledData
from tests.mocks.core.detection_trace import MockDetectionTrace


def test_fp_metric_init_valid() -> None:
    """
    Test successful initialization of FalsePositiveMetric with valid error margins.
    """

    margin = (3, 3)
    metric: FalsePositiveMetric[DetectionTrace, LabeledData[Any]] = FalsePositiveMetric(error_margin=margin)

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
def test_fp_metric_init_invalid(invalid_margin: tuple[int, int]) -> None:
    """
    Test that FalsePositiveMetric validation raises an error on negative margins.
    """

    with pytest.raises((ValueError, AttributeError), match="must be non-negative"):
        FalsePositiveMetric(error_margin=invalid_margin)


@pytest.mark.parametrize(
    "detected, true_cps, margin, expected_fp",
    [
        # Scenario 1: Empty lists
        ([], [], (2, 2), 0),
        ([], [10], (2, 2), 0),  # True CP missed (FN), but no false alarms -> 0 FP
        ([10], [], (2, 2), 1),  # Detection exists, but no true CPs -> 1 FP
        # Scenario 2: Perfect match
        ([10, 20], [10, 20], (0, 0), 0),
        # Scenario 3: Order independence (arrays are not sorted)
        # 10 matches 10, 30 matches 30. 50 is out of bounds -> 1 FP.
        ([50, 30, 10], [10, 20, 30], (1, 1), 1),
        # Scenario 4: Multiple detections covering a SINGLE true change point
        # True CP is 10. Detections 9, 10, 11 all fall into the window.
        # Based on the base ClassificationMetric match logic, all these are captured
        # by CP 10. Thus, none are left "unmatched", so FP = 0.
        ([9, 10, 11], [10], (2, 2), 0),
        # Scenario 5: Detections completely outside any true CP margin
        # 8 and 12 match 10. 25 and 40 don't match anything -> 2 FPs.
        ([8, 12, 25, 40], [10], (2, 2), 2),
        # Scenario 6: Mixed complex scenario
        # True CPs: 10, 30.
        # Detected: 10 (matches 10), 15 (FP), 31 and 32 (matches 30), 40 (FP), 45 (FP).
        # We expect 3 FPs (15, 40, 45).
        ([40, 10, 15, 31, 32, 45], [30, 10], (2, 2), 3),
    ],
)
def test_fp_metric_compute(
    detected: Sequence[int], true_cps: Sequence[int], margin: tuple[int, int], expected_fp: int
) -> None:
    """
    Test the `compute` classmethod for correct False Positive calculation.
    """

    result = FalsePositiveMetric.compute(detected_changes=detected, true_changes=true_cps, error_margin=margin)
    assert result == expected_fp


def test_fp_metric_evaluate() -> None:
    """
    Test the `evaluate` instance method using LabeledData and DetectionTrace mocks.
    It should extract the arrays, call `compute`, and return a float.
    """

    margin = (2, 2)
    metric: FalsePositiveMetric[MockDetectionTrace, MockLabeledData] = FalsePositiveMetric(error_margin=margin)

    # 10 matches 10.
    # 20 matches 20.
    # 50, 60, 70 do not match anything (False Positives).
    trace_mock = MockDetectionTrace(detected_change_points=[10, 20, 50, 60, 70])
    data_mock = MockLabeledData(change_points=[10, 20, 30])

    result = metric.evaluate(trace=trace_mock, data=data_mock)

    # We expect exactly 3.0 FPs (points 50, 60, 70), and the type must be float.
    assert result == 3.0
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
    result_unsorted = FalsePositiveMetric.compute(detected, true_cps, margin)

    # Compute on strictly sorted lists
    result_sorted = FalsePositiveMetric.compute(sorted_detected, sorted_true, margin)

    # The result must be identical
    assert result_unsorted == result_sorted
