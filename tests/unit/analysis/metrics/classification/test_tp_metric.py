# -*- coding: ascii -*-

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any

import pytest

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.classification.tp_metric import TruePositiveMetric
from pysatl_cpd.core.detection_trace import DetectionTrace
from tests.mocks.analysis.labeled_data import MockLabeledData
from tests.mocks.core.detection_trace import MockDetectionTrace


def test_tp_metric_init_valid() -> None:
    """
    Test successful initialization of TruePositiveMetric with valid error margins.
    """

    margin = (5, 5)
    metric = TruePositiveMetric[DetectionTrace, LabeledData[Any]](error_margin=margin)

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
def test_tp_metric_init_invalid(invalid_margin: tuple[int, int]) -> None:
    """
    Test that TruePositiveMetric validation raises an error on negative margins.
    """

    with pytest.raises((ValueError, AttributeError), match="must be non-negative"):
        TruePositiveMetric(error_margin=invalid_margin)


@pytest.mark.parametrize(
    "detected, true_cps, margin, expected_tp",
    [
        # Scenario 1: Empty lists
        ([], [], (2, 2), 0.0),
        ([10], [], (2, 2), 0.0),
        ([], [10], (2, 2), 0.0),
        # Scenario 2: Perfect match
        ([10, 20], [10, 20], (0, 0), 2.0),
        # Scenario 3: Order independence (arrays are not sorted)
        # Detected and true points are provided in random orders.
        ([30, 10, 20], [20, 30, 10], (1, 1), 3.0),
        ([5, 15, 25], [26, 6, 14], (2, 2), 3.0),
        # Scenario 4: Multiple detections covering a SINGLE true change point
        # True CP is 10. Detections 9, 10, 11 all fall into the window.
        # It still counts as exactly ONE True Positive.
        ([9, 10, 11], [10], (2, 2), 1.0),
        # Scenario 5: One detection overlapping MULTIPLE true change points
        # Margin is (3,3). Detection 12 fits for both 10 [7..13] and 15 [12..18].
        # Due to greedy matching, it binds to 10. 15 remains unmatched. Output: 1 TP.
        ([12], [10, 15], (3, 3), 1.0),
        # Scenario 6: False Positives don't inflate True Positives
        # 10 is matched, 50 is out of bounds (False Positive). Output: 1 TP.
        ([10, 50], [10], (2, 2), 1.0),
        # Scenario 7: Mixed complex scenario
        # True CPs: 10, 20, 30.
        # Detected: 8 (matches 10), 22 (matches 20), 40 (FP), 45 (FP).
        # 30 is missed (FN).
        ([45, 8, 40, 22], [30, 10, 20], (2, 2), 2.0),
    ],
)
def test_tp_metric_compute(
    detected: Sequence[int], true_cps: Sequence[int], margin: tuple[int, int], expected_tp: float
) -> None:
    """
    Test the `compute` classmethod for correct True Positive calculation.
    """

    result = TruePositiveMetric.compute(detected_changes=detected, true_changes=true_cps, error_margin=margin)
    assert result == expected_tp


def test_tp_metric_evaluate() -> None:
    """
    Test the `evaluate` instance method using LabeledData and DetectionTrace mocks.
    It should extract the arrays, call `compute`, and return a float.
    """

    margin = (2, 2)
    metric = TruePositiveMetric[MockDetectionTrace, MockLabeledData](error_margin=margin)

    trace_mock = MockDetectionTrace(detected_change_points=[9, 21])
    data_mock = MockLabeledData(change_points=[10, 20, 30])

    result = metric.evaluate(trace=trace_mock, data=data_mock)

    # 9 matches 10, 21 matches 20, 30 is missed.
    # We expect exactly 2.0 TPs, and the type must be float.
    assert result == 2.0
    assert isinstance(result, float)
