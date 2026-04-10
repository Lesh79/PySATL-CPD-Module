# -*- coding: ascii -*-

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any

import pytest

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.core.detection_trace import DetectionTrace
from tests.mocks.analysis.metrics.classification.simple import MockClassificationMetric


def test_classification_metric_init_valid() -> None:
    """
    Test successful initialization of the metric with valid (non-negative) error margins.
    """
    margin = (2, 3)
    metric = MockClassificationMetric[DetectionTrace, LabeledData[Any]](error_margin=margin)

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
def test_classification_metric_init_invalid(invalid_margin: tuple[int, int]) -> None:
    """
    Test that initialization raises an AttributeError (or ValueError)
    when error_margin contains negative values.
    """
    with pytest.raises(ValueError, match="must be non-negative"):
        MockClassificationMetric(error_margin=invalid_margin)


@pytest.mark.parametrize(
    "detected, true_cps, margin, expected",
    [
        # 1. Empty input arrays
        ([], [], (1, 1), {}),
        ([5], [], (1, 1), {}),
        ([], [5], (1, 1), {5: set()}),
        # 2. Exact matches (no deviation)
        ([5], [5], (1, 1), {5: {5}}),
        # 3. Margin boundary conditions
        # True CP is 10, margin (2, 2) -> valid range [8, 12]
        ([7, 8, 12, 13], [10], (2, 2), {10: {8, 12}}),
        # 4. Asymmetric margins
        # True CP is 10, margin (1, 3) -> valid range [9, 13]
        ([8, 9, 13, 14], [10], (1, 3), {10: {9, 13}}),
        # 5. Multiple detections falling into the window of a single True CP
        ([9, 10, 11], [10], (1, 1), {10: {9, 10, 11}}),
        # 6. Greediness and unused detections validation
        # Margin (3, 3). Detection 12 falls into both 10's window and 15's window.
        # It should be assigned to the first true CP (10), leaving nothing for 15.
        ([12], [10, 15], (3, 3), {10: {12}, 15: set()}),
        # 7. False Positives (detections out of bounds) are ignored in the match output dict
        ([2, 20], [10], (2, 2), {10: set()}),
        # 8. Complex scenario: multiple true CPs and multiple detections
        # For 10 -> catches [8, 12]. For 15 -> catches [14]. Detection 20 is ignored.
        ([8, 12, 14, 20], [10, 15], (2, 2), {10: {8, 12}, 15: {14}}),
        # 9. Greediness with multiple available points
        # True: 10, 20. Detected: 12, 18. Margin (5, 5).
        # 10 takes 12 (range 5-15). 20 takes 18 (range 15-25).
        ([12, 18], [10, 20], (5, 5), {10: {12}, 20: {18}}),
    ],
)
def test_classification_metric_match(
    detected: Sequence[int], true_cps: Sequence[int], margin: tuple[int, int], expected: dict[int, set[int]]
) -> None:
    """
    Test the static `match` method to ensure correct alignment of detected
    change points to true change points according to the specified error margin.
    """
    result = MockClassificationMetric.match(
        detected_change_points=detected, true_change_points=true_cps, error_margin=margin
    )

    assert result == expected
