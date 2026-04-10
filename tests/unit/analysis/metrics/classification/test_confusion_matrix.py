# -*- coding: ascii -*-

"""
Tests for ConfusionMatrix metric class.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from typing import Any

import hypothesis.strategies as st
import pytest
from hypothesis import given

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.classification.confusion_matrix import ConfusionMatrix
from pysatl_cpd.core.detection_trace import DetectionTrace
from tests.mocks.analysis.labeled_data import MockLabeledData
from tests.mocks.core.detection_trace import MockDetectionTrace


def test_confusion_matrix_init_valid() -> None:
    """
    Test successful initialization of ConfusionMatrix with valid error margins.
    """

    margin = (5, 5)
    metric: ConfusionMatrix[DetectionTrace, LabeledData[Any]] = ConfusionMatrix(error_margin=margin)

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
def test_confusion_matrix_init_invalid(invalid_margin: tuple[int, int]) -> None:
    """
    Test that ConfusionMatrix validation raises an error on negative margins.
    """

    with pytest.raises(ValueError, match="must be non-negative"):
        ConfusionMatrix(error_margin=invalid_margin)


@pytest.mark.parametrize(
    "detected, true_cps, margin, expected_tp, expected_fp, expected_fn",
    [
        # Scenario 1: Empty lists
        ([], [], (2, 2), 0.0, 0.0, 0.0),
        # Scenario 2: Only false alarms (FP)
        ([10, 20], [], (2, 2), 0.0, 2.0, 0.0),
        # Scenario 3: Only missed detections (FN)
        ([], [10, 20], (2, 2), 0.0, 0.0, 2.0),
        # Scenario 4: Perfect match (only TP)
        ([10, 20], [10, 20], (0, 0), 2.0, 0.0, 0.0),
        # Scenario 5: Multiple detections covering a single true CP
        # True CP is 10. Detections 9, 10, 11 are all matched to 10.
        # TP = 1 (one true CP covered)
        # FP = 0 (all detections used)
        # FN = 0 (all true CPs covered)
        ([9, 10, 11], [10], (2, 2), 1.0, 0.0, 0.0),
        # Scenario 6: Mixed complex scenario
        # True CPs: 10, 20, 30.
        # Detections: 8, 22, 40, 45. Margin (2, 2).
        # - 8 matches 10 (TP = 1)
        # - 22 matches 20 (TP = +1 -> 2)
        # - 30 is missed (FN = 1)
        # - 40 and 45 match nothing (FP = 2)
        ([8, 22, 40, 45], [10, 20, 30], (2, 2), 2.0, 2.0, 1.0),
        # Scenario 7: Greedy matching conflict
        # True CPs: 10, 15. Margin: (3, 3).
        # Detection: 12.
        # 12 binds to 10.
        # TP = 1 (CP 10). FN = 1 (CP 15 missed). FP = 0.
        ([12], [10, 15], (3, 3), 1.0, 0.0, 1.0),
    ],
)
def test_confusion_matrix_evaluate(
    detected: list[int],
    true_cps: list[int],
    margin: tuple[int, int],
    expected_tp: float,
    expected_fp: float,
    expected_fn: float,
) -> None:
    """
    Test the `evaluate` method for calculating TP, FP, and FN correctly.
    """

    metric: ConfusionMatrix[MockDetectionTrace, MockLabeledData] = ConfusionMatrix(error_margin=margin)

    trace_mock = MockDetectionTrace(detected_change_points=detected)
    data_mock = MockLabeledData(change_points=true_cps)

    result = metric.evaluate(trace=trace_mock, data=data_mock)

    assert isinstance(result, dict)

    # Check keys exist
    assert "tp" in result
    assert "fp" in result
    assert "fn" in result

    # Check values and types
    assert result["tp"] == expected_tp
    assert isinstance(result["tp"], float)

    assert result["fp"] == expected_fp
    assert isinstance(result["fp"], float)

    assert result["fn"] == expected_fn
    assert isinstance(result["fn"], float)


@given(
    detected=st.lists(st.integers(min_value=1, max_value=1000), unique=True).map(sorted),
    true_cps=st.lists(st.integers(min_value=1, max_value=1000), unique=True).map(sorted),
    margin=st.tuples(st.integers(min_value=0, max_value=100), st.integers(min_value=0, max_value=100)),
)
def test_confusion_matrix_invariants(detected: list[int], true_cps: list[int], margin: tuple[int, int]) -> None:
    """
    Hypothesis property-based test to verify mathematical invariants of Confusion Matrix.
    """

    metric: ConfusionMatrix[MockDetectionTrace, MockLabeledData] = ConfusionMatrix(error_margin=margin)

    trace_mock = MockDetectionTrace(detected_change_points=detected)
    data_mock = MockLabeledData(change_points=true_cps)

    res = metric.evaluate(trace=trace_mock, data=data_mock)

    tp = res["tp"]
    fp = res["fp"]
    fn = res["fn"]

    # Invariant 1: Every true change point is either detected (TP) or missed (FN)
    assert tp + fn == float(len(true_cps))

    # Invariant 2: False positives cannot exceed total detections
    assert fp <= float(len(detected))

    # Invariant 3: All metrics must be non-negative
    assert tp >= 0.0
    assert fp >= 0.0
    assert fn >= 0.0
