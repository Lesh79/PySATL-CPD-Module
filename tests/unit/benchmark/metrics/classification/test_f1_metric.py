# -*- coding: ascii -*-

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.analysis.metrics.classification.confusion_matrix import ConfusionMatrix
from pysatl_cpd.benchmark.metrics.classification.f1_metric import F1Metric
from tests.mocks.analysis.labeled_data import MockLabeledData
from tests.mocks.core.detection_trace import MockDetectionTrace


def test_f1_metric_initialization() -> None:
    """Test that the metric initializes correctly and sets up the base metric."""
    metric: F1Metric[MockDetectionTrace, MockLabeledData] = F1Metric(error_margin=(1, 1))
    assert isinstance(metric.base_metric, ConfusionMatrix)
    assert metric.base_metric._error_margin == (1, 1)


def test_f1_metric_invalid_margin() -> None:
    """Test that invalid error margins are caught during initialization."""
    with pytest.raises(ValueError, match="non-negative numbers"):
        F1Metric(error_margin=(-1, 1))


@pytest.mark.parametrize(
    "values, expected_f1",
    [
        ([], 0.0),
        ([{"tp": 0.0, "fp": 0.0, "fn": 0.0}], 0.0),  # Zero denominator protection
        ([{"tp": 0.0, "fp": 10.0, "fn": 10.0}], 0.0),  # P=0, R=0
        ([{"tp": 10.0, "fp": 0.0, "fn": 0.0}], 1.0),  # Perfect F1
        (
            [
                {"tp": 5.0, "fp": 5.0, "fn": 0.0},  # Run 1
                {"tp": 5.0, "fp": 0.0, "fn": 5.0},  # Run 2
            ],
            0.6666666,
        ),  # Global TP=10, FP=5, FN=5 -> P=0.66, R=0.66 -> F1=0.66
    ],
)
def test_f1_aggregate(values: list[dict[str, float]], expected_f1: float) -> None:
    """Test the aggregation logic independently of the evaluation."""
    metric: F1Metric[MockDetectionTrace, MockLabeledData] = F1Metric(error_margin=(0, 0))
    assert metric.aggregate(values) == pytest.approx(expected_f1)


@pytest.mark.parametrize(
    "true_cps, detected_cps, margin, expected",
    [
        ([], [], (1, 1), 0.0),
        ([10], [10, 15], (1, 1), 2.0 / 3.0),  # TP=1, FP=1, FN=0 -> P=0.5, R=1.0 -> F1=0.666
    ],
)
def test_f1_evaluate_boundaries(
    true_cps: list[int], detected_cps: list[int], margin: tuple[int, int], expected: float
) -> None:
    """Test the evaluation logic using various boundaries and edge cases."""
    metric: F1Metric[MockDetectionTrace, MockLabeledData] = F1Metric(error_margin=margin)
    trace: MockDetectionTrace = MockDetectionTrace(detected_change_points=detected_cps)
    data: MockLabeledData = MockLabeledData(change_points=true_cps)
    assert metric.evaluate([(trace, data)]) == pytest.approx(expected)


def test_f1_evaluate_multiple_runs() -> None:
    """Test evaluation across an entire dataset consisting of multiple runs."""
    metric: F1Metric[MockDetectionTrace, MockLabeledData] = F1Metric(error_margin=(1, 1))
    runs: list[tuple[MockDetectionTrace, MockLabeledData]] = [
        (MockDetectionTrace([11, 15]), MockLabeledData([10])),  # TP=1, FP=1, FN=0
        (MockDetectionTrace([20]), MockLabeledData([20, 30])),  # TP=1, FP=0, FN=1
        (MockDetectionTrace([50]), MockLabeledData([40])),  # TP=0, FP=1, FN=1
    ]
    # Global TP = 2, FP = 2, FN = 2
    # Precision = 2/4 = 0.5. Recall = 2/4 = 0.5. F1 = 2*0.25/1 = 0.5
    assert metric.evaluate(runs) == 0.5
