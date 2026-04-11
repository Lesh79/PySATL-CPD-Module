# -*- coding: ascii -*-

import pytest

from pysatl_cpd.analysis.metrics.classification.confusion_matrix import ConfusionMatrix
from pysatl_cpd.benchmark.metrics.classification.precision_metric import PrecisionMetric
from tests.mocks.analysis.labeled_data import MockLabeledData
from tests.mocks.core.detection_trace import MockDetectionTrace


def test_precision_metric_initialization() -> None:
    """Test that the metric initializes correctly and sets up the base metric."""
    metric: PrecisionMetric[MockDetectionTrace, MockLabeledData] = PrecisionMetric(error_margin=(1, 1))
    assert isinstance(metric.base_metric, ConfusionMatrix)
    assert metric.base_metric._error_margin == (1, 1)


def test_precision_metric_invalid_margin() -> None:
    """Test that invalid error margins are caught during initialization."""
    with pytest.raises(ValueError, match="non-negative numbers"):
        PrecisionMetric(error_margin=(-1, 1))


@pytest.mark.parametrize(
    "values, expected_precision",
    [
        ([], 0.0),
        ([{"tp": 0.0, "fp": 0.0, "fn": 0.0}], 0.0),  # Zero denominator protection
        ([{"tp": 0.0, "fp": 10.0, "fn": 0.0}], 0.0),  # Precision 0 (0/10)
        ([{"tp": 10.0, "fp": 0.0, "fn": 5.0}], 1.0),  # Perfect Precision (10/10)
        ([{"tp": 2.0, "fp": 1.0, "fn": 0.0}, {"tp": 4.0, "fp": 3.0, "fn": 0.0}], 0.6),  # Global: 6 / (6+4) = 0.6
    ],
)
def test_precision_aggregate(values: list[dict[str, float]], expected_precision: float) -> None:
    """Test the aggregation logic independently of the evaluation."""
    metric: PrecisionMetric[MockDetectionTrace, MockLabeledData] = PrecisionMetric(error_margin=(0, 0))
    assert metric.aggregate(values) == pytest.approx(expected_precision)


@pytest.mark.parametrize(
    "true_cps, detected_cps, margin, expected",
    [
        ([], [], (1, 1), 0.0),
        ([10], [10, 15], (1, 1), 0.5),  # 1 TP (10), 1 FP (15) -> 1 / 2 = 0.5
    ],
)
def test_precision_evaluate_boundaries(
    true_cps: list[int], detected_cps: list[int], margin: tuple[int, int], expected: float
) -> None:
    """Test the evaluation logic using various boundaries and edge cases."""
    metric: PrecisionMetric[MockDetectionTrace, MockLabeledData] = PrecisionMetric(error_margin=margin)
    trace: MockDetectionTrace = MockDetectionTrace(detected_change_points=detected_cps)
    data: MockLabeledData = MockLabeledData(change_points=true_cps)
    assert metric.evaluate([(trace, data)]) == pytest.approx(expected)


def test_precision_evaluate_multiple_runs() -> None:
    """Test evaluation across an entire dataset consisting of multiple runs."""
    metric: PrecisionMetric[MockDetectionTrace, MockLabeledData] = PrecisionMetric(error_margin=(1, 1))
    runs: list[tuple[MockDetectionTrace, MockLabeledData]] = [
        (MockDetectionTrace([11, 15]), MockLabeledData([10])),  # TP=1, FP=1, FN=0
        (MockDetectionTrace([]), MockLabeledData([20])),  # TP=0, FP=0, FN=1
        (MockDetectionTrace([30]), MockLabeledData([])),  # TP=0, FP=1, FN=0
    ]
    # Global: TP = 1, FP = 2. Precision = 1 / (1 + 2) = 1/3
    assert metric.evaluate(runs) == pytest.approx(1.0 / 3.0)
