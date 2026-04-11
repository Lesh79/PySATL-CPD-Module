# -*- coding: ascii -*-

import pytest

from pysatl_cpd.analysis.metrics.classification.tp_metric import TruePositiveMetric as SingleTP
from pysatl_cpd.benchmark.metrics.classification.tp_metric import TruePositiveMetric
from tests.mocks.analysis.labeled_data import MockLabeledData
from tests.mocks.core.detection_trace import MockDetectionTrace


def test_tp_metric_initialization() -> None:
    """Test that the metric initializes correctly and sets up the base metric."""
    metric: TruePositiveMetric[MockDetectionTrace, MockLabeledData] = TruePositiveMetric(error_margin=(1, 1))
    assert isinstance(metric.base_metric, SingleTP)
    assert metric.base_metric._error_margin == (1, 1)


def test_tp_metric_invalid_margin() -> None:
    """Test that invalid error margins are caught during initialization."""
    with pytest.raises(ValueError, match="non-negative numbers"):
        TruePositiveMetric(error_margin=(0, -5))


@pytest.mark.parametrize(
    "values, expected",
    [
        ([], 0.0),
        ([0.0, 0.0], 0.0),
        ([1.0, 2.0, 3.0], 6.0),
    ],
)
def test_tp_metric_aggregate(values: list[float], expected: float) -> None:
    """Test the aggregation logic independently of the evaluation."""
    metric: TruePositiveMetric[MockDetectionTrace, MockLabeledData] = TruePositiveMetric(error_margin=(0, 0))
    assert metric.aggregate(values) == expected


@pytest.mark.parametrize(
    "true_cps, detected_cps, margin, expected_tp",
    [
        ([], [], (2, 2), 0.0),
        ([10], [], (2, 2), 0.0),
        ([], [10], (2, 2), 0.0),
        ([10], [8], (2, 2), 1.0),  # Exactly on the left boundary
        ([10], [12], (2, 2), 1.0),  # Exactly on the right boundary
        ([10], [7], (2, 2), 0.0),  # Missed left boundary by 1
        ([10], [13], (2, 2), 0.0),  # Missed right boundary by 1
        ([10], [9, 10, 11], (2, 2), 1.0),  # Multiple detections mapping to one true CP -> 1 TP
        ([10, 12], [11], (2, 2), 1.0),  # 11 matches 10 (the first). 12 is left unmatched -> 1 TP
    ],
)
def test_tp_metric_evaluate_boundaries(
    true_cps: list[int], detected_cps: list[int], margin: tuple[int, int], expected_tp: float
) -> None:
    """Test the evaluation logic using various boundaries and edge cases."""
    metric: TruePositiveMetric[MockDetectionTrace, MockLabeledData] = TruePositiveMetric(error_margin=margin)
    trace: MockDetectionTrace = MockDetectionTrace(detected_change_points=detected_cps)
    data: MockLabeledData = MockLabeledData(change_points=true_cps)
    assert metric.evaluate([(trace, data)]) == expected_tp


def test_tp_metric_evaluate_multiple_runs() -> None:
    """Test evaluation across an entire dataset consisting of multiple runs."""
    metric: TruePositiveMetric[MockDetectionTrace, MockLabeledData] = TruePositiveMetric(error_margin=(1, 1))
    runs: list[tuple[MockDetectionTrace, MockLabeledData]] = [
        (MockDetectionTrace([11, 15]), MockLabeledData([10])),  # 11 is TP (TP=1)
        (MockDetectionTrace([20, 31]), MockLabeledData([20, 30])),  # Both hit (TP=2)
        (MockDetectionTrace([50]), MockLabeledData([40])),  # Missed (TP=0)
    ]
    # Expected: 1 + 2 + 0 = 3.0
    assert metric.evaluate(runs) == 3.0
