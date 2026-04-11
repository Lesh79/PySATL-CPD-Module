# -*- coding: ascii -*-

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.analysis.metrics.classification.fn_metric import FalseNegativeMetric as SingleFN
from pysatl_cpd.benchmark.metrics.classification.fn_metric import FalseNegativeMetric
from tests.mocks.analysis.labeled_data import MockLabeledData
from tests.mocks.core.detection_trace import MockDetectionTrace


def test_fn_metric_initialization() -> None:
    """Test that the metric initializes correctly and sets up the base metric."""
    metric: FalseNegativeMetric[MockDetectionTrace, MockLabeledData] = FalseNegativeMetric(error_margin=(2, 2))
    assert isinstance(metric.base_metric, SingleFN)
    assert metric.base_metric._error_margin == (2, 2)


def test_fn_metric_invalid_margin() -> None:
    """Test that invalid error margins are caught during initialization."""
    with pytest.raises(ValueError, match="non-negative numbers"):
        FalseNegativeMetric(error_margin=(-1, -1))


@pytest.mark.parametrize("values, expected", [([], 0.0), ([1.0, 2.0, 0.0], 3.0)])
def test_fn_metric_aggregate(values: list[float], expected: float) -> None:
    """Test the aggregation logic independently of the evaluation."""
    metric: FalseNegativeMetric[MockDetectionTrace, MockLabeledData] = FalseNegativeMetric(error_margin=(0, 0))
    assert metric.aggregate(values) == expected


@pytest.mark.parametrize(
    "true_cps, detected_cps, margin, expected_fn",
    [
        ([], [], (2, 2), 0.0),  # No true points -> nothing to miss
        ([], [10, 20], (2, 2), 0.0),  # Detections exist, but no true points -> FN=0
        ([10, 20], [], (2, 2), 2.0),  # Two missed -> FN=2
        ([10], [8], (2, 2), 0.0),  # Caught on the border -> FN=0
        ([10], [7], (2, 2), 1.0),  # Outside the border -> Missed (FN=1)
        ([10, 20], [10], (1, 1), 1.0),  # 10 is caught, 20 is missed -> FN=1
    ],
)
def test_fn_metric_evaluate_boundaries(
    true_cps: list[int], detected_cps: list[int], margin: tuple[int, int], expected_fn: float
) -> None:
    """Test the evaluation logic using various boundaries and edge cases."""
    metric: FalseNegativeMetric[MockDetectionTrace, MockLabeledData] = FalseNegativeMetric(error_margin=margin)
    trace: MockDetectionTrace = MockDetectionTrace(detected_change_points=detected_cps)
    data: MockLabeledData = MockLabeledData(change_points=true_cps)
    assert metric.evaluate([(trace, data)]) == expected_fn


def test_fn_metric_evaluate_multiple_runs() -> None:
    """Test evaluation across an entire dataset consisting of multiple runs."""
    metric: FalseNegativeMetric[MockDetectionTrace, MockLabeledData] = FalseNegativeMetric(error_margin=(1, 1))
    runs: list[tuple[MockDetectionTrace, MockLabeledData]] = [
        (MockDetectionTrace([11, 15]), MockLabeledData([10])),  # 10 is caught (FN=0)
        (MockDetectionTrace([20]), MockLabeledData([20, 30])),  # 20 is caught, 30 is missed (FN=1)
        (MockDetectionTrace([50]), MockLabeledData([40])),  # 40 is missed (FN=1)
    ]
    # Expected: 0 + 1 + 1 = 2.0
    assert metric.evaluate(runs) == 2.0
