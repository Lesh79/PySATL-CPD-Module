# -*- coding: ascii -*-

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.analysis.metrics.classification.fp_metric import FalsePositiveMetric as SingleFP
from pysatl_cpd.benchmark.metrics.classification.fp_metric import FalsePositiveMetric
from tests.mocks.analysis.labeled_data import MockLabeledData
from tests.mocks.core.detection_trace import MockDetectionTrace


def test_fp_metric_initialization() -> None:
    """Test that the metric initializes correctly and sets up the base metric."""
    metric: FalsePositiveMetric[MockDetectionTrace, MockLabeledData] = FalsePositiveMetric(error_margin=(2, 3))
    assert isinstance(metric.base_metric, SingleFP)
    assert metric.base_metric._error_margin == (2, 3)


def test_fp_metric_invalid_margin() -> None:
    """Test that invalid error margins are caught during initialization."""
    with pytest.raises(ValueError, match="non-negative numbers"):
        FalsePositiveMetric(error_margin=(-1, 2))


@pytest.mark.parametrize("values, expected", [([], 0.0), ([0.0, 0.0], 0.0), ([1.0, 5.0, 2.0], 8.0)])
def test_fp_metric_aggregate(values: list[float], expected: float) -> None:
    """Test the aggregation logic independently of the evaluation."""
    metric: FalsePositiveMetric[MockDetectionTrace, MockLabeledData] = FalsePositiveMetric(error_margin=(0, 0))
    assert metric.aggregate(values) == expected


@pytest.mark.parametrize(
    "true_cps, detected_cps, margin, expected_fp",
    [
        ([], [], (2, 2), 0.0),  # Empty data
        ([10], [], (2, 2), 0.0),  # No detections -> 0 FP
        ([], [10, 20], (2, 2), 2.0),  # No true points -> all detections are FP
        ([10], [8], (2, 3), 0.0),  # Exactly on the left boundary (10-2=8) -> TP, FP=0
        ([10], [13], (2, 3), 0.0),  # Exactly on the right boundary (10+3=13) -> TP, FP=0
        ([10], [7], (2, 3), 1.0),  # Outside the left boundary (7 < 8) -> FP=1
        ([10], [14], (2, 3), 1.0),  # Outside the right boundary (14 > 13) -> FP=1
        ([10], [7, 8, 13, 14], (2, 3), 2.0),  # 8 and 13 are TP, 7 and 14 are FP (Total 2 FP)
    ],
)
def test_fp_metric_evaluate_boundaries(
    true_cps: list[int], detected_cps: list[int], margin: tuple[int, int], expected_fp: float
) -> None:
    """Test the evaluation logic using various boundaries and edge cases."""
    metric: FalsePositiveMetric[MockDetectionTrace, MockLabeledData] = FalsePositiveMetric(error_margin=margin)
    trace: MockDetectionTrace = MockDetectionTrace(detected_change_points=detected_cps)
    data: MockLabeledData = MockLabeledData(change_points=true_cps)
    assert metric.evaluate([(trace, data)]) == expected_fp


def test_fp_metric_evaluate_multiple_runs() -> None:
    """Test evaluation across an entire dataset consisting of multiple runs."""
    metric: FalsePositiveMetric[MockDetectionTrace, MockLabeledData] = FalsePositiveMetric(error_margin=(1, 1))
    runs: list[tuple[MockDetectionTrace, MockLabeledData]] = [
        (MockDetectionTrace([11, 15]), MockLabeledData([10])),  # 11 is TP, 15 is FP (FP=1)
        (MockDetectionTrace([]), MockLabeledData([20])),  # No detections (FP=0)
        (MockDetectionTrace([30]), MockLabeledData([])),  # False detection (FP=1)
    ]
    # Expected: 1 + 0 + 1 = 2.0
    assert metric.evaluate(runs) == 2.0
