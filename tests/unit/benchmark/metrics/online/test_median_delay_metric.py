# -*- coding: ascii -*-

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.analysis.metrics.online.delay_metric import DelayMetric as SingleDelayMetric
from pysatl_cpd.benchmark.metrics.online.delay_metric import MedianDelayMetric
from tests.mocks.analysis.labeled_data import MockLabeledData
from tests.mocks.core.online.online_detection_trace import MockOnlineDetectionTrace


def test_median_delay_initialization() -> None:
    """Test that the MedianDelayMetric initializes correctly and sets up the base metric."""
    max_delay: int = 10
    metric: MedianDelayMetric[MockOnlineDetectionTrace, MockLabeledData] = MedianDelayMetric(max_delay=max_delay)

    assert isinstance(metric.base_metric, SingleDelayMetric)
    assert metric.base_metric._max_delay == max_delay
    assert metric._MedianDelayMetric__max_delay == max_delay  # type: ignore


def test_median_delay_invalid_max_delay() -> None:
    """Test that negative max_delay raises a ValueError."""
    # Assuming the implementation checks for negative max_delay and raises ValueError
    with pytest.raises(ValueError):
        MedianDelayMetric(max_delay=-5)


@pytest.mark.parametrize(
    "values, expected_median",
    [
        ([], 5.0),  # Empty sequence returns max_delay
        ([[], []], 5.0),  # Multiple runs but no ground truth -> max_delay
        ([[0, 10], [100]], 10.0),  # Flattened: [0, 10, 100]. Median: 10.0 (Odd count)
        ([[1, 2], [3, 4]], 2.5),  # Flattened: [1, 2, 3, 4]. Median: (2+3)/2 = 2.5 (Even count)
        ([[5, 5], [5, 5]], 5.0),  # All missed (penalty = 5). Median: 5.0
    ],
)
def test_median_delay_aggregate(values: list[list[int]], expected_median: float) -> None:
    """Test the median aggregation logic independently of the base metric evaluation."""
    metric: MedianDelayMetric[MockOnlineDetectionTrace, MockLabeledData] = MedianDelayMetric(max_delay=5)
    assert metric.aggregate(values) == pytest.approx(expected_median)


@pytest.mark.parametrize(
    "true_cps, detected_cps, max_delay, expected_median",
    [
        ([], [10, 20], 5, 5.0),  # No true CPs -> returns max_delay
        ([10], [10], 5, 0.0),  # Perfect hit. Delay: 0
        ([10], [15], 5, 5.0),  # Hit exactly on the right boundary (10+5). Delay: 5
        ([10], [16], 5, 5.0),  # Missed (Late detection). Penalty: 5
        ([10], [9], 5, 5.0),  # Missed (Early detection is out of window). Penalty: 5
        ([10], [12, 14], 5, 2.0),  # Multiple hits in window. Minimum delay is used: 2
        ([10, 20, 30], [11, 25, 30], 5, 1.0),  # Delays: [1, 5(penalty), 0]. Median of [0, 1, 5] is 1.0
    ],
)
def test_median_delay_evaluate_boundaries(
    true_cps: list[int], detected_cps: list[int], max_delay: int, expected_median: float
) -> None:
    """Test evaluation logic using various boundaries, penalties, and early/late hits."""
    metric: MedianDelayMetric[MockOnlineDetectionTrace, MockLabeledData] = MedianDelayMetric(max_delay=max_delay)
    trace: MockOnlineDetectionTrace = MockOnlineDetectionTrace(detected_change_points=detected_cps)
    data: MockLabeledData = MockLabeledData(change_points=true_cps)

    assert metric.evaluate([(trace, data)]) == pytest.approx(expected_median)


def test_median_delay_evaluate_multiple_runs() -> None:
    """Test integration across an entire dataset consisting of multiple runs."""
    metric: MedianDelayMetric[MockOnlineDetectionTrace, MockLabeledData] = MedianDelayMetric(max_delay=10)

    runs: list[tuple[MockOnlineDetectionTrace, MockLabeledData]] = [
        # True: 10, 20. Det: 12 (Delay: 2), 35 (Late -> Penalty: 10) -> Delays for run 1: [2, 10]
        (MockOnlineDetectionTrace([12, 35]), MockLabeledData([10, 20])),
        # True: None. Det: 15 -> Ignored. Delays for run 2: []
        (MockOnlineDetectionTrace([15]), MockLabeledData([])),
        # True: 30. Det: 30 (Delay: 0) -> Delays for run 3: [0]
        (MockOnlineDetectionTrace([30]), MockLabeledData([30])),
        # True: 40. Det: 41 (Delay: 1) -> Delays for run 4: [1]
        (MockOnlineDetectionTrace([41]), MockLabeledData([40])),
    ]

    # Global flattened delays: [2, 10, 0, 1]
    # Sorted: [0, 1, 2, 10]
    # Median is the mean of two middle elements: (1 + 2) / 2 = 1.5
    assert metric.evaluate(runs) == 1.5
