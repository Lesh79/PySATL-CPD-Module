# -*- coding: ascii -*-

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import pytest

from pysatl_cpd.analysis.metrics.online.delay_metric import DelayMetric as SingleDelayMetric
from pysatl_cpd.benchmark.metrics.online.delay_metric import MeanDelayMetric
from tests.mocks.analysis.labeled_data import MockLabeledData
from tests.mocks.core.online.online_detection_trace import MockOnlineDetectionTrace


def test_mean_delay_initialization() -> None:
    """Test that the MeanDelayMetric initializes correctly and sets up the base metric."""
    max_delay: int = 10
    metric: MeanDelayMetric[MockOnlineDetectionTrace, MockLabeledData] = MeanDelayMetric(max_delay=max_delay)

    assert isinstance(metric.base_metric, SingleDelayMetric)
    assert metric.base_metric._max_delay == max_delay
    assert metric._MeanDelayMetric__max_delay == max_delay  # type: ignore


def test_mean_delay_invalid_max_delay() -> None:
    """Test that negative max_delay raises a ValueError."""
    # Assuming the implementation checks for negative max_delay and raises ValueError
    with pytest.raises(ValueError):
        MeanDelayMetric(max_delay=-5)


@pytest.mark.parametrize(
    "values, expected_mean",
    [
        ([], 5.0),  # Empty sequence returns max_delay
        ([[], []], 5.0),  # Multiple runs but no ground truth -> max_delay
        ([[0, 2], [4]], 2.0),  # Flattened: [0, 2, 4]. Mean: 6 / 3 = 2.0
        ([[5, 5], [5, 5]], 5.0),  # All missed (penalty = 5). Mean: 5.0
    ],
)
def test_mean_delay_aggregate(values: list[list[int]], expected_mean: float) -> None:
    """Test the aggregation logic independently of the base metric evaluation."""
    metric: MeanDelayMetric[MockOnlineDetectionTrace, MockLabeledData] = MeanDelayMetric(max_delay=5)
    assert metric.aggregate(values) == pytest.approx(expected_mean)


@pytest.mark.parametrize(
    "true_cps, detected_cps, max_delay, expected_mean",
    [
        ([], [10, 20], 5, 5.0),  # No true CPs -> returns max_delay
        ([10], [10], 5, 0.0),  # Perfect hit. Delay: 0
        ([10], [15], 5, 5.0),  # Hit exactly on the right boundary (10+5). Delay: 5
        ([10], [16], 5, 5.0),  # Missed (Late detection). Penalty: 5
        ([10], [9], 5, 5.0),  # Missed (Early detection is out of window). Penalty: 5
        ([10], [12, 14], 5, 2.0),  # Multiple hits in window. Minimum delay is used: 2
        ([10, 20], [12, 23], 5, 2.5),  # Two points. Delays: 2 and 3. Mean: 2.5
    ],
)
def test_mean_delay_evaluate_boundaries(
    true_cps: list[int], detected_cps: list[int], max_delay: int, expected_mean: float
) -> None:
    """Test evaluation logic using various boundaries, penalties, and early/late hits."""
    metric: MeanDelayMetric[MockOnlineDetectionTrace, MockLabeledData] = MeanDelayMetric(max_delay=max_delay)
    trace: MockOnlineDetectionTrace = MockOnlineDetectionTrace(detected_change_points=detected_cps)
    data: MockLabeledData = MockLabeledData(change_points=true_cps)

    assert metric.evaluate([(trace, data)]) == pytest.approx(expected_mean)


def test_mean_delay_evaluate_multiple_runs() -> None:
    """Test integration across an entire dataset consisting of multiple runs."""
    metric: MeanDelayMetric[MockOnlineDetectionTrace, MockLabeledData] = MeanDelayMetric(max_delay=10)

    runs: list[tuple[MockOnlineDetectionTrace, MockLabeledData]] = [
        # True: 10, 20. Det: 12 (Delay: 2), 35 (Late -> Penalty: 10) -> Delays for run 1: [2, 10]
        (MockOnlineDetectionTrace([12, 35]), MockLabeledData([10, 20])),
        # True: None. Det: 15 -> Ignored. Delays for run 2: []
        (MockOnlineDetectionTrace([15]), MockLabeledData([])),
        # True: 30. Det: 30 (Delay: 0) -> Delays for run 3: [0]
        (MockOnlineDetectionTrace([30]), MockLabeledData([30])),
    ]

    # Global flattened delays: [2, 10, 0]
    # Mean: (2 + 10 + 0) / 3 = 12 / 3 = 4.0
    assert metric.evaluate(runs) == 4.0
