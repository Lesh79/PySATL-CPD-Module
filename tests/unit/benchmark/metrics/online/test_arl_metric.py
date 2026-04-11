# -*- coding: ascii -*-

import math
import warnings

import pytest

from pysatl_cpd.analysis.metrics.online.run_length_metric import RunLengthMetric
from pysatl_cpd.benchmark.metrics.online.arl_metric import ARLMetric
from tests.mocks.analysis.labeled_data import MockLabeledData
from tests.mocks.core.online.online_detection_trace import MockOnlineDetectionTrace


def test_arl_metric_initialization() -> None:
    """Test that the ARL metric initializes correctly and sets up the base metric."""
    metric: ARLMetric[MockOnlineDetectionTrace, MockLabeledData] = ARLMetric()
    assert isinstance(metric.base_metric, RunLengthMetric)


@pytest.mark.parametrize(
    "values, expected_arl",
    [
        ([], math.inf),  # Empty sequence of runs
        ([[], []], math.inf),  # Multiple runs, but no detections in any
        ([[15]], 15.0),  # Single run, single detection distance
        ([[10, 5, 15]], 10.0),  # Single run, multiple distances -> (10+5+15)/3
        ([[10, 20], [], [15]], 15.0),  # Multiple runs -> flattened: (10+20+15)/3
    ],
)
def test_arl_metric_aggregate(values: list[list[int]], expected_arl: float) -> None:
    """Test the aggregation logic independently of the evaluation and ground truth."""
    metric: ARLMetric[MockOnlineDetectionTrace, MockLabeledData] = ARLMetric()
    result: float = metric.aggregate(values)

    if math.isinf(expected_arl):
        assert math.isinf(result)
    else:
        assert result == pytest.approx(expected_arl)


@pytest.mark.parametrize(
    "detected_cps, expected_arl",
    [
        ([], math.inf),  # No detections at all -> Infinity
        ([10], 10.0),  # Distances: [10] -> mean: 10
        ([10, 25, 30], 10.0),  # Distances: [10, 15, 5] -> mean: (30)/3 = 10
        ([25, 10, 30], 10.0),  # Unsorted input should be sorted by base metric -> [10, 15, 5]
    ],
)
def test_arl_metric_evaluate_single_run(detected_cps: list[int], expected_arl: float) -> None:
    """Test evaluation logic on a single trace, including sorting robustness."""
    warnings.filterwarnings("ignore")
    metric: ARLMetric[MockOnlineDetectionTrace, MockLabeledData] = ARLMetric()
    trace: MockOnlineDetectionTrace = MockOnlineDetectionTrace(detected_change_points=detected_cps)
    # Ground truth data is ignored by ARL, so we pass an empty list
    data: MockLabeledData = MockLabeledData(change_points=[])

    runs: list[tuple[MockOnlineDetectionTrace, MockLabeledData]] = [(trace, data)]
    result: float = metric.evaluate(runs)

    if math.isinf(expected_arl):
        assert math.isinf(result)
    else:
        assert result == pytest.approx(expected_arl)


def test_arl_metric_evaluate_multiple_runs() -> None:
    """Test evaluation across an entire dataset consisting of multiple runs."""
    metric: ARLMetric[MockOnlineDetectionTrace, MockLabeledData] = ARLMetric()

    runs: list[tuple[MockOnlineDetectionTrace, MockLabeledData]] = [
        # Run 1: Detections [10, 20] -> Distances: [10, 10]
        (MockOnlineDetectionTrace([10, 20]), MockLabeledData([])),
        # Run 2: No detections -> Distances: []
        (MockOnlineDetectionTrace([]), MockLabeledData([])),
        # Run 3: Detection [5] -> Distances: [5]
        (MockOnlineDetectionTrace([5]), MockLabeledData([])),
    ]

    # Flattened distances: [10, 10, 5]
    # Mean: (10 + 10 + 5) / 3 = 25 / 3 = 8.333333333333334
    expected_mean: float = 25.0 / 3.0

    assert metric.evaluate(runs) == pytest.approx(expected_mean)
