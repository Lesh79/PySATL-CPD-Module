# -*- coding: ascii -*-

import pytest

from pysatl_cpd.analysis.metrics.classification.confusion_matrix import ConfusionMatrix
from pysatl_cpd.benchmark.metrics.classification.classification_report import ClassificationReport
from tests.mocks.analysis.labeled_data import MockLabeledData
from tests.mocks.core.detection_trace import MockDetectionTrace


def test_classification_report_initialization() -> None:
    """Test that the metric initializes correctly and sets up the base metric."""
    report: ClassificationReport[MockDetectionTrace, MockLabeledData] = ClassificationReport(error_margin=(1, 1))
    assert isinstance(report.base_metric, ConfusionMatrix)
    assert report.base_metric._error_margin == (1, 1)


def test_classification_report_invalid_margin() -> None:
    """Test that invalid error margins are caught during initialization."""
    with pytest.raises(ValueError, match="non-negative numbers"):
        ClassificationReport(error_margin=(-1, 1))


@pytest.mark.parametrize(
    "values, expected",
    [
        ([], {"tp": 0.0, "fp": 0.0, "fn": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}),
        (
            [{"tp": 0.0, "fp": 0.0, "fn": 0.0}],  # Zero denominator
            {"tp": 0.0, "fp": 0.0, "fn": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0},
        ),
        (
            [{"tp": 10.0, "fp": 0.0, "fn": 0.0}],  # Perfect score
            {"tp": 10.0, "fp": 0.0, "fn": 0.0, "precision": 1.0, "recall": 1.0, "f1": 1.0},
        ),
        (
            [{"tp": 5.0, "fp": 2.0, "fn": 1.0}, {"tp": 5.0, "fp": 3.0, "fn": 4.0}],  # Mixed scenario: TP=10, FP=5, FN=5
            {"tp": 10.0, "fp": 5.0, "fn": 5.0, "precision": 10 / 15, "recall": 10 / 15, "f1": 10 / 15},
        ),
    ],
)
def test_classification_report_aggregate(values: list[dict[str, float]], expected: dict[str, float]) -> None:
    """Test the aggregation logic independently of the evaluation."""
    report: ClassificationReport[MockDetectionTrace, MockLabeledData] = ClassificationReport(error_margin=(0, 0))
    result: dict[str, float] = report.aggregate(values)

    for key in expected:
        assert result[key] == pytest.approx(expected[key])


@pytest.mark.parametrize(
    "true_cps, detected_cps, margin, expected_tp, expected_fp, expected_fn",
    [
        ([10], [10, 15], (1, 1), 1.0, 1.0, 0.0),
    ],
)
def test_classification_report_evaluate_boundaries(
    true_cps: list[int],
    detected_cps: list[int],
    margin: tuple[int, int],
    expected_tp: float,
    expected_fp: float,
    expected_fn: float,
) -> None:
    """Test the evaluation logic using various boundaries and edge cases."""
    report: ClassificationReport[MockDetectionTrace, MockLabeledData] = ClassificationReport(error_margin=margin)
    trace: MockDetectionTrace = MockDetectionTrace(detected_change_points=detected_cps)
    data: MockLabeledData = MockLabeledData(change_points=true_cps)

    result: dict[str, float] = report.evaluate([(trace, data)])
    assert result["tp"] == expected_tp
    assert result["fp"] == expected_fp
    assert result["fn"] == expected_fn


def test_classification_report_evaluate_multiple_runs() -> None:
    """Test evaluation across an entire dataset consisting of multiple runs."""
    report: ClassificationReport[MockDetectionTrace, MockLabeledData] = ClassificationReport(error_margin=(1, 1))
    runs: list[tuple[MockDetectionTrace, MockLabeledData]] = [
        (MockDetectionTrace([11, 15]), MockLabeledData([10])),  # TP=1, FP=1, FN=0
        (MockDetectionTrace([20]), MockLabeledData([20, 30])),  # TP=1, FP=0, FN=1
        (MockDetectionTrace([50]), MockLabeledData([40])),  # TP=0, FP=1, FN=1
    ]
    # Global: TP=2, FP=2, FN=2. P=0.5, R=0.5, F1=0.5
    result: dict[str, float] = report.evaluate(runs)

    assert result["tp"] == 2.0
    assert result["fp"] == 2.0
    assert result["fn"] == 2.0
    assert result["precision"] == 0.5
    assert result["recall"] == 0.5
    assert result["f1"] == 0.5
