from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.classification.classification_metric import ClassificationMetric
from pysatl_cpd.analysis.metrics.run_metric import RunMetric
from pysatl_cpd.core.detection_trace import DetectionTrace


class ConfusionMatrix[T: DetectionTrace, D: LabeledData[Any]](RunMetric[T, D, dict[str, float]]):
    def __init__(self, error_margin: tuple[int, int]) -> None:
        self.__error_margin = error_margin

    def evaluate(self, trace: T, data: D) -> dict[str, float]:
        detected_changes = trace.detected_change_points
        true_changes = data.change_points

        tp = len(ClassificationMetric.match(detected_changes, true_changes, self.__error_margin))
        fn = len(true_changes) - tp
        fp = len(detected_changes) - tp

        return {"tp": float(tp), "fp": float(fp), "fn": float(fn)}
