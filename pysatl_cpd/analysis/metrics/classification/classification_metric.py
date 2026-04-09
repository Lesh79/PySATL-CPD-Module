from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.run_metric import RunMetric
from pysatl_cpd.core.detection_trace import DetectionTrace


class ClassificationMetric[T: DetectionTrace, D: LabeledData[Any]](RunMetric[T, D, float]):
    def __init__(self, error_margin: tuple[int, int]) -> None:
        self._error_margin = error_margin

    @staticmethod
    def match(detected_changes: Sequence[int], true_changes: Sequence[int], error_margin: tuple[int, int]) -> set[int]:
        left, right = error_margin
        used_detections = set()

        for true_change in true_changes:
            for detected_change in detected_changes:
                if detected_change in used_detections:
                    continue
                if true_change - left <= detected_change <= true_change + right:
                    used_detections.add(detected_change)
                    break

        return used_detections

    @abstractmethod
    def evaluate(self, trace: T, data: D) -> float:
        pass
