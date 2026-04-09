from collections.abc import Sequence
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.classification.classification_metric import ClassificationMetric
from pysatl_cpd.core.detection_trace import DetectionTrace


class FalseNegativeMetric[T: DetectionTrace, D: LabeledData[Any]](ClassificationMetric[T, D]):
    @classmethod
    def compute(
        cls, detected_changes: Sequence[int], true_changes: Sequence[int], error_margin: tuple[int, int]
    ) -> int:
        return len(true_changes) - len(cls.match(detected_changes, true_changes, error_margin))

    def evaluate(self, trace: T, data: D) -> float:
        return float(
            self.compute(
                trace.detected_change_points,
                data.change_points,
                self._error_margin,
            )
        )
