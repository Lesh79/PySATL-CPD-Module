from abc import abstractmethod
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.core.detection_trace import DetectionTrace


class RunMetric[T: DetectionTrace, D: LabeledData[Any], R]:
    @abstractmethod
    def evaluate(self, trace: T, data: D) -> R:
        pass
