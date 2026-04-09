# -*- coding: ascii -*-

from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.classification.classification_metric import ClassificationMetric
from pysatl_cpd.core.detection_trace import DetectionTrace


class MockClassificationMetric(ClassificationMetric[DetectionTrace, LabeledData[Any]]):
    """
    Mock implementation of the abstract ClassificationMetric class for testing purposes.
    """

    def evaluate(self, trace: DetectionTrace, data: LabeledData[Any]) -> float:
        """
        Dummy implementation of the abstract evaluate method.
        """
        return 0.0
