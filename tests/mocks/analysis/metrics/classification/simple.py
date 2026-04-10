# -*- coding: ascii -*-

"""
Mock classification metrics for testing.

This module provides a mock implementation of ClassificationMetric
for testing metric evaluations without performing actual computations.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.classification.classification_metric import ClassificationMetric
from pysatl_cpd.core.detection_trace import DetectionTrace


class MockClassificationMetric[TraceT: DetectionTrace, ProviderT: LabeledData[Any]](
    ClassificationMetric[TraceT, ProviderT]
):
    """
    Mock implementation of the ClassificationMetric class for testing purposes.

    This metric always returns a constant evaluation score (0.0) regardless
    of the provided trace and labeled data.
    """

    def evaluate(self, trace: DetectionTrace, data: LabeledData[Any]) -> float:
        """
        Dummy implementation of the evaluate method.

        Parameters
        ----------
        trace : DetectionTrace
            The detection trace produced by a change-point detection algorithm.
        data : LabeledData[Any]
            The ground truth labeled data.

        Returns
        -------
        float
            A constant dummy metric value (0.0).
        """

        return 0.0
