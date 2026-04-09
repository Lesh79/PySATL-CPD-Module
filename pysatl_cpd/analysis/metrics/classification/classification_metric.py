# -*- coding: ascii -*-

"""
Base module for classification-based evaluation metrics.

This module provides the `ClassificationMetric` base class and the core
matching algorithm used to align predicted change points with true change points
within a defined tolerance window.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.analysis.metrics.run_metric import RunMetric
from pysatl_cpd.core.detection_trace import DetectionTrace


class ClassificationMetric[T: DetectionTrace, D: LabeledData[Any]](RunMetric[T, D, float]):
    """
    Base class for classification metrics (TP, FP, FN) in change point detection.

    Parameters
    ----------
    error_margin : tuple[int, int]
        A tuple `(left, right)` representing the tolerance window around a true
        change point. A detected change point is considered a match if it falls
        within `[true_change - left, true_change + right]`.
    """

    def __init__(self, error_margin: tuple[int, int]) -> None:
        self._error_margin = error_margin

    @staticmethod
    def match(detected_changes: Sequence[int], true_changes: Sequence[int], error_margin: tuple[int, int]) -> set[int]:
        """
        Match detected change points to true change points within a given error margin.

        Establishes a 1-to-1 mapping where the first available detected point within
        the tolerance window of a true point is considered a match.

        Parameters
        ----------
        detected_changes : Sequence[int]
            The sequence of predicted change point indices.
        true_changes : Sequence[int]
            The sequence of actual change point indices.
        error_margin : tuple[int, int]
            Tolerance window `(left, right)`.

        Returns
        -------
        set[int]
            A set of detected change points that successfully matched with true change points.
        """

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
        """
        Evaluate the metric.

        Parameters
        ----------
        trace : T
            The trace containing detected change points.
        data : D
            The ground truth data containing actual change points.

        Returns
        -------
        float
            The computed metric value.
        """
