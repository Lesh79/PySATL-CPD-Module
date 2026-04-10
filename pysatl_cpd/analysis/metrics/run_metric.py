# -*- coding: ascii -*-

"""
Base module defining the interface for all evaluation metrics.

This module provides the generic `RunMetric` base class, which establishes
the standard evaluation protocol for change point detection algorithms.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData
from pysatl_cpd.core.detection_trace import DetectionTrace


class RunMetric[TraceT: DetectionTrace, ProviderT: LabeledData[Any], ResultT](ABC):
    """
    Base class for all run evaluation metrics in change point detection.

    Provides a generic interface to evaluate a detection trace against
    labeled ground truth data.
    """

    @abstractmethod
    def evaluate(self, trace: TraceT, data: ProviderT) -> ResultT:
        """
        Evaluate the detection trace against the provided labeled data.

        Parameters
        ----------
        trace : TraceT
            The trace containing detected change points.
        data : ProviderT
            The ground truth data containing actual change points.

        Returns
        -------
        ResultT
            The computed metric result. The type depends on the specific metric
            implementation (e.g., float, dict, or sequence of integers).
        """
