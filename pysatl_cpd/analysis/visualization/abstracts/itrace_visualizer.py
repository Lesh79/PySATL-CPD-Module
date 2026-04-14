# -*- coding: ascii -*-
"""
Trace visualizer interface.

This module defines the abstract base class for visualizers that render
detection traces, including detection function values and algorithm metadata.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from abc import ABC, abstractmethod
from typing import Self

from pysatl_cpd.analysis.visualization.abstracts.ivisualizer import IVisualizer
from pysatl_cpd.core.detection_trace import DetectionTrace


class ITraceVisualizer[DetectionTraceT: DetectionTrace](IVisualizer, ABC):
    """
    Abstract base class for trace visualizers.

    Visualizers of this type render detection results, including detection
    function values and algorithm metadata.

    Type Parameters
    ---------------
    DetectionTraceT : DetectionTrace
        The detection trace type bound by DetectionTrace. This allows
        visualizers to work with specific trace implementations such as
        OnlineDetectionTrace or OfflineDetectionTrace.

    Notes
    -----
    The type parameter DetectionTraceT is bound to DetectionTrace to ensure
    that any concrete implementation works with valid detection traces.
    Due to mypy limitations with generic bounds, a type-ignore comment
    is used on the TypeVar definition.
    """

    @abstractmethod
    def set_trace(self, trace: DetectionTraceT) -> Self:
        """
        Set the detection trace to visualize.

        Parameters
        ----------
        trace : DetectionTraceT
            Detection results containing change-point indices, scores,
            and algorithm-specific metadata.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        raise NotImplementedError
