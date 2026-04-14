# -*- coding: ascii -*-
"""
Online visualization components.

This module provides visualizers and components for rendering online
change-point detection results, including detection traces and algorithm
state evolution.

The module is organized into:
- Core online visualizer for detection traces (detection function, processing time)
- State visualizers for algorithm-specific internal state evolution

Components in this module work together with the composable architecture:
- OnlineTraceVisualizer renders the primary detection results
- State visualizers (e.g., DummyStateVisualizer) can be composed to add
  algorithm state visualization panels

Classes
-------
OnlineTraceVisualizer
    Visualizer for online detection trace results including detection function
    and processing time subplots.
IOnlineStateVisualiser
    Abstract base interface for online state visualizers.
DummyStateVisualizer
    Placeholder state visualizer that performs no rendering, suitable for
    testing or when state visualization is not required.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.analysis.visualization.online.online_trace_visualizer import (
    DetectionFuncDrawOpts,
    DetectionFuncPlotOpts,
    OnlineTraceVisualizer,
    ProcessingTimeDrawOpts,
    ProcessingTimePlotOpts,
    ThresholdDrawOpts,
)
from pysatl_cpd.analysis.visualization.online.states import (
    DummyStateVisualizer,
    IOnlineStateVisualizer,
)

__all__ = [
    "OnlineTraceVisualizer",
    "DetectionFuncPlotOpts",
    "DetectionFuncDrawOpts",
    "ThresholdDrawOpts",
    "ProcessingTimePlotOpts",
    "ProcessingTimeDrawOpts",
    "IOnlineStateVisualizer",
    "DummyStateVisualizer",
]
