# -*- coding: ascii -*-
"""
Visualization module for change-point detection.

This module provides classes and functions for visualizing change-point
detection results, including time series data, detection traces, and
algorithm performance metrics. It supports both Matplotlib and Plotly
backends for flexible rendering options.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.visualization.abstracts import IVisualComponent
from pysatl_cpd.analysis.visualization.benchmarking import (
    ARLBasedMetricVisualizer,
    BenchmarkPlotter,
    MetricVisualizer,
    PrAucVisualizer,
    ThresholdBasedMetricVisualizer,
)
from pysatl_cpd.analysis.visualization.components import (
    VerticalFillComponent,
    VerticalLineVisualComponent,
)
from pysatl_cpd.analysis.visualization.online import (
    DetectionFuncDrawOpts,
    DetectionFuncPlotOpts,
    DummyStateVisualizer,
    IOnlineStateVisualizer,
    OnlineTraceVisualizer,
    ProcessingTimeDrawOpts,
    ProcessingTimePlotOpts,
    ThresholdDrawOpts,
)
from pysatl_cpd.analysis.visualization.plotters import OnlineCpdPlotter
from pysatl_cpd.analysis.visualization.timeseries import (
    TimeseriesDrawOpts,
    TimeseriesPlotOpts,
    UnivariateTimeseriesVisualizer,
)
from pysatl_cpd.analysis.visualization.typedefs import DrawBackend

__all__ = [
    # Coordinator
    "OnlineCpdPlotter",
    # Backend typedefs
    "DrawBackend",
    # Time series visualizers
    "UnivariateTimeseriesVisualizer",
    "TimeseriesPlotOpts",
    "TimeseriesDrawOpts",
    # Online trace visualizers
    "OnlineTraceVisualizer",
    "DetectionFuncPlotOpts",
    "DetectionFuncDrawOpts",
    "ThresholdDrawOpts",
    "ProcessingTimePlotOpts",
    "ProcessingTimeDrawOpts",
    "IOnlineStateVisualizer",
    "DummyStateVisualizer",
    # Benchmark visualizers
    "BenchmarkPlotter",
    "MetricVisualizer",
    "PrAucVisualizer",
    "ThresholdBasedMetricVisualizer",
    "ARLBasedMetricVisualizer",
    # Components
    "IVisualComponent",
    "VerticalLineVisualComponent",
    "VerticalFillComponent",
]
