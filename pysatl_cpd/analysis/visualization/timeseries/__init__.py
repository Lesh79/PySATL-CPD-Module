# -*- coding: ascii -*-
"""
Time series visualizers for change-point detection.

This module provides visualizers for rendering univariate and multivariate
time series data with change point annotations, period fills, and ground truth markers.

The visualizers follow the abstract interfaces defined in `analysis.visualization.abstracts`
and provide both Matplotlib and Plotly backend implementations.

Classes
-------
UnivariateTimeseriesVisualizer
    Visualizer for univariate time series data with configurable plot and drawing options.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.visualization.timeseries.univariate_timeseries_visualizer import (
    TimeseriesDrawOpts,
    TimeseriesPlotOpts,
    UnivariateTimeseriesVisualizer,
)

__all__ = [
    "UnivariateTimeseriesVisualizer",
    "TimeseriesPlotOpts",
    "TimeseriesDrawOpts",
]
