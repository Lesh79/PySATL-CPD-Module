# -*- coding: ascii -*-
"""
Univariate time series visualizer implementation.

This module provides a visualizer for rendering univariate time series data
with change point annotations, period fills, and ground truth markers.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from typing import Self, TypedDict, Unpack

import plotly.graph_objs as go

from pysatl_cpd.analysis.visualization.abstracts import (
    ITimeseriesVisualizer,
)
from pysatl_cpd.analysis.visualization.typedefs import (
    DrawBackend,
    GoAxMapping,
    GoFigure,
    PltAxMapping,
    PltFigure,
)
from pysatl_cpd.core.data_providers import DataProvider
from pysatl_cpd.core.typedefs import Number

# Plotly layout constants
PLOTLY_GRID_COLOR = "lightgray"
PLOTLY_GRID_WIDTH = 1
PLOTLY_TITLE_X = 0.5
PLOTLY_TITLE_Y = 0.95

# Matplotlib constants
MPL_GRID_ALPHA = 0.3


class TimeseriesPlotOpts(TypedDict, total=False):
    """Plot options for time series subplot."""

    xlabel: str
    ylabel: str
    grid: bool
    legend: bool


class TimeseriesDrawOpts(TypedDict, total=False):
    """Drawing options for time series line."""

    color: str
    linewidth: float
    alpha: float
    label: str


class UnivariateTimeseriesVisualizer(ITimeseriesVisualizer[DataProvider[Number]]):
    """
    Visualizer for univariate time series with change point annotations.

    This visualizer renders the original time series data with optional
    detected change points, forced change points, ground truth markers,
    learning periods, skip periods, and margin windows.
    """

    def __init__(self, backend: DrawBackend) -> None:
        """
        Initialize the univariate time series visualizer.

        Parameters
        ----------
        backend : DrawBackend
            Plotting backend to use for rendering.
        """
        super().__init__(backend)
        self._data_provider: DataProvider[Number] | None = None

        # Options storage with defaults
        self._plot_opts: TimeseriesPlotOpts = {
            "xlabel": "Time Index",
            "ylabel": "Value",
            "grid": True,
            "legend": True,
        }
        self._draw_opts: TimeseriesDrawOpts = {
            "color": "black",
            "linewidth": 1.5,
            "alpha": 1.0,
            "label": "Time Series",
        }

    def set_data_provider(self, data_provider: DataProvider[Number]) -> Self:
        """Set the data provider containing the time series observations."""
        self._data_provider = data_provider
        return self

    def set_plot_opts(self, **options: Unpack[TimeseriesPlotOpts]) -> Self:
        """Set general plot options for time series subplot."""
        self._plot_opts.update(options)
        return self

    def set_draw_opts(self, **options: Unpack[TimeseriesDrawOpts]) -> Self:
        """Set drawing options for time series line."""
        self._draw_opts.update(options)
        return self

    @property
    def axes(self) -> set[str]:
        """Declare the subplot names required by this visualizer."""
        return {"timeseries"}

    def _draw_matplotlib(self, figure: PltFigure, axes: PltAxMapping) -> PltFigure:
        """Draw using Matplotlib backend."""
        if "timeseries" not in axes:
            return figure

        ax = axes["timeseries"]

        # Draw time series data
        if self._data_provider is not None:
            time_points = list(range(len(self._data_provider)))
            values = list(self._data_provider)

            ax.plot(
                time_points,
                values,
                color=self._draw_opts["color"],
                linewidth=self._draw_opts["linewidth"],
                alpha=self._draw_opts["alpha"],
                label=self._draw_opts["label"] if self._plot_opts["legend"] else None,
            )

        # Configure axes
        ax.set_xlabel(self._plot_opts["xlabel"])
        ax.set_ylabel(self._plot_opts["ylabel"])

        if self._plot_opts["grid"]:
            ax.grid(True, alpha=MPL_GRID_ALPHA)

        return figure

    def _draw_plotly(self, figure: GoFigure, axes: GoAxMapping) -> GoFigure:
        """Draw using Plotly backend."""
        if "timeseries" not in axes:
            return figure

        row, col = axes["timeseries"]

        # Draw time series data
        if self._data_provider is not None:
            time_points = list(range(len(self._data_provider)))
            values = list(self._data_provider)

            figure.add_trace(
                go.Scatter(
                    x=time_points,
                    y=values,
                    mode="lines",
                    name=self._draw_opts["label"],
                    line={
                        "color": self._draw_opts["color"],
                        "width": self._draw_opts["linewidth"],
                    },
                    opacity=self._draw_opts["alpha"],
                    showlegend=self._plot_opts["legend"],
                ),
                row=row,
                col=col,
            )

        # Configure axes
        figure.update_xaxes(
            title_text=self._plot_opts["xlabel"],
            showgrid=self._plot_opts["grid"],
            gridwidth=PLOTLY_GRID_WIDTH,
            gridcolor=PLOTLY_GRID_COLOR,
            row=row,
            col=col,
        )
        figure.update_yaxes(
            title_text=self._plot_opts["ylabel"],
            showgrid=self._plot_opts["grid"],
            gridwidth=PLOTLY_GRID_WIDTH,
            gridcolor=PLOTLY_GRID_COLOR,
            row=row,
            col=col,
        )
        return figure
