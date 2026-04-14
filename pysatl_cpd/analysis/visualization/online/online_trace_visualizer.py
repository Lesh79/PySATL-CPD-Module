# -*- coding: ascii -*-
"""
Online trace visualizer implementation.

This module provides concrete visualizer for rendering online detection
traces including detection function values and processing times.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from typing import Self, TypedDict, Unpack

import numpy as np
import plotly.graph_objects as go

from pysatl_cpd.analysis.visualization.abstracts import ITraceVisualizer
from pysatl_cpd.analysis.visualization.online.states import IOnlineStateVisualizer
from pysatl_cpd.analysis.visualization.typedefs import (
    DrawBackend,
    GoAxes,
    GoAxMapping,
    GoFigure,
    PltAxes,
    PltAxMapping,
    PltFigure,
)
from pysatl_cpd.analysis.visualization.utils import translate_linestyle
from pysatl_cpd.core.online import OnlineAlgorithmState, OnlineDetectionTrace

# Plotly layout constants
PLOTLY_GRID_WIDTH = 1
PLOTLY_GRID_COLOR = "lightgray"

# Matplotlib constants
MPL_GRID_ALPHA = 0.3


class DetectionFuncPlotOpts(TypedDict, total=False):
    """Plot options for detection function subplot."""

    xlabel: str
    ylabel: str
    grid: bool


class DetectionFuncDrawOpts(TypedDict, total=False):
    """Drawing options for detection function line."""

    color: str
    linewidth: float
    label: str


class ThresholdDrawOpts(TypedDict, total=False):
    """Drawing options for threshold line."""

    color: str
    linestyle: str
    linewidth: float
    alpha: float
    label: str


class ProcessingTimePlotOpts(TypedDict, total=False):
    """Plot options for processing time subplot."""

    xlabel: str
    ylabel: str
    grid: bool


class ProcessingTimeDrawOpts(TypedDict, total=False):
    """Drawing options for processing time line."""

    color: str
    linewidth: float
    fill_alpha: float
    label: str


class OnlineTraceVisualizer[StateT: OnlineAlgorithmState](ITraceVisualizer[OnlineDetectionTrace[StateT]]):
    """
    Visualizer for online detection trace results.

    This visualizer renders detection function values and processing times.
    Annotations such as change points, learning periods, and skip periods
    should be added by the caller using separate visual components.
    """

    def __init__(
        self,
        backend: DrawBackend,
        state_visualizer: IOnlineStateVisualizer[StateT],
    ) -> None:
        """
        Initialize the online trace visualizer.

        Parameters
        ----------
        backend : DrawBackend
            Plotting backend to use for rendering.
        state_visualizer : IOnlineStateVisualiser[StateT]
            Visualizer for algorithm state evolution.
        """
        super().__init__(backend)
        self._trace: OnlineDetectionTrace[StateT] | None = None
        self._state_visualizer = state_visualizer

        # Options storage with defaults
        self._detection_func_plot_opts: DetectionFuncPlotOpts = {
            "xlabel": "Time Index",
            "ylabel": "Detection Statistic",
            "grid": True,
        }
        self._detection_func_draw_opts: DetectionFuncDrawOpts = {
            "color": "blue",
            "linewidth": 1,
            "label": "Detection Function",
        }
        self._threshold_draw_opts: ThresholdDrawOpts = {
            "color": "green",
            "linestyle": "-" if backend == DrawBackend.MATPLOTLIB else "solid",
            "linewidth": 1,
            "alpha": 0.5,
            "label": "Threshold",
        }
        self._processing_time_plot_opts: ProcessingTimePlotOpts = {
            "xlabel": "Time Index",
            "ylabel": "Time (seconds)",
            "grid": True,
        }
        self._processing_time_draw_opts: ProcessingTimeDrawOpts = {
            "color": "green",
            "linewidth": 1,
            "fill_alpha": 0.3,
            "label": "Processing Time",
        }

    def set_trace(self, trace: OnlineDetectionTrace[StateT]) -> Self:
        """
        Set the detection trace to visualize.

        Parameters
        ----------
        trace : OnlineDetectionTrace[Any, StateT]
            Detection results containing detection function values,
            processing times, and algorithm states.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._trace = trace
        self._state_visualizer.set_states(trace.algorithm_states)
        return self

    def set_detection_func_plot_opts(self, **options: Unpack[DetectionFuncPlotOpts]) -> Self:
        """
        Set general plot options for detection function subplot.

        Parameters
        ----------
        **options : Unpack[DetectionFuncPlotOpts]
            xlabel : str
                X-axis label.
            ylabel : str
                Y-axis label.
            grid : bool
                Whether to show grid lines.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._detection_func_plot_opts.update(options)
        return self

    def set_detection_func_draw_opts(self, **options: Unpack[DetectionFuncDrawOpts]) -> Self:
        """
        Set drawing options for detection function line.

        Parameters
        ----------
        **options : Unpack[DetectionFuncDrawOpts]
            color : str
                Line color.
            linewidth : float
                Line width in points.
            label : str
                Legend label for the detection function line.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._detection_func_draw_opts.update(options)
        return self

    def set_threshold_draw_opts(self, **options: Unpack[ThresholdDrawOpts]) -> Self:
        """
        Set drawing options for threshold line.

        Parameters
        ----------
        **options : Unpack[ThresholdDrawOpts]
            color : str
                Line color.
            linestyle : str
                Line style ('solid', 'dash', etc.).
            linewidth : float
                Line width in points.
            alpha : float
                Line opacity between 0 and 1.
            label : str
                Legend label for the threshold line.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._threshold_draw_opts.update(options)
        return self

    def set_processing_time_plot_opts(self, **options: Unpack[ProcessingTimePlotOpts]) -> Self:
        """
        Set general plot options for processing time subplot.

        Parameters
        ----------
        **options : Unpack[ProcessingTimePlotOpts]
            xlabel : str
                X-axis label.
            ylabel : str
                Y-axis label.
            grid : bool
                Whether to show grid lines.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._processing_time_plot_opts.update(options)
        return self

    def set_processing_time_draw_opts(self, **options: Unpack[ProcessingTimeDrawOpts]) -> Self:
        """
        Set drawing options for processing time line.

        Parameters
        ----------
        **options : Unpack[ProcessingTimeDrawOpts]
            color : str
                Line color.
            linewidth : float
                Line width in points.
            fill_alpha : float
                Opacity of fill under the line between 0 and 1.
            label : str
                Legend label for the processing time line.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._processing_time_draw_opts.update(options)
        return self

    @property
    def axes(self) -> set[str]:
        """
        Declare the subplot names required by this visualizer.

        Returns
        -------
        set[str]
            Set containing "detection_function", "processing_time" subplot names,
            and axes from state visualizer.
        """
        return {"detection_function", "processing_time"} | self._state_visualizer.axes

    def _draw_matplotlib(self, figure: PltFigure, axes: PltAxMapping) -> PltFigure:
        """Draw using Matplotlib backend."""
        if self._trace is None:
            return figure

        # Draw detection function
        if "detection_function" in axes:
            self._mpl_draw_detection_function(axes["detection_function"])

        # Draw processing time
        if "processing_time" in axes:
            self._mpl_draw_processing_time(axes["processing_time"])

        # Draw state evolution
        state_axes = {k: v for k, v in axes.items() if k in self._state_visualizer.axes}
        if state_axes:
            self._state_visualizer._draw_matplotlib(figure, state_axes)

        return figure

    def _mpl_draw_detection_function(self, ax: PltAxes) -> None:
        """
        Draw detection function on Matplotlib axes.

        Parameters
        ----------
        ax : PltAxes
            The Matplotlib axes to draw on.
        """
        if self._trace is None:
            return

        time_points = list(range(len(self._trace.detection_function)))

        # Draw detection function line
        ax.plot(
            time_points,
            self._trace.detection_function,
            color=self._detection_func_draw_opts["color"],
            linewidth=self._detection_func_draw_opts["linewidth"],
            label=self._detection_func_draw_opts.get("label", "Detection Function"),
        )

        # Draw threshold line
        if self._trace.threshold is not None and not np.isnan(self._trace.threshold):
            ax.axhline(
                y=self._trace.threshold,
                color=self._threshold_draw_opts["color"],
                linestyle=translate_linestyle(self._threshold_draw_opts["linestyle"]),
                linewidth=self._threshold_draw_opts["linewidth"],
                alpha=self._threshold_draw_opts["alpha"],
                label=self._threshold_draw_opts.get("label", f"Threshold = {self._trace.threshold:.4f}"),
            )

        # Configure axes
        ax.set_xlabel(self._detection_func_plot_opts["xlabel"])
        ax.set_ylabel(self._detection_func_plot_opts["ylabel"])

        if self._detection_func_plot_opts["grid"]:
            ax.grid(True, alpha=MPL_GRID_ALPHA)

        # Add legend if there are labeled elements
        lines = ax.get_lines()
        if len(lines) > 0:
            ax.legend(loc="best")

    def _mpl_draw_processing_time(self, ax: PltAxes) -> None:
        """
        Draw processing time on Matplotlib axes.

        Parameters
        ----------
        ax : PltAxes
            The Matplotlib axes to draw on.
        """
        if self._trace is None:
            return

        time_points = list(range(len(self._trace.processing_time)))

        label = self._processing_time_draw_opts.get("label", "Processing Time")

        ax.plot(
            time_points,
            self._trace.processing_time,
            color=self._processing_time_draw_opts["color"],
            linewidth=self._processing_time_draw_opts["linewidth"],
            label=label,
        )
        ax.fill_between(
            time_points,
            0,
            self._trace.processing_time,
            alpha=self._processing_time_draw_opts["fill_alpha"],
            color=self._processing_time_draw_opts["color"],
        )

        ax.set_xlabel(self._processing_time_plot_opts["xlabel"])
        ax.set_ylabel(self._processing_time_plot_opts["ylabel"])

        if self._processing_time_plot_opts["grid"]:
            ax.grid(True, alpha=MPL_GRID_ALPHA)

        # Add legend
        ax.legend(loc="best")

    def _draw_plotly(self, figure: GoFigure, axes: GoAxMapping) -> GoFigure:
        """Draw using Plotly backend."""
        if self._trace is None:
            return figure

        # Draw detection function
        if "detection_function" in axes:
            self._plotly_draw_detection_function(figure, axes["detection_function"])

        # Draw processing time
        if "processing_time" in axes:
            self._plotly_draw_processing_time(figure, axes["processing_time"])

        # Draw state evolution
        state_axes = {k: v for k, v in axes.items() if k in self._state_visualizer.axes}
        if state_axes:
            self._state_visualizer._draw_plotly(figure, state_axes)

        return figure

    def _plotly_draw_detection_function(
        self,
        figure: GoFigure,
        ax_pos: GoAxes,
    ) -> None:
        """
        Draw detection function on Plotly subplot.

        Parameters
        ----------
        figure : GoFigure
            The Plotly figure containing the subplot.
        ax_pos : GoAxes
            The subplot position (row, column) to draw on.
        """
        if self._trace is None:
            return

        row, col = ax_pos
        time_points = list(range(len(self._trace.detection_function)))

        # Draw detection function line
        figure.add_trace(
            go.Scatter(
                x=time_points,
                y=self._trace.detection_function,
                mode="lines",
                name=self._detection_func_draw_opts.get("label", "Detection Function"),
                line={
                    "color": self._detection_func_draw_opts["color"],
                    "width": self._detection_func_draw_opts["linewidth"],
                },
                hovertemplate="Step: %{x}, Value: %{y}",
            ),
            row=row,
            col=col,
        )

        # Draw threshold line
        if self._trace.threshold is not None and not np.isnan(self._trace.threshold):
            figure.add_hline(
                y=self._trace.threshold,
                line_color=self._threshold_draw_opts["color"],
                line_dash=self._threshold_draw_opts["linestyle"],
                line_width=self._threshold_draw_opts["linewidth"],
                opacity=self._threshold_draw_opts["alpha"],
                row=row,
                col=col,
                name=self._threshold_draw_opts.get("label", "Threshold"),
            )

        # Configure axes
        figure.update_xaxes(
            title_text=self._detection_func_plot_opts["xlabel"],
            showgrid=self._detection_func_plot_opts["grid"],
            gridwidth=PLOTLY_GRID_WIDTH,
            gridcolor=PLOTLY_GRID_COLOR,
            row=row,
            col=col,
        )
        figure.update_yaxes(
            title_text=self._detection_func_plot_opts["ylabel"],
            showgrid=self._detection_func_plot_opts["grid"],
            gridwidth=PLOTLY_GRID_WIDTH,
            gridcolor=PLOTLY_GRID_COLOR,
            row=row,
            col=col,
        )

    def _plotly_draw_processing_time(
        self,
        figure: GoFigure,
        ax_pos: GoAxes,
    ) -> None:
        """
        Draw processing time on Plotly subplot.

        Parameters
        ----------
        figure : GoFigure
            The Plotly figure containing the subplot.
        ax_pos : GoAxes
            The subplot position (row, column) to draw on.
        """
        if self._trace is None:
            return

        row, col = ax_pos
        time_points = list(range(len(self._trace.processing_time)))

        line_color = self._processing_time_draw_opts["color"]
        line_width = self._processing_time_draw_opts["linewidth"]
        fill_alpha = self._processing_time_draw_opts["fill_alpha"]

        # Convert fill color to rgba
        if line_color.startswith("#"):
            # Convert hex to rgba
            r = int(line_color[1:3], 16)
            g = int(line_color[3:5], 16)
            b = int(line_color[5:7], 16)
            fill_color = f"rgba({r}, {g}, {b}, {fill_alpha})"
        elif line_color.startswith("rgb"):
            fill_color = line_color.replace("rgb", "rgba").replace(")", f", {fill_alpha})")
        else:
            # Named color - use with opacity
            fill_color = line_color

        figure.add_trace(
            go.Scatter(
                x=time_points,
                y=self._trace.processing_time,
                mode="lines",
                name=self._processing_time_draw_opts.get("label", "Processing Time"),
                line={"color": line_color, "width": line_width},
                fill="tozeroy",
                fillcolor=fill_color,
            ),
            row=row,
            col=col,
        )

        # Configure axes
        figure.update_xaxes(
            title_text=self._processing_time_plot_opts["xlabel"],
            showgrid=self._processing_time_plot_opts["grid"],
            gridwidth=PLOTLY_GRID_WIDTH,
            gridcolor=PLOTLY_GRID_COLOR,
            row=row,
            col=col,
        )
        figure.update_yaxes(
            title_text=self._processing_time_plot_opts["ylabel"],
            showgrid=self._processing_time_plot_opts["grid"],
            gridwidth=PLOTLY_GRID_WIDTH,
            gridcolor=PLOTLY_GRID_COLOR,
            row=row,
            col=col,
        )
