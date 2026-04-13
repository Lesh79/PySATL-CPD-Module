"""
Shewhart state visualizer implementation.

This module provides a visualizer for rendering Shewhart control chart
algorithm state evolution over time, including running mean, control limits,
and sliding window mean.
"""

from collections.abc import Sequence
from typing import Self, TypedDict, Unpack

import numpy as np
import plotly.graph_objects as go

from pysatl_cpd.algorithms.online.shewhart_control_chart import ShewhartControlChartState
from pysatl_cpd.analysis.visualization.online.states.ionline_state_visualizer import (
    IOnlineStateVisualizer,
)
from pysatl_cpd.analysis.visualization.typedefs import (
    DrawBackend,
    DrawOption,
    GoAxMapping,
    GoFigure,
    PltAxMapping,
    PltFigure,
)

# Plotly constants
PLOTLY_GRID_WIDTH = 1
PLOTLY_GRID_COLOR = "lightgray"

# Matplotlib constants
MPL_GRID_ALPHA = 0.3


class ShewhartStatePlotOpts(TypedDict, total=False):
    """Plot options for Shewhart state subplot."""

    xlabel: str
    ylabel: str
    grid: bool


class ShewhartStateDrawOpts(TypedDict, total=False):
    """Drawing options for Shewhart state lines."""

    mean_color: str
    mean_linewidth: float
    mean_label: str
    control_limit_color: str
    control_limit_linestyle: str
    control_limit_linewidth: float
    control_limit_label: str
    window_mean_color: str
    window_mean_linewidth: float
    window_mean_label: str
    fill_alpha: float


class ShewhartStateBandOpts(TypedDict, total=False):
    """Band calculation options for Shewhart control limits."""

    band_size: float


class ShewhartStateVisualizer(IOnlineStateVisualizer[ShewhartControlChartState]):
    """
    Visualizer for Shewhart control chart algorithm state evolution.

    This visualizer renders:
    - Running mean (μ)
    - Control limits (μ ± k * σ / √w) where w is window size
    - Sliding window mean (x̄_w)

    Parameters
    ----------
    backend : DrawBackend
        Plotting backend to use for rendering.
    """

    def __init__(self, backend: DrawBackend) -> None:
        """
        Initialize the Shewhart state visualizer.

        Parameters
        ----------
        backend : DrawBackend
            Plotting backend to use for rendering.
        """
        super().__init__(backend)
        self._states: Sequence[ShewhartControlChartState | None] = []

        # Options storage with defaults
        self._plot_opts: ShewhartStatePlotOpts = {
            "xlabel": "Time Index",
            "ylabel": "Value",
            "grid": True,
        }
        self._draw_opts: ShewhartStateDrawOpts = {
            "mean_color": "blue",
            "mean_linewidth": 1.5,
            "mean_label": "Running Mean (μ)",
            "control_limit_color": "red",
            "control_limit_linestyle": "--" if backend == DrawBackend.MATPLOTLIB else "dash",
            "control_limit_linewidth": 1,
            "control_limit_label": "Control Limits (μ ± k·σ/√w)",
            "window_mean_color": "green",
            "window_mean_linewidth": 1,
            "window_mean_label": "Window Mean (x̄_w)",
            "fill_alpha": 0.2,
        }
        self._band_opts: ShewhartStateBandOpts = {
            "band_size": 3.0,
        }

    @property
    def axes(self) -> set[str]:
        """
        Declare the subplot names required by this visualizer.

        Returns
        -------
        set[str]
            Set containing "shewhart_state" subplot name.
        """
        return {"shewhart_state"}

    def set_states(
        self,
        states: Sequence[ShewhartControlChartState | None],
        **draw_options: DrawOption,
    ) -> Self:
        """
        Set the sequence of algorithm states to visualize.

        Parameters
        ----------
        states : Sequence[ShewhartControlChartState | None]
            Sequence of Shewhart state snapshots for each observation step.
            None values indicate steps where state was not captured.
        **draw_options : DrawOption
            Additional backend-specific drawing options.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._states = states

        # Extract data from states
        self._time_points = np.arange(len(self._states))
        self._means = np.array(
            [
                state.mean if (state is not None and not state.is_in_learning_period) else np.nan
                for state in self._states
            ]
        )
        self._stds = np.array(
            [
                state.standard_deviation if (state is not None and not state.is_in_learning_period) else np.nan
                for state in self._states
            ]
        )
        self._window_means = np.array(
            [
                state.window_mean if (state is not None and not state.is_in_learning_period) else np.nan
                for state in self._states
            ]
        )
        self._window_sizes = np.array(
            [
                state.window_size if (state is not None and not state.is_in_learning_period) else 1
                for state in self._states
            ]
        )

        # Calculate control limits: μ ± k * σ / √w
        band_scale = self._band_opts["band_size"]
        self._half_band = band_scale * self._stds / np.sqrt(self._window_sizes)
        self._upper_limits = self._means + self._half_band
        self._lower_limits = self._means - self._half_band

        return self

    def set_plot_opts(self, **options: Unpack[ShewhartStatePlotOpts]) -> Self:
        """
        Set general plot options for Shewhart state subplot.

        Parameters
        ----------
        **options : Unpack[ShewhartStatePlotOpts]
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
        self._plot_opts.update(options)
        return self

    def set_draw_opts(self, **options: Unpack[ShewhartStateDrawOpts]) -> Self:
        """
        Set drawing options for Shewhart state lines.

        Parameters
        ----------
        **options : Unpack[ShewhartStateDrawOpts]
            mean_color : str
                Color of the running mean line.
            mean_linewidth : float
                Width of the running mean line.
            mean_label : str
                Legend label for running mean.
            control_limit_color : str
                Color of the control limit lines.
            control_limit_linestyle : str
                Line style for control limits.
            control_limit_linewidth : float
                Width of the control limit lines.
            control_limit_label : str
                Legend label for control limits.
            window_mean_color : str
                Color of the window mean line.
            window_mean_linewidth : float
                Width of the window mean line.
            window_mean_label : str
                Legend label for window mean.
            fill_alpha : float
                Opacity of the fill between control limits.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._draw_opts.update(options)
        return self

    def set_band_opts(self, **options: Unpack[ShewhartStateBandOpts]) -> Self:
        """
        Set band calculation options for control limits.

        Parameters
        ----------
        **options : Unpack[ShewhartStateBandOpts]
            band_size : float
                Multiplier k for control limits (μ ± k * σ / √w).
                Default is 3.0.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._band_opts.update(options)
        # Recalculate control limits with new band size
        if hasattr(self, "_states") and self._states:
            band_scale = self._band_opts["band_size"]
            self._half_band = band_scale * self._stds / np.sqrt(self._window_sizes)
            self._upper_limits = self._means + self._half_band
            self._lower_limits = self._means - self._half_band
        return self

    def _draw_matplotlib(self, figure: PltFigure, axes: PltAxMapping) -> PltFigure:
        """Draw using Matplotlib backend."""
        if "shewhart_state" not in axes:
            return figure

        if len(self._states) == 0:
            return figure

        ax = axes["shewhart_state"]

        # Draw window mean first (bottom layer)
        ax.plot(
            self._time_points,
            self._window_means,
            color=self._draw_opts["window_mean_color"],
            linewidth=self._draw_opts["window_mean_linewidth"],
            label=self._draw_opts["window_mean_label"],
            zorder=1,
        )

        # Draw fill between control limits
        ax.fill_between(
            self._time_points,
            self._lower_limits,
            self._upper_limits,
            alpha=self._draw_opts["fill_alpha"],
            color=self._draw_opts["control_limit_color"],
            label="Control Band",
            zorder=2,
        )

        # Draw control limits (upper and lower)
        ax.plot(
            self._time_points,
            self._upper_limits,
            color=self._draw_opts["control_limit_color"],
            linestyle=self._draw_opts["control_limit_linestyle"],
            linewidth=self._draw_opts["control_limit_linewidth"],
            label=self._draw_opts["control_limit_label"],
            zorder=3,
        )

        ax.plot(
            self._time_points,
            self._lower_limits,
            color=self._draw_opts["control_limit_color"],
            linestyle=self._draw_opts["control_limit_linestyle"],
            linewidth=self._draw_opts["control_limit_linewidth"],
            zorder=3,
        )

        # Draw running mean on top
        ax.plot(
            self._time_points,
            self._means,
            color=self._draw_opts["mean_color"],
            linewidth=self._draw_opts["mean_linewidth"],
            label=self._draw_opts["mean_label"],
            zorder=4,
        )

        # Configure axes
        ax.set_xlabel(self._plot_opts["xlabel"])
        ax.set_ylabel(self._plot_opts["ylabel"])

        if self._plot_opts["grid"]:
            ax.grid(True, alpha=MPL_GRID_ALPHA, zorder=0)

        ax.legend(loc="best")

        return figure

    def _draw_plotly(self, figure: GoFigure, axes: GoAxMapping) -> GoFigure:
        """Draw using Plotly backend."""
        if "shewhart_state" not in axes:
            return figure
        if len(self._states) == 0:
            return figure

        row, col = axes["shewhart_state"]

        # Draw window mean first (bottom layer)
        figure.add_trace(
            go.Scatter(
                x=self._time_points,
                y=self._window_means,
                mode="lines",
                name=self._draw_opts["window_mean_label"],
                line={
                    "color": self._draw_opts["window_mean_color"],
                    "width": self._draw_opts["window_mean_linewidth"],
                },
            ),
            row=row,
            col=col,
        )

        # Draw fill between control limits
        figure.add_trace(
            go.Scatter(
                x=np.concatenate([self._time_points, self._time_points[::-1]]),
                y=np.concatenate([self._upper_limits, self._lower_limits[::-1]]),
                fill="toself",
                fillcolor=f"rgba(255, 0, 0, {self._draw_opts['fill_alpha']})",
                line={"color": "rgba(255, 0, 0, 0)"},
                name="Control Band",
                showlegend=True,
            ),
            row=row,
            col=col,
        )

        # Draw upper control limit
        figure.add_trace(
            go.Scatter(
                x=self._time_points,
                y=self._upper_limits,
                mode="lines",
                name=self._draw_opts["control_limit_label"],
                line={
                    "color": self._draw_opts["control_limit_color"],
                    "dash": self._draw_opts["control_limit_linestyle"],
                    "width": self._draw_opts["control_limit_linewidth"],
                },
            ),
            row=row,
            col=col,
        )

        # Draw lower control limit (without label to avoid duplicate legend entries)
        figure.add_trace(
            go.Scatter(
                x=self._time_points,
                y=self._lower_limits,
                mode="lines",
                name=None,
                showlegend=False,
                line={
                    "color": self._draw_opts["control_limit_color"],
                    "dash": self._draw_opts["control_limit_linestyle"],
                    "width": self._draw_opts["control_limit_linewidth"],
                },
            ),
            row=row,
            col=col,
        )

        # Draw running mean on top
        figure.add_trace(
            go.Scatter(
                x=self._time_points,
                y=self._means,
                mode="lines",
                name=self._draw_opts["mean_label"],
                line={
                    "color": self._draw_opts["mean_color"],
                    "width": self._draw_opts["mean_linewidth"],
                },
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
