# -*- coding: ascii -*-
"""
Online Change-Point Detection Plotter.

This module provides the OnlineCpdPlotter class that coordinates multiple
visualizers and components to create comprehensive visualizations for
change-point detection analysis.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from typing import Any, Self

from pysatl_cpd.analysis.visualization.abstracts import IVisualComponent
from pysatl_cpd.analysis.visualization.online import DummyStateVisualizer, OnlineTraceVisualizer
from pysatl_cpd.analysis.visualization.timeseries import UnivariateTimeseriesVisualizer
from pysatl_cpd.analysis.visualization.typedefs import AxMapping, DrawBackend, Figure
from pysatl_cpd.core.data_providers import DataProvider
from pysatl_cpd.core.online import OnlineDetectionTrace


class OnlineCpdPlotter:
    """
    Coordinator class for online change-point detection visualizations.

    This class manages visualizers and components to create comprehensive
    visualizations. It provides a simplified API for setting up visualizations
    with configurable defaults.

    Parameters
    ----------
    backend : DrawBackend
        The plotting backend to use (MATPLOTLIB or PLOTLY).
    data_provider : DataProvider, optional
        The data provider containing observations.
    detection_trace : OnlineDetectionTrace, optional
        The detection trace from online algorithm execution.
    """

    def __init__(
        self,
        backend: DrawBackend,
        data_provider: DataProvider[Any] | None = None,
        detection_trace: OnlineDetectionTrace[Any] | None = None,
    ) -> None:
        """
        Initialize the plotter with the specified backend.

        Parameters
        ----------
        backend : DrawBackend
            The plotting backend to use for rendering.
        data_provider : DataProvider, optional
            The data provider containing observations.
        detection_trace : OnlineDetectionTrace, optional
            The detection trace from online algorithm execution.
        """
        self._backend = backend
        self._data_provider = data_provider
        self._detection_trace = detection_trace

        # Visualizers
        self._timeseries_visualizer: UnivariateTimeseriesVisualizer | None = None
        self._trace_visualizer: OnlineTraceVisualizer[Any] | None = None

        # Components storage: name -> (component, axes_names, show_legend)
        self._components: dict[str, tuple[IVisualComponent, list[str], bool]] = {}

        # Initialize defaults
        self._create_default_visualizers()

    @property
    def backend(self) -> DrawBackend:
        """Return the plotting backend."""
        return self._backend

    @property
    def timeseries_visualizer(self) -> UnivariateTimeseriesVisualizer | None:
        """Return the timeseries visualizer."""
        return self._timeseries_visualizer

    @property
    def trace_visualizer(self) -> OnlineTraceVisualizer[Any] | None:
        """Return the trace visualizer."""
        return self._trace_visualizer

    def set_data_provider(self, data_provider: DataProvider[Any]) -> Self:
        """
        Set the data provider for the timeseries visualizer.

        Parameters
        ----------
        data_provider : DataProvider
            Data provider containing observations.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._data_provider = data_provider
        if self._timeseries_visualizer is not None:
            self._timeseries_visualizer.set_data_provider(data_provider)
        return self

    def set_detection_trace(self, detection_trace: OnlineDetectionTrace[Any]) -> Self:
        """
        Set the detection trace for the trace visualizer.

        Parameters
        ----------
        detection_trace : OnlineDetectionTrace
            Detection trace from online algorithm execution.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._detection_trace = detection_trace
        if self._trace_visualizer is not None:
            self._trace_visualizer.set_trace(detection_trace)
        return self

    def _create_default_visualizers(self) -> None:
        """Create default visualizer instances."""
        # Create timeseries visualizer
        self._timeseries_visualizer = UnivariateTimeseriesVisualizer(backend=self._backend)
        if self._data_provider is not None:
            self._timeseries_visualizer.set_data_provider(self._data_provider)

        state_visualizer = DummyStateVisualizer[Any](backend=self._backend)
        self._trace_visualizer = OnlineTraceVisualizer[Any](backend=self._backend, state_visualizer=state_visualizer)
        if self._detection_trace is not None:
            self._trace_visualizer.set_trace(self._detection_trace)

    def configure_timeseries(
        self,
        xlabel: str = "Time Index",
        ylabel: str = "Value",
        grid: bool = True,
        legend: bool = True,
        color: str = "black",
        linewidth: float = 1.5,
        alpha: float = 0.7,
    ) -> Self:
        """
        Configure the timeseries visualizer.

        Parameters
        ----------
        xlabel : str, default="Time Index"
            X-axis label.
        ylabel : str, default="Value"
            Y-axis label.
        grid : bool, default=True
            Whether to show grid lines.
        legend : bool, default=True
            Whether to show legend.
        color : str, default="black"
            Line color.
        linewidth : float, default=1.5
            Line width.
        alpha : float, default=0.7
            Line opacity.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        if self._timeseries_visualizer is not None:
            self._timeseries_visualizer.set_plot_opts(xlabel=xlabel, ylabel=ylabel, grid=grid, legend=legend)
            self._timeseries_visualizer.set_draw_opts(color=color, linewidth=linewidth, alpha=alpha)
        return self

    def configure_detection_func(
        self,
        xlabel: str = "Time Index",
        ylabel: str = "Detection Statistic",
        grid: bool = True,
        color: str = "blue",
        linewidth: float = 1,
        threshold_color: str = "red",
        threshold_linestyle: str = "dash",
        threshold_linewidth: float = 2,
        threshold_alpha: float = 0.8,
    ) -> Self:
        """
        Configure the detection function subplot.

        Parameters
        ----------
        xlabel : str, default="Time Index"
            X-axis label.
        ylabel : str, default="Detection Statistic"
            Y-axis label.
        grid : bool, default=True
            Whether to show grid lines.
        color : str, default="blue"
            Detection function line color.
        linewidth : float, default=1
            Detection function line width.
        threshold_color : str, default="red"
            Threshold line color.
        threshold_linestyle : str, default="dash"
            Threshold line style.
        threshold_linewidth : float, default=2
            Threshold line width.
        threshold_alpha : float, default=0.8
            Threshold line opacity.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        if self._trace_visualizer is not None:
            self._trace_visualizer.set_detection_func_plot_opts(xlabel=xlabel, ylabel=ylabel, grid=grid)
            self._trace_visualizer.set_detection_func_draw_opts(color=color, linewidth=linewidth)
            self._trace_visualizer.set_threshold_draw_opts(
                color=threshold_color,
                linestyle=threshold_linestyle,
                linewidth=threshold_linewidth,
                alpha=threshold_alpha,
            )
        return self

    def configure_processing_time(
        self,
        xlabel: str = "Time Index",
        ylabel: str = "Time (seconds)",
        grid: bool = True,
        color: str = "purple",
        linewidth: float = 1,
        fill_alpha: float = 0.3,
    ) -> Self:
        """
        Configure the processing time subplot.

        Parameters
        ----------
        xlabel : str, default="Time Index"
            X-axis label.
        ylabel : str, default="Time (seconds)"
            Y-axis label.
        grid : bool, default=True
            Whether to show grid lines.
        color : str, default="purple"
            Processing time line color.
        linewidth : float, default=1
            Processing time line width.
        fill_alpha : float, default=0.3
            Opacity of fill under the line.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        if self._trace_visualizer is not None:
            self._trace_visualizer.set_processing_time_plot_opts(
                xlabel=xlabel,
                ylabel=ylabel,
                grid=grid,
            )
            self._trace_visualizer.set_processing_time_draw_opts(
                color=color, linewidth=linewidth, fill_alpha=fill_alpha
            )
        return self

    def add_component(
        self,
        name: str,
        component: IVisualComponent,
        axes_names: list[str],
        show_legend: bool = True,
    ) -> Self:
        """
        Add a visual component to the plotter.

        Parameters
        ----------
        name : str
            Name to identify the component.
        component : IVisualComponent
            Component instance to add.
        axes_names : list[str]
            Names of axes where the component should be drawn. Must be provided.
        show_legend : bool, default=True
            Whether to show legend for this component.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._components[name] = (component, axes_names, show_legend)
        return self

    def remove_component(self, name: str) -> Self:
        """
        Remove a component from the plotter.

        Parameters
        ----------
        name : str
            Name of the component to remove.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        if name in self._components:
            del self._components[name]
        return self

    def get_component(self, name: str) -> IVisualComponent | None:
        """
        Get a component by name.

        Parameters
        ----------
        name : str
            Name of the component to retrieve.

        Returns
        -------
        IVisualComponent | None
            The component instance or None if not found.
        """
        component_data = self._components.get(name)
        if component_data:
            return component_data[0]
        return None

    def draw(self, figure: Figure, axes: AxMapping) -> Figure:
        """
        Coordinate drawing of all visualizers and components.

        Parameters
        ----------
        figure : Figure
            The figure to draw on (Matplotlib or Plotly).
        axes : AxMapping
            Mapping from subplot names to their axes objects or positions.

        Returns
        -------
        Figure
            The modified figure with all visual elements drawn.
        """
        # Draw all visualizers
        if self._timeseries_visualizer is not None:
            figure = self._timeseries_visualizer.draw(figure=figure, axes=axes)

        if self._trace_visualizer is not None:
            figure = self._trace_visualizer.draw(figure=figure, axes=axes)

        # Draw all components
        for component, axes_names, show_legend in self._components.values():
            for axes_name in axes_names:
                if axes_name in axes:
                    component.draw(figure, axes[axes_name], add_legend=show_legend)

        return figure

    @property
    def required_axes(self) -> set[str]:
        """
        Return the set of all axes names required by visualizers and components.

        Returns
        -------
        set[str]
            Set of subplot names needed for drawing.
        """
        axes_set: set[str] = set()

        # Add axes from visualizers
        if self._timeseries_visualizer is not None:
            axes_set.update(self._timeseries_visualizer.axes)
        if self._trace_visualizer is not None:
            axes_set.update(self._trace_visualizer.axes)

        # Add axes from components
        for _, axes_names, _ in self._components.values():
            axes_set.update(axes_names)

        return axes_set
