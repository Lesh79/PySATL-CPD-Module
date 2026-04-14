# -*- coding: ascii -*-
"""
Visualization interfaces for change-point detection results.

This module defines abstract base classes for visualizers that render
time series data, detection traces, and algorithm performance metrics
using either Matplotlib or Plotly backends.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from abc import ABC, abstractmethod
from typing import cast, overload

from pysatl_cpd.analysis.visualization.typedefs import (
    AxMapping,
    DrawBackend,
    Figure,
    GoAxMapping,
    GoFigure,
    PltAxMapping,
    PltFigure,
)


class IVisualizer(ABC):
    """
    Abstract base class for all visualizers.

    This interface defines the contract for visualizers that render data
    onto specific subplots within a figure. Each visualizer declares which
    subplot axes it requires and provides backend-specific drawing methods.

    The visualizer pattern enables composable figure construction, where
    a coordinator creates a figure with named subplots, and each visualizer
    draws its content onto its designated axes.

    Parameters
    ----------
    backend : DrawBackend
        Plotting backend to use for rendering (MATPLOTLIB or PLOTLY).
    """

    def __init__(self, backend: DrawBackend) -> None:
        """
        Initialize the visualizer with the specified backend.

        Parameters
        ----------
        backend : DrawBackend
            Plotting backend to use for rendering.

        Raises
        ------
        ValueError
            If the provided backend is not a valid DrawBackend value.
        """
        if not isinstance(backend, DrawBackend):
            raise ValueError(f"Unsupported backend: {backend}")
        self.__backend = backend

    @property
    def backend(self) -> DrawBackend:
        """
        Return the plotting backend used by this visualizer.

        Returns
        -------
        DrawBackend
            Current backend (MATPLOTLIB or PLOTLY).
        """
        return self.__backend

    @backend.setter
    def backend(self, value: str) -> None:
        if value == DrawBackend.MATPLOTLIB:
            self.__backend = DrawBackend.MATPLOTLIB
        if value == DrawBackend.PLOTLY:
            self.__backend = DrawBackend.PLOTLY
        raise ValueError(f"Unknown backend {value}")

    @property
    @abstractmethod
    def axes(self) -> set[str]:
        """
        Declare the subplot names required by this visualizer.

        Returns
        -------
        set[str]
            Set of subplot identifiers that this visualizer will draw onto.
            These names must correspond to axes provided in the AxMapping
            when the draw method is called.
        """
        raise NotImplementedError

    @overload
    def draw(self, *, figure: PltFigure, axes: PltAxMapping) -> PltFigure: ...

    @overload
    def draw(self, *, figure: GoFigure, axes: GoAxMapping) -> GoFigure: ...

    def draw(self, *, figure: Figure, axes: AxMapping) -> Figure:
        """
        Draw content onto the specified axes.

        This method dispatches to the appropriate backend-specific
        implementation based on the backend argument. The figure is
        inferred from the axes objects (for Matplotlib) or created
        automatically (for Plotly).

        Parameters
        ----------
        axes : AxMapping
            Mapping from subplot names to their axes objects or positions.

        Returns
        -------
        Figure
            The modified figure object.

        Raises
        ------
        ValueError
            If the axes mapping type does not match the backend.
        """
        if self.backend == DrawBackend.PLOTLY:
            go_axes = cast(GoAxMapping, axes)
            return self._draw_plotly(figure, go_axes)

        if self.backend == DrawBackend.MATPLOTLIB:
            plt_axes = cast(PltAxMapping, axes)
            return self._draw_matplotlib(figure, plt_axes)

        # This line should never be reached due to validation in __init__
        raise ValueError(f"Unsupported backend: {self.backend}")

    @abstractmethod
    def _draw_matplotlib(self, figure: PltFigure, axes: PltAxMapping) -> PltFigure:
        """
        Draw using Matplotlib backend.

        Parameters
        ----------
        axes : PltAxMapping
            Mapping from subplot names to Matplotlib axes objects.

        Returns
        -------
        PltFigure
            The figure containing the drawn axes.
        """
        raise NotImplementedError

    @abstractmethod
    def _draw_plotly(self, figure: GoFigure, axes: GoAxMapping) -> GoFigure:
        """
        Draw using Plotly backend.

        Parameters
        ----------
        axes : GoAxMapping
            Mapping from subplot names to Plotly subplot positions.

        Returns
        -------
        GoFigure
            The modified Plotly figure.
        """
        raise NotImplementedError
