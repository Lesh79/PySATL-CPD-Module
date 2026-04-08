# -*- coding: ascii -*-
"""
Visualization component interfaces.

This module defines the abstract base class for composable visualization
components that draw specific elements (change points, segments, annotations)
onto a single subplot. Components are backend-agnostic and provide separate
implementations for Matplotlib and Plotly backends.

Components must support optional legend entries with the ability to toggle
visibility interactively in Plotly through legend groups.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from typing import Self, overload

from pysatl_cpd.analysis.visualization.typedefs import DrawBackend, Figure, GoAxes, GoFigure, PltAxes, PltFigure


class IVisualComponent(ABC):
    """
    Base interface for composable visualization components.

    Components are responsible for drawing specific elements (e.g., change points,
    segments, annotations) onto a single subplot. They are backend-agnostic and
    implement both Matplotlib and Plotly drawing methods.
    """

    def __init__(self, backend: DrawBackend) -> None:
        """
        Initialize the component with a specific backend.

        Parameters
        ----------
        backend : DrawBackend
            The backend to use for drawing operations.
        """
        self.__backend = backend
        self._legend_label: str | None = None

    @property
    def backend(self) -> DrawBackend:
        return self.__backend

    @backend.setter
    def backend(self, value: str) -> None:
        if value == DrawBackend.MATPLOTLIB:
            self.__backend = DrawBackend.MATPLOTLIB
            return
        if value == DrawBackend.PLOTLY:
            self.__backend = DrawBackend.PLOTLY
            return
        raise ValueError(f"Unknown backend {value}")

    def set_legend_label(self, legend_label: str) -> Self:
        self._legend_label = legend_label
        return self

    @overload
    def draw(self, figure: PltFigure, axes: PltAxes, add_legend: bool = False) -> None:
        """
        Draw component on a Matplotlib axes.

        Parameters
        ----------
        figure : PltFigure
            The Matplotlib figure containing the axes.
        axes : PltAxes
            The Matplotlib axes to draw on.
        add_legend : bool, default=False
            Whether to add legend entry for this component.
        """

    @overload
    def draw(self, figure: GoFigure, axes: GoAxes, add_legend: bool = False) -> None:
        """
        Draw component on a Plotly subplot.

        Parameters
        ----------
        figure : GoFigure
            The Plotly figure containing the subplot.
        axes : GoAxes
            The subplot position (row, col) to draw on.
        add_legend : bool, default=False
            Whether to display legend entry for this component.
        """

    def draw(self, figure: Figure, axes: GoAxes | PltAxes, add_legend: bool = False) -> None:
        """
        Draw component on a subplot.

        Parameters
        ----------
        figure : Figure
            The figure containing the subplot (Matplotlib or Plotly).
        axes : GoAxes | PltAxes
            The subplot data to draw on.
        add_legend : bool, default=False
            Whether to add (for Matplotlib) or display (for Plotly)
            legend entry for this component

        Notes
        ----------
        For Plotly backend it is assumed that legend entry always
        added within legendgroup with same label. This ensures that
        interactive disabling/enabling visualization of corresponding
        traces works.

        In Plotly, it is possible assign a trace to only one legendgroup,
        but  one can achieve use nested legend groups with Plotly buttons.
        This allows to group traces by multiple criteria and toggle them
        collectively.
        """
        if self._legend_label is None and add_legend:
            raise ValueError("Can not draw with legend: label is set to None")

        if self.__backend == DrawBackend.MATPLOTLIB:
            if not isinstance(figure, PltFigure.__value__):
                raise TypeError(f"Expected PltFigure for Matplotlib backend, got {type(figure)}")
            if not isinstance(axes, PltAxes.__value__):
                raise TypeError(f"Expected PltAxes for Matplotlib backend, got {type(axes)}")
            self._draw_matplotlib(figure, axes, add_legend)
        elif self.__backend == DrawBackend.PLOTLY:
            if not isinstance(figure, GoFigure.__value__):
                raise TypeError(f"Expected GoFigure for Plotly backend, got {type(figure)}")
            if not (
                isinstance(axes, tuple) and (len(axes) == 2) and isinstance(axes[0], int) and isinstance(axes[1], int)
            ):
                raise TypeError(f"Expected tuple of ints for Plotly backend, got {axes}")
            self._draw_plotly(figure, axes, add_legend)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

    @abstractmethod
    def _draw_matplotlib(self, figure: PltFigure, axes: PltAxes, add_legend: bool = False) -> None:
        """
        Backend-specific drawing implementation for Matplotlib.
        """
        raise NotImplementedError

    @abstractmethod
    def _draw_plotly(self, figure: GoFigure, axes: GoAxes, add_legend: bool = False) -> None:
        """
        Backend-specific drawing implementation for Plotly.
        """
        raise NotImplementedError
