# -*- coding: ascii -*-
"""
Vertical fill visualization component.

This module provides a component for visualizing filled vertical regions
between x-coordinates.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Sequence
from typing import Self, TypedDict, Unpack

import plotly.graph_objects as go

from pysatl_cpd.analysis.visualization.abstracts.icomponent import IVisualComponent
from pysatl_cpd.analysis.visualization.typedefs import DrawBackend, GoAxes, GoFigure, PltAxes, PltFigure
from pysatl_cpd.analysis.visualization.utils import get_subplot_y_limits


class VerticalFillStyle(TypedDict, total=True):
    """Style configuration for vertical fills."""

    fill_color: str
    fill_alpha: float


class VerticalFillComponent(IVisualComponent):
    """
    Component for visualizing filled vertical regions between x-coordinates.
    """

    def __init__(self, backend: DrawBackend) -> None:
        super().__init__(backend)
        self._regions: list[tuple[float, float]] = []

        self._style: VerticalFillStyle = {
            "fill_color": "gray",
            "fill_alpha": 0.3,
        }

    def set_regions(self, regions: Sequence[tuple[float, float]]) -> Self:
        """
        Set vertical regions to fill.

        Parameters
        ----------
        regions : Sequence[tuple[float, float]]
            List of (start, end) x-coordinate pairs defining vertical regions.
        """
        self._regions = list(regions)
        return self

    def set_style(self, **style: Unpack[VerticalFillStyle]) -> Self:
        """
        Set style for vertical fills.

        Parameters
        ----------
        **style : Unpack[VerticalFillStyle]
            fill_color : str
                Color for the filled region.
            fill_alpha : float
                Opacity of the fill between 0 and 1.
        """
        self._style.update(style)
        return self

    def _draw_matplotlib(self, figure: PltFigure, axes: PltAxes, add_legend: bool = False) -> None:
        """
        Draw vertical fills on Matplotlib axes.

        Parameters
        ----------
        figure : PltFigure
            The Matplotlib figure containing the axes.
        axes : PltAxes
            The Matplotlib axes to draw on.
        add_legend : bool, default=False
            Whether to add legend entry for these fills.
        """
        if not self._regions:
            return
        plot_opts = {
            "alpha": self._style["fill_alpha"],
            "color": self._style["fill_color"],
        }
        for i, region in enumerate(self._regions):
            axes.axvspan(
                *region,
                **plot_opts,  # type: ignore
                label=self._legend_label if (add_legend and i == 0) else None,
            )

    def _draw_plotly(self, figure: GoFigure, axes: GoAxes, add_legend: bool = False) -> None:
        """
        Draw vertical fills on Plotly subplot.

        Parameters
        ----------
        figure : GoFigure
            The Plotly figure containing the subplot.
        axes : GoAxes
            The subplot position (row, column) to draw on.
        add_legend : bool, default=False
            Whether to add legend entry for these fills.
        """
        if not self._regions:
            return
        plot_opts = {
            "fill": "toself",
            "fillcolor": self._style["fill_color"],
            "opacity": self._style["fill_alpha"],
        }

        legend_opts = {
            "name": self._legend_label,
            "legendgroup": self._legend_label,
        }

        # Build coordinates for all rectangles in one trace
        x_coords = []
        y_coords = []
        y_min, y_max = get_subplot_y_limits(figure, axes)

        # Build hover points
        hover_x = []
        hover_y = []
        hover_t = []
        spacing = 11
        y_hover_poses = [y_min + float(x) / spacing * (y_max - y_min) for x in range(spacing)]

        for start, end in self._regions:
            # Add rectangle coordinates with None separators
            x_coords.extend([start, start, end, end, start, None])
            y_coords.extend([y_min, y_max, y_max, y_min, y_min, None])

            # Add multiple hover points within the rectangle area
            x_mid = (start + end) / 2
            hover_x.extend([x_mid] * spacing)
            hover_y.extend(y_hover_poses)
            hover_t.extend([f"[{start:.0f}, {end:.0f}] {self._legend_label}"] * spacing)

        # Add rectangle trace (no hover)
        figure.add_trace(
            go.Scatter(
                x=x_coords,
                y=y_coords,
                mode="lines",
                line={"width": 0},
                showlegend=add_legend,
                hoverinfo="skip",  # Rectangle doesn't show hover
                **plot_opts,
                **legend_opts,
            ),
            row=axes[0],
            col=axes[1],
        )

        # Add invisible points for hover detection
        figure.add_trace(
            go.Scatter(
                x=hover_x,
                y=hover_y,
                mode="markers",
                marker={
                    "color": self._style["fill_color"],
                    "size": 8,
                    "opacity": 0,  # Completely invisible
                },
                hoverinfo="text",
                hovertext=hover_t,
                hovertemplate="%{hovertext}<extra></extra>",
                showlegend=False,
                legendgroup=self._legend_label,
            ),
            row=axes[0],
            col=axes[1],
        )

    def clear(self) -> Self:
        """
        Clear all vertical regions.

        Returns
        -------
        Self
            Returns self to allow method chaining.
        """
        self._regions.clear()
        return self
