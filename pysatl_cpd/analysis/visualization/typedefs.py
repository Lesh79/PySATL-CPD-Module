# -*- coding: ascii -*-
"""
Visualization type definitions.

This module defines common types, enumerations, and aliases used throughout
the visualization module for consistent type hints and backend abstraction.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from enum import StrEnum, auto
from typing import Any

import matplotlib as mpl
import plotly.graph_objs as pltgo

type PltFigure = mpl.figure.Figure
"""
Type alias for Matplotlib figure objects.
"""

type GoFigure = pltgo.Figure
"""
Type alias for Plotly graph objects figure.
"""

type PltAxes = mpl.axes.Axes
"""
Type alias for Matplotlib axes object.
"""

type GoAxes = tuple[int, int]
"""
Type alias for Plotly subplot position.

Each value is a tuple (row, column) specifying the location of the subplot
within the figure's grid layout.
"""

type PltAxMapping = dict[str, PltAxes]
"""
Mapping from subplot names to Matplotlib axes objects.
"""

type GoAxMapping = dict[str, GoAxes]
"""
Mapping from subplot names to Plotly subplot positions.
"""

type AxMapping = PltAxMapping | GoAxMapping
"""
Union type for axes mappings from either Matplotlib or Plotly backends.
"""

type Figure = GoFigure | PltFigure
"""
Union type for figures from either Matplotlib or Plotly backends.
"""


class DrawBackend(StrEnum):
    """
    Supported plotting backends.

    This enumeration defines the available backends for rendering visualizations.
    Each backend provides distinct capabilities: Matplotlib for static publication-
    ready figures, and Plotly for interactive web-based visualizations.

    Attributes
    ----------
    MATPLOTLIB : auto
        Matplotlib backend for static figure generation. Suitable for papers,
        reports, and offline rendering.
    PLOTLY : auto
        Plotly backend for interactive figure generation. Supports zooming,
        panning, and hover tooltips in web environments.
    """

    MATPLOTLIB = auto()
    PLOTLY = auto()


type DrawOption = Any
"""
Type alias for backend-specific drawing options.

This dictionary can contain arbitrary keyword arguments to be passed to
backend drawing functions, such as color schemes, line styles, or rendering
parameters.
"""
