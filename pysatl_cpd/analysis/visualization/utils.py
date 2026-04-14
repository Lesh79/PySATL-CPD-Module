# -*- coding: ascii -*-
"""
Utility functions for visualization module.

This module provides helper functions for common visualization tasks,
including backend-agnostic style translation and figure manipulation.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.analysis.visualization.typedefs import GoAxes, GoFigure

# TODO: Move to plotply color here

def translate_linestyle(linestyle: str) -> str:
    """
    Translate linestyle strings from Plotly to Matplotlib format.

    Parameters
    ----------
    linestyle : str
        Linestyle string in source format (Plotly).

    Returns
    -------
    str
        Linestyle string in target format.

    Raises
    ------
    ValueError
        If linestyle is not recognized.

    Examples
    --------
    >>> translate_linestyle("dash")
    '--'
    >>> translate_linestyle("dashdot")
    '-.'
    """
    # Mapping from Plotly to Matplotlib
    PLOTLY_TO_MPL = {
        "solid": "-",
        "dash": "--",
        "dot": ":",
        "dashdot": "-.",
        "longdash": "--",
        "longdashdot": "-.",
    }

    if linestyle in PLOTLY_TO_MPL:
        return PLOTLY_TO_MPL[linestyle]
    else:
        raise ValueError(f"Unrecognized linestyle: {linestyle}")


def get_subplot_y_limits(fig: GoFigure, axes: GoAxes) -> tuple[float, float]:
    """
    Extract min and max y-values from actual data traces for a Plotly subplot.

    This function analyzes all traces associated with a specific subplot
    and computes the data range, adding a small padding to ensure comfortable
    viewing margins.

    Parameters
    ----------
    fig : GoFigure
        Plotly figure containing the subplot.
    axes : GoAxes
        Subplot position as a tuple (row, column). Row and column indices
        are 1-based as used in Plotly's subplot addressing.

    Returns
    -------
    tuple[float, float]
        (y_min, y_max) representing the recommended y-axis limits for the
        subplot, with padding applied. Returns (0.0, 1.0) if no data found.

    Notes
    -----
    The function uses a hack with dynamic attributes to track whether padding
    has already been applied for a given subplot. Padding is applied only once
    per subplot to avoid recursive expansion.

    The y-axis mapping is determined using Plotly's internal `_grid_ref`
    structure. If not available, a fallback mapping based on row number is used.

    Examples
    --------
    >>> import plotly.graph_objects as go
    >>> fig = go.Figure()
    >>> fig.add_trace(go.Scatter(x=[0,1,2], y=[5,10,15]), row=1, col=1)
    >>> y_min, y_max = get_subplot_y_limits(fig, (1, 1))
    >>> print(f"{y_min:.2f}, {y_max:.2f}")
    4.75, 15.25
    """
    # Extract row and col (1-based indices)
    row, col = axes

    # Determine if padding has already been applied for this subplot
    padding_flag_attr = f"_hack_padding_flag_{row}_{col}"
    padding = 0.00 if hasattr(fig, padding_flag_attr) else 0.05
    if not hasattr(fig, padding_flag_attr):
        setattr(fig, padding_flag_attr, True)

    # Map subplot to yaxis key using _grid_ref
    if hasattr(fig, "_grid_ref") and fig._grid_ref:
        # Find which yaxis this subplot uses
        grid_row = fig._grid_ref[row - 1][col - 1]
        yaxis_key = grid_row[0].layout_keys[1]  # Second element is yaxis key
    else:
        # Fallback mapping: yaxis for first row, yaxis2, yaxis3, etc.
        yaxis_key = "yaxis" if row == 1 else f"yaxis{row}"

    # Collect all y values from traces that use this yaxis
    all_y = []
    for trace in fig.data:
        trace_yaxis = getattr(trace, "yaxis", "y")
        trace_yaxis_key = f"yaxis{trace_yaxis[1:]}" if trace_yaxis != "y" else "yaxis"

        if trace_yaxis_key == yaxis_key and hasattr(trace, "y") and trace.y is not None:
            y_vals = trace.y
            if hasattr(y_vals, "__iter__") and not isinstance(y_vals, str):
                all_y.extend([y for y in y_vals if y is not None])
            elif y_vals is not None:
                all_y.append(y_vals)

    if all_y:
        y_min, y_max = min(all_y), max(all_y)
        y_range = y_max - y_min
        if y_range == 0:
            y_range = abs(y_min) if y_min != 0 else 1
        y_min = y_min - (y_range * padding)
        y_max = y_max + (y_range * padding)
        return y_min, y_max

    return 0.0, 1.0
