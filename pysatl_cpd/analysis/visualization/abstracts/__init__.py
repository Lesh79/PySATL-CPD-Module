# -*- coding: ascii -*-
"""
Abstract visualization interfaces.

This module exports the abstract base classes that define the contracts for
visualizers and components in the PySATL CPD visualization system.

Architecture Patterns
---------------------
The visualization module implements two complementary design patterns:

1. **Strategy Pattern** (IVisualizer hierarchy)
   - Visualizers are responsible for rendering complete subplots from data
   - Each visualizer encapsulates the drawing logic for a specific data type
     (time series, detection trace, algorithm state)
   - Backend selection (Matplotlib/Plotly) determines which drawing strategy
     is invoked through the template method pattern

2. **Component Pattern** (IVisualComponent)
   - Components add discrete visual elements (change point markers, period fills,
     annotation lines) to existing subplots
   - Multiple components can be composed on the same subplot
   - Components are lightweight and focused on a single visual concern
   - Supports optional legend entries with interactive toggling in Plotly

Separation of Concerns
----------------------
- **Visualizers**: Own their subplot completely, configure axes properties,
  and draw the primary data content. They do not know about other visualizers.
- **Components**: Draw supplementary annotations on subplots created by visualizers.
  They are independent and can be combined arbitrarily.
- **Coordinator** (not part of this module): Creates figure layout, instantiates
  visualizers and components, and orchestrates drawing order.

Backend Abstraction
-------------------
All visualizers and components implement both `_draw_matplotlib()` and
`_draw_plotly()` methods. The public `draw()` method dispatches to the
appropriate backend-specific implementation based on the `backend` property,
ensuring type safety and consistent behavior across plotting libraries.

Legend Management
-----------------
- Components support optional legend labels via `set_legend_label()`
- When `add_legend=True` is passed to `draw()`, the component adds its
  legend entry to the subplot
- For Plotly, legend entries are grouped by label, allowing interactive
  toggling of all traces with the same label

Class Overview
--------------
- `IVisualizer`: Base interface for all visualizers that manage complete subplots
- `ITimeseriesVisualizer`: Interface for time series data visualizers
- `ITraceVisualizer`: Interface for detection trace visualizers
- `IVisualComponent`: Interface for composable drawing components that add
  specific elements (change points, segments, annotations) to existing subplots
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.analysis.visualization.abstracts.icomponent import IVisualComponent
from pysatl_cpd.analysis.visualization.abstracts.itimeseries_visualizer import (
    ITimeseriesVisualizer,
)
from pysatl_cpd.analysis.visualization.abstracts.itrace_visualizer import ITraceVisualizer
from pysatl_cpd.analysis.visualization.abstracts.ivisualizer import IVisualizer

__all__ = [
    "IVisualizer",
    "ITimeseriesVisualizer",
    "ITraceVisualizer",
    "IVisualComponent",
]
