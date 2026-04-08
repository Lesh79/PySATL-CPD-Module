# -*- coding: ascii -*-
"""
State visualizers for online algorithm state evolution.

This module provides visualizers for rendering algorithm state evolution over time.
State visualizers implement the IOnlineStateVisualizer interface and are responsible
for displaying internal algorithm statistics (e.g., running means, control limits,
window statistics) as the algorithm processes observations sequentially.

The module follows a composable architecture where state visualizers can be
combined with other visualizers (e.g., OnlineTraceVisualizer) to create complete
detection visualizations.

Classes
-------
IOnlineStateVisualizer
    Abstract base interface for all online state visualizers.
DummyStateVisualizer
    Placeholder visualizer that performs no rendering, useful for testing
    or when state visualization is not required.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.analysis.visualization.online.states.ionline_state_visualizer import (
    IOnlineStateVisualizer,
)
from pysatl_cpd.analysis.visualization.online.states.state_dummy_visualizer import (
    DummyStateVisualizer,
)

__all__ = [
    "IOnlineStateVisualizer",
    "DummyStateVisualizer",
]
