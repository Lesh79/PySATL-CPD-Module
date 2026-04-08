# -*- coding: ascii -*-
"""
Visualization components.

This module provides reusable components for common visualization elements
such as vertical lines and vertical fills. These components are backend-agnostic
and can be composed together to build complex visualizations.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.analysis.visualization.components.vert_fill import VerticalFillComponent
from pysatl_cpd.analysis.visualization.components.vert_line import VerticalLineVisualComponent

__all__ = [
    "VerticalLineVisualComponent",
    "VerticalFillComponent",
]
