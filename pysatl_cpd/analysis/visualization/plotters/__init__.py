# -*- coding: ascii -*-
"""
Plotter classes for change-point detection visualization.

This module provides coordinator classes that manage multiple visualizers
and components to create comprehensive visualizations.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.analysis.visualization.plotters.default import OnlineCpdPlotter

__all__ = [
    "OnlineCpdPlotter",
]
