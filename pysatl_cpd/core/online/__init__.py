# -*- coding: ascii -*-
"""
Core online detection components.

This module contains the core interfaces and classes for online change-point detection,
including algorithm base classes, solvers, and result containers.
"""

__author__ = "Alexey Tatyanenko, Vladimir Kutuev, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from pysatl_cpd.core.online.ionline_algorithm import (
    OnlineAlgorithm,
    OnlineAlgorithmConfiguration,
    OnlineAlgorithmState,
)
from pysatl_cpd.core.online.online_cpd_solver import OnlineCpdSolver
from pysatl_cpd.core.online.online_detection_trace import (
    OnlineDetectionStepResult,
    OnlineDetectionTrace,
)

__all__ = [
    "OnlineAlgorithm",
    "OnlineAlgorithmState",
    "OnlineAlgorithmConfiguration",
    "OnlineCpdSolver",
    "OnlineDetectionStepResult",
    "OnlineDetectionTrace",
]
