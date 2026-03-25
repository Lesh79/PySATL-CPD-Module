# -*- coding: ascii -*-
"""
Core module for PySATL CPD library.

This module contains the fundamental building blocks for change-point detection,
including data providers, detection traces, and base classes for online algorithms.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from pysatl_cpd.core import data_providers, online, typedefs
from pysatl_cpd.core.detection_trace import DetectionTrace
from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace

__all__ = [
    "data_providers",
    "online",
    "typedefs",
    "DetectionTrace",
    "OnlineDetectionTrace",
]
