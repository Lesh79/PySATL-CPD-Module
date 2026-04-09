# -*- coding: ascii -*-

"""
Mock core components for testing.

This module provides basic mock implementations of core data structures,
such as detection traces, used across testing suites.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence

from pysatl_cpd.core.detection_trace import DetectionTrace


class MockDetectionTrace(DetectionTrace):
    """
    Mock implementation of DetectionTrace for testing purposes.

    Provides a simple trace with predefined detected change points,
    a dummy algorithm name, and a dummy configuration hash.

    Parameters
    ----------
    detected_change_points : Sequence[int]
        A sequence of indices representing detected change points.
    """

    def __init__(self, detected_change_points: Sequence[int]):
        super().__init__(
            detected_change_points=detected_change_points, algorithm_name="MockAlgorithm", configuration_hash=12345
        )
