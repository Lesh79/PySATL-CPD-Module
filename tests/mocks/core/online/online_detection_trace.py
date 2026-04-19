# -*- coding: ascii -*-

"""
Mock online core components for testing.

This module provides mock implementations of online data structures,
specifically online detection traces.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence
from typing import Any

import numpy as np

from pysatl_cpd.core.online.online_detection_trace import OnlineDetectionTrace


class MockOnlineDetectionTrace(OnlineDetectionTrace[Any]):
    """
    Mock implementation of OnlineDetectionTrace for testing purposes.

    Provides a simple online trace with predefined detected change points,
    along with empty arrays for processing times, detection functions,
    and algorithm states.

    Parameters
    ----------
    detected_change_points : Sequence[int]
        A sequence of indices representing detected change points.
    """

    def __init__(self, detected_change_points: Sequence[int]):
        super().__init__(
            detected_change_points=detected_change_points,
            algorithm_name="MockOnlineAlgorithm",
            configuration_hash=12345,
            processing_time=np.array([]),
            detection_function=np.array([]),
            algorithm_states=[],
        )

    def slice(self, start: int, end: int) -> "MockOnlineDetectionTrace":
        """
        Mock implementation of slice.

        Returns a new MockOnlineDetectionTrace containing only the change points
        that fall within [start, end], shifted relative to `start`.
        """

        shifted_cps: list[int] = [cp - start for cp in self.detected_change_points if start <= cp <= end]
        return MockOnlineDetectionTrace(detected_change_points=shifted_cps)
