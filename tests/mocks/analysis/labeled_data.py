# -*- coding: ascii -*-

"""
Mock objects for labeled data analysis testing.

This module provides mock implementations of labeled data structures
used in testing change-point detection analysis components.
"""

__author__ = "Danil Totmyanin"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Sequence

from pysatl_cpd.analysis.labeled_data import LabeledData


class MockLabeledData(LabeledData[float]):
    """
    Mock implementation of LabeledData for testing purposes.

    This class generates a dummy sequence of raw data based on the
    provided change points to satisfy the base class requirements.

    Parameters
    ----------
    change_points : Sequence[int]
        A sequence of indices representing known change points.
    name : str, default="MockLabeledData"
        The name of the mock labeled data instance.
    """

    def __init__(self, change_points: Sequence[int], name: str = "MockLabeledData"):
        max_idx = max(change_points) if change_points else 0
        dummy_raw_data = [0.0] * max_idx
        super().__init__(raw_data=dummy_raw_data, change_points=change_points, name=name)


class MockLabeledDataWithPadding(LabeledData[float]):
    """
    Mock LabeledData where raw data length exceeds the maximum change point index.

    Unlike MockLabeledData (where len == max_cp), this mock adds padding so
    that the last observation index is not a change point. This prevents
    algorithms from producing detections at index 0 due to insufficient data.

    Parameters
    ----------
    change_points : Sequence[int]
        Known change point indices (1-based, must be positive).
    padding : int, default=10
        Number of extra observations to append after the last change point.
    name : str, default="MockLabeledDataWithPadding"
        Dataset identifier.
    """

    def __init__(
        self,
        change_points: Sequence[int],
        padding: int = 10,
        name: str = "MockLabeledDataWithPadding",
    ) -> None:
        max_idx = max(change_points) if change_points else 0
        dummy_raw_data = [0.0] * (max_idx + padding)
        super().__init__(raw_data=dummy_raw_data, change_points=change_points, name=name)
