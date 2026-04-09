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
from typing import Any

from pysatl_cpd.analysis.labeled_data import LabeledData


class MockLabeledData(LabeledData[Any]):
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
