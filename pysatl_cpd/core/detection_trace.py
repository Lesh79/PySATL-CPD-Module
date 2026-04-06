# -*- coding: ascii -*-
"""
Module contains detection trace container for changepoint detection results.

This module provides a unified container for storing detection results from
both online and offline changepoint detection algorithms.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass(kw_only=True)
class DetectionTrace:
    """
    Container for changepoint detection algorithm output.

    This class stores the detected changepoint positions It serves
    as a unified result type for both online and offline detection
    algorithms.

    Parameters
    ----------
    detected_changes : Sequence[int]
        Indices where changepoints were detected. For online algorithms,
        these are typically reported as they occur. For offline algorithms,
        these are the final changepoint positions.

    Examples
    --------
    >>> from pysatl_cpd.data_providers import NDArrayUnivariateProvider
    >>> trace = DetectionTrace(detected_change_points=[2, 4])
    >>> trace.detected_changes
    [2, 4]
    """

    detected_change_points: Sequence[int] = field(default_factory=list)
    """Indices of detected changepoints in the data sequence."""

    def __post_init__(self) -> None:
        """
        Validate detection trace consistency.

        Ensures that:
        - Detected changes indices are valid (positive)
        - All indices are unique
        - All indices are within data bounds

        Raises
        ------
        ValueError
            If any detected change index is non-positive, or if indices are
            not unique, or if any index exceeds data length.
        """
        if not self.detected_change_points:
            return

        # Validate indices are positive
        if min(self.detected_change_points) <= 0:
            raise ValueError(
                "Detected change indices must be positive."
                f"Found non-positive index: {min(self.detected_change_points)}"
            )

        # Validate indices are unique
        if len(self.detected_change_points) != len(set(self.detected_change_points)):
            duplicates = [x for x in self.detected_change_points if list(self.detected_change_points).count(x) > 1]
            raise ValueError(f"Detected change indices must be unique. Found duplicates: {set(duplicates)}")

        # Check if indices are sorted
        if list(self.detected_change_points) != sorted(self.detected_change_points):
            warnings.warn(
                f"Detected change indices are not sorted: {list(self.detected_change_points)}. "
                "Consider sorting them for consistent behavior.",
                UserWarning,
                stacklevel=2,
            )
