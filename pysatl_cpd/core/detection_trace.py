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
from typing import Any

from pysatl_cpd.core.data_providers import DataProvider


@dataclass(kw_only=True)
class DetectionTrace[DataProviderT: DataProvider[Any]]:
    """
    Container for changepoint detection algorithm output.

    This class stores the detected changepoint positions along with the data
    that was analyzed. It serves as a unified result type for both online
    and offline detection algorithms.

    Parameters
    ----------
    data : DataProviderT
        The data provider containing the analyzed time series.
    detected_changes : Sequence[int]
        Indices where changepoints were detected. For online algorithms,
        these are typically reported as they occur. For offline algorithms,
        these are the final changepoint positions.

    Examples
    --------
    >>> from pysatl_cpd.data_providers import NDArrayUnivariateProvider
    >>> data = NDArrayUnivariateProvider(np.array([1, 2, 3, 4, 5]))
    >>> trace = DetectionTrace(data=data, detected_changes=[2, 4])
    >>> trace.detected_changes
    [2, 4]
    """

    data: DataProviderT
    detected_changes: Sequence[int] = field(default_factory=list)
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
        if not self.detected_changes:
            return

        # Validate indices are positive
        if min(self.detected_changes) <= 0:
            raise ValueError(
                f"Detected change indices must be positive. Found non-positive index: {min(self.detected_changes)}"
            )

        # Validate indices are unique
        if len(self.detected_changes) != len(set(self.detected_changes)):
            duplicates = [x for x in self.detected_changes if list(self.detected_changes).count(x) > 1]
            raise ValueError(f"Detected change indices must be unique. Found duplicates: {set(duplicates)}")

        # Validate indices are within data bounds
        data_length = len(self.data)
        if max(self.detected_changes) >= data_length:
            raise ValueError(
                f"Detected change index {max(self.detected_changes)} exceeds data length {data_length - 1}"
            )

        # Check if indices are sorted
        if list(self.detected_changes) != sorted(self.detected_changes):
            warnings.warn(
                f"Detected change indices are not sorted: {list(self.detected_changes)}. "
                "Consider sorting them for consistent behavior.",
                UserWarning,
                stacklevel=2,
            )

    def __str__(self) -> str:
        """
        Return a human-readable string representation.

        Returns
        -------
        str
            String representation with number of detected changepoints.
        """
        changes_count = len(self.detected_changes)
        return f"DetectionTrace(changes={changes_count})"
