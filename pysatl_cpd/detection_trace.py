# -*- coding: ascii -*-
"""
Module contains detection trace container for changepoint detection results.

This module provides a unified container for storing detection results from
both online and offline changepoint detection algorithms.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Collection, Sequence
from dataclasses import dataclass


@dataclass(kw_only=True)
class DetectionTrace[T]:
    """
    Container for changepoint detection algorithm output.

    This class stores the detected changepoint positions and, optionally,
    the observation scores that led to those detections. It serves as a
    unified result type for both online and offline detection algorithms.

    Parameters
    ----------
    detected_changes : Sequence[int]
        Indices where changepoints were detected. For online algorithms,
        these are typically reported as they occur. For offline algorithms,
        these are the final changepoint positions.
    observation_scores : Collection[T] | None, optional
        Scores associated with each observation that contributed to changepoint
        detection. The interpretation of scores depends on the specific
        detection algorithm. For online algorithms, this may be a streaming
        collection; for offline algorithms, this may be a complete array
        of scores. Default is None.

    Examples
    --------
    >>> from pysatl_cpd.core import DetectionTrace
    >>> trace = DetectionTrace(detected_changes=[10, 25, 42])
    >>> trace.detected_changes
    [10, 25, 42]
    >>> trace.observation_scores is None
    True
    """

    detected_changes: Sequence[int]
    """Indices of detected changepoints in the data sequence."""

    observation_scores: Collection[T] | None = None
    """Scores associated with each observation (algorithm-specific)."""

    def __post_init__(self) -> None:
        """
        Validate detection trace consistency.

        Ensures that detected changes indices are valid (non-negative)
        when present. Note that empty sequence is allowed for cases where
        no changepoints were detected.

        Raises
        ------
        ValueError
            If any detected change index is negative.
        """
        # Validate that detected change indices are non-negative
        if self.detected_changes and min(self.detected_changes) < 0:
            raise ValueError(
                f"Detected change indices must be non-negative. Found negative index: {min(self.detected_changes)}"
            )

    def __len__(self) -> int:
        """
        Return the number of detected changepoints.

        Returns
        -------
        int
            Number of changepoints detected.
        """
        return len(self.detected_changes)

    def __str__(self) -> str:
        """
        Return a human-readable string representation.

        Returns
        -------
        str
            String representation with number of detected changepoints.
        """
        changes_count = len(self.detected_changes)
        scores_info = "with scores" if self.observation_scores is not None else "without scores"
        return f"DetectionTrace(changes={changes_count}, {scores_info})"
