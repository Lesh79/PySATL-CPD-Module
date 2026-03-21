"""
Module contains wrapper for labeled dataset.
"""

__author__ = "Artem Romanyuk, Vladimir Kutuev, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2025 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Collection, Iterator, Sequence
from dataclasses import dataclass
from typing import TypeVar

from pysatl_cpd.data_providers import DataProvider

T = TypeVar("T")


@dataclass
class LabeledData(DataProvider[T]):
    """
    Container for labeled time series data with known change point locations.

    This class wraps raw data and associated change point indices, providing
    iteration over observations and metadata about change point positions.
    Change points are represented as indices where a regime change occurs,
    with each index indicating the first observation after the change.

    Parameters
    ----------
    raw_data : Collection[T]
        The sequential observations forming the time series.
    change_points : Sequence[int]
        Indices of change points in the data. Each index must be positive
        (>= 1) and less than or equal to len(raw_data). An index i indicates
        that a regime change occurs between observations i-1 and i.
    name : str | None, optional
        Optional identifier for the dataset. Default is None.

    Raises
    ------
    ValueError
        If any change point index is less than or equal to 0.
    ValueError
        If any change point index exceeds the length of raw_data.

    Examples
    --------
    >>> data = [1, 2, 3, 10, 11, 12]
    >>> labeled = LabeledData(data, change_points=[3])
    >>> list(labeled)
    [1, 2, 3, 10, 11, 12]
    >>> len(labeled)
    6
    """

    raw_data: Collection[T]
    change_points: Sequence[int]
    name: str | None = None

    def __post_init__(self) -> None:
        """
        Validate change point indices after initialization.

        Verifies that all change point indices are positive and within
        the bounds of the raw data. Change points must be >= 1 because
        they represent the first observation index after a regime change,
        and index 0 would imply a change before any data exists.

        Raises
        ------
        ValueError
            If any change point index is <= 0.
        ValueError
            If any change point index exceeds the length of raw_data.
        """
        # Validate that change points are positive
        if self.change_points and min(self.change_points) <= 0:
            raise ValueError(f"Change point indices must be positive (>= 1). Found index: {min(self.change_points)}")

        # Validate that change points are within data bounds
        max_index = max(self.change_points) if self.change_points else 0
        if max_index > len(self.raw_data):
            raise ValueError(
                f"Change point index exceeds data length. Max index: {max_index}, data length: {len(self.raw_data)}"
            )

    def __iter__(self) -> Iterator[T]:
        """
        Return an iterator over the raw observations.

        Returns
        -------
        Iterator[T]
            Iterator yielding each observation in sequence.
        """
        return iter(self.raw_data)

    def __len__(self) -> int:
        """
        Return the number of observations in the dataset.

        Returns
        -------
        int
            Total number of observations.
        """
        return len(self.raw_data)

    def __str__(self) -> str:
        """
        Return a human-readable string representation.

        Returns
        -------
        str
            String representation with dataset name (if provided) and length.
        """
        if self.name is not None:
            return f"{self.name} (len = {len(self)})"
        return f"Labeled Data (len = {len(self)})"
