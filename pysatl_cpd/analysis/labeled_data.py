# -*- coding: ascii -*-
"""
Module contains wrapper for labeled dataset.
"""

__author__ = "Artem Romanyuk, Vladimir Kutuev, Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Collection, Iterator, Sequence

from pysatl_cpd.core.data_providers import DataProvider


class LabeledData[T](DataProvider[T]):
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

    def __init__(self, raw_data: Collection[T], change_points: Sequence[int], name: str | None = None):
        super().__init__(name)

        # Validate that change points are positive
        if change_points and min(change_points) <= 0:
            raise ValueError(f"Change point indices must be positive (>= 1). Found index: {min(change_points)}")

        # Validate that change points are within data bounds
        max_index = max(change_points) if change_points else 0
        if max_index > len(raw_data):
            raise ValueError(
                f"Change point index exceeds data length. Max index: {max_index}, data length: {len(raw_data)}"
            )

        self.__raw_data = raw_data
        self.__change_points = change_points

    def __iter__(self) -> Iterator[T]:
        """
        Return an iterator over the raw observations.

        Returns
        -------
        Iterator[T]
            Iterator yielding each observation in sequence.
        """
        return iter(self.__raw_data)

    def __len__(self) -> int:
        """
        Return the number of observations in the dataset.

        Returns
        -------
        int
            Total number of observations.
        """
        return len(self.__raw_data)

    def __str__(self) -> str:
        """
        Return a human-readable string representation.

        Returns
        -------
        str
            String representation with dataset name (if provided) and length.
        """
        return f"{self.name} (len = {len(self)})"

    @property
    def raw_data(self) -> Collection[T]:
        return self.__raw_data

    @property
    def change_points(self) -> Sequence[int]:
        return self.__change_points
