# -*- coding: ascii -*-

"""
Mock dirty data providers for testing.

These providers wrap existing data providers and inject NaN or Inf values
at specified positions to test algorithm robustness.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from collections.abc import Iterator

from pysatl_cpd.core.data_providers import DataProvider
from pysatl_cpd.core.typedefs import Number


class MockUnivariateDirtyDataProvider(DataProvider[Number]):
    """
    Mock data provider that injects dirty values into univariate data.

    Wraps an existing univariate data provider and replaces values at
    specified indices with NaN or Inf.

    Parameters
    ----------
    source : DataProvider[Number]
        The source data provider to wrap.
    nan_indices : list[int], optional
        Indices where NaN should be injected.
    inf_indices : list[int], optional
        Indices where Inf should be injected.
    """

    def __init__(
        self,
        source: DataProvider[Number],
        nan_indices: list[int] | None = None,
        inf_indices: list[int] | None = None,
    ) -> None:
        self._source = source
        self._nan_indices = set(nan_indices or [])
        self._inf_indices = set(inf_indices or [])
        self._call_count = 0

        # Validate no overlapping indices
        overlapping = self._nan_indices & self._inf_indices
        if overlapping:
            raise ValueError(f"Indices cannot be both NaN and Inf: {overlapping}")

        # Pre-load data to allow indexing
        self._data = list(source)

    def __iter__(self) -> Iterator[Number]:
        """Return iterator with dirty values injected."""
        self._call_count += 1
        result = []
        for i, value in enumerate(self._data):
            if i in self._nan_indices:
                result.append(float("nan"))
            elif i in self._inf_indices:
                result.append(float("inf"))
            else:
                result.append(value)
        return iter(result)

    def __len__(self) -> int:
        """Return number of observations."""
        return len(self._data)

    def __getitem__(self, index: int) -> Number:
        """Get observation at specific index with dirty injection."""
        if index < 0 or index >= len(self._data):
            raise IndexError("Index out of range")

        if index in self._nan_indices:
            return float("nan")
        elif index in self._inf_indices:
            return float("inf")
        return self._data[index]

    def get_call_count(self) -> int:
        """Return number of times __iter__ was called."""
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the call counter."""
        self._call_count = 0

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"MockUnivariateDirtyDataProvider("
            f"source={self._source!r}, "
            f"nan_indices={sorted(self._nan_indices)}, "
            f"inf_indices={sorted(self._inf_indices)})"
        )


class MockMultivariateDirtyDataProvider(DataProvider[list[Number]]):
    """
    Mock data provider that injects dirty values into multivariate data.

    Wraps an existing multivariate data provider and replaces specific
    elements at specified (observation, dimension) positions with NaN or Inf.

    Parameters
    ----------
    source : DataProvider[list[Number]]
        The source data provider to wrap.
    nan_positions : list[tuple[int, int]], optional
        List of (observation_index, dimension_index) where NaN should be injected.
    inf_positions : list[tuple[int, int]], optional
        List of (observation_index, dimension_index) where Inf should be injected.
    """

    def __init__(
        self,
        source: DataProvider[list[Number]],
        nan_positions: list[tuple[int, int]] | None = None,
        inf_positions: list[tuple[int, int]] | None = None,
    ) -> None:
        self._source = source
        self._nan_positions = set(nan_positions or [])
        self._inf_positions = set(inf_positions or [])
        self._call_count = 0

        # Validate no overlapping positions
        overlapping = self._nan_positions & self._inf_positions
        if overlapping:
            raise ValueError(f"Positions cannot be both NaN and Inf: {overlapping}")

        # Pre-load data to allow indexing
        self._data = [list(obs) for obs in source]

        # Validate dimensions are consistent
        if self._data:
            dims = {len(obs) for obs in self._data}
            if len(dims) != 1:
                raise ValueError(f"All observations must have same dimensions, got {dims}")
            self._dimensions = dims.pop()
        else:
            self._dimensions = 0

        # Validate positions are within bounds
        for obs_idx, dim_idx in self._nan_positions | self._inf_positions:
            if obs_idx < 0 or obs_idx >= len(self._data):
                raise IndexError(f"Observation index {obs_idx} out of range (0-{len(self._data)-1})")
            if dim_idx < 0 or dim_idx >= self._dimensions:
                raise IndexError(f"Dimension index {dim_idx} out of range (0-{self._dimensions-1})")

    def __iter__(self) -> Iterator[list[Number]]:
        """Return iterator with dirty values injected."""
        self._call_count += 1

        for obs_idx, observation in enumerate(self._data):
            result = observation.copy()
            for dim_idx in range(len(result)):
                if (obs_idx, dim_idx) in self._nan_positions:
                    result[dim_idx] = float("nan")
                elif (obs_idx, dim_idx) in self._inf_positions:
                    result[dim_idx] = float("inf")
            yield result

    def __len__(self) -> int:
        """Return number of observations."""
        return len(self._data)

    def __getitem__(self, index: int) -> list[Number]:
        """Get observation vector at specific index with dirty injection."""
        if index < 0 or index >= len(self._data):
            raise IndexError("Index out of range")

        observation = self._data[index].copy()
        for dim_idx in range(len(observation)):
            if (index, dim_idx) in self._nan_positions:
                observation[dim_idx] = float("nan")
            elif (index, dim_idx) in self._inf_positions:
                observation[dim_idx] = float("inf")
        return observation

    def get_call_count(self) -> int:
        """Return number of times __iter__ was called."""
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the call counter."""
        self._call_count = 0

    @property
    def dimensions(self) -> int:
        """Return number of dimensions (variables)."""
        return self._dimensions

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"MockMultivariateDirtyDataProvider("
            f"source={self._source!r}, "
            f"nan_positions={sorted(self._nan_positions)}, "
            f"inf_positions={sorted(self._inf_positions)})"
        )
