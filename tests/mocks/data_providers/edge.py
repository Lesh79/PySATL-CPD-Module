"""
Mock edge case data providers for testing.
"""

from collections.abc import Iterator, Sequence

from pysatl_cpd._typing import Number
from pysatl_cpd.data_providers import DataProvider


class MockNaNDataProvider(DataProvider[Number]):
    """
    Mock data provider that yields NaN values.

    Useful for testing algorithm robustness with missing data.

    Parameters
    ----------
    length : int
        Number of observations to yield.
    """

    def __init__(self, length: int) -> None:
        self._length = length
        self._call_count = 0

    def __iter__(self) -> Iterator[Number]:
        """Return iterator yielding NaN values."""
        self._call_count += 1
        return iter([float("nan")] * self._length)

    def __len__(self) -> int:
        """Return number of observations."""
        return self._length

    def __getitem__(self, index: int) -> Number:
        """Get NaN at specific index."""
        if 0 <= index < self._length:
            return float("nan")
        raise IndexError("Index out of range")

    def get_call_count(self) -> int:
        """Return number of times __iter__ was called."""
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the call counter."""
        self._call_count = 0

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MockNaNDataProvider(length={self._length})"


class MockInfDataProvider(DataProvider[Number]):
    """
    Mock data provider that yields Inf values.

    Useful for testing algorithm robustness with infinite values.

    Parameters
    ----------
    length : int
        Number of observations to yield.
    """

    def __init__(self, length: int) -> None:
        self._length = length
        self._call_count = 0

    def __iter__(self) -> Iterator[Number]:
        """Return iterator yielding Inf values."""
        self._call_count += 1
        return iter([float("inf")] * self._length)

    def __len__(self) -> int:
        """Return number of observations."""
        return self._length

    def __getitem__(self, index: int) -> Number:
        """Get Inf at specific index."""
        if 0 <= index < self._length:
            return float("inf")
        raise IndexError("Index out of range")

    def get_call_count(self) -> int:
        """Return number of times __iter__ was called."""
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the call counter."""
        self._call_count = 0

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MockInfDataProvider(length={self._length})"


class MockNegativeDataProvider(DataProvider[Number]):
    """
    Mock data provider that yields negative values.

    Useful for testing algorithms with negative observations.

    Parameters
    ----------
    values : Sequence[Number]
        Sequence of negative observations to yield.
    """

    def __init__(self, values: Sequence[Number]) -> None:
        # Validate all values are negative
        for v in values:
            if v >= 0:
                raise ValueError(f"Expected negative value, got {v}")
        self._data = list(values)
        self._call_count = 0

    def __iter__(self) -> Iterator[Number]:
        """Return iterator over negative values."""
        self._call_count += 1
        return iter(self._data)

    def __len__(self) -> int:
        """Return number of observations."""
        return len(self._data)

    def __getitem__(self, index: int) -> Number:
        """Get observation at specific index."""
        return self._data[index]

    def get_call_count(self) -> int:
        """Return number of times __iter__ was called."""
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the call counter."""
        self._call_count = 0

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MockNegativeDataProvider(length={len(self)})"


class MockZeroDataProvider(DataProvider[Number]):
    """
    Mock data provider that yields zeros.

    Useful for testing algorithms with zero observations.

    Parameters
    ----------
    length : int
        Number of observations to yield.
    """

    def __init__(self, length: int) -> None:
        self._length = length
        self._call_count = 0

    def __iter__(self) -> Iterator[Number]:
        """Return iterator yielding zeros."""
        self._call_count += 1
        return iter([0] * self._length)

    def __len__(self) -> int:
        """Return number of observations."""
        return self._length

    def __getitem__(self, index: int) -> Number:
        """Get zero at specific index."""
        if 0 <= index < self._length:
            return 0
        raise IndexError("Index out of range")

    def get_call_count(self) -> int:
        """Return number of times __iter__ was called."""
        return self._call_count

    def reset_call_count(self) -> None:
        """Reset the call counter."""
        self._call_count = 0

    def __repr__(self) -> str:
        """Return string representation."""
        return f"MockZeroDataProvider(length={self._length})"


class MockMultivariateEdgeDataProvider(DataProvider[list[Number]]):
    """
    Mock data provider for multivariate edge cases.

    Allows injecting NaN/Inf values into specific positions in observation vectors.

    Parameters
    ----------
    data : Sequence[Sequence[Number]]
        Base data sequence.
    nan_positions : list[tuple[int, int]], optional
        List of (observation_index, dimension_index) where NaN should be injected.
    inf_positions : list[tuple[int, int]], optional
        List of (observation_index, dimension_index) where Inf should be injected.
    """

    def __init__(
        self,
        data: Sequence[Sequence[Number]],
        nan_positions: list[tuple[int, int]] | None = None,
        inf_positions: list[tuple[int, int]] | None = None,
    ) -> None:
        # Convert to list of lists for mutability
        self._data = [list(obs) for obs in data]

        if not self._data:
            self._dimensions = 0
        else:
            # Verify all observations have same length
            dims = {len(obs) for obs in self._data}
            if len(dims) != 1:
                raise ValueError(f"All observations must have same dimensions, got {dims}")
            self._dimensions = dims.pop()

            # Inject NaN values
            if nan_positions:
                for obs_idx, dim_idx in nan_positions:
                    if 0 <= obs_idx < len(self._data) and 0 <= dim_idx < self._dimensions:
                        self._data[obs_idx][dim_idx] = float("nan")

            # Inject Inf values
            if inf_positions:
                for obs_idx, dim_idx in inf_positions:
                    if 0 <= obs_idx < len(self._data) and 0 <= dim_idx < self._dimensions:
                        self._data[obs_idx][dim_idx] = float("inf")

        self._call_count = 0

    def __iter__(self) -> Iterator[list[Number]]:
        """Return iterator over the data rows."""
        self._call_count += 1
        return iter(self._data)

    def __len__(self) -> int:
        """Return number of observations."""
        return len(self._data)

    def __getitem__(self, index: int) -> list[Number]:
        """Get observation vector at specific index."""
        return self._data[index]

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
        return f"MockMultivariateEdgeDataProvider(observations={len(self)}, dimensions={self.dimensions})"
