"""
Core Type Definitions

Fundamental types and data structures used throughout the PySATL CPD library.
"""

__author__ = "Mikhail Mikhailov"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"


from typing import Any

import numpy as np

type NumPyNumber = np.floating[Any] | np.integer[Any]
"""Type alias for NumPy numeric types."""

type Number = NumPyNumber | int | float
"""Type alias for all numeric types."""

type UnivariateNumericArray = np.ndarray[tuple[int], np.dtype[NumPyNumber]]
"""Type alias for numeric arrays."""

type MultivariateNumericArray = np.ndarray[tuple[int, int], np.dtype[NumPyNumber]]
"""Type alias for array of vectors."""

type NumericArray = np.ndarray[tuple[int, ...], np.dtype[NumPyNumber]]
"""Type alias for generic NumPy numeric array"""

type BoolArray = np.typing.NDArray[np.bool_]
"""Type alias for boolean arrays."""
