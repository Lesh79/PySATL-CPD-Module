"""
Module for the SSA decomposition step base class.
"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass
class SVD:
    """
    Dataclass of the singular value decomposition.
    """

    U: npt.NDArray[np.float64]
    sigma: npt.NDArray[np.float64]


class IDecomposition(ABC):
    """
    Abstract class of the second step of SSA (Decomposition step).
    """

    @abstractmethod
    def decompose(self, X: npt.NDArray[np.float64]) -> SVD:
        """
        Decomposes the trajectory matrix into elementary matrices.
        :param X: trajectory matrix from embedding step.
        :return: matrix decomposition for grouping step.
        """
