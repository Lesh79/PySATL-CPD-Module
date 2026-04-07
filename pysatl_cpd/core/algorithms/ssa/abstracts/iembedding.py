"""
Module for the SSA embedding step base class.
"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class IEmbedding(ABC):
    """
    Abstract class of the first step of SSA (Embedding step).
    """

    @abstractmethod
    def transform(
        self, segment: npt.NDArray[np.float64], L: int
    ) -> npt.NDArray[np.float64]:
        """
        Converts a segment of the time series into a trajectory matrix.
        :param segment: segment of the time series.
        :param L: window width.
        :return: trajectory matrix for decomposition step.
        """
