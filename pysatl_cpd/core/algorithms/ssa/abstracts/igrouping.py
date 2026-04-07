"""
Module for the SSA grouping step base class.
"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.ssa.abstracts.idecomposition import SVD


class IGrouping(ABC):
    """
    Abstract class of the third step of SSA (Grouping step).
    """

    @abstractmethod
    def group(self, svd: SVD) -> npt.NDArray[np.float64]:
        """
        Groups vectors to define a subspace of the time series.
        :param svd: decomposition of trajectory matrix.
        :return: vector group characterizing a subspace.
        """
