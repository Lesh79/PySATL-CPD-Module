"""
Module for implementation of SSA constant grouping step.
"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.ssa.abstracts import SVD, IGrouping


class ConstantGrouping(IGrouping):
    """
    Class of SSA constant grouping step.
    """

    def __init__(self, M: int) -> None:
        """
        Initializes SSA constant grouping step with given number of vectors.
        :param M: number of vectors in the main group.
        """
        self._M = M

    def group(self, svd: SVD) -> npt.NDArray[np.float64]:
        """
        Groups vectors to define a subspace of the time series with constant number of vectors.
        :param svd: decomposition of trajectory matrix.
        :return: vector group characterizing a subspace.
        """
        U = svd.U[np.argsort(svd.sigma)]
        return U[:, : self._M]
