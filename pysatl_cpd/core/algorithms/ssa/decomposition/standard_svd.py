"""
Module for implementation of basic SSA decomposition step.
"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.ssa.abstracts import SVD, IDecomposition


class BasicSVD(IDecomposition):
    """
    Class of basic SSA decomposition step based on SVD.
    """

    def decompose(self, X: npt.NDArray[np.float64]) -> SVD:
        """
        Decomposes the trajectory matrix using SVD.
        :param X: trajectory matrix from embedding step.
        :return: matrix decomposition for grouping step.
        """
        U, sigma, _ = np.linalg.svd(X, full_matrices=False)
        return SVD(U=U, sigma=sigma)
