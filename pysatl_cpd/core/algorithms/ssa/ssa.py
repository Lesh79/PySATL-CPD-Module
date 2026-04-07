"""
Module for the SSA method.
"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.ssa.abstracts.idecomposition import IDecomposition
from pysatl_cpd.core.algorithms.ssa.abstracts.iembedding import IEmbedding
from pysatl_cpd.core.algorithms.ssa.abstracts.igrouping import IGrouping


class SSA:
    """
    Class for the SSA method.
    """

    def __init__(
        self,
        embedding_step: IEmbedding,
        decomposition_step: IDecomposition,
        grouping_step: IGrouping,
    ) -> None:
        """
        Initializes the steps of the SSA method.
        :param embedding_step: step for constructing the trajectory matrix.
        :param decomposition_step: step of decomposing the trajectory matrix into elementary matrices.
        :param grouping_step: step of grouping the main matrices.
        """
        self.__embedding_step = embedding_step
        self.__decomposition_step = decomposition_step
        self.__grouping_step = grouping_step

    def subspace(
        self, segment: npt.NDArray[np.float64], L: int
    ) -> npt.NDArray[np.float64]:
        """
        Determines the subspace vectors based on a segment of the time series.
        :param segment: segment of the time series.
        :param L: window width for the SSA method.
        :return: matrix of subspace vectors, where the number of columns equals the number of subspace vectors
        """
        X = self.__embedding_step.transform(segment, L)
        svd = self.__decomposition_step.decompose(X)

        return self.__grouping_step.group(svd)
