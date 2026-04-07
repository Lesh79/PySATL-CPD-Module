"""
Module for implementation of basic SSA embedding step.
"""

__author__ = "Mark Dubrovchenko"
__copyright__ = "Copyright (c) 2026 PySATL project"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import numpy.typing as npt

from pysatl_cpd.core.algorithms.ssa.abstracts.iembedding import IEmbedding


class BasicEmbedding(IEmbedding):
    """
    Class of basic SSA embedding step based on a sliding window.
    """

    def transform(
        self, segment: npt.NDArray[np.float64], L: int
    ) -> npt.NDArray[np.float64]:
        """
        Converts a segment of the time series into a trajectory matrix based on a sliding window.
        :param segment: segment of the time series.
        :param L: window width.
        :return: trajectory matrix for decomposition step.
        """
        K = len(segment) - L + 1
        X = np.zeros((L, K))
        for i in range(K):
            X[:, i] = segment[i : i + L]

        return X
