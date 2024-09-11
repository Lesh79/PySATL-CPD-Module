"""
Module for implementation of CPD algorithm based on nearest neighbours.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

from collections.abc import Iterable

import numpy as np

from CPDShell.Core.algorithms.abstract_algorithm import Algorithm


class KNNAlgorithm(Algorithm):
    """
    The class implementing change point detection algorithm based on nearest neighbours.
    """

    def __init__(
        self, metric: tp.Callable[[Observation, Observation], float], k=3, threshold: float = 0.5
    ) -> None:
        self._k = k
        self._threshold = threshold
        self._metric = metric

        self.__change_points: list[int] = []
        self.__change_points_count = 0

    def detect(self, window: Iterable[float | np.float64]) -> int:
        """Finds change points in window.

        :param window: part of global data for finding change points.
        :return: the number of change points in the window.
        """
        self.__process_data(False, window)
        return self.__change_points_count

    def localize(self, window: Iterable[float | np.float64]) -> list[int]:
        """Finds coordinates of change points (localizes them) in window.

        :param window: part of global data for finding change points.
        :return: list of window change points.
        """
        self.__process_data(True, window)
        return self.__change_points.copy()

