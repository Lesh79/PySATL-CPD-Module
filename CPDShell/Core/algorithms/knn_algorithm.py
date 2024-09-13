"""
Module for implementation of CPD algorithm based on nearest neighbours.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import numpy as np
import typing as tp

from collections import deque
from collections.abc import Iterable

from CPDShell.Core.algorithms.KNNCPD.cpd import KNNCPD
from CPDShell.Core.algorithms.abstract_algorithm import Algorithm


class KNNAlgorithm(Algorithm):
    """
    The class implementing change point detection algorithm based on nearest neighbours.
    """

    def __init__(
        self, metric: tp.Callable[[float, float], float] | tp.Callable[[np.float64, np.float64], float], k=3, threshold: float = 0.5
    ) -> None:
        self.__k = k
        self.__threshold = threshold
        self.__metric = metric

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

    def __process_data(self, with_localization: bool, window: Iterable[float | np.float64]) -> None:
        """
        Processes a window of data to detect/localize all change points depending on working mode.
        :param with_localization: boolean flag representing whether function needs to localize a change point.
        :param window: part of global data for change points analysis.
        """
        self.__prepare()

        sample = deque(window)
        sample_size = len(sample)
        if sample_size == 0:
            return
        
        knncpd = KNNCPD(len(window), self.__metric, window, self.__k, self.__threshold)
        knncpd.process_sample()

        return knncpd.change_points if with_localization else knncpd.change_points_count
        # build a graph and examine each of the points in sample
        # If the window size will be managed outside algorithm, algorithm should get
        # whole window and build a graph on the whole window not point by point.
        # I should make graph code more efficient. But probably first i should integrate it and then if i would have a time to optimize it.

    def __prepare(self) -> None:
        """
        Clear algorithm's state (including change points and time related information) before data processing.
        :param sample_size: an overall size of the sample.
        """
        self.__change_points = []
        self.__change_points_count = 0
