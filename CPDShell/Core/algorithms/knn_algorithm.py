"""
Module for implementation of CPD algorithm based on nearest neighbours.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import typing as tp
from collections import deque
from collections.abc import Iterable
from math import sqrt

import numpy as np

import CPDShell.Core.algorithms.KNNCPD.knn_graph as knngraph
from CPDShell.Core.algorithms.abstract_algorithm import Algorithm


class KNNAlgorithm(Algorithm):
    """
    The class implementing change point detection algorithm based on nearest neighbours.
    """

    def __init__(
        self,
        metric: tp.Callable[[float, float], float] | tp.Callable[[np.float64, np.float64], float],
        k=3,
        threshold: float = 0.5,
    ) -> None:
        """
        Initializes a new instance of KNN change point algorithm.

        :param metric: function for calculating distance between points in time series.
        :param k: number of neighbours in graph relative to each point.
        :param threshold: threshold that statistics should overcome to fix change point.
        """
        self.__k = k
        self.__metric = metric
        self.__threshold = threshold

        self.__change_points: list[int] = []
        self.__change_points_count = 0

        self.__knngraph: knngraph.KNNGraph | None = None

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
        self.__process_data(window)
        return self.__change_points.copy()

    def __process_data(self, window: Iterable[float | np.float64]) -> None:
        """
        Processes a window of data to detect/localize all change points depending on working mode.

        :param window: part of global data for change points analysis.
        """
        sample = deque(window)
        sample_size = len(sample)
        if sample_size == 0:
            return

        # Preparing.
        self.__change_points: list[int] = []
        self.__change_points_count = 0

        # Building graph.
        self.__knngraph = knngraph.KNNGraph(window, self.__metric, self.__k)
        self.__knngraph.build()

        # Examining each point.
        # Boundaries are always change points.
        for time in range(1, len(window) - 1):
            statistics = self.__calculate_statistics_in_point(time, len(window))

            if self.__check_change_point(statistics):
                self.__change_points.append(time)
                self.__change_points_count += 1

    def __calculate_statistics_in_point(self, time: int, window_size: int) -> float:
        """
        Calculate the statistics of the KNN graph in specified point.

        :param time: index of point in the given sample to calculate statistics relative to it.
        :param window_size: size of sample to analyze.
        """
        assert self.__knngraph is not None, "Graph should not be None."

        k = self.__k
        n = window_size
        n_1 = time
        n_2 = n - time

        h = 4 * (n_1 - 1) * (n_2 - 1) / ((n - 2) * (n - 3))

        sum_1 = (1 / n) * sum(
            self.__knngraph.check_for_neighbourhood(i, j) * self.__knngraph.check_for_neighbourhood(j, i)
            for i in range(window_size)
            for j in range(window_size)
        )

        sum_2 = (1 / n) * sum(
            self.__knngraph.check_for_neighbourhood(j, i) * self.__knngraph.check_for_neighbourhood(m, i)
            for i in range(window_size)
            for j in range(window_size)
            for m in range(window_size)
        )

        expectation = 4 * k * n_1 * (n_2) / (n - 1)
        variance = (expectation / k) * (h * (sum_1 + k - (2 * k**2 / (n - 1))) + (1 - h) * (sum_2 - k**2))
        deviation = sqrt(variance)

        permutation: np.array = np.arange(window_size)
        # np.random.shuffle(permutation) # It seems that random permutation spoils the result

        statistics = -(self.__calculate_random_variable(permutation, time, window_size) - expectation) / deviation

        return statistics

    def __check_change_point(self, statistics: float) -> bool:
        """
        Check if calculated statistics is more than a given threshold to find out if it is a change point or not.

        :return: True if change point occurs, False otherwise.
        """
        return statistics > self.__threshold

    def __calculate_random_variable(self, permutation: np.array, t: int, window_size: int) -> int:
        """
        Calculate a random variable from a permutation and a fixed point.

        :param permutation: random permutation of observations.
        :param t: fixed point that splits the permutation.
        :return: value of the random variable.
        """

        def b(i: int, j: int) -> bool:
            pi = permutation[i]
            pj = permutation[j]
            return (pi <= t < pj) or (pj <= t < pi)

        s = sum(
            (self.__knngraph.check_for_neighbourhood(i, j) + self.__knngraph.check_for_neighbourhood(j, i)) * b(i, j)
            for i in range(window_size)
            for j in range(window_size)
        )

        return s
