import typing as tp
import numpy as np

from math import sqrt
from collections import deque
from collections.abc import Iterable

import knn_graph as knngraph
from observations.observation import Observation, Observations

class KNNCPD:
    def __init__(
        self, metric: tp.Callable[[float, float], float] | tp.Callable[[np.float64, np.float64], float], k=3, threshold: float = 0.5
    ) -> None:
        self.__k = k
        self.__metric = metric
        self.__threshold = threshold

        self.__change_points: list[int] = []
        self.__change_points_count = 0

        self.__knngraph: knngraph.KNNGraph | None = None

    @property
    def change_points_count(self) -> int:
        return self.__change_points_count
    
    @property
    def change_points(self) -> list[int]:
        return self.__change_points

    def process_sample(self, window: Iterable[float | np.float64]) -> None:
        if len(self.window) == 0:
            return

        self.__knngraph = knngraph.KNNGraph(window, self.__metric, self.__k)
        self.__knngraph.build()

        # Boundaries are always change points
        for time in range(1, len(window) - 1):
            statistics = self.__calculate_statistics_in_point(time, len(window))

            if self.__check_change_point(statistics):
                self.__change_points.append(time)
                self.__change_points_count += 1

    def __calculate_statistics_in_point(self, time: int, window_size: int) -> float:
        """
        Calculate the statistics of the KNN graph in specified point.
        """
        assert self.__knngraph is not None, "Graph should not be None."

        k = self.__k
        n = window_size
        n_1 = time
        n_2 = n - time

        h = 4 * (n_1 - 1) * (n_2 - 1) / ((n - 2) * (n - 3))

        sum_1 = (1 / n) * sum(self.__knngraph.check_for_neighbourhood(i, j) * self.__knngraph.check_for_neighbourhood(j, i)
                     for i in range(window_size) for j in range(window_size))
        
        sum_2 = (1 / n) * sum(self.__knngraph.check_for_neighbourhood(j, i) * self.__knngraph.check_for_neighbourhood(l, i)
                     for i in range(window_size)
                       for j in range(window_size)
                         for l in range (window_size))
        
        expectation = 4 * k * n_1 * (n_2) / (n - 1)
        variance = (expectation / k) * (h * (sum_1 + k - (2 * k**2 / (n-1))) + (1 - h) * (sum_2 - k**2))
        deviation = sqrt(variance)

        permutation: np.array = np.arange(window_size)
        np.random.shuffle(permutation)

        statistics = -(self.__calculate_random_variable(permutation, time) - expectation) / deviation

        return statistics

    def __check_change_point(self, statistics) -> bool:
        """
        Check if change point occurs in current sequence.
        :return: True if change point occurs, False otherwise.
        """
        return statistics > self.__threshold

    def __calculate_random_variable(self, permutation: np.array, t: int, window_size: int) -> int:
        """
        Calculate a random variable from a permutation and a fixed point.
        :param permutation: permutation of observations.
        :param t: fixed point that splits the permutation.
        :return: value of the random variable.
        """
        def b(i: int, j: int) -> bool:
            pi = permutation[i]
            pj = permutation[j]
            return (pi <= t < pj) or (pj <= t < pi)

        s = sum((self.__knngraph.check_for_neighbourhood(i, j) + self.__knngraph.check_for_neighbourhood(j, i)) * b(i, j)
                     for i in range(window_size) for j in range(window_size))

        return s
