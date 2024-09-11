import typing as tp
import numpy as np
from statistics import mean
from math import sqrt
from collections import deque

import knn_graph as knngraph
from observations.observation import Observation, Observations


class KNNCPD:
    def __init__(self, window_size: int, metric: tp.Callable[[Observation, Observation], float],
                 observations: Observations | None = None, k=3,
                 threshold: float = 0.5) -> None:
        self._k = k
        self._metric = metric
        self._threshold = threshold
        self._observations = observations
        self._window_size = window_size
        self._statistics: float = 0.0
        if observations is not None and len(observations) >= window_size:
            self._knngraph = knngraph.KNNGraph(window_size, observations, metric)
            self._knngraph.build()
        else:
            self._knngraph = None

    @property
    def statistics(self) -> float:
        """
        Get the statistics of the current model
        :return: The statistics value
        """
        return self._statistics

    def update(self, observation: Observation) -> None:
        """
        Add an observation to the KNN graph
        :param observation: New observation
        """
        if self._observations is None:
            self._observations = deque([observation])
            return
        else:
            self._observations.append(observation)

        if self._knngraph is None and len(self._observations) >= self._window_size:
            self._knngraph = knngraph.KNNGraph(self._window_size, self._observations, self._metric)
            self._knngraph.build()
            self._statistics = self.calculate_statistics()
        elif self._knngraph is not None:
            self._knngraph.update(observation)
            self._statistics = self.calculate_statistics()

    def calculate_random_variable(self, permutation: np.array, t: int) -> int:
        """
        Calculate a random variable from a permutation and a fixed point
        :param permutation: permutation of observations
        :param t: fixed point that splits the permutation
        :return: value of the random variable
        """
        def b(i: int, j: int) -> bool:
            pi = permutation[i]
            pj = permutation[j]
            return (pi <= t < pj) or (pj <= t < pi)

        s = 0

        for i in range(self._window_size):
            for j in range(self._window_size):
                s += (self._knngraph.check_neighbour(i, j) + self._knngraph.check_neighbour(j, i)) * b(i, j)

        return s

    def calculate_statistics(self) -> float:
        """
        Calculate the statistics of the KNN graph in specified window
        :return: statistics value
        """
        if self._observations is None or len(self._observations) < self._window_size:
            return 0.0

        permutation: np.array = np.arange(self._window_size)
        np.random.shuffle(permutation)

        expectation = mean(self.calculate_random_variable(permutation, i) for i in range(self._window_size))
        expectation_sqr = mean(self.calculate_random_variable(permutation, i)**2 for i in range(self._window_size))
        deviation = sqrt(expectation_sqr - expectation**2)
        statistics = -(self.calculate_random_variable(permutation, self._window_size // 2) - expectation) / deviation

        return statistics

    def check_change_point(self) -> bool:
        """
        Check if change point occurs in current sequence
        :return: True if change point occurs, False otherwise
        """
        return self._statistics > self._threshold
