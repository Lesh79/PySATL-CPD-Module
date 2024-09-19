"""
Module for implementation of nearest neighbours heap.
"""

__author__ = "Artemii Patov"
__copyright__ = "Copyright (c) 2024 Artemii Patov"
__license__ = "SPDX-License-Identifier: MIT"

import heapq
import typing as tp

from .abstracts.observation import Observation, Neighbour


class NNHeap:
    """
    The class implementing nearest neighbours heap --- helper abstraction for KNN graph.
    Internally uses auxillary heap of the same size as the main one for optimization purposes.
    """

    def __init__(
        self, size: int, metric: tp.Callable[[Observation, Observation], float], main_observation: Observation
    ) -> None:
        """
        Initializes a new instance of NNHeap.

        :param size: size of the heap.
        :param metric: function for calculating distance between two observations.
        :param main_observation: the central point relative to which the nearest neighbours are sought.
        """
        self._size = size
        self._metric = metric
        self._main_observation = main_observation

        self._heap: list[Neighbour] = []
        self._auxiliary_heap: list[Neighbour] = []

    def build(self, neighbours: list[Observation]) -> None:
        """
        Builds a nearest neighbour heap relative to the main observation with the given neighbours.

        :param neigbours: list of neighbours.
        """
        for neighbour in neighbours:
            self.__add(neighbour)

    def find_in_heap(self, observation: Observation) -> bool:
        """
        Checks if the given observation is among the nearest neighbours of the main observation.

        :param observation: observation to test.
        """
        def predicate(x: Neighbour) -> bool:
            return x.observation.value is observation.value

        return any(predicate(i) for i in self._heap)

    def __add(self, observation: Observation) -> None:
        """
        Adds observation to heap. Also if the observation is not getting into the main heap it can get into the auxiliary one.

        :param observation: observation to add.
        """
        if observation is self._main_observation:
            return

        neg_distance = -self._metric(self._main_observation, observation)
        neighbour = Neighbour(neg_distance, observation)

        if len(self._heap) == self._size and neighbour.distance >= self._heap[0].distance:
            heapq.heapreplace(self._heap, neighbour)
        elif len(self._heap) < self._size:
            heapq.heappush(self._heap, neighbour)