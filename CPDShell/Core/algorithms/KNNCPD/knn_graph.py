import numpy as np
import typing as tp
from collections import deque
from collections.abc import Iterable

from observations.observation import Observation
from observations.observation_heap import NNHeap


class KNNGraph:
    def __init__(
        self, window: Iterable[float | np.float64], metric: tp.Callable[[float, float], float] | tp.Callable[[np.float64, np.float64], float], k=3
    ) -> None:
        self.__window: list[Observation] = [Observation(t, v) for t, v in enumerate(window)]
        self.__metric: tp.Callable[[Observation, Observation], float] = lambda obs1, obs2: metric(obs1.value, obs2.value) 
        self.__k = k

        self.__window_size = len(window)
        self.__graph: deque[NNHeap] = deque(maxlen=self.__window_size)

    def build(self) -> None:
        """
        Build KNN graph according to the given parameters.
        """
        for i in range(self.__window_size):
            heap = NNHeap(self.__k, self.__metric, self.__window[-i - 1])
            heap.build(self.__window)
            self.__graph.appendleft(heap)

    def check_for_neighbourhood(self, first_index: int, second_index: int) -> bool:
        """
        Checks if the second observation is among the k nearest neighbours of the first observation.
        :param first_index:
        :param second_index:
        :return:
        """
        neighbour = self.__window[second_index]
        return self.__graph[first_index].find_in_heap(neighbour)
